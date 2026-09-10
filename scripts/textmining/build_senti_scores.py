"""
研报摘要 FinBERT 情感打分（对齐华泰 AI 41 研报情感因子）
========================================================

AI 41《基于BERT的分析师研报情感因子》(20210118) 流程：
1. 预训练中文 BERT → 用带情感标注金融舆情微调 → 对无标注研报摘要预测情感
2. 摘要预处理：剔除转义字符、删除"风险提示"后内容
3. 逐句打分，取 P(正面) 作为情感概率

我们的近似（2026-09-09，如实声明与研报的差异）：
- 模型：yiyanghkust/finbert-tone-chinese（已在金融文本上微调的三分类
  Neutral/Positive/Negative），跳过自微调（与 encode_fadt_bert 同一近似）
- 粒度：**报告级**而非逐句——研报摘要均长 ~1000 字符，逐句编码在本机
  CPU（22 线程，无 GPU）需 ~11 小时；报告级截断 256 token 约 3-4 小时。
  报告情感得分 = P(pos) - P(neg)（三分类边际差，中性 ≈ 0，与研报
  "P(正面)-0.5" 同为以 0 为中点的情感得分）。逐句与报告级口径的
  一致性在 build_senti_factors 抽样验证。
- 输入：标题 + 摘要头部（max_len 256 token，约摘要前 250 字）。
  研报摘要结论前置，头部信息密度最高；尾部多为财务表格与风险提示
  （后者本就按研报口径剔除）。

用法（双分片并行，约 3 小时）：
    python -m scripts.textmining.build_senti_scores --pool zz1000 --shard 0 --num-shards 2
    python -m scripts.textmining.build_senti_scores --pool zz1000 --shard 1 --num-shards 2

产出：
    reports/textmining/senti/features/senti_scores_{pool}.parquet
    列：row_idx, code, date, n_chars, p_neu, p_pos, p_neg, title_head
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cli_common import setup_logging  # noqa: E402
from scripts.textmining._paths import Out  # noqa: E402

OUT_DIR = Out("senti")
log = setup_logging("senti_scores")

RISK_PAT = re.compile(r"风险提示")


def clean_text(title: str, summary: str) -> str:
    """AI 41 预处理近似：拼接标题+摘要，剔除转义字符，删除'风险提示'后内容。"""
    t = (str(title).strip() + " " + str(summary).strip())
    t = t.replace("\\r###", "。").replace("\\r", "").replace("\\n", "")
    m = RISK_PAT.search(t)
    if m:
        t = t[: m.start()]
    return re.sub(r"\s+", " ", t).strip()


def load_model(max_len: int = 256):
    """加载 FinBERT 三分类模型（含微调分类头，CPU 推理）。"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    try:
        from config import Config
        d = Config.get().get("textmining", {}).get("bert_model_dir")
    except Exception:
        d = None
    model_dir = str(d).replace("//", "/") if d else r"E:/data/models/finbert_tone_chinese"
    torch.set_num_threads(max(1, args_threads[0]))
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tok


args_threads = [22]  # 由 main 写入，load_model 读取（避免改 transformers 内部签名）


def run(pool: str = "zz1000", begin: str = "20200701", max_len: int = 256,
        batch_size: int = 48, shard: int = 0, num_shards: int = 1,
        threads: int = 22) -> pd.DataFrame:
    args_threads[0] = threads
    out_path = OUT_DIR / f"senti_scores_{pool}.parquet"
    tmp_dir = OUT_DIR / f"senti_scores_{pool}_parts"
    tmp_dir.mkdir(exist_ok=True)

    # ── 语料：池内研报（AI 41 用全部研报摘要，不限调整事件）──
    from config import Config
    cache_root = Path(str(Config.cache()["root"]).replace("//", "/"))
    if pool == "zz1000":
        daily_path = cache_root / "archive_zz1000" / "daily_zz1000_only.parquet"
    else:
        daily_path = cache_root / f"daily_{pool}.parquet"
    codes = set(pd.read_parquet(daily_path, columns=[])
                .index.get_level_values("code").unique())

    rep = pd.read_parquet(cache_root / "text_ths_report.parquet")
    rep = rep[rep["code"].isin(codes)].copy()
    rep["date"] = pd.to_datetime(rep["date"]).dt.normalize()
    rep = rep[rep["date"] >= pd.Timestamp(begin)].reset_index(drop=True)
    log.info("池 %s 研报 %d 条 / %d 只（%s 起）", pool, len(rep),
             rep["code"].nunique(), begin)

    rep["text"] = [clean_text(t, s) for t, s in
                   zip(rep["title"].fillna(""), rep["summary"].fillna(""))]
    rep = rep[rep["text"].str.len() > 8].reset_index(names="row_idx")
    log.info("清洗后有效文本 %d 条", len(rep))

    # 分片（双进程并行时各拿一半）+ 按长度排序只用于批内动态 padding，
    # 输出顺序按 row_idx 恢复
    shard_idx = np.array([i for i in range(len(rep)) if i % num_shards == shard])
    rep_shard = rep.iloc[shard_idx].reset_index(drop=True)
    log.info("分片 %d/%d: %d 条", shard + 1, num_shards, len(rep_shard))

    # ── 编码（分 chunk 断点续跑）──
    model, tok = load_model(max_len)
    import torch

    chunk = 2000
    n_chunks = (len(rep_shard) + chunk - 1) // chunk
    t0 = pd.Timestamp.now()
    done_rows = 0
    for ci in range(n_chunks):
        part_path = tmp_dir / f"part_{shard:02d}_{ci:04d}.parquet"
        if part_path.exists():
            done_rows += chunk
            continue
        sub = rep_shard.iloc[ci * chunk:(ci + 1) * chunk]
        texts = sub["text"].tolist()
        lens = np.array([len(tok(t)["input_ids"]) for t in texts])
        order = np.argsort(lens)  # 长度排序 → 批内 padding 最小
        probs = np.zeros((len(texts), 3), dtype=np.float32)  # 0 neu, 1 pos, 2 neg
        for i in range(0, len(order), batch_size):
            sel = order[i:i + batch_size]
            enc = tok([texts[j] for j in sel], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            with torch.no_grad():
                logits = model(**enc).logits
            probs[sel] = torch.softmax(logits, dim=1).numpy()
        part = pd.DataFrame({
            "row_idx": sub["row_idx"].values,
            "code": sub["code"].values,
            "date": sub["date"].values,
            "n_chars": sub["text"].str.len().values,
            "p_neu": probs[:, 0], "p_pos": probs[:, 1], "p_neg": probs[:, 2],
            "title_head": sub["title"].str.slice(0, 60).values,
        })
        part.to_parquet(part_path, compression="snappy")
        done_rows += len(sub)
        rate = done_rows / max((pd.Timestamp.now() - t0).total_seconds(), 1)
        eta_min = (len(rep_shard) - done_rows) / max(rate, 0.1) / 60
        log.info("  分片%d chunk %d/%d 完成，速率 %.1f 篇/s，剩余约 %.0f 分钟",
                 shard + 1, ci + 1, n_chunks, rate, eta_min)

    # ── 合并（全部分片完成后由最后一个进程写出总表）──
    parts = sorted(tmp_dir.glob("part_*.parquet"))
    expected = n_chunks * num_shards
    if len(parts) < expected:
        log.info("尚有 %d/%d 个 chunk 未完成，跳过合并", len(parts), expected)
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.sort_values("row_idx").reset_index(drop=True)
    df.to_parquet(out_path, compression="snappy")
    log.info("情感得分已存: %s (%d 行, P(pos)均值 %.3f)",
             out_path, len(df), df["p_pos"].mean())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="zz1000", choices=["hs300", "zz1000"])
    ap.add_argument("--begin", default="20200701")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=22)
    args = ap.parse_args()
    run(args.pool, args.begin, args.max_len, args.batch_size,
        args.shard, args.num_shards, args.threads)
