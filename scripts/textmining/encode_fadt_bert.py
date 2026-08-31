"""
FinBERT CLS 编码（对齐华泰 AI 63 文本表示升级）
================================================

研报方案（AI 63）：
1. 用带情感标注的万得新闻微调 FinBERT（14.8 万条，准确率 95%+）
2. 研报文本预处理：标题+摘要拼接，截长补短 N=500（基础模型参数）
3. 输入 FinBERT，去掉微调头（4-7 层），输出 CLS 层 768 维编码
4. 用 768 维编码替代词频向量作为 XGBoost 二次训练的特征

我们的近似（2026-08-18）：
- 无万得标注数据 + CPU 环境（无法 Adapter 微调）
- 用已微调中文金融 FinBERT（yiyanghkust/finbert-tone-chinese）直接编码，
  取其 CLS 层（768 维），跳过自微调步骤
- 如实声明：研报微调版 27.5% vs 不微调 23.0% 有 ~4.5pct 差距，我们用
  已微调通用金融模型近似"微调版"，实际效果介于两者之间

用法：
    python -m scripts.textmining.encode_fadt_bert --task fadt --pool zz1000 --max-len 500
    python -m scripts.textmining.encode_fadt_bert --task sue --pool hs300

产出：
    reports/textmining/{task}_cls_{pool}.parquet  (code, event_date, cls_0..cls_767)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cli_common import setup_logging  # noqa: E402

OUT_DIR = ROOT / "reports" / "textmining"
# 本地模型目录（hf-mirror 下载，沙箱无法用 huggingface_hub 缓存管理）；
# 路径真源在 config/settings.yaml 的 textmining.bert_model_dir
def _bert_model_dir() -> str:
    try:
        from config import Config
        d = Config.get().get("textmining", {}).get("bert_model_dir")
        if d:
            return str(d).replace("//", "/")
    except Exception:
        pass
    return r"E:/data/models/finbert_tone_chinese"

MODEL_DIR = _bert_model_dir()
log = setup_logging("encode_bert")

# tokenizer 缺失时手动加载的中文 BERT 词表（bert-base-chinese 同款）——不使用，
# 直接依赖 transformers 的 AutoTokenizer


def load_model(max_len: int = 500):
    """加载 FinBERT + tokenizer（CPU 推理）。"""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tok


def encode_batch(model, tok, texts: list[str], max_len: int = 500,
                 batch_size: int = 8) -> np.ndarray:
    """编码一批文本，返回 (n, 768) CLS 向量。

    研报：截长补短（前截断保留标题+首段核心观点），输入 [CLS] text [SEP]。
    batch_size 默认 8（CPU 内存受限，FADT 全量 2.7 万行长文本曾 OOM）。
    """
    import torch
    out = []
    n = len(texts)
    for i in range(0, n, batch_size):
        chunk_texts = texts[i:i + batch_size]
        enc = tok(chunk_texts, padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            last_hidden = model(**enc).last_hidden_state
        # CLS 层 = 第 0 个 token 的隐藏层输出（768 维）
        cls = last_hidden[:, 0, :].numpy()
        out.append(cls)
    return np.vstack(out)


def run(task: str = "fadt", pool: str = "zz1000", max_len: int = 500,
        force: bool = False, limit: int | None = None,
        chunk: int = 1000) -> pd.DataFrame:
    sp = OUT_DIR / (f"{task}_samples_{pool}.parquet" if task == "fadt"
                    else f"sue_txt_samples_{pool}.parquet")
    out_path = OUT_DIR / f"{task}_cls_{pool}.parquet"
    if out_path.exists() and not force:
        log.info("编码缓存已存在: %s", out_path)
        return pd.read_parquet(out_path)

    samples = pd.read_parquet(sp)
    samples["event_date"] = pd.to_datetime(samples["event_date"]).dt.normalize()
    log.info("样本 %d 行 / %d 只", len(samples), samples["code"].nunique())

    # 文本 = 标题 + 摘要（研报图表25：标题与摘要拼接，前截断保留核心）
    if "title" in samples.columns and "summary" in samples.columns:
        samples["text"] = (samples["title"].fillna("") + " " +
                           samples["summary"].fillna(""))
    else:
        samples["text"] = samples["title"].fillna("")
    # 清洗：移除 \r### 段落分隔符、\r、\n（保留语义分隔为句号）
    samples["text"] = (samples["text"].str.replace(r"\\r###", "。", regex=True)
                       .str.replace(r"\\r", "", regex=True)
                       .str.replace(r"\\n", "", regex=True)
                       .str.replace(r"\s+", " ", regex=True))
    # 研报 AI 63：每条研报独立编码（同事件多研报各自输入，二次训练按研报行）。
    # 不做事件级去重——保留全部研报行；保留原始行索引 row_idx 供训练时对齐。
    uni = samples.reset_index(names="row_idx")[["row_idx", "code", "event_date", "text"]]
    if limit:
        uni = uni.head(limit)
    log.info("待编码研报 %d 条", len(uni))

    t0 = time.time()
    model, tok = load_model(max_len)
    # 分块编码：每 chunk 条存一次临时 parquet（断点续跑），最后合并。
    # 全量 2.7 万行长文本一次性编码会 OOM（CPU）。
    tmp_dir = OUT_DIR / f"{task}_cls_{pool}_parts"
    tmp_dir.mkdir(exist_ok=True)
    done = 0
    n_chunks = (len(uni) + chunk - 1) // chunk
    for ci in range(n_chunks):
        part_path = tmp_dir / f"part_{ci:04d}.parquet"
        if part_path.exists() and not force:
            log.info("  chunk %d 已存在，跳过", ci)
            done += 1
            continue
        sub = uni.iloc[ci * chunk:(ci + 1) * chunk].copy()
        emb = encode_batch(model, tok, sub["text"].tolist(), max_len=max_len)
        cols = {f"cls_{i}": emb[:, i] for i in range(emb.shape[1])}
        # 注意：sub 保留了原始 index（可能非 0 起始），必须 reset 对齐
        # DataFrame(cols) 的 0..n-1 index，否则 concat axis=1 会按 index
        # 错位产生 NaN 行（实测 part_0001+ 全错位，2026-08-18 修复）
        sub = sub.reset_index(drop=True)
        part = pd.concat([sub[["row_idx", "code", "event_date"]],
                          pd.DataFrame(cols)], axis=1)
        part.to_parquet(part_path, compression="snappy")
        done += 1
        log.info("  chunk %d/%d 完成 (%d 行)", ci + 1, n_chunks, len(part))
    log.info("编码全部完成，合并 %d 个 chunk", done)

    parts = sorted(tmp_dir.glob("part_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(out_path, compression="snappy")
    log.info("编码已存: %s (%d 行)", out_path, len(df))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="fadt", choices=["sue", "fadt"])
    ap.add_argument("--pool", default="zz1000", choices=["hs300", "zz1000"])
    ap.add_argument("--max-len", type=int, default=500)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()


    run(args.task, args.pool, args.max_len, args.force, args.limit)
