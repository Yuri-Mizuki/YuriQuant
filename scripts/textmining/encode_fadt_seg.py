"""
分段编码（AI 63 消融：截断 vs 分段）
====================================

AI 63 扩展测试 1："截断 vs 分段（500/200 字）——均有效，分段样本量更多更稳"。
研报对长文本的处理：截断 N=500 vs 把文本切成多段分别编码。

本实现：把每条研报（标题+摘要，与 encode_fadt_bert 同一清洗）按句子边界
切成 ≤ SEG_MAX_TOKEN 的段（最多 MAX_SEG 段），逐段过 FinBERT 取 CLS，
段向量取均值作为该研报的编码（保持样本行数不变、与截断版可比；
"段即样本"的变体会改变训练集构成，不利于单变量对照）。

与截断版的差异只在编码方式，训练/标签/因子构建完全一致
（复用 train_fadt_bert.run，--cls-file 指向本产出）。

用法：
    python -m scripts.textmining.encode_fadt_seg --pool zz1000

产出：
    reports/textmining/fadt_cls_seg_{pool}.parquet （列名 cls_*，均值编码）
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

OUT_DIR = Out("fadt")
log = setup_logging("encode_seg")

SEG_MAX_TOKEN = 250   # AI 63 分段口径 ~200-500 字/段，取 250 token
MAX_SEG = 3           # 最多 3 段（覆盖 p95 长度，控制 CPU 时长）


def split_segments(text: str, tok, seg_max: int = SEG_MAX_TOKEN,
                   max_seg: int = MAX_SEG) -> list[str]:
    """按句子边界切段：贪心装填，段内 token 数 ≤ seg_max。"""
    sents = [s for s in re.split(r"(?<=[。！？；])", text) if s.strip()]
    segs: list[str] = []
    cur = ""
    for s in sents:
        cand = cur + s
        if cur and len(tok(cand)["input_ids"]) > seg_max:
            segs.append(cur)
            cur = s
        else:
            cur = cand
        if len(segs) >= max_seg:
            break
    if cur and len(segs) < max_seg:
        segs.append(cur)
    elif len(segs) >= max_seg and cur:
        pass  # 已满 max_seg 段，剩余截断（研报亦截断处理尾部）
    return segs or [text]


def run(pool: str = "zz1000", batch_size: int = 32, threads: int = 22,
        force: bool = False, chunk: int = 2000):
    out_path = OUT_DIR / f"fadt_cls_seg_{pool}.parquet"
    if out_path.exists() and not force:
        log.info("已存在: %s", out_path)
        return
    import torch
    torch.set_num_threads(threads)
    from transformers import AutoModel, AutoTokenizer
    try:
        from config import Config
        d = Config.get().get("textmining", {}).get("bert_model_dir")
    except Exception:
        d = None
    model_dir = str(d).replace("//", "/") if d else r"E:/data/models/finbert_tone_chinese"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).eval()

    samples = pd.read_parquet(OUT_DIR / f"fadt_samples_{pool}.parquet")
    samples["event_date"] = pd.to_datetime(samples["event_date"]).dt.normalize()
    text = (samples["title"].fillna("") + " " + samples["summary"].fillna(""))
    text = (text.str.replace(r"\\r###", "。", regex=True)
            .str.replace(r"\\r", "", regex=True)
            .str.replace(r"\\n", "", regex=True)
            .str.replace(r"\s+", " ", regex=True))
    uni = samples.reset_index(names="row_idx")[["row_idx", "code", "event_date"]]
    uni["text"] = text
    log.info("样本 %d 行，切段（≤%d token × %d 段）...", len(uni),
             SEG_MAX_TOKEN, MAX_SEG)

    # 切段展开：一行研报 → 1..MAX_SEG 段
    seg_rows = []
    for i, t in enumerate(uni["text"].tolist()):
        segs = split_segments(t, tok)
        for si, s in enumerate(segs):
            seg_rows.append((i, si, s))
    seg_df = pd.DataFrame(seg_rows, columns=["row_pos", "seg_id", "text"])
    log.info("切段完成：%d 行 → %d 段（均值 %.2f 段/行）",
             len(uni), len(seg_df), len(seg_df) / len(uni))

    tmp_dir = OUT_DIR / f"fadt_cls_seg_{pool}_parts"
    tmp_dir.mkdir(exist_ok=True)
    n_chunks = (len(seg_df) + chunk - 1) // chunk
    for ci in range(n_chunks):
        part_path = tmp_dir / f"part_{ci:04d}.parquet"
        if part_path.exists():
            continue
        sub = seg_df.iloc[ci * chunk:(ci + 1) * chunk].reset_index(drop=True)
        lens = np.array([len(tok(t)["input_ids"]) for t in sub["text"]])
        order = np.argsort(lens)
        emb = np.zeros((len(sub), 768), dtype=np.float32)
        for i in range(0, len(order), batch_size):
            sel = order[i:i + batch_size]
            enc = tok([sub["text"].iloc[j] for j in sel], padding=True,
                      truncation=True, max_length=SEG_MAX_TOKEN, return_tensors="pt")
            with torch.no_grad():
                h = model(**enc).last_hidden_state[:, 0, :]
            emb[sel] = h.numpy()
        part = sub[["row_pos", "seg_id"]].copy()
        for k in range(768):
            part[f"cls_{k}"] = emb[:, k]
        part.to_parquet(part_path, compression="snappy")
        done = sum(1 for _ in tmp_dir.glob("part_*.parquet"))
        log.info("  chunk %d/%d 完成（%d/%d）", ci + 1, n_chunks, done, n_chunks)

    parts = sorted(tmp_dir.glob("part_*.parquet"))
    seg_emb = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    mean_cols = [f"cls_{k}" for k in range(768)]
    agg = seg_emb.groupby("row_pos")[mean_cols].mean()
    assert len(agg) == len(uni)
    out_df = pd.concat([uni.drop(columns=["text"]).reset_index(drop=True),
                        agg.reset_index(drop=True)], axis=1)
    out_df.to_parquet(out_path, compression="snappy")
    log.info("分段均值编码已存: %s (%d 行)", out_path, len(out_df))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="zz1000", choices=["hs300", "zz1000"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=22)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(args.pool, args.batch_size, args.threads, args.force)
