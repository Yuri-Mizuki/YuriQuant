"""
Pooler（全连接层）编码推导（AI 63 消融：CLS vs 全连接层）
========================================================

AI 63 扩展测试 3 对比 "CLS 层编码 vs 全连接层编码"。全连接层（BERT pooler）
输出 = tanh(h_CLS · W_pᵀ + b_p)，其中 h_CLS 是 CLS token 的最后一层隐藏向量
——正是 fadt_cls_{pool}.parquet 里存的内容。因此 pooler 编码可由现有 CLS
文件 **零推理成本** 线性变换得到（本脚本），并与 live 前向抽样核对一致性。

用法：
    python -m scripts.textmining.derive_pooler_encoding --task fadt --pool zz1000

产出：
    reports/textmining/{task}_cls_pooler_{pool}.parquet （列名 cls_*，供训练复用）
"""
from __future__ import annotations

import argparse
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
log = setup_logging("pooler_enc")


def _pooler_weights() -> tuple[np.ndarray, np.ndarray]:
    from config import Config
    try:
        d = Config.get().get("textmining", {}).get("bert_model_dir")
    except Exception:
        d = None
    model_dir = str(d).replace("//", "/") if d else r"E:/data/models/finbert_tone_chinese"
    from safetensors import safe_open
    with safe_open(str(Path(model_dir) / "model.safetensors"), framework="pt") as fh:
        w = fh.get_tensor("bert.pooler.dense.weight").numpy()
        b = fh.get_tensor("bert.pooler.dense.bias").numpy()
    return w, b


def run(task: str = "fadt", pool: str = "zz1000", check_live: bool = True):
    src = OUT_DIR / f"{task}_cls_{pool}.parquet"
    out = OUT_DIR / f"{task}_cls_pooler_{pool}.parquet"
    if out.exists():
        log.info("已存在: %s", out)
        return
    cls = pd.read_parquet(src)
    X = cls[[c for c in cls.columns if c.startswith("cls_")]].values
    W, b = _pooler_weights()
    P = np.tanh(X @ W.T + b)
    meta = cls[["row_idx", "code", "event_date"]]
    pd.concat([meta, pd.DataFrame(P, columns=[f"cls_{i}" for i in range(P.shape[1])])],
              axis=1).to_parquet(out, compression="snappy")
    log.info("pooler 编码已存: %s (%d 行)", out, len(cls))

    if check_live:
        # 抽 4 条与 live 前向核对（tokenizer 截断 max_len=500，与 encode 一致）

        import torch
        from transformers import AutoModel, AutoTokenizer

        from config import Config
        try:
            d = Config.get().get("textmining", {}).get("bert_model_dir")
        except Exception:
            d = None
        model_dir = str(d).replace("//", "/") if d else r"E:/data/models/finbert_tone_chinese"
        sp = OUT_DIR / f"{task}_samples_{pool}.parquet"
        samples = pd.read_parquet(sp).reset_index(names="row_idx").head(4)
        texts = (samples["title"].fillna("") + " " + samples["summary"].fillna(""))
        texts = (texts.str.replace(r"\\r###", "。", regex=True)
                 .str.replace(r"\\r", "", regex=True)
                 .str.replace(r"\\n", "", regex=True)
                 .str.replace(r"\s+", " ", regex=True))
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir).eval()
        enc = tok(texts.tolist(), padding=True, truncation=True, max_length=500,
                  return_tensors="pt")
        with torch.no_grad():
            o = model(**enc)
        live_pooler = o.pooler_output.numpy()
        mine = P[:4]
        diff = np.abs(live_pooler - mine).max()
        log.info("live pooler vs 推导 pooler 最大绝对差: %.2e（应 ~1e-6）", diff)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="fadt", choices=["sue", "fadt"])
    ap.add_argument("--pool", default="zz1000", choices=["hs300", "zz1000"])
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    run(args.task, args.pool, check_live=not args.no_check)
