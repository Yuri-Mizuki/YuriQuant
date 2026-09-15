"""
原始语气得分直接作为 FADT 因子（AI 63 扩展测试 5 的无微调对照）
================================================================

AI 63 测试 5："仅 FinBERT 微调（直接学 AR 标签）无效"。本地无微调版本：
**跳过 XGBoost 二次训练**，直接把 FinBERT 分类头的语气得分
（P(pos) − P(neg)，AI 41 打分产物）作为 FADT 调整事件的因子值，
聚合口径与 FADT 完全一致（月末回溯 3 个月、个股全部事件得分均值、无衰减）。

回答的问题：XGBoost 二次训练（fadt_bert）是否提取了"超越语气本身"的
信息——若 rawtone ≈ fadt_bert，说明该管线的增量主要是语气重新包装。

用法：
    python -m scripts.textmining.build_rawtone_factor --pool zz1000

产出：
    reports/textmining/fadt/features/fadt_factor_rawtone_xgb_{pool}.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.cli_common import setup_logging  # noqa: E402
from scripts.textmining._paths import TM, Out  # noqa: E402
from scripts.textmining.build_sue_txt_samples import _to_naive  # noqa: E402
from scripts.textmining.train_fadt import build_factor_from_pred  # noqa: E402

OUT_DIR = Out("fadt")
log = setup_logging("rawtone")


def run(pool: str = "zz1000"):
    samples = pd.read_parquet(OUT_DIR / f"fadt_samples_{pool}.parquet")
    samples["event_date"] = pd.to_datetime(samples["event_date"]).dt.normalize()
    samples["title_head"] = samples["title"].fillna("").str.slice(0, 60)

    senti_dir = TM / "senti" / "features"
    sc = pd.read_parquet(senti_dir / f"senti_scores_{pool}.parquet")
    sc["date"] = _to_naive(sc["date"]).dt.normalize()
    sc["tone"] = sc["p_pos"] - sc["p_neg"]

    m = samples.merge(sc[["code", "date", "title_head", "tone"]],
                      left_on=["code", "event_date", "title_head"],
                      right_on=["code", "date", "title_head"], how="inner")
    dup = m.duplicated(["code", "event_date", "title_head"]).sum()
    if dup:
        m = m.drop_duplicates(["code", "event_date", "title_head"])
    log.info("FADT 事件 %d 行中命中语气得分 %d 行（无得分为 2020-07 前事件）",
             len(samples), len(m))

    pred = m.rename(columns={"event_date": "event_date"})[
        ["code", "event_date", "tone"]].rename(columns={"tone": "sue0"})
    factor = build_factor_from_pred(pred, "rawtone_xgb", pool)
    out = OUT_DIR / f"fadt_factor_rawtone_xgb_{pool}.parquet"
    factor.to_parquet(out, compression="snappy")
    log.info("rawtone 因子面板: %d 行, 覆盖 %d 只 → %s",
             len(factor), factor.index.get_level_values("code").nunique(), out)
    return factor


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="zz1000", choices=["hs300", "zz1000", "all_a"])
    args = ap.parse_args()
    run(args.pool)
