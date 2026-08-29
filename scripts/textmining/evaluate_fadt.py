"""
FADT 因子评估（对齐华泰 AI 57）
===============================

与 evaluate_sue_txt 相同框架：覆盖度 / 5 层分层（月度调仓等权，基准中证500）/
RankIC / 与 2 日 AR 因子对比。因子文件为 fadt_factor_{model}_{pool}.parquet。

用法：
    python -m scripts.textmining.evaluate_fadt --model xgb --pool zz1000
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
from scripts.textmining.evaluate_sue_txt import (  # noqa: E402
    coverage,
    load_bench_ret,
    load_next_ret,
    rank_ic,
    stratified_backtest,
)

OUT_DIR = ROOT / "reports" / "textmining"


def load_factor(model: str, pool: str = "zz1000") -> pd.DataFrame:
    p = OUT_DIR / f"fadt_factor_{model}_{pool}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} 不存在，先跑 train_fadt")
    f = pd.read_parquet(p)
    f["date"] = pd.to_datetime(f.index.get_level_values("date"))
    f["code"] = f.index.get_level_values("code")
    return f.reset_index(drop=True)


def main(model: str = "xgb", pool: str = "zz1000"):
    f = load_factor(model, pool)
    print(f"== FADT forecast_adj_txt ({model}) 因子评估 ==")
    print(f"因子行数: {len(f)}, 覆盖 {f['code'].nunique()} 只, "
          f"{f['date'].min().date()} ~ {f['date'].max().date()}")

    cov = coverage(f)
    print(f"\n[覆盖度] 月均 {cov['n_stocks_mean'].iloc[0]:.0f} 只, "
          f"中位 {cov['n_stocks_median'].iloc[0]:.0f}, "
          f"区间 [{cov['min'].iloc[0]:.0f}, {cov['max'].iloc[0]:.0f}], "
          f"{cov['n_months'].iloc[0]} 个月")

    print("\n[分层回测 5 层]（月度调仓等权，下月收益）")
    sb = stratified_backtest(f, n_layers=5)
    print(sb.to_string(index=False))

    print("\n[RankIC]")
    ic = rank_ic(f)
    print(f"IC 均值 {ic['ic_mean'].iloc[0]:.4f} | ICIR {ic['icir'].iloc[0]:.2f} | "
          f"正占比 {ic['positive_ratio'].iloc[0]:.0%} | {ic['n_months'].iloc[0]} 个月")

    # 与 2 日 AR 因子对比（研报结论：forecast_adj_txt 强于传统因子）
    sp = OUT_DIR / f"fadt_samples_{pool}.parquet"
    if sp.exists():
        samples = pd.read_parquet(sp)
        samples["event_date"] = pd.to_datetime(samples["event_date"])
        samples["month"] = samples["event_date"].dt.to_period("M")
        ar_f = samples.groupby(["month", "code"])["ar"].mean().reset_index()
        ar_f["date"] = ar_f["month"].dt.to_timestamp() + pd.offsets.MonthEnd(0)
        m2 = ar_f.merge(f[["date", "code", "factor"]], on=["date", "code"], how="inner")
        if len(m2) > 50:
            corr = m2["factor"].corr(m2["ar"])
            print(f"\n[与 2 日 AR 因子相关] {corr:.4f}")
    else:
        corr = np.nan

    out = OUT_DIR / f"fadt_eval_{model}_{pool}.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"== FADT forecast_adj_txt ({model}) 因子评估 ==\n")
        fh.write(f"覆盖度: {cov.to_string()}\n")
        fh.write(f"分层: \n{sb.to_string()}\n")
        fh.write(f"RankIC: {ic.to_string()}\n")
        if not np.isnan(corr):
            fh.write(f"与AR相关: {corr:.4f}\n")
    print(f"\n评估已存: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgb")
    ap.add_argument("--pool", default="zz1000", choices=["hs300", "zz1000"])
    args = ap.parse_args()
    main(args.model, args.pool)
