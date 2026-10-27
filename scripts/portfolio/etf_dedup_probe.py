"""
ETF 轮动·选标层动态去重 A/B（2026-10-27）
=========================================

池子有 13 对 >0.95 的"双胞胎"，Top3 组合里 27.9% 的持仓日同时押两只同质标的。
本探针在**选标层**做动态去重：按动量从高到低贪心，若候选与已选标的的
``dedup_window`` 日相关 ≥ ``dedup_corr`` 就跳过——只用当日及之前的收益，无前视。

三套方案各测三档阈值：
  - 比赛 A（Top3 周频满仓）、比赛 B（Top2 双周满仓）；
  - 长线 champ（Top5 月频、逆波动、15% 目标波动、债金防御腿）。

指标：全样本年化/Sharpe/MDD，2 月窗口中位/p10/最差，以及诊断量
（双胞胎共持天数、同类占比）。成本统一按实际费率 1bp + 5bp。

用法:
    python -m scripts.portfolio.etf_dedup_probe
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.cache import DataCache
from data.datasource import create_datasource
from scripts.portfolio.etf_champ_signals import level_from_returns
from scripts.portfolio.etf_contest_probe import two_month_table
from scripts.portfolio.etf_rotation_p2 import (
    build_defense,
    build_extra_panels,
    month_first_dates,
)
from scripts.portfolio.etf_rotation_probe import (
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "etf_dedup"

COST = {"commission_bp": 1.0, "slippage_bp": 5.0}
CONTEST_A = {"signal": "mom_crowd_f20", "freq": "W", "top_k": 3,
             "absolute": True, "weighting": "equal", **COST}
CONTEST_B = {"signal": "mom_crowd_f20", "freq": "W", "top_k": 2, "every_n": 10,
             "absolute": True, "weighting": "equal", **COST}
CHAMP = {"signal": "mom_crowd_f20", "freq": "M", "top_k": 5, "absolute": True,
         "weighting": "inverse_vol", "target_vol": 0.15, "defense": "mix", **COST}

SPECS = [
    {**CONTEST_A, "name": "A_nodd"},
    {**CONTEST_A, "name": "A_dd85", "dedup_corr": 0.85},
    {**CONTEST_A, "name": "A_dd90", "dedup_corr": 0.90},
    {**CONTEST_A, "name": "A_dd95", "dedup_corr": 0.95},
    {**CONTEST_B, "name": "B_nodd"},
    {**CONTEST_B, "name": "B_dd90", "dedup_corr": 0.90},
    {**CHAMP, "name": "champ_nodd"},
    {**CHAMP, "name": "champ_dd90", "dedup_corr": 0.90},
]


def main() -> None:
    cache = DataCache(create_datasource())
    uni, close, amount, bench_raw = load_bundle(cache, WARMUP_BEGIN, None)
    raw = (cache.read_daily("etf")["close"].unstack("code")
           .reindex(index=close.index, columns=close.columns))
    panels = build_extra_panels(close, build_panels(close, amount), raw)
    extra = pd.DataFrame({"DEF": level_from_returns(build_defense(close)["mix"])})
    bench = (1.0 + bench_raw.fillna(0.0)).cumprod()
    mfs = month_first_dates(close.index, 20200101,
                            int(close.index.max().strftime("%Y%m%d")))
    c2 = close.pct_change(fill_method=None).loc["2024-01-01":].corr()
    codes = list(c2.columns)
    arr = c2.to_numpy()
    dup = [(codes[i], codes[j]) for i, j in zip(*np.where(arr > 0.95)) if i < j]
    cat_of = {c: uni.category(c) for c in close.columns}

    def diag(w: pd.DataFrame) -> tuple[float, float]:
        n = sum(1 for _, row in w.iterrows()
                if any(row.get(a, 0) > 0 and row.get(b, 0) > 0 for a, b in dup))
        share = []
        for _, row in w.iterrows():
            live = [c for c in row.index if row[c] > 0 and c not in ("DEF", "CASH")]
            if len(live) > 1:
                share.append(float(pd.Series([cat_of.get(c, "?") for c in live])
                                   .value_counts(normalize=True).iloc[0]))
        return 100.0 * n / len(w), float(np.mean(share)) if share else np.nan

    rows = []
    for spec in SPECS:
        rb = None
        if spec.get("every_n"):
            idx = close.index[close.index >= pd.Timestamp("20200101")]
            rb = set(idx[:: int(spec["every_n"])])
        r = run_variant((uni, close, amount, bench_raw), panels, spec,
                        20200101, None, extra_close=extra, rebalance_days=rb)
        tab = two_month_table(r["equity"], bench, mfs)
        dup_pct, cat_share = diag(r["weights"])
        rows.append({
            "name": spec["name"], "dedup": spec.get("dedup_corr"),
            "ann": r["annual_return"], "sharpe": r["sharpe"],
            "mdd": r["max_drawdown"], "turnover": r["avg_turnover"],
            "2m_median": tab["ret"].median(), "2m_p10": tab["ret"].quantile(0.10),
            "2m_worst": tab["ret"].min(),
            "dup_days_pct": dup_pct, "max_cat_share": cat_share,
        })
        print(f"{spec['name']:<10} 年化 {r['annual_return']:6.2%} "
              f"Sharpe {r['sharpe']:5.2f} MDD {r['max_drawdown']:7.2%} | "
              f"2M 中位 {tab['ret'].median():+6.2%} p10 {tab['ret'].quantile(.1):+7.2%} "
              f"最差 {tab['ret'].min():+7.2%} | 双胞胎日 {dup_pct:4.1f}% "
              f"同类占比 {cat_share:.2f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "dedup_summary.csv", index=False,
                              encoding="utf-8-sig")
    print(f"\n写入 {OUT / 'dedup_summary.csv'}")


if __name__ == "__main__":
    main()
