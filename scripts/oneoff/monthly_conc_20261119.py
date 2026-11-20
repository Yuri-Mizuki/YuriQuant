"""monthly 档权重集中度改进（2026-11-19）
==========================================

背景：``monthly`` 是「逆波动 + 12% 目标波动 + 债金防御腿」。50 只池里有 1~2% 波动
的国债/短融 ETF 与 25% 的行业 ETF 混排，``w ∝ 1/σ`` 会在债券走强的年份把单只
国债 ETF 顶到权益腿的 80%+（2023-03-02 511260 占 83.9%）。本探针在同一选标、
同一频率下只改权重分配，比较：

    base（现状） / max_weight 0.50/0.40/0.35/0.30 / weight_vol_floor 0.10/0.15
    / cap40 + vf10

口径：50 只池、T+1 open、月内前 5 个交易日相位各跑一遍（相位平均）。
集中度指标按**权益腿归一化后**计算（DEF 那一半不参与），另记 DEF 占比。

用法::

    python -m scripts.oneoff.monthly_conc_20261119
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.portfolio.etf_champ_signals import PROFILES  # noqa: E402
from scripts.portfolio.etf_contest5_eval import two_month_windows  # noqa: E402
from scripts.portfolio.etf_rotation_panels import (  # noqa: E402
    build_defense,
    level_from_returns,
)
from scripts.portfolio.etf_weight_probe import load_env, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "etf_contest5"

VARIANTS: dict[str, dict] = {
    "base（现状）": {},
    "cap50": {"max_weight": 0.50},
    "cap40": {"max_weight": 0.40},
    "cap35": {"max_weight": 0.35},
    "cap30": {"max_weight": 0.30},
    "vf10": {"weight_vol_floor": 0.10},
    "vf15": {"weight_vol_floor": 0.15},
    "cap40+vf10": {"max_weight": 0.40, "weight_vol_floor": 0.10},
}


def _ann_seg(eq: pd.Series, lo: str, hi: str) -> float:
    s = eq.loc[lo:hi]
    if len(s) < 30:
        return float("nan")
    r = s.pct_change().dropna()
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)


def _conc(w: pd.DataFrame) -> dict:
    """权益腿（剔除 CASH/DEF）归一化后的集中度 + DEF 占比。"""
    held = w[[c for c in w.columns if c not in ("CASH", "DEF")]]
    live = held[held.abs().sum(axis=1) > 1e-9]
    max_norm, ratio, defs = [], [], []
    for d, row in live.iterrows():
        v = row[row > 0]
        if v.empty:
            continue
        nw = v / float(v.sum())
        max_norm.append(float(nw.max()))
        ratio.append(float(nw.max() / nw.min()))
        defs.append(float(w.at[d, "DEF"]) if "DEF" in w.columns else 0.0)
    return {
        "max_norm": float(np.mean(max_norm)),
        "ratio_p90": float(pd.Series(ratio).quantile(0.9)),
        "ratio_gt3": float((pd.Series(ratio) > 3.0).mean()),
        "def_share": float(np.mean(defs)),
    }


def _phases(close: pd.DatetimeIndex) -> list[set]:
    """月内第 k 个交易日（k=1..5）的调仓日集合。"""
    per = pd.Series(list(close), index=close).groupby(close.to_period("M"))
    out = []
    for k in range(5):
        rb = set()
        for _, g in per:
            v = list(g)
            if len(v) > k:
                rb.add(v[k])
        out.append(rb)
    return out


def main() -> int:
    print("加载环境…", flush=True)
    env = load_env()
    env["extra_close"] = pd.DataFrame(
        {"DEF": level_from_returns(build_defense(env["close"])["mix"])})
    phases = _phases(env["close"].index)
    rows = []
    for name, patch in VARIANTS.items():
        spec = {**PROFILES["monthly"], **patch}
        for k, rb in enumerate(phases, start=1):
            _r0, r1 = run(env, spec, rebalance_days=rb)
            c = _conc(r1["weights"])
            wt = two_month_windows(r1["equity"], env["mfs"])
            rows.append({
                "variant": name, "phase": k,
                "ann": r1["annual_return"], "sharpe": r1["sharpe"],
                "mdd": r1["max_drawdown"], "turnover": r1["avg_turnover"],
                "win2m": float(wt.mean()),
                "ann_early": _ann_seg(r1["equity"], "2020-01-01", "2023-12-31"),
                "ann_late": _ann_seg(r1["equity"], "2024-01-01", "2026-12-31"),
                **c,
            })
        print(f"  {name} 完成", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "monthly_conc.csv", index=False, encoding="utf-8-sig")
    agg = d.groupby("variant", sort=False).agg(
        ann=("ann", "mean"), sr=("sharpe", "mean"), mdd=("mdd", "mean"),
        mdd_worst=("mdd", "max"), win2m=("win2m", "mean"),
        turn=("turnover", "mean"), max_norm=("max_norm", "mean"),
        ratio_p90=("ratio_p90", "mean"), gt3=("ratio_gt3", "mean"),
        defs=("def_share", "mean"), early=("ann_early", "mean"),
        late=("ann_late", "mean"), ann_rng=("ann", lambda x: x.max() - x.min()),
    )
    agg["calmar"] = agg["ann"] / agg["mdd"]
    print("\n== monthly 权重分配（5 相位均值，50 只池，T+1 open）==")
    print(f"  {'方案':<12}{'年化':>8}{'SR':>6}{'MDD':>8}{'MDD最差':>9}{'Calmar':>8}"
          f"{'换手':>7}{'最大单只':>9}{'极差p90':>9}{'>3x占比':>9}"
          f"{'DEF':>7}{'前段':>8}{'后段':>8}{'相位极差':>9}")
    for name, x in agg.iterrows():
        print(f"  {name:<12}{x['ann']:>8.2%}{x['sr']:>6.2f}{x['mdd']:>8.2%}"
              f"{x['mdd_worst']:>9.2%}{x['calmar']:>8.2f}{x['turn']*12:>6.0%}"
              f"{x['max_norm']:>9.1%}{x['ratio_p90']:>9.1f}{x['gt3']:>9.1%}"
              f"{x['defs']:>7.0%}{x['early']:>8.2%}{x['late']:>8.2%}"
              f"{x['ann_rng']:>8.2%}")
    print(f"\n换手列为年化（月频 ×12）；明细已落盘 {OUT / 'monthly_conc.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
