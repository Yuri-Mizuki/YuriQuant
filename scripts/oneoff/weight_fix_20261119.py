"""balanced 档权重方法改进（2026-11-19）
=========================================

背景：`balanced`（Top5 · 周频 · HRP）在 50 只池、60 日协方差下给出极端权重——
实盘信号板出现 69.5% / 0.18% / 0.17%（后者在 1000 万账户里各约 7000 元，只贡献佣金）。
本探针在**同一选标、同一频率**下只换权重规则，比较：

    hrp（现状） / hrp_shrunk / erc / inverse_vol / equal
    + 三个加 ``max_weight=0.35`` 的版本

口径：50 只池、T+1 open、每周前 5 个交易日相位各跑一遍（相位平均），
另记「单只最大权重均值」与「<0.5% 碎仓占比」两个集中度指标。

用法::

    python -m scripts.oneoff.weight_fix_20261119
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.portfolio.etf_champ_signals import PROFILES  # noqa: E402
from scripts.portfolio.etf_contest5_eval import two_month_windows  # noqa: E402
from scripts.portfolio.etf_weight_probe import load_env, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "etf_contest5"

VARIANTS: dict[str, dict] = {
    "hrp（现状）": {},
    "hrp_shrunk": {"weighting": "hrp_shrunk"},
    "erc": {"weighting": "erc"},
    "inverse_vol": {"weighting": "inverse_vol"},
    "equal": {"weighting": "equal"},
    "hrp+cap35": {"max_weight": 0.35},
    "erc+cap35": {"weighting": "erc", "max_weight": 0.35},
    "iv+cap35": {"weighting": "inverse_vol", "max_weight": 0.35},
}


def _ann_seg(eq: pd.Series, lo: str, hi: str) -> float:
    s = eq.loc[lo:hi]
    if len(s) < 30:
        return float("nan")
    r = s.pct_change().dropna()
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)


def _conc(w: pd.DataFrame) -> tuple[float, float]:
    """返回（单只最大权重均值, <0.5% 碎仓占比均值）。只算多头腿。"""
    held = w[[c for c in w.columns if c not in ("CASH", "DEF")]]
    live = held[held.abs().sum(axis=1) > 1e-9]
    if live.empty:
        return float("nan"), float("nan")
    mx, dust = [], []
    for _, row in live.iterrows():
        v = row[row > 0]
        if v.empty:
            continue
        mx.append(float(v.max()))
        dust.append(float((v < 0.005).mean()))
    return float(np.mean(mx)), float(np.mean(dust))


def _phases(close: pd.DatetimeIndex) -> list[set]:
    per = pd.Series(list(close), index=close).groupby(close.to_period("W"))
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
    phases = _phases(env["close"].index)
    rows = []
    for name, patch in VARIANTS.items():
        spec = {**PROFILES["balanced"], **patch}
        for k, rb in enumerate(phases, start=1):
            _r0, r1 = run(env, spec, rebalance_days=rb)
            mx, dust = _conc(r1["weights"])
            wt = two_month_windows(r1["equity"], env["mfs"])
            rows.append({
                "variant": name, "phase": k,
                "ann": r1["annual_return"], "sharpe": r1["sharpe"],
                "mdd": r1["max_drawdown"], "turnover": r1["avg_turnover"],
                "win2m": float(wt.mean()), "max_w": mx, "dust": dust,
                "ann_early": _ann_seg(r1["equity"], "2020-01-01", "2023-12-31"),
                "ann_late": _ann_seg(r1["equity"], "2024-01-01", "2026-12-31"),
            })
        print(f"  {name} 完成", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "weight_fix_balanced.csv", index=False, encoding="utf-8-sig")
    agg = d.groupby("variant", sort=False).agg(
        ann=("ann", "mean"), sr=("sharpe", "mean"), mdd=("mdd", "mean"),
        mdd_worst=("mdd", "max"), win2m=("win2m", "mean"),
        turn=("turnover", "mean"), max_w=("max_w", "mean"),
        dust=("dust", "mean"),
        ann_rng=("ann", lambda x: x.max() - x.min()),
        early=("ann_early", "mean"), late=("ann_late", "mean"),
    )
    agg["calmar"] = agg["ann"] / agg["mdd"]
    print("\n== balanced 权重规则（5 相位均值，50 只池，T+1 open）==")
    hdr = (f"  {'规则':<12}{'年化':>8}{'SR':>6}{'MDD':>8}{'MDD最差':>9}"
           f"{'Calmar':>8}{'换手':>7}{'最大单只':>9}{'碎仓':>7}{'前段':>8}{'后段':>8}")
    print(hdr)
    for name, x in agg.iterrows():
        print(f"  {name:<12}{x['ann']:>8.2%}{x['sr']:>6.2f}{x['mdd']:>8.2%}"
              f"{x['mdd_worst']:>9.2%}{x['calmar']:>8.2f}{x['turn']*52:>6.0%}"
              f"{x['max_w']:>9.1%}{x['dust']:>7.1%}{x['early']:>8.2%}{x['late']:>8.2%}")
    print(f"\n明细已落盘 {OUT / 'weight_fix_balanced.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
