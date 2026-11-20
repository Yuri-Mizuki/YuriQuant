"""动量回看窗口扫描（2026-11-19）
====================================

问题：生产两档都用 12-1 动量（monthly = 12-1 + 拥挤硬过滤，balanced = 12-1 + 软惩罚
+ pre10）。历史证据只有「缩短窗口更差」的周频 IC（README §7.1），从未做组合回测，
也没测过更长窗口与 12-0。本探针在同一套流水线里只替换动量的回看窗口：

    6-1 / 9-1 / 12-1（现状）/ 12-0 / 18-1 / 24-1

口径：50 只池、T+1 open、5 相位均值。24-1 在 2020 年初因暖机不足缺 2 个月，
额外给所有窗口一致的「共同窗」指标（从 24-1 的首个有效日算起），避免首尾错位。

用法::

    python -m scripts.oneoff.signal_lookback_20261119
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.portfolio.etf_champ_signals import PROFILES  # noqa: E402
from scripts.portfolio.etf_rotation_panels import (  # noqa: E402
    build_defense,
    level_from_returns,
)
from scripts.portfolio.etf_weight_probe import load_env, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "etf_contest5"

# (标签, 月数, 跳过最近交易日数)。1 个月 ≈ 21 个交易日。
WINDOWS = [("6-1", 6, 21), ("9-1", 9, 21), ("12-1", 12, 21),
           ("12-0", 12, 0), ("18-1", 18, 21), ("24-1", 24, 21)]


def _mom(close: pd.DataFrame, months: int, skip: int) -> pd.DataFrame:
    n = months * 21
    return close.shift(skip) / close.shift(n) - 1.0


def _alt_panels(panels: dict, close: pd.DataFrame, months: int, skip: int) -> dict:
    """只替换动量族面板，拥挤度 / pre10 / 掩码全部沿用。"""
    p = dict(panels)
    mom = _mom(close, months, skip)
    crowd = panels["crowd"]
    over = ((crowd - 0.80) / 0.20).clip(lower=0.0, upper=1.0).fillna(0.0)
    soft = mom - 0.10 * over
    p["mom12_1"] = mom
    p["mom_crowd_f20"] = mom.where(crowd <= 0.80)
    p["mom12_1_crowdsoft_pre10"] = soft.where(panels["mom20"] <= 0.10)
    return p


def _phase_days(idx: pd.DatetimeIndex, period: str) -> list[set]:
    s = pd.Series(list(idx), index=idx)
    out = []
    for k in range(5):
        rb = set()
        for _, g in s.groupby(s.index.to_period(period)):
            v = list(g)
            if len(v) > k:
                rb.add(v[k])
        out.append(rb)
    return out


def _metrics(eq: pd.Series) -> tuple[float, float, float]:
    r = eq.pct_change().dropna()
    if len(r) < 30:
        return float("nan"), float("nan"), float("nan")
    ann = float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)
    sr = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")
    mdd = float((eq / eq.cummax() - 1.0).min())
    return ann, sr, mdd


def _weekly_ic(mom: pd.DataFrame, close: pd.DataFrame, mask: pd.DataFrame):
    idx = close.index
    s = pd.Series(list(idx), index=idx)
    rb = pd.DatetimeIndex([list(g)[0] for _, g in s.groupby(s.index.to_period("W"))])
    ics = []
    for t, tn in zip(rb[:-1], rb[1:]):
        f = mom.loc[t].where(mask.loc[t])
        r = close.loc[tn] / close.loc[t] - 1.0
        ok = f.notna() & r.notna()
        if int(ok.sum()) < 8:
            continue
        ics.append(f[ok].corr(r[ok], method="spearman"))
    a = pd.Series(ics).dropna()
    if len(a) < 10:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.mean() / (a.std() / np.sqrt(len(a))))


def main() -> int:
    print("加载环境…", flush=True)
    env = load_env()
    ex = pd.DataFrame({"DEF": level_from_returns(build_defense(env["close"])["mix"])})
    env_m = {**env, "extra_close": ex}
    ph = {"balanced": _phase_days(env["close"].index, "W"),
          "monthly": _phase_days(env["close"].index, "M")}
    pnls = {}
    common_lo = None
    for prof in ("balanced", "monthly"):
        for tag, months, skip in WINDOWS:
            pnl = _alt_panels(env["panels"], env["close"], months, skip)
            pnls[(prof, tag)] = pnl
            first = pnl["mom12_1"].notna().sum(axis=1).ge(8)
            d0 = first[first].index[0] if first.any() else env["close"].index[0]
            common_lo = d0 if common_lo is None else max(common_lo, d0)
    print(f"共同窗起点 {pd.Timestamp(common_lo).date()}（24-1 的暖机下限）", flush=True)
    rows, eqs = [], {}
    for prof in ("balanced", "monthly"):
        for tag, _months, _skip in WINDOWS:
            e = env_m if prof == "monthly" else env
            spec = dict(PROFILES[prof])
            pnl = pnls[(prof, tag)]
            for k, rb in enumerate(ph[prof], start=1):
                r0, r1 = run({**e, "panels": pnl}, spec, rebalance_days=rb)
                _ = r0
                eq = r1["equity"]
                sub = eq.loc[eq.index >= max(common_lo, eq.index[0])]
                cann, csr, cmdd = _metrics(sub) if len(sub) > 100 else (np.nan,) * 3
                rows.append({"profile": prof, "window": tag, "phase": k,
                             "ann": r1["annual_return"], "sharpe": r1["sharpe"],
                             "mdd": r1["max_drawdown"],
                             "turnover": r1["avg_turnover"],
                             "ann_common": cann, "sr_common": csr,
                             "mdd_common": cmdd})
                eqs[(prof, tag, k)] = eq
            print(f"  {prof} {tag} 完成", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "signal_lookback.csv", index=False, encoding="utf-8-sig")
    ic_rows = []
    for prof in ("balanced", "monthly"):
        for tag, months, skip in WINDOWS:
            m = _mom(env["close"], months, skip)
            ic, t = _weekly_ic(m, env["close"], env["panels"]["mask"])
            ic_rows.append({"profile": prof, "window": tag, "ic": ic, "ic_t": t})
    ic = pd.DataFrame(ic_rows)

    print(f"\n== 动量回看窗口（5 相位均值，50 只池，T+1 open；共同窗起于 "
          f"{pd.Timestamp(common_lo).date()}）==")
    for prof in ("balanced", "monthly"):
        print(f"\n-- {prof} --")
        print(f"  {'窗口':<6}{'年化':>8}{'SR':>6}{'MDD':>8}{'Calmar':>8}{'换手':>7}"
              f"{'相位极差':>9} | {'共同窗年化':>10}{'SR':>7}{'MDD':>8} | {'周IC':>7}{'t':>6}")
        for tag, _, _ in WINDOWS:
            g = d[(d.profile == prof) & (d.window == tag)]
            ann, sr, mdd = g.ann.mean(), g.sharpe.mean(), g.mdd.mean()
            x = ic[(ic.profile == prof) & (ic.window == tag)].iloc[0]
            print(f"  {tag:<6}{ann:>8.2%}{sr:>6.2f}{mdd:>8.2%}"
                  f"{ann / mdd:>8.2f}{g.turnover.mean()*12 if prof == 'monthly' else g.turnover.mean():>7.0%}"
                  f"{g.ann.max() - g.ann.min():>8.2%} | "
                  f"{g.ann_common.mean():>10.2%}{g.sr_common.mean():>7.2f}"
                  f"{g.mdd_common.mean():>8.2%} | {x.ic:>7.3f}{x.ic_t:>6.1f}")
    print(f"\n明细 {OUT / 'signal_lookback.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
