"""ETF 磐石（``champ``）· ``target_vol`` 扫描 + 相位独立信息量（n_eff）实测。

两件事一个脚本，因为它们共用一个环境：

## A. ``target_vol`` 是不是一个真正的优化维度？

``reports/etf_contest5/README.md`` §5.1.5 把 ``target_vol`` 列为唯一没扫过的旋钮。
它的重要性在于**余量的落点**：``strategy/etf_rotation.py`` 里

    scale = min(1.0, target_vol / vol.mean());  w *= scale
    w[cash_code] = 1 - w.sum()          # 磐石 cash_code == "DEF"

⇒ 降杠杆的余量落在 **`DEF`（50% 国债 + 50% 黄金的合成腿，有正收益）**，不是零收益现金。
所以我此前反复引用的「降杠杆是免费前沿、Calmar 是降杠杆不变量」在这里**可能不成立**：
把一个赚钱的低波腿加进来，Calmar 是可能变好的。本脚本就是把这条验证掉。

判据（事前定）：若 Calmar 随 tvol 显著上升 ⇒ 这是一个真优化维度；
若 Calmar 在噪声内平坦 ⇒ 它只是「选风险档位」，不是优化。

## B. 「多相位重测」到底值多少独立样本？

同屏输出 5 个相位净值的两两相关，算有效独立数

    n_eff = k / (1 + (k-1) * rho)

三锋实测 rho=0.825（五子账户摊平，§七 #32）⇒ n_eff ≈ 1.16。
这决定了「把判否的实验拿到 5 个相位上重测」能提高多少检验力。

## 用法

    python -m scripts.portfolio.etf_champ_tvol_scan
    python -m scripts.portfolio.etf_champ_tvol_scan --tvols 0.10,0.15,0.20
    python -m scripts.portfolio.etf_champ_tvol_scan --from-csv

产物：``reports/etf_champ/tvol_scan.csv``。
环境：**系统 Python 3.12**（``.venv`` 缺 ``AmazingData``）。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.portfolio.etf_champ_phase_matrix import (
    build_env,
    month_phase_rebalance_days,
    run_champ,
)
from scripts.portfolio.etf_champ_phase_robust import _monthly, _roll_mdd
from scripts.portfolio.etf_champ_signals import PROFILES

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "reports/etf_champ/tvol_scan.csv"

PHASES = (1, 2, 3, 4, 5)

#: 定版 target_vol。0.25 附近起 ``scale = min(1, tvol/vol.mean())`` 触顶（vol.mean ≈ 0.248）
#: ⇒ 再往上加完全等价于满仓，故上限取 0.25。
DEFAULT_TVOLS = (0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.25)
BASE_TVOL = 0.15


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    agg = d.groupby("tvol", sort=True).agg(
        ann_mean=("ann", "mean"), ann_min=("ann", "min"), ann_max=("ann", "max"),
        sr_mean=("sharpe", "mean"), sr_min=("sharpe", "min"),
        mdd_mean=("mdd", "mean"), mdd_worst=("mdd", "max"),
        turnover=("turnover", "mean"),
        gross=("gross", "median"), defw=("defw", "median"),
    ).reset_index()
    agg["calmar"] = agg["ann_mean"] / agg["mdd_mean"].abs()
    agg["ann_range"] = (agg["ann_max"] - agg["ann_min"]) * 100
    return agg


def print_summary(agg: pd.DataFrame) -> None:
    print("\n== 磐石 · target_vol 扫描（相位平均，T+1 open，2020 起）==")
    print(f"  {'tvol':>6}{'年化均':>9}{'区间':>16}{'极差':>7}{'SR均':>7}"
          f"{'MDD均':>8}{'MDD最差':>9}{'Calmar':>8}{'换手':>7}"
          f"{'多头仓':>8}{'DEF仓':>7}")
    for _, x in agg.iterrows():
        print(f"  {x['tvol']:>6.2f}{x['ann_mean']:>9.2%}"
              f"{x['ann_min']:>7.2%}~{x['ann_max']:>7.2%}"
              f"{x['ann_range']:>6.1f}pp{x['sr_mean']:>7.2f}"
              f"{x['mdd_mean']:>8.2%}{x['mdd_worst']:>9.2%}{x['calmar']:>8.2f}"
              f"{x['turnover']:>7.1%}{x['gross']:>8.3f}{x['defw']:>7.3f}")

    b = agg[np.isclose(agg.tvol, BASE_TVOL)]
    if not len(b):
        return
    b = b.iloc[0]
    print(f"\n  定版 tvol={BASE_TVOL}：年化 {b['ann_mean']:.2%} / SR {b['sr_mean']:.2f} / "
          f"MDD {b['mdd_mean']:.2%} / Calmar {b['calmar']:.2f}")
    print(f"  {'tvol':>6}{'Δ年化':>10}{'ΔMDD':>9}{'ΔCalmar':>10}{'ΔSR':>8}")
    for _, x in agg.iterrows():
        if np.isclose(x["tvol"], BASE_TVOL):
            continue
        print(f"  {x['tvol']:>6.2f}{(x['ann_mean'] - b['ann_mean']) * 100:>+9.2f}pp"
              f"{(x['mdd_mean'] - b['mdd_mean']) * 100:>+8.2f}pp"
              f"{x['calmar'] - b['calmar']:>+10.2f}"
              f"{x['sr_mean'] - b['sr_mean']:>+8.2f}")

    # 判据：Calmar 有单调趋势 ⇒ 真维度；否则只是风险档位
    c = agg["calmar"].to_numpy()
    t = agg["tvol"].to_numpy()
    slope = float(np.polyfit(t, c, 1)[0])
    print(f"\n  Calmar 对 tvol 的线性斜率 = {slope:+.2f}/单位tvol，"
          f"极差 {c.max() - c.min():.2f}（{c.min():.2f}~{c.max():.2f}）")
    if c.max() - c.min() > 0.10 and abs(slope) > 0.5:
        best = agg.loc[agg["calmar"].idxmax()]
        print(f"  🚨 Calmar 非常数 ⇒ target_vol 是**真维度**，最优 tvol={best['tvol']:.2f}"
              f"（Calmar {best['calmar']:.2f} vs 定版 {b['calmar']:.2f}）")
    else:
        print("  ✅ Calmar 在噪声内平坦 ⇒ target_vol 只是「选风险档位」，不是优化维度")


def print_judge(eq: dict, tvols: tuple[float, ...], base: float) -> None:
    print(f"\n== ① 月度配对 t（vs 定版 tvol={base}，同相位）==")
    print(f"  {'tvol':>6}{'k':>3}{'Δ月均值':>10}{'t':>7}{'胜率':>8}")
    for t in tvols:
        if np.isclose(t, base):
            continue
        for k in PHASES:
            a, b = _monthly(eq[(t, k)]), _monthly(eq[(base, k)])
            j = a.index.intersection(b.index)
            dv = (a[j] - b[j]).dropna()
            if len(dv) < 10:
                continue
            tt = dv.mean() / dv.std(ddof=1) * np.sqrt(len(dv))
            print(f"  {t:>6.2f}{k:>3}{dv.mean() * 100:>+9.2f}%{tt:>7.2f}"
                  f"{(dv > 0).mean():>8.1%}")

    print(f"\n== ② 12 个月滚动最差回撤（vs 定版 tvol={base}）==")
    print(f"  {'tvol':>6}{'k':>3}{'中位':>8}{'p90':>8}{'最差':>8}{'Δ中位':>10}")
    for k in PHASES:
        ref = _roll_mdd(eq[(base, k)])
        for t in tvols:
            r = _roll_mdd(eq[(t, k)])
            j = r.index.intersection(ref.index)
            dm = r[j].median() - ref[j].median()
            tag = "" if np.isclose(t, base) else f"{dm * 100:>+9.2f}pp"
            print(f"  {t:>6.2f}{k:>3}{r.median():>8.2%}{r.quantile(0.9):>8.2%}"
                  f"{r.max():>8.2%}{tag}")
        print()


def print_phase_independence(eq: dict, base: float) -> None:
    """5 个相位净值的两两相关 ⇒ 有效独立样本数 n_eff。"""
    cols = {}
    for k in PHASES:
        if (base, k) in eq:
            cols[k] = eq[(base, k)]
    if len(cols) < 2:
        return
    e = pd.DataFrame(cols).dropna()
    r = e.pct_change().dropna()
    c = r.corr()
    iu = np.triu_indices_from(c.to_numpy(), 1)
    vals = c.to_numpy()[iu]
    k = len(cols)
    print(f"\n== ③ 相位独立信息量（tvol={base}，月内第 k 个交易日）==")
    print("  相位日收益两两相关：")
    print(c.round(3).to_string())
    print(f"\n  平均相关 rho = {vals.mean():.3f}（{vals.min():.3f}~{vals.max():.3f}）")
    n_eff = k / (1.0 + (k - 1) * vals.mean())
    print(f"  有效独立样本 n_eff = {k}/(1+{k - 1}×{vals.mean():.3f}) = **{n_eff:.2f}**")
    print(f"  ⇒ 「跑到 5 个相位」实际只值 {n_eff:.2f} 个独立样本；"
          f"名义放大 {k:.0f}×，真实放大 {n_eff:.2f}×")
    # 绝对净值末值相关（含趋势，通常更高）
    c2 = e.corr()
    iu2 = np.triu_indices_from(c2.to_numpy(), 1)
    print(f"  净值（水平）相关均值 = {c2.to_numpy()[iu2].mean():.3f} "
          f"⇒ 两者都指向「同一注的不同起点」")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tvols", default=",".join(str(t) for t in DEFAULT_TVOLS))
    ap.add_argument("--csv", default=str(OUT_CSV))
    ap.add_argument("--from-csv", action="store_true")
    args = ap.parse_args(argv)

    if args.from_csv:
        d = pd.read_csv(Path(args.csv), encoding="utf-8-sig")
        print_summary(summarize(d))
        print("\n（--from-csv 不重跑 ⇒ 配对 t / 滚动回撤 / 相位相关不可得）")
        return 0

    tvols = tuple(float(t) for t in args.tvols.split(",") if t.strip())

    print("加载环境（全量 ETF 面板 + DEF 合成腿，约 1-2 分钟）…", flush=True)
    t0 = time.time()
    env = build_env()
    rb_map = month_phase_rebalance_days(env["close"].index)
    print(f"环境就绪 {time.time() - t0:.0f}s；{len(tvols)} 档 tvol × {len(PHASES)} 相位 "
          f"= {len(tvols) * len(PHASES)} 次回测", flush=True)

    base = dict(PROFILES["champ"])
    rows: list[dict] = []
    eq: dict[tuple[float, int], pd.Series] = {}

    for tv in tvols:
        spec = dict(base)
        spec["target_vol"] = tv
        for k in PHASES:
            t1 = time.time()
            _r0, r1 = run_champ(env, spec, rb_map[k])
            eq[(tv, k)] = r1["equity"].dropna()
            w = r1["weights"]
            held = [c for c in w.columns if c not in ("CASH", "DEF")]
            live = w[w.abs().sum(axis=1) > 1e-9]
            leg = live[held].sum(axis=1)
            dfw = live["DEF"] if "DEF" in live.columns else pd.Series(
                0.0, index=live.index)
            rows.append({
                "tvol": tv, "phase": k,
                "ann": r1["annual_return"], "sharpe": r1["sharpe"],
                "mdd": r1["max_drawdown"], "turnover": r1["avg_turnover"],
                "gross": float(leg.median()), "defw": float(dfw.median()),
                "n_rebalance": r1.get("n_rebalance", 0),
            })
            print(f"  tvol={tv:.2f} k={k}  年化 {r1['annual_return']:>7.2%}  "
                  f"SR {r1['sharpe']:>5.2f}  MDD {r1['max_drawdown']:>7.2%}  "
                  f"多头仓 {float(leg.median()):.3f}  ({time.time() - t1:.0f}s)",
                  flush=True)

    d = pd.DataFrame(rows)
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False, encoding="utf-8-sig")

    print_summary(summarize(d))
    print_judge(eq, tvols, BASE_TVOL)
    print_phase_independence(eq, BASE_TVOL)
    print(f"\n明细已落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
