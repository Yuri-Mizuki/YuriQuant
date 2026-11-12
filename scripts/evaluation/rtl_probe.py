"""调仓时点运气（Rebalance Timing Luck, RTL）—— 相位极差的文献口径复现。

回答 2026-11-10 的问题：「调仓日放在哪一天有没有公开研究？感觉我们很多策略的
调仓日极差挺大的。」

为什么需要这个脚本
------------------
此前只扫了 7 个相位（`solo_freq_probe --only month_phase`），且只用**极差**描述。
文献（Hoffstein et al. 2019）的 RTL 定义是各相位年化的**标准差**，两者差 3.6 倍
（我们 4.97% vs 17.85pp），**混用会让人以为我们比文献差很多**。本脚本做三件事：

1. **21 相位全扫**（月内每个交易日一个相位），给出标准差口径的 RTL；
2. **等权基准同相位对照** —— 分离「市场日历效应」与「策略结构放大」。
   🚨 这是本脚本的核心：若基准极差也很大 ⇒ 是市场日历效应；若基准极差很小
   ⇒ 极差全部来自策略把微小差异放大。
3. **21 相位叠加（tranching）** —— 文献的标准解法，看能降多少。
   同时给出「无选择偏差」的收益估计（叠加值 / 中位数），用于校正「报最大值」的偏差。

文献参照
--------
- Hoffstein, Sibears & Faber (2019), *Rebalance Timing Luck: The Difference between
  Hired and Fired*, J. Index Investing. 100 只半年调仓：动量 4.4%、价值 2.5%、
  质量 2.0%、低波 1.1%（均为**年化标准差**）。三个放大器：低频 / 集中 / 快衰减信号。
- Zarattini & Pagani (2025), *The Tranching Dilemma*：20 只月度动量，1991-2024，
  最好 vs 最差调仓日 **CAGR 差 ≈ 350bp**；小资金 tranching 不划算。

输出 ``reports/rtl_rebalance_timing/``
--------------------------------------
  phase21.csv   21 相位明细（solo 年化/SR/回撤 + 等权基准年化）
  summary.csv   三个口径汇总（标准差 / 极差 / 建仓偏差校正）

用法::

    python -m scripts.evaluation.rtl_probe
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scripts.evaluation.caliber as cal
from research.benchmarks import segmented_equal_weight_returns
from scripts.evaluation.solo_freq_probe import (
    CAPITAL,
    END,
    LOT,
    SOLO_BUF,
    SOLO_K,
    industry_panel,
    price_panel,
    stats,
)
from strategy.cross_section import SectorCappedBufferedTopKLongOnly

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "rtl_rebalance_timing"
N_PHASE = 21          # 月内交易日数上限


def _eqw_ann(ctx: cal.CaliberContext, phase: int, full_close: pd.DataFrame) -> float:
    """等权基准：与 solo **同一相位**调仓、段内权重随价漂移。"""
    rb = cal.rebalance_dates(ctx.pred.index, freq="M", phase=phase)
    s = segmented_equal_weight_returns(full_close, rebalance_dates=rb,
                                       index=ctx.pred.index)
    n_years = len(s) / 252.0
    return float((1 + s).prod() ** (1 / n_years) - 1)


def _max_dd(dr: pd.Series) -> float:
    eq = (1 + dr).cumprod()
    return float((eq / eq.cummax() - 1).min())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = cal.load_context("gbdt_mid13", window="prod", end=END, execution="open")
    cost = cal.cost_profile("production")
    px = price_panel(ctx)
    ind = industry_panel(ctx)
    full_close = pd.read_parquet(cal.BASE_DIR / "close_adj.parquet")
    print(f"窗口 {ctx.pred.index[0].date()}..{ctx.pred.index[-1].date()} "
          f"| 资金 {CAPITAL:,.0f} | 整手 {LOT} | k={SOLO_K} b={SOLO_BUF}", flush=True)

    rows, rets = [], {}
    for p in range(N_PHASE):
        n_reb = len(cal.rebalance_dates(ctx.pred.index, freq="M", phase=p))
        strat = SectorCappedBufferedTopKLongOnly(
            ind, k=SOLO_K, exit_mult=SOLO_BUF, max_per_sector=1)
        bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                initial_capital=CAPITAL, costs=cost)
        res = cal.run_once(bt, ctx, "open", signal=None, freq="M", phase=p,
                           price_panel=px, lot_size=LOT)
        r = stats(res, n_reb)
        rets[p] = res.daily_returns
        bench = _eqw_ann(ctx, p, full_close)
        rows.append({"phase": p, "n_reb": n_reb, "solo_ann": r["ann"],
                     "solo_sr": r["sharpe_arith"], "solo_dd": r["max_dd"],
                     "eqw_ann": bench, "excess": r["ann"] - bench})
        print(f"  phase={p:<3} solo={r['ann'] * 100:7.2f}% SR={r['sharpe_arith']:6.3f} "
              f"等权基准={bench * 100:7.2f}% 超额={(r['ann'] - bench) * 100:6.2f}%",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "phase21.csv", index=False, encoding="utf-8-sig")

    # 叠加（tranching）：21 条腿等权
    R = pd.DataFrame(rets)
    blend = R.mean(axis=1)
    n_y = len(blend) / 252.0
    blend_ann = float((1 + blend).cumprod().iloc[-1] ** (1 / n_y) - 1)
    blend_sr = float(blend.mean() / blend.std(ddof=1) * np.sqrt(252))
    blend_dd = _max_dd(blend)

    def stat(col: str) -> dict:
        s = df[col]
        return {"mean": s.mean(), "median": s.median(),
                "std": s.std(ddof=1),
                "range": s.max() - s.min(), "min": s.min(), "max": s.max()}

    sm = pd.DataFrame({
        "solo_ann": stat("solo_ann"),
        "eqw_ann": stat("eqw_ann"),
        "excess": stat("excess"),
    }).T
    sm.to_csv(OUT / "summary.csv", encoding="utf-8-sig")

    print("\n=== 汇总（solo vs 等权基准）===", flush=True)
    for name in ("solo_ann", "eqw_ann", "excess"):
        x = sm.loc[name]
        print(f"  {name:<10} 均值 {x['mean'] * 100:7.2f}%  标准差 {x['std'] * 100:6.2f}%  "
              f"极差 {x['range'] * 100:6.2f}pp  (min {x['min'] * 100:6.2f} / "
              f"max {x['max'] * 100:6.2f})", flush=True)

    ratio = sm.loc["solo_ann", "std"] / sm.loc["eqw_ann", "std"]
    print(f"\n  放大倍数（solo 标准差 / 基准标准差）= {ratio:.1f}x", flush=True)
    print("\n=== 叠加（tranching，21 腿等权）===", flush=True)
    print(f"  ann={blend_ann * 100:7.2f}%  SR_arith={blend_sr:6.3f}  "
          f"DD={blend_dd * 100:5.1f}%", flush=True)
    print(f"  对照 phase=0: ann={df.loc[0, 'solo_ann'] * 100:.2f}%  "
          f"SR={df.loc[0, 'solo_sr']:.3f}", flush=True)

    # ---- 文献口径的 RTL：相对叠加组合的**月度跟踪误差**年化 ----
    # Hoffstein et al. (2019) 原文：std of monthly return dispersion between the
    # different rebalance date variations of the overlapping portfolio solution,
    # then annualize. 🚨 这与「年化收益的横截面 std」是**两个不同的量**（实测差约 2.2 倍），
    # 只有这个才可与文献 4.4% 比较。
    tes = {}
    for p in R.columns:
        diff = R[p] - blend
        mon = (1 + diff).resample("ME").prod() - 1
        tes[p] = float(mon.std(ddof=1) * np.sqrt(12))
    te = pd.Series(tes)
    print("\n=== 文献口径 RTL（月度跟踪误差 × √12）===", flush=True)
    print(f"  21 相位：均值 {te.mean() * 100:.2f}%  中位 {te.median() * 100:.2f}%  "
          f"范围 {te.min() * 100:.2f}~{te.max() * 100:.2f}%", flush=True)
    print(f"  文献对照 Hoffstein 100 只半年动量 = 4.4%  ⇒ 我们 {te.mean() / 0.044:.1f}x",
          flush=True)
    te.rename("rtl_te").to_frame().assign(phase=te.index).to_csv(
        OUT / "rtl_tracking_error.csv", index=False, encoding="utf-8-sig")

    # 选择偏差：随机极大值期望（正态近似）
    mu, sd = sm.loc["solo_ann", "mean"], sm.loc["solo_ann", "std"]
    rng = np.random.default_rng(20261110)
    e_max = float(rng.standard_normal((200_000, N_PHASE)).max(axis=1).mean())
    print("\n=== 选择偏差校正 ===", flush=True)
    print(f"  E0 phase=0（定版，21 个里的最大值） {df.loc[0, 'solo_ann'] * 100:7.2f}%",
          flush=True)
    print(f"  E1 随机极大值期望 μ+σ·E[max]={e_max:.3f}  {(mu + sd * e_max) * 100:7.2f}%",
          flush=True)
    print(f"  E2 21 相位叠加（实测）              {blend_ann * 100:7.2f}%", flush=True)
    print(f"  E3 21 相位中位数                    {sm.loc['solo_ann', 'median'] * 100:7.2f}%",
          flush=True)
    print(f"\n产物 → {OUT}", flush=True)


if __name__ == "__main__":
    main()