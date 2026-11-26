"""追顶风险控制手段测试（2026-11-26 建）。

背景
----
归因/择时检验（``reports/style_attribution``）显示：三档超额来自**慢速风格动量**
——TopK 篮子的 12 个月事前暴露与过去 12 月风格收益正相关（momentum 0.47~0.51、
liquidity 0.38、nl_size 0.32~0.39、btop 0.36~0.47）⇒ 信号在追「已经涨过的风格」。
regime 反转（2024-01 微盘崩）时这套暴露正好在顶部，表现为**涨月 β 0.65、跌月 β
1.02~1.04**。本脚本只测**风控 overlay**，不碰信号本身：目标是「用年化 −1~2pp
换反转期回撤 −10pp」。

四条规则（月度决策、次月生效）
------------------------------
=========  ==========================================================
R0         无约束（w=1）
R1 断路器   滚动 6 月超额 < −8pp ⇒ 后续至少 3 个月 w=0.7；
           退出需「已满 3 个月 + 连续 2 个正超额月」（快减慢加）
R2 过热     12 月风格收益的**扩张分位** >90%（liquidity / nl_size /
           momentum / beta 任一）⇒ 下月 w=0.8
R3          R1 + R2 取 min
R4 暴露上限  (仅集中档) 篮子 nl_size / momentum 截面 z 暴露 > 扩张
           均值+1σ ⇒ 下月 w=0.8
=========  ==========================================================

口径
----
- 日净值：``reports/yq_core/nav_attrib.csv``（f50 数据态、uniform 1e6 无整手）。
- 基准：``reports/alla_rolling/_base/bench_eqw.parquet``（全 A 日频等权）。
- 风格收益：``style_attribution.style_monthly()``（Barra WLS 月频，已行业中性）。
- 月度收益一律 **月末对月末**（日收益按月复利），最后一个月若被 END 截断则剔除。
- 降权部分视为**现金，收益 0**（与全项目 rf=0 口径一致）。
- 权重变化的额外成本：``|Δw| × 13.5bp``（单边），计在当月首个交易日。

产物：``reports/topping_risk/{single_path.csv,episodes.csv,phase_overlay.csv,summary.csv}``。

用法::

    python -m scripts.portfolio.topping_risk_test              # 单路径四档 × 5 规则
    python -m scripts.portfolio.topping_risk_test --phases 5   # 21 相位（R0 vs R3）
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.metrics import (  # noqa: E402
    annual_return, annual_volatility, max_drawdown, sharpe_ratio)
from scripts.portfolio.style_attribution import (  # noqa: E402
    TIERS, style_monthly)
from scripts.portfolio.topping_overlay import (  # noqa: E402
    expanding_pct, hot_flags, to_weekly, weekly_weights)

OUT = ROOT / "reports" / "topping_risk"
NAV_CSV = ROOT / "reports" / "yq_core" / "nav_attrib.csv"
BENCH = ROOT / "reports" / "alla_rolling" / "_base" / "bench_eqw.parquet"
BASKET = ROOT / "reports" / "style_attribution" / "basket_exposures.csv"

TRIG_STYLES = ("liquidity", "nl_size", "momentum", "beta")
EXPO_STYLES = ("nl_size", "momentum")
COST_SIDE = 13.5e-4          # 单边换手成本：13.5bp
Q_HOT = 0.90                 # 12 月风格收益分位阈值
LOOKBACK = 6                 # 断路器回看月数
DD_THRESH = -0.08            # 滚动 6 月超额触发阈值
W_CUT, W_HOT, W_EXPO = 0.7, 0.8, 0.8
MIN_CUT, NEED_POS = 3, 2

#: 反转型回撤窗口（样本内；2018Q4 不在 prod 窗口 2019 起内，无法评估）
EPISODES: dict[str, tuple[str, str]] = {
    "2024-01 微盘崩": ("2024-01-01", "2024-02-08"),
    "2021-02 抱团瓦解": ("2021-02-10", "2021-03-09"),
    "2022-01 拥挤反转": ("2022-01-04", "2022-03-15"),
    "2020-02 新冠": ("2020-02-20", "2020-03-23"),
    "2019-04 回撤": ("2019-04-19", "2019-06-06"),
}


# ---------------------------------------------------------------------------
# 月度工具
# ---------------------------------------------------------------------------
def month_ret(daily: pd.Series) -> pd.Series:
    """日收益 → 月末对月末月收益（PeriodIndex 对齐，避免两套日期标签错位）。"""
    per = daily.index.to_period("M")
    return (1 + daily.fillna(0.0)).groupby(per).prod() - 1


def trim_partial(daily: pd.Series, bench_d: pd.Series | None = None,
                 min_frac: float = 0.9, freq: str = "M") -> pd.Series:
    """剔除被 END 截断的末月（``freq="W-FRI"`` 时是末周）。

    只判**末期**：样本中部的短月（春节等）是真实短月，两侧同样短，不能删。
    """
    if bench_d is None:
        return daily
    per = daily.index.to_period(freq)
    cnt = pd.Series(1, index=per).groupby(level=0).sum()
    pb = bench_d.index.to_period(freq)
    cb = pd.Series(1, index=pb).groupby(level=0).sum()
    last = cnt.index[-1]
    if last in cb.index and cnt.loc[last] < min_frac * cb.loc[last]:
        return daily[per != last]
    return daily


# ---------------------------------------------------------------------------
# 规则（过热触发与落地实现共用一份：scripts/portfolio/topping_overlay.py）
# ---------------------------------------------------------------------------
def style_hot(styles: pd.DataFrame, q: float = Q_HOT,
              min_periods: int = 24) -> pd.Series:
    """12 月风格收益的扩张分位 > q（任一触发风格）⇒ 过热。"""
    return hot_flags(styles, TRIG_STYLES, q, 12, min_periods)


def breaker_weights(excess: pd.Series, lookback: int = LOOKBACK,
                    thresh: float = DD_THRESH, cut: float = W_CUT,
                    min_cut: int = MIN_CUT, need_pos: int = NEED_POS) -> pd.Series:
    """滚动 ``lookback`` 月超额跌破 ``thresh`` ⇒ 至少 ``min_cut`` 个月 w=cut。

    「快减慢加」：触发即降，退出要满足最短降仓期 + 连续 ``need_pos`` 个正超额月。
    """
    roll = (1 + excess.fillna(0.0)).rolling(lookback).apply(np.prod, raw=True) - 1
    w = pd.Series(1.0, index=excess.index)
    state, cut_applied, pos = 0, 0, 0
    for i in range(len(excess)):
        w.iloc[i] = cut if state == 1 else 1.0
        ex = float(excess.iloc[i])
        if state == 0:
            r = roll.iloc[i]
            if np.isfinite(r) and r < thresh:
                state, cut_applied, pos = 1, 0, 0
        else:
            cut_applied += 1
            pos = pos + 1 if ex > 0 else 0
            if cut_applied >= min_cut and pos >= need_pos:
                state = 0
    return w


def hot_weights(hot: pd.Series, w: float = W_HOT) -> pd.Series:
    """t 月末过热 ⇒ t+1 月降权。"""
    out = pd.Series(1.0, index=hot.index)
    out[hot.shift(1).fillna(False).astype(bool)] = w
    return out


def exposure_hot(tier: str, min_periods: int = 24) -> pd.Series | None:
    """篮子 nl_size / momentum 截面 z 暴露 > 扩张均值+1σ（t 月末信息）。"""
    if not BASKET.exists():
        return None
    be = pd.read_csv(BASKET, index_col=0, parse_dates=True)
    cols = [f"{tier}|{s}" for s in EXPO_STYLES]
    if any(c not in be.columns for c in cols):
        return None
    df = be[cols]
    hot = (df > df.expanding(min_periods=min_periods).mean()
           + df.expanding(min_periods=min_periods).std()).any(axis=1)
    hot.index = hot.index.to_period("M")
    return hot


# ---------------------------------------------------------------------------
# 施加权重
# ---------------------------------------------------------------------------
def apply_weights(daily: pd.Series, w_month: pd.Series) -> tuple[pd.Series, dict]:
    """月度目标权重 → 日收益（降权部分现金）+ 调仓成本（当月首个交易日扣）。"""
    per = daily.index.to_period("M")
    wmap = dict(w_month.items())
    wd = np.array([float(wmap.get(p, 1.0)) for p in per])
    r = daily.fillna(0.0).to_numpy(dtype="float64")

    prev = w_month.shift(1).fillna(1.0)
    dw = (w_month - prev).abs()
    first_mask = ~per.duplicated()
    dc = dw.reindex(pd.PeriodIndex(per[first_mask], freq="M")).fillna(0.0).to_numpy()
    cost = np.zeros(len(daily))
    cost[first_mask] = dc * COST_SIDE

    net = pd.Series(r * wd - cost, index=daily.index)
    info = {"n_cut_month": int((w_month < 1.0 - 1e-9).sum()),
            "n_switch": int((dw > 1e-9).sum()),
            "w_mean": float(wd.mean()),
            "cost_ann": float(cost.sum() / max(len(daily), 1) * 244)}
    return net, info


def apply_weights_weekly(daily: pd.Series,
                         w_week: pd.Series) -> tuple[pd.Series, dict]:
    """周频目标权重 → 日收益（降权部分现金）+ 成本（每周首个交易日扣）。

    ``w_week`` 已在 ``weekly_weights`` 里 ``shift(1)`` ⇒ 第 k 周倍率由第 k-1 周
    周五收盘的信息决定（下周一开盘生效），周内每日同倍率。
    """
    per = daily.index.to_period("W-FRI")
    wmap = dict(w_week.items())
    wd = np.array([float(wmap.get(p, 1.0)) for p in per])
    r = daily.fillna(0.0).to_numpy(dtype="float64")

    dw = (w_week - w_week.shift(1).fillna(1.0)).abs()
    first_mask = ~per.duplicated()
    dc = dw.reindex(pd.PeriodIndex(per[first_mask], freq="W-FRI")).fillna(0.0)
    cost = np.zeros(len(daily))
    cost[first_mask] = dc.to_numpy() * COST_SIDE

    net = pd.Series(r * wd - cost, index=daily.index)
    info = {"n_cut_week": int((w_week < 1.0 - 1e-9).sum()),
            "n_switch": int((dw > 1e-9).sum()),
            "w_mean": float(wd.mean()),
            "cost_ann": float(cost.sum() / max(len(daily), 1) * 244)}
    return net, info


def stats(daily: pd.Series) -> dict:
    return {"ann": float(annual_return(daily)),
            "vol": float(annual_volatility(daily)),
            "sharpe": float(sharpe_ratio(daily)),
            "mdd": float(max_drawdown(daily)),
            "total": float((1 + daily).prod() - 1)}


def episode(daily: pd.Series, lo: str, hi: str) -> tuple[float, float]:
    s = daily.loc[lo:hi]
    if s.empty:
        return float("nan"), float("nan")
    eq = (1 + s).cumprod()
    return float(eq.iloc[-1] - 1), float((eq / eq.cummax() - 1).min())


# ---------------------------------------------------------------------------
# 单路径
# ---------------------------------------------------------------------------
def build_rules(excess: pd.Series, styles: pd.DataFrame,
                tier: str, thresh: float = DD_THRESH,
                q: float = Q_HOT) -> dict[str, pd.Series]:
    hot = style_hot(styles, q).reindex(excess.index).fillna(False)
    w1 = breaker_weights(excess, thresh=thresh)
    w2 = hot_weights(hot)
    w3 = pd.concat([w1, w2], axis=1).min(axis=1)
    rules = {"R0": pd.Series(1.0, index=excess.index),
             "R1": w1, "R2": w2, "R3": w3}
    eh = exposure_hot(tier)
    if eh is not None:
        rules["R4"] = hot_weights(eh.reindex(excess.index).fillna(False), W_EXPO)
    return rules


def single_path() -> int:
    nav = pd.read_csv(NAV_CSV, index_col=0, parse_dates=True)
    bench_d = pd.read_parquet(BENCH)["ret"]
    daily_all = nav.pct_change(fill_method=None).fillna(0.0)[list(TIERS)]
    styles = style_monthly()
    styles.index = styles.index.to_period("M")

    rows, ep_rows = [], []
    for tier in TIERS:
        daily = trim_partial(daily_all[tier], bench_d)
        m_t = month_ret(daily)
        m_b = month_ret(bench_d).reindex(m_t.index)
        keep = m_b.notna()
        m_t, m_b = m_t[keep], m_b[keep]
        daily = daily[daily.index.to_period("M").isin(m_t.index)]
        excess = m_t - m_b
        print(f"\n=== {tier}  月度 {m_t.index.min()} ~ {m_t.index.max()} "
              f"（{len(m_t)} 月）基准月均 {m_b.mean():.2%}")

        rules = build_rules(excess, styles, tier)
        for name, w in list(rules.items()) + [
                ("S2", pd.Series(float(rules["R2"].mean()), index=excess.index))]:
            if name == "S2":       # 对照：与 R2 同平均敞口的**常数**仓位（无择时）
                net = daily * float(w.iloc[0])
                info = {"n_cut_month": 0, "n_switch": 0,
                        "w_mean": float(w.iloc[0]), "cost_ann": 0.0}
            else:
                net, info = apply_weights(daily, w)
            s = stats(net)
            row = {"tier": tier, "rule": name, **s, **info}
            rows.append(row)
            print(f"  {name}  ann {s['ann']:7.2%}  SR {s['sharpe']:6.3f}  "
                  f"MDD {s['mdd']:7.2%}  降仓 {info['n_cut_month']:2d} 月 / "
                  f"切换 {info['n_switch']:2d} 次  w̄ {info['w_mean']:.3f}")
            for ep, (lo, hi) in EPISODES.items():
                r, dd = episode(net, lo, hi)
                ep_rows.append({"tier": tier, "rule": name, "episode": ep,
                                "ret": r, "mdd": dd})

    # 灵敏度：阈值太松的规则不会触发 ⇒ 给一张「年化代价 vs 2024-01 改善」表
    sens = []
    for tier in TIERS:
        daily = trim_partial(daily_all[tier], bench_d)
        m_t = month_ret(daily)
        m_b = month_ret(bench_d).reindex(m_t.index)
        keep = m_b.notna()
        m_t, m_b = m_t[keep], m_b[keep]
        daily = daily[daily.index.to_period("M").isin(m_t.index)]
        excess = m_t - m_b
        r0 = stats(daily)
        r0_24, _ = episode(daily, *EPISODES["2024-01 微盘崩"])
        for thresh in (-0.04, -0.06, -0.08, -0.10):
            net, info = apply_weights(daily, breaker_weights(excess, thresh=thresh))
            s = stats(net)
            r24, _ = episode(net, *EPISODES["2024-01 微盘崩"])
            sens.append({"tier": tier, "rule": f"brk{abs(thresh):.0%}",
                         "ann": s["ann"], "mdd": s["mdd"],
                         "ep2024_ret": r24, "n_cut": info["n_cut_month"],
                         "d_ann_pp": (s["ann"] - r0["ann"]) * 100,
                         "d_2024_pp": (r24 - r0_24) * 100})
        hot_base = style_hot(styles).reindex(excess.index)
        for q in (0.80, 0.85, 0.90):
            hot = (hot_base if q == Q_HOT else
                   style_hot(styles, q)).reindex(excess.index).fillna(False)
            net, info = apply_weights(daily, hot_weights(hot))
            s = stats(net)
            r24, _ = episode(net, *EPISODES["2024-01 微盘崩"])
            sens.append({"tier": tier, "rule": f"hot{int(q * 100)}",
                         "ann": s["ann"], "mdd": s["mdd"],
                         "ep2024_ret": r24, "n_cut": info["n_cut_month"],
                         "d_ann_pp": (s["ann"] - r0["ann"]) * 100,
                         "d_2024_pp": (r24 - r0_24) * 100})

    OUT.mkdir(parents=True, exist_ok=True)
    sp = pd.DataFrame(rows)
    ep = pd.DataFrame(ep_rows)
    sd = pd.DataFrame(sens)
    sp.to_csv(OUT / "single_path.csv", index=False)
    ep.to_csv(OUT / "episodes.csv", index=False)
    sd.to_csv(OUT / "sensitivity.csv", index=False)

    print("\n=== 各规则 MDD")
    print(sp.pivot(index="tier", columns="rule", values="mdd").round(4).to_string())
    print("\n=== 反转期累计收益")
    print(ep.pivot_table(index=["tier", "episode"], columns="rule",
                         values="ret").round(4).to_string())
    print("\n=== 断路器阈值灵敏度（Δ年化 pp / Δ2024-01 pp）")
    print(sd[sd.rule.str.startswith("brk")][
        ["tier", "rule", "n_cut", "d_ann_pp", "d_2024_pp"]].round(2).to_string(index=False))
    print("\n=== 过热分位灵敏度")
    print(sd[sd.rule.str.startswith("hot")][
        ["tier", "rule", "n_cut", "d_ann_pp", "d_2024_pp"]].round(2).to_string(index=False))

    print(f"\n产物：{OUT}")
    return 0


# ---------------------------------------------------------------------------
# 周频决策（周五收盘 → 下周一开盘）
# ---------------------------------------------------------------------------
def weekly_path() -> int:
    """月频 vs 周频：**同一段日收益**，只换过热降仓的决策频率。

    月频：t 月末判 → t+1 月生效（滞后最长 21 天）；
    周频：第 k 周周五收盘判 → 第 k+1 周一生效（滞后最长 5 天）。
    """
    nav = pd.read_csv(NAV_CSV, index_col=0, parse_dates=True)
    bench_d = pd.read_parquet(BENCH)["ret"]
    daily_all = nav.pct_change(fill_method=None).fillna(0.0)[list(TIERS)]
    styles = style_monthly()
    styles.index = styles.index.to_period("M")
    w_week = weekly_weights()

    rows = []
    for tier in TIERS:
        daily = trim_partial(daily_all[tier], bench_d)
        m_t = month_ret(daily)
        m_b = month_ret(bench_d).reindex(m_t.index)
        keep = m_b.notna()
        m_t, m_b = m_t[keep], m_b[keep]
        daily = daily[daily.index.to_period("M").isin(m_t.index)]
        excess = m_t - m_b
        hot = style_hot(styles).reindex(excess.index).fillna(False)
        w_m = hot_weights(hot)

        print(f"\n=== {tier}  周频 vs 月频  "
              f"{daily.index.min().date()} ~ {daily.index.max().date()}")
        net_m, info_m = apply_weights(daily, w_m)
        net_w, info_w = apply_weights_weekly(daily, w_week)
        combos = {
            "R0": (daily, {"n_cut": 0, "n_switch": 0,
                           "w_mean": 1.0, "cost_ann": 0.0}),
            "R2m": (net_m, {"n_cut": info_m["n_cut_month"],
                            "n_switch": info_m["n_switch"],
                            "w_mean": info_m["w_mean"],
                            "cost_ann": info_m["cost_ann"]}),
            "R2w": (net_w, {"n_cut": info_w["n_cut_week"],
                            "n_switch": info_w["n_switch"],
                            "w_mean": info_w["w_mean"],
                            "cost_ann": info_w["cost_ann"]}),
            # 同平均敞口的常数仓对照（分开算，因为月/周的平均敞口略有差异）
            "S2m": (daily * info_m["w_mean"],
                    {"n_cut": 0, "n_switch": 0,
                     "w_mean": info_m["w_mean"], "cost_ann": 0.0}),
            "S2w": (daily * info_w["w_mean"],
                    {"n_cut": 0, "n_switch": 0,
                     "w_mean": info_w["w_mean"], "cost_ann": 0.0}),
        }
        for name, (net, info) in combos.items():
            s = stats(net)
            r24, _ = episode(net, *EPISODES["2024-01 微盘崩"])
            r22, _ = episode(net, *EPISODES["2022-01 拥挤反转"])
            rows.append({"tier": tier, "rule": name, **s,
                         "ep2024_ret": r24, "ep2022_ret": r22, **info})
            print(f"  {name:4s} ann {s['ann']:7.2%}  SR {s['sharpe']:6.3f}  "
                  f"MDD {s['mdd']:7.2%}  2024-01 {r24:7.2%}  "
                  f"2022-01 {r22:7.2%}  降仓 {info['n_cut']:3d} / "
                  f"切换 {info['n_switch']:3d}  w̄ {info['w_mean']:.3f}")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "weekly.csv", index=False)
    print("\n=== 月频 vs 周频（Δ = 周 − 月，pp）")
    comp = []
    for tier in TIERS:
        w = df[df.tier == tier].set_index("rule")
        comp.append({"tier": tier,
                     "d_ann_pp": (w.loc["R2w", "ann"] - w.loc["R2m", "ann"]) * 100,
                     "d_sharpe": w.loc["R2w", "sharpe"] - w.loc["R2m", "sharpe"],
                     "d_mdd_pp": (w.loc["R2w", "mdd"] - w.loc["R2m", "mdd"]) * 100,
                     "d_2024_pp": (w.loc["R2w", "ep2024_ret"]
                                   - w.loc["R2m", "ep2024_ret"]) * 100,
                     "d_2022_pp": (w.loc["R2w", "ep2022_ret"]
                                   - w.loc["R2m", "ep2022_ret"]) * 100,
                     "n_cut_m": int(w.loc["R2m", "n_cut"]),
                     "n_cut_w": int(w.loc["R2w", "n_cut"]),
                     "n_switch_m": int(w.loc["R2m", "n_switch"]),
                     "n_switch_w": int(w.loc["R2w", "n_switch"]),
                     "cost_m_bp": w.loc["R2m", "cost_ann"] * 1e4,
                     "cost_w_bp": w.loc["R2w", "cost_ann"] * 1e4})
    cdf = pd.DataFrame(comp)
    cdf.to_csv(OUT / "weekly_compare.csv", index=False)
    print(cdf.round(2).to_string(index=False))

    cut = w_week[w_week < 1.0]
    hot22 = cut.loc[pd.Period("2021-10", "W-FRI"):pd.Period("2022-02", "W-FRI")]
    print(f"\n周频降仓 {len(cut)} 周 / 全样本 {len(w_week)} 周；"
          f"2021-10~2022-02 窗口：")
    print(hot22.to_string() if len(hot22) else "  （无降仓周）")
    print(f"\n产物：{OUT / 'weekly.csv'}  {OUT / 'weekly_compare.csv'}")
    return 0


# ---------------------------------------------------------------------------
# 21 相位复核（R0 vs R3）
# ---------------------------------------------------------------------------
def phase_overlay(phases: int, weekly: bool = False) -> int:
    from scripts.evaluation import caliber as cal
    from scripts.portfolio.focus_signal_compare import (  # noqa: WPS433
        ARMS, CAPS, lean_ctx, panel_for, price_panel)
    from strategy.registry import get_tier

    bench_d = pd.read_parquet(BENCH)["ret"]
    styles = style_monthly()
    styles.index = styles.index.to_period("M")
    w_week = weekly_weights() if weekly else None
    cost = cal.cost_profile("production")

    rows = []
    ctx = lean_ctx(ARMS["f50"])
    pp = price_panel(ctx)
    ind = panel_for("industry", ctx)
    print(f"\n=== 21 相位复核  arm f50  指纹 {ctx.fingerprint}  "
          f"{len(ctx.pred)} 日 × {ctx.pred.shape[1]} 只")
    for tier, (cap, lot) in CAPS.items():
        spec = get_tier(tier)
        overrides = {spec.panel: ind} if spec.panel else {}
        for ph in range(phases):
            strat = spec.build(**overrides)
            bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                    initial_capital=float(cap), costs=cost)
            res = cal.run_once(bt, ctx, "open", signal=ctx.pred,
                               price_panel=pp if lot else None,
                               lot_size=lot, phase=ph)
            daily = trim_partial(res.daily_returns.fillna(0.0), bench_d)
            m_t = month_ret(daily)
            m_b = month_ret(bench_d).reindex(m_t.index)
            keep = m_b.notna()
            m_t, m_b = m_t[keep], m_b[keep]
            daily = daily[daily.index.to_period("M").isin(m_t.index)]
            excess = m_t - m_b
            rules = build_rules(excess, styles, tier)
            wb = float(rules["R2"].mean())          # 同平均敞口的常数仓位对照
            rules["S2"] = pd.Series(wb, index=excess.index)
            nets = {"R0": apply_weights(daily, rules["R0"]),
                    "R2": apply_weights(daily, rules["R2"]),
                    "R3": apply_weights(daily, rules["R3"]),
                    "S2": (daily * wb, {"n_cut_month": 0})}
            if weekly:
                net_w, info_w = apply_weights_weekly(daily, w_week)
                nets["R2w"] = (net_w, info_w)
                nets["S2w"] = (daily * float(info_w["w_mean"]),
                               {"n_cut_month": 0})
            for name, (net, info) in nets.items():
                s = stats(net)
                r24, dd24 = episode(net, *EPISODES["2024-01 微盘崩"])
                rows.append({"tier": tier, "phase": ph, "rule": name,
                             "ann": s["ann"], "sharpe": s["sharpe"],
                             "mdd": s["mdd"], "ep2024_ret": r24,
                             "ep2024_mdd": dd24,
                             "n_cut": info.get("n_cut_month",
                                               info.get("n_cut_week", 0))})
            del res
        gc.collect()
    del ctx, pp, ind
    gc.collect()

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "phase_overlay.csv", index=False)

    summ = (df.groupby(["tier", "rule"])
            .agg(ann=("ann", "mean"), sharpe=("sharpe", "mean"),
                 mdd=("mdd", "mean"), ep2024_ret=("ep2024_ret", "mean"),
                 ep2024_mdd=("ep2024_mdd", "mean"),
                 n_cut=("n_cut", "mean")).reset_index())
    summ.to_csv(OUT / "summary.csv", index=False)
    print("\n=== 相位均值")
    print(summ.round(4).to_string(index=False))

    paired = []
    for tier in CAPS:
        w = df[df.tier == tier]
        for rule, base in (("R2", "S2"), ("R2w", "S2w")):
            if rule not in set(w.rule):
                continue
            for col in ("ann", "sharpe", "mdd", "ep2024_ret", "ep2024_mdd"):
                a = w[w.rule == rule].set_index("phase")[col]
                b = w[w.rule == "R0"].set_index("phase")[col]
                c = w[w.rule == base].set_index("phase")[col]
                d = (a - c).dropna()
                t = (float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
                     if len(d) > 1 else float("nan"))
                paired.append({"tier": tier, "rule": rule, "metric": col,
                               "n": int(len(d)),
                               "r0": float(b.mean()), "s2": float(c.mean()),
                               "r2": float(a.mean()),
                               "diff_r2_r0": float((a - b).mean()),
                               "diff_r2_s2": float(d.mean()),
                               "t_vs_s2": t,
                               "n_win_vs_s2": int((d > 0).sum())})
    pdf = pd.DataFrame(paired)
    pdf.to_csv(OUT / "paired.csv", index=False)
    print("\n=== 逐相位配对（R2 vs R0 / R2 vs S2 同敞口常数仓）")
    print(pdf.round(4).to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phases", type=int, default=0,
                    help=">0 时跑 21 相位复核（R0 vs R3）；0 = 只出单路径")
    ap.add_argument("--weekly", action="store_true",
                    help="加跑周频决策（周五收盘 → 下周一开盘）对比")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    single_path()
    if a.weekly:
        weekly_path()
    if a.phases > 0:
        phase_overlay(a.phases, weekly=a.weekly)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
