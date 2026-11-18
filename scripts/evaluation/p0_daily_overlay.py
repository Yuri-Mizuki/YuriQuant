"""P0：日频信号能不能给 `yqcore-focus` 做**风控叠加**（而非选股替代）
====================================================================

背景
----
`reports/intraday_daily/README.md` §三~§九 已经把所有「日频信号自己选股」的
方向判死：日频 Top10（H=1）净 +9.87%/SR 0.480，打不过 `yqcore-focus`
（29.60%/1.113），且换手 216.9/年、往返 27bp 即被成本吃光。

**唯一未被否的一条是「日频降回撤」**：日频策略 MDD −28.1% vs focus −38.2%，
浅约 10pp；且两者 top10 重合率仅 5.8% ⇒ **可能是低相关的两条腿**。

本脚本正面测两件事（这是「日频有没有残余价值」的最后一问）：

- **E1 两腿组合**：日频 Top10 与 focus 的**日收益相关性**，以及 70/30、80/20
  等权重下的组合指标。判据：ρ 低且组合 SR > focus 单腿 ⇒ 日频有组合价值。
- **E2 focus 持仓内剔除**：在 focus 的持仓里，按日频信号排序，把最差的
  ``--drop-frac`` 换成 focus 里的次优（或直接降到现金/等权稀释），看回撤与 SR。
  判据：回撤显著变浅且 SR 不降 ⇒ 日频有风控价值。

口径（**铁律**）
----------------
- focus 腿**只能**走 ``caliber.load_context`` + ``PROFILES["production"]``
  （27bp 时变 / T+1 open / 1e6 / 掩码 mask-lag 1），**不得自造基线**。
- 日频腿走 ``pit_intraday_probe.run``，成本同为单边 13.5bp（往返 27bp），
  与 §三~§九 逐位一致。
- 组合收益 = ``w·r_focus + (1-w)·r_daily``，**在扣完各自成本之后**相加。

🚨 三个必须先说清的边界（跑之前定死，不许事后找补）
--------------------------------------------------
1. **窗口只能到 2022-01~2026-06（1085 日 / 4.3 年）**。日频特征需要 5 分钟数据，
   PIT 池从 2022-01 才有；focus 信号从 2018 起。所以组合只能在重叠窗口上测，
   **不是 focus 的完整业绩期**（2019 起 1864 日）。focus 在 2019~2021 的表现不进本测。
2. **两条腿的池子不同**：日频池 421 只（HS300 PIT），focus 池 5826 只（全A）。
   这意味着「低相关」里有一部分只是**池子差异**，不全是「信号互补」。
   要有对照：同一个 421 池内，focus 信号自己跑一遍的日收益，才叫同池对照。
3. **组合是「事后配权」**：w 在全样本上扫，没有样本外。若某一 w 明显更好，
   必须报「w 的邻域是否平坦」，否则就是拟合。

用法::

    .venv/Scripts/python.exe -m scripts.evaluation.p0_daily_overlay
    .venv/Scripts/python.exe -m scripts.evaluation.p0_daily_overlay --stage corr
    .venv/Scripts/python.exe -m scripts.evaluation.p0_daily_overlay --stage blend
    .venv/Scripts/python.exe -m scripts.evaluation.p0_daily_overlay --stage drop

⚠️ 特征缓存由 pandas 3.0.5（venv）写出，系统 Python 读不了 ⇒ 必须 ``.venv`` 跑。
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

from scripts.common.cli_common import setup_logging  # noqa: E402
from scripts.evaluation import caliber as cal  # noqa: E402
from scripts.evaluation.pit_intraday_probe import (  # noqa: E402
    ONE_WAY_BP,
    OUT,
    ic_stats,
    load_context,
    orient,
)
from scripts.evaluation.pit_weekly_probe import (  # noqa: E402
    daily_ic,
    fwd_returns,
    run_periodic,
)

log = setup_logging("p0_daily_overlay")

ANCHOR = "mom_last_bar"      # §三~§九 的唯一日频锚因子
TOP = 10
FOCUS_ID = "yqcore-focus-k15"


# ----------------------------------------------------------------------
# 两条腿
# ----------------------------------------------------------------------
def focus_daily(signal: str | None = None, end: int | None = 20260630
                ) -> tuple[pd.Series, dict]:
    """按**生产口径**跑 focus，返回 (日收益, 指标)。

    `end` 缺省截到 20260630 —— 与日频特征的末日对齐，否则 focus 多出 4 个月
    而日频腿是 NaN，合并时会静默缩短样本或错位。
    """
    from strategy.registry import get_tier

    spec = get_tier(FOCUS_ID)
    sig = signal or spec.runnable_signal
    ctx = cal.load_context(signal=sig, window="prod", execution="open", end=end)
    cost = cal.cost_profile("production")
    kw: dict = {}
    if spec.panel is not None:
        pf = cal.BASE_DIR / f"{spec.panel}.parquet"
        if pf.exists():
            kw[spec.panel] = (pd.read_parquet(pf)
                              .reindex(index=ctx.pred.index,
                                       columns=ctx.pred.columns))
    strat = spec.build(**kw)
    bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                            initial_capital=1e6, costs=cost)
    res = cal.run_once(bt, ctx, "open", freq="M")
    return res.daily_returns.dropna(), cal.metrics_row(res, 1e6)


def daily_leg(feats: dict[str, pd.DataFrame], fwds: dict[int, pd.DataFrame],
              name: str = ANCHOR, top: int = TOP) -> tuple[pd.Series, dict]:
    """日频 Top10（H=1）净收益序列 + 指标，与 §三~§九 同口径。"""
    p = feats[name]
    ic = ic_stats(daily_ic(p, fwds[1]))
    sig = orient(p, float(ic["ic"]))
    r = run_periodic(sig, fwds[1], hold=1, top=top, cost_bp=ONE_WAY_BP)
    st = _daily_stats(r["net"])
    return r["net"].dropna(), st


def _daily_stats(x: pd.Series) -> dict:
    """日频腿的指标（与 caliber.metrics_row 同名同算法，便于并列）。"""
    from backtest.metrics import annual_return, annual_volatility, max_drawdown, sharpe_arithmetic
    x = x.dropna()
    sd = float(x.std(ddof=1))
    return {
        "ann": float(annual_return(x)),
        "sharpe": float(x.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
        "sharpe_arith": sharpe_arithmetic(x),
        "vol": float(annual_volatility(x)),
        "max_dd": float(max_drawdown(x)),
        "n_days": int(len(x)),
    }


def _stats_series(x: pd.Series) -> dict:
    return _daily_stats(x)


# ----------------------------------------------------------------------
# E1 相关性 + 组合
# ----------------------------------------------------------------------
def stage_corr(rf: pd.Series, rd: pd.Series) -> pd.DataFrame:
    idx = rf.index.intersection(rd.index)
    a, b = rf.loc[idx], rd.loc[idx]
    print(f"\n=== E1 相关性（重叠 {len(idx)} 日："
          f"{idx.min().date()} ~ {idx.max().date()}）===")
    print(f"  日收益 Pearson rho = {a.corr(b):+.4f}")
    print(f"  日收益 Spearman rho = {a.corr(b, method='spearman'):+.4f}")
    # 月度层面（降噪后相关性通常更高，也更接近「两条腿」的真实互补性）
    am = (1 + a).resample("ME").prod() - 1
    bm = (1 + b).resample("ME").prod() - 1
    m = am.index.intersection(bm.index)
    print(f"  月收益 Pearson rho = {am.loc[m].corr(bm.loc[m]):+.4f}（{len(m)} 个月）")
    # 逐年
    rows = []
    for y, sub in a.groupby(a.index.year):
        sb = b.loc[sub.index]
        rows.append({"year": y,
                     "focus": float((1 + sub).prod() - 1),
                     "daily": float((1 + sb).prod() - 1),
                     "rho": float(sub.corr(sb))})
    return pd.DataFrame(rows)


def stage_blend(rf: pd.Series, rd: pd.Series,
                weights: tuple[float, ...] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6,
                                              0.7, 0.8, 0.9, 1.0)) -> pd.DataFrame:
    """w = focus 权重；组合日收益 = w·focus + (1−w)·daily（各自已扣成本）。"""
    idx = rf.index.intersection(rd.index)
    a, b = rf.loc[idx], rd.loc[idx]
    print(f"\n=== E1 两腿组合（w = focus 权重，{len(idx)} 日）===")
    print("  %6s %9s %8s %8s %8s %8s" % ("w", "年化", "SR", "SR_arith",
                                         "波动", "最大回撤"))
    rows = []
    for w in weights:
        x = w * a + (1 - w) * b
        st = _stats_series(x)
        rows.append({"w_focus": w, **st})
        print("  %6.1f %+8.2f%% %8.3f %8.3f %7.2f%% %+8.2f%%" % (
            w, st["ann"] * 100, st["sharpe"], st["sharpe_arith"],
            st["vol"] * 100, st["max_dd"] * 100))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# E2 focus 持仓内用日频信号剔除
# ----------------------------------------------------------------------
def stage_drop(rf: pd.Series, feats: dict[str, pd.DataFrame],
               fwds: dict[int, pd.DataFrame], name: str = ANCHOR,
               fractions: tuple[float, ...] = (0.0, 0.2, 0.3, 0.5)
               ) -> pd.DataFrame:
    """在 focus **持仓内**按日频信号剔除最差的 ``frac``，剩余等权。

    🚨 这是最关键的一个测法：它**不改选股池**（仍是 focus 的 15 只），
    只改变权重分配。所以若有效，证明日频信号在 focus 的持仓里有**增量排序信息**；
    若无效，则说明 focus 的 15 只内部已经无可优化空间（这也与 §八
    「alpha 集中在极端端、锚排序最优」的结论一致）。
    """
    from strategy.registry import get_tier

    spec = get_tier(FOCUS_ID)
    ctx = cal.load_context(signal=spec.runnable_signal, window="prod",
                           execution="open", end=20260630)
    cost = cal.cost_profile("production")
    kw: dict = {}
    if spec.panel is not None:
        pf = cal.BASE_DIR / f"{spec.panel}.parquet"
        if pf.exists():
            kw[spec.panel] = (pd.read_parquet(pf)
                              .reindex(index=ctx.pred.index,
                                       columns=ctx.pred.columns))
    strat = spec.build(**kw)
    bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                            initial_capital=1e6, costs=cost)
    res = cal.run_once(bt, ctx, "open", freq="M")

    wh = res.weights_history
    held = (wh > cal.N_ACTIVE_TOL)
    print(f"\n=== E2 focus 持仓内按日频信号剔除（持仓矩阵 {held.shape}）===")
    print("  %6s %9s %8s %8s %8s" % ("剔除", "年化", "SR", "波动", "最大回撤"))

    # 日频信号（调仓日 T 收盘可得）
    p = feats[name]
    fwd1 = fwds[1]
    ic = ic_stats(daily_ic(p, fwd1))
    sig = orient(p, float(ic["ic"])).reindex(columns=ctx.pred.columns)

    rows = []
    for frac in fractions:
        daily_r, prev_w = [], None
        dates = wh.index
        for i, dt in enumerate(dates[:-1]):
            cur = held.loc[dt]
            names = list(cur[cur].index)
            if not names:
                continue
            if frac > 0 and dt in sig.index:
                sv = sig.loc[dt].reindex(names).dropna()
                if len(sv) >= 3:
                    # 信号低的先剔（orient 已保证值越大越好）
                    n_drop = int(round(len(names) * frac))
                    drop = sv.sort_values().index[:n_drop]
                    keep = [c for c in names if c not in set(drop)]
                    names = keep if keep else names
            w = pd.Series(1.0 / len(names), index=names)
            # 用下一交易日的 fwd 收益（T+1 open → T+2 open）
            nxt = dates[i + 1]
            if nxt not in fwd1.index:
                continue
            rr = fwd1.loc[nxt, names].dropna()
            if not len(rr):
                continue
            gross = float(rr.mean())
            # 换手 + 成本
            turns = 0.0
            if prev_w is not None:
                allc = w.index.union(prev_w.index)
                dw = w.reindex(allc, fill_value=0) - prev_w.reindex(allc, fill_value=0)
                turns = float(dw.abs().sum() / 2)
            daily_r.append((nxt, gross - turns * ONE_WAY_BP / 1e4))
            prev_w = w
        if not daily_r:
            continue
        ser = pd.Series(dict(daily_r)).sort_index()
        st = _stats_series(ser)
        rows.append({"drop_frac": frac, **st})
        print("  %5.0f%% %+8.2f%% %8.3f %7.2f%% %+8.2f%%" % (
            frac * 100, st["ann"] * 100, st["sharpe"],
            st["vol"] * 100, st["max_dd"] * 100))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="P0：日频信号 overlay 到 focus")
    ap.add_argument("--stage", default="all",
                    choices=["all", "corr", "blend", "drop"])
    ap.add_argument("--factor", default=ANCHOR)
    args = ap.parse_args()

    print("=" * 70)
    print("P0：日频信号能否给 focus 做风控叠加")
    print("=" * 70)

    store, _codes, daily, _ = load_context()
    fwds = fwd_returns(daily, [1])
    feats = pd.read_pickle(OUT / "xs_nl_feats.pkl")
    pool = [c for c in store.codes if c in fwds[1].columns]
    feats = {k: v.reindex(columns=pool) for k, v in feats.items()}

    rd, st_d = daily_leg(feats, fwds, args.factor)
    rf, st_f = focus_daily()
    print(f"\n日频腿（{args.factor} Top{TOP} H=1）：净 {st_d['ann']*100:+.2f}% / "
          f"SR {st_d['sharpe']:.3f} / MDD {st_d['max_dd']*100:+.2f}% / {st_d['n_days']} 日")
    print(f"focus 腿（{FOCUS_ID}，production）：净 {st_f['ann']*100:+.2f}% / "
          f"SR {st_f['sharpe']:.3f} / MDD {st_f['max_dd']*100:+.2f}% / "
          f"{st_f['n_days']} 日")

    idx = rf.index.intersection(rd.index)
    print(f"\n⚠️ 重叠窗口 {idx.min().date()} ~ {idx.max().date()}（{len(idx)} 日）"
          f"—— 日频需 5 分钟数据，PIT 池从 2022-01 才有")
    # 同窗口重算 focus（否则并列不公平）
    rf_w = rf.loc[idx]
    st_fw = _stats_series(rf_w)
    st_dw = _stats_series(rd.loc[idx])
    print(f"  同窗口 focus：{st_fw['ann']*100:+.2f}% / SR {st_fw['sharpe']:.3f} / "
          f"MDD {st_fw['max_dd']*100:+.2f}%")
    print(f"  同窗口日频 ：{st_dw['ann']*100:+.2f}% / SR {st_dw['sharpe']:.3f} / "
          f"MDD {st_dw['max_dd']*100:+.2f}%")

    if args.stage in ("all", "corr", "blend"):
        yr = stage_corr(rf, rd)
        print("\n  逐年：")
        for _, r in yr.iterrows():
            print("    %d  focus %+7.2f%%  daily %+7.2f%%  rho %+.3f" % (
                r["year"], r["focus"] * 100, r["daily"] * 100, r["rho"]))
        bl = stage_blend(rf, rd)
        bl.to_csv(OUT / "p0_blend.csv", index=False, encoding="utf-8-sig")

    if args.stage in ("all", "drop"):
        dr = stage_drop(rf, feats, fwds, args.factor)
        dr.to_csv(OUT / "p0_drop.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()