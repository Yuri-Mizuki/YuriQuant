"""前置检验：分钟频结构能不能预测 **focus 未来的相对表现**（不是涨跌）
=====================================================================

为什么先做这一步
----------------
补 488 只非 HS300 股票的分钟数据要约 28 分钟，但补完若是白补就亏了。
所以先把**唯一值得做的用法**（用法 C：整体仓位/风格择时）的信息检验跑掉 ——
不需要新数据，`focus@421` 的持仓与日频特征都已经在手上。

要测的不是「明天大盘涨不涨」（那是择时，本项目已多次判负），而是：

> **分钟频结构能不能预测「focus 相对其自身池子的未来超额」？**

若能，则可以用它调 focus 的**总仓位**（零额外换股成本）；
若不能，则补数据去做 E2 也只是在赌 `mom_last_bar` 那一个 t=−2.39 的边际信号。

三层检验（**逐层过，任一层不过即止损**）
--------------------------------------
- **T1 先行性**：结构指标 vs **未来** focus 超额收益，用**重叠窗口校正后**的 t。
  🚨 必须 `/√h` 或做 NW 校正，否则重叠窗口会把 t 机械放大 √h。
- **T2 稳定性**：按年拆。只在 1~2 个年份有效 = 一次风格，不是信号。
- **T3 可比对照**：把结构指标换成**同期已实现的 focus 波动/收益**做同样的预测，
  若「已实现波动」和「分钟频结构」效果一样 ⇒ 结构指标是**换皮**，没有增量。
  这是本项目最贵的教训（§八 三个「显著」因子全是旧因子换皮，ρ 0.83~0.97）。

相位对齐（**别搞错，这是最容易出错的一步**）
------------------------------------------
`engine.daily_returns[t]` 与 `pct_change[t]` 同标签：T 日收益由 T-1 收盘后定的
权重赚取。所以 **T 日收盘可得的信号 → 预测 T+1 起的收益**。
`fwd_returns(h=1)` 也已经返回 `T+1 open → T+2 open`，
两侧同相位，**不需要再 shift**。多 shift 一次就变成「用明天信息预测今天」。

用法::

    .venv/Scripts/python.exe -m scripts.evaluation.p0_timing_precheck
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

from scripts.common.cli_common import setup_logging  # noqa: E402,I001
from scripts.evaluation import caliber as cal  # noqa: E402
from scripts.evaluation.caliber import next_day_rebalance, rebalance_dates  # noqa: E402
from scripts.evaluation.pit_intraday_probe import (  # noqa: E402
    OUT,
    load_context as load_daily_ctx,
)
from scripts.evaluation.pit_weekly_probe import fwd_returns  # noqa: E402

log = setup_logging("p0_timing_precheck")

FOCUS_ID = "yqcore-focus-k15"
K = 15


# ----------------------------------------------------------------------
# 数据装配
# ----------------------------------------------------------------------
def build_ctx_421():
    """focus 信号在 **421 池内部**跑（同信号/同执行/同成本），持仓 100% 可算日频特征。"""
    from backtest.engine import build_execution_split
    from strategy.examples import BufferedTopKLongOnly

    ctx = cal.load_context(signal="dingban_fundind_local", window="prod",
                           execution="open", end=20260630)
    feats = pd.read_pickle(OUT / "xs_nl_feats.pkl")
    cols421 = [c for c in next(iter(feats.values())).columns
               if c in ctx.pred.columns]

    m421 = pd.DataFrame(False, index=ctx.mask.index, columns=ctx.mask.columns)
    m421[cols421] = ctx.mask[cols421]

    cost = cal.cost_profile("production")
    sig = ctx.pred.shift(1).where(m421)
    split = build_execution_split(ctx.fill, ctx.close,
                                  set(rebalance_dates(ctx.pred.index, freq="M")))
    rb = next_day_rebalance(ctx.pred.index, freq="M")
    bt = cal.VectorBacktest(strategy=BufferedTopKLongOnly(k=K, exit_mult=2.0),
                            rebalance_freq="M", initial_capital=1e6, costs=cost)
    res = bt.run(sig, ctx.fwd, horizon=1, rebalance_days=rb,
                 execution_split=split, adv_panel=ctx.amt, vol_panel=ctx.vol60)
    return ctx, feats, cols421, res, cost


def structure_series(feats: dict[str, pd.DataFrame], cols: list[str]) -> pd.DataFrame:
    """把日频特征压成**全池横截面聚合的时间序列**（择时用的输入）。

    每个因子取「全池中位数」和「全池标准差」两个聚合 —— 中位数代表「市场状态」，
    标准差代表「分化度」。都只用当日 bar，无前视。
    """
    out = {}
    for name in ("mom_last_bar", "evt_down_tail_share", "vol_extreme_bar_share",
                 "nl_vol_cluster_acf1", "ret_std", "nl_zero_cross_rate"):
        if name not in feats:
            continue
        p = feats[name].reindex(columns=cols)
        out[f"{name}__med"] = p.median(axis=1)
        out[f"{name}__std"] = p.std(axis=1)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# 统计工具
# ----------------------------------------------------------------------
def nw_t(x: pd.Series, lags: int) -> tuple[float, float]:
    """Newey-West 校正的均值 t 值（重叠窗口必用）。返回 (mean, t)。"""
    x = x.dropna()
    n = len(x)
    if n < 30:
        return float("nan"), float("nan")
    mu = float(x.mean())
    e = (x - mu).to_numpy()
    g0 = float((e @ e) / n)
    var = g0
    for k in range(1, min(lags, n - 1) + 1):
        gk = float((e[k:] @ e[:-k]) / n)
        var += 2.0 * (1.0 - k / (lags + 1.0)) * gk
    if var <= 0:
        return mu, float("nan")
    return mu, mu / np.sqrt(var / n)


def _sr(r: pd.Series) -> float:
    """年化 SR（**算术**口径，与本项目 ``_arith`` 一致）。"""
    r = r.dropna()
    s = float(r.std())
    return float(r.mean() / s * np.sqrt(252)) if s > 0 else float("nan")


def _mdd(r: pd.Series) -> float:
    """最大回撤（负数）。"""
    eq = (1 + r.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def _overlay(base: pd.Series, sig: pd.Series, sign: float,
             mode: str) -> tuple[pd.Series, pd.Series]:
    """仓位 overlay：``ret = p·base``（p=0 即空仓）。

    ``binary``：信号高于自身滚动中位数则满仓，否则空仓。
    ``linear``：信号滚动分位映射到 [0,1]。
    🚨 ``sig`` 是 **T 日收盘**值，必须 ``shift(1)`` 才作用到 T+1 的 ``base``。
    """
    q = (sig * sign).rolling(252, min_periods=120).rank(pct=True)
    p = (q > 0.5).astype(float) if mode == "binary" else q.clip(0, 1)
    p = p.shift(1).reindex(base.index).fillna(1.0)
    return p * base, p


def ic_by_day(sig: pd.Series, fwd: pd.Series, min_n: int = 30) -> float:
    """单日 spearman（成对非空）。

    🚨 ``min_n`` 必须按用途给：全池横截面 IC 用 30，**focus 持仓内只有 K=15 只**，
    用 30 会恒返回 NaN（这正是本脚本第一版 T1 一行不输出的原因）。
    """
    m = sig.notna() & fwd.notna()
    if m.sum() < min_n:
        return float("nan")
    return sig[m].corr(fwd[m], method="spearman")


# ----------------------------------------------------------------------
# T1 先行性
# ----------------------------------------------------------------------
def t1_cs(feat: pd.DataFrame, f1: pd.DataFrame, dates: pd.Index,
          scope: str, held: pd.DataFrame, min_n: int, h: int) -> tuple[float, float, int]:
    """``h`` 日累计收益上的逐日横截面 IC 均值 + NW t。

    ``scope='held'`` 只在 focus 持仓内；``'pool'`` 用全 421 池（稳健性旁证）。
    ``h=1`` 直接用 ``f1``（已是 T+1 open→T+2 open，与 T 日收盘信号同相位）；
    ``h>1`` 用 ``shift(-(h-1))`` 拼出 T+1..T+h，**不再多 shift**。
    """
    fwd = f1 if h == 1 else (1 + f1).rolling(h).apply(np.prod, raw=True).shift(-(h - 1)) - 1
    ics = []
    for d in dates:
        if d not in feat.index or d not in fwd.index:
            continue
        if scope == "held":
            nm = list(held.columns[held.loc[d].to_numpy()])
        else:
            nm = list(feat.columns)
        c = ic_by_day(feat.loc[d, nm], fwd.loc[d, nm], min_n=min_n)
        if pd.notna(c):
            ics.append(c)
    if len(ics) < 30:
        return float("nan"), float("nan"), len(ics)
    mu, t = nw_t(pd.Series(ics), lags=max(h * 2, 5))
    return mu, t, len(ics)


def t1_series(sig_ts: pd.Series, ret: pd.Series, horizon: int) -> tuple[float, float]:
    """时间序列版：聚合指标的 T 值 vs 未来 horizon 日累计收益（相关性）。"""
    fwd_cum = (1 + ret).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1
    df = pd.concat([sig_ts.rename("s"), fwd_cum.rename("r")], axis=1).dropna()
    if len(df) < 60:
        return float("nan"), float("nan")
    c = float(df["s"].corr(df["r"]))
    # 重叠窗口 ⇒ 用 NW 校正的回归 t
    x = (df["s"] - df["s"].mean()).to_numpy()
    y = (df["r"] - df["r"].mean()).to_numpy()
    beta = float(x @ y / (x @ x))
    e = y - beta * x
    n = len(x)
    g0 = float((e @ e) / n)
    var = g0
    for k in range(1, min(horizon * 2, n - 1) + 1):
        gk = float((e[k:] @ e[:-k]) / n)
        var += 2.0 * (1.0 - k / (horizon * 2 + 1.0)) * gk
    se = np.sqrt(max(var, 1e-18) / n) / float(np.sqrt((x @ x) / n))
    return c, beta / se if se > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="分钟频择时信息的前置检验")
    ap.add_argument("--max-h", type=int, default=20)
    ap.parse_args()

    print("=" * 72)
    print("前置检验：分钟频结构能否预测 focus 的未来相对表现")
    print("=" * 72)

    ctx, feats, cols421, res, cost = build_ctx_421()
    held = res.weights_history > cal.N_ACTIVE_TOL
    dr = res.daily_returns.dropna()
    _, _c, daily, _ = load_daily_ctx()
    f1 = fwd_returns(daily, [1])[1].reindex(index=ctx.pred.index,
                                           columns=ctx.pred.columns)

    print(f"focus@{len(cols421)} 池：日收益 {len(dr)} 日，"
          f"平均持仓 {held.sum(axis=1).mean():.1f} 只")

    # ---- 目标：持仓组合的次日收益 ----
    # abs = 等权持仓的绝对次日收益；exc = 减去全 421 池等权（即「focus 相对其池子的超额」）
    pool_ew = f1.reindex(columns=cols421).mean(axis=1)
    hold_ew = pd.Series({d: float(f1.loc[d, list(held.loc[d][held.loc[d]].index)]
                                  .dropna().mean())
                         for d in held.index
                         if d in f1.index and held.loc[d].sum() >= 5}).dropna()
    tgt = hold_ew
    exc = (hold_ew - pool_ew.reindex(hold_ew.index)).dropna()
    # 🚨 日频特征只到 2022 起，取交集否则 T1 前半段全 NaN
    idx = exc.index.intersection(hold_ew.index).intersection(f1.index)

    # ================= T1 =================
    print("\n" + "-" * 72)
    print("T1 先行性：单股横截面 IC（信号 T 收盘，收益自 T+1 起）")
    print("   held = focus 持仓内（IC 用 K=15 只，min_n=8）· pool = 全 421 池（min_n=30）")
    print("-" * 72)
    print("  %-22s %5s %6s %9s %8s %6s  %s" % ("因子", "scope", "h", "IC", "t(NW)", "n", "判定"))
    rows = []
    for name in ("mom_last_bar", "evt_down_tail_share", "vol_extreme_bar_share",
                 "ret_std", "nl_vol_cluster_acf1"):
        if name not in feats:
            continue
        p = feats[name].reindex(index=idx, columns=cols421)
        for scope, min_n in (("held", 8), ("pool", 30)):
            for h in (1, 5, 20):
                mu, t, n = t1_cs(p, f1, idx, scope, held, min_n, h)
                if n == 0:
                    continue
                verdict = "✅" if abs(t) > 1.96 else "—"
                print("  %-22s %6s %6d %+9.4f %+8.2f %6d  %s" % (
                    name, scope, h, mu, t, n, verdict))
                rows.append({"factor": name, "scope": scope, "h": h,
                             "ic": mu, "t": t, "n": n})

    # ================= T1b 聚合指标（真正能用来择时的那条）=================
    print("\n" + "-" * 72)
    print("T1b 聚合指标 → 未来 focus 组合收益（时间序列，NW 校正）")
    print("-" * 72)
    agg = structure_series(feats, cols421)
    print("  %-34s %6s %9s %8s  %s" % ("聚合指标", "h", "corr", "t(NW)", "判定"))
    agg_rows = []
    # 两个目标都测：绝对收益（择时总仓位）与 相对 421 池超额（选风格）
    for tgt_name, tgt_s in (("abs", tgt), ("exc", exc)):
        for col in agg.columns:
            for h in (1, 5, 20):
                c, t = t1_series(agg[col].reindex(idx), tgt_s.reindex(idx), h)
                if np.isnan(t):
                    continue
                if abs(t) > 1.96:
                    print("  %-34s %6d %+9.4f %+8.2f  %s %s" % (
                        col, h, c, t, tgt_name, "✅"))
                    agg_rows.append({"agg": col, "tgt": tgt_name, "h": h,
                                     "corr": c, "t": t})
    if not agg_rows:
        print("  （无任何聚合指标过 |t|>1.96）")
    else:
        best = sorted(agg_rows, key=lambda r: -abs(r["t"]))[:5]
        print("\n  过线的前 5 个：")
        for r in best:
            print("    %-30s %-4s h=%-3d corr %+.3f  t %+.2f" % (
                r["agg"], r["tgt"], r["h"], r["corr"], r["t"]))

    # ================= T2 稳定性 =================
    print("\n" + "-" * 72)
    print("T2 稳定性：分年拆（只在 1~2 年有效 = 一次风格，不是信号）")
    print("-" * 72)
    if agg_rows:
        keep = [r for r in sorted(agg_rows, key=lambda r: -abs(r["t"]))[:4]]
        for r in keep:
            s = agg[r["agg"]].reindex(idx)
            y = tgt.reindex(idx) if r["tgt"] == "abs" else exc.reindex(idx)
            df = pd.concat([s.rename("s"), y.rename("y")], axis=1).dropna()
            cells = []
            for yr, g in df.groupby(df.index.year):
                if len(g) < 100:
                    continue
                c = float(g["s"].corr(g["y"]))
                cells.append(f"{yr}:{c:+.3f}")
            print("  %-28s %-4s h=%-3d  %s" % (r["agg"], r["tgt"], r["h"],
                                               "  ".join(cells)))
        print("  （判读：全正/全负且量级相近才算稳定；正负翻转为风格切换）")

    # ================= T4 经济检验（仓位 overlay）=================
    print("\n" + "-" * 72)
    print("T4 经济检验：把结构指标做成仓位 overlay，扣成本后能不能赚钱")
    print("   p_t ∈ [0,1] 由 T 日收盘结构指标定，作用于 T+1 起的 focus 收益")
    print("   只在调仓日改仓位（与 focus 月频一致，避免日频换手成本）")
    print("-" * 72)
    base = dr.reindex(idx).dropna()
    print("  基准 focus@421：年化 %+.2f%%  SR %.3f  MDD %.2f%%" % (
        100 * ((1 + base).prod() ** (252 / len(base)) - 1),
        _sr(base), 100 * _mdd(base)))
    if agg_rows:
        for r in sorted(agg_rows, key=lambda r: -abs(r["t"]))[:4]:
            s = agg[r["agg"]].reindex(idx)
            y = tgt.reindex(idx) if r["tgt"] == "abs" else exc.reindex(idx)
            sign = 1.0 if r["corr"] > 0 else -1.0
            for mode in ("binary", "linear"):
                ov = _overlay(base, s, sign, mode)
                ret, pos = ov
                print("  %-26s %-4s %-6s 年化 %+7.2f%%  SR %6.3f  MDD %6.2f%%  "
                      "平均仓位 %.2f" % (
                          r["agg"], r["tgt"], mode,
                          100 * ((1 + ret).prod() ** (252 / len(ret)) - 1),
                          _sr(ret), 100 * _mdd(ret), pos.mean()))
        print("  （判读：SR 不显著高于基准 %0.3f ⇒ 该指标无经济价值）"
              % _sr(base))
    print("\n" + "-" * 72)
    print("T3 换皮对照：用「同期已实现的 focus 波动」做同样的预测")
    print("   若已实现波动 ≥ 分钟频结构，则结构指标无增量")
    print("-" * 72)
    rv = tgt.rolling(20).std().shift(1)
    for tgt_name, tgt_s in (("abs", tgt), ("exc", exc)):
        for h in (5, 20):
            c, t = t1_series(rv.reindex(idx), tgt_s.reindex(idx), h)
            print("  %-30s %-4s %6d %+9.4f %+8.2f  %s" % (
                "realized_vol_20d", tgt_name, h, c, t,
                "✅" if abs(t) > 1.96 else "—"))

    pd.DataFrame(rows).to_csv(OUT / "p0_precheck_ic.csv", index=False,
                              encoding="utf-8-sig")
    pd.DataFrame(agg_rows).to_csv(OUT / "p0_precheck_agg.csv", index=False,
                                  encoding="utf-8-sig")
    print("\n已写 p0_precheck_ic.csv / p0_precheck_agg.csv")


if __name__ == "__main__":
    main()