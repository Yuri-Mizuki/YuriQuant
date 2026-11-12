"""行业去重的增益在多相位下是否还在 —— 以及哪个变体 RTL 最小。

回答 2026-11-10 的追问：`rtl_rebalance_timing` 已证明「35.57% 是 21 相位最大值」，
那么**建立在单点上的所有增益结论都得复核**：
1. 行业去重（``max_per_sector=1``）的 **+6.48pp** 在多相位平均下还在吗？
   还是也是「抽到好签」的产物？
2. 还有别的办法能拿到增益吗（`exit_mult` 缓冲带 / `max_per_sector` 放宽）？
3. 哪个变体的 **RTL 最小**（= 最不依赖调仓日运气）？

为什么必须「配对」而不是「比两个均值」
--------------------------------------
相位是一个**共同冲击**：同一个月里不同相位吃到的市场段不同，所有臂在同相位上
共享这份噪声。直接比「m1 的 21 相位均值」与「plain 的 21 相位均值」会把这份
共同噪声算进误差、把检验功效吃掉。正确估计量是**逐相位配对差** ``d_p = ann_A(p) − ann_B(p)``，
再对 ``{d_p}`` 取均值 ± 标准误。本脚本主表给配对差，臂均值只作展示。

口径
----
production：固定 10bp / T+1 open / 整手 100 / 资金 1e5 / 窗口 2019-01-02..2026-09-07
/ 信号 ``gbdt_mid13`` / Sharpe **算术**。相位 0~20（21 个），每相位 93 期。

⚠️ **``phase=19/20`` 不可靠**：每月交易日中位数只有 21 天，``rebalance_dates`` 在
「该月交易日不足 phase+1」时会**回退到月末**（防丢月），实测 phase=20 与 phase=19
有 43/93 个月撞同一天 ⇒ 后段相位不是独立抽样，会**低估** RTL。主表因此并列
「phase ≤ 15」子样本（该段无月份回退，撞日 ≤ 5/93）。

输出 ``reports/rtl_rebalance_timing/``
--------------------------------------
  variant_grid.csv      10 臂 × 21 相位汇总（均值/中位/std/极差/文献口径 RTL）
  variant_phase.csv     长表：每臂每相位明细
  variant_paired.csv    关键对照的**逐相位配对差**（含 t）

用法::

    python -m scripts.evaluation.rtl_variant_probe            # 全跑（约 5 分钟）
    python -m scripts.evaluation.rtl_variant_probe --phases 0,1,2   # 抽样调试
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.evaluation.caliber as cal
from research.benchmarks import segmented_equal_weight_returns
from scripts.evaluation.solo_freq_probe import (
    CAPITAL,
    END,
    LOT,
    industry_panel,
    price_panel,
    stats,
)
from strategy.cross_section import BufferedTopKLongOnly, SectorCappedBufferedTopKLongOnly

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "rtl_rebalance_timing"
N_PHASE = 21
#: 该相位及以前**不会**触发「月内交易日不足 ⇒ 回退月末」（实测撞日 ≤ 5/93）
SAFE_PHASE = 15

#: (臂名, 每行业上限（None = 不加行业约束）, k, exit_mult)
#: ``plain_k6`` 是**持仓数匹配对照** —— solo 实持 5.9 只，与 plain k8（8 只）比会
#: 把「行业去重」和「持仓更少」混在一起，必须再加一个「实持≈6 只」的对照。
ARMS: list[tuple[str, int | None, int, float]] = [
    ("plain_k6_b1.0", None, 6, 1.0),
    ("plain_k8_b1.0", None, 8, 1.0),
    ("plain_k8_b2.0", None, 8, 2.0),
    ("m1_k8_b1.0", 1, 8, 1.0),
    ("m2_k8_b1.0", 2, 8, 1.0),
    ("m3_k8_b1.0", 3, 8, 1.0),
    ("m1_k8_b2.0", 1, 8, 2.0),
    ("m2_k8_b2.0", 2, 8, 2.0),
    ("m1_k8_b3.0", 1, 8, 3.0),
    ("m2_k8_b3.0", 2, 8, 3.0),
]

#: 配对对照（A, B, 备注）—— 逐相位做 ``A − B``
PAIRS: list[tuple[str, str, str]] = [
    ("m1_k8_b1.0", "plain_k8_b1.0", "行业去重（同参 k8 b1.0）"),
    ("m1_k8_b1.0", "plain_k6_b1.0", "行业去重 vs 持仓数匹配对照"),
    ("m2_k8_b1.0", "plain_k8_b1.0", "放宽到每行业 2 只 vs 无约束"),
    ("m1_k8_b1.0", "m2_k8_b1.0", "去重松一格（1→2）"),
    ("m2_k8_b1.0", "m3_k8_b1.0", "去重松一格（2→3）"),
    ("m1_k8_b2.0", "m1_k8_b1.0", "加缓冲带 b1.0→2.0（去重臂）"),
    ("m2_k8_b2.0", "m2_k8_b1.0", "加缓冲带 b1.0→2.0（m2 臂）"),
    ("plain_k8_b2.0", "plain_k8_b1.0", "加缓冲带 b1.0→2.0（无约束臂）"),
    ("m1_k8_b3.0", "m1_k8_b2.0", "缓冲带 b2.0→3.0"),
]


def _max_dd(dr: pd.Series) -> float:
    eq = (1 + dr).cumprod()
    return float((eq / eq.cummax() - 1).min())


def annualized(dr: pd.Series) -> float:
    n_y = len(dr) / 252.0
    return float((1 + dr).prod() ** (1 / n_y) - 1)


def _sr(dr: pd.Series) -> float:
    return float(dr.mean() / dr.std(ddof=1) * np.sqrt(252))


def build_arm(arm: tuple[str, int | None, int, float], ind: pd.DataFrame):
    _, m, k, buf = arm
    if m is None:
        return BufferedTopKLongOnly(k=k, exit_mult=buf)
    return SectorCappedBufferedTopKLongOnly(ind, k=k, exit_mult=buf, max_per_sector=m)


def _hold_sets(res, exec_days: list[pd.Timestamp]) -> dict:
    """执行日当天的持仓集合（权重 > 1e-9）。"""
    wh = res.weights_history
    out = {}
    for d in exec_days:
        if d not in wh.index:
            continue
        row = wh.loc[d]
        out[d] = set(row.index[row.to_numpy() > 1e-9])
    return out


def _adj_jaccard(by_phase: dict[int, dict]) -> float:
    """相邻相位之间、同月份的持仓 Jaccard 均值。"""
    vals = []
    phases = sorted(by_phase)
    for a, b in zip(phases, phases[1:]):
        ma = {d.to_period("M"): s for d, s in by_phase[a].items()}
        mb = {d.to_period("M"): s for d, s in by_phase[b].items()}
        for per in set(ma) & set(mb):
            x, y = ma[per], mb[per]
            if not x and not y:
                continue
            vals.append(len(x & y) / len(x | y))
    return float(np.mean(vals)) if vals else float("nan")


def main(phases: list[int]) -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = cal.load_context("gbdt_mid13", window="prod", end=END, execution="open")
    cost = cal.cost_profile("production")
    px = price_panel(ctx)
    ind = industry_panel(ctx)
    full_close = pd.read_parquet(cal.BASE_DIR / "close_adj.parquet")
    print(f"窗口 {ctx.pred.index[0].date()}..{ctx.pred.index[-1].date()} | "
          f"资金 {CAPITAL:,.0f} | 整手 {LOT} | 臂 {len(ARMS)} × 相位 {len(phases)}",
          flush=True)

    # ---- 等权基准（与臂无关，只算一遍）----
    eqw = {}
    for p in phases:
        rb = cal.rebalance_dates(ctx.pred.index, freq="M", phase=p)
        s = segmented_equal_weight_returns(full_close, rebalance_dates=rb,
                                           index=ctx.pred.index)
        eqw[p] = annualized(s)

    long_rows, daily, holds, arm_sum = [], {}, {}, {}
    for arm in ARMS:
        name = arm[0]
        ta = time.time()
        daily[name], holds[name], rows = {}, {}, []
        for p in phases:
            n_reb = len(cal.rebalance_dates(ctx.pred.index, freq="M", phase=p))
            bt = cal.VectorBacktest(strategy=build_arm(arm, ind), rebalance_freq="M",
                                    initial_capital=CAPITAL, costs=cost)
            res = cal.run_once(bt, ctx, "open", freq="M", phase=p,
                               price_panel=px, lot_size=LOT)
            r = stats(res, n_reb)
            daily[name][p] = res.daily_returns
            exec_days = sorted(cal.next_day_rebalance(ctx.pred.index, "M", p))
            holds[name][p] = _hold_sets(res, exec_days)
            rows.append({"arm": name, "phase": p, "ann": r["ann"],
                         "sr": r["sharpe_arith"], "dd": r["max_dd"],
                         "vol": r["vol"], "nhold": r["nhold_mean"],
                         "invest": r["invest_mean"], "turnover_ann": r["turnover_ann"],
                         "eqw_ann": eqw[p], "excess": r["ann"] - eqw[p]})
        df = pd.DataFrame(rows)
        long_rows.append(df)

        # 叠加（= 该臂 21 腿 tranching，无选择偏差的估计量）
        R = pd.DataFrame(daily[name])
        blend = R.mean(axis=1)
        # 文献口径 RTL：相对**本臂叠加组合**的月度跟踪误差年化
        te = {}
        for p in R.columns:
            mon = (1 + (R[p] - blend)).resample("ME").prod() - 1
            te[p] = float(mon.std(ddof=1) * np.sqrt(12))
        te_s = pd.Series(te)

        safe = df[df["phase"] <= SAFE_PHASE]
        arm_sum[name] = {
            "ann_mean": df["ann"].mean(), "ann_median": df["ann"].median(),
            "ann_std": df["ann"].std(ddof=1),
            "ann_range": df["ann"].max() - df["ann"].min(),
            "ann_min": df["ann"].min(), "ann_max": df["ann"].max(),
            "sr_mean": df["sr"].mean(), "dd_mean": df["dd"].mean(),
            "excess_mean": df["excess"].mean(),
            "blend_ann": annualized(blend), "blend_sr": _sr(blend), "blend_dd": _max_dd(blend),
            "rtl_te": te_s.mean(),
            "rtl_te_safe": safe["ann"].std(ddof=1),
            "ann_range_safe": safe["ann"].max() - safe["ann"].min(),
            "nhold_mean": df["nhold"].mean(), "invest_mean": df["invest"].mean(),
            "turnover_ann": df["turnover_ann"].mean(),
            "jaccard_adj": _adj_jaccard(holds[name]),
            "rtl_over_vol": te_s.mean() / df["vol"].mean(),
        }
        g = arm_sum[name]
        print(f"  {name:<14} 均值 {g['ann_mean'] * 100:6.2f}%  中位 {g['ann_median'] * 100:6.2f}% "
              f"std {g['ann_std'] * 100:5.2f}%  极差 {g['ann_range'] * 100:5.2f}pp  "
              f"RTL {g['rtl_te'] * 100:5.2f}%  实持 {g['nhold_mean']:4.1f}  "
              f"Jaccard {g['jaccard_adj']:.3f}  [{time.time() - ta:.0f}s]",
              flush=True)

    grid = pd.DataFrame(arm_sum).T
    grid.index.name = "arm"
    grid.to_csv(OUT / "variant_grid.csv", encoding="utf-8-sig")
    long = pd.concat(long_rows, ignore_index=True)
    long.to_csv(OUT / "variant_phase.csv", index=False, encoding="utf-8-sig")

    # ---- 配对差（主结论用这个，不用臂均值的差）----
    wide = long.pivot(index="phase", columns="arm", values="ann")
    prows = []
    print("\n=== 逐相位配对差（均值 ± SE，t 用 √n 近似；相位间有相关性 ⇒ 偏乐观）===",
          flush=True)
    for a, b, note in PAIRS:
        d = (wide[a] - wide[b]).dropna()
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else np.nan
        safe = d[d.index <= SAFE_PHASE]
        prows.append({"a": a, "b": b, "note": note, "n": len(d),
                      "diff_mean": d.mean(), "diff_se": se, "t": t,
                      "n_pos": int((d > 0).sum()), "diff_min": d.min(),
                      "diff_max": d.max(), "diff_safe_mean": safe.mean(),
                      "n_pos_safe": int((safe > 0).sum()), "n_safe": len(safe)})
        print(f"  {note:<28} {a:>12} − {b:<12} = {d.mean() * 100:+6.2f}pp "
              f"± {se * 100:4.2f} (t={t:5.1f})  正 {int((d > 0).sum()):2d}/{len(d)}  "
              f"[≤{SAFE_PHASE}: {safe.mean() * 100:+6.2f}pp "
              f"正 {int((safe > 0).sum())}/{len(safe)}]", flush=True)
    pd.DataFrame(prows).to_csv(OUT / "variant_paired.csv", index=False,
                               encoding="utf-8-sig")

    print(f"\n总耗时 {time.time() - t0:.0f}s  产物 → {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="", help="逗号分隔，缺省 0..20 全扫")
    args = ap.parse_args()
    main([int(x) for x in args.phases.split(",")] if args.phases
         else list(range(N_PHASE)))