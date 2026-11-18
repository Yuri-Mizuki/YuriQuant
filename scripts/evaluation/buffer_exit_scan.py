"""缓冲带（``exit_mult``）× K 二维扫描：成本能省多少，代价是多少

背景
----
`reports/focus_riskbudget/` 已把两条降回撤的路堵死：
① 个股配权（inv_vol / risk_parity）—— 有效独立注数只有 1.9，配对检验不显著；
② 整仓缩放（vol_target）—— 纯杠杆旋钮，SR/仓位单调下降。

⇒ 唯一有**确定机制**的漏点是成本：K=8 毛年化 40.45% → 净 30.16%，
**成本吃掉 25%**（10.29%/年，换手 10.56/年）。降低换手不需要统计功效，
省下 1pp 就是 1pp 净收益。这是「确定性减法」，与「预测」是两类问题。

``exit_mult`` 是什么
--------------------
``BufferedTopKLongOnly`` 的离场门槛倍数。每期：
    保留 = 已持仓中 排名 ≤ K×exit_mult 的
    新进 = 未持仓中 排名 ≤ K 的（**进场门槛恒为 K，不受 exit_mult 影响**）
⇒ ``exit_mult`` 只调**离场侧黏性**。1.0 = 硬切 TopK；2.0 = 现行生产值。

判据（**必须三项同报，只看成本必错**）
--------------------------------------
降换手本身一定「好看」，但可能是**毛收益掉得更多**换来的。故每格同时报：
    ① 净年化 / SR（生产 27bp）
    ② 毛年化 + 成本（同参数零成本重跑，成本 = 毛 − 净的可分解量）
    ③ 换手
并做**配对检验**（逐日收益差 vs 现行 K×2.0），不显著即不得声称改进。

🚨 相位铁律：定版收益是 phase 0 单点，**21 相位均值才可期望**。本脚本先跑 phase 0
探方向，报告里引用的任何数字必带「phase 0」字样。

用法::

    .venv/Scripts/python.exe -m scripts.evaluation.buffer_exit_scan --capital 1e5
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

log = setup_logging("buffer_exit_scan")

OUT = ROOT / "reports" / "buffer_exit_scan"


def main() -> None:
    ap = argparse.ArgumentParser(description="缓冲带 exit_mult × K 扫描")
    ap.add_argument("--capital", type=float, default=1e5)
    ap.add_argument("--ks", type=str, default="5,8,15",
                    help="逗号分隔的 K（默认 5,8,15：小资金 / 中资金 / 定版档）")
    ap.add_argument("--mults", type=str, default="1.0,1.5,2.0,3.0,4.0",
                    help="逗号分隔的 exit_mult")
    ap.add_argument("--end", type=int, default=20260907)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    mults = [float(x) for x in args.mults.split(",") if x.strip()]

    cost = cal.cost_profile("production")
    # 真实零成本对照（factor_cost=False 会把 commission_min 也置零，2026-10-19 审计 P2）
    from backtest.costs import default_costs
    zero = default_costs(factor_cost=False)

    ctx = cal.load_context(signal="gbdt_mid13", window="prod",
                           execution="open", end=args.end)
    from strategy.examples import BufferedTopKLongOnly

    rows: list[dict] = []
    daily: dict[str, pd.Series] = {}

    for k in ks:
        for mult in mults:
            tag = f"k{k}_x{mult:g}"
            # 每次必须是全新实例：有状态策略跨回测复用会把上期末持仓泄漏为前视
            bt = cal.VectorBacktest(
                strategy=BufferedTopKLongOnly(k=k, exit_mult=mult),
                rebalance_freq="M", initial_capital=args.capital, costs=cost)
            res = cal.run_once(bt, ctx, "open", freq="M",
                               price_panel=ctx.close, lot_size=100)
            m = cal.metrics_row(res, args.capital)
            dr_net = res.daily_returns.dropna()

            # 同参数零成本重跑 ⇒ 毛收益与成本可分解（不做「净+成本」的算术拼接，
            # 因为成本是在复利路径上逐期扣的，几何口径必须重跑才准）
            bt0 = cal.VectorBacktest(
                strategy=BufferedTopKLongOnly(k=k, exit_mult=mult),
                rebalance_freq="M", initial_capital=args.capital, costs=zero)
            res0 = cal.run_once(bt0, ctx, "open", freq="M",
                                price_panel=ctx.close, lot_size=100)
            m0 = cal.metrics_row(res0, args.capital)

            rows.append({
                "k": k, "exit_mult": mult, "tag": tag,
                "ann_net": m["ann"], "sharpe": m["sharpe"],
                "sharpe_arith": m["sharpe_arith"], "max_dd": m["max_dd"],
                "turnover": m["turnover"], "cost_bp": m["cost_bp"],
                "ann_gross": m0["ann"], "sharpe_gross": m0["sharpe"],
                "turnover_gross": m0["turnover"],
                "n_days": m["n_days"],
            })
            daily[tag] = dr_net
            log.info("%-10s 净 %.2f%% / SR %.4f / MDD %.2f%% / 换手 %.3f | 毛 %.2f%%",
                     tag, 100 * m["ann"], m["sharpe"], 100 * m["max_dd"],
                     m["turnover"], 100 * m0["ann"])

    df = pd.DataFrame(rows)
    df["cost_ann"] = df["ann_gross"] - df["ann_net"]

    # ---- 配对检验：每格 vs 同 K 的现行基线（exit_mult=2.0）----
    from scipy import stats
    prows = []
    for k in ks:
        base_tag = f"k{k}_x2"
        if base_tag not in daily:
            continue
        b = daily[base_tag]
        for mult in mults:
            tag = f"k{k}_x{mult:g}"
            if tag == base_tag or tag not in daily:
                continue
            j = pd.concat([daily[tag].rename("a"), b.rename("b")], axis=1).dropna()
            diff = j["a"] - j["b"]
            t, p = stats.ttest_1samp(diff, 0)
            prows.append({"k": k, "exit_mult": mult, "vs": "x2.0",
                          "mean_diff": float(diff.mean()), "t": float(t),
                          "p": float(p), "n": int(len(diff))})
    dpair = pd.DataFrame(prows)

    # ---- 输出 ----
    for k in ks:
        sub = df[df["k"] == k].sort_values("exit_mult")
        print("\n" + "=" * 92)
        print(f"K={k} · 资金 {args.capital:,.0f} · 整手100 · production 27bp · phase 0 单点")
        print("=" * 92)
        print("  %-6s %9s %9s %9s %8s %8s %9s %9s" % (
            "mult", "净年化", "毛年化", "成本/年", "SR", "MDD", "换手", "成本占比"))
        for _, r in sub.iterrows():
            cost_share = r["cost_ann"] / r["ann_gross"] if r["ann_gross"] > 0 else np.nan
            print("  %-6.1f %+8.2f%% %+8.2f%% %+8.2f%% %8.4f %+7.2f%% %9.3f %8.1f%%" % (
                r["exit_mult"], 100 * r["ann_net"], 100 * r["ann_gross"],
                100 * r["cost_ann"], r["sharpe"], 100 * r["max_dd"],
                r["turnover"], 100 * cost_share))

    if len(dpair):
        print("\n" + "=" * 92)
        print("  配对检验 vs 同 K 的 exit_mult=2.0（现行生产值）")
        print("=" * 92)
        print("  %-4s %-6s %11s %9s %8s  %s" % ("K", "mult", "日均差", "配对t", "p", "判定"))
        for _, r in dpair.iterrows():
            v = "✅ 显著" if r["p"] < 0.05 else "—"
            print("  %-4d %-6.1f %+10.5f%% %+9.3f %8.4f  %s" % (
                r["k"], r["exit_mult"], 100 * r["mean_diff"], r["t"], r["p"], v))

    # ---- 成本弹性：每多 1 单位 exit_mult 省多少成本、丢多少毛 ----
    print("\n" + "=" * 92)
    print("  成本弹性：exit_mult 从 2.0 继续放松，省的成本 vs 丢的毛收益")
    print("=" * 92)
    print("  %-4s %-6s %11s %11s %11s  %s" % (
        "K", "mult", "Δ成本/年", "Δ毛/年", "Δ净/年", "净效应"))
    for k in ks:
        b = df[(df["k"] == k) & (df["exit_mult"] == 2.0)]
        if not len(b):
            continue
        b = b.iloc[0]
        for mult in mults:
            if mult == 2.0:
                continue
            r = df[(df["k"] == k) & (df["exit_mult"] == mult)]
            if not len(r):
                continue
            r = r.iloc[0]
            d_cost = r["cost_ann"] - b["cost_ann"]
            d_gross = r["ann_gross"] - b["ann_gross"]
            d_net = r["ann_net"] - b["ann_net"]
            verdict = "省成本 > 丢毛 ⇒ 净改善" if d_net > 0 else "丢毛 > 省成本 ⇒ 净受损"
            print("  %-4d %-6.1f %+10.2f%% %+10.2f%% %+10.2f%%  %s" % (
                k, mult, 100 * d_cost, 100 * d_gross, 100 * d_net, verdict))

    df.to_csv(OUT / "buffer_exit_metrics.csv", index=False, encoding="utf-8-sig")
    if len(dpair):
        dpair.to_csv(OUT / "buffer_exit_paired.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(daily).to_csv(OUT / "buffer_exit_daily.csv", encoding="utf-8-sig")
    print(f"\n已写 {OUT.name}/buffer_exit_{{metrics,paired,daily}}.csv")


if __name__ == "__main__":
    main()