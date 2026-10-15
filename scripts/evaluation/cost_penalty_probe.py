"""成本感知优化探针（P0-2，2026-10-13）。

把 ``optimize.solver`` 里「默认 0、从未校准」的 ``turnover_penalty``（线性冲击）与
``quadratic_cost``（二次冲击，A-C 型）填上由真实 ADV/波动率推来的值，在**真实 882
回测口径**上做两件事：

1. **定标表**（廉价，必做）：由逐股 ``c_i = η·σ_i·√(capital/ADV_i)`` 汇出
   ``turnover_penalty=a`` / ``quadratic_cost=b=a/x*`` 的理论起点，扫多个资金规模。
2. **真实回测对照**（较重）：逐调仓日取 Top-N 候选 → 单日 QP（``solve_portfolio``，
   带 ``prev_weights`` 使换手惩罚真正生效）→ 权重表喂回 ``PrecomputedWeightsStrategy``
   + ``VectorBacktest``，比较**净年化 / 换手 / 成本**。

⚠️ 为什么逐日切片：``optimize_weights_qp`` 的面板入口对**全列**建 Σ，Top-N 并集可膨胀到
数千列 ⇒ ~40s/日（2026-10-13 实测），全期扫描不可行；单日 ``solve_portfolio`` 在
Top-300 下仅 ~2s。故本脚本自行循环调仓日、逐日限集。

🚨 **2026-10-13 起，加载 / 成本 / 指标 / 调仓日一律走 :mod:`scripts.evaluation.caliber`**
（本项目**统一评估口径**的唯一真源）。本脚本只保留「定标表 + 逐日 QP 权重表」这两个
探针专属算法。历史上正是因为在自家重写这些原语，才有过 5 处口径不同源 ⇒ 结论三次反转。
口径与边界见 ``reports/docs/design/定版主模型与统一口径.md``。

用法::

    # 只出定标表（秒级）
    D:/python/Python312/python.exe scripts/evaluation/cost_penalty_probe.py --calib-only

    # 定标表 + 真实回测对照（~10 分钟，本机）
    D:/python/Python312/python.exe scripts/evaluation/cost_penalty_probe.py \\
        --begin 20180101 --end 20260901 --n-names 300 --scales 0,1,2 --capital 1e8
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

from backtest.metrics import annual_return  # noqa: E402
from optimize.cost_penalty import calibrate_cost_penalty  # noqa: E402
from optimize.solver import rolling_covariance, solve_portfolio  # noqa: E402
from scripts.evaluation import caliber  # noqa: E402
from scripts.evaluation.caliber import (  # noqa: E402
    WINDOWS,
    CaliberContext,
    cost_profile,
    evaluate_strategy,
    evaluate_weights,
    load_context,
    rebalance_dates,
    sharpe_pair,
)
from strategy.examples import BufferedTopFracLongOnly, TopFracLongOnly  # noqa: E402

# 口径收敛：以下别名仅供既有测试/外部引用（tests/test_cost_penalty_aim.py）。
_sharpe_pair = sharpe_pair
_rebalance_dates = rebalance_dates


def ann(dr: pd.Series) -> float:
    dr = dr.dropna()
    return float(annual_return(dr)) if len(dr) else float("nan")


# ---------------------------------------------------------------------------
# 1) 定标表
# ---------------------------------------------------------------------------
def calibration_table(amt: pd.DataFrame, ret: pd.DataFrame, capitals,
                      eta: float, n_names: int, x_star: float | None) -> pd.DataFrame:
    adv_mean = amt.mean(axis=0).to_numpy(dtype=float)
    vol_mean = ret.rolling(60, min_periods=20).std().mean(axis=0).to_numpy(dtype=float)
    rows = []
    for cap in capitals:
        cal = calibrate_cost_penalty(adv_mean, vol_mean, float(cap), x_star=x_star,
                                     eta=eta, n_names=n_names)
        rows.append({"capital": float(cap), **cal})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2) 逐调仓日 QP 权重表
# ---------------------------------------------------------------------------
def qp_weight_panel(pred, ret, mask, *, n_names: int, k1: float, k2: float,
                    max_weight: float, window: int, min_periods: int,
                    method: str = "min_var", risk_aversion: float = 1.0,
                    max_turnover: float | None = None,
                    alpha_scale: float | str | None = None) -> pd.DataFrame:
    """逐调仓日单期 QP 权重表。

    ``ret`` **必须是不截断到回测窗口**的收益面板（:attr:`CaliberContext.ret_hist`）——
    否则窗口最早的调仓日历史不足 ⇒ ``rolling_covariance`` 返回 ``None`` ⇒ 权重置 0、
    组合静默空仓（2026-10-13 实测 −2.7pp/年）。本函数对跳过的期数**打印告警**。
    """
    reb = rebalance_dates(pred.index)
    codes_all = pred.columns
    rows, prev = [], None
    n_skip_empty = n_skip_hist = 0
    for t in reb:
        row = pred.loc[t]
        if row.notna().sum() == 0:
            rows.append(pd.Series(0.0, index=codes_all))
            prev = None
            n_skip_empty += 1
            continue
        m = mask.loc[t].reindex(row.index).fillna(True)
        cand = row.where(m).dropna().nlargest(n_names)
        if cand.empty:
            rows.append(pd.Series(0.0, index=codes_all))
            prev = None
            n_skip_empty += 1
            continue
        codes = cand.index
        Sigma = rolling_covariance(ret.reindex(columns=codes), t, window=window,
                                   min_periods=min_periods)
        if Sigma is None:
            rows.append(pd.Series(0.0, index=codes_all))
            prev = None
            n_skip_hist += 1
            continue
        pw = prev.reindex(codes).fillna(0.0) if prev is not None else None
        w = solve_portfolio(cand, Sigma, method=method, risk_aversion=risk_aversion,
                            alpha_scale=alpha_scale,
                            max_weight=max_weight, prev_weights=pw,
                            max_turnover=max_turnover,
                            turnover_penalty=k1, quadratic_cost=k2)
        w = w.reindex(codes_all).fillna(0.0)
        rows.append(w)
        prev = w
    if n_skip_hist:
        print(f"  🚨 警告：{n_skip_hist}/{len(reb)} 个调仓日因**协方差历史不足**被置 0 权重"
              f"（组合空仓）——传入的 `ret` 面板是否被截断到回测窗口？"
              f"应改用 CaliberContext.ret_hist（2026-10-13 实测该缺陷压低约 2.7pp/年）")
    if n_skip_empty:
        print(f"  ℹ️ {n_skip_empty}/{len(reb)} 个调仓日无可用候选（pred/掩码全缺）→ 空仓")
    return pd.DataFrame(rows, index=reb, columns=codes_all).fillna(0.0)


def eqw_topk_panel(pred, mask, k: int) -> pd.DataFrame:
    """等权 Top-K 权重面板（**与 QP 臂同候选口径**：同 pred、同掩码、同调仓日）。

    用途 = 分离两个机制：QP 落后等权，到底是①**集中度**（少数股权重高 ⇒ 平方根律
    冲击贵）还是②**优化器分配**（弱 alpha 项把仓位配到低收益名字上）。
    """
    reb = rebalance_dates(pred.index)
    codes_all = pred.columns
    rows = []
    for t in reb:
        row = pred.loc[t]
        m = mask.loc[t].reindex(row.index).fillna(True)
        cand = row.where(m).dropna().nlargest(k)
        w = pd.Series(0.0, index=codes_all)
        if not cand.empty:
            w.loc[cand.index] = 1.0 / len(cand)
        rows.append(w)
    return pd.DataFrame(rows, index=reb, columns=codes_all).fillna(0.0)


def run_precomputed(target_w: pd.DataFrame, ctx: CaliberContext, *, costs,
                    capital: float) -> dict:
    """薄封装 → :func:`caliber.evaluate_weights`（保留旧名与旧返回键）。"""
    return evaluate_weights(target_w, ctx, costs, capital)


# ---------------------------------------------------------------------------
# 3) CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--begin", type=int, default=None,
                    help="回测起点 YYYYMMDD；缺省 = 生产定版窗口 20180101")
    ap.add_argument("--end", type=int, default=20260901)
    ap.add_argument("--signal", default="ens_h1h5",
                    help="信号源（caliber.SIGNALS 的键）")
    ap.add_argument("--capital", type=float, default=1e8)
    ap.add_argument("--capitals", default="1e7,3e7,1e8,3e8,1e9",
                    help="定标表扫的资金规模")
    ap.add_argument("--n-names", type=int, default=300,
                    help="逐调仓日 QP 候选池上限（Top-N by pred）")
    ap.add_argument("--scales", default="0,1,2",
                    help="k1_scale 扫描；0=无成本惩罚基线")
    ap.add_argument("--eta", type=float, default=None,
                    help="冲击系数 η；缺省取 Config.costs（impact_eta）")
    ap.add_argument("--buy-mult", type=float, default=None)
    ap.add_argument("--max-weight", type=float, default=0.1)
    ap.add_argument("--method", default="min_var",
                    help="QP 方法：min_var（良态，无 alpha 项）/ mvo / tev。"
                         "⚠️ mvo/tev 在 alpha_scale=none 下量级失配 ⇒ 顶点解 bang-bang"
                         "（2026-10-13 实测）；做「风险-收益权衡」须同时给 --alpha-scale auto")
    ap.add_argument("--alpha-scale", default="none",
                    help="α 项尺度：none（默认，零回归）/ auto（自动标定）/ 数值。"
                         "仅 mvo/tev 生效")
    ap.add_argument("--risk-aversion", type=float, default=1.0)
    ap.add_argument("--max-turnover", type=float, default=None,
                    help="组合级换手硬约束（0.5 = 项目对照脚本的做法）")
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--min-periods", type=int, default=60)
    ap.add_argument("--buffer", default="0.15:0.40",
                    help="缓冲带等权臂（生产定版形态，默认 0.15:0.40 —— §十五.1："
                         "任何「QP vs 基线」对照都必须含 buffer 臂；传空串关闭）。"
                         "如 `0.15:0.40`，逗号分隔多组。用 `BufferedTopFrac"
                         "LongOnly` 跑**同一逐股冲击成本口径**")
    ap.add_argument("--eqw-topk", default="",
                    help="等权 Top-K 对照臂（逗号分隔，如 300,21），与 QP 臂**同 pred / "
                         "同掩码 / 同调仓日**")
    ap.add_argument("--calib-only", action="store_true")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()
    alpha_scale: float | str | None
    if args.alpha_scale == "none":
        alpha_scale = None
    elif args.alpha_scale == "auto":
        alpha_scale = "auto"
    else:
        alpha_scale = float(args.alpha_scale)

    begin = args.begin if args.begin is not None else int(WINDOWS["prod"])
    window = "iso2019" if begin >= 20190101 else "prod"
    ctx = load_context(signal=args.signal, window=window, begin=begin, end=args.end)
    print(caliber.describe(ctx, "impact"))

    # --- 1) 定标表 ---
    capitals = [float(x) for x in args.capitals.split(",") if x.strip()]
    eta = float(args.eta) if args.eta is not None else caliber.Config.costs()["impact_eta"]
    calib = calibration_table(ctx.amt, ctx.fwd, capitals, eta=eta,
                              n_names=args.n_names, x_star=None)
    out = ROOT / "reports" / "cost_penalty_probe"
    out.mkdir(parents=True, exist_ok=True)
    suf = f"_{args.out_tag}" if args.out_tag else ""
    calib.to_csv(out / f"calibration{suf}.csv", index=False)
    print("\n=== 定标表（η=%.2f, x*=1/N=%d只）===" % (eta, args.n_names))
    print(calib.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    if args.calib_only:
        print(f"\n产物: {out / f'calibration{suf}.csv'}")
        return

    # --- 2) 真实回测对照 ---
    c = calibrate_cost_penalty(
        ctx.amt.mean(axis=0).to_numpy(dtype=float),
        ctx.fwd.rolling(60, min_periods=20).std().mean(axis=0).to_numpy(dtype=float),
        args.capital, eta=eta, n_names=args.n_names)
    k1_c, k2_c = c["k1"], c["k2"]
    print("\n=== 定标起点（capital=%.0f）: k1=%.6g, k2=%.6g, c̄=%.6g ==="
          % (args.capital, k1_c, k2_c, c["c_mean"]))

    costs = cost_profile("impact", eta=eta, buy_mult=args.buy_mult)
    eq_costs = cost_profile("flat")

    rows = []
    # 等权基线：同时报「固定 10bp」（项目历史假设）与「逐股冲击」（公平对照口径）
    for tag, cc in (("eqw_top10pct_fixed10bp", eq_costs),
                    ("eqw_top10pct_impact", costs)):
        m = evaluate_strategy(TopFracLongOnly(frac=0.10), ctx, cc, args.capital)
        rows.append({"arm": tag, "k1_scale": np.nan, "turnover_penalty": 0.0,
                     "quadratic_cost": 0.0, "alpha_scale": "", **m})
        print("=== %-24s 年化 %.2f%% Sharpe %.3f 换手 %.3f 成本 %.1fbp ==="
              % (tag, m["ann"] * 100, m["sharpe"], m["turnover"], m["cost_bp"]))

    # 缓冲带等权臂（生产定版执行层形态）——同逐股冲击口径
    for spec in [x for x in args.buffer.split(",") if x.strip()]:
        a_s, b_s = spec.split(":")
        ent, ext = float(a_s), float(b_s)
        m = evaluate_strategy(BufferedTopFracLongOnly(ent, ext), ctx, costs, args.capital)
        rows.append({"arm": f"eqw_top10pct_buf{ent:g}_{ext:g}", "k1_scale": np.nan,
                     "turnover_penalty": 0.0, "quadratic_cost": 0.0,
                     "alpha_scale": "", **m})
        print("=== %-24s 年化 %.2f%% Sharpe %.3f 换手 %.3f 成本 %.1fbp 活跃 %.0f只 ==="
              % (rows[-1]["arm"], m["ann"] * 100, m["sharpe"], m["turnover"],
                 m["cost_bp"], m["n_active"]))

    # 等权 Top-K 对照臂（同候选口径）——分离「集中度」与「优化器分配」
    for k in [int(x) for x in args.eqw_topk.split(",") if x.strip()]:
        tw_k = eqw_topk_panel(ctx.pred, ctx.mask, k)
        m = run_precomputed(tw_k, ctx, costs=costs, capital=args.capital)
        rows.append({"arm": f"eqw_top{k}", "k1_scale": np.nan, "turnover_penalty": 0.0,
                     "quadratic_cost": 0.0, "alpha_scale": "", **m})
        print("=== %-24s 年化 %.2f%% Sharpe %.3f 换手 %.3f 成本 %.1fbp 活跃 %.0f只 ==="
              % (f"eqw_top{k}", m["ann"] * 100, m["sharpe"], m["turnover"],
                 m["cost_bp"], m["n_active"]))

    for sc in [float(x) for x in args.scales.split(",") if x.strip()]:
        k1, k2 = k1_c * sc, k2_c * sc
        tw = qp_weight_panel(ctx.pred, ctx.ret_hist, ctx.mask, n_names=args.n_names,
                             k1=k1, k2=k2, max_weight=args.max_weight,
                             window=args.window, min_periods=args.min_periods,
                             method=args.method, risk_aversion=args.risk_aversion,
                             max_turnover=args.max_turnover, alpha_scale=alpha_scale)
        m = run_precomputed(tw, ctx, costs=costs, capital=args.capital)
        rows.append({"arm": "qp_cost_aware", "k1_scale": sc, "turnover_penalty": k1,
                     "quadratic_cost": k2, "alpha_scale": str(alpha_scale), **m})
        print("  scale=%.2f  k1=%.3g k2=%.3g  年化 %.2f%%  Sharpe %.3f  换手 %.3f  "
              "成本 %.1fbp  活跃 %.0f只"
              % (sc, k1, k2, m["ann"] * 100, m["sharpe"], m["turnover"],
                 m["cost_bp"], m["n_active"]))

    df = pd.DataFrame(rows)
    df.to_csv(out / f"summary{suf}.csv", index=False)
    print(f"\n产物: {out / f'summary{suf}.csv'}")


if __name__ == "__main__":
    main()
