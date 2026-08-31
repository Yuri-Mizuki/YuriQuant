"""
报告23 vs 基线：GP 适应度模式效果对比
=====================================

在同一数据上对比三种适应度口径（华泰研报21 基线 vs 报告23 两个新指标），
并从"线性预测 / 非线性信息 / 分层形态 / 线性化救活"四个维度评估挖出的因子。

用法:
    python scripts/compare_htai_fitness.py                          # mock（快）
    python scripts/compare_htai_fitness.py --real                   # 真实 HS300
    python scripts/compare_htai_fitness.py --fitness-modes rankic_mean,mutual_info
    python scripts/compare_htai_fitness.py --pop 300 --gen 3 --jobs 4

评估维度（对每个模式 hof 前 top-K 因子取平均）:
    lin_ic      : 线性预测力 —— 月频 20 日 rank IC 均值（华泰报告21 的适应度口径）
    mi          : 非线性信息 —— 离散化互信息均值（华泰报告23 的适应度口径）
    top_excess  : 多头超额收益（报告23 第二个适应度）
    convex      : 分层"中间凸"度 = 中间层超额 - 两端层超额（>0 表示非线性、ML 可利用）
    mono_rho    : 分层单调性 |spearman(层号, 层超额)|（高 = 线性因子）
    poly_gain   : 多项式拟合法线性化后的 |IC| 相对原始 |IC| 的倍数（非线性因子被"救活"的程度）

输出: reports/_htai_gp/compare_htai_fitness.csv + 终端对比表
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("compare_htai_fitness")

from scipy import stats


def build_panel(args) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """构建面板 + 华泰特征补全 + 中性化协变量（复用 mine_factors 的构造逻辑）。"""
    from data.cache_helpers import build_htai_neutral_panels, build_real_panel
    from data.mock import gen_mock_panel_with_signal
    if args.real:
        from config import Config
        cfg = Config.get()
        begin = args.begin or cfg["fetch"]["begin_date"]
        end = args.end or cfg.get("end_date")
        panel, rets = build_real_panel(cfg, begin, end)
    else:
        panel = gen_mock_panel_with_signal()
        rets = panel["close"].pct_change().shift(-1)
    if "returns" not in panel:
        panel["returns"] = panel["close"].pct_change()
    if "vwap" not in panel and "amount" in panel and "volume" in panel:
        panel["vwap"] = panel["amount"] / panel["volume"]
    neutral_panels = build_htai_neutral_panels(panel, real=args.real)
    return panel, rets, neutral_panels


def layer_excess_profile(fp: pd.DataFrame, rets: pd.DataFrame,
                         n_layers: int = 10) -> pd.Series:
    """10 层等分：每层平均未来 20 日收益相对全池等权的超额（全期均值）。

    华泰报告23 分层测试口径：因子值从大到小排序等分 N 层、层内等权，
    基准为全池等权。这里用月频收益近似（不跑完整净值回测）。
    """
    from factor.genetic_mining import _monthly_forward_returns
    r_forward = _monthly_forward_returns(rets)
    prof = []
    for d in fp.index:
        f = fp.loc[d]
        r = r_forward.loc[d]
        m = f.notna() & r.notna()
        if int(m.sum()) < n_layers * 2:
            continue
        fv = f[m]
        rv = r[m]
        q = fv.rank(pct=True)
        base = float(rv.mean())
        row = {}
        ok = True
        for li in range(n_layers):
            lo, hi = li / n_layers, (li + 1) / n_layers
            sel = (q > lo) & (q <= hi) if li < n_layers - 1 else (q > lo)
            if int(sel.sum()) == 0:      # 某层无股票 → 该天跳过（避免 NaN 污染曲线）
                ok = False
                break
            row[li] = float(rv[sel].mean()) - base
        if ok:
            prof.append(row)
    if not prof:
        return pd.Series(dtype=float)
    return pd.DataFrame(prof).mean()


def evaluate_factor(formula: str, panel: dict, returns_panel: pd.DataFrame,
                    neutral_panels: dict) -> dict:
    """单个因子的四维评估（华泰环内预处理后）。"""
    from factor.formula import formula_builder
    from factor.genetic_mining import (_htai_preprocess, _monthly_forward_returns,
                                       _mutual_info_series, _top_excess_series,
                                       polynomial_transform)
    from research.factor_analysis import calc_ic_series

    feats = list(panel.keys())
    fp = formula_builder(formula, features=feats)(panel)
    fpp = _htai_preprocess(fp, neutral_panels=neutral_panels)
    r_month = _monthly_forward_returns(returns_panel)
    ic = calc_ic_series(fpp, r_month, method="spearman").dropna()
    if len(ic) == 0:
        return {"formula": formula}
    lin_ic = float(ic.mean())
    lin_t = float(lin_ic / (ic.std() / np.sqrt(len(ic)))) if ic.std() > 0 else 0.0
    mi = _mutual_info_series(fpp, r_month).dropna()
    t_ex, b_ex, _ = _top_excess_series(fpp, r_month, top_frac=0.1)
    prof = layer_excess_profile(fpp, returns_panel)
    convex = float("nan")
    mono = float("nan")
    if len(prof) >= 10:
        mid = prof.loc[[4, 5]].mean()
        ends = prof.loc[[0, 1, 8, 9]].mean()
        convex = float(mid - ends)
        mono = float(abs(stats.spearmanr(np.arange(10), prof.values).statistic))
    # 多项式拟合法线性化：转换后月频 |IC| / 原始 |IC|（非线性救活倍数）
    poly_gain = float("nan")
    try:
        fp_poly = polynomial_transform(fpp, returns_panel, fit_window=100, refit=20)
        ic_p = calc_ic_series(fp_poly, r_month, method="spearman").dropna()
        if len(ic_p) > 10 and abs(lin_ic) > 1e-8:
            poly_gain = float(abs(ic_p.mean()) / abs(lin_ic))
    except Exception:
        pass
    return {
        "formula": formula,
        "lin_ic": lin_ic, "lin_t": lin_t, "n": len(ic),
        "mi": float(mi.mean()) if len(mi) else float("nan"),
        "top_excess": t_ex, "bot_excess": b_ex,
        "convex": convex, "mono_rho": mono, "poly_gain": poly_gain,
    }


def main():
    ap = argparse.ArgumentParser(description="华泰报告23 vs 基线：GP 适应度效果对比")
    ap.add_argument("--real", action="store_true", help="真实数据（默认 mock）")
    ap.add_argument("--begin", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--pop", type=int, default=300, help="每个模式的种群规模")
    ap.add_argument("--gen", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=1, help="GP 并行进程数")
    ap.add_argument("--top", type=int, default=10, help="每个模式评估 hof 前 N 个因子")
    ap.add_argument("--fitness-modes", default="rankic_mean,mutual_info,top_excess",
                    help="对比的适应度模式，逗号分隔")
    ap.add_argument("--out", default=None, help="结果 CSV 路径")
    args = ap.parse_args()

    from factor.genetic_mining import run_gp_mining
    from factor.operators import DEFAULT_WINDOWS

    panel, rets, neutral_panels = build_panel(args)
    feats = list(panel.keys())
    log.info("面板: %d 日 × %d 股, 特征 %d 个, 中性化协变量=%s",
             len(rets), panel["close"].shape[1], len(feats), list(neutral_panels.keys()))

    modes = [m.strip() for m in args.fitness_modes.split(",") if m.strip()]
    per_mode: dict[str, pd.DataFrame] = {}
    for mode in modes:
        log.info("==> GP fitness_mode=%s (pop=%d gen=%d) ...", mode, args.pop, args.gen)
        df, hof = run_gp_mining(
            panel, rets, features=feats, windows=DEFAULT_WINDOWS,
            population=args.pop, generations=args.gen, min_depth=1, max_depth=4,
            tournament=20, train_frac=1.0, htai=True, neutral_panels=neutral_panels,
            fitness_mode=mode, n_jobs=args.jobs, seed=0, verbose=True,
        )
        per_mode[mode] = df

    # 统一四维评估
    rows = []
    for mode, df in per_mode.items():
        for _, r in df.head(args.top).iterrows():
            ev = evaluate_factor(r["formula"], panel, rets, neutral_panels)
            ev["mode"] = mode
            rows.append(ev)
    ev_df = pd.DataFrame(rows)

    # 汇总对比表
    summary = []
    for mode in modes:
        sub = ev_df[ev_df["mode"] == mode]
        summary.append({
            "mode": mode,
            "n_factors": int(len(sub)),
            "avg|lin_ic|": float(sub["lin_ic"].abs().mean()),
            "avg_lin_t": float(sub["lin_t"].abs().mean()),
            "avg_mi": float(sub["mi"].mean()),
            "avg_top_excess": float(sub["top_excess"].mean()),
            "avg_convex": float(sub["convex"].mean()),
            "avg_mono_rho": float(sub["mono_rho"].mean()),
            "avg_poly_gain": float(sub["poly_gain"].mean()),
        })
    sum_df = pd.DataFrame(summary)
    print(f"\n===== 适应度模式效果对比（hof 前 {args.top} 因子平均）=====")
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.4f}"):
        print(sum_df.to_string(index=False))
    print("\n指标含义: lin_ic=月频线性RankIC | mi=互信息(非线性) | top_excess=多头超额")
    print("convex=分层中间凸度(>0 非线性可被ML利用) | mono_rho=分层单调性 | poly_gain=多项式线性化后|IC|倍数")

    out_path = Path(args.out or "reports/_htai_gp/compare_htai_fitness.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ev_df.to_csv(out_path, index=False)
    sum_df.to_csv(out_path.with_name(out_path.stem + "_summary.csv"), index=False)
    log.info("结果已保存: %s (+ _summary.csv)", out_path)


if __name__ == "__main__":
    main()
