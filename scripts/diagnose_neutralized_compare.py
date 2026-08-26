"""决定性实验：中性化口径下 模型 vs 最强单因子。

问题：
1. 单因子 range20/alpha191_159/vol60 月频超额 +20%~+36%，但调仓 IC 是负的
   —— 怀疑是 2024-2026 上行期高 beta 风格暴露，不是预测力。
2. 模型 IC=0.070 最高，但分层无单调、跑输指数。

实验设计：
- 候选：模型预测(中性化) / range20 / alpha191_159 / vol60 / alpha191_167
- 每个候选做两种处理：原样 vs 五因子中性化（市值/行业/mom20/vol20/turn20 取残差）
- 统一：月频调仓 top-50 等权、修复后引擎、费后、2024-01~2026-08
- 输出：总收益 / Sharpe / beta / 超额 + 调仓日IC + 分层单调性(单调相关系数)

预期：若单因子中性化后大幅缩水 -> 风格假象确认，模型相对最优。
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("diagnose_neu")

from scripts.e2e_common import (  # noqa: E402
    HORIZON, build_labels, compute_classic_features, drop_stale_factors,
    load_daily_data, select_features,
)
from scripts.e2e_backtest import (  # noqa: E402
    run_equal_weight_backtest, perf_stats, walk_forward_predictions,
)
from scripts.investment_report import load_index_returns  # noqa: E402
from scripts.optimize_e2e import build_neutral_covariates, neutralize_predictions  # noqa: E402
from research.factor_analysis import calc_ic_series  # noqa: E402

BT_START = "2024-01-01"
TOP_N = 50


def build_mock_data(n_days=400, n_codes=30, seed=1):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    px = {}
    px["close"] = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, (n_days, n_codes)), axis=0)),
        index=dates, columns=codes)
    px["open"] = px["close"] * (1 + rng.rand(n_days, n_codes) * 0.01)
    px["high"] = px[["open", "close"]].max(axis=1) * (1 + rng.rand(n_days, n_codes) * 0.01)
    px["low"] = px[["open", "close"]].min(axis=1) * (1 - rng.rand(n_days, n_codes) * 0.01)
    px["volume"] = pd.DataFrame(rng.rand(n_days, n_codes) * 1e6, index=dates, columns=codes)
    px["amount"] = px["volume"] * px["close"]
    px["vwap"] = px["close"] * (1 + rng.rand(n_days, n_codes) * 0.005)
    return px


def monotonic_corr(layer_nav):
    """分层单调性：Q1~Q5 终点累计收益与组序的 Spearman 相关。"""
    ends = layer_nav.dropna(how="all").iloc[-1]
    vals = (ends - 1).values
    # 组间差距平均
    diffs = np.diff(vals)
    monotonic = np.corrcoef(np.arange(5), vals)[0, 1] if len(vals) == 5 else np.nan
    return monotonic, diffs.mean(), vals


def quantile_backtest_simple(factor: pd.DataFrame, fwd_ret: pd.DataFrame, n=5):
    """逐日分层净值（日频未来一期收益正确复利，与报告口径一致）。"""
    common = fwd_ret.dropna(how="all").index.intersection(factor.dropna(how="all").index)
    nav = pd.DataFrame(1.0, index=common, columns=[f"Q{i+1}" for i in range(n)])
    ls = pd.Series(1.0, index=common)
    for t in common:
        f = factor.loc[t].dropna()
        r = fwd_ret.loc[t].reindex(f.index)
        cc = f.index.intersection(r.dropna().index)
        if len(cc) < 20:
            continue
        q = pd.qcut(f[cc], n, labels=False, duplicates="drop")
        for g in range(n):
            m = q == g
            if m.sum() >= 3:
                nav.loc[t, f"Q{g+1}"] = nav.loc[t, f"Q{g+1}"] * (1 + r[cc][m].mean())
        m1, m5 = q == 0, q == n - 1
        if m1.sum() >= 3 and m5.sum() >= 3:
            ls.loc[t] = ls.loc[t] * (1 + r[cc][m5].mean() - r[cc][m1].mean())
    nav = nav.cumprod() if False else nav  # 已逐日乘
    return nav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--out", default="reports/diagnose_neu")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.real:
        px, lib_feats = load_daily_data(begin=20220101)
        classic = compute_classic_features(px)
        all_feats = {**classic, **lib_feats}
        all_feats = drop_stale_factors(all_feats, px["close"].index[-1])
        close = px["close"]
    else:
        px = build_mock_data()
        close = px["close"]
        all_feats = compute_classic_features(px)

    returns = close.pct_change(fill_method=None)
    labels, fwd = build_labels(close, horizon=HORIZON)

    common = None
    for f in all_feats.values():
        d = f.dropna(how="all").index
        common = d if common is None else common.intersection(d)
    common = common.intersection(labels.dropna(how="all").index)
    bt_days = common[common >= pd.Timestamp(BT_START)]
    s = pd.Series(bt_days, index=bt_days)
    reb_days = list(s.groupby(s.index.to_period("M")).first())
    log.info("回测 %s ~ %s, %d 次调仓", bt_days[0].date(), bt_days[-1].date(), len(reb_days))

    # 特征选择（回测前窗口）
    all_days = close.index
    sel_days = all_days[all_days < pd.Timestamp(BT_START)]
    feats, _ = select_features(all_feats, fwd, sel_days, max_features=30)
    log.info("特征选择: %d -> %d", len(all_feats), len(feats))

    # 模型预测（与报告一致）
    pred = walk_forward_predictions(feats, labels, reb_days, common, model="gbdt")

    # 中性化协变量
    mc, ind, extra = build_neutral_covariates(px, close, args.real)
    pred_neu = neutralize_predictions(pred, mc, ind, extra)

    # 单因子候选（因子库）
    cands = {"模型预测": pred}
    if args.real:
        from scripts.e2e_common import load_library_factors
        lib = load_library_factors(exclude_model=True)
        for name in ["range20", "alpha191_159", "vol60", "alpha191_167", "mom20"]:
            if name in lib:
                cands[name] = lib[name].reindex(common)
        # 经典因子（vol60/range20 可能不在库，从经典特征里取）
        for name in ["vol60", "range20", "mom20"]:
            if name in all_feats:
                cands[name] = all_feats[name]

    bench = None
    if args.real:
        bench = load_index_returns("000300.SH", 20240101, 20260821, real=True).reindex(bt_days)

    rows = []
    mono_rows = []
    for name, f in cands.items():
        for neu_tag, ff in [("原样", f), ("中性化", neutralize_predictions(f, mc, ind, extra))]:
            tag = f"{name}[{neu_tag}]"
            # 调仓日 IC
            rp = ff.reindex(reb_days)
            ic = calc_ic_series(rp, fwd.reindex(reb_days)).dropna().mean()
            # 回测
            r = run_equal_weight_backtest(ff, returns, bt_days, args.top)
            st = perf_stats(r.daily_returns, tag)
            ex = np.nan
            if bench is not None:
                df = pd.DataFrame({"s": r.daily_returns, "b": bench}).dropna()
                ex = float((1 + (df["s"] - df["b"])).prod() - 1)
            rows.append({
                "候选": tag, "调仓IC": round(float(ic), 4),
                "总收益": round(st["total_return"], 4),
                "Sharpe": round(st["sharpe"], 3),
                "回撤": round(st["max_drawdown"], 4),
                "超额vs指数": round(ex, 4) if not np.isnan(ex) else None,
            })
            # 分层单调性（稀疏面板）
            try:
                nav = quantile_backtest_simple(ff.reindex(common), fwd.reindex(common))
                mon, gap, vals = monotonic_corr(nav)
                mono_rows.append({
                    "候选": tag,
                    "单调相关": round(mon, 3) if not np.isnan(mon) else None,
                    "Q5-Q1(pp)": round((vals[-1] - vals[0]) * 100, 1),
                    "Q1": round(vals[0] * 100, 1), "Q3": round(vals[2] * 100, 1),
                    "Q5": round(vals[4] * 100, 1),
                })
            except Exception as e:
                mono_rows.append({"候选": tag, "单调相关": None, "Q5-Q1(pp)": None,
                                  "Q1": None, "Q3": None, "Q5": None, "err": str(e)[:60]})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "neutralized_compare.csv", index=False, encoding="utf-8-sig")
    print("\n=== 中性化口径对比（月频 top-%d 等权，费后）===" % args.top)
    print(df.to_string(index=False))

    mf = pd.DataFrame(mono_rows)
    mf.to_csv(out_dir / "monotonicity.csv", index=False, encoding="utf-8-sig")
    print("\n=== 分层单调性（Q1=最低组, Q5=最高组）===")
    print(mf.to_string(index=False))

    # 模型预测的 IC 对比（稀疏 vs 持仓）
    print("\n=== 模型预测 IC ===")
    for neu_tag, ff in [("原样", pred), ("中性化", pred_neu)]:
        rp = ff.reindex(reb_days)
        ic = calc_ic_series(rp, fwd.reindex(reb_days)).dropna()
        ic_h = calc_ic_series(ff.reindex(common).ffill(), fwd.reindex(common)).dropna()
        print(f"  {neu_tag}: 稀疏IC={ic.mean():.4f} (n={len(ic)}) | 持仓IC={ic_h.mean():.4f} (n={len(ic_h)})")

    json.dump({"bt_start": str(bt_days[0].date()), "bt_end": str(bt_days[-1].date()),
               "n_rebalance": len(reb_days), "top_n": args.top,
               "n_features": len(feats)},
              open(out_dir / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
