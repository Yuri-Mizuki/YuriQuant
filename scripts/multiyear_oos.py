"""多年度 OOS 稳健性实验：模型 × horizon × 调仓频率 × 年份

把固化链路的配置（gbdt+中性化+TopFrac 重仓多头）在 2023/2024/2025 各年做
walk-forward OOS 验证。目标：确认 horizon=1 + 月度调仓的最优性在多个年份
是否稳健，而非只在 2025 成立。

纪律：
  - 特征选择固定在定型期（<=valid_end），不随目标年份重选，避免前视选择。
  - 每年 OOS：用该年之前的全部历史 rolling 训练，只预测该年（n_folds=4 季度折）。
  - 目标年份 = 2023/2024/2025（2022 是最早训练段，无前置历史，不作目标年）。
  - 调仓频率合法组合：horizon=1 → D/W/M；horizon=5 → 仅 M（引擎守卫约束）。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

log = logging.getLogger("multiyear")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

DATASET = "hs300_2022_2025"
MODEL_PARAMS = {
    "gbdt": dict(n_estimators=150, learning_rate=0.03, num_leaves=15,
                 min_child_samples=50, seed=0),
    "ridge": {},
    "ranker": dict(n_estimators=200, learning_rate=0.05, num_leaves=15,
                   min_child_samples=50, seed=0, labels_bins=2, objective="rank_xendcg"),
}
N_FOLDS = 4
YEARS = [2023, 2024, 2025]
FRACS = 0.20
OUT = Path("reports") / "multiyear"


def build_features(close, valid_days, dev_days):
    from factor.preprocessing import standardize_zscore
    from research.factor_library import FactorLibrary
    from research.factor_analysis import calc_ic_series
    from model.features import build_feature_set

    feats = FactorLibrary(dataset=DATASET).load_library_features()
    feats = {k: v for k, v in feats.items() if v.index[0].year <= 2022}
    feats = {k: standardize_zscore(v.reindex(close.index)) for k, v in feats.items()}
    # 质量分只用定型期 valid 段（固定，不随目标年变化 -> 无前视）
    fwd_v = close.pct_change(1, fill_method=None).shift(-1).loc[valid_days]
    quality = {}
    for nm, p in feats.items():
        ic = calc_ic_series(p.loc[valid_days], fwd_v).dropna()
        if len(ic) >= 10:
            quality[nm] = abs(float(ic.mean()))
    sel = build_feature_set(
        {k: v.loc[dev_days] for k, v in feats.items()},
        min_coverage=0.5, dedup_corr=0.7, max_features=50,
        quality=pd.Series(quality) if quality else None)
    return {k: feats[k] for k in sel}


def run_year(year, model, horizon, features, close, all_days):
    """walk-forward 预测 year 全年，返回 OOS 预测面板 + 该年 fwd 面板。"""
    from model.predictor import PREDICTORS
    from scripts.walk_forward_model import rolling_oos
    from model.labels import build_labels

    test_days = all_days[(all_days >= pd.Timestamp(f"{year}-01-01")) &
                         (all_days <= pd.Timestamp(f"{year}-12-31"))]
    labels, embargo = build_labels(close, horizon=horizon, mode="rank")
    params = MODEL_PARAMS.get(model, {})
    pred = rolling_oos(PREDICTORS[model], features, labels, test_days, all_days,
                       n_folds=N_FOLDS, embargo_days=embargo, min_train_days=120, **params)
    # 回测收益口径（engine 约定）：h=1 传未 shift 的 pct_change()（与指数基准
    # 日标签对齐）；h>1 传 forward 段累计面板。
    if horizon == 1:
        fwd = close.pct_change(fill_method=None).loc[test_days]
    else:
        fwd = close.pct_change(horizon, fill_method=None).shift(-horizon).loc[test_days]
    return pred.loc[test_days], fwd, test_days


def main():
    from scripts.run_model_portfolio import (
        load_index_benchmark, build_style_covariates_panel, neutralize_panel,
    )
    from strategy.examples import TopFracLongOnly
    from backtest.engine import VectorBacktest
    from backtest.costs import TransactionCosts
    from data.cache_helpers import build_panel
    from research.factor_analysis import calc_ic_series

    t0 = time.time()
    disc = Config.discipline()
    panel, _ = build_panel(Config.get(), disc["begin"], 20261231, offline=True,
                           include_market_cap=True)
    close = panel["close"]
    all_days = close.index
    valid_days = all_days[(all_days > pd.Timestamp(str(disc["train_end"]))) &
                          (all_days <= pd.Timestamp(str(disc["valid_end"])))]
    dev_days = all_days[all_days <= pd.Timestamp(str(disc["valid_end"]))]

    features = build_features(close, valid_days, dev_days)
    log.info("特征 %d 个（定型期固定选择）| valid %s~%s", len(features),
             valid_days[0].date(), valid_days[-1].date())

    cov = build_style_covariates_panel(panel)
    costs = TransactionCosts(commission_rate=0.0003, stamp_duty=0.001, slippage_bp=10.0)
    strat = TopFracLongOnly(frac=FRACS, weight_mode="equal")

    rows = []
    for year in YEARS:
        test_days = all_days[(all_days >= pd.Timestamp(f"{year}-01-01")) &
                             (all_days <= pd.Timestamp(f"{year}-12-31"))]
        bench = load_index_benchmark(test_days).dropna()
        bench_annual = (1 + bench).prod() ** (252 / len(bench)) - 1
        log.info("===== 年份 %d 沪深300年化=%.2f%% (交易日 %d) =====",
                 year, bench_annual * 100, len(bench))
        for model in ["gbdt", "ridge", "ranker"]:
            for horizon in [1, 5]:
                st = time.time()
                pred, fwd, _ = run_year(year, model, horizon, features, close, all_days)
                ic = float(calc_ic_series(pred, fwd).mean())
                sig = neutralize_panel(pred, cov)
                freqs = ["M"] if horizon > 1 else ["D", "W", "M"]
                for freq in freqs:
                    bt = VectorBacktest(strategy=strat, rebalance_freq=freq,
                                        initial_capital=1_000_000.0, costs=costs)
                    res = bt.run(sig, fwd, horizon=horizon)
                    m = res.metrics(benchmark_returns=bench)
                    rows.append({
                        "year": year, "model": model, "horizon": horizon, "freq": freq,
                        "oos_ic": ic, "annual": m.get("annual_return", 0),
                        "excess": m.get("excess_return", 0), "sharpe": m.get("sharpe", 0),
                        "ir": m.get("information_ratio", 0),
                        "max_dd": m.get("max_drawdown", 0),
                        "turnover": m.get("avg_turnover", 0),
                        "bench_annual": bench_annual,
                    })
                    log.info("[%d] %s h%d %s | IC=%.4f 超额=%+.2f%% 换手=%.1f%% | %.0fs",
                             year, model, horizon, freq, ic,
                             rows[-1]["excess"] * 100, rows[-1]["turnover"], time.time() - st)

    table = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("多年度 OOS 稳健性（中性化 Top20% 多头, 含成本）")
    print("=" * 78)
    for year in YEARS:
        sub = table[table.year == year]
        ba = sub["bench_annual"].iloc[0]
        print(f"\n--- {year} 年 沪深300={ba:.2%} ---")
        cols = ["model", "horizon", "freq", "oos_ic", "annual", "excess", "sharpe", "max_dd", "turnover"]
        with pd.option_context("display.width", 200, "display.float_format",
                               lambda v: f"{v:.4f}"):
            print(sub.sort_values("excess", ascending=False)[cols].to_string(index=False))
    # 按 (model,horizon,freq) 汇总三年
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "multiyear_result.csv", index=False, encoding="utf-8-sig")
    print("\n===== 三年平均超额（按配置，正值=跑赢指数）=====")
    g = table.groupby(["model", "horizon", "freq"])["excess"].agg(["mean", "std", "count"])
    g["pos_years"] = table.groupby(["model", "horizon", "freq"])["excess"].apply(
        lambda s: int((s > 0).sum()))
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(g.sort_values("mean", ascending=False).round(4).to_string())
    print(f"\n耗时 {time.time()-t0:.0f}s | 保存 {OUT}")


if __name__ == "__main__":
    main()