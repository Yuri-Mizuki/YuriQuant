"""
模型增强组合回测（正式固化入口）
================================

把「模型信号 → 风格中性化 → TopFrac 重仓多头」整条链路固化为一个正式可复用入口，
参数统一从 ``config/settings.yaml`` 的 ``model_portfolio`` 段读取（2026-08-25 精修固化）。

默认配置（gbdt tuneda）：horizon=1, model=gbdt, strategy=topfrac_lo, frac=0.20,
月度调仓。2025 test 段成本后超额沪深300 +4.85%（见 reports/gbdt_tune 网格）。

用法:
    # 用配置默认跑（真实本地数据, 完整 walk-forward）
    python -m scripts.run_model_portfolio

    # 覆盖模型 / 持仓比例
    python -m scripts.run_model_portfolio --model ranker --frac 0.25

    # 只回测、跳过 walk-forward 训练（直接吃现有 OOS 面板——需先跑一次全流程）
    python -m scripts.run_model_portfolio --offline-panel reports/model_portfolio/gbdt_pred.parquet
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

log = logging.getLogger("model_portfolio")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

DATASET = "hs300_2022_2025"
DEFAULT_MODEL_PARAMS = {
    "gbdt":   dict(n_estimators=150, learning_rate=0.03, num_leaves=15,
                   min_child_samples=50, seed=0),
    "ridge":  {},
    "ranker": dict(n_estimators=200, learning_rate=0.05, num_leaves=15,
                   min_child_samples=50, seed=0, labels_bins=2,
                   objective="rank_xendcg"),
}
OUT_DIR = Path("reports") / "model_portfolio"


def _mp_cfg() -> dict:
    """合并 config 的 model_portfolio 段（默认数据回填）。"""
    cfg = Config.get().get("model_portfolio", {})
    defaults = dict(horizon=1, strategy="topfrac_lo", frac=0.20,
                    rebalance_freq="M", model="gbdt", neutralize=True,
                    cost_slippage_bp=10, cost_commission=0.0003, cost_stamp=0.001)
    defaults.update({k: v for k, v in cfg.items() if v is not None})
    return defaults


def load_index_benchmark(test_days: pd.DatetimeIndex) -> pd.Series:
    code = str(Config.get()["backtest"]["benchmark"])
    safe = code.replace(".", "_")
    root = Path(str(Config.cache()["root"]).replace("//", "/"))
    kline = pd.read_parquet(root / f"index_daily_{safe}.parquet")
    if "code" in kline.index.names:
        close = kline.xs(code, level="code")["close"]
    else:
        close = kline["close"]
    close = close.reindex(test_days).ffill()
    ret = close.pct_change(fill_method=None)
    ret.name = code
    return ret


def build_style_covariates_panel(panel):
    from factor.preprocessing import build_style_covariates
    from data.industry import IndustryClassification
    from data.offline import OfflineQuietDataSource
    from data.cache import DataCache
    cache = DataCache(OfflineQuietDataSource())
    payload = {
        "close": panel["close"], "volume": panel["volume"],
        "tot_share": panel["market_cap"] / panel["close"].where(panel["close"] > 0),
    }
    ind = IndustryClassification(cache, level=1).get_industry_panel(
        panel["close"].columns, panel["close"].index)
    return build_style_covariates(payload, market_cap_panel=panel["market_cap"],
                                  industry_panel=ind)


def neutralize_panel(signal, cov):
    from factor.preprocessing import neutralize
    size = cov.get("size")
    ind = cov.get("industry")
    extra = {k: v for k, v in cov.items() if k not in ("size", "industry")}
    return neutralize(signal, market_cap_panel=size, industry_panel=ind,
                      extra_covariates=extra)


def build_model_panel(model: str, horizon: int, test_days: pd.DatetimeIndex):
    """walk-forward 训练 model，返回 (test 段 OOS 预测面板, panel)."""
    from model.predictor import PREDICTORS
    from scripts.walk_forward_model import rolling_oos
    from factor.preprocessing import standardize_zscore
    from model.labels import build_labels
    from research.factor_analysis import calc_ic_series
    from model.features import build_feature_set
    from data.cache_helpers import build_panel

    disc = Config.discipline()
    cfg = Config.get()
    panel, _ = build_panel(cfg, disc["begin"], 20261231, offline=True,
                           include_market_cap=True)
    close = panel["close"]
    all_days = close.index
    valid_days = all_days[(all_days > pd.Timestamp(str(disc["train_end"]))) &
                          (all_days <= pd.Timestamp(str(disc["valid_end"])))]
    dev_days = all_days[: len(all_days) - len(test_days)]

    from research.factor_library import FactorLibrary
    feats = FactorLibrary(dataset=DATASET).load_library_features()
    feats = {k: v for k, v in feats.items() if v.index[0].year <= 2022}
    feats = {k: standardize_zscore(v.reindex(close.index)) for k, v in feats.items()}

    labels, embargo = build_labels(close, horizon=horizon, mode="rank")
    fwd_v = close.pct_change(horizon, fill_method=None).shift(-horizon).loc[valid_days]
    quality = {}
    for nm, p in feats.items():
        ic = calc_ic_series(p.loc[valid_days], fwd_v).dropna()
        if len(ic) >= 10:
            quality[nm] = abs(float(ic.mean()))
    feats_sel = build_feature_set(
        {k: v.loc[dev_days] for k, v in feats.items()},
        min_coverage=0.5, dedup_corr=0.7, max_features=50,
        quality=pd.Series(quality) if quality else None)
    sel = {k: feats[k] for k in feats_sel}

    params = DEFAULT_MODEL_PARAMS.get(model, {})
    pred = rolling_oos(PREDICTORS[model], sel, labels, test_days, all_days,
                       n_folds=12, embargo_days=embargo, min_train_days=120, **params)
    # 回测收益口径（engine 约定）：h=1 传未 shift 的 pct_change()（第 i 行 =
    # i-1→i 单日收益，与指数基准日标签对齐）；h>1 传 forward 段累计面板。
    fwd = close.pct_change(fill_method=None) if horizon == 1 \
        else close.pct_change(horizon, fill_method=None).shift(-horizon)
    return pred, panel, fwd


def _metrics(res, bench_daily) -> dict:
    return res.metrics(benchmark_returns=bench_daily)


def run_backtest(signal, fwd, bench_daily, frac, horizon, factor_cost: bool):
    from strategy.examples import TopFracLongOnly
    from backtest.engine import VectorBacktest
    from backtest.costs import TransactionCosts
    cfg = _mp_cfg()
    if factor_cost:
        costs = TransactionCosts(commission_rate=cfg["cost_commission"],
                                 stamp_duty=cfg["cost_stamp"],
                                 slippage_bp=cfg["cost_slippage_bp"])
    else:
        costs = TransactionCosts(commission_rate=0.0, stamp_duty=0.0, slippage_bp=0.0)
    strat = TopFracLongOnly(frac=frac, weight_mode="equal")
    bt = VectorBacktest(strategy=strat, rebalance_freq=cfg["rebalance_freq"],
                        initial_capital=1_000_000.0, costs=costs)
    res = bt.run(signal, fwd, horizon=horizon)
    return res, _metrics(res, bench_daily)


def save_report(table, curves, bench_annual, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "portfolio_result.csv", index=False, encoding="utf-8-sig")
    for name, eq in curves.items():
        eq.to_csv(out_dir / f"equity_{name}.csv", encoding="utf-8-sig")
    # 精简 txt 摘要
    lines = ["===== 模型增强组合（固化配置）=====",
             f"沪深300指数年化: {bench_annual:.2%}"]
    for _, row in table.iterrows():
        lines.append(
            f"{row['config']}: 年化={row['annual']:.2%} 超额={row['excess']:+.2%} "
            f"Sharpe={row['sharpe']:.2f} IR={row['ir']:.2f} MaxDD={row['max_dd']:.2%}")
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log.info("结果已保存到 %s", out_dir)


def main():
    parser = argparse.ArgumentParser(description="模型增强组合回测（固化入口）")
    parser.add_argument("--model", default=None, help="信号模型: gbdt/ridge/ranker")
    parser.add_argument("--frac", type=float, default=None, help="持仓比例（默认读配置 0.20）")
    parser.add_argument("--horizon", type=int, default=None, help="预测视野（默认读配置 1）")
    parser.add_argument("--pre-cost", action="store_true", help="同时输出成本前口径")
    args = parser.parse_args()

    cfg = _mp_cfg()
    if args.model:
        cfg["model"] = args.model
    if args.frac is not None:
        cfg["frac"] = args.frac
    if args.horizon is not None:
        cfg["horizon"] = args.horizon
    log.info("模型组合配置: model=%s horizon=%d frac=%.2f strategy=%s 调仓=%s 中性化=%s",
             cfg["model"], cfg["horizon"], cfg["frac"], cfg["strategy"],
             cfg["rebalance_freq"], cfg["neutralize"])

    t0 = time.time()
    disc = Config.discipline()
    from data.cache_helpers import build_panel
    close0, _ = build_panel(Config.get(), disc["begin"], 20261231, offline=True)
    test_days = close0["close"].index[
        close0["close"].index > pd.Timestamp(str(disc["valid_end"]))]
    test_days = test_days[test_days <= pd.Timestamp("2025-12-31")]
    bench_daily = load_index_benchmark(test_days).dropna()
    bench_annual = (1 + bench_daily).prod() ** (252 / len(bench_daily)) - 1

    pred, panel, fwd_all = build_model_panel(cfg["model"], cfg["horizon"], test_days)
    fwd = fwd_all.loc[test_days]
    pred = pred.loc[test_days].reindex(columns=fwd.columns)
    sig = pred
    if cfg["neutralize"]:
        cov = build_style_covariates_panel(panel)
        sig = neutralize_panel(pred, cov)
        log.info("已做风格中性化协变量: %s", sorted(k for k in cov))

    rows, curves = [], {}
    for tag2, cost in (("net", True), ("pre", False)):
        res, m = run_backtest(sig, fwd, bench_daily, cfg["frac"], cfg["horizon"],
                              factor_cost=cost)
        curves[f"{cfg['model']}_{tag2}"] = res.equity_curve
        rows.append({"config": f"{cfg['model']}_{tag2}", "cost": tag2,
                     "annual": m.get("annual_return", 0),
                     "excess": m.get("excess_return", 0),
                     "sharpe": m.get("sharpe", 0),
                     "ir": m.get("information_ratio", 0),
                     "max_dd": m.get("max_drawdown", 0),
                     "turnover": m.get("avg_turnover", 0)})
        if tag2 == "pre" and not args.pre_cost:
            break  # 默认只保存成本后
    table = pd.DataFrame(rows)
    print(f"\n===== 模型增强组合（{cfg['model']}, h={cfg['horizon']}, frac={cfg['frac']}）=====")
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.4f}"):
        print(table.to_string(index=False))
    print(f"沪深300指数年化: {bench_annual:.2%} | 交易日 {len(bench_daily)} | 总耗时 {time.time()-t0:.0f}s")

    save_report(table, curves, bench_annual, OUT_DIR)


if __name__ == "__main__":
    main()