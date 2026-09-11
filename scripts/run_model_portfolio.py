"""
模型增强组合（正式固化入口，2026-09-09 升级为全A正交化管线）
==========================================================

固化为实验最优配方（三臂对比 + 消融 + 集成实验，2018-2026 样本外含成本，
详见 reports/alla_rolling{,_ortho,_mixed}/ 与 README「已完成 2026-09-07」段）：

    全A（含退市回补） → 因子层全正交（panels_neu）→ DPP 选择+基本面族保留席位
    → gbdt h1+h5 秩平均集成 → raw 信号（信号层不再中性化）→ 月频 Top10% 等权

头部实现 2018-2026 样本外：年化 15.5%、超额上证 +13.3%/年、Sharpe 0.63。
旧版 HS300 口径（dataset=hs300_2022_2025 + 信号层风格中性化）已退役——
其「信号层中性化」在全A上被实证为过度中性化（−5.4pp/年）。

数据前置（_base 与因子面板由 rolling_grid_alla 管线维护，本入口直接消费）：
    python scripts/rolling_grid_alla.py --stage prep        # _base 基础面板
    python scripts/builders/build_alla_alpha_panels.py ...    # 因子面板（已存在则跳过）

用法:
    python -m scripts.run_model_portfolio                       # 全流程（当年预测+回测+今日选股）
    python -m scripts.run_model_portfolio --frac 0.20           # 覆盖持仓比例
    python -m scripts.run_model_portfolio --refresh-base        # 先重建 _base 再跑
    python -m scripts.run_model_portfolio --no-train            # 复用上次预测面板（只回测/选股）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.costs import default_costs  # noqa: E402
from backtest.engine import VectorBacktest  # noqa: E402
from backtest.metrics import PERIODS_PER_YEAR  # noqa: E402
from config import Config  # noqa: E402
from model.params import DEFAULT_MODEL_PARAMS  # noqa: E402
from scripts.cli_common import setup_logging  # noqa: E402
from scripts.portfolio_common import neutralize_panel  # noqa: E402
from strategy.examples import TopFracLongOnly  # noqa: E402

log = setup_logging("model_portfolio")

OUT_DIR = Path("reports") / "model_portfolio"


def _mp_cfg() -> dict:
    """合并 config 的 model_portfolio 段（默认数据回填）。"""
    cfg = Config.get().get("model_portfolio", {})
    defaults = dict(
        pipeline="alla_ortho",
        dataset="all_a_2018_2026",
        ensemble_horizons=[1, 5],
        model="gbdt", strategy="topfrac_lo", frac=0.10,
        rebalance_freq="M", neutralize=False,
        selection_cut="auto",
        train_window=500, n_folds=4, quality_window=500,
        benchmark="000001.SH")
    defaults.update({k: v for k, v in cfg.items() if v is not None})
    return defaults


def load_benchmarks(test_days: pd.DatetimeIndex, base: dict) -> dict:
    """基准组：配置指数（默认上证）+ 全A等权（_base 内置）。"""
    from data.cache_helpers import load_index_returns
    code = str(_mp_cfg()["benchmark"])
    ret = load_index_returns(code, begin=int(test_days[0].strftime("%Y%m%d")),
                             reindex_to=test_days)
    if ret is None:
        raise FileNotFoundError(
            f"指数 {code} 无缓存：请先 update_data 拉取指数日线")
    return {f"idx_{code[:6]}": ret.fillna(0.0),
            "eqw_alla": base["bench_eqw"].reindex(test_days).fillna(0.0)}


def build_ensemble_panel(cfg: dict, base: dict, force_retrain: bool):
    """当年 walk-forward 训练集成成员（h1+h5 gbdt）→ 秩平均集成信号。

    复用 rolling_grid_alla 的实验验证组件（单一真源）：
    FeatureStore（ortho 加载 panels_neu）/ select_features_for_year（DPP+保留席位，
    生产口径 cut=最新完整日）/ rolling_window_oos（500 日窗季度折）/ existence_mask。
    """
    import scripts.rolling_grid_alla as RG
    from model.labels import build_labels
    from model.predictor import PREDICTORS

    close = base["close"]
    all_days = close.index
    year = int(all_days[-1].year)
    test_days = all_days[all_days >= pd.Timestamp(f"{year}-01-01")]
    store = RG.FeatureStore(RG.ds_root() / "panels_neu")   # 因子层正交化口径

    # 生产口径的特征选择：质量窗截止 = 最新完整交易日（selection_cut: auto）
    cut = None
    if str(cfg.get("selection_cut", "auto")) != "auto":
        cut = pd.Timestamp(str(cfg["selection_cut"]))
        test_days = test_days[test_days <= cut]

    preds: dict[int, pd.DataFrame] = {}
    for h in cfg["ensemble_horizons"]:
        cache_path = OUT_DIR / f"pred_h{h}.parquet"
        if force_retrain or not cache_path.exists():
            registry = pd.read_csv(RG.ds_root() / "registry.csv")
            ic_cache = pd.read_parquet(RG.ds_root() / f"ic_h{h}.parquet")
            sel_cut = (cut if cut is not None
                       else pd.Timestamp(f"{year}-12-31"))
            names = RG.select_features_for_year(
                year + 1, h, ic_cache, registry, store, all_days, cut=sel_cut)
            feats = {k: v.reindex(index=all_days, columns=close.columns)
                     for k, v in store.get_many(names).items()}
            labels, _embargo = build_labels(close, horizon=h, mode="rank")
            params = dict(DEFAULT_MODEL_PARAMS[cfg["model"]])
            pred = RG.rolling_window_oos(
                PREDICTORS["gbdt"], params, feats, labels,
                test_days, all_days, h, cfg["train_window"])
            valid = RG.existence_mask(feats, close, test_days)
            pred = pred.where(valid)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            pred.astype(np.float32).to_parquet(cache_path)
            log.info("[h%d] 预测完成并缓存: %s（%d 天 × %d 股）", h, cache_path.name,
                     len(pred), pred.shape[1])
        else:
            pred = pd.read_parquet(cache_path)
            log.info("[h%d] 复用缓存预测 %s（--no-train 关闭复用）", h, cache_path.name)
        preds[h] = pred

    ens = RG._rank_average(list(preds.values()), min_panels=len(preds))
    return ens, preds


def export_picks(signal: pd.DataFrame, mask: pd.DataFrame, frac: float) -> Path:
    """最新完整交易日的 TopFrac 选股清单（信号×可交易掩码，等权 1/k）。"""
    d = signal.index[-1]
    vals = signal.loc[d].dropna()
    executable = mask.loc[d]
    vals = vals[vals.index.intersection(executable[executable].index)]
    k = max(1, int(round(frac * len(vals))))
    top = vals.sort_values(ascending=False).head(k)
    out = OUT_DIR / f"picks_{d.date()}.csv"
    pd.DataFrame({"rank": range(1, len(top) + 1), "score": top.values,
                  "weight": 1.0 / len(top)},
                 index=top.index.rename("code")).to_csv(out, encoding="utf-8-sig")
    log.info("今日选股（%s）: Top%d（%.0f%% 截面）-> %s", d.date(), len(top),
             frac * 100, out)
    return out


def main():
    parser = argparse.ArgumentParser(description="模型增强组合（全A正交化管线，固化入口）")
    parser.add_argument("--frac", type=float, default=None,
                        help="持仓比例（默认读配置 0.10）")
    parser.add_argument("--freq", default=None, help="调仓频率（默认读配置 M）")
    parser.add_argument("--pre-cost", action="store_true", help="同时输出成本前口径")
    parser.add_argument("--refresh-base", action="store_true",
                        help="先重建 _base 基础面板（日线有更新后需要）")
    parser.add_argument("--no-train", action="store_true",
                        help="复用上次预测缓存（快速回测/选股）")
    args = parser.parse_args()

    cfg = _mp_cfg()
    if args.frac is not None:
        cfg["frac"] = args.frac
    if args.freq is not None:
        cfg["rebalance_freq"] = args.freq
    cfg["ensemble_horizons"] = [int(h) for h in cfg["ensemble_horizons"]]
    log.info("管线配置: pipeline=%s 集成视野=%s frac=%.2f 调仓=%s 信号=%s",
             cfg["pipeline"], cfg["ensemble_horizons"], cfg["frac"],
             cfg["rebalance_freq"], "raw" if not cfg["neutralize"] else "neut")

    t0 = time.time()

    if args.refresh_base:
        from scripts.rolling_grid_alla import stage_prep
        stage_prep()

    import scripts.rolling_grid_alla as RG
    base = RG.load_base()
    close = base["close"]
    log.info("数据面板: %d 日 × %d 股（%s ~ %s）", len(close), close.shape[1],
             close.index[0].date(), close.index[-1].date())

    ens, _preds = build_ensemble_panel(cfg, base, force_retrain=not args.no_train)
    test_days = ens.index
    fwd = close.pct_change(fill_method=None).loc[test_days]
    bench = load_benchmarks(test_days, base)
    bench_main = list(bench.values())[0]

    # 信号层口径：主配方 raw（neutralize=false）；如配置 true 则信号层风格中性化
    sig = ens.reindex(columns=close.columns)
    if cfg["neutralize"]:
        sig = neutralize_panel(sig, base["cov"])

    mask = base["mask"].reindex(index=test_days, columns=close.columns).fillna(True)

    rows, curves = [], {}
    for tag, cost in (("net", True), ("pre", False)):
        strat = TopFracLongOnly(frac=cfg["frac"], weight_mode="equal")
        bt = VectorBacktest(strategy=strat, rebalance_freq=cfg["rebalance_freq"],
                            initial_capital=1_000_000.0, costs=default_costs(cost))
        res = bt.run(sig, fwd, executable_mask=mask, horizon=cfg["ensemble_horizons"][0])
        m = res.metrics(benchmark_returns=bench_main)
        curves[f"ens_h{'h'.join(str(h) for h in cfg['ensemble_horizons'])}_{tag}"] = \
            res.equity_curve
        rows.append({"config": f"ens_h{''.join(map(str, cfg['ensemble_horizons']))}_{tag}",
                     "cost": tag, "annual": m.get("annual_return", 0),
                     "excess": m.get("excess_return", 0),
                     "sharpe": m.get("sharpe", 0),
                     "ir": m.get("information_ratio", 0),
                     "max_dd": m.get("max_drawdown", 0),
                     "turnover": m.get("avg_turnover", 0)})
        if tag == "pre" and not args.pre_cost:
            break

    table = pd.DataFrame(rows)
    print(f"\n===== 模型增强组合（全A正交化管线 ens_h{'h'.join(map(str, cfg['ensemble_horizons']))}"
          f"，frac={cfg['frac']}，{cfg['rebalance_freq']}）=====")
    with pd.option_context("display.width", 200,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(table.to_string(index=False))
    for name, s in bench.items():
        b = s.dropna()
        print(f"基准 {name} 年化: {(1 + b).prod() ** (PERIODS_PER_YEAR / max(1, len(b))) - 1:.2%}"
              f"（{len(b)} 日）")
    print(f"样本 {test_days[0].date()} ~ {test_days[-1].date()} | 总耗时 {time.time()-t0:.0f}s")

    # 回测与选股落盘
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "portfolio_result.csv", index=False, encoding="utf-8-sig")
    for name, eq in curves.items():
        eq.to_csv(OUT_DIR / f"equity_{name}.csv", encoding="utf-8-sig")
    ens.round(6).to_parquet(OUT_DIR / "ens_pred.parquet")
    export_picks(sig, mask, cfg["frac"])
    log.info("结果已保存到 %s", OUT_DIR)


if __name__ == "__main__":
    main()
