"""
从因子库加载 raw 因子 → 四种合成 → 评估对比 + 入库（带血缘）
================================================================

与 scripts/synthesize_factors.py --from-library 等价，但**不需要 SDK**：
组件面板直接从因子库 panels 读回，returns 面板从本地日线缓存计算。
适合 SDK 不可用 / 纯离线研究场景。

合成方式：IC 加权 / PCA / Gram-Schmidt 正交 / ML Stacking。
入库名：composite36_ic_weighted 等（v2 前缀区分旧 10 因子版合成），
parents 记录全部参与合成的 raw 因子（血缘可追溯）。

用法
----
    python -m scripts.synthesize_library --dataset hs300_2025
    python -m scripts.synthesize_library --dataset hs300_2025 --methods ic_weighted,orthogonal
    python -m scripts.synthesize_library --dataset mock --begin 20230103 --end 20241231
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from data.cache import DataCache
from data.cache_helpers import returns_from_cache
from data.offline import OfflineDataSource
from factor.preprocessing import standardize_zscore
from factor.synthesis import (
    CompositeInput,
    rebuild_train_weights,
    synthesize_ic_weighted,
    synthesize_orthogonal,
    synthesize_pca,
    synthesize_stacking,
    synthesize_stacking_gbdt,
    synthesize_stacking_gbdt_tuned,
    synthesize_stacking_lambdarank,
)
from research.factor_library import FactorLibrary
from backtest.engine import VectorBacktest
from strategy.examples import TopKLongShort

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("synthesize_library")

METHODS = ["ic_weighted", "pca", "orthogonal", "stacking", "stacking_gbdt", "lambdarank"]


def main():
    parser = argparse.ArgumentParser(description="因子库内因子合并合成（离线）")
    parser.add_argument("--dataset", default="hs300_2025")
    parser.add_argument("--methods", default="all",
                        help="ic_weighted,pca,orthogonal,stacking,stacking_gbdt,lambdarank")
    parser.add_argument("--tune-trials", type=int, default=0,
                        help="stacking_gbdt 用 optuna 自动调参的 trials 数（>0 启用；"
                             "默认 0=固定参数）")
    parser.add_argument("--target-mode", choices=["raw", "rank"], default="raw",
                        help="stacking 系拟合目标：raw=次日收益值；rank=当日截面收益"
                             "百分比秩（与 rank IC 评价口径一致，方案 A）")
    parser.add_argument("--begin", type=int, default=20250101)
    parser.add_argument("--end", type=int, default=20251231)
    parser.add_argument("--train-frac", type=float, default=0.7,
                        help="前段比例用作训练段，其 IC 决定 ic_weighted/orthogonal 的"
                             "权重与符号；后段（1-train_frac）为测试段，仅用于外推评估"
                             "（防未来函数，2026-08-17）")
    parser.add_argument("--out", default=None, help="对比报告 CSV（默认 reports/）")
    parser.add_argument("--no-save", action="store_true", help="只对比不入库")
    args = parser.parse_args()

    cache = DataCache(OfflineDataSource())
    lib = FactorLibrary(dataset=args.dataset)
    reg = lib.list_all(kind="raw")
    if reg.empty:
        log.error("数据集 %s 无 raw 因子", args.dataset)
        sys.exit(1)

    components: list[CompositeInput] = []
    for _, row in reg.iterrows():
        panel = lib.get_panel(row["name"])
        if panel is None or panel.empty:
            continue
        components.append(CompositeInput(
            name=row["name"], panel=standardize_zscore(panel),
            ic=float(row.get("ic_mean", 0.0) or 0.0),
            ir=float(row.get("ic_ir", 0.0) or 0.0),
        ))
    # 覆盖率过滤：非空率 < 50% 的稀疏因子不参与合成（避免 NaN 传播稀释
    # 复合面板；2026-08-04 分红/股东类因子覆盖不均引入此问题）
    before = len(components)
    components = [c for c in components
                  if c.panel.notna().mean().mean() >= 0.5]
    if len(components) < before:
        log.info("覆盖率过滤: %d -> %d（非空率<50%% 的因子已剔除）",
                 before, len(components))
    components.sort(key=lambda c: abs(c.ic), reverse=True)
    log.info("从库加载 %d 个 raw 因子（%s）", len(components), args.dataset)

    returns_panel = returns_from_cache(cache, args.begin, args.end)
    returns_panel = returns_panel.reindex(index=components[0].panel.index)
    log.info("returns 面板: %d 日 × %d 股", returns_panel.shape[0], returns_panel.shape[1])

    names = [c.name for c in components]
    method_list = METHODS if args.methods == "all" else [m.strip() for m in args.methods.split(",")]

    # 训练/测试段切分（2026-08-17 防未来函数）：训练段只用于决定合成权重/符号，
    # 测试段仅用于外推评估，绝不参与任何权重估计。
    all_dates = returns_panel.index
    train_frac = min(max(args.train_frac, 0.1), 0.9)
    n_train = max(1, int(len(all_dates) * train_frac))
    train_dates = all_dates[:n_train]
    test_dates = all_dates[n_train:]
    if len(test_dates) < 20:
        log.error("测试段交易日过少（%d < 20），请调大 --train-frac 或扩大 --end",
                  len(test_dates))
        sys.exit(1)
    log.info("切分: 训练段 %d 日(%s~%s) / 测试段 %d 日(%s~%s)",
             len(train_dates), train_dates[0], train_dates[-1],
             len(test_dates), test_dates[0], test_dates[-1])
    returns_train = returns_panel.loc[train_dates]
    returns_test = returns_panel.loc[test_dates]
    # 权重/符号来源：ic_weighted / orthogonal 只用训练段 IC，杜绝全样本 look-ahead
    train_components = rebuild_train_weights(components, returns_panel, train_dates)

    # 基线：最优单因子（测试段外推）
    best = train_components[0]
    ic_best_test = calc_ic_series(best.panel.loc[test_dates], returns_test).dropna()
    rows = [{
        "method": f"best_single({best.name})",
        "ic_mean_test": ic_best_test.mean() if len(ic_best_test) else float("nan"),
        "ir_test": calc_ir(ic_best_test) if len(ic_best_test) >= 2 else 0.0,
    }]

    panels_out: dict[str, pd.DataFrame] = {}
    for m in method_list:
        # 仅 ic_weighted / orthogonal 的权重由训练段 IC 决定（train_components）；
        # pca / stacking 系列内部已是时序防泄漏，传全样本面板即可。
        if m == "ic_weighted":
            comp = synthesize_ic_weighted(train_components, weight_by="ic_abs")
        elif m == "pca":
            comp = synthesize_pca(components, n_components=min(3, len(components)), returns_panel=returns_panel)
        elif m == "orthogonal":
            comp = synthesize_orthogonal(train_components, weight_by="ic_abs")
        elif m == "stacking":
            comp = synthesize_stacking(components, returns_panel, n_splits=5,
                                       target_mode=args.target_mode)
        elif m == "stacking_gbdt":
            if getattr(args, "tune_trials", 0) and args.tune_trials > 0:
                comp = synthesize_stacking_gbdt_tuned(
                    components, returns_panel, n_trials=args.tune_trials,
                    target_mode=args.target_mode)
            else:
                comp = synthesize_stacking_gbdt(components, returns_panel, n_splits=5,
                                                target_mode=args.target_mode)
        elif m == "lambdarank":
            comp = synthesize_stacking_lambdarank(components, returns_panel, n_splits=5)
        else:
            log.warning("未知方法 %s", m)
            continue
        # 评估一律在测试段（真实外推能力）
        comp_test = comp.loc[test_dates].reindex(columns=returns_test.columns)
        ic_test = calc_ic_series(comp_test, returns_test).dropna()
        bt = VectorBacktest(TopKLongShort(k=30), rebalance_freq="M")
        res = bt.run(comp_test, returns_test)
        mtr = res.metrics()
        rows.append({
            "method": m,
            "ic_mean_test": ic_test.mean() if len(ic_test) else float("nan"),
            "ir_test": calc_ir(ic_test) if len(ic_test) >= 2 else 0.0,
            "sharpe": mtr.get("sharpe"), "annual_return": mtr.get("annual_return"),
            "max_drawdown": mtr.get("max_drawdown"),
        })
        panels_out[m] = comp

    report = pd.DataFrame(rows)
    print("\n===== 合成对比·测试段外推（%d 个 raw 因子参与）=====" % len(components))
    with pd.option_context("display.width", 160, "display.float_format", lambda v: f"{v:.4f}"):
        print(report.to_string(index=False))

    if args.no_save:
        log.info("未入库（--no-save）")
        return

    out_path = Path(args.out) if args.out else Path("reports") / f"synthesis_library_{args.dataset}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    log.info("对比报告: %s", out_path)

    # 入库（v2 命名 + 血缘）。关键：入库用【训练段】returns，使库内 ic_mean/ic_ir
    # 不掺入测试段（未来）信息，避免后续迭代把带 look-ahead 的 IC 再当权重。
    for m, panel in panels_out.items():
        name = f"composite2_{m}"
        lib.register(
            name=name,
            panel=standardize_zscore(panel),
            returns_panel=returns_train,
            kind="composite",
            formula=f"synthesis:{m}({len(components)} raw 因子; train_frac={train_frac})",
            parents=names,
            source=f"synthesis_library:{args.dataset}",
        )
        log.info("已入库 %s（库内 IC 基于训练段）", name)
    log.info("完成。数据集 %s 现有 %d 个因子", args.dataset, len(lib.list_all()))


if __name__ == "__main__":
    main()
