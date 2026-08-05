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
from factor.preprocessing import standardize_zscore
from factor.synthesis import (
    CompositeInput,
    synthesize_ic_weighted,
    synthesize_orthogonal,
    synthesize_pca,
    synthesize_stacking,
)
from research.factor_analysis import calc_ic_series, calc_ir
from research.factor_library import FactorLibrary
from backtest.engine import VectorBacktest
from strategy.examples import TopKLongShort

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("synthesize_library")

METHODS = ["ic_weighted", "pca", "orthogonal", "stacking"]


class _OfflineDataSource:
    def _raise(self, *a, **k):
        raise RuntimeError("offline 模式不连接数据源")

    get_calendar = get_code_list = get_index_constituent = _raise
    get_daily_kline = get_minute_kline = get_adj_factor = get_backward_factor = _raise
    get_code_info = get_history_stock_status = get_industry_classification = _raise
    get_equity_structure = get_balance_sheet = get_cash_flow = get_income = _raise


def returns_from_cache(cache, begin: int, end: int) -> pd.DataFrame:
    """从日线缓存构造次日收益面板（与因子库 IC 口径一致）。"""
    cal = cache.get_calendar(begin, end)
    if not cal:
        raise RuntimeError("交易日历为空")
    d = pd.read_parquet(Path(cache.root) / "daily.parquet")
    close_w = d.reset_index().pivot(index="date", columns="code", values="close").sort_index()
    close_w = close_w.loc[str(begin): str(end)]
    return close_w.pct_change().shift(-1)


def main():
    parser = argparse.ArgumentParser(description="因子库内因子合并合成（离线）")
    parser.add_argument("--dataset", default="hs300_2025")
    parser.add_argument("--methods", default="all", help="ic_weighted,pca,orthogonal,stacking")
    parser.add_argument("--begin", type=int, default=20250101)
    parser.add_argument("--end", type=int, default=20251231)
    parser.add_argument("--out", default=None, help="对比报告 CSV（默认 reports/）")
    parser.add_argument("--no-save", action="store_true", help="只对比不入库")
    args = parser.parse_args()

    cache = DataCache(_OfflineDataSource())
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

    # 基线：最优单因子
    best = components[0]
    ic_best = calc_ic_series(best.panel, returns_panel)
    rows = [{
        "method": f"best_single({best.name})",
        "ic_mean": ic_best.mean(), "ir": calc_ir(ic_best),
    }]

    panels_out: dict[str, pd.DataFrame] = {}
    for m in method_list:
        if m == "ic_weighted":
            comp = synthesize_ic_weighted(components, weight_by="ic_abs", returns_panel=returns_panel)
        elif m == "pca":
            comp = synthesize_pca(components, n_components=min(3, len(components)), returns_panel=returns_panel)
        elif m == "orthogonal":
            comp = synthesize_orthogonal(components, weight_by="ic_abs")
        elif m == "stacking":
            comp = synthesize_stacking(components, returns_panel, n_splits=5)
        else:
            log.warning("未知方法 %s", m)
            continue
        ic = calc_ic_series(comp, returns_panel)
        bt = VectorBacktest(TopKLongShort(k=30), rebalance_freq="M")
        res = bt.run(comp, returns_panel)
        mtr = res.metrics()
        rows.append({
            "method": m, "ic_mean": ic.mean(), "ir": calc_ir(ic),
            "sharpe": mtr.get("sharpe"), "annual_return": mtr.get("annual_return"),
            "max_drawdown": mtr.get("max_drawdown"),
        })
        panels_out[m] = comp

    report = pd.DataFrame(rows)
    print("\n===== 合成对比（%d 个 raw 因子参与）=====" % len(components))
    with pd.option_context("display.width", 160, "display.float_format", lambda v: f"{v:.4f}"):
        print(report.to_string(index=False))

    if args.no_save:
        log.info("未入库（--no-save）")
        return

    out_path = Path(args.out) if args.out else Path("reports") / f"synthesis_library_{args.dataset}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    log.info("对比报告: %s", out_path)

    # 入库（v2 命名 + 血缘）
    for m, panel in panels_out.items():
        name = f"composite2_{m}"
        lib.register(
            name=name,
            panel=standardize_zscore(panel),
            returns_panel=returns_panel,
            kind="composite",
            formula=f"synthesis:{m}({len(components)} raw 因子)",
            parents=names,
            source=f"synthesis_library:{args.dataset}",
        )
        log.info("已入库 %s", name)
    log.info("完成。数据集 %s 现有 %d 个因子", args.dataset, len(lib.list_all()))


if __name__ == "__main__":
    main()
