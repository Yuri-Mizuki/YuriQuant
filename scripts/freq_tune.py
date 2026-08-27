"""调仓频率补充精修：horizon ∈ {1,5} × rebalance ∈ {D,W,M}

在已固化的 gbdt + 中性化 + Top20% 基础上，放开"调仓频率"维度的网格，
回答：horizon=1 时哪档换仓节奏真正最优，并确认换手是否会吞掉收益。

口径：test 2025，含成本（佣金3bp+印花0.1%+滑点10bp），Signal = 中性化 gbdt。
复用 scripts/run_model_portfolio 的构建函数，避免重复实现。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from config import Config

log = logging.getLogger("freq_tune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

HORIZONS = [1, 5]
# 只有 调仓区间跨度 >= horizon 的组合合法（引擎守卫会拦截非法组合）：
#   h=1 跨度=1 => D/W/M 全合法；h=5：月度跨度17合法，周度跨度可能<5(节假日) 被拦
VALID = {(1, "D"), (1, "W"), (1, "M"), (5, "M")}
OUT = Path("reports") / "freq_tune"


def main():
    from scripts.run_model_portfolio import (
        load_index_benchmark, build_model_panel, build_style_covariates_panel,
        neutralize_panel, DEFAULT_MODEL_PARAMS,
    )
    from strategy.examples import TopFracLongOnly
    from backtest.engine import VectorBacktest
    from backtest.costs import TransactionCosts
    from data.cache_helpers import build_panel

    t0 = time.time()
    disc = Config.discipline()
    close0, _ = build_panel(Config.get(), disc["begin"], 20261231, offline=True)
    test_days = close0["close"].index[
        close0["close"].index > pd.Timestamp(str(disc["valid_end"]))]
    test_days = test_days[test_days <= pd.Timestamp("2025-12-31")]
    bench_daily = load_index_benchmark(test_days).dropna()
    bench_annual = (1 + bench_daily).prod() ** (252 / len(bench_daily)) - 1

    costs = TransactionCosts(commission_rate=0.0003, stamp_duty=0.001, slippage_bp=10.0)
    strat_fixed = TopFracLongOnly(frac=0.20, weight_mode="equal")

    rows = []
    for hz in HORIZONS:
        # 每个 horizon 训练一次 gbdt（标签前瞻不同）
        pred, panel, fwd_all = build_model_panel("gbdt", hz, test_days)
        fwd = fwd_all.loc[test_days]
        pred = pred.loc[test_days].reindex(columns=fwd.columns)
        cov = build_style_covariates_panel(panel)
        sig = neutralize_panel(pred, cov)

        for freq in ["D", "W", "M"]:
            if (hz, freq) not in VALID:
                continue  # 非法组合（跨度<horizon）已被引擎守卫拦截，跳过
            bt = VectorBacktest(strategy=strat_fixed, rebalance_freq=freq,
                                initial_capital=1_000_000.0, costs=costs)
            res = bt.run(sig, fwd, horizon=hz)
            m = res.metrics(benchmark_returns=bench_daily)
            rows.append({
                "horizon": hz, "freq": freq,
                "annual": m.get("annual_return", 0),
                "excess": m.get("excess_return", 0),
                "sharpe": m.get("sharpe", 0),
                "ir": m.get("information_ratio", 0),
                "max_dd": m.get("max_drawdown", 0),
                "turnover": m.get("avg_turnover", 0),
            })
            log.info(f"h={hz} freq={freq} | 年化={rows[-1]['annual']:.2%} "
                     f"超额={rows[-1]['excess']:+.2%} 换手={rows[-1]['turnover']:.1%} | "
                     f"{time.time()-t0:.0f}s")

    table = pd.DataFrame(rows)
    print(f"\n===== 调仓频率精修（gbdt中性化Top20%, test 2025, 含成本）=====")
    print(f"沪深300指数年化: {bench_annual:.2%}\n")
    for hz in HORIZONS:
        print(f"--- horizon={hz} ---")
        with pd.option_context("display.width", 200, "display.float_format",
                               lambda v: f"{v:.4f}"):
            print(table[table.horizon == hz].sort_values("excess", ascending=False)
                  .to_string(index=False))
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "freq_tune.csv", index=False, encoding="utf-8-sig")
    print(f"总耗时 {time.time()-t0:.0f}s | 保存 {OUT}")


if __name__ == "__main__":
    main()