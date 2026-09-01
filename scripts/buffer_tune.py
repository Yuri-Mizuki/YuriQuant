"""执行阻尼实验：排名缓冲带 vs 硬切 Top20%（2026-09-01）

在 freq_tune 的固化口径（gbdt + 风格中性化 + Top20% 重仓多头, h=1,
test 2025, 含成本）下比较三种执行方式：
  - naive_top20 : TopFracLongOnly(0.20)，现行配置
                  （freq_tune 基准：超额 +6.1% / Sharpe 1.55 / 单次换手 66%）
  - buffer_20_30: 进 Top20% / 跌出 Top30% 才卖
  - buffer_20_40: 进 Top20% / 跌出 Top40% 才卖

回答：把进出门槛分离、压掉截止线附近的排名噪声换手，能否在不动模型的前提下
提升成本后收益。turnover 口径 = 单边、按调仓事件平均（BUG-2 修复后）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.metrics import PERIODS_PER_YEAR  # noqa: E402
from config import Config  # noqa: E402
from scripts.cli_common import setup_logging  # noqa: E402

log = setup_logging("buffer_tune")

OUT = Path("reports") / "buffer_tune"


def main():
    from backtest.engine import VectorBacktest
    from data.cache_helpers import build_panel
    from scripts.run_model_portfolio import (
        build_model_panel,
        build_style_covariates_panel,
        default_costs,
        load_index_benchmark,
        neutralize_panel,
    )
    from strategy.examples import BufferedTopFracLongOnly, TopFracLongOnly

    t0 = time.time()
    disc = Config.discipline()
    close0, _ = build_panel(Config.get(), disc["begin"], 20261231, offline=True)
    test_days = close0["close"].index[
        close0["close"].index > pd.Timestamp(str(disc["valid_end"]))]
    test_days = test_days[test_days <= pd.Timestamp("2025-12-31")]
    bench_daily = load_index_benchmark(test_days).dropna()
    bench_annual = (1 + bench_daily).prod() ** (PERIODS_PER_YEAR / len(bench_daily)) - 1

    costs = default_costs()
    pred, panel, fwd_all = build_model_panel("gbdt", 1, test_days)
    # 回测收益口径（engine 约定）：h=1 用未 shift 的 pct_change()
    fwd = panel["close"].pct_change(fill_method=None).reindex(test_days)
    pred = pred.loc[test_days].reindex(columns=fwd.columns)
    cov = build_style_covariates_panel(panel)
    sig = neutralize_panel(pred, cov)

    variants = [
        ("naive_top20", lambda: TopFracLongOnly(0.20)),
        ("buffer_20_30", lambda: BufferedTopFracLongOnly(0.20, 0.30)),
        ("buffer_20_40", lambda: BufferedTopFracLongOnly(0.20, 0.40)),
    ]
    rows = []
    for name, make_strat in variants:
        # 有状态策略：每次回测必须全新实例（跨回测复用 = 期末持仓泄漏为前视）
        bt = VectorBacktest(strategy=make_strat(), rebalance_freq="M",
                            initial_capital=1_000_000.0, costs=costs)
        res = bt.run(sig, fwd, horizon=1)
        m = res.metrics(benchmark_returns=bench_daily)
        rows.append({
            "variant": name,
            "annual": m.get("annual_return", 0),
            "excess": m.get("excess_return", 0),
            "sharpe": m.get("sharpe", 0),
            "ir": m.get("information_ratio", 0),
            "max_dd": m.get("max_drawdown", 0),
            "turnover": m.get("avg_turnover", 0),
        })
        log.info("%s | 年化=%.2f%% 超额=%+.2f%% 换手=%.1f%% | %.0fs",
                 name, rows[-1]["annual"] * 100, rows[-1]["excess"] * 100,
                 rows[-1]["turnover"] * 100, time.time() - t0)

    table = pd.DataFrame(rows)
    print("\n===== 执行阻尼实验（gbdt中性化Top20%, h=1, M, test 2025, 含成本）=====")
    print(f"沪深300指数年化: {bench_annual:.2%}\n")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.4f}"):
        print(table.to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "buffer_tune.csv", index=False, encoding="utf-8-sig")
    print(f"总耗时 {time.time()-t0:.0f}s | 保存 {OUT}")


if __name__ == "__main__":
    main()
