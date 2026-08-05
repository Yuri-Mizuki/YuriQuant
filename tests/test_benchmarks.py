"""基准序列模块测试：等权/买入持有基准 + 对照指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.benchmarks import (
    benchmark_returns, buy_hold_returns, compare_to_benchmark,
    equal_weight_returns, with_benchmark_metrics,
)


@pytest.fixture
def price_panel():
    rng = np.random.default_rng(21)
    idx = pd.date_range("2023-01-01", periods=120, freq="B")
    codes = ["A", "B", "C", "D"]
    # 确定性的价格：A 强趋势、B 弱趋势、C/D 横盘
    base = pd.DataFrame({
        "A": 10 * (1.0008 ** np.arange(120)),
        "B": 10 * (1.0002 ** np.arange(120)),
        "C": 10 + np.sin(np.arange(120) / 10),
        "D": 10 + np.cos(np.arange(120) / 10),
    }, index=idx)
    return base + rng.normal(0, 0.01, (120, 4))


def test_equal_weight_returns_shape(price_panel):
    r = equal_weight_returns(price_panel)
    assert len(r) == len(price_panel)
    assert r.iloc[0] == pytest.approx(0.0) or np.isnan(r.iloc[0])  # 首日无收益


def test_buy_hold_vs_equal_weight(price_panel):
    """强趋势股权重下，买入持有（权重漂移）年化应 ≥ 等权（再平衡）。"""
    ew = equal_weight_returns(price_panel).dropna()
    bh = buy_hold_returns(price_panel).dropna()
    ew_ann = (1 + ew).prod() ** (252 / len(ew)) - 1
    bh_ann = (1 + bh).prod() ** (252 / len(bh)) - 1
    # A 股强趋势，买入持有应跑赢每日再平衡
    assert bh_ann >= ew_ann - 1e-9


def test_benchmark_returns_modes(price_panel):
    ew = benchmark_returns(price_panel, mode="equal_weight")
    bh = benchmark_returns(price_panel, mode="buy_hold")
    assert len(ew) == len(bh) == len(price_panel)


def test_compare_to_benchmark_basic(price_panel):
    strategy = price_panel["A"].pct_change()          # 强趋势股收益
    bench = equal_weight_returns(price_panel)
    out = compare_to_benchmark(strategy, bench)
    assert {"excess_annual", "information_ratio", "tracking_error",
            "correlation", "beta"}.issubset(out.keys())
    assert out["excess_annual"] > 0                   # A 强趋势 > 等权
    assert 0.0 <= out["correlation"] <= 1.0
    assert np.isfinite(out["tracking_error"])


def test_with_benchmark_metrics(price_panel):
    from backtest.metrics import calc_all_metrics
    strategy = price_panel["A"].pct_change().dropna()
    metrics = calc_all_metrics(strategy)
    bench = equal_weight_returns(price_panel).dropna()
    enriched = with_benchmark_metrics(metrics, strategy, bench)
    assert enriched["benchmark"]["excess_annual"] > 0
    # 原 dict 未被修改
    assert "benchmark" not in metrics
