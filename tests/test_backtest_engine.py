"""
回测引擎测试
============

覆盖 VectorBacktest.run() 的 executable_mask 参数：不可执行标的的权重
应被强制置零并重新归一化，不传 mask 时行为与原有实现完全一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import VectorBacktest, _apply_executable_mask
from strategy.base import Strategy


class EqualWeightTop2(Strategy):
    """测试用策略：等权做多因子值最大的 2 只。"""

    name = "equal_top2"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        top = factor_values.sort_values(ascending=False).index[:2]
        return pd.Series(0.5, index=top)


@pytest.fixture
def small_panels():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    codes = ["A", "B", "C"]
    factor_panel = pd.DataFrame(
        [[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1]],
        index=dates, columns=codes, dtype=float,
    )
    returns_panel = pd.DataFrame(0.01, index=dates, columns=codes)
    return dates, codes, factor_panel, returns_panel


def test_apply_executable_mask_zeroes_and_renormalizes():
    weights = np.array([0.5, 0.5, 0.0])
    executable = np.array([True, False, True])
    out = _apply_executable_mask(weights, executable)
    # B 不可执行 -> 权重归零；A 独占多头，重新归一到原多头之和 1.0
    assert out[1] == 0.0
    assert out[0] == pytest.approx(1.0)
    assert out[2] == 0.0


def test_apply_executable_mask_long_short_preserves_sign_groups():
    weights = np.array([0.5, 0.5, -0.5, -0.5])
    executable = np.array([True, False, True, True])
    out = _apply_executable_mask(weights, executable)
    # 多头一侧只剩 A，归一到原多头之和 1.0；空头两侧都可执行，权重不变
    assert out[0] == pytest.approx(1.0)
    assert out[1] == 0.0
    assert out[2] == pytest.approx(-0.5)
    assert out[3] == pytest.approx(-0.5)


def test_run_without_mask_matches_baseline(small_panels):
    dates, codes, factor_panel, returns_panel = small_panels
    bt = VectorBacktest(strategy=EqualWeightTop2(), rebalance_freq="D", initial_capital=1.0)
    result = bt.run(factor_panel, returns_panel)
    # A、B 权重各 0.5，C 始终 0
    assert (result.weights_history["C"] == 0).all()
    assert result.weights_history.loc[dates[0], "A"] == pytest.approx(0.5)


def test_run_with_mask_excludes_unexecutable_stock(small_panels):
    dates, codes, factor_panel, returns_panel = small_panels
    # A 在第 3 天开始不可执行（例如涨停封板）
    mask = pd.DataFrame(True, index=dates, columns=codes)
    mask.loc[dates[2]:, "A"] = False

    bt = VectorBacktest(strategy=EqualWeightTop2(), rebalance_freq="D", initial_capital=1.0)
    result = bt.run(factor_panel, returns_panel, executable_mask=mask)

    # 第 3 天起 A 权重应为 0，B 顶替占满原多头权重
    assert result.weights_history.loc[dates[2], "A"] == 0.0
    assert result.weights_history.loc[dates[2], "B"] == pytest.approx(1.0)
    # 第 1、2 天（mask 全 True）行为与不加 mask 时一致
    assert result.weights_history.loc[dates[0], "A"] == pytest.approx(0.5)
    assert result.weights_history.loc[dates[0], "B"] == pytest.approx(0.5)
