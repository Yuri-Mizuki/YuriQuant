"""
绩效指标测试
============

用已知输入验证核心指标的计算公式没有被破坏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    annual_return,
    annual_volatility,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    win_rate,
)


def test_annual_return_constant_daily_return():
    # 每日固定收益 r，252 个交易日的年化收益应为 (1+r)^252 - 1
    r = 0.001
    daily = pd.Series([r] * 252)
    expected = (1 + r) ** 252 - 1
    assert annual_return(daily) == pytest.approx(expected, rel=1e-9)


def test_annual_volatility_zero_for_constant_returns():
    daily = pd.Series([0.001] * 100)
    assert annual_volatility(daily) == pytest.approx(0.0)


def test_max_drawdown_simple_path():
    # 净值路径 1 -> 1.2 -> 0.9 -> 1.1，最大回撤 = (1.2-0.9)/1.2 = 0.25
    daily = pd.Series([0.2, -0.25, 0.2222222])
    assert max_drawdown(daily) == pytest.approx(0.25, rel=1e-4)


def test_sharpe_ratio_zero_when_volatility_exactly_zero():
    # annual_volatility 的 0 值分支只在标准差恰好为 0 时触发；用真正常量（std() 严格为 0）
    # 而不是"看起来恒定但存在浮点误差"的序列。
    daily = pd.Series(np.zeros(100))
    assert sharpe_ratio(daily) == 0.0


def test_win_rate_basic():
    daily = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
    assert win_rate(daily) == pytest.approx(2 / 5)


def test_calmar_ratio_zero_when_no_drawdown():
    daily = pd.Series([0.0] * 10)
    assert calmar_ratio(daily) == 0.0
