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
    sortino_ratio,
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


# ---------------------------------------------------------------------------
# Sortino：下行偏差必须用半方差（2026-08-03 修复，原实现误用负收益日 std）
# ---------------------------------------------------------------------------
def test_sortino_uses_downside_deviation():
    """Sortino 分母 = 半方差 sqrt(mean(min(excess,0)^2))，而非负收益日 std。"""
    daily = pd.Series([0.02, -0.05, -0.03, 0.01, -0.04])
    r = sortino_ratio(daily, periods_per_year=1)
    ar = annual_return(daily, periods_per_year=1)   # 年化口径与实现一致
    dd = float(np.sqrt(np.mean(np.minimum(daily.values, 0.0) ** 2)))
    assert r == pytest.approx(ar / dd)


def test_sortino_differs_from_negative_day_std():
    """构造"负收益日之间离差小但亏损深"的序列，验证口径差异可感知。

    负收益日 std 度量亏损日离散度（≈0），半方差度量亏损深度（>0），
    修复后 Sortino 不应再等于 年化收益/(负收益日std×√252)。
    """
    daily = pd.Series([0.01] * 40 + [-0.10, -0.10, -0.10])   # 亏损日几乎相同
    r = sortino_ratio(daily)
    excess = daily.values
    dd = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    ar = (1 + daily).prod() ** (252 / len(daily)) - 1
    assert r == pytest.approx(ar / (dd * np.sqrt(252)))
    # 旧口径（负收益日 std）会远小于半方差 → 旧 Sortino 会大很多
    neg_std = daily[daily < 0].std()
    assert neg_std < dd, "构造前提：负收益日离差 < 半方差"


def test_sortino_zero_when_no_downside():
    daily = pd.Series([0.01] * 10)
    assert sortino_ratio(daily) == 0.0
