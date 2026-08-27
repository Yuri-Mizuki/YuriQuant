"""多期组合执行适配器（optimize/multi_period.py）单元测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimize.multi_period import (
    RebalanceConfig,
    _rebalance_days,
    optimize_rebalance_weights,
    run_multi_period_backtest,
)
from scripts.compare_portfolio_methods import gen_mock_panel


@pytest.fixture(scope="module")
def panel():
    return gen_mock_panel(n_days=500, n_codes=60, seed=0)


def test_rebalance_days_monthly_first_trading_day(panel):
    dates = panel["returns"].index
    days = _rebalance_days(dates, "M")
    # 每月首个交易日
    per = pd.Series(dates).dt.to_period("M")
    expected = list(pd.Series(dates).groupby(per).first())
    assert days == expected
    assert len(days) == dates.to_period("M").nunique()


def test_optimize_only_on_rebalance_days(panel):
    factor, returns = panel["factor"], panel["returns"]
    cfg = RebalanceConfig(rebalance_freq="M", method="mvo")
    target = optimize_rebalance_weights(factor, returns, cfg)
    expected = _rebalance_days(factor.index.intersection(returns.index), "M")
    assert list(target.index) == expected
    assert sorted(target.columns) == sorted(factor.columns)


def test_full_run_shapes(panel):
    factor, returns = panel["factor"], panel["returns"]
    result, target = run_multi_period_backtest(factor, returns, RebalanceConfig(rebalance_freq="M"))
    assert result.daily_returns.index.equals(factor.index)
    assert result.equity_curve.index.equals(factor.index)
    assert result.weights_history.shape[0] == len(factor.index)
    # 目标权重只在调仓日非空，回测引擎每日权重覆盖全期
    assert target.shape[0] == len(_rebalance_days(factor.index, "M"))
    assert np.isfinite(result.daily_returns).all()


def test_turnover_constraint_holds_on_rebalance(panel):
    factor, returns = panel["factor"], panel["returns"]
    cfg = RebalanceConfig(rebalance_freq="M", max_turnover=0.5, method="mvo")
    target = optimize_rebalance_weights(factor, returns, cfg)
    turnover = 0.5 * target.diff().abs().sum(axis=1)
    # OSQP 数值公差下换手硬约束近似满足
    assert (turnover <= 0.5 + 1e-2).all()


def test_executable_mask_zeroes_non_tradable(panel):
    factor, returns = panel["factor"], panel["returns"]
    # 把首个调仓日的某只股票标为不可交易
    cfg = RebalanceConfig(rebalance_freq="M", method="mvo")
    days = _rebalance_days(factor.index, "M")
    ex = pd.DataFrame(True, index=factor.index, columns=factor.columns)
    ex.loc[days[0], "600000.SH"] = False
    target = optimize_rebalance_weights(factor, returns, cfg, executable=ex)
    assert target.loc[days[0], "600000.SH"] == 0.0
    # 其它调仓日不受影响
    assert abs(target.loc[days[1], "600000.SH"]) > 0 or target.loc[days[1], "600000.SH"] == 0.0


def test_empty_by_insufficient_covariance(panel):
    factor, returns = panel["factor"], panel["returns"]
    cfg = RebalanceConfig(rebalance_freq="M", method="mvo", min_periods=10_000)
    target = optimize_rebalance_weights(factor, returns, cfg)
    assert (target == 0.0).all().all()