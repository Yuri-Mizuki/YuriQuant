"""次日执行价模式单测：build_execution_split + 引擎 execution_split 结算。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.costs import TransactionCosts
from backtest.engine import VectorBacktest, build_execution_split
from strategy.examples import TopFracLongOnly

ZERO_COSTS = TransactionCosts(commission_rate=0.0, stamp_duty=0.0, slippage_bp=0.0, commission_min=0.0)  # 含最低佣金一并置零


def _panels():
    """3 只股票 × 4 天的手工价量面板（open/close），便于手算校验。"""
    days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    cols = ["A", "B", "C"]
    close = pd.DataFrame(
        [[10.0, 20.0, 30.0], [11.0, 19.0, 33.0],
         [12.0, 18.0, 36.0], [11.0, 21.0, 34.0]],
        index=days, columns=cols)
    open_ = pd.DataFrame(
        [[10.0, 20.0, 30.0], [10.5, 20.5, 31.0],
         [11.5, 19.5, 34.0], [12.0, 20.0, 35.0]],
        index=days, columns=cols)
    return close, open_


def test_build_execution_split_values():
    close, open_ = _panels()
    rb = {pd.Timestamp("2024-01-02")}          # 信号日 01-02，执行日 01-03
    on, idr = build_execution_split(open_, close, rb)
    # 执行日 01-03：隔夜 = O/C(-1)-1 = 10.5/10-1；日内 = C/O-1 = 11/10.5-1
    assert on.loc["2024-01-03", "A"] == pytest.approx(10.5 / 10.0 - 1)
    assert idr.loc["2024-01-03", "A"] == pytest.approx(11.0 / 10.5 - 1)
    # 非执行日：隔夜 = cc，日内 = 0
    assert on.loc["2024-01-04", "A"] == pytest.approx(12.0 / 11.0 - 1)
    assert idr.loc["2024-01-04", "A"] == 0.0
    # 调仓日当天（01-02）不受影响：隔夜 = NaN（首日）
    assert pd.isna(on.loc["2024-01-02", "A"])


def test_engine_next_open_matches_manual():
    """正确配方：信号 shift(1) + 调仓日=执行日。手算对账。"""
    close, open_ = _panels()
    signal = pd.DataFrame({"A": [1.0, np.nan, np.nan, np.nan],
                           "B": [np.nan] * 4, "C": [np.nan] * 4},
                          index=close.index)
    # 信号 01-02 收盘产生 → shift(1) 后在 01-03（执行日）生效
    sig_lag = signal.shift(1)
    exec_day = pd.Timestamp("2024-01-03")
    rb = {exec_day}
    on, idr = build_execution_split(open_, close, {pd.Timestamp("2024-01-02")})

    bt = VectorBacktest(strategy=TopFracLongOnly(frac=1.0, weight_mode="equal"),
                        rebalance_freq="M", initial_capital=1_000_000.0,
                        costs=ZERO_COSTS)
    res = bt.run(sig_lag, close / close.shift(1) - 1.0, horizon=1,
                 rebalance_days=rb, check_convention=False,
                 execution_split=(on, idr))
    eq = res.equity_curve
    # 01-02: 未执行（信号次日生效）→ 空仓
    assert eq.loc["2024-01-02"] == pytest.approx(1.0)
    # 01-03: 开盘 1.0 全仓 A（fill=开盘价 10.5）→ 日内 11/10.5-1
    assert eq.loc["2024-01-03"] == pytest.approx(11.0 / 10.5)
    # 01-04: 收盘持有 12/11
    assert eq.loc["2024-01-04"] == pytest.approx(12.0 / 10.5)
    # 01-05: 11（回吐 01-04 的涨幅）
    assert eq.loc["2024-01-05"] == pytest.approx(11.0 / 10.5)


def test_engine_next_open_vs_close_fill_gap_priced_in():
    """开盘买入价高于信号收盘价时，隔夜缺口被如实计入（对比收盘成交口径）。"""
    close, open_ = _panels()
    signal = pd.DataFrame({"A": [1.0, np.nan, np.nan, np.nan],
                           "B": [np.nan] * 4, "C": [np.nan] * 4},
                          index=close.index)
    rb_signal = {pd.Timestamp("2024-01-02")}
    on, idr = build_execution_split(open_, close, rb_signal)

    # T 收盘成交：01-02 收盘以 10.0 买入 → 01-03 净值 = 11/10
    bt_close = VectorBacktest(strategy=TopFracLongOnly(frac=1.0, weight_mode="equal"),
                              rebalance_freq="M", initial_capital=1.0, costs=ZERO_COSTS)
    r_close = bt_close.run(signal, close / close.shift(1) - 1.0, horizon=1,
                           rebalance_days=rb_signal, check_convention=False)
    # T+1 开盘成交：01-03 以开盘价 10.5 买入 → 净值 = 11/10.5
    bt_open = VectorBacktest(strategy=TopFracLongOnly(frac=1.0, weight_mode="equal"),
                             rebalance_freq="M", initial_capital=1.0, costs=ZERO_COSTS)
    r_open = bt_open.run(signal.shift(1), close / close.shift(1) - 1.0, horizon=1,
                         rebalance_days={pd.Timestamp("2024-01-03")},
                         check_convention=False,
                         execution_split=(on, idr))
    c_close = r_close.equity_curve.loc["2024-01-03"]
    c_open = r_open.equity_curve.loc["2024-01-03"]
    assert c_close == pytest.approx(11.0 / 10.0)          # 收盘成交：赚满 10→11
    assert c_open == pytest.approx(11.0 / 10.5)           # 开盘成交：缺口 10→10.5 未吃，
                                                          # 只赚开盘 10.5→收盘 11 的日内段
    assert c_open < c_close                               # 隔夜缺口被计入


def test_execution_split_rejects_horizon_gt1():
    days = pd.to_datetime([f"2024-01-{d:02d}" for d in range(2, 13)])   # 10 个交易日
    close = pd.DataFrame(100.0, index=days, columns=["A", "B"])
    open_ = close.copy()
    signal = pd.DataFrame(1.0, index=days, columns=["A", "B"])
    on, idr = build_execution_split(open_, close, {days[0]})
    bt = VectorBacktest(strategy=TopFracLongOnly(frac=1.0, weight_mode="equal"),
                        rebalance_freq="M", initial_capital=1.0, costs=ZERO_COSTS)
    with pytest.raises(NotImplementedError, match="仅支持 horizon=1"):
        bt.run(signal, close / close.shift(1) - 1.0, horizon=5,
               rebalance_days={days[0]}, check_convention=False,
               execution_split=(on, idr))
