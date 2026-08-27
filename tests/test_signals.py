"""每日交易信号生成（optimize/signals.py）+ 方向可交易掩码（data/tradability.py）测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.tradability import build_directional_masks
from optimize.signals import build_signal_frame


@pytest.fixture
def price() -> pd.Series:
    return pd.Series({"600001.SH": 10.0, "600002.SH": 20.0, "600003.SH": 5.0})


def test_basic_buy_and_hold(price):
    target = pd.Series({"600001.SH": 0.5, "600002.SH": 0.5, "600003.SH": 0.0})
    frame = build_signal_frame("2025-01-31", target, price, capital=1_000_000)
    row1 = frame[frame["code"] == "600001.SH"].iloc[0]
    assert row1["direction"] == "BUY"
    assert row1["qty_target"] == int(0.5 * 1_000_000 / 10.0)  # 50000
    assert row1["qty_order"] == 50000          # 整手化 100 股整数
    assert row1["status"] == "OK"
    row3 = frame[frame["code"] == "600003.SH"].iloc[0]
    assert row3["direction"] == "FLAT"
    assert row3["qty_order"] == 0


def test_lot_rounding_toward_zero(price):
    target = pd.Series({"600002.SH": 0.01})  # 目标市值 10000 → 500 股
    cur0 = pd.Series({"600002.SH": 0.0})
    frame = build_signal_frame("2025-01-31", target, price,
                               capital=1_000_000, current_positions=cur0)
    row = frame.iloc[0]
    assert row["qty_target"] == 500
    assert row["qty_order"] == 500
    # 净变化 449 → 整手 400（向零）
    target2 = pd.Series({"600002.SH": 0.01})
    f2 = build_signal_frame("2025-01-31", target2, price, capital=1_000_000,
                            current_positions=pd.Series({"600002.SH": 51.0}))
    assert f2.iloc[0]["qty_delta"] == 500 - 51
    assert f2.iloc[0]["qty_order"] == 400


def test_short_sell_direction(price):
    target = pd.Series({"600001.SH": -0.3})
    frame = build_signal_frame("2025-01-31", target, price, capital=1_000_000)
    row = frame.iloc[0]
    assert row["direction"] == "SELL"
    assert row["qty_target"] == -int(0.3 * 1_000_000 / 10.0)
    assert row["qty_order"] == -30000   # 负向整手


def test_up_limit_blocks_buy(price):
    target = pd.Series({"600001.SH": 0.6, "600002.SH": 0.2})
    buyable = pd.Series({"600001.SH": False, "600002.SH": True})   # 涨停：买不进
    sellable = pd.Series({"600001.SH": True, "600002.SH": True})
    frame = build_signal_frame("2025-01-31", target, price, capital=1_000_000,
                               buyable=buyable, sellable=sellable)
    r = frame[frame["code"] == "600001.SH"].iloc[0]
    assert r["status"] == "BLOCKED_BUY"
    assert r["qty_order"] == 0
    assert "涨停" in r["note"]
    assert (frame[frame["code"] == "600002.SH"].iloc[0]["status"]) == "OK"


def test_down_limit_blocks_sell(price):
    # 当前持仓想卖出，但跌停卖不出
    price2 = pd.Series({"600003.SH": 5.0})
    target = pd.Series({"600003.SH": 0.0})                # 目标清仓
    current = pd.Series({"600003.SH": 20000})             # 有持仓
    buyable = pd.Series({"600003.SH": True})
    sellable = pd.Series({"600003.SH": False})            # 跌停：卖不出
    frame = build_signal_frame("2025-01-31", target, price2, capital=1_000_000,
                               buyable=buyable, sellable=sellable,
                               current_positions=current)
    r = frame[frame["code"] == "600003.SH"].iloc[0]
    assert r["direction"] == "SELL"
    assert r["status"] == "BLOCKED_SELL"
    assert r["qty_order"] == 0
    assert "跌停" in r["note"]


def test_suspension_blocks_both(price):
    target = pd.Series({"600001.SH": 0.3})
    tradable = pd.Series({"600001.SH": False})            # 停牌
    frame = build_signal_frame("2025-01-31", target, price, capital=1_000_000, tradable=tradable)
    r = frame[frame["code"] == "600001.SH"].iloc[0]
    assert r["status"] == "BLOCKED_BUY"
    assert "停牌" in r["note"]


def test_directional_masks_construction():
    dates = pd.DatetimeIndex(["2025-01-02", "2025-01-03"])
    codes = pd.Index(["600001.SH", "600002.SH"])
    d0 = dates[0]
    status = pd.DataFrame({
        "date": [d0, d0],
        "code": ["600001.SH", "600002.SH"],
        "is_suspended": [False, True],          # 600002 停牌
        "high_limited": [11.0, np.nan],          # 600001 涨停价
        "low_limited": [9.0, np.nan],
    })
    close = pd.DataFrame({"600001.SH": [11.0, 10.0], "600002.SH": [5.0, 5.0]}, index=dates)
    buyable, sellable = build_directional_masks(status, dates, codes, close_panel=close)
    # 600001 首日涨停封板 → 只禁买不禁卖
    assert not buyable.at[d0, "600001.SH"]
    assert bool(sellable.at[d0, "600001.SH"])
    # 600002 停牌 → 禁买也禁卖
    assert not buyable.at[d0, "600002.SH"]
    assert not sellable.at[d0, "600002.SH"]
    # 无状态默认全可交易
    b0, s0 = build_directional_masks(None, dates, codes)
    assert b0.all().all() and s0.all().all()