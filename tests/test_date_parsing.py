"""日期列解析：`pd.to_datetime(int)` 的 1970 陷阱（2026-11-07）。

`pd.to_datetime` 只按**纳秒**解释整数，而这批 SDK 表的日期列有**三种**形态：
datetime64、epoch 整数（实测微秒，`1_453_334_400_000_000` = 2016-01-21）、
`YYYYMMDD` 整数。按纳秒读，前两种全部落到 1970。

这个错**不抛异常**，症状是"面板整块恒定"：所有事件日期都早于任何交易日，
PIT 的 ffill 于是把每只股票的最后一个截面铺满全历史。2026-11-07 全库体检发现
31 块面板正是这个形状（其中 4 个还进了 920 模型的特征清单），
根因就在这九张事件表的日期列上。

所以这里既测解析器，也测**后果**（`build_pit_panel` 出来的面板必须随时间变化）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.financials import build_pit_panel, to_datetime_col

# 2016-01-21 00:00:00 UTC 的三种整数写法（本库 SDK 落盘用微秒）
_US = 1_453_334_400_000_000
_MS = 1_453_334_400_000
_S = 1_453_334_400


@pytest.mark.parametrize("raw", [_US, _MS, _S], ids=["us", "ms", "s"])
def test_epoch_integers_of_any_unit_land_in_2016(raw):
    got = to_datetime_col(pd.Series([raw]))
    assert got.dt.year.tolist() == [2016]


def test_yyyymmdd_integers_are_not_read_as_epoch():
    got = to_datetime_col(pd.Series([20261106, 19991231]))
    assert got.dt.strftime("%Y-%m-%d").tolist() == ["2026-11-06", "1999-12-31"]


def test_yyyymmdd_strings_and_iso_strings_both_work():
    a = to_datetime_col(pd.Series(["20261106"]))
    b = to_datetime_col(pd.Series(["2026-11-06"]))
    assert a.iloc[0] == b.iloc[0]


def test_datetime_columns_pass_through():
    s = pd.to_datetime(pd.Series(["2026-11-06", "2025-11-06"]))
    assert to_datetime_col(s).tolist() == s.tolist()


def test_nan_and_nat_survive():
    got = to_datetime_col(pd.Series([_US, np.nan]))
    assert got.notna().tolist() == [True, False]


def test_pit_panel_is_not_a_single_broadcast_cross_section():
    """端到端：ann_date 是 epoch 微秒时，PIT 面板必须**随时间变化**。

    解析错误（落到 1970）时这块面板的每一行都等于最后一行 —— 一个截面复制 N 遍。
    """
    cal = pd.bdate_range("2016-01-01", "2016-12-31")
    codes = ["A", "B"]
    # 两只股票各两条公告，日期相差半年
    rows = []
    for code, (d1, d2), (v1, v2) in zip(
            codes,
            [(pd.Timestamp("2016-02-01"), pd.Timestamp("2016-08-01"))] * 2,
            [(1.0, 2.0), (10.0, 20.0)]):
        for d, v in ((d1, v1), (d2, v2)):
            rows.append({"code": code, "ann_date": int(d.value // 1_000),
                         "x": v})
    df = pd.DataFrame(rows)

    panel = build_pit_panel(df, cal, "x")
    assert panel.shape[1] == 2
    assert panel.loc["2016-03-01", "A"] == pytest.approx(1.0)
    assert panel.loc["2016-09-01", "A"] == pytest.approx(2.0)
    assert panel.loc["2016-09-01", "B"] == pytest.approx(20.0)
    # 关键：不是"全历史一个截面"
    assert panel["A"].nunique() > 1
    assert not panel.fillna(-1).nunique(axis=1).eq(1).all()


def test_pit_panel_does_not_backfill_before_the_first_announcement():
    """1970 那个错还会让**公告日之前**就有值（ffill 从 1970 起）。"""
    cal = pd.bdate_range("2016-01-01", "2016-12-31")
    df = pd.DataFrame([{"code": "A",
                        "ann_date": int(pd.Timestamp("2016-08-01").value // 1_000),
                        "x": 5.0}])
    panel = build_pit_panel(df, cal, "x")
    assert pd.isna(panel.loc["2016-03-01", "A"])
    assert panel.loc["2016-09-01", "A"] == pytest.approx(5.0)
