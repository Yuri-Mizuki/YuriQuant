"""`year_offset` 的错位方向（前视 bug 的唯一防线，2026-11-07）。

为什么必须有这个文件
--------------------
2026-11-04 的样本外验证查出 `year_offset()` 取的是**未来一年**的报表
（``_off(p) = v(p) − v(p+1y)``），TODO 里标了「已修复」；11-07 复查发现**代码里没有**
（``git log -S "DateOffset(years=1)" --all`` 全库只有一条，方向是 ``−1y``）。后果：

* 10-17 建的那批面板仍然是污染态（`asset_growth_yoy` 近 250 日有 98% 的股票一动不动）；
* 若照原计划直接重建，只是把同一份 bug 重算一遍。

这个 bug 难抓的地方在于**两种写法在正常年份上都出数**：都非空、都"像"同比，
差别只在最近一年（写错时最近 4 个报告期永远缺 ``p+1y`` ⇒ 尾部冻结）。所以断言必须
直接钉住方向，不能只测"有没有算出来"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.builders.build_alla_constructed_factors import year_offset

N_Q = 24                      # 6 年季度
CODES = ["000001.SZ", "600000.SH"]
FIRST = pd.Timestamp("2019-03-31")


def _panel(values: np.ndarray) -> pd.DataFrame:
    """把 ``values[code, quarter]`` 摊成 year_offset 吃的长表（季末报告期）。"""
    rows = []
    periods = pd.date_range(FIRST, periods=N_Q, freq="QE")
    for i, code in enumerate(CODES):
        for j, p in enumerate(periods):
            rows.append({"code": code, "ann_date": p + pd.Timedelta(days=30),
                         "report_period": p, "V": float(values[i, j])})
    return pd.DataFrame(rows)


def _idx(period: pd.Timestamp) -> int:
    """报告期 → 该股在面板里的列号。"""
    return (period.year - FIRST.year) * 4 + (period.month - 3) // 3


def _quarterly(values: np.ndarray, mode: str = "delta") -> pd.DataFrame:
    out = year_offset(_panel(values), "V", mode)
    return out.sort_values(["code", "report_period"]).reset_index(drop=True)


def test_offset_is_positive_when_the_series_only_grows():
    """单调递增序列 ⇒ 同比差必须全是正的。

    写反方向时 ``_off(p) = v(p) − v(p+1y)`` 恒为负——一个符号就能分开两种实现，
    不需要知道具体数值。
    """
    vals = np.tile(np.arange(1.0, N_Q + 1), (len(CODES), 1))
    got = _quarterly(vals)["_off_V"].dropna()
    assert len(got) > 0
    assert (got > 0).all(), "同比差出现负值 ⇒ 取的是未来一年"


def test_offset_equals_current_minus_same_quarter_last_year():
    vals = np.tile(np.arange(1.0, N_Q + 1), (len(CODES), 1))
    for _, row in _quarterly(vals).dropna(subset=["_off_V"]).iterrows():
        j = _idx(row["report_period"])
        assert row["_off_V"] == pytest.approx(vals[0, j] - vals[0, j - 4])


def test_the_last_four_report_periods_are_not_empty():
    """最近一年必须有值。

    这是 bug 最直接的症状：``p+1y`` 还不存在 ⇒ 写反时最近 4 期 `_off` 全空，
    PIT 展开只能拿一年前的旧值前填，面板于是"冻结"。
    """
    vals = np.tile(np.arange(1.0, N_Q + 1), (len(CODES), 1))
    tail = _quarterly(vals).groupby("code", group_keys=False).tail(4)
    assert tail["_off_V"].notna().all()


def test_changing_a_future_report_does_not_move_the_past():
    """扰动 ``p+1y`` 的值，``p`` 的同比差必须一毫不变。

    前面三条测"方向看着对不对"，这条测因果：只要未来的数字进了 ``_off(p)`` 的算式，
    面板就是前视的，无论它当下长得像不像。
    """
    base = np.tile(np.arange(1.0, N_Q + 1), (len(CODES), 1))
    bumped = base.copy()
    bumped[:, 12:] *= 10.0                     # 第 4 年之后的报表整体放大
    a = _quarterly(base).set_index(["code", "report_period"])["_off_V"]
    b = _quarterly(bumped).set_index(["code", "report_period"])["_off_V"]
    cut = FIRST + pd.DateOffset(years=3)
    early = a.index.get_level_values("report_period") < cut
    pd.testing.assert_series_equal(a[early], b[early], check_names=False)


def test_ratio_mode_matches_current_over_last_year_minus_one():
    vals = np.tile(np.arange(1.0, N_Q + 1), (len(CODES), 1))
    out = year_offset(_panel(vals), "V", "ratio")
    for _, row in out.dropna(subset=["_off_V"]).iterrows():
        j = _idx(row["report_period"])
        assert row["_off_V"] == pytest.approx(vals[0, j] / vals[0, j - 4] - 1.0)


def test_missing_field_is_a_no_op():
    df = _panel(np.ones((len(CODES), N_Q)))
    assert year_offset(df, "NOT_THERE", "delta") is df


def test_row_count_is_preserved():
    vals = np.tile(np.arange(1.0, N_Q + 1), (len(CODES), 1))
    df = _panel(vals)
    out = year_offset(df, "V", "delta")
    assert len(out) == len(df)
    assert not out.duplicated(subset=["code", "ann_date", "report_period"]).any()
