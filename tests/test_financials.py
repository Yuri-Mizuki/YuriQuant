"""财务数据 PIT 展开测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.financials import build_pit_panel, build_pit_panels, INCOME_FIELDS


@pytest.fixture
def calendar():
    return pd.date_range("2024-01-01", "2024-01-25", freq="B")


@pytest.fixture
def reports():
    # code A: 1/10 公布 100，1/20 公布 200（修订/新版）
    # code B: 1/15 公布 50
    return pd.DataFrame({
        "code": ["A", "A", "B"],
        "ann_date": pd.to_datetime(["2024-01-10", "2024-01-20", "2024-01-15"]),
        "field": [100.0, 200.0, 50.0],
    })


def test_pit_no_lookahead_before_report(calendar, reports):
    p = build_pit_panel(reports, calendar, "field")
    # 公告前应为 NaN（不能偷看未来）
    assert np.isnan(p.loc[pd.Timestamp("2024-01-09"), "A"])
    assert np.isnan(p.loc[pd.Timestamp("2024-01-12"), "B"])


def test_pit_uses_announcement_date(calendar, reports):
    p = build_pit_panel(reports, calendar, "field")
    # A 在 1/12 看到的应是 1/10 公布的 100
    assert p.loc[pd.Timestamp("2024-01-12"), "A"] == 100.0
    # A 在 1/22 看到的应是 1/20 公布的 200（最新版）
    assert p.loc[pd.Timestamp("2024-01-22"), "A"] == 200.0
    # B 在 1/16 看到 1/15 公布的 50
    assert p.loc[pd.Timestamp("2024-01-16"), "B"] == 50.0


def test_pit_forward_fill(calendar, reports):
    p = build_pit_panel(reports, calendar, "field")
    # 持续前向填充
    assert p.loc[pd.Timestamp("2024-01-25"), "A"] == 200.0
    assert p.loc[pd.Timestamp("2024-01-25"), "B"] == 50.0


def test_pit_unknown_field_raises(calendar, reports):
    with pytest.raises(KeyError):
        build_pit_panel(reports, calendar, "NOT_A_FIELD")


def test_build_pit_panels_multi(calendar, reports):
    rep = reports.rename(columns={"field": "rev"}).copy()
    rep["earnings"] = rep["rev"] * 0.1
    out = build_pit_panels(rep, calendar, ["rev", "earnings"])
    assert set(out.keys()) == {"rev", "earnings"}
    assert out["rev"].loc[pd.Timestamp("2024-01-22"), "A"] == 200.0


def test_income_fields_present():
    # 关键基本面字段应在白名单中
    assert "OPERA_REV" in INCOME_FIELDS
    assert "NET_PRO_INCL_MIN_INT_INC" in INCOME_FIELDS
    assert "BASIC_EPS" in INCOME_FIELDS


def test_pit_int_calendar_not_epoch_nanos(calendar, reports):
    """回归：get_calendar 返回 int 型 YYYYMMDD 列表，必须按日期解析，
    不能把整数当成"自 epoch 起的纳秒"（会生成 1970 年的垃圾时间戳）。"""
    int_cal = [int(d.strftime("%Y%m%d")) for d in calendar]
    p = build_pit_panel(reports, int_cal, "field")
    # 索引应为真正的 2024 交易日，而非 1970
    assert p.index[0].year == 2024
    assert np.isnan(p.loc[pd.Timestamp("2024-01-09"), "A"])  # 1/10 公告前为 NaN
    assert np.isnan(p.loc[pd.Timestamp("2024-01-12"), "B"])  # 1/15 公告前为 NaN
    assert p.loc[pd.Timestamp("2024-01-12"), "A"] == 100.0   # 1/10 公布后前向填充

