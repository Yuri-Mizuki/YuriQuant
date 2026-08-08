"""
数据层通路测试
==============

覆盖：config 加载、DataSource 抽象接口、DataCache 增量缓存、
Universe point-in-time 成分股查询。用 MockDataSource 模拟数据，不依赖真实凭证。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import Config
from data.cache import DataCache
from data.universe import Universe


def test_config_loads():
    cfg = Config.load()
    assert cfg["datasource"]["type"] in ("amazing_data", "csv")
    assert "begin_date" in cfg["fetch"]


def test_calendar_not_empty(tmp_path: Path, mock_ds):
    cache = DataCache(mock_ds, cache_root=tmp_path)
    cal = cache.get_calendar(20230101, 20240131)
    assert len(cal) > 0
    assert cal[0] >= 20230101
    assert cal[-1] <= 20240131


def test_code_info_schema(mock_ds):
    """get_code_info 返回列与基类 docstring 一致（schema 断言，防实现跑偏）。

    基类约定：index=code，列 symbol/status/pre_close/high_limited/low_limited/price_tick。
    Amazing 实现会把 SDK 的 security_status 归一化为 status；Mock 应保持一致。
    """
    df = mock_ds.get_code_info()
    assert list(df.columns) == [
        "symbol", "status", "pre_close", "high_limited", "low_limited", "price_tick",
    ]
    # index 为股票代码
    assert len(df) == len(mock_ds.MOCK_CODES)
    assert set(df.index) == set(mock_ds.MOCK_CODES)
    # 数值列合理：涨停价 >= 昨收 >= 跌停价，最小变动单位 > 0
    assert (df["high_limited"] >= df["pre_close"]).all()
    assert (df["low_limited"] <= df["pre_close"]).all()
    assert (df["price_tick"] > 0).all()


def test_code_info_rename_normalization(tmp_path: Path, mock_ds):
    """Amazing 实现的归一化路径：security_status → status 由 DataCache 落盘后可见。

    用 cache.get_code_info 落盘（Mock 直通），验证列名统一为小写 schema；
    真实 SDK 的 security_status 列由 datasource 层 rename（无法在无 SDK 环境
    直接单测，映射关系见 data/datasource.py 的 get_code_info 实现）。
    """
    cache = DataCache(mock_ds, cache_root=tmp_path)
    df = cache.get_code_info()
    assert "status" in df.columns
    assert "security_status" not in df.columns


def test_hs300_constituent_count(tmp_path: Path, mock_ds):
    cache = DataCache(mock_ds, cache_root=tmp_path)
    uni = Universe(cache)
    codes = uni.get_hs300(20240101)
    assert len(codes) == 50


def test_daily_kline_first_full_fetch(tmp_path: Path, mock_ds):
    cache = DataCache(mock_ds, cache_root=tmp_path)
    uni = Universe(cache)
    codes = uni.get_hs300(20240101)

    kline = cache.get_daily_kline(codes, 20230101, 20240131)
    assert not kline.empty
    assert kline.index.get_level_values("code").nunique() == len(codes)


def test_daily_kline_incremental_update_only_adds_new_rows(tmp_path: Path, mock_ds):
    cache = DataCache(mock_ds, cache_root=tmp_path)
    uni = Universe(cache)
    codes = uni.get_hs300(20240101)

    kline1 = cache.get_daily_kline(codes, 20230101, 20240131)
    n1 = len(kline1)

    # 模拟数据源新增了 20240201 以后的数据
    mock_ds._cal = mock_ds._gen_calendar(20230101, 20240301)
    kline2 = cache.get_daily_kline(codes, 20230101, 20240301)
    n2 = len(kline2)

    assert n2 > n1
    # 增量更新不应重新产生重复的 (date, code) 行
    assert not kline2.index.duplicated().any()


def test_daily_kline_cache_hit_respects_requested_dates(tmp_path: Path, mock_ds):
    cache = DataCache(mock_ds, cache_root=tmp_path)
    codes = Universe(cache).get_hs300(20240101)

    cache.get_daily_kline(codes, 20230101, 20240131)
    result = cache.get_daily_kline(codes, 20230115, 20230120)

    dates = result.index.get_level_values("date")
    assert dates.min() >= pd.Timestamp("2023-01-15")
    assert dates.max() <= pd.Timestamp("2023-01-20")


def test_backward_factor_cached(tmp_path: Path, mock_ds):
    cache = DataCache(mock_ds, cache_root=tmp_path)
    uni = Universe(cache)
    codes = uni.get_hs300(20240101)

    backward = cache.get_backward_factor(codes)
    assert not backward.empty
    assert set(backward.columns) == set(codes)
    assert (cache.root / "backward_factor.parquet").exists()


def test_history_stock_status_cached(tmp_path: Path, mock_ds):
    cache = DataCache(mock_ds, cache_root=tmp_path)
    uni = Universe(cache)
    codes = uni.get_hs300(20240101)

    status = cache.get_history_stock_status(codes, 20230101, 20240131)
    assert not status.empty
    assert {"high_limited", "low_limited", "is_suspended"}.issubset(
        status.reset_index().columns
    )
    assert (cache.root / "history_stock_status.parquet").exists()


def test_membership_mask_point_in_time(tmp_path: Path, mock_ds):

    cache = DataCache(mock_ds, cache_root=tmp_path)
    uni = Universe(cache)
    dates = pd.date_range("2023-01-01", "2023-01-10", freq="B")

    mask = uni.get_membership_mask("000300.SH", dates)
    # MockDataSource 的成分股 in_date 是 2023-01-03，之前的日期应全为 False
    before = dates[dates < pd.Timestamp("2023-01-03")]
    after = dates[dates >= pd.Timestamp("2023-01-03")]
    if len(before) > 0:
        assert not mask.loc[before].any().any()
    assert mask.loc[after].all().all()


def test_narrow_fetch_does_not_overwrite_cache(tmp_path: Path, mock_ds):
    """窄区间 + 少量 code 的增量查询不得覆盖丢失缓存里其余 code / 日期。

    回归 P0：早期 _refresh_long_table 把按 (codes, 日期) 过滤后的子集写回
    parquet，一次窄查询会永久销毁其余股票 / 历史日期。
    """
    cache = DataCache(mock_ds, cache_root=tmp_path)
    codes = Universe(cache).get_hs300(20240101)
    cache.get_daily_kline(codes, 20230101, 20240131)
    p = cache.root / "daily.parquet"
    full_rows = len(pd.read_parquet(p))
    full_codes = set(pd.read_parquet(p).index.get_level_values("code").unique())

    # 扩展数据源后，仅查 2 只票、半个月新区间（触发 fetch + 写盘）
    mock_ds._cal = mock_ds._gen_calendar(20230101, 20240301)
    cache.get_daily_kline(codes[:2], 20240201, 20240215)

    after = pd.read_parquet(p)
    # 其余 code 与历史日期仍在，未被窄查询覆盖
    assert set(after.index.get_level_values("code").unique()) == full_codes
    assert len(after) >= full_rows


def test_narrow_calendar_query_does_not_truncate(tmp_path: Path, mock_ds):
    """窄区间日历查询不得覆盖丢失已有的完整日历（回归 P0）。"""
    cache = DataCache(mock_ds, cache_root=tmp_path)
    cache.get_calendar(20230101, 20241231)
    p = cache.root / "calendar.parquet"
    full_len = len(pd.read_parquet(p))
    # 窄区间再查一次
    cache.get_calendar(20240101, 20240131)
    assert len(pd.read_parquet(p)) == full_len
