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
    import pandas as pd

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
