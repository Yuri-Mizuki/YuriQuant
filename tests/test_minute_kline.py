"""
分钟K线数据层测试
=================

覆盖：分钟K线 mock 生成、缓存增量/幂等/窄区间查询不丢历史/新代码回填/
增量扩展、档位校验。对应 data/datasource.get_minute_kline + data/cache.get_minute_kline
（AmazingData 手册 3.5.4.2 query_kline + 4.1.7 Period）。
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.cache import DataCache
from data.datasource import validate_minute_period


def _cache(mock_ds, tmp_path) -> DataCache:
    return DataCache(mock_ds, cache_root=tmp_path)


# ---- 档位校验 ----

def test_validate_minute_period():
    for p in (1, 3, 5, 10, 15, 30, 60, 120):
        assert validate_minute_period(p) == p
    for p in (2, 4, 7, 20, 45, 90, 0, -5):
        with pytest.raises(ValueError):
            validate_minute_period(p)


# ---- mock 数据源生成 ----

def test_mock_min5_shape(mock_ds):
    df = mock_ds.get_minute_kline(mock_ds.MOCK_CODES[:5], 20230103, 20230106, period=5)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["kline_time", "code"]
    # 每交易日 48 根（240//5），4 个交易日 × 5 只
    assert len(df) == 4 * 48 * 5
    # OHLC 一致性：low <= min(open, close)，high >= max(open, close)
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["volume"] > 0).all()
    assert (df["amount"] > 0).all()
    # bar 时间在 A 股交易时段内，且同一 (kline_time, code) 无重复
    times = df.index.get_level_values("kline_time")
    assert times.hour.min() >= 9
    assert times.hour.max() <= 15
    assert not df.index.duplicated().any()


def test_mock_period_bar_counts(mock_ds):
    """各档位每交易日 bar 数 = 240 // period。"""
    codes = mock_ds.MOCK_CODES[:3]
    expected = {1: 240, 3: 80, 5: 48, 15: 16, 30: 8, 60: 4, 120: 2}
    for period, bars in expected.items():
        df = mock_ds.get_minute_kline(codes, 20230103, 20230103, period=period)
        assert len(df) == bars * len(codes), f"period={period}"


def test_mock_minute_sorted_and_intraday_times(mock_ds):
    """5 分钟 bar 时间戳应为 9:30 起每 5 分钟（9:30, 9:35, ... 11:25, 13:00, ... 14:55）。"""
    times = mock_ds._intraday_times(20230103, 5)
    assert len(times) == 48
    assert times[0] == pd.Timestamp("2023-01-03 09:30")
    assert times[23] == pd.Timestamp("2023-01-03 11:25")
    assert times[24] == pd.Timestamp("2023-01-03 13:00")
    assert times[-1] == pd.Timestamp("2023-01-03 14:55")
    assert all((t.minute % 5 == 0) for t in times)


def test_mock_minute_begin_end_time_filter(mock_ds):
    """get_minute_kline 的 begin_time/end_time 日内时段过滤（与 AmazingDataSource 一致）。"""
    codes = mock_ds.MOCK_CODES[:3]
    full = mock_ds.get_minute_kline(codes, 20230103, 20230103, period=5)
    assert len(full) == 48 * 3
    # 只取上午 9:30-11:30（begin_time=930, end_time=1130）
    morning = mock_ds.get_minute_kline(codes, 20230103, 20230103, period=5,
                                       begin_time=930, end_time=1130)
    assert len(morning) == 24 * 3
    assert (morning.index.get_level_values("kline_time").hour < 12).all()
    # 只取尾盘 14:00 之后
    tail = mock_ds.get_minute_kline(codes, 20230103, 20230103, period=5, begin_time=1400)
    assert (tail.index.get_level_values("kline_time").hour == 14).all()
    assert len(tail) == 12 * 3


# ---- 缓存：幂等 / 增量 / 不丢历史 ----

def test_cache_minute_power_idempotent(mock_ds, tmp_path):
    cache = _cache(mock_ds, tmp_path)
    codes = mock_ds.MOCK_CODES[:10]
    df1 = cache.get_minute_kline(codes, 20230103, 20230203, period=5)
    df2 = cache.get_minute_kline(codes, 20230103, 20230203, period=5)
    assert len(df2) == len(df1)
    assert not df2.index.duplicated().any()
    assert (tmp_path / "min5_hs300.parquet").exists()
    on_disk = pd.read_parquet(tmp_path / "min5_hs300.parquet")
    assert len(on_disk) == len(df1)


def test_cache_minute_narrow_query_keeps_history(mock_ds, tmp_path):
    """窄区间查询返回窄区间，但缓存文件保留全量（防历史丢失回归）。"""
    cache = _cache(mock_ds, tmp_path)
    codes = mock_ds.MOCK_CODES[:10]
    full = cache.get_minute_kline(codes, 20230103, 20231231, period=5)
    narrow = cache.get_minute_kline(codes, 20230601, 20230630, period=5)
    # 返回值只含窄区间
    days = {d.date() for d in narrow.index.get_level_values("kline_time").unique()}
    assert 0 < len(days) <= 22
    # 缓存文件仍保留全量
    on_disk = pd.read_parquet(tmp_path / "min5_hs300.parquet")
    assert len(on_disk) == len(full)


def test_cache_minute_new_code_backfill(mock_ds, tmp_path):
    """新增代码应从 begin 全量回填，不被全局 last_date 跳过。"""
    cache = _cache(mock_ds, tmp_path)
    codes_a = mock_ds.MOCK_CODES[:5]
    cache.get_minute_kline(codes_a, 20230103, 20230331, period=5)
    codes_b = mock_ds.MOCK_CODES[5:8]
    df = cache.get_minute_kline(codes_a + codes_b, 20230103, 20230331, period=5)
    new_codes = df.index.get_level_values("code").unique()
    assert all(c in new_codes for c in codes_b)
    # 新票也有完整区间数据（约 60 交易日 × 48 根）
    per_code = df.groupby(level="code").size()
    assert (per_code > 48 * 50).all()


def test_cache_minute_incremental_extension(mock_ds, tmp_path):
    """增量扩展：只补新日期，旧日期行数不变（last_inclusive 从 last 当天重拉）。"""
    cache = _cache(mock_ds, tmp_path)
    codes = mock_ds.MOCK_CODES[:5]
    df1 = cache.get_minute_kline(codes, 20230103, 20230131, period=5)
    df2 = cache.get_minute_kline(codes, 20230103, 20230228, period=5)
    assert len(df2) > len(df1)
    assert len(df2) == len(pd.read_parquet(tmp_path / "min5_hs300.parquet"))
    jan = df2[df2.index.get_level_values("kline_time") < "2023-02-01"]
    assert len(jan) == len(df1)


def test_cache_minute_last_inclusive_backfills_partial_day(mock_ds, tmp_path):
    """半拉一天的场景：分钟 bar 可按日内时段部分拉取，重拉时当天会被补全。"""
    cache = _cache(mock_ds, tmp_path)
    codes = mock_ds.MOCK_CODES[:5]

    # 模拟某天只拉了上午（fetch_fn 截断到 11:30 前的 bar）
    def partial_fetch(c, b, e, period=5):
        df = mock_ds.get_minute_kline(c, b, e, period=period)
        if df.empty:
            return df
        return df[df.index.get_level_values("kline_time").hour < 12]

    # 第一次：整个区间只有 1 天（2023-01-03），且该天只有上午 24 根
    full1 = cache._refresh_long_table(
        "min5.parquet", "min5", codes, 20230103, 20230103, partial_fetch,
        time_col="kline_time", last_inclusive=True,
    )
    assert (full1.index.get_level_values("kline_time").hour < 12).all()

    # 第二次：正常拉全量，2023-01-03 的下午 bar 应被补全（幂等去重）
    full2 = cache.get_minute_kline(codes, 20230103, 20230103, period=5)
    assert len(full2) == 48 * len(codes)
    assert full2.index.get_level_values("kline_time").hour.max() >= 13


def test_cache_minute_multiple_periods_isolated(mock_ds, tmp_path):
    """不同档位缓存文件互相独立，互不串扰。"""
    cache = _cache(mock_ds, tmp_path)
    codes = mock_ds.MOCK_CODES[:5]
    df5 = cache.get_minute_kline(codes, 20230103, 20230106, period=5)
    df15 = cache.get_minute_kline(codes, 20230103, 20230106, period=15)
    assert (tmp_path / "min5_hs300.parquet").exists()
    assert (tmp_path / "min15_hs300.parquet").exists()
    assert len(df5) == 4 * 48 * 5
    assert len(df15) == 4 * 16 * 5


def test_cache_minute_offline_reuse(mock_ds, tmp_path):
    """缓存命中时不再调用数据源（离线可用）。"""
    calls = {"n": 0}

    class TrackingDS:
        def __init__(self, inner):
            self._inner = inner
        def get_minute_kline(self, c, b, e, period=5, **kw):
            calls["n"] += 1
            return self._inner.get_minute_kline(c, b, e, period=period, **kw)
        # 半拉天/覆盖检测会走 get_calendar（cache._refresh_long_table 用交易日
        # 对齐判断"请求区间是否已被本地覆盖"），桩必须透传，否则 AttributeError
        def get_calendar(self, begin=20100101, end=None):
            return self._inner.get_calendar(begin, end)

    cache = DataCache(TrackingDS(mock_ds), cache_root=tmp_path)
    codes = mock_ds.MOCK_CODES[:5]
    cache.get_minute_kline(codes, 20230103, 20230131, period=5)
    assert calls["n"] == 1
    # 第二次拉更早区间（2022 年，本地没有）→ 仍会请求数据源
    cache.get_minute_kline(codes, 20220103, 20220131, period=5)
    assert calls["n"] == 2
    # 第三次拉已缓存区间 → 不再请求数据源（fetch_begin=last=2023-01-31 > end=2023-01-15）
    cache.get_minute_kline(codes, 20230103, 20230115, period=5)
    assert calls["n"] == 2
