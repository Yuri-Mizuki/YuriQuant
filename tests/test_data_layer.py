"""
数据层通路测试
==============

覆盖：config 加载、DataSource 抽象接口、DataCache 增量缓存、
Universe point-in-time 成分股查询。用 MockDataSource 模拟数据，不依赖真实凭证。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

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


def test_wide_table_narrow_request_keeps_cached_columns(tmp_path: Path, mock_ds):
    """窄池请求不得丢弃已缓存列：每日增量更新只传当期成分并集，
    若按请求列过滤落盘，历史成员的复权因子列会被永久清掉（幸存者偏差）。"""
    cache = DataCache(mock_ds, cache_root=tmp_path)
    uni = Universe(cache)
    codes = sorted(uni.get_hs300(20240101))

    full = cache.get_backward_factor(codes)
    subset = codes[: len(codes) // 2]
    cache.get_backward_factor(subset)

    cached = pd.read_parquet(cache.root / "backward_factor.parquet")
    assert set(cached.columns) == set(full.columns)

    # 请求列的数据仍以新拉取为准（keep="last" 去重）
    again = cache.get_backward_factor(subset)
    assert set(again.columns) == set(subset)
    assert again.notna().all().all()


@pytest.mark.parametrize(
    "table", ["equity_structure", "dividend", "share_holder", "holder_num", "balance_sheet"]
)
def test_sparse_table_narrow_request_keeps_cached_rows(tmp_path: Path, mock_ds, table: str):
    """窄池请求不得丢弃已缓存行：稀疏事件表（股本结构/分红/股东/财务）按 code
    整表覆盖，若落盘只写本次请求 code 的行，历史成员的事件会被永久清掉
    （幸存者偏差——已退出成分的基本面因子构建拿不到历史公告数据）。"""
    cache = DataCache(mock_ds, cache_root=tmp_path)
    codes = sorted(mock_ds.MOCK_CODES)
    getter = getattr(cache, f"get_{table}")

    full = getter(codes)
    assert not full.empty
    subset = codes[: len(codes) // 2]
    getter(subset)

    cached = pd.read_parquet(cache.root / f"{table}.parquet")
    assert set(cached["code"].unique()) == set(codes)

    # 返回值仍按请求 code 过滤（请求 code 的行以新拉取为准）
    again = getter(subset)
    assert set(again["code"].unique()) == set(subset)


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
    p = cache.root / "daily_hs300.parquet"
    full_rows = len(pd.read_parquet(p))
    full_codes = set(pd.read_parquet(p).index.get_level_values("code").unique())

    # 扩展数据源后，仅查 2 只票、半个月新区间（触发 fetch + 写盘）
    mock_ds._cal = mock_ds._gen_calendar(20230101, 20240301)
    cache.get_daily_kline(codes[:2], 20240201, 20240215)

    after = pd.read_parquet(p)
    # 其余 code 与历史日期仍在，未被窄查询覆盖
    assert set(after.index.get_level_values("code").unique()) == full_codes
    assert len(after) >= full_rows


def test_build_panel_core(tmp_path: Path, mock_ds):
    """build_panel 返回 (panel, returns)，含基础行情 + 财务 PIT 字段，口径一致。"""
    cache = DataCache(mock_ds, cache_root=tmp_path)
    cfg = {"universe": {"index_code": "000300.SH", "adjust": "backward"}}

    from data.cache_helpers import build_panel
    panel, returns = build_panel(cfg, 20230101, 20231231, cache=cache)

    # 基础行情字段
    for f in ("close", "open", "high", "low", "volume", "amount"):
        assert f in panel, f"缺字段 {f}"
        assert panel[f].shape == panel["close"].shape
    assert panel["close"].index.is_monotonic_increasing
    # 财务 PIT 字段（Mock 的 income 含 OPERA_REV / balance 含 TOTAL_ASSETS）
    assert "OPERA_REV" in panel
    assert "TOTAL_ASSETS" in panel
    assert "TOT_SHARE_EQUITY_EXCL_MIN_INT" in panel
    # returns = close.pct_change().shift(-1)，形状/索引与 close 对齐
    assert returns.shape == panel["close"].shape
    assert returns.index.equals(panel["close"].index)
    assert returns.columns.equals(panel["close"].columns)
    # returns 与"用同一 close 现算的次日收益"逐元素一致（口径不漂移）
    expected = panel["close"].pct_change().shift(-1)
    pd.testing.assert_frame_equal(returns, expected)
    # 至少存在有效收益（首行因 mask/复权可能为 NaN，不强制）
    assert returns.notna().any().any()


def test_build_panel_market_cap(tmp_path: Path, mock_ds):
    """include_market_cap=True 时 panel 额外含 close_m / market_cap / mask。"""
    cache = DataCache(mock_ds, cache_root=tmp_path)
    cfg = {"universe": {"index_code": "000300.SH", "adjust": "backward"}}

    from data.cache_helpers import build_panel
    panel, _ = build_panel(cfg, 20230101, 20231231, cache=cache, include_market_cap=True)

    assert "close_m" in panel
    assert "market_cap" in panel
    assert "mask" in panel
    # close_m 与 close 形状一致；market_cap = TOT_SHARE × close_m，均为正或 NaN
    assert panel["close_m"].shape == panel["close"].shape
    assert panel["market_cap"].shape == panel["close"].shape
    assert panel["mask"].shape == panel["close"].shape
    mc_pos = panel["market_cap"].dropna(how="all")
    assert (mc_pos > 0).all().all()


def test_build_panel_returns_masked(tmp_path: Path, mock_ds):
    """close 已应用 membership mask：成分 in_date(2023-01-03) 前的行情应为 NaN。"""
    cache = DataCache(mock_ds, cache_root=tmp_path)
    cfg = {"universe": {"index_code": "000300.SH", "adjust": "backward"}}

    from data.cache_helpers import build_panel
    panel, _ = build_panel(cfg, 20230101, 20230110, cache=cache)
    close = panel["close"]
    before = close.index[close.index < pd.Timestamp("2023-01-03")]
    after = close.index[close.index >= pd.Timestamp("2023-01-03")]
    if len(before) > 0:
        assert close.loc[before].isna().all().all()
    assert close.loc[after].notna().all().all()


def test_narrow_calendar_query_does_not_truncate(tmp_path: Path, mock_ds):
    """窄区间日历查询不得覆盖丢失已有的完整日历（回归 P0）。"""
    cache = DataCache(mock_ds, cache_root=tmp_path)
    cache.get_calendar(20230101, 20241231)
    p = cache.root / "calendar.parquet"
    full_len = len(pd.read_parquet(p))
    # 窄区间再查一次
    cache.get_calendar(20240101, 20240131)
    assert len(pd.read_parquet(p)) == full_len
