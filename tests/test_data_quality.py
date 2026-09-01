"""数据质量检查 + 缓存数据指纹测试。"""
from __future__ import annotations

import pytest

from data.cache import DataCache
from data.quality import (
    check_adjust_factor_jumps, check_coverage, check_financial_nan,
    check_kline_missing,
)


def _cache(tmp_path, mock_ds) -> DataCache:
    return DataCache(mock_ds, cache_root=tmp_path / "parquet")


# ---------------------------------------------------------------------------
# 缓存数据指纹（P1-1）
# ---------------------------------------------------------------------------
def test_fingerprint_stable_and_sensitive(tmp_path, mock_ds):
    cache = _cache(tmp_path, mock_ds)
    cache.get_calendar(20230101, 20231231)
    codes = mock_ds.get_index_constituent("000300.SH")["con_code"].tolist()[:10]
    cache.get_daily_kline(codes, 20230101, 20231231)
    fp1 = cache.get_fingerprint()
    fp2 = cache.get_fingerprint()
    assert fp1 == fp2 and len(fp1) == 12
    # 拉取新数据（meta 变化）→ 指纹变化
    cache.get_daily_kline(codes, 20230101, 20240630)
    fp3 = cache.get_fingerprint()
    assert fp3 != fp1


def test_fingerprint_meta_only(tmp_path, mock_ds):
    """未拉任何数据时指纹仍稳定（基于 meta + 文件 stat）。"""
    cache = _cache(tmp_path, mock_ds)
    assert len(cache.get_fingerprint()) == 12


# ---------------------------------------------------------------------------
# 数据质量检查（P1-2）
# ---------------------------------------------------------------------------
def test_kline_missing_on_mock(tmp_path, mock_ds):
    cache = _cache(tmp_path, mock_ds)
    cal = cache.get_calendar(20230101, 20231231)
    codes = mock_ds.get_index_constituent("000300.SH")["con_code"].tolist()
    df = check_kline_missing(cache, codes, cal)
    assert not df.empty
    assert {"code", "n_expected", "n_actual", "missing_rate"}.issubset(df.columns)
    # mock 数据完整 → 缺失率全 0
    assert (df["missing_rate"] == 0.0).all()


def test_financial_nan_on_mock(tmp_path, mock_ds):
    cache = _cache(tmp_path, mock_ds)
    codes = mock_ds.get_index_constituent("000300.SH")["con_code"].tolist()
    df = check_financial_nan(cache, codes)
    assert not df.empty
    assert "max_nan_rate" in df.columns


def test_coverage_on_mock(tmp_path, mock_ds):
    cache = _cache(tmp_path, mock_ds)
    cal = cache.get_calendar(20230101, 20231231)
    codes = mock_ds.get_index_constituent("000300.SH")["con_code"].tolist()
    cov = check_coverage(cache, codes, cal)
    assert cov == pytest.approx(1.0)   # mock 成分全覆盖


def test_adjust_factor_jumps_on_mock(tmp_path, mock_ds):
    """mock 复权因子为平滑随机游走（无 >15% 跳变）→ 检查返回空。"""
    cache = _cache(tmp_path, mock_ds)
    cal = cache.get_calendar(20230101, 20231231)
    codes = mock_ds.get_index_constituent("000300.SH")["con_code"].tolist()
    df = check_adjust_factor_jumps(cache, codes, cal)
    assert df.empty
