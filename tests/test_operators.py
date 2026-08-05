"""算子空间单元测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.operators import (
    ELEMENT_OPS, TS_OPS, CS_OPS, all_operators, op_registry,
    abs_, sign, log_, div, add, ts_mean, ts_ref, ts_delta, ts_arg_max,
    ts_corr, ts_slope, ts_ema, cs_rank, cs_zscore, cs_normalize, cs_scale,
    cs_rank_normalize, cs_demean, DEFAULT_WINDOWS,
)


@pytest.fixture
def panel():
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    cols = ["000001.SZ", "000002.SZ", "600000.SH", "600519.SH"]
    return pd.DataFrame(np.random.randn(80, 4), idx, cols)


@pytest.fixture
def panel2(panel):
    np.random.seed(7)
    return pd.DataFrame(np.random.randn(80, 4), panel.index, panel.columns)


# ---- 形状保持 ----
def test_registry_complete():
    ops = all_operators()
    assert len(ops) == len(ELEMENT_OPS) + len(TS_OPS) + len(CS_OPS)
    reg = op_registry()
    assert "ts_mean" in reg and "cs_rank" in reg and "div" in reg
    # 名称唯一
    names = [o.name for o in ops]
    assert len(names) == len(set(names))


def test_element_ops_preserve_shape(panel, panel2):
    for op in (abs_, sign, log_):
        out = op(panel)
        assert out.shape == panel.shape
    for op in (add, div):
        out = op(panel, panel2)
        assert out.shape == panel.shape


def test_ts_ops_preserve_shape(panel, panel2):
    assert ts_mean(panel, 20).shape == panel.shape
    assert ts_ref(panel, 5).shape == panel.shape
    assert ts_corr(panel, panel2, 20).shape == panel.shape
    assert ts_slope(panel, 20).shape == panel.shape


def test_cs_ops_preserve_shape(panel):
    for op in (cs_rank, cs_zscore, cs_normalize, cs_scale, cs_demean, cs_rank_normalize):
        assert op(panel).shape == panel.shape


# ---- NaN 安全 ----
def test_div_by_zero_yields_nan(panel):
    out = div(panel, panel * 0)
    # 不应出现 inf，应为 NaN
    assert not np.isinf(out.replace([np.nan], 0)).any().any()
    assert out.isna().all().all()


def test_log_nonpositive_nan():
    df = pd.DataFrame([[ -1.0, 2.0, 0.0]], columns=["a", "b", "c"])
    out = log_(df)
    assert np.isnan(out.iloc[0, 0])
    assert np.isnan(out.iloc[0, 2])
    assert out.iloc[0, 1] == pytest.approx(np.log(2.0))


def test_ts_min_periods(panel):
    # 前 n-1 行应为 NaN（min_periods=window）
    out = ts_mean(panel, 20)
    assert out.iloc[:19].isna().all().all()
    assert out.iloc[-1].notna().all()


# ---- 时序语义 ----
def test_ts_ref_equals_shift(panel):
    pd.testing.assert_frame_equal(ts_ref(panel, 3), panel.shift(3))


def test_ts_delta(panel):
    pd.testing.assert_frame_equal(ts_delta(panel, 5), panel - panel.shift(5))


def test_ts_arg_max_range(panel):
    out = ts_arg_max(panel, 20).iloc[-1]
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_ts_corr_in_range(panel, panel2):
    out = ts_corr(panel, panel2, 20).iloc[-1]
    assert out.min() >= -1.0001 and out.max() <= 1.0001


# ---- 截面语义 ----
def test_cs_zscore_centered(panel):
    row = cs_zscore(panel).iloc[-1].dropna()
    assert row.mean() == pytest.approx(0.0, abs=1e-9)
    assert row.std() == pytest.approx(1.0, abs=1e-6)


def test_cs_normalize_unit_interval(panel):
    row = cs_normalize(panel).iloc[-1].dropna()
    assert row.min() == pytest.approx(0.0, abs=1e-9)
    assert row.max() == pytest.approx(1.0, abs=1e-9)


def test_cs_scale_unit_norm(panel):
    row = cs_scale(panel).iloc[-1].dropna()
    assert (row ** 2).sum() == pytest.approx(1.0, abs=1e-6)


def test_cs_rank_in_unit_interval(panel):
    row = cs_rank(panel).iloc[-1].dropna()
    assert row.min() > 0.0 and row.max() <= 1.0


# ---- 窗口默认值 ----
def test_default_windows_sane():
    assert DEFAULT_WINDOWS == (5, 10, 20, 60)
