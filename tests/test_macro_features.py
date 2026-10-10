"""model.macro_features 测试 —— 宏观日历市场状态特征（altf P1 落地）。

重点锁四件事（与 test_market_features 同纪律）：
1. **防前视**：扰动 t 日之后的序列值 → t 日特征逐位不变（expanding 回看）；
2. **同日同值不被误标准化**：不做截面 zscore（FeatureStore 通道会全 NaN）
   ——广播面板行内唯一值 = 1 锁定；
3. **零信息/预热诚实 NaN**：常数序列 → expanding sd=0 → NaN；预热期 NaN；
4. **市场级序列提取**：部分列缺值时仍能正确取到（mean 忽略 NaN），
   且面板名映射 ``alt_macro_x`` → ``mac_x``。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.macro_features import (
    MACRO_FEAT_VERSION,
    build_macro_state_features,
)
from model.market_features import Z_MIN_PERIODS


def _biz_days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=n)


def _broadcast_panel(dates, codes, vals) -> pd.DataFrame:
    """构造广播面板：同行同值（宏观日历面板的真实形态）。"""
    v = np.asarray(vals, dtype="float64")
    return pd.DataFrame(np.tile(v[:, None], (1, len(codes))),
                        index=dates, columns=list(codes))


def _wave(n: int) -> np.ndarray:
    """确定性、有波动的市场级序列（避免常数导致 sd=0 全 NaN）。"""
    return np.sin(np.arange(n) / 7.0) + np.arange(n) * 0.01


class TestBuildMacroStateFeatures:
    def test_keys_names_and_shape(self):
        d, codes = _biz_days(400), ["c0", "c1", "c2"]
        panel = _broadcast_panel(d, codes, _wave(400))
        out = build_macro_state_features({"alt_macro_surprise_str_5d": panel},
                                         d, codes)
        assert list(out) == ["mac_surprise_str_5d"]      # 前缀 alt_macro_ → mac_
        assert out["mac_surprise_str_5d"].shape == (400, 3)

    def test_causal_future_perturbation(self):
        """扰动未来值不影响过去特征（防前视核心锁）。"""
        n = 400
        d, codes = _biz_days(n), ["c0", "c1"]
        vals = _wave(n)
        base = build_macro_state_features(
            {"alt_macro_x": _broadcast_panel(d, codes, vals)}, d, codes)
        v2 = vals.copy()
        v2[n // 2:] += 5.0
        pert = build_macro_state_features(
            {"alt_macro_x": _broadcast_panel(d, codes, v2)}, d, codes)
        a = base["mac_x"].iloc[: n // 2, 0]
        b = pert["mac_x"].iloc[: n // 2, 0]
        valid = a.notna() & b.notna()
        assert valid.any()
        pd.testing.assert_series_equal(a[valid], b[valid], check_names=False)

    def test_constant_series_all_nan(self):
        """恒定序列 → expanding sd=0 → 诚实 NaN（不写防除零常数）。"""
        d, codes = _biz_days(400), ["c0", "c1"]
        panel = _broadcast_panel(d, codes, np.full(400, 3.0))
        out = build_macro_state_features({"alt_macro_c": panel}, d, codes)
        assert out["mac_c"].notna().sum().sum() == 0

    def test_warmup_nan(self):
        """expanding z 预热期（Z_MIN_PERIODS）内置 NaN。"""
        d, codes = _biz_days(400), ["c0"]
        panel = _broadcast_panel(d, codes, _wave(400))
        s = build_macro_state_features({"alt_macro_w": panel}, d, codes)["mac_w"].iloc[:, 0]
        assert s.iloc[: Z_MIN_PERIODS - 1].isna().all()
        assert s.iloc[Z_MIN_PERIODS:].notna().any()

    def test_broadcast_same_row_value(self):
        """同日全市场同值（广播语义）。"""
        d, codes = _biz_days(400), ["c0", "c1", "c2", "c3"]
        panel = _broadcast_panel(d, codes, _wave(400))
        out = build_macro_state_features({"alt_macro_b": panel}, d, codes)["mac_b"]
        rows = out.dropna(how="any")
        assert (rows.nunique(axis=1) == 1).all()

    def test_not_cross_sectional_zscored(self):
        """不做截面 zscore：行内 std=0（FeatureStore 通道会把整行变 NaN）。"""
        d, codes = _biz_days(400), ["c0", "c1", "c2"]
        panel = _broadcast_panel(d, codes, _wave(400))
        out = build_macro_state_features({"alt_macro_z": panel}, d, codes)["mac_z"]
        rows = out.dropna(how="any")
        assert (rows.std(axis=1) == 0).all()

    def test_partial_nan_columns_still_extracted(self):
        """部分列缺值（覆盖差异）时市场序列仍正确（mean 忽略 NaN）。"""
        n = 400
        d, codes = _biz_days(n), ["c0", "c1"]
        panel = _broadcast_panel(d, codes, _wave(n))
        panel.loc[d[200:250], "c1"] = np.nan
        out = build_macro_state_features({"alt_macro_p": panel}, d, codes)
        mid = out["mac_p"].iloc[200:250, 0]
        assert mid.notna().all()          # c0 列完好 → mean 仍取到值

    def test_all_nan_panel_skipped(self):
        d, codes = _biz_days(400), ["c0", "c1"]
        panel = _broadcast_panel(d, codes, np.full(400, np.nan))
        out = build_macro_state_features({"alt_macro_n": panel}, d, codes)
        assert out == {}

    def test_float32_output(self):
        d, codes = _biz_days(400), ["c0", "c1"]
        panel = _broadcast_panel(d, codes, _wave(400))
        out = build_macro_state_features({"alt_macro_f": panel}, d, codes)
        assert out["mac_f"].dtypes.iloc[0] == np.float32


def test_empty_panels_raises():
    with pytest.raises(ValueError, match="为空"):
        build_macro_state_features({}, _biz_days(10), ["c0"])


def test_version_constant_stable():
    """特征集版本是显式常量（改定义须手动 +1，防静默混口径）。"""
    assert isinstance(MACRO_FEAT_VERSION, int) and MACRO_FEAT_VERSION >= 1
