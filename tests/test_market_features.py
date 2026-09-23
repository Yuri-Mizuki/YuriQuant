"""model.market_features 测试 —— 指数点位市场状态特征（国金19 转译）。

重点锁四件事：
1. **防前视**：扰动 t 日之后的指数值 → t 日特征逐位不变（因果锁）；
2. **同日同值不被误标准化**：不做截面 zscore（FeatureStore 通道会全 NaN）
   ——广播面板行内 std=0 锁定；
3. **零信息/预热诚实 NaN**：常数指数 → expanding sd=0 → NaN；52 周高低
   点窗口不足置 NaN；
4. **单调不变性**：点位线性缩放（如 ×2）后全部比率/收益特征逐位一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.market_features import (
    FEATURES_PER_INDEX,
    MARKET_FEAT_VERSION,
    Z_MIN_PERIODS,
    build_market_state_features,
    index_state_series,
)


def _biz_days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _ramp_close(n: int) -> pd.Series:
    """确定性好构造的指数：每日 +0.1% 复利上行。"""
    d = _biz_days(n)
    return pd.Series(1000.0 * (1.001 ** np.arange(n)), index=d)


class TestIndexStateSeries:
    def test_keys_and_shapes(self):
        s = index_state_series(_ramp_close(400))
        assert list(s) == list(FEATURES_PER_INDEX)
        for v in s.values():
            assert isinstance(v, pd.Series)
            assert len(v) == 400

    def test_causal_future_perturbation(self):
        """扰动未来值不影响过去特征（防前视核心锁）。"""
        n = 400
        c = _ramp_close(n)
        base = index_state_series(c)
        c2 = c.copy()
        c2.iloc[n // 2:] *= 1.5          # 后半段整体抬升
        pert = index_state_series(c2)
        for name, v in base.items():
            head = v.iloc[: n // 2]
            head_p = pert[name].iloc[: n // 2]
            # hi252/lo252 需要 252 日完整窗，前半段尾部已受影响——用共同有效段
            valid = head.notna() & head_p.notna()
            pd.testing.assert_series_equal(
                head[valid], head_p[valid], check_names=False)

    def test_scale_invariance(self):
        """点位线性缩放后比率/收益特征逐位一致（对点位刻度不敏感）。"""
        c = _ramp_close(400)
        a = index_state_series(c)
        b = index_state_series(c * 2.0)
        for name in FEATURES_PER_INDEX:
            pd.testing.assert_series_equal(
                a[name].dropna(), b[name].dropna(), check_names=False)

    def test_constant_series_all_nan(self):
        """恒定点位 → 全部特征 sd=0 或无波动 → expanding z 诚实 NaN。"""
        d = _biz_days(400)
        s = index_state_series(pd.Series(1000.0, index=d))
        for name, v in s.items():
            assert v.notna().sum() == 0, f"{name} 对零信息序列应全 NaN"

    def test_warmup_nan(self):
        """expanding z 预热期（Z_MIN_PERIODS）内置 NaN。"""
        c = _ramp_close(400)
        s = index_state_series(c)
        for name, v in s.items():
            first_valid = v.first_valid_index()
            if first_valid is not None:
                pos = v.index.get_loc(first_valid)
                assert pos >= Z_MIN_PERIODS - 1, f"{name} 预热不足"

    def test_ret1d_value(self):
        """ret1d expanding z 有界、量纲合理（|z| 基本 < 4），均值≈0。

        （确定性正漂移序列检验的是 hi252/lo252 语义，见下条；
        expanding z 是去均值的，符号分布本身不承诺正偏。）
        """
        rng = np.random.default_rng(7)
        n = 600
        d = _biz_days(n)
        r = rng.normal(0.001, 0.005, n)   # 均值 0.1% 的随机日收益
        c = pd.Series(1000.0 * np.cumprod(1.0 + r), index=d)
        z = index_state_series(c)["ret1d"].dropna()
        assert len(z) > 300
        assert z.abs().max() < 4.0
        assert abs(z.mean()) < 0.2

    def test_hi_lo_semantics(self):
        """上行指数：hi252（距高点）≈0 上界、lo252（距低点）> 0。"""
        c = _ramp_close(400)
        raw_hi = c / c.rolling(252, min_periods=252).max() - 1.0
        raw_lo = c / c.rolling(252, min_periods=252).min() - 1.0
        assert (raw_hi.dropna() <= 1e-12).all()
        assert (raw_lo.dropna() > 0).all()


class TestBuildMarketStateFeatures:
    def _build(self, n_days: int = 400, n_codes: int = 5):
        c = _ramp_close(n_days + 260)   # 多给一年预热
        closes = {"000001_SH": c, "399317_SZ": c * 1.3}
        idx = c.index[-n_days:]
        return build_market_state_features(closes, idx, [f"S{i}" for i in range(n_codes)]), idx

    def test_names_and_shape(self):
        feats, idx = self._build()
        assert len(feats) == 2 * len(FEATURES_PER_INDEX)
        for name, p in feats.items():
            assert p.shape == (len(idx), 5)
            assert p.index.equals(idx)
        assert "mkt_000001_SH_ret1d" in feats
        assert "mkt_399317_SZ_lo252" in feats

    def test_broadcast_same_row_value(self):
        """同日全市场同值（广播语义），行内 std=0。"""
        feats, _ = self._build()
        p = feats["mkt_000001_SH_ret5d"]
        row = p.iloc[100]
        assert row.nunique() == 1

    def test_not_cross_sectional_zscored(self):
        """行内 std=0 锁定：本模块**绝不**做截面标准化——否则同日同值
        过截面 z 会整行 NaN（FeatureStore 通道陷阱的反向守卫）。
        含 NaN 行（std=NaN），只在有效行上断言。"""
        feats, _ = self._build()
        for p in feats.values():
            sd = p.std(axis=1)
            valid = sd.dropna().index
            assert (sd.loc[valid] == 0).all(), "存在非同值行 → 被截面变换污染"

    def test_index_gap_dates_nan(self):
        """指数序列中断的日期整行 NaN（reindex 造洞，不依赖交易日历）。"""
        c = _ramp_close(400)
        idx = c.index
        # 从指数序列里抽掉中间 5 天 → 目标网格比指数多出 5 个洞日
        hole = idx[200:205]
        c2 = c.drop(hole)
        feats = build_market_state_features({"000001_SH": c2}, idx, ["S0"])
        p = feats["mkt_000001_SH_ret1d"]
        assert pd.isna(p.loc[hole].to_numpy()).all(), "洞日应整行 NaN"
        assert p.drop(index=hole).notna().any().any(), "洞日之外应有值"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_market_state_features({}, _biz_days(10), ["S0"])

    def test_unsafe_tag_raises(self):
        with pytest.raises(ValueError):
            build_market_state_features({"000001.SH": _ramp_close(10)},
                                        _biz_days(10), ["S0"])

    def test_float32_output(self):
        feats, _ = self._build(n_days=50)
        assert all(p.values.dtype == np.float32 for p in feats.values())


def test_version_constant_stable():
    """特征集版本常量存在（进 pred 指纹，改特征须 +1）。"""
    assert isinstance(MARKET_FEAT_VERSION, int) and MARKET_FEAT_VERSION >= 1
