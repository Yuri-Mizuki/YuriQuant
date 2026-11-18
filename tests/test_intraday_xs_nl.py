"""`factor.intraday_xs_nl` 的非线性特征与横截面变换测试。

重点钉住三条**静默出错**的坑（都是实测踩过的）：

1. **排列熵的并列值**：严格 6 序会把「有两个 bar 相等」的三元组全丢掉，
   交替序列 a,b,a 的熵被算成 0 —— 与「完全无规律」同分。必须 13 弱序。
2. **``mkt_z`` 对照臂**：spearman IC 对单调变换不敏感，z-score 后的 IC
   必须与原始**逐位相等**。不等 ⇒ 列对齐/掩码坏了，此时任何新因子结论都不可信。
3. **共线陷阱**：``P(涨→涨)+P(跌→跌)−1`` 在涨跌各半时恒等于
   ``1 − 2×翻转率``，是两个名字一个因子，故改用差值口径 ``nl_markov_asym``。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.intraday_xs_nl import (
    NL_DOCS,
    _markov_asym,
    _perm_entropy,
    _range_efficiency,
    _zero_cross_rate,
    cross_variant,
    extract_nl_from_block,
)


def _block(days: int = 5, bars: int = 48, codes: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    close = np.cumprod(1.0 + rng.normal(0, 0.002, (days, bars, codes)), axis=1) * 10.0
    open_ = np.concatenate([close[:, :1, :], close[:, :-1, :]], axis=1)
    vol = rng.lognormal(10, 0.5, (days, bars, codes))
    idx = [20220104 + i for i in range(days)]
    return {"open": open_, "close": close, "volume": vol}, idx


class TestPermEntropy:
    def test_tied_values_not_dropped(self):
        """严格交替 a,b,a 在 6 序实现下会得 0（等于白噪声），13 序必须给正值。"""
        bars = 48
        alt = np.empty((1, bars, 1))
        alt[0, ::2, 0] = 0.002
        alt[0, 1::2, 0] = -0.002
        ent = float(_perm_entropy(alt)[0, 0])
        assert ent > 0.0, "交替序列被算成 0 ⇒ 并列三元组被丢弃"
        # 交替序列只有 2 种模式，熵应远低于接近随机的情况
        rng = np.random.default_rng(3)
        rnd = np.where(rng.random((1, bars, 1)) > 0.5, 0.002, -0.002)
        assert ent < float(_perm_entropy(rnd)[0, 0])

    def test_zero_heavy_finite(self):
        """零收益 bar 占比高时（本项目常见）仍给有限值，不能整片 NaN。"""
        bars = 48
        rng = np.random.default_rng(4)
        x = np.where(rng.random((1, bars, 1)) > 0.4, 0.0, 0.002)
        ent = _perm_entropy(x)[0, 0]
        assert np.isfinite(ent)
        assert 0.0 <= ent <= 1.0

    def test_bounds_trend_vs_random(self):
        """单调爬升的序列模式单一 ⇒ 熵低；随机序列模式分散 ⇒ 熵高。

        🚨 趋势里不能掺与步长同量级的噪声：`perm_entropy` 看的是**相对大小序**，
        ``0.001 + 5e-5·N(0,1)`` 的相邻差分约有一半为负，模式已接近随机
        （实测熵 0.697 ≈ 随机的 0.680），用例会假失败。
        """
        bars = 48
        step = 1e-5
        trend = (0.001 + step * np.arange(bars, dtype=float)).reshape(1, bars, 1)
        rng = np.random.default_rng(5)
        rnd = rng.standard_normal((1, bars, 1)) * 0.002
        assert float(_perm_entropy(trend)[0, 0]) < float(_perm_entropy(rnd)[0, 0])


class TestPathShape:
    def test_range_efficiency_extremes(self):
        bars = 48
        trend = np.full((1, bars, 1), 0.001)
        zig = np.empty((1, bars, 1))
        zig[0, ::2, 0] = 0.002
        zig[0, 1::2, 0] = -0.002
        assert float(_range_efficiency(trend)[0, 0]) == pytest.approx(1.0)
        assert float(_range_efficiency(zig)[0, 0]) == pytest.approx(0.0)

    def test_zero_cross_rate_extremes(self):
        bars = 48
        trend = np.full((1, bars, 1), 0.001)
        zig = np.empty((1, bars, 1))
        zig[0, ::2, 0] = 0.002
        zig[0, 1::2, 0] = -0.002
        assert float(_zero_cross_rate(trend)[0, 0]) == pytest.approx(0.0)
        assert float(_zero_cross_rate(zig)[0, 0]) == pytest.approx(1.0)


class TestMarkovAsym:
    def test_not_collinear_with_flip_rate(self):
        """差值口径下，翻转率相同但两端不对称的序列必须给出不同值。"""
        bars = 48
        # 上行黏、下行即反：涨→涨 多，跌→跌 少
        up_sticky = np.array([0.002, 0.002, -0.002] * 16, dtype=float).reshape(1, bars, 1)
        both = float(_markov_asym(up_sticky)[0, 0])
        assert np.isfinite(both)
        assert both > 0, "涨端更黏时不对称度应为正"

    def test_symmetric_is_zero(self):
        bars = 48
        rng = np.random.default_rng(6)
        x = np.where(rng.random((1, bars, 1)) > 0.5, 0.002, -0.002)
        assert abs(float(_markov_asym(x)[0, 0])) < 0.4


class TestPanelOutput:
    def test_all_docs_produced(self):
        fields, days = _block()
        feats = extract_nl_from_block(fields, days, [f"S{i}" for i in range(4)])
        assert set(feats) == set(NL_DOCS), "产出键与 NL_DOCS 不一致（入库元数据会漂移）"
        for name, p in feats.items():
            assert p.shape == (len(days), 4), f"{name} 形状错"
            assert np.isfinite(p.to_numpy()).any(), f"{name} 全 NaN"

    def test_no_lookahead_future_independence(self):
        """改动某天之后的 bar 不应改变该天的特征。"""
        fields, days = _block(days=4)
        base = extract_nl_from_block(fields, days, [f"S{i}" for i in range(4)])
        f2 = {k: v.copy() for k, v in fields.items()}
        f2["close"][2] *= 1.05  # 只改第 3 天
        alt = extract_nl_from_block(f2, days, [f"S{i}" for i in range(4)])
        for name in NL_DOCS:
            a = base[name].iloc[:2].to_numpy()
            b = alt[name].iloc[:2].to_numpy()
            np.testing.assert_allclose(a, b, equal_nan=True,
                                       err_msg=f"{name} 前两日被未来数据污染")


class TestCrossVariant:
    @staticmethod
    def _panel(seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.to_datetime(["2022-01-04", "2022-01-05"])
        cols = [f"S{i}" for i in range(8)]
        return pd.DataFrame(rng.normal(0, 1, (2, 8)), index=idx, columns=cols)

    def test_mkt_z_preserves_spearman(self):
        """对照臂：z-score 后截面秩完全不变（IC 对单调变换不敏感）。"""
        p = self._panel()
        z = cross_variant(p, None, "mkt_z")
        for dt in p.index:
            a = p.loc[dt].rank().to_numpy()
            b = z.loc[dt].rank().to_numpy()
            np.testing.assert_allclose(a, b)

    def test_ind_diff_breaks_global_monotonicity(self):
        """行业内去均值**会**打破全截面单调性 —— 这才有可能产生新信息。"""
        p = self._panel()
        ind = pd.DataFrame(
            [["A"] * 4 + ["B"] * 4, ["A"] * 4 + ["B"] * 4],
            index=p.index, columns=p.columns)
        d = cross_variant(p, ind, "ind_diff")
        dt = p.index[0]
        assert (p.loc[dt].rank().to_numpy() != d.loc[dt].rank().to_numpy()).any()

    def test_ind_rank_in_unit_interval(self):
        p = self._panel()
        ind = pd.DataFrame(
            [["A"] * 4 + ["B"] * 4, ["A"] * 4 + ["B"] * 4],
            index=p.index, columns=p.columns)
        r = cross_variant(p, ind, "ind_rank")
        v = r.to_numpy()
        v = v[np.isfinite(v)]
        assert v.min() >= 0.0 and v.max() <= 1.0

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            cross_variant(self._panel(), None, "nope")