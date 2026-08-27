"""
factor/technical.py 自研技术指标库测试（2026-08-17 抽取自 build_technical_factors）。

覆盖 calc_indicators（9 指标）与 calc_sar（Wilder 抛物线）。mock 数据，不依赖 SDK。
核心关注：无未来函数（指标只反映 t 及以前信息）+ 数值方向合理性 + 与 close 索引对齐。
"""
import numpy as np
import pandas as pd
import pytest

from factor.technical import calc_indicators, calc_sar


def _series(n=120, seed=5, trend=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    ret = rng.normal(trend, 0.01, n)
    close = pd.Series(100 * np.exp(np.cumsum(ret)), index=idx)
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.uniform(-0.002, 0.002, n))
    volume = pd.Series(rng.integers(1e5, 1e6, n).astype(float), index=idx)
    return close, high, low, open_, volume


def test_calc_indicators_returns_all_nine():
    c, h, l, o, v = _series()
    res = calc_indicators(c, h, l, o, v)
    expected = {"macd_hist", "rsi_12", "kdj_j", "trix_12", "obv_dev",
                "wad_dev", "asi_26", "cho", "sar_dev"}
    assert set(res) == expected
    for k, s in res.items():
        assert isinstance(s, pd.Series)
        assert s.index.equals(c.index)


def test_no_lookahead_on_jump_day():
    """构造前段平稳、第 50 日单日跳涨：所有指标在第 50 日应反映该跳变，
    但第 50 日之前的任何值不因未来跳变而改变（逐日只用到当日及以前）。"""
    n = 80
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = pd.Series(100.0, index=idx) + np.arange(n) * 0.1
    # 第 50 日跳涨
    close.iloc[50:] += 5.0
    high = close + 0.2
    low = close - 0.2
    open_ = close - 0.1
    volume = pd.Series(1e5, index=idx)
    res = calc_indicators(close, high, low, open_, volume)
    # 跳变发生在 50，第 49 日（跳变前一日）的指标应等同于"没有后续跳变"的对照
    def baseline():
        return calc_indicators(close.iloc[:50].copy(), high.iloc[:50].copy(),
                               low.iloc[:50].copy(), open_.iloc[:50].copy(),
                               volume.iloc[:50].copy())
    base = baseline()
    for k in res:
        # 前 49 日与"截断到 50 日再算"的前 49 日一致（无未来泄漏）
        full_49 = res[k].iloc[:49]
        base_49 = base[k].iloc[:49]
        pd.testing.assert_series_equal(full_49, base_49, check_names=False)


def test_rsi_bounded_upward_trend():
    c, h, l, o, v = _series(trend=0.02)  # 强上行
    res = calc_indicators(c, h, l, o, v)
    r = res["rsi_12"].dropna()
    assert len(r) > 0
    # RSI 有界 (0,100)
    assert ((r > 0) & (r <= 100)).all()
    # 持续上涨 → RSI 偏高（均值 > 60）
    assert r.mean() > 60


def test_calc_sar_aligned_and_valid():
    c, h, l, o, v = _series()
    sar = calc_sar(c, h, l)
    assert isinstance(sar, pd.Series)
    assert sar.index.equals(c.index)
    # 前 n 日 NaN，其后有值
    assert sar.iloc[:4].isna().all()
    assert sar.iloc[4:].notna().any()
    # 上涨趋势中 SAR 应低于价格（多头持仓期，支撑线在下方）
    up_c, up_h, up_l, _, _ = _series(trend=0.02)
    up_sar = calc_sar(up_c, up_h, up_l)
    valid = (up_c.notna() & up_sar.notna())
    tail = valid.tail(30)
    assert (up_sar[tail.index] < up_c[tail.index]).all()


def test_macd_hist_positive_in_uptrend():
    c, h, l, o, v = _series(trend=0.02)
    res = calc_indicators(c, h, l, o, v)
    hist = res["macd_hist"].dropna().tail(30)
    assert (hist > 0).all()
