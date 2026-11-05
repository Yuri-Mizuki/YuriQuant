"""基准序列模块测试：等权/买入持有/分段等权基准 + 对照指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.benchmarks import (
    benchmark_returns,
    buy_hold_returns,
    compare_to_benchmark,
    equal_weight_returns,
    monthly_first_trading_days,
    segmented_equal_weight_returns,
    with_benchmark_metrics,
)


@pytest.fixture
def price_panel():
    rng = np.random.default_rng(21)
    idx = pd.date_range("2023-01-01", periods=120, freq="B")
    # 确定性的价格：A 强趋势、B 弱趋势、C/D 横盘
    base = pd.DataFrame({
        "A": 10 * (1.0008 ** np.arange(120)),
        "B": 10 * (1.0002 ** np.arange(120)),
        "C": 10 + np.sin(np.arange(120) / 10),
        "D": 10 + np.cos(np.arange(120) / 10),
    }, index=idx)
    return base + rng.normal(0, 0.01, (120, 4))


def test_equal_weight_returns_shape(price_panel):
    r = equal_weight_returns(price_panel)
    assert len(r) == len(price_panel)
    assert r.iloc[0] == pytest.approx(0.0) or np.isnan(r.iloc[0])  # 首日无收益


def test_buy_hold_vs_equal_weight(price_panel):
    """强趋势股权重下，买入持有（权重漂移）年化应 ≥ 等权（再平衡）。"""
    ew = equal_weight_returns(price_panel).dropna()
    bh = buy_hold_returns(price_panel).dropna()
    ew_ann = (1 + ew).prod() ** (252 / len(ew)) - 1
    bh_ann = (1 + bh).prod() ** (252 / len(bh)) - 1
    # A 股强趋势，买入持有应跑赢每日再平衡
    assert bh_ann >= ew_ann - 1e-9


def test_benchmark_returns_modes(price_panel):
    ew = benchmark_returns(price_panel, mode="equal_weight")
    bh = benchmark_returns(price_panel, mode="buy_hold")
    assert len(ew) == len(bh) == len(price_panel)


def test_compare_to_benchmark_basic(price_panel):
    strategy = price_panel["A"].pct_change()          # 强趋势股收益
    bench = equal_weight_returns(price_panel)
    out = compare_to_benchmark(strategy, bench)
    assert {"excess_annual", "information_ratio", "tracking_error",
            "correlation", "beta"}.issubset(out.keys())
    assert out["excess_annual"] > 0                   # A 强趋势 > 等权
    assert 0.0 <= out["correlation"] <= 1.0
    assert np.isfinite(out["tracking_error"])


def test_with_benchmark_metrics(price_panel):
    from backtest.metrics import calc_all_metrics
    strategy = price_panel["A"].pct_change().dropna()
    metrics = calc_all_metrics(strategy)
    bench = equal_weight_returns(price_panel).dropna()
    enriched = with_benchmark_metrics(metrics, strategy, bench)
    assert enriched["benchmark"]["excess_annual"] > 0
    # 原 dict 未被修改
    assert "benchmark" not in metrics


# =============================================================================
# segmented_equal_weight_returns —— 池化基准的唯一实现
#
# 下面几条分别锁定 2026-11-05 池子探针踩过的口径 bug。它们必须能在
# "手写版本"上失败，否则这些测试就没有防线价值。
#
# 约定：测试面板**首行是前值锚日**（窗口外），窗口从第二行起 —— 与真实调用
# 一致（框架要求传完整面板 + `index=` 裁输出）。
# =============================================================================


def _panel(prices: dict[str, list[float]], start: str = "2023-01-03"):
    """构造 date × code 价格面板；首行为前值锚。返回 (panel, window_index)。"""
    n = len(next(iter(prices.values())))
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(prices, index=idx), idx[1:]


def test_segmented_weights_drift_not_rebalanced():
    """段内**权重随价漂移**，不是每日再平衡到等权。

    A 100→150→200（+100%）、B 100→75→50（−50%）。
    买入持有：1.125 × 1.1111 = 1.25 ⇒ 累计 +25%。
    每日再平衡：1.125 × 1.0 = 1.125 ⇒ 累计 +12.5%。
    两者差一倍，口径错了这条必炸。
    """
    px, win = _panel({"A": [100.0, 100.0, 150.0, 200.0],
                      "B": [100.0, 100.0, 75.0, 50.0]})
    r = segmented_equal_weight_returns(px, index=win)
    assert (1 + r).prod() == pytest.approx(1.25, rel=1e-9)

    daily_rb = (1 + px.pct_change(fill_method=None).mean(axis=1).fillna(0)).prod()
    assert daily_rb == pytest.approx(1.125, rel=1e-9)   # 与漂移口径明确不同


def test_segmented_no_dilution_by_empty_columns():
    """空列（当年尚无价格）**不得**占权重。

    8 列里只有 2 列有价格、且都涨 10%。若按全部 8 列归一，基准会被稀释成
    +2.5% —— 这正是 2018 年全A 等权 −29.5% 被算成 −18.4% 的机制。
    """
    px, win = _panel({"A": [100.0, 100.0, 105.0, 110.0],
                      "B": [100.0, 100.0, 105.0, 110.0]})
    for k in range(6):
        px[f"EMPTY{k}"] = np.nan
    r = segmented_equal_weight_returns(px, index=win)
    assert (1 + r).prod() == pytest.approx(1.10, rel=1e-9)
    assert (1 + r).prod() != pytest.approx(1.025, rel=1e-3)


def test_segmented_window_index_keeps_first_month():
    """输出索引裁窗**不得**丢段：窗口首日恰为调仓日时，该段仍须有收益。

    场景即 2024-01 微盘暴跌：窗口从 1 月首个交易日开始，若从裁剪后的面板算
    首日收益 ⇒ 首日 NaN ⇒ 整月被 skip，基准虚高 24pp。
    """
    idx = pd.bdate_range("2023-12-01", periods=45)
    px = pd.DataFrame({"A": np.linspace(100, 121, len(idx)),
                       "B": np.linspace(100, 88, len(idx))}, index=idx)
    window = idx[22:]                                   # 从中间开始截

    full = segmented_equal_weight_returns(px, index=window)
    assert full.index.equals(window)
    assert full.iloc[0] != 0.0, "窗口首日收益为 0 ⇒ 首段被跳过"
    # 裁窗不得改变任何一天的收益值
    ref = segmented_equal_weight_returns(px).reindex(window)
    assert full.to_numpy() == pytest.approx(ref.to_numpy())


def test_segmented_raises_on_truncated_panel():
    """**中间段**首日大面积缺失 ⇒ 报错，而不是静默按少数股票算。"""
    idx = pd.bdate_range("2023-01-02", periods=70)
    px = pd.DataFrame({"A": np.linspace(100, 130, len(idx)),
                       "B": np.linspace(100, 90, len(idx))}, index=idx)
    broken = px.copy()
    # 让 2 月首个交易日（第 22 个交易日附近）有效列数为 0
    feb_start = monthly_first_trading_days(idx)[1]
    broken.loc[feb_start, :] = np.nan
    with pytest.raises(ValueError, match="有效列数"):
        segmented_equal_weight_returns(broken)


def test_segmented_warns_on_panel_without_prev_close():
    """面板首日无前值 ⇒ 丢该日 + warning（提示可能是裁剪面板）。"""
    idx = pd.bdate_range("2023-01-02", periods=5)
    px = pd.DataFrame({"A": np.linspace(100, 110, 5),
                       "B": np.linspace(100, 95, 5)}, index=idx)
    with pytest.warns(RuntimeWarning, match="无前值"):
        segmented_equal_weight_returns(px)


def test_segmented_pool_filter_via_investable():
    """池过滤走 ``investable=``：池外股票不参与，且不占权重。"""
    px, win = _panel({"IN1": [100.0, 100, 110, 121, 133, 146],
                      "IN2": [100.0, 100, 100, 100, 100, 100],
                      "OUT": [100.0, 100, 200, 400, 800, 1600]})
    mask = pd.DataFrame(True, index=px.index, columns=px.columns)
    mask["OUT"] = False

    r_pool = segmented_equal_weight_returns(px, index=win, investable=mask)
    r_all = segmented_equal_weight_returns(px, index=win)
    # 池内 = IN1(+46%) 与 IN2(0%) 等权 ⇒ 0.5×1.46 + 0.5×1.00 = 1.23
    assert (1 + r_pool).prod() == pytest.approx(1.23, rel=1e-9)
    assert (1 + r_all).prod() == pytest.approx((1.46 + 1.0 + 16.0) / 3, rel=1e-9)
    assert (1 + r_all).prod() > (1 + r_pool).prod()


def test_segmented_where_mask_breaks_first_day():
    """``close.where(mask)`` 的陷阱：窗口前一日价格被抹 ⇒ 首段算不出来。

    这条测试是为了**记住为什么改用 investable=**。若哪天有人把 probe 改回
    ``where(mask)``，这里会提醒他：首月会静默变成 0 收益。
    """
    px, win = _panel({"A": [100.0, 100, 110, 121, 133], "B": [100.0] * 5})
    # mask 只在窗口内有值（真实调用就是这样，因为 mask 来自信号面板的 index）
    mask = pd.DataFrame(True, index=px.index, columns=px.columns)
    mask.loc[px.index[0]] = False                       # 窗口前一日不在 mask 里

    with pytest.raises(ValueError, match="有效列数"):
        segmented_equal_weight_returns(px.where(mask), index=win)


def test_segmented_monthly_reset():
    """段边界 = 每月首个交易日；跨月权重重置回等权。"""
    idx = pd.bdate_range("2023-01-02", periods=46)
    starts = monthly_first_trading_days(idx)
    assert starts[0] == pd.Timestamp("2023-01-02")
    assert all(s.day <= 5 for s in starts)              # 月初
    px = pd.DataFrame({"A": 100 * 1.001 ** np.arange(len(idx)),
                       "B": 100 * 0.999 ** np.arange(len(idx))}, index=idx)
    r = segmented_equal_weight_returns(px)
    assert r.notna().all()
    assert len(starts) >= 2

    # 跨月权重确实被重置：A 每月抹平相对 B 的累积优势
    long_run = 100 * 1.05 ** np.arange(60)
    px2 = pd.DataFrame({"A": long_run, "B": np.full(60, 100.0)},
                       index=pd.bdate_range("2023-01-02", periods=60))
    seg = segmented_equal_weight_returns(px2).dropna()
    bh = buy_hold_returns(px2).dropna()
    assert (1 + seg).prod() < (1 + bh).prod(), "分段重置应弱于全程买入持有"
