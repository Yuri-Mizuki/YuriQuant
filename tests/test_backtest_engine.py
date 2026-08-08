"""
回测引擎测试
============

覆盖 VectorBacktest.run() 的 executable_mask 参数：不可执行标的的权重
应被强制置零并重新归一化，不传 mask 时行为与原有实现完全一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import VectorBacktest, _apply_executable_mask
from backtest.costs import ShortCostModel
from strategy.base import Strategy
from strategy.examples import TopKLongOnly, TopKLongShort


class EqualWeightTop2(Strategy):
    """测试用策略：等权做多因子值最大的 2 只。"""

    name = "equal_top2"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        top = factor_values.sort_values(ascending=False).index[:2]
        return pd.Series(0.5, index=top)


@pytest.fixture
def small_panels():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    codes = ["A", "B", "C"]
    factor_panel = pd.DataFrame(
        [[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1]],
        index=dates, columns=codes, dtype=float,
    )
    returns_panel = pd.DataFrame(0.01, index=dates, columns=codes)
    return dates, codes, factor_panel, returns_panel


def test_apply_executable_mask_zeroes_and_renormalizes():
    weights = np.array([0.5, 0.5, 0.0])
    executable = np.array([True, False, True])
    out = _apply_executable_mask(weights, executable)
    # B 不可执行 -> 权重归零；A 独占多头，重新归一到原多头之和 1.0
    assert out[1] == 0.0
    assert out[0] == pytest.approx(1.0)
    assert out[2] == 0.0


def test_apply_executable_mask_long_short_preserves_sign_groups():
    weights = np.array([0.5, 0.5, -0.5, -0.5])
    executable = np.array([True, False, True, True])
    out = _apply_executable_mask(weights, executable)
    # 多头一侧只剩 A，归一到原多头之和 1.0；空头两侧都可执行，权重不变
    assert out[0] == pytest.approx(1.0)
    assert out[1] == 0.0
    assert out[2] == pytest.approx(-0.5)
    assert out[3] == pytest.approx(-0.5)


def test_run_without_mask_matches_baseline(small_panels):
    dates, codes, factor_panel, returns_panel = small_panels
    bt = VectorBacktest(strategy=EqualWeightTop2(), rebalance_freq="D", initial_capital=1.0)
    result = bt.run(factor_panel, returns_panel)
    # A、B 权重各 0.5，C 始终 0
    assert (result.weights_history["C"] == 0).all()
    assert result.weights_history.loc[dates[0], "A"] == pytest.approx(0.5)


def test_run_with_mask_excludes_unexecutable_stock(small_panels):
    dates, codes, factor_panel, returns_panel = small_panels
    # A 在第 3 天开始不可执行（例如涨停封板）
    mask = pd.DataFrame(True, index=dates, columns=codes)
    mask.loc[dates[2]:, "A"] = False

    bt = VectorBacktest(strategy=EqualWeightTop2(), rebalance_freq="D", initial_capital=1.0)
    result = bt.run(factor_panel, returns_panel, executable_mask=mask)

    # 第 3 天起 A 权重应为 0，B 顶替占满原多头权重
    assert result.weights_history.loc[dates[2], "A"] == 0.0
    assert result.weights_history.loc[dates[2], "B"] == pytest.approx(1.0)
    # 第 1、2 天（mask 全 True）行为与不加 mask 时一致
    assert result.weights_history.loc[dates[0], "A"] == pytest.approx(0.5)
    assert result.weights_history.loc[dates[0], "B"] == pytest.approx(0.5)


def test_transaction_costs_double_sided_and_zero_turnover():
    """交易成本应按双边（佣金/滑点）+ 单边卖出（印花税）计费，零换手零成本。

    回归 P0：早期实现佣金/滑点按单边、印花税按四分之一边（少算一倍），
    且零换手时仍按最低佣金扣费。
    """
    from backtest.costs import TransactionCosts

    c = TransactionCosts(commission_rate=0.0001, commission_min=5.0,
                         stamp_duty=0.001, slippage_bp=5.0)
    # 零换手 → 零成本
    assert c.calc(np.zeros(3), np.zeros(3), 0.0, 1_000_000.0) == 0.0

    # turnover=0.1 单边, capital=1e6 → 买入额=卖出额=1e5, 双边=2e5
    cost = c.calc(np.zeros(3), np.array([0.1, 0.0, 0.0]), 0.1, 1_000_000.0)
    expected_commission = max(2e5 * 0.0001, 5.0)   # 20
    expected_stamp = 1e5 * 0.001                    # 100
    expected_slip = 2e5 * 5.0 / 10000               # 10
    assert cost == pytest.approx(expected_commission + expected_stamp + expected_slip)


def test_daily_returns_net_of_cost():
    """daily_returns 应为净收益（扣成本），与 equity_curve 一致。

    回归 P0：早期实现 daily_returns 只存毛收益、成本仅从 capital 扣，
    导致 Sharpe / 年化 / 最大回撤系统性虚高。
    注：引擎对齐修复（2026-07-30）后，调仓日先建仓再赚当日收益，故首日
    既有建仓成本又有收益，净收益 = 毛收益(0.01) - 成本 < 0.01。
    """
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    codes = ["A", "B"]
    # 因子恒定 → 每个调仓日权重不变，仅首日建仓有换手有成本
    factor_panel = pd.DataFrame([[3, 2]] * 4, index=dates, columns=codes, dtype=float)
    returns_panel = pd.DataFrame(0.01, index=dates, columns=codes)

    bt = VectorBacktest(strategy=EqualWeightTop2(), rebalance_freq="D", initial_capital=1_000_000.0)
    res = bt.run(factor_panel, returns_panel)

    # 首日建仓产生成本 → 首日净收益低于毛收益 0.01（成本拖累）
    assert res.daily_returns.iloc[0] < 0.01
    # 之后权重不变 → 换手 0 → 无成本，净收益等于毛收益 0.01
    assert res.daily_returns.iloc[1] == pytest.approx(0.01)
    assert res.daily_returns.iloc[2] == pytest.approx(0.01)
    # equity_curve 与 daily_returns 自洽：(1+r).cumprod() ≈ equity
    recon = (1 + res.daily_returns).cumprod()
    assert np.allclose(recon.values, res.equity_curve.values, rtol=1e-6, atol=1e-9)


# ===========================================================================
# 空头腿成本（借券费 / 保证金占用）
# ===========================================================================
def _ls1_panels():
    """A 因子最高、B 最低 → TopKLongShort(k=1) 多 A 空 B，多空各 1 倍权重。"""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    codes = ["A", "B"]
    factor_panel = pd.DataFrame([[3.0, 1.0]] * 4, index=dates, columns=codes)
    returns_panel = pd.DataFrame(0.01, index=dates, columns=codes)
    return dates, codes, factor_panel, returns_panel


def test_short_borrow_fee_charged_daily_and_net_of_cost():
    """多空策略空头腿按日计提借券费；净收益自洽；首日净收益低于毛收益（成本拖累）。

    回归：旧引擎只计交易成本，空头持仓的借券费被完全忽略 → 多空收益虚高。
    """
    dates, codes, fp, rp = _ls1_panels()
    bt = VectorBacktest(strategy=TopKLongShort(k=1), rebalance_freq="D",
                        initial_capital=1_000_000.0,
                        short_costs=ShortCostModel(borrow_rate=0.08, margin_ratio=1.0))
    res = bt.run(fp, rp)

    # 每日借券费非零（空头 1 倍名义持续持有）
    assert (res.borrow_fee_series > 0).all()
    # 毛收益 = 1.0×0.01 + (-1.0)×0.01 = 0 → 首日净收益为负（交易成本 + 借券费拖累）
    assert res.daily_returns.iloc[0] < 0
    # 净收益与净值自洽（daily_returns 已是扣借券费后的净收益）
    recon = (1 + res.daily_returns).cumprod()
    assert np.allclose(recon.values, res.equity_curve.values, rtol=1e-6, atol=1e-9)
    # metrics 并入敞口/借券费指标（每日实际敞口口径：全年多空各 1 倍 → 敞口 1.0、占用 2.0）
    m = res.metrics()
    assert m["avg_short_exposure"] == pytest.approx(1.0)
    assert m["avg_margin_usage"] == pytest.approx(2.0)
    assert m["borrow_fee_total"] == pytest.approx(res.borrow_fee_series.sum())
    assert m["borrow_fee_drag_annual"] > 0


def test_borrow_fee_amount_daily_accrual():
    """借券费金额 = 空头名义(1.0) × 当日资金 × 年化费率/252；非调仓日无交易成本干扰。"""
    dates, codes, fp, rp = _ls1_panels()
    bt = VectorBacktest(strategy=TopKLongShort(k=1), rebalance_freq="M",  # 月调仓：仅首日换手
                        initial_capital=1_000_000.0,
                        short_costs=ShortCostModel(borrow_rate=0.08, margin_ratio=1.0))
    res = bt.run(fp, rp)
    fee1 = res.borrow_fee_series.iloc[1]
    capital_after_day0 = res.equity_curve.iloc[0] * 1_000_000.0
    assert fee1 == pytest.approx(capital_after_day0 * 0.08 / 252)


def test_no_borrow_fee_for_long_only():
    """纯多头无空头 → 借券费恒为 0，保证金占用 = 1 倍。"""
    dates, codes, fp, rp = _ls1_panels()
    bt = VectorBacktest(strategy=TopKLongOnly(k=1), rebalance_freq="D",
                        short_costs=ShortCostModel(borrow_rate=0.08, margin_ratio=1.0))
    res = bt.run(fp, rp)
    assert (res.borrow_fee_series == 0).all()
    assert res.margin_usage_series.iloc[0] == pytest.approx(1.0)


def test_margin_usage_reports_two_x_for_long_short():
    """多空各满仓 1 倍 → 保证金占用 = 1 + 1×保证金比例(1.0) = 2（隐含 2 倍资金需求）。"""
    dates, codes, fp, rp = _ls1_panels()
    bt = VectorBacktest(strategy=TopKLongShort(k=1), rebalance_freq="D",
                        short_costs=ShortCostModel(borrow_rate=0.08, margin_ratio=1.0))
    res = bt.run(fp, rp)
    assert res.margin_usage_series.iloc[0] == pytest.approx(2.0)
    assert res.metrics()["max_margin_usage"] == pytest.approx(2.0)


def test_deleverage_caps_margin_usage_at_one():
    """1 倍资金约束：原权重 [1, -1]（保证金需求 2）→ 缩放为 [0.5, -0.5]，占用 ≤ 1。"""
    dates, codes, fp, rp = _ls1_panels()
    bt = VectorBacktest(strategy=TopKLongShort(k=1), rebalance_freq="D",
                        short_costs=ShortCostModel(borrow_rate=0.08, margin_ratio=1.0),
                        deleverage=True)
    res = bt.run(fp, rp)
    assert res.weights_history.iloc[0]["A"] == pytest.approx(0.5)
    assert res.weights_history.iloc[0]["B"] == pytest.approx(-0.5)
    assert res.margin_usage_series.iloc[0] == pytest.approx(1.0)
    assert (res.margin_usage_series <= 1.0 + 1e-9).all()


def test_short_costs_disabled_matches_old_behavior():
    """borrow_rate=0（或显式关闭）→ 借券费恒 0，与旧口径一致。"""
    dates, codes, fp, rp = _ls1_panels()
    bt = VectorBacktest(strategy=TopKLongShort(k=1), rebalance_freq="D",
                        short_costs=ShortCostModel(borrow_rate=0.0))
    res = bt.run(fp, rp)
    assert (res.borrow_fee_series == 0).all()
    assert res.metrics().get("borrow_fee_total", 0.0) == 0.0
