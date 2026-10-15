"""逐股冲击成本（P0-1）与回测整手约束（P1）的回归测试（2026-10-13）。

覆盖：
- ImpactCostModel 平方根律、买卖非对称、无效 adv/vol 不计成本、上限截断；
- TransactionCosts 在挂了 impact 且给了逐股 adv/vol 时，滑点项改走冲击成本；
- VectorBacktest 透传 adv/vol 面板 → impact_cost_series；
- _apply_lot_rounding 整手取整、不足一手归零、空头方向。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.costs import ImpactCostModel, TransactionCosts, impact_from_config
from backtest.engine import VectorBacktest, _apply_lot_rounding
from strategy.base import Strategy


class EqualWeightTop2(Strategy):
    name = "equal_top2"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        top = factor_values.sort_values(ascending=False).index[:2]
        return pd.Series(0.5, index=top)


# ---------------------------------------------------------------------------
# ImpactCostModel
# ---------------------------------------------------------------------------
def test_impact_buy_costs_more_than_sell():
    """A 股买入冲击 = 卖出 × buy_multiplier。"""
    m = ImpactCostModel(eta=0.4, buy_multiplier=1.5)
    adv = np.array([1e8])
    vol = np.array([0.03])
    buy = m.calc_per_name(np.array([0.0]), np.array([0.1]), 1e6, adv, vol)[0]
    sell = m.calc_per_name(np.array([0.1]), np.array([0.0]), 1e6, adv, vol)[0]
    assert buy == pytest.approx(1.5 * sell)


def test_impact_sqrt_law_and_magnitude():
    """impact_frac = η·σ·√(Q/ADV)，量级合理（bp 级）。"""
    m = ImpactCostModel(eta=0.4, buy_multiplier=1.0)
    capital, w = 1e6, 0.1
    adv, vol = 1e8, 0.03
    cost = m.calc_per_name(np.array([0.0]), np.array([w]), capital,
                           np.array([adv]), np.array([vol]))[0]
    notional = w * capital
    part = notional / adv
    expected_frac = 0.4 * vol * np.sqrt(part)
    assert cost == pytest.approx(notional * expected_frac)
    # 约 3.8bp 量级
    assert 2e-4 < cost / notional < 6e-4


def test_impact_scales_with_participation():
    """参与率越高冲击越大（单调）。"""
    m = ImpactCostModel(eta=0.4, buy_multiplier=1.0)
    lo = m.calc_per_name(np.array([0.0]), np.array([0.1]), 1e6,
                         np.array([1e9]), np.array([0.03]))[0]
    hi = m.calc_per_name(np.array([0.0]), np.array([0.1]), 1e6,
                         np.array([1e7]), np.array([0.03]))[0]
    assert hi > lo


def test_impact_ignores_invalid_adv_vol():
    """adv/vol 缺失或非正 → 该股不计冲击（不制造假成本）。"""
    m = ImpactCostModel()
    out = m.calc_per_name(np.array([0.0, 0.0, 0.0]),
                          np.array([0.1, 0.1, 0.1]), 1e6,
                          np.array([np.nan, 0.0, 1e8]),
                          np.array([0.03, 0.03, np.nan]))
    assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0


def test_impact_capped_at_max():
    """极端参与率被 max_impact 截断。"""
    m = ImpactCostModel(eta=0.4, buy_multiplier=1.0, max_impact=0.10)
    cost = m.calc_per_name(np.array([0.0]), np.array([1.0]), 1e6,
                           np.array([1e6]), np.array([0.5]))[0]
    assert cost == pytest.approx(1e6 * 0.10)


def test_impact_dimension_mismatch_raises():
    m = ImpactCostModel()
    with pytest.raises(ValueError):
        m.calc_per_name(np.array([0.0, 0.0]), np.array([0.1]), 1e6,
                        np.array([1e8]), np.array([0.03]))


# ---------------------------------------------------------------------------
# TransactionCosts 集成
# ---------------------------------------------------------------------------
def test_costs_impact_replaces_slippage_term():
    """挂 impact + 给 adv/vol → 滑点项 = 冲击成本；佣金/印花不变。"""
    adv, vol = np.array([1e8, 1e8]), np.array([0.03, 0.03])
    old = np.array([0.5, 0.5])
    new = np.array([0.5, 0.5])  # 不动仓

    base = TransactionCosts(commission_rate=0.0003, stamp_duty=0.001, slippage_bp=10)
    impact = ImpactCostModel()
    with_imp = TransactionCosts(commission_rate=0.0003, stamp_duty=0.001,
                                slippage_bp=10, impact=impact)

    # 换手 0.1（单边）：单边额=1e5，双边=2e5
    total_base = base.calc(old, new, 0.1, 1e6)
    total_imp = with_imp.calc(old, new, 0.1, 1e6, adv=adv, vol=vol)

    commission = max(2e5 * 0.0003, 5.0)
    stamp = 1e5 * 0.001
    expected_slip = impact.calc(old, new, 1e6, adv, vol)
    assert total_imp == pytest.approx(commission + stamp + expected_slip)
    assert total_base != pytest.approx(total_imp)


def test_costs_without_adv_falls_back_to_slippage():
    """挂了 impact 但没给 adv/vol → 退回固定滑点（零回归保证）。"""
    impact = ImpactCostModel()
    c = TransactionCosts(commission_rate=0.0003, stamp_duty=0.001,
                         slippage_bp=10, impact=impact)
    flat = TransactionCosts(commission_rate=0.0003, stamp_duty=0.001, slippage_bp=10)
    old, new = np.array([0.5, 0.3]), np.array([0.2, 0.6])
    assert c.calc(old, new, 0.3, 1e6) == pytest.approx(flat.calc(old, new, 0.3, 1e6))


def test_impact_from_config_default_off():
    """impact_model=none → 不启用；=sqrt → 构建且参数对齐。"""
    assert impact_from_config({"impact_model": "none"}) is None
    m = impact_from_config({"impact_model": "sqrt", "impact_eta": 0.35,
                            "impact_buy_multiplier": 1.4, "impact_max": 0.08})
    assert m is not None
    assert m.eta == pytest.approx(0.35)
    assert m.buy_multiplier == pytest.approx(1.4)
    assert m.max_impact == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# VectorBacktest 透传
# ---------------------------------------------------------------------------
@pytest.fixture
def panels():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    codes = ["A", "B", "C"]
    factor = pd.DataFrame([[3, 2, 1]] * 6, index=dates, columns=codes, dtype=float)
    returns = pd.DataFrame(0.0, index=dates, columns=codes)
    adv = pd.DataFrame(1e8, index=dates, columns=codes)
    vol = pd.DataFrame(0.03, index=dates, columns=codes)
    return dates, codes, factor, returns, adv, vol


def test_engine_impact_series_populated(panels):
    dates, codes, factor, returns, adv, vol = panels
    bt = VectorBacktest(strategy=EqualWeightTop2(), rebalance_freq="D",
                        initial_capital=1e6,
                        costs=TransactionCosts(commission_rate=0.0003,
                                               stamp_duty=0.001, slippage_bp=10,
                                               impact=ImpactCostModel()))
    res = bt.run(factor, returns, adv_panel=adv, vol_panel=vol)
    assert res.impact_cost_series.sum() > 0
    assert res.config["impact_model"] == "sqrt"
    # 冲击**必须进入实际扣费**（总成本 ≥ 冲击分量）；回归 2026-10-13 的
    # 「只报告、不扣费」引擎接线 bug。
    assert res.cost_series.sum() >= res.impact_cost_series.sum() - 1e-9


def test_engine_total_cost_excludes_slippage_when_impact_on(panels):
    """启用冲击后，总成本应 = 佣金+印花+冲击，而非 佣金+印花+固定滑点。"""
    dates, codes, factor, returns, adv, vol = panels
    imp = ImpactCostModel()
    # 故意把固定滑点设到极大：若它仍生效，总成本会爆表。
    bt_imp = VectorBacktest(strategy=EqualWeightTop2(), rebalance_freq="D",
                            initial_capital=1e6,
                            costs=TransactionCosts(commission_rate=0.0003,
                                                   stamp_duty=0.001, slippage_bp=1e5,
                                                   impact=imp))
    res_imp = bt_imp.run(factor, returns, adv_panel=adv, vol_panel=vol)
    bt_flat = VectorBacktest(strategy=EqualWeightTop2(), rebalance_freq="D",
                             initial_capital=1e6,
                             costs=TransactionCosts(commission_rate=0.0003,
                                                    stamp_duty=0.001, slippage_bp=1e5))
    res_flat = bt_flat.run(factor, returns, adv_panel=adv, vol_panel=vol)
    # 冲击臂的总成本 ≈ 佣金+印花+冲击（≪ 假固定滑点），证明滑点项被替换而非叠加
    assert res_imp.cost_series.sum() < res_flat.cost_series.sum() / 10
    assert res_imp.cost_series.sum() >= res_imp.impact_cost_series.sum() - 1e-9


def test_engine_without_adv_has_zero_impact(panels):
    dates, codes, factor, returns, adv, vol = panels
    bt = VectorBacktest(strategy=EqualWeightTop2(), rebalance_freq="D",
                        initial_capital=1e6,
                        costs=TransactionCosts(commission_rate=0.0003,
                                               stamp_duty=0.001, slippage_bp=10,
                                               impact=ImpactCostModel()))
    res = bt.run(factor, returns)  # 不给 adv
    assert res.impact_cost_series.sum() == 0.0
    assert res.config["impact_model"] == "sqrt"  # 模型仍挂着，只是退回固定滑点


# ---------------------------------------------------------------------------
# 整手约束
# ---------------------------------------------------------------------------
def test_lot_rounding_basic():
    """100 元现金、1 元/股、lot=100：w=0.5 → 50 股 → 向零取整为 0。"""
    w = np.array([0.5, 0.5])
    prices = np.array([1.0, 1.0])
    out = _apply_lot_rounding(w, prices, capital=100.0, lot_size=100)
    assert (out == 0.0).all()


def test_lot_rounding_rounds_down():
    """1e6 元、10 元/股、lot=100：w=0.5 → 5万股（整手）恰好 0.5。"""
    w = np.array([0.5])
    prices = np.array([10.0])
    out = _apply_lot_rounding(w, prices, capital=1e6, lot_size=100)
    assert out[0] == pytest.approx(0.5)


def test_lot_rounding_floor_and_cash_left():
    """w=0.3333、100元/股、cap=1e6 → 3333 股取整为 3300 股 → 权重 0.33。"""
    w = np.array([0.3333])
    out = _apply_lot_rounding(w, np.array([100.0]), capital=1e6, lot_size=100)
    assert out[0] == pytest.approx(3300 * 100.0 / 1e6)


def test_lot_rounding_short_direction():
    """空头向零取整（不超卖）。"""
    w = np.array([-0.3333])
    out = _apply_lot_rounding(w, np.array([100.0]), capital=1e6, lot_size=100)
    assert out[0] == pytest.approx(-3300 * 100.0 / 1e6)


def test_lot_rounding_missing_price_zeroes():
    """价格缺失的持仓 → 归零（无法建仓）。"""
    w = np.array([0.5, 0.5])
    out = _apply_lot_rounding(w, np.array([np.nan, 10.0]), capital=1e6, lot_size=100)
    assert out[0] == 0.0 and out[1] > 0


def test_engine_lot_reduces_exposure(panels):
    """整手约束下组合欠配（Σ|w| 下降），净收益相应变化。"""
    dates, codes, factor, returns, adv, vol = panels
    price = pd.DataFrame({"A": 999.0, "B": 1.0, "C": 1.0}, index=dates)
    bt = VectorBacktest(strategy=EqualWeightTop2(), rebalance_freq="D",
                        initial_capital=100.0)  # 小资金 → 整手摩擦显著
    res_lot = bt.run(factor, returns, price_panel=price, lot_size=100)
    bt2 = VectorBacktest(strategy=EqualWeightTop2(), rebalance_freq="D",
                         initial_capital=100.0)
    res_no = bt2.run(factor, returns)
    # A 单价 999，100 元只能买 0 手 → A 被整手归零
    assert res_lot.weights_history["A"].abs().sum() == 0.0
    assert res_no.weights_history["A"].abs().sum() > 0.0
