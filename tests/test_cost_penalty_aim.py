"""成本感知优化（P0-2）与 α 衰减感知 aim（P0-3）的回归测试（2026-10-13）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimize.aim import (
    aim_target,
    aim_weights,
    optimize_weights_aim,
    persistence_from_horizons,
    persistence_scaled_alpha,
)
from optimize.cost_penalty import (
    calibrate_cost_penalty,
    cost_aware_weights,
    scan_cost_aware,
    typical_impact_c,
)


# ---------------------------------------------------------------------------
# cost_penalty
# ---------------------------------------------------------------------------
def test_typical_impact_c_formula_and_invalid():
    c = typical_impact_c(np.array([1e8, 1e8, 0.0, np.nan]),
                         np.array([0.03, 0.03, 0.03, 0.03]), capital=1e6, eta=0.4)
    assert c[0] == pytest.approx(0.4 * 0.03 * np.sqrt(1e6 / 1e8))
    assert c[1] == pytest.approx(c[0])
    assert np.isnan(c[2]) and np.isnan(c[3])


def test_calibrate_b_is_a_over_xstar():
    """核心推导：b = a / x*。"""
    cal = calibrate_cost_penalty(np.array([1e8, 5e7]), np.array([0.03, 0.04]),
                                 capital=1e6, x_star=0.02)
    assert cal["k1"] > 0 and cal["k2"] > 0
    assert cal["k2"] == pytest.approx(cal["k1"] / 0.02)


def test_calibrate_default_xstar_equal_weight():
    cal = calibrate_cost_penalty(np.array([1e8] * 50), np.array([0.03] * 50),
                                 capital=1e6)
    assert cal["x_star"] == pytest.approx(1 / 50)


def test_calibrate_all_invalid_returns_zero():
    cal = calibrate_cost_penalty(np.array([np.nan]), np.array([np.nan]), capital=1e6)
    assert cal["k1"] == 0.0 and cal["k2"] == 0.0


@pytest.fixture
def qp_panels():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    codes = [f"S{i}" for i in range(5)]
    factor = pd.DataFrame(rng.normal(size=(30, 5)), index=dates, columns=codes)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(30, 5)), index=dates, columns=codes)
    adv = pd.DataFrame(1e8, index=dates, columns=codes)
    vol = pd.DataFrame(0.03, index=dates, columns=codes)
    return factor, returns, adv, vol


def test_cost_aware_weights_runs(qp_panels):
    factor, returns, adv, vol = qp_panels
    w = cost_aware_weights(factor, returns, adv_panel=adv, vol_panel=vol,
                           capital=1e6, window=10, min_periods=5)
    assert w.shape == factor.shape
    assert np.isfinite(w.to_numpy()).all()
    assert (w.to_numpy() >= -1e-9).all()


def test_scan_cost_aware_returns_per_scale(qp_panels):
    factor, returns, adv, vol = qp_panels
    out = scan_cost_aware(factor, returns, adv_panel=adv, vol_panel=vol,
                          capital=1e6, scales=[0.0, 1.0],
                          window=10, min_periods=5)
    assert set(out) == {0.0, 1.0}
    assert out[0.0].shape == factor.shape


# ---------------------------------------------------------------------------
# aim
# ---------------------------------------------------------------------------
def _psd_sigma(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    S = A @ A.T / n * 1e-4
    return S + 1e-8 * np.eye(n)


def test_persistence_scaled_alpha_monotone():
    alpha = pd.Series([1.0, 1.0], index=["A", "B"])
    lo = persistence_scaled_alpha(alpha, 0.2)
    hi = persistence_scaled_alpha(alpha, 0.8)
    assert hi.iloc[0] > lo.iloc[0]


def test_persistence_scaled_alpha_per_name():
    alpha = pd.Series([1.0, 1.0], index=["A", "B"])
    phi = pd.Series([0.1, 0.9], index=["A", "B"])
    out = persistence_scaled_alpha(alpha, phi)
    assert out["B"] > out["A"]


def test_aim_target_long_only_sum_and_cap():
    n = 6
    alpha = pd.Series(np.linspace(1, 3, n), index=[f"S{i}" for i in range(n)])
    Sigma = _psd_sigma(n)
    w = aim_target(alpha, Sigma, gamma=1.0, max_weight=0.4, budget=1.0)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)
    assert (w <= 0.4 + 1e-9).all()


def test_aim_weights_partial_adjustment_between_prev_and_target():
    n = 6
    codes = [f"S{i}" for i in range(n)]
    alpha = pd.Series(np.linspace(1, 3, n), index=codes)
    Sigma = _psd_sigma(n)
    prev = pd.Series(1.0 / n, index=codes)
    w = aim_weights(alpha, Sigma, phi=0.5, prev_weights=prev)
    tgt = aim_target(persistence_scaled_alpha(alpha, 0.5), Sigma)
    # 偏调整结果应落在 prev 与 target 之间（同向且不超过端点）
    for c in codes:
        lo, hi = sorted([prev[c], tgt[c]])
        assert lo - 1e-9 <= w[c] <= hi + 1e-9


def test_aim_higher_cost_lower_turnover():
    n = 8
    codes = [f"S{i}" for i in range(n)]
    rng = np.random.default_rng(3)
    alpha = pd.Series(rng.normal(size=n), index=codes)
    Sigma = _psd_sigma(n)
    prev = pd.Series(rng.dirichlet(np.ones(n)), index=codes)
    w_lo = aim_weights(alpha, Sigma, phi=0.5, cost_kappa=0.0, prev_weights=prev)
    w_hi = aim_weights(alpha, Sigma, phi=0.5, cost_kappa=4.0, prev_weights=prev)
    to_lo = np.abs(w_lo - prev).sum()
    to_hi = np.abs(w_hi - prev).sum()
    assert to_hi < to_lo


def test_aim_empty_prev_jumps_to_target():
    n = 5
    codes = [f"S{i}" for i in range(n)]
    alpha = pd.Series(np.linspace(1, 2, n), index=codes)
    Sigma = _psd_sigma(n)
    w = aim_weights(alpha, Sigma, phi=0.5, prev_weights=None)
    tgt = aim_target(persistence_scaled_alpha(alpha, 0.5), Sigma)
    pd.testing.assert_series_equal(w, tgt)


def test_persistence_from_horizons_range_and_direction():
    codes = ["A", "B"]
    preds = {
        1: pd.Series([3.0, 1.0], index=codes),
        20: pd.Series([1.0, 3.0], index=codes),  # A 随 horizon 衰减，B 走强
    }
    phi = persistence_from_horizons(preds)
    assert phi.between(0.0, 0.95).all()
    assert phi["B"] > phi["A"]


def test_optimize_weights_aim_panel_runs():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    codes = [f"S{i}" for i in range(5)]
    factor = pd.DataFrame(rng.normal(size=(25, 5)), index=dates, columns=codes)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(25, 5)), index=dates, columns=codes)
    w = optimize_weights_aim(factor, returns, window=10, min_periods=5, phi=0.5)
    assert w.shape == factor.shape
    assert (w.to_numpy() >= -1e-9).all()
    assert np.isfinite(w.to_numpy()).all()


# ---------------------------------------------------------------------------
# 协方差暖机口径（2026-10-13 排查结论，防「窗口起点=数据起点」静默空仓回归）
# ---------------------------------------------------------------------------
def _panel(start, periods, n_codes, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    cols = [f"S{i}" for i in range(n_codes)]
    return pd.DataFrame(rng.normal(size=(periods, n_codes)), index=idx, columns=cols), cols


def test_qp_panel_truncated_history_goes_to_cash(capsys):
    """ret 面板被截断到回测窗口 ⇒ 早期调仓日 Σ 不足 ⇒ 权重全 0（会**静默空仓**）。

    这是 2026-10-13 在 ``cost_penalty_probe`` 上实测到的缺陷形态（当时压低 QP 约
    2.7pp/年）；本测试锁死「必须报警」这一行为，防它再次静默。
    """
    from scripts.evaluation.cost_penalty_probe import qp_weight_panel

    n = 240
    ret, cols = _panel("2020-01-01", n, 6, seed=1)
    pred = ret * 0.0 + 1.0                      # 常数信号 ⇒ 排序稳定、全期有值
    ret.iloc[:150] = np.nan                     # 收益面板「缺少」前 150 天历史
    mask = pd.DataFrame(True, index=ret.index, columns=cols)

    tw = qp_weight_panel(pred, ret, mask, n_names=5, k1=0.0, k2=0.0,
                         max_weight=1.0, window=60, min_periods=30,
                         method="min_var")
    out = capsys.readouterr().out
    assert "协方差历史不足" in out
    # 首个调仓日（历史为 0 行）→ 空仓
    first = tw.index[0]
    assert abs(tw.loc[first].sum()) == 0.0


def test_qp_panel_uses_history_outside_window(capsys):
    """ret 面板**长于** pred 窗口 ⇒ 第一个调仓日就有 Σ ⇒ 有仓位、无告警。"""
    from scripts.evaluation.cost_penalty_probe import qp_weight_panel

    ret, cols = _panel("2019-01-01", 400, 6, seed=2)
    pred = pd.DataFrame(np.nan, index=ret.index[120:], columns=cols)
    pred.iloc[:, 0] = 2.0                       # 全部给 S0 最高分
    pred.iloc[:, 1:] = 1.0
    mask = pd.DataFrame(True, index=pred.index, columns=cols)

    tw = qp_weight_panel(pred, ret, mask, n_names=5, k1=0.0, k2=0.0,
                         max_weight=1.0, window=60, min_periods=30,
                         method="min_var")
    out = capsys.readouterr().out
    assert "协方差历史不足" not in out
    first = tw.index[0]
    assert tw.loc[first].sum() == pytest.approx(1.0)
    assert tw.loc[first, "S0"] > 0


def test_eqw_topk_panel_weights_and_size():
    from scripts.evaluation.cost_penalty_probe import eqw_topk_panel

    ret, cols = _panel("2020-01-01", 120, 10, seed=3)
    mask = pd.DataFrame(True, index=ret.index, columns=cols)
    tw = eqw_topk_panel(ret, mask, 4)
    row = tw.iloc[-1]
    assert (row > 0).sum() == 4
    assert row.sum() == pytest.approx(1.0)
    assert row.max() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Sharpe 口径（2026-10-13 晚核对：探针初版用算术口径、与生产几何口径不可并列）
# ---------------------------------------------------------------------------
def test_sharpe_pair_matches_production_convention():
    """``_sharpe_pair`` 的 ``sharpe`` 必须与生产同源（``backtest.metrics.sharpe_ratio``），
    且**恒 ≤ 算术口径** —— 初版误用算术口径使所有对比虚高 0.05~0.13。"""
    from backtest.metrics import sharpe_ratio
    from scripts.evaluation.caliber import sharpe_pair as _sharpe_pair

    rng = np.random.default_rng(7)
    dr = pd.Series(rng.normal(0.0008, 0.012, 900),
                   index=pd.bdate_range("2020-01-02", periods=900))
    out = _sharpe_pair(dr)
    assert out["sharpe"] == pytest.approx(sharpe_ratio(dr))
    assert out["sharpe_arith"] == pytest.approx(dr.mean() / dr.std() * np.sqrt(252.0))
    assert out["sharpe_arith"] > out["sharpe"]        # 几何 ≤ 算术（波动拖累）


def test_sharpe_pair_degenerate_is_nan():
    from scripts.evaluation.caliber import sharpe_pair as _sharpe_pair

    dr = pd.Series([0.0] * 50, index=pd.bdate_range("2021-01-04", periods=50))
    out = _sharpe_pair(dr)
    assert np.isnan(out["sharpe"]) and np.isnan(out["sharpe_arith"])
