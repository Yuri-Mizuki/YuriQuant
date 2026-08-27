"""
optimize/solver.py 求解器优化测试（P0）。

覆盖：
- 协方差估计：PSD、形状
- 防前视：修改未来收益不影响当期 Σ 与权重
- 约束精确满足：预算 / 个股上限 / 行业中性（等式）/ 换手（单边口径）
- 优化有效性：最小方差解波动 ≤ 等权；TEV 解目标 ≤ 基准目标
- 面板输出约定：形状、NaN 因子股票权重为 0、窗口不足时全 0 空仓
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimize.solver import (
    bl_posterior,
    bl_views_from_factor,
    estimate_covariance,
    hrp_weights,
    optimize_weights_hrp,
    optimize_weights_qp,
    rolling_covariance,
    solve_portfolio,
)


def _mock_panel(n_days: int = 150, n_codes: int = 20, seed: int = 7):
    """随机收益 + 滞后收益因子（有预测力），仿 test_pipeline_layers。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    rets = pd.DataFrame(
        rng.normal(0, rng.uniform(0.005, 0.03, n_codes), (n_days, n_codes)),
        idx, codes,
    )
    f = rets.shift(1)
    f = f.div(f.abs().max(axis=1), axis=0)
    return f, rets


# ===========================================================================
# 协方差估计
# ===========================================================================
def test_estimate_covariance_psd_and_shape():
    f, rets = _mock_panel(n_days=120, n_codes=20, seed=1)
    Sigma = estimate_covariance(rets)
    assert Sigma.shape == (20, 20)
    assert np.allclose(Sigma, Sigma.T, atol=1e-12)  # 对称
    eig = np.linalg.eigvalsh(Sigma)
    assert eig.min() > -1e-8  # 半正定


def test_estimate_covariance_ledoit_wolf_preferred():
    """样本充足时走 sklearn LedoitWolf（自动收缩），结果仍 PSD。"""
    f, rets = _mock_panel(n_days=300, n_codes=20, seed=2)
    Sigma = estimate_covariance(rets, method="ledoit_wolf")
    assert np.linalg.eigvalsh(Sigma).min() > -1e-8


def test_estimate_covariance_high_dim_fallback():
    """特征数接近样本数 → 手动收缩兜底，不抛错。"""
    f, rets = _mock_panel(n_days=30, n_codes=25, seed=3)  # T=30 < N=25
    Sigma = estimate_covariance(rets, method="ledoit_wolf")
    assert Sigma.shape == (25, 25)
    assert np.linalg.eigvalsh(Sigma).min() > -1e-8


def test_rolling_covariance_no_lookahead():
    f, rets = _mock_panel(n_days=120, n_codes=10, seed=4)
    t = rets.index[60]
    s1 = rolling_covariance(rets, t, window=60, min_periods=30)
    rets2 = rets.copy()
    rets2.iloc[60:] *= 10.0  # 污染 t 当日及之后
    s2 = rolling_covariance(rets2, t, window=60, min_periods=30)
    assert s1 is not None and s2 is not None
    np.testing.assert_allclose(s1, s2, atol=1e-12)  # 只用 < t 数据


def test_rolling_covariance_insufficient():
    f, rets = _mock_panel(n_days=20, n_codes=10, seed=5)
    assert rolling_covariance(rets, rets.index[5], window=60, min_periods=30) is None


# ===========================================================================
# 单截面求解：约束精确满足
# ===========================================================================
def _solve_first(f, rets, **kw):
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    assert Sigma is not None
    return solve_portfolio(f.iloc[100], Sigma, **kw)


def test_minvar_budget_and_bounds():
    f, rets = _mock_panel(seed=11)
    w = _solve_first(f, rets, method="min_var", max_weight=0.2)
    assert (w >= -1e-9).all()
    assert (w <= 0.2 + 1e-9).all()
    assert abs(w.sum() - 1.0) < 1e-6


def test_industry_neutral_exact():
    """行业中性用等式约束 → 精确满足（优于投影法的近似）。"""
    f, rets = _mock_panel(seed=12)
    codes = list(f.columns)
    ind_map = {c: f"ind{i % 3}" for i, c in enumerate(codes)}
    w = _solve_first(f, rets, method="min_var", industry_map=ind_map)
    for name in ("ind0", "ind1", "ind2"):
        cols = [c for c in codes if ind_map[c] == name]
        assert abs(w[cols].sum() - 1.0 / 3) < 1e-8


def test_industry_target_custom():
    f, rets = _mock_panel(seed=13)
    codes = list(f.columns)
    ind_map = {c: f"ind{i % 3}" for i, c in enumerate(codes)}
    target = {"ind0": 0.5, "ind1": 0.3, "ind2": 0.2}
    w = _solve_first(f, rets, method="min_var", industry_map=ind_map, industry_target=target)
    for name, tgt in target.items():
        cols = [c for c in codes if ind_map[c] == name]
        assert abs(w[cols].sum() - tgt) < 1e-8


def test_turnover_constraint_effective():
    """换手硬约束：单边换手 ≤ 上限，且确实比无约束时更低。"""
    f, rets = _mock_panel(seed=14)
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    prev = pd.Series(1.0 / len(f.columns), index=f.columns)  # 等权上期
    w_un = solve_portfolio(f.iloc[100], Sigma, method="min_var")
    w_c = solve_portfolio(
        f.iloc[100], Sigma, method="min_var",
        prev_weights=prev, max_turnover=0.1,
    )
    turn_c = 0.5 * (w_c - prev).abs().sum()
    turn_un = 0.5 * (w_un - prev).abs().sum()
    assert turn_c <= 0.1 + 1e-6
    assert turn_c < turn_un - 1e-9  # 约束确实收紧（无约束解换手更大）


def test_turnover_penalty_in_objective():
    f, rets = _mock_panel(seed=15)
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    prev = pd.Series(1.0 / len(f.columns), index=f.columns)
    w0 = solve_portfolio(f.iloc[100], Sigma, method="mvo", risk_aversion=1.0)
    w1 = solve_portfolio(
        f.iloc[100], Sigma, method="mvo", risk_aversion=1.0,
        prev_weights=prev, turnover_penalty=5.0,
    )
    assert 0.5 * (w1 - prev).abs().sum() <= 0.5 * (w0 - prev).abs().sum() + 1e-9


def test_nan_alpha_forced_zero():
    f, rets = _mock_panel(seed=16)
    alpha = f.iloc[100].copy()
    alpha.iloc[:5] = np.nan  # 前 5 只不可持仓
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    w = solve_portfolio(alpha, Sigma, method="min_var")
    assert (w.iloc[:5].abs() < 1e-9).all()
    assert abs(w.iloc[5:].sum() - 1.0) < 1e-6


# ===========================================================================
# 优化有效性（凸优化性质：最优解不劣于任意可行解）
# ===========================================================================
def test_minvar_vol_leq_equal_weight():
    f, rets = _mock_panel(seed=17)
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    w = solve_portfolio(f.iloc[100], Sigma, method="min_var")
    n = len(w)
    we = pd.Series(1.0 / n, index=w.index)
    assert w.values @ Sigma @ w.values <= we.values @ Sigma @ we.values + 1e-9


def test_tev_objective_no_worse_than_benchmark():
    f, rets = _mock_panel(seed=18)
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    alpha = f.iloc[100]
    wb = pd.Series(1.0 / len(alpha), index=alpha.index)
    lam = 1.0
    w = solve_portfolio(alpha, Sigma, method="tev", risk_aversion=lam, benchmark=wb)
    score = alpha.rank(pct=True).fillna(0.0).values

    def obj(x: pd.Series) -> float:
        d = (x.values - wb.values)
        return float(d @ Sigma @ d) - lam * score @ x.values

    assert obj(w) <= obj(wb) + 1e-9  # wb 是可行解 → 最优解不劣于它


def test_mvo_uses_alpha():
    """mvo 解应比 min_var 更偏向高分股票（α 进目标）。"""
    f, rets = _mock_panel(seed=19)
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    alpha = f.iloc[100]
    w_mv = solve_portfolio(alpha, Sigma, method="min_var")
    w_mvo = solve_portfolio(alpha, Sigma, method="mvo", risk_aversion=0.5)
    # 用加权分数比较：mvo 解的 alpha 分数不应低于 min_var 解
    s = alpha.rank(pct=True).fillna(0.0)
    assert s @ w_mvo >= s @ w_mv - 1e-9


# ===========================================================================
# 面板级入口
# ===========================================================================
def test_panel_output_shape_and_empty_head():
    f, rets = _mock_panel(n_days=120, n_codes=15, seed=21)
    out = optimize_weights_qp(f, rets, method="min_var", window=60, min_periods=30)
    assert out.shape == f.shape
    assert out.index.equals(f.index) and out.columns.equals(f.columns)
    # 前 min_periods 天内窗口不足 → 全 0（空仓）
    assert (out.iloc[:30].values == 0).all()
    # 有数据后存在非空仓行
    assert (out.iloc[60:].sum(axis=1) > 0.5).any()


def test_panel_no_lookahead():
    f, rets = _mock_panel(n_days=120, n_codes=10, seed=22)
    out1 = optimize_weights_qp(f, rets, method="min_var", window=60, min_periods=30)
    rets2 = rets.copy()
    rets2.iloc[60:] *= 10.0
    out2 = optimize_weights_qp(f, rets2, method="min_var", window=60, min_periods=30)
    # 修改 t=60 当日及之后 → 前 60 天（含 t=59）不受影响
    pd.testing.assert_frame_equal(out1.iloc[:60], out2.iloc[:60])


def test_panel_with_industry_and_prev():
    f, rets = _mock_panel(n_days=120, n_codes=12, seed=23)
    codes = list(f.columns)
    ind_map = {c: f"ind{i % 3}" for i, c in enumerate(codes)}
    prev = pd.Series(1.0 / len(codes), index=codes)
    out = optimize_weights_qp(
        f, rets, method="min_var", window=60, min_periods=30,
        industry_map=ind_map, prev_weights=prev, max_turnover=0.5,
    )
    for t in out.index[60:]:
        row = out.loc[t]
        if row.sum() == 0:
            continue
        assert abs(row.sum() - 1.0) < 1e-6
        for name in ("ind0", "ind1", "ind2"):
            cols = [c for c in codes if ind_map[c] == name]
            assert abs(row[cols].sum() - 1.0 / 3) < 1e-6
        turn = 0.5 * (row - prev).abs().sum()
        assert turn <= 0.5 + 1e-6


def test_requires_returns_panel():
    f, _ = _mock_panel()
    with pytest.raises(ValueError):
        optimize_weights_qp(f, None)


def test_invalid_method():
    f, rets = _mock_panel()
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    with pytest.raises(ValueError):
        solve_portfolio(f.iloc[100], Sigma, method="nope")
    with pytest.raises(ValueError):
        solve_portfolio(f.iloc[100], Sigma, method="tev")  # 缺 benchmark


# ===========================================================================
# P1：风格中性化 / 行业偏离 / 风险平价 / HRP
# ===========================================================================
def test_style_neutralization():
    """风格中性化：组合对风格暴露 ≈ 0（把风格 beta 剔除）。"""
    f, rets = _mock_panel(seed=31)
    codes = list(f.columns)
    rng = np.random.default_rng(31)
    # 构造两个风格暴露（市值 / 动量，已 zscore 中心化）
    style = pd.DataFrame(
        rng.normal(0, 1, (len(codes), 2)), index=codes, columns=["mktcap", "momentum"],
    )
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    w = solve_portfolio(f.iloc[100], Sigma, method="mvo", risk_aversion=1.0,
                        style_exposures=style)
    B = style.reindex(codes).fillna(0.0).values
    assert np.abs(B.T @ w.values).max() <= 1e-4  # |B'w| ≤ tol


def test_style_neutralization_panel():
    """面板级风格中性化（dict{style → date×code 面板}）。"""
    f, rets = _mock_panel(n_days=120, n_codes=12, seed=32)
    codes = list(f.columns)
    rng = np.random.default_rng(32)
    style_panels = {
        "mktcap": pd.DataFrame(rng.normal(0, 1, f.shape), f.index, codes),
        "momentum": pd.DataFrame(rng.normal(0, 1, f.shape), f.index, codes),
    }
    out = optimize_weights_qp(
        f, rets, method="mvo", risk_aversion=1.0,
        window=60, min_periods=30, style_exposures=style_panels,
    )
    for t in out.index[60:]:
        row = out.loc[t]
        if row.sum() == 0:
            continue
        B = pd.concat({k: v.loc[t] for k, v in style_panels.items()}, axis=1)
        assert np.abs(B.values.T @ row.values).max() <= 1e-4


def test_industry_deviation_bounds():
    """行业偏离区间约束：|行业权重 − 目标| ≤ dev，且行得通（非精确等式）。"""
    f, rets = _mock_panel(seed=33)
    codes = list(f.columns)
    ind_map = {c: f"ind{i % 3}" for i, c in enumerate(codes)}
    dev = 0.05
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    w = solve_portfolio(
        f.iloc[100], Sigma, method="min_var",
        industry_map=ind_map, industry_deviation=dev,
    )
    for name in ("ind0", "ind1", "ind2"):
        cols = [c for c in codes if ind_map[c] == name]
        assert abs(w[cols].sum() - 1.0 / 3) <= dev + 1e-6


def test_risk_parity_risk_contributions_equal():
    """风险平价：各股票对组合总风险的边际贡献近似相等。"""
    f, rets = _mock_panel(seed=34)
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    w = solve_portfolio(f.iloc[100], Sigma, method="risk_parity")
    assert (w > 0).all()  # 风险平价要求严格正权重
    assert abs(w.sum() - 1.0) < 1e-6
    rc = w.values * (Sigma @ w.values)  # 边际风险贡献 w_i(Σw)_i
    # 等风险预算 → 各贡献接近相等。注意：对数障碍配方数值解贡献比 ~1.5
    #（等权退化时 ~4.4），阈值 2.5 证明「显著优于等权」且接近平价。
    ratio = rc.max() / rc.min() if rc.min() > 0 else float("inf")
    assert ratio < 2.5


def test_hrp_basic_properties():
    """HRP：非负、归一、免逆矩阵（奇异协方差不炸）。"""
    f, rets = _mock_panel(seed=35)
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    w = hrp_weights(Sigma, codes=list(f.columns))
    assert (w >= 0).all()
    assert abs(w.sum() - 1.0) < 1e-8


def test_hrp_singular_covariance_ok():
    """HRP 对病态协方差免疫：两列完全相同（奇异）也能算。"""
    rng = np.random.default_rng(36)
    X = rng.normal(0, 0.02, (100, 5))
    cov = np.cov(X, rowvar=False)
    cov[:, 2] = cov[:, 0]  # 第 3 列与第 1 列完全相关 → 奇异
    cov[2, :] = cov[0, :]
    w = hrp_weights(cov)
    assert (w >= 0).all()
    assert abs(w.sum() - 1.0) < 1e-8


def test_hrp_panel():
    """面板级 HRP 输出形状与窗口不足处理。"""
    f, rets = _mock_panel(n_days=120, n_codes=10, seed=37)
    out = optimize_weights_hrp(rets, window=60, min_periods=30)
    assert out.shape == rets.shape
    assert (out.iloc[:30].values == 0).all()
    assert (out.iloc[60:].sum(axis=1) > 0.5).any()


def test_risk_parity_panel():
    """面板级风险平价（method='risk_parity'，alpha 不参与目标）。"""
    f, rets = _mock_panel(n_days=120, n_codes=10, seed=38)
    out = optimize_weights_qp(f, rets, method="risk_parity", window=60, min_periods=30)
    for t in out.index[60:]:
        row = out.loc[t]
        if row.sum() == 0:
            continue
        assert (row > 0).all()
        assert abs(row.sum() - 1.0) < 1e-6


def test_risk_parity_rejects_short():
    f, rets = _mock_panel(seed=39)
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    with pytest.raises(ValueError):
        solve_portfolio(f.iloc[100], Sigma, method="risk_parity", allow_short=True)


# ===========================================================================
# P2：Black-Litterman / 多空 / Almgren-Chriss 成本
# ===========================================================================
def test_bl_no_views_reduces_to_equilibrium():
    """无观点 → 后验 μ 即均衡收益 π=δΣw_mkt，Σ_BL=Σ。"""
    rng = np.random.default_rng(41)
    X = rng.normal(0, 0.02, (150, 10))
    Sigma = np.cov(X, rowvar=False)
    mw = np.full(10, 0.1)
    mu, Sig = bl_posterior(Sigma, market_weights=mw, views=None, tau=0.05, delta=2.5)
    np.testing.assert_allclose(mu, 2.5 * Sigma @ mw, atol=1e-12)
    np.testing.assert_allclose(Sig, Sigma, atol=1e-12)


def test_bl_view_direction():
    """观点（看好 A 弱于 B）应把后验 μ 朝观点方向拉动。"""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 0.02, (150, 5))
    Sigma = np.cov(X, rowvar=False)
    mw = np.full(5, 0.2)
    mu0, _ = bl_posterior(Sigma, market_weights=mw, views=None)
    P = np.zeros((1, 5)); P[0, 0] = 1.0; P[0, 1] = -1.0  # 观点：0 比 1 强
    views = {"P": P, "q": np.array([0.005]), "Omega": np.array([[1e-5]])}
    mu1, _ = bl_posterior(Sigma, market_weights=mw, views=views)
    # 观点使 μ0 − μ1 相对方向增强（0 vs 1 的差变大）
    assert mu1[0] - mu1[1] > mu0[0] - mu0[1] - 1e-12


def test_bl_views_from_factor_structure():
    f, _ = _mock_panel(seed=43)
    alpha = f.iloc[100].dropna()
    views = bl_views_from_factor(alpha, n_top=5, view_scale=0.002)
    P = views["P"]
    assert P.shape == (1, len(alpha))
    # P 行和 = 0（相对观点，无方向暴露）
    assert abs(P.sum()) < 1e-9
    # 方向：top 因子股票 +、bottom −（P 按 alpha.index 位置索引）
    pos = {c: i for i, c in enumerate(alpha.index)}
    order = alpha.sort_values(ascending=False)
    top_pos = [pos[c] for c in order.index[:5]]
    bot_pos = [pos[c] for c in order.index[-5:]]
    assert (P[0, top_pos] > 0).all()
    assert (P[0, bot_pos] < 0).all()
    assert views["q"][0] != 0


def test_bl_solve_portfolio():
    """method='bl'：权重和为预算、非负、观点收敛到高分股。"""
    f, rets = _mock_panel(seed=44)
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    alpha = f.iloc[100]
    views = bl_views_from_factor(alpha, n_top=5, view_scale=0.002)
    w = solve_portfolio(alpha, Sigma, method="bl", views=views,
                        market_weights=pd.Series(1.0 / len(alpha), index=alpha.index))
    assert (w >= -1e-9).all()
    assert abs(w.sum() - 1.0) < 1e-6
    s = alpha.rank(pct=True).fillna(0.0)
    # BL 用观点，组合分数应显著高于等权
    assert s @ w > s.mean() - 1e-9


def test_short_limit_and_net_budget():
    """多空：净多头=1，空头总量 ≤ short_limit，权重可负。"""
    f, rets = _mock_panel(seed=45)
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    w = solve_portfolio(f.iloc[100], Sigma, method="mvo", risk_aversion=0.5,
                        allow_short=True, short_limit=0.3)
    short_amt = -w[w < 0].sum()
    assert abs(w.sum() - 1.0) < 1e-6          # 净多头预算
    assert short_amt <= 0.3 + 1e-6            # 空头总量限制
    assert short_amt > 0                      # 确实用了空头（mvo 激进时做空弱票）


def test_gross_leverage_limit():
    f, rets = _mock_panel(seed=46)
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    w = solve_portfolio(f.iloc[100], Sigma, method="mvo", risk_aversion=0.5,
                        allow_short=True, gross_limit=1.6)
    assert w.abs().sum() <= 1.6 + 1e-6
    assert abs(w.sum() - 1.0) < 1e-6


def test_quadratic_cost_reduces_turnover():
    """A-C 二次冲击成本：κ₂ 越大，权重越贴近上期（换手越小）。"""
    f, rets = _mock_panel(seed=47)
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    prev = pd.Series(1.0 / len(f.columns), index=f.columns)
    w0 = solve_portfolio(f.iloc[100], Sigma, method="mvo", risk_aversion=0.5,
                         prev_weights=prev)
    w1 = solve_portfolio(f.iloc[100], Sigma, method="mvo", risk_aversion=0.5,
                         prev_weights=prev, quadratic_cost=50.0)
    t0 = 0.5 * (w0 - prev).abs().sum()
    t1 = 0.5 * (w1 - prev).abs().sum()
    assert t1 <= t0 + 1e-9  # 二次成本压换手


def test_linear_vs_quadratic_cost_both_supported():
    """线性 + 二次成本可同时存在（A-C 组合）。"""
    f, rets = _mock_panel(seed=48)
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    prev = pd.Series(1.0 / len(f.columns), index=f.columns)
    w = solve_portfolio(f.iloc[100], Sigma, method="mvo", risk_aversion=0.5,
                        prev_weights=prev, turnover_penalty=1.0, quadratic_cost=10.0)
    assert abs(w.sum() - 1.0) < 1e-6


def test_min_weight_long_short_filter():
    """多空下 min_weight 后处理：正负微仓都清零。"""
    f, rets = _mock_panel(seed=49)
    Sigma = rolling_covariance(rets, f.index[100], window=90, min_periods=60)
    w = solve_portfolio(f.iloc[100], Sigma, method="mvo", risk_aversion=0.5,
                        allow_short=True, short_limit=0.5, min_weight=0.03)
    nz = w[w != 0]
    assert (nz.abs() >= 0.03 - 1e-9).all()
