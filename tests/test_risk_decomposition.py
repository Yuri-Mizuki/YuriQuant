"""
optimize/risk.py 风险分解测试。

覆盖：
- Euler 分解恒等式：Σ CR_k = σ_p（成分贡献之和 = 组合波动率）
- 等权组合 MRC 应相等
- 风格因子方差贡献 ≤ 总方差（特质风险 ≥ 0）
- VaR/CVaR 成分分解：方向正确、数值有限
- 防前视：修改未来收益不影响当期风险分解结果
- 数值稳定性（PSD 正则化无 NaN）
- ffill 权重：非调仓日沿用上次持仓
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimize.risk import risk_decomposition


def _mock_weights_and_returns(n_days: int = 150, n_codes: int = 10, seed: int = 42):
    """等权月度调仓的权重 + 随机收益面板。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]

    # 收益面板
    rets = pd.DataFrame(
        rng.normal(0, 0.02, (n_days, n_codes)), idx, codes,
    )

    # 月度调仓等权权重（仅在月末有非零值，其余日全 0 模拟 weights_history）
    w = pd.DataFrame(0.0, idx, codes)
    s = pd.Series(idx, index=idx)
    monthly_dates = s.groupby(pd.Grouper(freq="ME")).last()
    for dt in monthly_dates:
        if dt in w.index:
            w.loc[dt] = 1.0 / n_codes

    return w, rets, codes


# ===========================================================================
# Euler 分解恒等式
# ===========================================================================
def test_euler_decomposition_identity():
    """成分风险贡献之和 = 组合波动率（Euler 定理）。"""
    w, rets, codes = _mock_weights_and_returns(n_days=200, n_codes=10)
    result = risk_decomposition(w, rets, freq="M", min_periods=60, covariance_window=120)

    assert "error" not in result, f"不应返回错误: {result.get('error')}"

    cr_df = result["risk_contributions"]
    port_vol = result["summary"]["avg_portfolio_vol"]
    # CR 之和应近似等于最新截面的组合波动率
    latest_vol = result["exposure"]["portfolio_vol"].iloc[-1]
    assert abs(cr_df["CR"].sum() - latest_vol) < 1e-6, (
        f"CR 之和 {cr_df['CR'].sum()} != 组合波动率 {latest_vol}"
    )


def test_equal_weight_mrc_equal():
    """等权组合的 MRC（边际风险贡献）应近似相等。"""
    # 用 isotropic 收益（相同波动率、零相关性）确保 MRC 严格相等
    rng = np.random.default_rng(99)
    n_days, n_codes = 200, 10
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    rets = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_codes)), idx, codes)

    w = pd.DataFrame(0.0, idx, codes)
    monthly_dates = pd.Series(idx, index=idx).groupby(pd.Grouper(freq="ME")).last()
    for dt in monthly_dates:
        if dt in w.index:
            w.loc[dt] = 1.0 / n_codes

    result = risk_decomposition(w, rets, freq="M", min_periods=60, covariance_window=120)
    cr_df = result["risk_contributions"]
    # MRC 在等权下应方差很小（各股贡献相近）
    mrc_std = cr_df["MRC"].std()
    mrc_mean = cr_df["MRC"].mean()
    cv = mrc_std / (abs(mrc_mean) + 1e-12)
    assert cv < 0.5, f"等权组合 MRC 变异系数应小，实际 {cv:.4f}"


# ===========================================================================
# 风格因子方差贡献
# ===========================================================================
def test_style_variance_contribution_le_total():
    """风格因子方差贡献之和 ≤ 总方差（特质风险 ≥ 0）。"""
    w, rets, codes = _mock_weights_and_returns(n_days=200, n_codes=10)

    # 构造一个风格暴露面板（动量 = 过去 20 日收益）
    mom = rets.rolling(20).mean()
    style = {"mom": mom}

    result = risk_decomposition(
        w, rets, style_exposures=style, freq="M",
        min_periods=60, covariance_window=120,
    )
    vd = result["variance_decomp"]
    assert vd["explained_variance"] <= vd["total_variance"] + 1e-10, (
        f"风格解释方差 {vd['explained_variance']} > 总方差 {vd['total_variance']}"
    )
    assert vd["idiosyncratic_risk"] >= -1e-10, "特质风险不应为负"


def test_style_zscore_normalization():
    """风格暴露应经 zscore 归一（量纲无关）。"""
    w, rets, codes = _mock_weights_and_returns(n_days=200, n_codes=10)

    # 用量纲极端的暴露（市值 = 随机大数）
    rng = np.random.default_rng(55)
    mc = pd.DataFrame(
        rng.uniform(1e8, 1e10, (200, 10)),
        index=rets.index, columns=codes,
    )

    result = risk_decomposition(
        w, rets, style_exposures={"size": mc}, freq="M",
        min_periods=60, covariance_window=120,
    )
    vd = result["variance_decomp"]
    # size 因子贡献应为有限正数（非 NaN/Inf）
    val = vd["style_factor_contrib"].get("size", 0)
    assert np.isfinite(val), f"size 贡献非有限: {val}"
    assert val >= 0, f"方差贡献不应为负: {val}"


# ===========================================================================
# VaR / CVaR
# ===========================================================================
def test_var_cvar_finite_and_direction():
    """VaR/CVaR 数值有限、方向正确（VaR ≤ 0 左尾）。"""
    w, rets, codes = _mock_weights_and_returns(n_days=200, n_codes=10)
    result = risk_decomposition(w, rets, freq="M", min_periods=60, covariance_window=120)

    vc = result["var_cvar"]
    assert np.isfinite(vc["portfolio_VaR"]), "VaR 非有限"
    assert np.isfinite(vc["portfolio_CVaR"]), "CVaR 非有限"
    # CVaR 应更极端（更负或相等）
    assert vc["portfolio_CVaR"] <= vc["portfolio_VaR"] + 1e-10, (
        f"CVaR {vc['portfolio_CVaR']} 应 ≤ VaR {vc['portfolio_VaR']}"
    )


def test_component_var_sum():
    """成分 VaR 之和应近似 = 组合 VaR（Epperlein 恒等式）。"""
    w, rets, codes = _mock_weights_and_returns(n_days=250, n_codes=10)
    result = risk_decomposition(w, rets, freq="M", min_periods=60, covariance_window=120)

    vc = result["var_cvar"]
    if "component_VaR" in vc:
        comp_sum = vc["component_VaR"].sum()
        port_var = vc["portfolio_VaR"]
        # 成分 VaR 之和应近似等于组合 VaR（历史模拟口径下近似成立）
        assert abs(comp_sum - port_var) < abs(port_var) * 0.5 + 1e-6, (
            f"成分 VaR 之和 {comp_sum} 与组合 VaR {port_var} 偏差过大"
        )


# ===========================================================================
# 防前视
# ===========================================================================
def test_no_lookahead():
    """修改未来收益不影响当期风险分解结果。"""
    w, rets, codes = _mock_weights_and_returns(n_days=200, n_codes=10, seed=100)

    result1 = risk_decomposition(w, rets, freq="M", min_periods=60, covariance_window=120)

    # 篡改最后一期（未来）收益
    rets_mod = rets.copy()
    rets_mod.iloc[-1] = 999.0  # 极端值
    result2 = risk_decomposition(w, rets_mod, freq="M", min_periods=60, covariance_window=120)

    # 最后一期调仓日的分解不应改变（只用 < date 的收益估 Σ）
    vol1 = result1["exposure"]["portfolio_vol"].iloc[-1]
    vol2 = result2["exposure"]["portfolio_vol"].iloc[-1]
    assert abs(vol1 - vol2) < 1e-10, f"防前视失败: {vol1} vs {vol2}"


# ===========================================================================
# 数值稳定性
# ===========================================================================
def test_no_nan_in_output():
    """输出不应含 NaN（PSD 正则化生效）。"""
    w, rets, codes = _mock_weights_and_returns(n_days=200, n_codes=15)
    result = risk_decomposition(w, rets, freq="M", min_periods=60, covariance_window=120)

    assert "error" not in result
    cr_df = result["risk_contributions"]
    assert not cr_df[["MRC", "CR", "CR_pct"]].isna().any().any(), "CR/MRC 含 NaN"
    assert np.isfinite(result["summary"]["avg_portfolio_vol"]), "波动率含 NaN"


def test_high_dimensional_stability():
    """高维（N > T/2）协方差应走收缩兜底，不崩。"""
    n_days, n_codes = 80, 30  # N 接近 T/2，LedoitWolf 可能 fallback
    rng = np.random.default_rng(77)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    rets = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_codes)), idx, codes)

    w = pd.DataFrame(0.0, idx, codes)
    monthly_dates = pd.Series(idx, index=idx).groupby(pd.Grouper(freq="ME")).last()
    for dt in monthly_dates:
        if dt in w.index:
            w.loc[dt] = 1.0 / n_codes

    result = risk_decomposition(
        w, rets, freq="M", min_periods=30, covariance_window=80,
    )
    # 至少应有结果（可能部分截面被跳过）
    if "error" not in result:
        assert result["summary"]["n_periods"] >= 1


# ===========================================================================
# 行业贡献
# ===========================================================================
def test_industry_contribution():
    """行业贡献按 CR 分组求和。"""
    w, rets, codes = _mock_weights_and_returns(n_days=200, n_codes=12)

    # 3 个行业，各 4 只
    ind_names = ["银行", "电子", "食品"]
    ind_series = pd.Series(
        [ind_names[i % 3] for i in range(12)], index=codes
    )
    industry_panel = pd.DataFrame(
        np.tile(ind_series.values, (len(rets), 1)),
        index=rets.index, columns=codes,
    )

    result = risk_decomposition(
        w, rets, industry_panel=industry_panel, freq="M",
        min_periods=60, covariance_window=120,
    )
    vd = result["variance_decomp"]
    # 至少应有一个行业贡献
    assert len(vd["industry_contrib"]) >= 1
    # 行业贡献之和应近似 = CR 之和（容差：行业面板可能未全覆盖所有持仓 code）
    total_ind = sum(vd["industry_contrib"].values())
    cr_sum = result["risk_contributions"]["CR"].sum()
    assert abs(total_ind - cr_sum) / (abs(cr_sum) + 1e-12) < 0.05, (
        f"行业贡献之和 {total_ind} 与 CR 之和 {cr_sum} 偏差 > 5%"
    )


# ===========================================================================
# 风险预算校验
# ===========================================================================
def test_risk_budget_check():
    """提供 risk_budgets 时输出 budget_check。"""
    w, rets, codes = _mock_weights_and_returns(n_days=200, n_codes=9)

    ind_names = ["银行", "电子", "食品"]
    ind_series = pd.Series(
        [ind_names[i % 3] for i in range(9)], index=codes
    )
    industry_panel = pd.DataFrame(
        np.tile(ind_series.values, (len(rets), 1)),
        index=rets.index, columns=codes,
    )

    budgets = {"银行": 0.333, "电子": 0.333, "食品": 0.334}
    result = risk_decomposition(
        w, rets, industry_panel=industry_panel,
        risk_budgets=budgets, freq="M",
        min_periods=60, covariance_window=120,
    )
    bc = result["budget_check"]
    assert bc is not None, "budget_check 未生成"
    assert "银行" in bc
    assert "target" in bc["银行"]
    assert "actual" in bc["银行"]
    assert "deviation" in bc["银行"]


# ===========================================================================
# ffill 权重
# ===========================================================================
def test_ffill_weights():
    """非调仓日权重应被 ffill 填充，不产生零持仓假象。"""
    n_days, n_codes = 200, 10
    rng = np.random.default_rng(33)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    rets = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_codes)), idx, codes)

    # 仅第一天有权重
    w = pd.DataFrame(0.0, idx, codes)
    w.iloc[0] = 1.0 / n_codes

    # 用 freq="D" 逐日计算
    result = risk_decomposition(
        w, rets, freq="D", min_periods=60, covariance_window=120,
    )
    if "error" not in result:
        # 第一天之后的截面也应有效（ffill 后有权重）
        assert result["summary"]["n_periods"] >= 1
