"""
收益归因单元测试
================

覆盖：
- Fama-MacBeth：能恢复已知因子溢价且 Newey-West t 显著；含截距项
- Brinson-Fachler：单期各效应与手算一致；多期 Carino 链接恒等式（recon_error≈0）
- α/β 分解：CAPM 恢复已知 α/β；多因子贡献表
"""
import numpy as np
import pandas as pd

from research.attribution import alpha_beta, brinson_attribution, fama_macbeth


# ---------------------------------------------------------------------------
# Fama-MacBeth
# ---------------------------------------------------------------------------
def test_fama_macbeth_recovers_premium():
    """构造截面收益 = 0.002×size + 噪声，FM 应恢复 premium≈0.002 且显著。"""
    rng = np.random.default_rng(6)
    n_days, n_codes = 250, 40
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"c{i:02d}" for i in range(n_codes)]
    size = rng.uniform(0.5, 1.5, n_codes)
    size_panel = pd.DataFrame(np.tile(size, (n_days, 1)), idx, codes)
    premium = 0.002
    rets = premium * size_panel + rng.normal(0, 0.01, (n_days, n_codes))
    future = rets.shift(-1)  # 未来一期收益（FM/IC 口径）

    res = fama_macbeth({"size": size_panel}, future)
    assert "intercept" in res.index and "size" in res.index
    assert abs(res.loc["size", "premium"] - premium) < 0.0005
    assert res.loc["size", "t_nw"] > 2.0          # NW 校正后仍显著
    assert res.loc["size", "n_periods"] == n_days - 1
    assert 0.0 <= res.loc["size", "mean_r2"] <= 1.0


def test_fama_macbeth_no_intercept():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-01", periods=60, freq="B")
    codes = ["a", "b", "c", "d"]
    f = pd.DataFrame(rng.normal(0, 1, (len(idx), len(codes))), idx, codes)
    rets = 0.001 * f + rng.normal(0, 0.01, (len(idx), len(codes)))
    future = rets.shift(-1)
    res = fama_macbeth({"f": f}, future, add_intercept=False)
    assert "intercept" not in res.index


# ---------------------------------------------------------------------------
# Brinson 归因
# ---------------------------------------------------------------------------
def _brinson_panels_1period():
    """单期（1 个交易日）构造：2 行业 × 2 股票，手算验证。

    行业 X：股 X1(r=0.08) X2(r=0.04)；行业 Y：股 Y1(r=0.03) Y2(r=0.01)
    基准：X 总权重 0.4(X1:0.2,X2:0.2)、Y 总权重 0.6(Y1:0.3,Y2:0.3)
    组合：X 总权重 0.6(X1:0.5,X2:0.1)、Y 总权重 0.4(Y1:0.3,Y2:0.1)
    手算：
      r_b,X=0.06, r_b,Y=0.02, R_b=0.036
      r_p,X=0.073333, r_p,Y=0.025, R_p=0.054
      Alloc_X=0.0048, Alloc_Y=0.0032, Sel_X=0.0053333, Sel_Y=0.003
      Int_X=0.0026667, Int_Y=-0.001, active=0.018
    """
    idx = pd.DatetimeIndex(["2023-01-03"])
    stocks = ["X1", "X2", "Y1", "Y2"]
    r = pd.DataFrame([[0.08, 0.04, 0.03, 0.01]], idx, stocks)
    pw = pd.DataFrame([[0.5, 0.1, 0.3, 0.1]], idx, stocks)
    bw = pd.DataFrame([[0.2, 0.2, 0.3, 0.3]], idx, stocks)
    cat = {"X1": "X", "X2": "X", "Y1": "Y", "Y2": "Y"}
    return r, pw, bw, cat


def test_brinson_single_period_hand_computed():
    r, pw, bw, cat = _brinson_panels_1period()
    df, summary = brinson_attribution(r, pw, bw, cat, freq="D", min_obs=1)

    assert summary["n_periods"] == 1
    assert abs(summary["portfolio_return"] - 0.054) < 1e-8
    assert abs(summary["benchmark_return"] - 0.036) < 1e-8
    assert abs(summary["recon_error"]) < 1e-10   # 三效应恒等式

    assert abs(df.loc["X", "allocation"] - 0.0048) < 1e-8
    assert abs(df.loc["Y", "allocation"] - 0.0032) < 1e-8
    assert abs(df.loc["X", "selection"] - 0.0053333333) < 1e-8
    assert abs(df.loc["Y", "selection"] - 0.003) < 1e-8
    assert abs(df.loc["X", "interaction"] - 0.0026666667) < 1e-8
    assert abs(df.loc["Y", "interaction"] + 0.001) < 1e-8
    assert abs(df.loc["TOTAL", "total"] - 0.018) < 1e-8


def test_brinson_multi_period_carino_recon():
    """多期（2 个月）Carino 链接：链接后的累计效应必须还原累计超额（恒等式）。"""
    idx = pd.date_range("2023-01-03", periods=2, freq="MS")  # 两个月初
    stocks = ["A", "B"]
    rng = np.random.default_rng(9)
    r1 = pd.DataFrame(rng.normal(0.001, 0.01, (2, 2)), idx, stocks)
    pw = pd.DataFrame([[0.7, 0.3], [0.4, 0.6]], idx, stocks)
    bw = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], idx, stocks)
    cat = {"A": "G1", "B": "G2"}
    df, summary = brinson_attribution(r1, pw, bw, cat, freq="M", min_obs=1)
    assert summary["n_periods"] == 2
    assert abs(summary["recon_error"]) < 1e-10
    assert abs(df.loc["TOTAL", "total"] - summary["active_return"]) < 1e-10


def test_brinson_multi_day_month_recon():
    """月度多日（每月 21 个交易日）恒等式必须严格成立（2026-08-03 修复）。

    旧实现用复利期收益，复利交叉项使单期恒等式不成立（recon_error ≠ 0）；
    修复为算术口径（期内日收益简单加总）后 recon_error 应精确为 0。
    注：旧测试每期只有 1 天，无复利，恰好掩盖了该问题。
    """
    idx = pd.date_range("2023-01-02", periods=42, freq="B")  # 2 个月 × 21 天
    stocks = ["A", "B", "C", "D"]
    rng = np.random.default_rng(12)
    r = pd.DataFrame(rng.normal(0.001, 0.01, (len(idx), 4)), idx, stocks)
    pw = pd.DataFrame(np.nan, index=idx, columns=stocks)
    bw = pd.DataFrame(np.nan, index=idx, columns=stocks)
    pw.iloc[0] = [0.5, 0.3, 0.1, 0.1]          # 月初权重
    pw.iloc[21] = [0.1, 0.1, 0.4, 0.4]
    bw.iloc[0] = [0.25, 0.25, 0.25, 0.25]
    bw.iloc[21] = [0.25, 0.25, 0.25, 0.25]
    cat = {"A": "G1", "B": "G1", "C": "G2", "D": "G2"}
    df, summary = brinson_attribution(r, pw, bw, cat, freq="M", min_obs=5)
    assert summary["n_periods"] == 2
    assert abs(summary["recon_error"]) < 1e-10
    assert abs(df.loc["TOTAL", "total"] - summary["active_return"]) < 1e-10


def test_brinson_unheld_category_no_selection():
    """组合完全回避某类别：该类别 selection/interaction 必须为 0（2026-08-03 修复）。

    旧实现把未持有类别的组合收益当 0，制造虚假 selection = -w_b,j·r_b,j；
    修复后 r_p,j 退回 r_b,j（无持仓 = 无选股决策），仅保留配置效应。
    """
    idx = pd.date_range("2023-01-02", periods=5, freq="B")
    r = pd.DataFrame({
        "A": [0.01, 0.01, 0.01, 0.01, 0.01],   # G1
        "B": [0.02, 0.02, 0.02, 0.02, 0.02],
        "C": [0.03, 0.03, 0.03, 0.03, 0.03],   # G2（基准持有，组合回避）
        "D": [0.04, 0.04, 0.04, 0.04, 0.04],
    }, index=idx)
    pw = pd.DataFrame(np.nan, index=idx, columns=["A", "B", "C", "D"])
    bw = pd.DataFrame(np.nan, index=idx, columns=["A", "B", "C", "D"])
    pw.iloc[0] = [0.6, 0.4, 0.0, 0.0]
    bw.iloc[0] = [0.2, 0.2, 0.3, 0.3]
    cat = {"A": "G1", "B": "G1", "C": "G2", "D": "G2"}
    df, summary = brinson_attribution(r, pw, bw, cat, freq="W", min_obs=3)
    assert abs(df.loc["G2", "selection"]) < 1e-10
    assert abs(df.loc["G2", "interaction"]) < 1e-10
    # G2 仍应有配置效应（组合欠配该类别）
    assert df.loc["G2", "allocation"] < 0
    assert abs(summary["recon_error"]) < 1e-10


# ---------------------------------------------------------------------------
# α/β 分解
# ---------------------------------------------------------------------------
def test_alpha_beta_recovers_capm():
    rng = np.random.default_rng(8)
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    m = rng.normal(0.0005, 0.01, n)
    alpha, beta = 0.001, 1.2
    p = alpha + beta * m + rng.normal(0, 0.005, n)
    res = alpha_beta(pd.Series(p, idx), pd.Series(m, idx))

    assert abs(res["alpha"] - alpha) < 0.0004
    assert abs(res["beta"] - beta) < 0.1
    assert res["alpha_t_nw"] > 2.0
    assert res["beta_t_nw"] > 5.0
    assert res["r2"] > 0.8
    assert res["n"] == n
    assert res["factors"].empty


def test_alpha_beta_multi_factor_contributions():
    rng = np.random.default_rng(10)
    n = 300
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    m = rng.normal(0.0005, 0.01, n)
    f1 = rng.normal(0.0002, 0.005, n)   # 风格因子
    alpha, beta_m, beta_f = 0.0005, 1.0, 0.8
    p = alpha + beta_m * m + beta_f * f1 + rng.normal(0, 0.003, n)

    fdf = pd.DataFrame({"size": f1}, index=idx)
    res = alpha_beta(pd.Series(p, idx), pd.Series(m, idx), factor_returns=fdf)

    assert abs(res["alpha"] - alpha) < 0.0003
    assert abs(res["beta"] - 1.0) < 0.1
    assert not res["factors"].empty
    assert abs(res["factors"].loc["size", "coef"] - 0.8) < 0.15
    # 贡献 = 系数 × 因子收益均值
    assert abs(res["factors"].loc["size", "contribution"]
               - res["factors"].loc["size", "coef"] * f1.mean()) < 1e-12


def test_alpha_beta_rf_conversion():
    """rf>0 时 α 应整体下移（扣除无风险收益）。

    构造 β=0 的组合（收益与市场独立）：α = E[r_p - rf]，rf 上升 → α 下降。
    """
    rng = np.random.default_rng(11)
    n = 300
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    m = rng.normal(0.0005, 0.01, n)
    p = 0.001 + rng.normal(0, 0.005, n)   # 与市场独立的组合
    res0 = alpha_beta(pd.Series(p, idx), pd.Series(m, idx), rf=0.0)
    res2 = alpha_beta(pd.Series(p, idx), pd.Series(m, idx), rf=0.05)
    assert abs(res0["alpha"] - 0.001) < 0.0004
    assert abs(res0["beta"]) < 0.2
    assert res2["alpha"] < res0["alpha"]
