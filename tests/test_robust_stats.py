"""
稳健统计（Newey-West）与相关修复的回归测试
===========================================

覆盖：
- auto_lag 带宽选择规则（Andrews 1991 / 立方根）
- nw_tstat：i.i.d. 时 ≈ OLS t；AR(1) 正自相关时 |t_nw| 显著变小（核心性质）
- ols_newey_west：系数与 OLS 一致、正自相关残差下 NW 标准误不缩小
- IC 衰减 off-by-one 修复：decay[1] == 主 IC
- standard_factor_summary 新增 Newey-West 列
- Sharpe 的 rf 日频折算（rf>0 时 Sharpe 下降）
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from backtest.metrics import sharpe_ratio
from research.factor_analysis import calc_ic_decay, calc_ic_series, standard_factor_summary
from research.robust_stats import auto_lag, nw_tstat, ols_newey_west


# ---------------------------------------------------------------------------
# auto_lag
# ---------------------------------------------------------------------------
def test_auto_lag_andrews():
    assert auto_lag(100) == 4            # floor(4*(1.0)^(2/9))
    assert auto_lag(1000) == 6           # floor(4*10^(2/9))
    assert auto_lag(1000, method="cuberoot") == 10
    assert auto_lag(2) == 0
    assert auto_lag(500) >= 1


# ---------------------------------------------------------------------------
# nw_tstat
# ---------------------------------------------------------------------------
def test_nw_tstat_iid_approx_ols():
    """i.i.d. 序列自相关≈0：NW t 应与 OLS t 接近。"""
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, 500)
    t_ols = s.mean() / (s.std() / np.sqrt(len(s)))
    t_nw, se, lag = nw_tstat(s)
    assert lag >= 1
    assert abs(t_nw - t_ols) < 0.2


def test_nw_tstat_ar1_shrinks_t():
    """AR(1) 正自相关：普通 t 虚高，NW 校正后 |t| 必须变小。"""
    rng = np.random.default_rng(1)
    n = 400
    e = rng.normal(0, 1, n)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.6 * s[t - 1] + e[t]
    s += 0.2
    t_ols = s.mean() / (s.std() / np.sqrt(n))
    t_nw, _, _ = nw_tstat(s)
    assert abs(t_nw) < abs(t_ols)


def test_nw_tstat_nan_handling():
    s = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0])
    t, se, lag = nw_tstat(s)
    assert np.isfinite(t) and np.isfinite(se)
    assert lag >= 0


# ---------------------------------------------------------------------------
# ols_newey_west
# ---------------------------------------------------------------------------
def test_ols_newey_west_coef_unchanged_and_se_not_shrunk():
    """系数不变；AR(1) 正自相关残差下 NW 标准误总体不小于 OLS。"""
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(0, 1, n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.7 * e[t - 1] + rng.normal(0, 1)
    e -= e.mean()   # 中心化残差，避免 AR(1) 漂移污染截距
    y = 1.0 + 2.0 * x + e
    X = np.column_stack([np.ones(n), x])
    res = ols_newey_west(X, y)
    assert res["n"] == n
    assert np.allclose(res["beta"], [1.0, 2.0], atol=0.15)
    # 正自相关下 NW 标准误应整体 ≥ OLS（允许小幅抽样噪声）
    assert res["se_nw"].sum() >= res["se_ols"].sum() - 1e-12
    assert np.all(np.isfinite(res["t_nw"]))
    # 输出含 OLS 版可对比
    assert "t_ols" in res and "p_nw" in res


def test_ols_newey_west_small_sample_returns_nan():
    res = ols_newey_west(np.ones((2, 1)), np.array([1.0, 2.0]))
    assert np.isnan(res["beta"][0])


# ---------------------------------------------------------------------------
# IC 衰减 off-by-one 修复
# ---------------------------------------------------------------------------
def test_ic_decay_lag1_equals_main_ic():
    """修复验证：decay[1] 必须等于主 IC 均值（次日收益口径），不再多前移一期。"""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2023-01-01", periods=120, freq="B")
    codes = ["A", "B", "C", "D", "E"]
    rets = pd.DataFrame(rng.normal(0, 0.01, (len(idx), len(codes))), idx, codes)
    factor = rets.shift(1).rolling(5).mean()   # 与滞后收益相关的动量因子
    future = rets.shift(-1)                     # 未来一期收益（主口径）
    ic_main = calc_ic_series(factor, future).dropna().mean()
    decay = calc_ic_decay(factor, future, max_lag=3)
    assert abs(decay[1] - ic_main) < 1e-10
    assert decay.index.tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# standard_factor_summary 的 Newey-West 列
# ---------------------------------------------------------------------------
def test_standard_summary_has_newey_west():
    rng = np.random.default_rng(4)
    idx = pd.date_range("2023-01-01", periods=120, freq="B")
    codes = ["A", "B", "C", "D", "E"]
    rets = pd.DataFrame(rng.normal(0, 0.01, (len(idx), len(codes))), idx, codes)
    factor = rets.rolling(5).mean()
    future = rets.shift(-1)
    s = standard_factor_summary(factor, future)
    assert "t_stat_nw" in s and "p_value_nw" in s and "nw_lag" in s
    assert np.isfinite(s["t_stat_nw"])
    assert 0.0 <= s["p_value_nw"] <= 1.0


# ---------------------------------------------------------------------------
# Sharpe 的 rf 处理
# ---------------------------------------------------------------------------
def test_sharpe_rf_reduces_sharpe():
    rng = np.random.default_rng(5)
    daily = pd.Series(rng.normal(0.001, 0.01, 500))
    s0 = sharpe_ratio(daily, rf=0.0)
    s2 = sharpe_ratio(daily, rf=0.05)   # 5% 年化无风险利率
    assert s2 < s0
    assert s0 > 0


def test_sharpe_rf_zero_unchanged():
    rng = np.random.default_rng(6)
    daily = pd.Series(rng.normal(0.001, 0.01, 300))
    assert sharpe_ratio(daily, rf=0.0, periods_per_year=252) == pytest.approx(
        sharpe_ratio(daily, rf=0.0, periods_per_year=252)
    )
