"""
绩效指标测试
============

用已知输入验证核心指标的计算公式没有被破坏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    annual_return,
    annual_volatility,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


def test_annual_return_constant_daily_return():
    # 每日固定收益 r，252 个交易日的年化收益应为 (1+r)^252 - 1
    r = 0.001
    daily = pd.Series([r] * 252)
    expected = (1 + r) ** 252 - 1
    assert annual_return(daily) == pytest.approx(expected, rel=1e-9)


def test_annual_volatility_zero_for_constant_returns():
    daily = pd.Series([0.001] * 100)
    assert annual_volatility(daily) == pytest.approx(0.0)


def test_max_drawdown_simple_path():
    # 净值路径 1 -> 1.2 -> 0.9 -> 1.1，最大回撤 = (1.2-0.9)/1.2 = 0.25
    daily = pd.Series([0.2, -0.25, 0.2222222])
    assert max_drawdown(daily) == pytest.approx(0.25, rel=1e-4)


def test_sharpe_ratio_zero_when_volatility_exactly_zero():
    # annual_volatility 的 0 值分支只在标准差恰好为 0 时触发；用真正常量（std() 严格为 0）
    # 而不是"看起来恒定但存在浮点误差"的序列。
    daily = pd.Series(np.zeros(100))
    assert sharpe_ratio(daily) == 0.0


def test_win_rate_basic():
    daily = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
    assert win_rate(daily) == pytest.approx(2 / 5)


def test_calmar_ratio_zero_when_no_drawdown():
    daily = pd.Series([0.0] * 10)
    assert calmar_ratio(daily) == 0.0


# ---------------------------------------------------------------------------
# Sortino：下行偏差必须用半方差（2026-08-03 修复，原实现误用负收益日 std）
# ---------------------------------------------------------------------------
def test_sortino_uses_downside_deviation():
    """Sortino 分母 = 半方差 sqrt(mean(min(excess,0)^2))，而非负收益日 std。"""
    daily = pd.Series([0.02, -0.05, -0.03, 0.01, -0.04])
    r = sortino_ratio(daily, periods_per_year=1)
    ar = annual_return(daily, periods_per_year=1)   # 年化口径与实现一致
    dd = float(np.sqrt(np.mean(np.minimum(daily.values, 0.0) ** 2)))
    assert r == pytest.approx(ar / dd)


def test_sortino_differs_from_negative_day_std():
    """构造"负收益日之间离差小但亏损深"的序列，验证口径差异可感知。

    负收益日 std 度量亏损日离散度（≈0），半方差度量亏损深度（>0），
    修复后 Sortino 不应再等于 年化收益/(负收益日std×√252)。
    """
    daily = pd.Series([0.01] * 40 + [-0.10, -0.10, -0.10])   # 亏损日几乎相同
    r = sortino_ratio(daily)
    excess = daily.values
    dd = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    ar = (1 + daily).prod() ** (252 / len(daily)) - 1
    assert r == pytest.approx(ar / (dd * np.sqrt(252)))
    # 旧口径（负收益日 std）会远小于半方差 → 旧 Sortino 会大很多
    neg_std = daily[daily < 0].std()
    assert neg_std < dd, "构造前提：负收益日离差 < 半方差"


def test_sortino_zero_when_no_downside():
    daily = pd.Series([0.01] * 10)
    assert sortino_ratio(daily) == 0.0


# ---------------------------------------------------------------------------
# t 统计量与证明年限（paperswithbacktest 复现口径，2026-09-08 新增）
# ---------------------------------------------------------------------------
from backtest.metrics import calc_all_metrics  # noqa: E402


def test_calc_all_metrics_sharpe_tstat_sign_and_scale():
    """已知均值的确定性序列：t 符号跟均值走，量级 ≈ mean/std*sqrt(n)。"""
    n = 504
    rng = np.random.default_rng(42)
    daily = pd.Series(rng.normal(0.001, 0.01, n))
    m = calc_all_metrics(daily)
    # 正均值 → 正 t；量级对照算术口径（NW 滞后校正后应十分接近）
    assert m["sharpe_t_stat"] > 0
    naive_t = daily.mean() / daily.std() * np.sqrt(n)
    assert m["sharpe_t_stat"] == pytest.approx(naive_t, rel=0.3)

    m_neg = calc_all_metrics(-daily)
    assert m_neg["sharpe_t_stat"] < 0


def test_calc_all_metrics_years_to_prove_monotonic_in_sharpe():
    """(1.96/|SR|)² 随夏普单调递减；SR=0 → inf。"""
    strong = pd.Series(np.random.default_rng(1).normal(0.002, 0.01, 504))
    weak = pd.Series(np.random.default_rng(2).normal(0.0002, 0.01, 504))
    m_strong = calc_all_metrics(strong)
    m_weak = calc_all_metrics(weak)
    assert m_strong["years_to_prove"] < m_weak["years_to_prove"]
    assert calc_all_metrics(pd.Series(np.zeros(100)))["years_to_prove"] == float("inf")


def test_calc_all_metrics_excess_tstat_requires_benchmark():
    """无基准时不出现 excess_t_stat；有基准时超额 t 与超额均值同号。"""
    n = 504
    daily = pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, n))
    bench = pd.Series(np.random.default_rng(4).normal(0.0000, 0.008, n))
    m = calc_all_metrics(daily)
    assert "excess_t_stat" not in m
    mb = calc_all_metrics(daily, benchmark_returns=bench)
    assert (mb["excess_t_stat"] > 0) == (mb["excess_return"] > 0)


# ---------------------------------------------------------------------------
# 算术 Sharpe / 算术口径证明年数（2026-10-28 新增）
# ``t = SR·√T`` 恒等式只认算术 SR；几何 SR 套公式会错配，且**方向不单调**。
# 下面用「精确控制均值」的构造序列，避免随机均值落到近零退化区间。
# ---------------------------------------------------------------------------
def _mean_exact(mu: float, n: int = 1008, sigma: float = 0.012, seed: int = 7):
    """构造日均值**精确**为 ``mu`` 的序列（去掉抽样噪声，测试才可判）。"""
    raw = np.random.default_rng(seed).normal(0.0, sigma, n)
    return pd.Series(raw - raw.mean() + mu)


def test_arith_sharpe_is_the_significance_scale():
    """算术 SR × √(n/252) 与 NW t 同口径（iid 序列下近似恒等）。

    参数照抄 ``v1w-ortho882`` 的量级（vol 24.8% / 算术 SR 0.675）——这是本项目
    的主流区间，也是几何/算术分歧最大的区间：实测几何 0.589 / 算术 0.675
    ⇒ 用几何算的 ``years_to_prove`` 虚高 31%。本用例锁死「算术版 ↔ t 统计量」
    这条不变式，防止后来人把 ``years_to_prove_arith`` 按几何口径重算。
    """
    n = 1008
    daily = _mean_exact(0.000664, n, sigma=0.01563)
    m = calc_all_metrics(daily)
    implied_t = m["sharpe_arith"] * np.sqrt(n / 252)
    assert implied_t == pytest.approx(m["sharpe_t_stat"], rel=0.15)
    # vol 高、SR 低 ⇒ |几何| < |算术| ⇒ 几何口径年数更长（偏保守）
    assert abs(m["sharpe"]) < abs(m["sharpe_arith"])
    assert m["years_to_prove"] > m["years_to_prove_arith"]


def test_negative_return_geo_years_shorter():
    """⚠️ 收益为负时方向**翻转**：几何更负 ⇒ |几何| > |算术| ⇒ 几何年数**虚短**。

    同参数（日波 1.56%）下把日均从 +0.066% 翻成 −0.066%：几何 SR −0.717 /
    算术 −0.665，年数 7.5 vs 8.7。⇒ **「几何版总比算术版保守」是错的**，
    只在正收益（``a>0``）时成立。
    """
    daily = _mean_exact(-0.000664, 1008, sigma=0.01563, seed=8)
    m = calc_all_metrics(daily)
    assert m["sharpe"] < 0 and m["sharpe_arith"] < 0
    assert abs(m["sharpe"]) > abs(m["sharpe_arith"])
    assert m["years_to_prove"] < m["years_to_prove_arith"]


def test_years_to_prove_degenerate_near_zero_sharpe():
    """⚠️ 边界：算术 SR≈0 时几何被 ``b = σ²/2`` 拖累放大，**方向反转**。

    日均精确 0 ⇒ 算术 SR≈0、``years_to_prove_arith`` 天文数字（≈7e29）；但几何
    年化被波动拖累拉成明显负值 ⇒ ``|几何| ≫ |算术|``、``years_to_prove`` 反而
    **虚短**（几百年级别而非天文）。⇒ 近零 Sharpe 上该指标没有解释力。
    """
    daily = _mean_exact(0.0, 1008, seed=9)
    m = calc_all_metrics(daily)
    assert abs(m["sharpe_arith"]) < 1e-9
    assert abs(m["sharpe"]) > 0.05            # 被 b 拖成明显负值
    assert m["years_to_prove_arith"] > 1e20   # 算术口径 ≈ inf（浮点残差非严格 0）
    assert m["years_to_prove"] < 1000         # 几何口径「虚短」——不可当保守估计


def test_years_to_prove_from_sharpe_edge_cases():
    """SR=1.96 → 1 年；SR=0 → inf；负 SR 用绝对值。"""
    from backtest.metrics import years_to_prove_from_sharpe
    assert years_to_prove_from_sharpe(0.0) == float("inf")
    assert years_to_prove_from_sharpe(1.96) == pytest.approx(1.0, abs=0.05)
    assert years_to_prove_from_sharpe(-1.96) == years_to_prove_from_sharpe(1.96)
