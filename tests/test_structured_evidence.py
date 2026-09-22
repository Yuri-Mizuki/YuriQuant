"""结构化证据统计量测试（evals 改造包 2026-09-22）。

覆盖 ``stats.ic.monotonicity_ratio``（分层收益单调性占比）与
``stats.ic.perturbation_fidelity``（PFS 扰动保真度）。

口径锁死要点：
- 单调性与 ``quantile_backtest`` 同口径（当日因子赚当日未来一期收益、
  有效观测 < 组数跳过、qcut duplicates=drop）；
- direction="auto" 对完美反号因子给 1.0（因子可取反，抓的是中间组乱序）；
- PFS 与 ``factor_autocorr`` 正交：时间延续性高 ≠ 噪声鲁棒；
- PFS 固定 seed → 同一面板重复调用结果逐位一致（入库可复现）；
- 极端值主导的因子 PFS 显著低于 ranks 稳定的因子（判别力）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stats.ic import (
    monotonicity_ratio,
    perturbation_fidelity,
    quantile_backtest,
)

N_DAYS, N_CODES = 120, 40


def _idx_codes():
    idx = pd.date_range("2024-01-01", periods=N_DAYS, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(N_CODES)]
    return idx, codes


# ---------------------------------------------------------------------------
# monotonicity_ratio
# ---------------------------------------------------------------------------

def test_monotonicity_perfect_linear_factor_is_high():
    """因子 = 未来收益线性叠加小噪声 → 单调占比应接近 1。"""
    rng = np.random.default_rng(7)
    idx, codes = _idx_codes()
    rets = pd.DataFrame(rng.normal(0, 0.02, (N_DAYS, N_CODES)), idx, codes)
    factor = rets * 20 + rng.normal(0, 1e-4, (N_DAYS, N_CODES))
    ratio = monotonicity_ratio(factor, rets, n_quantiles=5)
    assert ratio > 0.95


def test_monotonicity_noise_factor_is_coin_flip():
    """因子与收益独立 → 单调占比应接近随机水平（5 组随机单调概率 ≈ 2/24≈8%）。"""
    rng = np.random.default_rng(11)
    idx, codes = _idx_codes()
    factor = pd.DataFrame(rng.normal(0, 1, (N_DAYS, N_CODES)), idx, codes)
    rets = pd.DataFrame(rng.normal(0, 0.02, (N_DAYS, N_CODES)), idx, codes)
    ratio = monotonicity_ratio(factor, rets, n_quantiles=5)
    # 随机基线：5 组全序排列中严格单调的占 2/120；auto 取双向 ≈ 4/120。
    # 噪声日间波动大，给宽上限 0.35 防翻车，下限 0（不该显著高于随机）。
    assert 0.0 <= ratio <= 0.35


def test_monotonicity_direction_semantics():
    """direction='asc' 只认递增：完美反号因子 asc=0 / desc=1 / auto=1。"""
    rng = np.random.default_rng(3)
    idx, codes = _idx_codes()
    rets = pd.DataFrame(rng.normal(0, 0.02, (N_DAYS, N_CODES)), idx, codes)
    # 因子 = -未来收益（完美反号）
    factor = -rets * 20 + rng.normal(0, 1e-6, (N_DAYS, N_CODES))
    asc = monotonicity_ratio(factor, rets, direction="asc")
    desc = monotonicity_ratio(factor, rets, direction="desc")
    auto = monotonicity_ratio(factor, rets, direction="auto")
    assert asc < 0.05
    assert desc > 0.95
    assert auto > 0.95


def test_monotonicity_matches_quantile_backtest_convention():
    """口径一致性：单调比 ≈ 分层日收益差的符号率（同一 qcut 口径的独立复算）。

    对完美单调因子，quantile_backtest 的逐日 Q5-Q1 应恒为正——单调占比应为 1。
    """
    rng = np.random.default_rng(9)
    idx, codes = _idx_codes()
    rets = pd.DataFrame(rng.normal(0, 0.02, (N_DAYS, N_CODES)), idx, codes)
    factor = rets * 20
    ratio = monotonicity_ratio(factor, rets, n_quantiles=5)
    nav = quantile_backtest(factor, rets, n_quantiles=5)
    daily_spread = nav["Q5"].diff() - nav["Q1"].diff()
    pos_rate = float((daily_spread.dropna() > 0).mean())
    assert ratio == 1.0
    assert pos_rate > 0.95


def test_monotonicity_insufficient_data_returns_nan():
    """全部截面有效观测 < 组数 → NaN（不是 0）。"""
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    factor = pd.DataFrame(np.random.default_rng(1).normal(size=(10, 3)), idx,
                          ["A", "B", "C"])
    rets = factor.copy()
    assert np.isnan(monotonicity_ratio(factor, rets, n_quantiles=5))


# ---------------------------------------------------------------------------
# perturbation_fidelity
# ---------------------------------------------------------------------------

def test_pfs_fixed_seed_reproducible():
    """固定 seed → 同一面板两次调用逐位一致（入库可复现要求）。"""
    rng = np.random.default_rng(5)
    idx, codes = _idx_codes()
    factor = pd.DataFrame(rng.normal(0, 1, (N_DAYS, N_CODES)), idx, codes)
    a = perturbation_fidelity(factor)
    b = perturbation_fidelity(factor)
    assert a == b


def test_pfs_high_for_well_spread_factor():
    """正态因子排名分散 → 1% 扰动下 PFS 应非常高（>0.99）。"""
    rng = np.random.default_rng(5)
    idx, codes = _idx_codes()
    factor = pd.DataFrame(rng.normal(0, 1, (N_DAYS, N_CODES)), idx, codes)
    assert perturbation_fidelity(factor) > 0.99


def test_pfs_discriminates_extreme_value_dominated_factor():
    """少数极端值主导排名的因子 → PFS 显著更低（判别力测试）。

    构造：每行 95% 样本挤在 [0, 0.01] 的微小噪声带 + 5% 大极端值。
    极端值决定名次两端，但中间名次对同尺度扰动敏感度远高于正态因子。
    """
    rng = np.random.default_rng(13)
    idx, codes = _idx_codes()
    base = rng.uniform(0, 0.01, (N_DAYS, N_CODES))
    extreme_mask = rng.random((N_DAYS, N_CODES)) < 0.05
    vals = np.where(extreme_mask, rng.normal(0, 10, (N_DAYS, N_CODES)), base)
    extreme_factor = pd.DataFrame(vals, idx, codes)
    normal_factor = pd.DataFrame(rng.normal(0, 1, (N_DAYS, N_CODES)), idx, codes)
    pfs_extreme = perturbation_fidelity(extreme_factor, noise_scale=0.05)
    pfs_normal = perturbation_fidelity(normal_factor, noise_scale=0.05)
    assert pfs_extreme < pfs_normal


def test_pfs_t_distribution_stronger_than_gauss():
    """重尾扰动（t 分布）比高斯扰动破坏力更强 → PFS 更低。"""
    rng = np.random.default_rng(17)
    idx, codes = _idx_codes()
    factor = pd.DataFrame(rng.normal(0, 1, (N_DAYS, N_CODES)), idx, codes)
    pfs_gauss = perturbation_fidelity(factor, noise_scale=0.05, distribution="gauss")
    pfs_t = perturbation_fidelity(factor, noise_scale=0.05, distribution="t", df_t=2)
    assert pfs_t <= pfs_gauss


def test_pfs_empty_panel_returns_nan():
    empty = pd.DataFrame()
    assert np.isnan(perturbation_fidelity(empty))
