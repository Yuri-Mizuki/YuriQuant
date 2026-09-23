"""结构化证据统计量测试（evals 改造包 2026-09-22；09-23 口径修正后重写）。

覆盖 ``stats.ic.monotonicity_ratio``（分层收益单调性占比）与
``stats.ic.perturbation_fidelity``（PFS 扰动保真度）。

口径锁死要点（09-23 修正后）：
- **mono 只认完整分组**：qcut duplicates=drop 后组数 < n_quantiles 的日跳过，
  有效日 < min_valid_days 返回 NaN。回归锁：与收益无关的三值因子必须接近
  随机水平（旧口径曾给 0.96 的假满分——组数不足 artifact）；
- direction="auto" 对完美反号因子给 1.0（因子可取反，抓的是中间组乱序）；
- **PFS tie_aware（默认）**：同一截面内相同值共享同一扰动。回归锁：高占比
  并列面板 PFS=1.0（旧口径逐元素独立噪声曾把并列人为打散，pap_breakout_atr
  实测 0.145 的"低 PFS"纯属噪声模型 artifact）；
- PFS 与 ``factor_autocorr`` 正交：时间延续性高 ≠ 噪声鲁棒；
- PFS 固定 seed → 同一面板重复调用结果逐位一致（入库可复现）。
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


def test_monotonicity_low_cardinality_no_false_perfect():
    """【09-23 回归锁】与收益无关的三值因子 → mono 必须是 NaN（不是假满分）。

    旧口径只要 qcut 后 ≥2 组就计入该日：三值因子 qcut 后 2 组，auto 语义
    下单差分非正即负 → 该日必然记"单调"，实测 mono=0.96。新口径只认完整
    5 组：三值因子永远分不出 5 组 → 有效日=0 < min_valid_days → NaN。
    这是"组数不足 artifact"的直接反证。
    """
    rng = np.random.default_rng(23)
    idx, codes = _idx_codes()
    tri = pd.DataFrame(rng.choice([0.0, 1.0, 2.0], p=[.6, .3, .1],
                                  size=(N_DAYS, N_CODES)), idx, codes)
    rets = pd.DataFrame(rng.normal(0, 0.02, (N_DAYS, N_CODES)), idx, codes)
    ratio = monotonicity_ratio(tri, rets)
    assert np.isnan(ratio)


def test_monotonicity_binary_factor_returns_nan():
    """二值因子同理：qcut 后最多 2 组 → 全部塌缩日跳过 → NaN。"""
    rng = np.random.default_rng(29)
    idx, codes = _idx_codes()
    binary = pd.DataFrame((rng.normal(0, 1, (N_DAYS, N_CODES)) > 0).astype(float),
                          idx, codes)
    rets = pd.DataFrame(rng.normal(0, 0.02, (N_DAYS, N_CODES)), idx, codes)
    assert np.isnan(monotonicity_ratio(binary, rets))


def test_monotonicity_min_valid_days_floor():
    """有效日不足 min_valid_days → NaN（单日拼不出占比）。

    旧口径曾让 alpha191_004（有效日仅 1 天）给出 mono=1.0 的假满分。
    """
    rng = np.random.default_rng(31)
    idx, codes = _idx_codes()
    rets = pd.DataFrame(rng.normal(0, 0.02, (N_DAYS, N_CODES)), idx, codes)
    factor = rets * 20  # 完美单调因子，但只给 10 天有效日
    ratio = monotonicity_ratio(factor.iloc[:10], rets.iloc[:10],
                               min_valid_days=60)
    assert np.isnan(ratio)
    # 放宽下限后同一段数据有值
    ratio_ok = monotonicity_ratio(factor.iloc[:10], rets.iloc[:10],
                                  min_valid_days=5)
    assert 0.0 <= ratio_ok <= 1.0


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


def test_pfs_tie_aware_ties_share_perturbation():
    """【09-23 回归锁】高占比并列面板 → PFS=1.0（tie_aware 并列共享扰动）。

    旧口径逐元素独立噪声会把本应不可分的并列值人为打散：99.4% 并列面板
    实测 PFS=0.145。tie_aware 下并列名次集体移动、相对秩序不变 → PFS=1。
    """
    rng = np.random.default_rng(37)
    idx, codes = _idx_codes()
    # 99% 样本同值(1.0，非零——乘性噪声下 0 值天然不动，测不出差异)
    # + 1% 独立大值：旧口径 artifact 的重现构造
    vals = np.ones((N_DAYS, N_CODES))
    rare_mask = rng.random((N_DAYS, N_CODES)) < 0.01
    vals[rare_mask] = 10.0
    tied = pd.DataFrame(vals, idx, codes)
    pfs_tie = perturbation_fidelity(tied, tie_aware=True, n_trials=5)
    assert pfs_tie > 0.999
    # 旧口径对照：同一面板逐元素独立噪声 → 并列被人为打散 → 显著低于 1
    pfs_old = perturbation_fidelity(tied, tie_aware=False, n_trials=5)
    assert pfs_old < 0.95


def test_pfs_tie_aware_matches_old_on_continuous():
    """tie_aware 与旧口径在连续面板上应给出几乎相同的结果（并列极少）。

    连续正态面板 unique 数 ≈ 行长，并列可忽略——两种噪声模型在数学上
    近似同分布，均值差异应远小于 trial 间波动量级。回归锁：修正不能
    改变连续因子的既有读数（alpha158_MA5 实测 Δ=0）。
    """
    rng = np.random.default_rng(41)
    idx, codes = _idx_codes()
    factor = pd.DataFrame(rng.normal(0, 1, (N_DAYS, N_CODES)), idx, codes)
    pfs_new = perturbation_fidelity(factor, tie_aware=True, n_trials=5)
    pfs_old = perturbation_fidelity(factor, tie_aware=False, n_trials=5)
    assert abs(pfs_new - pfs_old) < 0.01


def test_pfs_discriminates_extreme_value_dominated_factor():
    """少数极端值主导排名的因子 → PFS 显著更低（判别力测试）。

    构造：每行 95% 样本挤在 [0, 0.01] 的微小噪声带 + 5% 大极端值。
    极端值决定名次两端，但中间名次对同尺度扰动敏感度远高于正态因子。
    （09-23 注：此为"连续重尾"型低 PFS 的正当来源——与并列打散
    artifact 不同，这里的微小间隙是真实的数据敏感性。）
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
