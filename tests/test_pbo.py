"""策略级过拟合检验（stats/pbo.py）单元测试。

覆盖：
- CSCV PBO 的统计行为锚点：纯噪声 → PBO≈0.5（无真本事时，样本内最优者
  样本外排名近似均匀 → 一半组合落入后半）；唯一真信号 → PBO=0；
- 确定性与结构字段（组合数 = C(S, S/2)、logits/omegas 形状）；
- 参数校验（块数、T 不足、NaN、单配置）；
- DSR 的解析锚点：零 SR + 零试验方差 → Φ(0)=0.5；单调性；偏度/试验数的影响；
- deflate_best 端到端。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stats.pbo import (
    _avg_rank,
    cscv_pbo,
    deflate_best,
    deflated_sharpe_ratio,
    trial_sharpes,
)


# ---------------------------------------------------------------------------
# CSCV PBO
# ---------------------------------------------------------------------------
def _noise_matrix(n_days: int = 1000, n_trials: int = 20, seed: int = 7):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, (n_days, n_trials))


def test_avg_rank_ties_averaged():
    ranks = _avg_rank(np.array([3.0, 1.0, 2.0, 1.0]))
    assert ranks.tolist() == [4.0, 1.5, 3.0, 1.5]


def test_pbo_random_noise_near_half():
    """无真本事：PBO 应回到 ~0.5 附近（不显著偏离）。"""
    res = cscv_pbo(_noise_matrix(), n_partitions=8)
    assert 0.3 <= res["pbo"] <= 0.7
    assert 0.35 <= res["omega_mean"] <= 0.65
    assert res["n_combinations"] == 70                  # C(8,4)
    assert res["logits"].shape == (70,)
    assert res["omegas"].shape == (70,)
    assert res["metric"] == "sharpe" and res["n_trials"] == 20


def test_pbo_real_signal_zero():
    """唯一真信号策略：样本内最优恒为它、样本外也最优 → PBO = 0。"""
    M = _noise_matrix(n_days=1500, n_trials=20, seed=11)
    M[:, 0] += 0.003                                    # 唯一有真实均值
    res = cscv_pbo(M, n_partitions=8)
    assert res["pbo"] == 0.0
    assert res["omega_mean"] > 0.9


def test_pbo_deterministic_and_dataframe_columns():
    M = _noise_matrix(n_days=400, n_trials=6, seed=3)
    df = pd.DataFrame(M, columns=[f"cfg{i}" for i in range(6)])
    r1 = cscv_pbo(df, n_partitions=8)
    r2 = cscv_pbo(df, n_partitions=8)
    assert np.array_equal(r1["logits"], r2["logits"])
    assert r1["columns"] == [f"cfg{i}" for i in range(6)]
    r3 = cscv_pbo(M, n_partitions=8)
    assert np.array_equal(r1["logits"], r3["logits"])   # DataFrame/ndarray 同结果


def test_pbo_mean_metric_runs():
    res = cscv_pbo(_noise_matrix(n_days=300, n_trials=5, seed=5),
                   n_partitions=6, metric="mean")
    assert res["n_combinations"] == 20                  # C(6,3) = 20
    assert 0.0 <= res["pbo"] <= 1.0


def test_pbo_input_validation():
    with pytest.raises(ValueError, match="偶数"):
        cscv_pbo(_noise_matrix(100, 3), n_partitions=7)
    with pytest.raises(ValueError, match="不足以切"):
        cscv_pbo(_noise_matrix(10, 3), n_partitions=16)
    with pytest.raises(ValueError, match="NaN"):
        M = _noise_matrix(100, 4)
        M[3, 1] = np.nan
        cscv_pbo(M, n_partitions=8)
    with pytest.raises(ValueError, match="N ≥ 2"):
        cscv_pbo(np.zeros((100, 1)), n_partitions=8)


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------
def test_dsr_zero_sr_zero_variance_is_exactly_half():
    """SR=0 且试验间无差异（SR₀=0）→ z=0 → DSR=Φ(0)=0.5。"""
    srs = np.full(10, 0.1)
    assert deflated_sharpe_ratio(0.0, srs, t_obs=500) == pytest.approx(0.5)


def test_dsr_high_sr_few_trials_significant():
    srs = np.full(5, 0.02)
    srs[0] = 0.30                                       # 明显赢家、试验间方差小
    dsr = deflated_sharpe_ratio(0.30, srs, t_obs=1000)
    assert dsr > 0.99


def test_dsr_many_diverse_trials_kills_edge():
    """试验数多且彼此差异大 → SR₀ 高（500 次试验 σ=0.08 时 SR₀≈0.245）
    → SR=0.20 虽是"赢家"仍被选择偏差解释，不再显著。"""
    rng = np.random.default_rng(1)
    srs = rng.normal(0.05, 0.08, 500)                   # 500 次试验的宽分布
    dsr = deflated_sharpe_ratio(0.20, srs, t_obs=1000)
    assert dsr < 0.3


def test_dsr_monotone_in_sr():
    rng = np.random.default_rng(2)
    srs = rng.normal(0.0, 0.05, 50)
    prev = -1.0
    for sr in np.linspace(0.0, 0.4, 9):
        d = deflated_sharpe_ratio(float(sr), srs, t_obs=800)
        assert d >= prev - 1e-12
        prev = d


def test_dsr_negative_skew_hurts():
    """同 SR 下，负偏度抬高 PSR 分母 → DSR 变小（尾部风险惩罚）。"""
    srs = np.full(8, 0.05)
    d0 = deflated_sharpe_ratio(0.25, srs, t_obs=600, skew=0.0, kurt=3.0)
    d1 = deflated_sharpe_ratio(0.25, srs, t_obs=600, skew=-1.0, kurt=3.0)
    assert d1 < d0


def test_dsr_input_validation():
    with pytest.raises(ValueError, match="≥2"):
        deflated_sharpe_ratio(0.2, np.array([0.2]), t_obs=100)
    with pytest.raises(ValueError, match="t_obs"):
        deflated_sharpe_ratio(0.2, np.full(5, 0.1), t_obs=1)


# ---------------------------------------------------------------------------
# deflate_best（端到端）
# ---------------------------------------------------------------------------
def test_deflate_best_picks_and_scores():
    rng = np.random.default_rng(9)
    M = rng.normal(0.0, 0.01, (1200, 12))
    M[:, 5] += 0.004                                    # 第 5 列是真赢家
    df = pd.DataFrame(M, columns=[f"a{i}" for i in range(12)])
    out = deflate_best(df)
    assert out["best_index"] == 5 and out["best_column"] == "a5"
    assert out["best_sr"] == pytest.approx(float(trial_sharpes(M)[5]))
    assert 0.9 < out["dsr"] <= 1.0
    assert out["n_trials"] == 12 and out["t_obs"] == 1200
    assert np.isfinite(out["skew"]) and out["kurt"] > 0


def test_deflate_best_rejects_zero_variance_column():
    M = _noise_matrix(300, 4, seed=1)
    M[:, 2] = 0.0
    with pytest.raises(ValueError, match="零方差"):
        deflate_best(M)
