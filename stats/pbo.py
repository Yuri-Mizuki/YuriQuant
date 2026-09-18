"""策略级过拟合检验（华泰 AI19 重采样 / AI22 回测过拟合概率，2026-09-18 接入）。

模型层的 CPCV（``scripts/evaluation/cpcv_eval.py``）检验的是**模型训练**的
稳健性；本模块检验的是**策略/参数选择**层面的过拟合——"在 N 个候选配置里
挑样本内最优，样本外大概率不再最优"的概率：

- :func:`cscv_pbo` —— **CSCV PBO**（Combinatorially Symmetric Cross-Validation,
  Bailey, Borwein, López de Prado & Zhu 2017, Journal of Computational Finance）。
  把 T 期收益切成 S 块，穷举 C(S, S/2) 种对半组合；每组合内取**样本内**最优配置
  n*，看它的**样本外**相对排名 ω̄。ω̄ ≤ 0.5（样本外落入后半）即记一次过拟合，
  PBO = 过拟合组合占比。
- :func:`deflated_sharpe_ratio` —— **DSR**（Bailey & López de Prado 2014,
  Journal of Portfolio Management）。在"N 次试验里挑出的最大 Sharpe"的零分布下
  问：观测 Sharpe 还有多大概率仍然显著。试验次数越多、试验间 Sharpe 方差越大，
  门槛越高。
- :func:`deflate_best` —— 一步到位：输入 T×N 收益矩阵（N 个候选配置），
  对样本内 Sharpe 最高的那个算 DSR。

与研报的对应：华泰 AI22《回测过拟合的概率》即 PBO 方法论；AI19 的重采样思想
体现在 CSCV 的组合对称切分（比 train/test 单次对半更充分）。研报复现口径为
"骨架 + 差异分析"：本实现按原论文公式落地，未逐页比对华泰两篇的行文细节。

引用注意：PBO/DSR 只回答"选择过程是否过拟合"，**不**回答策略本身是否有效；
两者都依赖输入收益的代表性与 iid 程度（NW 校正未引入，CSCV 的块切分已部分
吸收时序相关）。
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

#: Euler–Mascheroni 常数（DSR 期望最大值的阶量展开）
EULER_GAMMA = 0.5772156649015329


def _avg_rank(x: np.ndarray) -> np.ndarray:
    """升序平均秩（并列取平均），返回 1..N 的浮点秩。"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _block_aggregates(M: np.ndarray, blocks: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    """逐块聚合 (count, Σx, Σx²)，供组合内 O(S/2) 快速算 mean/std。"""
    cnt = np.array([len(b) for b in blocks], dtype=float)
    s1 = np.stack([M[b].sum(axis=0) for b in blocks])
    s2 = np.stack([(M[b].astype(float) ** 2).sum(axis=0) for b in blocks])
    return cnt, s1, s2


def _perf_on(cnt, s1, s2, block_ids: list[int], metric: str) -> np.ndarray:
    """给定块集合上各配置的表现（均值或 Sharpe）。"""
    n = cnt[block_ids].sum()
    m = s1[block_ids].sum(axis=0)
    if metric == "mean":
        return m / n
    if metric != "sharpe":
        raise ValueError(f"未知 metric {metric!r}（mean / sharpe）")
    q = s2[block_ids].sum(axis=0)
    var = (q - m**2 / n) / max(n - 1.0, 1.0)
    sd = np.sqrt(np.maximum(var, 1e-18))
    return (m / n) / sd


def cscv_pbo(
    returns: pd.DataFrame | np.ndarray,
    *,
    n_partitions: int = 16,
    metric: str = "sharpe",
) -> dict[str, Any]:
    """CSCV 概率过拟合检验（PBO）。

    Args:
        returns: (T × N) 逐期收益矩阵——T 期、N 个**候选配置**（参数网格 /
            因子臂 / 模型臂等）。列名有意义时透传进结果。
        n_partitions: 块数 S（偶数）。默认 16 → C(16,8) = 12870 个对半组合，
            与原论文一致；T 小时降到 8 或 6。
        metric: 排名用表现度量，``sharpe``（逐期均值/标准差）或 ``mean``。
            两侧必须同度量（对称性所在）。

    Returns:
        dict：``pbo``（λ≤0 的组合占比）、``n_combinations``、``logits``（逐组合
        logit）、``logit_median`` / ``logit_iqr``、``omega_mean``（IS 最优者 OOS
        相对秩的均值，<0.5 即整体偏过拟合）、``n_trials`` / ``n_partitions`` /
        ``columns``（输入为 DataFrame 时保留列名）。

    References:
        Bailey, Borwein, López de Prado & Zhu (2017), "The Probability of
        Backtest Overfitting", J. Computational Finance 20(4).
    """
    if isinstance(returns, pd.DataFrame):
        columns = [str(c) for c in returns.columns]
        M = returns.to_numpy(dtype=float)
    else:
        columns = None
        M = np.asarray(returns, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("returns 须为 (T×N) 且 N ≥ 2 个候选配置")
    T, N = M.shape
    if n_partitions < 2 or n_partitions % 2 != 0:
        raise ValueError("n_partitions 须为偶数 ≥ 2")
    if T < n_partitions * 2:
        raise ValueError(f"T={T} 不足以切 {n_partitions} 块（每块至少 2 期）")
    if np.isnan(M).any():
        raise ValueError("returns 含 NaN：请先对齐期数或剔除含缺失的配置")

    blocks = [np.asarray(b) for b in np.array_split(np.arange(T), n_partitions)]
    cnt, s1, s2 = _block_aggregates(M, blocks)
    half = n_partitions // 2

    logits = np.empty(len(list(combinations(range(n_partitions), half))),
                      dtype=float)
    omegas = np.empty_like(logits)
    for k, phi in enumerate(combinations(range(n_partitions), half)):
        test_blocks = [b for b in range(n_partitions) if b not in phi]
        r_is = _perf_on(cnt, s1, s2, list(phi), metric)
        n_star = int(np.argmax(r_is))
        r_oos = _perf_on(cnt, s1, s2, test_blocks, metric)
        # 原论文：相对秩 ω̄ = 秩/(N+1)（严格落在 (0,1)，logit 有限）
        omega = float(_avg_rank(r_oos)[n_star] / (N + 1.0))
        omegas[k] = omega
        logits[k] = float(np.log(omega / (1.0 - omega)))

    return {
        "pbo": float(np.mean(logits <= 0.0)),
        "n_combinations": int(len(logits)),
        "logits": logits,
        "omegas": omegas,
        "logit_median": float(np.median(logits)),
        "logit_iqr": (float(np.percentile(logits, 25)),
                      float(np.percentile(logits, 75))),
        "omega_mean": float(np.mean(omegas)),
        "metric": metric,
        "n_partitions": n_partitions,
        "n_trials": int(N),
        "columns": columns,
    }


def trial_sharpes(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    """各候选配置的**逐期** Sharpe（均值/标准差，未年化；年化是线性缩放，
    对 DSR 的排序无影响，但 SR₀ 与观测 SR 必须同口径）。"""
    M = returns.to_numpy(dtype=float) if isinstance(returns, pd.DataFrame) \
        else np.asarray(returns, dtype=float)
    if M.ndim != 2 or M.shape[1] < 1:
        raise ValueError("returns 须为 (T×N)")
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)
    return mu / sd


def deflated_sharpe_ratio(
    sr: float,
    trial_srs: np.ndarray | pd.Series,
    t_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """缩水夏普比率 DSR（Deflated Sharpe Ratio, Bailey & López de Prado 2014）。

    在「N 个试验里期望最大 Sharpe」SR₀ 的零分布下，给出现测 ``sr`` 仍显著
    的概率（PSR 形式）：DSR = Φ( (sr − SR₀)·√(T−1) / √(1 − γ₃·sr + (γ₄−1)/4·sr²) )。

    Args:
        sr: 候选策略的**逐期** Sharpe（与 trial_srs 同频率、未年化）。
        trial_srs: 全部 N ≥ 2 个候选的逐期 Sharpe（含候选自身）——其方差
            决定 SR₀（试验越多、方差越大，门槛越高）。
        t_obs: 收益观测期数 T。
        skew: 收益偏度（样本）。
        kurt: 收益峰度（**Pearson 口径**，正态 = 3，非超额峰度）。

    Returns:
        DSR ∈ (0, 1)。经验法则 DSR ≥ 0.95 才认为"挑出来的最优"在计入选择偏差后
        仍显著。分母 ≤ 0 时（极端负偏 + 高 SR 的病态组合）直接给 1.0/0.0
        （只在 sr 相对 SR₀ 同侧时发生）。

    References:
        Bailey & López de Prado (2014), "The Deflated Sharpe Ratio",
        Journal of Portfolio Management 40(5).
    """
    from scipy.stats import norm  # 惰性导入（sklearn 已带 scipy，但保持模块轻量）

    srs = np.asarray(trial_srs, dtype=float)
    srs = srs[np.isfinite(srs)]
    n = len(srs)
    if n < 2:
        raise ValueError("trial_srs 须含 ≥2 个候选（N=1 无选择偏差可言）")
    if t_obs < 2:
        raise ValueError("t_obs 须 ≥ 2")
    var_trials = float(np.var(srs, ddof=1))
    # 期望最大值：E[max SR] ≈ √V[SR]·[(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e))]
    sr0 = np.sqrt(var_trials) * (
        (1.0 - EULER_GAMMA) * float(norm.ppf(1.0 - 1.0 / n))
        + EULER_GAMMA * float(norm.ppf(1.0 - 1.0 / (n * np.e)))
    )
    denom = 1.0 - float(skew) * float(sr) + (float(kurt) - 1.0) / 4.0 * float(sr) ** 2
    if denom <= 0:
        return 1.0 if sr > sr0 else 0.0
    z = (float(sr) - sr0) * np.sqrt(t_obs - 1.0) / np.sqrt(denom)
    return float(norm.cdf(z))


def deflate_best(
    returns: pd.DataFrame | np.ndarray,
    *,
    metric: str = "sharpe",
) -> dict[str, Any]:
    """一步到位：对 (T×N) 候选收益矩阵，挑样本内最优配置并算其 DSR。

    返回 dict：``best_index`` / ``best_column`` / ``best_sr``（逐期）/
    ``dsr`` / ``n_trials`` / ``t_obs`` / ``skew`` / ``kurt`` /
    ``trial_srs``。典型读法：``dsr >= 0.95`` 且 PBO 低 → 选择过程可信。
    """
    if isinstance(returns, pd.DataFrame):
        columns = [str(c) for c in returns.columns]
        M = returns.to_numpy(dtype=float)
    else:
        columns = None
        M = np.asarray(returns, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("returns 须为 (T×N) 且 N ≥ 2")

    srs = trial_sharpes(M)
    if not np.isfinite(srs).all():
        raise ValueError("存在零方差/无效配置（Sharpe 非有限），请先剔除")
    k = int(np.argmax(srs))
    col = M[:, k]
    centered = col - col.mean()
    var = float(np.var(col, ddof=1))
    skew = float(np.mean(centered**3) / var**1.5) if var > 0 else 0.0
    kurt = float(np.mean(centered**4) / var**2) if var > 0 else 3.0
    return {
        "best_index": k,
        "best_column": columns[k] if columns else None,
        "best_sr": float(srs[k]),
        "dsr": deflated_sharpe_ratio(float(srs[k]), srs, len(M),
                                     skew=skew, kurt=kurt),
        "n_trials": int(M.shape[1]),
        "t_obs": int(len(M)),
        "skew": skew,
        "kurt": kurt,
        "trial_srs": srs,
    }
