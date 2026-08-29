"""
稳健统计推断（Robust Inference）
===============================

因子研究里最常见的"伪显著"来源：**IC 序列 / 因子溢价序列在时间上强自相关**，
普通 t 检验假设观测 i.i.d.，会系统性低估标准误、夸大 t 值。业界公认的做法是
用 **Newey-West (1987) HAC 标准误** 做自相关-异方差稳健推断：

    V_NW = (X'X)^{-1} Ω (X'X)^{-1},   Ω = Γ_0 + Σ_{j=1}^{L} w_j (Γ_j + Γ_j')

- Bartlett 核权重: w_j = 1 - j/(L+1)（保证半正定）
- 滞后截断 L 用 Andrews (1991) 经验规则: L = floor(4*(T/100)^(2/9))
  （备选 L = floor(T^(1/3))，R sandwich 包默认 plug-in 规则）
- 系数估计不变，只改标准误与 t/p 值

本模块提供三个入口（全部纯 numpy，无第三方依赖）：

- ``nw_tstat``        : 单序列均值检验（IC 序列 / Fama-MacBeth β 序列）的 NW t 统计量
- ``ols_newey_west``  : 通用 OLS + NW HAC 标准误（α/β 回归、多因子回归）
- ``auto_lag``        : 滞后截断的自适应选择规则

约定：所有函数对 NaN 按"缺失观测剔除"处理；样本太少时返回 NaN/0 而非报错。

真源历史：2026-08-29 自 research/robust_stats.py 原样下沉（research 层保留
re-export 转出口）。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

__all__ = ["auto_lag", "nw_tstat", "ols_newey_west"]


def auto_lag(n: int, method: str = "andrews") -> int:
    """自适应选择 Newey-West 滞后截断 L。

    Args:
        n: 样本量 T。
        method:
            - 'andrews'  : L = floor(4*(T/100)^(2/9))（Andrews 1991，业界默认）
            - 'cuberoot' : L = floor(T^(1/3))（常见备选）
    Returns:
        整数滞后阶数，至少 1（T 很小时取 0 则无校正意义，统一回 1）。
    """
    n = int(n)
    if n <= 2:
        return 0
    if method == "cuberoot":
        return max(1, int(np.floor(n ** (1 / 3) + 1e-9)))
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2 / 9) + 1e-9)))


def _clean_series(series: Sequence[float]) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    return s[~np.isnan(s)]


def nw_tstat(
    series: Sequence[float],
    lag: int | None = None,
    method: str = "andrews",
) -> tuple[float, float, int]:
    """单序列均值的 Newey-West 校正 t 统计量。

    用于 IC 序列 / Fama-MacBeth 系数序列的显著性检验：
    检验 H0: mean(series) = 0，对序列自相关与异方差稳健。

    Args:
        series: 一维数值序列（IC 或 β_t）。NaN 自动剔除。
        lag:   滞后截断。None → 按 ``method`` 自动选择。
        method: auto_lag 的规则（'andrews' | 'cuberoot'）。
    Returns:
        (t_stat, se, lag_used)：NW t 值、NW 标准误（均值估计量的）、实际使用滞后。
        样本不足时返回 (0.0, 0.0, lag)。
    """
    s = _clean_series(series)
    n = len(s)
    if n < 2:
        return (0.0, 0.0, 0)
    if lag is None:
        lag = auto_lag(n, method)
    lag = int(min(max(lag, 0), n - 2))

    mean = float(s.mean())
    u = s - mean
    gamma0 = float(np.mean(u * u))
    omega = gamma0
    for j in range(1, lag + 1):
        g = float(np.mean(u[j:] * u[:-j]))
        w = 1.0 - j / (lag + 1.0)
        omega += 2.0 * w * g
    se = float(np.sqrt(max(omega, 0.0) / n))
    if se <= 0:
        t = 0.0
    else:
        t = mean / se
    return (t, se, lag)


def ols_newey_west(
    X: np.ndarray,
    y: np.ndarray,
    lag: int | None = None,
    method: str = "andrews",
    df_adjust: bool = True,
) -> dict:
    """OLS 回归 + Newey-West HAC 标准误（通用）。

    用于 α/β 分解（CAPM/多因子回归）：y = Xβ + u，对残差的自相关与
    异方差稳健地推断系数显著性。系数估计与普通 OLS 相同，仅标准误不同。

    Args:
        X: (T, k) 设计矩阵（含截距列请自行拼接 np.column_stack([ones, ...])）。
        y: (T,) 因变量。
        lag: 滞后截断，None → 自动（Andrews 规则）。
        method: auto_lag 规则。
        df_adjust: True 时 p 值用 t 分布（df=T-k-1），False 用正态近似。
    Returns:
        dict: beta / se_nw / t_nw / p_nw / resid / n / lag_used。
        同时附 OLS 版 se_ols / t_ols / p_ols 便于对比（自相关越强，两者差异越大）。
    """
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa[:, None]
    valid = ~np.isnan(ya)
    if Xa.ndim == 2:
        valid &= ~np.isnan(Xa).any(axis=1)
    Xa, ya = Xa[valid], ya[valid]
    n, k = Xa.shape
    if n < k + 2:
        return {"beta": np.full(k, np.nan), "se_nw": np.full(k, np.nan),
                "t_nw": np.full(k, np.nan), "p_nw": np.full(k, np.nan),
                "resid": np.array([]), "n": n, "lag_used": 0,
                "se_ols": np.full(k, np.nan), "t_ols": np.full(k, np.nan),
                "p_ols": np.full(k, np.nan)}

    # OLS 系数（不变）
    XtX = Xa.T @ Xa
    Xty = Xa.T @ ya
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(XtX, Xty, rcond=None)
    resid = ya - Xa @ beta

    if lag is None:
        lag = auto_lag(n, method)
    lag = int(min(max(lag, 0), n - 2))

    # 普通 OLS 方差（供对比）
    sigma2 = float(resid @ resid / (n - k))
    v_ols = np.linalg.inv(XtX) * sigma2
    se_ols = np.sqrt(np.maximum(np.diag(v_ols), 0.0))

    # NW meat：Ω = Σ u_t^2 x_t x_t' + Σ_j w_j Σ_t u_t u_{t-j} (x_t x_{t-j}' + x_{t-j} x_t')
    omega = np.zeros((k, k))
    for t in range(n):
        xt = Xa[t][:, None]
        omega += resid[t] ** 2 * (xt @ xt.T)
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)
        for t in range(j, n):
            xt = Xa[t][:, None]
            xtj = Xa[t - j][:, None]
            c = resid[t] * resid[t - j]
            omega += w * c * (xt @ xtj.T + xtj @ xt.T)

    bread = np.linalg.inv(XtX)
    v_nw = bread @ omega @ bread
    se_nw = np.sqrt(np.maximum(np.diag(v_nw), 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_nw = beta / se_nw
        t_ols = beta / np.where(se_ols > 0, se_ols, np.nan)
    df = n - k - 1 if df_adjust else np.inf
    p_nw = 2.0 * (1.0 - stats.t.cdf(np.abs(t_nw), df=df))
    p_ols = 2.0 * (1.0 - stats.t.cdf(np.abs(t_ols), df=df))

    return {
        "beta": beta, "se_nw": se_nw, "t_nw": t_nw, "p_nw": p_nw,
        "se_ols": se_ols, "t_ols": t_ols, "p_ols": p_ols,
        "resid": resid, "n": n, "lag_used": lag,
    }
