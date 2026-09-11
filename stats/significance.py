"""
显著性判定（p 值与多重检验校正）
==================================

与 ``stats.robust_stats`` 的分工：

- ``stats.robust_stats`` 负责 **估计** —— Newey-West HAC 标准误与 t 统计量；
- 本模块负责 **判定** —— 由 t 得到双侧 p 值，以及由一批 p 值做
  **Benjamini-Hochberg FDR** 多重检验校正。

合起来构成完整的显著性口径。三个入口：

- ``t_pvalue``            : t 统计量 → 双侧 p 值（可给自由度，缺省正态近似）
- ``benjamini_hochberg``  : p 值数组 + 目标 q → 每个假设是否拒绝（bool 数组）
- ``mean_inference``      : 单序列均值检验的**一步式**入口，同时给出
                            OLS t/p 与 NW t/p（因子 IC 序列 / 因子溢价序列的标配）

真源历史（2026-09-10 收敛，第二批口径统一）
--------------------------------------------
此前「同一个判定」散在三处、口径无从对照：

- ``factor/mining.py`` 私有函数 ``_benjamini_hochberg`` + worker/串行两条路径
  各写一遍 ``p = 2*(1-t.cdf(|t|, n-1))``（批量挖掘用 BH-FDR，q=0.05）；
- ``research/factor_library.py`` **完全不算 p 值**，直接用 ``|t_nw| > 2.0``
  判显著（单因子入库，无多重检验校正）；
- ``research/factor_analysis.py`` / ``research/attribution.py`` /
  ``model/evaluation.py`` 各自内联 p 值公式。

统一的**表述方式**是：

    "统一"= 一份实现 + 显式声明，不是把所有地方压成同一个阈值。

两处的**多重检验族（family）本来就不同**，因此保留两种语义、但都走本模块：

- **批量挖掘**（``factor/mining.evaluate_candidates``）：一次评估成百上千个候选，
  族 = 这一批候选 → **BH-FDR(q)**，控制"判显著者里假发现的比例"。
- **单因子入库**（``research.factor_library.register``）：一次只登记一个因子，
  调用内没有可言的族 → 存**原始 NW 显著性**并记录 ``p_value_nw``；若要在
  "整个库"这个族上做校正，走 ``FactorLibrary.significance_table()`` /
  ``load_significant_features(correction="fdr")``（族 = 库内全部因子），
  而不是在登记时假装存在一个批次。

判据与数值均为纯 numpy/scipy，无第三方依赖。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

from stats.robust_stats import nw_tstat

__all__ = ["t_pvalue", "benjamini_hochberg", "mean_inference"]


def t_pvalue(t: float | np.ndarray, df: float | None = None) -> float | np.ndarray:
    """t 统计量 → **双侧** p 值。

    Args:
        t: t 统计量（标量或数组）。负号自动取绝对值（双侧检验）。
        df: 自由度（标量，或与 ``t`` 同形状的数组——如按行不同的 ``n - 1``）。
            ``None`` 或非有限 → 用标准正态近似；``<= 0`` → NaN。
            小样本（IC 序列常为几百个交易日）建议传 ``n - 1``。

    Returns:
        与输入同形状的 p 值；输入为 NaN 或 df <= 0 时返回 NaN。
    """
    ta = np.asarray(t, dtype=float)
    a = np.abs(ta)
    if df is None or (np.ndim(df) == 0 and not np.isfinite(df)):
        # 标量 df 非有限（或未给）→ 标准正态近似
        with np.errstate(invalid="ignore"):
            p = 2.0 * stats.norm.sf(a)
    elif np.ndim(df) == 0:
        if df <= 0:
            p = np.full(a.shape, np.nan)
        else:
            with np.errstate(invalid="ignore"):
                p = 2.0 * stats.t.sf(a, df=df)
    else:
        # 数组 df（逐元素自由度，如 registry 补算时的 n_dates - 1）：语义与标量
        # 路径一致 —— 有限且 >0 走 t 分布；有限但 <=0 → NaN；非有限 → 正态近似。
        dfa = np.broadcast_to(np.asarray(df, dtype=float), a.shape).astype(float)
        pos = dfa > 0
        fin = np.isfinite(dfa)
        with np.errstate(invalid="ignore"):
            p_pos = 2.0 * stats.t.sf(a, df=np.where(pos, dfa, 1.0))
            p_norm = 2.0 * stats.norm.sf(a)
        p = np.where(pos, p_pos, np.where(fin, np.nan, p_norm))
    p = np.where(np.isnan(a), np.nan, p)
    return float(p) if p.ndim == 0 else p


def benjamini_hochberg(pvalues: Sequence[float], q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR 多重检验校正。

    给定一族假设的 p 值，返回**每个假设是否拒绝原假设**（即是否显著）。
    控制的是被拒绝者中假发现的期望比例不超过 ``q``。

    实现（标准 BH 步进）：

        k* = max{ k : p_(k) <= k * q / m },   m = 族内有效假设数

    拒绝集合 = 最小的 k* 个 p 值对应的假设（单调闭包）。

    Args:
        pvalues: 一族 p 值（一维）。**NaN 视为不参与检验**，返回 False，
            且不计入 m —— 缺失观测不该让整族阈值变严。
        q: 目标 FDR 水平（默认 0.05）。

    Returns:
        与 ``pvalues`` 等长的 bool 数组，True = 该假设显著。
        空输入返回空数组。
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    out = np.zeros(p.shape, dtype=bool)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv, kind="stable")
    sorted_p = pv[order]
    # k*: 最大的 k 使 p_(k) <= k*q/m
    k_max = 0
    for i in range(m):
        if sorted_p[i] <= (i + 1) * q / m:
            k_max = i + 1
    if k_max > 0:
        keep = order[:k_max]
        idx = np.flatnonzero(ok)[keep]
        out[idx] = True
    return out


def mean_inference(
    series: Sequence[float],
    robust: bool = True,
    method: str = "andrews",
) -> dict:
    """**单序列均值检验**的一步式入口：H0 为 mean(series) = 0。

    适用对象是因子研究里的时间序列：日度 IC 序列、Fama-MacBeth 的 β_t 序列、
    组合超额收益序列等。一次调用同时给出

    - 朴素 OLS 口径：``t_stat`` / ``p_value``（假设观测 i.i.d.）；
    - Newey-West 稳健口径：``t_stat_nw`` / ``p_value_nw`` / ``lag``
      （对自相关与异方差稳健，强自相关 IC 序列的标配）。

    两者并排输出是有意的：自相关越强，OLS t 越虚高，两列差距就是"伪显著"的
    直接证据。**判显著请用 NW 那一列。**

    Args:
        series: 一维数值序列。NaN 自动剔除。
        robust: False 时不算 NW（省一次 O(N·L) 计算），NW 列返回 NaN。
        method: NW 滞后截断规则（见 ``robust_stats.auto_lag``）。

    Returns:
        dict:
        - ``n``        有效观测数
        - ``mean`` / ``std``  样本均值 / 样本标准差（**ddof=1**，与 pandas 默认一致）
        - ``t_stat`` / ``p_value``      OLS 口径
        - ``t_stat_nw`` / ``p_value_nw`` NW 口径；``lag`` 为实际使用的滞后阶
        - ``se_nw``    NW 标准误（均值估计量的）
        样本不足（n < 2）时各统计量返回 NaN / 0，不抛异常。
    """
    s = np.asarray(series, dtype=float).ravel()
    s = s[~np.isnan(s)]
    n = int(s.size)
    nan = float("nan")
    if n < 2:
        return {"n": n, "mean": float(s.mean()) if n else nan, "std": nan,
                "t_stat": nan, "p_value": nan,
                "t_stat_nw": nan, "p_value_nw": nan, "se_nw": nan, "lag": 0}

    m = float(s.mean())
    sd = float(np.std(s, ddof=1))          # ddof=1：与 pandas Series.std() 一致
    t_ols = m / (sd / np.sqrt(n)) if sd > 0 else 0.0
    p_ols = t_pvalue(t_ols, df=n - 1)

    if robust:
        t_nw, se_nw, lag = nw_tstat(s, method=method)
        p_nw = t_pvalue(t_nw, df=n - 1)
    else:
        t_nw, se_nw, lag, p_nw = nan, nan, 0, nan

    return {"n": n, "mean": m, "std": sd,
            "t_stat": float(t_ols), "p_value": float(p_ols),
            "t_stat_nw": float(t_nw), "p_value_nw": float(p_nw),
            "se_nw": float(se_nw), "lag": int(lag)}
