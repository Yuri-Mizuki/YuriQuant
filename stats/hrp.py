"""HRP（López de Prado 2016）：层次聚类 + 递归二分 + 逆方差。**唯一真源**。

从 ``optimize.solver`` 下沉（2026-11-25）：``strategy.etf_rotation`` 也要 HRP，
而分层规则禁止 ``strategy`` import ``optimize``。算法本身只依赖
numpy/pandas/scipy，属统计原语。

历史坑：曾有两份同名实现，``linkage_method`` 一个 ``single`` 一个 ``ward``、
簇方差一个用协方差子块求和、一个用 IVP 加权 ⇒ 同一协方差矩阵最大权重差 0.18，
两族数字不可互引。现统一为原文口径（single + IVP）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def quasi_diag(link: np.ndarray) -> list[int]:
    """把层次聚类 linkage 的叶节点重排为「准对角」顺序（聚类块连续）。"""
    link = link.astype(int)
    num_items = int(link[-1, 3])
    sort_ix = [int(link[-1, 0]), int(link[-1, 1])]
    while max(sort_ix) >= num_items:
        new_ix: list[int] = []
        for ix in sort_ix:
            if ix >= num_items:
                j = ix - num_items
                new_ix += [int(link[j, 0]), int(link[j, 1])]
            else:
                new_ix.append(ix)
        sort_ix = new_ix
    return sort_ix


def cluster_order(cov: np.ndarray, linkage_method: str = "single") -> list[int]:
    """相关距离 → 层次聚类 → 准对角叶子顺序（HRP 的聚类骨架）。

    ``hrp_weights`` 与 ``strategy.etf_rotation._hrp_cluster_equal`` 共用
    （后者只换掉权重分配，不改聚类）。
    """
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform

    n = cov.shape[0]
    if n <= 1:
        return list(range(n))
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    if np.any(std <= 0):
        return list(range(n))
    corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))  # 相关距离
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    link = sch.linkage(squareform(dist, checks=False), method=linkage_method)
    return quasi_diag(link)


def cluster_variance(sub_cov: np.ndarray, pos: list[int]) -> float:
    """簇内方差：用逆方差分配（IVP）加权。"""
    inv_diag = 1.0 / np.maximum(np.diag(sub_cov), 1e-12)
    ivp = inv_diag[pos] / inv_diag.sum()
    return float(ivp @ sub_cov[np.ix_(pos, pos)] @ ivp)


def _recursive(cov: np.ndarray, idx: list[int]) -> dict[int, float]:
    """递归二分：每层按左右簇方差比例分配权重，叶子返回。"""
    if len(idx) <= 1:
        return {idx[0]: 1.0}
    mid = len(idx) // 2
    left, right = idx[:mid], idx[mid:]
    sub = cov[np.ix_(idx, idx)]
    lv = cluster_variance(sub, list(range(mid)))
    rv = cluster_variance(sub, list(range(mid, len(idx))))
    alpha = 1.0 - lv / (lv + rv) if (lv + rv) > 0 else 0.5
    out: dict[int, float] = {}
    for k, v in _recursive(cov, left).items():
        out[k] = alpha * v
    for k, v in _recursive(cov, right).items():
        out[k] = (1.0 - alpha) * v
    return out


def hrp_weights(cov: np.ndarray, codes=None,
                linkage_method: str = "single") -> pd.Series:
    """HRP 权重：全程不需求 Σ⁻¹（病态协方差免疫），天然非负、和为 1。

    缺点：不用收益信号（纯风险结构）。
    """
    n = cov.shape[0]
    index = codes if codes is not None else range(n)
    if n == 1:
        return pd.Series([1.0], index=index)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    if np.any(std <= 0):
        return pd.Series(np.full(n, 1.0 / n), index=index)
    out = np.zeros(n)
    for pos, wgt in _recursive(cov, cluster_order(cov, linkage_method)).items():
        out[pos] = wgt
    s = out.sum()
    if s > 0:
        out /= s
    return pd.Series(out, index=index)
