"""因子拥挤度 / 相关性监控（P3，2026-08-20）。

库级宏观指标：对库内全部因子（GP + 模型）的 IC 序列做相关性分析，
衡量"分散度是否真实"。真实里 23 个 GP 因子同族、19 个同时 break，
正是同质化的写照——单项因子各自"健康"，但彼此高度相关 => 合并后
仍押同一方向，分散度是假的。

指标：
- corr_mean: IC 序列两两 Spearman 相关均值（上三角）
- pc1_share: 相关矩阵最大特征值占比（第一主成分解释度）
- corr_n   : 参与计算的因子数

该矩阵同时是后续 alpha 收敛（多因子合并）的前置数据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from monitoring.metrics import MonitorMetrics

MIN_PERIODS = 30  # 至少 30 个重叠交易日才算相关，否则丢弃该对


def compute_crowding(
    ic_history: dict[str, pd.Series],
    name: str = "cluster:crowding",
) -> MonitorMetrics | None:
    """把各因子 IC 序列折叠成一条库级拥挤度 MonitorMetrics(category="crowd")。"""
    m = MonitorMetrics(name=name, category="crowd")
    pairs = {k: v for k, v in ic_history.items() if len(v) >= MIN_PERIODS}
    if len(pairs) < 2:
        return None
    df = pd.DataFrame(pairs)  # 自动按 index 对齐
    corr = df.corr(method="spearman", min_periods=MIN_PERIODS)
    m.corr_n = int(corr.shape[0])

    off = corr.values[np.triu_indices_from(corr.values, k=1)]
    off = off[~np.isnan(off)]
    if len(off) == 0:
        return None
    m.corr_mean_full = float(np.clip(np.nanmean(off), -1.0, 1.0))

    # 特征值分析：对角线恒为 1、缺对置 0，再做 eig（顺序很重要，否则对角被污染）
    mat = corr.values.copy()
    np.fill_diagonal(mat, 1.0)
    mat = np.nan_to_num(mat, nan=0.0)
    eigvals = np.linalg.eigvalsh(mat)
    total = max(float(eigvals.sum()), 1e-12)
    m.pc1_share_full = float(eigvals[-1] / total)
    return m