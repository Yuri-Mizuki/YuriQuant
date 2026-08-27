"""统一交叉验证纪律 —— 挖因子 / 评估因子共用的切分基础设施。

三层切分体系
============

1. **三段式（train / valid / test）** —— 挖因子层
   - ``split_three_periods``：按日期边界切 train/valid/test，test 冻结。
   - 所有挖因子方法（GP / GFlowNet / 暴力枚举）统一调用，保证公平对比。

2. **Purged K-Fold** —— 筛选 / 选参层
   - ``purged_kfold``：时序折，测试折可在中间（非仅末尾），训练集取两侧，
     边界处 purge 掉标签时间区间重叠的样本 + embargo 隔离带。
   - 比 ``_time_folds``（expanding，测试折只在末尾）更灵活：研究评估时
     每个时段都能当一次测试段，充分利用数据。

3. **CPCV（Combinatorial Purged Cross-Validation）** —— 选型 / 最终评估层
   - ``cpcv``：把全历史切 N 组，取 k 组当测试的所有组合 → φ=C(N,k) 条 OOS 路径。
   - 每条路径独立 purge + embargo，产出 IC 分布而非单点。
   - 用于判断算法间相对优劣是实力还是运气（选型决策依据）。

设计原则
--------
- 所有切分按【交易日边界】操作，绝不把同一天劈开（复用 ``_time_folds``
  的日期边界纪律）。
- embargo_days 应 >= 标签 horizon，防标签前视泄漏。
- 返回值统一为 ``list[Fold]``，``Fold = (train_days, test_days)``，
  ``train_days / test_days`` 均为 ``pd.DatetimeIndex``（升序）。
"""
from __future__ import annotations

from itertools import combinations
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

__all__ = [
    "Fold", "CPCVPath",
    "split_three_periods",
    "purged_kfold",
    "cpcv",
]


class Fold(NamedTuple):
    """单折切分结果。

    Attributes:
        train_days: 训练段交易日（升序 DatetimeIndex）。
        test_days:  测试段交易日（升序 DatetimeIndex）。
    """
    train_days: pd.DatetimeIndex
    test_days: pd.DatetimeIndex


@dataclass
class CPCVPath:
    """CPCV 单条路径。

    Attributes:
        path_id:    路径编号（0-based）。
        test_groups: 本路径的测试组索引列表（如 [0, 3]）。
        train_days: 训练段交易日（purge + embargo 后）。
        test_days:  测试段交易日（各组拼接，升序）。
    """
    path_id: int
    test_groups: list[int]
    train_days: pd.DatetimeIndex
    test_days: pd.DatetimeIndex


# ---------------------------------------------------------------------------
# 1. 三段式 —— 挖因子层统一接口
# ---------------------------------------------------------------------------
def split_three_periods(
    index: pd.DatetimeIndex,
    train: tuple[int, int] = (20220101, 20231231),
    valid: tuple[int, int] = (20240101, 20241231),
    test: tuple[int, int] = (20250101, 20251231),
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """按日期边界切 train / valid / test 三段（test 冻结）。

    与 ``gp_tune_budget.split_dates`` 同口径，但返回 DatetimeIndex
    并统一到本模块，供所有挖因子方法调用。

    Args:
        index: 全量交易日索引（升序）。
        train: (begin, end) 日期整数，如 (20220101, 20231231)。
        valid: (begin, end) 日期整数。
        test:  (begin, end) 日期整数。

    Returns:
        (train_days, valid_days, test_days) —— 三个升序 DatetimeIndex。
    """
    def rng(b: int, e: int) -> pd.DatetimeIndex:
        return index[(index >= pd.Timestamp(str(b)))
                     & (index <= pd.Timestamp(str(e)))]

    return rng(*train), rng(*valid), rng(*test)


# ---------------------------------------------------------------------------
# 2. Purged K-Fold —— 筛选 / 选参层
# ---------------------------------------------------------------------------
def purged_kfold(
    index: pd.DatetimeIndex,
    n_splits: int = 5,
    embargo_days: int = 5,
) -> list[Fold]:
    """时序 Purged K-Fold：测试折可在中间，训练集取两侧 + purge + embargo。

    与 ``factor.synthesis._time_folds`` 的区别：
    - ``_time_folds``：expanding window，测试折只在末尾（forward-chaining）。
    - 本函数：每个折的测试段可在任意位置，训练集取其**两侧**剩余日期，
      边界处 purge 掉标签重叠 + embargo 隔离。

    适用于研究评估（如 valid 段内做 5 折 CV 选超参），每段数据都能当一次测试。

    Args:
        index: 全量交易日索引（升序）。
        n_splits: 折数（>=2）。
        embargo_days: 测试折前后各剔除的天数（防标签前视 + 特征自相关）。

    Returns:
        ``list[Fold]``，长度 ``n_splits``。
    """
    days = pd.DatetimeIndex(sorted(index.unique()))
    n = len(days)
    if n < n_splits:
        raise ValueError(f"日期数 {n} < n_splits {n_splits}")

    # 按交易日边界均匀切 n_splits 段
    edges = np.linspace(0, n, n_splits + 1).astype(int)
    folds: list[Fold] = []
    for s in range(n_splits):
        d0 = max(0, int(edges[s]))
        d1 = min(n, int(edges[s + 1]))
        if d1 <= d0:
            continue
        test_days = pd.DatetimeIndex(days[d0:d1])

        # 训练集 = 全部日期 - 测试段 - embargo 带
        purge_start = max(0, d0 - embargo_days)
        purge_end = min(n, d1 + embargo_days)
        purge_set = set(days[purge_start:purge_end])
        train_days = pd.DatetimeIndex(
            [d for d in days if d not in purge_set]
        )
        folds.append(Fold(train_days, test_days))
    return folds


# ---------------------------------------------------------------------------
# 3. CPCV —— 选型 / 最终评估层
# ---------------------------------------------------------------------------
def cpcv(
    index: pd.DatetimeIndex,
    n_groups: int = 6,
    k: int = 2,
    embargo_days: int = 5,
) -> list[CPCVPath]:
    """Combinatorial Purged Cross-Validation。

    把全历史切 ``n_groups`` 个连续组，取 ``k`` 组当测试集的所有组合
    φ = C(n_groups, k) 条路径。每条路径：
    - 测试段 = 这 k 个组的日期拼接。
    - 训练段 = 其余组的日期，减去与测试组相邻的 embargo 带。

    经典配置 n_groups=6, k=2 → 15 条路径。

    Args:
        index: 全量交易日索引（升序）。
        n_groups: 连续组数（>=2）。
        k: 每条路径的测试组数（>=1, <=n_groups-1）。
        embargo_days: 测试组边界处剔除的训练天数。

    Returns:
        ``list[CPCVPath]``，长度 C(n_groups, k)。
    """
    from math import comb

    days = pd.DatetimeIndex(sorted(index.unique()))
    n = len(days)
    if n_groups < 2:
        raise ValueError(f"n_groups >= 2, got {n_groups}")
    if k < 1 or k >= n_groups:
        raise ValueError(f"need 1 <= k < n_groups, got k={k}, n_groups={n_groups}")
    if n < n_groups:
        raise ValueError(f"日期数 {n} < n_groups {n_groups}")

    # 切 n_groups 个连续组（按交易日边界）
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    groups: list[pd.DatetimeIndex] = []
    for s in range(n_groups):
        d0 = max(0, int(edges[s]))
        d1 = min(n, int(edges[s + 1]))
        if d1 <= d0:
            raise ValueError(
                f"组 {s} 为空（日期数 {n} 不足以切 {n_groups} 组）")
        groups.append(pd.DatetimeIndex(days[d0:d1]))

    total_paths = comb(n_groups, k)
    paths: list[CPCVPath] = []
    for pid, combo in enumerate(combinations(range(n_groups), k)):
        test_set = set()
        for gi in combo:
            test_set.update(groups[gi])
        test_days = pd.DatetimeIndex(sorted(test_set))

        # 训练集 = 全部日期 - 测试段 - 每个测试组边界的 embargo 带
        purge_set = set(test_days)
        for gi in combo:
            g = groups[gi]
            # 找该组在 days 中的起止位置
            g_start_idx = days.get_loc(g[0])
            g_end_idx = days.get_loc(g[-1])
            # 前后各 embargo_days 天
            emb_start = max(0, g_start_idx - embargo_days)
            emb_end = min(n, g_end_idx + embargo_days + 1)
            purge_set.update(days[emb_start:emb_end])

        train_days = pd.DatetimeIndex(
            sorted(set(days) - purge_set)
        )
        paths.append(CPCVPath(
            path_id=pid,
            test_groups=list(combo),
            train_days=train_days,
            test_days=test_days,
        ))
    return paths


# ---------------------------------------------------------------------------
# 辅助：对任意切分跑评估器（供 CPCV 多路径评估脚本调用）
# ---------------------------------------------------------------------------
def run_cv_paths(
    predictor_cls,
    features: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    paths: list[Fold] | list[CPCVPath],
    min_train_days: int = 120,
    **predictor_params,
) -> pd.DataFrame:
    """对一组路径（Fold 或 CPCVPath）逐路径 fit→predict，拼接 OOS 面板。

    返回 date×code 面板：每条路径的 test_days 段填入 OOS 预测，
    多路径重叠区取最后一条路径的结果（CPCV 评估时通常按路径分别算 IC
    再聚合，这里拼接面板仅用于快速查看）。

    对于 CPCV，推荐调用方按路径分别计算 IC 再看分布，而非拼接后算全局 IC。
    """
    all_days = sorted(set().union(*[set(p.test_days) for p in paths]))
    cols = labels.columns
    out = pd.DataFrame(np.nan, index=pd.DatetimeIndex(all_days), columns=cols)

    for p in paths:
        train_days = p.train_days
        test_days = p.test_days
        if len(train_days) < min_train_days or len(test_days) == 0:
            continue
        model = predictor_cls(**predictor_params)
        model.fit(
            {k: v.loc[train_days] for k, v in features.items()},
            labels.loc[train_days],
        )
        pred = model.predict(
            {k: v.loc[test_days] for k, v in features.items()}
        )
        out.loc[test_days] = pred.reindex(index=test_days, columns=cols)
    return out
