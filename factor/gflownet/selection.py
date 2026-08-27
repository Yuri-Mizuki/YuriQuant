"""
低相关性因子筛选（Phase 1）
===========================

研报对齐（系列之二十二 §2.3 末尾）：**「最终基于测试集进行因子筛选时，我们会
限定入选因子与已入选因子 spearman 相关性低于 0.4」**。

实现：按奖励 R 降序贪心入选 —— 新因子与**任一已入选因子**的面板 spearman 相关
> 0.4 则跳过；全部通过才入选。相关度量 = 面板 flatten 向量（date×code）的
spearman（成对剔除 NaN），与 :func:`factor.gflownet.tb.evaluate_samples` 一致。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from factor.formula import formula_builder

__all__ = ["select_low_corr", "panel_flat_corr"]


def panel_flat_corr(a: pd.DataFrame, b: pd.DataFrame,
                    min_overlap: int = 200) -> float:
    """两因子面板的 spearman 相关（flatten 后成对有效样本，NaN 返回 0）。"""
    av, bv = a.to_numpy().flatten(), b.to_numpy().flatten()
    m = ~(np.isnan(av) | np.isnan(bv))
    if m.sum() < min_overlap:
        return 0.0
    ra = np.argsort(np.argsort(av[m]))
    rb = np.argsort(np.argsort(bv[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def select_low_corr(samples: list[tuple[str, float]],
                    panel: dict[str, pd.DataFrame], features: list[str],
                    threshold: float = 0.4,
                    evaluator: Optional[Callable] = None,
                    min_overlap: int = 200,
                    progress: bool = False) -> list[tuple[str, float]]:
    """按 R 降序贪心入选，入选因子与已入选因子相关 > ``threshold`` 则跳过。

    Args:
        samples: [(formula, R)]，调用方保证已按 R 降序（``sample_formulas`` 输出即降序）。
        panel: 特征面板 dict。
        features: 特征名列表（formula_builder 用）。
        threshold: 相关性上限（研报 0.4）。
        evaluator: 可注入 mock 求值器（formula -> 面板）。
        progress: 是否打印筛选进度。

    Returns:
        入选的 [(formula, R)]（保持 R 降序）。
    """
    selected: list[tuple[str, float, pd.DataFrame]] = []
    for i, (formula, r) in enumerate(samples):
        if evaluator is not None:
            fp = evaluator(formula)
        else:
            fp = formula_builder(formula, features=features)(panel)
        if fp is None or fp.empty:
            continue
        ok = True
        for _, _, prev_fp in selected:
            if abs(panel_flat_corr(fp, prev_fp, min_overlap=min_overlap)) > threshold:
                ok = False
                break
        if ok:
            selected.append((formula, r, fp))
        if progress and (i + 1) % 50 == 0:
            print(f"  筛选 {i + 1}/{len(samples)} 已入选 {len(selected)}", flush=True)
    return [(f, r) for f, r, _ in selected]
