"""
低相关性因子筛选（Phase 1）
===========================

研报对齐（系列之二十二 §2.3 末尾 + 系列之二十四 §3.2）：

- **低相关贪心**：「最终基于测试集进行因子筛选时，我们会限定入选因子与已入选
  因子 spearman 相关性低于 0.4」。按奖励 R 降序贪心入选 —— 新因子与**任一
  已入选因子**的面板 spearman 相关 > 0.4 则跳过。相关度量 = 面板 flatten 向量
  （date×code）的 spearman，与 :func:`factor.gflownet.tb.evaluate_samples` 一致。
- **RRE 秩稳定性门槛**（系列之二十四，AlphaEval 的 Rank-based Robustness
  Evaluation）：截面排名自相关过低的因子每日大换血 → 高换手、成本吃掉 alpha。
  ``min_autocorr`` 默认关闭（0），可按需开启作为硬门槛。

实现：:func:`select_low_corr` 按 R 降序逐因子评估 —— 先过秩稳定性门槛，
再过低相关贪心；两个条件都通过才入选。
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from factor.formula import formula_builder

log = logging.getLogger(__name__)

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
                    progress: bool = False,
                    min_autocorr: float = 0.0) -> list[tuple[str, float]]:
    """按 R 降序贪心入选，入选因子与已入选因子相关 > ``threshold`` 则跳过。

    Args:
        samples: [(formula, R)]，调用方保证已按 R 降序（``sample_formulas`` 输出即降序）。
        panel: 特征面板 dict。
        features: 特征名列表（formula_builder 用）。
        threshold: 相关性上限（研报之二十二 0.4）。
        evaluator: 可注入 mock 求值器（formula -> 面板）。
        progress: 是否打印筛选进度。
        min_autocorr: **RRE 秩稳定性门槛**（研报之二十四）：因子面板的截面
            排名自相关下限，低于该值剔除（降换手）。0 = 关闭（默认，
            保持原行为）；经验取值参考换手目标，如 0.2~0.5。

    Returns:
        入选的 [(formula, R)]（保持 R 降序）。
    """
    from factor.gflownet.reward import rank_stability

    selected: list[tuple[str, float, pd.DataFrame]] = []
    n_rre_dropped = 0
    for i, (formula, r) in enumerate(samples):
        if evaluator is not None:
            fp = evaluator(formula)
        else:
            fp = formula_builder(formula, features=features)(panel)
        if fp is None or fp.empty:
            continue
        if min_autocorr > 0.0:
            ac = rank_stability(fp)
            if not np.isfinite(ac) or ac < min_autocorr:
                n_rre_dropped += 1
                if progress:
                    log.info("[RRE] %d/%d autocorr=%.3f < %.2f 剔除",
                             i + 1, len(samples), ac, min_autocorr)
                continue
        ok = True
        for _, _, prev_fp in selected:
            if abs(panel_flat_corr(fp, prev_fp, min_overlap=min_overlap)) > threshold:
                ok = False
                break
        if ok:
            selected.append((formula, r, fp))
        if progress and (i + 1) % 50 == 0:
            log.info("筛选 %d/%d 已入选 %d", i + 1, len(samples), len(selected))
    if min_autocorr > 0.0 and n_rre_dropped:
        log.info("RRE 秩稳定性门槛 min_autocorr=%s: 共剔除 %d 个",
                 min_autocorr, n_rre_dropped)
    return [(f, r) for f, r, _ in selected]
