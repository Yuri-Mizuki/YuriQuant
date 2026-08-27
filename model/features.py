"""特征供给 —— 模型层 ① FeatureStore。

把因子库面板（或任意 ``{name: date×code 面板}`` 集合）整理为对齐的模型特征集。
三级选择漏斗（对齐模型层蓝图 Step 1）：

1. 白/黑名单过滤（include / exclude）
2. 覆盖率过滤（低历史覆盖的特征直接剔除，避免拖垮对齐网格）
3. 相关性去冗余（贪心 pairwise dedup：|corr| > 阈值的因子只留一个，
   复用 ``research.dpp_selection`` 与因子库冗余预检同一口径）
4. 上限截断（max_features；有质量分时按质量降序保留，否则按独立性顺序）

对齐约定：特征对齐到所有面板的**交集网格**（date×code），与
``factor.synthesis._long_matrix`` 同口径；面板间不重叠的格子为 NaN，
由 Predictor 按各自算法处理（GBDT 原生容忍 / ridge 剔行）。

输入面板应已截面标准化（因子库 panels 即此约定；mock/原始特征请先过
``standardize_zscore``）。
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("model.features")

__all__ = ["build_feature_set"]


def _aligned_grid(panels: dict[str, pd.DataFrame]) -> tuple[pd.Index, pd.Index]:
    """所有面板的 (date, code) 交集网格。"""
    idx = None
    cols = None
    for p in panels.values():
        idx = p.index if idx is None else idx.intersection(p.index)
        cols = p.columns if cols is None else cols.intersection(p.columns)
    return idx, cols


def build_feature_set(
    panels: dict[str, pd.DataFrame],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    min_coverage: float = 0.5,
    dedup_corr: float | None = 0.7,
    max_features: int | None = None,
    quality: pd.Series | None = None,
    dedup_sample_days: int = 240,
) -> dict[str, pd.DataFrame]:
    """整理对齐特征集（选择漏斗：过滤 → 去冗余 → 截断）。

    Args:
        panels: {name: date×code 面板}（建议已截面标准化）。
        include: 白名单（给了则只用这些）。
        exclude: 黑名单。
        min_coverage: 特征在对齐网格上的非 NaN 覆盖率下限（低于剔除）。
        dedup_corr: 相关性去冗余阈值（None 关闭；与因子库冗余预检同口径，
            贪心入选，|corr| 超阈值者剔除）。
        max_features: 特征数上限。
        quality: {name: 质量分}（如 valid 段 |IC|）。给了则去冗余与截断
            都按质量降序优先保留；否则按独立性（平均 |corr| 升序）。
        dedup_sample_days: 去冗余时最多抽样的日期数（提速；<=0 用全部）。

    Returns:
        对齐后的 {name: 面板}（网格交集，缺失为 NaN）。
    """
    if not panels:
        raise ValueError("panels 为空")

    names = list(panels.keys())
    if include is not None:
        names = [n for n in names if n in set(include)]
    if exclude is not None:
        names = [n for n in names if n not in set(exclude)]
    if not names:
        raise ValueError("白/黑名单过滤后特征为空")
    sel = {n: panels[n] for n in names}
    log.info("特征漏斗·过滤: %d -> %d", len(panels), len(sel))

    # ---- 对齐网格 + 覆盖率过滤 ----
    idx, cols = _aligned_grid(sel)
    if len(idx) == 0 or len(cols) == 0:
        raise ValueError("特征面板无公共 (date, code) 网格")
    aligned = {n: p.reindex(index=idx, columns=cols) for n, p in sel.items()}

    if min_coverage and min_coverage > 0:
        kept = []
        for n, p in aligned.items():
            cov = float(p.notna().mean().mean())
            if cov >= min_coverage:
                kept.append(n)
            else:
                log.info("特征漏斗·覆盖率: 剔除 %s (coverage=%.2f)", n, cov)
        if not kept:
            raise ValueError(f"覆盖率过滤后特征为空（min_coverage={min_coverage}）")
        aligned = {n: aligned[n] for n in kept}
    log.info("特征漏斗·对齐网格: %d 日 × %d 股, 特征 %d 个", len(idx), len(cols), len(aligned))

    # ---- 相关性去冗余 ----
    if dedup_corr and len(aligned) > 1:
        from research.dpp_selection import corr_matrix, pairwise_dedup

        sample = aligned
        if dedup_sample_days and len(idx) > dedup_sample_days:
            step = max(1, len(idx) // dedup_sample_days)
            sample_days = idx[::step]
            sample = {n: p.loc[sample_days] for n, p in aligned.items()}
        corr = corr_matrix(sample)
        order = None
        if quality is not None:
            q = quality.reindex(list(aligned.keys())).dropna()
            order = list(q.sort_values(ascending=False).index)
            order += [n for n in aligned if n not in set(order)]
        kept = pairwise_dedup(corr, order=order, threshold=dedup_corr)
        dropped = [n for n in aligned if n not in set(kept)]
        if dropped:
            log.info("特征漏斗·去冗余(阈值 %.2f): %d -> %d, 剔除 %s",
                     dedup_corr, len(aligned), len(kept), dropped)
        aligned = {n: aligned[n] for n in kept if n in aligned}

    # ---- 上限截断 ----
    if max_features and len(aligned) > max_features:
        if quality is not None:
            order = list(quality.reindex(list(aligned.keys()))
                         .sort_values(ascending=False).index)
        else:
            order = list(aligned.keys())
        cut = order[:max_features]
        log.info("特征漏斗·截断: %d -> %d", len(aligned), len(cut))
        aligned = {n: aligned[n] for n in cut if n in aligned}

    return aligned
