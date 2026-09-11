"""
组合构建约束与面板级权重算子
============================

**组合构建层的唯一真源**（2026-09-11 由 `optimize/portfolio.py` 下沉）。
这里装的是"**不需要风险模型**的权重生产算子"——信号构建与工程约束投影，
全部为面板级（date×code）纯函数，**零三方依赖**（只要 numpy/pandas）。

与 :mod:`strategy.base` / :mod:`strategy.examples` 的关系：

- ``strategy.examples``：**截面级**策略类，实现 ``Strategy`` 契约
  （``get_weights(Series) -> Series``），由回测引擎在调仓日逐日调用。
- ``strategy.constraints``（本模块）：**面板级**算子，一次性处理整块面板，
  供 ``optimize.portfolio.optimize_weights`` 这类批量入口编排使用。

两处不是简单重复，粒度与 tie 语义都不同：

- ``equal_topk`` 与 :class:`strategy.examples.TopKLongOnly` **目标相同、tie-break
  不同**——本模块用 ``rank(method="first", ascending=False)``（按**列序确定性**），
  后者用 ``sort_values()``（quicksort，**不稳定**）。实测真实因子库约 **4.4% 的
  截面**在 top-k 边界存在 tie（离散型因子接近 100%），两者在这些截面会选出
  **不同**的持仓集合。故本模块保留确定性实现，刻意**不委托**给策略类。
- ``factor_weighted``（全池 rank 百分比归一化）在 strategy 策略类中**没有对应物**：
  ``TopKLongOnly(weight_mode="factor")`` 只在前 k 只内按因子值归一化，语义不同。

约束叠加顺序（固定，勿改）::

    信号权重 → 行业中性 → 权重上下限 → 换手收缩
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "build_signal_weights",
    "neutralize_industry",
    "apply_bounds",
    "apply_turnover",
    "apply_constraints",
]


# ===========================================================================
# 1) 信号构建（面板级向量化）
# ===========================================================================
def build_signal_weights(
    factor_panel: pd.DataFrame,
    method: str = "factor_weighted",
    k: int | None = None,
) -> pd.DataFrame:
    """由因子面板生成原始信号权重（未经任何约束）。

    Args:
        factor_panel: date×code 因子值（方向已对齐：值越大越看好）。
        method:
            - ``"factor_weighted"``: 截面 rank 百分比加权（0..1 归一），
              全池生效（strategy 策略类中无对应物，见模块 docstring）。
            - ``"equal_topk"``: 每截面取 top-k 等权 ``1/k``（k 必填）。
              与 :class:`strategy.examples.TopKLongOnly` 目标相同，但 tie-break
              用 ``rank(method="first", ascending=False)``（按**列序**确定性），
              而后者用 ``sort_values()``（quicksort，不稳定）。真实因子库约
              **4.4% 的截面**在 top-k 边界存在 tie（离散型因子接近 100%），
              两者在这些截面会选出**不同**的持仓集合。此处取确定性实现。
        k: equal_topk 的持仓数。

    Returns:
        DataFrame(date×code) 权重；非空行权重和 ≈ 1（等权/归一化后），
        全 NaN 行退化为 0。
    """
    if method == "equal_topk":
        if not k or k <= 0:
            raise ValueError("equal_topk 需要 k > 0")
        ranks = factor_panel.rank(axis=1, ascending=False, method="first")
        w = ranks.le(k).astype(float)
        w[ranks.gt(k)] = 0.0
        w = w.div(w.sum(axis=1), axis=0).fillna(0.0)
    elif method == "factor_weighted":
        w = factor_panel.rank(axis=1, pct=True)
        w = w.where(factor_panel.notna())
        w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    else:
        raise ValueError(
            f"未知组合优化方法 {method!r}，可选: factor_weighted / equal_topk"
        )
    return w


# ===========================================================================
# 2) 约束算子（面板级，纯函数）
# ===========================================================================
def neutralize_industry(
    w: pd.DataFrame,
    industry_map: Mapping[str, str] | pd.Series,
    target: Mapping[str, float] | pd.Series | None = None,
) -> pd.DataFrame:
    """行业中性化（投影法）：每行业总权重 = 目标（默认等权），行业内按信号分配。

    ``industry_map`` 未覆盖的 code 权重保持不变（不参与投影）。
    """
    ind = pd.Series(industry_map).reindex(w.columns)
    known = ind.notna()
    if not known.any():
        return w
    inds = ind[known].unique()
    # 目标行业权重
    if target is None:
        tgt = pd.Series(1.0 / len(inds), index=inds)
    else:
        tgt = pd.Series(target).reindex(inds).fillna(0.0)
        s = tgt.sum()
        if s > 0:
            tgt = tgt / s
    # 组内权重总和（date × known_code），转置按行业分组求和
    sub = w.loc[:, known] if known.all() else w.loc[:, ind[known].index]
    ind_sum = sub.T.groupby(ind[known].values).transform("sum").T
    # 每个 code 的目标行业权重
    tgt_code = ind[known].map(tgt).values
    adj = sub * (tgt_code / ind_sum.replace(0, np.nan))
    out = w.copy()
    out.loc[:, ind[known].index] = adj.fillna(0.0)
    return out


def apply_bounds(
    w: pd.DataFrame,
    max_weight: float | None = None,
    min_weight: float | None = None,
) -> pd.DataFrame:
    """权重上下限：上限裁剪超额不回补（等价现金）；下限过滤微仓（清零）。"""
    if max_weight is not None and max_weight < 1.0:
        w = w.clip(upper=max_weight)
    if min_weight is not None and min_weight > 0:
        w = w.where((w >= min_weight) | (w == 0), 0.0)
    return w


def apply_turnover(
    w: pd.DataFrame,
    prev: pd.Series | pd.DataFrame,
    max_turnover: float,
) -> pd.DataFrame:
    """换手约束：w' = prev + α·(w − prev)，α 使单边换手 ≤ max_turnover。"""
    if max_turnover is None or max_turnover <= 0:
        return w
    if isinstance(prev, pd.Series):
        prev = pd.DataFrame(
            np.broadcast_to(prev.reindex(w.columns).fillna(0.0).values, w.shape),
            index=w.index, columns=w.columns,
        )
    else:
        prev = prev.reindex(index=w.index, columns=w.columns).ffill().bfill().fillna(0.0)
    delta = (w - prev).abs().sum(axis=1) * 0.5  # 单边换手
    alpha = (max_turnover / delta.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
    return prev + w.sub(prev).mul(alpha, axis=0)


def apply_constraints(
    w: pd.DataFrame,
    industry_map: Mapping[str, str] | pd.Series | None = None,
    industry_target: Mapping[str, float] | pd.Series | None = None,
    max_weight: float | None = None,
    min_weight: float | None = None,
    prev_weights: pd.Series | pd.DataFrame | None = None,
    max_turnover: float | None = None,
) -> pd.DataFrame:
    """按固定顺序叠加全部工程约束（组合构建的统一出口）。

    顺序：**行业中性 → 权重上下限 → 换手收缩**（与历史行为一致，勿改）。
    行业中性后若再裁剪，行业总权重会略降但行业**比例**基本保持（近似约束）。
    """
    if industry_map is not None:
        w = neutralize_industry(w, industry_map, industry_target)
    w = apply_bounds(w, max_weight, min_weight)
    if prev_weights is not None and max_turnover is not None:
        w = apply_turnover(w, prev_weights, max_turnover)
    return w
