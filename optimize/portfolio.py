"""
组合优化 —— 03 优化层「组合优化」。

约束增强版（2026-08-07）：在信号权重基础上叠加工程约束，全部为
**无求解器的启发式投影**，输出直接喂 backtest.VectorBacktest：

1. 行业中性（industry_map + industry_target）：把每行行业总权重投影到
   目标行业权重（默认等权行业），行业内再按信号权重分配 —— 投影法。
2. 权重上下限（max_weight / min_weight）：上限裁剪超额不回补（等价现金）；
   下限过滤微仓（< min_weight 清零）。
3. 换手约束（prev_weights + max_turnover）：新权重向旧持仓线性收缩，
   把单边换手率（0.5·Σ|Δw|，与回测成本模型口径一致）压到上限内。

约束叠加顺序：信号权重 → 行业中性 → 上下限 → 换手收缩。
行业中性后若再裁剪，行业总权重会略降但行业**比例**基本保持（近似约束）。

TODO（待建）：
- 均值方差 / 风险平价（协方差估计 + 求解器）
- 换手惩罚进目标函数（而非硬约束收缩）
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

__all__ = ["optimize_weights"]


# ===========================================================================
# 约束算子（面板级，date×code）
# ===========================================================================
def _neutralize_industry(
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


def _apply_turnover(
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


# ===========================================================================
# 主入口
# ===========================================================================
def optimize_weights(
    factor_panel: pd.DataFrame,
    method: str = "factor_weighted",
    k: int | None = None,
    max_weight: float | None = None,
    min_weight: float | None = None,
    industry_map: Mapping[str, str] | pd.Series | None = None,
    industry_target: Mapping[str, float] | pd.Series | None = None,
    prev_weights: pd.Series | pd.DataFrame | None = None,
    max_turnover: float | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """生成截面权重（组合优化默认入口，约束增强版）。

    Args:
        factor_panel: date×code 因子值（方向已对齐：值越大越看好）。
        method:
            - "factor_weighted": 截面 rank 百分比加权（0..1 归一）。
            - "equal_topk": 每截面取 top-k 等权 1/k（k 必填）。
        k: equal_topk 的持仓数。
        max_weight: 个股权重上限；裁剪后超额不回补（权重和可能 <1 = 现金仓位）。
        min_weight: 微仓过滤：权重 < min_weight 的持仓清零（不重新归一）。
        industry_map: code → 行业。提供时启用行业中性（默认等权行业，
            industry_target 可指定目标行业权重，如基准行业权重）。
        prev_weights: 上一期权重（Series 单期广播 / DataFrame 按行对齐）。
            提供 max_turnover 时启用换手约束。
        max_turnover: 单边换手率上限（0.5·Σ|Δw|）。

    Returns:
        DataFrame(date×code) 权重；非空行权重和 ≤ 1（超额裁剪/换手收缩
        后可能小于 1，等价现金仓位）。
    """
    # 1. 信号权重
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
        raise ValueError(f"未知组合优化方法 {method!r}，可选: factor_weighted / equal_topk")

    # 2. 行业中性（投影到目标行业权重）
    if industry_map is not None:
        w = _neutralize_industry(w, industry_map, industry_target)

    # 3. 权重上下限
    if max_weight is not None and max_weight < 1.0:
        w = w.clip(upper=max_weight)
    if min_weight is not None and min_weight > 0:
        w = w.where((w >= min_weight) | (w == 0), 0.0)

    # 4. 换手约束（向上一期持仓收缩）
    if prev_weights is not None and max_turnover is not None:
        w = _apply_turnover(w, prev_weights, max_turnover)

    return w
