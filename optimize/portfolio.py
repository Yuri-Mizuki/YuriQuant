"""
组合优化 —— 03 优化层「组合优化（无求解器门面）」。

**本模块已退化为薄门面**（2026-09-11 第 3 批口径统一，方案 A）：信号构建与
工程约束投影的真源下沉到 :mod:`strategy.constraints`，这里只保留"面板进 /
面板出"的批量编排签名，供历史调用方（`scripts/compare_portfolio_methods.py`
的 projection 基线、`tests/test_pipeline_layers.py`）无缝继续使用。

为什么下沉：这些算子**不需要风险模型**（无协方差、无求解器），属于"组合构建"
而非"组合优化"；放在 optimize/ 会让 strategy/ 与 optimize/ 的职责边界模糊。
只有需要 Σ 的路径（:mod:`optimize.solver` 的 QP / HRP / BL）才真正属于优化层。

执行链路::

    factor_panel
      → strategy.constraints.build_signal_weights   （信号权重）
      → strategy.constraints.apply_constraints      （行业中性 → 上下限 → 换手）

约束叠加顺序固定为 **信号权重 → 行业中性 → 上下限 → 换手收缩**；
行业中性后若再裁剪，行业总权重会略降但行业**比例**基本保持（近似约束）。

需要协方差驱动的精确优化（min_var / tev / mvo / risk_parity / bl）或 HRP，请改用
:func:`optimize.solver.optimize_weights_qp` / :func:`optimize.solver.optimize_weights_hrp`。
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from strategy.constraints import apply_constraints, build_signal_weights

__all__ = ["optimize_weights"]


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
    """生成截面权重（组合构建默认入口；约束增强版）。

    真源在 :mod:`strategy.constraints`（``build_signal_weights`` +
    ``apply_constraints``），本函数不再持有任何独立实现。

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
    w = build_signal_weights(factor_panel, method=method, k=k)
    return apply_constraints(
        w,
        industry_map=industry_map,
        industry_target=industry_target,
        max_weight=max_weight,
        min_weight=min_weight,
        prev_weights=prev_weights,
        max_turnover=max_turnover,
    )
