"""
持续监控 —— 03 优化层「持续监控」。

因子/模型上线后的漂移监控。当前提供骨架 + 复用 research/factor_analysis
的滚动 IC / 衰减 / 自相关能力：

- rolling_ic：滚动窗口 IC（窗口太小噪音大，建议 ≥ 40 交易日）
- monitor_report：全期 vs 近期对比（IC 均值漂移、IR、衰减、自相关、近 1/3 段斜率）

TODO（待建）：
- 拥挤度监控（因子截面相关性、同族因子拥挤）
- 预警阈值配置 + 自动化定时跑（可挂 WorkBuddy 自动化）
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.factor_analysis import calc_ic_decay, calc_ic_series, calc_ir, factor_autocorr
from research.robust_stats import nw_tstat

__all__ = ["rolling_ic", "monitor_report"]


def rolling_ic(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    """滚动窗口 IC 均值序列（监控因子预测力的时变特征）。

    Args:
        factor_panel / returns_panel: date×code（returns 为未来一期收益）。
        window: 滚动窗口交易日数（默认 60 ≈ 一个季度）。
    Returns:
        Series(index=date)，rolling mean of daily IC。
    """
    ic = calc_ic_series(factor_panel, returns_panel)
    return ic.rolling(window, min_periods=max(10, window // 2)).mean()


def monitor_report(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    window: int = 60,
    max_lag: int = 10,
) -> dict[str, Any]:
    """因子/模型漂移监控报告。

    对比「全期」与「最近 window 交易日」两段的 IC 特征，量化漂移：

    - ic_mean_full / ic_mean_recent：全期 vs 近期 IC 均值
    - ic_drift = 近期 - 全期（正=增强，负=衰减）
    - ic_ir_full / ic_ir_recent
    - ic_t_nw_full / ic_t_nw_recent：Newey-West 显著性（口径同因子库）
    - ic_decay：全期 IC 衰减
    - autocorr：因子自相关（换手代理）
    - n_days / recent_n_days：样本量
    """
    ic = calc_ic_series(factor_panel, returns_panel).dropna()
    recent = ic.tail(window) if len(ic) > window else ic

    def _stats(s: pd.Series) -> dict[str, float]:
        t_nw, _se_nw, _lag = nw_tstat(s.values) if len(s) > 1 else (0.0, 0.0, 0)
        from scipy import stats as _stats
        p_nw = 2.0 * (1.0 - _stats.t.cdf(abs(t_nw), df=max(len(s) - 1, 1))) if len(s) > 1 else float("nan")
        return {"mean": float(s.mean()), "ir": float(calc_ir(s)),
                "t_nw": float(t_nw), "p_nw": float(p_nw)}

    full = _stats(ic)
    rec = _stats(recent)
    decay = calc_ic_decay(factor_panel, returns_panel, max_lag=max_lag)

    return {
        "ic_mean_full": full["mean"], "ic_mean_recent": rec["mean"],
        "ic_drift": rec["mean"] - full["mean"],
        "ic_ir_full": full["ir"], "ic_ir_recent": rec["ir"],
        "ic_t_nw_full": full["t_nw"], "ic_p_nw_full": full["p_nw"],
        "ic_t_nw_recent": rec["t_nw"], "ic_p_nw_recent": rec["p_nw"],
        "ic_decay": decay,
        "autocorr": float(factor_autocorr(factor_panel)),
        "n_days": int(len(ic)), "recent_n_days": int(len(recent)),
        "status": "normal" if rec["mean"] >= 0.5 * full["mean"] else "warning",
    }
