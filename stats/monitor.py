"""
IC 漂移监控统计
===============

因子/模型上线后的漂移监控（纯统计部分，被 monitoring/ 生产化包装消费）：

- rolling_ic：滚动窗口 IC（窗口太小噪音大，建议 ≥ 40 交易日）
- monitor_ic_series：对现成 IC 序列做全期 vs 近期漂移对比（因子库复用）
- monitor_report：面板级监控报告（IC 漂移、IR、衰减、自相关）

真源历史：2026-08-29 自 optimize/monitor.py 下沉（optimize 层保留 re-export
转出口；monitoring/metrics 与 research/factor_library 改为直接消费本模块，
research↔optimize 包级循环就此解开）。
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from stats.ic import calc_ic_decay, calc_ic_series, calc_ir, factor_autocorr
from stats.robust_stats import nw_tstat

__all__ = ["rolling_ic", "monitor_ic_series", "monitor_report"]


def _stats(s: pd.Series) -> dict[str, float]:
    t_nw, _se_nw, _lag = nw_tstat(s.values) if len(s) > 1 else (0.0, 0.0, 0)
    from scipy import stats as _stats
    p_nw = 2.0 * (1.0 - _stats.t.cdf(abs(t_nw), df=max(len(s) - 1, 1))) if len(s) > 1 else float("nan")
    return {"mean": float(s.mean()), "ir": float(calc_ir(s)),
            "t_nw": float(t_nw), "p_nw": float(p_nw)}


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


def monitor_ic_series(
    ic_series: pd.Series,
    window: int = 60,
) -> dict[str, Any]:
    """对现成 IC 序列做漂移监控（因子库生命周期监控复用，无需重算面板）。

    对比「全期」与「最近 window 交易日」两段：
    ic_mean_full / ic_mean_recent / ic_drift / ic_ir_full / ic_ir_recent /
    ic_t_nw_full / ic_p_nw_full / ic_t_nw_recent / ic_p_nw_recent /
    n_days / recent_n_days / status（recent 均值 < 全期一半 → warning）。
    """
    ic = ic_series.dropna()
    if len(ic) < 2:
        return {"ic_mean_full": float("nan"), "ic_mean_recent": float("nan"),
                "ic_drift": float("nan"), "ic_ir_full": float("nan"),
                "ic_ir_recent": float("nan"), "ic_t_nw_full": float("nan"),
                "ic_p_nw_full": float("nan"), "ic_t_nw_recent": float("nan"),
                "ic_p_nw_recent": float("nan"), "n_days": 0, "recent_n_days": 0,
                "status": "unknown"}
    recent = ic.tail(window) if len(ic) > window else ic
    full = _stats(ic)
    rec = _stats(recent)
    return {
        "ic_mean_full": full["mean"], "ic_mean_recent": rec["mean"],
        "ic_drift": rec["mean"] - full["mean"],
        "ic_ir_full": full["ir"], "ic_ir_recent": rec["ir"],
        "ic_t_nw_full": full["t_nw"], "ic_p_nw_full": full["p_nw"],
        "ic_t_nw_recent": rec["t_nw"], "ic_p_nw_recent": rec["p_nw"],
        "n_days": int(len(ic)), "recent_n_days": int(len(recent)),
        "status": "normal" if rec["mean"] >= 0.5 * full["mean"] else "warning",
    }


def monitor_report(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    window: int = 60,
    max_lag: int = 10,
) -> dict[str, Any]:
    """因子/模型漂移监控报告（面板级，需重算 IC）。

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
    base = monitor_ic_series(ic, window=window)
    decay = calc_ic_decay(factor_panel, returns_panel, max_lag=max_lag)
    base["ic_decay"] = decay
    base["autocorr"] = float(factor_autocorr(factor_panel))
    return base
