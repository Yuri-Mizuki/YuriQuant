"""
模型评价 —— 02 模型层「模型评价」。

统一评价入口：输入预测面板（模型输出）→ 输出 IC/IR、Newey-West 显著性、
IC 衰减、分层单调性。复用现有研究能力：

- research.factor_analysis：calc_ic_series / calc_ir / calc_ic_decay / quantile_backtest
- research.robust_stats：Newey-West t（防自相关伪显著，与因子库判定口径一致）

口径约定：``returns_panel`` 为**未来一期收益**（与全项目 IC 口径一致）。
组合级评价（净值/回撤/基准对比）请走 backtest（scripts/run_backtest、
select_stocks），模型层评价聚焦预测能力的 IC 口径。
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from research.factor_analysis import calc_ic_decay, calc_ic_series, calc_ir, quantile_backtest
from research.robust_stats import nw_tstat

__all__ = ["evaluate_model"]


def evaluate_model(
    pred_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    max_lag: int = 10,
    n_quantiles: int = 5,
) -> dict[str, Any]:
    """对模型预测面板做一套标准评价（模型评价阶段的标准报告）。

    Args:
        pred_panel: date×code 模型预测值（复合因子 / 预测分数）。
        returns_panel: date×code 未来一期收益（与 IC 口径一致）。
        max_lag: IC 衰减最大 lag。
        n_quantiles: 分层组数。

    Returns:
        dict: ic_series / ic_mean / ic_ir / ic_t_nw / ic_p_nw / ic_decay /
              quantile_backtest(DataFrame)。
    """
    ic_series = calc_ic_series(pred_panel, returns_panel)
    ic = ic_series.dropna()
    t_nw, _se_nw, _lag = nw_tstat(ic.values) if len(ic) > 1 else (0.0, 0.0, 0)
    # p 值口径与 research/factor_analysis.standard_factor_summary 一致
    from scipy import stats as _stats
    if len(ic) > 1:
        p_nw = 2.0 * (1.0 - _stats.t.cdf(abs(t_nw), df=max(len(ic) - 1, 1)))
    else:
        p_nw = float("nan")

    return {
        "ic_series": ic_series,
        "ic_mean": float(ic.mean()),
        "ic_ir": calc_ir(ic_series),
        "ic_t_nw": float(t_nw),
        "ic_p_nw": float(p_nw),
        "ic_decay": calc_ic_decay(pred_panel, returns_panel, max_lag=max_lag),
        "quantile_backtest": quantile_backtest(pred_panel, returns_panel, n_quantiles=n_quantiles),
    }
