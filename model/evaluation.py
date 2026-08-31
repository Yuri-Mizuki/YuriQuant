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

import numpy as np
import pandas as pd

from stats.ic import calc_ic_decay, calc_ic_series, calc_ir, quantile_backtest
from stats.robust_stats import nw_tstat

__all__ = ["evaluate_model", "eval_row", "monthly_ic", "fit_predict_valid_ic"]


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


def eval_row(tag: str, pred: pd.DataFrame, fwd: pd.DataFrame,
             days: pd.DatetimeIndex) -> dict:
    """单个预测面板的 OOS 评价行（IC/ICIR/NW-t/p/胜率/多空日均）。

    2026-08-31 自 ml_synthesis_experiment._eval_row 下沉（三实验共用口径）。
    """
    pred = pred.loc[days]
    tgt = fwd.loc[days]
    ic = calc_ic_series(pred, tgt).dropna()
    t_nw, _, _ = nw_tstat(ic.values) if len(ic) > 1 else (0.0, 0.0, 0)
    from scipy import stats as st
    p = 2 * (1 - st.t.cdf(abs(t_nw), df=max(len(ic) - 1, 1)))
    ev = evaluate_model(pred, tgt)
    qb = ev["quantile_backtest"]
    ls = None
    if qb is not None and len(qb) > 1:
        # Q5(预测最高组) - Q1(最低组) 的日均收益（组内日收益 = 净值日差）
        ret = qb.diff().dropna(how="all")
        if "Q5" in ret.columns and "Q1" in ret.columns:
            ls = float((ret["Q5"] - ret["Q1"]).mean())
    return {
        "name": tag, "ic_mean": float(ic.mean()),
        "ic_ir": float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() > 0 else np.nan,
        "ic_t_nw": float(t_nw), "ic_p_nw": float(p),
        "ic_win_rate": float((ic > 0).mean()),
        "n_days": int(len(ic)),
        "ls_daily": ls,
    }


def monthly_ic(pred: pd.DataFrame, fwd: pd.DataFrame,
               days: pd.DatetimeIndex) -> pd.Series:
    """按自然月聚合的 rank IC 均值序列。"""
    ic = calc_ic_series(pred.loc[days], fwd.loc[days]).dropna()
    return ic.groupby(ic.index.to_period("M")).mean()


def fit_predict_valid_ic(predictor, feats_tr, labels_tr, feats_va, labels_va) -> float:
    """train 拟合 → valid 预测 → valid 段 rank IC 均值（调参目标函数）。"""
    p = predictor()
    p.fit(feats_tr, labels_tr)
    pred = p.predict(feats_va)
    ic = calc_ic_series(pred, labels_va).dropna()
    return float(ic.mean()) if len(ic) else float("nan")
