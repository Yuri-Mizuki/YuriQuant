"""
因子分析
========

因子检验指标:
- IC (Information Coefficient): 因子值与未来收益的截面相关系数
- IR (Information Ratio): IC 均值 / IC 标准差 × √252
- IC 衰减: 不同持有期的 IC 变化
- 分层回测: 按因子值分 5 组，看各组收益单调性

所有函数接收:
- factor_panel: DataFrame(date, code), 因子值
- returns_panel: DataFrame(date, code), 日收益率
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def calc_ic_series(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """计算每日 IC（因子值与未来收益的截面相关系数）。

    Args:
        factor_panel: DataFrame(date, code), 因子值。
        returns_panel: DataFrame(date, code), 未来一期收益。
        method: 'pearson' 或 'spearman'（默认，Rank IC）。
    Returns:
        Series(index=date), 每日 IC。
    """
    common_dates = factor_panel.index.intersection(returns_panel.index)
    common_codes = factor_panel.columns.intersection(returns_panel.columns)
    fp = factor_panel.loc[common_dates, common_codes]
    rp = returns_panel.loc[common_dates, common_codes]

    ic_list = []
    for date in common_dates:
        f = fp.loc[date].dropna()
        r = rp.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 5:
            ic_list.append(np.nan)
            continue
        f_aligned = f.loc[common]
        r_aligned = r.loc[common]
        if method == "spearman":
            corr, _ = stats.spearmanr(f_aligned, r_aligned)
        else:
            corr, _ = stats.pearsonr(f_aligned, r_aligned)
        ic_list.append(corr)

    return pd.Series(ic_list, index=common_dates, name="ic")


def calc_ir(ic_series: pd.Series, periods_per_year: int = 252) -> float:
    """信息比率 = IC均值 / IC标准差 × √252。"""
    ic = ic_series.dropna()
    if len(ic) < 2 or ic.std() == 0:
        return 0.0
    return ic.mean() / ic.std() * np.sqrt(periods_per_year)


def calc_ic_decay(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    max_lag: int = 10,
) -> pd.Series:
    """IC 衰减: lag=1~max_lag 日的 IC 变化。"""
    decay = {}
    for lag in range(1, max_lag + 1):
        shifted_returns = returns_panel.shift(-lag)
        ic = calc_ic_series(factor_panel, shifted_returns)
        decay[lag] = ic.mean()
    return pd.Series(decay, name="ic_decay")


def quantile_backtest(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """分层回测: 按因子值分 N 组，计算各组累计收益。

    Returns:
        DataFrame(index=date, columns=quantile_1~N), 各组累计净值。
    """
    common_dates = factor_panel.index.intersection(returns_panel.index)
    common_codes = factor_panel.columns.intersection(returns_panel.columns)
    fp = factor_panel.loc[common_dates, common_codes]
    rp = returns_panel.loc[common_dates, common_codes]

    group_returns = pd.DataFrame(0.0, index=common_dates, columns=[f"Q{i+1}" for i in range(n_quantiles)])

    for i, date in enumerate(common_dates):
        if i == 0:
            continue
        f = fp.iloc[i - 1].dropna()  # 用前一日因子值分组
        r = rp.iloc[i]              # 当日收益

        common = f.index.intersection(r.index)
        if len(common) < n_quantiles:
            continue
        f_aligned = f.loc[common]
        r_aligned = r.loc[common]

        try:
            groups = pd.qcut(f_aligned, n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        n_actual = groups.nunique()
        for g in range(n_actual):
            mask = groups == g
            if mask.sum() > 0:
                group_returns.iloc[i, g] = r_aligned[mask].mean()

    # 累计净值
    cum = (1 + group_returns).cumprod()
    return cum


def factor_summary(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
) -> dict:
    """因子检验汇总报告。"""
    ic = calc_ic_series(factor_panel, returns_panel)
    ir = calc_ir(ic)
    decay = calc_ic_decay(factor_panel, returns_panel, max_lag=5)
    layers = quantile_backtest(factor_panel, returns_panel, n_quantiles=5)

    return {
        "ic_mean": ic.mean(),
        "ic_std": ic.std(),
        "ic_win_rate": (ic > 0).mean(),
        "ir": ir,
        "ic_decay": decay.to_dict(),
        "layer_returns": layers.iloc[-1].to_dict(),
        "ic_series": ic,
        "layer_nav": layers,
    }
