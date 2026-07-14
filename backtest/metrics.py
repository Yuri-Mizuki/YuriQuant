"""
绩效评估指标
============

计算回测结果的常用绩效指标:
- 年化收益率、累计收益
- 夏普比率、信息比率
- 最大回撤、卡玛比率
- 胜率、盈亏比
- 换手率

所有函数接收日频收益率 Series(index=date)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def annual_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """年化收益率。"""
    total = (1 + daily_returns).prod() - 1
    n_years = len(daily_returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return (1 + total) ** (1 / n_years) - 1


def annual_volatility(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """年化波动率。"""
    return daily_returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """夏普比率（无风险利率默认0）。"""
    vol = annual_volatility(daily_returns, periods_per_year)
    if vol == 0:
        return 0.0
    return (annual_return(daily_returns, periods_per_year) - rf) / vol


def max_drawdown(daily_returns: pd.Series) -> float:
    """最大回撤（正值，如0.15表示回撤15%）。"""
    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return abs(dd.min())


def calmar_ratio(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """卡玛比率: 年化收益 / 最大回撤。"""
    mdd = max_drawdown(daily_returns)
    if mdd == 0:
        return 0.0
    return annual_return(daily_returns, periods_per_year) / mdd


def information_ratio(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """信息比率: 超额收益年化 / 跟踪误差年化。"""
    excess = daily_returns - benchmark_returns
    te = excess.std() * np.sqrt(periods_per_year)
    if te == 0:
        return 0.0
    ir = excess.mean() * periods_per_year / te
    return ir


def win_rate(daily_returns: pd.Series) -> float:
    """日胜率。"""
    if len(daily_returns) == 0:
        return 0.0
    return (daily_returns > 0).sum() / len(daily_returns)


def turnover_rate(weights_history: pd.DataFrame) -> float:
    """平均换手率。
    weights_history: DataFrame(index=date, columns=code), 每日权重。
    """
    if len(weights_history) < 2:
        return 0.0
    diffs = weights_history.diff().abs().sum(axis=1) / 2
    return diffs.mean()


def calc_all_metrics(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    weights_history: pd.DataFrame | None = None,
    periods_per_year: int = 252,
) -> dict:
    """一次性计算所有指标。"""
    m = {
        "annual_return": annual_return(daily_returns, periods_per_year),
        "annual_volatility": annual_volatility(daily_returns, periods_per_year),
        "sharpe": sharpe_ratio(daily_returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(daily_returns),
        "calmar": calmar_ratio(daily_returns, periods_per_year),
        "win_rate": win_rate(daily_returns),
        "total_return": (1 + daily_returns).prod() - 1,
        "n_days": len(daily_returns),
    }
    if benchmark_returns is not None:
        # 对齐
        aligned = daily_returns.align(benchmark_returns, join="inner")[0]
        bench_aligned = daily_returns.align(benchmark_returns, join="inner")[1]
        m["benchmark_annual_return"] = annual_return(bench_aligned, periods_per_year)
        m["information_ratio"] = information_ratio(aligned, bench_aligned, periods_per_year)
        m["excess_return"] = m["annual_return"] - m["benchmark_annual_return"]
    if weights_history is not None:
        m["avg_turnover"] = turnover_rate(weights_history)
    return m


def format_metrics(metrics: dict) -> str:
    """格式化为可读字符串。"""
    lines = []
    for k, v in metrics.items():
        if isinstance(v, float):
            if abs(v) < 100:
                lines.append(f"  {k:<28s} {v:>10.4f}")
            else:
                lines.append(f"  {k:<28s} {v:>10.0f}")
        else:
            lines.append(f"  {k:<28s} {v}")
    return "\n".join(lines)
