"""
基准序列模块（Benchmarks）
=========================

为回测/因子报告提供**统一的基准对照**：把"策略赚的是 alpha 还是 beta"一眼
看清楚。三种基准（均基于现有 close 面板，无额外数据依赖）：

- ``equal_weight``：全市场等权（每日再平衡）——研究最常见的朴素基准。
- ``buy_hold``   ：期初等权买入持有，权重随价格漂移（不调仓）。
- ``index``      ：指数收益（预留接口；当前数据层无指数行情，可传入外部序列）。

对照指标（``compare_to_benchmark``）：超额年化 / 信息比率 / 跟踪误差 /
相关性 / 简化 beta —— 与 ``backtest.metrics.calc_all_metrics`` 无缝合并。

口径：所有收益序列为日频，index=date。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.metrics import PERIODS_PER_YEAR

__all__ = ["equal_weight_returns", "buy_hold_returns", "benchmark_returns",
           "compare_to_benchmark", "with_benchmark_metrics"]


def equal_weight_returns(close_panel: pd.DataFrame) -> pd.Series:
    """全市场等权日收益（每日再平衡，等权是默认行业/市值中性化的朴素对照）。

    对每只股票先算日收益（pct_change），再逐日取横截面均值。
    """
    return close_panel.pct_change().mean(axis=1, skipna=True)


def buy_hold_returns(close_panel: pd.DataFrame) -> pd.Series:
    """期初等权买入持有：权重随价格漂移、期内不调仓（成本最低的持有基准）。

    组合净值 = Σ w_i × (close_i / close_i0)；收益 = 净值 pct_change。
    """
    first = close_panel.iloc[0].replace(0, np.nan)
    nav_i = close_panel.div(first, axis=1)          # 各股归一净值
    port_nav = nav_i.mean(axis=1, skipna=True)      # 期初等权 → 权重随价格漂移
    return port_nav.pct_change()


def benchmark_returns(close_panel: pd.DataFrame, mode: str = "equal_weight") -> pd.Series:
    """基准序列入口。mode: 'equal_weight'(默认) | 'buy_hold'。"""
    if mode == "buy_hold":
        return buy_hold_returns(close_panel)
    return equal_weight_returns(close_panel)


def compare_to_benchmark(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict:
    """策略 vs 基准的对照指标（日频序列已对齐）。"""
    from backtest.metrics import annual_return

    a, b = daily_returns.align(benchmark_returns, join="inner")
    if len(a) < 2:
        return {"excess_annual": np.nan, "information_ratio": np.nan,
                "tracking_error": np.nan, "correlation": np.nan, "beta": np.nan}
    excess = a - b
    te = float(excess.std() * np.sqrt(periods_per_year))
    ir = float(excess.mean() * periods_per_year / te) if te > 0 else 0.0
    corr = float(a.corr(b)) if a.std() > 0 and b.std() > 0 else np.nan
    beta = float(a.cov(b) / b.var()) if b.var() > 0 else np.nan
    return {
        "excess_annual": float(annual_return(a, periods_per_year) - annual_return(b, periods_per_year)),
        "information_ratio": ir,
        "tracking_error": te,
        "correlation": corr,
        "beta": beta,
    }


def with_benchmark_metrics(
    metrics: dict,
    daily_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict:
    """把基准对照指标并入 ``calc_all_metrics`` 的指标 dict（原 dict 不修改）。"""
    out = dict(metrics)
    out["benchmark"] = compare_to_benchmark(daily_returns, benchmark_returns, periods_per_year)
    return out
