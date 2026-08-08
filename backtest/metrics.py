"""
绩效评估指标
============

计算回测结果的常用绩效指标（含中英文标签）:
- 年化收益率 Annual Return
- 累计收益 Total Return
- 年化波动率 Annual Volatility
- 夏普比率 Sharpe Ratio
- 索提诺比率 Sortino Ratio
- 最大回撤 Max Drawdown
- 卡玛比率 Calmar Ratio
- 胜率 Win Rate
- 盈亏比 Profit/Loss Ratio
- 平均日收益 Avg Daily Return
- 平均换手率 Avg Turnover
- 信息比率 Information Ratio（需基准）
- 超额年化 Excess Return（需基准）

所有函数接收日频收益率 Series(index=date)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def annual_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """年化收益率 Annual Return。"""
    total = (1 + daily_returns).prod() - 1
    n_years = len(daily_returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return (1 + total) ** (1 / n_years) - 1


def annual_volatility(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """年化波动率 Annual Volatility。"""
    return daily_returns.std() * np.sqrt(periods_per_year)


def _rf_daily(rf: float, periods_per_year: int) -> float:
    """年化 rf → 日频 rf（复利折算），用于超额收益序列。"""
    if rf <= 0:
        return 0.0
    return (1 + rf) ** (1 / periods_per_year) - 1


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """夏普比率 Sharpe Ratio。

    rf 按日频复利折算为超额收益序列（(1+r)/(1+rf_daily)-1），再用几何年化口径：
    Sharpe = 年化超额收益 / 年化波动率。rf=0 时与经典实现完全一致。
    """
    rfd = _rf_daily(rf, periods_per_year)
    excess = (1 + daily_returns) / (1 + rfd) - 1
    vol = annual_volatility(daily_returns, periods_per_year)
    if vol == 0:
        return 0.0
    return annual_return(excess, periods_per_year) / vol


def sortino_ratio(daily_returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """索提诺比率 Sortino Ratio: 只用下行波动。

    下行偏差 = 半方差 downside deviation = sqrt(mean(min(excess, 0)²))，
    衡量亏损深度（相对目标收益 0）；早期实现误用"负收益日样本的 std"，
    度量的是亏损日之间的离散度，系统性高估 |Sortino|（实证 ~30%）。
    rf 按日频折算为超额序列后再计算（与 Sharpe 同口径）。
    """
    rfd = _rf_daily(rf, periods_per_year)
    excess = (1 + daily_returns) / (1 + rfd) - 1
    ar = annual_return(excess, periods_per_year)
    dd = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    if dd == 0:
        return 0.0
    downside_vol = dd * np.sqrt(periods_per_year)
    return ar / downside_vol


def max_drawdown(daily_returns: pd.Series) -> float:
    """最大回撤 Max Drawdown（正值）。"""
    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return abs(dd.min())


def calmar_ratio(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """卡玛比率 Calmar Ratio: 年化收益 / 最大回撤。"""
    mdd = max_drawdown(daily_returns)
    if mdd == 0:
        return 0.0
    return annual_return(daily_returns, periods_per_year) / mdd


def information_ratio(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """信息比率 Information Ratio。"""
    excess = daily_returns - benchmark_returns
    te = excess.std() * np.sqrt(periods_per_year)
    if te == 0:
        return 0.0
    return excess.mean() * periods_per_year / te


def win_rate(daily_returns: pd.Series) -> float:
    """胜率 Win Rate。"""
    if len(daily_returns) == 0:
        return 0.0
    return (daily_returns > 0).sum() / len(daily_returns)


def profit_loss_ratio(daily_returns: pd.Series) -> float:
    """盈亏比 Profit/Loss Ratio: 平均盈利 / 平均亏损。"""
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    if len(losses) == 0 or losses.mean() == 0:
        return 0.0
    return wins.mean() / abs(losses.mean())


def turnover_rate(weights_history: pd.DataFrame) -> float:
    """平均换手率 Avg Turnover。"""
    if len(weights_history) < 2:
        return 0.0
    diffs = weights_history.diff().abs().sum(axis=1) / 2
    return diffs.mean()


def calc_short_metrics(
    weights_history: pd.DataFrame,
    borrow_fee_series: pd.Series,
    initial_capital: float = 1.0,
    margin_ratio: float = 1.0,
    n_days: int = 0,
    periods_per_year: int = 252,
    exposure: pd.DataFrame | None = None,
) -> dict:
    """空头成本与资金占用指标（配合 ShortCostModel 使用）。

    Args:
        weights_history: 权重历史（date × code），多头 > 0、空头 < 0。
        borrow_fee_series: 每日借券费（元）。
        initial_capital: 初始资金（元），用于把费用折算为年化拖累。
        margin_ratio: 融券保证金比例（与回测引擎一致）。
        n_days: 交易日数（用于年化拖累；缺省取 borrow_fee_series 长度）。
        periods_per_year: 年化周期数。
        exposure: 每日实际敞口（DataFrame，列 long/short/margin，index=date）。
                  由回测引擎在 run() 内按日记录，**比 weights_history（仅调仓日有
                  记录）更准确地反映全年真实资金占用**；缺省时回退 weights_history。
    """
    if exposure is not None and len(exposure) > 0:
        long_exp = exposure["long"].astype(float)
        short_exp = exposure["short"].astype(float)
        margin = exposure["margin"].astype(float)
    else:
        if weights_history is None or len(weights_history) == 0:
            return {}
        w = weights_history.fillna(0.0)
        long_exp = w.clip(lower=0).sum(axis=1)
        short_exp = (-w.clip(upper=0)).sum(axis=1)
        margin = long_exp + short_exp * margin_ratio
    fee = borrow_fee_series.reindex(long_exp.index).fillna(0.0)
    n = n_days or len(long_exp)
    n_years = n / periods_per_year
    return {
        "avg_long_exposure": float(long_exp.mean()),
        "avg_short_exposure": float(short_exp.mean()),
        "avg_margin_usage": float(margin.mean()),
        "max_margin_usage": float(margin.max()),
        "borrow_fee_total": float(fee.sum()),
        "borrow_fee_drag_annual": (
            float(fee.sum() / initial_capital / n_years)
            if n_years > 0 and initial_capital > 0 else 0.0
        ),
    }


# ===========================================================================
# 指标中英文标签
# ===========================================================================
METRIC_LABELS = {
    "annual_return":       "年化收益率 Annual Return",
    "total_return":        "累计收益 Total Return",
    "annual_volatility":   "年化波动率 Annual Volatility",
    "sharpe":              "夏普比率 Sharpe Ratio",
    "sortino":             "索提诺比率 Sortino Ratio",
    "max_drawdown":        "最大回撤 Max Drawdown",
    "calmar":              "卡玛比率 Calmar Ratio",
    "win_rate":            "胜率 Win Rate",
    "profit_loss_ratio":   "盈亏比 Profit/Loss Ratio",
    "avg_daily_return":    "平均日收益 Avg Daily Return",
    "avg_turnover":        "平均换手率 Avg Turnover",
    "n_days":              "交易日数 Trading Days",
    "benchmark_annual_return": "基准年化 Benchmark Annual Return",
    "excess_return":       "超额年化 Excess Return",
    "information_ratio":   "信息比率 Information Ratio",
    "ic_mean":             "IC均值 IC Mean",
    "ic_std":              "IC标准差 IC Std",
    "ic_win_rate":         "IC胜率 IC Win Rate",
    "ir":                  "信息比率 IR (IC-based)",
    "avg_long_exposure":   "平均多头敞口 Avg Long Exposure",
    "avg_short_exposure":  "平均空头敞口 Avg Short Exposure",
    "avg_margin_usage":    "平均保证金占用 Avg Margin Usage",
    "max_margin_usage":    "最大保证金占用 Max Margin Usage",
    "borrow_fee_total":    "借券费总额 Borrow Fee Total (CNY)",
    "borrow_fee_drag_annual": "借券费年化拖累 Borrow Fee Drag (Ann.)",
}


def calc_all_metrics(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    weights_history: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    rf: float = 0.0,
) -> dict:
    """一次性计算所有指标。

    Args:
        daily_returns: 日收益序列。
        benchmark_returns: 基准日收益（可选，计算 IR / 超额收益）。
        weights_history: 权重历史（可选，计算平均换手率）。
        periods_per_year: 年化周期数（默认 252 交易日）。
        rf: 年化无风险利率（默认 0），用于 Sharpe / Sortino 的超额收益。
    """
    m = {
        "annual_return": annual_return(daily_returns, periods_per_year),
        "total_return": (1 + daily_returns).prod() - 1,
        "annual_volatility": annual_volatility(daily_returns, periods_per_year),
        "sharpe": sharpe_ratio(daily_returns, rf=rf, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(daily_returns, rf=rf, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(daily_returns),
        "calmar": calmar_ratio(daily_returns, periods_per_year),
        "win_rate": win_rate(daily_returns),
        "profit_loss_ratio": profit_loss_ratio(daily_returns),
        "avg_daily_return": daily_returns.mean(),
        "n_days": len(daily_returns),
    }
    if benchmark_returns is not None:
        aligned = daily_returns.align(benchmark_returns, join="inner")[0]
        bench_aligned = daily_returns.align(benchmark_returns, join="inner")[1]
        m["benchmark_annual_return"] = annual_return(bench_aligned, periods_per_year)
        m["information_ratio"] = information_ratio(aligned, bench_aligned, periods_per_year)
        m["excess_return"] = m["annual_return"] - m["benchmark_annual_return"]
    if weights_history is not None:
        m["avg_turnover"] = turnover_rate(weights_history)
    return m


def format_metrics(metrics: dict) -> str:
    """格式化为可读字符串（中英文标签）。"""
    lines = []
    for k, v in metrics.items():
        label = METRIC_LABELS.get(k, k)
        if isinstance(v, float):
            if k in ("win_rate", "ic_win_rate"):
                lines.append(f"  {label:<35s} {v:>10.2%}")
            elif abs(v) < 100:
                lines.append(f"  {label:<35s} {v:>10.4f}")
            else:
                lines.append(f"  {label:<35s} {v:>10.0f}")
        else:
            lines.append(f"  {label:<35s} {v}")
    return "\n".join(lines)
