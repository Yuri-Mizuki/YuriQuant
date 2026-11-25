"""绩效指标的纯计算真源。

从 ``backtest/metrics`` 下沉（2026-11-25）：这些函数只依赖 numpy/pandas，属统计
原语而非回测逻辑。下沉后 ``strategy/`` 层可以算年化/Sharpe 而不必 import
``backtest``（分层测试禁止），``backtest.metrics`` 对它们做 re-export 保持
历史 import 路径可用。

口径铁律
--------
- ``sharpe_ratio`` = **几何年化收益 / 年化波动** → 报数用（生产定版）。
- ``sharpe_arithmetic`` = ``mean/std·√N`` → **换算显著性用**；文献的
  ``t = SR·√T`` 只认算术口径。正收益时几何偏小（保守），负收益/近零时几何偏大，
  两组「证明年数」不可比。引用任何外部显著性公式前先确认它要哪种 SR。

所有函数接收日频收益率 Series（index=date）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stats import PERIODS_PER_YEAR


def annual_return(daily_returns: pd.Series,
                  periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """年化收益率（几何）。"""
    total = (1 + daily_returns).prod() - 1
    n_years = len(daily_returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return (1 + total) ** (1 / n_years) - 1


def annual_volatility(daily_returns: pd.Series,
                      periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """年化波动率。"""
    return daily_returns.std() * np.sqrt(periods_per_year)


# --- 探针/分场景统计用的 NaN 语义包装 -------------------------------
# 生产口径对空序列返回 0.0，便于表格展示；但探针把逐月/逐窗指标再求均值时，
# 0.0 会被当成真实观测。下列包装只改**退化输入的返回值**，公式与生产完全一致。
def annual_return_nan(daily_returns: pd.Series,
                      periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """空序列返回 NaN 的年化收益（公式同 :func:`annual_return`）。"""
    r = daily_returns.dropna()
    return float("nan") if r.empty else annual_return(r, periods_per_year)


def sharpe_ratio_nan(daily_returns: pd.Series, rf: float = 0.0,
                     periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """空序列或零波动返回 NaN 的几何夏普（公式同 :func:`sharpe_ratio`）。"""
    r = daily_returns.dropna()
    if r.empty or annual_volatility(r, periods_per_year) <= 0:
        return float("nan")
    return sharpe_ratio(r, rf, periods_per_year)


def sharpe_arithmetic_nan(daily_returns: pd.Series, rf: float = 0.0,
                          periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """空序列或零波动返回 NaN 的算术夏普（公式同 :func:`sharpe_arithmetic`）。"""
    r = daily_returns.dropna()
    if r.empty or float(r.std()) <= 0:
        return float("nan")
    return sharpe_arithmetic(r, rf, periods_per_year)


def _rf_daily(rf: float, periods_per_year: int) -> float:
    """年化 rf → 日频 rf（复利折算），用于超额收益序列。"""
    if rf <= 0:
        return 0.0
    return (1 + rf) ** (1 / periods_per_year) - 1


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0,
                 periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """夏普比率（几何口径：年化超额收益 / 年化波动），**报数用**。"""
    rfd = _rf_daily(rf, periods_per_year)
    excess = (1 + daily_returns) / (1 + rfd) - 1
    vol = annual_volatility(daily_returns, periods_per_year)
    if vol == 0:
        return 0.0
    return annual_return(excess, periods_per_year) / vol


def sharpe_arithmetic(daily_returns: pd.Series, rf: float = 0.0,
                      periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """**算术**夏普 ``mean(excess)/std(excess)·√N``，**换算显著性用**。

    只与 ``t = SR·√T`` 同口径；用几何 SR 套该公式会错配（正收益高估年数、
    负收益/近零低估）。详见模块文档。
    """
    rfd = _rf_daily(rf, periods_per_year)
    excess = (1 + daily_returns.dropna()) / (1 + rfd) - 1
    sd = float(excess.std())
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def years_to_prove_from_sharpe(sr: float, z: float = 1.96) -> float:
    """``(z/|SR|)²`` —— 让 ``t = SR·√T`` 达到 ``z`` 所需的样本年数。

    **必须在算术 SR 上传入才与 t 统计量同口径**；``SR=0`` → ``inf``。
    """
    return round((z / abs(sr)) ** 2, 1) if sr else float("inf")


def sortino_ratio(daily_returns: pd.Series, rf: float = 0.0,
                  periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """索提诺比率：年化收益 / 年化下行偏差。

    下行偏差 = 半方差 ``sqrt(mean(min(excess, 0)²))``，不是「负收益日的 std」
    （后者度量亏损日之间的离散度，系统性高估 |Sortino| 约 30%）。
    """
    rfd = _rf_daily(rf, periods_per_year)
    excess = (1 + daily_returns) / (1 + rfd) - 1
    ar = annual_return(excess, periods_per_year)
    dd = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    if dd == 0:
        return 0.0
    return ar / (dd * np.sqrt(periods_per_year))


def drawdown_series(nav: pd.Series) -> pd.Series:
    """回撤曲线（≤0）：``nav / nav.cummax() - 1``。全项目唯一实现。"""
    return nav / nav.cummax() - 1.0


def max_drawdown(daily_returns: pd.Series) -> float:
    """最大回撤（正值）。"""
    return abs(drawdown_series((1 + daily_returns).cumprod()).min())


def calmar_ratio(daily_returns: pd.Series,
                 periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """卡玛比率：年化收益 / 最大回撤。"""
    mdd = max_drawdown(daily_returns)
    if mdd == 0:
        return 0.0
    return annual_return(daily_returns, periods_per_year) / mdd


def information_ratio(daily_returns: pd.Series, benchmark_returns: pd.Series,
                      periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """信息比率：年化超额 / 年化跟踪误差。"""
    excess = daily_returns - benchmark_returns
    te = excess.std() * np.sqrt(periods_per_year)
    if te == 0:
        return 0.0
    return excess.mean() * periods_per_year / te


def win_rate(daily_returns: pd.Series) -> float:
    """胜率。"""
    if len(daily_returns) == 0:
        return 0.0
    return (daily_returns > 0).sum() / len(daily_returns)


def profit_loss_ratio(daily_returns: pd.Series) -> float:
    """盈亏比：平均盈利 / 平均亏损（绝对值）。"""
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    if len(losses) == 0 or losses.mean() == 0:
        return 0.0
    return wins.mean() / abs(losses.mean())


def turnover_rate(weights_history: pd.DataFrame) -> float:
    """平均单边换手率。"""
    if len(weights_history) < 2:
        return 0.0
    diffs = weights_history.diff().abs().sum(axis=1) / 2
    return diffs.mean()
