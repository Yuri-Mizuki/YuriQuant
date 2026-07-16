"""
向量化回测引擎
==============

日频向量化回测，信号→权重→收益直接矩阵运算。

流程:
1. 输入: 因子值面板(date, code) + 收益率面板(date, code)
2. 按调仓频率(rebalance_freq)取出截面
3. 每个调仓日: 用策略算权重 → 持有至下个调仓日
4. 计算组合日收益 = Σ(weight_i * return_i)
5. 减去交易成本

输出:
- daily_returns: Series(index=date), 组合日收益
- weights_history: DataFrame(index=date, columns=code), 持仓权重
- equity_curve: Series(index=date), 净值曲线
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.costs import TransactionCosts
from backtest.metrics import calc_all_metrics, format_metrics
from config import Config
from strategy.base import Strategy


@dataclass
class BacktestResult:
    """回测结果容器。"""
    daily_returns: pd.Series
    weights_history: pd.DataFrame
    equity_curve: pd.Series
    turnover_series: pd.Series
    cost_series: pd.Series
    config: dict = field(default_factory=dict)

    def metrics(self, benchmark_returns: pd.Series | None = None) -> dict:
        return calc_all_metrics(
            self.daily_returns,
            benchmark_returns,
            self.weights_history,
        )

    def summary(self, benchmark_returns: pd.Series | None = None) -> str:
        m = self.metrics(benchmark_returns)
        return format_metrics(m)


class VectorBacktest:
    """向量化回测引擎。"""

    def __init__(
        self,
        strategy: Strategy,
        rebalance_freq: str = "M",  # D / W / M
        initial_capital: float = 1_000_000.0,
        costs: TransactionCosts | None = None,
    ):
        self.strategy = strategy
        self.rebalance_freq = rebalance_freq
        self.initial_capital = float(initial_capital)
        if costs is None:
            cfg = Config.get().get("backtest", {})
            costs = TransactionCosts(
                commission_rate=cfg.get("commission_rate", 0.0001),
                commission_min=cfg.get("commission_min", 5.0),
                stamp_duty=cfg.get("stamp_duty", 0.001),
                slippage_bp=cfg.get("slippage_bp", 5.0),
            )
        self.costs = costs

    def run(
        self,
        factor_panel: pd.DataFrame,
        returns_panel: pd.DataFrame,
        executable_mask: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """执行回测。

        Args:
            factor_panel: DataFrame(index=date, columns=code), 因子值。
            returns_panel: DataFrame(index=date, columns=code), 日收益率。
                          通常 = close.pct_change()，用次日收益（信号日次日）。
            executable_mask: DataFrame(index=date, columns=code), bool，
                          True=当日可开仓/调仓。为 None 时不做任何限制（原有行为）。
                          用于把停牌、涨跌停封板、不在当日成分股内的标的强制权重置零，
                          避免回测悄悄假设这些不可能成交的仓位能够建立。
        Returns:
            BacktestResult
        """
        # 对齐日期和代码
        common_dates = factor_panel.index.intersection(returns_panel.index)
        common_codes = factor_panel.columns.intersection(returns_panel.columns)
        fp = factor_panel.loc[common_dates, common_codes]
        rp = returns_panel.loc[common_dates, common_codes]

        dates = fp.index
        codes = fp.columns
        n_days = len(dates)
        n_codes = len(codes)

        mask_values = None
        if executable_mask is not None:
            mask_aligned = executable_mask.reindex(index=dates, columns=codes).fillna(False)
            mask_values = mask_aligned.values.astype(bool)

        # 调仓日
        rebalance_days = self._get_rebalance_days(dates)

        # 用 numpy 数组存储结果，避免 pandas 链式赋值问题
        rp_values = rp.values  # (n_days, n_codes)
        fp_values = fp.values

        daily_ret_arr = np.zeros(n_days, dtype=np.float64)
        turnover_arr = np.zeros(n_days, dtype=np.float64)
        cost_arr = np.zeros(n_days, dtype=np.float64)
        equity_arr = np.full(n_days, self.initial_capital, dtype=np.float64)
        weights_arr = np.zeros((n_days, n_codes), dtype=np.float64)

        current_weights = np.zeros(n_codes, dtype=np.float64)
        capital = self.initial_capital

        for i in range(n_days):
            # 1) 当日收益: 用前一日权重 × 当日收益
            if i > 0:
                day_ret = np.nansum(current_weights * rp_values[i])
                daily_ret_arr[i] = day_ret
                capital *= (1 + day_ret)
                equity_arr[i] = capital

            # 2) 调仓: 在调仓日更新权重
            if dates[i] in rebalance_days and i < n_days - 1:
                factor_vals = fp.iloc[i].dropna()
                if len(factor_vals) > 0:
                    new_weights = self.strategy.get_weights(factor_vals)
                    new_arr = new_weights.reindex(codes).fillna(0.0).values.astype(np.float64)
                else:
                    new_arr = np.zeros(n_codes, dtype=np.float64)

                if mask_values is not None:
                    new_arr = _apply_executable_mask(new_arr, mask_values[i])

                # 换手
                turnover = np.abs(new_arr - current_weights).sum() / 2
                turnover_arr[i] = turnover

                # 成本
                cost = self.costs.calc(current_weights, new_arr, turnover, capital)
                cost_arr[i] = cost
                capital -= cost
                equity_arr[i] = capital

                # 更新权重
                current_weights = new_arr
                weights_arr[i] = current_weights

        # 转 pandas
        daily_rets = pd.Series(daily_ret_arr, index=dates, name="portfolio_return")
        equity_curve = pd.Series(equity_arr / self.initial_capital, index=dates, name="equity")
        turnover_s = pd.Series(turnover_arr, index=dates, name="turnover")
        cost_s = pd.Series(cost_arr, index=dates, name="cost")
        weights_df = pd.DataFrame(weights_arr, index=dates, columns=codes)

        return BacktestResult(
            daily_returns=daily_rets,
            weights_history=weights_df,
            equity_curve=equity_curve,
            turnover_series=turnover_s,
            cost_series=cost_s,
            config={
                "strategy": self.strategy.name,
                "rebalance_freq": self.rebalance_freq,
                "initial_capital": self.initial_capital,
            },
        )

    def _get_rebalance_days(self, dates: pd.DatetimeIndex) -> set:
        """根据频率返回调仓日集合。"""
        if self.rebalance_freq == "D":
            return set(dates)
        elif self.rebalance_freq == "W":
            s = pd.Series(dates, index=dates)
            return set(s.groupby(s.index.to_period("W")).first())
        elif self.rebalance_freq == "M":
            s = pd.Series(dates, index=dates)
            return set(s.groupby(s.index.to_period("M")).first())
        else:
            return set(dates)


def _apply_executable_mask(weights: np.ndarray, executable: np.ndarray) -> np.ndarray:
    """把不可执行标的的权重强制置零，再按原符号结构重新归一化。

    多头部分（正权重）归一到原多头权重之和，空头部分（负权重）归一到原空头
    权重之和的绝对值；任何一侧被完全清零时该侧直接保持为 0（不做除零重分配）。
    """
    masked = weights * executable
    out = np.zeros_like(masked)

    pos_mask = masked > 0
    neg_mask = masked < 0

    orig_pos_sum = weights[weights > 0].sum()
    orig_neg_sum = weights[weights < 0].sum()

    masked_pos_sum = masked[pos_mask].sum()
    masked_neg_sum = masked[neg_mask].sum()

    if masked_pos_sum > 0 and orig_pos_sum > 0:
        out[pos_mask] = masked[pos_mask] * (orig_pos_sum / masked_pos_sum)
    if masked_neg_sum < 0 and orig_neg_sum < 0:
        out[neg_mask] = masked[neg_mask] * (orig_neg_sum / masked_neg_sum)

    return out

