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


def _apply_executable_mask(
    weights: np.ndarray,
    executable: np.ndarray,
) -> np.ndarray:
    """将不可执行标的权重置零，并按多空组独立重新归一化。

    多头（weights > 0）和空头（weights < 0）各自独立归一化，
    保持原有资金在多空两端的分配比例。
    """
    weights = weights.copy()
    weights[~executable] = 0.0

    long_mask = weights > 0
    if long_mask.any():
        weights[long_mask] /= weights[long_mask].sum()

    short_mask = weights < 0
    if short_mask.any():
        weights[short_mask] /= abs(weights[short_mask].sum())

    return weights


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
            executable_mask: DataFrame(index=date, columns=code), dtype=bool，
                            True 表示当日该股票可交易。为 None 时不做过滤。
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

        # 对齐 executable_mask
        if executable_mask is not None:
            executable_mask = executable_mask.reindex(
                index=common_dates, columns=common_codes
            ).fillna(True)

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
            cap_before = capital

            # 1) 调仓: 在调仓日按【当日】因子值设权重。
            #    信号在 close[t] 已知 → 当日建仓 → 捕获 return(t, t+1)（= returns_panel[t]），
            #    与 IC 口径（factor[t] vs returns_panel[t]）严格对齐，无 1 日错位。
            #    （旧实现先按旧权重赚 return[i] 再调仓，致 weights(t) 赚 returns_panel[t+1]，
            #     对 1 日反转因子符号翻转——已于 2026-07-30 修复。）
            if dates[i] in rebalance_days and i < n_days - 1:
                factor_vals = fp.iloc[i].dropna()
                if len(factor_vals) > 0:
                    new_weights = self.strategy.get_weights(factor_vals)
                    new_arr = new_weights.reindex(codes).fillna(0.0).values.astype(np.float64)
                    # 用 executable_mask 将不可执行标的权重归零并按多空组重新归一化
                    if executable_mask is not None:
                        new_arr = _apply_executable_mask(
                            new_arr, executable_mask.iloc[i].values.astype(bool)
                        )
                else:
                    new_arr = np.zeros(n_codes, dtype=np.float64)

                # 换手
                turnover = np.abs(new_arr - current_weights).sum() / 2
                turnover_arr[i] = turnover

                # 成本
                cost = self.costs.calc(current_weights, new_arr, turnover, capital)
                cost_arr[i] = cost
                capital -= cost

                # 更新权重
                current_weights = new_arr
                weights_arr[i] = current_weights

            # 2) 当日收益: 用当前权重 × 当日收益（与 IC 同口径：factor[t] 赚 return(t,t+1)）。
            #    最后一日 returns_panel 通常为 NaN（无未来收益）→ nansum 为 0。
            gross_ret = np.nansum(current_weights * rp_values[i])
            capital *= (1 + gross_ret)

            # 3) 净日收益 = 当日资金变动（毛收益 - 交易成本），与 equity_curve 一致。
            #    使 (1+daily_returns).cumprod() 严格等于 equity_curve。
            daily_ret_arr[i] = (capital / cap_before - 1.0) if cap_before > 0 else 0.0
            equity_arr[i] = capital

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
