"""
交易成本模型
============

支持:
- 佣金: 费率制 + 最低5元
- 印花税: 卖出千1
- 滑点: 按基点扣除

用法:
    costs = TransactionCosts(commission_rate=0.0001, commission_min=5.0,
                             stamp_duty=0.001, slippage_bp=5)
    fee = costs.calc(old_weights, new_weights, prices, shares_outstanding)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TransactionCosts:
    """交易成本计算。"""

    def __init__(
        self,
        commission_rate: float = 0.0001,
        commission_min: float = 5.0,
        stamp_duty: float = 0.001,
        slippage_bp: float = 5.0,
    ):
        self.commission_rate = commission_rate
        self.commission_min = commission_min
        self.stamp_duty = stamp_duty
        self.slippage_bp = slippage_bp

    def calc(
        self,
        old_weights,
        new_weights,
        turnover: float,
        capital: float,
    ) -> float:
        """计算单期交易成本（金额）。

        Args:
            old_weights: 调仓前权重（pd.Series 或 np.ndarray）
            new_weights: 调仓后权重（pd.Series 或 np.ndarray）
            turnover: 单边换手率 (|new - old| 之和 / 2)
            capital: 当前总资金

        Returns:
            成本总额（元）
        """
        # 换手产生的交易金额
        trade_value = float(turnover) * capital

        # 佣金: max(成交额 × 费率, 最低佣金)
        # 简化: 按总交易额算佣金，实际每笔单独算最低5元
        commission = max(trade_value * self.commission_rate, self.commission_min)

        # 印花税: 仅卖出方
        # 简化: 卖出额 = turnover * capital / 2 (一半是卖出)
        sell_value = trade_value / 2
        stamp = sell_value * self.stamp_duty

        # 滑点: 按基点
        slip = trade_value * self.slippage_bp / 10000

        return commission + stamp + slip
