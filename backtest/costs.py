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
            turnover: 单边换手率 (|new - old| 之和 / 2)，即买入额 = 卖出额
            capital: 当前总资金

        Returns:
            成本总额（元）

        说明：
            ``turnover`` 为单边换手率，对应买入金额 = 卖出金额 = turnover*capital。
            佣金 / 滑点对买卖双边计费，印花税仅对卖出单边计费。早期实现三者均少算
            一倍（佣金/滑点按单边、印花税按四分之一边），且零换手时仍按最低佣金扣费。
        """
        # 单边交易金额 = 买入额 = 卖出额
        single_sided_value = float(turnover) * capital
        if single_sided_value <= 0:
            return 0.0
        two_sided_value = 2.0 * single_sided_value  # 买 + 卖

        # 佣金: 买卖双边，不低于最低佣金（仅在实际有交易时收取）
        # 简化: 按总交易额算佣金，实际每笔单独算最低5元
        commission = max(two_sided_value * self.commission_rate, self.commission_min)

        # 印花税: 仅卖出方（单边）
        stamp = single_sided_value * self.stamp_duty

        # 滑点: 买卖双边，按基点
        slip = two_sided_value * self.slippage_bp / 10000

        return commission + stamp + slip
