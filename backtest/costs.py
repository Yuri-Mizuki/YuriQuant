"""
交易成本模型
============

支持:
- 佣金: 费率制 + 最低5元
- 印花税: 卖出千1
- 滑点: 按基点扣除
- 空头成本（ShortCostModel）: 借券费率按日计提 + 融券保证金占用

用法:
    costs = TransactionCosts(commission_rate=0.0001, commission_min=5.0,
                             stamp_duty=0.001, slippage_bp=5)
    fee = costs.calc(old_weights, new_weights, prices, shares_outstanding)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.metrics import PERIODS_PER_YEAR


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


class EtfCosts(TransactionCosts):
    """ETF 场内交易成本：免印花税、佣金按场内基金口径（更低且最低档 0.1 元）、滑点同股票。

    与股票档（含印花税千 1）不同：ETF 买卖免印花税，且佣金费率与最低门槛更低。
    用于 ETF 轮动回测（VectorBacktest 要求成本对象实现 ``calc`` 接口）。
    """

    def __init__(
        self,
        commission_rate: float = 0.0005,
        commission_min: float = 0.1,
        slippage_bp: float = 5.0,
    ):
        super().__init__(
            commission_rate=commission_rate,
            commission_min=commission_min,
            stamp_duty=0.0,  # ETF 免印花税
            slippage_bp=slippage_bp,
        )


@dataclass
class ShortCostModel:
    """空头腿成本模型：借券费率（按日计提）+ 融券保证金占用（资金效率报告）。

    修正空头腿乐观偏差：旧引擎只计交易成本（佣金/印花税/滑点），空头持仓的
    借券费完全忽略、保证金占用不报告 —— 多空策略（TopKLongShort 等）的收益
    系统性虚高，长持空头时偏差尤其明显。

    - borrow_rate: 年化借券费率。A 股融券常见 8%~10%，个股差异大，可配置；
      设为 0 等价于关闭借券费（旧口径）。
    - margin_ratio: 融券保证金比例，监管最低 100%（=1.0）。保证金占用 =
      多头名义 + 空头名义 × margin_ratio，相对 1 倍资金计。
    - days_per_year: 年化天数（交易日口径）。
    """

    borrow_rate: float = 0.08
    margin_ratio: float = 1.0
    days_per_year: int = PERIODS_PER_YEAR

    def daily_borrow_fee(self, short_exposure: float, capital: float) -> float:
        """按日计提借券费（金额）：空头名义金额 × 年化费率 / 交易日数。

        Args:
            short_exposure: 空头名义敞口，相对资金的权重和（如 1.0 = 满仓做空 1 倍）。
            capital: 当前总资金（元）。
        """
        if short_exposure <= 0.0 or capital <= 0.0 or self.borrow_rate <= 0.0:
            return 0.0
        return short_exposure * capital * self.borrow_rate / self.days_per_year

    def margin_usage(self, long_exposure: float, short_exposure: float) -> float:
        """保证金占用倍数：多头名义 + 空头名义×保证金比例（相对 1 倍资金）。

        > 1 表示组合隐含超过 1 倍资金的杠杆需求（如等权多空各 1 倍 = 2.0，
        需要 2 倍资金才能同时满仓两端）。
        """
        return long_exposure + short_exposure * self.margin_ratio
