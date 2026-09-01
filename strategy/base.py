"""
策略基类
========

Strategy: 将因子值转换为组合权重。
- get_weights(factor_table, date) -> pd.Series(code -> weight)

支持的权重模式:
- long_short_topk: 取因子 top-K 做多，bottom-K 做空
- long_only_topk: 只做多 top-K
- long_short_quantile: 按分位做多空
- long_short_all: 全标的多空（因子值归一化为权重）
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """策略抽象基类。"""

    name: str = "base"

    @abstractmethod
    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        """将单日因子值截面转换为权重。

        Args:
            factor_values: Series(index=code, values=因子值)，单日截面。
        Returns:
            Series(index=code, values=权重)，权重之和应接近 0（多空）或 1（纯多头）。
        """
        ...

    def get_weights_at(self, date: pd.Timestamp, factor_values: pd.Series) -> pd.Series:
        """带调仓日的权重计算（默认退化为 get_weights；需要日期的策略可覆写）。"""
        return self.get_weights(factor_values)


