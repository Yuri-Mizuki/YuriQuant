"""
因子基类
========

Factor 定义:
- name: 因子名称
- calc(panel) -> pd.DataFrame: 输入日频面板(index=date,columns=code)，
  输出与输入同形状的因子值矩阵。
- 所有因子返回 multi-index DataFrame (date, code) 或 panel(date, code)。

panel 数据结构约定:
    index  = 交易日期 (Timestamp)
    columns = 证券代码 (str, 如 '600000.SH')
    values  = 因子值 / 行情值
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd


class Factor(ABC):
    """因子抽象基类。"""

    name: str = "base"

    @abstractmethod
    def calc(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """计算因子值。

        Args:
            panel: 字典，key 为字段名(close/open/volume/...),
                   value 为 DataFrame(index=date, columns=code)。
        Returns:
            DataFrame(index=date, columns=code), 因子值。
        """
        ...

    def __repr__(self) -> str:
        return f"Factor({self.name})"


class FactorEngine:
    """批量计算因子，返回 factor_table(date, code, value)。

    用法:
        engine = FactorEngine()
        engine.register(Momentum(20))
        engine.register(Volatility(20))
        table = engine.run(panel)  # DataFrame: index=date, columns=[code, factor, value]
    """

    def __init__(self):
        self._factors: list[Factor] = []

    def register(self, factor: Factor) -> None:
        self._factors.append(factor)

    def register_many(self, factors: Sequence[Factor]) -> None:
        self._factors.extend(factors)

    def run(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """计算所有因子，返回长表。"""
        results = []
        for f in self._factors:
            vals = f.calc(panel)
            if isinstance(vals.index, pd.MultiIndex):
                vals = vals.unstack("code")
            long = vals.stack().reset_index()
            long.columns = ["date", "code", "value"]
            long["factor"] = f.name
            results.append(long)
        if not results:
            return pd.DataFrame(columns=["date", "code", "factor", "value"])
        out = pd.concat(results, ignore_index=True)
        return out[["date", "code", "factor", "value"]]
