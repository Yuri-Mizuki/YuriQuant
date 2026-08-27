"""
ETF 轮动策略
============

经典动量轮动：对候选 ETF 的截面动量打分排序，等权持有动量最高的前 K 只，
动量转负（入选标的动量均值 ≤ 0）时退回现金。仅做多、不持有空头。

与股票截面策略的差异：轮动只选方向（哪类 ETF），不选标的；空头不对称，
以现金作为"无机会"的默认出口。
"""
from __future__ import annotations

import pandas as pd

from strategy.base import Strategy


class EtfRotation(Strategy):
    """动量轮动策略。

    Args:
        top_k: 持仓数量（动量最高的前 K 只）。
        cash_filter: True 时，当入选的 top_k 动量均值 <= 0 则全现金（风向过滤）。
        name: 策略名（自动生成，可覆盖）。
    """

    def __init__(self, top_k: int = 5, cash_filter: bool = True):
        self.top_k = int(top_k)
        self.cash_filter = bool(cash_filter)
        self.name = f"etf_rotation_topk{self.top_k}{'_cf' if self.cash_filter else ''}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if vals.empty:
            return pd.Series(dtype=float)

        k = min(self.top_k, len(vals))
        top = vals.sort_values().index[-k:]

        # 风向过滤：强势资产也整体转负 -> 无赚钱效应，退回现金
        if self.cash_filter and float(vals.loc[top].mean()) <= 0.0:
            return pd.Series(dtype=float)

        return pd.Series(1.0 / k, index=top)

    def get_weights_at(self, date: pd.Timestamp, factor_values: pd.Series) -> pd.Series:
        return self.get_weights(factor_values)