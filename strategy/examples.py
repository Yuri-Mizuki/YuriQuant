"""
策略示例
========

常用策略实现，继承 Strategy。
"""
from __future__ import annotations

import pandas as pd

from strategy.base import Strategy


class TopKLongShort(Strategy):
    """Top-K 多空策略: 做多因子值最大的 K 只，做空最小的 K 只。

    Args:
        k: 每边持仓数量
        weight_mode: 'equal' 等权 / 'factor' 按因子值加权
    """

    def __init__(self, k: int = 30, weight_mode: str = "equal"):
        self.k = k
        self.weight_mode = weight_mode
        self.name = f"topk_ls_{k}_{weight_mode}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if len(vals) < 2 * self.k:
            # 标的不足，按实际数量的 1/3 取
            k = max(1, len(vals) // 3)
        else:
            k = self.k

        sorted_vals = vals.sort_values()
        short_codes = sorted_vals.index[:k]
        long_codes = sorted_vals.index[-k:]

        if self.weight_mode == "equal":
            w_long = pd.Series(1.0 / k, index=long_codes)
            w_short = pd.Series(-1.0 / k, index=short_codes)
        else:
            # 按因子值绝对值归一化
            long_vals = vals.loc[long_codes]
            short_vals = vals.loc[short_codes]
            w_long = long_vals / long_vals.abs().sum()
            w_short = -short_vals.abs() / short_vals.abs().sum()

        return pd.concat([w_long, w_short])


class TopKLongOnly(Strategy):
    """Top-K 纯多头策略: 只做多因子值最大的 K 只。"""

    def __init__(self, k: int = 30, weight_mode: str = "equal"):
        self.k = k
        self.weight_mode = weight_mode
        self.name = f"topk_lo_{k}_{weight_mode}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        k = min(self.k, len(vals))
        long_codes = vals.sort_values().index[-k:]

        if self.weight_mode == "equal":
            return pd.Series(1.0 / k, index=long_codes)
        else:
            long_vals = vals.loc[long_codes]
            return long_vals / long_vals.abs().sum()


class QuantileLongShort(Strategy):
    """分位多空策略: 做多最高分位，做空最低分位。"""

    def __init__(self, n_quantiles: int = 5):
        self.n = n_quantiles
        self.name = f"quantile_ls_{n_quantiles}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if vals.empty:
            return pd.Series(dtype=float)

        # 分位分组
        try:
            groups = pd.qcut(vals, self.n, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(dtype=float)
        n_actual = groups.nunique()
        if n_actual < 2:
            return pd.Series(dtype=float)

        long_mask = groups == groups.max()
        short_mask = groups == groups.min()

        long_codes = vals[long_mask].index
        short_codes = vals[short_mask].index

        w_long = pd.Series(1.0 / len(long_codes), index=long_codes)
        w_short = pd.Series(-1.0 / len(short_codes), index=short_codes)
        return pd.concat([w_long, w_short])
