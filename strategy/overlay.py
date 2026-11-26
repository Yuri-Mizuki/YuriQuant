"""策略层权重 overlay（2026-11-26 建）。

用途
----
把「目标仓位倍率」乘到内层策略的权重上，降仓部分自然留作现金。
放在策略层而不是事后改收益序列，是为了让**换手率、成本、整手约束都由回测引擎
自己算** —— 事后缩放日收益会漏掉「降仓那 20% 也要付一次交易成本」，也会和整手
约束脱钩。

频率
----
- 倍率索引为 ``PeriodIndex(freq="M")`` 时按**日历月**取（月频决策，次月生效）；
- 索引为 ``PeriodIndex(freq="W-FRI")`` 时按**自然周**取（周频决策，周五收盘判、
  下周一生效）。
- 周频时调仓日会多于选股日：``inner_dates`` 给**该重新选股的日子**（月频调仓的
  执行日）；其余被引擎叫到的日子复用上一次的选股权重，只换倍率 ⇒ 不会把
  「周频降仓」变成「周频重选股」。

约束
----
- 倍率必须由调用方预先算好（本模块不读数据，见 tests/test_layering.py）。
- 只能表达「事前已知」的规则（风格过热、估值分位等）；依赖自身已实现收益的
  规则（如滚动超额断路器）无法用本类表达，必须走事后路径。

用法::

    from strategy.overlay import WeightScaleOverlay
    strat = WeightScaleOverlay(inner=spec.build(**kw), weights=w_weekly,
                               inner_dates=monthly_exec_days)
"""
from __future__ import annotations

import pandas as pd


class WeightScaleOverlay:
    """按倍率缩放内层策略权重（多头），余下部分视为现金。

    Args:
        inner: 内层策略，需实现 ``get_weights_at(date, factor_values)``。
        weights: 倍率序列，``PeriodIndex(freq="M")`` 或 ``"W-FRI"``；缺省按 1.0。
        name: 展示名（回测标签用）。
        inner_dates: 需要**重新选股**的日期集合（引擎调仓执行日）。为 None 时
            每次调用都重新选股（月频 overlay 的既有行为）；给集合时，不在集合
            里的日期复用上一次的选股权重（周频 overlay 的额外调仓日）。
    """

    def __init__(self, inner, weights: pd.Series, name: str = "weight_overlay",
                 inner_dates=None):
        self.inner = inner
        self.weights = weights
        self.name = name
        idx = weights.index
        self._freq = idx.freqstr if isinstance(idx, pd.PeriodIndex) else "M"
        self._inner_dates = (None if inner_dates is None
                             else {pd.Timestamp(d) for d in inner_dates})
        self._last: pd.Series | None = None

    def _scale(self, date) -> float:
        per = pd.Timestamp(date).to_period(self._freq)
        return float(self.weights.get(per, 1.0))

    def get_weights_at(self, date: pd.Timestamp, factor_values: pd.Series) -> pd.Series:
        if self._inner_dates is None or pd.Timestamp(date) in self._inner_dates:
            self._last = self.inner.get_weights_at(date, factor_values)
        elif self._last is None:
            self._last = self.inner.get_weights_at(date, factor_values)
        return self._last * self._scale(date)

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        raise RuntimeError(
            "WeightScaleOverlay 的倍率依赖调仓日；请调用 get_weights_at"
            "（回测引擎默认走它，直接调 get_weights 会静默漏掉 overlay）")
