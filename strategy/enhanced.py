"""指数增强策略 —— 在指数基准权重上叠加信号偏移
====================================================

组合权重 = 指数成分基准权重（锚）+ ETA × 信号偏离。

- ``IndexEnhancedLongOnly``：以指数市值权重为基准（date×code 面板），
  叠加截面标准化的信号偏离，得到"贴近指数 + α 驱动"的权重。
  信号应为**中性化后**的纯 alpha（见 NeutralEnhancedLongOnly / 外部中性化）。

与 TopK 等权组合的本质区别：权重以指数为锚，天然保留指数的市值/行业
暴露（beta），只在锚上做小幅偏离（tilt），因此收益 = 指数 beta +
模型 alpha —— 解决"高 IC 但等权组合跑输指数"的问题。

构造:
    base_weight: date×code 指数基准权重面板（每行和为 1），
                 通常 = 当日成分股市值 / 当日指数总市值。
    eta: 增强强度（默认 0.1），控制信号偏离幅度。权重
         w_i = base_i + eta * (s_i - mean_s)，s 为截面标准化的信号。
"""
from __future__ import annotations

import pandas as pd

from strategy.base import Strategy


class IndexEnhancedLongOnly(Strategy):
    """指数增强纯多头：指数基准权重 + 信号 tilt 偏离。

    信号归一：调用方可传入任何因子/模型面板；策略内对单日截面做 z-score
    （或 rank→[−1,1]），再乘 ETA 叠到基准权重上。若已中性化，信号为纯 alpha，
    tilt 不会改变组合的市值/行业暴露。
    """

    name = "index_enhanced"

    def __init__(
        self,
        base_weight: pd.DataFrame,
        eta: float = 0.1,
        center: str = "zscore",
        min_weight: float | None = None,
    ):
        self.base_weight = base_weight.astype(float)
        self.eta = float(eta)
        self.center = center
        self.min_weight = min_weight

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        raise NotImplementedError(
            "指数增强策略需要调仓日信息，请用 get_weights_at(date, factor_values)"
        )

    def get_weights_at(self, date: pd.Timestamp, factor_values: pd.Series) -> pd.Series:
        """给定调仓日和信号截面，返回指数增强权重（和 ≈ 1）。"""
        base = self.base_weight.loc[date] if date in self.base_weight.index else None
        sig = factor_values.dropna()

        if base is None or len(base) == 0:
            # 无基准权重（如非成分日）：退化为等权（兜底，不应出现在正式调用）
            return pd.Series(1.0 / len(sig), index=sig.index)

        codes = base.index
        b = base.reindex(codes).fillna(0.0)

        # 信号在共同代码上归一
        s = sig.reindex(codes)

        if self.center == "zscore":
            std = s.std()
            s = (s - s.mean()) / std if std and std > 0 else (s - s.mean())
        elif self.center == "rank":
            s = s.rank(pct=True) - 0.5
        s = s.fillna(0.0)

        w = b.add(s * self.eta, fill_value=0.0)

        if self.min_weight is not None:
            w = w.clip(lower=self.min_weight)

        # 长腿归一为纯多头（和=1）；若出现负权重，先做非负裁剪再归一
        if (w < 0).any():
            w = w.clip(lower=0.0)
        total = w.sum()
        if total > 0:
            w = w / total
        return w


__all__ = ["IndexEnhancedLongOnly"]