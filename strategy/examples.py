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
        if len(vals) < 2:
            # 只有 1 只：纯多头，避免 long/short 重叠导致权重 index 重复
            return pd.Series(1.0, index=vals.index)
        if len(vals) < 2 * self.k:
            # 标的不足：每边最多取 len//2，保证多空不重叠
            k = max(1, min(self.k, len(vals) // 2))
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


class TopFracLongOnly(Strategy):
    """Top-Frac 重仓多头: 按比例 selected top-股做等权多头。

    与 TopKLongOnly（固定 k）的区别：k = round(frac * 当日有效信号数)，
    随股票池规模自适应。研究结论：中性化后的模型信号用 Top20% 重仓多头
    是 alpha 变现充分的组合形式（gbdt_tune 实验：frac=0.20,horizon=1
    于 2025 test 段成本后超额 +4.85%）。
    """

    name = "topfrac_lo"

    def __init__(self, frac: float = 0.20, weight_mode: str = "equal"):
        if not 0 < frac <= 1:
            raise ValueError(f"frac 必须在 (0,1]: {frac}")
        self.frac = float(frac)
        self.weight_mode = weight_mode
        self.name = f"topfrac_lo_{frac:.2f}_{weight_mode}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if len(vals) == 0:
            return pd.Series(dtype=float)
        k = max(1, min(round(self.frac * len(vals)), len(vals)))
        top = vals.sort_values().index[-k:]

        if self.weight_mode == "equal":
            return pd.Series(1.0 / k, index=top)
        top_vals = vals.loc[top]
        return top_vals / top_vals.abs().sum()

    def get_weights_at(self, date: pd.Timestamp, factor_values: pd.Series) -> pd.Series:
        return self.get_weights(factor_values)


class BufferedTopFracLongOnly(Strategy):
    """带排名缓冲带的 Top-Frac 多头（低换手变体）。

    指数式缓冲带规则（每期目标持仓数 N = round(frac_entry × 有效截面数)）：
    - 现有持仓：排名仍在前 frac_exit × n 名以内则保留（[entry, exit) 缓冲带内不换出）；
    - 空缺名额由排名最高的非持仓补足（仅考虑排名在 frac_entry × n 以内的候选）；
    - 截面超量收缩导致保留数 > N 时，按排名裁到 N；
    - 信号缺失（当日截面无值）的持仓卖出（无法排名；停牌场景由引擎
      executable_mask 另行处理）；
    - 等权 1/N。

    目的：TopFracLongOnly 每期按分数硬切 TopN，截止线附近的排名噪声直接变成
    交易（2025 实测月频单次单边换手 66%，其中大量无信息增量）。缓冲带把
    "进出门槛"分离，换手随 (frac_exit − frac_entry) 带宽下降。

    **有状态**：引擎按调仓日时序逐次调用 get_weights，实例内部维护上一期持仓。
    每个 VectorBacktest 必须使用全新实例——跨回测复用同一实例会把上一次回测的
    期末持仓泄漏为本次期初持仓（前视）。
    """

    def __init__(self, frac_entry: float = 0.20, frac_exit: float = 0.30):
        if not 0 < frac_entry <= 1:
            raise ValueError(f"frac_entry 必须在 (0,1]: {frac_entry}")
        if not frac_entry <= frac_exit <= 1:
            raise ValueError(
                f"需要 0 < frac_entry <= frac_exit <= 1: entry={frac_entry}, exit={frac_exit}")
        self.frac_entry = float(frac_entry)
        self.frac_exit = float(frac_exit)
        self.name = f"buffered_topfrac_lo_{frac_entry:.2f}_{frac_exit:.2f}"
        self._prev: set = set()

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if len(vals) == 0:
            self._prev = set()
            return pd.Series(dtype=float)
        n = len(vals)
        n_entry = max(1, min(round(self.frac_entry * n), n))
        n_exit = max(n_entry, min(round(self.frac_exit * n), n))
        ranks = vals.rank(ascending=False, method="first")

        keep = [c for c in self._prev if c in ranks.index and ranks[c] <= n_exit]
        candidates = [c for c in ranks.sort_values().index
                      if c not in set(keep) and c not in self._prev and ranks[c] <= n_entry]
        portfolio = keep + candidates[:max(0, n_entry - len(keep))]
        if len(portfolio) > n_entry:
            portfolio = sorted(portfolio, key=lambda c: ranks[c])[:n_entry]

        self._prev = set(portfolio)
        return pd.Series(1.0 / len(portfolio), index=portfolio)


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


def build_strategy(name: str, k: int = 30, frac: float | None = None):
    """按名称构造策略实例（run_backtest / select_stocks 共用工厂，2026-08-05 统一）。

    Args:
        name: topk_ls（TopK 多空）| topk_lo（TopK 纯多）| quantile（分位多空）
              | topfrac_lo（Top-Frac 比例多头）
        k: TopK 持仓数（quantile 忽略；topfrac_lo 忽略）。
        frac: topfrac_lo 持仓比例（默认 0.20）。
    """
    if name == "topk_ls":
        return TopKLongShort(k=k)
    elif name == "topk_lo":
        return TopKLongOnly(k=k)
    elif name == "quantile":
        return QuantileLongShort(n_quantiles=5)
    elif name == "topfrac_lo":
        return TopFracLongOnly(frac=frac if frac is not None else 0.20)
    raise ValueError(f"未知策略: {name}（可选 topk_ls / topk_lo / quantile / topfrac_lo）")
