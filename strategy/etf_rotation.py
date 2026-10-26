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


class EtfRotationAdvanced(Strategy):
    """ETF 轮动的可配置变体（P1 探针，2026-10-26）。

    在 ``EtfRotation`` 的等权 TopK 之上叠加四个常被引用的改进，全部为**纯多头**：

    - ``absolute_filter``：只保留自身动量 > 0 的标的（Dual Momentum 的绝对腿，
      相对排序之外再要求绝对收益为正）；
    - ``weighting="inverse_vol"``：入选标的按 1/波动率加权；
    - ``target_vol``：按入选标的的平均年化波动缩放总仓位
      （``min(1, target/vol)``，不足 1 的部分留现金；用**事前**波动，不做
      策略自身收益的递归估计）；
    - ``enter_rank`` / ``exit_rank``：缓冲区换手（已持有者排名跌出
      ``exit_rank`` 才卖出，新标的须进入 ``enter_rank`` 才买入）。

    ``vol_panel`` 为 date × code 的年化波动率面板（与因子面板同索引）；
    不传则退化为等权、无波动缩放。
    """

    def __init__(
        self,
        top_k: int = 5,
        absolute_filter: bool = False,
        cash_filter: bool = True,
        weighting: str = "equal",
        vol_panel: pd.DataFrame | None = None,
        target_vol: float | None = None,
        enter_rank: int | None = None,
        exit_rank: int | None = None,
        cash_code: str | None = None,
    ):
        self.top_k = int(top_k)
        self.absolute_filter = bool(absolute_filter)
        self.cash_filter = bool(cash_filter)
        self.weighting = str(weighting)
        self.vol_panel = vol_panel
        self.target_vol = float(target_vol) if target_vol is not None else None
        self.enter_rank = int(enter_rank) if enter_rank is not None else None
        self.exit_rank = int(exit_rank) if exit_rank is not None else None
        self.cash_code = cash_code
        self._held: list[str] = []
        self.name = (
            f"etf_adv_topk{self.top_k}"
            f"{'_abs' if self.absolute_filter else ''}"
            f"{'_iv' if self.weighting == 'inverse_vol' else ''}"
            f"{'_vt' if self.target_vol else ''}"
            f"{'_buf' if self.enter_rank else ''}"
        )

    # ---- 选标 ----
    def _select(self, vals: pd.Series) -> list[str]:
        if self.absolute_filter:
            vals = vals[vals > 0]
        if vals.empty:
            return []
        ranked = vals.sort_values(ascending=False)
        rank = {c: i for i, c in enumerate(ranked.index)}
        if not self.enter_rank:
            return list(ranked.index[: self.top_k])

        held = [c for c in self._held if c in rank]
        keep = [c for c in held if rank[c] < self.exit_rank]
        add = [c for c in ranked.index if c not in keep and rank[c] < self.enter_rank]
        sel = keep + [c for c in add if c not in keep]
        if len(sel) < self.top_k:
            sel += [c for c in ranked.index if c not in sel]
        return sel[: self.top_k]

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        return self._weights(factor_values, None)

    def get_weights_at(self, date: pd.Timestamp, factor_values: pd.Series) -> pd.Series:
        return self._weights(factor_values, date)

    def _weights(self, factor_values: pd.Series, date: pd.Timestamp | None) -> pd.Series:
        vals = factor_values.dropna()
        if vals.empty:
            self._held = []
            return pd.Series(dtype=float)
        sel = self._select(vals)
        if not sel:
            self._held = []
            return pd.Series(dtype=float)
        if self.cash_filter and float(vals.loc[sel].mean()) <= 0.0:
            self._held = []
            return pd.Series(dtype=float)

        n = len(sel)
        w = pd.Series(1.0 / n, index=sel)
        vol = None
        if self.vol_panel is not None and date is not None:
            try:
                vol = self.vol_panel.loc[date].reindex(sel)
            except KeyError:
                vol = None
        if vol is not None and self.weighting == "inverse_vol":
            inv = (1.0 / vol.replace(0.0, float("nan"))).dropna()
            if len(inv) == n:
                w = inv / inv.sum()
        if vol is not None and self.target_vol is not None:
            v = vol.replace(0.0, float("nan")).dropna()
            if len(v):
                scale = min(1.0, self.target_vol / float(v.mean()))
                w = w * scale
                # 引擎在施加可交易掩码后会把多头腿**重新归一化到 1**，
                # 因此「留现金」必须显式落在一个 0 收益的现金资产上。
                if scale < 1.0 and self.cash_code:
                    w[self.cash_code] = 1.0 - float(w.sum())

        self._held = [c for c in w.index if c != self.cash_code]
        return w
