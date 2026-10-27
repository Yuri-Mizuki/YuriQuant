"""
ETF 轮动策略
============

经典动量轮动：对候选 ETF 的截面动量打分排序，等权持有动量最高的前 K 只，
动量转负（入选标的动量均值 ≤ 0）时退回现金。仅做多、不持有空头。

与股票截面策略的差异：轮动只选方向（哪类 ETF），不选标的；空头不对称，
以现金作为"无机会"的默认出口。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.base import Strategy


def _erc_weights(cov: np.ndarray) -> np.ndarray:
    """等风险贡献（ERC）权重：最小化各资产风险贡献的平方偏差。"""
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    var = np.diag(cov).copy()
    if np.any(var <= 0):
        return np.ones(n) / n
    x0 = (1.0 / np.sqrt(var)) / (1.0 / np.sqrt(var)).sum()

    def _obj(w: np.ndarray) -> float:
        port = float(w @ cov @ w)
        if port <= 0:
            return 1e6
        rc = w * (cov @ w) / port
        return float(np.sum((rc - 1.0 / n) ** 2)) * 1e4

    from scipy.optimize import minimize

    res = minimize(_obj, x0, method="SLSQP",
                   bounds=[(1e-4, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 200, "ftol": 1e-10})
    w = res.x if res.success else x0
    w = np.clip(w, 0, None)
    return w / w.sum()


def _quasi_diag(link: np.ndarray) -> list[int]:
    """把 scipy linkage 矩阵转成 HRP 需要的叶子顺序。"""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _hrp_weights(cov: np.ndarray) -> np.ndarray:
    """层次风险平价（HRP，López de Prado 2016）：单链接聚类 + 递归二分。"""
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0):
        return np.ones(n) / n
    corr = cov / np.outer(sd, sd)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    link = linkage(squareform(dist, checks=False), method="single")
    order = _quasi_diag(link)
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        new: list[list[int]] = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            c0, c1 = c[:half], c[half:]
            v0 = float(cov[np.ix_(c0, c0)].sum())
            v1 = float(cov[np.ix_(c1, c1)].sum())
            alpha = 1.0 - v0 / (v0 + v1) if (v0 + v1) > 0 else 0.5
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
            new += [c0, c1]
        clusters = new
    out = w.sort_index().to_numpy()
    return out / out.sum()


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
        abs_threshold: float = 0.0,
        cash_filter: bool = True,
        weighting: str = "equal",
        vol_panel: pd.DataFrame | None = None,
        target_vol: float | None = None,
        enter_rank: int | None = None,
        exit_rank: int | None = None,
        cash_code: str | None = None,
        ret_panel: pd.DataFrame | None = None,
        cov_window: int = 60,
        max_weight: float | None = None,
        dedup_corr: float | None = None,
        dedup_window: int = 120,
        dedup_fill: bool = False,
    ):
        self.top_k = int(top_k)
        self.absolute_filter = bool(absolute_filter)
        self.abs_threshold = float(abs_threshold)
        self.cash_filter = bool(cash_filter)
        self.weighting = str(weighting)
        self.vol_panel = vol_panel
        self.target_vol = float(target_vol) if target_vol is not None else None
        self.enter_rank = int(enter_rank) if enter_rank is not None else None
        self.exit_rank = int(exit_rank) if exit_rank is not None else None
        self.cash_code = cash_code
        self.ret_panel = ret_panel
        self.cov_window = int(cov_window)
        self.max_weight = float(max_weight) if max_weight is not None else None
        self.dedup_corr = float(dedup_corr) if dedup_corr is not None else None
        self.dedup_window = int(dedup_window)
        self.dedup_fill = bool(dedup_fill)
        self._held: list[str] = []
        self.name = (
            f"etf_adv_topk{self.top_k}"
            f"{'_abs' if self.absolute_filter else ''}"
            f"{'_iv' if self.weighting == 'inverse_vol' else ''}"
            f"{'_rd' if self.weighting == 'rank_decay' else ''}"
            f"_{self.weighting}" if self.weighting in ("erc", "hrp") else ""
            f"{'_vt' if self.target_vol else ''}"
            f"{'_buf' if self.enter_rank else ''}"
            f"{'_mw' if self.max_weight else ''}"
            f"{f'_dd{int(self.dedup_corr * 100)}' if self.dedup_corr else ''}"
        )

    # ---- 选标 ----
    def _dedup(self, ranked: list[str], date: pd.Timestamp | None) -> list[str]:
        """按动量从高到低贪心去重：与已选标的的 120 日相关 ≥ 阈值则跳过。

        没有前视——只用 ``date`` 及之前的收益；历史不足 ``dedup_window/2`` 的
        标的相关系数缺失，按不重复处理（保留）。
        """
        if self.dedup_corr is None or self.ret_panel is None or date is None:
            return ranked
        win = self.ret_panel.loc[:date].tail(self.dedup_window)
        win = win.reindex(columns=[c for c in ranked if c in win.columns])
        if len(win) < max(40, self.dedup_window // 2) or win.shape[1] < 2:
            return ranked
        corr = win.corr(min_periods=40)
        keep: list[str] = []
        for c in ranked:
            dup = False
            for k in keep:
                if c in corr.index and k in corr.columns:
                    v = corr.at[c, k]
                    if pd.notna(v) and v >= self.dedup_corr:
                        dup = True
                        break
            if not dup:
                keep.append(c)
        return keep

    def _select(self, vals: pd.Series, date: pd.Timestamp | None = None) -> list[str]:
        if self.absolute_filter:
            vals = vals[vals > self.abs_threshold]
        if vals.empty:
            return []
        ranked = vals.sort_values(ascending=False)
        raw = list(ranked.index)
        ranked = self._dedup(raw, date)
        if self.dedup_fill and len(ranked) < self.top_k:
            # 硬性持仓只数约束（如比赛要求 ≥5 只）：去重后不足时，用原排名补回，
            # 宁可持同质双胞胎也不降低只数。
            ranked = ranked + [c for c in raw if c not in ranked]
        rank = {c: i for i, c in enumerate(ranked)}
        if not self.enter_rank:
            return list(ranked[: self.top_k])

        held = [c for c in self._held if c in rank]
        keep = [c for c in held if rank[c] < self.exit_rank]
        add = [c for c in ranked if c not in keep and rank[c] < self.enter_rank]
        sel = keep + [c for c in add if c not in keep]
        if len(sel) < self.top_k:
            sel += [c for c in ranked if c not in sel]
        return sel[: self.top_k]

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        return self._weights(factor_values, None)

    def get_weights_at(self, date: pd.Timestamp, factor_values: pd.Series) -> pd.Series:
        return self._weights(factor_values, date)

    # ---- 组合构造 ----
    def _risk_parity(self, sel: list[str], date: pd.Timestamp,
                     fallback: pd.Series) -> pd.Series:
        """在``sel``上按协方差矩阵做 ERC / HRP 加权；样本不足时退回``fallback``。

        协方差取``ret_panel``截至``date``（含）最近 ``cov_window`` 个交易日，
        年化（×252）。要求至少 20 个有效交易日，否则不改变原权重。
        """
        sub = self.ret_panel.loc[:date].tail(self.cov_window).reindex(columns=sel)
        sub = sub.dropna(how="all").dropna(axis=1, how="all")
        if len(sub) < 20 or sub.shape[1] != len(sel):
            return fallback
        cov = sub.cov().to_numpy() * 252.0
        w = _erc_weights(cov) if self.weighting == "erc" else _hrp_weights(cov)
        return pd.Series(w, index=sel)

    def _cap_weight(self, w: pd.Series) -> pd.Series:
        """逐项截断到 ``max_weight`` 并把超额按比例回补，直到全部满足。"""
        w = w.clip(lower=0.0)
        total = float(w.sum())
        if total <= 0:
            return w
        w = w / total
        cap = float(self.max_weight)
        if cap <= 0 or cap * len(w) < 1.0:
            return w
        for _ in range(50):
            over = w > cap
            if not over.any():
                break
            excess = float((w[over] - cap).sum())
            w[over] = cap
            free = ~over
            if not free.any():
                break
            w[free] = w[free] + excess * w[free] / float(w[free].sum())
        return w / float(w.sum())

    def _weights(self, factor_values: pd.Series, date: pd.Timestamp | None) -> pd.Series:
        vals = factor_values.dropna()
        if vals.empty:
            self._held = []
            return pd.Series(dtype=float)
        sel = self._select(vals, date)
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
        if self.weighting == "rank_decay" and n > 1:
            # 排名线性衰减：n, n-1, ..., 1 归一化。用于「必须持够 5 只、
            # 但想保留前几名集中度」的规则约束（如比赛要求 ≥5 只产品）。
            base = np.arange(n, 0, -1, dtype=float)
            w = pd.Series(base / base.sum(), index=sel)
        if self.weighting in ("erc", "hrp") and self.ret_panel is not None and date is not None:
            w = self._risk_parity(sel, date, w)
        if self.max_weight is not None and n > 1:
            w = self._cap_weight(w)
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
