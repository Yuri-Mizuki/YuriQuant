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


def _shrunk_cov(cov: np.ndarray, delta: float = 0.2) -> np.ndarray:
    """Ledoit-Wolf 式收缩：Σ ← (1−δ)Σ + δ·diag(Σ)。

    收缩到对角 = 承认"协方差的非对角项估计噪声最大"。用固定 δ 而非
    样本估计，避免在 n=3、60 日窗这种极小样本上再引入一层估计误差。
    """
    d = np.diag(np.diag(cov))
    return (1.0 - delta) * cov + delta * d


def _inv_cov_weights(cov: np.ndarray, mu: np.ndarray | None = None) -> np.ndarray:
    """``w ∝ Σ⁻¹·mu`` 的统一实现（mu=1 → GMV；mu=σ → 最大分散化）。

    奇异/非正定时加对角抖动重试，仍失败则退化为逆方差。
    """
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    var = np.diag(cov).copy()
    if np.any(var <= 0):
        return np.ones(n) / n
    b = np.ones(n) if mu is None else np.asarray(mu, float)
    eye = np.eye(n) * (var.mean() * 1e-6)
    for jitter in (0.0, 1e-8, 1e-6, 1e-4):
        try:
            inv = np.linalg.inv(cov + eye * (jitter / 1e-6 if jitter else 0.0))
            w = inv @ b
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(w)) and float(w.sum()) > 1e-12:
            w = np.clip(w, 0.0, None)
            if float(w.sum()) > 1e-12:
                return w / float(w.sum())
    w = 1.0 / var
    return w / w.sum()


def _risk_budget_weights(cov: np.ndarray, budget: np.ndarray) -> np.ndarray:
    """风险预算：使各资产的风险贡献占比 ≈ ``budget``（ERC 是其特例 1/n）。"""
    from scipy.optimize import minimize

    n = cov.shape[0]
    b = np.asarray(budget, float)
    b = b / b.sum()
    if n == 1:
        return np.ones(1)
    var = np.diag(cov)
    if np.any(var <= 0):
        return np.ones(n) / n
    x0 = b.copy()

    def _obj(w: np.ndarray) -> float:
        port = float(w @ cov @ w)
        if port <= 0:
            return 1e6
        rc = w * (cov @ w) / port
        return float(np.sum((rc - b) ** 2)) * 1e4

    res = minimize(_obj, x0, method="SLSQP", bounds=[(1e-4, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 200, "ftol": 1e-10})
    w = res.x if res.success else x0
    w = np.clip(w, 0.0, None)
    return w / w.sum()


def _hrp_cluster_equal(cov: np.ndarray) -> np.ndarray:
    """HRP 的聚类骨架 + **簇间/簇内等权**（去掉逆方差分配腿）。

    用来把 HRP 的两个成分拆开：若集中度主要来自「逆方差分配」而非「聚类」，
    这一档的集中度应明显低于 HRP。
    """
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0):
        return np.ones(n) / n
    corr = np.nan_to_num(cov / np.outer(sd, sd), nan=0.0)
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
            w[c0] *= 0.5
            w[c1] *= 0.5
            new += [c0, c1]
        clusters = new
    out = w.sort_index().to_numpy()
    return out / out.sum()


def _hrp_avg_corr(cov: np.ndarray) -> np.ndarray:
    """把相关矩阵整体替换为「平均相关」单因子结构，再做 HRP。

    等价于假设"所有标的共享一个共同因子"，只保留波动率差异。用于检验
    两两相关的估计噪声是否在起反作用（n=3、60 日窗时相关矩阵只有 3 个参数，
    但每个都由 ~60 个观测估出，噪声并不小）。
    """
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0):
        return np.ones(n) / n
    corr = np.nan_to_num(cov / np.outer(sd, sd), nan=0.0)
    iu = np.triu_indices(n, k=1)
    rbar = float(np.mean(corr[iu])) if len(iu[0]) else 0.0
    rbar = float(np.clip(rbar, 0.0, 0.99))
    flat = np.full((n, n), rbar)
    np.fill_diagonal(flat, 1.0)
    cov2 = flat * np.outer(sd, sd)
    return _hrp_weights(cov2)


#: 需要读协方差面板的权重规则（``_risk_parity`` 统一分发）。
#: ``equal`` / ``inverse_vol`` / ``rank_decay`` 只看单只统计量，不在此列。
COV_WEIGHTINGS = frozenset({
    "hrp",          # 定版：层次风险平价（n=3 时实为「压到最不合群那只」）
    "erc",          # 等风险贡献
    "inv_var",      # 逆方差
    "min_var",      # 全局最小方差（GMV）
    "max_div",      # 最大分散化（MDP）
    "risk_budget",  # 风险预算，按动量排名递减分配风险
    "hrp_shrunk",   # 收缩协方差 + HRP（降噪）
    "hrp_avgcorr",  # 单因子相关结构 + HRP（降噪）
    "hrp_ew",       # HRP 聚类骨架 + 等权分配
    "hrp_anticorr",  # HRP 的方向镜像：最大份额压给**最相关**那只（安慰剂）
})


def _hrp_anticorr(cov: np.ndarray) -> np.ndarray:
    """HRP 的**方向镜像**：同样的权重分布，但把最大的那份压到**最相关**的那只。

    用于检验「压最不合群」这个方向本身有没有信息：若 ``hrp`` 与
    ``hrp_anticorr`` 表现接近，说明收益只来自集中度，与「选哪只」无关。
    """
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    w = _hrp_weights(cov)
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0):
        return w
    corr = np.nan_to_num(cov / np.outer(sd, sd), nan=0.0)
    np.fill_diagonal(corr, 1.0)
    mean_corr = (corr.sum(axis=1) - 1.0) / max(n - 1, 1)
    out = np.empty(n)
    out[np.argsort(-mean_corr)] = w[np.argsort(-w)]
    return out


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
        fill_to: int | None = None,
        fill_codes: list[str] | None = None,
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
        self.fill_to = int(fill_to) if fill_to is not None else None
        self.fill_codes = list(fill_codes) if fill_codes else []
        self._held: list[str] = []
        self.name = (
            f"etf_adv_topk{self.top_k}"
            f"{'_abs' if self.absolute_filter else ''}"
            f"{'_iv' if self.weighting == 'inverse_vol' else ''}"
            f"{'_rd' if self.weighting == 'rank_decay' else ''}"
            f"_{self.weighting}" if self.weighting in COV_WEIGHTINGS else ""
            f"{'_vt' if self.target_vol else ''}"
            f"{'_buf' if self.enter_rank else ''}"
            f"{'_mw' if self.max_weight else ''}"
            f"{'_dd' + str(int(self.dedup_corr * 100)) if self.dedup_corr else ''}"
            f"{'_fill' if self.fill_to else ''}"
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
        rule = self.weighting
        if rule == "erc":
            w = _erc_weights(cov)
        elif rule == "inv_var":
            var = np.diag(cov).copy()
            if np.any(var <= 0):
                return fallback
            w = 1.0 / var
            w = w / w.sum()
        elif rule == "min_var":
            w = _inv_cov_weights(cov)
        elif rule == "max_div":
            w = _inv_cov_weights(cov, np.sqrt(np.diag(cov)))
        elif rule == "risk_budget":
            # 风险预算按**动量排名**递减：第 1 名拿 n 份、第 2 名 n−1 份 ……
            # ``sel`` 的顺序就是 ``_select`` 的动量排名（fill 占位腿排在末尾，
            # 自然拿最小预算）。n=3 时 = 50% / 33% / 17% 的风险贡献。
            base = np.arange(len(sel), 0, -1, dtype=float)
            w = _risk_budget_weights(cov, base)
        elif rule == "hrp_shrunk":
            w = _hrp_weights(_shrunk_cov(cov))
        elif rule == "hrp_avgcorr":
            w = _hrp_avg_corr(cov)
        elif rule == "hrp_ew":
            w = _hrp_cluster_equal(cov)
        elif rule == "hrp_anticorr":
            w = _hrp_anticorr(cov)
        else:
            w = _hrp_weights(cov)
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
        # 只数不足时的占位腿：用国债/黄金等低相关资产补足（如比赛要求 ≥5 只）。
        # 占位标的可能不在动量候选内（绝对动量过滤后候选为空），因此从完整
        # 截面索引取，不要求其因子非 NaN。
        fill: list[str] = []
        if self.fill_to and len(sel) < self.fill_to and self.fill_codes:
            fill = [c for c in self.fill_codes if c not in sel]
            fill = fill[: self.fill_to - len(sel)]
        if not sel and not fill:
            self._held = []
            return pd.Series(dtype=float)
        core = vals.reindex(sel).dropna()
        if self.cash_filter and len(core) and float(core.mean()) <= 0.0 and not fill:
            self._held = []
            return pd.Series(dtype=float)
        sel = sel + fill

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
        if self.weighting in COV_WEIGHTINGS and self.ret_panel is not None and date is not None:
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
