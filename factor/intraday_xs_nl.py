"""
日内因子的两个缺维度：横截面相对量 + 时序非线性结构
=====================================================

`factor/intraday_features.py` 的 95 个特征全是「**同日截面绝对量价**」的二阶
统计（mean/std/skew/acf），缺两类结构：

**① 横截面相对量（``xs_*``）**——把个股放到同业里看，而不是只看自己。
现有 `vol_rv`、`pv_vol_skew` 都是个股自身绝对量，同一天小盘股整体波动高、大盘股
整体波动低，绝对量的截面排序里混着「市值/行业」这个共同因子。

**② 时序非线性结构（``nl_*``）**——bar 序列的**依赖结构**，而不是分布形状。
skew/kurt 描述的是边际分布；acf 只看了 1~2 阶线性相关。三条真结构没碰：
波动率聚集（|r| 的自相关）、杠杆效应（涨跌对后续波动的不对称传导）、
路径复杂度（排列熵 / 游程 / 转移概率）与量价交互（Kyle λ）。

设计约束
--------
- 输入与 `intraday_features` 完全同构：``{field: [D, B, C]}`` 稠密块，纯 numpy
  整块向量化，不做 groupby-apply（逐组回调在 421×48 上要分钟级）。
- **无未来函数**：全部只用当日 bar；跨日结构（lag）不在此层——已验证为伪维度。
- NaN 语义沿用 `intraday_features`：有效 bar 不足 ``min_bar_frac * B`` 记 NaN。

🚨 **一条会被反复踩的数学事实**：IC 用 spearman，对**任何单调变换都不敏感**。
所以「截面 rank / z-score 标准化」本身**不可能**产生新信息——
``xs_mkt_z`` 就是专门留的**对照臂**，它的 IC 必须与原始因子**逐位相等**。
只有「按行业分组去均值」这类**打破了全截面单调性**的变换才可能有用。
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from factor.intraday_features import (
    MIN_BAR_FRAC,
    _autocorr,
    _bar_returns,
    _n,
)

#: IC 计算所需最少有效 bar 数（低于则当日该特征记 NaN）
_MIN_BARS = 10


# ----------------------------------------------------------------------
# 向量化原语（沿 bar 轴，[D, B, C] → [D, C]）
# ----------------------------------------------------------------------
def _corr_bar(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """成对 NaN 感知的相关系数（沿 bar 轴）。"""
    m = np.isfinite(x) & np.isfinite(y)
    xx = np.where(m, x, np.nan)
    yy = np.where(m, y, np.nan)
    n = m.sum(axis=1)
    mx = np.nansum(xx, axis=1) / n
    my = np.nansum(yy, axis=1) / n
    dx = xx - mx[:, None, :]
    dy = yy - my[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        num = np.nansum(dx * dy, axis=1)
        den = np.sqrt(np.nansum(dx * dx, axis=1) * np.nansum(dy * dy, axis=1))
        out = num / den
    return np.where(n >= _MIN_BARS, out, np.nan)


def _perm_entropy(x: np.ndarray, order: int = 3) -> np.ndarray:
    """order 阶排列熵（Bandt-Pompe，**并列感知**），归一到 [0, 1]。

    只看 bar 收益的**相对大小序**，对量纲/波动率水平完全不敏感——
    这是真正意义上的「路径复杂度」，skew/kurt/acf 都测不到。

    🚨 **必须用 13 种弱序而不是 6 种严格序**。5 分钟 bar 的零收益占比不低
    （见 `pv_zero_ret_share`），严格序会把「有两个 bar 相等」的三元组
    全部丢弃 ⇒ 交替序列 a,b,a 的熵被算成 **0**（实测踩过），
    即「最有规律的序列」与「最没规律的序列」得分相同。弱序把
    「a=b<c」「a<b=c」等 6 个含并列模式 + 全等 1 个一并编码，无信息丢失。
    """
    if order != 3:
        raise ValueError("当前只实现 order=3")
    a, b, c = x[:, :-2, :], x[:, 1:-1, :], x[:, 2:, :]
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    n = ok.sum(axis=1)
    # 6 严格序 + 6 含并列 + 1 全等 = 13 种弱序（3 元素的全部可能序关系）
    pats = (
        ok & (a < b) & (b < c),                  # a<b<c
        ok & (a < c) & (c < b),                  # a<c<b
        ok & (b < a) & (a < c),                  # b<a<c
        ok & (b < c) & (c < a),                  # b<c<a
        ok & (c < a) & (a < b),                  # c<a<b
        ok & (c < b) & (b < a),                  # c<b<a
        ok & (a == b) & (b < c),                 # a=b<c
        ok & (a == b) & (b > c),                 # a=b>c
        ok & (a < b) & (b == c),                 # a<b=c
        ok & (a > b) & (b == c),                 # a>b=c
        ok & (a == c) & (a < b),                 # a=c<b
        ok & (a == c) & (a > b),                 # a=c>b
        ok & (a == b) & (b == c),                # 全等
    )
    ent = np.zeros(n.shape, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        for p in pats:
            k = p.sum(axis=1)
            pk = k / n
            ent -= np.where(k > 0, pk * np.log(np.where(pk > 0, pk, 1.0)), 0.0)
        ent = ent / np.log(float(len(pats)))
    return np.where(n >= _MIN_BARS, ent, np.nan)


def _markov_asym(r: np.ndarray) -> np.ndarray:
    """符号链的**不对称**持续性 ``P(涨→涨) − P(跌→跌)`` ∈ [−1, 1]。

    正 ⇒ 「涨之后更容易继续涨」强于「跌之后继续跌」（多头惯性占优）；
    负 ⇒ 下跌更有黏性（阴跌惯性）。0 ⇒ 两侧对称。

    🚨 刻意**不用** ``P(涨→涨)+P(跌→跌)−1``：那个在涨跌各半时恒等于
    ``1 − 2×nl_zero_cross_rate``，与翻转频率**完全共线**，是两个名字一个因子。
    这个差值只保留「两端的不对称」，与翻转频率不共线。
    """
    s = np.sign(np.where(np.isfinite(r), r, np.nan))
    a, b = s[:, :-1, :], s[:, 1:, :]
    ok = np.isfinite(r[:, :-1, :]) & np.isfinite(r[:, 1:, :]) & (a != 0) & (b != 0)
    uu = (ok & (a > 0) & (b > 0)).sum(axis=1)
    ud = (ok & (a > 0) & (b < 0)).sum(axis=1)
    dd = (ok & (a < 0) & (b < 0)).sum(axis=1)
    du = (ok & (a < 0) & (b > 0)).sum(axis=1)
    nu, nd = uu + ud, dd + du
    with np.errstate(invalid="ignore", divide="ignore"):
        out = uu / nu - dd / nd
    # 两个方向各至少 3 个起点，否则条件概率是 0/小数的噪声
    return np.where((nu >= 3) & (nd >= 3), out, np.nan)


def _kyle_lambda(r: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Kyle λ：收益对**标准化带符号成交量**的回归斜率（价格冲击强度）。

    带符号量 = volume × sign(r)（主动买为+），先在日内 z 标准化再回归，
    消除成交量量纲与个股规模差别。λ 高 ⇒ 同样一笔主动流推动的价格更大
    ⇒ 流动性更差。`pv_amihud` 只用了 |r|/amount 的均值，不含方向，
    也不捕捉「买盘推动 vs 卖盘推动」的不对称。
    """
    sv = volume * np.sign(np.where(np.isfinite(r), r, 0.0))
    m = np.isfinite(r) & np.isfinite(volume)
    sv = np.where(m, sv, np.nan)
    ok_sv = np.isfinite(sv)
    n = ok_sv.sum(axis=1)
    mu = np.nansum(sv, axis=1) / n
    sd = np.sqrt(np.nansum((sv - mu[:, None, :]) ** 2, axis=1) / n)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (sv - mu[:, None, :]) / sd[:, None, :]
        num = np.nansum(r * z, axis=1)
        den = np.nansum(z * z, axis=1)
        out = num / den
    return np.where(n >= _MIN_BARS, out, np.nan)


def _range_efficiency(r: np.ndarray) -> np.ndarray:
    """路径效率 ``|净位移| / 总路程`` ∈ [0, 1]。

    1 ⇒ 单边直冲（趋势）；0 ⇒ 来回磨（往返）。与 `ret_abs_sum`（总路程）
    和 `mom_intraday_ret`（净位移）单看都不等价——它俩之比才是形状。
    """
    net = np.abs(np.nansum(r, axis=1))
    path = np.nansum(np.abs(r), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = net / path
    return np.where(_n(r) >= _MIN_BARS, out, np.nan)


def _zero_cross_rate(r: np.ndarray) -> np.ndarray:
    """符号翻转频率（穿 0 次数 / 可判定对数）。

    `shape_crossing_cnt` 穿的是**日内均值**（方向依赖整体漂移），
    这个穿 **0**（方向无偏），是纯“来回抖动”度量。
    """
    a, b = r[:, :-1, :], r[:, 1:, :]
    ok = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    n = ok.sum(axis=1)
    flip = (ok & (np.sign(a) != np.sign(b))).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = flip / n
    return np.where(n >= _MIN_BARS, out, np.nan)


def _vol_stability(r: np.ndarray, groups: int = 6) -> np.ndarray:
    """日内波动的**不稳定性**：各时段 RMS 的变异系数。

    同样是全天 RV 相同，波动集中在某一时段（如开盘砸盘后死水）与均匀分布
    是两件事。`vol_extreme_bar_share` 看的是单根 bar 的极端程度，
    这个看的是**时段之间的再分布**。
    """
    D, B, C = r.shape
    k = B // groups
    if k < 2:
        return np.full((D, C), np.nan)
    r2 = np.where(np.isfinite(r), r * r, np.nan)
    blocks = r2[:, :groups * k, :].reshape(D, groups, k, C).mean(axis=2)
    mu = np.nanmean(blocks, axis=1)
    sd = np.nanstd(blocks, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sd / mu
    return np.where(_n(r) >= _MIN_BARS, out, np.nan)


def _updown_vol_ratio(r: np.ndarray) -> np.ndarray:
    """上涨 bar 与下跌 bar 的 RMS 之比（取对数）。

    “涨得猛还是跌得猛”——`vol_rsj` 用二阶矩差，对规模的敏感度不同；
    这个是**尺度无关**的不对称波动比，跨股票可比性更好。
    """
    up = np.where(r > 0, r * r, np.nan)
    dn = np.where(r < 0, r * r, np.nan)
    nu = np.isfinite(up).sum(axis=1)
    nd = np.isfinite(dn).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.sqrt(np.nansum(up, axis=1) / nu)
        md = np.sqrt(np.nansum(dn, axis=1) / nd)
        out = np.log(mu / md)
    ok = (_n(r) >= _MIN_BARS) & (nu >= 3) & (nd >= 3) & (mu > 0) & (md > 0)
    return np.where(ok, out, np.nan)


def _leverage_corr(r: np.ndarray) -> np.ndarray:
    """杠杆效应：``corr(r_t, |r_{t+1}|)``。

    经典杠杆效应是**负**的（跌之后波动更大）。注意这是**跨期**相关，
    与 `vol_vol_corr`（同期量价）和 `shape_autocorr`（r 与 r 的跨期）
    都不是一个东西——它把「方向」和「后续波动的幅度」交叉起来。
    """
    a = r[:, :-1, :]
    b = np.abs(r[:, 1:, :])
    return _corr_bar(a, b)


def _vol_cluster(r: np.ndarray) -> np.ndarray:
    """波动率聚集：``|r|`` 的一阶自相关。

    高 ⇒ 大波动之后还是大波动（GARCH 型记忆）。`shape_autocorr_l1`
    是 **r** 的自相关（方向记忆），这个是 **|r|**（幅度记忆），两回事。
    """
    return _autocorr(np.abs(r), 1)


#: 非线性特征名 → 中文说明（入库/报告用）
NL_DOCS: dict[str, str] = {
    "nl_vol_cluster_acf1": "|bar收益| 一阶自相关（波动率聚集/幅度记忆）",
    "nl_leverage_corr": "corr(r_t, |r_{t+1}|)（杠杆效应，跌后波动更大）",
    "nl_perm_entropy": "bar收益 3 阶排列熵（路径复杂度，归一化 0~1）",
    "nl_markov_asym": "符号链不对称持续性 P(涨→涨)−P(跌→跌)",
    "nl_kyle_lambda": "收益对标准化带符号成交量的回归斜率（价格冲击）",
    "nl_range_efficiency": "|净位移|/总路程（路径效率，趋势 vs 往返）",
    "nl_zero_cross_rate": "符号翻转频率（穿 0 次数占比）",
    "nl_vol_stability": "时段 RMS 的变异系数（日内波动再分布）",
    "nl_updown_vol_ratio": "上涨/下跌 bar 的 RMS 之比（对数，不对称波动）",
}


# ----------------------------------------------------------------------
# 提取入口
# ----------------------------------------------------------------------
def extract_nl_from_block(
    fields: dict[str, np.ndarray],
    days: Sequence[int],
    codes: Sequence[str],
    min_bar_frac: float = MIN_BAR_FRAC,
) -> dict[str, pd.DataFrame]:
    """单块 [D, B, C] → 9 个非线性结构面板。"""
    close, open_ = fields.get("close"), fields.get("open")
    volume = fields.get("volume")
    if close is None or open_ is None:
        raise ValueError("非线性特征需要 close/open")
    B = close.shape[1]
    idx = pd.to_datetime([str(d) for d in days], format="%Y%m%d")
    cols = list(codes)
    r = _bar_returns(close, open_)
    ok = _n(r) >= max(2, int(min_bar_frac * B))

    def panel(a: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(np.where(ok, a, np.nan), index=idx, columns=cols, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {
            "nl_vol_cluster_acf1": panel(_vol_cluster(r)),
            "nl_leverage_corr": panel(_leverage_corr(r)),
            "nl_perm_entropy": panel(_perm_entropy(r)),
            "nl_markov_asym": panel(_markov_asym(r)),
            "nl_kyle_lambda": (panel(_kyle_lambda(r, volume))
                               if volume is not None else None),
            "nl_range_efficiency": panel(_range_efficiency(r)),
            "nl_zero_cross_rate": panel(_zero_cross_rate(r)),
            "nl_vol_stability": panel(_vol_stability(r)),
            "nl_updown_vol_ratio": panel(_updown_vol_ratio(r)),
        }
    return {k: v for k, v in out.items() if v is not None}


def extract_nl_all(store, begin_date: int, end_date: int,
                   block_days: int = 128) -> dict[str, pd.DataFrame]:
    """流式提取全区间非线性特征（与 `extract_all` 同构，按日期拼接）。"""
    parts: dict[str, list[pd.DataFrame]] = {}
    for chunk_days, block in store.iter_blocks(
            ("open", "high", "low", "close", "volume", "amount"),
            block_days=block_days, begin_date=begin_date, end_date=end_date):
        feats = extract_nl_from_block(block, chunk_days, store.codes)
        for name, df in feats.items():
            parts.setdefault(name, []).append(df)
    return {name: pd.concat(frames).sort_index() for name, frames in parts.items()}


# ----------------------------------------------------------------------
# 横截面变体（面板层，需行业 PIT 面板）
# ----------------------------------------------------------------------
#: 横截面变换名 → 说明
XS_DOCS: dict[str, str] = {
    "mkt_z": "全截面 z-score —— 🚨 对照臂，IC 必须与原始因子逐位相等",
    "ind_diff": "减同行业当日中位数（行业相对量）",
    "ind_rank": "行业内百分位排名",
    "ind_z": "行业内 z-score",
}


def cross_variant(panel: pd.DataFrame, industry: pd.DataFrame, how: str) -> pd.DataFrame:
    """把一个 date×code 面板转成横截面变体。

    ``how="mkt_z"`` 是**对照臂**：spearman IC 对它的变换完全不敏感，
    所以它的 IC 必须与原始因子相等。若不等，说明代码对齐/掩码出了问题。
    """
    idx, cols = panel.index, panel.columns
    if how == "mkt_z":
        mu = panel.mean(axis=1)
        sd = panel.std(axis=1)
        return panel.sub(mu, axis=0).div(sd, axis=0)
    if how not in ("ind_diff", "ind_rank", "ind_z"):
        raise ValueError(f"未知横截面变换 {how}")
    if industry is None:
        raise ValueError(f"{how} 需要行业面板")

    ind = industry.reindex(index=idx, columns=cols)
    flat = panel.stack()
    key = ind.stack().reindex(flat.index)
    df = pd.DataFrame({
        "x": flat.to_numpy(dtype=float),
        "ind": key.to_numpy(),
        "dt": flat.index.get_level_values(0),
        "cd": flat.index.get_level_values(1),
    })
    gb = df.groupby(["dt", "ind"], observed=True)["x"]
    if how == "ind_diff":
        df["v"] = df["x"] - gb.transform("median")
    elif how == "ind_rank":
        df["v"] = gb.rank(pct=True)
    elif how == "ind_z":
        mu = gb.transform("mean")
        sd = gb.transform("std")
        df["v"] = (df["x"] - mu) / sd.where(sd > 0)
    else:
        raise ValueError(f"未知横截面变换 {how}")
    return df.pivot(index="dt", columns="cd", values="v").reindex(
        index=idx, columns=cols)