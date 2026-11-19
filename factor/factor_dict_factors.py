"""飞书《因子字典》候选因子复现（2026-11-19）
============================================

从 606 条「因子字典」里按 覆盖维度 / 数据可得性 / doc IC 三轴筛出的首批候选，
全部按公众号原文的构造实现（原文给的是链接 + 代码片段，没有公式）：

====================  ==========================================================
因子                      构造（原文口径）
====================  ==========================================================
``fd_tcc``            时间网络相对中心度（全截面版）：日收益**全截面** zscore →
                      平方 → 过去 21 日均值 → 取倒数。
``fd_tcc_g``          同一因子的「邪修版」：zscore 改为在 **申万一级行业 × 市值
                      三档**（小 30% / 中 40% / 大 30%）组内做，其余不变。
``fd_grp_abn``        分组异常收益：日收益减去 **行业 × 市值三档** 组内均值后取
                      绝对值，再取过去 21 日**最大值**。
``fd_rnbr``           自信溢出：按代码排序取相邻 10 只（前 4 + 自身 + 后 5）的
                      当日收益均值作 ``NBR_ret``，对自身收益做**带截距**截面 OLS，
                      残差取过去 21 日**标准差**。
``fd_ent30``          单一成交额占比熵（30 分钟）：把 5 分钟 bar 重采样成 30 分钟，
                      ``p_t = close_t/Σclose × vol_t/Σvol``，``-Σ p_t·ln p_t``。
====================  ==========================================================

口径与边界
----------
- 日收益用**后复权 close** 的 ``pct_change()``（原文 ``close × factor``）；
  全部只用当日及更早信息，无未来函数。
- 行业用申万一级 **PIT** 面板（``data.industry``），市值 = PIT 总股本 × 未复权
  close；无行业归属的票不进入分组类因子（不是填 0）。
- ``fd_ent30`` 的 close 是**未复权分钟价**，但分子分母同属一天、只做比值，
  乘性复权因子会约掉，不需要除权掩码。
- 原文作者自己标注：``fd_ent30`` 在 30 分钟最优（60 次之、15 更差、5 分钟很差），
  因此这里固定 30 分钟；本仓库只有 5 分钟面板，故做重采样。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

#: 市值分档切点（与原文 ``for q in [0.3, 0.7, 1]`` 一致：小 30% / 中 40% / 大 30%）
_BUCKET_QS = (0.3, 0.7, 1.0)


def _group_buckets(cap: pd.Series) -> list[np.ndarray]:
    """按市值升序把一只票的位置切成三档，返回每档的**行位置**数组。"""
    order = np.argsort(cap.to_numpy(dtype=float), kind="stable")
    out: list[np.ndarray] = []
    start = 0
    for q in _BUCKET_QS:
        end = int(len(order) * q)
        if end > start:
            out.append(order[start:end])
        start = end
    return out


def _xs_group_stat(ret: pd.Series, ind: pd.Series, cap: pd.Series,
                   mode: str) -> pd.Series:
    """单日截面：行业内按市值分三档，返回组内 zscore（``mode='z'``）或
    对组均值的绝对偏离（``mode='abs_dev'``）。

    与原文一致的两处细节：① 分组**先行业后市值**（组内再分三档）；
    ② 组内 zscore 用样本标准差（``ddof=1``），标准差为 0 / 样本<2 记 NaN。
    """
    df = pd.DataFrame({"r": ret.to_numpy(dtype=float),
                       "ind": ind.to_numpy(),
                       "cap": cap.to_numpy(dtype=float)},
                      index=ret.index)
    ok = np.isfinite(df["r"].to_numpy()) & np.isfinite(df["cap"].to_numpy()) \
        & df["ind"].notna().to_numpy()
    out = pd.Series(np.nan, index=ret.index, dtype=float)
    if not ok.any():
        return out
    sub = df[ok]
    for _, g in sub.groupby("ind", sort=False):
        if len(g) < 2:
            continue
        for pos in _group_buckets(g["cap"]):
            seg = g.iloc[pos]
            mu = float(seg["r"].mean())
            if mode == "z":
                sd = float(seg["r"].std(ddof=1))
                if not np.isfinite(sd) or sd <= 0:
                    continue
                vals = (seg["r"] - mu) / sd
            else:
                vals = (seg["r"] - mu).abs()
            out.loc[seg.index] = vals
    return out


def tcc_panel(ret: pd.DataFrame, industry: pd.DataFrame,
              market_cap: pd.DataFrame, window: int = 21,
              by_group: bool = True) -> pd.DataFrame:
    """时间网络相对中心度（``by_group=False`` 原版 / ``True`` 行业市值分组版）。

    ``ret`` 为当日收益面板（**不 shift**），返回 ``1 / mean_{window}(z²)``。
    取倒数后仍是一个"标准化收益越稳定、因子值越高"的单调量。
    """
    ind = industry.reindex(index=ret.index, columns=ret.columns) \
        if industry is not None else None
    cap = market_cap.reindex(index=ret.index, columns=ret.columns)
    z2 = {}
    for d in ret.index:
        if by_group:
            if ind is None:
                continue
            z = _xs_group_stat(ret.loc[d], ind.loc[d], cap.loc[d], mode="z")
        else:
            r = ret.loc[d]
            r = r[np.isfinite(r.to_numpy(dtype=float))]
            if len(r) < 2:
                continue
            sd = float(r.std(ddof=1))
            if not np.isfinite(sd) or sd <= 0:
                continue
            z = (r - float(r.mean())) / sd
        z2[d] = np.square(z)
    if not z2:
        return pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    z2 = pd.DataFrame(z2).T.reindex(index=ret.index, columns=ret.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 / z2.rolling(window, min_periods=window).mean()
    # 21 日 z² 均值恰为 0（连续 21 天收益完全相同）时倒数是 inf，不是因子值
    return out.replace([np.inf, -np.inf], np.nan)


def group_abnormal_panel(ret: pd.DataFrame, industry: pd.DataFrame,
                         market_cap: pd.DataFrame,
                         window: int = 21) -> pd.DataFrame:
    """分组异常收益 21 日最大：行业内市值三档，``|r − 组均值|`` 后滚动取最大。"""
    ind = industry.reindex(index=ret.index, columns=ret.columns)
    cap = market_cap.reindex(index=ret.index, columns=ret.columns)
    dev = {}
    for d in ret.index:
        dev[d] = _xs_group_stat(ret.loc[d], ind.loc[d], cap.loc[d], mode="abs_dev")
    if not dev:
        return pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    dev = pd.DataFrame(dev).T.reindex(index=ret.index, columns=ret.columns)
    return dev.rolling(window, min_periods=window).max()


def rnbr_panel(ret: pd.DataFrame, nbr: int = 10,
               window: int = 21) -> pd.DataFrame:
    """自信溢出：代码邻接残差，21 日滚动标准差。

    ``nbr`` 为邻域宽度（原文 10：前 4 + 自身 + 后 5）；首尾各 ``nbr//2`` 只用
    边缘 ``nbr`` 只的均值（与原文 ``rtn.iloc[:10].mean()`` 一致）。
    残差用**带截距**的最小二乘，跨全部当日有效股票估计一条斜率。
    """
    half = nbr // 2
    rows = {}
    for d in ret.index:
        r = ret.loc[d]
        r = r[np.isfinite(r.to_numpy(dtype=float))].sort_index()
        n = len(r)
        if n < nbr + half * 2:
            continue
        y = r.to_numpy(dtype=float)
        cs = np.concatenate([[0.0], np.cumsum(y)])
        x = np.empty(n, dtype=float)
        x[:half] = y[:nbr].mean()
        x[n - half:] = y[-nbr:].mean()
        i = np.arange(half, n - half)
        # 原文窗口 rtn.iloc[i-4:i+6]（含 i−4 到 i+5，共 nbr 个）
        x[i] = (cs[i + half + 1] - cs[i - half + 1]) / nbr
        yc, xc = y - y.mean(), x - x.mean()
        vx = float((xc * xc).sum())
        if vx <= 0:
            continue
        beta = float((xc * yc).sum()) / vx
        rows[d] = pd.Series(yc - beta * xc, index=r.index)
    if not rows:
        return pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    res = pd.DataFrame(rows).T.reindex(index=ret.index, columns=ret.columns)
    return res.rolling(window, min_periods=window).std()


def entropy_amount_share(fields: dict[str, np.ndarray],
                         days: Sequence[int],
                         codes: Sequence[str],
                         period: int = 5,
                         target: int = 30) -> pd.DataFrame:
    """单一成交额占比熵（``target`` 分钟口径，本仓库由 5 分钟 bar 重采样）。

    Args:
        fields: 至少含 ``close`` / ``volume``，形状 ``[D, B, C]``。
        days: 与第 0 维对齐的交易日（int YYYYMMDD）。
        codes: 与第 2 维对齐的代码。
        period: 原始 bar 分钟数（面板口径）。
        target: 目标 bar 分钟数（原文 30 分钟最优）。

    Returns:
        date×code 面板（无未来函数，只用当日 bar）。
    """
    close = np.asarray(fields["close"], dtype=np.float64)
    vol = np.asarray(fields["volume"], dtype=np.float64)
    D, B, C = close.shape
    k = int(target) // int(period)
    if k <= 0 or B % k:
        raise ValueError(f"bar 数 {B} 无法按 {target}/{period} 分钟整除重采样")
    nb = B // k
    c3 = close.reshape(D, nb, k, C)
    v3 = vol.reshape(D, nb, k, C)
    # 段内任一 bar 缺失（停牌/半拉天）⇒ 该 30 分钟段不可用
    good = np.isfinite(c3).all(axis=2) & np.isfinite(v3).all(axis=2)
    c30 = np.where(good, c3[:, :, -1, :], np.nan)
    v30 = np.where(good, np.nansum(v3, axis=2), np.nan)
    sc = np.nansum(c30, axis=1, keepdims=True)
    sv = np.nansum(v30, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (c30 / sc) * (v30 / sv)
    ok = np.isfinite(p) & (p > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(ok, p * np.log(np.where(ok, p, 1.0)), 0.0),
                         axis=1)
    # 段不全的日子不出值：否则 nansum 会拿残缺的 8 段算出一个不可比的熵
    ent = np.where(good.all(axis=1), ent, np.nan)
    dates = pd.to_datetime([str(int(d)) for d in days])
    return pd.DataFrame(ent, index=dates, columns=list(codes))


__all__ = [
    "entropy_amount_share", "group_abnormal_panel", "rnbr_panel", "tcc_panel",
]
