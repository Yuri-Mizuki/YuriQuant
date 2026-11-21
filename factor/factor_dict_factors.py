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
``fd_csad21``          CSAD 模型 ``std_21``：过去 120 日相关矩阵取相关性 Top10 构成板块，
                      板块内 ``CSAD_t = mean|r_i − r_self|``，取最近 21 日 CSAD 的标准差。
``fd_mmd21``           收益率不对称性：过去 60 个日收益与镜像分布 ``−x+2μ`` 的 MMD（RBF 核），
                      再取 21 日均值。**日频近似**（原文用当日分钟收益）。
``fd_tr_resid``        特质换手波动率：换手率对 ``[市场 / SMB / HML / MOM]`` 换手溢价做
                      21 日时序回归，因子 = 残差标准差。
``fd_illiq_ds``        非流动性（DS 改造版）：``|0.6·(H−L)/O + 0.4·(O_t/C_{t−1}−1)|^1.5``
                      ``/ (成交额^0.7 + 5e6)``。
``fd_umr_tr``          UMR2.0（换手风险）：超额收益 × ``(mean_21(换手) − 换手)``，再取 21 日均值。
``fd_umr_trv``         UMR2.0（真实波动风险）：同上，风险代理换成 True Range。
``fd_fear_std``        原始惊恐因子：``std_21(惊恐度 × 收益)``，
                      惊恐度 ``= |r−r_m| / (|r|+|r_m|+0.1)``。
``fd_hf_cs``           CS 价差（分钟）：逐 bar Corwin-Schultz 估计取日内均值，
                      再取 21 日标准差。
``fd_smart_vwap``      聪明钱：成交量 Top10 时刻的后一根 vwap ÷ 全日 vwap。
``fd_net_support``     净支撑成交量：``(支撑量 − 阻力量) / 总股本``，
                      支撑 = 分钟收盘价低于当日分钟均价。
``fd_utr``             优加换手率：量小 ``mean_20(换手)`` 与量稳 ``std_20(换手)``
                      各自去极值/zscore/市值中性化后，按 STR 中位数分两组，
                      稳定组量小降序、不稳定组升序，两段名次相加再归一。
``fd_ind_mom``         行业联合动量 CMC：个股过去 20 日收益排序，取最大 5 日与
                      最小 15 日对应的**行业收益**分别加权求和后相减。
``fd_price_delay``     价格时滞：个股收益对 ``[当期 + 滞后 3 期]`` 市场收益回归，
                      ``1 − R²_受限 / R²_全``（受限 = 滞后项系数置 0）。
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


def csad_std21_panel(ret: pd.DataFrame, window: int = 120, n_top: int = 10,
                     recent: int = 21, min_nonzero: int = 20) -> pd.DataFrame:
    """CSAD 模型 ``std_21``（《基于市场羊群效应的股票alpha研究》）。

    逐日、逐票：过去 ``window`` 日收益相关矩阵 → 取相关性最高的 ``n_top`` 只
    （含自身，自身 ρ=1）构成板块 → 板块内 ``CSAD_t = mean_i |r_i,t − r_self,t|``
    → 取最近 ``recent`` 日 CSAD 的标准差（``ddof=1``）。

    与原文的两处偏差（都为控成本）：
    ① 要求窗口内 **120 日全有效**（原文允许 20 个 NaN）；② 剔除零收益日 ≥20 的
    停牌票（原文同款过滤）。相关系数用 Pearson（原文 ``DataFrame.corr()``）。
    """
    vals = ret.to_numpy(dtype=np.float64)
    T, _N = vals.shape
    out = np.full(vals.shape, np.nan)
    for i in range(window - 1, T):
        blk = vals[i - window + 1: i + 1]
        ok = np.isfinite(blk).all(axis=0)
        if int(ok.sum()) < n_top + 1:
            continue
        cols = np.flatnonzero(ok)
        sub = blk[:, cols]
        nz = (sub != 0).sum(axis=0) >= min_nonzero
        if int(nz.sum()) < n_top + 1:
            continue
        cols, sub = cols[nz], sub[:, nz]
        sd = sub.std(axis=0, ddof=1)
        keep = sd > 0
        if int(keep.sum()) < n_top + 1:
            continue
        cols, sub, sd = cols[keep], sub[:, keep], sd[keep]
        z = ((sub - sub.mean(axis=0)) / sd).astype(np.float32)
        corr = (z.T @ z) / np.float32(window - 1)
        # 每列（每只票）取相关性最高的 n_top 只；自身 ρ=1 必在列内
        neigh = np.argpartition(-corr, n_top - 1, axis=0)[:n_top]
        sector = sub[:, neigh]                       # (window, n_top, n)
        csad = np.abs(sector - sub[:, None, :]).mean(axis=1)   # (window, n)
        out[i, cols] = csad[-recent:].std(axis=0, ddof=1)
    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


def mmd_asym_panel(ret: pd.DataFrame, window: int = 60, smooth: int = 21,
                   gamma: float = 1.0, chunk: int = 512) -> pd.DataFrame:
    """收益率不对称性 MMD（《收益率的非对称分布与尾部蕴含的Alpha》）。

    ``MMD²(x, −x+2μ) = E k(x,x) + E k(y,y) − 2 E k(x,y)``，``k`` 为 RBF 核，
    ``y = −x+2μ`` 是围绕样本均值的镜像分布；样本是**窗口内的单个收益**，
    核均值取窗口内全部 (t,s) 配对（V 统计量，含对角）。

    与原文的两处偏差：
    ① 原文用**当日分钟收益**算 MMD 再 21 日低频化，本仓库 5 分钟面板只有
       48 根/日（原文要求 ≥60），故改为**过去 60 个日收益**的滚动 MMD，再取
       21 日均值——日频近似，不是逐字实现；
    ② 窗口内收益先做**个股标准化**（减均值除样本标准差）再算核：标准化后镜像
       分布恰好是 ``−z``，带宽也随之无量纲（``γ=1`` 对应 ``1/E[d²]=1/2``，
       d 为标准化样本间距）。不标准化的话日收益量级 ~2%，``exp(−γd²)`` 恒为 1，
       因子退化成浮点噪声。
    """
    vals = ret.to_numpy(dtype=np.float64)
    T, _N = vals.shape
    daily = np.full(vals.shape, np.nan)
    # 标准化后两独立样本的平均平方距离为 2，故默认带宽取 1/2
    g = 0.5 * float(gamma)
    for i in range(window - 1, T):
        blk = vals[i - window + 1: i + 1]
        ok = np.isfinite(blk).all(axis=0)
        if int(ok.sum()) < 10:
            continue
        x = blk[:, ok]
        mu = x.mean(axis=0)
        sd = x.std(axis=0, ddof=1)
        keep = sd > 0
        if int(keep.sum()) < 10:
            continue
        cols = np.flatnonzero(ok)[keep]
        # (n, window)：每行是一只票的窗口收益，标准化后镜像分布 = −z
        z = ((x[:, keep] - mu[keep]) / sd[keep]).T.astype(np.float32)
        mmd2 = np.empty(z.shape[0], dtype=np.float64)
        for s in range(0, z.shape[0], chunk):
            zz = z[s: s + chunk]                              # (m, window)
            sq = zz * zz                                      # (m, window)
            # 样本是标量：d²(t,s) = z_t² + z_s² − 2 z_t z_s
            gram = zz[:, :, None] @ zz[:, None, :]            # (m, window, window)
            dxx = np.maximum(sq[:, :, None] + sq[:, None, :] - 2.0 * gram, 0.0)
            # y = −z ⇒ ⟨z_t, y_s⟩ = −z_t z_s，平方距离 = z_t² + z_s² + 2 z_t z_s
            dxy = np.maximum(sq[:, :, None] + sq[:, None, :] + 2.0 * gram, 0.0)
            k_xx = np.exp(-g * dxx, dtype=np.float32).mean(axis=(1, 2))
            k_xy = np.exp(-g * dxy, dtype=np.float32).mean(axis=(1, 2))
            mmd2[s: s + chunk] = np.maximum(2.0 * (k_xx - k_xy), 0.0)
        daily[i, cols] = mmd2
    frame = pd.DataFrame(daily, index=ret.index, columns=ret.columns)
    return frame.rolling(smooth, min_periods=smooth).mean()


def turnover_resid_vol_panel(turnover: pd.DataFrame, pb: pd.DataFrame,
                             market_cap: pd.DataFrame, ret: pd.DataFrame,
                             window: int = 21) -> pd.DataFrame:
    """特质换手波动率（中银《价值组合因子》）。

    每日截面构造四个换手率溢价（与原文 ``cal_premium`` 同）::

        mkt  = 全市场换手均值
        tsmb = 市值后 1/3 换手均值 − 前 1/3
        thml = pb   前 1/3 换手均值 − 后 1/3
        tmom = 过去 21 日收益前 1/3 换手均值 − 后 1/3

    对每只票做 ``turnover ~ 1 + [tmom, thml, tsmb, mkt]`` 的 21 日时序回归，
    因子 = 残差标准差（``ddof=0``，原文 ``np.nanstd``）。窗口内换手必须全有效，
    否则该日不出值。
    """
    tr = turnover.to_numpy(dtype=np.float64)
    cap = market_cap.reindex_like(turnover).to_numpy(dtype=np.float64)
    pbv = pb.reindex_like(turnover).to_numpy(dtype=np.float64)
    mom = pd.DataFrame(np.expm1(np.log1p(ret).rolling(window, min_periods=window).sum()),
                       index=ret.index, columns=ret.columns)
    momv = mom.reindex_like(turnover).to_numpy(dtype=np.float64)
    T, N = tr.shape

    prem = np.full((T, 4), np.nan)
    for i in range(T):
        t = tr[i]
        ok = np.isfinite(t)
        if ok.sum() < 9:
            continue
        out = [np.nan, np.nan, np.nan, float(t[ok].mean())]

        def _third(values: np.ndarray, key: np.ndarray, *, flip: bool) -> float:
            g = np.isfinite(values) & np.isfinite(key)
            if g.sum() < 3:
                return np.nan
            v, k = values[g], key[g]
            order = np.argsort(k, kind="stable")
            v = v[order]
            l = len(v) // 3
            if l < 1:
                return np.nan
            lo, hi = float(v[:l].mean()), float(v[-l:].mean())
            return (lo - hi) if flip else (hi - lo)

        out[0] = _third(t, momv[i], flip=False)   # tmom：收益高 → 高
        out[1] = _third(t, pbv[i], flip=True)     # thml：pb 高 − pb 低
        out[2] = _third(t, cap[i], flip=True)     # tsmb：市值小 − 大
        prem[i] = out

    design = np.column_stack([np.ones(T), prem])
    res = np.full((T, N), np.nan)
    for i in range(window - 1, T):
        xw = design[i - window + 1: i + 1]
        if not np.isfinite(xw).all():
            continue
        y = tr[i - window + 1: i + 1]
        good = np.isfinite(y).all(axis=0)
        if not good.any():
            continue
        yg = y[:, good]
        beta, *_ = np.linalg.lstsq(xw, yg, rcond=None)
        res[i, np.flatnonzero(good)] = (yg - xw @ beta).std(axis=0, ddof=0)
    return pd.DataFrame(res, index=turnover.index, columns=turnover.columns)


def illiq_panel(opn: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                close: pd.DataFrame, amount: pd.DataFrame,
                w_intra: float = 0.6, w_gap: float = 0.4,
                amount_power: float = 0.7,
                amount_floor: float = 5e6) -> pd.DataFrame:
    """非流动性（DeepSeek 改造版 Amihud，日频）。

    ``最优收益率 = 0.6 × (high−low)/open + 0.4 × (open_t/close_{t−1} − 1)``，
    因子 ``= |最优收益率|^1.5 / (成交额^0.7 + 5e6)``。

    原文对成交额做过分段处理（低于 21 日 25% 分位乘 2、高于 75% 分位开方
    乘 1.5、中间取 0.7 次方），作者实测「直接取 0.7 次方」更好，这里按他的
    最终选择实现。价格用**后复权**序列：日内比值与隔夜跳空在乘性复权下不变，
    但能避免除权日被误判成跳空。
    """
    gap = opn / close.shift(1) - 1.0
    intraday = (high - low) / opn
    best = w_intra * intraday + w_gap * gap
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = best.abs().pow(1.5) / (amount.clip(lower=0).pow(amount_power)
                                     + amount_floor)
    return out.replace([np.inf, -np.inf], np.nan)


def umr_panel(excess_ret: pd.DataFrame, risk: pd.DataFrame,
              window: int = 21) -> pd.DataFrame:
    """UMR2.0 动量反转统一框架（东方证券 2023，原文 d=m=21）。

    ``risk_adj = mean_{window}(risk) − risk_t``：高于自身近期均值的风险日
    收益被视为风险补偿、未来偏反转，低于均值的偏动量。
    因子 ``= mean_{window}(excess_ret × risk_adj)``。

    ``risk`` 传换手率得 ``tr_umr``，传真实波动得 ``true_range_umr``。
    ``excess_ret`` 为相对**市值加权**市场收益的超额。
    """
    adj = risk.rolling(window, min_periods=window).mean() - risk
    return (excess_ret * adj).rolling(window, min_periods=window).mean()


def fear_panel(ret: pd.DataFrame, market_ret: pd.Series | None = None,
               window: int = 21, eps: float = 0.1) -> pd.DataFrame:
    """原始惊恐因子（方正证券 2022《草木皆兵》）。

    ``惊恐度 = |r_i − r_m| / (|r_i| + |r_m| + eps)``，
    因子 ``= std_{window}(惊恐度 × 个股收益)``。

    ``r_m`` 缺省取全 A 等权日收益（与库内基准同口径）。原文比较了均值 /
    标准差 / 两者等权三种合成，实测标准差最好，这里取标准差（``ddof=1``）。
    """
    rm = (ret.mean(axis=1) if market_ret is None else market_ret).reindex(ret.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        fear = ret.sub(rm, axis=0).abs() / (ret.abs().add(rm.abs(), axis=0) + eps)
    return (fear * ret).rolling(window, min_periods=window).std()


def _minute_frame(arr: np.ndarray, days: Sequence[int],
                  codes: Sequence[str]) -> pd.DataFrame:
    dates = pd.to_datetime([str(int(d)) for d in days])
    return pd.DataFrame(arr, index=dates, columns=list(codes))


def cs_spread_panel(fields: dict[str, np.ndarray], days: Sequence[int],
                    codes: Sequence[str], window: int = 21,
                    min_bars: int = 24) -> pd.DataFrame:
    """CS 价差（Corwin-Schultz）分钟频版 → 21 日标准差。

    逐 bar 对 ``(b−1, b)`` 计算：``beta = mean(log(h/l)²)``、
    ``gamma = log(max(h)/min(l))²``，``num = 3−2√2``，
    ``alpha = (√(2β)−√β)/num − √(γ/num)``，``S = 2e^{α−1}/(1+e^{α})``。
    日内取 ``S`` 均值得到日值，再取 ``window`` 日标准差（原文：分钟数据用
    标准差合成优于均值，且标准差合成的 IC 为负）。
    """
    high = np.asarray(fields["high"])
    low = np.asarray(fields["low"])
    parts = []
    for i in range(0, high.shape[0], _BAR_BLOCK_DAYS):
        parts.append(_cs_spread_daily(high[i:i + _BAR_BLOCK_DAYS],
                                      low[i:i + _BAR_BLOCK_DAYS], min_bars))
    daily = np.vstack(parts)
    return _minute_frame(daily, days, codes).rolling(
        window, min_periods=window).std()


#: bar 级中间量按天分块的大小：64 天 × 48 bar × 5,692 码 float64 ≈ 140 MB，
#: 全 A 5.3 亿格整表展开 ~4.25 GB/步必 OOM（2026-11-20 实测）。
_BAR_BLOCK_DAYS = 64


def _cs_spread_daily(high: np.ndarray, low: np.ndarray,
                     min_bars: int = 24) -> np.ndarray:
    """CS 价差日值内核：[B, n_bars, C] → [B, C]。

    bar 级中间量**日内封闭**（beta/gamma 只耦合同日相邻 bar），按天分块计算
    与整表计算逐位等价；float64 上转换放到块内，全 A 规模下不爆内存。
    """
    h = np.asarray(high, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    ok = np.isfinite(h) & np.isfinite(lo) & (h > 0) & (lo > 0)
    h = np.where(ok, h, np.nan)
    lo = np.where(ok, lo, np.nan)
    lh2 = np.log(h / lo) ** 2
    beta = np.full_like(h, np.nan)
    beta[:, 1:, :] = 0.5 * (lh2[:, 1:, :] + lh2[:, :-1, :])
    hi2 = np.fmax(h[:, 1:, :], h[:, :-1, :])
    lo2 = np.fmin(lo[:, 1:, :], lo[:, :-1, :])
    gamma = np.full_like(h, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma[:, 1:, :] = np.log(hi2 / lo2) ** 2
        num = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / num - np.sqrt(gamma / num)
        s = 2.0 * np.exp(alpha - 1.0) / (1.0 + np.exp(alpha))
    cnt = np.isfinite(s).sum(axis=1)
    daily = np.nansum(s, axis=1) / np.maximum(cnt, 1)
    return np.where(cnt >= min_bars, daily, np.nan)


def smart_vwap_panel(fields: dict[str, np.ndarray], days: Sequence[int],
                     codes: Sequence[str], top: int = 10, skip: int = 3,
                     follow: int = 1, dedupe: int = 2) -> pd.DataFrame:
    """聪明钱「跟随时刻 vwap / 全日 vwap」（5 分钟近似）。

    原文用 1 分钟数据：去掉前 15 分钟 → 取成交量最大的 10 个时刻 →
    相距不足 5 分钟只留前一个 → 跟随时刻（紧随的 5 分钟）成交额/成交量
    作为 vwap → 除以全日 vwap。

    本仓库只有 5 分钟 bar，故 ``skip=3``（去掉前 15 分钟）、``follow=1``
    （紧随 1 根，即 5 分钟）、``dedupe=2``（10 分钟内算同一簇）。
    """
    vol = np.asarray(fields["volume"])
    amt = np.asarray(fields["amount"])
    D, B, C = vol.shape
    out = np.full((D, C), np.nan)
    for d in range(D):
        # float64 上转换放逐日内（整表展开 ~8.5 GB 会 OOM；float32→float64
        # 精确无损失，逐日结果与整表上转换逐位一致）
        v_all, a_all = vol[d].astype(np.float64), amt[d].astype(np.float64)
        v, a = v_all[skip:], a_all[skip:]
        nb = v.shape[0]
        if nb <= 0:
            continue
        dv = np.nansum(v_all, axis=0)
        da = np.nansum(a_all, axis=0)
        order = np.argsort(-np.where(np.isfinite(v), v, -np.inf), axis=0)[:top]
        for c in range(C):
            if not (np.isfinite(dv[c]) and dv[c] > 0 and da[c] > 0):
                continue
            pos = np.sort(order[:, c])
            pos = pos[np.isfinite(v[pos, c])]
            if pos.size == 0:
                continue
            keep = [int(pos[0])]
            for p in pos[1:]:
                if p - keep[-1] > dedupe:
                    keep.append(int(p))
            fv = fa = 0.0
            for p in keep:
                q = slice(p + 1, min(p + 1 + follow, nb))
                fv += np.nansum(v[q, c])
                fa += np.nansum(a[q, c])
            if fv > 0 and fa > 0:
                out[d, c] = (fa / fv) / (da[c] / dv[c])
    return _minute_frame(out, days, codes)


def net_support_panel(fields: dict[str, np.ndarray], days: Sequence[int],
                      codes: Sequence[str], shares: np.ndarray,
                      min_bars: int = 24) -> pd.DataFrame:
    """净支撑成交量：(支撑量 − 阻力量) / 总股本。

    日内收盘价低于当日分钟收盘价均值的成交量计入支撑，高于的计入阻力
    （《"量价淘金"系列》净支撑量因子）。原文用流通股本，本仓库缓存无流通股，
    改用 PIT 总股本（与内部换手率同口径）。
    """
    close = np.asarray(fields["close"])
    vol = np.asarray(fields["volume"])
    shares = np.asarray(shares, dtype=np.float64)
    parts = []
    for i in range(0, close.shape[0], _BAR_BLOCK_DAYS):
        j = i + _BAR_BLOCK_DAYS
        parts.append(_net_support_daily(close[i:j], vol[i:j], shares[i:j],
                                        min_bars))
    return _minute_frame(np.vstack(parts), days, codes)


def _net_support_daily(close: np.ndarray, vol: np.ndarray,
                       shares: np.ndarray, min_bars: int = 24) -> np.ndarray:
    """净支撑量日值内核：[B, n_bars, C] → [B, C]（日内封闭，同 cs 内核）。"""
    c = np.where(np.isfinite(close), close.astype(np.float64), np.nan)
    v = np.where(np.isfinite(vol), vol.astype(np.float64), np.nan)
    cnt = np.isfinite(c).sum(axis=1)
    mean_c = (np.nansum(c, axis=1, keepdims=True)
              / np.maximum(cnt, 1)[:, None, :])
    sup = np.nansum(np.where(c < mean_c, v, 0.0), axis=1)
    res = np.nansum(np.where(c > mean_c, v, 0.0), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (sup - res) / shares
    return np.where(cnt >= min_bars, val, np.nan)


def _xs_neutralize(x: pd.Series, cap: pd.Series,
                   q: float = 0.01) -> pd.Series:
    """截面 去极值(q/q) → zscore → 对 log(市值) 回归取残差。

    原文（东吴 2021，优加换手率）对量小/量稳换手率各自做「横截面去极值、
    标准化、市值中性化」，且强调**顺序不能颠倒**（先分别中性化再合成）。
    """
    good = x.notna() & cap.notna()
    if good.sum() < 10:
        return pd.Series(np.nan, index=x.index)
    v = x[good]
    lo, hi = v.quantile(q), v.quantile(1 - q)
    v = v.clip(lo, hi)
    sd = v.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    z = (v - v.mean()) / sd
    lc = np.log(cap[good].clip(lower=1e-6))
    X = np.c_[np.ones(len(lc)), lc.to_numpy(dtype=float)]
    beta = np.linalg.lstsq(X, z.to_numpy(dtype=float), rcond=None)[0]
    return pd.Series(X @ beta, index=v.index).reindex(x.index)


def utr_panel(turnover: pd.DataFrame, market_cap: pd.DataFrame,
              window: int = 20) -> pd.DataFrame:
    """优加换手率 UTR（东吴 2021《解决 1+1＜2 的难题》）。

    量小 ``Turn20 = mean_20(换手)``、量稳 ``STR = std_20(换手)``，两者分别做
    去极值 / 标准化 / 市值中性化；当日按 STR 升序给分 ``Score1``，再按 STR
    中位数切两半——**稳定组**（前半）Turn20 降序给分、**不稳定组**（后半）
    升序给分，``UTR = (Score1 + Score2) / N``。原文方向：**UTR 越小越好**。

    换手率口径为 成交量 / PIT 总股本（缓存无自由流通股）。
    """
    turn = turnover.rolling(window, min_periods=window).mean()
    stab = turnover.rolling(window, min_periods=window).std()
    out = pd.DataFrame(np.nan, index=turn.index, columns=turn.columns)
    for dt in turn.index:
        t = _xs_neutralize(turn.loc[dt], market_cap.loc[dt])
        s = _xs_neutralize(stab.loc[dt], market_cap.loc[dt])
        pair = pd.concat([t.rename("t"), s.rename("s")], axis=1).dropna()
        if len(pair) < 20:
            continue
        pair = pair.sort_values("s", kind="mergesort")
        n = len(pair)
        score1 = pd.Series(np.arange(1, n + 1), index=pair.index)
        mid = n // 2
        score2 = pd.Series(np.nan, index=pair.index)
        score2.iloc[:mid] = pair["t"].iloc[:mid].rank(ascending=False)
        score2.iloc[mid:] = pair["t"].iloc[mid:].rank(ascending=True)
        out.loc[dt, pair.index] = (score1 + score2) / n
    return out


def ind_joint_momentum_panel(ret: pd.DataFrame, market_cap: pd.DataFrame,
                             industry: pd.DataFrame, window: int = 20,
                             top: int = 5, bottom: int = 15) -> pd.DataFrame:
    """行业联合动量 CMC（国信 2024《个股与行业的共振》）。

    行业内按市值加权得到行业收益 ``sector_ret``；对个股过去 ``window`` 日
    收益升序排列，取**最大 5 日**对应的行业收益按 ``w5 = [0.5,0.59,0.7,0.84,1]``
    加权得 ``ICM``，取**最小 15 日**按 ``2^(-i/14)`` 加权得 ``ICR``，
    ``CMC = ICM − ICR``。原文权重不归一化；行业收益用申万一级内市值加权
    （原文用中信一级指数，作者实测等权效果更差）。
    """
    w5 = np.array([0.5, 0.59, 0.7, 0.84, 1.0])[:top]
    w15 = np.array([2.0 ** (-(i) / 14.0) for i in range(bottom)])
    # 行业收益：逐日行业内市值加权
    sec = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    for dt in ret.index:
        r = ret.loc[dt]
        ind = industry.loc[dt] if dt in industry.index else None
        cap = market_cap.loc[dt]
        if ind is None:
            continue
        g = pd.DataFrame({"r": r, "c": cap, "i": ind}).dropna()
        if g.empty:
            continue
        w = g["c"] / g.groupby("i")["c"].transform("sum")
        sec.loc[dt, g.index] = (g["r"] * w).groupby(g["i"]).transform("sum")
    r_arr, s_arr = ret.to_numpy(dtype=float), sec.to_numpy(dtype=float)
    n_day, n_col = r_arr.shape
    out = np.full((n_day, n_col), np.nan)
    for k in range(window - 1, n_day):
        R = r_arr[k - window + 1:k + 1]
        S = s_arr[k - window + 1:k + 1]
        valid = np.isfinite(R) & np.isfinite(S)
        cnt = valid.sum(axis=0)
        key = np.where(valid, R, np.inf)
        order = np.argsort(key, axis=0, kind="stable")
        cols = np.arange(n_col)[None, :]
        bot = S[order[:bottom], cols] * valid[order[:bottom], cols] * w15[:, None]
        topv = S[order[-top:], cols] * valid[order[-top:], cols] * w5[:, None]
        icr, icm = bot.sum(axis=0), topv.sum(axis=0)
        ok = cnt >= max(top + bottom - 2, 10)
        out[k, ok] = (icm - icr)[ok]
    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


def price_delay_panel(ret: pd.DataFrame, market_ret: pd.Series | None = None,
                      window: int = 21, lags: int = 3,
                      min_valid: int = 15) -> pd.DataFrame:
    """价格时滞（Hou-Moskowitz 2005 / 东方证券 2015）。

    以个股日收益为 y、``[1, 当期市场收益, 滞后 1..lags 期市场收益]`` 为 x 做
    时序回归：全模型 R² 与「滞后项系数全置 0」的受限 R² 之比，
    因子 ``= 1 − R²_受限 / R²_全``。时滞越大、对市场信息的反应越慢，因子越大。

    逐列用带掩码的矩矩阵批量求解（不用逐股循环）；有效样本 < ``min_valid``
    或 R² 全 ≤0 时不出值。``market_ret`` 缺省取全 A 等权。
    """
    rm = (ret.mean(axis=1) if market_ret is None else market_ret).reindex(ret.index)
    x = np.column_stack([np.ones(len(ret))] + [
        rm.shift(j).to_numpy(dtype=float) for j in range(lags + 1)])
    ok_x = np.isfinite(x).all(axis=1)
    Y = ret.to_numpy(dtype=float)
    n_day, n_col = Y.shape
    out = np.full((n_day, n_col), np.nan)
    for k in range(window - 1, n_day):
        rows = np.arange(k - window + 1, k + 1)
        X, Yw = x[rows], Y[rows]
        if not ok_x[rows].all():
            continue
        M = np.isfinite(Yw).astype(float)
        Yf = np.nan_to_num(Yw)
        XtX = np.einsum("rn,rp,rq->npq", M, X, X)
        XtY = np.einsum("rn,rp,rn->np", M, X, Yf)
        cnt = M.sum(axis=0)
        good = cnt >= min_valid
        if not good.any():
            continue
        XtX, XtY, Yw2, M2 = XtX[good], XtY[good], Yw[:, good], M[:, good]
        try:
            beta = np.linalg.solve(XtX + 1e-10 * np.eye(X.shape[1]),
                                   XtY[..., None])[..., 0]
            beta_r = np.linalg.solve(
                XtX[:, :2, :2] + 1e-10 * np.eye(2),
                XtY[:, :2, None])[..., 0]
        except np.linalg.LinAlgError:
            continue
        fit = np.einsum("rp,np->rn", X, beta)
        fit_r = np.einsum("rp,np->rn", X[:, :2], beta_r)
        cnt_g = cnt[good]
        sse = (M2 * (Yw2 - fit) ** 2).sum(axis=0)
        sse_r = (M2 * (Yw2 - fit_r) ** 2).sum(axis=0)
        mean_y = (M2 * Yw2).sum(axis=0) / cnt_g
        sst = (M2 * (Yw2 - mean_y) ** 2).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = 1.0 - sse / sst
            r2_r = 1.0 - sse_r / sst
            val = 1.0 - r2_r / r2
        val = np.where((r2 > 1e-8) & np.isfinite(val), val, np.nan)
        idx = np.flatnonzero(good)
        out[k, idx] = val
    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


__all__ = [
    "cs_spread_panel", "csad_std21_panel", "entropy_amount_share", "fear_panel",
    "group_abnormal_panel", "illiq_panel", "ind_joint_momentum_panel",
    "mmd_asym_panel", "net_support_panel", "price_delay_panel", "rnbr_panel",
    "smart_vwap_panel", "tcc_panel", "turnover_resid_vol_panel", "umr_panel",
    "utr_panel",
]
