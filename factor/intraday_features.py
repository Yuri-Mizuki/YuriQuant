"""
分钟→日频统计特征层（tsfresh 风格，向量化）
==========================================

日内因子挖掘的"降维"层（参照 Alpha掘金 22 的做法：先用约 40 个统计指标把
日内分钟数据压成日频特征，再复用日频挖掘框架；统计量选取参照 tsfresh 的
feature calculators）。与 ``scripts/factors/build_intraday_factors.py`` 的 14 个
**经济含义** 因子互补：本层是面向挖掘的**统计特征族**原料。

设计约束
--------
- 输入是 ``data.intraday.MinutePanelStore`` 产出的稠密块
  ``{field: [D, B, C] ndarray}``（D=交易日, B=日内bar, C=代码），纯 numpy
  整块向量化——长表 groupby-apply 的逐组 Python 回调在 hs300 全量上要分钟
  级，这里的目标是秒级。
- **只用当日分钟信息，无未来函数**；bar 内收益口径与 build_intraday_factors
  一致（首根 close/open-1，其余 close/prev_close-1，不跨日拼接）。
- 跨日滚动（量比、RV-Z）与日线依赖（隔夜收益）不在本层——它们是日频面板
  上的 rolling，留给上层（见 build_intraday_factors / 日频挖掘框架）。
- NaN 语义：停牌/缺失 bar 为 NaN，所有统计量 NaN 感知（跳过缺失 bar）；
  当日有效 bar 数不足 ``min_bar_frac * B`` 时特征记 NaN（半拉天不产出）。

tsfresh 对应关系（feature_calculators 名称）
-------------------------------------------
mean / standard_deviation / skewness / kurtosis / minimum / maximum /
median → ret_* 系列；linear_trend([{attr:'slope'/'rvalue'}]) → mom_trend_*；
autocorrelation(lag_n) → shape_autocorr_l*；number_crossing_m →
shape_crossing_cnt；longest_strike_above_mean → shape_longest_pos_run（变体：
以 0 为界）；absolute_sum_of_changes → ret_abs_sum；root_mean_square →
ret_rms。量价类（ret_vol_corr 等）为 Alpha掘金 22 的量价维度，tsfresh 无
直接对应。
2026-10-14 增：时段切片矩阵族（20 分钟窗 × 动量/波动/量，`*_segNN_*` 36 个，
见 ``_segment_slices``）——「分钟频挖掘第三层扩原料」第一项（TODO.md §二 资源墙后置）。
2026-10-23 增：成交量占比自相关族 3 个（`pv_volshare_*`）+ 量价相关方向拆分 2 个
（`pv_ret_vol_corr_{up,dn}`）——「高频数据低频化：为什么量比价更重要」P0 落点，
见 `reports/docs/research_notes/短板匹配与微信文章研读_2026-09-22.md` §2b；同批再加
路径 `path_*` 2 个、分域显著性 `dom_*` 1 个、稀疏事件 `evt_*` 3 个（P1 落点）。
2026-10-23 再增：条件分布族 3 个（`cond_ret_{hi,lo}_mean` / `cond_ret_hi_lo_diff`）——
「高频研究系列六」第 4 章，P3-③ 落点；量价转移熵 2 个（`pv_te_{vol_ret,ret_vol}`）——
「高频研究系列八」`te_Comb`，P3-④ 落点；零收益 bar 占比 1 个（`pv_zero_ret_share`）——
P3-⑤ 非同步交易探针的正向落点（零成交维度不可用，改用价格粘滞度）。
共 59 个非时段特征 + 36 个时段特征。
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from data.intraday import MinutePanelStore

#: 每 bar 至少需覆盖的日内时长比例（低于则当日特征记 NaN）
MIN_BAR_FRAC = 0.5


# ----------------------------------------------------------------------
# NaN 感知的向量化统计原语（沿 bar 轴，[D, B, C] → [D, C]）
# ----------------------------------------------------------------------
def _n(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x).sum(axis=1)


def _mean(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(x, axis=1) / _n(x)


def _moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (m2, m3)：中心二阶/三阶矩（NaN 感知）。"""
    m1 = _mean(x)
    d = x - m1[:, None, :]
    m2 = np.nansum(d * d, axis=1) / _n(x)
    m3 = np.nansum(d ** 3, axis=1) / _n(x)
    return m2, m3


def _std(x: np.ndarray) -> np.ndarray:
    m2, _ = _moments(x)
    return np.sqrt(m2)


def _skew(x: np.ndarray) -> np.ndarray:
    m2, m3 = _moments(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        return m3 / np.where(m2 > 0, m2 ** 1.5, np.nan)


def _kurt(x: np.ndarray) -> np.ndarray:
    m1 = _mean(x)
    d = x - m1[:, None, :]
    m2 = np.nansum(d * d, axis=1) / _n(x)
    m4 = np.nansum(d ** 4, axis=1) / _n(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(m2 > 0, m4 / m2 ** 2, np.nan) - 3.0


def _ols_trend(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对 bar 序号 t=0..B-1 做 OLS：返回 (slope, r²)（NaN 感知）。"""
    B = x.shape[1]
    t = np.arange(B, dtype=np.float64)
    W = np.isfinite(x)
    n = W.sum(axis=1)
    tm = np.nansum(np.where(W, t[None, :, None], np.nan), axis=1) / n
    m1 = _mean(x)
    dt = np.where(W, t[None, :, None] - tm[:, None, :], np.nan)
    dx = x - m1[:, None, :]
    cov = np.nansum(dt * dx, axis=1)
    var_t = np.nansum(dt * dt, axis=1)
    var_x = np.nansum(dx * dx, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = cov / var_t
        r2 = (cov * cov) / (var_t * var_x)
    return slope, r2


def _autocorr(x: np.ndarray, lag: int) -> np.ndarray:
    """lag 阶自相关（对日内均值中心化；NaN 成对跳过）。"""
    m1 = _mean(x)
    d = x - m1[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(d[:, lag:] * d[:, :-lag], axis=1) / np.nansum(d * d, axis=1)


def _acf_disp(x: np.ndarray, max_lag: int = 5) -> np.ndarray:
    """lag1..max_lag 的 |自相关| 均值（多阶自相关离散度）。

    高频低频化文章的多阶自相关变体：单看某一阶会被噪声误导，取前几阶
    绝对值的均值刻画"序列是否长期扎堆"，对阶数选择不敏感。
    """
    acs = np.stack([_autocorr(x, k) for k in range(1, max_lag + 1)], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(np.abs(acs), axis=0)


def _ar1_resid_autocorr(x: np.ndarray) -> np.ndarray:
    """AR(1) 残差的 lag-1 自相关（NaN 感知，成对有效）。

    对每个 (日, 码) 用 x_{t-1} 回归 x_t，再取残差 e_t 的一阶自相关。
    这是「高频数据低频化」排名第一的构造本身（输入换成成交量占比后
    RankIC 0.0649 / 11 年年年正）。
    """
    a, b = x[:, :-1, :], x[:, 1:, :]
    W = np.isfinite(a) & np.isfinite(b)
    n = W.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = np.nansum(np.where(W, a, np.nan), axis=1) / n
        mb = np.nansum(np.where(W, b, np.nan), axis=1) / n
        da = np.where(W, a - ma[:, None, :], np.nan)
        db = np.where(W, b - mb[:, None, :], np.nan)
        slope = np.nansum(da * db, axis=1) / np.nansum(da * da, axis=1)
    resid = db - slope[:, None, :] * da
    return _autocorr(resid, 1)


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两序列沿 bar 轴的相关系数（NaN 感知）。"""
    ma, mb = _mean(a), _mean(b)
    da = a - ma[:, None, :]
    db = b - mb[:, None, :]
    cov = np.nansum(da * db, axis=1)
    va = np.nansum(da * da, axis=1)
    vb = np.nansum(db * db, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(va * vb)


def _discretize(x: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """按当日截面（每 (日,码) 自身时序）分位离散化，返回 int8 分箱（NaN → -1）。"""
    qs = np.nanquantile(x, np.arange(1, n_bins) / n_bins, axis=1, keepdims=True)
    b = np.zeros(x.shape, dtype=np.int8)
    for k in range(n_bins - 1):
        b += (x > qs[k]).astype(np.int8)
    b[~np.isfinite(x)] = -1
    return b


def _transfer_entropy(src: np.ndarray, dst: np.ndarray,
                      n_bins: int = 3) -> np.ndarray:
    """转移熵 TE(src→dst)：[D, B, C] 序列对，返回 [D, C]（单位 nats）。

    TE = Σ p(s_t, d_t, d_{t+1}) · ln[ p(s_t,d_t,d_{t+1})·p(d_t)
                                    / (p(s_t,d_t)·p(d_t,d_{t+1})) ]

    刻画"知道 src 的当期值后，对 dst 下一期的不确定性减少多少"。分箱按
    (日,码) 自己的三分位，天然消除量纲；Laplace +0.5 平滑避免空盒
    log(0)。只允许 lag=1（48 bar/日，更长 lag 的样本量撑不住 27 个盒子）。
    """
    D, B, C = src.shape
    nb = n_bins
    bs, bd = _discretize(src, nb), _discretize(dst, nb)
    s_t, d_t, d_n = bs[:, :-1, :], bd[:, :-1, :], bd[:, 1:, :]
    valid = (s_t >= 0) & (d_t >= 0) & (d_n >= 0)
    cell = (s_t.astype(np.int64) * nb + d_t) * nb + d_n
    base = (np.arange(D)[:, None, None] * C
            + np.arange(C)[None, None, :]) * (nb ** 3)
    idx = (base + np.where(valid, cell, 0)).ravel()
    w = valid.ravel().astype(np.float64)
    counts = np.bincount(idx, weights=w, minlength=D * C * nb ** 3)
    p = counts.reshape(D, C, nb, nb, nb) + 0.5
    p /= p.sum(axis=(2, 3, 4), keepdims=True)
    p_sd = p.sum(axis=4)                          # (s_t, d_t)
    p_dd = p.sum(axis=2)                          # (d_t, d_{t+1})
    p_d = p.sum(axis=(2, 4))                      # d_t 边际
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (p * p_d[:, :, None, :, None]
                 / (p_sd[:, :, :, :, None] * p_dd[:, :, None, :, :]))
        return np.nansum(p * np.log(ratio), axis=(2, 3, 4))


def _longest_run(mask: np.ndarray) -> np.ndarray:
    """bool [D, B, C] 沿 bar 轴的最长连续 True 段长。

    技巧：位置 i 的连续 True 长度 = i - 上一个 False 的位置（前缀无 False
    则为 i+1，用哨兵 -1 统一）；NaN bar 在调用方已被置 False（打断连段）。
    """
    B = mask.shape[1]
    idx = np.arange(B)[None, :, None]
    last_false = np.maximum.accumulate(
        np.where(~mask, idx, -1), axis=1)
    run = np.where(mask, idx - last_false, 0)
    return run.max(axis=1).astype(np.float64)


def _segment_slices(bar_times: Sequence[str], period: int) -> list[np.ndarray] | None:
    """把日内 bar 分成 12 个 20 分钟窗（上午/下午各 6 窗），返回每窗的 bar 掩码。

    按**位置**分组：bar_times 是全日共用的时刻表（meta.json 48 档），bar 的
    位置即时段归属——bar 开始时刻源（0930..1125）与结束时刻源（0935..1130）
    都成立，与源语义无关。缺失 bar 在面板里是 NaN 占位，位置不塌缩。

    - 20 分钟不可被 period 整除（如 30/15 分钟 bar）→ 返回 None，调用方
      跳过时段特征（无意义的半窗切片）。
    - 假定标准 A 股两 session 各恰好 6 窗（4 小时 / 20 分钟）；bar 数不足
      一个 session（异常源）时同样返回 None，保证产出键与 FEATURE_DOCS 的
      12 窗命名一致。
    """
    if period <= 0 or 20 % period != 0:
        return None
    bars_per_seg = 20 // period
    am = np.array([bt <= "1130" for bt in bar_times])
    slices: list[np.ndarray] = []
    for session_mask in (am, ~am):
        idx = np.where(session_mask)[0]
        if len(idx) < bars_per_seg or len(idx) % bars_per_seg != 0:
            return None
        for k in range(len(idx) // bars_per_seg):
            m = np.zeros(len(bar_times), dtype=bool)
            m[idx[k * bars_per_seg:(k + 1) * bars_per_seg]] = True
            slices.append(m)
    if len(slices) != 12:
        return None
    return slices


def _bar_returns(close: np.ndarray, open_: np.ndarray) -> np.ndarray:
    """bar 内收益 [D, B, C]：首根 close/open-1，其余 close/prev_close-1。"""
    r = close[:, 1:, :] / close[:, :-1, :] - 1.0
    first = close[:, 0:1, :] / open_[:, 0:1, :] - 1.0
    return np.concatenate([first, r], axis=1)


# ----------------------------------------------------------------------
# 特征提取主入口
# ----------------------------------------------------------------------
def extract_from_block(
    fields: dict[str, np.ndarray],
    days: Sequence[int],
    codes: Sequence[str],
    bar_times: Sequence[str],
    period: int,
    min_bar_frac: float = MIN_BAR_FRAC,
) -> dict[str, pd.DataFrame]:
    """单个块 [D, B, C] → {特征名: date×code 面板}。

    fields 需至少含 open/high/low/close/volume/amount（字段缺失则相关特征
    直接不产出，不报错——允许裁剪字段的流式用法）。
    """
    close, open_, high, low = (fields.get(k) for k in ("close", "open", "high", "low"))
    volume, amount = fields.get("volume"), fields.get("amount")
    B = close.shape[1]
    idx = pd.to_datetime([str(d) for d in days], format="%Y%m%d")
    cols = list(codes)

    # 全 NaN 日（新股/停牌）在 nanmin/0 除等处触发 RuntimeWarning——NaN 本身
    # 就是预期产物（guard 会统一置 NaN），这里整体压制避免刷屏
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _extract_inner(close, open_, high, low, volume, amount,
                              B, idx, cols, bar_times, period, min_bar_frac)


def _extract_inner(close, open_, high, low, volume, amount,
                   B, idx, cols, bar_times, period, min_bar_frac):
    out: dict[str, pd.DataFrame] = {}

    def panel(a: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(a, index=idx, columns=cols, dtype=float)

    r = _bar_returns(close, open_)
    n_r = _n(r)
    ok = n_r >= max(2, int(min_bar_frac * B))  # 有效 bar 不足的当日置 NaN

    def guard(a: np.ndarray) -> np.ndarray:
        return np.where(ok, a, np.nan)

    # ---- 价格动量类 ----
    with np.errstate(invalid="ignore", divide="ignore"):
        out["mom_intraday_ret"] = panel(guard(close[:, -1, :] / close[:, 0, :] - 1.0))
        k = max(1, 30 // period)
        out["mom_open30_ret"] = panel(guard(np.nansum(r[:, :k, :], axis=1)))
        out["mom_close30_ret"] = panel(guard(np.nansum(r[:, -k:, :], axis=1)))
        # 上午 = 时刻 ≤ 11:30（bar 打结束时刻的源最后一根是 11:30，属上午；
        # 打开始时刻的源最后一根是 11:25，两种源该判定都正确）
        am = np.array([bt <= "1130" for bt in bar_times])
        out["mom_am_ret"] = panel(guard(np.nansum(r[:, am, :], axis=1)))
        out["mom_pm_ret"] = panel(guard(np.nansum(r[:, ~am, :], axis=1)))
        out["mom_am_pm_diff"] = out["mom_am_ret"] - out["mom_pm_ret"]
        out["mom_first_bar"] = panel(guard(r[:, 0, :]))
        out["mom_last_bar"] = panel(guard(r[:, -1, :]))
        lc = np.log(close)
        slope, r2 = _ols_trend(lc)
        # 斜率归一到日内总漂移（×B 后≈全日对数收益），消除 bar 数量纲
        out["mom_trend_drift"] = panel(guard(slope * B))
        out["mom_trend_r2"] = panel(guard(r2))
        hi = np.nanmax(high, axis=1)
        lo = np.nanmin(low, axis=1)
        out["mom_pos_in_range"] = panel(guard(
            (close[:, -1, :] - lo) / np.where(hi > lo, hi - lo, np.nan)))

    # ---- 收益分布统计类（tsfresh 基础统计量）----
    out["ret_mean"] = panel(guard(_mean(r)))
    out["ret_std"] = panel(guard(_std(r)))
    out["ret_skew"] = panel(guard(_skew(r)))
    out["ret_kurt"] = panel(guard(_kurt(r)))
    out["ret_min"] = panel(guard(np.nanmin(r, axis=1)))
    out["ret_max"] = panel(guard(np.nanmax(r, axis=1)))
    out["ret_median"] = panel(guard(np.nanmedian(r, axis=1)))
    out["ret_abs_sum"] = panel(guard(np.nansum(np.abs(r), axis=1)))
    out["ret_rms"] = panel(guard(np.sqrt(np.nansum(r * r, axis=1) / n_r)))
    out["ret_up_ratio"] = panel(guard((r > 0).sum(axis=1) / n_r))

    # ---- 条件分布类（兴业「高频研究系列六」第 4 章：先按波动状态筛选再统计收益）----
    # 日内 bar 的 |r| 分位作为「高波/低波」状态：上 20% 记高波、下 20% 记低波，
    # 分别取带符号收益均值。与 vol_rsj 的区别：RSJ 只用平方量（幅度），这里保留
    # 方向——"高波时涨"与"高波时跌"是相反信号，平方量会把它们抹平。
    absr = np.where(np.isfinite(r), np.abs(r), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        q_hi = np.nanquantile(absr, 0.8, axis=1, keepdims=True)
        q_lo = np.nanquantile(absr, 0.2, axis=1, keepdims=True)
        hi_m = np.isfinite(r) & (absr >= q_hi)
        lo_m = np.isfinite(r) & (absr <= q_lo)
        hi_mean = np.nansum(np.where(hi_m, r, np.nan), axis=1) / hi_m.sum(axis=1)
        lo_mean = np.nansum(np.where(lo_m, r, np.nan), axis=1) / lo_m.sum(axis=1)
    out["cond_ret_hi_mean"] = panel(guard(hi_mean))
    out["cond_ret_lo_mean"] = panel(guard(lo_mean))
    out["cond_ret_hi_lo_diff"] = panel(guard(hi_mean - lo_mean))

    # ---- 波动类 ----
    with np.errstate(invalid="ignore", divide="ignore"):
        up = np.where(r > 0, r * r, 0.0)
        dn = np.where(r < 0, r * r, 0.0)
        su, sd = np.nansum(up, axis=1), np.nansum(dn, axis=1)
        out["vol_rv"] = panel(guard(su + sd))
        out["vol_rsj"] = panel(guard((su - sd) / np.where(su + sd > 0, su + sd, np.nan)))
        out["vol_downside_rms"] = panel(guard(np.sqrt(
            np.nansum(np.where(r < 0, r * r, np.nan), axis=1) / (r < 0).sum(axis=1))))
        out["vol_range"] = panel(guard((hi - lo) / np.where(close[:, 0, :] > 0,
                                                            close[:, 0, :], np.nan)))
        big = np.abs(r) >= np.nanpercentile(np.where(np.isfinite(r), np.abs(r), np.nan),
                                            90, axis=1, keepdims=True)
        out["vol_extreme_bar_share"] = panel(guard(
            np.nansum(np.where(big, r * r, 0.0), axis=1) / np.where(su + sd > 0,
                                                                    su + sd, np.nan)))

    # ---- 形态/自相关类 ----
    out["shape_autocorr_l1"] = panel(guard(_autocorr(r, 1)))
    out["shape_autocorr_l2"] = panel(guard(_autocorr(r, 2)))
    m1 = _mean(r)
    cross = (r[:, :-1, :] < m1[:, None, :]) & (r[:, 1:, :] >= m1[:, None, :])
    out["shape_crossing_cnt"] = panel(guard(cross.sum(axis=1).astype(np.float64)))
    r_finite = np.isfinite(r)
    out["shape_longest_pos_run"] = panel(guard(_longest_run(r_finite & (r > 0))))
    out["shape_longest_neg_run"] = panel(guard(_longest_run(r_finite & (r < 0))))

    # ---- 路径：极值邻域伴随状态（记"当时伴随什么"，不是"发生在什么时候"）----
    hi_src = high if high is not None else close
    lo_src = low if low is not None else close
    pos = np.arange(B)[None, :, None]
    valid_hi = np.isfinite(hi_src).any(axis=1)
    valid_lo = np.isfinite(lo_src).any(axis=1)
    k_nb = max(1, 20 // period)  # ±20 分钟邻域
    with np.errstate(invalid="ignore", divide="ignore"):
        arg_hi = np.nanargmax(np.where(np.isfinite(hi_src), hi_src, -np.inf), axis=1)
        arg_lo = np.nanargmin(np.where(np.isfinite(lo_src), lo_src, np.inf), axis=1)
        near_hi = np.abs(pos - arg_hi[:, None, :]) <= k_nb
        near_lo = np.abs(pos - arg_lo[:, None, :]) <= k_nb
        rv_bar = np.where(np.isfinite(r), r * r, 0.0)
        rv_hi = np.nansum(np.where(near_hi, rv_bar, np.nan), axis=1)
        rv_lo = np.nansum(np.where(near_lo, rv_bar, np.nan), axis=1)
        out["path_extreme_rv_diff"] = panel(guard(np.where(
            valid_hi & valid_lo, rv_hi - rv_lo, np.nan)))

    # ---- 稀疏事件：只记冲击深度，不记修复速度 ----
    with np.errstate(invalid="ignore", divide="ignore"):
        # 最大反转幅度：日内低点之后回升的最大幅度（下档承接厚度）
        after_lo = pos > arg_lo[:, None, :]
        hi_after = np.nanmax(np.where(after_lo, close, np.nan), axis=1)
        lo_val = np.take_along_axis(close, arg_lo[:, None, :], axis=1)[:, 0, :]
        out["evt_max_reversal_depth"] = panel(guard(np.where(
            valid_lo, hi_after / lo_val - 1.0, np.nan)))
        # 冲击深度：自日内累计高点的最大回撤幅度
        cummax = np.maximum.accumulate(
            np.where(np.isfinite(close), close, -np.inf), axis=1)
        dd = np.where(np.isfinite(close) & (cummax > 0), close / cummax - 1.0, np.nan)
        out["evt_drawdown_depth"] = panel(guard(-np.nanmin(dd, axis=1)))
        # 极端下行 bar 到达率（尾部频次；与 vol_extreme_bar_share 的"幅度占比"互补）
        thr = np.nanpercentile(np.where(np.isfinite(r), r, np.nan), 5,
                               axis=1, keepdims=True)
        out["evt_down_tail_share"] = panel(guard(
            ((r <= thr) & np.isfinite(r)).sum(axis=1) / n_r))

    # ---- 量价类 ----
    if volume is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            vm = _mean(volume)
            # 条件与值都升到 (D,1,C)——2-D 条件 (D,C) 与 3-D 值 (D,1,C) 直接
            # np.where 会广播成 (D,D,C) 的错形状
            vol_norm = volume / np.where(vm[:, None, :] > 0, vm[:, None, :], np.nan)
            vslope, _ = _ols_trend(vol_norm)
            out["pv_vol_trend"] = panel(guard(vslope * B))
            out["pv_ret_vol_corr"] = panel(guard(_corr(r, volume)))
            # 价格粘滞度：零收益 bar 占比（非同步交易的日频代理）。5 分钟 bar 上
            # 零成交几乎不发生（87.8% 的 (日,码) 单元零成交 bar 数为 0），故用
            # "价格未动"而非"无成交"度量不活跃；与 ret_up_ratio 有机械负相关
            # （涨+跌+平=1），但剔除后仍有递增残差信息（见研读笔记 §2b.15）
            out["pv_zero_ret_share"] = panel(guard((r == 0).sum(axis=1) / n_r))
            vsum = np.nansum(volume, axis=1)
            third = max(1, B // 3)
            head = _mean(volume[:, :third, :])
            tail = _mean(volume[:, -third:, :])
            out["pv_vol_tail_ratio"] = panel(guard(tail / np.where(head > 0, head, np.nan)))
            # 大波动 bar 的成交量占比（前 10% |r| 的 bar）
            out["pv_big_move_vol_share"] = panel(guard(
                np.nansum(np.where(big, volume, 0.0), axis=1)
                / np.where(vsum > 0, vsum, np.nan)))
            out["pv_vol_skew"] = panel(guard(_skew(volume)))
            # ---- 成交量占比自相关族（「量比价更耐久」，见研读笔记 §2b.1 结论 1）----
            # 占比 = 该 bar 量 / 当日总量。自相关对常数缩放不变，故占比化不改变
            # 相关系数，只统一 AR(1) 斜率与残差的量纲；跨截面可比性由 zscore 承担。
            vshare = volume / np.where(vsum[:, None, :] > 0, vsum[:, None, :], np.nan)
            out["pv_volshare_autocorr_l1"] = panel(guard(_autocorr(vshare, 1)))
            out["pv_volshare_acf_disp"] = panel(guard(_acf_disp(vshare)))
            out["pv_volshare_ar1_resid_acf"] = panel(guard(_ar1_resid_autocorr(vshare)))
            # ---- 量价耦合按涨跌方向拆分（混算会让两种机制相互抵消）----
            up_m, dn_m = r > 0, r < 0
            out["pv_ret_vol_corr_up"] = panel(guard(_corr(
                np.where(up_m, r, np.nan), np.where(up_m, volume, np.nan))))
            # 下跌分钟量价相关通常为负，取负号使「放量下跌」= 高值（与上涨侧同向）
            out["pv_ret_vol_corr_dn"] = panel(guard(-_corr(
                np.where(dn_m, r, np.nan), np.where(dn_m, volume, np.nan))))
            # ---- 转移熵：量价非线性关联（系列八 te_Comb）----
            # 线性相关看不到"放量后下一 bar 涨跌"的非线性传导；TE 三分位离散后
            # 测条件不确定性下降，两个方向各自成因子（不合成，避免再出 ρ≈1 冗余）。
            out["pv_te_vol_ret"] = panel(guard(_transfer_entropy(volume, r)))
            out["pv_te_ret_vol"] = panel(guard(_transfer_entropy(r, volume)))
            # 极值邻域的成交量份额差（「当时伴随了多少量」，非「什么时候」）
            vol_ok = np.isfinite(volume)
            v_hi = np.nansum(np.where(near_hi & vol_ok, volume, np.nan), axis=1)
            v_lo = np.nansum(np.where(near_lo & vol_ok, volume, np.nan), axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["path_extreme_volshare_diff"] = panel(guard(
                    (v_hi - v_lo) / np.where(vsum > 0, vsum, np.nan)))
            # 分域显著性：按成交量 5 档测 RV 偏离全天均值的相对幅度
            # （文章结论：强度变量有效、方向变量无效，故只做波动/量）
            vv = vol_ok & np.isfinite(r)
            n_bkt = 5
            order = np.argsort(np.where(vv, volume, np.inf), axis=1, kind="stable")
            ranks = np.empty(order.shape, dtype=np.int64)
            np.put_along_axis(ranks, order, np.broadcast_to(
                np.arange(volume.shape[1])[None, :, None], order.shape).copy(), axis=1)
            n_ok = vv.sum(axis=1)
            bkt = np.clip((ranks * n_bkt) // np.maximum(n_ok[:, None, :], 1),
                          0, n_bkt - 1)
            rv_all = np.nansum(np.where(vv, rv_bar, np.nan), axis=1) \
                / np.maximum(n_ok, 1)
            sig = np.full(rv_all.shape, -np.inf)
            for b in range(n_bkt):
                m = vv & (bkt == b)
                mean_b = np.nansum(np.where(m, rv_bar, np.nan), axis=1) \
                    / np.maximum(m.sum(axis=1), 1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    sig = np.fmax(sig, mean_b / np.where(rv_all > 0, rv_all, np.nan))
            out["dom_vol_rv_sig"] = panel(guard(np.where(
                np.isfinite(sig), sig - 1.0, np.nan)))
        if amount is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                amihud = np.abs(r) / np.where(amount > 0, amount, np.nan)
                out["pv_amihud"] = panel(guard(_mean(amihud)))
                typical = (high + low + close) / 3.0
                vwap = np.nansum(typical * volume, axis=1) / np.where(vsum > 0, vsum, np.nan)
                out["pv_vwap_dev"] = panel(guard(close[:, -1, :] / vwap - 1.0))
        # 日内最高价出现在上午（1/0；当日 high 全缺失记 NaN——argmax 对
        # 全 NaN 序列返回 0，不处理会误判成"上午"）
        valid_hi = np.isfinite(high).any(axis=1)
        argmax_hi = np.nanargmax(np.where(np.isfinite(high), high, -np.inf), axis=1)
        out["candle_high_in_am"] = panel(guard(
            np.where(valid_hi, am[argmax_hi].astype(float), np.nan)))

    # ---- 蜡烛形态类（bar 级影线/收盘位置的日内均值）----
    with np.errstate(invalid="ignore", divide="ignore"):
        body_hi = np.maximum(open_, close)
        body_lo = np.minimum(open_, close)
        rng = high - low
        rng_safe = np.where(rng > 0, rng, np.nan)
        out["candle_upper_shadow"] = panel(guard(_mean((high - body_hi) / rng_safe)))
        out["candle_lower_shadow"] = panel(guard(_mean((body_lo - low) / rng_safe)))
        out["candle_close_pos"] = panel(guard(_mean((close - low) / rng_safe)))

    # ---- 时段切片矩阵类（20 分钟窗 × 动量/波动/量，TODO「第三层扩原料」）----
    # 12 窗命名 mom_seg01..12 / vol_seg01..12 / pv_seg01..12：
    # seg01=09:30-09:50 … seg06=11:10-11:30，seg07=13:00-13:20 … seg12=14:40-15:00。
    # 窗内无有效 bar → NaN（nansum 会把停牌段静默记 0，须按有效 bar 数门控）；
    # 窗内部分 bar 缺失时为部分和（半拉天语义，整体仍受 guard 门控）。
    seg_masks = _segment_slices(bar_times, period)
    if seg_masks is not None:
        r2 = np.where(np.isfinite(r), r * r, 0.0)
        rv_day = np.nansum(r2, axis=1)
        for k, m in enumerate(seg_masks, start=1):
            rk = r[:, m, :]
            nv = np.isfinite(rk).sum(axis=1)
            tag = f"{k:02d}"
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"mom_seg{tag}_ret"] = panel(guard(
                    np.where(nv > 0, np.nansum(rk, axis=1), np.nan)))
                rv_k = np.nansum(r2[:, m, :], axis=1)
                out[f"vol_seg{tag}_rvshare"] = panel(guard(np.where(
                    (nv > 0) & (rv_day > 0), rv_k / np.where(rv_day > 0, rv_day, np.nan),
                    np.nan)))
                if volume is not None:
                    v_k = np.nansum(volume[:, m, :], axis=1)
                    v_day = np.nansum(volume, axis=1)
                    out[f"pv_seg{tag}_volshare"] = panel(guard(np.where(
                        (nv > 0) & (v_day > 0),
                        v_k / np.where(v_day > 0, v_day, np.nan), np.nan)))

    return out


def extract_all(
    store: MinutePanelStore,
    begin_date: int | None = None,
    end_date: int | None = None,
    block_days: int = 128,
    min_bar_frac: float = MIN_BAR_FRAC,
) -> dict[str, pd.DataFrame]:
    """从面板 store 流式提取全区间特征（逐块载入，内存友好），按日期拼接。

    返回 {特征名: date×code DataFrame}，可直接喂给 FactorLibrary /
    genetic_mining 等日频挖掘框架（Alpha掘金 22 的"降维后复用日频框架"）。
    """
    parts: dict[str, list[pd.DataFrame]] = {}
    for chunk_days, block in store.iter_blocks(
            ("open", "high", "low", "close", "volume", "amount"),
            block_days=block_days, begin_date=begin_date, end_date=end_date):
        feats = extract_from_block(block, chunk_days, store.codes,
                                   store.bar_times, store.period, min_bar_frac)
        for name, df in feats.items():
            parts.setdefault(name, []).append(df)
    return {name: pd.concat(frames).sort_index() for name, frames in parts.items()}


#: 特征中文说明（入库/报告用；与 extract_from_block 的 key 一一对应）
FEATURE_DOCS: dict[str, str] = {
    "mom_intraday_ret": "日内收益 close末/close首-1",
    "mom_open30_ret": "开盘30分钟累计bar收益",
    "mom_close30_ret": "尾盘30分钟累计bar收益",
    "mom_am_ret": "上午累计bar收益",
    "mom_pm_ret": "下午累计bar收益",
    "mom_am_pm_diff": "上午-下午累计收益差",
    "mom_first_bar": "首根bar收益",
    "mom_last_bar": "末根bar收益",
    "mom_trend_drift": "log close对bar序号的OLS斜率×B（日内趋势总漂移）",
    "mom_trend_r2": "日内趋势拟合优度R²",
    "mom_pos_in_range": "收盘价在当日高低区间中的位置",
    "ret_mean": "bar收益均值（tsfresh mean）",
    "ret_std": "bar收益标准差（standard_deviation）",
    "ret_skew": "bar收益偏度（skewness）",
    "ret_kurt": "bar收益超额峰度（kurtosis）",
    "ret_min": "bar收益最小值（minimum）",
    "ret_max": "bar收益最大值（maximum）",
    "ret_median": "bar收益中位数（median）",
    "ret_abs_sum": "bar收益绝对值和（absolute_sum_of_changes近似）",
    "ret_rms": "bar收益均方根（root_mean_square）",
    "ret_up_ratio": "上涨bar占比",
    "cond_ret_hi_mean": "高波bar（|r|上20%）的带符号收益均值",
    "cond_ret_lo_mean": "低波bar（|r|下20%）的带符号收益均值",
    "cond_ret_hi_lo_diff": "高波-低波收益均值差（方向随波动状态的分化）",
    "vol_rv": "已实现波动率Σr²",
    "vol_rsj": "好坏波动差（RSJ）",
    "vol_downside_rms": "下行bar收益均方根",
    "vol_range": "日内振幅(high_max-low_min)/open首",
    "vol_extreme_bar_share": "最剧烈10% bar的贡献占RV比例",
    "shape_autocorr_l1": "bar收益1阶自相关（autocorrelation lag1）",
    "shape_autocorr_l2": "bar收益2阶自相关",
    "shape_crossing_cnt": "bar收益上穿日内均值次数（number_crossing_m）",
    "shape_longest_pos_run": "最长连续上涨bar数（longest_strike变体）",
    "shape_longest_neg_run": "最长连续下跌bar数",
    "pv_vol_trend": "成交量日内趋势斜率×B（归一化）",
    "pv_ret_vol_corr": "bar收益与成交量相关系数",
    "pv_zero_ret_share": "零收益bar占比（价格粘滞度，非同步交易的日频代理）",
    "pv_vol_tail_ratio": "尾1/3时段均量/首1/3时段均量",
    "pv_big_move_vol_share": "最剧烈10% bar的成交量占比",
    "pv_vol_skew": "成交量分布偏度（放量集中度）",
    "pv_volshare_autocorr_l1": "成交量占比序列一阶自相关（交易扎堆惯性）",
    "pv_volshare_acf_disp": "成交量占比 lag1~5 |自相关| 均值（多阶离散度）",
    "pv_volshare_ar1_resid_acf": "成交量占比 AR(1) 残差自相关（量比价耐久主构造）",
    "pv_ret_vol_corr_up": "上涨 bar 的量价相关（放量上涨=买盘推进）",
    "pv_ret_vol_corr_dn": "下跌 bar 量价相关的绝对值口径（取负，放量下跌=高值）",
    "pv_te_vol_ret": "量→收益转移熵（lag1，三分位离散，非线性传导强度）",
    "pv_te_ret_vol": "收益→量转移熵（lag1，反向传导强度）",
    "path_extreme_rv_diff": "最高/最低价 ±20 分钟邻域的已实现波动差（伴随状态）",
    "path_extreme_volshare_diff": "最高/最低价 ±20 分钟邻域的成交量份额差",
    "dom_vol_rv_sig": "按成交量 5 档的最大 RV 偏离（域内显著性，强度变量）",
    "evt_max_reversal_depth": "日内低点之后的极大回升幅度（下档承接厚度）",
    "evt_drawdown_depth": "自日内累计高点的最大回撤幅度（冲击深度）",
    "evt_down_tail_share": "极端下行 bar 到达率（尾部频次，不记速度）",
    "pv_amihud": "日内Amihud非流动性 mean(|r|/amount)",
    "pv_vwap_dev": "收盘价相对日内VWAP偏离",
    "candle_high_in_am": "当日最高价是否出现在上午(1/0)",
    "candle_upper_shadow": "bar级上影线占比均值",
    "candle_lower_shadow": "bar级下影线占比均值",
    "candle_close_pos": "收盘在bar区间内的位置均值",
}

# 时段切片矩阵（20 分钟窗 × 动量/波动/量）：12 窗命名固定，由
# _segment_slices 保证「两 session 各恰好 6 窗」时才产出，键集与 docs 一致。
for _k in range(1, 13):
    FEATURE_DOCS[f"mom_seg{_k:02d}_ret"] = f"第{_k}/12个20分钟窗累计bar收益"
    FEATURE_DOCS[f"vol_seg{_k:02d}_rvshare"] = f"第{_k}/12个20分钟窗RV占全天RV份额"
    FEATURE_DOCS[f"pv_seg{_k:02d}_volshare"] = f"第{_k}/12个20分钟窗成交量占全天份额"
