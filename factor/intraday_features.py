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
    "pv_vol_tail_ratio": "尾1/3时段均量/首1/3时段均量",
    "pv_big_move_vol_share": "最剧烈10% bar的成交量占比",
    "pv_vol_skew": "成交量分布偏度（放量集中度）",
    "pv_amihud": "日内Amihud非流动性 mean(|r|/amount)",
    "pv_vwap_dev": "收盘价相对日内VWAP偏离",
    "candle_high_in_am": "当日最高价是否出现在上午(1/0)",
    "candle_upper_shadow": "bar级上影线占比均值",
    "candle_lower_shadow": "bar级下影线占比均值",
    "candle_close_pos": "收盘在bar区间内的位置均值",
}
