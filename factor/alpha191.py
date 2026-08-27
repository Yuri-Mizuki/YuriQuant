"""国泰君安《基于短周期价量特征的多因子选股体系》(2017) 191 因子面板实现。

- 纯 pandas 面板实现（date×code），复用 factor/operators + alpha_base 算子。
- 公式文本以 BigQuant/JoinQuant 公开转录为准（两源一致），歧义处参考
  DolphinDB gtja191Alpha.dos 与 GitHub 公开实现，逐条注明。
- 语义约定：
  * RANK=截面百分位、TSRANK=时序百分位、SMA=通达信递归 SMA(X,N,M)、
    WMA/DECAYLINEAR=线性加权均值(近期权重更大)、REGBETA(X,SEQUENCE,N)=
    ts_slope(X,N)、MEAN/STD/SUM/DELTA/DELAY 同 factor/operators（STD 为样本std）；
  * MAX(X,N)/MIN(X,N) 按公式语境取 TSMAX/TSMIN（如 MAX(VWAP,15)）；
  * 布尔比较因子：原文 ``(cond) * -1`` 输出 0/-1，裸比较输出 0/1；
  * 部分原文有笔误（HGIH→HIGH、DELAT→DELTA、SMEAN→SMA、CLOSE:20→CLOSE,20、
    MAX(HIGH,9)-TSMAX(LOW,9)→TSMAX(HIGH,9)-TSMIN(LOW,9) 等），按显然意图修正。
- 跳过清单见 ``SKIPPED_191``：基准指数类、Fama-French 类、SELF 递归类、
  SUMAC 转写歧义类，以及与 alpha101 完全重复的 3 个因子（去重）。

用法：
    d = AlphaData(panels)
    panels_out = compute_alpha191(d)      # {name: 面板}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor.alpha_base import (  # noqa: F401
    AlphaData, abs_, and_, corr, count_, cov, decay_linear, delay, delta, eq,
    ge, gt, highday, if_else, le, log_, lowday, lt, max_, min_, or_, power,
    rank, sign, sma_tdx, stddev, sumif, tsmax, tsmin, tsr, w,
)
from factor.operators import ts_slope, ts_wma

__all__ = ["ALPHA191", "compute_alpha191", "SKIPPED_191"]

SKIPPED_191: dict[str, str] = {
    "alpha191_030": "REGRESI 需 MKT/SMB/HML（Fama-French）数据，跳过",
    "alpha191_032": "与 alpha101_032 公式完全相同，去重跳过",
    "alpha191_040": "与 alpha101_040 公式完全相同，去重跳过",
    "alpha191_075": "需基准指数行情（BANCHMARKINDEX*），跳过",
    "alpha191_139": "与 alpha101_006 公式完全相同，去重跳过",
    "alpha191_143": "公式含 SELF（递归引用），跳过",
    "alpha191_149": "需基准指数行情（BANCHMARKINDEX*），跳过",
    "alpha191_165": "SUMAC（累计和极值）原文转写歧义，主流实现均未实现，跳过",
    "alpha191_181": "需基准指数行情（BANCHMARKINDEX*），跳过",
    "alpha191_182": "需基准指数行情（BANCHMARKINDEX*），跳过",
    "alpha191_183": "SUMAC（累计和极值）原文转写歧义，主流实现均未实现，跳过",
}

ALPHA191: dict[str, object] = {}


def _a(name: str):
    def deco(fn):
        ALPHA191[name] = fn
        return fn
    return deco


def _sum(x, n):
    return x.rolling(n, min_periods=n).sum()


def _mean(x, n):
    return x.rolling(n, min_periods=n).mean()


def _clean(x: pd.DataFrame) -> pd.DataFrame:
    return x.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# 001-020
# ---------------------------------------------------------------------------
@_a("alpha191_001")
def _(d: AlphaData):
    return -1 * corr(rank(delta(log_(d.volume), 1)), rank((d.close - d.open) / d.open), 6)


@_a("alpha191_002")
def _(d: AlphaData):
    return -1 * delta((d.close - d.low - (d.high - d.close)) / (d.high - d.low), 1)


@_a("alpha191_003")
def _(d: AlphaData):
    dc = delay(d.close, 1)
    inner = if_else(
        eq(d.close, dc), 0.0,
        d.close - if_else(gt(d.close, dc), min_(d.low, dc), max_(d.high, dc)),
    )
    return _sum(inner, 6)


@_a("alpha191_004")
def _(d: AlphaData):
    m8, s8 = _mean(d.close, 8), stddev(d.close, 8)
    m2 = _mean(d.close, 2)
    return if_else(
        lt(m8 + s8, m2), -1.0,
        if_else(
            lt(m2, m8 - s8), 1.0,
            if_else(ge(d.volume / _mean(d.volume, 20), 1), 1.0, -1.0),
        ),
    )


@_a("alpha191_005")
def _(d: AlphaData):
    return -1 * tsmax(corr(tsr(d.volume, 5), tsr(d.high, 5), 5), 3)


@_a("alpha191_006")
def _(d: AlphaData):
    return rank(sign(delta(d.open * 0.85 + d.high * 0.15, 4))) * -1


@_a("alpha191_007")
def _(d: AlphaData):
    return (rank(tsmax(d.vwap - d.close, 3)) + rank(tsmin(d.vwap - d.close, 3))) * rank(delta(d.volume, 3))


@_a("alpha191_008")
def _(d: AlphaData):
    return rank(delta((d.high + d.low) / 2 * 0.2 + d.vwap * 0.8, 4) * -1)


@_a("alpha191_009")
def _(d: AlphaData):
    inner = ((d.high + d.low) / 2 - (delay(d.high, 1) + delay(d.low, 1)) / 2) * (d.high - d.low) / d.volume
    return sma_tdx(inner, 7, 2)


@_a("alpha191_010")
def _(d: AlphaData):
    inner = if_else(lt(d.returns, 0), stddev(d.returns, 20), d.close)
    return rank(tsmax(power(inner, 2), 5))


@_a("alpha191_011")
def _(d: AlphaData):
    return _sum((d.close - d.low - (d.high - d.close)) / (d.high - d.low) * d.volume, 6)


@_a("alpha191_012")
def _(d: AlphaData):
    return rank(d.open - _sum(d.vwap, 10) / 10) * (-1 * rank(abs_(d.close - d.vwap)))


@_a("alpha191_013")
def _(d: AlphaData):
    return power(d.high * d.low, 0.5) - d.vwap


@_a("alpha191_014")
def _(d: AlphaData):
    return d.close - delay(d.close, 5)


@_a("alpha191_015")
def _(d: AlphaData):
    return d.open / delay(d.close, 1) - 1


@_a("alpha191_016")
def _(d: AlphaData):
    return -1 * tsmax(rank(corr(rank(d.volume), rank(d.vwap), 5)), 5)


@_a("alpha191_017")
def _(d: AlphaData):
    return power(rank(d.vwap - tsmax(d.vwap, 15)), delta(d.close, 5))


@_a("alpha191_018")
def _(d: AlphaData):
    return d.close / delay(d.close, 5)


@_a("alpha191_019")
def _(d: AlphaData):
    dc = delay(d.close, 5)
    return if_else(
        lt(d.close, dc), (d.close - dc) / dc,
        if_else(eq(d.close, dc), 0.0, (d.close - dc) / d.close),
    )


@_a("alpha191_020")
def _(d: AlphaData):
    return (d.close - delay(d.close, 6)) / delay(d.close, 6) * 100


# ---------------------------------------------------------------------------
# 021-040
# ---------------------------------------------------------------------------
@_a("alpha191_021")
def _(d: AlphaData):
    # REGBETA(MEAN(CLOSE,6),SEQUENCE(6)) = 对 6 日均值做时序回归斜率
    return ts_slope(_mean(d.close, 6), 6)


@_a("alpha191_022")
def _(d: AlphaData):
    # 原文 SMEAN 为 SMA 笔误
    x = (d.close - _mean(d.close, 6)) / _mean(d.close, 6)
    return sma_tdx(x - delay(x, 3), 12, 1)


@_a("alpha191_023")
def _(d: AlphaData):
    dc = delay(d.close, 1)
    up = sma_tdx(if_else(gt(d.close, dc), stddev(d.close, 20), 0.0), 20, 1)
    dn = sma_tdx(if_else(ge(dc, d.close), stddev(d.close, 20), 0.0), 20, 1)
    return up / (up + dn) * 100


@_a("alpha191_024")
def _(d: AlphaData):
    return sma_tdx(d.close - delay(d.close, 5), 5, 1)


@_a("alpha191_025")
def _(d: AlphaData):
    inner = delta(d.close, 7) * (1 - rank(decay_linear(d.volume / _mean(d.volume, 20), 9)))
    return (-1 * rank(inner)) * (1 + rank(_sum(d.returns, 250)))


@_a("alpha191_026")
def _(d: AlphaData):
    return (_sum(d.close, 7) / 7 - d.close) + corr(d.vwap, delay(d.close, 5), 230)


@_a("alpha191_027")
def _(d: AlphaData):
    x = ((d.close - delay(d.close, 3)) / delay(d.close, 3) + (d.close - delay(d.close, 6)) / delay(d.close, 6)) * 100
    return ts_wma(x, 12)


@_a("alpha191_028")
def _(d: AlphaData):
    # 原文第二处 MAX(HIGH,9)-TSMAX(LOW,9) 为笔误，统一为 TSMAX(HIGH,9)-TSMIN(LOW,9)
    inner = (d.close - tsmin(d.low, 9)) / (tsmax(d.high, 9) - tsmin(d.low, 9)) * 100
    return 3 * sma_tdx(inner, 3, 1) - 2 * sma_tdx(sma_tdx(inner, 3, 1), 3, 1)


@_a("alpha191_029")
def _(d: AlphaData):
    return (d.close - delay(d.close, 6)) / delay(d.close, 6) * d.volume


@_a("alpha191_031")
def _(d: AlphaData):
    return (d.close - _mean(d.close, 12)) / _mean(d.close, 12) * 100


@_a("alpha191_033")
def _(d: AlphaData):
    lo5 = tsmin(d.low, 5)
    return (-1 * lo5 + delay(lo5, 5)) * rank((_sum(d.returns, 240) - _sum(d.returns, 20)) / 220) * tsr(d.volume, 5)


@_a("alpha191_034")
def _(d: AlphaData):
    return _mean(d.close, 12) / d.close


@_a("alpha191_035")
def _(d: AlphaData):
    return min_(
        rank(decay_linear(delta(d.open, 1), 15)),
        rank(decay_linear(corr(d.volume, d.open * 0.65 + d.open * 0.35, 17), 7)),
    ) * -1


@_a("alpha191_036")
def _(d: AlphaData):
    # 原文 CORR 缺窗口，按公开实现取 6；SUM 窗口 2
    return rank(_sum(corr(rank(d.volume), rank(d.vwap), 6), 2))


@_a("alpha191_037")
def _(d: AlphaData):
    x = _sum(d.open, 5) * _sum(d.returns, 5)
    return -1 * rank(x - delay(x, 10))


@_a("alpha191_038")
def _(d: AlphaData):
    return if_else(lt(_mean(d.high, 20), d.high), -1 * delta(d.high, 2), 0.0)


@_a("alpha191_039")
def _(d: AlphaData):
    return (
        rank(decay_linear(delta(d.close, 2), 8))
        - rank(decay_linear(corr(d.vwap * 0.3 + d.open * 0.7, _sum(_mean(d.volume, 180), 37), 14), 12))
    ) * -1


@_a("alpha191_041")
def _(d: AlphaData):
    return rank(tsmax(delta(d.vwap, 3), 5)) * -1


@_a("alpha191_042")
def _(d: AlphaData):
    return (-1 * rank(stddev(d.high, 10))) * corr(d.high, d.volume, 10)


@_a("alpha191_043")
def _(d: AlphaData):
    dc = delay(d.close, 1)
    return _sum(if_else(gt(d.close, dc), d.volume, if_else(lt(d.close, dc), -d.volume, 0.0)), 6)


# ---------------------------------------------------------------------------
# 044-070
# ---------------------------------------------------------------------------
@_a("alpha191_044")
def _(d: AlphaData):
    return tsr(decay_linear(corr(d.low, _mean(d.volume, 10), 7), 6), 4) + tsr(decay_linear(delta(d.vwap, 3), 10), 15)


@_a("alpha191_045")
def _(d: AlphaData):
    return rank(delta(d.close * 0.6 + d.open * 0.4, 1)) * rank(corr(d.vwap, _mean(d.volume, 150), 15))


@_a("alpha191_046")
def _(d: AlphaData):
    return (_mean(d.close, 3) + _mean(d.close, 6) + _mean(d.close, 12) + _mean(d.close, 24)) / (4 * d.close)


@_a("alpha191_047")
def _(d: AlphaData):
    return sma_tdx((tsmax(d.high, 6) - d.close) / (tsmax(d.high, 6) - tsmin(d.low, 6)) * 100, 9, 1)


@_a("alpha191_048")
def _(d: AlphaData):
    s = (sign(d.close - delay(d.close, 1))
         + sign(delay(d.close, 1) - delay(d.close, 2))
         + sign(delay(d.close, 2) - delay(d.close, 3)))
    return -1 * rank(s) * _sum(d.volume, 5) / _sum(d.volume, 20)


def _dm_4951(d: AlphaData):
    """049/050/051 共用：方向性波动的上下行分解（上/下行日各自贡献的和）。"""
    cur = d.high + d.low
    prev = delay(d.high, 1) + delay(d.low, 1)
    mx = max_(abs_(d.high - delay(d.high, 1)), abs_(d.low - delay(d.low, 1)))
    up = _sum(if_else(gt(cur, prev), mx, 0.0), 12)
    dn = _sum(if_else(lt(cur, prev), mx, 0.0), 12)
    return up, dn


@_a("alpha191_049")
def _(d: AlphaData):
    # 分子 SUM(cond_ge?0:X) = 下行日贡献 dn
    up, dn = _dm_4951(d)
    return dn / (dn + up)


@_a("alpha191_050")
def _(d: AlphaData):
    # SUM(cond_le?0:X)/(...) - SUM(cond_ge?0:X)/(...) = (up-dn)/(up+dn)
    up, dn = _dm_4951(d)
    return up / (up + dn) - dn / (dn + up)


@_a("alpha191_051")
def _(d: AlphaData):
    # 分子 SUM(cond_le?0:X) = 上行日贡献 up
    up, dn = _dm_4951(d)
    return up / (up + dn)


@_a("alpha191_052")
def _(d: AlphaData):
    # 原文 "-L" 为 LOW 笔误
    tp = (d.high + d.low + d.close) / 3
    return _sum(max_(0.0, d.high - delay(tp, 1)), 26) / _sum(max_(0.0, delay(tp, 1) - d.low), 26) * 100


@_a("alpha191_053")
def _(d: AlphaData):
    return count_(gt(d.close, delay(d.close, 1)), 12) / 12 * 100


@_a("alpha191_054")
def _(d: AlphaData):
    # 原文 STD 缺窗口，按 DolphinDB 实现取 10（与 CORR 窗口一致）
    return -1 * rank((stddev(abs_(d.close - d.open), 10) + (d.close - d.open)) + corr(d.close, d.open, 10))


def _dm_55137(d: AlphaData):
    """055/137 共用：K线下影/上影主导的分母分支。"""
    hc = abs_(d.high - delay(d.close, 1))
    lc = abs_(d.low - delay(d.close, 1))
    hl = abs_(d.high - delay(d.low, 1))
    co = abs_(delay(d.close, 1) - delay(d.open, 1))
    branch = if_else(
        and_(gt(hc, lc), gt(hc, hl)), hc + lc / 2 + co / 4,
        if_else(and_(gt(lc, hl), gt(lc, hc)), lc + hc / 2 + co / 4, hl + co / 4),
    )
    return hc, lc, branch


@_a("alpha191_055")
def _(d: AlphaData):
    hc, lc, branch = _dm_55137(d)
    numer = 16 * (d.close - delay(d.close, 1) + (d.close - d.open) / 2 + delay(d.close, 1) - delay(d.open, 1))
    return _sum(numer / branch * max_(hc, lc), 20)


@_a("alpha191_056")
def _(d: AlphaData):
    inner = rank(power(rank(corr(_sum((d.high + d.low) / 2, 19), _sum(_mean(d.volume, 40), 19), 13)), 5))
    return lt(rank(d.open - tsmin(d.open, 12)), inner)


@_a("alpha191_057")
def _(d: AlphaData):
    return sma_tdx((d.close - tsmin(d.low, 9)) / (tsmax(d.high, 9) - tsmin(d.low, 9)) * 100, 3, 1)


@_a("alpha191_058")
def _(d: AlphaData):
    return count_(gt(d.close, delay(d.close, 1)), 20) / 20 * 100


@_a("alpha191_059")
def _(d: AlphaData):
    dc = delay(d.close, 1)
    inner = if_else(
        eq(d.close, dc), 0.0,
        d.close - if_else(gt(d.close, dc), min_(d.low, dc), max_(d.high, dc)),
    )
    return _sum(inner, 20)


@_a("alpha191_060")
def _(d: AlphaData):
    return _sum((d.close - d.low - (d.high - d.close)) / (d.high - d.low) * d.volume, 20)


@_a("alpha191_061")
def _(d: AlphaData):
    return max_(
        rank(decay_linear(delta(d.vwap, 1), 12)),
        rank(decay_linear(rank(corr(d.low, _mean(d.volume, 80), 8)), 17)),
    ) * -1


@_a("alpha191_062")
def _(d: AlphaData):
    return -1 * corr(d.high, rank(d.volume), 5)


@_a("alpha191_063")
def _(d: AlphaData):
    dc = d.close - delay(d.close, 1)
    return sma_tdx(max_(dc, 0.0), 6, 1) / sma_tdx(abs_(dc), 6, 1) * 100


@_a("alpha191_064")
def _(d: AlphaData):
    return max_(
        rank(decay_linear(corr(rank(d.vwap), rank(d.volume), 4), 4)),
        rank(decay_linear(tsmax(corr(rank(d.close), rank(_mean(d.volume, 60)), 4), 13), 14)),
    ) * -1


@_a("alpha191_065")
def _(d: AlphaData):
    return _mean(d.close, 6) / d.close


@_a("alpha191_066")
def _(d: AlphaData):
    return (d.close - _mean(d.close, 6)) / _mean(d.close, 6) * 100


@_a("alpha191_067")
def _(d: AlphaData):
    dc = d.close - delay(d.close, 1)
    return sma_tdx(max_(dc, 0.0), 24, 1) / sma_tdx(abs_(dc), 24, 1) * 100


@_a("alpha191_068")
def _(d: AlphaData):
    inner = ((d.high + d.low) / 2 - (delay(d.high, 1) + delay(d.low, 1)) / 2) * (d.high - d.low) / d.volume
    return sma_tdx(inner, 15, 2)


@_a("alpha191_069")
def _(d: AlphaData):
    dop = delay(d.open, 1)
    dtm = if_else(gt(d.open, dop), max_(d.high - d.open, d.open - dop), 0.0)
    dbm = if_else(lt(d.open, dop), max_(d.open - d.low, dop - d.open), 0.0)
    s_dtm, s_dbm = _sum(dtm, 20), _sum(dbm, 20)
    return if_else(
        gt(s_dtm, s_dbm), (s_dtm - s_dbm) / s_dtm,
        if_else(eq(s_dtm, s_dbm), 0.0, (s_dtm - s_dbm) / s_dbm),
    )


@_a("alpha191_070")
def _(d: AlphaData):
    return stddev(d.amount, 6)


# ---------------------------------------------------------------------------
# 071-100
# ---------------------------------------------------------------------------
@_a("alpha191_071")
def _(d: AlphaData):
    return (d.close - _mean(d.close, 24)) / _mean(d.close, 24) * 100


@_a("alpha191_072")
def _(d: AlphaData):
    return sma_tdx((tsmax(d.high, 6) - d.close) / (tsmax(d.high, 6) - tsmin(d.low, 6)) * 100, 15, 1)


@_a("alpha191_073")
def _(d: AlphaData):
    return (
        tsr(decay_linear(decay_linear(corr(d.close, d.volume, 10), 16), 4), 5)
        - rank(decay_linear(corr(d.vwap, _mean(d.volume, 30), 4), 3))
    ) * -1


@_a("alpha191_074")
def _(d: AlphaData):
    return (
        rank(corr(_sum(d.low * 0.35 + d.vwap * 0.65, 20), _sum(_mean(d.volume, 40), 20), 7))
        + rank(corr(rank(d.vwap), rank(d.volume), 6))
    )


@_a("alpha191_076")
def _(d: AlphaData):
    x = abs_(d.returns) / d.volume
    return stddev(x, 20) / _mean(x, 20)


@_a("alpha191_077")
def _(d: AlphaData):
    return min_(
        rank(decay_linear((d.high + d.low) / 2 + d.high - (d.vwap + d.high), 20)),
        rank(decay_linear(corr((d.high + d.low) / 2, _mean(d.volume, 40), 3), 6)),
    )


@_a("alpha191_078")
def _(d: AlphaData):
    tp = (d.high + d.low + d.close) / 3
    return (tp - _mean(tp, 12)) / (0.015 * _mean(abs_(d.close - _mean(tp, 12)), 12))


@_a("alpha191_079")
def _(d: AlphaData):
    dc = d.close - delay(d.close, 1)
    return sma_tdx(max_(dc, 0.0), 12, 1) / sma_tdx(abs_(dc), 12, 1) * 100


@_a("alpha191_080")
def _(d: AlphaData):
    return (d.volume - delay(d.volume, 5)) / delay(d.volume, 5) * 100


@_a("alpha191_081")
def _(d: AlphaData):
    return sma_tdx(d.volume, 21, 2)


@_a("alpha191_082")
def _(d: AlphaData):
    return sma_tdx((tsmax(d.high, 6) - d.close) / (tsmax(d.high, 6) - tsmin(d.low, 6)) * 100, 20, 1)


@_a("alpha191_083")
def _(d: AlphaData):
    return -1 * rank(cov(rank(d.high), rank(d.volume), 5))


@_a("alpha191_084")
def _(d: AlphaData):
    dc = delay(d.close, 1)
    return _sum(if_else(gt(d.close, dc), d.volume, if_else(lt(d.close, dc), -d.volume, 0.0)), 20)


@_a("alpha191_085")
def _(d: AlphaData):
    return tsr(d.volume / _mean(d.volume, 20), 20) * tsr(-1 * delta(d.close, 7), 8)


@_a("alpha191_086")
def _(d: AlphaData):
    x = (delay(d.close, 20) - delay(d.close, 10)) / 10 - (delay(d.close, 10) - d.close) / 10
    return if_else(
        lt(0.25, x), -1.0,
        if_else(lt(x, 0), 1.0, -1 * (d.close - delay(d.close, 1))),
    )


@_a("alpha191_087")
def _(d: AlphaData):
    # (LOW*0.9)+(LOW*0.1) 恒等于 LOW
    inner = (d.low - d.vwap) / (d.open - (d.high + d.low) / 2)
    return (rank(decay_linear(delta(d.vwap, 4), 7)) + tsr(decay_linear(inner, 11), 7)) * -1


@_a("alpha191_088")
def _(d: AlphaData):
    return (d.close - delay(d.close, 20)) / delay(d.close, 20) * 100


@_a("alpha191_089")
def _(d: AlphaData):
    s13, s27 = sma_tdx(d.close, 13, 2), sma_tdx(d.close, 27, 2)
    return 2 * (s13 - s27 - sma_tdx(s13 - s27, 10, 2))


@_a("alpha191_090")
def _(d: AlphaData):
    return rank(corr(rank(d.vwap), rank(d.volume), 5)) * -1


@_a("alpha191_091")
def _(d: AlphaData):
    return (rank(d.close - tsmax(d.close, 5)) * rank(corr(_mean(d.volume, 40), d.low, 5))) * -1


@_a("alpha191_092")
def _(d: AlphaData):
    return max_(
        rank(decay_linear(delta(d.close * 0.35 + d.vwap * 0.65, 2), 3)),
        tsr(decay_linear(abs_(corr(_mean(d.volume, 180), d.close, 13)), 5), 15),
    ) * -1


@_a("alpha191_093")
def _(d: AlphaData):
    dop = delay(d.open, 1)
    return _sum(if_else(ge(d.open, dop), 0.0, max_(d.open - d.low, d.open - dop)), 20)


@_a("alpha191_094")
def _(d: AlphaData):
    dc = delay(d.close, 1)
    return _sum(if_else(gt(d.close, dc), d.volume, if_else(lt(d.close, dc), -d.volume, 0.0)), 30)


@_a("alpha191_095")
def _(d: AlphaData):
    return stddev(d.amount, 20)


@_a("alpha191_096")
def _(d: AlphaData):
    inner = (d.close - tsmin(d.low, 9)) / (tsmax(d.high, 9) - tsmin(d.low, 9)) * 100
    return sma_tdx(sma_tdx(inner, 3, 1), 3, 1)


@_a("alpha191_097")
def _(d: AlphaData):
    return stddev(d.volume, 10)


@_a("alpha191_098")
def _(d: AlphaData):
    x = delta(_mean(d.close, 100), 100) / delay(d.close, 100)
    return if_else(
        or_(lt(x, 0.05), eq(x, 0.05)),
        -1 * (d.close - tsmin(d.close, 100)),
        -1 * delta(d.close, 3),
    )


@_a("alpha191_099")
def _(d: AlphaData):
    return -1 * rank(cov(rank(d.close), rank(d.volume), 5))


@_a("alpha191_100")
def _(d: AlphaData):
    return stddev(d.volume, 20)


# ---------------------------------------------------------------------------
# 101-120
# ---------------------------------------------------------------------------
@_a("alpha191_101")
def _(d: AlphaData):
    a = rank(corr(d.close, _sum(_mean(d.volume, 30), 37), 15))
    b = rank(corr(rank(d.high * 0.1 + d.vwap * 0.9), rank(d.volume), 11))
    return if_else(lt(a, b), -1.0, 1.0)


@_a("alpha191_102")
def _(d: AlphaData):
    dv = d.volume - delay(d.volume, 1)
    return sma_tdx(max_(dv, 0.0), 6, 1) / sma_tdx(abs_(dv), 6, 1) * 100


@_a("alpha191_103")
def _(d: AlphaData):
    return (20 - lowday(d.low, 20)) / 20 * 100


@_a("alpha191_104")
def _(d: AlphaData):
    return -1 * delta(corr(d.high, d.volume, 5), 5) * rank(stddev(d.close, 20))


@_a("alpha191_105")
def _(d: AlphaData):
    return -1 * corr(rank(d.open), rank(d.volume), 10)


@_a("alpha191_106")
def _(d: AlphaData):
    return d.close - delay(d.close, 20)


@_a("alpha191_107")
def _(d: AlphaData):
    return (
        (-1 * rank(d.open - delay(d.high, 1)))
        * rank(d.open - delay(d.close, 1))
        * rank(d.open - delay(d.low, 1))
    )


@_a("alpha191_108")
def _(d: AlphaData):
    return power(
        rank(d.high - tsmin(d.high, 2)),
        rank(corr(d.vwap, _mean(d.volume, 120), 6)),
    ) * -1


@_a("alpha191_109")
def _(d: AlphaData):
    a = sma_tdx(d.high - d.low, 10, 2)
    return a / sma_tdx(a, 10, 2)


@_a("alpha191_110")
def _(d: AlphaData):
    dc = delay(d.close, 1)
    return _sum(max_(0.0, d.high - dc), 20) / _sum(max_(0.0, dc - d.low), 20) * 100


@_a("alpha191_111")
def _(d: AlphaData):
    v = d.volume * ((d.close - d.low) - (d.high - d.close)) / (d.high - d.low)
    return sma_tdx(v, 11, 2) - sma_tdx(v, 4, 2)


@_a("alpha191_112")
def _(d: AlphaData):
    dv = d.close - delay(d.close, 1)
    up = _sum(if_else(gt(dv, 0.0), dv, 0.0), 12)
    dn = _sum(if_else(lt(dv, 0.0), abs_(dv), 0.0), 12)
    return (up - dn) / (up + dn) * 100


@_a("alpha191_113")
def _(d: AlphaData):
    return -1 * (
        rank(_sum(delay(d.close, 5), 20) / 20)
        * corr(d.close, d.volume, 2)
        * rank(corr(_sum(d.close, 5), _sum(d.close, 20), 2))
    )


@_a("alpha191_114")
def _(d: AlphaData):
    hl5 = (d.high - d.low) / (_sum(d.close, 5) / 5)
    return rank(delay(hl5, 2)) * rank(rank(d.volume)) / (hl5 / (d.vwap - d.close))


@_a("alpha191_115")
def _(d: AlphaData):
    return power(
        rank(corr(d.high * 0.9 + d.close * 0.1, _mean(d.volume, 30), 10)),
        rank(corr(tsr((d.high + d.low) / 2, 4), tsr(d.volume, 10), 7)),
    )


@_a("alpha191_116")
def _(d: AlphaData):
    # REGBETA(CLOSE, SEQUENCE(20))
    return ts_slope(d.close, 20)


@_a("alpha191_117")
def _(d: AlphaData):
    return tsr(d.volume, 32) * (1 - tsr(d.close + d.high - d.low, 16)) * (1 - tsr(d.returns, 32))


@_a("alpha191_118")
def _(d: AlphaData):
    return _sum(d.high - d.open, 20) / _sum(d.open - d.low, 20) * 100


@_a("alpha191_119")
def _(d: AlphaData):
    return (
        rank(decay_linear(corr(d.vwap, _sum(_mean(d.volume, 5), 26), 5), 7))
        - rank(decay_linear(tsr(tsmin(corr(rank(d.open), rank(_mean(d.volume, 15)), 21), 9), 7), 8))
    )


@_a("alpha191_120")
def _(d: AlphaData):
    return rank(d.vwap - d.close) / rank(d.vwap + d.close)


# ---------------------------------------------------------------------------
# 121-140
# ---------------------------------------------------------------------------
@_a("alpha191_121")
def _(d: AlphaData):
    return power(
        rank(d.vwap - tsmin(d.vwap, 12)),
        tsr(corr(tsr(d.vwap, 20), tsr(_mean(d.volume, 60), 2), 18), 3),
    ) * -1


@_a("alpha191_122")
def _(d: AlphaData):
    a = sma_tdx(sma_tdx(sma_tdx(log_(d.close), 13, 2), 13, 2), 13, 2)
    return (a - delay(a, 1)) / delay(a, 1)


@_a("alpha191_123")
def _(d: AlphaData):
    a = rank(corr(_sum((d.high + d.low) / 2, 20), _sum(_mean(d.volume, 60), 20), 9))
    b = rank(corr(d.low, d.volume, 6))
    return if_else(lt(a, b), -1.0, 1.0)


@_a("alpha191_124")
def _(d: AlphaData):
    return (d.close - d.vwap) / decay_linear(rank(tsmax(d.close, 30)), 2)


@_a("alpha191_125")
def _(d: AlphaData):
    return (
        rank(decay_linear(corr(d.vwap, _mean(d.volume, 80), 17), 20))
        / rank(decay_linear(delta(d.close * 0.5 + d.vwap * 0.5, 3), 16))
    )


@_a("alpha191_126")
def _(d: AlphaData):
    return (d.close + d.high + d.low) / 3


@_a("alpha191_127")
def _(d: AlphaData):
    # 原文 MEAN 缺窗口，按 DolphinDB 取 12
    mx = tsmax(d.close, 12)
    x = 100 * (d.close - mx) / mx
    return power(_mean(power(x, 2), 12), 0.5)


@_a("alpha191_128")
def _(d: AlphaData):
    tp = (d.close + d.high + d.low) / 3
    dtp = delay(tp, 1)
    up = _sum(if_else(gt(tp, dtp), tp * d.volume, 0.0), 14)
    dn = _sum(if_else(lt(tp, dtp), tp * d.volume, 0.0), 14)
    return 100 - 100 / (1 + up / dn)


@_a("alpha191_129")
def _(d: AlphaData):
    dv = d.close - delay(d.close, 1)
    return _sum(if_else(lt(dv, 0.0), abs_(dv), 0.0), 12)


@_a("alpha191_130")
def _(d: AlphaData):
    return (
        rank(decay_linear(corr((d.high + d.low) / 2, _mean(d.volume, 40), 9), 10))
        / rank(decay_linear(corr(rank(d.vwap), rank(d.volume), 7), 3))
    )


@_a("alpha191_131")
def _(d: AlphaData):
    return power(rank(delta(d.vwap, 1)), tsr(corr(d.close, _mean(d.volume, 50), 18), 18))


@_a("alpha191_132")
def _(d: AlphaData):
    return _mean(d.amount, 20)


@_a("alpha191_133")
def _(d: AlphaData):
    return (20 - highday(d.high, 20)) / 20 * 100 - (20 - lowday(d.low, 20)) / 20 * 100


@_a("alpha191_134")
def _(d: AlphaData):
    return (d.close - delay(d.close, 12)) / delay(d.close, 12) * d.volume


@_a("alpha191_135")
def _(d: AlphaData):
    return sma_tdx(delay(d.close / delay(d.close, 20), 1), 20, 1)


@_a("alpha191_136")
def _(d: AlphaData):
    return -1 * rank(delta(d.returns, 3)) * corr(d.open, d.volume, 10)


@_a("alpha191_137")
def _(d: AlphaData):
    hc, lc, branch = _dm_55137(d)
    # -DELAY(CLOSE,1)+DELAY(CLOSE,1) 相消
    numer = 16 * (d.close + (d.close - d.open) / 2 - delay(d.open, 1))
    return numer / branch * max_(hc, lc)


@_a("alpha191_138")
def _(d: AlphaData):
    return (
        rank(decay_linear(delta(d.low * 0.7 + d.vwap * 0.3, 3), 20))
        - tsr(decay_linear(tsr(corr(tsr(d.low, 8), tsr(_mean(d.volume, 60), 17), 5), 19), 16), 7)
    ) * -1


@_a("alpha191_140")
def _(d: AlphaData):
    # MIN(X,Y) 两面板逐元素取小
    return min_(
        rank(decay_linear((rank(d.open) + rank(d.low)) - (rank(d.high) + rank(d.close)), 8)),
        tsr(decay_linear(corr(tsr(d.close, 8), tsr(_mean(d.volume, 60), 20), 8), 7), 3),
    )


# ---------------------------------------------------------------------------
# 141-170
# ---------------------------------------------------------------------------
@_a("alpha191_141")
def _(d: AlphaData):
    return rank(corr(rank(d.high), rank(_mean(d.volume, 15)), 9)) * -1


@_a("alpha191_142")
def _(d: AlphaData):
    return (
        (-1 * rank(tsr(d.close, 10)))
        * rank(delta(delta(d.close, 1), 1))
        * rank(tsr(d.volume / _mean(d.volume, 20), 5))
    )


@_a("alpha191_144")
def _(d: AlphaData):
    cond = lt(d.close, delay(d.close, 1))
    part = abs_(d.close / delay(d.close, 1) - 1) / d.amount
    return sumif(part, 20, cond) / count_(cond, 20)


@_a("alpha191_145")
def _(d: AlphaData):
    return (_mean(d.volume, 9) - _mean(d.volume, 26)) / _mean(d.volume, 12) * 100


@_a("alpha191_146")
def _(d: AlphaData):
    # 原文末项 SMA(X,60) 二参歧义，按 JoinQuant 校正值 SMA(dev^2,61,2)
    dev = d.returns - sma_tdx(d.returns, 61, 2)
    return _mean(dev, 20) * dev / sma_tdx(power(dev, 2), 61, 2)


@_a("alpha191_147")
def _(d: AlphaData):
    # REGBETA(MEAN(CLOSE,12), SEQUENCE(12))
    return ts_slope(_mean(d.close, 12), 12)


@_a("alpha191_148")
def _(d: AlphaData):
    a = rank(corr(d.open, _sum(_mean(d.volume, 60), 9), 6))
    b = rank(d.open - tsmin(d.open, 14))
    return if_else(lt(a, b), -1.0, 1.0)


@_a("alpha191_150")
def _(d: AlphaData):
    return (d.close + d.high + d.low) / 3 * d.volume


@_a("alpha191_151")
def _(d: AlphaData):
    return sma_tdx(d.close - delay(d.close, 20), 20, 1)


@_a("alpha191_152")
def _(d: AlphaData):
    inner = sma_tdx(delay(d.close / delay(d.close, 9), 1), 9, 1)
    return sma_tdx(_mean(delay(inner, 1), 12) - _mean(delay(inner, 1), 26), 9, 1)


@_a("alpha191_153")
def _(d: AlphaData):
    return (_mean(d.close, 3) + _mean(d.close, 6) + _mean(d.close, 12) + _mean(d.close, 24)) / 4


@_a("alpha191_154")
def _(d: AlphaData):
    cond = lt(d.vwap - tsmin(d.vwap, 16), corr(d.vwap, _mean(d.volume, 180), 18))
    return if_else(cond, 1.0, -1.0)


@_a("alpha191_155")
def _(d: AlphaData):
    a = sma_tdx(d.volume, 13, 2)
    b = sma_tdx(d.volume, 27, 2)
    return a - b - sma_tdx(a - b, 10, 2)


@_a("alpha191_156")
def _(d: AlphaData):
    base = d.open * 0.15 + d.low * 0.85
    return max_(
        rank(decay_linear(delta(d.vwap, 5), 3)),
        rank(decay_linear(-1 * delta(base, 2) / base, 3)),
    ) * -1


@_a("alpha191_157")
def _(d: AlphaData):
    # PROD(...,1)/SUM(...,1) 为 1 日窗口恒等变换，直接展开
    inner = -1 * rank(delta(d.close - 1.0, 5))
    y = rank(rank(log_(tsmin(rank(rank(inner)), 2))))
    return tsmin(y, 5) + tsr(delay(-1 * d.returns, 6), 5)


@_a("alpha191_158")
def _(d: AlphaData):
    a = sma_tdx(d.close, 15, 2)
    return ((d.high - a) - (d.low - a)) / d.close


@_a("alpha191_159")
def _(d: AlphaData):
    # 原文 HGIH 为 HIGH 笔误
    dc = delay(d.close, 1)
    lo = min_(d.low, dc)
    rng = max_(d.high, dc) - lo
    return (
        (d.close - _sum(lo, 6)) / _sum(rng, 6) * 12 * 24
        + (d.close - _sum(lo, 12)) / _sum(rng, 12) * 6 * 24
        + (d.close - _sum(lo, 24)) / _sum(rng, 24) * 6 * 24
    ) * 100 / (6 * 12 + 6 * 24 + 12 * 24)


@_a("alpha191_160")
def _(d: AlphaData):
    cond = le(d.close, delay(d.close, 1))
    return sma_tdx(if_else(cond, stddev(d.close, 20), 0.0), 20, 1)


def _atr_hl(d: AlphaData, n: int):
    """161/175 共用：MAX(MAX(HIGH-LOW,|DC-HIGH|),|DC-LOW|) 的 n 日均值。"""
    dc = delay(d.close, 1)
    return _mean(max_(max_(d.high - d.low, abs_(dc - d.high)), abs_(dc - d.low)), n)


@_a("alpha191_161")
def _(d: AlphaData):
    return _atr_hl(d, 12)


@_a("alpha191_162")
def _(d: AlphaData):
    dv = d.close - delay(d.close, 1)
    rsi = sma_tdx(max_(dv, 0.0), 12, 1) / sma_tdx(abs_(dv), 12, 1) * 100
    return (rsi - tsmin(rsi, 12)) / (tsmax(rsi, 12) - tsmin(rsi, 12))


@_a("alpha191_163")
def _(d: AlphaData):
    return rank((-1 * d.returns) * _mean(d.volume, 20) * d.vwap * (d.high - d.close))


@_a("alpha191_164")
def _(d: AlphaData):
    dc = delay(d.close, 1)
    inner = if_else(gt(d.close, dc), 1.0 / (d.close - dc), 1.0)
    return sma_tdx((inner - tsmin(inner, 12)) / (d.high - d.low) * 100, 13, 2)


@_a("alpha191_166")
def _(d: AlphaData):
    # 分母 SUM(MEAN(ratio,20)^2,20)，与 DolphinDB/JoinQuant 实现一致
    ratio = d.close / delay(d.close, 1)
    p1 = -20 * (20 - 1) ** 1.5 * _sum(d.returns - _mean(d.returns, 20), 20)
    p2 = (20 - 1) * (20 - 2) * power(_sum(power(_mean(ratio, 20), 2), 20), 1.5)
    return p1 / p2


@_a("alpha191_167")
def _(d: AlphaData):
    dv = d.close - delay(d.close, 1)
    return _sum(if_else(gt(dv, 0.0), dv, 0.0), 12)


@_a("alpha191_168")
def _(d: AlphaData):
    return -1 * d.volume / _mean(d.volume, 20)


@_a("alpha191_169")
def _(d: AlphaData):
    inner = sma_tdx(d.close - delay(d.close, 1), 9, 1)
    return sma_tdx(_mean(delay(inner, 1), 12) - _mean(delay(inner, 1), 26), 10, 1)


@_a("alpha191_170")
def _(d: AlphaData):
    return (
        (rank(1.0 / d.close) * d.volume / _mean(d.volume, 20))
        * (d.high * rank(d.high - d.close) / (_sum(d.high, 5) / 5))
        - rank(d.vwap - delay(d.vwap, 5))
    )


# ---------------------------------------------------------------------------
# 171-191
# ---------------------------------------------------------------------------
@_a("alpha191_171")
def _(d: AlphaData):
    return -1 * (d.low - d.close) * power(d.open, 5) / ((d.close - d.high) * power(d.close, 5))


def _adx_172186(d: AlphaData):
    """172/186 共用：DMI/ADX 形态。"""
    hd = d.high - delay(d.high, 1)
    ld = delay(d.low, 1) - d.low
    dc = delay(d.close, 1)
    tr = max_(max_(d.high - d.low, abs_(d.high - dc)), abs_(d.low - dc))
    s_tr = _sum(tr, 14)
    dm_p = _sum(if_else(and_(gt(hd, 0.0), gt(hd, ld)), hd, 0.0), 14) * 100 / s_tr
    dm_m = _sum(if_else(and_(gt(ld, 0.0), gt(ld, hd)), ld, 0.0), 14) * 100 / s_tr
    return _mean(abs_(dm_m - dm_p) / (dm_m + dm_p) * 100, 6)


@_a("alpha191_172")
def _(d: AlphaData):
    return _adx_172186(d)


@_a("alpha191_173")
def _(d: AlphaData):
    a = sma_tdx(d.close, 13, 2)
    b = sma_tdx(a, 13, 2)
    return 3 * a - 2 * b + sma_tdx(sma_tdx(sma_tdx(log_(d.close), 13, 2), 13, 2), 13, 2)


@_a("alpha191_174")
def _(d: AlphaData):
    cond = gt(d.close, delay(d.close, 1))
    return sma_tdx(if_else(cond, stddev(d.close, 20), 0.0), 20, 1)


@_a("alpha191_175")
def _(d: AlphaData):
    return _atr_hl(d, 6)


@_a("alpha191_176")
def _(d: AlphaData):
    rng = tsmax(d.high, 12) - tsmin(d.low, 12)
    return corr(rank((d.close - tsmin(d.low, 12)) / rng), rank(d.volume), 6)


@_a("alpha191_177")
def _(d: AlphaData):
    return (20 - highday(d.high, 20)) / 20 * 100


@_a("alpha191_178")
def _(d: AlphaData):
    return (d.close - delay(d.close, 1)) / delay(d.close, 1) * d.volume


@_a("alpha191_179")
def _(d: AlphaData):
    return rank(corr(d.vwap, d.volume, 4)) * rank(corr(rank(d.low), rank(_mean(d.volume, 50)), 12))


@_a("alpha191_180")
def _(d: AlphaData):
    dv = delta(d.close, 7)
    ts_part = -1 * tsr(abs_(dv), 60) * sign(dv)
    cond = lt(_mean(d.volume, 20), d.volume)
    return if_else(cond, ts_part, -1 * d.volume)


@_a("alpha191_184")
def _(d: AlphaData):
    return rank(corr(delay(d.open - d.close, 1), d.close, 200)) + rank(d.open - d.close)


@_a("alpha191_185")
def _(d: AlphaData):
    return rank(-1 * power(1.0 - d.open / d.close, 2))


@_a("alpha191_186")
def _(d: AlphaData):
    adx = _adx_172186(d)
    return (adx + delay(adx, 6)) / 2


@_a("alpha191_187")
def _(d: AlphaData):
    dop = delay(d.open, 1)
    part = if_else(le(d.open, dop), 0.0, max_(d.high - d.open, d.open - dop))
    return _sum(part, 20)


@_a("alpha191_188")
def _(d: AlphaData):
    hl = d.high - d.low
    a = sma_tdx(hl, 11, 2)
    return (hl - a) / a * 100


@_a("alpha191_189")
def _(d: AlphaData):
    return _mean(abs_(d.close - _mean(d.close, 6)), 6)


@_a("alpha191_190")
def _(d: AlphaData):
    thr = power(d.close / delay(d.close, 19), 1.0 / 20) - 1
    diff2 = power(d.returns - thr, 2)
    up = gt(d.returns, thr)
    dn = lt(d.returns, thr)
    numer = (count_(up, 20) - 1) * sumif(diff2, 20, dn)
    denom = count_(dn, 20) * sumif(diff2, 20, up)
    return log_(numer / denom)


@_a("alpha191_191")
def _(d: AlphaData):
    return corr(_mean(d.volume, 20), d.low, 5) + (d.high + d.low) / 2 - d.close


def compute_alpha191(d: AlphaData, skip: bool = True) -> dict[str, pd.DataFrame]:
    """计算全部 alpha191 因子面板（跳过清单内的因子默认不入结果）。"""
    out: dict[str, pd.DataFrame] = {}
    for name, fn in ALPHA191.items():
        if skip and name in SKIPPED_191:
            continue
        panel = fn(d)
        out[name] = panel.replace([np.inf, -np.inf], np.nan)
    return out

