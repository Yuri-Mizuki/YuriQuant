"""
自研技术指标函数库（单股序列 → 成品指标）
=========================================

纯 pandas/numpy 实现，**离线可用**，不依赖 AmazingData SDK。

定位（2026-08-17 抽取，收敛"第三份"实现）：
本模块原为 ``scripts/build_technical_factors.py`` 中的 ``_calc_indicators`` / ``_calc_sar``，
被因子构建（build_technical_factors）、walk-forward（walk_forward）与两段回测
（backtest_two_periods）三个调用方分别 import —— 属散落的重复逻辑。现收敛为
独立模块统一提供。

**与算子空间的职责边界**：
- ``factor/operators.py``：**挖掘/合成用**的原子算子空间（date×code 面板 → 面板，
  单输入可组合，供 GFlowNet / 遗传挖掘枚举）。
- ``factor/technical.py``（本模块）：**成品技术指标**（单股 5 序列 → 9 指标 dict，
  多字段、跨窗口、有内部状态），供"构建技术因子面板"与"回测逐股算因子"共用。

本模块的指标与 ``factor/technical_indicators.py``（AmazingData SDK 语义版）是
两套不同口径（如 RSI 平滑方式、OBV 返回形态），**不合并**；本模块是自研 pandas 版。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["calc_sar", "calc_indicators"]


def calc_sar(close: pd.Series, high: pd.Series, low: pd.Series,
             n: int = 4, step: float = 0.02, max_af: float = 0.2) -> pd.Series:
    """Wilder 抛物线 SAR（逐股迭代）。

    初始方向：前 n 日收盘净涨跌；多头 SAR 起点=前 n 日最低价，空头=最高价。
    AF 从 step 起，每创新极值 +step，上限 max_af；翻转时 AF 重置。
    返回值与 close 同索引的 SAR 序列（前 n 日 NaN）。
    """
    c, h, l = close.to_numpy(float), high.to_numpy(float), low.to_numpy(float)
    sar = np.full(len(c), np.nan)
    if len(c) <= n or not np.isfinite(h[:n]).any() or not np.isfinite(l[:n]).any():
        return pd.Series(sar, index=close.index)
    # 初始方向：前 n 日净涨跌
    bull = c[n - 1] >= c[0]
    if bull:
        ep = float(np.nanmax(h[:n]))
        sar_val = float(np.nanmin(l[:n]))
    else:
        ep = float(np.nanmin(l[:n]))
        sar_val = float(np.nanmax(h[:n]))
    af = step
    for i in range(n, len(c)):
        sar_val = sar_val + af * (ep - sar_val)
        if bull:
            # SAR 不高于前两日最低价（避免突进）
            lb = min(l[i - 1], l[i - 2]) if i >= 2 else l[i - 1]
            sar_val = min(sar_val, lb)
            if h[i] > ep:
                ep = h[i]
                af = min(af + step, max_af)
            if l[i] < sar_val:
                bull = False
                sar_val = ep
                ep = l[i]
                af = step
        else:
            hb = max(h[i - 1], h[i - 2]) if i >= 2 else h[i - 1]
            sar_val = max(sar_val, hb)
            if l[i] < ep:
                ep = l[i]
                af = min(af + step, max_af)
            if h[i] > sar_val:
                bull = True
                sar_val = ep
                ep = h[i]
                af = step
        sar[i] = sar_val
    return pd.Series(sar, index=close.index)


def calc_indicators(close: pd.Series, high: pd.Series, low: pd.Series,
                    open_: pd.Series, volume: pd.Series) -> dict[str, pd.Series]:
    """单只股票的 9 个指标序列（全部只用 t 及以前数据）。"""
    out: dict[str, pd.Series] = {}
    ref_c = close.shift(1)

    # MACD 柱
    dif = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(
        span=26, adjust=False, min_periods=26).mean()
    dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = 2 * (dif - dea)

    # RSI(12)
    diff = close - ref_c
    pos = diff.clip(lower=0.0)
    up = pos.ewm(alpha=1 / 12, adjust=False, min_periods=12).mean()
    dn = diff.abs().ewm(alpha=1 / 12, adjust=False, min_periods=12).mean()
    out["rsi_12"] = (up / dn.replace(0.0, np.nan) * 100).clip(0, 100)

    # KDJ-J
    llv = low.rolling(9, min_periods=9).min()
    hhv = high.rolling(9, min_periods=9).max()
    rsv = (close - llv) / (hhv - llv).replace(0.0, np.nan) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False, min_periods=9).mean()
    d = k.ewm(alpha=1 / 3, adjust=False, min_periods=9).mean()
    out["kdj_j"] = 3 * k - 2 * d

    # TRIX(12,9)
    mtr = close.ewm(span=12, adjust=False, min_periods=12).mean()
    mtr = mtr.ewm(span=12, adjust=False, min_periods=12).mean()
    mtr = mtr.ewm(span=12, adjust=False, min_periods=12).mean()
    out["trix_12"] = (mtr - mtr.shift(1)) / mtr.shift(1).replace(0.0, np.nan) * 100

    # OBV 乖离
    direction = np.sign(close - ref_c).fillna(0.0)
    obv = (direction * volume).cumsum()
    obv[obv.index[0]] = volume.iloc[0] if len(volume) else np.nan
    obv_ma = obv.rolling(30, min_periods=10).mean()
    out["obv_dev"] = obv / obv_ma.replace(0.0, np.nan) - 1.0

    # WAD 乖离
    mida = close - pd.concat([low, ref_c], axis=1).min(axis=1)
    midb = (close - pd.concat([ref_c, high], axis=1).max(axis=1)).where(close < ref_c, 0.0)
    wad_unit = mida.where(close > ref_c, midb)
    wad = wad_unit.cumsum()
    wad_ma = wad.rolling(30, min_periods=10).mean()
    out["wad_dev"] = wad / wad_ma.replace(0.0, np.nan) - 1.0

    # ASI(26,10)
    aa = (high - ref_c).abs()
    bb = (low - ref_c).abs()
    cc = (high - low.shift(1)).abs()
    dd = (ref_c - open_.shift(1)).abs()
    r_a = aa + bb / 2 + dd / 4
    r_b = bb + aa / 2 + dd / 4
    r_c = cc + dd / 4
    r = r_a.where((aa > bb) & (aa > cc), r_b.where((bb > cc) & (bb > aa), r_c))
    x = close - ref_c + (close - open_) / 2 + ref_c - open_.shift(1)
    si = 16 * x / r.replace(0.0, np.nan) * pd.concat([aa, bb], axis=1).max(axis=1)
    out["asi_26"] = si.rolling(26, min_periods=10).sum()

    # CHO
    mid = (volume * (2 * close - high - low) / (high + low).replace(0.0, np.nan)).cumsum()
    out["cho"] = (mid.rolling(10, min_periods=10).mean()
                  - mid.rolling(20, min_periods=10).mean()) / 100

    # SAR 偏离
    sar = calc_sar(close, high, low)
    out["sar_dev"] = (close - sar) / close.replace(0.0, np.nan)

    return out
