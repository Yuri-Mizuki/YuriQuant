"""Mock 行情面板生成（2026-08-31 从 scripts/e2e_common 下沉）。

无外部依赖的合成 OHLCV 面板，供测试/演示/离线冒烟。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_mock_data(n_days: int = 500, n_codes: int = 50, seed: int = 0) -> dict:
    """生成 mock OHLCV 面板 {open/high/low/close/volume/amount: date×code}。"""
    rng = np.random.RandomState(seed)
    codes = [f"{i:06d}.SZ" for i in range(n_codes)]
    dates = pd.bdate_range(end="2026-08-21", periods=n_days)
    close = 100 * np.exp(np.cumsum(rng.randn(n_days, n_codes) * 0.02, axis=0))
    px = {}
    px["close"] = pd.DataFrame(close, index=dates, columns=codes)
    px["open"] = px["close"] * (1 + rng.randn(n_days, n_codes) * 0.005)
    px["high"] = pd.DataFrame(
        np.maximum(px["open"].values, px["close"].values) * (1 + rng.rand(n_days, n_codes) * 0.01),
        index=dates, columns=codes)
    px["low"] = pd.DataFrame(
        np.minimum(px["open"].values, px["close"].values) * (1 - rng.rand(n_days, n_codes) * 0.01),
        index=dates, columns=codes)
    px["volume"] = pd.DataFrame(rng.randint(1e6, 1e8, (n_days, n_codes)),
                                index=dates, columns=codes)
    px["amount"] = px["volume"] * px["close"]
    return px


def gen_mock_panel_with_signal(n_days: int = 400, n_codes: int = 50, seed: int = 0) -> dict:
    """AR(1) 收益注入动量信号：rets[t] = phi*rets[t-1] + noise，使 ts_mean/momentum 类因子有正 IC。

    历史定义于 scripts/mine_factors.py，2026-08-31 随 mock 数据下沉归位 data/mock。
    含合成"财务字段" OPERA_REV（PIT 化：每 60 日更新一次，中间 ffill）。
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]

    phi = 0.25
    rets = np.zeros((n_days, n_codes))
    for t in range(1, n_days):
        rets[t] = phi * rets[t - 1] + rng.normal(0, 0.02, n_codes)

    base = 10.0 + rng.uniform(0, 50, n_codes)
    close = pd.DataFrame(base * np.exp(np.cumsum(rets, axis=0)), idx, codes)
    open_ = close * (1 + rng.normal(0, 0.005, (n_days, n_codes)))
    high = np.maximum(close.values, open_.values) * (1 + np.abs(rng.normal(0, 0.005, (n_days, n_codes))))
    low = np.minimum(close.values, open_.values) * (1 - np.abs(rng.normal(0, 0.005, (n_days, n_codes))))
    volume = pd.DataFrame(rng.lognormal(12, 0.5, (n_days, n_codes)), idx, codes)
    amount = volume * close

    # 合成"财务字段"：慢漂移的 OPERA_REV，PIT 化（每 60 日更新一次，中间 ffill）
    rev_raw = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, n_codes)), axis=0)), idx, codes)
    rev = rev_raw.copy()
    rev.loc[np.arange(n_days) % 60 != 0] = np.nan
    rev = rev.ffill()

    return {
        "close": close, "open": pd.DataFrame(open_, idx, codes),
        "high": pd.DataFrame(high, idx, codes), "low": pd.DataFrame(low, idx, codes),
        "volume": volume, "amount": amount, "OPERA_REV": rev,
    }
