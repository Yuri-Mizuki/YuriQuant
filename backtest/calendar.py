"""调仓日历唯一真源。

此前同一件事有三套实现，且 2M 的锚点不一致：

- ``VectorBacktest._get_rebalance_days``：``to_period("M").asi8 // 2``，锚在
  **1970-01**（月序奇偶）。样本起于偶数月时会连出两个月 ⇒ 首期只有 1 个月、
  多一次换手。
- ``caliber.rebalance_dates``：同上，另加 ``phase``。
- ``rolling_grid_alla._rebalance_days_validated``：``month_firsts()[::2]``，
  锚在**样本首月**。

三者对「频率效应」给出的不是同一个实验。现在只有本模块一套：以样本内第一个
自然周/月为起点计数，``[::per_len]`` 取日。

``phase`` 语义按频率不同（沿用 2026-11-10 相位扫描的既有定义）：

- ``W`` / ``M`` / ``2W``：周期内第 ``phase`` 个交易日；该周期交易日不足时回退
  到周期最后一日（防节假日把整周丢掉）。
- ``2M`` / ``Q``：从样本首月起偏移 ``phase`` 个自然月再取日。
"""
from __future__ import annotations

import pandas as pd

#: 引擎与 caliber 共同支持的调仓频率。
REBALANCE_FREQS: tuple[str, ...] = ("D", "W", "2W", "M", "2M", "Q")


def month_firsts(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每月**第一个交易日**（= 引擎 ``groupby(to_period("M")).first()``）。"""
    s = pd.Series(pd.DatetimeIndex(idx), index=pd.DatetimeIndex(idx))
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).first().to_numpy())


def _nth_of_period(idx: pd.DatetimeIndex, freq: str, phase: int) -> pd.DatetimeIndex:
    """每个周期取第 ``phase`` 个交易日；不足则回退最后一日。"""
    s = pd.Series(idx, index=idx)
    per = idx.to_period(freq)
    if phase == 0:
        return pd.DatetimeIndex(s.groupby(per).first().to_numpy())
    pos = pd.Series(idx.to_series().groupby(per).cumcount().to_numpy(), index=idx)
    nth = s[pos == phase]
    have = set(nth.index.to_period(freq))
    last = pd.DatetimeIndex(s.groupby(per).last().to_numpy())
    tail = last[~last.to_period(freq).isin(have)]
    return pd.DatetimeIndex(sorted(set(nth.to_numpy()) | set(tail.to_numpy())))


def rebalance_days(idx: pd.DatetimeIndex, freq: str = "M",
                   phase: int = 0) -> pd.DatetimeIndex:
    """调仓**信号日**（T）列表，升序、去重。

    Args:
        idx: 全量交易日（升序）。
        freq: ``D`` / ``W`` / ``2W`` / ``M`` / ``2M`` / ``Q``。
        phase: 周期内相位（见模块文档）。

    Raises:
        KeyError: 未知频率。
        ValueError: ``phase`` 超出该频率的合法范围。
    """
    if freq not in REBALANCE_FREQS:
        raise KeyError(f"未知调仓频率 {freq!r}，可选 {list(REBALANCE_FREQS)}")
    idx = pd.DatetimeIndex(idx).sort_values()
    if freq == "D":
        return idx
    if phase < 0:
        raise ValueError(f"phase 须 >= 0，收到 {phase}")

    if freq in ("2M", "Q"):
        per_len = 2 if freq == "2M" else 3
        if phase >= per_len:
            raise ValueError(f"freq={freq!r} 的 phase 须在 [0,{per_len})，收到 {phase}")
        return month_firsts(idx)[phase::per_len]

    out = _nth_of_period(idx, freq, phase)
    return out[::2] if freq == "2W" else out
