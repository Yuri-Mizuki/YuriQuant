"""组合/信号层监控（P3，2026-08-20）。

把每日交易信号产物（optimize.signals 导出的长表）折叠成一条组合层监控快照，
补 IC 层之外维度：信号新鲜度、股票覆盖、持仓集中度（HHI）、净换手、受阻比例。

为何只基于"目标权重信号"而非"组合净值"：当前不接实盘、无成交账本，可运营
的输入就是每日目标权重信号。净值/回撤等组合绩效属于离线多期回测（离线的评价），
不在此监听线上快照范围内。

输入 signals 长表列约定（optimize.signals 导出）：
    signal_date, code, target_weight, qty_target, qty_order, direction, status, note
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import pandas as pd

from monitoring.metrics import MonitorMetrics

_EPS = 1e-8


def load_signals(path: str | Path) -> pd.DataFrame:
    """按路径或 glob 读取信号长表，返回 signal_date 升序去重的 DataFrames。"""
    hits = sorted(glob.glob(str(path))) if any(ch in str(path) for ch in "*?[") else [str(path)]
    frames = []
    for h in hits:
        if Path(h).exists():
            frames.append(pd.read_csv(h))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if not {"signal_date", "code", "target_weight"}.issubset(df.columns):
        return df
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    return df.sort_values("signal_date").drop_duplicates(["signal_date", "code"])


def compute_signal_monitor(
    signals: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    name: str = "signal:latest",
) -> MonitorMetrics:
    """折叠信号序列为最新一期的组合层监控快照（指标 + 原始字段）。

    Args:
        signals: load_signals 输出，须含 signal_date/code/target_weight/status。
        calendar: 交易日日历（与因子库同源，如 returns 面板 index），算新鲜度。
        as_of: 监控基准日。
    Returns:
        MonitorMetrics(category="signal")，充填信号层字段；告警由 attach_alerts 触发。
    """
    m = MonitorMetrics(name=name, category="signal")
    if signals.empty:
        ref = calendar[-1] if len(calendar) else as_of
        m.signal_freshness_days = _trading_days(calendar, ref, as_of)
        return m

    last_date = signals["signal_date"].max()
    m.signal_date = str(pd.Timestamp(last_date).date())
    cur = signals[signals["signal_date"] == last_date]

    w = cur["target_weight"].fillna(0.0)
    wpos = w[w > _EPS]
    m.n_signal_stocks = int((wpos > 0).sum())
    if (wpos > 0).sum():
        wnorm = wpos / wpos.sum()
        m.hhi_recent = float((wnorm ** 2).sum())
    else:
        m.hhi_recent = float("nan")

    if "status" in cur.columns:
        n_blocked = int(cur["status"].astype(str).str.startswith("BLOCKED").sum())
        m.blocked_ratio = n_blocked / len(cur) if len(cur) else float("nan")

    prev = signals[signals["signal_date"] < last_date]["signal_date"].max()
    if pd.notna(prev):
        wp = signals[signals["signal_date"] == prev].set_index("code")[
            "target_weight"
        ].fillna(0.0)
        wc = signals[signals["signal_date"] == last_date].set_index("code")[
            "target_weight"
        ].fillna(0.0)
        idx = wc.index.union(wp.index)
        m.net_turnover = float(
            0.5
            * (wc.reindex(idx, fill_value=0.0) - wp.reindex(idx, fill_value=0.0))
            .abs()
            .sum()
        )
    else:
        m.net_turnover = float("nan")

    m.signal_freshness_days = _trading_days(calendar, last_date, as_of)
    return m


def _trading_days(
    calendar: pd.DatetimeIndex, from_ts: pd.Timestamp, to_ts: pd.Timestamp
) -> int:
    """from_ts 到 to_ts 之间的交易日数（含 to、不含 from；from>=to → 0）。"""
    if calendar is None or len(calendar) == 0:
        return 0
    ref = min(pd.Timestamp(to_ts), calendar[-1])
    fr = pd.Timestamp(from_ts)
    if ref <= fr:
        return 0
    return int(((calendar > fr) & (calendar <= ref)).sum())


def build_signal_metrics(
    signals: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    cfg: dict[str, Any] | None = None,
) -> MonitorMetrics:
    """便捷入口：compute + 触发告警（attach_alerts 写回 alerts/status）。"""
    from monitoring.alerts import attach_alerts
    return attach_alerts(compute_signal_monitor(signals, calendar, as_of), cfg or {})