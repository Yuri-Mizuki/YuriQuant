"""告警去抖 / 持续期确认（P3，2026-08-20）。

每次监控 run 是独立快照；若单期噪音就告警，会对整条流水线脱敏
（真实里一次就有 19 个 GP 因子同触发，其中可能含单期噪音）。

confirm_rows 用台账历史判断"该 (name, rule) 是否已连续触发 confirm_n 期"：
- 连续期数 >= confirm_n 才算确认告警，写入台账；
- 不足则归"观察中"(pending)，不落台账（避免单次噪音污染告警历史）。
恢复（本次未触发）天然不进入 current，无需额外处理。

"连续"定义为按 run_date 升序的最近 confirm_n 个监控 run 里该告警都出现。
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def confirm_rows(
    current_rows: list[dict[str, Any]],
    prev_alerts: pd.DataFrame,
    confirm_n: int = 3,
) -> tuple[list[dict[str, Any]], int]:
    """过滤出已连续触发 confirm_n 期的告警。

    Returns:
        (final_rows, pending_count): final_rows 为确认告警（落台账）；
        pending_count 为本次新触发但尚未形成连续证据的告警条数。
    """
    if confirm_n <= 1:
        return current_rows, 0
    prev = prev_alerts if prev_alerts is not None and not prev_alerts.empty else None

    runs = set()
    if prev is not None:
        runs |= set(prev["run_date"].astype(str))
    runs |= {str(r["run_date"]) for r in current_rows}
    if not runs:
        return current_rows, 0
    runs = sorted(runs)

    final: list[dict[str, Any]] = []
    pending = 0
    for r in current_rows:
        if _consecutive((r["name"], r["rule"]), prev, runs, confirm_n):
            final.append(r)
        else:
            pending += 1
    return final, pending


def _consecutive(
    key: tuple[str, str],
    prev: pd.DataFrame | None,
    runs: list[str],
    confirm_n: int,
) -> bool:
    hit = {run: False for run in runs}
    if prev is not None:
        sel = prev[(prev["name"] == key[0]) & (prev["rule"] == key[1])]
        if not sel.empty:
            for rd in sel["run_date"].astype(str):
                if rd in hit:
                    hit[rd] = True
    if runs:
        hit[runs[-1]] = True  # 当前 run 触发该告警

    cnt = 0
    for rd in reversed(runs):
        if hit[rd]:
            cnt += 1
        else:
            break
    return cnt >= confirm_n