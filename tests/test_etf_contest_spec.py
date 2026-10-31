"""定版参数一致性（2026-10-27 锁定）
=====================================

``etf_contest_spec.FINAL`` 是信号板与历史回测共用的唯一真源。任何一边改了参数而
另一边没跟上，这里会失败——这正是"回测 +4.44%、下单却是另一套权重"的防呆。

赛后变体探针（``etf_contest_probe`` 等）已于 2026-10-28 回收，本文件只校验
``FINAL`` ↔ 信号板 profile，以及仍在用的出场叠加调度（``etf_rotation_probe``）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.etf_universe import ETF_CANDIDATES
from scripts.portfolio.etf_champ_signals import PROFILES
from scripts.portfolio.etf_contest_spec import EXCLUDED_5, FINAL, POOL, POOL_SIZE
from scripts.portfolio.etf_rotation_probe import build_exit_schedule


def _strip_name(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "name"}


def test_signal_board_profile_is_final():
    assert "contest3" in PROFILES
    assert _strip_name(PROFILES["contest3"]) == FINAL


def test_excluded_five_are_distinct():
    assert len(EXCLUDED_5) == 5
    assert len(set(EXCLUDED_5)) == 5


def test_pool_size_is_45():
    """池规模唯一真源（2026-10-30 收口）：45 = 50 候选 − 5 屏蔽。

    锁死两个数，防止文档再次漂移成「46 只」（历史成因：拿行情表列数 51 直接减 5，
    而行情表多一个已剔出候选池的孤儿代码 512580 环保 ETF）。
    """
    assert len(ETF_CANDIDATES) == 50, "候选池应为 50 只"
    assert POOL_SIZE == 45, "可投池应为 45 只"
    assert len(POOL) == len(set(POOL)), "POOL 不得有重复"
    assert not set(EXCLUDED_5) & set(POOL), "屏蔽的 5 只不得进 POOL"
    assert set(POOL) | set(EXCLUDED_5) == set(ETF_CANDIDATES)
    # 孤儿代码不得出现在候选池里
    assert "512580.SH" not in ETF_CANDIDATES


def _flat_frame(values: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.DataFrame({"A": values}, index=idx)


def test_exit_schedule_trail_keeps_real_entry():
    """trail 触发在持仓期（非调仓日），且跨周重选不重置建仓日。"""
    px = [100.0] * 10 + list(np.linspace(100.0, 110.0, 5)) + [105.0] * 6 + [97.0] * 10
    close = _flat_frame(px)
    days = list(close.index)
    weekly = [days[0], days[10], days[20], days[30]]
    sel = {d: ["A"] for d in weekly}

    plan = build_exit_schedule(close, weekly, sel, {"rule": "trail", "dd": 0.10})

    assert list(plan["exits"]) == [days[21]]           # 非调仓日触发
    assert plan["exits"][days[21]] == {"A"}
    ev = plan["events"].iloc[0]
    assert ev["entry"] == days[0]                      # 真实建仓日，非当周调仓日
    assert ev["peak"] == 110.0


def test_exit_schedule_anchor_uses_entry_minus_21_not_rolling():
    """B1 锚定价必须锚在真实建仓日 t−21，不得随周频重选前滚。"""
    # 第 14 天 110 / 真实建仓日（第 25 天）的 t−21（第 4 天）= 100
    px = [105.0] * 50
    px[4], px[14], px[25] = 100.0, 110.0, 100.0
    close = _flat_frame(px)
    days = list(close.index)
    weekly = [days[25], days[35], days[45]]
    sel = {d: ["A"] for d in weekly}

    plan = build_exit_schedule(close, weekly, sel, {"rule": "anchor", "mult": 1.00})

    # 105 > 锚定价 100 ⇒ 不触发；若误用滚动 t−21（第 14 天 110）会全部触发。
    assert plan["events"].empty
    assert plan["exits"] == {}