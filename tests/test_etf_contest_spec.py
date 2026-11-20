"""定版参数一致性（2026-10-27 锁定 / 2026-11-19 去比赛化）
==========================================================

``etf_contest_spec.RETIRED_CORE`` 是退役档 core 的参数，也是历史回测与若干探针的
共用基准。任何一边改了参数而另一边没跟上，这里会失败——这正是"回测 +4.44%、
下单却是另一套权重"的防呆。

2026-11-19 起策略回到正常口径：池子 50 只全量，方案键改 ``core`` / ``balanced`` /
``monthly``（原 contest3 / contest5h / champ），原合规补位档 ``contest5`` 已删；
``EXCLUDED_5`` 仅用于复现比赛期 45 只历史数字。

同日核心档退役（相位均值 Calmar 0.36，被 monthly 支配）：``core`` 不再进
``PROFILES``，参数以 ``RETIRED_CORE`` 保留，本文件仍校验它与 ``FINAL`` 一致。

赛后变体探针（``etf_contest_probe`` 等）已于 2026-10-28 回收，本文件只校验
``FINAL`` ↔ 信号板 profile，以及仍在用的出场叠加调度（``etf_rotation_probe``）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.etf_universe import ETF_CANDIDATES
from scripts.portfolio.etf_champ_signals import PROFILES
from scripts.portfolio.etf_contest_spec import (
    EXCLUDED_5,
    FINAL,
    POOL,
    POOL_CONTEST,
    POOL_SIZE,
    RETIRED_CORE,
)
from scripts.portfolio.etf_rotation_probe import build_exit_schedule


def _strip_name(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "name"}


def test_signal_board_profile_is_final():
    """退役档 core 的参数必须与 ``FINAL`` 逐键一致（历史复现与探针基准共用）。"""
    assert _strip_name(RETIRED_CORE) == FINAL


def test_profile_keys_are_functional_names():
    """2026-11-19 退役 core 后，生产只剩 monthly（默认）与 balanced 两档。"""
    assert set(PROFILES) == {"balanced", "monthly"}
    assert PROFILES["balanced"]["name"] == "balanced"
    assert PROFILES["balanced"]["top_k"] == 5
    assert PROFILES["monthly"]["name"] == "monthly"
    assert PROFILES["monthly"]["freq"] == "M"
    # 2026-11-19：weighting 由 hrp 改 erc（§14.11）
    assert PROFILES["balanced"]["weighting"] == "erc"
    # 2026-11-19：tvol 由 0.15 降至 0.12（§14.9.1）
    assert PROFILES["monthly"]["target_vol"] == 0.12
    # 2026-11-19：逆波动加 σ 下限 10%，解决国债 ETF 占权益腿 80%+（§14.12）
    assert PROFILES["monthly"]["weight_vol_floor"] == 0.10
    assert "core" not in PROFILES


def test_combo_mix_not_adopted():
    """两档合并（60/40）已于 2026-11-19 判否：两腿调仓日 M/W 不同步，
    合并只等价于两个独立账户，§14.9.3 的 0.79 Calmar 是日频再平衡口径。"""
    import scripts.portfolio.etf_champ_signals as sig

    assert not hasattr(sig, "SLEEVE_MIX")
    assert "combo" not in sig.PROFILES


def test_excluded_five_are_distinct():
    assert len(EXCLUDED_5) == 5
    assert len(set(EXCLUDED_5)) == 5


def test_pool_size_is_50():
    """池规模唯一真源：正常口径 50 = 全部候选；比赛期口径 45 = 50 − 5。

    锁死两个数，防止文档再次漂移成「46 只」（历史成因：拿行情表列数 51 直接减 5，
    而行情表多一个已剔出候选池的孤儿代码 512580 环保 ETF）。
    """
    assert len(ETF_CANDIDATES) == 50, "候选池应为 50 只"
    assert POOL_SIZE == 50, "正常口径可投池应为 50 只"
    assert set(POOL) == set(ETF_CANDIDATES), "正常运行不再屏蔽任何标的"
    assert len(POOL) == len(set(POOL)), "POOL 不得有重复"
    assert len(POOL_CONTEST) == 45, "比赛期口径应为 45 只"
    assert not set(EXCLUDED_5) & set(POOL_CONTEST)
    assert set(POOL_CONTEST) | set(EXCLUDED_5) == set(ETF_CANDIDATES)
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