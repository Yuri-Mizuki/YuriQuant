"""定版参数一致性（2026-10-27 锁定）
=====================================

``etf_contest_spec.FINAL`` 是回测与实盘共用的唯一真源。任何一边改了参数而另一边
没跟上，这里会失败——这正是"回测 +4.44%、下单却是另一套权重"的防呆。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.portfolio.etf_champ_signals import PROFILES
from scripts.portfolio.etf_contest_probe import (
    CONC_SPECS,
    ENTRYLOW_SPECS,
    EXIT_SPECS,
    EXPLORE2_SPECS,
    EXPLORE_SPECS,
    SPECS,
    VOLPRE_SPECS,
)
from scripts.portfolio.etf_contest_spec import EXCLUDED_5, FINAL
from scripts.portfolio.etf_rotation_probe import build_exit_schedule

FINAL_NAME = "c3_final"


def _strip_name(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "name"}


def test_signal_board_profile_is_final():
    assert "contest3" in PROFILES
    assert _strip_name(PROFILES["contest3"]) == FINAL


def test_probe_final_variant_is_final():
    hits = [s for s in CONC_SPECS if s["name"] == FINAL_NAME]
    assert len(hits) == 1
    for key, value in FINAL.items():
        assert hits[0][key] == value, f"{key} 与 FINAL 不一致"


def test_spec_names_unique():
    names = [s["name"] for s in SPECS] + [s["name"] for s in CONC_SPECS]
    assert len(names) == len(set(names))


def test_excluded_five_are_distinct():
    assert len(EXCLUDED_5) == 5
    assert len(set(EXCLUDED_5)) == 5


def test_blindspot_variants_budget_and_derivation():
    """盲区/出场加速组：13 变体封顶（预注册预算），全部从 FINAL 派生。"""
    assert len(ENTRYLOW_SPECS) == 3
    assert len(EXIT_SPECS) == 10
    assert len(ENTRYLOW_SPECS) + len(EXIT_SPECS) == 13
    names = [s["name"] for s in ENTRYLOW_SPECS + EXIT_SPECS]
    assert len(names) == len(set(names))
    for spec in ENTRYLOW_SPECS + EXIT_SPECS:
        # A 组只改 signal（镜像低阈面板），其余参数必须与 FINAL 一致。
        keys = [k for k in FINAL if k != "signal"] if spec in ENTRYLOW_SPECS else FINAL
        for key in keys:
            assert spec[key] == FINAL[key], f"{spec['name']} 的 {key} 偏离 FINAL"
    for spec in ENTRYLOW_SPECS:
        assert spec["signal"].startswith(FINAL["signal"] + "_low")


def test_explore_group_budget_is_separate():
    """探索组是另立预算（4 条），不得并入预注册的 13 变体内。"""
    assert len(EXPLORE_SPECS) == 4
    names = [s["name"] for s in EXPLORE_SPECS]
    assert len(names) == len(set(names))
    assert not set(names) & {s["name"] for s in ENTRYLOW_SPECS + EXIT_SPECS}
    for spec in EXPLORE_SPECS:
        if spec["signal"] != FINAL["signal"]:
            assert spec["signal"].startswith(FINAL["signal"])


def test_explore2_group_budget_is_separate():
    """第二轮探索（参数邻域，左尾优先）5 条，同属另立预算且不重名。

    exposure 取值必须落在主流程已实现的 regime 模式内——拼错会让整条变体
    静默退化成满仓（与 c3_final 逐日相同），却仍然出现在报告里。
    """
    assert len(EXPLORE2_SPECS) == 5
    names = [s["name"] for s in EXPLORE2_SPECS]
    assert len(names) == len(set(names))
    others = {s["name"] for s in SPECS + CONC_SPECS + ENTRYLOW_SPECS
              + EXIT_SPECS + EXPLORE_SPECS}
    assert not set(names) & others
    for spec in EXPLORE2_SPECS:
        if spec["signal"] != FINAL["signal"]:
            assert spec["signal"].startswith(FINAL["signal"])
        if "exposure" in spec:
            assert spec["exposure"] in {"half", "half07", "vt20"}


def test_volpre_group_is_monotone_ladder():
    """波动率缩放阈词组：3 档 k 单调（50/75/100），且不与其它组重名。

    阈值 = max(10%, k × 20 日年化波动)，k 越大越宽松；档位顺序被打乱就等于
    失去"单调性"这条判据，故在这里锁死顺序。
    """
    assert [s["signal"] for s in VOLPRE_SPECS] == [
        "mom12_1_crowdsoft_volk50",
        "mom12_1_crowdsoft_volk75",
        "mom12_1_crowdsoft_volk100",
    ]
    names = [s["name"] for s in VOLPRE_SPECS]
    assert len(names) == len(set(names))
    others = {s["name"] for s in SPECS + CONC_SPECS + ENTRYLOW_SPECS
              + EXIT_SPECS + EXPLORE_SPECS + EXPLORE2_SPECS}
    assert not set(names) & others


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
