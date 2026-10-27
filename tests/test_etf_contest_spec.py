"""定版参数一致性（2026-10-27 锁定）
=====================================

``etf_contest_spec.FINAL`` 是回测与实盘共用的唯一真源。任何一边改了参数而另一边
没跟上，这里会失败——这正是"回测 +4.44%、下单却是另一套权重"的防呆。
"""
from __future__ import annotations

from scripts.portfolio.etf_champ_signals import PROFILES
from scripts.portfolio.etf_contest_probe import CONC_SPECS, SPECS
from scripts.portfolio.etf_contest_spec import EXCLUDED_5, FINAL

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
