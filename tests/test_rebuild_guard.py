"""月度重建守卫的目标日/自锁回归（2026-10-19 审计 P0-2）。

背景：``_prev_month_end_trading_day`` 原取 ``_base/close_adj`` 的日历，
``_base`` 一旦滞后一个整月，target 会等于 ``panels_neu`` 覆盖日 ⇒ 守卫永久
判「已覆盖」。本文件锁死两条：
1. target 来自**独立日历**（``calendar.parquet`` / ``daily_all_a``）；
2. ``_base`` 落后于 target 时**必须**判需要重建（即使 ``panels_neu`` 数字看着够）。
"""
from __future__ import annotations

import pandas as pd

from scripts.pipelines import rebuild_panel_library as R


def _cov(base: str, neu: str) -> dict[str, str]:
    return {"base(close_adj)": base, "panels_neu": neu}


def test_target_uses_independent_calendar():
    """独立日历含 10-30 时，target 必须是 10-30（不是 _base 的 10-15）。"""
    cal = pd.bdate_range("2026-09-01", "2026-10-30")
    assert R._prev_month_end_trading_day(
        today="2026-11-02", cal=cal) == "2026-10-30"


def test_stale_base_triggers_rebuild_despite_matching_neu():
    """_base 停 10-15、panels_neu 也停 10-15：target=10-30 ⇒ 必须重建。"""
    need, reasons = R.needs_rebuild(_cov("2026-10-15", "2026-10-15"), "2026-10-30")
    assert need
    assert any("_base" in r for r in reasons), reasons


def test_stale_base_triggers_rebuild_even_if_neu_looks_fresh():
    """panels_neu 看着新（10-30）但 _base 旧（10-15）也必须重建（版本耦合）。"""
    need, reasons = R.needs_rebuild(_cov("2026-10-15", "2026-10-30"), "2026-10-30")
    assert need
    assert any("_base" in r for r in reasons), reasons


def test_clean_state_does_not_rebuild():
    need, reasons = R.needs_rebuild(_cov("2026-10-30", "2026-10-30"), "2026-10-30")
    assert not need
    assert reasons == []


def test_missing_target_is_conservative():
    need, reasons = R.needs_rebuild(_cov("2026-10-30", "2026-10-30"), None)
    assert need
    assert reasons


def test_dates_from_int_date_column(tmp_path):
    """``calendar.parquet`` 是 date int64 列（无索引），解析不能被忽略。"""
    p = tmp_path / "calendar.parquet"
    pd.DataFrame({"date": [20260930, 20261015, 20261016]}).to_parquet(p)
    d = R._dates_from_parquet(p)
    assert list(d) == [pd.Timestamp("2026-09-30"),
                       pd.Timestamp("2026-10-15"),
                       pd.Timestamp("2026-10-16")]


def test_unreadable_panels_needs_rebuild():
    need, reasons = R.needs_rebuild(_cov("2026-10-30", "全部读取失败"), "2026-10-30")
    assert need
    assert any("panels_neu" in r for r in reasons), reasons


# --------------------------------------------------------------------------
# 重建链序列（2026-11-13）
# --------------------------------------------------------------------------

def test_dup_index_is_last_step():
    """冗余索引重建必须在**最后**：面板全量重算后旧索引已过期。

    这是反复踩了四轮的坑（`dedup_execution.md` 记了三次 P2 未做）。若把它
    插在中间，后面的中性化会产出新面板而索引仍指向旧签名 —— 入库查重
    （`_dup_check_fast`）与去重决策表都在读那份过期近邻。
    """
    steps = R._steps(workers=1)
    names = [n for n, _ in steps]
    assert names[-1] == "dup_index", names[-3:]
    assert names.count("dup_index") == 1


def test_dup_index_step_does_not_apply_registry_backfill():
    """重建链里的 dup_audit **不得带 --apply**。

    `--apply` 会回填 registry 的 `dup_*` 统计列。混进「重建面板」这个动作里，
    一旦出问题就无法二分「是面板变了还是统计列变了」；回填应当是独立决策。
    """
    steps = dict(R._steps(workers=1))
    cmd = steps["dup_index"]
    assert "--apply" not in cmd, cmd
    assert "scripts.factors.dup_audit" in cmd


def test_dup_index_step_targets_production_dataset():
    """索引必须建在生产数据集上，否则 `_dup_check_fast` 读不到。"""
    steps = dict(R._steps(workers=1))
    cmd = steps["dup_index"]
    assert "--dataset" in cmd
    assert cmd[cmd.index("--dataset") + 1] == "all_a_2018_2026"
