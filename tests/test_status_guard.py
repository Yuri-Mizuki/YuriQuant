"""状态表断更守卫的回归测试（2026-11-10 事故后新增）。

事故背景：``history_stock_status.parquet`` 止于 2026-10-22，10-23 起
可交易掩码的 ST/停牌剔除率从 3.4~4.8% 直接掉到 **0**，全链路无任何告警；
定版出榜连续多日混进 23~24 只 ST，11-02 的 solo Top10 里 5 只是 ST。

本文件锁死守卫的**判定语义**，重点是两件容易写错的事：

1. **按交易日算差，不按自然日**——否则每个周一都会被报成断更；
2. **日历是 int YYYYMMDD**——``pd.to_datetime(int)`` 会落到 1970（项目铁律），
   一旦踩中，所有日期差都会算成天文数字。
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.status_guard import (
    DEFAULT_ERROR_DAYS,
    StatusFreshness,
    _normalize_date,
    check_status_freshness,
)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _write_status(root, last_date: str, n_codes: int = 3) -> None:
    """写一个最小状态表（只有 is_st 一列，索引 (date, code)）。"""
    dates = pd.DatetimeIndex([pd.Timestamp(last_date)])
    codes = [f"{i:06d}.SZ" for i in range(n_codes)]
    idx = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])
    df = pd.DataFrame({"is_st": False, "is_suspended": False}, index=idx)
    df.to_parquet(root / "history_stock_status.parquet")


def _write_calendar(root, days: list[str]) -> None:
    """写 int YYYYMMDD 日历——与生产 calendar.parquet 的真实 dtype 一致。"""
    cal = pd.DataFrame({"date": [int(d.replace("-", "")) for d in days]})
    cal.to_parquet(root / "calendar.parquet")


# ---------------------------------------------------------------------------
# 日期归一化
# ---------------------------------------------------------------------------
def test_normalize_date_does_not_fall_into_1970():
    """int 日期必须拼字符串解析，不能当纳秒时间戳（项目 PIT 铁律）。"""
    for raw in (20261110, "20261110", "2026-11-10", pd.Timestamp("2026-11-10")):
        got = _normalize_date(raw)
        assert got == pd.Timestamp("2026-11-10"), f"{raw!r} -> {got}"


# ---------------------------------------------------------------------------
# 主判定
# ---------------------------------------------------------------------------
def test_freshness_counts_trading_days_not_calendar_days(tmp_path):
    """10-22 → 10-26 跨一个周末，只缺 1 个交易日（10-23），不是 4 个自然日。"""
    _write_status(tmp_path, "2026-10-22")
    _write_calendar(tmp_path, [
        "2026-10-21", "2026-10-22", "2026-10-23",
        "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30",
    ])
    res = check_status_freshness("2026-10-26", cache_root=tmp_path)
    assert res.missing_days == 1
    assert res.severity == "ok", "缺 1 天在 warn_days=1 阈值内，不应告警"
    assert res.last_date == pd.Timestamp("2026-10-22")


def test_freshness_escalates_to_error_after_a_week(tmp_path):
    """缺超过 5 个交易日 → error（事故现场：10-22 → 11-02 缺 6 天）。"""
    _write_status(tmp_path, "2026-10-22")
    _write_calendar(tmp_path, [
        "2026-10-22", "2026-10-23", "2026-10-26", "2026-10-27", "2026-10-28",
        "2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03",
    ])
    res = check_status_freshness("2026-11-02", cache_root=tmp_path)
    assert res.missing_days == 6
    assert res.severity == "error"
    assert res.is_stale
    assert not res, "is_stale 时 bool() 必须为 False"


def test_freshness_warn_band_between_thresholds(tmp_path):
    """缺 2~5 天 → warn：可用但需补拉。"""
    _write_status(tmp_path, "2026-10-22")
    _write_calendar(tmp_path, [
        "2026-10-22", "2026-10-23", "2026-10-26", "2026-10-27", "2026-10-28",
    ])
    res = check_status_freshness("2026-10-27", cache_root=tmp_path)
    assert res.missing_days == 2
    assert res.severity == "warn"


def test_freshness_ok_when_asof_is_last_day(tmp_path):
    _write_status(tmp_path, "2026-11-10")
    _write_calendar(tmp_path, ["2026-11-09", "2026-11-10"])
    res = check_status_freshness("2026-11-10", cache_root=tmp_path)
    assert res.missing_days == 0
    assert res.severity == "ok"


# ---------------------------------------------------------------------------
# 失败路径：守卫自身出问题不得阻断主流程
# ---------------------------------------------------------------------------
def test_missing_table_returns_unknown_not_exception(tmp_path):
    res = check_status_freshness("2026-11-02", cache_root=tmp_path)
    assert res.severity == "unknown"
    assert res.last_date is None
    assert not res.is_stale, "unknown 不等于断更，不应触发 strict 中止"


def test_corrupt_table_returns_unknown(tmp_path):
    (tmp_path / "history_stock_status.parquet").write_bytes(b"not a parquet")
    res = check_status_freshness("2026-11-02", cache_root=tmp_path)
    assert res.severity == "unknown"


def test_business_day_fallback_when_calendar_missing(tmp_path):
    """日历缺失时退化为工作日差：周末不误报，但仍能识别长断更。"""
    _write_status(tmp_path, "2026-10-22")
    res = check_status_freshness("2026-11-02", cache_root=tmp_path)
    # 10-23(五) + 10-26~30(5) + 11-02 不含端点 → 工作日 6 天
    assert res.missing_days == 6
    assert res.severity == "error"


def test_asof_none_only_checks_non_empty(tmp_path):
    """asof=None 只确认「表存在且非空」，不判新鲜度。"""
    _write_status(tmp_path, "2020-01-02")
    res = check_status_freshness(None, cache_root=tmp_path)
    assert res.severity == "ok"
    assert res.last_date == pd.Timestamp("2020-01-02")


def test_raise_on_error_is_opt_in(tmp_path):
    """默认不抛；显式 raise_on_error=True 才抛——守卫默认只诊断。"""
    _write_status(tmp_path, "2026-10-22")
    _write_calendar(tmp_path, ["2026-10-22"] + [
        f"2026-11-{d:02d}" for d in range(2, 12)
    ])
    res = check_status_freshness("2026-11-10", cache_root=tmp_path)
    assert res.is_stale, "前提：该日期应判为断更"
    with pytest.raises(RuntimeError, match="状态表断更"):
        check_status_freshness("2026-11-10", cache_root=tmp_path,
                              raise_on_error=True)


def test_dataclass_bool_semantics():
    """``if freshness:`` 只在 ok/unknown 为真，避免调用方写反。"""
    assert bool(StatusFreshness(severity="ok"))
    assert bool(StatusFreshness(severity="unknown"))
    assert not bool(StatusFreshness(severity="warn"))
    assert not bool(StatusFreshness(severity="error"))


def test_thresholds_are_ordered():
    assert 0 <= DEFAULT_ERROR_DAYS, "阈值必须非负"
    assert DEFAULT_ERROR_DAYS >= 1


# ---------------------------------------------------------------------------
# 覆盖率：防「部分补拉」造成的假 ok（2026-11-10 恢复补拉时暴露）
# ---------------------------------------------------------------------------
def _write_status_partial(root, last_date: str, full_codes: int,
                          last_codes: int) -> None:
    """写一张「历次单日 100 只、最后一天只有 10 只」的表。

    这正是分批补拉被打断后的形态：日期推进到了最后一天，但绝大多数
    标的仍停留在旧日期。
    """
    import pandas as pd_  # noqa: F401

    hist_idx = pd.MultiIndex.from_product(
        [pd.DatetimeIndex([pd.Timestamp("2026-10-22")]),
         [f"{i:06d}.SZ" for i in range(full_codes)]],
        names=["date", "code"])
    last_idx = pd.MultiIndex.from_product(
        [pd.DatetimeIndex([pd.Timestamp(last_date)]),
         [f"{i:06d}.SZ" for i in range(last_codes)]],
        names=["date", "code"])
    df = pd.DataFrame(
        {"is_st": False, "is_suspended": False},
        index=hist_idx.append(last_idx))
    df.to_parquet(root / "history_stock_status.parquet")


def test_partial_backfill_with_new_last_date_is_not_ok(tmp_path):
    """最后一天日期很新，但只覆盖 10% 标的 ⇒ 必须 warn，不能报 ok。

    只判日期会给出假健康：600/5559 只更新到最新日，其余仍是旧状态，
    下游按 code 取最新行时，89% 的标的判据其实是陈旧的。
    """
    _write_status_partial(tmp_path, "2026-11-10", full_codes=100, last_codes=10)
    _write_calendar(tmp_path, ["2026-10-22", "2026-11-10"])
    res = check_status_freshness("2026-11-10", cache_root=tmp_path)
    assert res.severity == "warn", res.message
    assert "覆盖" in res.message
    assert res.detail["coverage_last_day"] == 10
    assert res.detail["coverage_peak"] == 100
    assert res.detail["coverage_ratio"] == pytest.approx(0.1)


def test_full_coverage_on_last_day_stays_ok(tmp_path):
    """覆盖率完整时不应被新判据误伤（防过度告警）。"""
    _write_status_partial(tmp_path, "2026-11-10", full_codes=100, last_codes=100)
    _write_calendar(tmp_path, ["2026-10-22", "2026-11-10"])
    res = check_status_freshness("2026-11-10", cache_root=tmp_path)
    assert res.severity == "ok", res.message
    assert res.detail["coverage_ratio"] == pytest.approx(1.0)


def test_coverage_check_does_not_mask_real_staleness(tmp_path):
    """覆盖率低但日期差距已达 error 时，严重度仍是 error（不能被降级成 warn）。"""
    _write_status_partial(tmp_path, "2026-10-23", full_codes=100, last_codes=100)
    _write_calendar(tmp_path, ["2026-10-23"] + [
        f"2026-11-{d:02d}" for d in range(2, 12)
    ])
    res = check_status_freshness("2026-11-10", cache_root=tmp_path)
    assert res.severity == "error", res.message


def test_coverage_ratio_is_configurable(tmp_path):
    """阈值可调：放宽到 5% 后，10% 覆盖率不再告警。"""
    _write_status_partial(tmp_path, "2026-11-10", full_codes=100, last_codes=10)
    _write_calendar(tmp_path, ["2026-10-22", "2026-11-10"])
    res = check_status_freshness("2026-11-10", cache_root=tmp_path,
                                 coverage_min_ratio=0.05)
    assert res.severity == "ok", res.message