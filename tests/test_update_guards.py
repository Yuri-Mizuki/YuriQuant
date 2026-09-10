"""数据更新链路的守卫单测：盘中半拉日守卫 + 状态表分批拉取。"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from scripts.cli_common import complete_day_target


def _cal(*days: str) -> list[int]:
    return [int(d.replace("-", "")) for d in days]


def test_complete_day_target_intraday_rolls_back():
    # target=今天、盘中（<17:00）→ 回退到上一交易日
    now = dt.datetime(2026, 9, 8, 10, 30)
    cal = _cal("2026-09-04", "2026-09-07", "2026-09-08")
    target, rolled = complete_day_target(cal, 20260908, now=now)
    assert rolled is True
    assert target == 20260907


def test_complete_day_target_after_cutoff_keeps_today():
    # 已过 17:00 → 当日数据视为完整，不回退
    now = dt.datetime(2026, 9, 8, 18, 30)
    cal = _cal("2026-09-04", "2026-09-07", "2026-09-08")
    target, rolled = complete_day_target(cal, 20260908, now=now)
    assert rolled is False
    assert target == 20260908


def test_complete_day_target_non_today_noop():
    # target 不是今天（周末跑、或日历未推进）→ 不干预
    now = dt.datetime(2026, 9, 8, 10, 30)
    cal = _cal("2026-09-04", "2026-09-07")
    target, rolled = complete_day_target(cal, 20260907, now=now)
    assert rolled is False
    assert target == 20260907


def test_complete_day_target_allow_intraday():
    now = dt.datetime(2026, 9, 8, 10, 30)
    cal = _cal("2026-09-07", "2026-09-08")
    target, rolled = complete_day_target(cal, 20260908, allow_intraday=True, now=now)
    assert rolled is False
    assert target == 20260908


def test_complete_day_target_target_none():
    now = dt.datetime(2026, 9, 8, 10, 30)
    cal = _cal("2026-09-07", "2026-09-08")
    target, rolled = complete_day_target(cal, None, now=now)
    assert rolled is True
    assert target == 20260907


class _FlakyStatusDS:
    """模拟 SDK 状态表：整清单单查抛错，分批（<=200）才成功。"""

    def __init__(self):
        self.max_batch_seen = 0
        self.calls = 0

    def get_history_stock_status(self, code_list, begin_date, end_date):
        self.calls += 1
        codes = list(code_list)
        self.max_batch_seen = max(self.max_batch_seen, len(codes))
        if len(codes) > 200:
            raise RuntimeError("SDK hang simulation")
        idx = pd.MultiIndex.from_product(
            [pd.to_datetime([str(begin_date)]), codes], names=["date", "code"])
        return pd.DataFrame({"is_st": [False] * len(codes)}, index=idx)


def test_status_fetch_batched(tmp_path):
    from data.cache import DataCache

    ds = _FlakyStatusDS()
    cache = DataCache(ds, cache_root=str(tmp_path))
    codes = [f"c{i:04d}" for i in range(550)]     # > 200 → 走分批
    df = cache.get_history_stock_status(codes, 20260104, 20260104)
    assert len(df) == 550
    assert ds.max_batch_seen <= 200
    # 落盘 + 水位
    assert (tmp_path / "history_stock_status.parquet").exists()
    assert cache._get_last_date("history_stock_status") == 20260104


def test_status_fetch_small_list_direct(tmp_path):
    from data.cache import DataCache

    ds = _FlakyStatusDS()
    cache = DataCache(ds, cache_root=str(tmp_path))
    codes = [f"c{i}" for i in range(50)]          # <= 200 → 直连（不分批）
    df = cache.get_history_stock_status(codes, 20260104, 20260104)
    assert len(df) == 50
    assert ds.max_batch_seen == 50
