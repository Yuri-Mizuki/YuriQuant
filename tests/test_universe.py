"""Universe.get_all_a 的 PIT 语义测试（2026-10-23 立）。

锁定 10-23 回归修复：缓存数据止于 T 而查询日 > T（晚间 update 的常态——
target=今天、缓存止于上一交易日）时，**不得返回空池**。空池会让
update_data 误判「本地无缓存」→ 4 年全量拉 → 超预算死循环（10-20~22
三晚出榜故障根因）。
"""
from __future__ import annotations

import pandas as pd
import pytest


class _StubCache:
    def __init__(self, df: pd.DataFrame | None):
        self._df = df

    def read_daily(self, pool: str):
        return self._df


def _daily():
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-01-05"), "A"), (pd.Timestamp("2026-01-06"), "A"),
         (pd.Timestamp("2026-01-05"), "B"), (pd.Timestamp("2026-01-06"), "B"),
         (pd.Timestamp("2026-01-05"), "D")],   # D：只有 01-05 一天（退市近似）
        names=["date", "code"])
    return pd.DataFrame({"close": [1.0] * 5}, index=idx)


def test_pit_filters_delisted():
    from data.universe import Universe
    uni = Universe(_StubCache(_daily()))
    assert set(uni.get_all_a(20260106)) == {"A", "B"}   # D 已无 01-06 行情
    assert set(uni.get_all_a(20260105)) == {"A", "B", "D"}


def test_query_after_cache_end_not_empty():
    """查询日晚于缓存末日（数据滞后一天）⇒ 钳制右界，不返回空池。"""
    from data.universe import Universe
    uni = Universe(_StubCache(_daily()))
    assert set(uni.get_all_a(20260109)) == {"A", "B"}   # 修复前这里是 set()


def test_missing_cache_returns_empty_with_warning():
    from data.universe import Universe
    uni = Universe(_StubCache(None))
    with pytest.warns(UserWarning, match="daily_all_a"):
        assert uni.get_all_a(20260106) == []
