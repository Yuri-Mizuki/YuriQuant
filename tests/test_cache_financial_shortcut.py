"""``DataCache._get_financial`` 的水位短路（2026-11-20）。

财报接口没有增量参数（SDK 的 ``get_income(code_list)`` 收完代码就返回整段全历史），
没有短路时每次 ``update_data`` 都会把 5,830 只重拉一遍（约 36GB 流量）。这里锁住
三条行为：① 本地覆盖且未过期 → 不回源；② ``max_age_days=0`` → 强制回源；
③ 本地缺代码超过 5% 容差 → 回源。
"""
from __future__ import annotations

import pandas as pd

from data.cache import DataCache


class _FakeDataSource:
    """只实现 get_income，记录回源次数。"""

    def __init__(self) -> None:
        self.calls = 0

    def get_income(self, code_list, begin_date=None, end_date=None):
        self.calls += 1
        return pd.DataFrame({
            "code": list(code_list),
            "ann_date": pd.Timestamp("2025-10-31"),
            "report_period": pd.Timestamp("2025-09-30"),
            "revenue": 1.0,
        })


def _cache(tmp_path) -> tuple[DataCache, _FakeDataSource]:
    ds = _FakeDataSource()
    return DataCache(ds, cache_root=tmp_path), ds


def test_short_circuit_when_fresh_and_covered(tmp_path):
    cache, ds = _cache(tmp_path)
    codes = ["000001.SZ", "600000.SH"]

    first = cache.get_income(codes, max_age_days=7)
    assert ds.calls == 1 and len(first) == 2

    again = cache.get_income(["000001.SZ"], max_age_days=7)
    assert ds.calls == 1, "未过期且已覆盖，不应再回源"
    assert list(again["code"]) == ["000001.SZ"]
    assert cache._get_refreshed_on("income") != ""


def test_max_age_zero_forces_refetch(tmp_path):
    cache, ds = _cache(tmp_path)
    cache.get_income(["000001.SZ"], max_age_days=7)
    cache.get_income(["000001.SZ"], max_age_days=0)
    assert ds.calls == 2


def test_missing_codes_beyond_tolerance_refetch(tmp_path):
    cache, ds = _cache(tmp_path)
    cache.get_income(["000001.SZ"], max_age_days=7)
    # 请求 10 只、本地只有 1 只 → 缺 9 只远超 5% 容差，必须回源
    cache.get_income([f"00000{i}.SZ" for i in range(1, 11)], max_age_days=7)
    assert ds.calls == 2
