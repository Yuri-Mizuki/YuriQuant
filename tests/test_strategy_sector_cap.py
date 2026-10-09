"""SectorCappedTopKLongOnly 行业上限约束单测（2026-10-09 新增）。

覆盖：上限生效 / 行业不足时欠仓 / 与 TopK 一致（分散时）/ tie-break 口径 /
哨兵桶（行业缺失）/ 未知日期 / 空截面 / 接口与参数校验。
"""
import pandas as pd
import pytest

from strategy.examples import SectorCappedTopKLongOnly, TopKLongOnly

D = pd.Timestamp("2026-09-04")


def _sectors(mapping: dict) -> pd.DataFrame:
    """构造单日 date × code 行业面板。"""
    return pd.DataFrame([mapping], index=[D])


def test_cap_limits_per_sector_picks():
    """行业 A 有 4 只高分股，上限 2 ⇒ 只取 A 的前 2，再由 B 补足。"""
    cs = pd.Series({"a1": 10, "a2": 9, "b1": 8, "b2": 7, "a3": 6, "a4": 5})
    sec = _sectors({"a1": "A", "a2": "A", "b1": "B", "b2": "B",
                    "a3": "A", "a4": "A"})
    s = SectorCappedTopKLongOnly(sec, k=4, max_per_sector=2)
    w = s.get_weights_at(D, cs)
    assert set(w.index) == {"a1", "a2", "b1", "b2"}
    assert w.sum() == pytest.approx(1.0)


def test_cap_underfills_when_sectors_scarce():
    """行业数 × 上限 < k ⇒ 欠仓，不越界补足。"""
    cs = pd.Series({"a1": 10, "a2": 9, "a3": 8, "a4": 7})
    sec = _sectors({c: "A" for c in cs.index})
    s = SectorCappedTopKLongOnly(sec, k=4, max_per_sector=2)
    w = s.get_weights_at(D, cs)
    assert set(w.index) == {"a1", "a2"}


def test_matches_topk_when_cap_not_binding():
    """每只独立行业、上限 1 ⇒ 与 TopK 完全一致（同一 tie-break）。"""
    cs = pd.Series({c: 10 - i for i, c in enumerate("abcdefghij")})
    sec = _sectors({c: c for c in cs.index})
    s = SectorCappedTopKLongOnly(sec, k=3, max_per_sector=1)
    w = s.get_weights_at(D, cs)
    ref = TopKLongOnly(k=3).get_weights(cs)
    assert list(w.index) == list(ref.index)


def test_tie_break_follows_column_order():
    """并列时按列序（与全仓唯一 rank(method='first') 一致）。"""
    cs = pd.Series({"z": 5.0, "a": 5.0, "m": 5.0})
    sec = _sectors({c: c for c in cs.index})
    s = SectorCappedTopKLongOnly(sec, k=1, max_per_sector=1)
    assert list(s.get_weights_at(D, cs).index) == ["z"]


def test_nan_sector_goes_to_sentinel_bucket():
    """行业缺失归入同一哨兵桶，同样受上限约束。"""
    cs = pd.Series({"x1": 10, "x2": 9, "x3": 8, "k1": 7})
    sec = _sectors({"x1": float("nan"), "x2": float("nan"),
                    "x3": float("nan"), "k1": "K"})
    s = SectorCappedTopKLongOnly(sec, k=3, max_per_sector=2)
    assert set(s.get_weights_at(D, cs).index) == {"x1", "x2", "k1"}


def test_unknown_date_all_sentinel():
    """日期不在行业面板 ⇒ 全部归哨兵桶，按上限只取 1 只。"""
    cs = pd.Series({"a": 10, "b": 9, "c": 8})
    sec = _sectors({"a": "A", "b": "B", "c": "C"})
    s = SectorCappedTopKLongOnly(sec, k=3, max_per_sector=1)
    assert len(s.get_weights_at(pd.Timestamp("2030-01-01"), cs)) == 1


def test_empty_cross_section_returns_empty():
    s = SectorCappedTopKLongOnly(_sectors({"a": "A"}), k=3, max_per_sector=2)
    assert s.get_weights_at(D, pd.Series(dtype=float)).empty


def test_get_weights_without_date_raises():
    s = SectorCappedTopKLongOnly(_sectors({"a": "A"}), k=3)
    with pytest.raises(NotImplementedError):
        s.get_weights(pd.Series({"a": 1.0}))


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        SectorCappedTopKLongOnly(_sectors({"a": "A"}), k=0)
    with pytest.raises(ValueError):
        SectorCappedTopKLongOnly(_sectors({"a": "A"}), max_per_sector=0)
