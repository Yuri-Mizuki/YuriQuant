"""`SectorCappedBufferedTopKLongOnly` 单测（2026-11-10 新增）。

这个类是小资金档 `yqcore-solo` 的实现，锁四件事：

1. **行业去重真的生效** —— 每行业最多 `max_per_sector` 只，且欠仓（不越界补足）。
2. **缓冲带与行业名额叠加时语义不变** —— 上期持仓只要排名仍在 `k·exit_mult` 内、
   且行业有名额就保留。
3. **`min_holdings` 是"留现金"不是"补股票"** —— 持仓不足时每只按 1/min_holdings 配，
   权重和 < 1，差额是现金。
4. **`gross_exposure` 是纯缩放** —— 权重和 = gross_exposure，相对权重不变。

另有一条**等价性**测试：`max_per_sector=1 + exit_mult=1 + min_holdings=1` 时应与
`SectorCappedTopKLongOnly(max_per_sector=1)` 选出同一批股票（同一 tie-break）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.examples import (
    BufferedTopKLongOnly,
    SectorCappedBufferedTopKLongOnly,
    SectorCappedTopKLongOnly,
)

D = pd.Timestamp("2026-09-04")
D2 = pd.Timestamp("2026-10-09")


def _sec(mapping: dict, date: pd.Timestamp = D) -> pd.DataFrame:
    return pd.DataFrame([mapping], index=[date])


def test_cap_one_per_sector():
    """行业 A 占前 3 高分，上限 1 ⇒ 只取 A 的第 1；B/C 的股票仍在 TopK 内 ⇒ 一并纳入。"""
    cs = pd.Series({"a1": 10, "a2": 9, "a3": 8, "b1": 7, "c1": 6})
    sec = _sec({"a1": "A", "a2": "A", "a3": "A", "b1": "B", "c1": "C"})
    s = SectorCappedBufferedTopKLongOnly(sec, k=5, exit_mult=1.0, max_per_sector=1)
    w = s.get_weights_at(D, cs)
    assert set(w.index) == {"a1", "b1", "c1"}
    assert w.sum() == pytest.approx(1.0)


def test_does_not_reach_beyond_topk_to_fill_k():
    """🚨 语义边界：被行业挡掉的空缺**不往下补**（只在 TopK 内选）。

    这是「实际持仓 < k」的来源，也是本类与 `SectorCappedTopKLongOnly` 的差别所在：
    后者会继续往下扫直到凑满 k。要不要改成补满见 `reports/focus_micro/README.md`。
    """
    cs = pd.Series({"a1": 10, "a2": 9, "a3": 8, "b1": 7, "c1": 6, "d1": 5})
    sec = _sec({"a1": "A", "a2": "A", "a3": "A",
                "b1": "B", "c1": "C", "d1": "D"})
    s = SectorCappedBufferedTopKLongOnly(sec, k=3, exit_mult=1.0, max_per_sector=1)
    w = s.get_weights_at(D, cs)
    assert set(w.index) == {"a1"}, "TopK3 全是 A 行业 ⇒ 只留 a1，不拿 b1/c1 补满"


def test_underfills_when_industries_scarce():
    """全部同行业 + 上限 1 ⇒ 只买 1 只（不越界凑数），这是"实际持仓<k"的来源。"""
    cs = pd.Series({"a1": 10, "a2": 9, "a3": 8})
    sec = _sec({c: "A" for c in cs.index})
    s = SectorCappedBufferedTopKLongOnly(sec, k=3, max_per_sector=1)
    w = s.get_weights_at(D, cs)
    assert set(w.index) == {"a1"}
    assert w.sum() == pytest.approx(1.0)     # min_holdings=1 ⇒ 满仓押一只


def test_buffer_keeps_holding_within_exit_band():
    """上期持有 a1（本期排名 3，在 2×2=4 的保留带内）⇒ 保留；a2 掉到带外 ⇒ 换掉。"""
    sec = _sec({"a1": "A", "a2": "B", "b1": "B", "c1": "C", "d1": "D"})
    s = SectorCappedBufferedTopKLongOnly(sec, k=2, exit_mult=2.0, max_per_sector=1)
    w1 = s.get_weights_at(D, pd.Series({"a1": 10, "a2": 9, "b1": 8, "c1": 7, "d1": 6}))
    assert set(w1.index) == {"a1", "a2"}
    # 下期：b1 升到第 1，a2 掉到第 5（>4），a1 第 3
    sec2 = _sec({"a1": "A", "a2": "B", "b1": "B", "c1": "C", "d1": "D"}, D2)
    s.sectors = pd.concat([sec, sec2])
    w2 = s.get_weights_at(D2, pd.Series({"b1": 10, "c1": 9, "a1": 8, "d1": 7, "a2": 6}))
    assert "a1" in w2.index, "a1 排名 3 在保留带(≤4)内，应保留"
    assert "a2" not in w2.index, "a2 排名 5 超出保留带，应换掉"
    assert "b1" in w2.index, "b1 是本期第 1，应进"


def test_min_holdings_leaves_cash():
    """持仓 2 只但 min_holdings=5 ⇒ 每只 1/5，权重和 0.4（其余现金）。"""
    cs = pd.Series({"a1": 10, "b1": 9})
    sec = _sec({"a1": "A", "b1": "B"})
    s = SectorCappedBufferedTopKLongOnly(sec, k=5, max_per_sector=1, min_holdings=5)
    w = s.get_weights_at(D, cs)
    assert set(w.index) == {"a1", "b1"}
    assert w.sum() == pytest.approx(2 / 5)
    assert np.allclose(w.to_numpy(), 0.2)


def test_gross_exposure_scales_weights_only():
    """gross_exposure=0.75 ⇒ 权重和 0.75，相对权重不变。"""
    cs = pd.Series({"a1": 10, "b1": 9, "c1": 8})
    sec = _sec({"a1": "A", "b1": "B", "c1": "C"})
    full = SectorCappedBufferedTopKLongOnly(sec, k=3, max_per_sector=1)
    part = SectorCappedBufferedTopKLongOnly(
        sec, k=3, max_per_sector=1, gross_exposure=0.75)
    wf, wp = full.get_weights_at(D, cs), part.get_weights_at(D, cs)
    assert wf.sum() == pytest.approx(1.0)
    assert wp.sum() == pytest.approx(0.75)
    pd.testing.assert_series_equal(wp / wp.sum(), wf / wf.sum(), check_names=False)


def test_na_industry_shares_one_slot():
    """无行业分类（NaN）共用一个 `__NA__` 名额，不是每只一个。"""
    cs = pd.Series({"x1": 10, "x2": 9, "a1": 8})
    sec = _sec({"x1": None, "x2": None, "a1": "A"})
    s = SectorCappedBufferedTopKLongOnly(sec, k=3, max_per_sector=1)
    w = s.get_weights_at(D, cs)
    assert set(w.index) == {"x1", "a1"}


def test_is_subset_of_sector_cap_when_selection_not_binding():
    """`SectorCappedTopKLongOnly` 会往下补满 k，本类只在 TopK 内选 ⇒ 本类是它的子集。"""
    cs = pd.Series({"a1": 10, "a2": 9, "b1": 8, "b2": 7, "c1": 6, "d1": 5})
    sec = _sec({"a1": "A", "a2": "A", "b1": "B", "b2": "B", "c1": "C", "d1": "D"})
    ref = SectorCappedTopKLongOnly(sec, k=4, max_per_sector=1).get_weights_at(D, cs)
    new = SectorCappedBufferedTopKLongOnly(
        sec, k=4, exit_mult=1.0, max_per_sector=1).get_weights_at(D, cs)
    assert set(ref.index) == {"a1", "b1", "c1", "d1"}, "旧类补满 4 只"
    assert set(new.index) == {"a1", "b1"}, "新类只在 TopK4 内去重（Top4 = a1,a2,b1,b2）"
    assert set(new.index) < set(ref.index)


def test_matches_plain_buffer_when_every_name_distinct_industry():
    """每只都在不同行业 + 上限 1 ⇒ 与 `BufferedTopKLongOnly` 同选股。"""
    codes = [f"s{i}" for i in range(6)]
    cs = pd.Series({c: 10.0 - i for i, c in enumerate(codes)})
    sec = _sec({c: c.upper() for c in codes})
    ref = BufferedTopKLongOnly(k=3, exit_mult=1.0).get_weights_at(D, cs)
    new = SectorCappedBufferedTopKLongOnly(
        sec, k=3, exit_mult=1.0, max_per_sector=1).get_weights_at(D, cs)
    assert set(ref.index) == set(new.index)


def test_empty_cross_section_and_missing_date():
    s = SectorCappedBufferedTopKLongOnly(_sec({"a": "A"}), k=3, max_per_sector=1)
    assert s.get_weights_at(D, pd.Series(dtype=float)).empty
    # 面板里没有的日期 ⇒ 行业全 NaN，仍能出权重（都进 __NA__）
    w = s.get_weights_at(pd.Timestamp("2026-11-02"), pd.Series({"a": 1.0}))
    assert list(w.index) == ["a"]


def test_get_weights_without_date_raises():
    s = SectorCappedBufferedTopKLongOnly(_sec({"a": "A"}), k=3)
    with pytest.raises(NotImplementedError):
        s.get_weights(pd.Series({"a": 1.0}))


def test_invalid_params_raise():
    sec = _sec({"a": "A"})
    with pytest.raises(ValueError):
        SectorCappedBufferedTopKLongOnly(sec, k=0)
    with pytest.raises(ValueError):
        SectorCappedBufferedTopKLongOnly(sec, exit_mult=0.5)
    with pytest.raises(ValueError):
        SectorCappedBufferedTopKLongOnly(sec, max_per_sector=0)
    with pytest.raises(ValueError):
        SectorCappedBufferedTopKLongOnly(sec, min_holdings=0)
    with pytest.raises(ValueError):
        SectorCappedBufferedTopKLongOnly(sec, gross_exposure=0.0)
    with pytest.raises(ValueError):
        SectorCappedBufferedTopKLongOnly(sec, gross_exposure=1.5)


# ---- min_sectors：行业数地板（2026-11-10 新增）----

def test_min_sectors_extends_beyond_topk_for_new_industries():
    """TopK 内只剩 1 个行业 ⇒ 从 TopK 之外补「新行业」各 1 只，补到 min_sectors 为止。"""
    cs = pd.Series({"a1": 10, "a2": 9, "a3": 8, "b1": 7, "c1": 6, "d1": 5, "e1": 4})
    sec = _sec({"a1": "A", "a2": "A", "a3": "A",
                "b1": "B", "c1": "C", "d1": "D", "e1": "E"})
    # 无地板：TopK3 全 A ⇒ 只剩 a1
    off = SectorCappedBufferedTopKLongOnly(sec, k=3, max_per_sector=1)
    assert set(off.get_weights_at(D, cs).index) == {"a1"}
    # 地板 3：补 b1（B）、c1（C）；d1 排名 6 > k·1.0 会被上限挡住 ⇒ 不补
    on = SectorCappedBufferedTopKLongOnly(
        sec, k=3, max_per_sector=1, min_sectors=3)
    w = on.get_weights_at(D, cs)
    assert set(w.index) == {"a1", "b1", "c1"}
    assert w.sum() == pytest.approx(1.0)


def test_min_sectors_respects_rank_cap():
    """``sector_rank_mult`` 是硬上限：新行业候选排名超过 k·mult 就停止下探。"""
    cs = pd.Series({"a1": 10, "a2": 9, "a3": 8, "b1": 7, "z1": 1})
    sec = _sec({"a1": "A", "a2": "A", "a3": "A", "b1": "B", "z1": "Z"})
    # k=3, mult=1.0 ⇒ 候选上限排名 3 ⇒ b1(4) 已越界，只能只有 a1
    w = SectorCappedBufferedTopKLongOnly(
        sec, k=3, max_per_sector=1, min_sectors=2,
        sector_rank_mult=1.0).get_weights_at(D, cs)
    assert set(w.index) == {"a1"}
    # mult=3.0 ⇒ 上限排名 9 ⇒ b1 可用
    w2 = SectorCappedBufferedTopKLongOnly(
        sec, k=3, max_per_sector=1, min_sectors=2,
        sector_rank_mult=3.0).get_weights_at(D, cs)
    assert set(w2.index) == {"a1", "b1"}


def test_min_sectors_never_exceeds_k_holdings():
    """补位受 ``len(port) < k`` 约束，不会为了凑行业数超过 k 只。"""
    cs = pd.Series({f"s{i}": 100 - i for i in range(12)})
    sec = _sec({f"s{i}": f"X{i}" for i in range(12)})   # 12 个不同行业
    w = SectorCappedBufferedTopKLongOnly(
        sec, k=4, max_per_sector=1, min_sectors=4,
        sector_rank_mult=5.0).get_weights_at(D, cs)
    assert len(w.index) == 4


def test_min_sectors_zero_is_noop():
    """默认 0 ⇒ 与不传完全一致（历史数字逐位不变的保证）。"""
    cs = pd.Series({"a1": 10, "a2": 9, "b1": 8, "c1": 7})
    sec = _sec({"a1": "A", "a2": "A", "b1": "B", "c1": "C"})
    base = SectorCappedBufferedTopKLongOnly(sec, k=3, max_per_sector=1)
    zero = SectorCappedBufferedTopKLongOnly(sec, k=3, max_per_sector=1, min_sectors=0)
    pd.testing.assert_series_equal(base.get_weights_at(D, cs),
                                   zero.get_weights_at(D, cs))


def test_min_sectors_does_not_override_max_per_sector():
    """补进来的股票也必须守 ``max_per_sector``：已满的行业不会再补第 2 只。"""
    cs = pd.Series({"a1": 10, "a2": 9, "b1": 8, "b2": 7, "c1": 6})
    sec = _sec({"a1": "A", "a2": "A", "b1": "B", "b2": "B", "c1": "C"})
    w = SectorCappedBufferedTopKLongOnly(
        sec, k=4, max_per_sector=1, min_sectors=3,
        sector_rank_mult=3.0).get_weights_at(D, cs)
    got = [("A" if c.startswith("a") else c[0].upper()) for c in w.index]
    assert len(got) == len(set(got)), f"行业重复: {list(w.index)}"


def test_min_sectors_gt_k_raises():
    with pytest.raises(ValueError):
        SectorCappedBufferedTopKLongOnly(_sec({"a": "A"}), k=3, min_sectors=4)
    with pytest.raises(ValueError):
        SectorCappedBufferedTopKLongOnly(_sec({"a": "A"}), k=3, min_sectors=-1)
    with pytest.raises(ValueError):
        SectorCappedBufferedTopKLongOnly(_sec({"a": "A"}), k=3, sector_rank_mult=0.5)
