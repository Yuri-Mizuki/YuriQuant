"""top-k 选取的 tie-break 确定性（2026-09-11 统一后锁死）

背景：``strategy/examples.py`` 三处 top-k 原用 ``vals.sort_values()``
（pandas 默认 quicksort，不稳定），与 ``strategy.constraints.build_signal_weights``
的 ``rank(ascending=False, method="first")`` 语义相反。真实因子库实测约 4.4% 的
截面在 top-k 边界并列，最坏情况下两种语义选出的持仓完全无交集
（``scripts/oneoff/probe_tie_stable_sort.py``）。

现全仓统一为 ``rank(method="first")``——并列时**列序靠前者优先**。本文件把
"确定性 tie 语义"钉死，防止回潮。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategy.constraints import build_signal_weights
from strategy.examples import (
    BufferedTopFracLongOnly,
    TopFracLongOnly,
    TopKLongOnly,
    TopKLongShort,
)


def _codes(n: int = 10) -> list[str]:
    return [f"C{i:02d}" for i in range(n)]


def _rowwise(panel: pd.DataFrame, strat) -> pd.DataFrame:
    return pd.concat([strat.get_weights(row) for _, row in panel.iterrows()],
                     axis=1).T.fillna(0.0)


# ===========================================================================
# 1) 并列时按列序（method="first"）确定性取股
# ===========================================================================
def test_topk_longonly_tie_break_column_order():
    """全同值：取列序最靠前的 k 只。"""
    vals = pd.Series(1.0, index=_codes())
    w = TopKLongOnly(k=3, weight_mode="equal").get_weights(vals)
    assert list(w.index) == ["C00", "C01", "C02"]
    assert np.allclose(w.values, 1 / 3)


def test_topfrac_longonly_tie_break_column_order():
    vals = pd.Series(1.0, index=_codes())
    w = TopFracLongOnly(frac=0.30).get_weights(vals)   # k = round(0.3*10) = 3
    assert list(w.index) == ["C00", "C01", "C02"]


def test_topk_longshort_tie_break_column_order():
    """全同值：多头取列序前 k，空头取列序后 k（不重叠）。"""
    vals = pd.Series(1.0, index=_codes())
    w = TopKLongShort(k=2, weight_mode="equal").get_weights(vals)
    long_codes = [c for c in w.index if w[c] > 0]
    short_codes = [c for c in w.index if w[c] < 0]
    assert long_codes == ["C00", "C01"]
    assert sorted(short_codes) == ["C08", "C09"]
    assert not set(long_codes) & set(short_codes)


def test_mixed_values_ties_filled_by_column_order():
    """最高值优先，其余名额按列序补足。"""
    vals = pd.Series(1.0, index=_codes())
    vals["C07"] = 5.0
    w = TopKLongOnly(k=3, weight_mode="equal").get_weights(vals)
    assert set(w.index) == {"C07", "C00", "C01"}


# ===========================================================================
# 2) 与 constraints 的面板级实现同语义（2026-09-11 统一）
# ===========================================================================
@pytest.mark.parametrize("k", [1, 3, 5, 9])
def test_topk_longonly_matches_equal_topk_under_ties(k):
    """tie 截面下 ``TopKLongOnly`` 与 ``build_signal_weights(equal_topk)`` 逐位一致。

    这是本批统一的核心断言——旧版二者在 tie 上刻意不同（本测试会红）。
    """
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    panel = pd.DataFrame(1.0, index=idx, columns=_codes())   # 全并列
    panel.iloc[1, 7] = 5.0                                   # 行内混合
    panel.iloc[2, :] = np.arange(10)                         # 严格递增，无 tie
    panel.iloc[3, [0, 1, 2]] = np.nan                        # 含缺失

    a = build_signal_weights(panel, "equal_topk", k=k)
    b = (_rowwise(panel, TopKLongOnly(k, "equal"))
         .reindex(columns=a.columns).fillna(0.0))
    assert np.allclose(a.values, b.values, atol=1e-15)


# ===========================================================================
# 3) 边界与幂等
# ===========================================================================
def test_empty_section_returns_empty_not_all():
    """空截面返回空权重（旧实现 ``index[-0:]`` 会返回全部索引 → inf 权重）。"""
    empty = pd.Series(dtype=float)
    assert TopKLongOnly(k=5).get_weights(empty).empty
    assert TopKLongOnly(k=5).get_weights(pd.Series([np.nan, np.nan])).empty
    assert TopFracLongOnly(0.2).get_weights(empty).empty


def test_repeated_calls_are_identical():
    """同一输入重复调用结果逐位一致（可复现性）。"""
    vals = pd.Series(1.0, index=_codes())
    s = TopFracLongOnly(frac=0.20)
    first = s.get_weights(vals)
    for _ in range(5):
        assert s.get_weights(vals).equals(first)


def test_buffered_is_deterministic_on_ties():
    """``BufferedTopFracLongOnly`` 一直是确定的（内部已用 ``method="first"``）。"""
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    panel = pd.DataFrame(1.0, index=idx, columns=_codes())

    def run() -> pd.DataFrame:
        s = BufferedTopFracLongOnly(frac_entry=0.2, frac_exit=0.4)
        return pd.concat([s.get_weights(r) for _, r in panel.iterrows()],
                         axis=1).T.fillna(0.0)

    assert run().equals(run())


# ===========================================================================
# 4) 静态守卫：不得再对原始因子值用不稳定的 sort_values()
# ===========================================================================
def test_no_unstable_sort_on_raw_values():
    src = (Path(__file__).resolve().parents[1] / "strategy" / "examples.py"
           ).read_text(encoding="utf-8")
    assert "vals.sort_values()" not in src, (
        "top-k 选取不得再对原始因子值调用 sort_values()（quicksort 不稳定）；"
        "请走 _top_k_indices / _bottom_k_indices"
    )
