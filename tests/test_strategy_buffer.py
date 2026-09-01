"""BufferedTopFracLongOnly 缓冲带行为单测。"""
import pandas as pd
import pytest

from strategy.examples import BufferedTopFracLongOnly


def _cross(values: dict) -> pd.Series:
    return pd.Series(values)


# 10 只股票、分数 10(最高)~1(最低)；entry=round(0.2*10)=2, exit=round(0.4*10)=4
_CS10 = {c: 10 - i for i, c in enumerate("abcdefghij")}


def test_first_rebalance_is_naive_top_entry():
    s = BufferedTopFracLongOnly(0.2, 0.4)
    w = s.get_weights(_cross(_CS10))
    assert list(w.index) == ["a", "b"]
    assert w.sum() == pytest.approx(1.0)


def test_buffer_band_retains_holding_naive_would_drop():
    s = BufferedTopFracLongOnly(0.2, 0.4)
    s.get_weights(_cross(_CS10))
    # b 从第 2 掉到第 3（entry 带外、exit 带内）→ 保留，且无空缺时不补新
    cs2 = _cross({"a": 10, "c": 9, "b": 8, "d": 7, "e": 6,
                  "f": 5, "g": 4, "h": 3, "i": 2, "j": 1})
    w = s.get_weights(cs2)
    assert set(w.index) == {"a", "b"}
    assert w.sum() == pytest.approx(1.0)


def test_holding_below_exit_band_sold_and_vacancy_filled_by_best_candidate():
    s = BufferedTopFracLongOnly(0.2, 0.4)
    s.get_weights(_cross(_CS10))
    # b 掉到第 5（exit 带外）→ 卖出；空缺由非持仓中排名最高且 rank<=entry 的 c 补
    cs2 = _cross({"a": 10, "c": 9, "d": 8, "e": 7, "b": 6,
                  "f": 5, "g": 4, "h": 3, "i": 2, "j": 1})
    w = s.get_weights(cs2)
    assert set(w.index) == {"a", "c"}


def test_missing_signal_holding_is_sold():
    s = BufferedTopFracLongOnly(0.2, 0.4)
    s.get_weights(_cross(_CS10))
    # b 当日无信号（截面缺值）→ 卖出，由 c 补足
    cs2 = _cross({"a": 10, "c": 9, "d": 8, "e": 7, "f": 6,
                  "g": 5, "h": 4, "i": 3, "j": 2})
    w = s.get_weights(cs2)
    assert "b" not in w.index
    assert "c" in w.index


def test_empty_cross_section_resets_state():
    s = BufferedTopFracLongOnly(0.2, 0.4)
    s.get_weights(_cross(_CS10))
    w = s.get_weights(pd.Series(dtype=float))
    assert w.empty
    # 状态已重置：下一期视为建仓期
    w2 = s.get_weights(_cross(_CS10))
    assert set(w2.index) == {"a", "b"}


def test_invalid_entry_exit_raises():
    with pytest.raises(ValueError):
        BufferedTopFracLongOnly(0.4, 0.2)
