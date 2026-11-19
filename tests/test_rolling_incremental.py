"""增量补 pred 的前提：数据延长不得改变历史折。

`rolling_grid_alla --stage predict --years <末一年>` 只重算最后一年，其余年份
沿用既有面板拼接。这条捷径唯一的安全性来源是：每折训练集 = 折起点之前的
历史（再按 ``max_train`` 截尾），**追加未来数据永远进不了历史折**。

若这条不成立，拼接出来的面板就是"早年份用旧数据训、末年份用新数据训"的
混代产物 —— 而它看起来完全正常（文件名、列、日期区间都对）。所以这里用
折集合的逐日相等来钉住，而不是靠注释承诺。

反向验证：把 ``forward_roll_folds`` 的 ``tr = all_days[all_days < cut]`` 改成
``<= cut`` 或让 ``tr`` 取全量，本文件第 1 条立即变红。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor.cv import forward_roll_folds  # noqa: E402

OLD_END = pd.Timestamp("2026-09-23")
NEW_END = pd.Timestamp("2026-11-06")
BEGIN = pd.Timestamp("2016-07-01")


def _days(end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(BEGIN, end)


def _folds(horizon: int, end: pd.Timestamp, year: int):
    all_days = _days(end)
    test_days = all_days[(all_days.year == year)]
    n = 2 if horizon >= 10 else 4
    return forward_roll_folds(all_days, test_days, n, embargo_days=horizon)


@pytest.mark.parametrize("horizon", [1, 5, 10, 20])
@pytest.mark.parametrize("year", [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
def test_history_folds_are_immune_to_data_extension(year: int, horizon: int) -> None:
    """早年份的 train/test 集合在数据延长前后必须逐日相同。

    ``year <= 2025`` 且 ``OLD_END`` 已覆盖该年 ⇒ 延长末端只影响 2026。
    """
    old = _folds(horizon, OLD_END, year)
    new = _folds(horizon, NEW_END, year)
    assert len(old) == len(new)
    for fo, fn in zip(old, new):
        assert fo.test_days.equals(fn.test_days), f"{year} h{horizon} test 段变了"
        assert fo.train_days.equals(fn.train_days), f"{year} h{horizon} train 段变了"


@pytest.mark.parametrize("horizon", [1, 5, 10, 20])
def test_last_year_follows_data_end(horizon: int) -> None:
    """末年份的折**应该**随数据延长而变 —— 它是要重算的那一年。"""
    old = _folds(horizon, OLD_END, 2026)
    new = _folds(horizon, NEW_END, 2026)
    assert new[-1].test_days[-1] == _days(NEW_END)[-1]
    assert old[-1].test_days[-1] < _days(NEW_END)[-1]
    assert len(new[-1].test_days) > len(old[-1].test_days)


@pytest.mark.parametrize("horizon", [1, 5, 10, 20])
def test_no_fold_trains_on_its_own_future(horizon: int) -> None:
    """任何折的训练段都必须严格早于该折测试起点（前视守卫）。"""
    for fold in _folds(horizon, NEW_END, 2026):
        assert fold.train_days.max() < fold.test_days.min()
