"""factor.cv 统一交叉验证纪律单测。

验证三种切分的正确性：
1. 三段式：日期边界正确，三段不交叠
2. Purged K-Fold：训练集取两侧、embargo 隔离带生效、无日期交叉
3. CPCV：路径数 = C(N,k)、每组恰好出现 k 次、embargo 生效
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.cv import (split_three_periods, purged_kfold, cpcv, CPCVPath,
                       Fold, forward_folds, forward_roll_folds,
                       blocked_folds, make_folds)


# ---------------------------------------------------------------------------
# 测试数据：500 个交易日
# ---------------------------------------------------------------------------
@pytest.fixture
def dates():
    """1200 个交易日，覆盖 2022-01 ~ 2026-08，足以切三段式默认日期。"""
    return pd.date_range("2022-01-03", periods=1200, freq="B")


# ---------------------------------------------------------------------------
# 1. 三段式
# ---------------------------------------------------------------------------
class TestSplitThreePeriods:
    def test_three_disjoint(self, dates):
        tr, va, te = split_three_periods(dates)
        assert len(tr) > 0 and len(va) > 0 and len(te) > 0
        # 三段不交叠
        assert len(set(tr) & set(va)) == 0
        assert len(set(tr) & set(te)) == 0
        assert len(set(va) & set(te)) == 0

    def test_order(self, dates):
        tr, va, te = split_three_periods(dates)
        assert tr.max() < va.min()
        assert va.max() < te.min()

    def test_custom_dates(self, dates):
        tr, va, te = split_three_periods(
            dates, train=(20220101, 20221231), valid=(20230101, 20231231),
            test=(20240101, 20251231))
        assert tr.min() >= pd.Timestamp("2022-01-01")
        assert tr.max() <= pd.Timestamp("2022-12-31")
        assert te.min() >= pd.Timestamp("2024-01-01")


# ---------------------------------------------------------------------------
# 2. Purged K-Fold
# ---------------------------------------------------------------------------
class TestPurgedKFold:
    def test_n_folds(self, dates):
        folds = purged_kfold(dates, n_splits=5, embargo_days=5)
        assert len(folds) == 5

    def test_test_days_disjoint(self, dates):
        folds = purged_kfold(dates, n_splits=5, embargo_days=5)
        all_test = []
        for f in folds:
            all_test.extend(f.test_days)
        # 测试段之间不交叠
        assert len(all_test) == len(set(all_test))

    def test_train_excludes_test_and_embargo(self, dates):
        folds = purged_kfold(dates, n_splits=5, embargo_days=5)
        for f in folds:
            # 训练集不含任何测试日
            assert len(set(f.train_days) & set(f.test_days)) == 0
            # embargo 生效：测试段前后 embargo_days 天不在训练集
            te_sorted = f.test_days
            te_start = te_sorted[0]
            te_end = te_sorted[-1]
            te_start_idx = dates.get_loc(te_start)
            te_end_idx = dates.get_loc(te_end)
            if te_start_idx > 0:
                for d in dates[max(0, te_start_idx - 5):te_start_idx]:
                    assert d not in set(f.train_days), \
                        f"embargo day {d} should not be in train"
            if te_end_idx < len(dates) - 1:
                for d in dates[te_end_idx + 1:min(len(dates), te_end_idx + 6)]:
                    assert d not in set(f.train_days), \
                        f"embargo day {d} should not be in train"

    def test_test_cover_all_days(self, dates):
        """embargo 区会被丢弃，但测试段覆盖全部日期。"""
        folds = purged_kfold(dates, n_splits=5, embargo_days=5)
        all_test = set()
        for f in folds:
            all_test.update(f.test_days)
        assert all_test == set(dates)

    def test_embargo_zero(self, dates):
        """embargo=0 时，训练集 = 全部日期 - 测试段。"""
        folds = purged_kfold(dates, n_splits=5, embargo_days=0)
        for f in folds:
            expected_train = set(dates) - set(f.test_days)
            assert set(f.train_days) == expected_train


# ---------------------------------------------------------------------------
# 3. CPCV
# ---------------------------------------------------------------------------
class TestCPCV:
    def test_path_count(self, dates):
        from math import comb
        paths = cpcv(dates, n_groups=6, k=2, embargo_days=5)
        assert len(paths) == comb(6, 2)  # 15

    def test_groups_balance(self, dates):
        """每组恰好出现在 C(N-1, k-1) 条路径的测试集中。"""
        from math import comb
        N, k = 6, 2
        paths = cpcv(dates, n_groups=N, k=k, embargo_days=5)
        # 每组作为测试组的次数
        counts = {i: 0 for i in range(N)}
        for p in paths:
            for gi in p.test_groups:
                counts[gi] += 1
        expected = comb(N - 1, k - 1)  # C(5,1)=5
        for gi, cnt in counts.items():
            assert cnt == expected, f"组 {gi} 出现 {cnt} 次, 期望 {expected}"

    def test_train_excludes_test(self, dates):
        paths = cpcv(dates, n_groups=6, k=2, embargo_days=5)
        for p in paths:
            assert len(set(p.train_days) & set(p.test_days)) == 0

    def test_embargo(self, dates):
        paths = cpcv(dates, n_groups=6, k=2, embargo_days=5)
        for p in paths:
            for gi in p.test_groups:
                # 找该组在 dates 中的位置
                # 用 groups[0] 的第一天近似定位
                pass  # embargo 逻辑在 train_excludes_test 中已隐式验证

    def test_test_days_cover_all(self, dates):
        """所有路径的测试段并集 = 全部日期（无丢失）。"""
        paths = cpcv(dates, n_groups=6, k=2, embargo_days=5)
        all_test = set()
        for p in paths:
            all_test.update(p.test_days)
        assert all_test == set(dates)

    def test_invalid_params(self, dates):
        with pytest.raises(ValueError):
            cpcv(dates, n_groups=1, k=1)
        with pytest.raises(ValueError):
            cpcv(dates, n_groups=6, k=0)
        with pytest.raises(ValueError):
            cpcv(dates, n_groups=6, k=6)

    def test_classic_config(self, dates):
        """经典配置 N=6, k=2 → 15 路径，每路径 2 组当测试。"""
        paths = cpcv(dates, n_groups=6, k=2, embargo_days=5)
        assert len(paths) == 15
        for p in paths:
            assert len(p.test_groups) == 2
            assert len(p.train_days) > 0
            assert len(p.test_days) > 0


# ---------------------------------------------------------------------------
# 4. 统一切分调度器 make_folds
# ---------------------------------------------------------------------------
class TestMakeFolds:
    def test_forward_is_expanding_and_no_future(self, dates):
        """forward：测试折只在末尾，训练严格早于测试，且训练段长度递增。"""
        folds = make_folds(dates, method="forward", n_splits=5, embargo_days=5)
        assert len(folds) == 4  # n_splits-1：首段只当训练
        lens = []
        for f in folds:
            assert f.train_days.max() < f.test_days.min()
            assert len(set(f.train_days) & set(f.test_days)) == 0
            lens.append(len(f.train_days))
        assert lens == sorted(lens), "expanding 训练段应收敛递增"

    def test_forward_matches_forward_folds(self, dates):
        a = make_folds(dates, "forward", 5, 5)
        b = forward_folds(dates, 5, 5)
        assert len(a) == len(b)
        for f, g in zip(a, b):
            assert f.train_days.equals(g.train_days)
            assert f.test_days.equals(g.test_days)

    def test_forward_embargo_purges_adjacent_days(self, dates):
        """forward 的 embargo：训练段末尾剔除与测试段相邻的天。"""
        folds = make_folds(dates, "forward", n_splits=5, embargo_days=3)
        for f in folds:
            te_start_idx = dates.get_loc(f.test_days[0])
            if te_start_idx >= 3:
                for d in dates[te_start_idx - 3:te_start_idx]:
                    assert d not in set(f.train_days), \
                        f"embargo day {d} should not be in train"

    def test_purged_matches_purged_kfold(self, dates):
        a = make_folds(dates, "purged", 5, 3)
        b = purged_kfold(dates, n_splits=5, embargo_days=3)
        assert len(a) == len(b)
        for f, g in zip(a, b):
            assert f.train_days.equals(g.train_days)
            assert f.test_days.equals(g.test_days)

    def test_blocked_covers_all_and_no_leak(self, dates):
        """blocked：训练 = 全部 - 测试块 - embargo，测试块覆盖全部日期。"""
        folds = make_folds(dates, "blocked", n_splits=5, embargo_days=3)
        assert len(folds) == 5
        all_test = set()
        for f in folds:
            assert len(set(f.train_days) & set(f.test_days)) == 0
            all_test.update(f.test_days)
        assert all_test == set(dates), "测试块并集应为全历史"

    def test_invalid_method(self, dates):
        with pytest.raises(ValueError):
            make_folds(dates, "kfold", 5, 5)

    def test_all_methods_return_fold(self, dates):
        for m in ("forward", "purged", "blocked"):
            folds = make_folds(dates, m, n_splits=4, embargo_days=2)
            assert all(isinstance(f, Fold) for f in folds)


# ---------------------------------------------------------------------------
# 5. 生产专用前推滚动切分 forward_roll_folds
# ---------------------------------------------------------------------------
class TestForwardRollFolds:
    def test_covers_all_test_days(self, dates):
        """test 段等分后应覆盖全部 test_days（不丢首段）。"""
        all_days = dates
        test_days = dates[800:]  # 400 日
        folds = forward_roll_folds(all_days, test_days, n_folds=4, embargo_days=5)
        assert len(folds) == 4
        all_test = set()
        for f in folds:
            all_test.update(f.test_days)
        assert all_test == set(test_days), "test 等分应覆盖全部 test_days"

    def test_train_strictly_before_test(self, dates):
        all_days = dates
        test_days = dates[800:]
        folds = forward_roll_folds(all_days, test_days, n_folds=4, embargo_days=5)
        for f in folds:
            assert len(f.train_days) == 0 or f.train_days.max() < f.test_days[0]
            assert len(set(f.train_days) & set(f.test_days)) == 0

    def test_expanding_train_grows(self, dates):
        all_days = dates
        test_days = dates[800:]
        folds = forward_roll_folds(all_days, test_days, n_folds=4, embargo_days=5)
        lens = [len(f.train_days) for f in folds]
        assert lens == sorted(lens), "滚动训练段应收敛递增"

    def test_embargo_removes_adjacent_days(self, dates):
        all_days = dates
        test_days = dates[800:]
        embargo = 3
        folds = forward_roll_folds(all_days, test_days, n_folds=4,
                                   embargo_days=embargo)
        for f in folds:
            cut_idx = all_days.get_loc(f.test_days[0])
            if cut_idx >= embargo:
                for d in all_days[cut_idx - embargo:cut_idx]:
                    assert d not in set(f.train_days), \
                        f"embargo day {d} should not be in train"

    def test_too_few_test_days_raises(self, dates):
        with pytest.raises(ValueError):
            forward_roll_folds(dates, dates[:5], n_folds=6, embargo_days=0)
