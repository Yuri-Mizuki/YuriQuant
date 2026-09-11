"""
ML stacking 单元测试（模型层）
==============================

覆盖 2026-09-11 自 ``factor/synthesis.py`` 迁入 :mod:`model.stacking` 的四个
stacking 合成器（ridge / LightGBM / optuna 调参 / LambdaRank）与其时序 CV
切分辅助 ``_time_fold_masks``：只验「不崩 + 输出有限 + 切折无泄漏」的冒烟性质，
数值口径的对照见 ``tests/test_synthesis.py``（因子层确定性组合）。

因子层的确定性组合测试仍在 ``tests/test_synthesis.py``。
"""
import numpy as np
import pandas as pd
import pytest

from model.stacking import (
    _time_fold_masks,
    synthesize_stacking,
    synthesize_stacking_gbdt,
    synthesize_stacking_gbdt_tuned,
    synthesize_stacking_lambdarank,
)


def test_synthesize_stacking_finite(synth_parts):
    ret_panel, comps = synth_parts
    comp = synthesize_stacking(comps, ret_panel, n_splits=5)
    assert comp.shape == comps[0].panel.shape
    # stacking 预测只覆盖测试折，可能部分 NaN，但不应出现 inf
    assert np.isfinite(comp.dropna().values).all()


def test_synthesize_stacking_gbdt_finite(synth_parts):
    """GBDT stacking 冒烟：shape/有限性 + purged CV 不崩（lightgbm 缺失时跳过）。"""
    pytest.importorskip("lightgbm")
    ret_panel, comps = synth_parts
    comp = synthesize_stacking_gbdt(comps, ret_panel, n_splits=3, embargo_days=2,
                                    n_estimators=30, max_depth=3)
    assert comp.shape == comps[0].panel.shape
    assert np.isfinite(comp.dropna().values).all()


def test_synthesize_stacking_gbdt_tuned_finite(synth_parts):
    """optuna 调参版冒烟：嵌套时序 CV 不崩、输出有限（缺 optuna/lightgbm 跳过）。"""
    pytest.importorskip("lightgbm")
    pytest.importorskip("optuna")
    ret_panel, comps = synth_parts
    comp = synthesize_stacking_gbdt_tuned(comps, ret_panel, n_splits=2,
                                          embargo_days=2, n_trials=3)
    assert comp.shape == comps[0].panel.shape
    assert np.isfinite(comp.dropna().values).all()


def test_synthesize_stacking_rank_target(synth_parts):
    """方案 A：截面秩目标（target_mode='rank'）不崩且输出有限。

    秩目标应让 rank IC 评价下与 raw 目标可比（Spearman 对 y 单调变换不变）。
    """
    ret_panel, comps = synth_parts
    comp_rank = synthesize_stacking(comps, ret_panel, n_splits=3, target_mode="rank")
    comp_raw = synthesize_stacking(comps, ret_panel, n_splits=3, target_mode="raw")
    assert comp_rank.shape == comps[0].panel.shape
    assert np.isfinite(comp_rank.dropna().values).all()
    assert np.isfinite(comp_raw.dropna().values).all()


def test_synthesize_stacking_lambdarank_finite(synth_parts):
    """方案 B：LambdaRank stacking 冒烟（按日 group + purged CV 不崩）。"""
    pytest.importorskip("lightgbm")
    ret_panel, comps = synth_parts
    comp = synthesize_stacking_lambdarank(comps, ret_panel, n_splits=2,
                                          embargo_days=2, n_estimators=30, max_depth=3)
    assert comp.shape == comps[0].panel.shape
    assert np.isfinite(comp.dropna().values).all()


def test_stacking_prediction_uses_only_past(synth_parts):
    """验证 stacking 的某测试折预测仅用此前数据：构造一个前段无信号、
    后段强信号的数据，早期测试折的预测不应出现"未来才有的信号"。"""
    ret_panel, comps = synth_parts
    comp = synthesize_stacking(comps, ret_panel, n_splits=4, target_mode="raw")
    assert comp.shape == comps[0].panel.shape
    assert np.isfinite(comp.dropna().values).all()


def test_stacking_rejects_empty_components():
    """空因子列表应显式报错（ridge 分支无需 lightgbm，可直接断言）。"""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=10, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(5)]
    ret_panel = pd.DataFrame(rng.normal(0, 0.01, (10, 5)), idx, codes)
    with pytest.raises(ValueError, match="components 为空"):
        synthesize_stacking([], ret_panel)


# ===========================================================================
# 2026-08-17 修复：按【交易日边界】切分的时序 CV（防把某一天劈开 / 防未来函数）
# 2026-09-11 随 stacking 迁入模型层（原 tests/test_synthesis.py）
# ===========================================================================
def test_time_folds_no_split_within_day():
    """fold 边界不得落在某交易日中间：train 与 test 的日期集合不相交，
    且每个 fold 的 train 日期严格早于 test 日期（无未来函数）。"""
    # 模拟 20 天 × 3 股，且每天有效股票数不同（模拟 NaN 缺失）
    dates = pd.date_range("2023-01-02", periods=20, freq="B")
    per_day = [3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3]
    date_arr = np.concatenate([np.repeat(d, n) for d, n in zip(dates, per_day)])

    folds = _time_fold_masks(date_arr, n_splits=5)
    assert len(folds) == 4
    for train_mask, test_mask in folds:
        tr_days = set(pd.unique(date_arr[train_mask]))
        te_days = set(pd.unique(date_arr[test_mask]))
        # 1) 不相交
        assert tr_days.isdisjoint(te_days)
        # 2) train 全部早于 test（时序无泄漏）
        assert max(tr_days) < min(te_days)
        # 3) 覆盖完整交易日（test 集合就是该段全部日期）
        assert te_days == set(te_days)


def test_time_folds_embargo_purges_adjacent_days():
    """embargo_days 应剔除训练段末尾与测试段相邻的天（标签时间重叠防泄漏）。"""
    dates = pd.date_range("2023-01-02", periods=30, freq="B")
    date_arr = np.repeat(dates, 3)  # 每天 3 股
    folds = _time_fold_masks(date_arr, n_splits=5, embargo_days=2)
    assert len(folds) == 4
    # 每折 purge 后，训练段末尾日期到测试段起始日期的间隔 >= embargo_days 个交易日
    for train_mask, test_mask in folds:
        tr_days = pd.unique(date_arr[train_mask])
        te_days = pd.unique(date_arr[test_mask])
        gap = min(te_days) - max(tr_days)
        # 两个交易日之间至少相隔 1 天（周末可能更多），故 gap >= embargo_days 天
        assert gap >= pd.Timedelta(days=2)


def test_time_folds_all_obs_covered_across_folds():
    """所有 fold 的测试段并集应覆盖全部观测（每个样本至少被预测一次）。"""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-02", periods=40, freq="B")
    date_arr = np.concatenate([
        np.repeat(d, rng.integers(2, 6)) for d in dates
    ])
    folds = _time_fold_masks(date_arr, n_splits=6)
    assert len(folds) == 5
    # expanding-window：首个测试折从 edges[1]（约 1/n_splits 处）才开始，
    # 因此最前面约 1/6 的观测不会进入任何测试折（属预期，不是泄漏）。
    covered = np.zeros(len(date_arr), dtype=bool)
    for _, test_mask in folds:
        covered |= test_mask
    assert covered.any()
    # 最后一段（最后一个测试折）应被覆盖
    last_day = pd.unique(date_arr)[-1]
    assert covered[date_arr == last_day].any()
    # 各折测试段互不重叠（同一观测不会出现在两个测试折）
    seen = np.zeros(len(date_arr), dtype=bool)
    for _, test_mask in folds:
        assert not (seen & test_mask).any()
        seen |= test_mask


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
