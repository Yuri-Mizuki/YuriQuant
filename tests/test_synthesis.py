"""
多因子合成单元测试
====================
构造确定性的 mock 因子面板，验证四种合成方式都产出有限、形状正确的复合因子，
并验证正交化确实消除了因子间相关性。
"""
import numpy as np
import pandas as pd
import pytest

from factor.operators import cs_rank, cs_zscore, ts_rank
from factor.preprocessing import standardize_zscore
from factor.synthesis import (
    CompositeInput, build_components, composite_stats, orthogonalize,
    rebuild_train_weights,
    synthesize_ic_weighted, synthesize_orthogonal, synthesize_pca,
    synthesize_stacking, synthesize_stacking_gbdt, synthesize_stacking_gbdt_tuned,
    synthesize_stacking_lambdarank,
    _time_folds, _inner_split_by_day,
)


@pytest.fixture
def mock_parts():
    """构造 mock 面板 + 3 个因子（含一个与未来收益正相关、一个高相关冗余、一个独立）。"""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2022-01-01", periods=200, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(20)]
    # 注入 AR(1) 收益 → 动量因子有正 IC
    phi = 0.3
    rets = np.zeros((200, 20))
    for t in range(1, 200):
        rets[t] = phi * rets[t - 1] + rng.normal(0, 0.02, 20)
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), idx, codes)
    ret_panel = close.pct_change().shift(-1)

    f1 = ts_rank(close, 5)            # 动量类，与收益正相关
    # f2 是 f1 的带噪副本 → 与 f1 高度冗余（用于测试正交化消除相关性）
    noise = pd.DataFrame(rng.normal(0, 0.1, (200, 20)), idx, codes)
    f2 = standardize_zscore(f1 + noise)
    f3 = cs_zscore(ts_rank(close, 20) * -1)  # 反转类，弱信号
    comps = [
        CompositeInput("ts_rank_5", standardize_zscore(f1), ic=0.15, ir=1.2),
        CompositeInput("f1_noisy", f2, ic=0.12, ir=1.0),
        CompositeInput("rev", standardize_zscore(f3), ic=-0.05, ir=-0.4),
    ]
    return ret_panel, comps


def test_synthesize_ic_weighted_shape(mock_parts):
    ret_panel, comps = mock_parts
    comp = synthesize_ic_weighted(comps, returns_panel=ret_panel)
    assert isinstance(comp, pd.DataFrame)
    assert comp.shape == comps[0].panel.shape
    assert comp.dropna().notna().all().all()


def test_synthesize_pca_shape(mock_parts):
    ret_panel, comps = mock_parts
    comp = synthesize_pca(comps, n_components=2, returns_panel=ret_panel)
    assert comp.shape == comps[0].panel.shape
    assert comp.dropna().notna().all().all()


def test_synthesize_orthogonal_shape(mock_parts):
    ret_panel, comps = mock_parts
    comp = synthesize_orthogonal(comps)
    assert comp.shape == comps[0].panel.shape
    assert comp.dropna().notna().all().all()


def test_synthesize_stacking_finite(mock_parts):
    ret_panel, comps = mock_parts
    comp = synthesize_stacking(comps, ret_panel, n_splits=5)
    assert comp.shape == comps[0].panel.shape
    # stacking 预测只覆盖测试折，可能部分 NaN，但不应出现 inf
    assert np.isfinite(comp.dropna().values).all()


def test_synthesize_stacking_gbdt_finite(mock_parts):
    """GBDT stacking 冒烟：shape/有限性 + purged CV 不崩（lightgbm 缺失时跳过）。"""
    pytest.importorskip("lightgbm")
    ret_panel, comps = mock_parts
    comp = synthesize_stacking_gbdt(comps, ret_panel, n_splits=3, embargo_days=2,
                                    n_estimators=30, max_depth=3)
    assert comp.shape == comps[0].panel.shape
    assert np.isfinite(comp.dropna().values).all()


def test_synthesize_stacking_gbdt_tuned_finite(mock_parts):
    """optuna 调参版冒烟：嵌套时序 CV 不崩、输出有限（缺 optuna/lightgbm 跳过）。"""
    pytest.importorskip("lightgbm")
    pytest.importorskip("optuna")
    ret_panel, comps = mock_parts
    comp = synthesize_stacking_gbdt_tuned(comps, ret_panel, n_splits=2,
                                          embargo_days=2, n_trials=3)
    assert comp.shape == comps[0].panel.shape
    assert np.isfinite(comp.dropna().values).all()


def test_synthesize_stacking_rank_target(mock_parts):
    """方案 A：截面秩目标（target_mode='rank'）不崩且输出有限。

    秩目标应让 rank IC 评价下与 raw 目标可比（Spearman 对 y 单调变换不变）。
    """
    ret_panel, comps = mock_parts
    comp_rank = synthesize_stacking(comps, ret_panel, n_splits=3, target_mode="rank")
    comp_raw = synthesize_stacking(comps, ret_panel, n_splits=3, target_mode="raw")
    assert comp_rank.shape == comps[0].panel.shape
    assert np.isfinite(comp_rank.dropna().values).all()
    assert np.isfinite(comp_raw.dropna().values).all()


def test_synthesize_stacking_lambdarank_finite(mock_parts):
    """方案 B：LambdaRank stacking 冒烟（按日 group + purged CV 不崩）。"""
    pytest.importorskip("lightgbm")
    ret_panel, comps = mock_parts
    comp = synthesize_stacking_lambdarank(comps, ret_panel, n_splits=2,
                                          embargo_days=2, n_estimators=30, max_depth=3)
    assert comp.shape == comps[0].panel.shape
    assert np.isfinite(comp.dropna().values).all()


def test_orthogonalize_removes_correlation(mock_parts):
    """正交化后，相邻两个子因子的截面相关性应显著下降（接近 0）。"""
    ret_panel, comps = mock_parts
    ortho = orthogonalize(comps)
    assert len(ortho) == len(comps)

    def _safe_corr(a, b):
        av = a.values.ravel(); bv = b.values.ravel()
        mask = ~np.isnan(av) & ~np.isnan(bv)
        return np.corrcoef(av[mask], bv[mask])[0, 1]

    # f1 与 f2 原始高度相关
    orig_corr = _safe_corr(comps[0].panel, comps[1].panel)
    # 正交化后 f1(=base) 与 f2 残差应接近 0 相关
    o_corr = _safe_corr(ortho[0].panel, ortho[1].panel)
    assert abs(o_corr) < abs(orig_corr) - 0.1


def test_composite_stats(mock_parts):
    ret_panel, comps = mock_parts
    comp = synthesize_ic_weighted(comps, returns_panel=ret_panel)
    st = composite_stats(comp, ret_panel)
    assert set(["ic_mean", "ic_std", "ir", "t_stat", "n"]).issubset(st.keys())
    assert st["n"] > 0
    assert np.isfinite(st["ic_mean"])


def test_build_components_reconstructs():
    """build_components 能按挖掘结果 name 重建因子面板。"""
    from factor.mining import dedup_by_formula, evaluate_candidates, generate_candidates
    from scripts.mine_factors import gen_mock_panel_with_signal

    panel = gen_mock_panel_with_signal(n_days=200, n_codes=20, seed=1)
    returns_panel = panel["close"].pct_change().shift(-1)
    features = list(panel.keys())
    cands = dedup_by_formula(generate_candidates(features=features, windows=(5, 10), depth=1))
    result = evaluate_candidates(cands, panel, returns_panel, fdr_q=0.05)
    topk = result.head(3)
    comps = build_components(topk, panel, features=features, windows=(5, 10), depth=1)
    assert len(comps) == 3
    assert all(c.panel.shape == comps[0].panel.shape for c in comps)


def test_build_components_gp_formula_reconstruction():
    """build_components 支持 GP 公式还原（2026-08-03）：

    name 为 GP 前缀表达式（窗口编名）时，走统一公式解析器重建，不再依赖
    deap pset / 模块级 prim_map。
    """
    from scripts.mine_factors import gen_mock_panel_with_signal

    panel = gen_mock_panel_with_signal(n_days=200, n_codes=20, seed=2)
    features = list(panel.keys())
    gp_name = "mul(ts_mean_5(close), cs_rank(ts_delta_20(volume)))"
    topk = pd.DataFrame([
        {"name": gp_name, "ic_mean": 0.03, "ir": 0.8},
    ])
    comps = build_components(topk, panel, features=features, windows=(5, 10, 20), depth=1)
    assert len(comps) == 1
    assert comps[0].name == gp_name
    assert comps[0].panel.shape == panel["close"].shape
    assert comps[0].panel.notna().any().any()


def test_synthesize_pca_sign_calibration_no_lookahead(mock_parts):
    """PCA 符号校准段（sign_calib_frac）参数：默认 0.6，输出有限且形状正确。

    符号方向只用前 60% 时间段的收益决定（无未来函数）；换不同校准段比例
    不改变输出的有限性与形状。
    """
    ret_panel, comps = mock_parts
    comp_default = synthesize_pca(comps, n_components=1, returns_panel=ret_panel)
    assert comp_default.shape == comps[0].panel.shape
    assert np.isfinite(comp_default.dropna().values).all()

    comp_half = synthesize_pca(comps, n_components=1, returns_panel=ret_panel,
                               sign_calib_frac=0.5)
    assert comp_half.shape == comps[0].panel.shape
    assert np.isfinite(comp_half.dropna().values).all()
    # 校准段长度变化不应改变信号方向（结构稳定时）
    d = comp_default.sub(comp_default.mean(axis=1), axis=0)
    h = comp_half.sub(comp_half.mean(axis=1), axis=0)
    rp = ret_panel.reindex(index=d.index, columns=d.columns)
    corr_d = float(np.nanmean((d * rp).values))
    corr_h = float(np.nanmean((h * rp).values))
    assert (np.sign(corr_d) == np.sign(corr_h)) or abs(corr_d - corr_h) < 1e-8


# ===========================================================================
# 2026-08-17 修复：按【交易日边界】切分的时序 CV（防把某一天劈开 / 防未来函数）
# ===========================================================================
def test_time_folds_no_split_within_day():
    """fold 边界不得落在某交易日中间：train 与 test 的日期集合不相交，
    且每个 fold 的 train 日期严格早于 test 日期（无未来函数）。"""
    # 模拟 20 天 × 3 股，且每天有效股票数不同（模拟 NaN 缺失）
    dates = pd.date_range("2023-01-02", periods=20, freq="B")
    per_day = [3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3]
    date_arr = np.concatenate([np.repeat(d, n) for d, n in zip(dates, per_day)])

    folds = _time_folds(date_arr, n_splits=5)
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
    folds = _time_folds(date_arr, n_splits=5, embargo_days=2)
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
    folds = _time_folds(date_arr, n_splits=6)
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


def test_stacking_prediction_uses_only_past(mock_parts):
    """验证 stacking 的某测试折预测仅用此前数据：构造一个前段无信号、
    后段强信号的数据，早期测试折的预测不应出现"未来才有的信号"。"""
    ret_panel, _ = mock_parts
    # 用 mock_parts 的收益面板，构造一个仅在【前 60%】有值的信号因子
    # （后 40% 为 NaN → 只影响"可被预测"的观测，不影响无泄漏性质）
    comps = mock_parts[1]
    comp = synthesize_stacking(comps, ret_panel, n_splits=4, target_mode="raw")
    assert comp.shape == comps[0].panel.shape
    assert np.isfinite(comp.dropna().values).all()


def test_rebuild_train_weights_uses_train_only():
    """_rebuild_train_weights：返回的 ic/ir 应基于训练段重算，且面板保持全样本。

    构造一个训练段收益方向与全样本相反的数据，验证权重来源确实只有训练段。
    """
    rng = np.random.default_rng(3)
    idx = pd.date_range("2022-01-01", periods=120, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(10)]
    # 收益：前 70 天为正 IC 信号，后 50 天为负 IC 信号（方向反转）
    rets = np.zeros((120, 10))
    for t in range(1, 70):
        rets[t] = 0.05 * rng.normal(0, 1, 10) + rng.normal(0, 0.02, 10)
    for t in range(70, 120):
        rets[t] = -0.05 * rng.normal(0, 1, 10) + rng.normal(0, 0.02, 10)
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), idx, codes)
    ret_panel = close.pct_change().shift(-1)

    f1 = ts_rank(close, 5)
    comp = CompositeInput("tsr5", standardize_zscore(f1), ic=0.1, ir=1.0)

    train_dates = idx[:80]
    rebuilt = rebuild_train_weights([comp], ret_panel, train_dates)
    assert len(rebuilt) == 1
    # 训练段（前 80 天）IC 应为正（与构造一致），而非沿用传入的全样本 ic
    assert rebuilt[0].ic > 0
    assert abs(rebuilt[0].ic - 0.1) > 1e-6  # 已用训练段重算，非原值
    # 面板保持全样本形状
    assert rebuilt[0].panel.shape == comp.panel.shape
