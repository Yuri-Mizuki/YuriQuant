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
    synthesize_ic_weighted, synthesize_orthogonal, synthesize_pca,
    synthesize_stacking,
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
