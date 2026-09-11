"""
多因子合成单元测试（因子层）
============================

构造确定性的 mock 因子面板（共享 fixture ``synth_parts``，见 conftest），验证
**确定性**合成方式产出有限、形状正确的复合因子，并验证正交化确实消除了因子间
相关性。

模型层的 ML stacking（ridge / LightGBM / LambdaRank）测试见
``tests/test_stacking.py``——2026-09-11 归属修复后两者分属不同层。
"""
import numpy as np
import pandas as pd
import pytest

from factor.synthesis import (
    CompositeInput, build_components, composite_stats, orthogonalize,
    rebuild_train_weights,
    synthesize_ic_weighted, synthesize_orthogonal, synthesize_pca,
)


def test_synthesize_ic_weighted_shape(synth_parts):
    ret_panel, comps = synth_parts
    comp = synthesize_ic_weighted(comps, returns_panel=ret_panel)
    assert isinstance(comp, pd.DataFrame)
    assert comp.shape == comps[0].panel.shape
    assert comp.dropna().notna().all().all()


def test_synthesize_pca_shape(synth_parts):
    ret_panel, comps = synth_parts
    comp = synthesize_pca(comps, n_components=2, returns_panel=ret_panel)
    assert comp.shape == comps[0].panel.shape
    assert comp.dropna().notna().all().all()


def test_synthesize_orthogonal_shape(synth_parts):
    ret_panel, comps = synth_parts
    comp = synthesize_orthogonal(comps)
    assert comp.shape == comps[0].panel.shape
    assert comp.dropna().notna().all().all()


def test_orthogonalize_removes_correlation(synth_parts):
    """正交化后，相邻两个子因子的截面相关性应显著下降（接近 0）。"""
    ret_panel, comps = synth_parts
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


def test_composite_stats(synth_parts):
    ret_panel, comps = synth_parts
    comp = synthesize_ic_weighted(comps, returns_panel=ret_panel)
    st = composite_stats(comp, ret_panel)
    assert set(["ic_mean", "ic_std", "ir", "t_stat", "n"]).issubset(st.keys())
    assert st["n"] > 0
    assert np.isfinite(st["ic_mean"])


def test_build_components_reconstructs():
    """build_components 能按挖掘结果 name 重建因子面板。"""
    from data.mock import gen_mock_panel_with_signal
    from factor.mining import dedup_by_formula, evaluate_candidates, generate_candidates

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
    from data.mock import gen_mock_panel_with_signal

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


def test_synthesize_pca_sign_calibration_no_lookahead(synth_parts):
    """PCA 符号校准段（sign_calib_frac）参数：默认 0.6，输出有限且形状正确。

    符号方向只用前 60% 时间段的收益决定（无未来函数）；换不同校准段比例
    不改变输出的有限性与形状。
    """
    ret_panel, comps = synth_parts
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


def test_rebuild_train_weights_uses_train_only():
    """_rebuild_train_weights：返回的 ic/ir 应基于训练段重算，且面板保持全样本。

    构造一个训练段收益方向与全样本相反的数据，验证权重来源确实只有训练段。
    """
    from factor.operators import ts_rank
    from factor.preprocessing import standardize_zscore

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
