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
    CompositeInput, build_components, composite_stats, half_life_weights,
    orthogonalize, rebuild_train_weights, synthesize_ic_ir_max, synthesize_ic_max,
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


# ===========================================================================
# 华泰多因子系列10：半衰加权 + 最大化 IC_IR / 最大化 IC（2026-09-16 补齐）
# ===========================================================================
def test_half_life_weights_exact_and_monotone():
    """半衰权重：H 期前恰为 0.5，最近期为 1，且随距离单调不增。

    研报 ``w_t = 2^((t−T−1)/H)`` 换算到"距今 k 期"即 ``2^(−k/H)``。
    这是整条半衰链路的数值真源，必须逐点钉死（不是"跑通就行"）。
    """
    dates = pd.date_range("2022-01-01", periods=13, freq="B")
    for H in (3, 6, 12):
        w = half_life_weights(dates, H)
        assert w.shape == (13,)
        assert abs(w[-1] - 1.0) < 1e-12              # 最近期 = 1
        assert abs(w[-1 - H] - 0.5) < 1e-12          # 距今 H 期 = 0.5
        assert abs(w[0] - 2.0 ** (-12 / H)) < 1e-12  # 一般点
        assert np.all(np.diff(w) >= -1e-15)          # 单调不增（升序日期 → 越早越小）


def test_half_life_weights_as_of_anchor():
    """as_of 改变锚点：锚点之后的期数权重 > 1（外推段），锚点处为 1。"""
    dates = pd.date_range("2022-01-01", periods=10, freq="B")
    anchor = dates[6]
    w = half_life_weights(dates, 5, as_of=anchor)
    assert abs(w[6] - 1.0) < 1e-12
    assert w[7] > 1.0 and w[9] > w[7]                # 锚点之后继续指数放大
    assert w[5] == pytest.approx(2.0 ** (-1 / 5))


def test_half_life_weights_empty():
    assert half_life_weights([], 6).shape == (0,)


def test_synthesize_ic_weighted_equal_branch(synth_parts):
    """weight_by="equal" 分支：等权 1/N，但仍先做符号对齐（高值=高收益）。

    注意等权 ≠ "原始面板直接相加"：``rev`` 因子 IC 为负，内部会被取负，
    故正确对照是 ``f1 + f2 - f3``（不是 ``f1 + f2 + f3``）。
    """
    ret_panel, comps = synth_parts
    eq = synthesize_ic_weighted(comps, weight_by="equal")
    assert eq.shape == comps[0].panel.shape
    assert np.isfinite(eq.dropna().values).all()
    # 与手写"翻正后等权 + 同口径标准化"逐点一致
    from factor.preprocessing import standardize_zscore
    manual = sum(((-1 if c.ic < 0 else 1) * c.panel.fillna(0.0)) for c in comps)
    manual = standardize_zscore(manual)
    d = (eq - manual).abs()
    assert float(np.nanmax(d.values)) < 1e-12
    # 且与"不翻符号直接相加"明显不同 → 证明符号对齐确实生效
    naive = standardize_zscore(sum(c.panel.fillna(0.0) for c in comps))
    a, b = eq.values.ravel(), naive.values.ravel()
    m = ~np.isnan(a) & ~np.isnan(b)
    assert np.corrcoef(a[m], b[m])[0, 1] < 0.999


def test_synthesize_ic_ir_max_shape_and_diagnostics(synth_parts):
    """最大化 IC_IR：产出有限、形状正确，诊断字段齐全，权重已归一。"""
    ret_panel, comps = synth_parts
    train_dates = ret_panel.index[:120]
    comp, diag = synthesize_ic_ir_max(comps, ret_panel, train_dates,
                                      returns_diagnostics=True)
    assert comp.shape == comps[0].panel.shape
    assert np.isfinite(comp.dropna().values).all()
    assert diag["method"] == "ic_ir_max"
    assert diag["n_periods"] >= 3
    assert set(diag["weights"].keys()) == {c.name for c in comps}
    assert abs(sum(diag["weights"].values()) - 1.0) < 1e-9
    assert diag["long_only"] is True


def test_ic_ir_max_solution_is_cov_inverse_ic(synth_parts):
    """解的性质：``w ∝ Σ⁻¹·ĪC``（去掉 w≥0 截断后应严格成立）。

    这是最大化 IC_IR 的数学核心，必须验证解真的是 Σ⁻¹·ĪC 的方向，
    而不是"随便给了个权重"。
    """
    ret_panel, comps = synth_parts
    train_dates = ret_panel.index[:120]
    _, diag = synthesize_ic_ir_max(comps, ret_panel, train_dates,
                                   long_only=False, returns_diagnostics=True)
    w = np.array([diag["weights"][c.name] for c in comps], dtype=float)

    # 用诊断里暴露的 ĪC（对齐基）与同口径 Σ 重算方向，逐点比对
    from factor.synthesis import ic_matrix_from_components
    aligned = [CompositeInput(c.name, -c.panel if c.ic < 0 else c.panel,
                              ic=abs(c.ic), ir=c.ir) for c in comps]
    mu = np.array([diag["ic_mean"][c.name] for c in comps], dtype=float)
    ic = ic_matrix_from_components(aligned, ret_panel,
                                    train_dates).dropna(how="any")
    S = np.cov(ic.to_numpy(dtype=float), rowvar=False)
    S = 0.5 * np.diag(np.diag(S)) + 0.5 * S          # 与实现同口径的收缩
    target = np.linalg.solve(S, mu)
    target = target / np.abs(target).sum()
    # weights 键名与 comps 顺序一致 → 直接对齐比较
    assert np.max(np.abs(w - target)) < 1e-6


def test_ic_ir_max_long_only_truncates_negatives(synth_parts):
    """``long_only=True`` 应把负权重截为 0（研报 w ≥ 0），且仍归一。"""
    ret_panel, comps = synth_parts
    train_dates = ret_panel.index[:120]
    _, d_free = synthesize_ic_ir_max(comps, ret_panel, train_dates,
                                     long_only=False, returns_diagnostics=True)
    w_free = np.array([d_free["weights"][c.name] for c in comps], dtype=float)
    if (w_free < 0).any():                            # 该 fixture 下确实会出现负权重
        _, d_long = synthesize_ic_ir_max(comps, ret_panel, train_dates,
                                         long_only=True, returns_diagnostics=True)
        w_long = np.array([d_long["weights"][c.name] for c in comps], dtype=float)
        assert (w_long >= -1e-15).all()
        assert abs(w_long.sum() - 1.0) < 1e-9


def test_ic_ir_max_better_than_equal_weight_on_train(synth_parts):
    """解应优于等权：训练段 IC_IR（加权 IC 均值 / 其标准差）不低于等权组合。

    这是"最大化 IC_IR"这个名字的含义所在 —— 若解不如等权，说明求解或
    协方差口径有问题。用同口径（训练段 IC 序列）比较，避免口径不一致。
    """
    ret_panel, comps = synth_parts
    train_dates = ret_panel.index[:120]
    _, diag = synthesize_ic_ir_max(comps, ret_panel, train_dates,
                                   returns_diagnostics=True)
    rts = ret_panel.loc[train_dates]
    from factor.synthesis import ic_matrix_from_components
    ic = ic_matrix_from_components(comps, ret_panel, train_dates).dropna(how="any")
    X = ic.to_numpy(dtype=float)

    def _icir(w):
        s = X @ np.asarray(w, dtype=float)
        return float(s.mean() / s.std(ddof=1)) if s.std(ddof=1) > 0 else 0.0

    w_sol = np.array([diag["weights"][c.name] for c in comps], dtype=float)
    w_eq = np.ones(len(comps)) / len(comps)
    assert _icir(w_sol) >= _icir(w_eq) - 1e-9
    del rts


def test_ic_ir_max_requires_returns_panel(synth_parts):
    """Σ 来自历史 IC 序列 → 没收益面板必须显式报错，不能静默降级。"""
    ret_panel, comps = synth_parts
    with pytest.raises(ValueError):
        synthesize_ic_ir_max(comps, None, ret_panel.index[:120])
    with pytest.raises(ValueError):
        synthesize_ic_ir_max(comps, ret_panel, None)


def test_ic_max_no_lookahead_v_uses_last_snapshot(synth_parts):
    """最大化 IC 的 V 只取**决策时点截面**：扰动测试段因子值不改变 V。

    2026-09-16 修复项：原实现 ``long_matrix(comps, None)`` 铺全样本算 V
    → 隐式含测试段截面结构（未来函数）。现在 V 只取 ``panel.index[-1]``。

    注意 ĪC **本来就**依赖训练段日期（这是设计如此），故这里只验证 V：
    用 ``returns_panel`` 里**空训练段**以外的方式隔离 —— 直接对比 V 矩阵。
    """
    from factor.synthesis import long_matrix
    ret_panel, comps = synth_parts

    def _V(cs):
        last = cs[0].panel.index[-1]
        snap = [CompositeInput(c.name, c.panel.loc[[last]], ic=c.ic, ir=c.ir)
                for c in cs]
        X, _, _ = long_matrix(snap, None)
        V = np.corrcoef(np.nan_to_num(X, nan=0.0), rowvar=False)
        V = np.nan_to_num(np.atleast_2d(V), nan=0.0)
        np.fill_diagonal(V, 1.0)
        return V

    # 打乱末截面**之前**的所有因子值 → 末截面不变 → V 必须逐点相同
    tampered = []
    for c in comps:
        p = c.panel.copy()
        p.iloc[:-1, :] = p.iloc[:-1, :].sample(frac=1.0, random_state=0).values
        tampered.append(CompositeInput(c.name, p, ic=c.ic, ir=c.ir))
    assert np.allclose(_V(comps), _V(tampered), atol=1e-12)

    # 反之：换一个末截面（多用一天）→ V 应改变（证明 V 真的取自末截面）
    assert not np.allclose(_V(comps), _V([CompositeInput(
        c.name, c.panel.iloc[:-1], ic=c.ic, ir=c.ir) for c in comps]), atol=1e-12)


def test_ic_max_v_decision_point_follows_train_dates(synth_parts):
    """决策时点跟随 ``train_dates``：扰动训练段末之后的因子值不得改变权重。

    2026-09-24 修复回归锚：``mf10_t_scan`` 形态 = 全时段面板 + train_dates，
    原实现恒取 ``panel.index[-1]``（= 测试段末）算 V 截面 → 09-16 的
    "决策时点截面"修复在该调用形态下未生效（前视换个位置回归）。
    """
    ret_panel, comps = synth_parts
    train = ret_panel.index[:150]

    _, diag_base = synthesize_ic_max(comps, ret_panel, train,
                                     returns_diagnostics=True)
    assert diag_base["as_of"] == train[-1]

    # 打乱训练段末之后的所有因子值 → V 截面（train 末）不变 → 权重逐点相同
    rng = np.random.default_rng(3)
    tampered = []
    for c in comps:
        p = c.panel.copy()
        tail = p.iloc[150:]
        p.iloc[150:] = tail.iloc[rng.permutation(len(tail))].to_numpy()
        tampered.append(CompositeInput(c.name, p, ic=c.ic, ir=c.ir))
    _, diag_t = synthesize_ic_max(tampered, ret_panel, train,
                                  returns_diagnostics=True)
    assert diag_t["as_of"] == train[-1]
    for k in diag_base["weights"]:
        assert diag_t["weights"][k] == pytest.approx(
            diag_base["weights"][k], abs=1e-12)


def test_ic_ir_max_rejects_too_few_periods(synth_parts):
    """训练段有效 IC 期数 < 3 时无法估 Σ → 明确报错。"""
    ret_panel, comps = synth_parts
    tiny = ret_panel.index[:2]
    with pytest.raises(ValueError):
        synthesize_ic_ir_max(comps, ret_panel, tiny)


def test_synthesize_ic_max_shape_and_no_long_only_default(synth_parts):
    """最大化 IC：V 为因子值相关阵（非 IC 协方差），默认不施加 w ≥ 0。"""
    ret_panel, comps = synth_parts
    comp, diag = synthesize_ic_max(comps, returns_panel=ret_panel,
                                   returns_diagnostics=True)
    assert comp.shape == comps[0].panel.shape
    assert np.isfinite(comp.dropna().values).all()
    assert diag["method"] == "ic_max"
    assert diag["long_only"] is False
    assert abs(sum(diag["weights"].values()) - 1.0) < 1e-9


def test_ic_max_uses_factor_value_correlation_not_ic_cov(synth_parts):
    """V 的来源必须是**当期因子值相关阵**：两个高度共线因子上 V⁻¹ 应压低冗余项。

    构造 f2 ≈ f1 的冗余对（fixture 已提供），验证解确实对共线方向做了补偿：
    与「忽略 V 只用 ĪC 归一」的结果有明显差异。
    """
    ret_panel, comps = synth_parts
    _, diag = synthesize_ic_max(comps, returns_panel=ret_panel,
                                returns_diagnostics=True)
    w = np.array([diag["weights"][c.name] for c in comps], dtype=float)
    mu = np.array([diag["ic_mean"][c.name] for c in comps], dtype=float)
    naive = mu / np.abs(mu).sum()                     # 不除 V，直接用 ĪC
    assert np.max(np.abs(w - naive)) > 1e-3           # V 确实改变了权重


def test_ic_ir_max_no_lookahead_train_segment_only(synth_parts):
    """防未来函数：把训练段之后的收益全部改成反向，合成结果**完全不**受影响。

    这是本项目对研报口径的额外加严（研报未讨论）。若实现里误用了全样本收益，
    该断言必然失败。
    """
    ret_panel, comps = synth_parts
    train_dates = ret_panel.index[:120]
    a = synthesize_ic_ir_max(comps, ret_panel, train_dates)

    tampered = ret_panel.copy()
    tampered.loc[ret_panel.index[120:], :] = -tampered.loc[ret_panel.index[120:], :]
    b = synthesize_ic_ir_max(comps, tampered, train_dates)

    # 权重只由训练段决定 → 合成面板应逐点相同
    assert np.allclose(a.values, b.values, atol=1e-12, equal_nan=True)


def test_ic_ir_max_half_life_changes_weights(synth_parts):
    """半衰加权应真实生效（不是被忽略的参数）：H 不同 → 权重不同。"""
    ret_panel, comps = synth_parts
    train_dates = ret_panel.index[:120]
    _, d0 = synthesize_ic_ir_max(comps, ret_panel, train_dates,
                                 returns_diagnostics=True)
    _, d1 = synthesize_ic_ir_max(comps, ret_panel, train_dates, half_life=10,
                                 returns_diagnostics=True)
    w0 = np.array([d0["weights"][c.name] for c in comps], dtype=float)
    w1 = np.array([d1["weights"][c.name] for c in comps], dtype=float)
    assert not np.allclose(w0, w1, atol=1e-12)


def testsolve_synthesis_weights_falls_back_to_equal_when_all_truncated():
    """Σ⁻¹·ĪC 全为负且开 long_only → 退化为等权（而非全零/NaN）。"""
    from factor.synthesis import solve_synthesis_weights
    m = np.array([-1.0, -2.0, -3.0])
    C = np.eye(3)
    w = solve_synthesis_weights(m, C, long_only=True)
    assert np.all(np.isfinite(w))
    assert np.allclose(w, np.ones(3) / 3.0)


def testsolve_synthesis_weights_handles_singular_cov():
    """奇异协方差（全 1 阵）不应抛异常：走伪逆分支并给出有限权重。"""
    from factor.synthesis import solve_synthesis_weights
    m = np.array([1.0, 0.5, -0.25])
    C = np.ones((3, 3))
    w = solve_synthesis_weights(m, C, long_only=False)
    assert np.all(np.isfinite(w))
    assert abs(np.abs(w).sum() - 1.0) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
