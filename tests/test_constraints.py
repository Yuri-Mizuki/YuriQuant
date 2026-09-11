"""
组合构建约束与面板级权重算子测试（2026-09-11 口径统一第三批）
================================================================

`optimize/portfolio.py` 的信号构建与工程约束被判定为**组合构建**（不需要风险
模型），已下沉为 `strategy.constraints`。本模块锁死三件事：

1. 面板级算子的基本性质（行和 / 非负 / 上下限 / 换手上限）；
2. 与 `strategy.examples` 截面级策略类的关系——`equal_topk` 的 tie-break 是
   **按列序确定性**的（characterization，刻意与 `TopKLongOnly` 的
   `sort_values()` 不同）；
3. `optimize.portfolio.optimize_weights` 是**薄门面**：行为必须逐位等于
   `apply_constraints(build_signal_weights(...))`，且不得重新定义任何算子。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.constraints import (
    apply_bounds,
    apply_constraints,
    apply_turnover,
    build_signal_weights,
    neutralize_industry,
)


def _panel(n_days=30, n_codes=12, seed=0, ties=False, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    codes = [f"C{i:02d}" for i in range(n_codes)]
    v = rng.normal(0, 1, (n_days, n_codes))
    if ties:
        v = np.round(v, 0)
    df = pd.DataFrame(v, index=idx, columns=codes)
    if nan_frac:
        df = df.mask(rng.random((n_days, n_codes)) < nan_frac)
    return df


def _rowwise(panel, strat):
    rows = {d: strat.get_weights(row) for d, row in panel.iterrows()}
    return pd.DataFrame(rows).T.reindex(columns=panel.columns).fillna(0.0)


# ===========================================================================
# 1) 信号构建
# ===========================================================================
def test_factor_weighted_properties():
    p = _panel(nan_frac=0.2)
    w = build_signal_weights(p, "factor_weighted")
    sums = w.sum(axis=1)
    assert (sums[sums > 0] - 1.0).abs().max() < 1e-9
    assert (w.values >= 0).all()
    # 因子为 NaN 的格子权重为 0
    assert (w.where(p.notna(), 0.0) - w).abs().max().max() == 0.0


def test_factor_weighted_is_rank_pct_normalized():
    """口径：全池 rank(pct) 后再除以行和（不是原始因子值加权）。"""
    p = _panel(seed=3)
    w = build_signal_weights(p, "factor_weighted")
    expect = p.rank(axis=1, pct=True)
    expect = expect.div(expect.sum(axis=1), axis=0).fillna(0.0)
    assert np.allclose(w.values, expect.values, atol=1e-15)


def test_equal_topk_properties():
    p = _panel(nan_frac=0.15)
    w = build_signal_weights(p, "equal_topk", k=4)
    sums = w.sum(axis=1)
    assert (sums[sums > 0] - 1.0).abs().max() < 1e-9
    row = w.iloc[3]
    assert (row > 0).sum() == 4
    assert np.allclose(row[row > 0].values, 0.25)


def test_build_signal_weights_unknown_method():
    p = _panel()
    with pytest.raises(ValueError, match="未知组合优化方法"):
        build_signal_weights(p, "nope")


def test_equal_topk_requires_positive_k():
    p = _panel()
    with pytest.raises(ValueError, match="k > 0"):
        build_signal_weights(p, "equal_topk", k=0)


# ===========================================================================
# 2) 与截面级策略类的关系（characterization）
# ===========================================================================
def test_equal_topk_matches_topk_longonly_without_ties():
    """无 tie 时与 ``TopKLongOnly(k, "equal")`` 逐位一致——差异只在 tie 边界。"""
    from strategy.examples import TopKLongOnly

    p = _panel(seed=9)                      # 连续随机值，无重复
    a = build_signal_weights(p, "equal_topk", k=5)
    b = _rowwise(p, TopKLongOnly(5, "equal"))
    assert np.allclose(a.values, b.values, atol=1e-15)


def test_equal_topk_tie_break_is_deterministic_by_column_order():
    """**characterization**：tie 时按**列序**取前 k（确定性、可复现）。

    2026-09-11 起 ``strategy.examples`` 的 top-k 已统一到与本模块**同一** tie
    语义（``rank(ascending=False, method="first")``，并列时列序靠前者优先），
    ``TopKLongOnly`` / ``TopFracLongOnly`` / ``TopKLongShort`` 不再用不稳定的
    ``sort_values()``。这里钉住本模块自身的行为；策略类的确定性覆盖见
    ``tests/test_strategy_tie.py``。
    """
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    codes = [f"C{i:02d}" for i in range(10)]
    p = pd.DataFrame(1.0, index=idx, columns=codes)   # 全同值

    w = build_signal_weights(p, "equal_topk", k=3)
    picked = list(w.columns[w.iloc[0] > 0])
    assert picked == ["C00", "C01", "C02"]            # 列序前 3
    assert w.equals(build_signal_weights(p, "equal_topk", k=3))   # 幂等

    # 混合值：最高值优先，其余名额按列序补足（输出列序仍是原始列序）
    p2 = p.copy()
    p2.loc[:, "C07"] = 5.0
    w2 = build_signal_weights(p2, "equal_topk", k=3)
    picked2 = list(w2.columns[w2.iloc[0] > 0])
    assert picked2 == ["C00", "C01", "C07"]           # 含 C07（最高）+ 列序前两个


# ===========================================================================
# 3) 约束算子
# ===========================================================================
def test_neutralize_industry_projects_to_equal_target():
    p = _panel(seed=5)
    codes = list(p.columns)
    ind = {c: f"ind{i % 3}" for i, c in enumerate(codes)}
    w = neutralize_industry(build_signal_weights(p, "factor_weighted"), ind)
    mask = w.sum(axis=1) > 0
    for g in ("ind0", "ind1", "ind2"):
        cols = [c for c in codes if ind[c] == g]
        assert np.allclose(w.loc[mask, cols].sum(axis=1).values, 1 / 3, atol=1e-9)


def test_neutralize_industry_leaves_uncovered_codes_untouched():
    p = _panel(seed=6)
    codes = list(p.columns)
    w0 = build_signal_weights(p, "factor_weighted")
    w1 = neutralize_industry(w0, {codes[0]: "only"})
    assert np.allclose(w1[codes[1]].values, w0[codes[1]].values, atol=1e-15)


def test_neutralize_industry_noop_without_map():
    p = _panel(seed=7)
    w0 = build_signal_weights(p, "factor_weighted")
    assert neutralize_industry(w0, pd.Series(dtype=object)).equals(w0)


def test_apply_bounds_clip_and_floor():
    p = _panel(seed=8, nan_frac=0.1)
    w = build_signal_weights(p, "factor_weighted")

    capped = apply_bounds(w, max_weight=0.05)
    assert (capped.values <= 0.05 + 1e-12).all()
    sums = capped.sum(axis=1)
    assert (sums[sums > 0] <= 1.0 + 1e-12).all()      # 超额不回补 → 现金仓位

    floored = apply_bounds(w, min_weight=0.09)
    nz = floored[floored > 0].stack().dropna()
    assert (nz >= 0.09 - 1e-12).all()

    assert apply_bounds(w).equals(w)                  # 无参数 → 恒等


def test_apply_turnover_caps_and_shrinks():
    p = _panel(seed=10)
    codes = list(p.columns)
    w = build_signal_weights(p, "factor_weighted")
    prev = pd.Series(0.0, index=codes)
    out = apply_turnover(w, prev, 0.1)
    turn = (out - prev).abs().sum(axis=1) * 0.5
    assert (turn <= 0.1 + 1e-12).all()
    turn_raw = (w - prev).abs().sum(axis=1) * 0.5
    assert turn.max() < turn_raw.max()                # 确实被压低
    # max_turnover 非正 → 恒等
    assert apply_turnover(w, prev, 0.0).equals(w)


def test_apply_constraints_matches_sequential_composition():
    """组合入口 == 逐步调用（顺序固定：行业中性 → 上下限 → 换手）。"""
    p = _panel(seed=11, nan_frac=0.15)
    codes = list(p.columns)
    ind = {c: f"ind{i % 3}" for i, c in enumerate(codes)}
    prev = pd.Series(0.0, index=codes)

    w = build_signal_weights(p, "factor_weighted")
    step = neutralize_industry(w, ind)
    step = apply_bounds(step, max_weight=0.12, min_weight=0.02)
    step = apply_turnover(step, prev, 0.3)

    combo = apply_constraints(w, industry_map=ind, max_weight=0.12,
                              min_weight=0.02, prev_weights=prev, max_turnover=0.3)
    assert np.allclose(step.values, combo.values, atol=1e-15)


def test_apply_constraints_without_args_is_identity():
    p = _panel(seed=12)
    w = build_signal_weights(p, "factor_weighted")
    assert apply_constraints(w).equals(w)


# ===========================================================================
# 4) optimize_weights 是薄门面
# ===========================================================================
def test_optimize_weights_is_exactly_facade_composition():
    """门面输出必须逐位等于 `apply_constraints(build_signal_weights(...))`。"""
    from optimize.portfolio import optimize_weights

    p = _panel(seed=13, nan_frac=0.1)
    codes = list(p.columns)
    ind = {c: f"ind{i % 3}" for i, c in enumerate(codes)}
    prev = pd.Series(0.0, index=codes)
    kw = dict(industry_map=ind, max_weight=0.15, min_weight=0.01,
              prev_weights=prev, max_turnover=0.25)

    for method_kw in (dict(method="factor_weighted"), dict(method="equal_topk", k=4)):
        a = optimize_weights(p, **method_kw, **kw)
        b = apply_constraints(build_signal_weights(p, **method_kw), **kw)
        assert np.allclose(a.values, b.values, atol=1e-15)


def test_optimize_weights_keeps_error_semantics():
    from optimize.portfolio import optimize_weights

    with pytest.raises(ValueError, match="未知组合优化方法"):
        optimize_weights(_panel(), method="nope")
