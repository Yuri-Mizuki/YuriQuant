"""
统一公式解析器测试
==================

覆盖：
- exhaustive 语法（ts_mean(feat,5) / 嵌套 / ts_corr 三参）还原与原始 build 闭包一致
- GP 语法（ts_mean_5(feat) / ts_corr_20 双参 / 深层嵌套）还原与 eval_tree 一致
- 未知算子报错 / 参数数校验
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.formula import formula_builder, parse_formula


@pytest.fixture
def mock_panel():
    rng = np.random.default_rng(5)
    n_days, n_codes = 120, 15
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"c{i:02d}" for i in range(n_codes)]
    rets = pd.DataFrame(rng.normal(0.0003, 0.01, (n_days, n_codes)), idx, codes)
    close = 10.0 * np.exp(rets.cumsum())
    volume = pd.DataFrame(rng.lognormal(12, 0.5, (n_days, n_codes)), idx, codes)
    amount = volume * close
    return {"close": close, "volume": volume, "amount": amount}


def _rebuild_via_generate_candidates(formula, panel, windows=(5, 20)):
    """用 generate_candidates 的 build 闭包作为基准（exhaustive 命名）。"""
    from factor.mining import dedup_by_formula, generate_candidates
    cands = dedup_by_formula(generate_candidates(
        features=list(panel.keys()), windows=windows, depth=2))
    by_name = {c.name: c for c in cands}
    assert formula in by_name, f"候选空间应包含 {formula}"
    return by_name[formula].build(panel)


def test_formula_exhaustive_ts_unary(mock_panel):
    f = "ts_delta(amount,20)"
    ref = _rebuild_via_generate_candidates(f, mock_panel)
    got = formula_builder(f, features=list(mock_panel.keys()))(mock_panel)
    assert np.allclose(got.values, ref.values, equal_nan=True)


def test_formula_exhaustive_nested_cs_rank(mock_panel):
    f = "cs_rank(ts_mean(close,5))"
    ref = _rebuild_via_generate_candidates(f, mock_panel)
    got = formula_builder(f, features=list(mock_panel.keys()))(mock_panel)
    assert np.allclose(got.values, ref.values, equal_nan=True)


def test_formula_exhaustive_div_nested(mock_panel):
    f = "div(close,ts_mean(close,20))"
    ref = _rebuild_via_generate_candidates(f, mock_panel)
    got = formula_builder(f, features=list(mock_panel.keys()))(mock_panel)
    assert np.allclose(got.values, ref.values, equal_nan=True)


def test_formula_exhaustive_ts_corr_3arg(mock_panel):
    f = "ts_corr(close,volume,20)"
    ref = _rebuild_via_generate_candidates(f, mock_panel)
    got = formula_builder(f, features=list(mock_panel.keys()))(mock_panel)
    assert np.allclose(got.values, ref.values, equal_nan=True)


def test_formula_gp_windowed_name(mock_panel):
    """GP 窗口编名语法与显式窗口参数语法等价。"""
    gp_f = "ts_mean_5(close)"
    ex_f = "ts_mean(close,5)"
    got_gp = formula_builder(gp_f, features=list(mock_panel.keys()))(mock_panel)
    got_ex = formula_builder(ex_f, features=list(mock_panel.keys()))(mock_panel)
    assert np.allclose(got_gp.values, got_ex.values, equal_nan=True)


def test_formula_gp_deep_nested_matches_eval_tree(mock_panel):
    """GP 深层嵌套公式还原与 eval_tree（DEAP 原语求值）一致。"""
    pytest.importorskip("deap")
    from deap import gp as deap_gp
    from factor.genetic_mining import build_primitive_set, eval_tree

    formula = "mul(ts_mean_5(close), cs_rank(ts_delta_20(volume)))"
    pset, prim_map = build_primitive_set(["close", "volume", "amount"], (5, 20))
    tree = deap_gp.PrimitiveTree.from_string(formula, pset)
    ref = eval_tree(tree, mock_panel, prim_map)
    got = formula_builder(formula, features=list(mock_panel.keys()))(mock_panel)
    assert np.allclose(got.values, ref.values, equal_nan=True)


def test_formula_gp_ts_corr_2arg(mock_panel):
    """GP 风格 ts_corr_20(close, volume)：窗口在算子名，只传 2 个面板参数。"""
    formula = "ts_corr_20(close,volume)"
    got = formula_builder(formula, features=list(mock_panel.keys()))(mock_panel)
    assert got.shape == mock_panel["close"].shape
    assert got.notna().any().any()


def test_formula_unknown_operator_raises():
    with pytest.raises(ValueError, match="未知算子"):
        parse_formula("ts_meean(close,5)", features=["close"])


def test_formula_wrong_arg_count_raises():
    # 少了窗口参数且算子名无窗口后缀 → 缺窗口
    with pytest.raises(ValueError, match="窗口"):
        parse_formula("ts_mean(close)", features=["close"])
    # 参数数完全不符（ts_corr 需要 2 面板 + 1 窗口；只给 1 个）
    with pytest.raises(ValueError, match="参数数不符"):
        parse_formula("ts_corr(close)", features=["close", "volume"])


def test_formula_feature_fallback_without_features(mock_panel):
    """features 未传时未知裸 token 宽容视为特征名（panel 有该键即可）。"""
    got = formula_builder("reverse(close)")(mock_panel)
    assert np.allclose(got.values, -mock_panel["close"].values, equal_nan=True)
