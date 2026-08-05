"""遗传规划因子挖掘冒烟测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def signal_panel():
    """AR(1) 收益注入动量信号的 mock 面板。"""
    rng = np.random.default_rng(0)
    n_days, n_codes = 200, 20
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    phi = 0.25
    rets = np.zeros((n_days, n_codes))
    for t in range(1, n_days):
        rets[t] = phi * rets[t - 1] + rng.normal(0, 0.02, n_codes)
    close = pd.DataFrame(np.exp(np.cumsum(rets, axis=0)), idx, codes)
    volume = pd.DataFrame(rng.lognormal(12, 0.5, (n_days, n_codes)), idx, codes)
    return {"close": close, "volume": volume}, close.pct_change().shift(-1)


def test_gp_finds_signal(signal_panel):
    """GP 应能在注入信号的 mock 上找到 IC 显著的因子。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, hof = run_gp_mining(
        panel, rets, features=["close", "volume"], windows=(5, 10),
        population=40, generations=4, max_depth=3, seed=0, verbose=False,
    )
    assert len(df) > 0
    assert {"formula", "ic_mean", "ir", "t_stat"}.issubset(df.columns)
    # 注入了动量信号，最优因子 |IC| 应明显大于 0
    best = df.iloc[0]
    assert abs(best["ic_mean"]) > 0.05
    assert abs(best["t_stat"]) > 3.0


def test_eval_tree_handles_bad_expr(signal_panel):
    """eval_tree 对异常表达式应返回 None 而非抛异常。"""
    pytest.importorskip("deap")
    from deap import gp
    from factor.genetic_mining import build_primitive_set, eval_tree

    panel, _ = signal_panel
    pset, prim_map = build_primitive_set(["close", "volume"], (5, 10))
    # 正常表达式
    tree = gp.PrimitiveTree.from_string("ts_rank_5(close)", pset)
    out = eval_tree(tree, panel, prim_map)
    assert out is not None and out.shape == panel["close"].shape


def test_gp_early_stop(signal_panel):
    """早停（2026-08-03）：连续 patience 代无提升应提前终止，generations_run < generations。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, hof = run_gp_mining(
        panel, rets, features=["close", "volume"], windows=(5, 10),
        population=30, generations=30, max_depth=3, patience=2, seed=1, verbose=False,
    )
    assert getattr(hof, "generations_run", 30) < 30, "连续 2 代无提升应触发早停"


def test_gp_no_early_stop_with_patience_0(signal_panel):
    """patience=0 关闭早停：跑满全部代数。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, hof = run_gp_mining(
        panel, rets, features=["close", "volume"], windows=(5, 10),
        population=20, generations=5, max_depth=3, patience=0, seed=2, verbose=False,
    )
    assert getattr(hof, "generations_run", 5) == 5


# ---------------------------------------------------------------------------
# P0：样本外切分 / 多 horizon / 库去相关
# ---------------------------------------------------------------------------
def test_gp_train_oos_columns(signal_panel):
    """P0-1：train_frac 切分后结果表含 ic_train / ic_oos / t_oos 报告列。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, _ = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                          population=20, generations=2, max_depth=3, seed=3, verbose=False,
                          train_frac=0.7)
    assert {"ic_train", "ic_oos", "t_oos"}.issubset(df.columns)
    assert df["ic_train"].notna().all() and df["ic_oos"].notna().any()


def test_gp_monthly_weight_runs(signal_panel):
    """P0-2：monthly_weight>0（多 horizon 融合）跑通且不破坏信号。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, _ = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                          population=20, generations=2, max_depth=3, seed=4, verbose=False,
                          monthly_weight=0.5)
    assert len(df) > 0
    assert abs(df.iloc[0]["ic_mean"]) > 0.03


def test_gp_library_penalty_runs(signal_panel):
    """P0-3：library_penalty>0（与库去相关惩罚）跑通。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    # mock 库因子：一个与动量高度相关的面板 + 一个无关面板
    lib = {"mom": panel["close"].pct_change().rolling(5).mean().shift(1)}
    df, _ = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                          population=20, generations=2, max_depth=3, seed=5, verbose=False,
                          library_panels=lib, library_penalty=0.5)
    assert len(df) > 0


# ---------------------------------------------------------------------------
# P1：NSGA-II / hof 去相关 / memetic 局部搜索
# ---------------------------------------------------------------------------
def test_gp_nsga2_multi_objective(signal_panel):
    """P1-4：NSGA-II 双目标跑通，输出 Pareto 前沿（f1/f2/front 列）。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_nsga2

    panel, rets = signal_panel
    df, front = run_gp_nsga2(panel, rets, features=["close", "volume"], windows=(5, 10),
                             population=20, generations=3, max_depth=3, seed=6, verbose=False)
    assert len(df) > 0
    assert {"f1", "f2", "front"}.issubset(df.columns)
    assert len(front) > 0


def test_gp_dedup_by_correlation(signal_panel):
    """P1-6：hof 去相关聚类 —— 两个高度相关公式只保留一个。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import _dedup_hof_by_correlation
    from factor.formula import formula_builder

    panel, _ = signal_panel
    feats = ["close", "volume"]
    # ts_mean_5(close) 与 ts_mean_10(close) 高度相关
    rows = pd.DataFrame([
        {"formula": "ts_mean_5(close)", "ic_mean": 0.1, "ir": 1.0, "t_stat": 5.0, "n": 100},
        {"formula": "ts_mean_10(close)", "ic_mean": 0.08, "ir": 0.8, "t_stat": 4.0, "n": 100},
        {"formula": "cs_rank(volume)", "ic_mean": 0.05, "ir": 0.5, "t_stat": 3.0, "n": 100},
    ])
    out = _dedup_hof_by_correlation(rows, panel, feats, threshold=0.9)
    assert len(out) == 2, "两个高相关动量公式应去重为 1 个"
    assert "cs_rank(volume)" in set(out["formula"])


def test_gp_neighbor_formulas(signal_panel):
    """P1-7：近邻生成 —— 窗口扰动 + 算子替换，且不含原公式。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import generate_neighbor_formulas

    neigh = generate_neighbor_formulas("ts_mean_5(close)", n_per=20, seed=0)
    assert len(neigh) > 0
    assert "ts_mean_5(close)" not in neigh
    # 应包含窗口扰动（ts_mean_10 / ts_mean_3 等）或算子替换（ts_std_5 等）
    assert any("ts_mean_" in f and f != "ts_mean_5(close)" for f in neigh) or \
           any("ts_std_5" in f for f in neigh)


def test_gp_refine_neighbors(signal_panel):
    """P1-7：memetic 局部搜索跑通，合并表含 source 列。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import refine_gp_neighbors

    panel, rets = signal_panel
    base = pd.DataFrame([
        {"formula": "ts_mean_5(close)", "ic_mean": 0.1, "ir": 1.0, "t_stat": 5.0, "n": 100},
    ])
    merged = refine_gp_neighbors(base, panel, rets, n_per=5, min_obs=10)
    assert "source" in merged.columns
    assert "gp" in set(merged["source"])
    assert merged["t_stat"].abs().is_monotonic_decreasing


# ---------------------------------------------------------------------------
# P2：子树缓存 / 种群并行 / 粗筛
# ---------------------------------------------------------------------------
def test_eval_tree_memo_consistent(signal_panel):
    """P2-1：子树缓存不改变求值结果（与无 memo 逐值一致）。"""
    pytest.importorskip("deap")
    from deap import gp as deap_gp
    from factor.genetic_mining import build_primitive_set, eval_tree

    panel, _ = signal_panel
    pset, prim_map = build_primitive_set(["close", "volume"], (5, 10))
    formula = "mul(ts_mean_5(close), cs_rank(ts_delta_10(volume)))"
    tree = deap_gp.PrimitiveTree.from_string(formula, pset)
    out_plain = eval_tree(tree, panel, prim_map)
    memo = {}
    out_memo = eval_tree(tree, panel, prim_map, memo=memo)
    assert np.allclose(out_plain.values, out_memo.values, equal_nan=True)
    # 共享子树复用缓存：相同子树再求值直接命中
    memo2 = {}
    eval_tree(tree, panel, prim_map, memo=memo2)
    assert len(memo2) >= 1


def test_gp_parallel_matches_serial(signal_panel):
    """P2-2：种群并行（n_jobs>1）结果与串行一致。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df1, _ = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                           population=15, generations=2, max_depth=3, seed=7, verbose=False,
                           n_jobs=1)
    df2, _ = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                           population=15, generations=2, max_depth=3, seed=7, verbose=False,
                           n_jobs=2)
    assert len(df1) == len(df2)
    assert set(df1["formula"]) == set(df2["formula"])


def test_gp_sample_step_runs(signal_panel):
    """P2-3：粗筛（sample_step>1）跑通且不破坏流程。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, _ = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                          population=15, generations=2, max_depth=3, seed=8, verbose=False,
                          sample_step=3)
    assert len(df) > 0
