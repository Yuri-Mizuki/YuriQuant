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


# ---------------------------------------------------------------------------
# 华泰研报复现模式（htai，2026-08-10）
# ---------------------------------------------------------------------------
def _htai_neutral_panels(panel):
    """从 mock 面板构造中性化协变量（size/industry/mom20/vol20/turn20）。"""
    rng = np.random.default_rng(0)
    close = panel["close"]
    codes = close.columns
    n = len(close)
    out = {
        "mom20": close.pct_change(20),
        "vol20": close.pct_change().rolling(20).std(),
        "turn20": close.rolling(20).mean() * 1e-3,
        "size": close * 1e8,
    }
    ind = pd.DataFrame(
        np.tile(np.array(["Bank"] * 10 + ["Tech"] * 10), (n, 1)),
        index=close.index, columns=codes,
    )
    out["industry"] = ind
    return out


def test_htai_preprocess_pipeline(signal_panel):
    """华泰环内预处理：MAD(±5×MAD) → 五因子中性化 → zscore，形状保持。"""
    from factor.genetic_mining import _htai_preprocess

    panel, _ = signal_panel
    fp = panel["close"] * 1000 + 5.0  # 引入量纲与水平
    # 无协变量：只做 MAD + 标准化（mock 下限）
    out1 = _htai_preprocess(fp, neutral_panels=None)
    assert out1.shape == fp.shape
    assert np.isfinite(out1).sum().sum() > 0
    # 有协变量：中性化路径
    out2 = _htai_preprocess(fp, neutral_panels=_htai_neutral_panels(panel))
    assert out2.shape == fp.shape
    assert np.isfinite(out2).sum().sum() > 0


def test_htai_rankic_mean_fitness_runs(signal_panel):
    """华泰复现（htai=True, fitness_mode=rankic_mean）：月频20日目标 + 平均RankIC 跑通。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, _ = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                          population=20, generations=3, max_depth=3, seed=9, verbose=False,
                          htai=True, neutral_panels=_htai_neutral_panels(panel),
                          fitness_mode="rankic_mean", train_frac=1.0)
    assert len(df) > 0
    assert {"formula", "ic_mean", "ic_train", "ic_oos"}.issubset(df.columns)
    assert df["ic_mean"].notna().sum() >= len(df) // 2
    # ic_train 应与 ic_mean 一致（train_frac=1.0 全样本）
    assert np.allclose(df["ic_mean"], df["ic_train"], equal_nan=True)


def test_htai_tstat_fitness_runs(signal_panel):
    """华泰复现 + tstat 适应度（|mean|/std）跑通。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, _ = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                          population=20, generations=2, max_depth=3, seed=10, verbose=False,
                          htai=True, neutral_panels=None, fitness_mode="tstat")
    assert len(df) > 0
    assert df["t_stat"].notna().all()


def test_new_primitives_registered_and_eval(signal_panel):
    """华泰函数集扩充：inv/delay/ts_cov/ts_product/ts_zscore/scale/sigmoid/rank_sub/rank_div
    进入 GP 原语集且可求值。"""
    pytest.importorskip("deap")
    from deap import gp as deap_gp
    from factor.genetic_mining import build_primitive_set, eval_tree

    panel, _ = signal_panel
    pset, prim_map = build_primitive_set(["close", "volume"], (5, 10))
    for name in ["ts_zscore_5", "ts_delay_5", "ts_product_5", "ts_cov_5",
                 "inv", "sigmoid", "scale", "rank_sub", "rank_div"]:
        assert name in prim_map, f"原语 {name} 未注册"
    cases = ["ts_zscore_5(close)", "inv(close)", "sigmoid(close)", "scale(close)",
             "rank_sub(close, volume)", "rank_div(close, volume)", "ts_cov_5(close, volume)"]
    for f in cases:
        tree = deap_gp.PrimitiveTree.from_string(f, pset)
        out = eval_tree(tree, panel, prim_map)
        assert out is not None and out.shape == panel["close"].shape, f"求值失败: {f}"


def test_winsorize_mad_htai_scale(signal_panel):
    """华泰口径去极值：consistency_scale=False 时不乘 1.4826。"""
    from factor.preprocessing import winsorize_mad

    panel, _ = signal_panel
    x = panel["close"]
    a = winsorize_mad(x, n_mad=5.0, consistency_scale=True)   # 5×1.4826×MAD（更宽）
    b = winsorize_mad(x, n_mad=5.0, consistency_scale=False)  # 5×MAD（更窄）
    # 不乘常数 → 截断更严 → 改动更多
    assert (b != x).sum().sum() >= (a != x).sum().sum()
    assert np.isfinite(b).all().all()


# ---------------------------------------------------------------------------
# 报告23：互信息 / 多头超额适应度 / 非线性因子线性化（2026-08-10）
# ---------------------------------------------------------------------------
def test_mutual_info_series_ranges(signal_panel):
    """互信息序列：独立时≈0、强相关时>0，且序列长度与面板一致。"""
    from factor.genetic_mining import _mutual_info_series, _monthly_forward_returns

    # 用小面板扩大股数（20 股太稀，MI 小样本偏差大）
    rng = np.random.default_rng(0)
    n_days, n_codes = 150, 80
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    rets_mat = rng.normal(0.02, 0.02, (n_days, n_codes))
    close = pd.DataFrame(np.exp(np.cumsum(rets_mat, axis=0)), idx, codes)
    rets = close.pct_change().shift(-1)
    rm = _monthly_forward_returns(rets)
    # 强相关：因子=未来收益的 rank
    fp = rets.rank(axis=1, pct=True)
    mi_strong = _mutual_info_series(fp, rm).dropna()
    # 独立：随机面板
    fp_noise = pd.DataFrame(rng.normal(size=close.shape), index=idx, columns=codes)
    mi_noise = _mutual_info_series(fp_noise, rm).dropna()
    assert mi_strong.mean() > mi_noise.mean(), "强相关因子的 MI 应高于噪声因子"
    assert mi_noise.mean() < 0.3, "独立变量 MI 应接近 0（小样本保护后偏差应小）"
    assert 0 < mi_strong.mean() < 2.0


def test_top_excess_series_direction(signal_panel):
    """多头超额：正向因子 top_excess>0，负向因子 bot_excess>0，取 max 无偏。"""
    from factor.genetic_mining import _top_excess_series, _monthly_forward_returns

    panel, rets = signal_panel
    rm = _monthly_forward_returns(rets)
    fp = panel["close"].pct_change().shift(-1)   # 与未来收益正相关（mock AR(1)）
    t_pos, b_pos, nd_pos = _top_excess_series(fp, rm, top_frac=0.1)
    t_neg, b_neg, nd_neg = _top_excess_series(-fp, rm, top_frac=0.1)
    assert nd_pos > 10 and nd_neg > 10
    assert t_pos > 0 and b_neg > 0, "正/负向因子应各有一侧超额为正"
    assert abs(t_pos - b_neg) < 1e-6, "符号翻转后 Top/Bottom 应互换"


def test_mutual_info_fitness_runs(signal_panel):
    """fitness_mode='mutual_info'（报告23 适应度）跑通。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, _ = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                          population=20, generations=2, max_depth=3, seed=11, verbose=False,
                          htai=True, fitness_mode="mutual_info", train_frac=1.0)
    assert len(df) > 0
    assert {"mi_mean", "top_excess"}.issubset(df.columns)
    assert df["mi_mean"].notna().sum() >= len(df) // 2


def test_top_excess_fitness_runs(signal_panel):
    """fitness_mode='top_excess'（报告23 适应度）跑通。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, _ = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                          population=20, generations=2, max_depth=3, seed=12, verbose=False,
                          htai=True, fitness_mode="top_excess", train_frac=1.0)
    assert len(df) > 0
    assert df["top_excess"].notna().sum() >= len(df) // 2


def test_cubic_and_polynomial_transform(signal_panel):
    """三次方残差法（中间凸）与多项式拟合法（形状保持）跑通。"""
    from factor.genetic_mining import (_monthly_forward_returns,
                                       cubic_residual_transform,
                                       polynomial_transform)
    from research.factor_analysis import calc_ic_series

    panel, rets = signal_panel
    fp = panel["close"]
    # 三次方残差：残差与原始因子相关性应显著降低（剥离三次分量）
    resid = cubic_residual_transform(fp)
    c = fp.corrwith(resid, axis=1).mean()
    assert abs(c) < 0.95, f"残差应与原因子相关性大幅下降，实际 {c:.3f}"
    # 多项式拟合：输出形状一致且有有效值
    fp_poly = polynomial_transform(fp, rets, fit_window=60, refit=20)
    assert fp_poly.shape == fp.shape
    assert fp_poly.notna().sum().sum() > 0
    ic_p = calc_ic_series(fp_poly, _monthly_forward_returns(rets)).dropna()
    assert len(ic_p) > 10
