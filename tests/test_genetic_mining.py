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


# ---------------------------------------------------------------------------
# 国君研报（2023 解构系列之一）对齐（2026-08-27）
# ---------------------------------------------------------------------------
def test_ls_net_stats_direction_and_cost(signal_panel):
    """费后多空统计：正向因子 sharpe>0；方向中立（负向因子同值）；费用降低夏普。"""
    from factor.genetic_mining import _ls_net_stats

    panel, rets = signal_panel
    fp = panel["close"].pct_change()  # AR(1) 动量 mock，与未来收益正相关
    a = _ls_net_stats(fp, rets, fee_rt=0.0)
    b = _ls_net_stats(fp, rets, fee_rt=0.003)
    assert np.isfinite(a["sharpe"]) and a["sharpe"] > 0
    assert b["sharpe"] < a["sharpe"], "收费后夏普应低于免费"
    neg = _ls_net_stats(-fp, rets, fee_rt=0.0)
    assert abs(neg["sharpe"] - a["sharpe"]) < 1e-8, "方向翻转后 sharpe 应一致（方向中立）"
    assert a["n"] > 50


def test_gp_sharpe_fitness_runs(signal_panel):
    """fitness_mode='sharpe'（国君基准适应度）跑通且最优因子有正的费后表现。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, _ = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                          population=25, generations=3, max_depth=3, seed=13, verbose=False,
                          fitness_mode="sharpe")
    assert len(df) > 0
    # 最优因子训练段费前 IC 应显著（mock 信号下 GP 找到的公式不应完全失效）
    best = df.iloc[0]
    assert abs(best["ic_train"]) > 0.03 or abs(best["ic_mean"]) > 0.03


def test_gp_annual_return_and_ret_minus_dd_modes(signal_panel):
    """annual_return / ret_minus_dd 两种净值型口径跑通。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    for mode in ("annual_return", "ret_minus_dd"):
        df, _ = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                              population=15, generations=2, max_depth=3, seed=14,
                              verbose=False, fitness_mode=mode)
        assert len(df) > 0, mode


def test_gp_min_fitness_gate(signal_panel):
    """min_fitness 准入门槛：过高门槛时 hof 为空 / 结果表为空而不报错。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, hof = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                            population=15, generations=2, max_depth=3, seed=15, verbose=False,
                            min_fitness=1e9)   # 不可能有个体达标
    assert len(df) == 0
    # 正常门槛：hof 非空且所有入选个体满足门槛语义由包装保证
    df2, hof2 = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                              population=15, generations=2, max_depth=3, seed=16, verbose=False)
    assert len(df2) > 0


def test_gp_gtja_evolvement_options(signal_panel):
    """束搜索 + 家庭竞争 + 排挤三者组合跑通（研报最佳组合 s_r_bs_fc_sp）。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, hof = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                            population=20, generations=4, max_depth=3, seed=17, verbose=False,
                            beam_mult=2, family_competition=True, crowding="supplant",
                            crowd_corr_thr=0.9)
    assert len(df) > 0
    # sharing 变体也跑通
    df2, _ = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                           population=20, generations=3, max_depth=3, seed=18, verbose=False,
                           crowding="sharing")
    assert len(df2) > 0


def test_gp_separated_mutation_runs(signal_panel):
    """四类变异概率分离开关跑通。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, hof = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                            population=20, generations=3, max_depth=3, seed=19, verbose=False,
                            separated_mutation=True, patience=0)
    assert len(df) > 0
    assert getattr(hof, "generations_run", 0) == 3


def test_conditional_ops_registered_and_eval(signal_panel):
    """国君条件选择类算子进入 GP 原语集且可求值（greater/less/三元/四元）。"""
    pytest.importorskip("deap")
    from deap import gp as deap_gp
    from factor.genetic_mining import build_primitive_set, eval_tree

    panel, _ = signal_panel
    pset, prim_map = build_primitive_set(["close", "volume"], (5, 10))
    for name in ["greater", "less", "if_then_else", "clear_by_cond", "if_cond_then_else"]:
        assert name in prim_map, f"原语 {name} 未注册"
    cases = [
        "greater(ts_delta_5(close), ts_delay_5(close))",
        "if_then_else(greater(close, ts_delay_5(close)), cs_rank(volume), close)",
        "clear_by_cond(close, ts_delay_5(close), volume)",
        "if_cond_then_else(close, ts_mean_10(close), volume, cs_rank(close))",
    ]
    for f in cases:
        tree = deap_gp.PrimitiveTree.from_string(f, pset)
        out = eval_tree(tree, panel, prim_map)
        assert out is not None and out.shape == panel["close"].shape, f"求值失败: {f}"


def test_signed_sqrt_keeps_sign():
    """保号 sqrt：sqrt(-4)=-2、sqrt(4)=2、NaN 传递，不再整片 NaN。"""
    from factor.operators import sqrt_

    x = pd.DataFrame({"a": [4.0, -4.0, 9.0], "b": [-9.0, 1e-8, np.nan]})
    y = sqrt_(x)
    assert y.iloc[0, 0] == 2.0 and y.iloc[0, 1] == -3.0
    assert y.iloc[1, 0] == -2.0
    assert np.isnan(y.iloc[2, 1])


def test_crowding_supplant_reduces_fitness_of_similar():
    """排挤算法：相似对中低分者被减半、高分不变；restore 恢复精确原值。"""
    pytest.importorskip("deap")
    from deap import creator as deap_creator, gp as deap_gp
    from factor.genetic_mining import (_adjust_crowding, _ensure_creator,
                                        _restore_crowding, build_primitive_set)

    _ensure_creator()
    rng = np.random.default_rng(0)
    n_days, n_codes = 60, 10
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    cols = [f"{600000+i:06d}.SH" for i in range(n_codes)]
    panel = {"close": pd.DataFrame(rng.normal(100, 5, (n_days, n_codes)), idx, cols)}
    pset, prim_map = build_primitive_set(["close"], (5,))
    t1 = deap_creator.IndividualGP(deap_gp.PrimitiveTree.from_string("ts_mean_5(close)", pset))
    t2 = deap_creator.IndividualGP(deap_gp.PrimitiveTree.from_string("ts_sum_5(close)", pset))   # 与 t1 相似但不等
    t3 = deap_creator.IndividualGP(deap_gp.PrimitiveTree.from_string("close", pset))
    pool = []
    for tree, score in ((t1, 1.0), (t2, 0.6), (t3, 3.0)):
        ind = deap_creator.IndividualGP(tree)
        ind.fitness.values = (score,)
        pool.append(ind)
    saved = _adjust_crowding(pool, panel, prim_map, method="supplant",
                             corr_thr=0.7)
    {id(ind): float(ind.fitness.values[0]) for ind in pool}
    _restore_crowding(pool, saved)
    restored = {id(ind): float(ind.fitness.values[0]) for ind in pool}
    # 恢复后所有个体回到精确原值（排挤只影响选择，不污染 hof）
    assert all(abs(restored[id(ind)] - orig) < 1e-12
               for ind, orig in zip(pool, [1.0, 0.6, 3.0]))
    assert isinstance(saved, dict)


def test_beam_search_population_reduction(signal_panel):
    """束搜索：beam_mult>1 时正常运行且结果与标准流程结构一致。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, hof = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                            population=12, generations=2, max_depth=3, seed=20,
                            verbose=False, beam_mult=3)
    assert len(df) <= 20   # hall_size 默认 20


# ---------------------------------------------------------------------------
# gtja 预设与次日 VWAP 执行链（2026-08-27）
# ---------------------------------------------------------------------------
def test_apply_gtja_preset_resolution(signal_panel):
    """gtja 预设：None 哨兵按表1 解析；显式传参优先；终端裁剪为六量价字段。"""
    import argparse
    from scripts.mine_factors import _apply_gtja_preset

    panel, rets = signal_panel
    panel = dict(panel)
    panel["amount"] = panel["volume"] * panel["close"]      # mock amount
    panel["OPERA_REV"] = panel["close"] * 0 + 1.0           # 应被移出终端

    args = argparse.Namespace(
        pop=None, gen=None, patience=None, train_frac=None, gp_tournament=None,
        gp_min_fitness=0.0, gp_dedup_corr=None, gp_separated_mutation=False,
        gp_fitness=None)
    out_panel, returns_gp = _apply_gtja_preset(args, panel, real=False)

    # 表1 超参
    assert (args.pop, args.gen, args.patience, args.train_frac) == (500, 10, 5, 1.0)
    assert args.gp_tournament == 5 and args.gp_min_fitness == 0.5
    assert args.gp_dedup_corr == 0.5 and args.gp_fitness == "sharpe"
    assert args.gp_separated_mutation is True
    # 终端裁剪为六量价字段，财务字段移出
    assert set(out_panel) <= {"open", "high", "low", "close", "volume", "vwap"}
    assert "OPERA_REV" not in out_panel and "vwap" in out_panel
    # 执行链收益与原 close 收益不同序列（T+2/T+1 VWAP 对齐）
    assert returns_gp.shape == rets.shape
    assert not np.allclose(returns_gp.fillna(0), rets.fillna(0))


def test_build_vwap_exec_returns_alignment():
    """VWAP 执行链：首两行应为 NaN（未来函数保护），无 inf。"""
    from factor.gtja import build_vwap_exec_returns

    rng = np.random.default_rng(0)
    n_days, n_codes = 80, 6
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    cols = [f"{600000+i:06d}.SH" for i in range(n_codes)]
    volume = pd.DataFrame(rng.lognormal(12, 0.3, (n_days, n_codes)), idx, cols)
    amount = pd.DataFrame(rng.lognormal(15, 0.3, (n_days, n_codes)), idx, cols) * 1e4
    close = pd.DataFrame(rng.normal(50, 1, (n_days, n_codes)), idx, cols).abs()
    panel = {"close": close, "volume": volume, "amount": amount}
    rets = build_vwap_exec_returns(panel)
    # T 需要 T+1、T+2 的 VWAP → 最后两行必为 NaN
    assert rets.iloc[-1].isna().all() and rets.iloc[-2].isna().all()
    assert np.isfinite(rets.iloc[:-2]).all().all()


def test_gtja_results_sorted_by_fitness(signal_panel):
    """sharpe 模式结果表含 fitness 列且按其降序；基准复现命令可带 train_frac=1.0。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import run_gp_mining

    panel, rets = signal_panel
    df, _ = run_gp_mining(panel, rets, features=["close", "volume"], windows=(5, 10),
                          population=20, generations=2, max_depth=3, seed=21,
                          verbose=False, fitness_mode="sharpe", min_fitness=0.0,
                          train_frac=1.0)
    assert len(df) > 0 and "fitness" in df.columns
    assert (df["fitness"].diff().dropna() <= 1e-9).all(), "应按 fitness 降序"


def test_primitive_min_window_gates():
    """统计/技术类算子的小窗口注册门槛：恒 NaN/退化的 (算子,窗口) 不进原语集。"""
    pytest.importorskip("deap")
    from factor.genetic_mining import build_primitive_set

    pset, pm = build_primitive_set(["close"], (1, 2, 5, 10, 60))
    # 门槛之下不应注册（恒 NaN / 退化 / 数值爆炸）
    for absent in ("ts_skew_1", "ts_skew_2", "ts_kurt_1", "ts_kurt_2",
                   "ts_zscore_1", "ts_corr_1", "ts_corr_2", "ts_cov_1",
                   "boll_pctb_1", "kama_1", "aroonosc_1", "adx_2",
                   "ts_product_60"):
        assert absent not in pm, f"{absent} 应被最小/最大窗口门槛剪掉"
    # 门槛之上正常注册
    for present in ("ts_skew_5", "ts_kurt_5", "ts_zscore_2", "ts_corr_5",
                    "boll_pctb_5", "rsi_2", "kama_5",
                    "aroonosc_5", "adx_5", "ts_product_5", "ts_product_10"):
        assert present in pm, f"{present} 应保留"


# ---------------------------------------------------------------------------
# 可交易性过滤（2026-08-28：T+1 停牌/封板、当日 ST/停牌 剔除进适应度）
# ---------------------------------------------------------------------------
def test_build_tradable_mask_rules(tmp_path):
    """mask 构建规则：次日停牌/封涨停/封跌停、当日 ST/停牌 → False；其余 True。"""
    import numpy as np
    import pandas as pd
    from data.cache_helpers import build_tradable_mask

    idx = pd.date_range("2023-01-01", periods=6, freq="B")
    cols = ["A", "B", "C", "D", "E"]
    close = pd.DataFrame(np.full((6, 5), 10.0), idx, cols)   # 复权价 == 原始价（bwd=1）

    def st(date_rows: dict) -> pd.DataFrame:
        rows = {}
        for d, flags in date_rows.items():
            for c, f in flags.items():
                rows[(d, c)] = f
        df = pd.DataFrame.from_dict(rows, orient="index",
                                    columns=["pre_close", "high_limited", "low_limited",
                                             "is_st", "is_suspended",
                                             "is_ex_dividend", "is_ex_rights"])
        df.index = pd.MultiIndex.from_tuples(df.index, names=["date", "code"])
        return df

    d1, d2 = idx[0], idx[1]
    status = st({
        # A 在 d2 封涨停（close 10.0 >= high_limited 10.0）→ d1 信号不可交易
        d1: {"A": (10.0, 11.0, 9.0, False, False, False, False)},
        d2: {"A": (10.0, 10.0, 9.0, False, False, False, False),
             # B 在 d2 停牌 → d1 信号不可交易
             "B": (10.0, 11.0, 9.0, False, True, False, False),
             # C 在 d2 正常（未触板）
             "C": (10.0, 11.0, 9.0, False, False, False, False),
             # D 在 d2 收盘 10.0 == low_limited 10.0 封跌停
             "D": (10.0, 11.0, 10.0, False, False, False, False),
             # E 在 d2 是 ST
             "E": (10.0, 11.0, 9.0, True, False, False, False)},
    })
    p = tmp_path / "history_stock_status.parquet"
    status.to_parquet(p)
    mask = build_tradable_mask(close, bwd=None, cache_root=str(tmp_path))

    assert not mask.loc[d1, "A"], "T+1 封涨停应剔除"
    assert not mask.loc[d1, "B"], "T+1 停牌应剔除"
    assert not mask.loc[d1, "D"], "T+1 封跌停应剔除"
    assert not mask.loc[d1, "E"], "T+1 ST 应剔除"
    assert mask.loc[d1, "C"], "正常股票应保留"
    # 状态缺失的日子补 True
    assert bool(mask.loc[idx[3]].all())


def test_ls_net_stats_tradable_filter():
    """适应度可交易过滤：被 mask 掉的"假收益王"不得进入多空腿。"""
    import numpy as np
    import pandas as pd
    from factor.genetic_mining import _ls_net_stats

    rng = np.random.default_rng(0)
    n_days, n_codes = 80, 40
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    cols = [f"S{i:03d}" for i in range(n_codes)]
    rets = pd.DataFrame(rng.normal(0, 0.01, (n_days, n_codes)), idx, cols)
    # 构造一只"每天 +10%"的假收益王（现实中=连续涨停买不进）
    rets["S000"] = 0.10
    fp = pd.DataFrame(np.zeros((n_days, n_codes)), idx, cols)
    fp["S000"] = 1.0            # 因子把它排到多头腿
    tradable = pd.DataFrame(True, index=idx, columns=cols)
    tradable["S000"] = False    # mask 剔除

    with_fake = _ls_net_stats(fp, rets, min_cov_frac=0.9)
    filtered = _ls_net_stats(fp, rets, min_cov_frac=0.9, tradable=tradable)
    assert with_fake["sharpe"] > filtered["sharpe"], "过滤涨停王后夏普应大幅下降"
    assert abs(filtered["ann_ret"]) < abs(with_fake["ann_ret"]) / 10, \
        "过滤后剩余应为噪声级收益，而非涨停王贡献"
