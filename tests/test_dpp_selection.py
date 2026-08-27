"""
DPP 集合级筛选测试（2026-08-17）
================================
覆盖：三角相关结构（连锁误杀修复）、log-det 单调性、与两两去重对比、
随机面板降相关、质量项优先、FactorLibrary.select_diverse 集成。
mock 数据，不依赖 SDK。
"""
import numpy as np
import pandas as pd
import pytest

from research.dpp_selection import (corr_matrix, dpp_select, greedy_logdet_dpp,
                                    pairwise_dedup, similarity_kernel)
from research.factor_library import FactorLibrary


def _tri_corr():
    """A~B=0.8, B~C=0.8, A~C=0.1：两两去重会连锁误杀的经典结构。"""
    C = np.array([[1.0, 0.8, 0.1],
                  [0.8, 1.0, 0.8],
                  [0.1, 0.8, 1.0]])
    return pd.DataFrame(C, index=["A", "B", "C"], columns=["A", "B", "C"])


def test_dpp_selects_complementary_pair_in_triangle():
    """k=2 时 DPP 应选出 {A, C}（最互补），而不是顺序依赖的 {A, B}。"""
    corr = _tri_corr()
    res = dpp_select(corr, k=2)  # 纯多样性
    assert set(res["selected"]) == {"A", "C"}


def test_pairwise_dedup_order_dependence_kills_independent():
    """同结构下两两去重结果顺序依赖：order=[B,A,C] 时 A、C 都与 B 相关 0.8 被连锁
    误杀 → 只剩 {B}；DPP 从集合整体看 log-det，稳定选互补的 {A,C}。"""
    corr = _tri_corr()
    sel = pairwise_dedup(corr, order=["B", "A", "C"], threshold=0.7)
    assert sel == ["B"]  # 误杀：丢掉与 A 仅 0.1 相关的 C
    res = dpp_select(corr, k=2)
    assert set(res["selected"]) == {"A", "C"}


def test_logdet_trace_consistent_and_superior_to_random():
    """贪心 log-det 轨迹与直接 slogdet 一致；DPP 子集 log-det 显著高于随机子集。

    注：对单位对角核（exp 核，对角线=1），加入冗余因子的 Schur complement ≤ 1，
    log-det 增益 ≤ 0，因此轨迹**非增**；正确性质是「整体信息空间最大」。
    """
    rng = np.random.default_rng(7)
    n, m = 30, 12
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    base = rng.normal(0, 1, (n, 3))
    panels = {}
    for i in range(m):
        if i < 9:
            x = base[:, i % 3] * (0.8 + 0.3 * rng.random()) + rng.normal(0, 0.3, n)
        else:
            x = rng.normal(0, 1, n)
        panels[f"f{i}"] = pd.DataFrame({"c": x}, index=idx)
    corr = corr_matrix(panels, method="flat", min_overlap_dates=5, min_overlap_codes=1)
    L = similarity_kernel(corr)
    idx, trace = greedy_logdet_dpp(L, k=6)
    # 轨迹与直接 slogdet 一致（实现正确性）
    for t, kk in enumerate(range(1, 7)):
        sub = L[np.ix_(idx[:kk], idx[:kk])]
        assert abs(trace[t] - np.linalg.slogdet(sub)[1]) < 1e-9
    sel_logdet = trace[-1]
    # 随机同规模子集 log-det（200 次采样）：DPP 应显著优于随机（>95 分位）
    rand_l = []
    for _ in range(200):
        ridx = rng.choice(len(corr), 6, replace=False)
        sub = L[np.ix_(ridx, ridx)]
        rand_l.append(np.linalg.slogdet(sub)[1])
    assert sel_logdet > float(np.percentile(rand_l, 95)) + 1e-9


def test_dpp_reduces_max_corr_on_synthetic_panels():
    """随机面板：DPP 筛选后池内 max/mean |corr| 应显著下降。"""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2023-01-01", periods=150, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(20)]
    base = pd.DataFrame(rng.normal(0, 1, (len(idx), 3)), index=idx, columns=["u1", "u2", "u3"])
    panels = {}
    for i in range(15):
        comp = base[["u1", "u2", "u3"]].to_numpy() @ rng.normal(0, 1, 3)
        comp = comp + rng.normal(0, 0.5, (len(idx), 1)).ravel()
        df = pd.DataFrame(np.tile(comp[:, None], (1, len(codes))),
                          index=idx, columns=codes)
        df += pd.DataFrame(rng.normal(0, 0.1, df.shape), index=idx, columns=codes)
        panels[f"f{i:02d}"] = df
    corr = corr_matrix(panels, method="cross")
    res = dpp_select(corr, k=6)
    assert res["max_abs_corr_selected"] < res["max_abs_corr_pool"]
    assert res["mean_abs_corr_selected"] < res["mean_abs_corr_pool"]
    assert res["mean_abs_corr_selected"] < 0.5  # 强相关簇被拆散


def test_quality_biases_selection_toward_strong_factors():
    """质量项：IC 高的因子应优先入选（同等相关簇内）。"""
    rng = np.random.default_rng(3)
    # 两个强相关因子（相关 0.9），一个独立因子；k=2
    idx = pd.date_range("2023-01-01", periods=200, freq="B")
    base = rng.normal(0, 1, len(idx))
    f_hi = pd.DataFrame({"c": base}, index=idx)
    f_lo = pd.DataFrame({"c": base * 0.99 + rng.normal(0, 0.1, len(idx))}, index=idx)
    f_ind = pd.DataFrame({"c": rng.normal(0, 1, len(idx))}, index=idx)
    panels = {"hi": f_hi, "lo": f_lo, "ind": f_ind}
    corr = corr_matrix(panels, method="flat", min_overlap_dates=5, min_overlap_codes=1)
    ic = pd.Series({"hi": 0.05, "lo": 0.02, "ind": 0.03})
    res_no_q = dpp_select(corr, k=2)                      # 纯多样性
    res_q = dpp_select(corr, k=2, quality=ic.abs())       # 质量加权
    # 纯多样性在 hi/lo 中任选其一 + ind；质量加权应偏向 ic 更高的 hi
    assert "ind" in res_no_q["selected"]
    assert "ind" in res_q["selected"]
    assert "hi" in res_q["selected"]


def test_quality_length_mismatch_raises():
    with pytest.raises(ValueError, match="quality"):
        greedy_logdet_dpp(np.eye(4), k=2, quality=[1.0, 1.0, 1.0])


def test_k_clamped_to_n():
    corr = _tri_corr()
    res = dpp_select(corr, k=99)
    assert len(res["selected"]) == 3


# ===========================================================================
# FactorLibrary.select_diverse 集成
# ===========================================================================
def _mock_panel(n_days=200, n_codes=30, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    rets = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_codes)), idx, codes)
    factor = rets.shift(1)
    return factor, rets


def test_select_diverse_integration(tmp_path):
    lib = FactorLibrary(root=tmp_path / "flib")
    factor, returns = _mock_panel()
    for i in range(8):
        rng = np.random.default_rng(100 + i)
        f = factor * (0.5 + 0.5 * rng.random()) + pd.DataFrame(
            rng.normal(0, 0.05, factor.shape), index=factor.index, columns=factor.columns)
        lib.register(f"f{i}", f, returns)
    res = lib.select_diverse(k=3)
    assert len(res["selected"]) == 3
    assert res["n_pool"] == 8
    assert res["max_abs_corr_selected"] <= res["max_abs_corr_pool"] + 1e-9
    assert "quality_mean_selected" in res  # 质量保留率字段存在
    # 纯多样性模式
    res2 = lib.select_diverse(k=3, quality_col=None)
    assert len(res2["selected"]) == 3


def test_select_diverse_raises_without_panels(tmp_path):
    lib = FactorLibrary(root=tmp_path / "flib")
    with pytest.raises(RuntimeError, match="无可用因子面板"):
        lib.select_diverse()
