"""冗余审计的数值与口径测试（2026-11-03，与 dup_audit 一起建立）。

锁四件事：

1. **签名口径**：截面 rank + z-score 与 Spearman 等价，NaN 记 0。
2. **相关矩阵**：对角恒 1、完全相同的因子 1、完全反向 −1、正交 ≈0。
3. **重复簇不能有链式效应**：A~B、B~C 但 A~C 不相关时，三者不许并成一簇。
4. **入库门禁走签名查表时，答案要和"直接比面板"一致**。

第 4 条是这次改造的验收点：把 167 分钟的逐面板比对换成矩阵查表，
如果查表给的名字不对，等于门禁失效却还在报 dup_checked=True。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import research.factor_library as fl
from research.factor_library import FactorLibrary, _row_rank_zscore
from scripts.factors.dup_audit import (
    clusters,
    correlation_matrix,
    neighbors,
    vitality_table,
)


def _panel(n_dates=120, n_codes=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_dates)
    cols = [f"{i:06d}.SZ" for i in range(n_codes)]
    return pd.DataFrame(rng.normal(size=(n_dates, n_codes)), index=idx, columns=cols)


def _rets(n_dates=120, n_codes=60, seed=99):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_dates)
    cols = [f"{i:06d}.SZ" for i in range(n_codes)]
    return pd.DataFrame(rng.normal(0, 0.01, size=(n_dates, n_codes)),
                        index=idx, columns=cols)


# ---------------------------------------------------------------------------
# 1. 签名口径
# ---------------------------------------------------------------------------
def test_row_rank_zscore_matches_spearman():
    """两列 sigmoid 后的 z 值点积 / 归一，应等于这两列原始值的 Spearman。"""
    from scipy.stats import spearmanr

    a = _panel(seed=1)
    b = a * 3 + 5                       # 正单调变换 → Spearman 应为 1
    za, zb = _row_rank_zscore(a.to_numpy()), _row_rank_zscore(b.to_numpy())
    cos = float((za * zb).sum() / np.sqrt((za ** 2).sum() * (zb ** 2).sum()))
    assert cos == pytest.approx(1.0, abs=1e-5)

    c = _panel(seed=2)
    zc = _row_rank_zscore(c.to_numpy())
    cos2 = float((za * zc).sum() / np.sqrt((za ** 2).sum() * (zc ** 2).sum()))
    expect = float(np.nanmean([spearmanr(a.iloc[t], c.iloc[t])[0] for t in range(len(a))]))
    # 逐日 Spearman 的均值 vs 池化余弦：同一分布下应很接近
    assert cos2 == pytest.approx(expect, abs=0.05)


def test_row_rank_zscore_turns_constant_into_zero():
    """常数行（整列同值）rank 后全并列，z 必须全 0——这是识别死因子的依据。"""
    a = np.ones((4, 30), dtype=np.float64)
    z = _row_rank_zscore(a)
    assert np.all(z == 0)


def test_row_rank_zscore_keeps_nan_as_zero():
    a = np.array([[1.0, 2.0, np.nan, 4.0]])
    z = _row_rank_zscore(a)
    assert z.shape == a.shape
    assert z[0, 2] == 0.0
    assert np.isfinite(z).all()


# ---------------------------------------------------------------------------
# 2. 相关矩阵
# ---------------------------------------------------------------------------
def test_correlation_matrix_diagonal_and_extremes():
    a = _panel(seed=3)
    b = _panel(seed=4)
    Z = np.stack([
        _row_rank_zscore(a.to_numpy()),
        _row_rank_zscore(a.to_numpy()),           # 与 0 相同
        _row_rank_zscore((-a).to_numpy()),        # 与 0 完全反向
        _row_rank_zscore(b.to_numpy()),           # 独立
    ], axis=1)                                     # (K, F, C)
    corr = correlation_matrix(Z)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-5)
    assert corr[0, 1] == pytest.approx(1.0, abs=1e-5)
    assert corr[0, 2] == pytest.approx(-1.0, abs=1e-5)
    assert abs(corr[0, 3]) < 0.15
    assert np.allclose(corr, corr.T, atol=1e-6)


def test_neighbors_excludes_self_and_ranks():
    corr = np.array([[1.0, 0.9, 0.2],
                     [0.9, 1.0, 0.3],
                     [0.2, 0.3, 1.0]], dtype=np.float32)
    nb = neighbors(corr, ["a", "b", "c"], top_k=2)
    row = nb.set_index("name").loc["a"]
    assert row["top"] == "b"
    assert row["max_corr"] == pytest.approx(0.9)
    assert row["nb1"] == "b"


# ---------------------------------------------------------------------------
# 3. 重复簇不能链式
# ---------------------------------------------------------------------------
def test_clusters_does_not_chain_through_intermediate():
    """A~B 0.9、B~C 0.9，但 A~C 只有 0.1 —— 完全链接必须把它们拆开。

    单链接会把这三个串成一簇（实测在全库上串出过 323 个成员的巨簇，
    簇内相关跨 [0.08, 1.00]，完全无法回答"能合并哪些"）。
    """
    corr = np.array([[1.0, 0.9, 0.1],
                     [0.9, 1.0, 0.9],
                     [0.1, 0.9, 1.0]], dtype=np.float32)
    cl = clusters(corr, ["a", "b", "c"], thresh=0.85)
    labels = cl.set_index("name")["cluster"]
    assert labels["a"] != labels["c"]
    assert not (labels["a"] == labels["b"] == labels["c"])


def test_clusters_groups_true_duplicates():
    corr = np.array([[1.0, 0.98, 0.97, 0.1],
                     [0.98, 1.0, 0.96, 0.2],
                     [0.97, 0.96, 1.0, 0.0],
                     [0.1, 0.2, 0.0, 1.0]], dtype=np.float32)
    cl = clusters(corr, ["a", "b", "c", "d"], thresh=0.85)
    labels = cl.set_index("name")["cluster"]
    assert labels["a"] == labels["b"] == labels["c"]
    assert labels["d"] != labels["a"]


# ---------------------------------------------------------------------------
# 4. 死因子分类
# ---------------------------------------------------------------------------
def test_vitality_table_classifies_dead_factors():
    names = ["ok", "empty", "const", "sparse"]
    cover = np.array([0.9, 0.0, 0.8, 0.9])
    zcov = np.array([0.9, 0.0, 0.0, 0.01])
    v = vitality_table(names, cover, zcov).set_index("name")
    assert v.loc["ok", "issue"] == ""
    assert v.loc["empty", "issue"] == "面板全空"
    assert "常数" in v.loc["const", "issue"]
    assert "极稀疏" in v.loc["sparse", "issue"]


# ---------------------------------------------------------------------------
# 5. 入库门禁：签名查表 vs 逐面板比对
# ---------------------------------------------------------------------------
def _build_tiny_index(lib: FactorLibrary, panels: dict[str, pd.DataFrame]) -> None:
    """按 dup_audit 的格式手写一套索引文件（小规模，够测查表逻辑）。"""
    base = lib.base_root / lib.dataset
    base.mkdir(parents=True, exist_ok=True)
    names = list(panels)
    dates = list(panels[names[0]].index)
    codes = list(panels[names[0]].columns)
    Z = np.stack([_row_rank_zscore(panels[n].to_numpy()) for n in names], axis=1)
    np.save(base / "dup_signatures.npy", Z.astype(np.float32))
    corr = correlation_matrix(Z)
    np.savez_compressed(base / "dup_matrix.npz", corr=corr,
                        names=np.array(names, dtype="U64"),
                        dates=np.array([str(d.date()) for d in dates]))
    (base / "dup_meta.json").write_text(json.dumps({
        "dataset": lib.dataset, "built_at": "test", "n_factors": len(names),
        "n_dates": len(dates), "n_codes": len(codes),
        "dates": [str(d.date()) for d in dates], "codes": codes,
    }), encoding="utf-8")


def test_dup_fast_path_finds_near_duplicate(tmp_path):
    """新因子的面板 ≈ 库内 fb 时，查表应直接指认 fb。"""
    fl._DUP_INDEX_CACHE.clear()
    lib = FactorLibrary(root=tmp_path, dataset="t")
    rng = np.random.default_rng(7)
    a = _panel(seed=10)
    b = a + pd.DataFrame(rng.normal(0, 0.02, size=a.shape), index=a.index, columns=a.columns)
    solo = _panel(seed=11)
    _build_tiny_index(lib, {"fa": a, "fb": b, "fsolo": solo})

    near_a = a + pd.DataFrame(rng.normal(0, 0.02, size=a.shape),
                              index=a.index, columns=a.columns)
    row = lib._dup_check_fast(near_a, dup_corr=0.7, reject_dup=False, name="cand")
    assert row is not None
    assert row["dup_checked"] is True
    assert row["dup_top"] == "fa" or row["dup_top"] == "fb"
    assert row["dup_corr_max"] > 0.9

    rnd = _panel(seed=12)
    row2 = lib._dup_check_fast(rnd, dup_corr=0.7, reject_dup=False, name="rand")
    assert row2["dup_corr_max"] < 0.5
    fl._DUP_INDEX_CACHE.clear()


def test_dup_fast_path_rejects_when_over_threshold(tmp_path):
    fl._DUP_INDEX_CACHE.clear()
    lib = FactorLibrary(root=tmp_path, dataset="t")
    a = _panel(seed=20)
    _build_tiny_index(lib, {"fa": a})
    with pytest.raises(ValueError, match="reject_dup"):
        lib._dup_check_fast(a * 2 + 1, dup_corr=0.7, reject_dup=True, name="dup")
    fl._DUP_INDEX_CACHE.clear()


def test_dup_fast_path_returns_none_without_index(tmp_path):
    """没有索引文件时必须返回 None，让调用方回退到逐面板比对。"""
    fl._DUP_INDEX_CACHE.clear()
    lib = FactorLibrary(root=tmp_path, dataset="noidx")
    assert lib.dup_index() is None
    assert lib._dup_check_fast(_panel(), 0.7, False, name="x") is None


def test_dup_fast_path_matches_bruteforce(tmp_path):
    """查表结果与「逐面板逐日 rank 相关」必须一致到 0.05 以内。

    注意索引里**不能**放着待查面板本身——否则查到的是自己、相关恒为 1，
    测的就不是"查表准不准"而是"自己等于自己"。
    """
    fl._DUP_INDEX_CACHE.clear()
    lib = FactorLibrary(root=tmp_path, dataset="t")
    rng = np.random.default_rng(31)
    a = _panel(seed=30)
    solo = _panel(seed=33)
    b = a * 0.6 + pd.DataFrame(rng.normal(0, 1, size=a.shape),
                               index=a.index, columns=a.columns)
    _build_tiny_index(lib, {"fa": a, "fsolo": solo})

    fast = lib._dup_check_fast(b, dup_corr=0.7, reject_dup=False, name="cand")
    assert fast["dup_top"] == "fa"

    ra, rb = a.rank(axis=1), b.rank(axis=1)
    brute = float(ra.corrwith(rb, axis=1, method="pearson").mean())
    assert fast["dup_corr_max"] == pytest.approx(brute, abs=0.05)
    fl._DUP_INDEX_CACHE.clear()


def test_real_library_dup_index_is_consistent():
    """真实库若有索引，抽查它的自相关对角与形状必须自洽。"""
    lib = FactorLibrary(dataset="all_a_2018_2026")
    idx = lib.dup_index()
    if idx is None:
        pytest.skip("本机没有 dup 索引（先跑 scripts.factors.dup_audit）")
    corr, names = idx["corr"], idx["names"]
    assert corr.shape == (len(names), len(names))
    assert np.allclose(np.diag(corr), 1.0, atol=1e-4)
    assert np.allclose(corr, corr.T, atol=1e-4)
    assert np.isfinite(corr).all()
