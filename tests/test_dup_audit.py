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
from research.factor_library import FactorLibrary, row_rank_zscore
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
    za, zb = row_rank_zscore(a.to_numpy()), row_rank_zscore(b.to_numpy())
    cos = float((za * zb).sum() / np.sqrt((za ** 2).sum() * (zb ** 2).sum()))
    assert cos == pytest.approx(1.0, abs=1e-5)

    c = _panel(seed=2)
    zc = row_rank_zscore(c.to_numpy())
    cos2 = float((za * zc).sum() / np.sqrt((za ** 2).sum() * (zc ** 2).sum()))
    expect = float(np.nanmean([spearmanr(a.iloc[t], c.iloc[t])[0] for t in range(len(a))]))
    # 逐日 Spearman 的均值 vs 池化余弦：同一分布下应很接近
    assert cos2 == pytest.approx(expect, abs=0.05)


def test_row_rank_zscore_turns_constant_into_zero():
    """常数行（整列同值）rank 后全并列，z 必须全 0——这是识别死因子的依据。"""
    a = np.ones((4, 30), dtype=np.float64)
    z = row_rank_zscore(a)
    assert np.all(z == 0)


def test_row_rank_zscore_keeps_nan_as_zero():
    a = np.array([[1.0, 2.0, np.nan, 4.0]])
    z = row_rank_zscore(a)
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
        row_rank_zscore(a.to_numpy()),
        row_rank_zscore(a.to_numpy()),           # 与 0 相同
        row_rank_zscore((-a).to_numpy()),        # 与 0 完全反向
        row_rank_zscore(b.to_numpy()),           # 独立
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
    Z = np.stack([row_rank_zscore(panels[n].to_numpy()) for n in names], axis=1)
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
    """`a*2+1` 是**精确仿射变换** ⇒ 秩空间等价 ⇒ 必须拒。

    判据已从 ρ 换成秩空间残差（2026-11-12）：这里 ρ 也是 1.0，两种判据
    都会拒，但拒的**理由**不同——断言理由才能防止退回旧口径。
    """
    fl._DUP_INDEX_CACHE.clear()
    lib = FactorLibrary(root=tmp_path, dataset="t")
    a = _panel(seed=20)
    _build_tiny_index(lib, {"fa": a})
    with pytest.raises(ValueError, match="秩空间等价"):
        lib._dup_check_fast(a * 2 + 1, dup_corr=0.7, reject_dup=True, name="dup")
    fl._DUP_INDEX_CACHE.clear()


def test_dup_fast_path_allows_high_corr_with_increment(tmp_path):
    """🔑 ρ 高于 dup_corr 但秩空间有残差 ⇒ **不得**拒绝。

    这是 91 个误退的入库版同型错：旧实现 `top_corr > dup_corr` 就拒，
    而本库 |ρ|≥0.95 涉及 1142/950 个因子，等于把大量真信号挡在库外。
    """
    fl._DUP_INDEX_CACHE.clear()
    lib = FactorLibrary(root=tmp_path, dataset="t")
    a = _panel(seed=20)
    _build_tiny_index(lib, {"fa": a})
    rng = np.random.default_rng(99)
    noisy = a + 3.0 * pd.DataFrame(rng.normal(size=a.shape),
                                   index=a.index, columns=a.columns)
    row = lib._dup_check_fast(noisy, dup_corr=0.7, reject_dup=True, name="noisy")
    assert row is not None
    assert row["dup_verdict"] != "等价(秩空间退化)"
    fl._DUP_INDEX_CACHE.clear()


def test_dup_fast_path_rejects_via_signature_when_panel_missing(tmp_path):
    """🔑 最近邻没有独立面板时**仍须拒**等价——走索引签名兜底。

    真实库内 912 个批量入库（``merge_outputs``）的行 ``panel_path`` 为空，
    ``get_panel`` 返回 None。不兜底就等于「库内主体部分的最近邻一律放行」，
    等价检查**静默失效**，且日志长得像正常放行。
    """
    fl._DUP_INDEX_CACHE.clear()
    lib = FactorLibrary(root=tmp_path, dataset="t")
    a = _panel(seed=20)
    _build_tiny_index(lib, {"fa": a})          # 索引有签名，但**不写面板文件**
    assert lib.get_panel("fa") is None         # 确认走到兜底分支

    with pytest.raises(ValueError, match="秩空间等价"):
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


# ---------------------------------------------------------------------------
# 5. 判重必须覆盖**反向**重复（2026-11-12 复核）
# ---------------------------------------------------------------------------
def test_neighbors_abs_mode_catches_mirror_duplicate():
    """ρ=−1 的镜像对必须能被认出来。

    回归的是 2026-11-12 复核查出的 bug：``neighbors()`` 默认 ``argmax(corr)``
    只看正相关，``alpha158_SUMN30`` vs ``SUMP30``（精确 −1.000）的最近邻于是
    变成了另一个 ρ=+0.93 的因子 ⇒ max_corr=0.93 ⇒ 落进 T3/独立，整对逃过判重。
    ``_dup_check_fast`` 早已改成 ``argmax(|corr|)``，审计这条链当时漏了。
    """
    corr = np.array([[1.0, -1.0, 0.93],
                     [-1.0, 1.0, 0.10],
                     [0.93, 0.10, 1.0]], dtype=np.float32)
    names = ["mirror_a", "mirror_b", "other"]

    pos = neighbors(corr, names, top_k=1).set_index("name")
    assert pos.loc["mirror_a", "top"] == "other"          # 旧行为：看不到镜像
    assert pos.loc["mirror_a", "max_corr"] == pytest.approx(0.93, abs=1e-5)

    absn = neighbors(corr, names, top_k=1, abs_corr=True).set_index("name")
    assert absn.loc["mirror_a", "top"] == "mirror_b"      # 新行为：认出镜像
    assert absn.loc["mirror_a", "abs_corr"] == pytest.approx(1.0, abs=1e-5)
    # 符号仍保留在 max_corr 里，便于识别方向
    assert absn.loc["mirror_a", "max_corr"] == pytest.approx(-1.0, abs=1e-5)


def test_clusters_put_mirror_pair_together():
    """镜像对必须在同一簇。用 ``1−ρ`` 作距离时 ρ=−1 的距离是 2（最远）⇒ 分家。"""
    corr = np.array([[1.0, -1.0, 0.20],
                     [-1.0, 1.0, 0.15],
                     [0.20, 0.15, 1.0]], dtype=np.float32)
    lab = clusters(corr, ["a", "b", "c"], thresh=0.85).set_index("name")["cluster"]
    assert lab["a"] == lab["b"]
    assert lab["a"] != lab["c"]


# ---------------------------------------------------------------------------
# 6. 增量判据必须在**秩空间**算（2026-11-12 改口径）
# ---------------------------------------------------------------------------
def test_rank_resid_stats_kills_monotone_transform():
    """纯单调变换在秩空间必须判为退化（无增量）。

    回归的是 raw OLS 残差的假阳性：``alpha158_OPEN0`` 是 ``alpha101_033`` 的
    截面 rank（实测 2511 天逐日秩相关恒 = 1.000000），raw OLS 给出的却是
    t = −7.40 ⇒ 用 ``|t|≥2`` 当删除门槛会把它判成「有独立增量」而留下。
    """
    from scripts.factors.dup_audit import rank_resid_stats

    a = _panel(seed=7)
    rp = _rets(seed=8)
    # 严格单调但非线性：x**3。秩空间完全等价，原始值空间差很远。
    st = rank_resid_stats(a, {"base": (a ** 3)}, rp)
    assert st["rel_std"] < 1e-6
    assert st["degenerate"] is True

    # 镜像（×−1）同样必须退化
    st_neg = rank_resid_stats((-a), {"base": a}, rp)
    assert st_neg["rel_std"] < 1e-6
    assert st_neg["degenerate"] is True


def test_rank_resid_stats_detects_real_increment():
    """控制掉 base 后仍有残差的因子，不能被判成退化。"""
    from scripts.factors.dup_audit import rank_resid_stats

    rng = np.random.default_rng(11)
    a = _panel(seed=12)
    noise = pd.DataFrame(rng.normal(size=a.shape), index=a.index, columns=a.columns)
    mixed = a + 0.8 * noise          # 与 a 高度相关但**不是**它的单调变换
    st = rank_resid_stats(mixed, {"base": a}, _rets(seed=13))
    assert st["degenerate"] is False
    assert st["rel_std"] > 1e-3
    assert st["n_days"] >= 20


def test_rank_resid_stats_reports_insufficient_days_separately():
    """有效日不足必须返回 n_days<20 + NaN，**不能**返回 ic=0（那是「无增量」）。

    2026-11-12 的 8/8 假通过事故就是这条：退登记后 ``get_panel`` 返回 None，
    ``_resid_for`` 给 ``(nan, 0.0)``，0.0 被下游当成「无增量」。
    """
    from scripts.factors.dup_audit import rank_resid_stats

    a = _panel(n_dates=5, seed=14)
    rp = _rets(n_dates=5, seed=15)
    st = rank_resid_stats(a, {"base": _panel(n_dates=5, seed=16)}, rp)
    assert st["n_days"] == 0
    assert np.isnan(st["ic"])
    assert np.isnan(st["t"])
    assert st["degenerate"] is None


def test_rank_resid_stats_missing_panel_returns_nan_not_zero():
    """取不到面板时残差统计必须给 NaN，不许用 0.0 冒充结论。

    2026-11-12 的 8/8 假通过事故就是这条：退登记后 ``get_panel`` 返回 None，
    调用方把 ``0.0`` 读成「无增量」，于是什么都没测却全部判「可删」。
    这里直接测公开入口 ``rank_resid_stats``（空 DataFrame = 取不到面板）。
    """
    from scripts.factors.dup_audit import rank_resid_stats

    empty = pd.DataFrame()
    st = rank_resid_stats(empty, {"cover": empty}, empty)
    assert np.isnan(st["rel_std"])
    assert np.isnan(st["ic"])
    assert st["n_span"] == 0
    # 关键：不能是 0.0 —— 那会被下游当成「残差为 0 ⇒ 等价 ⇒ 可删」
    assert st["rel_std"] != 0.0


def test_degenerate_reports_zero_ic_days_but_positive_span():
    """精确退化时 ``n_days``(IC 有效日)=0 但 ``n_span``(参与回归的截面) 必须 >0。

    这是 2026-11-12 第二个坑：退化只在 IC 之前就能判（rel_std 恒 ~1e-15），
    若把两者混成一个字段，下游「n_days<20 ⇒ 无法检验」会把**精确等价**判成
    「有独立增量 ⇒ 必须留」，方向完全反了（实测 34 个）。
    """
    from scripts.factors.dup_audit import rank_resid_stats

    a = _panel(seed=21)
    st = rank_resid_stats(a, {"base": a}, _rets(seed=22))
    assert st["degenerate"] is True
    assert st["n_span"] >= 20          # 回归跑过了
    assert st["n_days"] == 0           # 但没有 IC 可算（残差恒 0）


def test_monotone_transform_with_full_span_still_degenerate_under_stride():
    """即使抽稀（stride），精确退化也必须仍被判为退化。

    抽稀提速的前提就是「退化是严格等式、与抽哪几天无关」；这条测试锁住这个前提。
    """
    from scripts.factors.dup_audit import rank_resid_stats

    a = _panel(n_dates=600, seed=23)
    rp = _rets(n_dates=600, seed=24)
    ran = a.rank(axis=1)               # 纯单调变换
    for stride in (1, 10, 20):
        st = rank_resid_stats(ran, {"base": a}, rp, stride=stride)
        assert st["degenerate"] is True, f"stride={stride} 漏判退化"
        assert st["rel_std"] < 1e-6
        assert st["n_span"] >= 30, f"stride={stride} 抽样后截面太少"
