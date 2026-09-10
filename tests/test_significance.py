"""显著性判定收口的回归测试（``stats.significance``）。

锁死三件事：

1. 与收敛前的**内联实现数值逐字一致** —— 包括 ``ddof=1`` 这个最易踩的口径
   （``np.std`` 默认 ddof=0 而 pandas ``.std()`` 是 1；收敛前各调用方传的是
   pandas Series，所以统一实现必须用 ddof=1，否则所有 t 值会静默变化）；
2. BH-FDR 的语义（单调闭包、NaN 不参与族、空输入）；
3. 因子库的**两种显著性口径**各有显式入口 —— 单因子原始（registry 的
   ``significant`` 列）与整库 BH-FDR（``significance_table`` /
   ``load_significant_features(correction="fdr")``），且两者不相同。

真源：2026-09-10 第二批口径统一（详见 ``stats/significance`` 模块 docstring）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from stats.robust_stats import nw_tstat
from stats.significance import benjamini_hochberg, mean_inference, t_pvalue


def _ar1(n=300, phi=0.4, seed=0):
    """构造有自相关的序列——模拟真实 IC 序列（自相关正是要 NW 校正的原因）。"""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, 0.05)
    return x


# --------------------------------------------------------------------------
# 1) t_pvalue
# --------------------------------------------------------------------------
@pytest.mark.parametrize("t,df", [(2.0, 99), (-2.3465, 217), (0.0, 10), (5.1, 1121)])
def test_t_pvalue_matches_inline_formula(t, df):
    """与收敛前各处的内联写法 2*(1 - t.cdf(|t|, df)) 逐字一致。"""
    old = 2.0 * (1.0 - sps.t.cdf(abs(t), df=df))
    assert t_pvalue(t, df=df) == pytest.approx(old, abs=1e-15)


def test_t_pvalue_normal_approximation_when_df_none():
    """df=None → 标准正态近似（大样本下的常用退化）。"""
    assert t_pvalue(1.96, df=None) == pytest.approx(2.0 * sps.norm.sf(1.96), abs=1e-15)


def test_t_pvalue_edge_cases():
    assert np.isnan(t_pvalue(float("nan"), df=10))
    assert np.isnan(t_pvalue(2.0, df=0))
    # df 非有限 → 按文档退化为正态近似（不是 NaN）
    assert t_pvalue(2.0, df=float("nan")) == pytest.approx(t_pvalue(2.0, df=None), abs=1e-15)
    # 数组输入逐元素
    got = t_pvalue(np.array([0.0, 2.0, np.nan]), df=100)
    assert got.shape == (3,)
    assert got[0] == pytest.approx(1.0) and np.isnan(got[2])


# --------------------------------------------------------------------------
# 2) mean_inference —— 与收敛前的内联实现逐字段一致
# --------------------------------------------------------------------------
def test_mean_inference_matches_legacy_inline_implementation():
    """逐字复刻收敛前 factor/mining.py 的写法并比对。

    这是本批最关键的一条：``ddof`` 取错会让 t 值整体偏移，且不会有任何报错。
    """
    ic = pd.Series(_ar1())                      # 调用方传的一律是 pandas Series
    n, m, s = len(ic), float(ic.mean()), float(ic.std())   # pandas .std() = ddof=1
    t_old = m / (s / np.sqrt(n)) if s > 0 else 0.0
    p_old = 2.0 * (1.0 - sps.t.cdf(abs(t_old), df=n - 1))
    t_nw_old, _se, _lag = nw_tstat(ic)
    p_nw_old = 2.0 * (1.0 - sps.t.cdf(abs(t_nw_old), df=n - 1))

    inf = mean_inference(ic)
    assert inf["n"] == n
    assert inf["mean"] == pytest.approx(m, abs=1e-15)
    assert inf["std"] == pytest.approx(s, abs=1e-15)          # ddof=1 ✓
    assert inf["t_stat"] == pytest.approx(t_old, abs=1e-12)
    assert inf["p_value"] == pytest.approx(p_old, abs=1e-15)
    assert inf["t_stat_nw"] == pytest.approx(t_nw_old, abs=1e-12)
    assert inf["p_value_nw"] == pytest.approx(p_nw_old, abs=1e-15)


def test_mean_inference_std_is_ddof1_not_ddof0():
    """显式钉死 ddof：与 np.std 默认值不同，差值应为 sqrt(n/(n-1)) 倍。"""
    x = np.arange(1.0, 11.0)
    inf = mean_inference(x)
    assert inf["std"] == pytest.approx(np.std(x, ddof=1), abs=1e-15)
    assert inf["std"] != pytest.approx(np.std(x, ddof=0), abs=1e-15)


def test_mean_inference_robust_false_returns_nan_nw():
    """robust=False 走"不算 NW"分支，与旧实现的 (NaN, NaN, 0) 一致。"""
    inf = mean_inference(_ar1(), robust=False)
    assert np.isnan(inf["t_stat_nw"]) and np.isnan(inf["p_value_nw"]) and inf["lag"] == 0
    assert np.isfinite(inf["t_stat"]) and np.isfinite(inf["p_value"])


def test_mean_inference_constant_series_and_degenerate_inputs():
    """常数序列 → t=0 / p=1（旧实现的 `else 0.0`）；样本不足不抛异常。"""
    inf = mean_inference([1.0] * 50)
    assert inf["t_stat"] == 0.0 and inf["p_value"] == pytest.approx(1.0)

    one = mean_inference([0.3])
    assert one["n"] == 1 and np.isnan(one["t_stat"]) and np.isnan(one["p_value_nw"])
    empty = mean_inference([])
    assert empty["n"] == 0 and np.isnan(empty["t_stat"])


def test_mean_inference_drops_nan():
    """NaN 剔除后再算（与 pandas mean/std 的 skipna 语义一致）。"""
    a = mean_inference(pd.Series([0.1, np.nan, 0.2, 0.3]))
    b = mean_inference(pd.Series([0.1, 0.2, 0.3]))
    assert a["n"] == b["n"] == 3
    assert a["t_stat"] == pytest.approx(b["t_stat"], abs=1e-12)


# --------------------------------------------------------------------------
# 3) benjamini_hochberg
# --------------------------------------------------------------------------
def _legacy_bh(pvalues, q):
    """收敛前 factor/mining.py 的私有实现（逐字复刻，用于比对）。"""
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    k_max = 0
    for i in range(n):
        if sorted_p[i] <= (i + 1) * q / n:
            k_max = i + 1
    passed = np.zeros(n, dtype=bool)
    if k_max > 0:
        passed[order[:k_max]] = True
    return passed


@pytest.mark.parametrize("seed", range(5))
def test_benjamini_hochberg_matches_legacy(seed):
    pv = np.random.default_rng(seed).uniform(0, 1, 200)
    assert (benjamini_hochberg(pv, 0.05) == _legacy_bh(pv, 0.05)).all()


def test_benjamini_hochberg_monotone_closure():
    """拒绝集 = 最小的 k* 个 p（若 p_i 被拒绝，则所有 p_j < p_i 也被拒绝）。"""
    pv = np.array([0.001, 0.008, 0.02, 0.5, 0.9])
    got = benjamini_hochberg(pv, 0.05)
    # m=5 的阈值 k*q/m = 0.01/0.02/0.03/0.04/0.05 → 前三个满足（0.02<=0.03）→ k*=3
    assert got.tolist() == [True, True, True, False, False]


def test_benjamini_hochberg_excludes_nan_from_family():
    """NaN 不该参与检验，也不该让整族阈值变严（不计入 m）。"""
    pv = np.array([0.01, 0.02, np.nan])
    got = benjamini_hochberg(pv, 0.05)
    assert not got[2]
    # m=2（NaN 不计入）：阈值 0.025 / 0.05 → 两个都过
    assert got[:2].tolist() == [True, True]


def test_benjamini_hochberg_empty_and_none_significant():
    assert benjamini_hochberg([], 0.05).size == 0
    assert not benjamini_hochberg([np.nan, np.nan], 0.05).any()
    assert not benjamini_hochberg([0.4, 0.6, 0.9], 0.05).any()
    assert benjamini_hochberg([1e-9], 0.05).tolist() == [True]


# --------------------------------------------------------------------------
# 4) 因子库：单因子 raw 与整库 FDR 两个口径
# --------------------------------------------------------------------------
def _lib_with_factors(tmp_path, n_days=260, n_codes=40):
    from research.factor_library import FactorLibrary

    lib = FactorLibrary(root=tmp_path / "flib")
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    rets = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_codes)), idx, codes)
    returns_panel = rets                      # 库内口径：第 i 行 = i→i+1 收益
    # 强信号：未来收益 + 等量噪声 → IC 高且有截面波动（标准差非 0，t 值有限）
    strong = returns_panel + pd.DataFrame(rng.normal(0, 0.02, (n_days, n_codes)), idx, codes)
    lib.register("strong", strong, returns_panel, kind="raw", formula="strong")
    # 纯噪声
    lib.register("noise", pd.DataFrame(rng.normal(0, 1, (n_days, n_codes)), idx, codes),
                 returns_panel, kind="raw", formula="noise")
    return lib


def test_registry_row_records_p_value_nw(tmp_path):
    lib = _lib_with_factors(tmp_path)
    reg = lib.list_all().set_index("name")
    assert "p_value_nw" in reg.columns
    p = float(reg.loc["strong", "p_value_nw"])
    assert 0.0 <= p <= 1.0


def test_significance_table_fdr_is_not_raw(tmp_path):
    """整库 FDR 与单因子 raw 是两个口径：FDR 结果须等于对 registry 的 p 值重做 BH。"""
    lib = _lib_with_factors(tmp_path)
    tbl = lib.significance_table(q=0.05)
    assert {"name", "t_stat_nw", "p_value_nw", "raw_significant",
            "fdr_significant"} <= set(tbl.columns)
    assert len(tbl) == len(lib.list_all())

    reg = lib.list_all()
    expected = benjamini_hochberg(np.asarray(reg["p_value_nw"], dtype=float), 0.05)
    by_name = dict(zip(reg["name"], expected))
    assert all(bool(r["fdr_significant"]) == bool(by_name[r["name"]])
               for _, r in tbl.iterrows())


def test_load_significant_features_correction_switch(tmp_path):
    """raw / fdr 两个口径都能跑通，且 fdr 只加载被判显著的面板。"""
    lib = _lib_with_factors(tmp_path)

    raw = lib.load_significant_features(correction="raw")
    assert raw, "至少 strong 应通过单因子原始显著性"
    assert set(raw) <= set(lib.list_all()["name"])

    fdr = lib.load_significant_features(correction="fdr", q=0.05)
    keep = set(lib.significance_table(q=0.05).query("fdr_significant")["name"])
    assert set(fdr) == keep & set(raw) | set(fdr)   # fdr 只可能是 FDR 判定通过者
    assert set(fdr) <= keep

    with pytest.raises(ValueError):
        lib.load_significant_features(correction="bogus")
