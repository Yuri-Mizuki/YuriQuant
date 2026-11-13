"""
因子库管理增强测试（2026-08-10）：
标签体系、入库冗余预检（相关性 + 正交残差 IC）、生命周期监控、分市场状态检验。
mock 数据，不依赖 SDK。
"""
import numpy as np
import pandas as pd
import pytest

from research.factor_library import FactorLibrary


def _mock_panel(n_days=200, n_codes=30, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    rets = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_codes)), idx, codes)
    factor = rets.shift(1)  # 昨日收益预测今日收益 → 正 IC
    return factor, rets


def _tmp_lib(tmp_path):
    return FactorLibrary(root=tmp_path / "flib")


def test_register_with_tags_and_dup_check(tmp_path):
    """入库自动查重：等价被拒、高相关但有增量放行。

    2026-11-12 起 ``check_dup`` 与 ``reject_dup`` **默认 True**（旧默认 False
    导致全库 ``dup_checked`` 0 行、门禁永远红）。判据也从 ρ 换成秩空间残差：
    这里 ``f2`` 是 ``factor*0.9+1e-6 噪声``，秩空间**精确等价** ⇒ 必拒。
    """
    lib = _tmp_lib(tmp_path)
    factor, returns = _mock_panel()
    # 注册带标签的因子
    row = lib.register("f1", factor, returns, kind="raw", formula="f1",
                       family="反转", frequency="日频", maturity="oos_verified",
                       note="mock 测试因子")
    assert row["family"] == "反转"
    assert row["maturity"] == "oos_verified"

    # ① 秩空间等价（同信号加微噪声，单调变换）⇒ 默认就被拒
    f2 = factor * 0.9 + np.random.default_rng(2).normal(0, 1e-6, factor.shape)
    f2 = pd.DataFrame(f2, index=factor.index, columns=factor.columns)
    with pytest.raises(ValueError, match="秩空间等价"):
        lib.register("f2_dup", f2, returns)
    # 显式关掉拒绝时才入库，且 dup 列写清结论
    row2 = lib.register("f2_dup", f2, returns, check_dup=True, reject_dup=False)
    assert row2["dup_checked"] is True or row2["dup_checked"] == "True"
    assert float(row2["dup_corr_max"]) > 0.7
    assert row2["dup_top"] == "f1"
    assert row2["dup_verdict"] == "等价(秩空间退化)"

    # ② 独立因子通过预检
    f_ind = pd.DataFrame(np.random.default_rng(3).normal(0, 1, factor.shape),
                         index=factor.index, columns=factor.columns)
    row3 = lib.register("f_ind", f_ind, returns, check_dup=True)
    assert float(row3["dup_corr_max"]) < 0.7


def test_high_corr_with_increment_is_not_rejected(tmp_path):
    """🔑 ρ 很高但秩空间有残差 ⇒ **不得**拒绝。这是 91 个误退的入库版同型错。"""
    lib = _tmp_lib(tmp_path)
    factor, returns = _mock_panel()
    lib.register("base", factor, returns)
    rng = np.random.default_rng(11)
    # 保留 base 的主要结构，但加入可观的重尾噪声 ⇒ 秩空间不再是单调变换
    noisy = factor + 3.0 * pd.DataFrame(
        rng.normal(size=factor.shape), index=factor.index, columns=factor.columns)
    row = lib.register("noisy", noisy, returns, check_dup=True, reject_dup=True)
    assert row["dup_checked"] is True or row["dup_checked"] == "True"
    # 判据是秩空间，不是 ρ：结论应为「有独立增量」而非「等价」
    assert row["dup_verdict"] != "等价(秩空间退化)"


def test_residual_ic_reports_incremental_info(tmp_path):
    """完全相同的因子：dup_corr_max≈1 且判为等价（残差无增量）。"""
    lib = _tmp_lib(tmp_path)
    factor, returns = _mock_panel()
    lib.register("f1", factor, returns)
    f2 = factor.copy()
    # 默认 reject_dup 已开 ⇒ 精确重复会被拒，这里显式放行以检查写入的列
    row = lib.register("f2_exact_dup", f2, returns, check_dup=True, reject_dup=False)
    assert float(row["dup_corr_max"]) > 0.99
    assert row["dup_verdict"] == "等价(秩空间退化)"


def test_set_tag(tmp_path):
    lib = _tmp_lib(tmp_path)
    factor, returns = _mock_panel()
    lib.register("f1", factor, returns)
    assert lib.set_tag("f1", family="动量", maturity="active", note="升级为 active")
    hit = lib.list_all()
    r = hit[hit["name"] == "f1"].iloc[0]
    assert r["family"] == "动量" and r["maturity"] == "active"
    assert not lib.set_tag("不存在", family="x")


def test_list_filter_by_tag(tmp_path):
    lib = _tmp_lib(tmp_path)
    factor, returns = _mock_panel()
    lib.register("f1", factor, returns, family="反转")
    lib.register("f2", factor * -1, returns, family="动量", maturity="retired")
    assert len(lib.list_all(family="反转")) == 1
    assert len(lib.list_all(maturity="retired")) == 1
    assert len(lib.list_all(family="反转", maturity="retired")) == 0


def test_monitor_status(tmp_path):
    lib = _tmp_lib(tmp_path)
    factor, returns = _mock_panel()
    lib.register("f1", factor, returns)
    df = lib.monitor(window=40)
    assert not df.empty
    assert {"name", "status", "ic_mean_full", "ic_mean_recent", "ic_drift"} <= set(df.columns)
    assert df["status"].isin(["normal", "warning"]).all()


def test_regime_analysis(tmp_path):
    lib = _tmp_lib(tmp_path)
    factor, returns = _mock_panel()
    lib.register("f1", factor, returns)
    market = returns.mean(axis=1)
    out = lib.regime_analysis("f1", market_returns=market)
    assert set(out.keys()) == {"熊/弱市", "震荡市", "牛/强市"}
    for v in out.values():
        assert v["n_days"] > 0
        assert "ic_mean" in v and "ir" in v
