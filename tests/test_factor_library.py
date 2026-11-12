"""
因子库单元测试（mock 数据，不依赖 SDK）。
"""
import numpy as np
import pandas as pd

from factor.synthesis import CompositeInput, synthesize_ic_weighted
from research.factor_library import FactorLibrary, _coerce_date


def _mock_panel(n_days=200, n_codes=30, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    # 一个有预测力的因子：滞后收益
    rets = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_codes)), idx, codes)
    factor = rets.shift(1)  # 昨日收益预测今日收益 → 有正 IC
    returns = rets
    return factor, returns


def _tmp_lib(tmp_path):
    return FactorLibrary(root=tmp_path / "flib")


def test_register_and_query(tmp_path):
    lib = _tmp_lib(tmp_path)
    factor, returns = _mock_panel()
    row = lib.register("f1", factor, returns, kind="raw", formula="f1")
    assert lib.has("f1")
    assert row["name"] == "f1"
    assert "ic_mean" in row
    # 面板可读取
    p = lib.get_panel("f1")
    assert p.shape == factor.shape


def test_evaluate_period_slices(tmp_path):
    lib = _tmp_lib(tmp_path)
    factor, returns = _mock_panel()
    lib.register("f1", factor, returns)
    # 全期
    full = lib.evaluate_period("f1")
    assert full["n_days"] > 0
    # 时间段切片
    sub = lib.evaluate_period("f1", start="20230301", end="20230630")
    assert sub["metrics"]["n_days"] <= full["metrics"]["n_days"]
    # 日期过滤生效：子区间交易日数少于全期
    assert sub["start"] == "2023-03-01 00:00:00"


def test_compare_ranking(tmp_path):
    lib = _tmp_lib(tmp_path)
    fa, ra = _mock_panel(seed=1)
    fb, rb = _mock_panel(seed=2)
    lib.register("fa", fa, ra)
    lib.register("fb", fb, rb)
    df = lib.compare(metric="sharpe")
    assert len(df) == 2
    # 列存在且按排序列（best_sharpe，compare 文档口径）降序
    assert "sharpe_ls_M" in df.columns
    assert "best_sharpe" in df.columns
    assert df["best_sharpe"].is_monotonic_decreasing


def test_composite_with_lineage(tmp_path):
    lib = _tmp_lib(tmp_path)
    fa, ra = _mock_panel(seed=1)
    fb, rb = _mock_panel(seed=2)
    lib.register("fa", fa, ra)
    lib.register("fb", fb, rb)
    comp = synthesize_ic_weighted([
        CompositeInput("fa", fa, ic=0.05),
        CompositeInput("fb", fb, ic=0.04),
    ])
    lib.register("composite_ic_weighted", comp, ra, kind="composite",
                 parents=["fa", "fb"], source="synthesis:ic_weighted")
    assert lib.has("composite_ic_weighted")
    assert lib.lineage("composite_ic_weighted") == ["fa", "fb"]
    comps = lib.list_composites()
    assert len(comps) == 1


def test_load_library_features_for_iteration(tmp_path):
    lib = _tmp_lib(tmp_path)
    fa, ra = _mock_panel(seed=1)
    lib.register("fa", fa, ra)
    feats = lib.load_library_features()
    assert "fa" in feats
    assert feats["fa"].shape == fa.shape


def test_delete(tmp_path):
    lib = _tmp_lib(tmp_path)
    fa, ra = _mock_panel(seed=1)
    lib.register("fa", fa, ra)
    assert lib.delete("fa")
    assert not lib.has("fa")
    assert not lib.delete("fa")  # 已删


def test_coerce_date():
    assert _coerce_date(20230301) == pd.Timestamp("2023-03-01")
    assert _coerce_date("2023-03-01") == pd.Timestamp("2023-03-01")
    assert _coerce_date(None) is None


def test_dataset_isolation(tmp_path):
    """按数据集分库根：不同 dataset 落到不同子目录，互不串扰，且默认库独立。"""
    base = tmp_path / "flib"
    lib_a = FactorLibrary(root=base, dataset="set_a")
    lib_b = FactorLibrary(root=base, dataset="set_b")
    fa, ra = _mock_panel(seed=1)
    fb, rb = _mock_panel(seed=2)
    lib_a.register("fa", fa, ra)
    lib_b.register("fb", fb, rb)
    # 互不串扰
    assert lib_a.has("fa") and not lib_a.has("fb")
    assert lib_b.has("fb") and not lib_b.has("fa")
    # 落到不同子目录
    assert (base / "set_a" / "registry.csv").exists()
    assert (base / "set_b" / "registry.csv").exists()
    # list_datasets 可见两者
    datasets = FactorLibrary.list_datasets(root=base)
    assert "set_a" in datasets and "set_b" in datasets
    # 默认（无 dataset）库独立、不含任一因子
    lib_default = FactorLibrary(root=base)
    assert len(lib_default.list_all()) == 0


def _write_ic_panels(lib, panels):
    """把 ic_h{h} 面板写到库根，模拟 builder 的 IC 融合产物。"""
    for h, df in panels.items():
        df.to_parquet(lib.root / f"ic_h{h}.parquet")


def test_ic_summary_reads_precomputed_panels(tmp_path):
    """ic_summary 直读库内 ic_h*.parquet，数值与手算一致（不重算 IC）。"""
    lib = _tmp_lib(tmp_path)
    idx = pd.date_range("2023-01-02", periods=10, freq="B")
    panels = {
        1: pd.DataFrame({"a": [0.1] * 10, "b": [-0.1] * 10}, idx),
        5: pd.DataFrame({"a": [0.2, 0.4] * 5, "b": [0.1, -0.3] * 5}, idx),
        10: pd.DataFrame({"a": [0.3] * 10, "b": [0.0] * 10}, idx),
        20: pd.DataFrame({"a": [0.5] * 10, "b": [0.2] * 10}, idx),
    }
    _write_ic_panels(lib, panels)

    s = lib.ic_summary()
    assert list(s.index) == ["a", "b"]
    # 均值逐位等于手算
    assert s.loc["a", "ic_h5"] == 0.30000000000000004 or abs(s.loc["a", "ic_h5"] - 0.3) < 1e-12
    assert abs(s.loc["b", "ic_h5"] - (-0.1)) < 1e-12
    # 正占比：b 在 h5 上 5/10 为正
    assert abs(s.loc["b", "pos_h5"] - 0.5) < 1e-12
    # 有效天数
    assert s.loc["a", "n_h20"] == 10
    # IR = mean / std
    assert abs(s.loc["a", "ir_h5"] - s.loc["a", "ic_h5"] / panels[5]["a"].std()) < 1e-12


def test_ic_matrix_window_and_names(tmp_path):
    """ic_matrix 支持窗口切片与列对齐（缺列补 NaN，不丢列）。"""
    lib = _tmp_lib(tmp_path)
    idx = pd.date_range("2023-01-02", periods=6, freq="B")
    _write_ic_panels(lib, {5: pd.DataFrame({"a": range(6)}, idx, dtype=float)})

    m = lib.ic_matrix(5)
    assert m.shape == (6, 1)
    m2 = lib.ic_matrix(5, names=["a", "ghost"])
    assert list(m2.columns) == ["a", "ghost"]
    assert m2["ghost"].isna().all()          # 缺列补 NaN，不 KeyError

    s = lib.ic_summary(window=("2023-01-03", "2023-01-04"))
    assert s.loc["a", "n_h5"] == 2           # 窗口内只有 2 个交易日


def test_ic_matrix_rejects_unknown_horizon(tmp_path):
    """非库内预计算的 horizon 必须显式报错，不得静默返回空表。"""
    import pytest

    lib = _tmp_lib(tmp_path)
    _write_ic_panels(lib, {5: pd.DataFrame({"a": [0.1]}, pd.date_range("2023-01-02", periods=1))})
    with pytest.raises(KeyError, match="ic_h3"):
        lib.ic_matrix(3)
    # 库内已登记的期限缺文件时返回 None + warning（不是崩）
    assert lib.ic_matrix(20) is None
