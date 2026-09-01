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
