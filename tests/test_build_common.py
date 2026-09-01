"""cli_common 公共 CI/CD 骨架的单元测试。"""
import argparse

import numpy as np
import pandas as pd

from scripts.cli_common import (
    add_build_args, make_data_context, print_no_save, register_panels,
    returns_from_daily,
)


def _daily() -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=5, freq="B")
    codes = ["000001.SH", "000002.SH"]
    rng = np.random.default_rng(0)
    frames = []
    for c in codes:
        px = 10 + rng.uniform(-0.5, 0.5, len(idx)).cumsum()
        frames.append(pd.DataFrame({
            "code": c, "date": idx, "open": px, "close": px,
            "high": px + 0.1, "low": px - 0.1, "volume": 100.0, "amount": 1000.0,
        }))
    return pd.concat(frames, ignore_index=True).set_index(["date", "code"])


def test_returns_from_daily_matches_manual():
    daily = _daily()
    ret = returns_from_daily(daily)
    d = daily.reset_index()
    close_w = d.pivot(index="date", columns="code", values="close").sort_index()
    expected = close_w.pct_change().shift(-1)
    pd.testing.assert_frame_equal(ret, expected)
    # 最后一行为 NaN（shift(-1) 越界）
    assert ret.iloc[-1].isna().all()


def test_add_build_args_and_make_data_context_mock():
    parser = argparse.ArgumentParser()
    add_build_args(parser)
    args = parser.parse_args(["--mock"])
    cache, uni, begin, end, dataset = make_data_context(args)
    # mock 默认区间与数据集
    assert begin == 20230103
    assert end == 20241231
    assert dataset == "mock"
    assert cache is not None and uni is not None


def test_make_data_context_offline_defaults():
    parser = argparse.ArgumentParser()
    add_build_args(parser)
    args = parser.parse_args(["--offline"])
    _, _, begin, end, dataset = make_data_context(args)
    assert begin == 20250101
    assert end == 20251231
    assert dataset == "hs300_2025"


def test_make_data_context_override_dataset():
    parser = argparse.ArgumentParser()
    add_build_args(parser)
    args = parser.parse_args(["--mock", "--begin", "20240101", "--end", "20240301",
                              "--dataset", "my_ds"])
    _, _, begin, end, dataset = make_data_context(args)
    assert begin == 20240101 and end == 20240301 and dataset == "my_ds"


def test_register_panels_skips_empty_and_registers():
    idx = pd.date_range("2023-01-02", periods=5, freq="B")
    codes = ["A", "B"]
    good = pd.DataFrame(np.random.rand(5, 2), index=idx, columns=codes)
    empty = pd.DataFrame(np.nan, index=idx, columns=codes)
    panels = {"g": good, "e": empty}
    defs = {"g": "公式g", "e": "公式e"}

    registered = []
    class FakeLib:
        def register(self, **kw):
            registered.append(kw)
            return {"ic_mean": 0.05, "t_stat_nw": 2.0,
                    "best_sharpe": 1.0, "best_config": "x"}

    returns = pd.DataFrame(0.01, index=idx, columns=codes)
    rows = register_panels(FakeLib(), panels, defs, returns, source="test")
    # 只注册 g（e 无有效数据被跳过）
    assert len(registered) == 1
    assert registered[0]["name"] == "g"
    assert registered[0]["formula"] == "公式g"
    assert registered[0]["source"] == "test"
    assert registered[0]["kind"] == "raw"
    assert len(rows) == 1


def test_register_panels_names_subset():
    idx = pd.date_range("2023-01-02", periods=5, freq="B")
    codes = ["A", "B"]  # 单列截面 std 恒为 0，会被"恒定值因子"守卫跳过
    panels = {"a": pd.DataFrame(np.random.rand(5, 2), index=idx, columns=codes),
              "b": pd.DataFrame(np.random.rand(5, 2), index=idx, columns=codes)}
    defs = {"a": "fa", "b": "fb"}
    registered = []
    class FakeLib:
        def register(self, **kw):
            registered.append(kw["name"])
            return {"ic_mean": 0.1, "t_stat_nw": 1.0,
                    "best_sharpe": 0.5, "best_config": "x"}
    returns = pd.DataFrame(0.01, index=idx, columns=codes)
    register_panels(FakeLib(), panels, defs, returns, "t", names=["b"])
    assert registered == ["b"]


def test_register_panels_on_fail_callback():
    idx = pd.date_range("2023-01-02", periods=5, freq="B")
    codes = ["A", "B"]
    panels = {"a": pd.DataFrame(np.random.rand(5, 2), index=idx, columns=codes)}
    defs = {"a": "fa"}
    calls = []
    class FakeLib:
        def register(self, **kw):
            raise RuntimeError("boom")
    returns = pd.DataFrame(0.01, index=idx, columns=codes)
    rows = register_panels(FakeLib(), panels, defs, returns, "t",
                           on_fail=lambda n, e: calls.append((n, str(e))))
    assert len(rows) == 0
    assert calls == [("a", "boom")]


def test_print_no_save(caplog):
    import logging
    idx = pd.date_range("2023-01-02", periods=5, freq="B")
    panels = {"a": pd.DataFrame(1.0, index=idx, columns=["A"]),
              "b": None}
    with caplog.at_level(logging.INFO):
        print_no_save(["a", "b"], panels)
    assert "未入库" in caplog.text
