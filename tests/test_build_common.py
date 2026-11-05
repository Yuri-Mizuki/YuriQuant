"""cli_common 公共 CI/CD 骨架的单元测试。"""
import argparse

import numpy as np
import pandas as pd
import pytest

from scripts.common.cli_common import (
    add_build_args,
    make_data_context,
    print_no_save,
    register_panels,
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


# ---------------------------------------------------------------------------
# 入库闸（factor.validation 的 I/O 编排层）
# ---------------------------------------------------------------------------
def _mk_panel(values, codes=200):
    idx = pd.date_range("2024-01-02", periods=values.shape[0], freq="B")
    return pd.DataFrame(values, index=idx,
                        columns=[f"{i:06d}.SZ" for i in range(codes)])


def test_gate_stats_rows_rejects_flat_and_logs(tmp_path):
    """截面恒常数的面板被拒，且明细落 factor_stats_{family}_rejected.jsonl。"""
    from scripts.builders.common import gate_stats_rows

    (tmp_path / "panels").mkdir()
    rng = np.random.default_rng(0)
    _mk_panel(rng.normal(size=(100, 200))).to_parquet(tmp_path / "panels" / "good.parquet")
    _mk_panel(np.ones((100, 200))).to_parquet(tmp_path / "panels" / "const.parquet")

    kept, rejected = gate_stats_rows(
        [{"name": "good"}, {"name": "const"}], tmp_path, "demo")

    assert [r["name"] for r in kept] == ["good"]
    assert [r["name"] for r in rejected] == ["const"]
    assert rejected[0]["reject_reason"] == "flat"
    log = (tmp_path / "factor_stats_demo_rejected.jsonl").read_text(encoding="utf-8")
    assert "const" in log and "flat" in log


def test_gate_stats_rows_passes_missing_panel(tmp_path):
    """面板文件缺失不在这里拦（由 select 阶段的完整性检查负责）。"""
    from scripts.builders.common import gate_stats_rows

    (tmp_path / "panels").mkdir()
    kept, rejected = gate_stats_rows([{"name": "absent"}], tmp_path, "demo")
    assert [r["name"] for r in kept] == ["absent"]
    assert rejected == []
    assert not (tmp_path / "factor_stats_demo_rejected.jsonl").exists()


# ---------------------------------------------------------------------------
# 中性化批处理：单点写盘失败不打断整批（Defender 占用 57MB parquet 的现场）
# ---------------------------------------------------------------------------
def _mk_neu_ds(tmp_path, names):
    """造一个最小数据集：registry.csv + panels/。"""
    (tmp_path / "panels").mkdir()
    rng = np.random.default_rng(0)
    for n in names:
        pd.DataFrame(rng.normal(size=(10, 30)),
                     index=pd.date_range("2024-01-02", periods=10, freq="B"),
                     columns=[f"{i:06d}.SZ" for i in range(30)]
                     ).to_parquet(tmp_path / "panels" / f"{n}.parquet")
    pd.DataFrame({"name": names, "coverage": [0.9] * len(names)}).to_csv(
        tmp_path / "registry.csv", index=False)


def test_run_neutralization_continues_after_single_write_failure(tmp_path, monkeypatch):
    """一个因子写盘失败（文件被占用）不能让另外 19 个白跑。"""
    from scripts.builders import common

    names = [f"f{i:02d}" for i in range(20)]
    _mk_neu_ds(tmp_path, names)

    def fake(task):
        if task[0] == "f07":
            raise PermissionError(13, "Permission denied")
        return task[0], 0.9

    monkeypatch.setattr(common, "neutralize_one", fake)
    n = common.run_neutralization(tmp_path, "panels_neu", workers=1, force=True)

    assert n == 19                                    # 其余全部完成
    lists = list(tmp_path.glob("*_failed_*.txt"))     # 失败清单落盘（深检不会告诉你少了谁）
    assert len(lists) == 1
    assert "f07" in lists[0].read_text(encoding="utf-8")


def test_run_neutralization_aborts_on_massive_failure(tmp_path, monkeypatch):
    """大面积失败 = 系统性故障（路径/权限/磁盘），必须中断而不是写半旧半新。"""
    from scripts.builders import common

    _mk_neu_ds(tmp_path, ["a", "b", "c"])

    def fake(task):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(common, "neutralize_one", fake)
    with pytest.raises(RuntimeError, match="失败率"):
        common.run_neutralization(tmp_path, "panels_neu", workers=1, force=True)
