"""实验管理（experiments.csv）单元测试。"""
from __future__ import annotations

import pytest

from research.experiments import Experiments, default_experiments_path


@pytest.fixture
def exp(tmp_path):
    return Experiments(tmp_path / "experiments.csv")


def test_record_and_list(exp):
    rid = exp.record(kind="mining", command="python -m scripts.mine_factors --depth 2",
                     params={"depth": 2, "windows": [5, 10]}, data_fingerprint="abc123",
                     result_path="reports/x.csv", metrics={"ic_mean": 0.03, "n": 100})
    df = exp.list()
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == rid
    assert df.iloc[0]["kind"] == "mining"
    assert df.iloc[0]["params"] == {"depth": 2, "windows": [5, 10]}   # JSON 已解析
    assert df.iloc[0]["metrics"]["ic_mean"] == 0.03


def test_kind_filter_and_latest(exp):
    exp.record(kind="mining", metrics={"n": 1})
    exp.record(kind="synthesis", metrics={"n": 2})
    exp.record(kind="mining", metrics={"n": 3})
    mining = exp.list(kind="mining")
    assert len(mining) == 2
    assert mining.iloc[0]["metrics"]["n"] == 3          # 时间倒序
    latest = exp.latest(kind="mining")
    assert latest is not None and latest["metrics"]["n"] == 3
    assert exp.latest(kind="backtest") is None


def test_query_metrics(exp):
    exp.record(kind="gp", metrics={"ic_mean": 0.05, "n": 50})
    exp.record(kind="gp", metrics={"ic_mean": 0.08, "n": 60})
    exp.record(kind="mining", metrics={"ic_mean": 0.02, "n": 70})
    df = exp.query(kind="gp", metrics__ic_mean=0.08)
    assert len(df) == 1
    assert df.iloc[0]["metrics"]["n"] == 60


def test_append_does_not_overwrite(exp):
    exp.record(kind="mining", metrics={"n": 1})
    exp.record(kind="mining", metrics={"n": 2})
    assert len(exp.list()) == 2


def test_load_empty(tmp_path):
    exp = Experiments(tmp_path / "nope.csv")
    assert exp.list().empty
    assert exp.latest() is None


def test_default_path_anchored_to_project_root():
    """默认实验日志锚定项目根（绝对路径），保证计划任务等任意 CWD 下稳定写入。"""
    p = default_experiments_path()
    assert p.is_absolute()
    assert p.name == "experiments.csv"
    assert "YuriQuant" in str(p)          # 落在项目根 reports/ 下，而非随 CWD 漂移
