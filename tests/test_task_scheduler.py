"""调度登记表与系统实际注册项的**漂移核对**单测（纯函数，不触碰系统）。

背景：``config/schedule.yaml`` 是注册唯一真源，但任务 XML 里的绝对路径/参数是**注册
时烧进去的**。只改表不重注册 ⇒ 系统仍跑旧口径（本项目历史最贵的一类事故）。
:func:`registry_drift` 把 ``query_task_xml`` 回读的字段与登记项比对，此文件锁住行为。
"""

from __future__ import annotations

from pathlib import Path

from scripts.common.task_scheduler import (
    _check_altdata_tables,
    _hhmm,
    _norm_path,
    _rc_hint,
    registry_drift,
)

_TASK = {
    "key": "alla_daily_rank",
    "name": "YuriQuant AllaDailyRank",
    "command": r"D:\python\Python312\python.exe",
    "arguments": r'"E:\YuriQuant\scripts\pipelines\alla_daily_rank.py" --preset dingban',
    "working_dir": r"E:\YuriQuant",
    "time": "17:30",
    "execution_limit": "PT3H",
}

_INFO = {
    "command": r"D:\python\Python312\python.exe",
    "arguments": r'"E:\YuriQuant\scripts\pipelines\alla_daily_rank.py" --preset dingban',
    "working_directory": r"E:\YuriQuant",
    "start_boundary": "2026-10-10T17:30:00",
    "execution_time_limit": "PT3H",
}


def test_norm_path_unifies_slashes_and_case():
    assert _norm_path(r"E:\YuriQuant") == _norm_path("e:/YuriQuant/")
    assert _norm_path(None) == ""


def test_hhmm_extracts_clock_time():
    assert _hhmm("2026-10-10T17:30:00") == "17:30"
    assert _hhmm("") == ""
    assert _hhmm(None) == ""


def test_no_drift_when_exactly_matching():
    assert registry_drift(_TASK, _INFO) == []


def test_case_only_difference_is_not_drift():
    """Windows 路径/参数大小写不敏感 —— 不能因大小写差异误报脱钩。"""
    info = dict(_INFO, command=r"d:\PYTHON\python312\PYTHON.EXE")
    assert registry_drift(_TASK, info) == []


def test_detects_command_drift():
    info = dict(_INFO, command=r"E:\YuriQuant\.venv\Scripts\python.exe")
    drift = registry_drift(_TASK, info)
    assert len(drift) == 1 and "命令不一致" in drift[0]


def test_detects_arguments_drift():
    """改了登记表没重注册的典型形态：系统仍是旧口径（缺 --preset dingban）。"""
    info = dict(_INFO, arguments=r'"E:\YuriQuant\scripts\pipelines\alla_daily_rank.py"')
    drift = registry_drift(_TASK, info)
    assert len(drift) == 1 and "参数不一致" in drift[0]


def test_detects_time_drift():
    info = dict(_INFO, start_boundary="2026-10-10T18:00:00")
    drift = registry_drift(_TASK, info)
    assert len(drift) == 1 and "计划时间不一致" in drift[0]


def test_detects_working_dir_drift():
    info = dict(_INFO, working_directory=r"D:\YuriQuant")
    drift = registry_drift(_TASK, info)
    assert len(drift) == 1 and "工作目录不一致" in drift[0]


def test_detects_execution_limit_drift():
    info = dict(_INFO, execution_time_limit="PT1H")
    drift = registry_drift(_TASK, info)
    assert len(drift) == 1 and "执行时长上限不一致" in drift[0]


def test_empty_info_yields_no_drift():
    """回读失败（任务不存在/解析不了）不应把每一项都报成漂移。"""
    assert registry_drift(_TASK, {}) == []


def test_multiple_drifts_all_reported():
    info = dict(_INFO, command=r"C:\other\py.exe", start_boundary="2026-10-10T09:00:00")
    drift = registry_drift(_TASK, info)
    assert len(drift) == 2


# ---------------------------------------------------------------------------
# 产物判据（2026-10-29 判据改造）
# ---------------------------------------------------------------------------
def test_rc_hint_translates_known_bad_codes():
    # 0xC000013A（无符号 3221225786 / 有符号 -1073741510）都要翻成人话
    assert "0xC000013A" in _rc_hint("-1073741510")
    assert "0xC000013A" in _rc_hint("3221225786")
    # 正常码与未知码不加注解（doctor 的 rc 列已有原值）
    assert _rc_hint("0") == ""
    assert _rc_hint("12345") == ""


def _seed_altdata(root: Path, *, news_day: str, snapshot_day: str) -> None:
    """造一份最小 altdata 缓存：7 张表齐全，news 分片与快照 mtime 可指定。"""
    import os
    from datetime import datetime

    import pandas as pd

    (root / "alt_cninfo_holder.parquet").write_bytes(b"")
    pd.DataFrame({"ann_date": [20261001, 20261028]}).to_parquet(
        root / "alt_cninfo_holder.parquet", index=False)
    pd.DataFrame({"date": [pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-28")]}).to_parquet(
        root / "alt_macro_calendar.parquet", index=False)

    (root / "alt_cls").mkdir()
    (root / "alt_cls" / "cls_20261028.parquet").write_bytes(b"")
    (root / "alt_news").mkdir()
    (root / "alt_news" / f"news_{news_day}.parquet").write_bytes(b"")

    snap_ts = datetime.strptime(snapshot_day, "%Y%m%d").replace(hour=18).timestamp()
    for t in ("holder_control", "inner_trade", "mgmt_hold"):
        p = root / f"alt_{t}.parquet"
        p.write_bytes(b"")
        os.utime(p, (snap_ts, snap_ts))


def _patch_cache_root(monkeypatch, root: Path) -> None:
    from config import Config

    monkeypatch.setattr(Config, "cache", classmethod(lambda cls: {"root": str(root)}))


def test_altdata_check_passes_when_all_tables_fresh(tmp_path, monkeypatch):
    _patch_cache_root(monkeypatch, tmp_path)
    _seed_altdata(tmp_path, news_day="20261028", snapshot_day="20261028")
    problems, _notes, summary = _check_altdata_tables(20261028)
    assert problems == []
    assert summary == "alt 7/7 表 @20261028"


def test_altdata_check_flags_stale_news_shard(tmp_path, monkeypatch):
    """10-28 事故形态：前三张快照表写成功、链尾 news 分片缺失。

    ``_alt_meta.json`` 此时是新的（逐表增量写），旧判据只看它的 mtime ⇒ 报 OK。
    """
    _patch_cache_root(monkeypatch, tmp_path)
    _seed_altdata(tmp_path, news_day="20261026", snapshot_day="20261028")
    problems, notes, summary = _check_altdata_tables(20261028)
    assert summary == "alt 6/7 表 @20261028"
    assert len(problems) == 1 and "news" in problems[0] and "20261026" in problems[0]
    assert any("news 例外" in n for n in notes)


def test_altdata_check_flags_stale_snapshot_mtime(tmp_path, monkeypatch):
    """快照表每天全量重拉 ⇒ mtime 落后即"本轮没走到"（ann_date 会长期不更新，不能用）。"""
    _patch_cache_root(monkeypatch, tmp_path)
    _seed_altdata(tmp_path, news_day="20261028", snapshot_day="20261020")
    problems, _notes, _summary = _check_altdata_tables(20261028)
    assert len(problems) == 3
    assert all("本轮未重拉" in p for p in problems)


def test_altdata_check_flags_stale_data_date(tmp_path, monkeypatch):
    _patch_cache_root(monkeypatch, tmp_path)
    _seed_altdata(tmp_path, news_day="20261028", snapshot_day="20261028")
    import pandas as pd

    pd.DataFrame({"ann_date": [20261001]}).to_parquet(
        tmp_path / "alt_cninfo_holder.parquet", index=False)
    problems, _notes, _summary = _check_altdata_tables(20261028)
    assert len(problems) == 1 and "cninfo_holder" in problems[0] and "20261001" in problems[0]
