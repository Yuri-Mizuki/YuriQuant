"""调度登记表与系统实际注册项的**漂移核对**单测（纯函数，不触碰系统）。

背景：``config/schedule.yaml`` 是注册唯一真源，但任务 XML 里的绝对路径/参数是**注册
时烧进去的**。只改表不重注册 ⇒ 系统仍跑旧口径（本项目历史最贵的一类事故）。
:func:`registry_drift` 把 ``query_task_xml`` 回读的字段与登记项比对，此文件锁住行为。
"""

from __future__ import annotations

from scripts.common.task_scheduler import _hhmm, _norm_path, registry_drift

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
