"""Windows 计划任务注册收口 + 调度体检（单一真源）。

为什么需要它
------------
2026-09-21 之前，任务注册逻辑在 ``alla_daily_rank`` / ``fetch_altdata_daily`` /
已归档的 ``daily_pipeline`` **三处各实现一遍**，且全部走 ``schtasks /Create`` 的
**裸参数**形式（``/SC DAILY /ST HH:MM /TR "..."``）。裸建的任务继承 Windows
默认设置，其中两条是致命的：

    <DisallowStartIfOnBatteries>true</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>true</StopIfGoingOnBatteries>

2026-09-21 实测：三个 YuriQuant 任务**同日同码**被杀
（``-1073741510`` = ``0xC000013A``），且分别使用系统 py312 与 ``.venv`` 两种
解释器 ⇒ 排除脚本/环境因素，指向任务宿主的统一终止。

本模块把注册收口到一处，并**改用 XML 定义导入**（``schtasks /Create /XML``）：

1. 显式写死上述两条为 ``false``，另设 ``StartWhenAvailable=true`` /
   ``ExecutionTimeLimit=PT3H`` / ``MultipleInstancesPolicy=IgnoreNew``；
2. 解释器 / 脚本 / 附加参数 / 工作目录集中登记在 ``config/schedule.yaml``，
   重注册不再是"照着文档手敲一行、漏一个参数就切口径"；
3. XML 文本节点里的引号**原样保留**，绕开 ``/TR`` 的**多层转义静默降级**坑
   （``fetch_altdata_daily._tr`` docstring 记录的那个：任务注册"成功"、列表也
   看得见，但运行时 Python 收到带反斜杠的路径直接失败）；
4. ``doctor`` 一条命令出体检表：登记↔系统漂移 / ``LastTaskResult`` / 产物新鲜度
   —— "任务显示已运行、结果码非 0、产物零更新"这种静默失败当场可见。

用法::

    from scripts.common.task_scheduler import install_task

    print(install_task(
        task_name="YuriQuant X",
        command=str(PY),
        arguments=f'"{SCRIPT}"',
        time_str="17:30",
        description="...",
        working_dir=str(ROOT),
    ))

CLI::

    python -m scripts.common.task_scheduler doctor            # 调度体检
    python -m scripts.common.task_scheduler list              # 系统里的 YuriQuant 任务
    python -m scripts.common.task_scheduler show <key>        # 打印登记项的 XML
    python -m scripts.common.task_scheduler install <key>     # 按登记表重注册
    python -m scripts.common.task_scheduler remove <key|任务名>
"""
from __future__ import annotations

import argparse
import glob as _glob
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as _esc

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 本项目任务名的统一前缀（doctor 反向漂移检测按它扫系统）
TASK_PREFIX = "YuriQuant"

#: 单一登记表
SCHEDULE_PATH = ROOT / "config" / "schedule.yaml"

_XML_NS = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"

__all__ = [
    "SCHEDULE_PATH",
    "TASK_PREFIX",
    "build_task_xml",
    "doctor",
    "install_task",
    "list_project_tasks",
    "load_schedule",
    "query_task_xml",
    "remove_task",
    "run_schtasks",
    "task_runtime_status",
]


# ---------------------------------------------------------------------------
# schtasks 调用与解码
# ---------------------------------------------------------------------------
def _decode(raw: bytes) -> str:
    """按编码逐个尝试解码 schtasks 输出。

    中文 Windows 下 ``/FO LIST`` 输出 GBK，``/XML`` 输出 UTF-16LE（带 BOM）。
    ``text=True``（默认 utf-8）会 UnicodeDecodeError —— 而任务其实已经建好了，
    异常栈把成功回显吞掉，看起来像注册失败（2026-09-17 踩过）。
    """
    if not raw:
        return ""
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16").lstrip("\ufeff")
    for enc in ("utf-8-sig", "utf-8", "gbk", "cp1252"):
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
        if "\x00" in text:  # 解出大量 NUL 说明选错了编码（典型是 UTF-16 被当 UTF-8）
            continue
        return text.lstrip("\ufeff")
    return raw.decode("utf-8", errors="replace")


def run_schtasks(cmd: str, *, timeout: int = 60) -> tuple[int, str]:
    """跑 schtasks，返回 ``(returncode, 解码后的输出)``。"""
    proc = subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)
    raw = (proc.stdout or b"") + (proc.stderr or b"")
    return proc.returncode, _decode(raw).strip()


# ---------------------------------------------------------------------------
# XML 构造
# ---------------------------------------------------------------------------
_DAILY_TRIGGER = """    <CalendarTrigger>
      <StartBoundary>{start}</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>{interval}</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>"""

_MINUTE_TRIGGER = """    <TimeTrigger>
      <StartBoundary>{start}</StartBoundary>
      <EndBoundary>{end}</EndBoundary>
      <Enabled>true</Enabled>
      <Repetition>
        <Interval>PT{minutes}M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
    </TimeTrigger>"""

_TASK_TEMPLATE = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{description}</Description>
  </RegistrationInfo>
  <Triggers>
{triggers}
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>{execution_limit}</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{command}</Command>
      <Arguments>{arguments}</Arguments>{workdir}
    </Exec>
  </Actions>
</Task>
"""


def build_task_xml(
    task_name: str,
    command: str,
    arguments: str = "",
    time_str: str = "17:30",
    *,
    description: str = "",
    working_dir: str = "",
    execution_limit: str = "PT3H",
    schedule: str = "daily",
    days_interval: int = 1,
    every_minutes: int = 10,
    end_time: str = "21:00",
    start_date: str | None = None,
) -> str:
    """构造任务 XML（UTF-16 文本内容，由 :func:`install_task` 落盘）。

    关键是在 ``<Settings>`` 里把两条电池条件显式关掉 —— 这是 2026-09-21
    三个任务同日全灭的头号嫌疑，也是"裸建"与"XML 导入"的唯一实质差异。
    """
    day = start_date or datetime.now().strftime("%Y-%m-%d")

    def _hhmm(text: str) -> str:
        parts = str(text).strip().split(":")
        hh, mm = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        return f"{hh:02d}:{mm:02d}:00"

    if schedule == "minute":
        triggers = _MINUTE_TRIGGER.format(
            start=f"{day}T{_hhmm(time_str)}",
            end=f"{day}T{_hhmm(end_time)}",
            minutes=int(every_minutes),
        )
    elif schedule == "daily":
        triggers = _DAILY_TRIGGER.format(
            start=f"{day}T{_hhmm(time_str)}", interval=int(days_interval)
        )
    else:
        raise ValueError(f"未知 schedule={schedule!r}（可用 daily / minute）")

    workdir = ""
    if working_dir:
        workdir = f"\n      <WorkingDirectory>{_esc(str(working_dir))}</WorkingDirectory>"

    return _TASK_TEMPLATE.format(
        description=_esc(description or task_name),
        triggers=triggers,
        execution_limit=execution_limit,
        command=_esc(str(command)),
        arguments=_esc(str(arguments)),
        workdir=workdir,
    )


# ---------------------------------------------------------------------------
# 注册 / 删除 / 查询
# ---------------------------------------------------------------------------
def install_task(
    task_name: str,
    command: str,
    arguments: str = "",
    time_str: str = "17:30",
    *,
    description: str = "",
    working_dir: str = "",
    execution_limit: str = "PT3H",
    schedule: str = "daily",
    days_interval: int = 1,
    every_minutes: int = 10,
    end_time: str = "21:00",
    dry_run: bool = False,
) -> str:
    """注册/覆盖一个计划任务（XML 导入），返回可读回显。

    与旧的 ``schtasks /Create /SC DAILY /TR`` 相比，寄存器里显式写入：
    电池不停、非电池也启动、可用即补跑、3 小时上限、新实例忽略。
    注册后**自动回读**``<Arguments>``核对引号是否裸的（防多层转义静默降级）。
    """
    xml_text = build_task_xml(
        task_name,
        command,
        arguments,
        time_str,
        description=description,
        working_dir=working_dir,
        execution_limit=execution_limit,
        schedule=schedule,
        days_interval=days_interval,
        every_minutes=every_minutes,
        end_time=end_time,
    )

    if dry_run:
        return f"# dry-run：{task_name}\n{xml_text}"

    fd, tmp_s = tempfile.mkstemp(prefix="yq_task_", suffix=".xml")
    tmp = Path(tmp_s)
    try:
        with os.fdopen(fd, "w", encoding="utf-16") as f:
            f.write(xml_text)
        # ⚠️ 必须先关闭句柄再调 schtasks（2026-09-21 实测）：mkstemp 的 fd 若
        # 未关闭，schtasks 读 XML 时撞上「另一个程序正在使用此文件」rc=1 ——
        # 正是 PermissionError WinError 32 同源。fdopen 的 with 已确保关闭。
        cmd = f'schtasks /Create /F /TN "{task_name}" /XML "{tmp}"'
        rc, out = run_schtasks(cmd)
    finally:
        # 占用降级（Windows WinError 32）：杀毒/安全钩子（WorkBuddy sitecustomize
        # _safe_remove 等）可能短暂持有共享读句柄。注册结果是主线，删临时文件只是
        # 卫生动作 —— 失败只警告，不吞掉注册结果（与 os.replace 占用降级同型）。
        try:
            tmp.unlink(missing_ok=True)
        except PermissionError as e:
            print(f"[warn] 临时文件未能删除（占用，可忽略）: {tmp} ({e})")

    lines = [
        f"task: {task_name}",
        f"time: {time_str}（schedule={schedule}）",
        f"cmd : {cmd}",
        f"schtasks rc={rc}",
        out,
    ]

    info = query_task_xml(task_name)
    if info is None:
        lines.append("核对：注册后回读失败（任务可能不存在）")
        return "\n".join(lines)

    args_text = info.get("arguments", "")
    if '\\"' in args_text:
        lines.append(
            f"核对：[警告] <Arguments> 含转义引号 {args_text!r} —— "
            "运行时 Python 会收到带反斜杠的路径（见 fetch_altdata_daily._tr）"
        )
    else:
        lines.append(f"核对：Arguments = {args_text}  引号正常")

    lines.append(
        "条件：DisallowStartIfOnBatteries={} / StopIfGoingOnBatteries={} / "
        "StartWhenAvailable={} / ExecutionTimeLimit={}".format(
            info.get("disallow_start_if_on_batteries", "?"),
            info.get("stop_if_going_on_batteries", "?"),
            info.get("start_when_available", "?"),
            info.get("execution_time_limit", "?"),
        )
    )
    return "\n".join(lines)


def remove_task(task_name: str) -> str:
    """删除一个计划任务。"""
    cmd = f'schtasks /Delete /F /TN "{task_name}"'
    rc, out = run_schtasks(cmd)
    return f"cmd: {cmd}\nrc={rc}\n{out}"


def query_task_xml(task_name: str) -> dict[str, str] | None:
    """回读任务 XML，返回关心的字段；任务不存在或解析失败返回 None。"""
    rc, out = run_schtasks(f'schtasks /Query /TN "{task_name}" /XML')
    if rc != 0 or "<Task" not in out:
        return None
    try:
        root = ET.fromstring(out)
    except ET.ParseError:
        return None

    def _text(tag: str) -> str:
        el = root.find(f".//{_XML_NS}{tag}")
        return (el.text or "").strip() if el is not None and el.text else ""

    return {
        "command": _text("Command"),
        "arguments": _text("Arguments"),
        "working_directory": _text("WorkingDirectory"),
        "start_boundary": _text("StartBoundary"),
        "days_interval": _text("DaysInterval"),
        "enabled": _text("Enabled"),
        "disallow_start_if_on_batteries": _text("DisallowStartIfOnBatteries"),
        "stop_if_going_on_batteries": _text("StopIfGoingOnBatteries"),
        "start_when_available": _text("StartWhenAvailable"),
        "execution_time_limit": _text("ExecutionTimeLimit"),
        "multiple_instances": _text("MultipleInstancesPolicy"),
    }


_STATUS_KEYS = {
    "state": ("计划任务状态", "Status", "Scheduled Task State"),
    "last_run": ("上次运行时间", "Last Run Time"),
    "last_result": ("上次结果", "Last Result"),
    "next_run": ("下次运行时间", "Next Run Time"),
}


def task_runtime_status(task_name: str) -> dict[str, str]:
    """取任务运行态（状态 / 上次运行 / 上次结果 / 下次运行）。缺字段给 ``?``。"""
    rc, out = run_schtasks(f'schtasks /Query /TN "{task_name}" /FO LIST /V')
    if rc != 0:
        return {k: "?" for k in _STATUS_KEYS}
    raw: dict[str, str] = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        raw[key.strip()] = val.strip()
    out_map: dict[str, str] = {}
    for key, candidates in _STATUS_KEYS.items():
        out_map[key] = next((raw[c] for c in candidates if c in raw), "?")
    return out_map


def list_project_tasks(prefix: str = TASK_PREFIX) -> list[str]:
    """列出系统里登记在册的 ``\\<prefix>*`` 任务名（去掉前导反斜杠）。"""
    rc, out = run_schtasks("schtasks /Query /FO CSV /NH")
    if rc != 0:
        return []
    names = []
    for line in out.splitlines():
        line = line.strip()
        if not line or not line.startswith('"'):
            continue
        name = line.split('","')[0].strip('"').lstrip("\\")
        if prefix.lower() in name.lower():
            names.append(name)
    return sorted(names)


# ---------------------------------------------------------------------------
# 登记表 + 体检
# ---------------------------------------------------------------------------
def load_schedule(path: Path | str | None = None) -> list[dict]:
    """读 ``config/schedule.yaml`` 的 ``tasks`` 列表。"""
    import yaml

    p = Path(path) if path else SCHEDULE_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    tasks = data.get("tasks") or []
    if not isinstance(tasks, list):
        raise ValueError(f"{p} 的 tasks 必须是列表")
    return tasks


def _latest_product(pattern: str) -> tuple[str, datetime] | None:
    """取 glob 命中的最新文件。``pattern`` 可以是项目内相对路径或绝对路径。"""
    p = Path(pattern)
    pat = str(p) if p.is_absolute() else str(ROOT / pattern)
    hits = [Path(x) for x in _glob.glob(pat)]
    hits = [h for h in hits if h.is_file()]
    if not hits:
        return None
    newest = max(hits, key=lambda x: x.stat().st_mtime)
    return newest.name, datetime.fromtimestamp(newest.stat().st_mtime)


def _date_in_name(name: str) -> str | None:
    """从产物文件名里取 8 位日期（``ranking_20260918.csv`` → ``20260918``）。

    判产物新鲜度必须用**文件名里的日期**，不能用 mtime：产物常在被杀之后由
    人工/次日补跑生成（如 09-18 的榜在 09-19 18:19 落盘），mtime 会晚于交易日
    从而把"缺当天产物"漏判成正常。
    """
    m = re.search(r"(20\d{6})", name)
    return m.group(1) if m else None


def _latest_trading_day() -> int | None:
    try:
        import pandas as pd

        from config import Config

        root = Path(str(Config.cache()["root"]).replace("//", "/"))
        p = root / "calendar.parquet"
        if not p.exists():
            return None
        return int(pd.read_parquet(p)["date"].max())
    except Exception:
        return None


def doctor(*, path: Path | str | None = None, verbose: bool = True) -> int:
    """调度体检。返回 0 = 无异常；1 = 有需要处理的项目。"""
    tasks = load_schedule(path)
    registered = set(list_project_tasks())
    latest_day = _latest_trading_day()
    declared = {t.get("name") for t in tasks if t.get("name")}
    problems: list[str] = []
    rows: list[str] = []

    if verbose:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        day_txt = str(latest_day) if latest_day else "?"
        print(f"调度体检  {stamp}   最近交易日 {day_txt}")
        print("-" * 96)

    for t in tasks:
        key = t.get("key", "?")
        name = t.get("name", "")
        system = t.get("system", "schtasks")
        status = t.get("status", "active")

        if system == "workbuddy":
            aid = t.get("automation_id", "?")
            rows.append(f"[--] {key:<20} WorkBuddy 自动化 {aid}（脚本无法枚举，需人工核对）")
            continue

        exists = name in registered
        if status == "planned":
            mark = "[--]" if not exists else "[!!]"
            note = "未启用（登记为 planned，属预期）" if not exists else "标为 planned 却已注册"
            if exists:
                problems.append(f"{key}: 标为 planned 但系统里已存在")
            rows.append(f"{mark} {key:<20} plan   {note}")
            continue

        if status == "retired":
            mark = "[--]" if not exists else "[!!]"
            note = "已退休" if not exists else "已退休但系统里仍存在"
            if exists:
                problems.append(f"{key}: 标为 retired 但仍注册着")
            rows.append(f"{mark} {key:<20} ret    {note}")
            continue

        if not exists:
            problems.append(f"{key}: 登记为 active 但系统里没有该任务")
            rows.append(f"[!!] {key:<20} ---    登记 active 但系统未注册")
            continue

        rt = task_runtime_status(name)
        info = query_task_xml(name) or {}
        rc_raw = rt.get("last_result", "?")
        bad_rc = rc_raw not in ("0", "?")
        state = rt.get("state", "?")
        warn = bad_rc or ("已禁用" in state)

        # 条件核对：电池两条必须是 false（2026-09-21 全灭事故的根因）
        cond_bad = (
            info.get("disallow_start_if_on_batteries") == "true"
            or info.get("stop_if_going_on_batteries") == "true"
        )
        if cond_bad:
            problems.append(f"{key}: 任务条件仍带电池限制（裸建遗留）")
            warn = True

        prod_txt = "-"
        pat = t.get("product_glob")
        if pat:
            hit = _latest_product(pat)
            if hit is None:
                prod_txt = "无产物"
                problems.append(f"{key}: 产物 {pat} 一个都没有")
                warn = True
            else:
                fname, ts = hit
                prod_txt = f"{fname} @{ts:%m-%d %H:%M}"
                if latest_day:
                    day = _date_in_name(fname) or ts.strftime("%Y%m%d")
                    if day < str(latest_day):
                        prod_txt += f" (未覆盖 {latest_day})"
                        problems.append(
                            f"{key}: 最新产物日期 {day} < 最近交易日 {latest_day}"
                        )
                        warn = True

        rows.append(
            f"{'[!!]' if warn else '[OK]'} {key:<20} "
            f"{t.get('time', '?'):<6} {state:<6} rc={rc_raw:<12} {prod_txt}"
        )

    unexpected = sorted(registered - declared)
    for n in unexpected:
        problems.append(f"系统里存在未登记的 YuriQuant 任务：{n}")

    if verbose:
        for r in rows:
            print(r)
        print("-" * 96)
        if problems:
            print(f"发现 {len(problems)} 项需处理：")
            for p in problems:
                print(f"  - {p}")
        else:
            print("全部正常。")

    return 1 if problems else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> int:
    ap = argparse.ArgumentParser(description="Windows 计划任务注册收口 + 调度体检")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="调度体检（登记↔系统漂移 / 结果码 / 产物新鲜度）")
    sub.add_parser("list", help="列出系统里的 YuriQuant 任务")

    p_show = sub.add_parser("show", help="打印登记项的 XML")
    p_show.add_argument("key")
    p_show.add_argument("--time", default=None, help="覆盖登记的时间，如 17:30")

    p_ins = sub.add_parser("install", help="按登记表重注册")
    p_ins.add_argument("key")
    p_ins.add_argument("--time", default=None, help="覆盖登记的时间，如 17:30")

    p_rm = sub.add_parser("remove", help="删除任务（登记 key 或任务名）")
    p_rm.add_argument("key")

    args = ap.parse_args()

    if args.cmd == "doctor":
        return doctor()
    if args.cmd == "list":
        for n in list_project_tasks():
            print(n)
        return 0

    tasks = load_schedule()
    by_key = {t.get("key"): t for t in tasks}

    if args.cmd == "show":
        t = by_key.get(args.key)
        if t is None:
            print(f"登记表里没有 key={args.key}；可用：{sorted(k for k in by_key if k)}")
            return 2
        print(build_task_xml(
            t["name"],
            t["command"],
            t.get("arguments", ""),
            args.time or t.get("time", "17:30"),
            description=t.get("description", ""),
            working_dir=t.get("working_dir", ""),
            execution_limit=t.get("execution_limit", "PT3H"),
            schedule=t.get("schedule", "daily"),
            days_interval=int(t.get("days_interval", 1)),
            every_minutes=int(t.get("every_minutes", 10)),
            end_time=t.get("end_time", "21:00"),
        ))
        return 0

    if args.cmd == "install":
        t = by_key.get(args.key)
        if t is None:
            print(f"登记表里没有 key={args.key}；可用：{sorted(k for k in by_key if k)}")
            return 2
        print(install_task(
            t["name"],
            t["command"],
            t.get("arguments", ""),
            args.time or t.get("time", "17:30"),
            description=t.get("description", ""),
            working_dir=t.get("working_dir", ""),
            execution_limit=t.get("execution_limit", "PT3H"),
            schedule=t.get("schedule", "daily"),
            days_interval=int(t.get("days_interval", 1)),
            every_minutes=int(t.get("every_minutes", 10)),
            end_time=t.get("end_time", "21:00"),
        ))
        return 0

    if args.cmd == "remove":
        name = by_key.get(args.key, {}).get("name", args.key)
        print(remove_task(name))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
