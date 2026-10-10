"""任务运行期间保持系统唤醒，防止被「现代待机」中断。

背景（2026-10-10 排查）
------------------------
YuriQuant 出榜（``YuriQuant AllaDailyRank``）/ 另类日更（``YuriQuant AltDataDaily``）
两个计划任务连续两周固定返回 ``0xC000013A``（= ``STATUS_CONTROL_C_EXIT``，进程被
强杀）、产物分别停在 09-22 / 09-23。按需触发（工作时段）时任务启动、跑动都正常
（``update_data`` 实时刷新缓存）⇒ 不是脚本层或任务条件问题。

查内核电源事件（``Microsoft-Windows-Kernel-Power``）发现本机只支持 **S0 现代待机**，
且傍晚 **18:30–19:43 固定进待机**（09-23 18:30 / 10-08 19:42 / 10-09 18:35、19:29）。
出榜任务 17:30 启动、全流程要跑 1–2h，正撞进待机窗口 —— 交互会话里的进程被挂起 /
终止，于是"任务已启动、结果码非 0、产物零更新"。

    ⚠️ 任务 XML 里的 ``DisallowStartIfOnBatteries`` / ``StopIfGoingOnBatteries``
    早已是 ``false``（2026-09-21 收口时显式写死），所以当日那句"电池条件致全灭"
    的归因对**本事故不适用**，别再照抄。

用法::

    from scripts.common.power import keep_system_awake

    with keep_system_awake():
        run(args)      # 期间系统不会因空闲进入待机

非 Windows 平台是 no-op（便于在 macOS / CI 上跑同一代码路径）。
"""
from __future__ import annotations

import contextlib
import sys

#: ``SetThreadExecutionState`` 标志位（winbase.h）
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _set_execution_state(flags: int) -> bool:
    """调 ``kernel32.SetThreadExecutionState``；非 Windows 返回 ``False``（no-op）。

    返回 ``True`` 仅表示调用成功（即"唤醒请求已生效"）。
    """
    if sys.platform != "win32":
        return False
    import ctypes

    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
        return True
    except Exception:  # pragma: no cover - 仅防御性，构造错误不该冒泡
        return False


@contextlib.contextmanager
def keep_system_awake():
    """上下文内请求系统保持唤醒（阻止空闲待机 / 睡眠），退出时恢复默认。

    必须在**长期存活的线程**上进入（``SetThreadExecutionState`` 是线程级的）——
    对项目单线程主流程即可。API 调用失败不抛异常（降级为"不阻止待机"），
    免得"防待机"这条保险本身变成新的失败点。

    产出（yield）一个 bool：``True`` = 唤醒请求已生效，``False`` = no-op/失败，
    便于调用方记日志。
    """
    active = _set_execution_state(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    try:
        yield active
    finally:
        if active:
            # 清掉 ES_SYSTEM_REQUIRED（只回写 ES_CONTINUOUS），恢复正常电源策略。
            _set_execution_state(_ES_CONTINUOUS)
