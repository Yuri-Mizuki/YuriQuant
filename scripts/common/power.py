"""任务运行期间请求系统保持唤醒（**在本机 S0 连接待机下无效，见下**）。

背景（2026-10-10 排查）
------------------------
YuriQuant 出榜（``YuriQuant AllaDailyRank``）/ 另类日更（``YuriQuant AltDataDaily``）
两个计划任务连续两周固定返回 ``0xC000013A``（= ``STATUS_CONTROL_C_EXIT``，进程被
强杀）、产物分别停在 09-22 / 09-23。按需触发（工作时段）时任务启动、跑动都正常
（``update_data`` 实时刷新缓存）⇒ 不是脚本层或任务条件问题。

查内核电源事件（``Microsoft-Windows-Kernel-Power``）发现本机只支持 **S0 现代待机**，
且傍晚 18:30 之后频繁进待机（09-23 18:30 / 10-08 19:42 / 10-09 18:35、19:29）。
出榜任务要跑 1–2h，正撞进待机窗口 —— 交互会话里的进程被挂起 / 终止，于是
"任务已启动、结果码非 0、产物零更新"。

    ⚠️ 任务 XML 里的 ``DisallowStartIfOnBatteries`` / ``StopIfGoingOnBatteries``
    早已是 ``false``（2026-09-21 收口时显式写死），所以当日那句"电池条件致全灭"
    的归因对**本事故不适用**，别再照抄。

🚨 2026-10-29 实证：**本保险对 S0 连接待机无效**（10-10 事故的"修复①"从未生效）
--------------------------------------------------------------------------------
- ``powercfg /a``：本机**只有 S0 低电量待机（Modern Standby / 连接待机）**，
  S1/S2/S3「固件不支持」，休眠已禁用。
- ``SetThreadExecutionState`` **只对传统 S1–S3 睡眠模型有效**；S0 的 CS 过渡由系统
  Modern Standby 状态机接管，应用层这个请求压不住。
- 实测（10-28）：API **返回非 0（"成功"）**、接电源 97%，18:43 照样进待机。
  ⇒ **返回值只能证明"调用成功"，不能证明"待机被阻止"**，别拿它当验证。
- ``powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE`` = ``0x0``（从不）⇒
  触发源是 **CS 空闲判定**而非标准超时，应用层无法覆盖。
- 近 30 天 ``Kernel-Power 506/507`` **每约 13 分钟无输入就一次**（10-28 当天
  11:27 / 11:41 / 15:47 / 18:09 / 18:43）。
- 10-19~10-26 出榜偶尔能跑通，**只是因为傍晚恰好有人在用电脑**（键盘唤醒），
  不是本保险起作用 —— 典型的"结论正确但因果错误"。

⇒ **唯一真解在系统电源策略层，脚本层无解，不要在这里加代码**：
``reg add HKLM\\SYSTEM\\CurrentControlSet\\Control\\Power /v PlatformAoAcOverride
/t REG_DWORD /d 0``（管理员 + 重启，把机器降级为传统 S3），或在电源 UI 把
「接通电源时使设备进入睡眠」设为「从不」。与"无线网卡掉线致 altdata 缺表"
（``Netwtw14`` 7021）是**两个独立问题**，别混。

用法::

    from scripts.common.power import keep_system_awake

    with keep_system_awake() as awake:
        # awake 只表示"唤醒请求已提交"，不代表待机会被阻止（见上）
        run(args)

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

    返回 ``True`` 仅表示**调用成功**（唤醒请求已提交），**不代表待机会被阻止** ——
    本机是 S0 连接待机，这个 API 压不住（见模块头 2026-10-29 实证）。
    """
    if sys.platform != "win32":
        return False
    import ctypes

    try:
        ret = ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
        return bool(ret)  # 0 = 调用失败；非 0 = 上一次的执行状态
    except Exception:  # pragma: no cover - 仅防御性，构造错误不该冒泡
        return False


@contextlib.contextmanager
def keep_system_awake():
    """上下文内向系统请求保持唤醒；退出时恢复默认电源策略。

    必须在**长期存活的线程**上进入（``SetThreadExecutionState`` 是线程级的）——
    对项目单线程主流程即可。API 调用失败不抛异常（降级为"不阻止待机"），
    免得"防待机"这条保险本身变成新的失败点。

    产出（yield）一个 bool：``True`` = 唤醒请求**已提交**，``False`` = no-op/失败。
    ⚠️ ``True`` **不等于**"待机已被阻止"（S0 连接待机下仍会被挂起），调用方记日志
    时不要说成"生效"。
    """
    active = _set_execution_state(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    try:
        yield active
    finally:
        if active:
            # 清掉 ES_SYSTEM_REQUIRED（只回写 ES_CONTINUOUS），恢复正常电源策略。
            _set_execution_state(_ES_CONTINUOUS)
