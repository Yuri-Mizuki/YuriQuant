"""每日因子流水线编排 —— 数据拉取 → 因子计算 → 监控报告

一条命令完成三步：
    1. update_data     增量拉日线/复权因子到本地缓存
    2. extend_library  把 alpha101/alpha191/gp 因子面板延长到最新交易日
    3. monitor          重算监控指标 + 生成 HTML 报告

用法：
    # 手动跑一次
    python scripts/daily_pipeline.py

    # 常驻模式（每日 17:30 自动跑一轮）
    python scripts/daily_pipeline.py --daemon 17:30

    # 注册 Windows 计划任务（每日 17:30，用系统 Python 3.12）
    python scripts/daily_pipeline.py --install-task

    # 只跑前两步（不生成报告）
    python scripts/daily_pipeline.py --skip-monitor

    # 只跑监控（因子已是最新）
    python scripts/daily_pipeline.py --skip-data --skip-extend
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("daily_pipeline")

# 真实数据链路需要自带 AmazingData SDK 与凭证的解释器：
# 默认当前解释器；计划任务部署到专用机器时用环境变量 YQ_SYSTEM_PY 指定
# （原硬编码 D:\python\Python312 换机即失效，2026-08-29 改为可配置）。
SYSTEM_PY = Path(os.environ.get("YQ_SYSTEM_PY") or sys.executable)
SCHEDULED_TASK = "YuriQuant DailyPipeline"

DEFAULT_DATASET = "hs300_2022_2025"


def _python() -> Path:
    """返回真实链路应使用的 Python 解释器。"""
    if SYSTEM_PY.exists():
        return SYSTEM_PY
    return Path(sys.executable).resolve()


def _run_step(name: str, cmd: list[str], timeout: int = 1800) -> bool:
    """跑一个子步骤，实时打印输出，返回是否成功。"""
    log.info("[%s] 开始: %s", name, " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), timeout=timeout,
            capture_output=True, text=True, encoding="utf-8",
        )
    except subprocess.TimeoutExpired:
        log.error("[%s] 超时（%d秒）", name, timeout)
        return False
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        # SDK 的 warning 走 stderr，不是真错误
        print(proc.stderr, end="", flush=True)
    if proc.returncode != 0:
        log.error("[%s] 失败 exit=%d", name, proc.returncode)
        return False
    log.info("[%s] 完成", name)
    return True


def run_pipeline(
    dataset: str = DEFAULT_DATASET,
    skip_data: bool = False,
    skip_extend: bool = False,
    skip_monitor: bool = False,
    python_path: str | None = None,
) -> dict:
    """跑完整流水线，返回各步骤状态。"""
    py = str(_python() if python_path is None else Path(python_path).resolve())
    results = {}

    # Step 1: 拉数据
    if not skip_data:
        ok = _run_step("update_data", [
            py, "-m", "scripts.update_data", "--no-minute",
        ])
        results["update_data"] = "ok" if ok else "fail"
        if not ok:
            log.error("数据拉取失败，后续步骤可能不完整")
    else:
        results["update_data"] = "skip"

    # Step 2: 延长因子面板
    if not skip_extend:
        ok = _run_step("extend_library", [
            py, "-m", "scripts.extend_factor_library", "--offline",
            "--sets", "alpha101,alpha191,alpha158,alpha360,gp",
        ], timeout=3600)
        results["extend_library"] = "ok" if ok else "fail"
        if not ok:
            log.error("因子延长失败，监控报告可能不包含最新数据")
    else:
        results["extend_library"] = "skip"

    # Step 3: 监控报告
    if not skip_monitor:
        ok = _run_step("monitor", [
            py, "-m", "scripts.monitor_performance", "--dataset", dataset,
        ])
        results["monitor"] = "ok" if ok else "fail"
    else:
        results["monitor"] = "skip"

    return results


def _next_run_time(now: datetime, hhmm: str) -> datetime:
    hh, mm = (int(x) for x in hhmm.split(":")[:2])
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return target if target > now else target + timedelta(days=1)


def task_scheduler_cmd(dataset: str = DEFAULT_DATASET,
                       python_path: str | None = None,
                       run_time: str = "17:30") -> str:
    """构造 schtasks 每日定时注册命令。"""
    py = str(_python() if python_path is None else Path(python_path).resolve())
    script = (ROOT / "scripts" / "daily_pipeline.py").resolve()
    tr = f'\\"{py}\\" \\"{script}\\" --dataset {dataset}'
    return (
        f'schtasks /Create /F /TN "{SCHEDULED_TASK}" '
        f'/SC DAILY /ST {run_time} /TR "{tr}"'
    )


def install_task(dataset: str = DEFAULT_DATASET,
                 python_path: str | None = None,
                 run_time: str = "17:30") -> str:
    """注册/更新 Windows 计划任务，返回 schtasks 输出。"""
    cmd = task_scheduler_cmd(dataset, python_path, run_time)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return f"cmd: {cmd}\n" + out.strip().strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="每日因子流水线：数据 → 因子 → 监控")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="因子库数据集")
    p.add_argument("--skip-data", action="store_true", help="跳过数据拉取")
    p.add_argument("--skip-extend", action="store_true", help="跳过因子延长")
    p.add_argument("--skip-monitor", action="store_true", help="跳过监控报告")
    p.add_argument("--python", default=None, help="覆盖 Python 解释器路径")
    p.add_argument(
        "--daemon", metavar="HH:MM", default=None,
        help="常驻模式：每日 HH:MM 自动跑一轮（Ctrl+C 退出）",
    )
    p.add_argument(
        "--install-task", action="store_true",
        help=f"注册 Windows 计划任务（每日 17:30，任务名 {SCHEDULED_TASK}）",
    )
    p.add_argument(
        "--task-time", default="17:30",
        help="计划任务运行时间 HH:MM（默认 17:30）",
    )
    p.add_argument(
        "--task-cmd", action="store_true",
        help="打印 schtasks 注册命令后退出",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.task_cmd:
        print(task_scheduler_cmd(args.dataset, args.python, args.task_time))
        return 0

    if args.install_task:
        print(install_task(args.dataset, args.python, args.task_time))
        return 0

    def _run() -> dict:
        t0 = time.time()
        results = run_pipeline(
            dataset=args.dataset,
            skip_data=args.skip_data,
            skip_extend=args.skip_extend,
            skip_monitor=args.skip_monitor,
            python_path=args.python,
        )
        elapsed = time.time() - t0
        log.info("流水线完成（%.0f秒）: %s", elapsed, results)
        return results

    if args.daemon:
        print(f"[pipeline] daemon 启动：每日 {args.daemon} 跑一轮（Ctrl+C 退出）")
        try:
            while True:
                nxt = _next_run_time(datetime.now(), args.daemon)
                wait = (nxt - datetime.now()).total_seconds()
                print(
                    f"[pipeline] 下一轮: {nxt:%Y-%m-%d %H:%M:%S}"
                    f"（{wait / 3600:.1f}h 后）",
                    flush=True,
                )
                time.sleep(max(wait, 1))
                try:
                    _run()
                except Exception as e:  # noqa: BLE001
                    log.error("本轮流水线失败（下一轮继续）: %s", e)
        except KeyboardInterrupt:
            print("\n[pipeline] daemon 已停止")
        return 0

    results = _run()
    # 任一关键步骤失败则非零退出（计划任务可据此告警）
    failed = [k for k, v in results.items() if v == "fail"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
