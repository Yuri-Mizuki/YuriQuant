"""生产化监控调度 CLI —— 因子与模型预测的性能监控。

用法：
    # 单次监控（cron / Windows 计划任务每日调用）
    python scripts/monitor_performance.py --dataset hs300_2022_2025
    python scripts/monitor_performance.py --dataset hs300_2022_2025 --as-of 20251231

    # 首次生产化：把 h=1 模型预测（ml_synthesis 实验 OOS 面板）回写因子库
    python scripts/monitor_performance.py --register-model-factors

    # 常驻调度（每日 17:30 自动跑一轮，stdlib 循环无额外依赖）
    python scripts/monitor_performance.py --daemon 17:30

    # 打印 Windows 计划任务注册命令（替代 daemon 的系统级方案）
    python scripts/monitor_performance.py --task-cmd

产物（reports/monitoring/）：
    snapshots.csv / alerts.csv / monitor_report.html
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def register_model_factors(dataset: str = "hs300_2022_2025") -> list[str]:
    """把 ml_synthesis h=1 实验 OOS 预测面板注册为因子库 model:* 因子。

    生产链路闭环：模型预测（serving 回写）→ 与普通因子同库 → 统一监控。
    model_id 取 ModelRegistry 最新同名记录（note 双向溯源）。
    """
    from config import Config
    from model.labels import forward_returns
    from model.registry import ModelRegistry
    from model.serving import register_model_as_factor
    from monitoring.metrics import load_close_panel

    pred_dir = ROOT / "reports" / "ml_synthesis_h1"
    feats = pd.read_csv(pred_dir / "selected_features.csv")
    parents = feats.iloc[:, 0].tolist() if len(feats) else []

    close = load_close_panel(Path(Config.cache()["root"]))
    fwd = forward_returns(close, horizon=1)

    registry = ModelRegistry(ROOT / "reports" / "models")
    reg_models = registry.list()
    registered = []
    for method in ("ridge", "gbdt"):
        pred_path = pred_dir / f"pred_{method}_holdout.parquet"
        if not pred_path.exists():
            logging.warning("预测面板缺失，跳过: %s", pred_path)
            continue
        pred = pd.read_parquet(pred_path)
        hit = reg_models[reg_models["name"] == f"{method}_holdout_h1"]
        model_id = str(hit.iloc[-1]["model_id"]) if not hit.empty else None
        name = f"model:{method}_h1"
        register_model_as_factor(
            name=name,
            pred_panel=pred,
            returns_panel=fwd,
            parents=parents,
            dataset=dataset,
            model_id=model_id,
            horizon=1,
            oos=True,
            note="ml_synthesis 三段纪律 test 段 OOS 预测",
        )
        registered.append(name)
        logging.info("已注册模型因子: %s (model_id=%s)", name, model_id)
    return registered


# 真实数据监控必须用该系统 Python（自带 AmazingData SDK 与凭证）；
# TRAE VM 的 Python 3.10 缺 SDK 无法跑真实链路，故计划任务优先指向它。
SYSTEM_PY = Path(r"D:\python\Python312\python.exe")
SCHEDULED_TASK = "YuriQuant Monitor"


def _run_python(python_path: str | None = None) -> Path:
    if python_path:
        return Path(python_path).resolve()
    if SYSTEM_PY.exists():
        return SYSTEM_PY
    return Path(sys.executable).resolve()


def task_scheduler_cmd(dataset: str = "hs300_2022_2025", python_path: str | None = None) -> str:
    """构造 schtasks 每日 17:30 计划任务注册命令（真实数据固定用系统 Python 3.12）。"""
    py = _run_python(python_path)
    script = (ROOT / "scripts" / "monitor_performance.py").resolve()
    tr = f'\\"{py}\\" \\"{script}\\" --dataset {dataset}'
    return f'schtasks /Create /F /TN "{SCHEDULED_TASK}" /SC DAILY /ST 17:30 /TR "{tr}"'


def install_task(dataset: str = "hs300_2022_2025", python_path: str | None = None) -> str:
    """注册/更新 Windows 计划任务，返回 schtasks 输出。"""
    import subprocess

    cmd = task_scheduler_cmd(dataset=dataset, python_path=python_path)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return (f"cmd: {cmd}\n" + out.strip()).strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="因子与模型预测性能监控（生产化调度）")
    p.add_argument("--dataset", default="hs300_2022_2025", help="因子库数据集")
    p.add_argument("--window", type=int, default=None, help="近期短窗（默认 config，如 60d）")
    p.add_argument("--window-long", type=int, default=None,
                   help="近期长窗（默认 config window_long，如 252d，稳健确认）")
    p.add_argument("--as-of", default=None, help="监控基准日 YYYYMMDD（默认数据源最新日）")
    p.add_argument("--ledger-root", default=None, help="账本输出目录（默认 config）")
    p.add_argument(
        "--signal-path",
        default=None,
        help="每日交易信号 CSV（路径或 glob），启用组合/信号层监控",
    )
    p.add_argument(
        "--confirm-n",
        type=int,
        default=None,
        help="告警须连续触发 N 期才确认（默认 config confirm_n，研究期可设 1 免去抖）",
    )
    p.add_argument(
        "--register-model-factors",
        action="store_true",
        help="把 ml_synthesis h=1 OOS 预测回写因子库后退出",
    )
    p.add_argument(
        "--daemon",
        metavar="HH:MM",
        default=None,
        help="常驻模式：每日 HH:MM 自动跑一轮（Ctrl+C 退出）",
    )
    p.add_argument("--task-cmd", action="store_true", help="打印 Windows 计划任务注册命令后退出")
    p.add_argument("--install-task", action="store_true",
                   help="直接注册/更新 Windows 每日计划任务（默认系统 Python 3.12）")
    p.add_argument("--python", default=None, help="计划任务使用的 Python 解释器路径（覆盖默认）")
    args = p.parse_args(argv)

    if args.task_cmd:
        print(task_scheduler_cmd(args.dataset, args.python))
        return 0
    if args.install_task:
        print(install_task(args.dataset, args.python))
        return 0
    if args.register_model_factors:
        names = register_model_factors(args.dataset)
        print(f"registered: {names}")
        return 0

    from config import Config
    from monitoring.runner import next_run_time, run_monitoring

    # 计划任务/任意 CWD 下跑时，相对 ledger_root 也要落在项目根（reports/monitoring）
    # 下，而非随工作目录漂移（否则账本散落到 system32 等位置）。
    conf = Config.monitoring()
    ledger_root = args.ledger_root
    if not ledger_root:
        rel = conf["ledger_root"]
        ledger_root = str(ROOT / rel) if not Path(rel).is_absolute() else rel

    def _run() -> None:
        summary = run_monitoring(
            dataset=args.dataset,
            window=args.window,
            window_long=args.window_long,
            as_of=args.as_of,
            ledger_root=ledger_root,
            signal_path=args.signal_path,
            confirm_n=args.confirm_n,
        )
        print(
            f"[monitor] as_of={summary['as_of']} factors={summary['n_factors']} "
            f"models={summary['n_models']} critical={summary['n_critical']} "
            f"warning={summary['n_warning']} -> {summary['report_path']}"
        )

    if args.daemon:
        print(f"[monitor] daemon 启动：每日 {args.daemon} 跑一轮（Ctrl+C 退出）")
        try:
            while True:
                nxt = next_run_time(pd.Timestamp.now().to_pydatetime(), args.daemon)
                wait = (nxt - pd.Timestamp.now().to_pydatetime()).total_seconds()
                print(
                    f"[monitor] 下一轮: {nxt:%Y-%m-%d %H:%M:%S}（{wait / 3600:.1f}h 后）",
                    flush=True,
                )
                time.sleep(max(wait, 1))
                try:
                    _run()
                except Exception as e:  # noqa: BLE001
                    logging.error("本轮监控失败（下一轮继续）: %s", e)
        except KeyboardInterrupt:
            print("\n[monitor] daemon 已停止")
        return 0

    _run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
