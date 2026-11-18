"""定版因子库健康度巡检（因子 + 模型预测）。

对 registry 全部因子算时间序列健康度（IC 漂移、中性化 IC、覆盖率、分位单调性），
触发阈值告警并落账本 + HTML 报告。与因子看板的区别：看板回答「谁更强」（横截面、
离线全量），本巡检回答「原来强的那些还正常吗」（时间序列、按日推进、带告警）。

用法::

    # 日常（秒级到分钟级）：只算 IC 层，仅对告警因子补面板指标
    python -m scripts.reporting.monitor_factor_health

    # 全库体检（分钟级）：读全部因子面板，覆盖率类问题才能全库发现
    python -m scripts.reporting.monitor_factor_health --panel-mode all

    # 只看 IC 漂移，完全不碰面板（最快）
    python -m scripts.reporting.monitor_factor_health --panel-mode none

产物（``--ledger-root``，默认 ``reports/monitoring``）：
    snapshots.csv          每因子一行 × 每次运行
    alerts.csv             每条告警一行 × 每次运行
    factor_health_report.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.cli_common import setup_logging  # noqa: E402

setup_logging("monitor_factor_health")

DEFAULT_DATASET = "all_a_2018_2026"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="定版因子库健康度巡检（因子 + 模型）")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"因子库数据集（默认 {DEFAULT_DATASET}）")
    p.add_argument("--as-of", default=None, help="监控基准日 YYYYMMDD（默认库基准面板末日）")
    p.add_argument("--window", type=int, default=None,
                   help="近期短窗交易日（默认 config monitoring.window，60d）")
    p.add_argument("--window-long", type=int, default=None,
                   help="近期长窗交易日（默认 config monitoring.window_long，252d）")
    p.add_argument("--panel-mode", default="alerted", choices=["alerted", "all", "none"],
                   help="面板读取策略：alerted=只给已告警因子补（默认）/ all=全库 / none=不读")
    p.add_argument("--ic-source", default="auto", choices=["auto", "board", "evals"],
                   help="IC 序列来源：auto=看板缓存优先、evals 兜底（默认）")
    p.add_argument("--board-cache", default=None,
                   help="因子看板缓存目录（默认 reports/factor_board/cache）")
    p.add_argument("--style-neutral-scope", default="alerted",
                   choices=["alerted", "model", "all", "none"],
                   help="中性化 IC 范围：alerted=只算告警因子（默认）/ model=只算 model: 因子"
                        " / all=全库 / none")
    p.add_argument("--ledger-root", default=None, help="账本与报告输出目录")
    p.add_argument("--neutral-max", type=int, default=20,
                   help="中性化诊断的因子上限（默认 20；单因子约 24 秒）")
    p.add_argument("--no-record", action="store_true", help="不写 experiments 账本")
    p.add_argument("--json", action="store_true", help="以 JSON 打印汇总")
    args = p.parse_args(argv)

    from monitoring.runner import run_monitoring

    summary = run_monitoring(
        dataset=args.dataset,
        window=args.window,
        window_long=args.window_long,
        as_of=args.as_of,
        ledger_root=args.ledger_root,
        record=not args.no_record,
        style_neutral_scope=args.style_neutral_scope,
        ic_source=args.ic_source,
        board_cache_dir=args.board_cache,
        panel_mode=args.panel_mode,
        neutral_max=args.neutral_max,
    )

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"[health] 基准日 {summary['as_of']} · 数据集 {summary['dataset']} · "
              f"面板策略 {summary['panel_mode']} · IC 来源 {summary['ic_source']}")
        print(f"[health] 纳入 {summary['n_factors']} 个（模型 {summary['n_models']}）· "
              f"跳过 {summary['n_skipped_no_ic']}（无 IC 序列）· "
              f"补面板 {summary['n_panel_filled']} · 中性化诊断 "
              f"{summary.get('n_neutral_diagnosed', 0)}")
        print(f"[health] critical {summary['n_critical']} · warning {summary['n_warning']} · "
              f"normal {summary['n_normal']} · 去抖待观察 {summary['n_pending']}")
        if summary["data_stale"]:
            print(f"[health] ⚠ 数据滞后 {summary['data_lag_days']} 个工作日，"
                  f"本轮指标不代表当日")
        print(f"[health] 报告 {summary['report_path']}")

    return 2 if summary["n_critical"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
