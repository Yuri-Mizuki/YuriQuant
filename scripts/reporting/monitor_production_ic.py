"""生产口径每日 IC / 风格暴露监控 CLI（全A · ens_h1h5 · ortho 预处理）。

用法::

    # 跑一轮（读每日出榜产物 + 回测基线，落 reports/monitoring/production_ic_daily.csv）
    python scripts/reporting/monitor_production_ic.py

    # 只打印摘要（近期 vs 全期 IC，回答「单日噪音 or 持续退化」）
    python scripts/reporting/monitor_production_ic.py --summary-only

产物：
    reports/monitoring/production_ic_daily.csv   逐日 IC / 中性化 IC / 风格暴露

设计动机见 ``monitoring/production_ic.py`` 模块 docstring。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.cli_common import setup_logging  # noqa: E402

setup_logging("monitor_production_ic")

DEFAULT_OUT = "reports/monitoring/production_ic_daily.csv"


def _fmt(v, width: int = 8, digits: int = 4) -> str:
    return f"{v:>{width}.{digits}f}" if v == v else f"{'—':>{width}}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="生产口径每日 IC / 风格暴露监控")
    p.add_argument("--rank-dir", default="reports/alla_daily",
                   help="每日出榜产物目录（ranking_<ds>.csv）")
    p.add_argument("--hist-dir", default="reports/alla_rolling_ortho",
                   help="回测基线目录（pred/ + _base/）")
    p.add_argument("--cache-root", default=None, help="行情缓存根（默认 config）")
    p.add_argument("--out", default=DEFAULT_OUT, help="输出 CSV 路径")
    p.add_argument("--live-begin", default="20260601", help="实时区间面板起点")
    p.add_argument("--window", type=int, default=60, help="摘要窗口（交易日）")
    p.add_argument("--summary-only", action="store_true", help="只打印摘要，不落盘")
    p.add_argument("--no-overlap-check", action="store_true",
                   help="跳过实时协变量与 _base 的重叠一致性校验")
    args = p.parse_args(argv)

    from monitoring.production_ic import run_production_ic, summarize

    df = run_production_ic(
        rank_dir=args.rank_dir,
        hist_dir=args.hist_dir,
        cache_root=args.cache_root,
        out_path=None if args.summary_only else args.out,
        live_begin=args.live_begin,
        check_overlap=not args.no_overlap_check,
    )
    if df.empty:
        print("[prod-ic] 无可用截面（榜单缺失或行情未更新）")
        return 1

    s = summarize(df, window=args.window)
    last = df.iloc[-1]
    print(f"[prod-ic] 逐日 {len(df)} 行 · 有效 IC {s.get('n_days', 0)} 日 · "
          f"最新 predict_date={last['predict_date'].date()} "
          f"next={last['next_date'].date() if pd.notna(last['next_date']) else '—'} "
          f"src={last['source']}")
    print(f"[prod-ic] IC 全期 {_fmt(s.get('ic_raw_full'))} / 近{args.window}日 "
          f"{_fmt(s.get('ic_raw_recent'))}  NW-t {_fmt(s.get('ic_raw_t_nw'), 7, 2)}"
          f"   |  中性化IC 全期 {_fmt(s.get('ic_neutral_full'))} / 近{args.window}日 "
          f"{_fmt(s.get('ic_neutral_recent'))}")
    if "top_excess_full" in s:
        print(f"[prod-ic] Top10% 超额 全期 {_fmt(s['top_excess_full'], 7, 4)} / "
              f"近{args.window}日 {_fmt(s['top_excess_recent'], 7, 4)}")

    tail = df.tail(10)[["predict_date", "source", "n_eval", "ic_raw", "ic_neutral",
                        "style_exposure_ratio", "top_excess", "z_size", "z_vol",
                        "z_turn"]]
    print("\n最近 10 个截面：")
    print(tail.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
