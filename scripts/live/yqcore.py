"""yqcore-solo 生产入口：出榜 / 看持仓，两个独立子命令

为什么分成两个子命令而不是一条命令跑完
--------------------------------------
`yqcore-solo` 是**月频**策略，但两个功能的频率差 20 倍：

- ``rank``（出榜）：**每天**盘后跑 —— 用最新数据给全市场重新打分排序；
- ``holdings``（看持仓）：**一个月一期** —— 信号日 = 当月首个交易日，
  成交日 = 其次一交易日开盘，整月不变。

若合成一条命令每天跑，会天天生成内容相同的 ``holdings_<日期>.csv``，
把「当前生效的是哪一份」搅浑。所以：

- ``rank`` 会落盘（每日一份 ranking）；
- ``holdings`` **默认只读**（查当期，按最新价重估，不写任何文件），
  只有显式 ``--commit`` 才生成新一期持仓。

用法
----
    # 功能一：出榜（每天盘后）
    python -m scripts.live.yqcore rank
    python -m scripts.live.yqcore rank --date 20261109 --preset dingban

    # 功能二：看持仓（随时查，只读）
    python -m scripts.live.yqcore holdings
    python -m scripts.live.yqcore holdings --all
    python -m scripts.live.yqcore holdings --date 20261102

    # 换仓日：生成新一期持仓（有副作用）
    python -m scripts.live.yqcore holdings --commit

时序口径（别混）
----------------
2026-11 这一期：

- ``11-02``（周一，当月第 **1** 个交易日）= **信号日**，收盘后算分
- ``11-03``（周二，当月第 **2** 个交易日）= **成交日**，开盘买入
- ``12-01`` = 下一期信号日

即「信号日 = 当月首个交易日，调仓（成交）= 次一交易日开盘」。
见 ``scripts/evaluation/caliber.py::rebalance_dates``（``groupby(月).first()``）。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(module: str, argv: list[str], label: str) -> int:
    """以子进程跑子功能（两个既有脚本各有独立 CLI，不重复实现其逻辑）。"""
    cmd = [sys.executable, "-m", module, *argv]
    print(f"[{label}] {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


# ---------------------------------------------------------------------------
# 功能一：出榜
# ---------------------------------------------------------------------------
def cmd_rank(args: argparse.Namespace) -> int:
    argv: list[str] = []
    if args.date:
        argv += ["--date", args.date]
    else:
        argv += ["--skip-update"]
    argv += ["--preset", args.preset]
    if args.out_tag:
        argv += ["--out-tag", args.out_tag]
    if args.log_file:
        argv += ["--log-file", args.log_file]
    if args.strict_status:
        argv += ["--strict-status"]
    return _run("scripts.pipelines.alla_daily_rank", argv, "rank")


# ---------------------------------------------------------------------------
# 功能二：看持仓
# ---------------------------------------------------------------------------
def cmd_holdings(args: argparse.Namespace) -> int:
    if args.commit:
        # 生成新一期：需要榜单路径；缺省自动取 live 目录已记录的最新榜单
        from scripts.live import show_holdings as sh

        ds = args.date or sh.latest_period()
        if ds is None:
            print("live/ 下还没有任何持仓期次，无法推断用哪份榜单；"
                  "请显式给 --ranking 与 --date", file=sys.stderr)
            return 2
        ranking = args.ranking
        if not ranking:
            meta = sh.load_meta(ds)
            ranking = meta.get("ranking", "")
            if not ranking:
                print(f"meta_{ds}.json 里没有记录榜单路径，请显式给 --ranking",
                      file=sys.stderr)
                return 2
        argv = ["--ranking", str(ranking), "--date", ds]
        if args.capital:
            argv += ["--capital", str(args.capital)]
        if args.k:
            argv += ["--k", str(args.k)]
        return _run("scripts.evaluation.solo_live", argv, "holdings --commit")

    # 只读查询
    argv = []
    if args.date:
        argv += ["--date", args.date]
    if args.all:
        argv += ["--all"]
    return _run("scripts.live.show_holdings", argv, "holdings")


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="yqcore",
        description="yqcore-solo 生产入口（出榜 / 看持仓，两功能独立）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rank", help="功能一：出榜（每日，落盘 ranking_<ds>.csv）")
    r.add_argument("--date", default=None, metavar="YYYYMMDD",
                   help="预测日；缺省 = 缓存最新交易日（此时跳过数据更新）")
    r.add_argument("--preset", default="dingban",
                   choices=["legacy", "dingban", "v1w2-cheap"],
                   help="口径预设（定版 = dingban）")
    r.add_argument("--out-tag", default=None, help="产物目录后缀，如 _solo")
    r.add_argument("--log-file", default=None, help="日志文件路径")
    r.add_argument("--strict-status", action="store_true",
                   help="状态表断更时中止出榜而非降级")
    r.set_defaults(func=cmd_rank)

    h = sub.add_parser("holdings", help="功能二：看持仓（默认只读，随时可查）")
    h.add_argument("--date", default=None, metavar="YYYYMMDD",
                   help="信号日；缺省 = 最新一期")
    h.add_argument("--all", action="store_true", help="列出全部期次一览")
    h.add_argument("--commit", action="store_true",
                   help="生成新一期持仓（有副作用，仅换仓日用）")
    h.add_argument("--ranking", default=None, help="--commit 时的榜单路径")
    h.add_argument("--capital", type=float, default=None, help="--commit 时资金")
    h.add_argument("--k", type=int, default=None, help="--commit 时目标只数")
    h.set_defaults(func=cmd_holdings)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()