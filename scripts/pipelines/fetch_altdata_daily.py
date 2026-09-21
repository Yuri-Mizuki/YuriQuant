"""另类数据每日抓取/存档入口（P0 Step 1a + Step 3）
=====================================================

一条链抓齐 P0 计划的数据块。**「免自建」与「可回补」是两件事，分开看**：

==========================  ==============================  ================  ============
表                          源                              历史              更新方式
==========================  ==============================  ================  ============
``holder_control``          巨潮（akshare）                  2010-12 起        全量快照
``inner_trade``             雪球（akshare）                  滚动 20.5 月      全量快照
``mgmt_hold``               巨潮（akshare）                  滚动 1 年         全量快照
``cninfo_holder``           **巨潮 p_sysapi1030（自建）**     **2010 起全历史**  **窗口回补**
``macro_calendar``          华尔街见闻（akshare）            2018-01 起        按日增量
``cls``                     **财联社原始端点（自建 6 行签名）** **≥ 2015-06**   **游标回补**
``news``                    同花顺 + 新浪（akshare 快照）    **无历史**        当日累积
==========================  ==============================  ================  ============

🚨 **``cninfo_holder`` 才是增减持的研究主表**（P0 Step 1b）：
akshare 的 ``stock_hold_management_detail_cninfo`` 把日期写死在函数体内，
只能拿 **1 年滚动**（即 ``mgmt_hold``）；自建直调后是 **2010 起全历史**
（增持 ≈16 万行 + 减持 ≈15 万行）。``mgmt_hold`` 保留作**交叉验证源**。
实现细节与三个接口怪癖见 :mod:`data.altdata.cninfo_holder`。

🚨 **``cls`` 是本 P0 里唯一「可回补历史」的新闻源，应作为新闻研究主表**
（2026-09-20 实测：``last_time`` 可往回翻页、``rn`` 上限 50、自带 ``level``
加红等级与 ``stock_list`` 关联个股）。``news``（同花顺/新浪）只有最新 20 条快照、
**丢了拿不回来**，仅作冗余备份。细节见 :mod:`data.altdata.source_cls`。

用法
----
    # 一次抓齐（事件表 + 日历 + 财联社前向增量）
    python scripts/pipelines/fetch_altdata_daily.py

    # 财联社全量回补（首次，≈240 万条 / 数小时，断点续传，可反复中断重跑）
    python scripts/pipelines/fetch_altdata_daily.py --tables cls --cls-backfill

    # 只更新财联社（每日增量，只用几页）
    python scripts/pipelines/fetch_altdata_daily.py --tables cls

    # 日历回补到 2018（≈2000 个交易日，断点续传，可分批）
    python scripts/pipelines/fetch_altdata_daily.py --tables macro_calendar --macro-begin 20180101

    # 注册计划任务（每日 18:00 全量 + 09:00-21:00 每 10 分钟快照新闻）
    python scripts/pipelines/fetch_altdata_daily.py --install-task 18:00
    python scripts/pipelines/fetch_altdata_daily.py --install-news-task 09:00 --every-minutes 10
    python scripts/pipelines/fetch_altdata_daily.py --remove-task

⚠️ **解释器**：本脚本依赖 akshare，装在 ``.venv``（**不是**系统 Python）。
计划任务的 TR 写 ``.venv/Scripts/python.exe`` 绝对路径；搬动仓库后须重新
``--install-task``（TR 内嵌绝对路径，与 ``alla_daily_rank`` 同一坑）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("fetch_altdata")

#: 计划任务名（三个：全量日更 / 财联社历史回补续跑 / 高频新闻快照）
TASK_NAME = "YuriQuant AltDataDaily"
CLS_BACKFILL_TASK = "YuriQuant AltDataClsBackfill"
NEWS_TASK_NAME = "YuriQuant AltDataNews"

#: 本脚本必须跑在装了 akshare 的解释器上
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"

SNAPSHOT_TABLES = ("holder_control", "inner_trade", "mgmt_hold")
ALL_TABLES = ("holder_control", "inner_trade", "mgmt_hold", "cninfo_holder",
              "macro_calendar", "cls", "news")


# ---------------------------------------------------------------------------
# 计划任务（2026-09-21 收口到 scripts.common.task_scheduler）
# ---------------------------------------------------------------------------
def install_task(time_str: str, tables: str = "all") -> str:
    """注册每日全量任务（默认盘后 18:00，出榜之后）。

    参数真源见 ``config/schedule.yaml`` 的 ``altdata_daily`` 条；注册改用 XML
    导入，顺带绕开旧的 ``/TR`` 多层转义坑：XML 文本节点里的引号原样保留，
    不会出现"注册成功但运行时路径带反斜杠"那种静默降级。
    """
    from scripts.common.task_scheduler import install_task as _install

    py = VENV_PY if VENV_PY.exists() else Path(sys.executable)
    script = (ROOT / "scripts" / "pipelines" / "fetch_altdata_daily.py").resolve()
    return _install(
        task_name=TASK_NAME,
        command=str(py),
        arguments=f'"{script}" --tables {tables}',
        time_str=time_str,
        description="YuriQuant 另类数据日更（akshare，必须 .venv 解释器）",
        working_dir=str(ROOT),
    )


def install_cls_backfill_task(time_str: str = "17:30", max_pages: int = 3000) -> str:
    """注册财联社历史回补续跑任务（默认 17:30，每日最多翻 ``max_pages`` 页）。

    为什么需要它：首次回补是「数小时级」的长任务，中途被杀是常态。
    ``backfill_cls`` 自带断点续传（``cls_backfill_cursor``），所以这个任务在
    回补期是「推进器」，回补完成后**自动变成空转**（游标已在 ``begin`` 之前
    ⇒ 首轮直接 ``reached_begin``，不做任何请求）⇒ 可长期挂着，不必删任务。
    """
    from scripts.common.task_scheduler import install_task as _install

    py = VENV_PY if VENV_PY.exists() else Path(sys.executable)
    script = (ROOT / "scripts" / "pipelines" / "fetch_altdata_daily.py").resolve()
    return _install(
        task_name=CLS_BACKFILL_TASK,
        command=str(py),
        arguments=(f'"{script}" --tables cls --cls-backfill '
                   f'--cls-max-pages {int(max_pages)}'),
        time_str=time_str,
        description="YuriQuant 财联社历史回补续跑推进器",
        working_dir=str(ROOT),
    )


def install_news_task(time_str: str, every_minutes: int = 10,
                      end_time: str = "21:00") -> str:
    """注册高频新闻快照任务（默认 09:00–21:00 每 10 分钟）。

    ⚠️ **引入财联社可回补源后，本任务的价值大幅下降**（见模块 docstring）：
    同花顺/新浪只有 20 条快照、无历史，10 分钟粒度仍会漏掉大量快讯；而财联社
    可按 ``last_time`` 精确回补到 2015。**默认不装**，仅在需要「多源旁证」或
    财联社端点失效时启用。
    """
    from scripts.common.task_scheduler import install_task as _install

    py = VENV_PY if VENV_PY.exists() else Path(sys.executable)
    script = (ROOT / "scripts" / "pipelines" / "fetch_altdata_daily.py").resolve()
    return _install(
        task_name=NEWS_TASK_NAME,
        command=str(py),
        arguments=f'"{script}" --tables news',
        time_str=time_str,
        description="YuriQuant 高频新闻快照（盘中每 10 分钟）",
        working_dir=str(ROOT),
        execution_limit="PT1H",
        schedule="minute",
        every_minutes=every_minutes,
        end_time=end_time,
    )


def remove_task(which: str = "all") -> str:
    from scripts.common.task_scheduler import remove_task as _remove

    table = {"all": [TASK_NAME, CLS_BACKFILL_TASK, NEWS_TASK_NAME],
             "daily": [TASK_NAME],
             "cls_backfill": [CLS_BACKFILL_TASK],
             "news": [NEWS_TASK_NAME]}
    return "\n".join(_remove(n) for n in table.get(which, [which]))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _check_interpreter() -> None:
    """提前检查 akshare 是否可用（缺失时给出可执行提示，而不是抓到一半炸）。"""
    try:
        import akshare  # noqa: F401
    except ImportError:
        log.error(
            "当前解释器 %s 没有 akshare。另类数据抓取必须用装了 altdata extra 的环境：\n"
            "  uv pip install -e \".[altdata]\" --python %s\n"
            "或直接指定：%s scripts/pipelines/fetch_altdata_daily.py",
            sys.executable, VENV_PY, VENV_PY)
        raise SystemExit(2)


def run_once(args) -> dict:
    from data.altdata import AltDataCache

    cache = AltDataCache(args.cache_root)
    tables = ALL_TABLES if args.tables == "all" else tuple(
        t.strip() for t in args.tables.split(",") if t.strip())
    unknown = [t for t in tables if t not in ALL_TABLES]
    if unknown:
        raise SystemExit(f"未知表 {unknown}；可用：{list(ALL_TABLES)} 或 all")

    out: dict = {"tables": {}, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for t in tables:
        t0 = time.time()
        try:
            if t == "holder_control":
                df = cache.get_holder_control(refresh=not args.no_refresh)
                info = {"rows": int(len(df))}
            elif t == "inner_trade":
                df = cache.get_inner_trade(refresh=not args.no_refresh)
                info = {"rows": int(len(df))}
            elif t == "mgmt_hold":
                df = cache.get_mgmt_hold(refresh=not args.no_refresh)
                info = {"rows": int(len(df))}
            elif t == "cninfo_holder":
                df = cache.get_cninfo_holder(refresh=not args.no_refresh,
                                             sleep=args.cninfo_sleep)
                info = {"rows": int(len(df)),
                        "stats": cache._meta.get("cninfo_holder_stats", {})}
            elif t == "macro_calendar":
                df = cache.get_macro_calendar(
                    begin=args.macro_begin, end=args.macro_end,
                    max_days=args.macro_max_days)
                miss = cache.macro_missing_forecast_days()
                info = {"rows": int(len(df)),
                        "done_days": len(cache._meta.get("macro_done_dates", [])),
                        "forecast_all_missing_days": int(len(miss)),
                        "missing_sample": miss["date"].tolist()[:5] if len(miss) else []}
            elif t == "cls":
                if args.cls_backfill:
                    st = cache.backfill_cls(
                        begin=args.cls_begin,
                        max_pages=args.cls_max_pages,
                        flush_every=args.cls_flush_every,
                        sleep=args.cls_sleep,
                        resume=not args.cls_no_resume)
                else:
                    st = cache.sync_cls_recent(
                        max_pages=args.cls_max_pages or 400,
                        sleep=args.cls_sleep)
                info = dict(st)
                info["coverage"] = cache.cls_coverage()
            elif t == "news":
                info = cache.append_news(sources=tuple(args.news_sources.split(",")))
            else:  # pragma: no cover
                continue
            info["seconds"] = round(time.time() - t0, 1)
            out["tables"][t] = info
            log.info("[altdata] %-16s OK  %s", t, info)
        except Exception as e:  # noqa: BLE001
            out["tables"][t] = {"error": f"{type(e).__name__}: {str(e)[:200]}",
                                "seconds": round(time.time() - t0, 1)}
            log.error("[altdata] %-16s FAIL %s: %s", t, type(e).__name__, str(e)[:200])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="另类数据每日抓取/存档")
    ap.add_argument("--tables", default="all",
                    help=f"逗号分隔或 all；可用：{','.join(ALL_TABLES)}")
    ap.add_argument("--cache-root", default=None, help="覆盖缓存根（默认 Config.cache().root）")
    ap.add_argument("--no-refresh", action="store_true",
                    help="事件表只读本地缓存、不重拉")
    ap.add_argument("--macro-begin", default=20180101, type=int,
                    help="宏观日历回补起点 YYYYMMDD（默认 20180101）")
    ap.add_argument("--macro-end", default=None,
                    help="宏观日历回补终点 YYYYMMDD（默认今天）")
    ap.add_argument("--macro-max-days", default=None, type=int,
                    help="本次宏观日历最多抓多少天（分批回补用）")
    ap.add_argument("--news-sources", default="ths,sina,em",
                    help="新闻快照源（默认 ths,sina,em；em 在本机代理下不可达）")
    ap.add_argument("--cls-backfill", action="store_true",
                    help="财联社**往回翻页回补历史**（默认只做前向增量）")
    ap.add_argument("--cls-begin", default="20150101",
                    help="财联社回补最早日期 YYYYMMDD（实测 ≥2015 有数据）")
    ap.add_argument("--cls-max-pages", default=None, type=int,
                    help="财联社本次最多翻多少页（1 页 = 50 条；限速/试跑用）")
    ap.add_argument("--cls-flush-every", default=20, type=int,
                    help="财联社每翻 N 页落盘一次（断点续传粒度）")
    ap.add_argument("--cls-sleep", default=0.25, type=float,
                    help="财联社每页间隔秒数（限速，防被封）")
    ap.add_argument("--cls-no-resume", action="store_true",
                    help="忽略续传水位，从 --cls-begin 之外重新翻（幂等，但更慢）")
    ap.add_argument("--cninfo-sleep", default=0.4, type=float,
                    help="巨潮增减持每次请求间隔秒数（不给间隔会被服务端 RST）")
    ap.add_argument("--loop", default=None, type=int, metavar="SECONDS",
                    help="常驻模式：每 N 秒跑一次（仅建议配合 --tables news）")
    ap.add_argument("--install-task", nargs="?", const="18:00", default=None,
                    metavar="HH:MM", help="注册每日全量计划任务并退出")
    ap.add_argument("--install-cls-backfill", nargs="?", const="17:30", default=None,
                    metavar="HH:MM", help="注册财联社历史回补续跑任务（回补完成后自动空转）")
    ap.add_argument("--cls-max-pages-per-run", default=3000, type=int,
                    help="回补续跑任务每次最多翻多少页（默认 3000，≈13 分钟）")
    ap.add_argument("--install-news-task", nargs="?", const="09:00", default=None,
                    metavar="HH:MM", help="注册高频新闻快照任务（有 cls 后一般不需要）")
    ap.add_argument("--every-minutes", default=10, type=int,
                    help="高频新闻任务的间隔分钟（默认 10）")
    ap.add_argument("--remove-task", action="store_true", help="删除计划任务并退出")
    ap.add_argument("--remove-which", default="all",
                    help="删除哪些：all / daily / cls_backfill / news")
    args = ap.parse_args()

    if args.install_task:
        print(install_task(args.install_task, tables=args.tables))
        return
    if args.install_cls_backfill:
        print(install_cls_backfill_task(args.install_cls_backfill,
                                        max_pages=args.cls_max_pages_per_run))
        return
    if args.install_news_task:
        print(install_news_task(args.install_news_task, every_minutes=args.every_minutes))
        return
    if args.remove_task:
        print(remove_task(args.remove_which))
        return

    _check_interpreter()

    if args.loop:
        log.info("常驻模式：每 %d 秒跑一次 tables=%s（Ctrl-C 退出）", args.loop, args.tables)
        while True:
            res = run_once(args)
            print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)
            time.sleep(args.loop)
    else:
        res = run_once(args)
        print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
