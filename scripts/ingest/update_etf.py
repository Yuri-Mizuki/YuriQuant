"""
ETF 数据更新脚本
================

用法:
    python -m scripts.ingest.update_etf              # 用系统 Python 3.12（含 AmazingData SDK/凭证）
    python -m scripts.ingest.update_etf --begin 20180101
    python -m scripts.ingest.update_etf --end 20260826

说明:
    - 拉取 ``data.etf_universe.ETF_CANDIDATES`` 候选池的日K线与后复权因子，
      落盘 daily_etf.parquet（日K线 × ETF）与 backward_factor.parquet（共用）。
    - 只做 ETF 轮动所需的最小子集（不拉个股的财务/状态/行业等）。
    - 增量更新：只拉本地缺失的日期段（复用 DataCache）。
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.cli_common import setup_logging  # noqa: E402


import argparse  # noqa: E402

from config import Config  # noqa: E402
from data.cache import DataCache  # noqa: E402
from data.datasource import create_datasource  # noqa: E402
from data.etf_universe import ETF_CANDIDATES, ETF_TABLE  # noqa: E402


log = setup_logging("update_etf")


def _pool_audit(kline, codes: list[str], min_amount: float = 3e7,
                min_years: float = 3.0) -> None:
    """P0 入池体检：逐只打印覆盖年份 + 近 60 日均成交额，并标出未达标项。

    只做**报告**不做剔除：剔除名单由人根据输出决定后改 ``ETF_CANDIDATES``，
    避免脚本静默改变研究池（池变更必须留痕到代码与计划文档）。
    """
    if kline is None or kline.empty:
        log.warning("日K线为空，跳过入池体检")
        return
    now = kline.index.get_level_values("date").max()
    fail = []
    for code in codes:
        try:
            sub = kline.xs(code, level="code")
        except KeyError:
            log.warning("  %s 无任何行情", code)
            fail.append((code, "无行情"))
            continue
        first, last = sub.index.min(), sub.index.max()
        years = (now - first).days / 365.25
        amt60 = float(sub["amount"].tail(60).mean())
        ok_hist = years >= min_years
        ok_liq = amt60 >= min_amount
        if not (ok_hist and ok_liq):
            fail.append((code, f"years={years:.1f} amt60={amt60/1e8:.2f}亿"))
        log.info("  %s %s -> %s (%.1f年) amt60=%.2f亿 %s",
                 code, first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"),
                 years, amt60 / 1e8, "OK" if (ok_hist and ok_liq) else "FAIL")
    if fail:
        log.warning("入池体检未通过 %d 只（需人工决定是否剔除）: %s", len(fail), fail)


def main():
    parser = argparse.ArgumentParser(description="YuriQuant ETF 数据更新")
    parser.add_argument("--begin", type=int, default=None, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", type=int, default=None, help="结束日期 YYYYMMDD，默认至今")
    args = parser.parse_args()

    cfg = Config.get()
    cfg["fetch"]
    begin = args.begin or 20180101
    codes = list(ETF_CANDIDATES.keys())

    ds = create_datasource()
    cache = DataCache(ds)

    # 与 SDK 全量 ETF 代码表比对，识别候选池中不在册的代码
    try:
        all_etf = set(ds.get_code_list("EXTRA_ETF"))
    except Exception as e:  # noqa: BLE001
        log.warning("获取 EXTRA_ETF 代码表失败（不影响拉取候选池）: %s", e)
        all_etf = set()
    missing = [c for c in codes if all_etf and c not in all_etf]
    if missing:
        log.warning("候选池中未在 EXTRA_ETF 代码表命中的代码: %s", missing)

    # 增量的结束日期：日历最新交易日
    end = args.end
    if end is None:
        cal = cache.get_calendar(begin, None)
        end = cal[-1] if cal else None

    log.info("候选 ETF: %d 只，区间 %s -> %s", len(codes), begin, end)

    log.info("增量拉取 ETF 日K线 (pool=%s) ...", ETF_TABLE)
    kline = cache.get_daily_kline(codes, begin, end, pool=ETF_TABLE)
    n_code = len(kline.index.get_level_values("code").unique()) if len(kline) else 0
    log.info("日K线行数: %d, 代码数: %d", len(kline), n_code)
    _pool_audit(kline, codes)

    log.info("拉取后复权因子 ...")
    backward = cache.get_backward_factor(codes)
    log.info("后复权因子列数: %d", backward.shape[1] if not backward.empty else 0)

    log.info("ETF 数据更新完成。缓存目录: %s", cache.root)


if __name__ == "__main__":
    main()