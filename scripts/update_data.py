"""
每日数据更新脚本
================

用法:
    python -m scripts.update_data              # 更新沪深300日K线
    python -m scripts.update_data --index 000905.SH  # 更新中证500
    python -m scripts.update_data --begin 20230101   # 指定起始日
    python -m scripts.update_data --minute 5         # 拉取 5 分钟K线
    python -m scripts.update_data --minute 1,5,15    # 拉取多档分钟K线
    python -m scripts.update_data --no-minute        # 跳过分钟K线（即使配置了）

说明:
    - 增量更新：只拉本地缺失的日期段。
    - 首次运行会从 config.fetch.begin_date 开始全量拉取。
    - 分钟K线默认按 config.minute.periods 拉取（默认 [5]），可用 --minute/--no-minute 覆盖。
    - 无 SDK 凭证时自动回退到 CSV 数据源（离线开发模式）。
"""
from __future__ import annotations

import argparse
import logging
import sys

from config import Config
from data.cache import DataCache
from data.datasource import create_datasource
from data.universe import Universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("update_data")


def _parse_minute_arg(raw: str | None) -> list[int] | None:
    """解析 --minute "1,5,15" → [1, 5, 15]；None 表示未指定（走配置）。

    档位合法性统一走 data.datasource.validate_minute_period（单一来源）。
    """
    if raw is None:
        return None
    from data.datasource import validate_minute_period
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [validate_minute_period(int(p)) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="YuriQuant 数据更新")
    parser.add_argument("--index", default=None, help="指数代码，默认取 config.universe.index_code")
    parser.add_argument("--begin", type=int, default=None, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", type=int, default=None, help="结束日期 YYYYMMDD，默认至今")
    parser.add_argument("--minute", default=None, help="拉取分钟K线档位，逗号分隔，如 5 或 1,5,15")
    parser.add_argument("--no-minute", action="store_true", help="跳过分钟K线拉取")
    args = parser.parse_args()

    # 1. 加载配置
    cfg = Config.get()
    fetch_cfg = cfg["fetch"]
    begin = args.begin or fetch_cfg["begin_date"]
    end = args.end or fetch_cfg.get("end_date")

    # 2. 创建数据源 + 缓存
    ds = create_datasource()
    cache = DataCache(ds)

    # 3. 确定股票池
    index_code = args.index or cfg["universe"]["index_code"]
    uni = Universe(cache)

    log.info("获取 %s 成分股 ...", index_code)
    # 用 end 日期获取当时的成分股；end 为 None 时用最新日历
    cal = cache.get_calendar(begin, end)
    if not cal:
        log.warning("交易日历为空，请检查数据源配置。")
        return
    target_date = end if end else cal[-1]
    codes = uni.get_constituent(index_code, target_date)
    log.info("成分股数量: %d", len(codes))

    # 4. 增量拉取日K线
    log.info("增量拉取日K线: %s -> %s", begin, target_date)
    kline = cache.get_daily_kline(codes, begin, target_date)
    log.info("日K线行数: %d, 代码数: %d", len(kline), kline.index.get_level_values("code").nunique())

    # 4.5 增量拉取分钟K线（日内研究）
    minute_periods = []
    if args.minute:
        minute_periods = _parse_minute_arg(args.minute)
    elif not args.no_minute:
        minute_periods = list(cfg.get("minute", {}).get("periods", [5]))
    minute_rows: dict[str, int] = {}
    for period in minute_periods:
        log.info("增量拉取 %d 分钟K线: %s -> %s", period, begin, target_date)
        mk = cache.get_minute_kline(codes, begin, target_date, period=period)
        n_codes = mk.index.get_level_values("code").nunique() if len(mk) else 0
        minute_rows[f"min{period}_rows"] = len(mk)
        log.info("%d 分钟K线行数: %d, 代码数: %d", period, len(mk), n_codes)

    # 5. 复权因子（单次复权因子 + 累积后复权因子）
    log.info("拉取复权因子 ...")
    adj = cache.get_adj_factor(codes)
    log.info("单次复权因子行数: %d, 列数: %d", len(adj), adj.shape[1])
    backward = cache.get_backward_factor(codes)
    log.info("后复权因子行数: %d, 列数: %d", len(backward), backward.shape[1])

    # 6. 历史涨跌停/停牌/ST 状态
    log.info("拉取历史涨跌停/停牌状态: %s -> %s", begin, target_date)
    status = cache.get_history_stock_status(codes, begin, target_date)
    log.info("历史状态行数: %d", len(status))

    # 7. 行业分类（因子行业中性化用）
    industry_level = int(cfg.get("preprocessing", {}).get("industry_level", 1))
    log.info("拉取行业分类 (level=%d) ...", industry_level)
    industry = cache.get_industry_classification(industry_level)
    log.info("行业分类行数: %d", len(industry))

    # 8. 股本结构（市值中性化用）
    log.info("拉取股本结构 ...")
    equity = cache.get_equity_structure(codes)
    log.info("股本结构行数: %d", len(equity))

    # 9. 财务三表（基本面因子 / 因子挖掘用）
    log.info("拉取利润表 ...")
    income = cache.get_income(codes)
    log.info("利润表行数: %d", len(income))

    log.info("拉取资产负债表 ...")
    balance = cache.get_balance_sheet(codes)
    log.info("资产负债表行数: %d", len(balance))

    log.info("拉取现金流量表 ...")
    cashflow = cache.get_cash_flow(codes)
    log.info("现金流量表行数: %d", len(cashflow))

    # 9b. 分红 / 十大股东 / 股东户数（股息率/股东类因子用，2026-08-04）
    log.info("拉取分红数据 ...")
    dividend = cache.get_dividend(codes)
    log.info("分红行数: %d", len(dividend))

    log.info("拉取十大股东 ...")
    share_holder = cache.get_share_holder(codes)
    log.info("十大股东行数: %d", len(share_holder))

    log.info("拉取股东户数 ...")
    holder_num = cache.get_holder_num(codes)
    log.info("股东户数行数: %d", len(holder_num))

    # 10. 实验记录（数据版本指纹 + 拉数摘要）
    try:
        from research.experiments import record_experiment
        fingerprint = cache.get_fingerprint()
        metrics = {"kline_rows": len(kline), "adj_cols": adj.shape[1],
                   "status_rows": len(status), "income_rows": len(income)}
        metrics.update(minute_rows)
        record_experiment(
            kind="data_update",
            command=" ".join(sys.argv),
            params={"index": index_code, "begin": begin, "end": target_date,
                    "n_codes": len(codes), "minute_periods": minute_periods},
            data_fingerprint=fingerprint,
            result_path=str(cache.root),
            metrics=metrics,
            note="数据更新完成",
        )
        log.info("数据指纹: %s", fingerprint)
    except Exception as e:
        log.warning("实验记录写入失败（不影响数据更新）: %s", e)

    log.info("数据更新完成。缓存目录: %s", cache.root)


if __name__ == "__main__":
    main()
