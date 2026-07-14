"""
每日数据更新脚本
================

用法:
    python -m scripts.update_data              # 更新沪深300日K线
    python -m scripts.update_data --index 000905.SH  # 更新中证500
    python -m scripts.update_data --begin 20230101   # 指定起始日

说明:
    - 增量更新：只拉本地缺失的日期段。
    - 首次运行会从 config.fetch.begin_date 开始全量拉取。
    - 无 SDK 凭证时自动回退到 CSV 数据源（离线开发模式）。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 把项目根目录加入 sys.path，支持直接 python scripts/update_data.py 运行
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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


def main():
    parser = argparse.ArgumentParser(description="YuriQuant 数据更新")
    parser.add_argument("--index", default=None, help="指数代码，默认取 config.universe.index_code")
    parser.add_argument("--begin", type=int, default=None, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", type=int, default=None, help="结束日期 YYYYMMDD，默认至今")
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

    # 5. 复权因子
    log.info("拉取复权因子 ...")
    adj = cache.get_adj_factor(codes)
    log.info("复权因子行数: %d, 列数: %d", len(adj), adj.shape[1])

    log.info("数据更新完成。缓存目录: %s", cache.root)


if __name__ == "__main__":
    main()
