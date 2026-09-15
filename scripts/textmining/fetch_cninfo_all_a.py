"""
巨潮业绩预告缓存全A补爬
========================

把巨潮"业绩预告"缓存从 hs300/zz1000 覆盖补到全A（daily_all_a 代码表）。
利用 TextMiningCache.get_cninfo_announcements 的缺失 code 增量短路：传全A
代码表时只抓缓存中没有的 code（约 3,200 只）。

分批调用（每批 ~200 code）避免单次 2 小时爬取中断全损——每批结束合并缓存
落盘一次，中断后重跑自动跳过已覆盖 code。

用法：
    python -m scripts.textmining.fetch_cninfo_all_a [--begin 20160101] [--batch 200]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("cninfo_all_a")


def run(begin: int = 20160101, batch: int = 200):
    from config import Config
    from data.textmining.fetch import TextMiningCache

    cache_root = Path(str(Config.cache()["root"]).replace("//", "/"))
    codes = sorted(set(
        pd.read_parquet(cache_root / "daily_all_a.parquet", columns=[])
        .index.get_level_values("code").unique()))
    log.info("全A代码表 %d 只", len(codes))

    cache = TextMiningCache()
    n = len(codes)
    t0 = time.time()
    for i in range(0, n, batch):
        chunk = codes[i:i + batch]
        df = cache.get_cninfo_announcements(
            codes=chunk, begin_date=begin, categories=["业绩预告"])
        done = min(i + batch, n)
        rate = done / max(time.time() - t0, 1)
        log.info("进度 %d/%d（%s~%s，本批缓存后 %d 行），速率 %.1f code/s，剩余约 %.0f 分钟",
                 done, n, chunk[0], chunk[-1], len(df), rate,
                 (n - done) / max(rate, 0.05) / 60)
        time.sleep(1.0)
    log.info("补爬完成")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--begin", type=int, default=20160101)
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()
    run(args.begin, args.batch)
