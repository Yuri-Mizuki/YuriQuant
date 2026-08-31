"""分段拉取全A历史状态表（涨跌停/停牌/ST/除权标记）——增量断点续拉版。

SDK 对大代码清单的单次查询会硬崩宿主进程且无 traceback（2026-08-28 实证：
5550 只单查挂死；分批拉至 3000 只时进程静默消失）。对策：

- 按 200 只/批拉取，**每批完成即合并落盘**——进程再崩也不丢进度；
- 启动时读现有 parquet，已覆盖 (code, date 区间完整) 的代码自动跳过；
- 单批 3 次重试，仍失败则跳过（最后统一报告缺口，重跑本脚本即可补）。

用法:
    python scripts/fetch_status_batched.py [--batch 200] [--begin 20220101] [--end 20251231]
"""

from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cli_common import setup_logging  # noqa: E402


log = setup_logging("fetch_status")

from config import Config  # 缓存根单一真源（原硬编码 e:/data/parquet）  # noqa: E402

CACHE_PATH = Path(str(Config.cache()["root"])) / "history_stock_status.parquet"

def _covered_codes(df: pd.DataFrame, begin: int, end: int) -> set[str]:
    """已有缓存中，日期覆盖达到 begin~end 目标交易日 80% 的代码。"""
    if df.empty:
        return set()
    dates = pd.to_datetime(df.index.get_level_values(0))
    in_range = (dates >= pd.Timestamp(str(begin))) & (dates <= pd.Timestamp(str(end)))
    sub = df[in_range]
    per_code = sub.groupby(level="code").size()
    n_days = pd.Timestamp(str(end)).dayofyear - pd.Timestamp(str(begin)).dayofyear
    approx_days = len(pd.bdate_range(str(begin), str(end)))
    ok = per_code[per_code >= approx_days * 0.8]
    return set(ok.index)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--begin", type=int, default=20220101)
    ap.add_argument("--end", type=int, default=20251231)
    args = ap.parse_args()

    from data.cache import DataCache
    from data.datasource import create_datasource
    from data.universe import Universe

    ds = create_datasource()
    cache = DataCache(ds)
    all_codes = Universe(cache).get_all_a(args.end)

    old = pd.read_parquet(CACHE_PATH) if CACHE_PATH.exists() else pd.DataFrame()
    covered = _covered_codes(old, args.begin, args.end)
    todo = [c for c in all_codes if c not in covered]
    log.info("全A %d 只 | 已覆盖 %d | 待拉 %d 只（%d/批）",
             len(all_codes), len(covered), len(todo), args.batch)
    if not todo:
        log.info("全部已覆盖，无需拉取")
        return

    merged = old.copy()
    done, failed = 0, []
    for i in range(0, len(todo), args.batch):
        batch = todo[i:i + args.batch]
        df = None
        for attempt in (1, 2, 3):
            try:
                df = ds.get_history_stock_status(batch, args.begin, args.end)
                break
            except Exception as exc:
                log.warning("batch %d 第 %d 次失败: %s", i, attempt, exc)
                time.sleep(5 * attempt)
        if df is None or len(df) == 0:
            failed.extend(batch)
            log.error("batch %d 放弃（%d 只）", i, len(batch))
            continue
        # 归一化为 (date, code) 多索引（SDK 实际返回平索引 + date/code 列）
        if "date" not in (df.index.names or []) and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index(["date", "code"]).sort_index()
        if not old.empty:
            df = df.reindex(columns=old.columns)      # 对齐历史 schema
        df = df[~df.index.duplicated(keep="last")]
        merged = pd.concat([merged, df])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.sort_index().to_parquet(CACHE_PATH)
        done += len(batch)
        log.info("batch %d-%d 落盘（本次累计 %d/%d，总行数 %d）",
                 i, i + len(batch), done, len(todo), len(merged))
        time.sleep(2)

    log.info("完成：本次新增 %d 只，失败 %d 只%s", done, len(failed),
             ("：" + ",".join(failed[:20]) + ("..." if len(failed) > 20 else "")) if failed else "")
    if failed:
        log.info("重跑本脚本可自动补拉失败批次")

if __name__ == "__main__":
    main()