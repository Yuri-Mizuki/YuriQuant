"""分段拉取全A历史状态表（涨跌停/停牌/ST/除权标记）——增量断点续拉版。

SDK 对大代码清单的单次查询会硬崩宿主进程且无 traceback（2026-08-28 实证：
5550 只单查挂死；分批拉至 3000 只时进程静默消失）。对策：

- 按 200 只/批拉取，**每批完成即合并落盘**——进程再崩也不丢进度；
- 启动时读现有 parquet，已覆盖 (code, date 区间完整) 的代码自动跳过；
- 单批 3 次重试，仍失败则跳过（最后统一报告缺口，重跑本脚本即可补）。

用法:
    python scripts/ingest/fetch_status_batched.py [--batch 200] [--begin 20220101] [--end 20251231]
"""

from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402


log = setup_logging("fetch_status")

from config import Config  # 缓存根单一真源（原硬编码 e:/data/parquet）  # noqa: E402

CACHE_PATH = Path(str(Config.cache()["root"])) / "history_stock_status.parquet"

def _covered_codes(df: pd.DataFrame, begin: int, end: int) -> set[str]:
    """已有缓存中，日期覆盖达到 begin~end 目标交易日 80% 的代码。

    注：``--begin`` 传"缺口首日"（而非全历史起点）时，区间内本地行数为 0
    （< 交易日数 × 0.8），池内所有 code 都会被判为待拉——这正是 update_data
    的增量调用方式（缺口首日由 `_status_incremental_begin` 算出）。
    """
    if df.empty:
        return set()
    dates = pd.to_datetime(df.index.get_level_values(0))
    in_range = (dates >= pd.Timestamp(str(begin))) & (dates <= pd.Timestamp(str(end)))
    sub = df[in_range]
    per_code = sub.groupby(level="code").size()
    approx_days = len(pd.bdate_range(str(begin), str(end)))
    ok = per_code[per_code >= approx_days * 0.8]
    return set(ok.index)


def _parts_dir() -> Path:
    """分片目录（与主表同盘，避免跨盘拷贝）。"""
    d = CACHE_PATH.parent / "_status_parts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _consume_parts(base: pd.DataFrame) -> pd.DataFrame:
    """把所有分片合并进主表并清空分片目录，返回合并后的表。

    **一次 O(n)**。旧写法是每批 ``concat(全表) + ~duplicated() + sort_index()``
    各一份全量拷贝，30 批即 O(n²)——1191 万行表（内存 1.1GB）下每批峰值 3GB+，
    实测本脚本在拉取首日即被 OOM 杀掉、主进程连带退出（2026-09-16 19:58 复现）。
    开头调用可消费上次被 kill 残留的分片，结尾调用合并本次全部。
    """
    files = sorted(_parts_dir().glob("part_*.parquet"))
    if not files:
        return base
    frames = [base] + [pd.read_parquet(f) for f in files]
    merged = pd.concat(frames, axis=0)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_parquet(CACHE_PATH)
    for f in files:
        f.unlink()
    return merged


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
    # 先消费上次被 kill 残留的分片（一次 O(n) 合并，而非每批一次）
    old = _consume_parts(old)
    covered = _covered_codes(old, args.begin, args.end)
    todo = [c for c in all_codes if c not in covered]
    log.info("全A %d 只 | 已覆盖 %d | 待拉 %d 只（%d/批）",
             len(all_codes), len(covered), len(todo), args.batch)
    if not todo:
        log.info("全部已覆盖，无需拉取")
        return

    parts = _parts_dir()
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
        # 只写本批分片（O(批大小)）。被 kill 也不丢——下次运行开头自动合并；
        # 旧写法每批 concat 全表，30 批 O(n²) 且在首日即 OOM（见 _consume_parts）。
        df.to_parquet(parts / f"part_{i:05d}.parquet")
        done += len(batch)
        log.info("batch %d-%d 落盘分片（本次累计 %d/%d）",
                 i, i + len(batch), done, len(todo))
        time.sleep(2)

    # 结束：一次性合并本次全部分片
    if done:
        old = _consume_parts(old)
        log.info("已合并全部增量，主表总行数 %d", len(old))

    log.info("完成：本次新增 %d 只，失败 %d 只%s", done, len(failed),
             ("：" + ",".join(failed[:20]) + ("..." if len(failed) > 20 else "")) if failed else "")
    if failed:
        log.info("重跑本脚本可自动补拉失败批次")

if __name__ == "__main__":
    main()