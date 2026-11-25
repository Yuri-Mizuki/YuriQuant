"""全 A 分钟数据拉取（5 分钟 / 1 分钟）
=====================================

与 ``fetch_minute_gap``（写死 HS300 PIT 成分、只补缺口）不同，本脚本面向
**任意池**：按全 A 日线里窗口内曾交易的代码取清单，按「年 × 代码块」分批请求，
每批落一个独立分片，支持断点续传，最后一次性合并。

为什么要分片而不是直接追加单表
------------------------------
长表的 O(n²) 陷阱：全 A 5 分钟一年约 6,300 万行（1.2 GB），若每批都
read-modify-write 单表，137 个代码块要重写 137 次（≈164 GB IO/年）。
所以先落到 ``_staging/<year>/chunk_XXXX.parquet``（≈9 MB/块），
``--compact`` 时每年只合并一次。

产物
----
    <cache>/intraday/_staging/min{period}_all_a/<year>/chunk_*.parquet   中间分片
    <cache>/intraday/min{period}_all_a_parts/part_<year>.parquet          按年合并
    <cache>/min{period}_all_a.parquet                                     单表（--compact --single；⚠️ 全 A 规模 concat ~50GB 会 OOM，面板构建走年分区 --from-parts）
    <cache>/intraday/min{period}_all_a/                                   稠密面板（另跑 build_minute_panel）

用法
----
    # 只出计划：不联网、不建目录
    python -m scripts.ingest.fetch_minute_all --plan
    python -m scripts.ingest.fetch_minute_all --period 1 --plan

    # 正式拉取（分片可断点续传；--plan 之外必须联网）
    python -m scripts.ingest.fetch_minute_all --begin 20180101 --end 20251231
    python -m scripts.ingest.fetch_minute_all --codes 000001.SZ,600000.SH

    # 合并分片 → 单表（面板构建读的是单表）
    python -m scripts.ingest.fetch_minute_all --compact
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.atomic_io import (  # noqa: E402
    write_json_atomic,
    write_parquet_atomic,
)
from scripts.common.cli_common import setup_logging  # noqa: E402
from scripts.ingest.fetch_minute_gap import data_root, year_spans  # noqa: E402

log = setup_logging("fetch_minute_all")

#: 实测（HS300 补缺）：长表 19 B/bar、约 13.5k 行/秒。用于计划期估算，不是承诺。
BYTES_PER_BAR = 19
ROWS_PER_SEC = 13_500
#: 每批股票数。40 只 × 1 年 5 分钟 ≈ 46 万行（≈9 MB），内存与请求耗时都合适。
DEFAULT_BATCH = 40


def trading_days(begin: int, end: int) -> int:
    """窗口内交易日数（同样从日线索引取，用于估算行数）。"""
    p = data_root() / "daily_all_a.parquet"
    d = pd.read_parquet(p, columns=["close"])
    dt = pd.to_datetime(d.index.get_level_values("date"))
    day = dt.strftime("%Y%m%d").astype("int64").to_numpy()
    return int(pd.unique(day[(day >= begin) & (day <= end)]).size)


def existing_coverage(period: int, pool: str) -> dict:
    """已有单表的覆盖（只读索引）。超过 2 GB 不读，避免计划阶段吃几 GB 内存。"""
    p = paths(period, pool)["single"]
    if not p.exists():
        return {"codes": set(), "first": None, "last": None, "size": 0}
    size = p.stat().st_size
    if size > 2 * 1024**3:
        return {"codes": None, "first": None, "last": None, "size": size}
    d = pd.read_parquet(p, columns=["close"])
    # 长表索引是 (kline_time, code)，不是 (date, code)
    dt = pd.to_datetime(d.index.get_level_values(0))
    return {"codes": set(d.index.get_level_values("code").unique()),
            "first": dt.min(), "last": dt.max(), "size": size}


def pool_codes(begin: int, end: int, pool: str) -> list[str]:
    """窗口内**曾交易**的代码（来自日线，天然含退市股，比当前成分更 PIT）。"""
    if pool == "hs300":
        from scripts.ingest.fetch_minute_gap import pit_codes
        return pit_codes(begin, end)
    if pool != "all_a":
        raise SystemExit(f"未知池 {pool}（支持 all_a / hs300，或改用 --codes）")
    p = data_root() / "daily_all_a.parquet"
    if not p.exists():
        raise SystemExit(f"缺 {p}：全 A 代码清单只能从日线推，请先更新日线")
    # 该表的 date/code 在 **MultiIndex** 上（不是列），只取索引避免整表装载
    d = pd.read_parquet(p, columns=["close"])
    dt = pd.to_datetime(d.index.get_level_values("date"))
    day = dt.strftime("%Y%m%d").astype("int64").to_numpy()
    code = d.index.get_level_values("code").to_numpy()
    m = (day >= begin) & (day <= end)
    return sorted(set(code[m].tolist()))


def paths(period: int, pool: str) -> dict[str, Path]:
    root = data_root()
    tag = f"min{period}_{pool}"
    return {
        "single": root / f"{tag}.parquet",
        "parts": root / "intraday" / f"{tag}_parts",
        "staging": root / "intraday" / "_staging" / tag,
        "meta": root / "_meta.json",
    }


def done_chunks(period: int, pool: str, year: int) -> set[str]:
    d = paths(period, pool)["staging"] / str(year)
    return {p.stem for p in d.glob("chunk_*.parquet")} if d.is_dir() else set()


def plan(period: int, pool: str, codes: list[str], begin: int, end: int,
         batch: int) -> None:
    spans = year_spans(begin, end)
    bars_per_day = 240 // period
    p = paths(period, pool)
    cov = existing_coverage(period, pool)
    n_days = trading_days(begin, end)

    # 已有单表能覆盖整年的代码，正式拉取时会跳过（见 fetch_year），计划里同步扣除
    have = cov["codes"] or set()
    covered = have if cov["first"] is not None else set()
    todo_codes = [c for c in codes if c not in covered]
    pending: list[tuple[int, int, int]] = []
    staged = 0
    for lo, hi in spans:
        y = lo // 10000
        done = len(done_chunks(period, pool, y))
        n_chunks = (len(todo_codes) + batch - 1) // batch
        todo = n_chunks - done
        staged += min(done, n_chunks)
        if todo > 0:
            pending.append((y, n_chunks, todo))

    total_chunks = sum(n for _, n, _ in pending)
    total_rows = len(todo_codes) * n_days * bars_per_day
    parts = sorted(p["parts"].glob("part_*.parquet")) if p["parts"].is_dir() else []
    covered_ratio = (1 - len(todo_codes) / len(codes)) if codes else 0

    print("\n[PLAN] 分钟数据拉取")
    print(f"  池            {pool}（窗口内曾交易 {len(codes)} 只）")
    print(f"  区间/频率     {begin} ~ {end} / {period} 分钟"
          f"（每日 {bars_per_day} 根 × {n_days} 个交易日）")
    print(f"  分批          {batch} 只/批 × {len(spans)} 年")
    if cov["first"] is not None:
        cov_desc = "有，覆盖 %d 只（%s ~ %s）" % (
            len(have), str(cov["first"])[:10], str(cov["last"])[:10])
    elif cov["size"]:
        cov_desc = "有，但 >2GB 未解析覆盖"
    else:
        cov_desc = "无"
    print(f"  已有单表      {cov_desc} / 年分区 {len(parts)} 个 / 已存分片 {staged} 个")
    print(f"  待拉代码      {len(todo_codes)} 只（已覆盖 {covered_ratio * 100:.0f}%）")
    print(f"  待拉行数      约 {total_rows / 1e8:.2f} 亿行，"
          f"{total_rows * BYTES_PER_BAR / 1024**3:.1f} GB（长表口径）")
    print(f"  请求次数      {total_chunks} 次")
    print(f"  预计耗时      约 {total_rows / ROWS_PER_SEC / 3600:.1f} 小时"
          f"（按实测 {ROWS_PER_SEC / 1000:.1f}k 行/秒，不含重试）")
    print("  磁盘方案      见 reports/docs/design/数据存储结构与规范_2026-11-20.md §4.3")
    if pending:
        print("  按年待拉:")
        for y, n, todo in pending:
            print(f"    {y}: {todo}/{n} 块")
    print("\n  [PLAN] 未联网、未创建目录、未写入任何文件\n")


def fetch_year(period: int, pool: str, codes: list[str], year: int,
               lo: int, hi: int, batch: int) -> int:
    """拉某一年 [lo, hi] 的全部分片，返回本次新增的分片数（已存在的跳过）。"""
    from data.datasource import create_datasource

    out_dir = paths(period, pool)["staging"] / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    have = done_chunks(period, pool, year)
    # 已有单表覆盖整年的代码直接跳过（新池首次拉取时没有单表，等于全量拉）
    cov = existing_coverage(period, pool)
    if cov["first"] is not None and str(cov["first"])[:10] <= f"{year}-01-01" \
            and str(cov["last"])[:10] >= f"{year}-12-31":
        codes = [c for c in codes if c not in cov["codes"]]
    if not codes:
        log.info("%d 年：全部代码已被单表覆盖，跳过", year)
        return 0
    ds = create_datasource()
    added = 0
    n_chunks = (len(codes) + batch - 1) // batch
    t0 = time.time()
    for i in range(0, len(codes), batch):
        name = f"chunk_{i // batch:04d}"
        if name in have:
            continue
        grp = codes[i:i + batch]
        df = ds.get_minute_kline(grp, lo, hi, period=period)
        if df is None or df.empty:
            log.warning("  %d 批 %d~%d 返回空，跳过", year, i, i + len(grp))
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        write_parquet_atomic(df, out_dir / f"{name}.parquet")
        added += 1
        el = time.time() - t0
        log.info("  %d 块 %d/%d：%d 行（%d 只）/ %.0fs",
                 year, i // batch + 1, n_chunks, len(df),
                 df.index.get_level_values("code").nunique(), el)
    return added


def fetch_tail(period: int, pool: str, begin: int, end: int,
               batch: int) -> int:
    """增量尾巴：只拉 ``begin~end`` 段、落 ``chunk_tail_*``、merge 进当年 part。

    与常规路径的三点不同（2026-11-22 周更链用）：
    ① **绕开块号陷阱**——常规分片按「块号」判已完成，增量段的代码块号与
       既有分片重名会被跳过；tail 用独立命名 ``chunk_tail_<begin>``，永不撞名。
    ② 请求区间只有增量段（几天），不是整年。
    ③ merge 只重写**有新数据的年份**的 ``part_<year>.parquet``，其余年份不动
       （全量 compact 的 concat 会把 8 年全部重写一遍，纯浪费）。
    """
    from data.datasource import create_datasource

    codes = pool_codes(begin, end, pool)
    if not codes:
        log.info("tail：代码清单为空")
        return 0
    ds = create_datasource()
    parts = paths(period, pool)["parts"]
    parts.mkdir(parents=True, exist_ok=True)
    touched: set[int] = set()
    added = 0
    t0 = time.time()
    for i in range(0, len(codes), batch):
        grp = codes[i:i + batch]
        df = ds.get_minute_kline(grp, begin, end, period=period)
        if df is None or df.empty:
            log.warning("  tail 批 %d~%d 返回空，跳过", i, i + len(grp))
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        name = f"chunk_tail_{begin}_{i // batch:04d}"
        write_parquet_atomic(
            paths(period, pool)["staging"] / "tail" / f"{name}.parquet", df)
        added += 1
        for y in pd.to_datetime(
                df.index.get_level_values("kline_time")).year.unique():
            touched.add(int(y))
        log.info("  tail 块 %d/%d：%d 行 / %.0fs",
                 i // batch + 1, (len(codes) + batch - 1) // batch, len(df),
                 time.time() - t0)
    # merge：touched 年份的 part 重写（part + staging/tail 对应段）
    tail_dir = paths(period, pool)["staging"] / "tail"
    for y in sorted(touched):
        part_f = parts / f"part_{y}.parquet"
        frames = []
        if part_f.exists():
            frames.append(pd.read_parquet(part_f))
        kept = []
        for f in sorted(tail_dir.glob(f"chunk_tail_{begin}_*.parquet")):
            d = pd.read_parquet(f)
            yr = pd.to_datetime(d.index.get_level_values("kline_time")).year
            k = d[yr == y]
            if not k.empty:
                kept.append(k)
        frames.extend(kept)
        merged = pd.concat(frames, ignore_index=False)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        write_parquet_atomic(part_f, merged)
        log.info("  part_%d 已合并增量：%d 行 / %d 只",
                 y, len(merged),
                 merged.index.get_level_values("code").nunique())
    # 清 tail 分片
    for f in tail_dir.glob("chunk_tail_*.parquet"):
        f.unlink()
    log.info("tail 完成：%d 分片，touch 年份 %s", added, sorted(touched))
    return added


def compact(period: int, pool: str, *, single: bool) -> None:
    """分片 → 年分区；``single=True`` 时再合并成单表（面板构建读单表）。"""
    p = paths(period, pool)
    p["parts"].mkdir(parents=True, exist_ok=True)
    years = sorted(d.name for d in p["staging"].iterdir() if d.is_dir()) \
        if p["staging"].is_dir() else []
    for y in years:
        chunks = sorted((p["staging"] / y).glob("chunk_*.parquet"))
        if not chunks:
            continue
        df = pd.concat([pd.read_parquet(c) for c in chunks])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        write_parquet_atomic(df, p["parts"] / f"part_{y}.parquet")
        log.info("%s 年合并：%d 片 → %d 行 / %d 只", y, len(chunks), len(df),
                 df.index.get_level_values("code").nunique())
    if not single:
        return
    parts = sorted(p["parts"].glob("part_*.parquet"))
    if not parts:
        log.warning("无年分区可合并")
        return
    df = pd.concat([pd.read_parquet(f) for f in parts])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    write_parquet_atomic(df, p["single"])
    mp = p["meta"]
    meta: dict = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    meta[f"min{period}_{pool}"] = {
        "pool": pool, "size": int(len(df)),
        "codes": int(df.index.get_level_values("code").nunique()),
        "first": str(df.index.get_level_values(0).min())[:10],
        "last": str(df.index.get_level_values(0).max())[:10],
        "universe_note": "窗口内曾交易的全部代码（含退市）",
    }
    write_json_atomic(meta, mp, indent=None)
    log.info("单表已写入 %s：%d 行 / %d 只", p["single"], len(df),
             df.index.get_level_values("code").nunique())
    print("\n下一步（重建稠密面板，全 A 规模用 --from-parts 流式逐年构建）：")
    print("  python -m scripts.factors.build_minute_panel --from-parts "
          f"--period {period} --universe {pool}")


def main() -> None:
    ap = argparse.ArgumentParser(description="全 A 分钟数据拉取（分片 + 断点续传）")
    ap.add_argument("--begin", type=int, default=20180101)
    ap.add_argument("--end", type=int, default=20251231)
    ap.add_argument("--period", type=int, default=5, choices=[1, 5, 15, 30])
    ap.add_argument("--pool", default="all_a", choices=["all_a", "hs300"])
    ap.add_argument("--codes", default="", help="逗号分隔；给了就忽略 --pool")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--plan", action="store_true", help="只出计划：不联网、不建目录")
    ap.add_argument("--compact", action="store_true", help="只做合并，不拉数据")
    ap.add_argument("--tail", type=int, default=None, metavar="BEGIN",
                    help="增量尾巴：拉 BEGIN~--end 段并直接 merge 进当年 part"
                         "（绕开块号命名陷阱；全量 --compact 仍可用但不必再跑）")
    ap.add_argument("--single", action="store_true",
                    help="与 --compact 同用：再合并成单表并更新 _meta.json")
    args = ap.parse_args()

    if args.compact:
        compact(args.period, args.pool, single=args.single)
        return
    if args.tail is not None:
        fetch_tail(args.period, args.pool, args.tail, args.end, args.batch)
        return

    codes = ([c.strip() for c in args.codes.split(",") if c.strip()]
             if args.codes else pool_codes(args.begin, args.end, args.pool))

    if args.plan:
        plan(args.period, args.pool, codes, args.begin, args.end, args.batch)
        return
    if not codes:
        raise SystemExit("代码清单为空")

    total = 0
    for lo, hi in year_spans(args.begin, args.end):
        total += fetch_year(args.period, args.pool, codes, lo // 10000,
                            lo, hi, args.batch)
    log.info("本次新增 %d 个分片；下一步 --compact（全 A 勿加 --single，面板走 --from-parts）", total)


if __name__ == "__main__":
    main()
