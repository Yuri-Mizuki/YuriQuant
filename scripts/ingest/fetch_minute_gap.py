"""补齐 HS300 动态池（PIT 成分）缺失的分钟数据
==========================================

**问题**：``min5_hs300.parquet`` 只有 **339 只**——正好是静态池名单，而
HS300 在 2022-01 ~ 2026-06 窗口内**曾属**的成分共 **421 只**。缺的 82 只
（19.5%）不是随机缺失，而是**被剔除出指数的差股票**（池审计：缺的这 82 只
中位累计收益 **−51.2%** vs 在库 +4.3%，p=5.7e−15）——它们正是日频策略
幸存者偏差的来源。不补齐就无法在 PIT 动态池上重算 ``im_*`` 因子。

**为什么要绕过 DataCache**：``DataCache._refresh_long_table`` 的覆盖检测
只比对**日期**（``local_days ⊇ req_days``），**不检查 code 覆盖**。补新股票
时请求区间已被日期覆盖 ⇒ 直接短路返回、不访问数据源，补数据静默失败。
所以本脚本直接调 ``DataSource.get_minute_kline``，自己合并落盘。

**做法**：
1. 从 ``index_constituent_000300_SH.parquet`` 逐条展开 (con_code, in_date,
   out_date)，取与 [begin, end] 有交集的全部成分（窗口内曾属）→ 421 只。
2. 与现有 parquet 的 code 集合求差 → 待补清单（82 只，减去备份可回补的）。
3. 按**年**分批向数据源请求（73 只 × 1 年 ≈ 85 万行/批，内存可控），
   每批落盘前与全量本地数据 concat + 去重。
4. 用 ``atomic_io.write_parquet_atomic`` 写回，更新 ``_meta.json``。
5. 提示重建 ``MinutePanelStore``（codes 从 339 → 421，必须重建）。

用法（须用系统 Python，需联网）::

    python -m scripts.ingest.fetch_minute_gap --plan          # 只报缺口，不拉
    python -m scripts.ingest.fetch_minute_gap                 # 补全量缺口
    python -m scripts.ingest.fetch_minute_gap --codes 000066.SZ,000069.SZ --plan
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

from config import Config  # noqa: E402
from scripts.common.atomic_io import write_json_atomic, write_parquet_atomic  # noqa: E402
from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("fetch_minute_gap")

#: 分钟缓存文件名（与 DataCache.get_minute_kline 的命名规则一致）
MIN_FILE = "min5_hs300.parquet"
INDEX_CODE = "000300.SH"
PERIOD = 5
BARS_PER_DAY = 240 // PERIOD


def data_root() -> Path:
    return Path(str(Config.cache()["root"]).replace("//", "/"))


def pit_codes(begin: int, end: int) -> list[str]:
    """窗口内曾属 HS300 的全部成分（PIT 逐条展开，非某一时点快照）。"""
    p = data_root() / f"index_constituent_{INDEX_CODE.replace('.', '_')}.parquet"
    c = pd.read_parquet(p)
    in_d = c["in_date"].astype("Int64")
    out_d = c["out_date"].astype("Int64")
    # 与 [begin, end] 有交集：进得够早 且 （还没出 或 出得够晚）
    m = (in_d <= end) & (out_d.isna() | (out_d >= begin))
    return sorted(set(c.loc[m, "con_code"]))


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def year_spans(begin: int, end: int) -> list[tuple[int, int]]:
    """按自然年切分 [begin, end]，返回 (起, 止) 的 YYYYMMDD 对。"""
    spans: list[tuple[int, int]] = []
    for y in range(begin // 10000, end // 10000 + 1):
        lo = max(begin, y * 10000 + 101)
        hi = min(end, y * 10000 + 1231)
        if lo <= hi:
            spans.append((lo, hi))
    return spans


def fetch_missing(codes: list[str], begin: int, end: int,
                  batch: int = 20) -> pd.DataFrame:
    """直接向数据源拉分钟数据（绕过缓存层的日期覆盖短路）。

    batch: 单次请求的股票数。实测 ~13.5k 行/秒，20 只 × 1 年 ≈ 23 万行/批。
    """
    from data.datasource import create_datasource

    ds = create_datasource()
    frames: list[pd.DataFrame] = []
    spans = year_spans(begin, end)
    total_est = len(codes) * len(spans)
    done = 0
    t0 = time.time()
    for lo, hi in spans:
        for i in range(0, len(codes), batch):
            grp = codes[i:i + batch]
            df = ds.get_minute_kline(grp, lo, hi, period=PERIOD)
            if df is not None and not df.empty:
                frames.append(df)
            done += 1
            el = time.time() - t0
            got = sum(len(f) for f in frames)
            log.info("  %d~%d 批 %d~%d：本批 %d 行，累计 %d 行 / %.0fs（%.0f 行/秒）",
                     lo, hi, i, i + len(grp), 0 if df is None else len(df),
                     got, el, got / el if el > 0 else 0)
            log.info("  进度 %d/%d 块", done, total_est)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="last")].sort_index()


def update_meta(path: Path, n_rows: int) -> None:
    """同步缓存元数据（last_date / size），使后续增量逻辑水位正确。"""
    mp = data_root() / "_meta.json"
    meta: dict = {}
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    key = f"min{PERIOD}_hs300"
    entry = meta.get(key, {})
    entry.update({"pool": "hs300", "size": int(n_rows), "pit": True,
                  "universe_note": "PIT 动态池：窗口内曾属 HS300 的全部成分"})
    meta[key] = entry
    write_json_atomic(meta, mp, indent=None)
    log.info("元数据已更新: %s size=%d", key, n_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="补齐 HS300 PIT 动态池的分钟数据缺口")
    ap.add_argument("--begin", type=int, default=20220101)
    ap.add_argument("--end", type=int, default=20260630)
    ap.add_argument("--codes", default="", help="逗号分隔的股票清单（默认=自动算缺口）")
    ap.add_argument("--batch", type=int, default=20, help="单次请求股票数")
    ap.add_argument("--plan", action="store_true", help="只报缺口不拉取")
    args = ap.parse_args()

    path = data_root() / MIN_FILE
    need_all = pit_codes(args.begin, args.end)
    existing = load_existing(path)
    have = (set(existing.index.get_level_values("code").unique())
            if not existing.empty else set())

    if args.codes:
        target = [c.strip() for c in args.codes.split(",") if c.strip()]
    else:
        target = [c for c in need_all if c not in have]

    log.info("PIT 动态池 %d 只 / 现有分钟覆盖 %d 只 / 待补 %d 只",
             len(need_all), len(have), len(target))
    if not target:
        log.info("无缺口，退出")
        return

    rows_est = len(target) * 1085 * BARS_PER_DAY
    log.info("待补清单: %s", ", ".join(target[:20]) + (" ..." if len(target) > 20 else ""))
    log.info("预估 %.0f 万行，按 13.5k 行/秒约 %.0f 分钟",
             rows_est / 1e4, rows_est / 13500 / 60)

    if args.plan:
        print(f"\n[PLAN] 待补 {len(target)} 只：")
        for c in target:
            print("  " + c)
        return

    add = fetch_missing(target, args.begin, args.end, batch=args.batch)
    if add.empty:
        log.warning("数据源返回空，未改动任何文件")
        return
    log.info("拉到 %d 行 / %d 只", len(add), add.index.get_level_values("code").nunique())

    merged = add if existing.empty else pd.concat([existing, add])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    write_parquet_atomic(merged, path)
    log.info("已写入 %s：%d 行 / %d 只",
             path, len(merged), merged.index.get_level_values("code").nunique())
    update_meta(path, len(merged))

    print("\n下一步（codes 由 %d → %d，必须重建面板）:" % (
        len(have), merged.index.get_level_values("code").nunique()))
    print("  python -m scripts.factors.build_minute_panel --offline")


if __name__ == "__main__":
    main()