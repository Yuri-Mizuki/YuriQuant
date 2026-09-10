"""全A文本数据补爬（backfill）。

把同花顺研报 + 巨潮业绩预告缓存从当前覆盖补到全A（在册全A ∪ daily_all_a
历史代码）。利用 TextMiningCache 增量水位：中断后重跑会自动跳过已覆盖 code。

用法:
    python -m scripts.textmining.fetch_all_a_backfill [--limit N]

一次性把全部 missing codes 传给 get_ths_reports / get_cninfo_announcements
（两者内部对已覆盖 code 自动短路，避免反复读全量缓存）。
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

from data.datasource import create_datasource  # noqa: E402
from data.textmining.fetch import TextMiningCache  # noqa: E402
from data.textmining.source_cninfo import fetch_cninfo_announcements  # noqa: E402
from data.textmining.source_ths import to_code6  # noqa: E402


def _backfill_cninfo_year(cache: TextMiningCache, codes: list[str], year: int) -> int:
    """非增量地抓取某年全年业绩预告并合并写回缓存。

    绕过 get_cninfo_announcements 的增量短路（其 missing/earliest 短路会让
    已缓存股票不补更早日期段），直接用底层 fetch 抓满一年全 A 后合并去重落盘。
    """
    p = cache.root / "text_cninfo_ann_cat_yjygjxz.parquet"
    local = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    print(f"抓取 {year} 全A业绩预告（{len(codes)} 只）...")
    t0 = time.time()
    df = fetch_cninfo_announcements(
        codes=codes, begin_date=int(f"{year}0101"), end_date=int(f"{year}1231"),
        categories=["业绩预告"])
    print(f"  抓取 {len(df)} 行（{time.time() - t0:.0f}s）")
    if df.empty:
        return len(local)
    combined = pd.concat([local, df], ignore_index=True) if not local.empty else df
    combined = combined.drop_duplicates(subset=["code", "date", "title"], keep="last")
    combined = (combined.sort_values(["code", "date"])
                       .reset_index(drop=True))
    combined.to_parquet(p, compression="snappy")
    return len(combined)


def target_codes(cache: TextMiningCache) -> list[str]:
    """补爬目标集：数据源在册全 A ∪ daily_all_a 历史代码。"""
    ds = create_datasource()
    full = {to_code6(c) for c in ds.get_code_list("EXTRA_STOCK_A")}
    try:
        da = pd.read_parquet(cache.root / "daily_all_a.parquet")
        full |= {to_code6(c) for c in da.index.get_level_values("code").unique()}
    except Exception as e:  # noqa: BLE001
        print(f"[warn] daily_all_a 读取失败,仅用在册集: {type(e).__name__}: {e}")
    return sorted(full)


def cached_ths(cache: TextMiningCache) -> set[str]:
    p = cache.root / "text_ths_report.parquet"
    if not p.exists():
        return set()
    return {to_code6(c) for c in pd.read_parquet(p, columns=["code"])["code"].unique()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="仅补前 N 只（冒烟用），默认全量")
    ap.add_argument("--skip-ths", action="store_true",
                    help="跳过 ths 研报补爬，仅做巨潮业绩预告回补")
    ap.add_argument("--cninfo-begin", type=int, default=20190101,
                    help="业绩预告回补起点 YYYYMMDD（如 20180101 补 2018 年）")
    ap.add_argument("--cninfo-year", type=int, default=None,
                    help="非增量抓取某年全年业绩预告并合并写盘（跳过增量短路）")
    args = ap.parse_args()

    cache = TextMiningCache()
    testp = cache.root / "_backfill_write_test.tmp"
    testp.write_text("ok", encoding="utf-8")
    testp.unlink()
    print(f"写权限 OK: {cache.root}")

    full = target_codes(cache)
    cur = cached_ths(cache)
    missing = [c for c in full if c not in cur]
    if args.limit:
        missing = missing[: args.limit]
    print(f"目标 {len(full)} 只, ths 已覆盖 {len(cur)} 只, 本次补 {len(missing)} 只")

    if not args.skip_ths:
        print("[1/2] 同花顺研报补爬 ...")
        t0 = time.time()
        df = cache.get_ths_reports(missing)
        print(f"  ths 缓存 {len(df)} 行 / {df['code'].nunique() if not df.empty else 0} 只"
              f"  ({time.time() - t0:.0f}s)")
    else:
        missing = []

    if args.cninfo_year:
        print("[2/2] 巨潮业绩预告非增量回补（按整年）...")
        rows = _backfill_cninfo_year(cache, full, args.cninfo_year)
        print(f"  合并写盘后缓存 {rows} 行")
    else:
        print("[2/2] 巨潮业绩预告补爬 ...")
        t0 = time.time()
        ann = cache.get_cninfo_announcements(
            codes=full, begin_date=args.cninfo_begin, end_date=20261231,
            categories=["业绩预告"])
        print(f"  业绩预告缓存 {len(ann)} 行 ({time.time() - t0:.0f}s)")
    print("DONE")


if __name__ == "__main__":
    main()