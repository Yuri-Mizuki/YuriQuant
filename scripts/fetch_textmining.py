"""
抓取文本挖掘数据（Phase 0 CLI）
==============================

用法：
    python -m scripts.fetch_textmining --universe hs300 --sources ths
    python -m scripts.fetch_textmining --codes 600519.SH,000001.SZ --sources ths,cninfo
    python -m scripts.fetch_textmining --universe hs300 --sources cninfo --categories 业绩预告 --with-pdf
    python -m scripts.fetch_textmining --codes 600519.SH --limit 1   # 调试：只抓 1 只

说明：
    - 主源 ths（同花顺研报）：一次返回全历史，按 code 增量缓存。
    - 辅源 cninfo（巨潮公告）：按日期段拉取，支持类别/关键词过滤 + PDF 全文。
    - 数据落 e:/data/parquet/text_ths_report.parquet 等，_meta.json 记水位。
"""
from __future__ import annotations

import argparse
import sys
import time

from data.textmining.fetch import fetch_docs, TextMiningCache


def _resolve_codes(args) -> list[str]:
    if args.codes:
        return [c.strip() for c in args.codes.split(",") if c.strip()]
    from data.universe import Universe  # noqa: PLC0415
    from data.cache import DataCache  # noqa: PLC0415
    from data.datasource import create_datasource  # noqa: PLC0415

    ds = create_datasource()
    cache = DataCache(ds)
    u = Universe(cache)
    codes = u.get_hs300(args.end) if args.universe == "hs300" else u.get_all_constituents()
    if args.limit:
        codes = codes[: args.limit]
    return codes


def main() -> None:
    ap = argparse.ArgumentParser(description="抓取文本挖掘数据（研报/公告）")
    ap.add_argument("--universe", default="hs300", help="股票池: hs300/zz500/all_a")
    ap.add_argument("--codes", default="", help="逗号分隔标准代码，优先于 --universe")
    ap.add_argument("--sources", default="ths", help="逗号分隔: ths(研报),cninfo(公告)")
    ap.add_argument("--begin", type=int, default=20190101, help="开始日期 YYYYMMDD")
    ap.add_argument("--end", type=int, default=20261231, help="结束日期 YYYYMMDD")
    ap.add_argument("--categories", default="", help="巨潮公告类别，逗号分隔（业绩预告/年报等）")
    ap.add_argument("--searchkey", default="", help="巨潮公告标题关键词")
    ap.add_argument("--with-pdf", action="store_true", help="下载巨潮公告 PDF 全文")
    ap.add_argument("--max-pages", type=int, default=None, help="巨潮单查询最多翻页数")
    ap.add_argument("--limit", type=int, default=0, help="最多抓 N 只（调试）")
    args = ap.parse_args()

    codes = _resolve_codes(args)
    if not codes:
        print("无股票代码，退出")
        sys.exit(1)
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    cats = [c.strip() for c in args.categories.split(",") if c.strip()] or None

    print(f"股票数: {len(codes)} | 源: {sources} | 区间: {args.begin}-{args.end}")
    t0 = time.time()
    docs = fetch_docs(
        codes, begin_date=args.begin, end_date=args.end, sources=sources,
        with_pdf=args.with_pdf, cninfo_categories=cats,
        cninfo_searchkey=args.searchkey, cninfo_max_pages=args.max_pages)
    for src, df in docs.items():
        if df.empty:
            print(f"[{src}] 无数据")
            continue
        print(f"[{src}] {len(df)} 条 | 覆盖 {df['code'].nunique()} 只 | "
              f"区间 {df['date'].min().date()} ~ {df['date'].max().date()}")
        if src == "ths":
            print(f"  覆盖度: 有研报的股票 {(df['code'].nunique() / len(codes)) * 100:.0f}%")
    print(f"总耗时: {time.time() - t0:.1f}s")
    print(f"缓存: {TextMiningCache().root}")


if __name__ == "__main__":
    main()
