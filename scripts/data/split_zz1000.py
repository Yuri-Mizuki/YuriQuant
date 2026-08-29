"""将 daily/min5 中异池代码分离归档（保留当前主池）。

背景（2026-08-26）：daily.parquet 曾为 HS300∪ZZ1000 双指数并集池（2700 只），
而财务/状态表只有 HS300 并集池（520 只），口径不一致。主池定为纯 HS300 后，
本脚本把异池代码分离到 archive_zz1000/（不删除，可随时合并回）。

注意：2026-08-26 池隔离扩展后，新数据按池分文件（daily_{pool}.parquet），
正常拉取不再产生混池，本脚本仅用于历史文件迁移或回退场景。

用法：python -m scripts.data.split_zz1000
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import Config  # noqa: E402  缓存根单一真源

ROOT = Path(str(Config.cache()["root"]))
ARCHIVE = ROOT / "archive_zz1000"


def main() -> None:
    ARCHIVE.mkdir(exist_ok=True)

    # ---- HS300 并集池（以财务表为准，520 只）----
    fin = pd.read_parquet(ROOT / "income.parquet")
    hs300_pool = set(fin["code"].unique())
    print(f"[1] HS300 并集池（财务表）: {len(hs300_pool)} 只")

    # ---- daily（优先新命名 daily_hs300.parquet，兼容旧 daily.parquet）----
    d_src = ROOT / "daily_hs300.parquet" if (ROOT / "daily_hs300.parquet").exists() else ROOT / "daily.parquet"
    if d_src.exists():
        d = pd.read_parquet(d_src).reset_index()
        keep = d[d["code"].isin(hs300_pool)]
        out = d[~d["code"].isin(hs300_pool)]
        index_part = out[out["code"].str.contains("^000\\d{3}\\.SH$", regex=True)]
        stock_zz1000 = out[~out["code"].str.contains("^000\\d{3}\\.SH$", regex=True)]
        print(f"[2] daily({d_src.name}) 原 {d['code'].nunique()} 只 -> 保留 {keep['code'].nunique()}, "
              f"归档股票 {stock_zz1000['code'].nunique()}, 指数 {index_part['code'].nunique()}")
        keep = keep.set_index(["date", "code"]).sort_index()
        keep.to_parquet(d_src, compression="snappy")
        if len(stock_zz1000):
            stock_zz1000.set_index(["date", "code"]).sort_index().to_parquet(
                ARCHIVE / "daily_zz1000_only.parquet", compression="snappy")
        if len(index_part):
            index_part.set_index(["date", "code"]).sort_index().to_parquet(
                ARCHIVE / "daily_index_000905SH.parquet", compression="snappy")

    # ---- min5 ----
    m_src = ROOT / "min5_hs300.parquet" if (ROOT / "min5_hs300.parquet").exists() else ROOT / "min5.parquet"
    if m_src.exists():
        m = pd.read_parquet(m_src).reset_index()
        m_keep = m[m["code"].isin(hs300_pool)]
        m_out = m[~m["code"].isin(hs300_pool)]
        print(f"[3] min5({m_src.name}) 原 {m['code'].nunique()} 只 -> 保留 {m_keep['code'].nunique()}, "
              f"归档 {m_out['code'].nunique()}")
        m_keep.set_index(["kline_time", "code"]).sort_index().to_parquet(m_src, compression="snappy")
        if len(m_out):
            m_out.set_index(["kline_time", "code"]).sort_index().to_parquet(
                ARCHIVE / "min5_non_pool.parquet", compression="snappy")

    print(f"\n归档目录: {ARCHIVE}")
    for p in sorted(ARCHIVE.glob("*.parquet")):
        print(f"  {p.name}: {p.stat().st_size/1e6:.1f}M")


if __name__ == "__main__":
    main()
