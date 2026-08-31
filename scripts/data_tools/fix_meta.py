"""修复 e:/data/parquet/_meta.json：对齐实际日期 + 补文本表 + 加 pool 口径。

背景（2026-08-26 数据整理）：
- 缓存按池分文件：daily_{pool}.parquet / min{period}_{pool}.parquet（2026-08-26 池隔离扩展）
- daily_hs300.parquet = HS300 并集池 520 只；ZZ1000 归档到 archive_zz1000/
- 文本表英文 slug 命名（text_cninfo_ann_cat_yjygjxz）

本脚本：
1. 重新扫描所有 parquet 的实际最新日期，更新 last_date
2. 每表增加 pool 口径字段：hs300（当前主池）| hs300_zz1000 | all_a | market | index_*
3. 记录 archive_zz1000 目录的存在（ZZ1000 数据离线归档，未删除）

用法：python -m scripts.data_tools.fix_meta
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import Config  # noqa: E402  缓存根单一真源

ROOT = Path(str(Config.cache()["root"]))

# 每表的池口径（2026-08-26 主池=纯 HS300；按池分文件后表名带 _hs300 后缀）
TABLE_POOL = {
    "calendar": "market",
    "daily_hs300": "hs300",
    "min5_hs300": "hs300",
    "history_stock_status": "hs300",
    "income": "hs300",
    "balance_sheet": "hs300",
    "cash_flow": "hs300",
    "equity_structure": "hs300",
    "dividend": "hs300",
    "share_holder": "hs300",
    "holder_num": "hs300",
    "index_constituent_000300_SH": "index_hs300",
    "index_constituent_000852_SH": "index_zz1000",
    "index_daily_000300_SH": "index_hs300",
    "industry_classification_level1": "all_a",
    "adj_factor": "hs300",
    "backward_factor": "hs300",
    "text_ths_report": "hs300_zz1000",  # 研报覆盖双指数
    "text_cninfo_ann_cat_yjygjxz": "hs300_zz1000",  # 公告覆盖双指数
}

# 无时间列（宽表）或 index 时间列名
WIDE_TABLES = {"adj_factor", "backward_factor"}


def _latest(parquet_path: Path) -> str | None:
    """返回表的最晚日期（YYYYMMDD 字符串），宽表返回 None。"""
    name = parquet_path.stem
    if name in WIDE_TABLES:
        return None
    df = pd.read_parquet(parquet_path)
    tcol = None
    for c in ["date", "kline_time", "ann_date", "report_date",
              "change_date", "ex_date", "in_date", "publish_date"]:
        if c in df.columns:
            tcol = c
            break
    if tcol is None and isinstance(df.index, pd.MultiIndex):
        tcol = df.index.names[0]
    if tcol is None:
        return None
    try:
        if isinstance(df.index, pd.MultiIndex) and tcol == df.index.names[0]:
            v = df.index.get_level_values(tcol).max()
        else:
            v = df[tcol].max()
        s = str(v)
        # 统一转 YYYYMMDD
        digits = "".join(ch for ch in s if ch.isdigit())[:8]
        return digits or None
    except Exception:
        return None


def main() -> None:
    meta_path = ROOT / "_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    print("=== 重新扫描各表实际最新日期 ===")
    new_meta: dict = {}
    for p in sorted(ROOT.glob("*.parquet")):
        name = p.stem
        last = _latest(p)
        old = meta.get(name, {})
        pool = TABLE_POOL.get(name, "unknown")
        entry = {
            "last_date": last if last is not None else old.get("last_date"),
            "pool": pool,
            "size": p.stat().st_size,  # 文件体积（字节），便于快速核对
        }
        # 保留原有额外字段（但剔除旧版遗留字段）
        for k, v in old.items():
            if k not in entry and k not in ("rows",):
                entry[k] = v
        new_meta[name] = entry
        print(f"  {name:<38} last_date={entry['last_date']:<10} pool={pool}")

    # 归档目录标记
    archive_dir = ROOT / "archive_zz1000"
    new_meta["_archive"] = {
        "note": "ZZ1000 专属数据离线归档（2026-08-26 从 daily/min5 分离），需要时从 archive_zz1000/*.parquet 合并回",
        "files": [f.name for f in sorted(archive_dir.glob("*.parquet"))] if archive_dir.exists() else [],
    }

    meta_path.write_text(json.dumps(new_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {meta_path}（{len(new_meta)-1} 张表 + 归档标记）")


if __name__ == "__main__":
    main()
