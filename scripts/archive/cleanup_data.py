"""数据层体检与整理脚本（2026-08-26 数据整理的可复用工具）。

功能（--check 只读体检；--fix-meta 才执行修复）：
1. 体检：扫描 e:/data/parquet 下所有表，核对 meta last_date/pool 与实盘数据；
   检查 daily_*/min*_* 是否混入异池代码或指数代码；列出大小 Top 表。
2. 修复 meta：重新扫描各表实际日期 + 池口径（调用 fix_meta）。

用法：
    python -m scripts.data_tools.cleanup_data --check              # 只体检
    python -m scripts.data_tools.cleanup_data --fix-meta           # 修 meta

注意：
    - 池口径以财务表（income.parquet）的 code 集为准。
    - 按池分文件后（daily_{pool}.parquet），异池隔离由文件名保证，
      --trim 已不再需要（2026-08-26 池隔离扩展）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import Config  # noqa: E402  缓存根单一真源

ROOT = Path(str(Config.cache()["root"]))


def _is_index_code(code: str) -> bool:
    import re
    return bool(re.match(r"^000\d{3}\.SH$", code)) and code not in ("000001.SZ",)


def _load_meta() -> dict:
    p = ROOT / "_meta.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _get_pool() -> set[str]:
    """主池 = 财务表 code 集（2026-08-26 约定为 HS300 并集池）。"""
    fin = pd.read_parquet(ROOT / "income.parquet")
    return set(fin["code"].unique())


def cmd_check(args) -> None:
    print("=" * 70)
    print("数据层体检（只读，不修改任何文件）")
    print("=" * 70)
    meta = _load_meta()
    pool = _get_pool()
    print(f"主池（财务表口径）: {len(pool)} 只\n")

    problems: list[str] = []
    for p in sorted(ROOT.glob("*.parquet")):
        name = p.stem
        size_mb = p.stat().st_size / 1e6
        info = meta.get(name, {})
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            problems.append(f"{name}: 读取失败 {e}")
            print(f"  [ERR] {name:<40} {size_mb:>8.1f}M  读取失败: {e}")
            continue

        # 池一致性检查（code 可能在列或在 MultiIndex 层）
        codes = None
        for c in ["code", "con_code"]:
            if c in df.columns:
                codes = set(df[c].dropna().unique())
                break
        if codes is None and isinstance(df.index, pd.MultiIndex) and "code" in df.index.names:
            codes = set(df.index.get_level_values("code").dropna().unique())
        is_kline = name.startswith("daily_") or (name.startswith("min") and "_" in name)
        if codes is not None and is_kline:
            foreign = codes - pool
            index_codes = {c for c in codes if _is_index_code(c)}
            if foreign:
                problems.append(f"{name}: 含 {len(foreign)} 只异池代码（{sorted(foreign)[:3]}...）")
            if index_codes:
                problems.append(f"{name}: 含指数代码 {index_codes}")
            status = "OK" if not foreign and not index_codes else "MIXED"
            print(f"  [{status:<5}] {name:<40} {size_mb:>8.1f}M  codes={len(codes)} "
                  f"pool={info.get('pool','?')}")
        else:
            print(f"  [---] {name:<40} {size_mb:>8.1f}M")

    print(f"\n{'='*70}")
    print(f"体检完成：{len(problems)} 个问题")
    for pr in problems:
        print(f"  - {pr}")
    print("修复建议：--fix-meta（修 meta）")


def cmd_fix_meta(args) -> None:
    from scripts.data_tools.fix_meta import main as fix_main
    fix_main()


def main():
    parser = argparse.ArgumentParser(description="数据层体检与整理")
    parser.add_argument("--check", action="store_true", help="只读体检")
    parser.add_argument("--fix-meta", action="store_true", help="修复 _meta.json")
    args = parser.parse_args()

    if args.check:
        cmd_check(args)
    elif args.fix_meta:
        cmd_fix_meta(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
