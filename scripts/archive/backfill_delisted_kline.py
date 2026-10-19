"""退市股 K线 回补：把 daily_all_a.parquet 缺失的退市股历史K线并入。

背景：daily_all_a 原按 get_code_list("EXTRA_STOCK_A")（当前上市）构建，261 只退市
股无 K线，导致全A实验存在幸存者偏差。此脚本按断点续跑方式补齐缺失代码。

用法:
    D:/python/Python312/python.exe -u scripts/builders/backfill_delisted_kline.py [--batch 100]
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
from data.datasource import create_datasource  # noqa: E402

BEGIN = 20150101
END = 20260902


def _load_daily_index(path: Path) -> set[str]:
    df = pd.read_parquet(path, columns=[]) if False else pd.read_parquet(path)
    if "code" not in df.columns:
        df = df.reset_index()
    return set(df["code"].astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--begin", type=int, default=BEGIN)
    ap.add_argument("--end", type=int, default=END)
    args = ap.parse_args()

    root = Path(str(Config.cache()["root"]))
    daily_path = root / "daily_all_a.parquet"
    scache = root / "_status_refetch"
    master_path = scache / "master_codes.parquet"

    local = _load_daily_index(daily_path)
    master = set(pd.read_parquet(master_path)["code"].astype(str))
    missing = sorted(master - local)
    print(f"本地 {len(local)} 只 | 主清单 {len(master)} 只 | 缺失(退市) {len(missing)} 只", flush=True)
    if not missing:
        print("无缺失，无需回补", flush=True)
        return

    ds = create_datasource()
    prog_path = root / "_fetch_progress_delist_kline.json"
    prog = set(json.loads(prog_path.read_text(encoding="utf-8"))) if prog_path.exists() else set()

    batches = [missing[i:i + args.batch] for i in range(0, len(missing), args.batch)]
    df = pd.read_parquet(daily_path)
    idx_names = df.index.names  # ['date','code']
    t0 = time.time()
    n_new = 0
    for bi, batch in enumerate(batches):
        if bi in prog:
            continue
        sub = None
        for attempt in (1, 2, 3):
            try:
                sub = ds.get_daily_kline(batch, args.begin, args.end)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"批{bi} 第{attempt}次失败: {str(exc)[:110]}", flush=True)
                time.sleep(5 * attempt)
        if sub is None or not len(sub):
            print(f"批{bi}: 无数据，跳过（{batch}）", flush=True)
            prog.add(bi)
            prog_path.write_text(json.dumps(sorted(prog)), encoding="utf-8")
            continue
        df = pd.concat([df, sub], axis=0)
        df = (df.reset_index().drop_duplicates(subset=idx_names, keep="last")
              .set_index(list(idx_names)).sort_index())
        n_new += len(sub)
        prog.add(bi)
        prog_path.write_text(json.dumps(sorted(prog)), encoding="utf-8")
        print(f"批{bi}/{len(batches)} 完成 {len(batch)} 只 +{len(sub)}行 "
              f"累计{len(df):,}行 [{time.time() - t0:.0f}s]", flush=True)
    print(f"[完成] 缺失 {len(missing)} 只 并入新行 {n_new} | 总 {len(df):,} 行",
          flush=True)
    df.to_parquet(daily_path)
    print(f"已写回 {daily_path}", flush=True)


if __name__ == "__main__":
    main()