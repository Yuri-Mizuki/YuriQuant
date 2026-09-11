"""退市股后复权因子回补：为 daily_all_a 中 backward_factor 缺失的退市股补宽表。

用法:
    D:/python/Python312/python.exe -u scripts/builders/backfill_delisted_backward.py [--batch 40]
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

from data.cache import DataCache  # noqa: E402
from data.datasource import create_datasource  # noqa: E402
from config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=40)
    args = ap.parse_args()
    root = Path(str(Config.cache()["root"]))

    bf = pd.read_parquet(root / "backward_factor.parquet")
    bfc = set(bf.columns.astype(str))
    panel = set(pd.read_parquet(root / "daily_all_a.parquet").reset_index()["code"].astype(str))
    missing = sorted(panel - bfc)
    print(f"backward {len(bfc)} 只 | 面板 {len(panel)} 只 | 缺失 {len(missing)} 只", flush=True)
    if not missing:
        print("无缺失", flush=True)
        return

    ds = create_datasource()
    cache = DataCache(ds, root)
    t0 = time.time()
    for i in range(0, len(missing), args.batch):
        batch = missing[i:i + args.batch]
        for attempt in (1, 2, 3):
            try:
                cache.get_backward_factor(batch)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"批{i//args.batch} 第{attempt}次失败: {str(exc)[:100]}", flush=True)
                time.sleep(4 * attempt)
        print(f"批{i//args.batch}/{(len(missing)+args.batch-1)//args.batch} "
              f"{len(batch)}只 [{time.time()-t0:.0f}s]", flush=True)
    bf2 = pd.read_parquet(root / "backward_factor.parquet")
    print(f"[完成] backward_factor 现 {bf2.shape[1]} 只代码", flush=True)


if __name__ == "__main__":
    main()