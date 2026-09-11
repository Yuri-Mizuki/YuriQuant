"""回补全A财务报表四表（income/balance_sheet/cash_flow/dividend）到含退市。

背景（2026-09-03）：B族（财务报表基本面因子：价值/质量/成长/规模/杠杆/周转）
需要全A的利润表/资产负债表/现金流量表/分红表。当前本地四表都是 HS300 窄池
（~520 只），全A主清单（含退市）5810 只缺~5290 只。本脚本按 master_codes
（_status_refetch/master_codes.parquet，含退市全历史代码）把四表补到全A。

与 backfill_holder_alla.py 同套机制：
- DataCache 的 get_* 是"稀疏事件表整表覆盖 + 保留未请求 code 本地行"，
  增量拉取天然保留已有数据，断点只需按批记录进度；
- 每批重试 3 次，失败批次跳过由进度文件保证续跑；
- 进度落 _fetch_progress_<table>.json。

用法:
    python -m scripts.builders.backfill_financial_alla                 # 补四表
    python -m scripts.builders.backfill_financial_alla --tables income cash_flow
    python -m scripts.builders.backfill_financial_alla --batch 100
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
from data.cache import DataCache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=120)
    ap.add_argument("--tables", nargs="+",
                    default=["income", "balance_sheet", "cash_flow", "dividend"],
                    help="要回补的财务表")
    args = ap.parse_args()

    root = Path(str(Config.cache()["root"]))
    master = sorted(pd.read_parquet(root / "_status_refetch" /
                                    "master_codes.parquet")["code"].astype(str))
    master = [c for c in master if c.endswith((".SH", ".SZ", ".BJ"))]
    print(f"主清单(含退市) {len(master)} 只", flush=True)

    from data.datasource import create_datasource
    ds = create_datasource()
    cache = DataCache(ds, root)

    for t in args.tables:
        p = root / f"{t}.parquet"
        if p.exists():
            have = set(pd.read_parquet(p)["code"].astype(str))
        else:
            have = set()
        missing = sorted(set(master) - have)
        print(f"[{t}] 已有 {len(have)} 只 | 缺失 {len(missing)} 只", flush=True)
        if not missing:
            print(f"[{t}] 无缺失，跳过", flush=True)
            continue

        prog_path = root / f"_fetch_progress_{t}.json"
        prog = set(json.loads(prog_path.read_text(encoding="utf-8"))
                   ) if prog_path.exists() else set()

        batches = [missing[i:i + args.batch]
                   for i in range(0, len(missing), args.batch)]
        t0 = time.time()
        n_new = 0
        ok = 0
        for bi, batch in enumerate(batches):
            if bi in prog:
                continue
            for attempt in (1, 2, 3):
                try:
                    out = getattr(cache, "get_" + t)(batch)
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f" 批{bi} 第{attempt}次失败: {str(exc)[:100]}", flush=True)
                    time.sleep(4 * attempt)
            if out is not None:
                n_new += len(out)
            ok += len(batch)
            prog.add(bi)
            prog_path.write_text(json.dumps(sorted(prog)), encoding="utf-8")
            print(f"[{t}] 批{bi}/{len(batches)} 累计完成{ok}/{len(missing)}只 "
                  f"({time.time()-t0:.0f}s)", flush=True)

        cur = pd.read_parquet(p)
        codes = cur["code"].astype(str).nunique() if "code" in cur.columns else 0
        print(f"[完成] {t} 现 {codes} 只 | 面板 {len(cur):,} 行", flush=True)


if __name__ == "__main__":
    main()