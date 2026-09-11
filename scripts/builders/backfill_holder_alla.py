"""回补股东结构数据（holder_num / share_holder）到全 A 口径（含退市）。

现有本地 `holder_num.parquet` / `share_holder.parquet` 仅覆盖 ~520 只
（早期按沪深300/zz1000池拉取的子集），覆盖率 ~9%，无法支撑全市场横截面
因子。本脚本用 master_codes.parquet（5810 只全A含退市清单）全量回补：

- DataCache.get_holder_num / get_share_holder 已原生支持"按新 code 追加并保留
  未请求 code 的本地行"（_merge_sparse_table 语义），直接复用即可；
- 断点续跑：每批完成后把批次下标记入 progress json，脚本可安全中断重续；
- 白名单：仅保留 .SH/.SZ/.BJ A股后缀（防历史代码表混入期货/期权）。

用法（真数据源，D:/python/Python312）:
    python -m scripts.builders.backfill_holder_alla --batch 200
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
from data.datasource import create_datasource  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--tables", nargs="+", default=["holder_num", "share_holder"],
                    help="要回补的表")
    args = ap.parse_args()

    root = Path(str(Config.cache()["root"]))
    master = sorted(pd.read_parquet(root / "_status_refetch" /
                                    "master_codes.parquet")["code"].astype(str))
    master = [c for c in master if c.endswith((".SH", ".SZ", ".BJ"))]
    print(f"主清单(含退市) {len(master)} 只", flush=True)

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
        for bi, batch in enumerate(batches):
            if bi in prog:
                continue
            for attempt in (1, 2, 3):
                try:
                    out = getattr(cache, "get_" + t)(batch)
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f" 批{bi} 第{attempt}次失败: {str(exc)[:100]}",
                          flush=True)
                    time.sleep(4 * attempt)
            if out is not None:
                n_new += len(out)
            prog.add(bi)
            prog_path.write_text(json.dumps(sorted(prog)), encoding="utf-8")
            print(f"[{t}] 批{bi}/{len(batches)} {len(batch)}只 累计+{n_new}行 "
                  f"({time.time()-t0:.0f}s)", flush=True)

        cur = pd.read_parquet(p)
        codes = cur["code"].astype(str).nunique() if "code" in cur.columns else 0
        print(f"[完成] {t} 现 {codes} 只 | 面板 {len(cur):,} 行",
              flush=True)


if __name__ == "__main__":
    main()