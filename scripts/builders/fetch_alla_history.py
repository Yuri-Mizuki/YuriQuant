"""全A历史数据回填（2015~now）——分批断点续拉版。

为「全A多年度滚动训练实验」补齐数据缺口（2026-09-01 盘点）：

| 表 | 现状 | 目标 | 手段 |
|---|---|---|---|
| daily_all_a | 2022-01~2025-12 | 2015-01~最新 | 分批拉 + 合并去重（SDK 大清单单查会挂死） |
| calendar | 2019-01~2026-08 | 2015-01~ | get_calendar 增量回补 |
| history_stock_status | 2019-01~ | 2015-01~2018-12 | 复用 fetch_status_batched.py（另跑） |
| index_daily_000001_SH | 无 | 2015-01~ | get_index_daily（全A基准） |

backward_factor / equity_structure / industry 已是全A覆盖，无需回填。

安全设计：
- 5553 只按 ``--batch``（默认 300）分批，每批 3 次重试，失败跳过（结尾报告）；
- 进度文件 ``_fetch_progress_alla_2015.json`` 记录已完成批次，重启自动跳过；
- 合并写盘保留全量本地数据（所有 code/日期），仅按 (date,code) 去重 keep=last。

用法:
    python scripts/builders/fetch_alla_history.py [--batch 300] [--begin 20150101]
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

from scripts.cli_common import setup_logging  # noqa: E402

log = setup_logging("fetch_alla_hist")

PROGRESS_KEYS = ("daily_batches_done",)


def _load_progress(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"daily_batches_done": []}


def _save_progress(path: Path, prog: dict) -> None:
    path.write_text(json.dumps(prog, ensure_ascii=False), encoding="utf-8")


def main():
    from config import Config
    from data.cache import DataCache
    from data.datasource import create_datasource

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=300)
    ap.add_argument("--begin", type=int, default=20150101)
    args = ap.parse_args()

    ds = create_datasource()
    cache = DataCache(ds)
    root = Path(str(Config.cache()["root"]))

    # ---- 1) 日历回补（先做：后续按最新交易日对齐） ----
    cal = cache.get_calendar(args.begin, None)
    log.info("日历: %s ~ %s（%d 日）", cal[0], cal[-1], len(cal))
    from scripts.cli_common import complete_day_target
    end_date, rolled = complete_day_target(cal, cal[-1])
    if rolled:
        log.info("盘中守卫: 拉取终点回退到 %s（今天数据未完整）", end_date)

    # ---- 2) 指数基准：上证指数 000001.SH（全A组合对照） ----
    for idx_code in ("000001.SH",):
        try:
            df_idx = cache.get_index_daily(idx_code, args.begin, end_date)
            log.info("指数 %s: %d 行（%s ~ %s）", idx_code, len(df_idx),
                     df_idx.index.get_level_values(0).min().date(),
                     df_idx.index.get_level_values(0).max().date())
        except Exception as exc:  # noqa: BLE001
            log.warning("指数 %s 拉取失败: %s", idx_code, exc)

    # ---- 3) 全A日线分批回填 ----
    codes_sdk = ds.get_code_list("EXTRA_STOCK_A")
    daily_path = root / "daily_all_a.parquet"
    local_codes: set[str] = set()
    if daily_path.exists():
        local_codes = set(pd.read_parquet(daily_path).index.get_level_values("code").unique())
    all_codes = sorted(set(codes_sdk) | local_codes)
    log.info("全A代码: SDK %d ∪ 本地 %d = %d 只", len(codes_sdk), len(local_codes), len(all_codes))

    batches = [all_codes[i:i + args.batch] for i in range(0, len(all_codes), args.batch)]
    prog_path = root / "_fetch_progress_alla_2015.json"
    prog = _load_progress(prog_path)
    done_batches = set(prog["daily_batches_done"])

    merged = pd.read_parquet(daily_path) if daily_path.exists() else pd.DataFrame()
    t0 = time.time()
    n_done_this_run = 0
    for bi, batch in enumerate(batches):
        if bi in done_batches:
            continue
        df = None
        for attempt in (1, 2, 3):
            try:
                df = ds.get_daily_kline(batch, args.begin, end_date)
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("批 %d/%d 第 %d 次失败: %s", bi + 1, len(batches),
                            attempt, str(exc)[:120])
                time.sleep(5 * attempt)
        if df is None:
            log.error("批 %d/%d 三次失败，跳过（重跑本脚本可补）", bi + 1, len(batches))
            continue
        if len(df):
            merged = pd.concat([merged, df], axis=0)
            # 全量本地合并去重（保留最新拉取）
            idx_names = merged.index.names
            merged = (merged.reset_index()
                      .drop_duplicates(subset=idx_names, keep="last")
                      .set_index(idx_names).sort_index())
        done_batches.add(bi)
        prog["daily_batches_done"] = sorted(done_batches)
        _save_progress(prog_path, prog)
        n_done_this_run += 1
        if n_done_this_run % 5 == 0:
            merged.to_parquet(daily_path)
            log.info("批 %d/%d 完成，已落盘 %d 行", bi + 1, len(batches), len(merged))
        log.info("批 %d/%d: +%d 行 (累计 %d, %.0fs)", bi + 1, len(batches),
                 len(df), len(merged), time.time() - t0)

    merged.to_parquet(daily_path)
    log.info("日线回填完成: %d 行, 代码 %d, %s ~ %s", len(merged),
             merged.index.get_level_values("code").nunique(),
             merged.index.get_level_values(0).min().date(),
             merged.index.get_level_values(0).max().date())

    # ---- 4) 更新 _meta.json 水位 ----
    meta_path = root / "_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["daily_all_a"] = {
        "last_date": int(pd.Timestamp(merged.index.get_level_values(0).max()).strftime("%Y%m%d")),
        "pool": "all_a",
        "size": int(len(merged)),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("meta 已更新: daily_all_a.last_date=%s", meta["daily_all_a"]["last_date"])
    log.info("全部完成，耗时 %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
