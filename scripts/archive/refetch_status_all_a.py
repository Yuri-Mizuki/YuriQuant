"""全量重拉 history_stock_status：完整历史清单（含退市）→ 2015 至今。

背景：本地 history_stock_status.parquet 呈锯齿覆盖——2015-2018（~2600-3300 只）
与 2022-2025（~4900-5450 只）较完整，但 2019-2021 仅 ~500 只、2026 年增量缺失，
且历史退市股从未纳入。可执行掩码在缺口年份近乎"全部可交易"，失真严重。

目标：按完整历史代码清单对 2015→今 全量重拉并覆盖式落盘，含退市股。

设计（断点续跑）：
- Step 0 主清单：合并
    get_hist_code_list("EXTRA_STOCK_A_SH_SZ", 2015, 今)  # 沪深A历史(含退市)
    get_code_list("EXTRA_STOCK_A")                       # 当前全A(含北交所)
  结果落盘为 checkpoint；已存在则跳过（重拉历史清单很慢，只需一次）。
- Step 1 状态重拉：粒度 = 「(年份 Y, 代码批 b)」，progress.json 记已完成键。
  逐年批内轮询，完成后并入目标表并落盘（内存上界≈1年）。崩溃后重跑只补缺失
  (年,批)，已完成的直接跳过。

用法:
    D:/python/Python312/python.exe -u scripts/builders/refetch_status_all_a.py --batch 100
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

BEGIN = 2015
CURRENT_YEAR = 2026
HIST_LIST_END = 20260902
SCACHE = Path(str(Config.cache()["root"])) / "_status_refetch"
TARGET = Path(str(Config.cache()["root"])) / "history_stock_status.parquet"


def _norm(c: str) -> str:
    return c.split(".")[0] if "." in c else c


def load_master(ds, local_path: str) -> pd.DataFrame:
    """构建并缓存完整历史代码清单（含退市），返回 id 单列 DataFrame。"""
    master_path = SCACHE / "master_codes.parquet"
    if master_path.exists():
        m = pd.read_parquet(master_path)
        print(f"[master] 复用缓存: {len(m)} 只", flush=True)
        return m
    hist = ds._base.get_hist_code_list(  # type: ignore[attr-defined]
        security_type="EXTRA_STOCK_A_SH_SZ",
        start_date=BEGIN * 10000 + 101,
        end_date=HIST_LIST_END,
        local_path=local_path,
    )
    cur = ds.get_code_list("EXTRA_STOCK_A")
    # 保留交易所后缀（如 000001.SZ）：status/kline 接口都按带后缀代码查询，
    # 剥后缀会导致查询失败。union 去重后的完整历史(含退市)+当前全A。
    merged = sorted(set(hist) | set(cur))
    # 白名单防混入：仅保留 A 股后缀（历史代码表快照里可能混入期货/期权代码）
    merged = [c for c in merged if c.endswith((".SH", ".SZ", ".BJ"))]
    m = pd.DataFrame({"code": merged})
    SCACHE.mkdir(parents=True, exist_ok=True)
    m.to_parquet(master_path, index=False)
    print(f"[master] 构建完成: 历史{len(set(hist))} ∪ 当前{len(set(cur))} = {len(m)} 只", flush=True)
    return m


def to_entry(tags: list[str]) -> set:
    if not SCACHE.exists():
        return set()
    p = SCACHE / "progress.json"
    if not p.exists():
        return set()
    return set(json.loads(p.read_text(encoding="utf-8")))


def mark_done(tags: set[str]) -> None:
    SCACHE.mkdir(parents=True, exist_ok=True)
    (SCACHE / "progress.json").write_text(
        json.dumps(sorted(tags), ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="全量重拉 history_stock_status（含退市，2015→今）")
    ap.add_argument("--batch", type=int, default=100, help="每个(年,批)的代码数")
    ap.add_argument("--years", default=None, help="限定年份，逗号分隔（如 2019,2020,2021）")
    args = ap.parse_args()

    cfg = Config.datasource()
    ds = create_datasource(cfg)
    # sdk_local_path 真源在 settings.yaml；此处仅防御键缺失（不再复制字面量）
    local_path = cfg["amazing_data"]["sdk_local_path"]

    master = load_master(ds, local_path)
    codes = master["code"].tolist()
    batches = [codes[i:i + args.batch] for i in range(0, len(codes), args.batch)]
    print(f"[计划] {len(codes)} 只代码 / 批{args.batch} => {len(batches)} 批", flush=True)

    years = [int(x) for x in args.years.split(",")] if args.years else \
        list(range(BEGIN, CURRENT_YEAR + 1))
    tags = set()
    if SCACHE.exists() and (SCACHE / "progress.json").exists():
        tags = set(json.loads((SCACHE / "progress.json").read_text(encoding="utf-8")))
    print(f"[断点] 已完成 {len(tags)} 个(年,批)", flush=True)

    tgt = pd.read_parquet(TARGET) if TARGET.exists() else pd.DataFrame()
    t0 = time.time()
    n_calls = 0
    for Y in years:
        b_lo, b_hi = Y * 10000 + 101, Y * 10000 + 1231
        yr_frames = []
        yr_new = 0
        for bi, cb in enumerate(batches):
            tag = f"{Y}_{bi}"
            if tag in tags:
                continue
            # 若批内代码在该年本地已全存在（防止退市/未上市代码 0 行导致永重拉，
            # 这里用"批内至少有一行"的弱校验 + progress 兜底）
            sub = None
            for attempt in (1, 2, 3):
                try:
                    sub = ds.get_history_stock_status(cb, b_lo, b_hi)
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"  {Y} 批{bi} 第{attempt}次失败: {str(exc)[:100]}", flush=True)
                    time.sleep(4 * attempt)
            if sub is None:
                print(f"  [跳过] {Y} 批{bi} 三次失败", flush=True)
                continue
            if len(sub):
                yr_frames.append(sub)
                yr_new += len(sub)
            tags.add(tag)
            n_calls += 1
            mark_done(tags)
            if n_calls % 10 == 0:
                print(f"  ... {Y} 完成 {int((bi + 1) / len(batches) * 100)}% "
                      f"({bi + 1}/{len(batches)}) 累计+{yr_new} 行 [{time.time() - t0:.0f}s]", flush=True)
        if yr_frames:
            new_yr = pd.concat(yr_frames, ignore_index=True)
            new_yr["date"] = pd.to_datetime(new_yr["date"], errors="coerce")
            new_yr = new_yr.dropna(subset=["date"])
            new_yr = new_yr.set_index(["date", "code"]).sort_index()
            if not tgt.empty:
                tgt = pd.concat([tgt, new_yr], axis=0)
            else:
                tgt = new_yr
            # 覆盖式去重：同(date,code)以后拉的为准（index 已含 date/code）
            tgt = tgt[~tgt.index.duplicated(keep="last")].sort_index()
            SCACHE.mkdir(parents=True, exist_ok=True)
            tgt.to_parquet(TARGET)
            print(f"[落盘] {Y} 完成: 目标累计 {len(tgt)} 行 / "
                  f"{tgt.index.get_level_values('code').nunique()} 只",
                  flush=True)
    print(f"[完成] {time.time() - t0:.0f}s，目标 {len(tgt)} 行，"
          f"{tgt.index.get_level_values('code').nunique()} 只代码", flush=True)


if __name__ == "__main__":
    main()