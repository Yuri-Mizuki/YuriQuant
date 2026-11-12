"""因子库回填：体检（dry-run）+ 并行补 canonical 评估。

背景（2026-11-02）
------------------
库有两条入库通道：``register()``（完整，131.4 s/因子，其中回测占 97%）与
builder 的 ``upsert_rows()``（轻量，只写 registry 行）。后者让 all_a 的
798 个因子既无 eval 也无标签，库分成两半。把 ``register`` 拆成
``ingest``（便宜）+ ``evaluate``（贵）之后，补评估不再需要重跑整条链路，
也不重写 45 GB 面板。

两种用法
--------
体检（只读，不写任何东西）::

    python -m scripts.factors.backfill_library --dataset all_a_2018_2026 --dry-run

补评估（并行，父进程统一写 registry）::

    python -m scripts.factors.backfill_library --dataset all_a_2018_2026 \
        --evaluate --top 100 --workers 6

``--top`` 选取口径是 ``|ic_mean_h20|`` 降序（库里唯一普遍有值的统计列）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402
from research.factor_library import FactorLibrary, name_slug  # noqa: E402

log = setup_logging("backfill_library")


def _num(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _blank(v) -> bool:
    """NaN / None / 空串 / 字符串 "nan" 都算空。

    registry 经 ``_align_registry`` 对齐后会缺列补 NaN，而 ``str(nan)="nan"``
    既非空串也非 None——直接 ``if not v`` 会把 798 个未标注因子判成"有标签"。
    """
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    s = str(v).strip()
    return s == "" or s.lower() == "nan"


def audit(lib: FactorLibrary) -> pd.DataFrame:
    """逐行体检：面板齐不齐、行数对不对、评估有没有、标签空不空。"""
    reg = lib._load_registry()
    rows = []
    for _, r in reg.iterrows():
        name = r["name"]
        issues: list[str] = []
        p = lib._resolve_panel_path(r)
        panel_rows, first, last = 0, "", ""
        if p is None:
            issues.append("panel_missing")
        else:
            try:
                idx = pd.read_parquet(p, columns=[]).index
                panel_rows = len(idx)
                if panel_rows:
                    first = str(pd.Timestamp(idx.min()).date())
                    last = str(pd.Timestamp(idx.max()).date())
            except Exception as e:  # noqa: BLE001
                issues.append(f"panel_unreadable:{type(e).__name__}")
            else:
                nd = _num(pd.Series([r.get("n_dates")])).iloc[0]
                if not np.isfinite(nd):
                    issues.append("n_dates_null")
                elif int(nd) != panel_rows:
                    issues.append("rows_mismatch")
        if _blank(r.get("evaluated_at")):
            issues.append("no_eval")
        if _blank(r.get("family")):
            issues.append("no_family")
        rows.append({
            "name": name,
            "set": r.get("set", ""),
            "maturity": r.get("maturity", ""),
            "eval_status": "(空)" if _blank(r.get("eval_status")) else r.get("eval_status"),
            "has_panel": p is not None,
            "panel_rows": panel_rows,
            "n_dates_reg": r.get("n_dates"),
            "first": first,
            "last": last,
            "ic_h20": _num(pd.Series([r.get("ic_mean_h20")])).iloc[0],
            "issue": "|".join(issues),
        })
    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame, lib: FactorLibrary) -> None:
    n = len(df)
    issue = df["issue"].fillna("")
    flag = lambda key: issue.str.split("|").apply(lambda xs: key in xs)  # noqa: E731

    print(f"\n数据集 {lib.dataset}：{n} 个因子")
    print(f"  有 canonical 评估：{int((~flag('no_eval')).sum())}/{n}")
    print(f"  面板文件齐全：{int((~flag('panel_missing') & ~issue.str.contains('panel_unreadable')).sum())}/{n}")
    print(f"  n_dates 已登记且吻合：{int((~flag('n_dates_null') & ~flag('rows_mismatch')).sum())}/{n}")
    print(f"  有 family 标签：{int((~flag('no_family')).sum())}/{n}")

    print("\n问题分布：")
    kinds = issue.replace("", "ok").str.split("|").explode().value_counts()
    for k, v in kinds.items():
        print(f"  {k:24s} {v}")


def _eval_chunk(task: tuple[str, str, list[str], dict]):
    """子进程：一次吃一整块因子，日线收益面板只读一次。

    按块分发而不是逐因子，是因为每个 worker 起手都要读 45 MB 面板 + 建收益
    面板（约 3 s）；逐因子分发等于把这份固定开销平摊到每个因子上。

    ``opts`` 原样透传给 :meth:`FactorLibrary.evaluate`（``use_mask`` /
    ``recompute_ic`` 等），保证父进程的口径开关真的落到 worker 里。
    """
    root, dataset, names, opts = task
    t0 = time.time()
    ok: list[tuple[str, dict]] = []
    fail: list[tuple[str, str]] = []
    lib = FactorLibrary(root=root, dataset=dataset)
    for name in names:
        try:
            ok.append((name, lib.evaluate(name, persist=False, **opts)))
        except Exception as e:  # noqa: BLE001
            fail.append((name, f"{type(e).__name__}: {e}"))
    return ok, fail, time.time() - t0


def _chunks(seq: list[str], n: int) -> list[list[str]]:
    k = max(1, -(-len(seq) // n))
    return [seq[i:i + k] for i in range(0, len(seq), k)]


def cmd_fill_ic(lib: FactorLibrary, args) -> None:
    """只补 registry 的 IC 检验列（``ic_mean`` / ``t_stat_nw`` …），不重跑回测。

    builder 通道当年只写 ``ic_mean_h1..h20``，``ic_mean`` / ``t_stat_nw`` 一直
    是空的——于是「按显著性排序」在半个库上失效。这些因子已经补过 eval，
    ``ic`` 序列就存在 eval 文件里，直接读出来算即可（毫秒级/个）。
    """
    from research.factor_library import ic_stats  # noqa: PLC0415

    reg = lib._load_registry()
    t = pd.to_numeric(reg["t_stat_nw"], errors="coerce")
    todo = reg[t.isna()]
    if args.limit:
        todo = todo.head(args.limit)
    if todo.empty:
        print("所有因子都已有 IC 检验统计，无需补。")
        return

    filled: dict[str, dict] = {}
    skipped = 0
    for _, r in todo.iterrows():
        p = lib._resolve_eval_path(r) or (lib.evals_dir / f"{name_slug(r['name'])}.parquet")
        try:
            ev = pd.read_parquet(p, columns=["ic"])
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        filled[r["name"]] = ic_stats(ev["ic"])

    if filled:
        # 分批落盘，避免单次 registry 太大 + 中途被杀丢全部
        items = list(filled.items())
        for i in range(0, len(items), 100):
            lib.apply_eval_many(dict(items[i:i + 100]))
    print(f"补齐 {len(filled)} 个因子的 IC 统计；缺 eval 文件跳过 {skipped} 个")


def cmd_sync_status(lib: FactorLibrary, args) -> None:
    """把"早就有 eval 但没状态列"的行对齐到状态机（不改任何指标）。

    ``hs300_2022_2025`` / ``zz1000_2022_2025`` 的 851 行是旧版 ``register``
    写的：eval 文件与 sharpe 都在，但 ``eval_status`` / ``evaluated_at``
    （2026-11-02 才加的列）为空。不补的话体检会把它们全报成"未评估"，
    两段式状态机在三个库上就不一致了。``evaluated_at`` 取 eval 文件 mtime。
    """
    from datetime import datetime as _dt  # noqa: PLC0415

    reg = lib._load_registry()
    sharpe = pd.to_numeric(reg["sharpe_ls_M"], errors="coerce")
    need = reg[reg["eval_status"].isna() | (reg["eval_status"].astype(str) == "")] \
        if "eval_status" in reg.columns else reg
    todo = need[sharpe.reindex(need.index).notna()]
    if args.limit:
        todo = todo.head(args.limit)

    payload: dict[str, dict] = {}
    for _, r in todo.iterrows():
        p = lib._resolve_eval_path(r) or (lib.evals_dir / f"{name_slug(r['name'])}.parquet")
        if not Path(p).exists():
            continue
        ts = _dt.fromtimestamp(Path(p).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        payload[r["name"]] = {"eval_status": "evaluated", "evaluated_at": ts}

    if payload:
        lib.apply_eval_many(payload)
    print(f"状态已对齐 {len(payload)} 行（有 eval 但缺 eval_status/evaluated_at）")


def cmd_dry_run(lib: FactorLibrary, args) -> None:
    df = audit(lib)
    _summarize(df, lib)

    todo = df[~df["eval_status"].eq("evaluated")].copy()
    todo["abs_ic"] = todo["ic_h20"].abs()
    todo = todo.sort_values("abs_ic", ascending=False)
    print(f"\n待补评估 {len(todo)} 个；|IC20| 头部 15 个：")
    cols = ["name", "set", "ic_h20", "panel_rows", "maturity", "issue"]
    print(todo[cols].head(15).to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n明细已写：{out}")


def cmd_evaluate(lib: FactorLibrary, args) -> None:
    df = audit(lib)
    pool = df[df["has_panel"]]
    if args.names:
        want = [s.strip() for s in args.names.split(",") if s.strip()]
        pool = pool[pool["name"].isin(want)]
        missing = sorted(set(want) - set(pool["name"]))
        if missing:
            log.warning("以下名字不存在或缺面板，已跳过：%s", missing)
    else:
        if not args.force:
            pool = pool[~pool["eval_status"].eq("evaluated")]
        pool = pool.assign(abs_ic=pool["ic_h20"].abs()).sort_values(
            "abs_ic", ascending=False)
    if args.top:
        pool = pool.head(args.top)
    if args.limit:
        pool = pool.head(args.limit)
    names = pool["name"].tolist()
    if not names:
        print("没有需要补评估的因子。")
        return
    n_w = max(1, min(args.workers, len(names)))
    # 块数取 2×进程数：既保持"每个 worker 只读一次日线"的摊薄效果，又把
    # 单块时长压到 ~5 分钟，落盘更频繁（进程被中断时丢的活更少）。
    chunks = _chunks(names, n_w * 2)
    print(f"待补评估 {len(names)} 个，{n_w} 进程 / {len(chunks)} 块"
          f"（每块约 {len(chunks[0])} 个；逐块落盘，中断可续跑）")

    root = str(lib.base_root)
    opts = {"use_mask": not args.no_mask,
            "mask_lag": args.mask_lag,
            "recompute_ic": args.recompute_ic}
    print(f"口径：use_mask={opts['use_mask']} mask_lag={opts['mask_lag']} "
          f"recompute_ic={opts['recompute_ic']}")
    tasks = [(root, lib.dataset, ch, opts) for ch in chunks]
    n_done = n_fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_w) as ex:
        futs = {ex.submit(_eval_chunk, t): len(t[2]) for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            ok, fail, dt = fut.result()
            if ok:
                lib.apply_eval_many(dict(ok))   # 逐块落盘：进程被杀也不丢已完成部分
                n_done += len(ok)
                last = ok[-1][1].get("best_sharpe", float("nan"))
                log.info("块 %d/%d：%d 个 / %.0fs（最新 best=%.3f）；累计 %d/%d",
                         i, len(chunks), len(ok), dt, last, n_done, len(names))
            if fail:
                n_fail += len(fail)
                for nm, err in fail[:3]:
                    log.warning("失败 %s: %s", nm, err)

    print(f"\n完成 {n_done}/{len(names)}，失败 {n_fail}，"
          f"总耗时 {(time.time()-t0)/60:.1f} 分钟")


def cmd_mark_invalid(lib: FactorLibrary, args) -> None:
    """把冗余审计查出的**死因子**标成 ``maturity=invalid``。

    「死」的判据在 ``dup_vitality.csv`` 里（面板全空 / 整列同值 / 同日跨股票
    只有 1 个取值 / 可排名率 <5%）。这批因子有评估、有 sharpe，看上去一切正常，
    实际截面排序全并列，任何选股链路拿到它们都只会随机挑票。

    为什么不用 ``retired``：retired 是"曾经能用、现在停用"；这批**从来没成立过**。
    也不用留空——空值现在意味着"还没验证"，而它们已经验证过了，结论是不可用。

    幂等：已经是 invalid 的不重复写；``--dry-run`` 只打印。
    """
    base = Path(str(lib.base_root)) / lib.dataset
    f = base / "dup_vitality.csv"
    if not f.exists():
        raise SystemExit(f"缺 {f}：先跑 python -m scripts.factors.dup_audit")
    v = pd.read_csv(f)
    issue = v["issue"].fillna("").astype(str).str.strip()
    bad = v[issue != ""]
    if bad.empty:
        print("没有死因子，无需处理。")
        return

    reg = lib._load_registry().set_index("name")
    stamp = "2026-11-04"
    payload: dict[str, dict] = {}
    for _, r in bad.iterrows():
        nm = r["name"]
        if nm not in reg.index:
            log.warning("死因子 %s 不在 registry，跳过", nm)
            continue
        if str(reg.at[nm, "maturity"]) == "invalid":
            continue
        old_note = "" if pd.isna(reg.at[nm, "note"]) else str(reg.at[nm, "note"])
        tag = f"[invalid {stamp}] {r['issue']}"
        payload[nm] = {
            "maturity": "invalid",
            "note": (old_note + " | " + tag).strip(" |"),
        }
    print(f"死因子 {len(bad)} 个，待标记 {len(payload)} 个（其余已是 invalid）：")
    for _, r in bad.head(30).iterrows():
        print(f"  {r['name']:34s} {r['issue']}")
    if args.dry_run:
        print("\n--dry-run：未写盘。")
        return
    if payload:
        lib.apply_eval_many(payload)
        print(f"\n已标记 {len(payload)} 个为 maturity=invalid。")


def main() -> None:
    ap = argparse.ArgumentParser(description="因子库体检与评估回填")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--root", default=None, help="库根（默认走 config）")
    ap.add_argument("--dry-run", action="store_true", help="只体检，不写任何东西")
    ap.add_argument("--evaluate", action="store_true", help="并行补 canonical 评估")
    ap.add_argument("--force", action="store_true",
                    help="重算已评估的因子（换口径迁移时用）")
    ap.add_argument("--no-mask", action="store_true",
                    help="回测不套可交易掩码（仅用于 A/B 对照）")
    ap.add_argument("--mask-lag", type=int, default=0,
                    help="掩码取样日：0=T 日状态（配 shift(-1) 收益），1=T+1")
    ap.add_argument("--recompute-ic", action="store_true",
                    help="按当前收益面板重算 IC 序列（换口径时必开）")
    ap.add_argument("--top", type=int, default=0, help="按 |ic_h20| 取前 N 个")
    ap.add_argument("--limit", type=int, default=0, help="最多处理 N 个")
    ap.add_argument("--names", default="", help="逗号分隔的因子名（覆盖 --top）")
    ap.add_argument("--workers", type=int, default=max(1, int((os.cpu_count() or 4) * 0.625)))
    ap.add_argument("--sync-status", action="store_true",
                    help="有 eval 但缺 eval_status 的老行，按格式对齐状态机")
    ap.add_argument("--fill-ic", action="store_true",
                    help="只补 registry 的 IC 检验列（读已有 eval，不跑回测）")
    ap.add_argument("--mark-invalid", action="store_true",
                    help="把 dup_vitality.csv 里的死因子标成 maturity=invalid")
    ap.add_argument("--out", default="", help="dry-run 明细 CSV 落盘路径")
    args = ap.parse_args()

    lib = FactorLibrary(root=args.root, dataset=args.dataset)
    if args.sync_status:
        cmd_sync_status(lib, args)
    elif args.fill_ic:
        cmd_fill_ic(lib, args)
    elif args.evaluate:
        cmd_evaluate(lib, args)
    elif args.mark_invalid:
        cmd_mark_invalid(lib, args)
    else:
        cmd_dry_run(lib, args)


if __name__ == "__main__":
    main()
