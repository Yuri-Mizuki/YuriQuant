"""库内去重的**执行层**：把决策表的 ``safe_drop`` 从 registry 退登记。

为什么"退登记"而不是"删文件"
------------------------------
`stage_select` 的候选池是 ``cands = [n for n in quality.index
if n in cov_ok.index and cov_ok[n] >= MIN_COVERAGE]``，其中:

- ``quality.index`` 来自 ``ic_h*.parquet`` 的列 → 删面板/删 ic 列**会改变输入**
- ``cov_ok.index`` 来自 ``registry.csv`` → **删 registry 行即可退出候选池**

``_extra``（基本面/另类补位池）同样从 ``cov_ok.index`` 取名，所以退登记对
四条入选路径一次生效。面板、evals、ic 列**全部保留**，回滚只需把
``registry_retired.csv`` 的行拼回 ``registry.csv``。

安全边界
--------
1. 只处理 ``action == safe_drop``；``review`` 一律不动（需组合层 A/B）。
2. 双重校验决策表的三条判据仍然成立（覆盖者存在且 ρ 达标、残差无增量、
   从未进定版）——防的是"拿旧决策表跑在新库上"。
3. 覆盖者必须在**退登记后仍保留**的集合里，否则会把一整支掏空。
4. 备份 + 原子写 + 事后再读一遍核对行数。

    # 看清楚要做什么（默认 dry-run）
    python -m scripts.factors.library_dedup_apply --dataset all_a_2018_2026

    # 执行
    python -m scripts.factors.library_dedup_apply --dataset all_a_2018_2026 --apply

    # 回滚（把某次退登记整批拼回）
    python -m scripts.factors.library_dedup_apply --rollback 20261112-140500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.factor_library import FactorLibrary  # noqa: E402

PLAN_DEFAULT = Path("reports/library_governance/dedup_decision.csv")
COVER_RHO_MIN = 0.95
RESID_IC_MAX = 0.02
RESID_T_MAX = 2.0


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """临时文件 + os.replace：Windows 上直接 to_csv 覆盖会偶发 PermissionError。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.3 * (2 ** attempt))


def _load_plan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"name", "action", "cover_by", "cover_rho", "resid_ic", "resid_t", "n_sel"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"决策表缺列：{sorted(miss)}")
    return df


def _verify_plan(sd: pd.DataFrame, keep: set[str], corr: np.ndarray,
                 names: list[str]) -> list[str]:
    """三重校验；返回违反项（空 = 通过）。"""
    pos = {n: i for i, n in enumerate(names)}
    bad: list[str] = []
    for r in sd.itertuples():
        if r.cover_by not in keep:
            bad.append(f"{r.name}: 覆盖者 {r.cover_by} 不在保留集")
            continue
        i, j = pos.get(r.name), pos.get(r.cover_by)
        if i is None or j is None:
            bad.append(f"{r.name}: 不在冗余索引里")
            continue
        rho = abs(float(corr[i, j]))
        if rho < COVER_RHO_MIN - 1e-6:
            bad.append(f"{r.name}: 覆盖 ρ={rho:.3f} < {COVER_RHO_MIN}")
        if abs(float(r.resid_ic)) >= RESID_IC_MAX or abs(float(r.resid_t)) >= RESID_T_MAX:
            bad.append(f"{r.name}: 残差有增量 ic={r.resid_ic} t={r.resid_t}")
        if int(r.n_sel) != 0:
            bad.append(f"{r.name}: 曾进定版 selection {int(r.n_sel)} 次")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description="库内去重执行（退登记）")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--plan", default=str(PLAN_DEFAULT))
    ap.add_argument("--apply", action="store_true", help="真正写盘；默认 dry-run")
    ap.add_argument("--rollback", default="", help="按 batch_id 回滚")
    ap.add_argument("--reason", default="dedup_v3: ρ覆盖且残差无增量且未进定版")
    args = ap.parse_args()

    lib = FactorLibrary(dataset=args.dataset)
    base = Path(lib.root)
    if not (base / "registry.csv").exists():
        base = base / args.dataset
    reg_path = base / "registry.csv"
    retired_path = base / "registry_retired.csv"

    # ---------- 回滚 ----------
    if args.rollback:
        if not retired_path.exists():
            raise SystemExit("没有 registry_retired.csv，无从回滚")
        old = pd.read_csv(retired_path)
        if "batch_id" not in old.columns:
            raise SystemExit("retired 台账缺 batch_id 列")
        back = old[old["batch_id"] == args.rollback]
        if back.empty:
            raise SystemExit(f"没有 batch_id={args.rollback} 的记录；"
                             f"现有 {sorted(old['batch_id'].unique())[:8]}")
        reg = pd.read_csv(reg_path)
        meta = ["batch_id", "retired_at", "reason", "cover_by", "cover_rho"]
        rows = back.drop(columns=[c for c in meta if c in back.columns])
        rows = rows.reindex(columns=reg.columns)
        merged = pd.concat([reg, rows], ignore_index=True).drop_duplicates(
            subset="name", keep="last")
        if args.apply:
            bak = reg_path.with_name(f"registry.csv.bak-{time.strftime('%Y%m%d-%H%M%S')}")
            reg.to_csv(bak, index=False, encoding="utf-8")
            _write_csv_atomic(merged, reg_path)
        print(f"回滚 {len(rows)} 行 → registry {len(reg)} → {len(merged)}"
              + ("" if args.apply else "（dry-run）"))
        return

    # ---------- 退登记 ----------
    plan = _load_plan(Path(args.plan))
    sd = plan[plan["action"] == "safe_drop"].copy()
    if sd.empty:
        raise SystemExit("决策表里没有 safe_drop")

    reg = pd.read_csv(reg_path)
    reg_names = set(reg["name"])
    present = [n for n in sd["name"] if n in reg_names]
    missing = sorted(set(sd["name"]) - reg_names)

    idx = lib.dup_index()
    if idx is None:
        raise SystemExit("冗余索引不可用（先跑 dup_audit），无法校验覆盖关系")
    keep = (set(plan.loc[plan["action"] != "safe_drop", "name"]) | (reg_names - set(sd["name"])))
    bad = _verify_plan(sd[sd["name"].isin(reg_names)], keep,
                       np.asarray(idx["corr"]), [str(n) for n in idx["names"]])
    if bad:
        print("!! 校验未通过，拒绝执行（前 10 条）：")
        for b in bad[:10]:
            print("   ", b)
        raise SystemExit(1)
    if missing:
        print(f"注：{len(missing)} 个已在库外（幂等跳过），如 {missing[:3]}")

    after = reg[~reg["name"].isin(present)].reset_index(drop=True)
    print(f"registry: {len(reg)} → {len(after)}（退登记 {len(present)}）")
    print(f"保留 {len(after)} 个；面板/evals/ic 列一律不动")
    if "family" in reg.columns:
        print("\n按家族:")
        print(reg[reg["name"].isin(present)]["family"].value_counts().to_string())
    print("\n按前缀:")
    pre = pd.Series(present).str.extract(r"^(alpha\d+|alt_\w+|\w+)")[0]
    print(pre.value_counts().rename(None).to_string())

    if not args.apply:
        print("\n（dry-run；加 --apply 执行）")
        return

    # 备份 → 台账 → 原子写
    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = reg_path.with_name(f"registry.csv.bak-{ts}")
    reg.to_csv(bak, index=False, encoding="utf-8")
    print(f"\n备份: {bak.name}")

    batch = f"dedup-{ts}"
    ledger = reg[reg["name"].isin(present)].copy()
    ledger["batch_id"] = batch
    ledger["retired_at"] = ts
    ledger["reason"] = args.reason
    ledger["cover_by"] = ledger["name"].map(dict(zip(sd["name"], sd["cover_by"])))
    ledger["cover_rho"] = ledger["name"].map(dict(zip(sd["name"], sd["cover_rho"])))
    if retired_path.exists():
        old = pd.read_csv(retired_path)
        ledger = pd.concat([old, ledger], ignore_index=True).drop_duplicates(
            subset=["name", "batch_id"], keep="last")
    _write_csv_atomic(ledger, retired_path)
    print(f"台账: {retired_path.name}（累计 {len(ledger)} 行，batch={batch}）")

    _write_csv_atomic(after, reg_path)

    # 复核（写完再读一遍，不信内存）
    chk = pd.read_csv(reg_path)
    residual = set(present) & set(chk["name"])
    print(f"\n复核: registry {len(chk)} 行 | 残留退登记名 {len(residual)} | "
          f"重复名 {int(chk['name'].duplicated().sum())}")
    if residual or len(chk) != len(after):
        raise SystemExit("!! 写盘复核失败，请从备份恢复")

    (base / f"dedup_batch_{ts}.json").write_text(json.dumps({
        "batch_id": batch, "n_retired": len(present), "before": len(reg),
        "after": len(chk), "plan": str(args.plan), "reason": args.reason,
        "names": present,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成。回滚命令: --rollback {ts}")


if __name__ == "__main__":
    main()
