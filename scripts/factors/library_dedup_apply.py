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
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.factor_library import FactorLibrary  # noqa: E402
from scripts.common.atomic_io import write_csv_atomic  # noqa: E402
from scripts.factors.dup_criteria import (  # noqa: E402
    CORR_TRIGGER,
    MIN_SPAN,
    REL_STD_DEGENERATE,
    classify,
)

PLAN_DEFAULT = Path("reports/library_governance/dedup_decision.csv")
#: 只用于**触发**残差检验的下限，不是判定门槛。判据见 ``dup_criteria``。
#: 🚨 2026-11-13：原为独立的 ``0.95``，与决策层的 ``CORR_TRIGGER=0.85`` **不一致**
#: ⇒ 决策表即便给出 ρ=0.8975 的 `safe_drop`（秩空间 rel_std=7.8e-16 的真等价），
#: 执行层也会以「覆盖 ρ < 0.95」拒批。这是「判据只在 ρ≥0.95 区域生效」的第四道闸。
#: 现收敛到唯一真源，改口径只需改 ``dup_criteria``。
COVER_RHO_MIN = CORR_TRIGGER


def _load_plan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"name", "action", "cover_by", "cover_rho", "n_sel"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"决策表缺列：{sorted(miss)}")
    return df


def _verify_plan(sd: pd.DataFrame, keep: set[str], corr: np.ndarray,
                 names: list[str]) -> list[str]:
    """三重校验；返回违反项（空 = 通过）。

    🚨 判据**不再用** registry 的 ``resid_ic`` / ``resid_t``（那是原始值空间、对
    ``dup_top`` 算的，已被证伪）。改用决策表的**秩空间**列 ``rank_rel_std`` /
    ``rank_resid_ic``，并按 ``dup_criteria`` 的分级判定。决策表若缺这两列
    （旧版本产物）直接拒绝——**拿旧表跑新库比不跑更危险**。
    """
    pos = {n: i for i, n in enumerate(names)}
    bad: list[str] = []
    need = {"rank_rel_std", "rank_resid_ic", "rank_n_days", "rank_note"}
    miss = need - set(sd.columns)
    if miss:
        return [f"决策表缺秩空间列 {sorted(miss)}——请用当前版本重跑 "
                f"`python -m scripts.factors.library_dedup_plan`"]
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
        v = classify({
            "rel_std": float(r.rank_rel_std) if pd.notna(r.rank_rel_std) else float("nan"),
            "ic": float(r.rank_resid_ic) if pd.notna(r.rank_resid_ic) else float("nan"),
            "t": float("nan"), "n_days": int(r.rank_n_days or 0),
            # n_span 决策表没单独存；退化判定只看 rel_std，不受影响
            "n_span": MIN_SPAN if pd.notna(r.rank_rel_std) else 0,
            "degenerate": (pd.notna(r.rank_rel_std)
                           and float(r.rank_rel_std) < REL_STD_DEGENERATE),
        })
        if not v.removable:
            bad.append(f"{r.name}: 秩空间仍有增量（{v.level}，{v.reason}）")
        # 🔑 「曾进定版 selection」**不再**是拒批理由（2026-11-13）。
        # 核实过：出榜链读静态 JSON（`alla_daily_rank.py::load_selection`），
        # 重建链明确不重建 selection ⇒ 退登记不改已落盘清单，只影响
        # 「主动 --stage select 重跑旧年份」。真正要护的是**当期在用**，
        # 而执行层只看决策表，无法独立判断当期 ⇒ 由 ``library_dedup_plan``
        # 把当期在用的判成 ``keep_prod``（永不进 safe_drop）来保证。
        # 这里只留一条「当期在用绝不能走到这」的强断言。
        if bool(getattr(r, "in_prod_now", False)):
            bad.append(f"{r.name}: 当期定版在用，禁止退登记")
    return bad


def _restore(lib, base: Path, reg_path: Path, retired_path: Path,
             names_file: Path, apply: bool, reason: str) -> None:
    """按名单把误退的因子恢复到 registry，**并从台账移除对应行**。

    与 ``--rollback`` 的区别：rollback 按 ``batch_id`` 整批退；restore 按**因子名**
    精确恢复（一次批量里可能只有一部分退错）。恢复后写
    ``restore_batches.jsonl`` 留痕，含当时账面依据，便于日后追溯。

    安全边界：
    - 只恢复「在 ``registry_retired.csv`` 里、且不在 ``registry.csv`` 里」的名字；
      已在册的幂等跳过（防止重复插入）。
    - 名字非法/找不到账本记录 ⇒ 报错退出，不做部分写。
    - 备份 + 原子写 + 事后再读一遍核对。
    """
    if not retired_path.exists():
        raise SystemExit("没有 registry_retired.csv，无从恢复")
    want = [s.strip() for s in names_file.read_text(encoding="utf-8").splitlines()
            if s.strip() and not s.strip().startswith("#")]
    if not want:
        raise SystemExit(f"名单为空：{names_file}")

    reg = pd.read_csv(reg_path)
    led = pd.read_csv(retired_path)
    in_reg, in_led = set(reg["name"]), set(led["name"])
    unknown = [n for n in want if n not in in_led]
    if unknown:
        raise SystemExit(f"{len(unknown)} 个不在台账里，拒绝执行：{unknown[:10]}")
    dup = [n for n in want if n in in_reg]
    todo = [n for n in want if n not in in_reg]
    if dup:
        print(f"注：{len(dup)} 个已在 registry（幂等跳过），如 {dup[:3]}")

    back = led[led["name"].isin(todo)].copy()
    meta = ["batch_id", "retired_at", "reason", "cover_by", "cover_rho"]
    rows = back.drop(columns=[c for c in meta if c in back.columns])
    rows = rows.reindex(columns=reg.columns)
    merged = pd.concat([reg, rows], ignore_index=True)
    if merged["name"].duplicated().any():
        raise SystemExit("!! 合并后出现重名，拒绝写盘")

    print(f"registry: {len(reg)} → {len(merged)}（恢复 {len(todo)}）")
    print(f"台账: {len(led)} → {len(led) - len(todo)}")
    if "family" in reg.columns:
        print("\n按家族:")
        print(back["family"].value_counts().to_string())
    print("\n按前缀:")
    pre = back["name"].str.extract(r"^(alpha\d+|alt_\w+|\w+)")[0]
    print(pre.value_counts().rename(None).to_string())

    if not apply:
        print("\n（dry-run；加 --apply 执行）")
        return

    ts = time.strftime("%Y%m%d-%H%M%S")
    reg_bak = reg_path.with_name(f"registry.csv.bak-{ts}")
    led_bak = retired_path.with_name(f"registry_retired.csv.bak-{ts}")
    reg.to_csv(reg_bak, index=False, encoding="utf-8")
    led.to_csv(led_bak, index=False, encoding="utf-8")
    print(f"\n备份: {reg_bak.name} / {led_bak.name}")

    write_csv_atomic(merged, reg_path)
    write_csv_atomic(led[~led["name"].isin(todo)].reset_index(drop=True), retired_path)

    chk = pd.read_csv(reg_path)
    led_chk = pd.read_csv(retired_path)
    missing = set(todo) - set(chk["name"])
    still = set(todo) & set(led_chk["name"])
    print(f"复核: registry {len(chk)} 行 | 未恢复 {len(missing)} | "
          f"台账残留 {len(still)} | 重复名 {int(chk['name'].duplicated().sum())}")
    if missing or still or chk["name"].duplicated().any():
        raise SystemExit("!! 写盘复核失败，请从备份恢复")

    audit = base / "restore_batches.jsonl"
    with audit.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "batch": f"restore-{ts}", "n": len(todo), "reason": reason,
            "before": len(reg), "after": len(chk), "names": todo,
            "source": str(names_file),
        }, ensure_ascii=False) + "\n")
    print(f"审计: {audit.name}（+1 行）")
    print(f"完成。回滚命令: --rollback-batch restore-{ts}")


def main() -> None:
    ap = argparse.ArgumentParser(description="库内去重执行（退登记 / 恢复）")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--plan", default=str(PLAN_DEFAULT))
    ap.add_argument("--apply", action="store_true", help="真正写盘；默认 dry-run")
    ap.add_argument("--rollback", default="", help="按 batch_id 整批回滚退登记")
    ap.add_argument("--restore", default="", help="按名单文件恢复误退因子（每行一个名）")
    ap.add_argument("--reason", default="dedup_v3: 秩空间等价或微小增量且未进定版")
    args = ap.parse_args()

    lib = FactorLibrary(dataset=args.dataset)
    base = Path(lib.root)
    if not (base / "registry.csv").exists():
        base = base / args.dataset
    reg_path = base / "registry.csv"
    retired_path = base / "registry_retired.csv"

    # ---------- 恢复（按名单） ----------
    if args.restore:
        _restore(lib, base, reg_path, retired_path, Path(args.restore),
                 args.apply, args.reason)
        return

    # ---------- 回滚（按 batch） ----------
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
            write_csv_atomic(merged, reg_path)
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
    write_csv_atomic(ledger, retired_path)
    print(f"台账: {retired_path.name}（累计 {len(ledger)} 行，batch={batch}）")

    write_csv_atomic(after, reg_path)

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
