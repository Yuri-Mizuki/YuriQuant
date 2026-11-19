"""E5 覆盖层体检：``panels_neu_fundind`` vs 基准层 ``panels_neu``。

背景
----
两层 950 个同名文件里 863 个**逐字节相同**（2026-11-18 全量 md5 实测）= 48.55 GB
纯重复：建库脚本按因子集整批落盘，没做「结果相同就不写」的判定。2026-11-18 起
``panels_neu_fundind`` 改为**覆盖层**语义（``alla_common.fundind_dirs()`` 优先读
覆盖层、未命中回落基准层），因此：

* 覆盖层里与基准层**内容相同**的文件 = 可直接删除的冗余（``redundant``）
* 覆盖层里与基准层**内容不同** = 有效覆盖（``override``，必须保留）
* 覆盖层有、基准层无 = 基准层缺口（``only_fundind``，保留，顺带提示补基准层）
* 基准层有、覆盖层无、且属基本面/股东族 = **口径错误**（会静默退化成全剥）

**例外（2026-11-19 加）**：基本面/股东族因子即使与基准层逐字节相同也**不判冗余**
（``pinned``）。因为 ``fundind_dirs()`` 要求这批因子在覆盖层显式存在，缺了会
``SystemExit``；实测 828 个冗余里只命中 1 个 ``div_yield_ttm``（两层都是全空面板，
删掉后口径检查立刻挂）。

用法
----
    python -m scripts.data_tools.audit_fundind_split              # 只报，不动
    python -m scripts.data_tools.audit_fundind_split --no-md5     # 只按 size 粗筛
    python -m scripts.data_tools.audit_fundind_split --prune      # 冗余项走回收站
    python -m scripts.data_tools.audit_fundind_split --prune --purge   # 永久删除

``--prune`` 前会把每个待删文件的 md5 写进
``e:/data/_archive/fundind_prune_<时间戳>/MANIFEST.json`` —— 被删的都是「与基准层
同名文件内容相同」的副本，恢复方式 = 从 ``panels_neu`` 复制回来，清单用于对账。

验收锚（2026-11-18）：prune 后 ``override=99``、``redundant=0``、可回收 0 GB。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.alla_common import (  # noqa: E402
    FUNDAMENTAL_FAMILY_SETS,
    FUNDIND_BASE_SUBDIR,
    FUNDIND_SUBDIR,
    ds_root,
)

ARCHIVE = Path(ds_root()).parent.parent / "_archive"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(ds: Path, use_md5: bool = True) -> dict:
    """把覆盖层每个文件归入 redundant / override / only_fundind。"""
    base, ov = ds / FUNDIND_BASE_SUBDIR, ds / FUNDIND_SUBDIR
    base_files = {p.stem: p for p in base.glob("*.parquet")}
    out = {"redundant": [], "override": [], "only_fundind": [], "detail": {}}
    for p in sorted(ov.glob("*.parquet")):
        b = base_files.get(p.stem)
        if b is None:
            out["only_fundind"].append(p.stem)
            out["detail"][p.stem] = {"kind": "only_fundind", "bytes": p.stat().st_size}
            continue
        same = b.stat().st_size == p.stat().st_size and (
            not use_md5 or md5_of(b) == md5_of(p))
        # 基本面/股东族**永不判冗余**：``fundind_dirs()`` 要求这些因子在覆盖层
        # 显式存在（缺了直接 SystemExit，防静默退化成「行业+市值全剥」），
        # 即使内容与基准层逐字节相同也必须留。实测 828 个冗余里只命中 1 个
        # （div_yield_ttm —— 两层都是全空面板，判冗余删掉后口径检查立挂）。
        pinned = p.stem in FUNDAMENTAL_FAMILY_SETS
        kind = "redundant" if (same and not pinned) else "override"
        out[kind].append(p.stem)
        out["detail"][p.stem] = {
            "kind": kind, "bytes": p.stat().st_size, "pinned": pinned,
            "md5": md5_of(p) if (use_md5 and not same) else None,
        }
    out["base_count"] = len(base_files)
    out["fundind_family_missing"] = sorted(
        n for n in FUNDAMENTAL_FAMILY_SETS
        if n in base_files and n not in {q.stem for q in ov.glob("*.parquet")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="E5 覆盖层体检（冗余/有效覆盖/口径风险）")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--no-md5", action="store_true",
                    help="只按文件大小粗筛（快，但同 size 不同内容会误判为冗余）")
    ap.add_argument("--prune", action="store_true",
                    help="把与基准层内容相同的覆盖文件走回收站")
    ap.add_argument("--purge", action="store_true",
                    help="配合 --prune：永久删除而不进回收站（被删项内容在基准层仍在）")
    ap.add_argument("--out", default=None, help="把结果写成 json")
    args = ap.parse_args()

    ds = ds_root().parent / args.dataset   # ds_root() 已含 DATASET，取其父目录
    print(f"数据集 {ds}", flush=True)
    res = classify(ds, use_md5=not args.no_md5)
    n_red = len(res["redundant"])
    gb = sum(res["detail"][n]["bytes"] for n in res["redundant"]) / 1e9
    print(f"基准层 {res['base_count']} 个 / 覆盖层 "
          f"{len(res['detail'])} 个（有效覆盖 {len(res['override'])}、"
          f"仅覆盖层 {len(res['only_fundind'])}、冗余 {n_red}）")
    if n_red:
        print(f"可回收 {gb:.2f} GB（{n_red} 个与基准层逐字节相同）")
        print(f"  样例：{res['redundant'][:5]}")
    if res["only_fundind"]:
        print(f"仅在覆盖层（基准层缺口）：{res['only_fundind'][:8]}")
    if res["fundind_family_missing"]:
        print(f"!! 基本面/股东族缺覆盖（会静默退化成全剥）："
              f"{res['fundind_family_missing'][:8]}", file=sys.stderr)
        return 1

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"结果 -> {args.out}")

    if args.prune and n_red:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = ARCHIVE / f"fundind_prune_{stamp}"
        snap.mkdir(parents=True, exist_ok=True)
        man = [{"name": n, "bytes": res["detail"][n]["bytes"],
                "md5": md5_of(ds / FUNDIND_SUBDIR / f"{n}.parquet"),
                "recover_from": str(ds / FUNDIND_BASE_SUBDIR / f"{n}.parquet")}
               for n in res["redundant"]]
        (snap / "MANIFEST.json").write_text(
            json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"清单 -> {snap / 'MANIFEST.json'}")

        if not args.purge:
            from scripts.data_tools.recycle import recycle

            paths = [str(ds / FUNDIND_SUBDIR / f"{n}.parquet")
                     for n in res["redundant"]]
            for i in range(0, len(paths), 10):
                rc = recycle(paths[i:i + 10])
                if rc != 0:
                    print(f"第 {i // 10 + 1} 批回收站返回 {rc}", file=sys.stderr)
                    return 1
                print(f"  回收站 {min(i + 10, len(paths))}/{len(paths)}", flush=True)
            print("注意：文件仍在回收站里占空间，清空回收站才真正释放。")
        else:
            for n in res["redundant"]:
                (ds / FUNDIND_SUBDIR / f"{n}.parquet").unlink()
            print(f"已永久删除 {n_red} 个（内容在基准层仍在，恢复见 MANIFEST）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
