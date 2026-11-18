"""registry 备份轮转：只保留「有用的快照」，其余走回收站。

为什么需要它
------------
``registry.csv`` 的写入方（``library_dedup_apply`` / ``formula_extract`` /
``backfill_library_metrics``）都遵循「改前先 ``cp`` 一份 ``registry.csv.bak-<ts>``」
的安全惯例，但**没有任何一方负责删除**。结果是两个月堆了 20 份、且同一次
变更里会有 5 分钟内连拍 5 份的中间态（formula 那批 11:42~11:44 五连拍）。

保留判据（三条任一命中即保留）
------------------------------
(a) **被点名**：文件名出现在 ``reports/**/*.md``、``TODO.md``、``scripts/**/*.py``
    里 —— 这些是文档里写的回滚命令、或代码里的 ``glob`` 依赖，删了会断链。
(b) **最新 N 份**（``--keep``，默认 5）：最近一次变更前的状态，是事故时的第一现场。
(c) ``--protect`` 显式点名的。

安全边界
--------
1. **默认 dry-run**，只打印计划；加 ``--apply`` 才动。
2. **不永久删除**：走 ``_recycle_batch.recycle()``（Windows 回收站）。
3. 只认 ``registry*.csv.bak-*`` / ``registry.backup_*`` 这两类命名，别的一概不碰。
4. ``--apply`` 前先把待删文件 ``cp`` 到 ``<data>/_archive/registry_bak_<ts>/`` 并写
   ``MANIFEST.json``（md5 校验），仓库外留底。

用法
----
    # 看计划（默认不动）
    python -m scripts.factors.prune_registry_backups --dataset all_a_2018_2026

    # 执行
    python -m scripts.factors.prune_registry_backups --dataset all_a_2018_2026 --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.alla_common import ds_root  # noqa: E402

#: 被点名扫描的文档/代码根（相对仓库根）
SCAN_GLOBS = (
    "reports/**/*.md",
    "TODO.md",
    "TODO_ARCHIVE.md",
    "scripts/**/*.py",
    "config/**/*.yaml",
    "monitoring/**/*.py",
    "research/**/*.py",
)
SCAN_SKIP_DIRS = {"archive", "__pycache__", ".venv"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_backup(p: Path) -> bool:
    n = p.name
    if not n.startswith("registry"):
        return False
    return ".bak-" in n or ".backup_" in n


def _scan_mentions(repo: Path) -> dict[str, list[str]]:
    """文件名 → 点名它的文件列表（相对仓库根）。"""
    hits: dict[str, list[str]] = {}
    for g in SCAN_GLOBS:
        for f in repo.glob(g):
            if not f.is_file() or any(s in f.parts for s in SCAN_SKIP_DIRS):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                if "registry" not in line or (".bak-" not in line and ".backup_" not in line):
                    continue
                for tok in line.replace('"', " ").replace("'", " ").replace("`", " ").split():
                    tok = tok.strip("(),;:")
                    if _is_backup(Path(tok)):
                        hits.setdefault(Path(tok).name, []).append(str(f.relative_to(repo)))
    return hits


def plan_backups(
    ds_dir: Path, keep: int, protect: set[str], repo: Path
) -> tuple[list[Path], list[Path], dict[str, list[str]]]:
    """返回 (待删, 保留, 点名表)。"""
    cands = sorted((p for p in ds_dir.iterdir() if p.is_file() and _is_backup(p)),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    mentions = _scan_mentions(repo)
    newest = {p.name for p in cands[:keep]}
    delete: list[Path] = []
    kept: list[Path] = []
    for p in cands:
        named = p.name in mentions
        if named or p.name in newest or p.name in protect:
            kept.append(p)
        else:
            delete.append(p)
    return delete, kept, mentions


def main() -> None:
    ap = argparse.ArgumentParser(description="registry 备份轮转（默认 dry-run）")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--keep", type=int, default=5, help="保留最新 N 份（默认 5）")
    ap.add_argument("--protect", default="", help="额外保留的文件名，逗号分隔")
    ap.add_argument("--apply", action="store_true", help="执行（默认只看计划）")
    ap.add_argument("--no-snapshot", action="store_true", help="跳过仓库外快照（不建议）")
    args = ap.parse_args()

    ds_dir = ds_root().parent / args.dataset
    if not ds_dir.is_dir():
        raise SystemExit(f"数据集不存在：{ds_dir}")

    protect = {s.strip() for s in args.protect.split(",") if s.strip()}
    delete, kept, mentions = plan_backups(ds_dir, args.keep, protect, _repo_root())

    print(f"数据集 {ds_dir.name}：共 {len(delete) + len(kept)} 份备份"
          f" → 删 {len(delete)} / 留 {len(kept)}")
    for p in kept:
        why = []
        if p.name in mentions:
            why.append("点名:" + ",".join(sorted(set(mentions[p.name]))[:2]))
        if p.name in protect:
            why.append("--protect")
        if not why:
            why.append("最新 N 份")
        print(f"  KEEP {p.stat().st_size / 1e6:7.3f}MB  {p.name:52s} {' | '.join(why)}")
    for p in delete:
        print(f"  DEL  {p.stat().st_size / 1e6:7.3f}MB  {p.name}")

    if not delete:
        print("\n没有可回收的备份。")
        return
    total = sum(p.stat().st_size for p in delete)
    print(f"\n可回收 {total / 1e6:.2f} MB")

    if not args.apply:
        print("\n（dry-run；加 --apply 执行，走回收站）")
        return

    if not args.no_snapshot:
        ts = time.strftime("%Y%m%d-%H%M%S")
        snap = ds_root().parent / "_archive" / f"registry_bak_{ts}"
        snap.mkdir(parents=True, exist_ok=True)
        man = []
        for p in delete:
            dst = snap / p.name
            shutil.copy2(p, dst)
            man.append({"name": p.name, "bytes": p.stat().st_size,
                        "md5": hashlib.md5(p.read_bytes()).hexdigest()})
        (snap / "MANIFEST.json").write_text(
            json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        bad = [m["name"] for m in man
               if hashlib.md5((snap / m["name"]).read_bytes()).hexdigest() != m["md5"]]
        if bad:
            raise SystemExit(f"!! 快照校验失败，中止：{bad[:5]}")
        print(f"快照 {len(man)} 个 → {snap}（md5 全对）")

    from scripts.data_tools.recycle import recycle  # 延迟导入：仅 --apply 需要

    rc = recycle([str(p) for p in delete])
    if rc != 0:
        raise SystemExit(f"!! 回收站返回 {rc}，有文件未删，请检查")
    left = [p.name for p in delete if p.exists()]
    if left:
        raise SystemExit(f"!! 仍有 {len(left)} 个未删：{left[:5]}")
    print(f"完成：{len(delete)} 个已入回收站，复核无残留。")


if __name__ == "__main__":
    main()
