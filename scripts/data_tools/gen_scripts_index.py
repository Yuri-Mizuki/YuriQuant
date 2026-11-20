"""生成 ``scripts/README.md`` 入口索引（2026-10-19 审计 ③）。

背景：全仓 190 个脚本，31 个「三类引用全零」——不是都能删（数据回补/研报
数据准备仍有人工流程），但至少要让每个入口可被发现。本生成器按目录列出
每个 tracked 脚本的首行 docstring。`archive/` 已于 2026-11-18 整体下线。

用法:
    python -m scripts.data_tools.gen_scripts_index          # 写入
    python -m scripts.data_tools.gen_scripts_index --check  # CI/本地校验
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scripts" / "README.md"
EXCLUDE_PARTS = {"archive", "__pycache__"}



def _tracked_scripts() -> list[str]:
    out = subprocess.run(["git", "ls-files", "scripts/*.py"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit("不在 git 工作区")
    rows = []
    for rel in out.stdout.splitlines():
        p = Path(rel)
        if p.name == "__init__.py" or EXCLUDE_PARTS & set(p.parts):
            continue
        # 已删除但尚未提交（git ls-files 仍会列出）的条目不算活跃入口
        if not (ROOT / rel).exists():
            continue
        rows.append(rel)
    return sorted(rows)


def _summary(path: Path) -> str:
    try:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    except SyntaxError:
        doc = None
    if not doc:
        return "—"
    line = next((s.strip() for s in doc.splitlines() if s.strip()), "—")
    return line.replace("|", "\\|")[:120]


def render() -> str:
    rows = _tracked_scripts()
    groups: dict[str, list[str]] = {}
    for rel in rows:
        parts = Path(rel).parts
        groups.setdefault(parts[1] if len(parts) > 2 else "scripts", []).append(rel)
    total = len(rows)
    out = [
        "# scripts 入口索引",
        "",
        f"> 自动生成（{total} 个活跃入口）。"
        "改脚本 docstring 后运行 "
        "`python -m scripts.data_tools.gen_scripts_index` 更新；"
        "`--check` 校验是否过期。",
        "",
        "约定：`python -m scripts.<目录>.<文件名>` 运行；"
        "`scripts/archive/` 已于 2026-11-18 整体下线（活代码已上移，其余删除）；"
        "实验结论查 `reports/README.md` §一「结论速查」，产物落盘规则见其 §1.6。",
        "",
    ]
    for g in sorted(groups):
        out += [f"## {g}", "", "| 脚本 | 说明 |", "|---|---|"]
        for rel in groups[g]:
            rel_p = Path(rel).relative_to("scripts").as_posix()
            out.append(f"| `{rel_p}` | {_summary(ROOT / rel)} |")
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args()
    text = render()
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT} 已过期：请运行 python -m scripts.data_tools.gen_scripts_index",
                  file=sys.stderr)
            return 1
        print(f"{OUT} 已是最新")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"写入 {OUT}（{len(_tracked_scripts())} 个入口）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
