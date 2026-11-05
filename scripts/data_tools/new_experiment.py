"""新建实验目录：一条命令完成「建目录 + 复制模板 + 登记判定」。

背景（2026-11-03）
------------------
实测 ``reports/`` 114 个目录的判定覆盖率只有 47%，而 ``docs_lint --check`` 全绿
——旧规则 1 只查「目录名在 55KB 全文出现过」，通配写法一行就能盖住 5 个目录。

根因不是「忘了清理」，而是**「建目录」和「写判定」是两个分离的动作**：前者一秒，
后者十分钟，于是永远只做了前者。攒够几十个之后，清理就得先花半天重新判断
「这目录是什么、能不能删」——清理的成本随目录数线性增长，而建目录的成本恒为一秒。

本脚本把两件事绑成一个原子操作：没有判定行就建不出目录。配合 ``docs_lint``
规则 5（无判定即 rc=1），「只建目录不登记」在流程上不可能发生。

用法::

    python -m scripts.data_tools.new_experiment my_probe --desc "我的假设一句话"
    python -m scripts.data_tools.new_experiment my_probe --script scripts/evaluation/my_probe.py

建完立刻可跑 ``python -m scripts.data_tools.docs_lint --check`` 确认是绿的。
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
INDEX = REPORTS / "README.md"
TEMPLATE = REPORTS / "docs" / "TEMPLATE_实验README.md"
SECTION = "### 结论速查"

VALID_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ROW = re.compile(r"^\|\s*`?([A-Za-z0-9_.\-/]+)`?\s*\|")


def find_table_bounds(lines: list[str]) -> tuple[int, int] | None:
    """返回 §结论速查 表格的数据行区间 [start, end)（不含表头与分隔行）。"""
    start = None
    for i, line in enumerate(lines):
        if line.startswith(SECTION):
            start = i
            continue
        if start is not None and line.startswith("#"):
            return _data_rows(lines, start, i)
    return _data_rows(lines, start, len(lines)) if start is not None else None


def _data_rows(lines: list[str], begin: int, end: int) -> tuple[int, int] | None:
    first = last = None
    for i in range(begin, end):
        if lines[i].startswith("|"):
            first = first if first is not None else i
            last = i
    if first is None or last is None:
        return None
    return first + 2, last + 1  # 跳过表头行与 |---| 分隔行


def insert_row(lines: list[str], name: str, desc: str, verdict: str, date: str) -> int:
    """按目录名字母序插入一行，返回插入位置。"""
    bounds = find_table_bounds(lines)
    if bounds is None:
        raise SystemExit(f"未在 {INDEX.name} 找到 `{SECTION}` 表格，中止（未建目录）")
    start, end = bounds
    row = f"| `{name}` | {desc} | {verdict} | {date} |"
    for i in range(start, end):
        m = ROW.match(lines[i])
        if m and m.group(1).rstrip("/").lower() > name.lower():
            lines.insert(i, row)
            return i
    lines.insert(end, row)
    return end


def render_readme(name: str, desc: str, script: str, date: str) -> str:
    text = TEMPLATE.read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->\s*", "", text, flags=re.S)
    text = text.replace("<实验名>", name).replace("<一句话判定>", desc)
    text = text.replace("YYYY-MM-DD", date)
    mod = script.removesuffix(".py").replace("/", ".").replace("\\", ".")
    text = text.replace("scripts/<域>/<脚本名>.py", script)
    text = text.replace("scripts.<域>.<脚本名>", mod)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", help="目录名（小写字母/数字/下划线/连字符）")
    ap.add_argument("--desc", default="（待收口）", help="结论速查表的「结论一句话」")
    ap.add_argument("--script", default="scripts/<域>/<脚本名>.py", help="产出脚本路径")
    ap.add_argument("--verdict", default="进行中", help="初始判定（默认 进行中）")
    ap.add_argument("--date", default=None, help="默认今天")
    args = ap.parse_args(argv)

    name = args.name.strip()
    if not VALID_NAME.match(name):
        print(f"目录名不合法：{name}（只允许小写字母/数字/下划线/连字符）", file=sys.stderr)
        return 2

    target = REPORTS / name
    if target.exists():
        print(f"目录已存在：{target.relative_to(ROOT)}（未做任何改动）", file=sys.stderr)
        return 2

    lines = INDEX.read_text(encoding="utf-8").splitlines()
    if any(re.search(rf"`{re.escape(name)}/?`", ln) for ln in lines):
        print(f"`{name}` 已在 {INDEX.name} 出现（可能已登记），请先核对，未做改动",
              file=sys.stderr)
        return 2

    date = args.date or dt.date.today().isoformat()
    pos = insert_row(lines, name, args.desc, args.verdict, date)  # 先改内存，失败即中止

    target.mkdir(parents=True)
    (target / "README.md").write_text(
        render_readme(name, args.desc, args.script, date), encoding="utf-8"
    )
    INDEX.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"已建 {target.relative_to(ROOT)}/README.md（四要素骨架，结论待填）")
    print(f"已在 {INDEX.name} §结论速查 第 {pos + 1} 行登记：{name} · {args.verdict} · {date}")
    print("\n下一步：")
    print("  1. 跑实验，产物只落本目录")
    print("  2. 收口时填 §〇 结论，并把判定从 `进行中` 改成 采纳/判负/归档/工具保留/事故记录")
    print("  3. python -m scripts.data_tools.docs_lint --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
