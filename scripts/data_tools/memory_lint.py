"""记忆层治理校验（memory_lint，2026-11-03）。

背景
----
2026-11-03 查伪：「优化层自评 20%」在全部项目 md 里零命中，唯一载体是
``USER.md`` 与 ``.workbuddy/memory/MEMORY.md`` 两份**记忆**。它由某次会话自造，
被反复引用成事实长达数月，直到有人当面质问才暴露。

根因不是「数字没更新」，而是**记忆层不在任何校验范围内**：``docs_lint`` 守护
``reports/`` 的目录登记与 README 四要素，唯独不碰记忆——而记忆每轮自动注入、
影响面最大、验证最少。同一事实在「研究笔记（带"自评"二字）→ 对话（转成断言）
→ 记忆（变成长期前提）」三份拷贝里逐级丢失限定词，且无一处可核验。

两条规则
--------
1. **失效路径**：记忆里反引号包住的项目路径必须存在（文件 / 目录 / glob）。
   目录归位、脚本改名后，记忆里的指针会静默失效，读记忆的人不会知道。
2. **禁用表述**（只审长期记忆）：``成熟度`` / ``完成度`` / ``自评 ≈N%`` 这类
   **无可核验分母**的进度百分比。记忆是写时快照，存百分比等于存一个必然过期
   的断言；要存就存指针（``见 reports/README.md §一``），不要存值。

日报的豁免
----------
``YYYY-MM-DD.md`` 按 ``AGENTS.md`` §1「报告产物是证据」处理：只做规则 1 且
**仅提示、不计入失败**；规则 2 完全不适用——那是当时的真实判断，事后回头看
措辞不好也不该改（改了就成篡改证据）。

归档区
------
``.workbuddy/memory/archive/`` 放已过期的日报（2026-11-10 起，超过约 1 个月的移入）。
本模块只 glob 顶层、**不递归**，故归档区天然不参与校验——那里的路径是历史事实，
不该因为脚本后来改名就报错。要查归档区自己另跑，别把它塞回顶层。

用法::

    python -m scripts.data_tools.memory_lint            # 报告（总是 rc=0）
    python -m scripts.data_tools.memory_lint --check    # 守卫（有违规 rc=1）
    python -m scripts.data_tools.memory_lint --verbose  # 列全部条目
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEM_DIR = ROOT / ".workbuddy" / "memory"
USER_HOME = Path.home() / ".workbuddy"

BACKTICK = re.compile(r"`([^`\n]{3,140})`")

#: 规则 2：无可核验分母的进度表述。只对长期记忆生效。
#: 必须「关键词 + 近处百分比」同现，否则「README 含完成度一节」这类描述会误报。
BANNED = (
    (re.compile(r"成熟度[^。\n]{0,12}?\d+\s*%"), "「成熟度 N%」"),
    (re.compile(r"完成度[^。\n]{0,12}?\d+\s*%"), "「完成度 N%」"),
    (re.compile(r"自评[^。\n]{0,10}?\d+\s*%"), "「自评 N%」"),
)

#: 该行在讲「禁止」本身（元讨论），不是违规断言。
NEGATION = ("不记", "禁用", "勿再", "禁止", "别再", "不要", "查伪", "无出处", "零命中")

#: 该行在描述「反例」（如「**没有** `scripts/data/`」），路径本就不该存在。
PATH_NEGATION = ("没有", "不存在", "已删", "已移", "禁用", "禁止", "移除", "别放")

#: 出现任一字样就说明不是路径（命令、URL、占位符、通配、代码片段、省略号）。
NOT_PATH = ("http", " ", "<", ">", "{", "}", "(", ")", "=", "|", "&", "*", "?", "$", ",", "…")

#: 项目内可校验路径的顶层前缀 —— 防止把目录树片段（`oneoff/`、`ic_mean/ic_ir`）当路径。
TOP_LEVEL = (
    "scripts/", "reports/", "factor/", "model/", "optimize/", "data/", "backtest/",
    "tests/", "config/", "monitoring/", "research/", "stats/", "strategy/",
    "docs/", "logs/", "_tmp/", ".workbuddy/", "memory/", "topics/",
)

#: 剥掉 Python 符号后缀（``::func``）与行号（``:118``）再校验。
STRIP_SUFFIX = re.compile(r"(::.+|:\d+(?:/\d+)*)$")


def _label(p: Path) -> str:
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


#: 盘符前缀（盘符 + 冒号 + 斜杠）。写成正则而非字符串字面量，是因为本文件同样受
#: ``tests/test_no_hardcoded_drive_paths`` 审计 —— 那里扫的就是「引号紧跟盘符」的
#: 形态，直接写死一个盘符前缀会让检测器把自己判成硬编码路径。
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:/")


def looks_like_path(tok: str) -> bool:
    """反引号内容是否是一个可校验的项目路径（宁缺勿滥，误报会让人放弃守卫）。"""
    if any(s in tok for s in NOT_PATH):
        return False
    if tok.startswith(("http", "~/")) or _DRIVE_PREFIX.match(tok):
        return True
    if "/" not in tok or tok.startswith(("-", "#")):
        return False
    t = STRIP_SUFFIX.sub("", tok)
    return t.startswith(TOP_LEVEL)


def resolve(tok: str, bases: tuple[Path, ...]) -> Path:
    """把记忆里的路径解析成绝对路径；多 base 依次试（仓库根 / 用户配置目录 / 文件所在目录）。"""
    t = STRIP_SUFFIX.sub("", tok).rstrip("/")
    if re.match(r"^[A-Za-z]:/", t):
        return Path(t)
    if t.startswith("~/"):
        return Path.home() / t[2:]
    for b in bases:
        p = b / t
        if p.exists():
            return p
    return bases[0] / t


def collect_files() -> tuple[list[Path], list[Path]]:
    """返回 (长期记忆, 日报)。长期记忆 = 用户级 USER.md/MEMORY.md + 项目 MEMORY*.md。"""
    long_term: list[Path] = []
    for name in ("USER.md", "MEMORY.md"):
        p = USER_HOME / name
        if p.is_file():
            long_term.append(p)
    daily: list[Path] = []
    if MEM_DIR.is_dir():
        for p in sorted(MEM_DIR.glob("MEMORY*.md")):
            if ".bak" not in p.name:
                long_term.append(p)
        daily = sorted(MEM_DIR.glob("20[0-9][0-9]-[0-9][0-9]-[0-9][0-9].md"))
    return long_term, daily


def check_paths(p: Path, text: str) -> list[str]:
    bases = tuple(dict.fromkeys([ROOT, USER_HOME, p.parent]))
    bad = []
    in_fence = False
    for i, line in enumerate(text.splitlines(), 1):
        head = line.lstrip()
        if head.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        # 围栏内是「引用的原文 / 代码示例」，不是当前有效指针，不校验
        if in_fence or any(n in line for n in PATH_NEGATION):
            continue
        for tok in BACKTICK.findall(line):
            if looks_like_path(tok) and not resolve(tok, bases).exists():
                bad.append(f"{_label(p)}:{i}  `{tok}` — 不存在")
    return bad


def check_banned(p: Path, text: str) -> list[str]:
    bad = []
    for i, line in enumerate(text.splitlines(), 1):
        if any(n in line for n in NEGATION):
            continue
        for pat, name in BANNED:
            if pat.search(line):
                bad.append(f"{_label(p)}:{i}  {name} — {line.strip()[:64]}")
                break
    return bad


def run(verbose: bool = False) -> int:
    long_term, daily = collect_files()
    if not long_term and not daily:
        print("未找到记忆文件（USER.md / MEMORY*.md / 日报），跳过")
        return 0

    dead, banned, stale = [], [], []
    for p in long_term:
        text = p.read_text(encoding="utf-8", errors="replace")
        dead += check_paths(p, text)
        banned += check_banned(p, text)
    for p in daily:
        stale += check_paths(p, p.read_text(encoding="utf-8", errors="replace"))

    print(f"长期记忆 {len(long_term)} 个 · 日报 {len(daily)} 个")
    groups = [
        ("规则 1 失效路径（长期记忆）", dead, True),
        ("规则 2 禁用表述（长期记忆）", banned, True),
        ("规则 1 失效路径（日报 · 仅提示）", stale, False),
    ]
    failed = 0
    for title, items, counted in groups:
        if items:
            print(f"\n{title}：{len(items)} 项")
            for x in (items if verbose else items[:30]):
                print(f"  - {x}")
            if not verbose and len(items) > 30:
                print(f"  … 另 {len(items) - 30} 项（--verbose 查看全部）")
        elif verbose:
            print(f"\n{title}：0 项")
        if counted:
            failed += len(items)

    print(f"\n合计违规 {failed} 项（日报 {len(stale)} 项仅提示，不计入）")
    if failed:
        print("修复：删数字 / 换指针（见 `reports/README.md §一`）；"
              "路径已归位的改指针，不要补新数字。")
    return failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="有违规返回 1（守卫模式）")
    ap.add_argument("--verbose", action="store_true", help="列全部条目")
    args = ap.parse_args(argv)
    failed = run(verbose=args.verbose)
    return 1 if (args.check and failed) else 0


if __name__ == "__main__":
    sys.exit(main())
