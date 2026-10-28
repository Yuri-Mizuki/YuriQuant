"""文档规范校验（docs 治理 P0，2026-10-28）。

背景
----
``reports/README.md`` §1.6 只管「产物放哪」，管不住「README 里该有哪几段」。
2026-10-28 全量实测：126 个实验目录仅 42 个有 README，其中四要素（结论 / 口径 /
复现 / 边界）全齐的只有 15 份；``reports/README.md`` §1.3 的文档索引已过期
（写「其余 12 篇」实际 18 篇、6 个 ``design/*.md`` 全部漏登、16 个目录从未登记）。
根因是**索引靠手写、无任何校验**——本脚本把四条规则变成可执行判据。

四条规则
--------
1. **目录登记**：``reports/<dir>/`` 必须在 ``reports/README.md`` 中出现。
2. **四要素**：``reports/<dir>/README.md`` 必须有结论 / 口径 / 复现 / 边界四类标题。
3. **标题带日期**：README 的 H1 必须含 ``YYYY-MM`` 或 ``YYYY-MM-DD``。
4. **无孤儿**：``reports/docs/`` 下的 ``.md`` 至少被一处引用（tracked 文件或
   ``reports/`` 内其它 md）。

存量豁免：按时间基线，不按白名单
--------------------------------
规则 2/3 只审**基线之后动过的 README**（``BASELINE`` 文件的 mtime 为准）。
判据：``README.md`` 的 mtime > ``BASELINE`` 的 mtime ⇒ 视为「新写或刚改」，
必须合规。存量文件永远豁免，不会有人为了过检查去硬加空标题；而一旦你因为
别的原因改动了某份 README，就必须顺手补齐——这正是「之后生成的文档统一
遵守规范」的可执行版本。

``GRANDFATHERED`` 只留给**明知旧但要求合规**的例外（当前为空）；规则 3 的
日期同理。换机器 / 重新 clone 后 mtime 全部刷新 ⇒ 退化为「全量豁免」，
不会误报（reports/ 是 gitignored 的本机证据库，mtime 在本机是稳定的）。

缺 README 的目录（规则 2/3 无对象）不算违规：「有没有 README」由
``MUST_HAVE_README_SKIP`` 白名单管，只对实验类目录提出建议（不计入失败）。

用法::

    python -m scripts.data_tools.docs_lint            # 报告（不改文件、总是 rc=0）
    python -m scripts.data_tools.docs_lint --check    # 守卫（有违规 rc=1）
    python -m scripts.data_tools.docs_lint --verbose  # 列全部条目而非只列违规
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORTS_INDEX = REPORTS / "README.md"
DOCS = REPORTS / "docs"

#: 四要素 → 标题关键词。判定只看标题（不看正文），避免正文提到「结论」二字就算有。
ELEMENTS = {
    "结论": ("结论", "摘要", "速览", "判定", "TL;DR", "一句话"),
    "口径": ("口径", "判据", "设定", "配置", "参数", "成本", "六件套"),
    "复现": ("复现", "重跑", "命令"),
    "边界": ("边界", "Caveat", "局限", "遗留", "风险", "限制"),
}

DATE_IN_TITLE = re.compile(r"20\d{2}-\d{2}")

#: 必须自建 README 的目录判定：非纯数据/纯产物目录的前缀白名单。
#: 生产臂目录（`alla_rolling_*`）只有 CSV/parquet，登记进 reports/README 即可。
MUST_HAVE_README_SKIP = (
    "alla_rolling", "alla_daily", "alla_arms", "alla_attribution", "model_portfolio",
    "models", "monitoring", "archive", "docs", "logs", "sync", "caliber",
    "capacity_table", "barra_cne6", "walk_forward", "two_periods", "freq_tune",
    "multiyear", "buffer_tune", "cpcv_h1", "gtja_repro", "investment_report",
    "ml_synthesis_h1", "risk_decomposition", "holder_dyn", "limit_pos",
    "e2e_p0_intraday", "panels_neu_backfill", "pbo_rolling_ortho", "horizon_mix",
    "prod_pipeline_gap", "alla_daily", "_htai_gp", "etf_rotation",
    "fundamental_blend", "stack_blend", "galaxy_l1", "portfolio_ppo_phase1",
    "deliverable_portfolio", "portfolio_phase2_smoke", "portfolio_phase2_mask_smoke",
    "kronos_smoke", "tabpfn3_smoke", "tabpfn_rolling_compare_smoke",
    "llm_mcts_phase0", "neutralize_vectorization", "structured_evidence_recalc",
    "signal_mktcap_drift", "altdata_macro", "altdata_cls_audit",
    "alt_paper_fingerprint", "clean_label_alt", "ic_headroom", "dpp_sigma_sweep",
    "size_neut_strength", "textmining", "daily_review", "daily_determinism",
)

#: 规则 1 豁免：确实不该进「实验结论速查」的目录（有其它登记渠道或纯中间产物）。
DIR_REGISTRY_SKIP = {
    "docs": "文档区，由 reports/docs/README.md 自管",
    "logs": "跨实验日志区，§1.5 已述",
    "archive": "归档区，§1.3 已述",
    "sync": "产物同步 zip，§1.2 已述",
}

#: 规则 2/3 的适用范围是「实验 README」。这些目录的 README 是**索引/导航页**，
#: 结构另有要求（目录地图、登记表），不适用四要素与日期标题。
INDEX_PAGE_READMES = {
    "docs": "文档区导航页，结构见 reports/docs/README.md §一–§四",
    "logs": "跨实验日志区",
}

#: 规则 2/3 的时间基线：mtime 严格大于它视为「新写或刚改」，必须合规。
BASELINE = DOCS / "TEMPLATE_实验README.md"

#: 明知旧但要求合规的例外（**只减不增，当前为空**）。
GRANDFATHERED: dict[str, list[str]] = {}

#: 规则 3 存量豁免：已删除（规则 2/3 统一按 mtime 基线判定，见模块头部）。

#: 规则 4 豁免：模板与被多处硬引用的单篇。
ORPHAN_SKIP = {
    "TEMPLATE_实验README.md": "模板本身，无上游引用是设计意图",
    "README.md": "目录索引",
}


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def _tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8")
    if out.returncode != 0:
        return []
    return [ROOT / f for f in out.stdout.split("\0")
            if f.endswith((".md", ".py", ".yaml", ".yml", ".toml"))]


def _corpus() -> dict[str, str]:
    """引用语料 = tracked 文本 + reports/ 下所有 md（后者 gitignored 但会互相引用）。"""
    files = _tracked_files()
    files += [p for p in REPORTS.rglob("*.md") if p.is_file()]
    corpus: dict[str, str] = {}
    for p in files:
        try:
            corpus[_rel(p)] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return corpus


def headings(text: str) -> list[str]:
    """提取全部 ``#``~``####`` 标题行（测试与模板自检共用，公开 API）。"""
    return [ln.strip() for ln in text.splitlines() if re.match(r"^#{1,4} ", ln)]


def h1(text: str) -> str:
    """取文档的一级标题（无则空串），供「标题是否带日期」判据使用。"""
    for ln in text.splitlines():
        if ln.startswith("# "):
            return ln[2:].strip()
    return ""


def check_dir_registry(index_text: str) -> list[str]:
    bad = []
    for d in sorted(p for p in REPORTS.iterdir() if p.is_dir()):
        if d.name in DIR_REGISTRY_SKIP:
            continue
        if d.name not in index_text:
            bad.append(f"{d.name}/ — 未在 reports/README.md §一/§1.2 登记")
    return bad


def check_readme(d: Path, baseline_mtime: float | None) -> tuple[list[str], list[str], bool]:
    """返回 (缺要素, 其它问题, 是否受基线约束)。"""
    readme = d / "README.md"
    if not readme.is_file():
        return [], [], False
    if baseline_mtime is not None and readme.stat().st_mtime <= baseline_mtime:
        return [], [], False  # 存量文档，不审
    text = readme.read_text(encoding="utf-8", errors="replace")
    heads = " ".join(headings(text))
    missing = [k for k, kws in ELEMENTS.items() if not any(kw in heads for kw in kws)]
    other = []
    if not DATE_IN_TITLE.search(h1(text)):
        other.append("H1 缺日期（YYYY-MM / YYYY-MM-DD）")
    return missing, other, True


def check_orphans(corpus: dict[str, str]) -> list[str]:
    bad = []
    if not DOCS.is_dir():
        return bad
    for p in sorted(DOCS.rglob("*.md")):
        if p.name in ORPHAN_SKIP:
            continue
        key = _rel(p)
        stem = p.stem
        hits = [k for k, v in corpus.items() if k != key and (p.name in v or stem in v)]
        if not hits:
            bad.append(f"{key} — 全项目零引用")
    return bad


def run(verbose: bool = False) -> int:
    if not REPORTS.is_dir():
        print("reports/ 不存在，跳过（本守卫只在本机有证据库时生效）")
        return 0
    index_text = REPORTS_INDEX.read_text(encoding="utf-8", errors="replace") \
        if REPORTS_INDEX.is_file() else ""
    corpus = _corpus()

    unregistered = check_dir_registry(index_text)
    baseline_mtime = BASELINE.stat().st_mtime if BASELINE.is_file() else None
    no_readme, missing_el, no_date = [], [], []
    audited = 0
    dirs = sorted(p for p in REPORTS.iterdir() if p.is_dir())
    for d in dirs:
        if not (d / "README.md").is_file():
            if not d.name.startswith(MUST_HAVE_README_SKIP):
                no_readme.append(f"{d.name}/ — 无 README")
            continue
        if d.name in INDEX_PAGE_READMES:
            if verbose:
                print(f"[跳过] {d.name}/ — {INDEX_PAGE_READMES[d.name]}")
            continue
        miss, other, in_scope = check_readme(d, baseline_mtime)
        if not in_scope:
            continue
        audited += 1
        allowed = set(GRANDFATHERED.get(d.name, []))
        real = [m for m in miss if m not in allowed]
        if real:
            missing_el.append(f"{d.name}/ — 缺 {'/'.join(real)}")
        elif verbose and miss:
            print(f"[豁免] {d.name}/ 缺 {'/'.join(miss)}")
        if other:
            no_date.append(f"{d.name}/ — {'; '.join(other)}")

    orphans = check_orphans(corpus)

    total = len(dirs)
    with_readme = sum(1 for d in dirs if (d / "README.md").is_file())
    print(f"reports/ 目录 {total} 个，其中有 README {with_readme} 个")
    print(f"规则 2/3 受基线约束（新写或刚改）的 README：{audited} 个")
    print(f"语料 {len(corpus)} 个文件")
    if baseline_mtime is None:
        print("⚠️ 未找到基线文件 docs/TEMPLATE_实验README.md ⇒ 规则 2/3 退化为全量豁免")

    groups = [
        ("规则 1 目录未登记", unregistered),
        ("规则 2 四要素不全", missing_el),
        ("规则 3 标题缺日期", no_date),
        ("规则 4 孤儿文档", orphans),
        ("建议补 README 的实验目录", no_readme),
    ]
    failed = 0
    for title, items in groups:
        # 「建议补 README」是提示项，不计入失败
        if title.startswith("建议") and not verbose:
            if items:
                print(f"\n{title}：{len(items)} 个（--verbose 查看）")
            continue
        print(f"\n{title}：{len(items)} 项")
        for x in items[:40]:
            print(f"  - {x}")
        if len(items) > 40:
            print(f"  … 另 {len(items) - 40} 项（--verbose 查看全部）")
        if items and not title.startswith("建议"):
            failed += len(items)

    print(f"\n合计违规 {failed} 项（豁免表 GRANDFATHERED 里 "
          f"{len(GRANDFATHERED)} 个目录不计入）")
    return failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="有违规返回 1（守卫模式）")
    ap.add_argument("--verbose", action="store_true", help="列全部条目与豁免明细")
    args = ap.parse_args(argv)
    failed = run(verbose=args.verbose)
    if args.check and failed:
        print("\n修复：补标题段 / 登记目录 / 挂引用指针；"
              "存量豁免只能减不能加（见 docs_lint.GRANDFATHERED）。",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())