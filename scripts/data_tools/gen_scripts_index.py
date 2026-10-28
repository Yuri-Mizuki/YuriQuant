"""生成 ``scripts/README.md`` 入口索引（2026-10-19 审计 ③）。

背景：全仓 190 个脚本，31 个「三类引用全零」——不是都能删（数据回补/研报
数据准备仍有人工流程），但至少要让每个入口可被发现。本生成器按目录列出
每个 tracked 脚本的首行 docstring；archive/oneoff 不计入活跃入口。

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
EXCLUDE_PARTS = {"archive", "oneoff", "__pycache__"}

ARCHIVED = [
    ("scripts/common/e2e_common.py", "HS300 时代 e2e 编排（数据/中性化/特征漏斗）"),
    ("scripts/evaluation/ai97_pbo_report.py", "AI97 PBO 报告（一次性）"),
    ("scripts/evaluation/pv_marginal_quality.py", "PV 边际质量探针（一次性）"),
    ("scripts/evaluation/tabfm_head_portfolio.py", "TabPFN/TabICL 头部组合验证"),
    ("scripts/evaluation/tabfm_head_study.py", "TabPFN/TabICL 头部诊断"),
    ("scripts/factors/phase2_ppo_diag.py", "Phase2 PPO 诊断"),
    ("scripts/factors/run_gflownet_phase0.py", "GFlowNet phase0 入口（已被 phase1 取代）"),
    ("scripts/reporting/factor_correlation.py", "因子相关性报告"),
    ("scripts/reporting/risk_decomposition_report.py", "风险分解报告"),
    ("scripts/evaluation/cheap_lever_probe.py", "廉价杠杆探针"),
    ("scripts/evaluation/impact_cost_probe.py", "冲击成本探针"),
    ("scripts/evaluation/multihorizon_fusion_probe.py", "多 horizon 融合探针"),
    ("scripts/builders/backfill_delisted_backward.py", "退市股后复权回补"),
    ("scripts/builders/backfill_delisted_equity.py", "退市股股本结构回补"),
    ("scripts/builders/backfill_delisted_kline.py", "退市股 K 线回补"),
    ("scripts/builders/build_alla_e4_interaction_factors.py", "E4 交互因子构建"),
    ("scripts/builders/refetch_status_all_a.py", "全A状态重取"),
    ("scripts/factors/fetch_textmining.py", "文本挖掘数据拉取"),
    ("scripts/factors/gflownet_library_ingest.py", "GFlowNet 因子入库"),
    ("scripts/data_tools/cleanup_data.py", "数据清理工具"),
    ("scripts/data_tools/split_zz1000.py", "ZZ1000 拆分工具"),
    ("scripts/textmining/build_enhanced_top25.py", "增强 Top25 构建"),
    ("scripts/textmining/build_rawtone_factor.py", "原始情绪因子构建"),
    ("scripts/textmining/encode_fadt_seg.py", "FADT 分段编码"),
    ("scripts/textmining/evaluate_fadt_ablation.py", "FADT 消融评估"),
    ("scripts/textmining/fetch_all_a_backfill.py", "全A文本数据回补"),
    ("scripts/textmining/register_text_factors.py", "文本因子入库"),
]


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
        f"> 自动生成（{total} 个活跃入口；archive/oneoff 不计）。"
        "改脚本 docstring 后运行 "
        "`python -m scripts.data_tools.gen_scripts_index` 更新；"
        "`--check` 校验是否过期。",
        "",
        "约定：`python -m scripts.<目录>.<文件名>` 运行；"
        "退役入口在 `scripts/archive/`（仍可运行，但不再维护）；"
        "实验结论查 `reports/README.md` §一「结论速查」，产物落盘规则见其 §1.6。",
        "",
    ]
    for g in sorted(groups):
        out += [f"## {g}", "", "| 脚本 | 说明 |", "|---|---|"]
        for rel in groups[g]:
            rel_p = Path(rel).relative_to("scripts").as_posix()
            out.append(f"| `{rel_p}` | {_summary(ROOT / rel)} |")
        out.append("")
    out += ["## 2026-10-19 归档（三类引用全零，无文档/产物引用）", "",
            "| 原路径 | 归档到 | 说明 |", "|---|---|---|"]
    for old, desc in ARCHIVED:
        name = Path(old).name
        out.append(f"| `{old}` | `scripts/archive/{name}` | {desc} |")
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
