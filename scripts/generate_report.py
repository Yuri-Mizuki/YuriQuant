"""
一键端到端研究报告生成
======================
收集 reports/ 下各层产物，组装为单个自包含 HTML 研究报告。

用法::

    python scripts/generate_report.py                              # 默认 hs300_2025
    python scripts/generate_report.py --dataset hs300_2025
    python scripts/generate_report.py --out reports/my_report.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cli_common import setup_logging  # noqa: E402


from research.report_pipeline import generate_research_report  # noqa: E402

log = setup_logging("generate_report")

def main() -> None:
    ap = argparse.ArgumentParser(description="一键端到端研究报告生成")
    ap.add_argument("--dataset", default="hs300_2025", help="数据集名")
    ap.add_argument("--out", default=None, help="输出 HTML 路径（默认带时间戳）")
    ap.add_argument("--report-dir", default="reports", help="reports 目录")
    ap.add_argument("--title", default=None, help="报告标题")
    args = ap.parse_args()

    out = generate_research_report(
        dataset=args.dataset,
        out=args.out,
        report_dir=args.report_dir,
        title=args.title,
    )
    print(f"\n报告已生成: {out.resolve()}")
    print(f"浏览器打开: file:///{str(out.resolve()).replace(chr(92), '/')}")

if __name__ == "__main__":
    main()