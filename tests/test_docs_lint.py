"""守卫：文档规范（docs 治理 P0，2026-10-28）。

背景
----
2026-10-28 全量审计：126 个实验目录仅 42 个有 README，四要素（结论 / 口径 /
复现 / 边界）全齐的只有 15 份；``reports/README.md`` §1.3 的索引已过期
（写「其余 12 篇」实际 18 篇、6 个 ``design/*.md`` 漏登、22 个目录从未登记、
10 篇文档全项目零引用）。根因是**索引靠手写、无校验**。

本测试只是把 ``scripts.data_tools.docs_lint`` 包一层，逻辑全在那支脚本里
（它可独立跑，用于给人看详细报告）。这里守四件事：

1. 目录已登记进 ``reports/README.md``；
2. 新写/刚改的实验 README 四要素齐备（存量按 mtime 基线豁免，见脚本文档）；
3. 新写/刚改的 README 标题带日期；
4. ``reports/docs/`` 下无孤儿文档。

设计取舍：**不校验存量旧文档**。强制补全 25 份旧 README 会让人为了过检查去
硬加空标题，反而更糟；按「改动即须合规」推进，存量随自然编辑逐步收敛。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINT = ROOT / "scripts" / "data_tools" / "docs_lint.py"


def test_docs_lint() -> None:
    if not (ROOT / "reports").is_dir():
        pytest.skip("无 reports/ 证据库（如 CI 精简环境），跳过")
    if not LINT.is_file():
        pytest.fail(f"缺少文档校验器 {LINT.relative_to(ROOT)}")

    proc = subprocess.run(
        [sys.executable, "-m", "scripts.data_tools.docs_lint", "--check"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert proc.returncode == 0, (
        "文档规范校验未通过：\n"
        + proc.stdout + proc.stderr
        + "\n\n修复方式："
          "\n  - 目录未登记 → 在 reports/README.md §一/§1.2 加一行；"
          "\n  - 四要素不全 → 照 reports/docs/TEMPLATE_实验README.md 补标题段；"
          "\n  - 标题缺日期 → H1 加 (YYYY-MM-DD)；"
          "\n  - 孤儿文档 → 在 reports/docs/README.md 加一行指向它。"
          "\n本地查看完整报告：python -m scripts.data_tools.docs_lint --verbose"
    )


def test_template_is_self_consistent() -> None:
    """模板自身必须满足它要求别人的四要素（否则复制出来就是错的）。"""
    template = ROOT / "reports" / "docs" / "TEMPLATE_实验README.md"
    if not template.is_file():
        pytest.skip("模板未落地")
    sys.path.insert(0, str(ROOT))
    from scripts.data_tools.docs_lint import DATE_IN_TITLE, ELEMENTS, h1, headings

    text = template.read_text(encoding="utf-8")
    heads = " ".join(headings(text))
    missing = [k for k, kws in ELEMENTS.items() if not any(kw in heads for kw in kws)]
    assert not missing, f"模板缺要素 {missing}——改模板后请同步 docs_lint.ELEMENTS"
    title = h1(text)
    # 模板里的日期是占位符，不能要求真实日期；只要求「有日期位」。
    assert DATE_IN_TITLE.search(title) or "YYYY-MM-DD" in title, \
        f"模板 H1 既无日期也无 YYYY-MM-DD 占位：{title!r}"