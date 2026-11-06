"""守卫：建实验目录工具的模板回退（2026-11-06）。

背景
----
``scripts/data_tools/new_experiment.py`` 的整个存在理由是把「建目录」和「写判定行」
绑成一个原子操作（见该脚本头部：只建目录不登记是 ``reports/`` 判定覆盖率长期只有
47% 的根因）。它依赖一份 README 模板，而模板原先只放在
``reports/docs/TEMPLATE_实验README.md``——``reports/*`` 整棵树被 gitignore
（只放行 ``reports/README.md``）⇒ **干净 clone 上模板不存在**，工具第一句
``read_text`` 就 FileNotFoundError。同一个缺失让 ``docs_lint`` 的规则 2/3
（四要素 / 标题带日期）**静默退化为全量豁免**：只打一行 ⚠️，rc 仍是 0。

本测试守两条：
1. 内置模板随 git 跟踪、渲染后四要素与日期都在（规则 2/3 的直接输入）；
2. ``docs_lint`` 的 mtime 基线解析到一个**真实存在的文件**（否则豁免是假绿）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_tools import docs_lint as DL  # noqa: E402
from scripts.data_tools import new_experiment as NE  # noqa: E402

ELEMENTS = ("结论", "口径", "复现", "边界")


def test_packaged_template_is_tracked_and_exists():
    """模板必须在 git 跟踪范围内（reports/ 之外），否则 clone 即缺。"""
    p = NE.TEMPLATE_PACKAGED
    assert p.is_file(), f"内置模板缺失：{p}"
    assert "reports" not in p.relative_to(ROOT).parts, \
        "内置模板不能放 reports/ 下（被 gitignore）"


def test_render_fills_placeholders_and_keeps_four_elements():
    text = NE.render_readme("my_probe", "判负：无增量", 
                            "scripts/evaluation/my_probe.py", "2026-11-06")
    for leftover in ("<实验名>", "<一句话判定>", "YYYY-MM-DD",
                     "scripts/<域>/<脚本名>.py", "scripts.<域>.<脚本名>"):
        assert leftover not in text, f"占位符未替换：{leftover}"
    assert "my_probe" in text and "判负：无增量" in text
    assert "scripts/evaluation/my_probe.py" in text
    assert "<!--" not in text, "HTML 注释应被剥掉"
    for el in ELEMENTS:
        assert el in text, f"四要素缺「{el}」标题"
    assert "2026-11-06" in text.splitlines()[0], "H1 必须带完整日期（规则 3）"


def test_local_template_overrides_packaged(tmp_path):
    """本地副本存在时优先用本地（允许个人改），不破坏既有习惯。"""
    local = tmp_path / "t.md"
    local.write_text("# <实验名>（YYYY-MM-DD）\n\n## 一、结论\n本地版\n",
                     encoding="utf-8")
    text = NE.render_readme("p", "d", "s.py", "2026-01-01", template=local)
    assert "本地版" in text and "2026-01-01" in text


def test_docs_lint_baseline_is_a_real_file():
    """基线文件必须真实存在；缺失会让规则 2/3 假绿（只打 ⚠️ 不报错）。"""
    assert DL.BASELINE.is_file(), f"基线缺失：{DL.BASELINE}"
    if DL.BASELINE_LOCAL.is_file():
        assert DL.BASELINE == DL.BASELINE_LOCAL
    else:
        assert DL.BASELINE == DL.BASELINE_PACKAGED
