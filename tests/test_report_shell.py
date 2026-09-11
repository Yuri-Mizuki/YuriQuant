"""报告渲染外壳守卫 + 主题 CSS 等价性（2026-09-11 报告渲染收口）。

背景：`research/html_report.py` 的 ``page()`` 是全仓唯一的 HTML 外壳实现
（``BASE_CSS`` / ``page`` / ``render_table`` / ``base_js`` / ``SORT_JS`` /
``svg_sparkline`` / ``embed_image_b64``）。2026-09-11 前仍有 3 个报告脚本
（``jq_style_report`` / ``rolling_grid_report`` / ``alla_excess_attribution``）
自己拼 ``<!DOCTYPE html>…</html>``，各写一份 head/body/style 外壳。

本测试钉住两件事：
1. **外壳唯一性**：除 `research/html_report.py` 外，任何模块（含 scripts/、
   monitoring/、research/）不得自拼 ``<!DOCTYPE`` / ``<html`` 外壳。
2. **主题 CSS 无损**：3 个被收编的脚本，其主题 CSS 必须**逐字节**出现在
   ``page(css=...)`` 的输出里——收编只换外壳，不许改主题。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: 允许自拼 HTML 外壳的唯一模块（外壳真源）
SHELL_SOURCE = ROOT / "research" / "html_report.py"
_SKIP_DIRS = {".venv", "venv", ".git", "oneoff", "archive", "node_modules",
              "tests", "__pycache__"}
_SHELL_RE = re.compile(r"<!DOCTYPE|<html\b", re.I)


def _code_without_comments(path: Path) -> str:
    """去掉整行注释与行末 ``#`` 注释后的代码（避免 docstring/注释里的字样误伤）。"""
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        stripped = ln.lstrip()
        if stripped.startswith("#"):
            continue
        # 粗略去行末注释：只在 # 前有代码时截断，且不处理字符串内的 #（足够本用）
        if "#" in ln:
            head = ln.split("#", 1)[0]
            if head.strip():
                ln = head
        out.append(ln)
    return "\n".join(out)


def test_only_html_report_owns_the_document_shell():
    """HTML 外壳只能有一份实现：research/html_report.py 的 ``page()``。"""
    offenders = []
    for p in sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT)
        if set(rel.parts) & _SKIP_DIRS:
            continue
        if p == SHELL_SOURCE:
            continue
        if _SHELL_RE.search(_code_without_comments(p)):
            offenders.append(str(rel))
    assert not offenders, (
        "这些模块自拼了 HTML 外壳（应用 research.html_report.page）:\n"
        + "\n".join(offenders))


def test_shell_source_still_exports_the_shared_primitives():
    """外壳模块必须继续导出公共原语（防被"重构"掉）。"""
    from research import html_report as hr

    for name in ("page", "BASE_CSS", "render_table", "base_js", "SORT_JS",
                 "svg_sparkline", "svg_sparkline_monthly", "embed_image_b64",
                 "render_sortable_table", "generate_html_report"):
        assert hasattr(hr, name), f"research.html_report 缺少公共原语 {name}"


def test_page_extra_css_composes_after_base():
    """``extra_css`` 是「继承 + 增量」：BASE_CSS 在前、增量在后（后者胜出）。"""
    from research.html_report import BASE_CSS, page

    out = page("t", body="<p>x</p>", extra_css=".z{color:#123456}")
    assert BASE_CSS in out, "extra_css 模式下 BASE_CSS 必须保留（继承语义）"
    assert out.index(BASE_CSS) < out.index(".z{color:#123456}"), (
        "增量 CSS 必须排在 BASE_CSS 之后，否则无法覆盖基础样式")


def test_page_css_replaces_base():
    """``css=`` 是整体替换（自带完整主题的报告语义，向后兼容）。"""
    from research.html_report import BASE_CSS, page

    out = page("t", body="", css=".only{color:#000}")
    assert BASE_CSS not in out
    assert ".only{color:#000}" in out


def test_page_emits_single_doctype_and_body():
    from research.html_report import page

    out = page("标题", meta="m", body="<p>b</p>")
    assert out.count("<!DOCTYPE html>") == 1
    assert out.count("<title>标题</title>") == 1
    assert out.rstrip().endswith("</body></html>")


@pytest.mark.parametrize("modname,title", [
    ("scripts.jq_style_report", "x · 收益曲线（vs y）"),
    ("scripts.rolling_grid_report", "全A滚动训练实验报告（2018~now）"),
    ("scripts.alla_excess_attribution", "全A主策略超额归因"),
])
def test_converted_reports_keep_theme_css_verbatim(modname, title):
    """收编的 3 个脚本：主题 CSS 必须逐字节进入 page() 输出（收编只换外壳）。"""
    import importlib

    from research.html_report import page

    mod = importlib.import_module(modname)
    css = mod._CSS
    assert "<style>" not in css and "</style>" not in css, (
        "主题 CSS 不应再自带 <style> 标签（外壳由 page() 提供）")
    assert len(css.strip()) > 100, "主题 CSS 疑似为空"

    cdn = getattr(mod, "_CHART_CDN", "")
    out = page(title, header="", body="<p>body</p>", css=css, head_extra=cdn)
    assert css in out, f"{modname} 的主题 CSS 未原样出现在输出中"
    if cdn:
        assert cdn in out, f"{modname} 的 Chart.js CDN 未注入"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
