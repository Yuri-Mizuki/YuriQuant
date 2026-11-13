"""因子看板渲染守卫：公式只在详情页，不占表格列（2026-11-13 二改）。

背景：2026-11-13 上午公式回填到 888/888 后，表格加了「公式」列显示前 50 字符。
但库内公式最长 317 字符，前 50 字符常常只到 ``let`` 别名（如 ``let inner = if_else(ret ...``），
读不出任何信息，还挤掉指标列宽 ⇒ 当天二改为**只在详情页展示**。

本测试钉住三件事：
1. **表格无公式列**：表头 ``<th>`` 与行模板 ``<td>`` 数量一致，且都不含公式单元格；
2. **payload 不截断**：详情页要能看全（原 400 字符截断是给表格列留的余量，已随列撤掉）；
3. **详情页分条渲染保留 ``let`` 别名**：曾经的 bug 是剥 ``head`` 时把首段右半一起丢掉，
   148 个带别名的公式因此只剩主表达式（``inner`` 无从解释）。

用 ``render()`` 而非生成整份 HTML：它是纯函数，不碰磁盘、不依赖巨型 payload。
"""
from __future__ import annotations

import re

import pandas as pd

from scripts.reporting.factor_dashboard import render

#: 一个覆盖三种形态的迷你 payload：带 let 别名 / 带比较符（需转义）/ 无公式
FORMULA_LONG = (
    "WorldQuant Alpha101 (Kakushadze 2016) #001 = "
    "let inner = if_else(returns < 0, stddev(returns, 20), close) | "
    "rank(argmax(power(inner, 2), 5)) - 0.5"
)


def _payload(formula: str = FORMULA_LONG) -> dict:
    return {
        "rows": [{"i": 0, "name": "alpha101_001", "set": "alpha101", "fam": "价量",
                  "grp": "量价", "cov": 0.99, "has_ic": False, "has_eq": False,
                  "sig": None, "w": {}, "nw": None}],
        "months": ["2026-01", "2026-02"],
        "groups": {"量价": ["alpha101"]},
        "monthly": {}, "equity": {}, "bench": {},
        "formula": {"alpha101_001": formula},
        "meta": {"alpha101_001": {"freq": "日频", "mat": "experimental",
                                  "sub": "", "af": "", "al": ""}},
        "hasLS": False, "hasW": False, "hasEx": False,
        "wins": ["ALL"], "winLabel": {"ALL": "全部"}, "winStart": {"ALL": ""},
    }


def _render(**kw) -> str:
    reg = pd.DataFrame([{"name": "alpha101_001", "set": "alpha101", "family": "价量"}])
    return render("all_a_2018_2026", 1, reg, _payload(**kw), ["2026-01", "2026-02"])


def _table_headers(html: str) -> list[str]:
    """只取主表的 thead，避免误抓详情抽屉里的 HTML。"""
    head = html.split('<table id="ftable">', 1)[1].split("</thead>", 1)[0]
    return re.findall(r"<th[^>]*>(.*?)</th>", head, re.S)


def _row_template(html: str) -> str:
    """renderRows 里拼 ``tr.innerHTML`` 的那段模板字符串。"""
    m = re.search(r"tr\.innerHTML =(.*?)tr\.onclick", html, re.S)
    assert m, "renderRows 的 tr.innerHTML 模板不见了（结构被改，需同步本测试）"
    return m.group(1)


def _render_ls(**kw) -> str:
    """带分层回测指标（hasLS=True, hasEx=True）的变体，用于列数对齐校验。"""
    p = _payload(**kw)
    p.update(hasLS=True, hasEx=True, hasW=True)
    reg = pd.DataFrame([{"name": "alpha101_001", "set": "alpha101", "family": "价量"}])
    return render("all_a_2018_2026", 1, reg, p, ["2026-01", "2026-02"])


class TestNoFormulaColumn:
    """表格不设公式列——公式只在详情页。"""

    def test_header_has_no_formula_column(self):
        for html in (_render(), _render_ls()):
            heads = _table_headers(html)
            assert "公式" not in heads, f"表头又出现公式列: {heads}"

    def test_row_template_has_no_formula_cell(self):
        tpl = _row_template(_render())
        assert "D.formula" not in tpl, "行模板又渲染公式了（表格列已撤）"
        assert 'class="fml"' not in tpl

    def test_header_and_cell_count_match(self):
        """表头列数 == 行模板 td 数，错位会让指标串列。

        行模板里 ``D.hasLS ? ... : ''`` 那一支的 td 是**条件渲染**的，静态数会多算，
        所以分两种情况各自对齐：无分层列时只看分支之前的部分，有分层列时数全量。
        """
        # ① 无分层回测列
        html = _render()
        base = _row_template(html).split("D.hasLS", 1)[0]
        n_th = len(_table_headers(html))
        n_td = base.count("<td")
        assert n_th == n_td == 11, f"无分层时表头 {n_th} 列 vs 行模板 {n_td} 个 td"

        # ② 有分层回测列（多空/多头/换手 + 超额条件分支）
        html_ls = _render_ls()
        n_th_ls = len(_table_headers(html_ls))
        n_td_ls = _row_template(html_ls).count("<td")
        # hasEx=True 时分支取真，故静态 td 数 == 表头列数（超额那列确实渲染）
        assert n_th_ls == n_td_ls == 15, f"有分层时表头 {n_th_ls} vs td {n_td_ls}"

    def test_no_fml_cell_css_left(self):
        """单元格样式随列一起撤掉；详情页的 .formula / .fmlLet 仍要保留。"""
        html = _render()
        assert not re.search(r"\.fml\s*\{", html), ".fml 单元格样式残留"
        assert ".formula" in html and ".fmlLet" in html


class TestDetailShowsFormula:
    """详情页：不截断、分条、保留 let 别名、转义比较符。"""

    def test_payload_not_truncated(self):
        """长公式必须全文进 payload（不截断才能"看全"）。"""
        html = _render()
        assert FORMULA_LONG in _payload()["formula"]["alpha101_001"]
        # render 把 payload 塞进 JS；公式原样出现在 HTML 里即证明没被截
        assert "rank(argmax(power(inner, 2), 5)) - 0.5" in html

    def test_detail_renders_into_dFml(self):
        html = _render()
        assert 'id="dFml"' in html
        assert "dFml" in html.split("function showDetail", 1)[-1], "详情页没写 dFml"

    def test_head_strip_keeps_first_let(self):
        """⚠️ 回归钉：剥 head 时首段右半必须放回 parts，否则首个 let 定义整段丢失。

        旧写法 ``parts.shift().split(' = ')[0]`` 会把 ``let inner = ...`` 一起丢掉，
        结果主表达式里的 ``inner`` 无从解释（148 个公式中招）。
        """
        js = _render()
        assert "parts[0].slice(eq + 3)" in js, "首段右半没放回 parts ⇒ 首个 let 会丢"
        assert "parts.shift().split(' = ')[0]" not in js, "旧写法回来了，会丢首段 let"

    def test_comparison_operators_escaped(self):
        """公式含 < > & ⇒ 必须经 esc() 才能拼 innerHTML，否则被当标签解析。"""
        js = _render()
        assert "const esc" in js
        assert "esc(p)" in js and "esc(head)" in js and "esc(nm)" in js


class TestSearchStillMatchesFormula:
    """列撤了，但按公式找因子要还能用。"""

    def test_search_includes_formula(self):
        tpl = _render()
        m = re.search(r"if \(q && !\((.*?)\)\) continue;", tpl, re.S)
        assert m, "搜索过滤条件不见了"
        assert "D.formula[r.name]" in m.group(1), "搜索不再匹配公式全文"