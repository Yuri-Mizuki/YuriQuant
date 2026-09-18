"""日报 markdown → 邮件 HTML（内联样式，适配 163 / QQ / Gmail 等客户端）。

背景：`agently-cli message +send` 对 `.md` / `.markdown` 文件按 **Markdown** 发送，
而国内多数邮箱客户端并不渲染 Markdown —— 收件人看到的是带 `#`、`|`、`**` 的
原始符号（实测 2026-09-17 首封即如此，用户反馈"不太可读"）。改发 `.html` 才能
正常显示表格。样式全部内联（不依赖 `<style>` 标签），避免被客户端剥离。

用法：
    python scripts/reporting/md_to_email_html.py --md reports/alla_daily/email_body_20260917.md
    python scripts/reporting/md_to_email_html.py --md in.md --out out.html

被 `build_daily_email_body.py --html-out` 复用（见下方 convert）。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import markdown

# ---- 内联样式（邮件客户端只可靠支持 style 属性，不支持外链/类名）----
_BODY = (
    "margin:0;padding:20px 12px;background:#f4f5f7;"
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',"
    "'Hiragino Sans GB','Microsoft YaHei',sans-serif;"
    "font-size:14px;line-height:1.75;color:#1f2329;"
)
_CARD = (
    "max-width:820px;margin:0 auto;background:#ffffff;border:1px solid #e5e7eb;"
    "border-radius:10px;padding:26px 28px;"
)
_H1 = (
    "font-size:19px;line-height:1.45;margin:0 0 18px;padding:0 0 12px;"
    "border-bottom:2px solid #2f6fed;color:#111827;font-weight:700;"
)
_H2 = (
    "font-size:15px;line-height:1.5;margin:26px 0 10px;padding:2px 0 2px 10px;"
    "border-left:3px solid #2f6fed;color:#1f2329;font-weight:700;"
)
_H3 = "font-size:14px;margin:18px 0 8px;color:#1f2329;font-weight:700;"
_P = "margin:9px 0;"
_TABLE = (
    "width:100%;border-collapse:collapse;margin:10px 0 14px;"
    "font-size:13px;font-variant-numeric:tabular-nums;"
)
_TH = (
    "background:#eef2f8;color:#243b53;font-weight:600;text-align:left;"
    "padding:8px 10px;border-bottom:1px solid #d7dee8;white-space:nowrap;"
)
_TD = "padding:7px 10px;border-bottom:1px solid #eef0f3;vertical-align:top;"
_BQ = (
    "margin:12px 0;padding:10px 14px;background:#f8fafc;"
    "border-left:3px solid #cbd5e1;color:#475569;font-size:13px;"
)
_LI = "margin:5px 0;"
_HR = "border:0;border-top:1px solid #e5e7eb;margin:22px 0;"
_EM = "color:#8a94a6;font-size:12px;font-style:normal;"
_STRONG = "color:#111827;"

_NUM = re.compile(r"^[+\-]?\d[\d,]*(?:\.\d+)?%?$")


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _enhance_tables(html: str) -> str:
    """表格：表头灰底、首列居中、数值右对齐、表体斑马纹。"""

    def fix_table(m: re.Match) -> str:
        tbl = m.group(0)
        tbl = tbl.replace(
            "<table>",
            f'<table cellpadding="0" cellspacing="0" border="0" style="{_TABLE}">',
        )
        tbl = tbl.replace("<th>", f'<th style="{_TH}">')
        zi = {"i": 0}

        def fix_row(rm: re.Match) -> str:
            row = rm.group(1)
            if "<th" in row:  # 表头行
                return f'<tr style="background:#eef2f8;">{row}</tr>'
            ci = {"j": 0}

            def fix_cell(cm: re.Match) -> str:
                j = ci["j"]
                ci["j"] += 1
                inner = cm.group(1)
                txt = _strip_tags(inner).replace(" ", "")
                if j == 0:
                    align = "center"
                elif _NUM.match(txt):
                    align = "right"
                else:
                    align = "left"
                return f'<td style="{_TD}text-align:{align};">{inner}</td>'

            row = re.sub(r"<td>(.*?)</td>", fix_cell, row, flags=re.S)
            bg = ' style="background:#fafbfc;"' if zi["i"] % 2 == 1 else ""
            zi["i"] += 1
            return f"<tr{bg}>{row}</tr>"

        return re.sub(r"<tr>(.*?)</tr>", fix_row, tbl, flags=re.S)

    return re.sub(r"<table>.*?</table>", fix_table, html, flags=re.S)


def _style_inline(html: str) -> str:
    """给 markdown 产出的裸标签注入内联样式。"""
    repl = [
        ("<h1>", f'<h1 style="{_H1}">'),
        ("<h2>", f'<h2 style="{_H2}">'),
        ("<h3>", f'<h3 style="{_H3}">'),
        ("<p>", f'<p style="{_P}">'),
        ("<blockquote>", f'<blockquote style="{_BQ}">'),
        ("<li>", f'<li style="{_LI}">'),
        ("<hr />", f'<hr style="{_HR}" />'),
        ("<hr/>", f'<hr style="{_HR}" />'),
        ("<em>", f'<em style="{_EM}">'),
        ("<strong>", f'<strong style="{_STRONG}">'),
    ]
    for a, b in repl:
        html = html.replace(a, b)
    # 引用块内的段落不需要额外外边距
    html = html.replace(f'<blockquote style="{_BQ}"><p style="{_P}">',
                        f'<blockquote style="{_BQ}"><p style="margin:2px 0;">')
    return html


def convert(md_text: str, title: str = "YuriQuant 每日选股预测") -> str:
    """markdown 正文 → 完整 HTML 文档字符串（可直接作为 --body-file 内容）。

    用 `nl2br`：日报头部若干「**字段**：值」行之间是软换行，不转 `<br>` 会被
    markdown 合并成一行。
    """
    body = markdown.markdown(
        md_text, extensions=["tables", "sane_lists", "nl2br", "fenced_code"]
    )
    body = _enhance_tables(body)
    body = _style_inline(body)
    safe_title = (title or "").replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<!DOCTYPE html>\n"
        '<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{safe_title}</title>\n</head>\n"
        f'<body style="{_BODY}">\n'
        f'<div style="{_CARD}">\n{body}\n</div>\n'
        "</body>\n</html>\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="日报 markdown → 邮件 HTML")
    ap.add_argument("--md", required=True, help="输入的 markdown 路径")
    ap.add_argument("--out", default=None, help="输出 html 路径（默认同名 .html）")
    ap.add_argument("--title", default="YuriQuant 每日选股预测", help="HTML 标题")
    args = ap.parse_args()

    src = Path(args.md)
    if not src.exists():
        raise SystemExit(f"输入文件不存在：{src}")
    out = Path(args.out) if args.out else src.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    html = convert(src.read_text(encoding="utf-8"), title=args.title)
    out.write_text(html, encoding="utf-8")
    print(f"OK  {src} -> {out}  ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
