"""把研报 PDF 抽成带页码标记的纯文本，便于按关键词定位口径参数。

用法::

    python -m scripts.data_tools.dump_pdf_text <pdf> <out.txt> [--start 1] [--end 30]

为什么不用 Read 直接读 PDF：``Read`` 对 PDF 走的是原始字节行模式（返回 ``%PDF-1.5``
之类的二进制行），无法定位正文；pdfplumber 才能按页抽文字。

典型用途：研报口径核对（如 AI39 / 多因子10 / AI97）——先把 PDF 抽成 txt，
再 grep 字段表、常数表、超参表逐项比对代码实现。抽出的文本建议落在仓库外的
临时目录或 ``reports/`` 下（后者已被 gitignore）。

依赖：``pdfplumber``。它**未列入 pyproject 依赖**（属研究期可选依赖），
故在此处**延迟导入**，缺席时给出明确报错而不是 import 失败。
项目内实测可用的解释器是 ``D:/python/Python312/python.exe``。
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _load_pdfplumber():
    try:
        import pdfplumber
    except ModuleNotFoundError as exc:  # pragma: no cover - 取决于本机环境
        raise RuntimeError(
            "需要 pdfplumber（研究期可选依赖，未列入 pyproject）："
            "请先 `pip install pdfplumber`，并改用装有该包的解释器运行"
            "（项目内为 D:/python/Python312/python.exe）。"
        ) from exc
    return pdfplumber


def dump(pdf: Path, out: Path, start: int = 1, end: int | None = None) -> int:
    """把 ``pdf`` 的第 start~end 页（1-based，含端点）抽成文本写入 ``out``。

    返回实际抽取的页数。``end=None`` 表示抽到末页。
    """
    pdfplumber = _load_pdfplumber()
    chunks: list[str] = []
    with pdfplumber.open(str(pdf)) as doc:
        n = len(doc.pages)
        end = n if end is None else min(end, n)
        for i in range(start - 1, end):
            txt = doc.pages[i].extract_text() or ""
            chunks.append(f"\n<<<PAGE {i + 1}/{n}>>>\n{txt}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(chunks), encoding="utf-8")
    return end - start + 1


def main() -> None:
    ap = argparse.ArgumentParser(description="把研报 PDF 抽成带页码标记的纯文本。")
    ap.add_argument("pdf")
    ap.add_argument("out")
    ap.add_argument("--start", type=int, default=1, help="起始页（1-based，含）")
    ap.add_argument("--end", type=int, default=None, help="结束页（含）；默认到末页")
    a = ap.parse_args()
    pages = dump(Path(a.pdf), Path(a.out), a.start, a.end)
    print(f"dumped {pages} pages -> {a.out}")


if __name__ == "__main__":
    main()
