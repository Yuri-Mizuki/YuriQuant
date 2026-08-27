"""
巨潮公告辅源抓取器（可选）
========================

从巨潮资讯网（cninfo）抓取上市公司公告（业绩预告/定期报告/调研纪要等）。

通道（2026-08-17 实测验证）：
    POST http://www.cninfo.com.cn/new/hisAnnouncement/query
    返回 JSON（announcements 数组），含 secCode/secName/announcementTitle/
    announcementTime/adjunctUrl（PDF 路径）。
    PDF 全文：http://static.cninfo.com.cn/{adjunctUrl}，pypdf 可解析文本。

用途（相对同花顺研报主源的补充）：
    - 业绩预告"业绩变动原因"原文（公司视角，与研报观点互补）
    - 定期报告"管理层讨论与分析"（MD&A）、"未来展望"
    - 投资者关系活动记录表（调研问答，Q&A 格式）
    - 事件时间戳精确到分钟，天然 PIT

注意：
    - 公告列表接口免费无鉴权，按日期段分页拉取；PDF 下载 + pypdf 解析较慢，
      默认关闭（with_pdf=False 只取元数据 + 标题）。
    - category 枚举见 _CATEGORY_MAP（巨潮内部编码，实测 '业绩预告' 等可用）。
"""
from __future__ import annotations

import io
import re
import time
from typing import Iterable

import pandas as pd
import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
PDF_URL = "http://static.cninfo.com.cn/{adjunct_url}"

# 巨潮 category 编码（2026-08-17 实测有效，完整字典见 akshare 内部 __get_category_dict）
_CATEGORY_MAP = {
    "年报": "category_ndbg_szsh",
    "半年报": "category_bndbg_szsh",
    "一季报": "category_yjdbg_szsh",
    "三季报": "category_sjdbg_szsh",
    "业绩预告": "category_yjygjxz_szsh",
    "权益分派": "category_qyfpxzcs_szsh",
    "董事会": "category_dshgg_szsh",
    "监事会": "category_jshgg_szsh",
    "股东大会": "category_gddh_szsh",
    "日常经营": "category_rcjy_szsh",
    "公司治理": "category_gszl_szsh",
    "中介报告": "category_zj_szsh",
    "首发": "category_sf_szsh",
    "增发": "category_zf_szsh",
    "股权激励": "category_gqjl_szsh",
    "配股": "category_pg_szsh",
    "解禁": "category_jj_szsh",
    "公司债": "category_gszq_szsh",
    "可转债": "category_kzzq_szsh",
    "其他融资": "category_qtrz_szsh",
    "股权变动": "category_gqbd_szsh",
    "补充更正": "category_bcgz_szsh",
    "澄清致歉": "category_cqdq_szsh",
    "风险提示": "category_fxts_szsh",
    "特别处理和退市": "category_tbclts_szsh",
    "退市整理期": "category_tszlq_szsh",
}

_STOCK_ID_CACHE: dict[str, str] = {}


def _get_stock_ids() -> dict[str, str]:
    """获取全市场 代码->orgId 映射（巨潮接口，首次调用拉取并缓存）。"""
    global _STOCK_ID_CACHE
    if _STOCK_ID_CACHE:
        return _STOCK_ID_CACHE
    # 巨潮的股票代码列表接口：分市场分页获取
    url = "http://www.cninfo.com.cn/new/data/szse_stock.json"
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        r.raise_for_status()
        data = r.json()
        for item in data.get("stockList", []):
            _STOCK_ID_CACHE[str(item["code"])] = str(item["orgId"])
    except Exception:
        _STOCK_ID_CACHE = {}
    return _STOCK_ID_CACHE


def to_code6(code: str) -> str:
    return str(code).split(".")[0].zfill(6)


def to_code_std(code6: str) -> str:
    c = str(code6).zfill(6)
    if c[0] in ("6", "9", "5"):
        return f"{c}.SH"
    if c[0] in ("0", "2", "3"):
        return f"{c}.SZ"
    if c[0] in ("4", "8"):
        return f"{c}.BJ"
    return c


def _headers() -> dict:
    return {
        "User-Agent": UA,
        "Referer": "http://www.cninfo.com.cn/",
        "X-Requested-With": "XMLHttpRequest",
    }


def _query_announcements(
    category_code: str,
    start_date: str,
    end_date: str,
    searchkey: str = "",
    stock_code: str = "",
    page_size: int = 30,
    max_pages: int | None = None,
    session: requests.Session | None = None,
    retries: int = 2,
) -> list[dict]:
    """分页查询巨潮公告列表。

    Parameters
    ----------
    category_code : str
        _CATEGORY_MAP 中的编码；'' 表示不限类别。
    start_date/end_date : str
        YYYY-MM-DD。
    searchkey : str
        标题关键词（如"业绩快报""投资者关系活动记录表"）。
    stock_code : str
        6 位代码（限单只时）；'' 为全市场。
    max_pages : int | None
        最多翻页数，None 为全部分页。
    """
    s = session or requests.Session()
    payload = {
        "pageNum": "1",
        "pageSize": str(page_size),
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": stock_code,
        "searchkey": searchkey,
        "secid": "",
        "category": category_code,
        "trade": "",
        "seDate": f"{start_date}~{end_date}",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    out: list[dict] = []
    page = 1
    while True:
        payload["pageNum"] = str(page)
        ok = False
        for _ in range(retries + 1):
            try:
                r = s.post(QUERY_URL, data=payload, headers=_headers(), timeout=20)
                r.raise_for_status()
                j = r.json()
                anns = j.get("announcements") or []
                out.extend(anns)
                total = int(j.get("totalAnnouncement", 0) or 0)
                ok = True
                break
            except Exception:
                time.sleep(1.0)
        if not ok:
            break
        if page * page_size >= total or len(anns) == 0:
            break
        if max_pages is not None and page >= max_pages:
            break
        page += 1
        time.sleep(0.3)
    return out


def fetch_cninfo_announcements(
    codes: Iterable[str] | None = None,
    begin_date: int = 20190101,
    end_date: int = 20261231,
    categories: list[str] | None = None,
    searchkey: str = "",
    with_pdf: bool = False,
    max_pages: int | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """抓取巨潮公告（按类别/关键词/个股范围）。

    Parameters
    ----------
    codes : Iterable[str] | None
        标准代码列表；None = 全市场。
    begin_date/end_date : int
        YYYYMMDD 日期段。
    categories : list[str] | None
        类别（_CATEGORY_MAP 的 key）；None 或空 = 不限。
    searchkey : str
        标题关键词（如"业绩快报"）。
    with_pdf : bool
        是否下载 PDF 并解析正文文本（慢，默认关）。
    max_pages : int | None
        每个查询最多翻页数（防全市场大表拉爆）。

    Returns
    -------
    pd.DataFrame
        列：code(标准)/date/title/category/searchkey/source(=cninfo)/url/
        text(可选，with_pdf=True 时含公告正文)。
    """
    codes6 = [to_code6(c) for c in codes] if codes is not None else None
    cats = {c: _CATEGORY_MAP[c] for c in (categories or []) if c in _CATEGORY_MAP}
    if categories and not cats:
        raise ValueError(f"未知类别 {categories}，可用: {list(_CATEGORY_MAP)}")

    stock_ids = _get_stock_ids()
    rows: list[dict] = []
    with requests.Session() as s:
        for cat_name, cat_code in (cats.items() if cats else [(None, "")]):
            if codes6 is None:
                stock_param = ""
            else:
                # 逐只查询（带 orgId 限定），避免全市场大表
                for c6 in codes6:
                    org = stock_ids.get(c6, "")
                    stock_param = f"{c6},{org}" if org else c6
                    anns = _query_announcements(
                        cat_code, _fmt(begin_date), _fmt(end_date),
                        searchkey=searchkey, stock_code=stock_param,
                        max_pages=max_pages, session=s)
                    for a in anns:
                        rows.append(_row(a, cat_name, searchkey))
                    if progress:
                        print(f"[cninfo] {cat_name or '全部'} {c6}: {len(anns)} 条")
                    time.sleep(0.3)
            if codes6 is None:
                anns = _query_announcements(
                    cat_code, _fmt(begin_date), _fmt(end_date),
                    searchkey=searchkey, max_pages=max_pages, session=s)
                for a in anns:
                    rows.append(_row(a, cat_name, searchkey))
                if progress:
                    print(f"[cninfo] {cat_name or '全部'}: {len(anns)} 条（第 1 页）")

    df = pd.DataFrame(rows)
    if df.empty:
        cols = ["code", "date", "title", "category", "searchkey", "source", "url", "text"]
        return pd.DataFrame(columns=cols)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["code", "date"]).reset_index(drop=True)
    if with_pdf and not df.empty:
        df["text"] = df["url"].map(lambda u: _download_pdf_text(u) if isinstance(u, str) and u else None)
    return df


def _row(a: dict, cat_name: str | None, searchkey: str) -> dict:
    ts = a.get("announcementTime")
    if isinstance(ts, (int, float)) and ts > 0:
        date = pd.to_datetime(ts, unit="ms", utc=True).tz_convert("Asia/Shanghai")
    else:
        date = pd.NaT
    return {
        "code": to_code_std(str(a.get("secCode", ""))),
        "date": date,
        "title": a.get("announcementTitle"),
        "category": cat_name or "",
        "searchkey": searchkey,
        "source": "cninfo",
        "url": ("http://static.cninfo.com.cn/" + a["adjunctUrl"]) if a.get("adjunctUrl") else None,
        "text": None,
    }


def _download_pdf_text(url: str, timeout: float = 90.0) -> str | None:
    """下载公告 PDF 并提取文本（pypdf）。失败返回 None。"""
    try:
        import pypdf
    except ImportError:
        return None
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        r.raise_for_status()
        if not r.content[:5] == b"%PDF-":
            return None
        pr = pypdf.PdfReader(io.BytesIO(r.content))
        parts = [p.extract_text() or "" for p in pr.pages]
        return "\n".join(parts)
    except Exception:
        return None


def _fmt(d: int) -> str:
    s = str(d)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"
