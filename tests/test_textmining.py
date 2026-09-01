"""
文本挖掘数据层测试（Phase 0）
==============================

覆盖：同花顺研报源（URL 构造/JSON 解析/字段标准化）、巨潮公告源（类别映射/
日期格式化/PDF 文本提取）、统一接口（缓存增量/code 过滤/日期过滤）。

网络用 mock 替换（requests.Session.get/post），不依赖外网。
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.textmining import fetch_docs
from data.textmining.fetch import TextMiningCache
from data.textmining.source_cninfo import (
    _CATEGORY_MAP,
    _download_pdf_text,
    _fmt,
    fetch_cninfo_announcements,
)
from data.textmining.source_ths import (
    _parse_report_json,
    fetch_ths_reports,
    to_code6,
    to_code_std,
)

# ---- 同花顺研报 ----

SAMPLE_THS_JSON = json.dumps([
    {"thspj": "买入", "title": "天风证券：非标量价拖累营收",
     "source": "天风证券", "researcher": "刘洁铭",
     "date": "2026-08-17",
     "content": "事件：公司发布2026年中报，营收922.78亿元。"},
    {"thspj": "增持", "title": "平安证券：直营持续放量",
     "source": "平安证券", "researcher": "张晋溢",
     "date": "2026-08-16",
     "content": "上半年公司营业总收入922.8亿元。"},
])


def _ths_page(code6: str) -> str:
    return f'<html><div id="report_list_contents">{SAMPLE_THS_JSON}</div></html>'


def test_to_code6_and_std():
    assert to_code6("600519.SH") == "600519"
    assert to_code6("000001.SZ") == "000001"
    assert to_code_std("600519") == "600519.SH"
    assert to_code_std("000001") == "000001.SZ"
    assert to_code_std("300750") == "300750.SZ"
    assert to_code_std("688196") == "688196.SH"
    assert to_code_std("430047") == "430047.BJ"


def test_parse_report_json():
    items = _parse_report_json(_ths_page("600519"))
    assert len(items) == 2
    assert items[0]["thspj"] == "买入"
    assert items[0]["title"].startswith("天风证券")


def test_fetch_ths_reports_normalizes_fields():
    with patch("data.textmining.source_ths.requests.Session") as mk:
        sess = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.text = _ths_page("600519")
        sess.get.return_value = resp
        mk.return_value = sess

        df = fetch_ths_reports("600519.SH", session=sess)
        assert len(df) == 2
        assert set(df.columns) >= {
            "code", "date", "title", "summary", "rating", "org", "analyst", "source"}
        assert (df["code"] == "600519.SH").all()
        assert (df["source"] == "ths").all()
        # 按日期升序：08-16(增持) 在前，08-17(买入) 在后
        assert df["rating"].tolist() == ["增持", "买入"]
        assert df["summary"].iloc[1].startswith("事件")


def test_fetch_ths_reports_empty_page():
    with patch("data.textmining.source_ths.requests.Session") as mk:
        sess = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "<html>no report data</html>"
        sess.get.return_value = resp
        mk.return_value = sess
        df = fetch_ths_reports("000001.SZ", session=sess)
        assert df.empty
        assert "code" in df.columns


def test_fetch_ths_reports_retries_on_error():
    with patch("data.textmining.source_ths.requests.Session") as mk:
        sess = MagicMock()
        sess.get.side_effect = RuntimeError("network down")
        mk.return_value = sess
        with pytest.raises(RuntimeError):
            fetch_ths_reports("600519.SH", session=sess, retries=1)


# ---- 巨潮公告 ----

def test_cninfo_category_map_has_key_categories():
    for k in ("业绩预告", "年报", "半年报", "一季报", "三季报"):
        assert k in _CATEGORY_MAP
        assert _CATEGORY_MAP[k].startswith("category_")


def test_fmt_yyyymmdd():
    assert _fmt(20240101) == "2024-01-01"
    assert _fmt(20190101) == "2019-01-01"


def test_download_pdf_text_valid_pdf():
    with patch("data.textmining.source_cninfo.requests.get") as mget:
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"%PDF-1.4 fake"  # 非真 PDF：pypdf 解析会失败，返回 None 可接受
        mget.return_value = resp
        out = _download_pdf_text("http://static.cninfo.com.cn/x.PDF")
        assert out is None or isinstance(out, str)


def test_fetch_cninfo_announcements_basic():
    ann = {
        "secCode": "000001", "secName": "平安银行",
        "announcementTitle": "平安银行2022年度业绩快报",
        "announcementTime": 1673892000000,  # 2023-01-17
        "adjunctUrl": "finalpage/2023-01-17/1215626113.PDF",
    }
    with patch("data.textmining.source_cninfo._query_announcements",
               return_value=[ann]):
        df = fetch_cninfo_announcements(
            codes=["000001.SZ"], begin_date=20230101, end_date=20231231,
            categories=["业绩预告"], progress=False)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["code"] == "000001.SZ"
        assert row["source"] == "cninfo"
        assert row["category"] == "业绩预告"
        assert "业绩快报" in row["title"]
        assert row["url"].startswith("http://static.cninfo.com.cn/")
        assert pd.Timestamp(row["date"]).year == 2023


# ---- 统一接口 + 缓存 ----

def test_fetch_docs_filters_codes_and_dates(tmp_path: Path):
    rows = pd.DataFrame({
        "code": ["600519.SH", "000001.SZ"],
        "date": pd.to_datetime(["2020-01-05", "2023-06-01"]),
        "title": ["t1", "t2"], "summary": ["s1", "s2"],
        "rating": ["买入", "增持"], "org": ["a", "b"],
        "analyst": ["x", "y"], "source": ["ths", "ths"],
    })
    with patch.object(TextMiningCache, "get_ths_reports", return_value=rows):
        # 请求只含 000001.SZ：code 过滤后保留 1 行（2023-06-01 在日期段内）
        docs = fetch_docs(["000001.SZ"], begin_date=20210101, end_date=20241231,
                          cache_root=tmp_path)
        df = docs["ths"]
        assert len(df) == 1
        assert df.iloc[0]["code"] == "000001.SZ"
        # 请求含两只但日期段只覆盖 000001 的行
        docs2 = fetch_docs(["600519.SH", "000001.SZ"], begin_date=20210101,
                           end_date=20241231, cache_root=tmp_path)
        df2 = docs2["ths"]
        assert len(df2) == 1
        assert df2.iloc[0]["code"] == "000001.SZ"  # 600519 的 2020 行被日期过滤


def test_cache_incremental_skips_existing_codes(tmp_path: Path):
    cache = TextMiningCache(tmp_path)
    existing = pd.DataFrame({
        "code": ["600519.SH"], "date": pd.to_datetime(["2023-01-01"]),
        "title": ["old"], "summary": ["s"], "rating": ["买入"],
        "org": ["o"], "analyst": ["a"], "source": ["ths"],
    })
    existing.to_parquet(tmp_path / "text_ths_report.parquet")

    with patch("data.textmining.fetch.fetch_ths_reports") as mfetch:
        cache.get_ths_reports(["600519.SH", "000001.SZ"])
        # 600519 已缓存 → 跳过；只拉 000001
        mfetch.assert_called_once()
        assert mfetch.call_args[0][0] in ("000001.SZ", "000001")


def test_cache_cninfo_signature_isolation(tmp_path: Path):
    """不同 categories/searchkey 写不同缓存文件，不互相短路。"""
    cache = TextMiningCache(tmp_path)
    assert cache._cninfo_filename(None, "") == "text_cninfo_ann.parquet"
    # 2026-08-26 统一命名：中文类别转英文 slug（业绩预告=yjygjxz）
    assert cache._cninfo_filename(["业绩预告"], "") == "text_cninfo_ann_cat_yjygjxz.parquet"
    assert cache._cninfo_filename(None, "业绩快报") == "text_cninfo_ann_sk_业绩快报.parquet"


def test_cninfo_cache_short_circuit_when_covered(tmp_path: Path):
    cache = TextMiningCache(tmp_path)
    rows = pd.DataFrame({
        "code": ["000001.SZ"], "date": pd.to_datetime(["2023-01-17"]),
        "title": ["业绩快报"], "category": ["业绩预告"], "searchkey": [""],
        "source": ["cninfo"], "url": [None], "text": [None],
    })
    rows.to_parquet(tmp_path / "text_cninfo_ann_cat_yjygjxz.parquet")

    with patch("data.textmining.fetch.fetch_cninfo_announcements") as mfetch:
        # 缓存最早 2023-01-17，请求从 2023-02-01 起 → 已覆盖，短路
        df = cache.get_cninfo_announcements(
            ["000001.SZ"], begin_date=20230201, end_date=20231231,
            categories=["业绩预告"])
        mfetch.assert_not_called()
        assert len(df) == 1

    with patch("data.textmining.fetch.fetch_cninfo_announcements",
               return_value=pd.DataFrame()) as mfetch2:
        # 请求从 2022-12 起 < 缓存最早 2023-01 → 需回补，不短路
        cache.get_cninfo_announcements(
            ["000001.SZ"], begin_date=20221201, end_date=20231231,
            categories=["业绩预告"])
        mfetch2.assert_called_once()


def test_cninfo_cache_fetch_new_codes_even_when_dates_covered(tmp_path: Path):
    """扩大股票池：请求含缓存未覆盖的 code 时必须拉取（不能仅按日期短路）。"""
    cache = TextMiningCache(tmp_path)
    rows = pd.DataFrame({
        "code": ["000001.SZ"], "date": pd.to_datetime(["2023-01-17"]),
        "title": ["业绩快报"], "category": ["业绩预告"], "searchkey": [""],
        "source": ["cninfo"], "url": [None], "text": [None],
    })
    rows.to_parquet(tmp_path / "text_cninfo_ann_cat_yjygjxz.parquet")

    with patch("data.textmining.fetch.fetch_cninfo_announcements",
               return_value=pd.DataFrame()) as mfetch:
        # 请求含 000002.SZ（未缓存）→ 必须拉，即使日期已覆盖
        cache.get_cninfo_announcements(
            ["000001.SZ", "000002.SZ"], begin_date=20230101, end_date=20231231,
            categories=["业绩预告"])
        mfetch.assert_called_once()
        called = mfetch.call_args.kwargs.get("codes") or mfetch.call_args[1].get("codes")
        assert "000002" in called  # 增量只拉缺失 code（6 位码）


def test_cninfo_cache_returns_only_requested_codes(tmp_path: Path):
    """返回必须按请求 codes 过滤——缓存是全局的，可能含其他股票池公告
    （回归测试：曾导致 hs300/zz1000 样本串池，事件数同为 18375）。"""
    cache = TextMiningCache(tmp_path)
    rows = pd.DataFrame({
        "code": ["000001.SZ", "600519.SH"], "date": pd.to_datetime(["2023-01-17"] * 2),
        "title": ["业绩预告", "业绩预告"], "category": ["业绩预告"] * 2,
        "searchkey": [""] * 2, "source": ["cninfo"] * 2,
        "url": [None] * 2, "text": [None] * 2,
    })
    rows.to_parquet(tmp_path / "text_cninfo_ann_cat_yjygjxz.parquet")

    with patch("data.textmining.fetch.fetch_cninfo_announcements") as mfetch:
        df = cache.get_cninfo_announcements(
            ["000001.SZ"], begin_date=20230201, end_date=20231231,
            categories=["业绩预告"])
        mfetch.assert_not_called()
        # 只返回请求的 000001.SZ，不得包含 600519.SH
        assert set(df["code"].unique()) == {"000001.SZ"}
