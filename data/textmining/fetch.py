"""
文本挖掘统一接口（Phase 0）
==========================

主源：同花顺研报（source_ths，标题+摘要全文+评级）
辅源：巨潮公告（source_cninfo，业绩预告/定期报告/调研纪要，可选）

设计（2026-08-17 调研结论：单源即可覆盖研报主线，巨潮为补充）：
    - 统一入口 ``fetch_docs(code, begin, end, sources=('ths', 'cninfo'))``
    - parquet 缓存（e:/data/parquet/，命名 <域>_<表>.parquet，对齐 data/cache.py）
    - 增量：同花顺一次返回全历史 → 本地已有该 code 则跳过，只拉缺失 code；
      巨潮按日期段查询 → 记录每 code 的最早日期，回补更早历史。
    - PIT：研报/公告均带精确发布日期，入库即按日期过滤，无前视。

缓存表：
    text_ths_report.parquet    # 同花顺研报（索引 date, code）
    text_cninfo_ann.parquet    # 巨潮公告（索引 date, code）
    _meta.json                 # 每 code 最早抓取日期（增量水位）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import Config

from data.textmining.source_ths import fetch_ths_reports, to_code6
from data.textmining.source_cninfo import fetch_cninfo_announcements

# 统一输出列（研报）
THS_COLS = ["code", "date", "title", "summary", "rating", "org", "analyst", "source"]
# 统一输出列（公告）
CNINFO_COLS = ["code", "date", "title", "category", "searchkey", "source", "url", "text"]


class TextMiningCache:
    """文本数据 parquet 缓存 + 增量拉取。"""

    def __init__(self, cache_root: Path | str | None = None):
        if cache_root is None:
            cache_root = Config.cache()["root"]
        self._root = Path(str(cache_root).replace("//", "/"))
        self._root.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._root / "_meta.json"
        self._meta: dict = self._load_meta()

    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            with open(self._meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_meta(self) -> None:
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2, ensure_ascii=False)

    @property
    def root(self) -> Path:
        return self._root

    # ---- 同花顺研报 ----
    def get_ths_reports(self, codes: Iterable[str]) -> pd.DataFrame:
        """按 code 增量拉取同花顺研报并缓存。

        同花顺一次返回全历史，故策略为：本地缺的 code 整拉，已有 code 跳过。
        返回全部已缓存的研报（调用方自行按日期过滤）。
        """
        p = self._root / "text_ths_report.parquet"
        local = pd.DataFrame()
        if p.exists():
            local = pd.read_parquet(p)
        cached_codes: set[str] = set()
        if not local.empty and "code" in local.columns:
            cached_codes = set(local["code"].unique())

        missing = [c for c in codes if to_code6(c) not in {to_code6(x) for x in cached_codes}]
        if missing:
            frames = [local] if not local.empty else []
            for c in missing:
                try:
                    df = fetch_ths_reports(c)
                    if not df.empty:
                        frames.append(df)
                except Exception as e:  # noqa: BLE001
                    print(f"[textmining] ths {c} 失败: {type(e).__name__}: {str(e)[:80]}")
            if len(frames) > 1 or (frames and local.empty):
                combined = pd.concat(frames, ignore_index=True)
                combined = combined.drop_duplicates(subset=["code", "date", "title"], keep="last")
                combined = combined.sort_values(["code", "date"]).reset_index(drop=True)
                combined.to_parquet(p, compression="snappy")
                local = combined
        return local

    # ---- 巨潮公告 ----
    # 中文类别 -> 英文 slug（2026-08-26 统一命名：缓存文件避免中文文件名，
    # 跨平台/脚本硬编码均更稳；slug 取自巨潮内部编码英文段，如 业绩预告=yjygjxz）。
    _CNINFO_CAT_SLUG = {
        "年报": "ndbg",
        "半年报": "bndbg",
        "一季报": "yjdbg",
        "三季报": "sjdbg",
        "业绩预告": "yjygjxz",
        "权益分派": "qyfpxzcs",
        "董事会": "dshgg",
        "监事会": "jshgg",
        "股东大会": "gddh",
        "日常经营": "rcjy",
        "公司治理": "gszl",
        "中介报告": "zj",
        "首发": "sf",
        "增发": "zf",
        "股权激励": "gqjl",
        "配股": "pg",
        "解禁": "jj",
    }

    @staticmethod
    def _cninfo_filename(categories: list[str] | None, searchkey: str) -> str:
        """缓存文件按过滤条件签名隔离，避免不同类别/关键词互相短路。

        类别名转英文 slug（见 _CNINFO_CAT_SLUG），未知类别回退为原始名。
        """
        name = "text_cninfo_ann"
        if categories:
            slugs = [TextMiningCache._CNINFO_CAT_SLUG.get(c, c) for c in categories]
            name += "_cat_" + "_".join(slugs)
        if searchkey:
            name += "_sk_" + searchkey
        return name + ".parquet"

    def get_cninfo_announcements(
        self,
        codes: Iterable[str] | None = None,
        begin_date: int = 20190101,
        end_date: int = 20261231,
        categories: list[str] | None = None,
        searchkey: str = "",
        with_pdf: bool = False,
        max_pages: int | None = None,
    ) -> pd.DataFrame:
        """按日期段增量拉取巨潮公告并缓存。

        策略：记录每 code 的最早抓取日期；请求 begin 早于缓存最早日期时回补。
        不同 categories/searchkey 写不同缓存文件（签名隔离）。
        """
        p = self._root / self._cninfo_filename(categories, searchkey)
        local = pd.DataFrame()
        if p.exists():
            local = pd.read_parquet(p)

        need_fetch = True
        earliest = None
        missing: list[str] = []
        if not local.empty and "code" in local.columns:
            cached_codes = set(local["code"].unique())
            if codes is not None:
                # 关键：请求 codes 中有未覆盖的 → 必须拉（不能只按日期短路，
                # 否则扩大股票池时新 code 被旧缓存误判为已覆盖）
                code6s = {to_code6(c) for c in codes}
                missing = [c for c in code6s
                           if to_code_std(c) not in cached_codes]
                if missing:
                    need_fetch = True
                    earliest = None  # 有缺失 code，跳过日期短路
                else:
                    local_sub = local[local["code"].isin(
                        [to_code_std(c) for c in code6s])]
                    if not local_sub.empty:
                        earliest = local_sub["date"].min()
            else:
                earliest = local["date"].min()
            if earliest is not None and need_fetch is not False:
                # 月度粒度短路：缓存最早到 begin 同月即可视为覆盖。
                # 日粒度会因节假日边界（如 begin=20190101 元旦 vs 缓存最早 2019-01-04）
                # 反复触发重拉，而公告事件按月度截面聚合，月内边界差可忽略。
                earliest_ym = int(pd.Timestamp(earliest).strftime("%Y%m"))
                begin_ym = int(str(begin_date)[:6])
                if earliest_ym <= begin_ym:
                    need_fetch = False

        if need_fetch:
            # 只拉缺失 code（增量），避免已有 code 被反复全量重拉
            fetch_codes = missing if (codes is not None and missing) else codes
            df = fetch_cninfo_announcements(
                codes=fetch_codes, begin_date=begin_date, end_date=end_date,
                categories=categories, searchkey=searchkey,
                with_pdf=with_pdf, max_pages=max_pages)
            if not df.empty:
                combined = pd.concat([local, df], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["code", "date", "title"], keep="last")
                combined = combined.sort_values(["code", "date"]).reset_index(drop=True)
                combined.to_parquet(p, compression="snappy")
                local = combined

        # 关键：按请求 codes 过滤返回（缓存是全局的，可能含其他股票池的公告；
        # 若不过滤，hs300 请求会返回 zz1000 的公告 → 样本串池）
        if codes is not None and not local.empty:
            code6s = {to_code6(c) for c in codes}
            local = local[local["code"].isin([to_code_std(c) for c in code6s])]
        return local


def to_code_std(code6: str) -> str:
    from data.textmining.source_ths import to_code_std as _std
    return _std(code6)


def fetch_docs(
    codes: Iterable[str],
    begin_date: int = 20190101,
    end_date: int = 20261231,
    sources: tuple[str, ...] = ("ths",),
    cache_root: Path | str | None = None,
    with_pdf: bool = False,
    cninfo_categories: list[str] | None = None,
    cninfo_searchkey: str = "",
    cninfo_max_pages: int | None = None,
) -> dict[str, pd.DataFrame]:
    """统一入口：抓取文本数据（研报/公告），返回 {source: DataFrame}。

    Parameters
    ----------
    codes : Iterable[str]
        标准代码列表（600519.SH）。
    begin_date/end_date : int
        YYYYMMDD 日期段。
    sources : tuple
        要抓的源，可取 'ths'（同花顺研报）/ 'cninfo'（巨潮公告）。
    with_pdf : bool
        cninfo 是否下载 PDF 全文（慢，默认关）。下载后 text 列为公告正文。
    cninfo_categories : list[str] | None
        巨潮公告类别过滤，如 ['业绩预告', '年报']；None = 全部公告。
    cninfo_searchkey : str
        巨潮公告标题关键词，如 '业绩快报'/'投资者关系活动记录表'。
    cninfo_max_pages : int | None
        巨潮单查询最多翻页数（防全市场大表拉爆），None = 全部。
    """
    cache = TextMiningCache(cache_root)
    result: dict[str, pd.DataFrame] = {}
    codes = list(codes)
    if not codes:
        return result

    if "ths" in sources:
        df = cache.get_ths_reports(codes)
        if not df.empty:
            req_codes = {to_code_std(to_code6(c)) for c in codes}
            df = df[df["code"].isin(req_codes)]
            df = df[(df["date"] >= pd.Timestamp(str(begin_date))) &
                    (df["date"] < pd.Timestamp(str(end_date)) + pd.Timedelta(days=1))]
        result["ths"] = df

    if "cninfo" in sources:
        df = cache.get_cninfo_announcements(
            codes=codes, begin_date=begin_date, end_date=end_date,
            categories=cninfo_categories, searchkey=cninfo_searchkey,
            with_pdf=with_pdf, max_pages=cninfo_max_pages)
        result["cninfo"] = df

    return result
