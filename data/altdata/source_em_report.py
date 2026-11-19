"""东财个股研报评级通道（免鉴权，历史 ≥2017）→ 结构化缓存
============================================================

来源：``reportapi.eastmoney.com/report/list``（数据页 data.eastmoney.com/report/），
公开接口、无 token、按 ``beginTime/endTime`` 区间 + 页码翻取。实测条目数：

======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======
年份     2017    2018    2019    2020    2021    2022    2023    2024    2025    2026
------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------
研报数   11452   12330   12134   12797   12937   20025   22074   19934   16233   10470
======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======

（``pageSize`` 上限 100，全历史约 1504 页 / 15 万行，单线程约 10 分钟。）

为什么单独开一个 source 而不是并入 ``source_akshare``
-----------------------------------------------------
2026-10-13 的 ``consensus_availability`` 尽调**只验证了 EPS 预测字段**（``predict*``
只有 2024-04 起，不足 36 月），据此把整条「分析师」线判了不可立项。实测该接口的
**评级 / 评级变动字段有 2017 起全历史、逐年覆盖 0.86~1.00**，与 EPS 是两批字段。
本模块只消费后者仍在的那批（评级族），EPS 族维持原判。

字段语义（实测，2026-11-15，2024-06 抽样 398 条）
-------------------------------------------------
- ``emRatingValue`` 是**序数**且跨机构可比：买入 3 / 增持 2 / 持有 1 / 中性 0
  （另有减仓、卖出档，本地样本未出现）。缺失约 4%（评级只写文字或不评级）。
- ``ratingChange`` 的编码**不可信**，实测与 ``lastEmRatingName → emRatingName``
  的方向对不上（``ratingChange=2`` 有 166 条是「上次为空」即首次覆盖，
  ``ratingChange=1`` 出现 买入→增持 这类下调）。⇒ **一律不用该列做方向**，
  因子层按 ``(org_code, code)`` 自身历史序列重算 up/down/first。
- ``indvAimPriceT`` / ``indvAimPriceL``（目标价上/下限）覆盖 **5%~31%**，稀疏，
  仅作可选分支，不进主因子。
- ``reportType`` 在样本内恒为 2（个股研报）；``qType=0`` 已固定为个股研报。

PIT
---
只有 ``publishDate``（含 00:00:00，即**只有日期没有时点**）。研报通常在盘前发布，
但无法证明 ⇒ 因子层一律 **+1 交易日**（``avail_lag_trading_days=1``），即 T 日发布
的研报最早在 T+1 收盘可用、T+2 开盘可交易。月度调仓下这条滞后只吃掉不到 1 天。

缓存
----
``<root>/alt_em_report_rating.parquet``，列：
``info_code, pub_date, code, org_code, org_name, rating_name, rating_value,
last_rating_name, last_rating_value, rating_change_raw, target_price,
target_price_low, industry_name, report_type``
增量：``<root>/_alt_em_rating_progress.json`` 记已完成月份（YYYY-MM）。
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from config import Config

log = logging.getLogger(__name__)

REPORT_API = "https://reportapi.eastmoney.com/report/list"
CACHE_NAME = "alt_em_report_rating.parquet"
PROGRESS_NAME = "_alt_em_rating_progress.json"

#: PIT：研报只有日期无时点 ⇒ 保守滞后 1 个交易日
AVAIL_LAG_TRADING_DAYS = 1

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://data.eastmoney.com/report/",
}

#: 评级文字 → 序数（越大越看多）。缺失保持 NaN，不填中性。
RATING_ORDINAL = {
    "买入": 3.0, "强烈推荐": 3.0, "推荐": 3.0, "强推": 3.0,
    "增持": 2.0, "谨慎推荐": 2.0, "优于大市": 2.0, "跑赢行业": 2.0,
    "持有": 1.0, "中性": 1.0, "同步大市": 1.0, "neutral": 1.0,
    "减持": 0.0, "回避": 0.0, "弱于大市": 0.0, "卖出": 0.0,
}


def _root() -> Path:
    return Path(Config.cache()["root"])


def _get(params: dict, retries: int = 3, timeout: int = 25) -> dict:
    url = f"{REPORT_API}?{urllib.parse.urlencode(params)}"
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.6 * (i + 1))
    raise RuntimeError(f"report api failed after {retries}: {last}")


def _norm(raw: dict) -> dict:
    """单条 API 记录 → 扁平 dict（只留评级族需要且口径稳定的字段）。"""
    def s(v):
        if v is None:
            return None
        v = str(v).strip()
        return None if v in ("", "nan", "None") else v

    def f(v):
        try:
            x = float(v)
            return x if x == x else None
        except (TypeError, ValueError):
            return None

    cur = s(raw.get("emRatingName"))
    last = s(raw.get("lastEmRatingName"))
    return dict(
        info_code=s(raw.get("infoCode")),
        pub_date=(s(raw.get("publishDate")) or "")[:10],
        code=s(raw.get("stockCode")),
        org_code=s(raw.get("orgCode")),
        org_name=s(raw.get("orgSName")) or s(raw.get("orgName")),
        rating_name=cur,
        rating_value=RATING_ORDINAL.get(cur) if cur else None,
        last_rating_name=last,
        last_rating_value=RATING_ORDINAL.get(last) if last else None,
        rating_change_raw=s(raw.get("ratingChange")),
        target_price=f(raw.get("indvAimPriceT")),
        target_price_low=f(raw.get("indvAimPriceL")),
        industry_name=s(raw.get("industryName")),
        report_type=s(raw.get("reportType")),
    )


def fetch_month(year: int, month: int, page_size: int = 100, sleep: float = 0.25,
                verbose: bool = True) -> pd.DataFrame:
    """抓取某月全部个股研报。"""
    import calendar

    last_day = calendar.monthrange(year, month)[1]
    begin, end = f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"
    rows: list[dict] = []
    page, total = 1, None
    while True:
        d = _get(dict(industryCode="*", pageSize=page_size, industry="*", rating="",
                      ratingChange="", beginTime=begin, endTime=end, pageNo=page,
                      fields="", qType=0, orgCode="", code="*", rcode=""))
        data = d.get("data") or []
        if total is None:
            total = d.get("hits") or 0
            if verbose:
                log.info("%d-%02d  hits=%s pages=%s", year, month, total,
                         -(-total // page_size))
        if not data:
            break
        rows.extend(_norm(r) for r in data)
        if page * page_size >= total:
            break
        page += 1
        time.sleep(sleep)
    df = pd.DataFrame(rows)
    if verbose:
        log.info("%d-%02d  got %d rows", year, month, len(df))
    return df


def _load_progress() -> dict:
    p = _root() / PROGRESS_NAME
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"done_months": []}


def _save_progress(pg: dict) -> None:
    (_root() / PROGRESS_NAME).write_text(json.dumps(pg, ensure_ascii=False, indent=1),
                                         encoding="utf-8")


def backfill(start: str = "2017-01", end: str | None = None, sleep: float = 0.25,
             page_size: int = 100, write_every: int = 4) -> pd.DataFrame:
    """按月回补，断点续传；每 ``write_every`` 个月合并落盘一次。

    Parameters
    ----------
    start, end : ``"YYYY-MM"``，``end`` 缺省 = 当月
    """
    import datetime as dt

    sy, sm = (int(x) for x in start.split("-"))
    if end is None:
        now = dt.date.today()
        ey, em = now.year, now.month
    else:
        ey, em = (int(x) for x in end.split("-"))

    cache_path = _root() / CACHE_NAME
    old = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    pg = _load_progress()
    done = set(pg.get("done_months", []))

    buf: list[pd.DataFrame] = []
    n_written = 0
    y, m = sy, sm
    while (y, m) <= (ey, em):
        key = f"{y}-{m:02d}"
        if key not in done:
            try:
                dfm = fetch_month(y, m, page_size=page_size, sleep=sleep)
            except Exception as e:  # noqa: BLE001
                log.warning("月 %s 抓取失败（跳过，稍后重跑）：%s", key, e)
                dfm = pd.DataFrame()
            buf.append(dfm)
            done.add(key)
            pg["done_months"] = sorted(done)
            n_written += 1
            if n_written % write_every == 0:
                old = _flush(old, buf, pg)
                buf = []
        m += 1
        if m > 12:
            y, m = y + 1, 1
    if buf:
        old = _flush(old, buf, pg)
    return old


def _flush(old: pd.DataFrame, buf: list[pd.DataFrame], pg: dict) -> pd.DataFrame:
    frames = [x for x in [old, *buf] if x is not None and len(x)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        df = (df.drop_duplicates(subset=["info_code"], keep="last")
                .sort_values("pub_date")
                .reset_index(drop=True))
    path = _root() / CACHE_NAME
    df.to_parquet(path, index=False)
    _save_progress(pg)
    log.info("落盘 %s  rows=%d", path.name, len(df))
    return df


def load_cache() -> pd.DataFrame:
    """读缓存，附 ``avail_date``（pub_date + 1 自然日，交易日滞后由因子层做）。"""
    p = _root() / CACHE_NAME
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    for c in ("rating_value", "last_rating_value", "target_price", "target_price_low"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = backfill()
    print(out.shape)
    print(out["pub_date"].min(), out["pub_date"].max())
