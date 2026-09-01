"""
同花顺研报主源抓取器
====================

从同花顺 F10 研报页抓取分析师研报（标题 + 摘要全文 + 评级 + 机构 + 研究员）。

通道（2026-08-17 实测验证）：
    https://basic.10jqka.com.cn/{code6}/report.html
    页面内嵌 JSON（id="report_list_contents"），一次请求返回该股全部历史研报
    （实测 600519 共 1868 条，2011-08 ~ 2026-08；002594 共 1325 条）。

注意：
    - 同花顺页面体积较大（单只 5~10MB），JSON 是其中主体，仅解析 JSON 段。
    - 东财系域名（reportapi.eastmoney.com 等）在当前网络不可达，本通道绕开。
    - 免费接口无 SLA，抓取频率建议控制（本项目 HS300 全量约 300 次请求，
      实测连续请求未见限流，仍建议 sleep 0.3~0.5s）。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Iterable

import pandas as pd
import requests

log = logging.getLogger(__name__)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
BASE_URL = "https://basic.10jqka.com.cn/{code6}/report.html"
_REPORT_JSON_RE = re.compile(r'id="report_list_contents">(\[.*?\])</div>', re.S)


def to_code6(code: str) -> str:
    """标准代码（600519.SH / 000001.SZ / 430047.BJ）转 6 位纯数字。"""
    return str(code).split(".")[0].zfill(6)


def to_code_std(code6: str) -> str:
    """6 位代码补全为标准格式（按 A 股代码段推断交易所）。"""
    c = str(code6).zfill(6)
    if c[0] in ("6", "9", "5"):
        return f"{c}.SH"
    if c[0] in ("0", "2", "3"):
        return f"{c}.SZ"
    if c[0] in ("4", "8"):
        return f"{c}.BJ"
    return c


def _parse_report_json(page_text: str) -> list[dict]:
    """从同花顺研报页 HTML 提取内嵌 JSON（研报列表）。"""
    m = _REPORT_JSON_RE.search(page_text)
    if not m:
        return []
    return json.loads(m.group(1))


def fetch_ths_reports(
    code: str,
    timeout: float = 60.0,
    retries: int = 2,
    sleep_s: float = 0.4,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """抓取单只股票的全部历史研报（同花顺一次返回全量）。

    Parameters
    ----------
    code : str
        标准代码（600519.SH）或 6 位代码（600519），自动归一。
    timeout : float
        单次请求超时（秒）。页面较大，默认 60s。
    retries : int
        失败重试次数。
    sleep_s : float
        请求间隔（秒），降低被限流概率。
    session : requests.Session | None
        复用 Session（连接复用，批量抓取更快）。

    Returns
    -------
    pd.DataFrame
        列：code(标准)/date/title/summary(摘要全文)/rating(评级)/org(机构)/
        analyst(研究员)/source(=ths)/title_raw。
        无数据时返回空 DataFrame（保留列）。
    """
    code6 = to_code6(code)
    url = BASE_URL.format(code6=code6)
    headers = {"User-Agent": UA}
    s = session or requests.Session()

    last_err: Exception | None = None
    for i in range(retries + 1):
        try:
            r = s.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            items = _parse_report_json(r.text)
            if not items:
                # 页面可达但无研报数据：可能页面结构变化或股票无研报覆盖
                return pd.DataFrame(
                    columns=["code", "date", "title", "summary", "rating",
                             "org", "analyst", "source", "title_raw"])
            rows = []
            for it in items:
                rows.append({
                    "code": to_code_std(code6),
                    "date": it.get("date"),
                    "title": it.get("title"),
                    "summary": it.get("content"),
                    "rating": it.get("thspj"),
                    "org": it.get("source"),
                    "analyst": it.get("researcher"),
                    "source": "ths",
                    "title_raw": it.get("title"),
                })
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            if i < retries:
                time.sleep(sleep_s * (i + 1))
    raise RuntimeError(f"同花顺研报抓取失败 {code6}: {last_err}") from last_err


def fetch_ths_reports_batch(
    codes: Iterable[str],
    timeout: float = 60.0,
    retries: int = 2,
    sleep_s: float = 0.4,
    max_codes: int | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """批量抓取多只股票的研报，合并为一个 DataFrame。

    Parameters
    ----------
    codes : Iterable[str]
        标准代码列表。
    max_codes : int | None
        最多抓取 N 只（调试用），None 为全量。
    progress : bool
        是否打印进度（tqdm 不可用时退化 print）。
    """
    codes = list(codes)
    if max_codes is not None:
        codes = codes[:max_codes]
    frames: list[pd.DataFrame] = []
    with requests.Session() as s:
        for i, c in enumerate(codes):
            try:
                df = fetch_ths_reports(c, timeout=timeout, retries=retries,
                                       sleep_s=sleep_s, session=s)
                frames.append(df)
            except Exception as e:  # noqa: BLE001
                log.warning("[ths] %s 失败: %s: %s", c, type(e).__name__, str(e)[:80])
            if progress and (i + 1) % 10 == 0:
                log.info("[ths] %d/%d", i + 1, len(codes))
    if not frames:
        return pd.DataFrame(columns=["code", "date", "title", "summary", "rating",
                                     "org", "analyst", "source", "title_raw"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["code", "date"]).reset_index(drop=True)
