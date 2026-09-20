"""另类数据 L1 传输层：新闻快讯（**快照型冗余源**）
=======================================================

🚨 **2026-09-20 二次修正：新闻并非「全都无历史」——本模块已降级为冗余备份。**
---------------------------------------------------------------------
上一轮在本模块写下「所有免费快讯源都只给最新快照、无历史可回补」。
**该结论对 akshare 封装成立，但对财联社原始端点不成立**：
``https://www.cls.cn/v1/roll/get_roll_list`` 支持 ``last_time`` 往回翻页，
**可回补到 ≥2015**，且原始响应自带 ``level``（加红等级）与 ``stock_list``
（关联个股）—— akshare 的 ``stock_info_global_cls`` 把这两样都丢了或写死了。

⇒ **新闻研究请用** :mod:`data.altdata.source_cls` **（主表，可回补）**；
本模块（同花顺 / 新浪快照）只用于：

1. 财联社端点变更导致主表抓取失败时的**旁证与兜底**；
2. 交叉验证（同一条快讯的两源发布时间差）。

⚠️ **快照深度很小（20 条/源）** —— 一天抓一次只能覆盖最近几十分钟的快讯。
所以本源的存档价值有限：**丢掉的时段拿不回来**。若仍要留档，须高频抓取
（``scripts/pipelines/fetch_altdata_daily.py`` 的 ``--loop`` / ``--install-news-task``）。

⚠️ **本模块实测状态（2026-09-20 复核，与上一轮结论相同部分）**
------------------------------------------------------------
======================  =============================  ==========================
接口                     本机实测                        结论
======================  =============================  ==========================
``stock_info_global_em``   ❌ SSLError (np-weblist)       **本机不可用**
``stock_info_global_ths``  ✅ 20 行 × 4 列，秒级时间戳       **可用**
``stock_info_global_sina`` ✅ 20 行 × 2 列，秒级时间戳       **可用**
``stock_info_global_cls``  ✅ 20 行 × 4 列（**丢 level**）   见 source_cls（直调更好）
======================  =============================  ==========================

统一输出 schema
---------------
``pub_time``  发布时间（datetime64，精确到秒）—— 存档的时间轴
``title``     标题（新浪源从内容首部 ``【】`` 提取；提取不到则 None）
``content``   正文
``url``       原文链接（新浪源不提供 ⇒ None）
``source``    来源标记（``ths`` / ``sina`` / ``em``）
"""
from __future__ import annotations

import logging
import re

import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["NEWS_COLS", "NEWS_SOURCES", "fetch_news", "fetch_news_all"]

NEWS_COLS = ["pub_time", "title", "content", "url", "source"]

#: 按优先级排列的可用源（2026-09-20 实测：em 本机不可用，保留以适配其他网络环境）
NEWS_SOURCES: tuple[str, ...] = ("ths", "sina", "em")

_TITLE_RE = re.compile(r"^【(.+?)】\s*")


def _ak():
    try:
        import akshare as ak
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "新闻存档依赖 akshare。安装：uv pip install -e \".[altdata]\" "
            "--python <repo>/.venv/Scripts/python.exe"
        ) from e
    return ak


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=NEWS_COLS)


def _split_title(content, title_col=None):
    """返回 (title, content)。有独立标题列则用；否则从 ``【...】`` 提取。"""
    if title_col is not None:
        t = str(title_col).strip() if pd.notna(title_col) else None
        return (t or None), (str(content) if pd.notna(content) else None)
    if pd.isna(content):
        return None, None
    s = str(content)
    m = _TITLE_RE.match(s)
    if m:
        return m.group(1).strip(), s[m.end():].strip() or None
    return None, s or None


def fetch_news(source: str = "ths") -> pd.DataFrame:
    """抓取单个源的当前快照（无历史，只能存）。

    Parameters
    ----------
    source : str
        ``ths``（同花顺全球财经直播）/ ``sina``（新浪全球财经快讯）/
        ``em``（东方财富全球快讯，**本机代理环境下不可达**）。

    Returns
    -------
    pd.DataFrame
        列为 :data:`NEWS_COLS`；失败返回空表（**不抛异常** —— 存档脚本要能
        在单源挂掉时继续抓其他源，但调用方应检查 ``source`` 列是否为空并告警）。
    """
    ak = _ak()
    fn_name = {"ths": "stock_info_global_ths", "sina": "stock_info_global_sina",
               "em": "stock_info_global_em"}.get(source)
    if fn_name is None:
        raise ValueError(f"未知新闻源 {source!r}；可用：{list(NEWS_SOURCES)}")
    fn = getattr(ak, fn_name, None)
    if fn is None:
        log.warning("[news] akshare 无接口 %s", fn_name)
        return _empty()
    try:
        raw = fn()
    except Exception as e:  # noqa: BLE001
        log.warning("[news] %s(%s) 失败: %s: %s", source, fn_name, type(e).__name__, str(e)[:100])
        return _empty()
    if raw is None or raw.empty:
        return _empty()

    if source == "ths":
        rows = [_split_title(c, t) for c, t in zip(raw.get("内容"), raw.get("标题"))]
        out = pd.DataFrame({
            "pub_time": raw.get("发布时间"),
            "title": [r[0] for r in rows],
            "content": [r[1] for r in rows],
            "url": raw.get("链接"),
        })
    elif source == "sina":
        rows = [_split_title(c) for c in raw.get("内容")]
        out = pd.DataFrame({
            "pub_time": raw.get("时间"),
            "title": [r[0] for r in rows],
            "content": [r[1] for r in rows],
            "url": None,                      # 新浪快讯不提供原文链接
        })
    else:  # em
        rows = [_split_title(c, t) for c, t in zip(raw.get("摘要"), raw.get("标题"))]
        out = pd.DataFrame({
            "pub_time": raw.get("发布时间"),
            "title": [r[0] for r in rows],
            "content": [r[1] for r in rows],
            "url": raw.get("链接"),
        })

    out["pub_time"] = pd.to_datetime(out["pub_time"], errors="coerce")
    out["source"] = source
    out = out.dropna(subset=["pub_time"])
    return out[NEWS_COLS].reset_index(drop=True)


def dedup_key(df: pd.DataFrame) -> pd.Series:
    """新闻幂等键：``(pub_time, title, content前80字)``。

    存档层与抓取层共用同一函数 —— 「重跑不重复」的判据必须只有一处定义，
    否则两层各写一份去重规则时会出现「抓取去了重、落盘又塞回重复」的错位。
    """
    return (df["pub_time"].astype(str) + "|" + df["title"].fillna("") + "|"
            + df["content"].fillna("").str.slice(0, 80))


def fetch_news_all(sources: tuple[str, ...] = NEWS_SOURCES) -> pd.DataFrame:
    """抓取全部可用源并合并去重（任一源失败不影响其他源）。"""
    frames = []
    for s in sources:
        df = fetch_news(s)
        if df.empty:
            log.warning("[news] 源 %s 无数据", s)
        else:
            frames.append(df)
    if not frames:
        return _empty()
    out = pd.concat(frames, ignore_index=True)
    out["_key"] = dedup_key(out)
    out = out.drop_duplicates(subset="_key", keep="first").drop(columns="_key")
    return out.sort_values("pub_time").reset_index(drop=True)
