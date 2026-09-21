"""另类数据 L1 传输层：财联社电报（**可回补历史**的新闻源）
=================================================================

🚨 2026-09-20 实测发现（**推翻本项目此前「新闻无历史可回补」的全部结论**）
-------------------------------------------------------------------
本项目原判断（``reports/docs/design/另类数据P0_实施计划.md`` §0.5 / §5.1）：
「所有免费快讯源都只给最新快照、无历史 ⇒ 新闻只能高频轮询存档，丢掉的时段拿不回来」。

**该结论对 akshare 封装成立，对原始端点不成立。** 实测（快照
``E:/_YuriQuant_snapshots/_tmp_20260921_0230/probe_cls_*.py``）：

============================  ==========================================  ==========
事项                           实测结果                                     影响
============================  ==========================================  ==========
历史深度                       **≥ 2015-06 可回补**（2013 起返回空）        **免攒一年**
``last_time`` 翻页             传 2018-01-10 真返回该日 09:31–11:59 电报     **可往回翻**
单日密度                       ≈ 580 条 / 日（2026-09-15 实测）            全量 ≈ 240 万条
``rn`` 上限                    **50**（50 可用，100 返回 0；akshare 写死 20） 请求数减半
``level`` 字段                 **A/B/C 加红等级，原始响应里有**            白嫖编辑重要度
``stock_list``                 **关联个股（代码 + 名称 + 当日涨幅）**        **个股映射免 NLP**
``subjects``                   主题/板块名（如「原油市场动态」「天然气」）   行业映射免 NLP
============================  ==========================================  ==========

⚠️ akshare 的 ``stock_info_global_cls`` **为什么没暴露这些**：
它把 ``last_time`` 写死成 ``int(time.time())``、``rn`` 写死成 20，并且在
``symbol="全部"`` 分支**把 ``level`` 列删掉了**（源码只返回 标题/内容/发布日期/发布时间）。
⇒ 要用上「历史 + 等级 + 个股」必须**直调原始端点**，代价只是一段 ``sign`` 计算（≈6 行）。

⚠️ akshare 封装**仍可用作「封装是否跟上」的哨兵**：若哪天它报错，说明端点变了，
本模块的 sign 逻辑也需要同步核对。

统一输出 schema（``CLS_COLS``）
-----------------------------
``news_id``      电报 id（**唯一主键**，断点续传与幂等都靠它）
``ctime``        发布时刻（**int unix 秒**，接口原值）—— 见下方"时区铁律"
``pub_time``     发布时间（datetime64，由 ``ctime`` 按 Asia/Shanghai 转，**无时区**）
``title``        标题（接口空标题时从正文 ``【】`` 前缀提取）
``content``      正文（已剥离 ``【标题】`` 前缀）
``level``        加红等级 ``A``/``B``/``C``（A 最高）
``stock_codes``  关联个股代码（逗号分隔，项目格式 ``600975.SH``）
``stock_names``  关联个股名称（逗号分隔，与 codes 同序）
``stock_rises``  关联个股当日涨幅（逗号分隔）
``subjects``     主题/板块名（逗号分隔）
``reading_num``  阅读数（**跨年口径不稳**，见下方"字段可用性年份边界"）
``share_num``    转发数
``comment_num``  评论数
``sort_score``   ⚠️ **不是排序分**：接口原值**恒等于 ``ctime``**（2026-09-20 全样本
                 实测 119.2 万行 **100.00% 逐位相同**，``sort_score − ctime`` 分布
                 全 0）。它是接口内部用于排序的时间戳副本，**纯冗余列，勿当因子**。
``url``          原文链接（``https://www.cls.cn/detail/<id>``）
``source``       固定 ``"cls"``

🚨 字段可用性年份边界（2026-09-20 全量 119.2 万条实测，别再假设"回补到 2015
就 2015 能用"）
---------------------------------------------------------------------
回补范围是 2014-12-31~今（4281 个日分片、逐年 364~366 天无断档），但**各字段的
真实可用起点差很多**：

============  ================  ==========  ==============  ==============
字段          2015–2017         2018        2019+           判断
============  ================  ==========  ==============  ==============
title/content ✅                ✅          ✅              全期可用
stock_codes   ❌ 0%             ❌ 0.6%      ✅ 16–29%       **2019 起**
subjects      ❌ 0%             ⚠️ 36.7%     ✅ 63–87%       **2018 起**
level(A/B)    ⚠️ 2015 全 C      ⚠️          ⚠️ A 类递减     弱，勿单独用
reading_num   ⚠️ 口径漂移        ⚠️          ⚠️              跨年不可比
comment_num   ⚠️ 25–59% 不稳      ⚠️          ⚠️              弱
============  ================  ==========  ==============  ==============

⇒ **要造「个股关联」类因子（依赖 ``stock_codes``），样本起点 = 2019，
且每年只有 1.8–3.3 万条带个股标注的电报**（占全年 ~20%）。2015–2018 的数据
只能支撑**市场级**用途（文本词频 / 主题 / 情绪），**不能关联到个股**。

⇒ ``reading_num`` 分年中位（非零）2015 年 174 万、2019 年 476 万、2025 年 278 万，
**非单调且 2015 年 max 反而最高（2180 万）** ⇒ 接口统计口径变过，
**跨年直接串联做因子会造出假信号**；要用须逐年 rank 标准化。

🚨 时区铁律（2026-09-20 实测踩坑，别再踩）
----------------------------------------
**游标 / 水位 / 去重时间戳一律只用 ``ctime``（int unix 秒），
绝不用 ``pub_time`` 的 ``.timestamp()`` 反推。**

原因：``pub_time`` 是**已转成北京时间的 naive datetime**，而 pandas 对 naive
Timestamp 调 ``.timestamp()`` 时**按 UTC 解释** ⇒ 反推出的 epoch 比真实值**多 8 小时**。
把它当游标喂回 ``last_time``，每翻一页游标就凭空前进 8 小时，
**游标单调递增、永远越不过终点 → 死循环**（实测把 2 天的回补跑成挂死）。

⇒ 本模块**同时输出 ``ctime`` 与 ``pub_time``**：``ctime`` 供机器算游标，
``pub_time`` 供人读和落盘。两者只允许「``ctime`` → ``pub_time``」单向转换。
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from urllib.parse import urlencode

import pandas as pd
import requests

log = logging.getLogger(__name__)

__all__ = [
    "CLS_COLS", "CLS_URL", "CLS_MAX_RN", "CLS_EARLIEST_DATE",
    "fetch_cls_page", "parse_cls_items", "normalize_stock_id",
]

CLS_URL = "https://www.cls.cn/v1/roll/get_roll_list"

CLS_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Referer": "https://www.cls.cn/telegraph",
    "Accept": "application/json, text/plain, */*",
}

#: 单页上限（实测 50 可用、100 返回 0 条；akshare 写死 20 属保守）
CLS_MAX_RN = 50

#: 实测可回补的最早日期（YYYYMMDD）。2015-06-15 有数据、2013-06-15 返回空。
#: 这是**保守下界**：真实边界在 2013–2015 之间，回补时以「翻到空页」自然终止。
CLS_EARLIEST_DATE = 20150101

CLS_COLS = [
    "news_id", "ctime", "pub_time", "title", "content", "level",
    "stock_codes", "stock_names", "stock_rises", "subjects",
    "reading_num", "share_num", "comment_num", "sort_score",
    "url", "source",
]

_TITLE_RE = re.compile(r"^【(.+?)】\s*")
_MARKET_SUFFIX = {"sh": "SH", "sz": "SZ", "bj": "BJ"}


# ---------------------------------------------------------------------------
# 请求
# ---------------------------------------------------------------------------
def _sign(params: dict) -> str:
    """复刻财联社网页端的签名：``md5(sha1(urlencode(params)))``。

    ⚠️ ``urlencode`` 的**键序必须与 params 插入序一致** —— 签名是对序列化串做的，
    调换键序会得到不同 sign，接口返回空。这里保持与 akshare 实现相同的插入顺序。
    """
    return hashlib.md5(
        hashlib.sha1(urlencode(params).encode("utf-8")).hexdigest().encode("utf-8")
    ).hexdigest()


def fetch_cls_page(
    last_time: int | float,
    rn: int = CLS_MAX_RN,
    *,
    session: requests.Session | None = None,
    timeout: int = 20,
    max_retries: int = 3,
) -> pd.DataFrame:
    """抓**一页**电报（20/50 条），返回 ``last_time`` 时点之前的最近 ``rn`` 条。

    Parameters
    ----------
    last_time : int | float
        unix 秒。返回 **``ctime`` ≤ ``last_time``** 的最近 ``rn`` 条 ⇒
        **往后翻历史 = 传上一页最老一条的 ctime − 1**。
    rn : int
        单页条数，上限 :data:`CLS_MAX_RN`。
    session : requests.Session | None
        复用连接（回补要发几万次请求，复用 session 避免每页重建 TLS）。
    timeout / max_retries : int
        网络容错。失败重试仍失败则**抛异常**（调用方决定是否中断/跳过），
        不返回空表 —— 因为「空表」在本接口语义里 = **已翻到尽头**，
        把网络错误当成尽头会**静默截断历史**。

    Returns
    -------
    pd.DataFrame
        ``CLS_COLS``；**空表 = 该时点之前没有数据（翻到尽头）**。
    """
    rn = int(min(rn, CLS_MAX_RN))
    params = {
        "app": "CailianpressWeb",
        "category": "",
        "last_time": int(last_time),
        "os": "web",
        "refresh_type": "1",
        "rn": str(rn),
        "sv": "8.4.6",
    }
    params["sign"] = _sign(params)

    getter = session.get if session is not None else requests.get
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = getter(CLS_URL, params=params, headers=CLS_HEADERS, timeout=timeout)
            r.raise_for_status()
            js = r.json()
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    else:  # pragma: no cover - 网络异常路径
        raise RuntimeError(f"cls roll_list 请求失败（{max_retries} 次）：{last_err}")

    roll = (js.get("data") or {}).get("roll_data") or []
    if not roll:
        return pd.DataFrame(columns=CLS_COLS)
    return parse_cls_items(roll)


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def normalize_stock_id(stock_id: str) -> str | None:
    """``"sh600975"`` → ``"600975.SH"``；``"sz300793"`` → ``"300793.SZ"``。

    财联社用**小写前缀式**（``sh``/``sz``/``bj``），与项目标准格式相反。
    非 A 股（``hk``/``us`` 等）或无法解析的返回原值（保留信息，不硬塞市场后缀）。
    """
    if not isinstance(stock_id, str) or not stock_id:
        return None
    s = stock_id.strip()
    pfx, body = s[:2].lower(), s[2:]
    if pfx in _MARKET_SUFFIX and body.isdigit() and len(body) == 6:
        return f"{body}.{_MARKET_SUFFIX[pfx]}"
    return s


def _join(items, key: str) -> str | None:
    """把 list[dict] 的某个键拼成逗号串；空 ⇒ None（alt 惯例：无事件不填 ""）。"""
    if not isinstance(items, list) or not items:
        return None
    vals = [str(it.get(key)).strip() for it in items
            if isinstance(it, dict) and it.get(key) not in (None, "")]
    return ",".join(vals) if vals else None


def _split_title(title, content) -> tuple[str | None, str | None]:
    """标题：优先接口 ``title`` 列；为空则从正文 ``【...】`` 提取。"""
    t = str(title).strip() if pd.notna(title) else ""
    c = str(content) if pd.notna(content) else ""
    body = c
    m = _TITLE_RE.match(c)
    if m:
        body = c[m.end():].strip()
        if not t:
            t = m.group(1).strip()
    return (t or None), (body or None)


def parse_cls_items(roll: list[dict]) -> pd.DataFrame:
    """把 ``roll_data`` 原始 list 规范成 :data:`CLS_COLS`。

    个股/主题字段**只做扁平化不做过滤** —— 用不用是使用层的事，
    存档层管的是「原文与原始标注不丢」（计划 §5.1 ① 全量落盘原则）。
    """
    rows = []
    for it in roll:
        if not isinstance(it, dict):
            continue
        nid = it.get("id")
        if nid is None:
            continue
        t, c = _split_title(it.get("title"), it.get("content"))
        stocks = it.get("stock_list")
        subj = it.get("subjects")
        ct = it.get("ctime")
        rows.append({
            "news_id": int(nid),
            "ctime": int(ct) if ct is not None else None,
            "pub_time": ct,
            "title": t,
            "content": c,
            "level": it.get("level"),
            "stock_codes": _join([{"c": normalize_stock_id(s.get("StockID"))}
                                  for s in stocks or [] if isinstance(s, dict)], "c"),
            "stock_names": _join(stocks, "name"),
            "stock_rises": _join(stocks, "RiseRange"),
            "subjects": _join(subj, "subject_name"),
            "reading_num": it.get("reading_num"),
            "share_num": it.get("share_num"),
            "comment_num": it.get("comment_num"),
            "sort_score": it.get("sort_score"),
            "url": f"https://www.cls.cn/detail/{int(nid)}",
            "source": "cls",
        })
    if not rows:
        return pd.DataFrame(columns=CLS_COLS)
    out = pd.DataFrame(rows)
    # 🚨 ctime 是 unix 秒（绝对时刻）⇒ 必须 ``unit="s", utc=True`` 再转上海时区，
    # **不要用本机时区**：换台机器/换时区跑会把整条时间轴平移。
    # 同时保留原始 ``ctime`` 供游标运算（见模块头「时区铁律」）。
    out["pub_time"] = pd.to_datetime(
        out["pub_time"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    out = out.dropna(subset=["ctime", "pub_time"])
    # 按 ctime（unix 秒）排序，不用 pub_time —— 两者单调性一致，但 ctime 是原值
    return out[CLS_COLS].sort_values("ctime").reset_index(drop=True)
