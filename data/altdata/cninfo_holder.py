"""另类数据 L1 传输层：巨潮「高管持股变动明细」全历史（P0 Step 1b）
=====================================================================

**为什么必须自建**：akshare 的 ``stock_hold_management_detail_cninfo`` 签名里
**没有日期参数** —— ``sdate``/``edate`` 写死在函数体内（``去年前今天 → 今天``），
所以免自建路径只能拿 **1 年滚动**。要 2010 起的全历史，只能直调
``p_sysapi1030``（本模块）。

与 akshare 实现的**三处关键差异**（每一处都是踩过的坑）
------------------------------------------------------
1. 🚨 **按 key 映射，不按位置映射。**
   响应 ``records`` 是 ``list[dict]``，键是 ``SECNAME`` / ``DECLAREDATE`` /
   ``F006N`` 这类**无语文缩码**。akshare 用
   ``pd.DataFrame(records)`` 之后 ``temp_df.columns = [16 个中文名]`` ——
   这依赖「服务端返回的键序恒定」。**键序一变，列名就整体错位，且不抛异常**。
   本模块显式按 key 取值，键序无关。
   （实测键：SECNAME DECLAREDATE HUMANNAME F009N F008N SECCODE F007N F006N
   ENDDATE F005N F004N F003V F002V F001V F011V F010V）

2. 🚨 **服务端单请求硬上限 20000 行，且截断不报错。**
   实测（快照 ``E:/_YuriQuant_snapshots/_tmp_20260921_0230/probe_cninfo1030_width.txt``）：

   ==========  ==========  =============  ==========================
   请求区间      总量 total   实际 records    结论
   ==========  ==========  =============  ==========================
   1 年          4 571       4 571          ✅ 完整
   2 年         12 283      12 283          ✅ 完整
   3 年         29 023      **20 000**      ❌ **被截断**
   8 年         95 458      **20 000**      ❌ 被截断
   ==========  ==========  =============  ==========================

   ⇒ **判据 = ``total > len(records)``**。本模块遇截断**自动二分区间递归**，
   绝不静默吞掉半段历史。

3. ⚠️ **区间过滤口径反直觉：加宽窗口是窄窗口的超集（实测 1年 − 2年 = 0），
   但切窄会丢行。** 实测 2019 分季取并集只有 3 450 条，而 2019 全年一次取有
   6 338 条 —— 少掉的 2 886 条正是「变动在年内、公告日推后」的行
   （2019 全年结果的 ``DECLAREDATE`` 一直延伸到 2021-06-28）。
   ⇒ 驱动层**用「带重叠的 2 年年段」（stride=1）而不是年/季切分**，
   见 :func:`iter_overlap_windows`。

⚠️ **PIT 红线**：``公告日期``（DECLAREDATE）是数据可得时点，
``截止日期``（ENDDATE）是变动发生时点。**因子对齐必须用 ``公告日期``**。

输出 schema（``CNINFO_HOLDER_COLS``）
------------------------------------
``code`` / ``sec_name``     证券代码（项目格式 ``000001.SZ``）/ 简称
``ann_date``                **公告日期** → int YYYYMMDD（PIT 对齐轴）
``chg_date``                截止日期（变动日）→ int YYYYMMDD
``direction``               ``"增持"`` / ``"减持"``
``subject_type``            主体类型（本接口恒为 ``"董监高"``）
``signed_qty``              **带符号股数**（增持 >0 / 减持 <0）—— 因子直接用这列
``chg_ratio``               变动比例（小数，如 0.0001 = 0.01%）
``price``                   成交均价
``amount``                  成交金额 = ``signed_qty * price``（price 缺失时为 NaN）
``qty_begin`` / ``qty_end`` 期初 / 期末持股数量
``mv_end``                  期末市值（实测 2019 年 0% 非空 ⇒ 基本不可用，保留占位）
``person``                  变动人（接口 HUMANNAME）
``related_person``          关联董监高姓名（接口 F001V）
``relation``                变动人与董监高关系（如「配偶」「兄弟姐妹」）
``duty``                    董监高职务
``data_source`` / ``reason`` 数据来源（临时公告/定期报告）/ 持股变动原因
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)

__all__ = [
    "CNINFO_URL", "SERVER_MAX_ROWS", "CNINFO_HOLDER_COLS",
    "fetch_holder_trades", "iter_overlap_windows", "get_enckey", "reset_session",
    "year_coverage", "assert_no_empty_years",
]

CNINFO_URL = "https://webapi.cninfo.com.cn/api/sysapi/p_sysapi1030"

#: 服务端单请求硬上限（实测：超过则 records 被截到 20000 且**不报错**）
SERVER_MAX_ROWS = 20000

#: 方向 → 接口 ``varytype``
VARYTYPE = {"增持": "B", "减持": "S"}

#: 无语文缩码 → 项目列名（**按 key 映射**，见模块头坑 1）
_KEYMAP = {
    "SECCODE": "code_raw",
    "SECNAME": "sec_name",
    "DECLAREDATE": "ann_date",
    "ENDDATE": "chg_date",
    "HUMANNAME": "person",
    "F001V": "related_person",
    "F002V": "duty",
    "F003V": "relation",
    "F004N": "qty_begin",
    "F005N": "qty_end",
    "F006N": "chg_qty_raw",
    "F007N": "chg_ratio",
    "F008N": "price",
    "F009N": "mv_end",
    "F010V": "reason",
    "F011V": "data_source",
}

CNINFO_HOLDER_COLS = [
    "code", "sec_name", "ann_date", "chg_date", "direction", "subject_type",
    "signed_qty", "chg_qty", "chg_ratio", "price", "amount",
    "qty_begin", "qty_end", "mv_end",
    "person", "related_person", "relation", "duty",
    "data_source", "reason",
]

_HEADERS_BASE = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Content-Length": "0",
    "Host": "webapi.cninfo.com.cn",
    "Origin": "https://webapi.cninfo.com.cn",
    "Pragma": "no-cache",
    "Proxy-Connection": "keep-alive",
    "Referer": "https://webapi.cninfo.com.cn/",
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"),
    "X-Requested-With": "XMLHttpRequest",
}

_ENckey: str | None = None


def get_enckey(force: bool = False) -> str:
    """生成 ``Accept-Enckey``。

    巨潮要求请求头带一个由 ``cninfo.js`` 算出的校验码。**复用 akshare 打包的
    ``cninfo.js`` + ``py_mini_racer``**，不自行 vendored 一份 —— 校验算法在
    服务端，副本迟早过期，跟着上游走比自己维护更可靠。

    ⚠️ mini-racer 初始化（跑一遍 JS）约 **0.5–1 s**，比一次 HTTP 还贵 ⇒
    模块级缓存；回补要发几十次请求，绝不能每次重算。
    """
    global _ENckey
    if _ENckey is None or force:
        import py_mini_racer
        from akshare.datasets import get_ths_js

        js = py_mini_racer.MiniRacer()
        with open(get_ths_js("cninfo.js"), encoding="utf-8") as f:
            js.eval(f.read())
        _ENckey = js.call("getResCode1")
    return _ENckey


def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS_BASE)
    return s


#: 模块级共享连接池。为什么不做成「调用方持有并传参」：
#: 递归树 + 窗口循环横跨多层，一旦某层遇到 RST 需要换连接，
#: 「换掉的那个 session」传不回上层 —— 上层会继续拿坏连接重试，
#: 表现为「偶发但反复」的失败。模块级单例 + 出错即整体重置，语义最简单。
_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _new_session()
    return _SESSION


def reset_session() -> None:
    """丢弃并重建共享连接池（连接级异常后调用）。"""
    global _SESSION
    if _SESSION is not None:
        try:
            _SESSION.close()
        except Exception:  # noqa: BLE001, S110
            pass
    _SESSION = None


def _post(sdate: str, edate: str, direction: str, timeout: int = 90,
          max_retries: int = 5, retry_sleep: float = 1.0) -> tuple[list, int]:
    """发一次请求，返回 ``(records, total)``。``total`` 是服务端认定的总量。

    ⚠️ **必须复用连接池**：本机代理环境下每次新建连接都会被
    ``ConnectionResetError(10054, '远程主机强迫关闭了现有的连接')`` 打断
    （实测：连发十几条后开始 RST，整轮回补因此中断）。复用 + 指数退避后稳定。
    """
    params = {"sdate": str(sdate), "edate": str(edate), "varytype": VARYTYPE[direction]}
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            headers = dict(_HEADERS_BASE, **{"Accept-Enckey": get_enckey()})
            r = _session().post(CNINFO_URL, headers=headers, params=params, timeout=timeout)
            if r.status_code >= 500:
                # 实测：区间过宽（如 2010-2026 一次拉）服务端直接 500
                raise RuntimeError(f"HTTP {r.status_code}（区间过宽？）")
            r.raise_for_status()
            j = r.json()
            return (j.get("records") or []), int(j.get("total") or 0)
        except Exception as e:  # noqa: BLE001
            last_err = e
            text = f"{type(e).__name__}: {e}"
            if "500" in text:
                get_enckey(force=True)          # enckey 过期也会 500，换一个再试
            if isinstance(e, (requests.ConnectionError, requests.Timeout)) or "500" in text:
                reset_session()                 # 连接池里的坏连接会被反复取出
            time.sleep(retry_sleep * (2 ** attempt))
    raise RuntimeError(f"p_sysapi1030 请求失败（{sdate}~{edate} {direction}）：{last_err}")


def _to_int_date(v) -> int | None:
    """``"2019-04-03"`` → ``20190403``（int）；非法/缺失 → None。"""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s or s.lower() in ("none", "nat", "nan"):
        return None
    s = s.replace("-", "").replace("/", "")[:8]
    return int(s) if s.isdigit() and len(s) == 8 else None


def _normalize(records: list[dict], direction: str) -> pd.DataFrame:
    """``records``（list[dict]，无语文缩码）→ :data:`CNINFO_HOLDER_COLS`。

    按 key 映射（不按位置）；``signed_qty = |chg_qty| * sign(direction)`` —— 
    与 ``source_akshare.fetch_mgmt_hold_change`` **同一符号约定**，
    这样两个源可以互换进同一套因子而不出现「一正一负互相抵消」。
    """
    if not records:
        return pd.DataFrame(columns=CNINFO_HOLDER_COLS)
    raw = pd.DataFrame([{_KEYMAP[k]: r.get(k) for k in _KEYMAP} for r in records])

    # 代码标准化：复用 source_akshare.std_code6（含北交所 .BJ 段修正）
    from data.altdata.source_akshare import std_code6

    out = pd.DataFrame()
    out["code"] = raw["code_raw"].map(
        lambda c: std_code6(str(c).strip()) if pd.notna(c) else None)
    out["sec_name"] = raw["sec_name"]
    out["ann_date"] = raw["ann_date"].map(_to_int_date)
    out["chg_date"] = raw["chg_date"].map(_to_int_date)
    out["direction"] = direction
    out["subject_type"] = "董监高"

    qty = pd.to_numeric(raw["chg_qty_raw"], errors="coerce")
    sign = 1.0 if direction == "增持" else -1.0
    out["signed_qty"] = qty.abs() * sign
    out["chg_qty"] = qty                            # 接口原值，保留以便对账
    out["chg_ratio"] = pd.to_numeric(raw["chg_ratio"], errors="coerce")
    out["price"] = pd.to_numeric(raw["price"], errors="coerce")
    out["amount"] = out["signed_qty"] * out["price"]
    out["qty_begin"] = pd.to_numeric(raw["qty_begin"], errors="coerce")
    out["qty_end"] = pd.to_numeric(raw["qty_end"], errors="coerce")
    out["mv_end"] = pd.to_numeric(raw["mv_end"], errors="coerce")
    out["person"] = raw["person"]
    out["related_person"] = raw["related_person"]
    out["relation"] = raw["relation"]
    out["duty"] = raw["duty"]
    out["data_source"] = raw["data_source"]
    out["reason"] = raw["reason"]

    # 无公告日的行不能用于 PIT 对齐 ⇒ 丢弃（同时把「丢了多少」记进日志）
    n0 = len(out)
    out = out.dropna(subset=["ann_date"])
    if len(out) < n0:
        log.warning("[cninfo_holder] %s 丢弃 %d 行无公告日期（无法 PIT 对齐）",
                    direction, n0 - len(out))
    return out[CNINFO_HOLDER_COLS].reset_index(drop=True)


def _fmt_date(ts: pd.Timestamp) -> int:
    return int(ts.strftime("%Y%m%d"))


def _parse_date(v: int | str) -> pd.Timestamp:
    """``20260915`` / ``"2026-09-15"`` → ``Timestamp``（绕开 int→1970 陷阱）。"""
    return pd.to_datetime(str(v).replace("-", "")[:8], format="%Y%m%d")


def fetch_holder_trades(
    begin: int | str,
    end: int | str,
    direction: str = "增持",
    *,
    timeout: int = 90,
    sleep: float = 0.4,
    _depth: int = 0,
    _max_depth: int = 8,
) -> pd.DataFrame:
    """取 ``[begin, end]`` 区间的高管增减持明细，**撞服务端上限时自动二分递归**。

    这是本模块存在的理由：把「服务端 20000 行静默截断」变成
    「自动切分 + 完整拼接」，调用方拿到的永远是全量。

    🚨 **二分必须切在真实日历日上**（2026-09-20 实测教训）：
    最初按 ``YYYYMMDD`` 的**整数中点**切，得到 ``20170666`` / ``20140383``
    这类**非法日期**。服务端不报错、照常返回数据，但过滤语义已被破坏 ——
    实测 2010–2026 减持回补后 **2010 / 2011 / 2013 三个整年凭空消失**
    （拿到 130 456 行，而服务端自报 total 153 312）。
    换成日历日中点后年份覆盖完整。

    Parameters
    ----------
    begin / end : int | str
        ``YYYYMMDD``。
    direction : str
        ``"增持"`` 或 ``"减持"``。
    sleep : float
        每次请求后的停顿（秒）。回补要发上百次请求，不给间隔会被服务端 RST。
    _depth : int
        递归深度（内部用）；超过 ``_max_depth`` 抛异常 ——
        宁可炸掉也不返回「疑似不完整」的数据。

    Examples
    --------
    >>> df = fetch_holder_trades(20100101, 20261231, "增持")   # doctest: +SKIP
    >>> df["signed_qty"].gt(0).all()                          # doctest: +SKIP
    True
    """
    if direction not in VARYTYPE:
        raise ValueError(f"direction 必须是 {list(VARYTYPE)}，收到 {direction!r}")
    return _fetch_rec(begin, end, direction, timeout=timeout,
                      sleep=sleep, depth=_depth, max_depth=_max_depth)

def _fetch_rec(begin, end, direction: str, *, timeout: int,
               sleep: float, depth: int, max_depth: int) -> pd.DataFrame:
    b_ts, e_ts = _parse_date(begin), _parse_date(end)
    if b_ts > e_ts:
        raise ValueError(f"begin({begin}) 晚于 end({end})")
    records, total = _post(_fmt_date(b_ts), _fmt_date(e_ts), direction, timeout=timeout)
    if sleep:
        time.sleep(sleep)

    if total > len(records):
        if depth >= max_depth:
            raise RuntimeError(
                f"区间 {begin}~{end} 递归 {depth} 层仍被截断（total={total}, "
                f"records={len(records)}）—— 拒绝返回不完整数据")
        span = (e_ts - b_ts).days
        if span < 1:
            raise RuntimeError(f"单日区间仍被截断（{begin}）：total={total}，接口口径可能已变")
        # 按**真实日历日**对半切（不能按 YYYYMMDD 整数中点 —— 会产生非法日期）
        mid = b_ts + pd.Timedelta(days=span // 2)
        log.info("[cninfo_holder] %s %s~%s 被截断 (%d/%d) ⇒ 二分 %s~%s + %s~%s",
                 direction, _fmt_date(b_ts), _fmt_date(e_ts), len(records), total,
                 _fmt_date(b_ts), _fmt_date(mid),
                 _fmt_date(mid + pd.Timedelta(days=1)), _fmt_date(e_ts))
        left = _fetch_rec(b_ts, mid, direction, timeout=timeout,
                          sleep=sleep, depth=depth + 1, max_depth=max_depth)
        right = _fetch_rec(mid + pd.Timedelta(days=1), e_ts, direction, timeout=timeout,
                           sleep=sleep, depth=depth + 1, max_depth=max_depth)
        return pd.concat([left, right], ignore_index=True)

    return _normalize(records, direction)


def iter_overlap_windows(years: list[int], span: int = 2):
    """产出**带重叠**的年段窗口 ``(begin, end)``（``YYYYMMDD`` int）。

    为什么要重叠（见模块头坑 3）：区间过滤口径**不是纯粹的『截止日在区间内』**
    —— 实测「2019 全年」含 ``公告日`` 到 2021 的行，而「2019 分季并集」比它少
    2 886 条。切窄会丢行，所以**用 stride=1 的宽窗口覆盖每个年份两次**，
    再靠全列去重。

    >>> list(iter_overlap_windows([2019, 2020, 2021]))
    [(20190101, 20201231), (20200101, 20211231)]
    """
    ys = sorted(set(years))
    for i in range(len(ys) - span + 1):
        yield ys[i] * 10000 + 101, ys[i + span - 1] * 10000 + 1231


def year_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """按 ``chg_date``（变动年）与 ``ann_date``（公告年）分别统计行数。

    用于**整年缺失检测** —— 2026-09-20 的非法日期二分事故就是整年消失，
    单看总行数（130 456）根本发现不了，必须按年对账。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["year", "chg_rows", "ann_rows"])
    chg = pd.to_datetime(df["chg_date"].astype("Int64").astype(str), format="%Y%m%d",
                         errors="coerce").dt.year
    ann = pd.to_datetime(df["ann_date"].astype("Int64").astype(str), format="%Y%m%d",
                         errors="coerce").dt.year
    years = sorted(set(chg.dropna().unique()) | set(ann.dropna().unique()))
    return pd.DataFrame([
        {"year": int(y),
         "chg_rows": int((chg == y).sum()),
         "ann_rows": int((ann == y).sum())}
        for y in years])


def assert_no_empty_years(df: pd.DataFrame, years: list[int],
                          column: str = "chg_date", min_rows: int = 1) -> None:
    """**完整性闸门**：``years`` 里每一年都必须有 ≥ ``min_rows`` 行，否则抛异常。

    这道闸门是本模块唯一的「静默丢年」防线。实测的非法日期二分事故让
    2010/2011/2013 三个整年消失而**不抛任何异常**；只有按年对账能抓到。
    """
    if df is None or df.empty:
        raise RuntimeError(f"空结果，无法覆盖 {years}")
    col = pd.to_datetime(df[column].astype("Int64").astype(str), format="%Y%m%d",
                         errors="coerce").dt.year
    empty = [y for y in years if int((col == y).sum()) < min_rows]
    if empty:
        raise RuntimeError(
            f"整年缺失（{column}）：{empty} —— 拒绝落盘，疑似切分边界非法或接口口径变更")
