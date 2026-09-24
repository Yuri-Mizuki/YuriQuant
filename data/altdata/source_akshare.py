"""另类数据 L1 传输层：akshare 通道（免自建四接口）
====================================================

覆盖 P0 计划 §〇.6.2 判定的「高价值 × 全面易得 × 单一数据源 × 免自建」
四条轴同时满足的 4 个接口 —— **零抓取器、零 enckey、零分页**：

======================================  ================================  ====================
接口                                     内容                              历史深度（实测）
======================================  ================================  ====================
``stock_hold_control_cninfo``             实际控制人持股变动               2010-12 起（一次全量）
``stock_inner_trade_xq``                  内部人交易（雪球）               2025-01 起（20.5 月）
``stock_hold_management_detail_cninfo``   高管增减持（巨潮，按方向两次）    滚动 1 年
``macro_info_ws``                         宏观日历（华尔街见闻，按日）      2018-01 起
======================================  ================================  ====================

🚨 PIT 质量三档（本模块最重要的口径标注，因子层必须区别对待）
------------------------------------------------------------------
四个接口的「数据可得时点」质量**不同**：

======================  ============================  ================================
接口                     PIT 时点                      评价
======================  ============================  ================================
``holder_control``       仅 ``变动日期``               ⚠️ **差**：实际披露在定期报告 /
                                                       权益变动报告中，晚于变动日最多数月
``inner_trade``          仅 ``变动日期``               🟡 中：法定披露期限为变动后 2 交易日
``mgmt_hold``            ``公告日期`` + ``截止日期``    ✅ 好：双时间戳，直接用公告日期
``macro_calendar``       ``时间``（分钟级）             ✅ 好：公布时点天然 PIT
======================  ============================  ================================

⇒ 前两个接口**没有公告日**。本模块原样返回 ``chg_date``（变动日），并在统一输出里
写一列 ``ann_date``（无公告日时回填 ``chg_date``，属**乐观可得日**）。
**因子层必须对它们施加保守滞后**，否则引入前视 —— 见 :data:`PIT_QUALITY`
与 :data:`SUGGESTED_LAG_TRADING_DAYS`。

统一输出 schema（标准化后交给 :class:`~data.altdata.fetch.AltDataCache` 落盘）
--------------------------------------------------------------------------------
- ``code``      标准代码（``600519.SH``），复用 ``data.textmining.source_ths.to_code_std``
                （与 ``source_cninfo.py`` 同一约定，不另造一份）
- ``ann_date``  数据可得日（PIT 唯一合法对齐轴）
- ``chg_date``  事件发生/统计日
- 数值列         一律 ``pd.to_numeric(errors="coerce")``；**无事件记 NaN，不填 0**

实测数据质量坑（2026-09-20，首轮真数据冒烟暴露）
--------------------------------------------------
1. **同表单位不一致**（``mgmt_hold``）：``变动数量`` 单位是**股**且**已带符号**
   （增持正 / 减持负），但 ``期初/期末持股数量`` 单位是**万股**。原样透传会得到
   双重取反的方向错误（实测减持曾产出 ``signed_qty=+5000``）。
2. **``期末市值`` 整列全空**（0 / 30317）⇒ **不能用它算金额**。
   ``成交均价`` 仅 52% 非空、``变动比例`` 仅 20% 非空。
   ⇒ 因子层一律走 **股数口径**（``chg_qty`` 100% 非空）再除流通股本，
   **不要**用 ``qty × price`` 造金额（会丢一半样本）。
3. **``inner_trade`` 的代码是交易所前缀式**（``SH600328`` / ``SZ301171``），
   不是标准格式；直接过 ``to_code_std`` 会产出 ``SH600328`` 这种废码 ——
   实测 join 项目面板的成功率只剩 **0.2%**。:func:`_code` 已统一处理三种写法。
4. **``holder_control`` 的 ``hold_qty`` 单位是万股**，且只有「变动时点」的行
   ⇒ 做水平量必须 ffill 且施加滞后（见 :data:`PIT_QUALITY`）。

列名防御
--------
akshare 升级可能改列名。所有映射都先过 :func:`_require` 做**显式失败**
（抛 RuntimeError 并列出实际列名），而不是静默产出全 NaN 面板 ——
后者正是本项目「静默降级」事故的同型（见 ``.workbuddy/memory/MEMORY.md``）。
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping

import pandas as pd

from data.textmining.source_ths import to_code6, to_code_std

log = logging.getLogger(__name__)

__all__ = [
    "PIT_QUALITY",
    "SUGGESTED_LAG_TRADING_DAYS",
    "fetch_holder_control",
    "fetch_inner_trade",
    "fetch_macro_calendar",
    "fetch_mgmt_hold_change",
]

# ---------------------------------------------------------------------------
# PIT 口径常量（因子层引用它们决定滞后，避免各处各写一份）
# ---------------------------------------------------------------------------
#: 各接口的可得时点类型。``has_ann_date`` = 有独立公告日，可直接 PIT；
#: ``no_ann_date`` = 只有变动日，须加保守滞后；``event_time`` = 公布时点。
PIT_QUALITY: dict[str, str] = {
    "holder_control": "no_ann_date",
    "inner_trade": "no_ann_date",
    "mgmt_hold": "has_ann_date",
    "cninfo_holder": "has_ann_date",
    "macro_calendar": "event_time",
    "cls": "event_time",
    "news": "event_time",
}

#: ``no_ann_date`` 接口的**建议保守滞后**（交易日）。
#: - ``holder_control``：变动日通常是定期报告统计期末，披露窗口最长到季报/年报公告
#:   （法定：年报 4 月末、一季报 4 月末、半年报 8 月末、三季报 10 月末）⇒ 取 60 交易日
#:   覆盖一个季度，是**宁可错杀不可前视**的保守值。
#: - ``inner_trade``：证券法/交易所规则要求董监高变动后 **2 个交易日内**公告 ⇒ 滞后 2。
#: - 其余有公告日/公布时点，无滞后。
#:
#: ⚠️ ``cninfo_holder`` 是本项目**自建**的全历史增减持表（:mod:`data.altdata.cninfo_holder`），
#: 它给出**真实的公告日期**（不是``mgmt_hold``那种乐观回填），故滞后 0 且 PIT 质量最高。
SUGGESTED_LAG_TRADING_DAYS: dict[str, int] = {
    "holder_control": 60,
    "inner_trade": 2,
    "mgmt_hold": 0,
    "cninfo_holder": 0,
    "macro_calendar": 0,
    "cls": 0,
    "news": 0,
}

# ---------------------------------------------------------------------------
# 统一输出列
# ---------------------------------------------------------------------------
HC_COLS = ["code", "name", "ann_date", "chg_date", "controller", "hold_qty",
           "hold_ratio", "direct_controller", "control_type", "source"]
IT_COLS = ["code", "name", "ann_date", "chg_date", "person", "chg_qty",
           "price", "qty_after", "relation", "position", "source"]
MH_COLS = ["code", "name", "ann_date", "chg_date", "person", "position",
           "relation", "direction", "qty_begin", "qty_end", "chg_qty",
           "signed_qty", "chg_ratio", "price", "mv_end", "reason", "source"]
MC_COLS = ["event_time", "date", "region", "event", "importance", "actual",
           "forecast", "previous", "source"]


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------
def _ak():
    """惰性导入 akshare（缺失时给出可直接执行的修复提示，不静默降级）。"""
    try:
        import akshare as ak
    except ImportError as e:  # pragma: no cover - 环境缺失路径
        raise ImportError(
            "另类数据抓取依赖 akshare，但当前解释器未安装。安装（.venv 由 uv 创建、"
            "内无 pip，必须走 uv）：\n"
            "  uv pip install -e \".[altdata]\" --python <repo>/.venv/Scripts/python.exe"
        ) from e
    return ak


def _require(df: pd.DataFrame, cols: Iterable[str], api: str) -> None:
    """列存在性检查 —— 缺列时显式失败并列出实际列名。

    防止 akshare 改列名后静默产出全 NaN 面板（静默降级同型事故）。
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{api} 返回列名与预期不符，缺少 {missing}；实际列 = {list(df.columns)}。"
            "akshare 可能升级改了列名，请核对后更新 data/altdata/source_akshare.py 的映射。"
        )


def _empty(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=cols)


def _opt(raw: pd.DataFrame, name: str) -> pd.Series:
    """取**可选**列：缺列时返回同索引的空 Series（不抛 KeyError、也不静默给错值）。

    与 ``_require`` 的分工：必需列缺失要**显式失败**，可选列缺失只降级为空。
    """
    return raw[name] if name in raw.columns else pd.Series(index=raw.index, dtype="object")


def _num(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _dt(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


#: 交易所前缀式代码（雪球返回 ``SH600328`` / ``SZ301171``，非项目标准格式）
_PREFIX_RE = re.compile(r"^(SH|SZ|BJ)\s*(\d{6})$", re.I)
#: 北交所代码段（2026-09-20 实测：项目 code_info 里 920xxx 记为 ``.BJ``，342 只）
_BJ_PREFIX2 = ("43", "83", "87", "88", "92")


def std_code6(c6: str) -> str:
    """6 位代码 → 项目标准格式（``600519.SH``）。

    ⚠️ 局部修正 ``source_ths.to_code_std`` 的一处口径 bug：它按首字符判市场
    （``'9' -> .SH``），但 **920xxx 是北交所新代码段**（项目 ``code_info.parquet``
    里 920375/920242 等 342 只均记为 ``.BJ``）。不修正则这些股票 join 不上
    项目面板（实测 inner_trade 的 join 率会被拉到 0.2%）。
    **公共工具是否统一修正另议** —— 改动会影响 textmining 已有缓存的 join 键，
    属于独立改动，不在本模块内静默改。
    """
    c = str(c6).zfill(6)
    if c[:2] in _BJ_PREFIX2:
        return f"{c}.BJ"
    return to_code_std(c)


def _code(series: pd.Series) -> pd.Series:
    """各类代码写法 → 项目标准代码（``600519.SH``）。

    实测需覆盖三种写法：``600328``（纯数字）/ ``600328.SH``（带后缀）/
    ``SH600328``（雪球的前缀式）。前两种走 ``to_code6``→``std_code6``，
    前缀式直接重组，避免 ``to_code6("SH600328")`` 原样返回后判错市场。
    """
    def conv(v):
        if pd.isna(v):
            return None
        s = str(v).strip().upper()
        if not s:
            return None      # 空串必须短路：否则 to_code6("") 会产出 "000000.SZ" 这种废码
        m = _PREFIX_RE.match(s)
        if m:
            # ⚠️ 不能直接重组 ``f"{code}.{exch}"``：雪球会把北交所个股标成
            # ``SH920242``（实测），而项目口径**按代码段定市场**（920xxx = .BJ）
            # ⇒ 前缀只用于剥离，判市场一律交给 _std_code6。
            return std_code6(m.group(2))
        return std_code6(to_code6(s))

    return series.map(conv)


# ---------------------------------------------------------------------------
# ① 实际控制人持股变动（巨潮数据中心）
# ---------------------------------------------------------------------------
def fetch_holder_control(symbol: str = "全部") -> pd.DataFrame:
    """实际控制人持股变动 —— 一次调用返回 2010-12 起全量（实测 5593 行 / 1.3s）。

    ⚠️ **无公告日**：只有 ``变动日期``（定期报告 / 权益变动报告口径的统计日），
    实际可得的时点晚于它 ⇒ ``ann_date`` 回填 ``chg_date`` 属**乐观可得日**，
    因子层须施加 :data:`SUGGESTED_LAG_TRADING_DAYS` 的滞后。
    """
    ak = _ak()
    raw = ak.stock_hold_control_cninfo(symbol=symbol)
    if raw is None or raw.empty:
        return _empty(HC_COLS)
    _require(raw, ["证券代码", "证券简称", "变动日期", "实际控制人名称",
                   "控股数量", "控股比例", "直接控制人名称", "控制类型"],
             "stock_hold_control_cninfo")
    out = pd.DataFrame({
        "code": _code(raw["证券代码"]),
        "name": raw["证券简称"].astype("string"),
        "chg_date": raw["变动日期"],
        "controller": raw["实际控制人名称"].astype("string"),
        "hold_qty": raw["控股数量"],
        "hold_ratio": raw["控股比例"],
        "direct_controller": raw["直接控制人名称"].astype("string"),
        "control_type": raw["控制类型"].astype("string"),
    })
    out = _dt(_num(out, ["hold_qty", "hold_ratio"]), ["chg_date"])
    # 无公告日 → ann_date 回填变动日（乐观可得日，见模块级 PIT_QUALITY）
    out["ann_date"] = out["chg_date"]
    out["source"] = "akshare:cninfo"
    out = out.dropna(subset=["code", "chg_date"])
    return out[HC_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# ② 内部人交易（雪球）
# ---------------------------------------------------------------------------
def fetch_inner_trade(name_to_code: Mapping[str, str] | None = None) -> pd.DataFrame:
    """内部人交易 —— 一次调用返回 20.5 个月全市场（实测 25987 行 / 2256 只 / 2.0s）。

    ⚠️ **无公告日**：只有 ``变动日期``；交易所规则要求变动后 2 个交易日内披露 ⇒
    因子层滞后 2 交易日即可（见 :data:`SUGGESTED_LAG_TRADING_DAYS`）。

    ⚠️ ``股票代码`` 非空率仅 **94.5%**（缺失集中在北交所等）⇒ 用 ``name_to_code``
    （项目 ``code_info.parquet`` 的名称映射）按 ``股票名称`` 回补；仍缺的置 NaN
    并在日志里报数（**不丢行** —— 名称仍可用于人工核对）。
    """
    ak = _ak()
    raw = ak.stock_inner_trade_xq()
    if raw is None or raw.empty:
        return _empty(IT_COLS)
    _require(raw, ["股票代码", "股票名称", "变动日期", "变动人", "变动股数",
                   "成交均价", "变动后持股数"], "stock_inner_trade_xq")

    code = _code(raw["股票代码"])
    n_missing = int(code.isna().sum())
    if n_missing and name_to_code:
        filled = raw["股票名称"].map(
            lambda n: name_to_code.get(str(n).strip()) if pd.notna(n) else None)
        # 回补路径同样必须走 _std_code6（北交所段修正）—— 用裸 to_code_std 会让
        # 回补出来的 920xxx 与主路径分歧
        code = code.where(code.notna(),
                          filled.map(lambda c: std_code6(to_code6(c)) if pd.notna(c) else None))
        log.info("[altdata] inner_trade 代码缺失 %d/%d，按名称回补后剩 %d",
                 n_missing, len(raw), int(code.isna().sum()))
    elif n_missing:
        log.warning("[altdata] inner_trade 代码缺失 %d/%d 且未提供名称映射，"
                    "这些行 code=NaN（仍保留）", n_missing, len(raw))

    out = pd.DataFrame({
        "code": code,
        "name": raw["股票名称"].astype("string"),
        "chg_date": raw["变动日期"],
        "person": raw["变动人"].astype("string"),
        "chg_qty": raw["变动股数"],
        "price": raw["成交均价"],
        "qty_after": raw["变动后持股数"],
        "relation": _opt(raw, "与董监高关系").astype("string"),
        "position": _opt(raw, "董监高职务").astype("string"),
    })
    out = _dt(_num(out, ["chg_qty", "price", "qty_after"]), ["chg_date"])
    out["ann_date"] = out["chg_date"]          # 乐观可得日（法定 T+2，因子层加滞后）
    out["source"] = "akshare:xq"
    out = out.dropna(subset=["chg_date"])      # code 允许 NaN（不丢行）
    return out[IT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# ③ 高管增减持（巨潮，按方向两次调用）
# ---------------------------------------------------------------------------
def fetch_mgmt_hold_change(directions: tuple[str, ...] = ("增持", "减持")) -> pd.DataFrame:
    """高管及董监高持股变动明细 —— 按方向调用后合并。

    实测（akshare 原样）：增持 13413 行 / 减持 16906 行 = 30319 行 / 3082 只，
    16 列含 ``成交均价 / 变动比例 / 期末市值 / 持股变动原因``。

    ⚠️ **仅滚动 1 年**：akshare 封装把 ``sdate/edate`` 写死在函数体内
    （``sdate = 去年前今天``）⇒ 要 2010 起全历史须走 P0 §3.2 的
    ``p_sysapi1030`` 自建通道（Step 1b，Δ 触发）。本函数是「零自建验证期」入口。

    ✅ **双时间戳**：``公告日期``（可得日）+ ``截止日期``（变动日），PIT 质量最好。
    """
    ak = _ak()
    frames: list[pd.DataFrame] = []
    for d in directions:
        raw = ak.stock_hold_management_detail_cninfo(symbol=d)
        if raw is None or raw.empty:
            log.warning("[altdata] mgmt_hold symbol=%s 返回空", d)
            continue
        _require(raw, ["证券代码", "证券简称", "截止日期", "公告日期", "变动数量"],
                 f"stock_hold_management_detail_cninfo({d})")
        sign = 1.0 if d == "增持" else -1.0
        # ⚠️ 实测（2026-09-20）：巨潮「变动数量」**已带符号**（增持正、减持负），单位「股」；
        # 而同一张表的「期初/期末持股数量」单位是「万股」—— 同表单位不一致。
        # 这里一律取绝对值再按 direction 重算符号：不依赖接口的符号约定，避免
        # 「接口已带负号 × sign=-1」的双重取反（实测 bug：减持曾产出 signed_qty=+5000）。
        qty = pd.to_numeric(raw["变动数量"], errors="coerce").abs()
        f = pd.DataFrame({
            "code": _code(raw["证券代码"]),
            "name": raw["证券简称"].astype("string"),
            "ann_date": raw["公告日期"],
            "chg_date": raw["截止日期"],
            "person": (_opt(raw, "高管姓名") if "高管姓名" in raw.columns
                       else _opt(raw, "董监高姓名")).astype("string"),
            "position": _opt(raw, "董监高职务").astype("string"),
            "relation": _opt(raw, "变动人与董监高关系").astype("string"),
            "direction": d,
            "qty_begin": _opt(raw, "期初持股数量"),
            "qty_end": _opt(raw, "期末持股数量"),
            "chg_qty": qty,
            "signed_qty": qty * sign,
            "chg_ratio": _opt(raw, "变动比例"),
            "price": _opt(raw, "成交均价"),
            "mv_end": _opt(raw, "期末市值"),
            "reason": _opt(raw, "持股变动原因").astype("string"),
        })
        frames.append(f)
    if not frames:
        return _empty(MH_COLS)
    out = pd.concat(frames, ignore_index=True)
    out = _dt(_num(out, ["qty_begin", "qty_end", "chg_qty", "signed_qty",
                         "chg_ratio", "price", "mv_end"]),
              ["ann_date", "chg_date"])
    out["source"] = "akshare:cninfo"
    out = out.dropna(subset=["code", "ann_date"])
    return out[MH_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# ④ 宏观日历 + 预期值（华尔街见闻，单日查询）
# ---------------------------------------------------------------------------
def fetch_macro_calendar(date: str | int) -> pd.DataFrame:
    """某一交易日的宏观日历（今值 / 预期 / 前值 + 重要性 1–4）。

    ⚠️ **单日查询**：接口无区间参数，回补须按日循环（2018-01-10 起 ≈2000 次）
    ⇒ 断点续传由 :meth:`AltDataCache.get_macro_calendar` 负责。

    ⚠️ **部分交易日「预期」列整列缺失**（实测 2026-09-14 预期 0 条、09-15 仅 1 条）
    ⇒ 入库必须做缺失日检测，否则会把「缺数据」误当成「无预期」。

    ✅ 用**单一源**（本接口）保证 surprise 分母 σ(历史意外) 口径一致；
    金十系接口已断更于 2025-09（见计划 §〇 表第 2 行），**禁止与本源混用**。

    Parameters
    ----------
    date : str | int
        ``YYYYMMDD``（int 也接受，内部转 str —— int 直接进 ``pd.to_datetime`` 会变 1970）。
    """
    ak = _ak()
    d = str(date).replace("-", "")
    raw = ak.macro_info_ws(date=d)
    if raw is None or raw.empty:
        return _empty(MC_COLS)
    _require(raw, ["时间", "地区", "事件", "重要性"], f"macro_info_ws({d})")
    out = pd.DataFrame({
        "event_time": raw["时间"],
        "region": raw.get("地区", pd.Series(index=raw.index, dtype="object")).astype("string"),
        "event": raw.get("事件", pd.Series(index=raw.index, dtype="object")).astype("string"),
        "importance": raw.get("重要性", pd.Series(index=raw.index, dtype="object")),
        "actual": raw.get("今值", pd.Series(index=raw.index, dtype="object")),
        "forecast": raw.get("预期", pd.Series(index=raw.index, dtype="object")),
        "previous": raw.get("前值", pd.Series(index=raw.index, dtype="object")),
    })
    out = _dt(_num(out, ["importance", "actual", "forecast", "previous"]), ["event_time"])
    out["date"] = pd.to_datetime(d, format="%Y%m%d")   # 日历日 int→datetime（不靠隐式解析）
    out["source"] = "akshare:ws"
    out = out.dropna(subset=["event_time"])
    return out[MC_COLS].sort_values("event_time").reset_index(drop=True)
