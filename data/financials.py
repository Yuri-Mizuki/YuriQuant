"""
财务数据 PIT 工具
================

把稀疏的财报（按报告期一行）展开成「按交易日」的面板，**严格 point-in-time**：
某交易日的取值 = 公告日 <= 该交易日的最新一份财报对应字段值。

这避免了两种常见未来函数：
1. 用 report_period（报告期末）而非 ann_date（公告日）对齐 → 财报在报告期
   结束后 1~4 个月才发布，提前用等于偷看未来。
2. 对未来财报修订版本未做处理 → 这里按 (code, ann_date) 去重保留最新报告期。

典型用法::

    income = cache.get_income(codes)
    cal = cache.get_calendar(begin, end)
    np_panel = build_pit_panel(income, cal, "NET_PRO_INCL_MIN_INT_INC")
    # np_panel: DataFrame(index=交易日, columns=code)，可直接喂给算子/因子
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# 各报表常用字段白名单（供 mining 层枚举基本面特征用）
BALANCE_SHEET_FIELDS = (
    "TOTAL_ASSETS",          # 资产总计
    "TOTAL_CUR_ASSETS",      # 流动资产合计
    "TOT_NONCUR_ASSETS",     # 非流动资产合计
    "TOTAL_LIAB",            # 负债合计
    "TOTAL_CUR_LIAB",        # 流动负债合计
    "TOTAL_NONCUR_LIAB",     # 非流动负债合计
    "TOT_SHARE_EQUITY_EXCL_MIN_INT",  # 股东权益(不含少数股东)
    "TOT_SHARE_EQUITY_INCL_MIN_INT",  # 股东权益(含少数股东)
    "CURRENCY_CAP",          # 货币资金
    "INV",                   # 存货
    "FIXED_ASSETS",          # 固定资产
    "INTANGIBLE_ASSETS",     # 无形资产
    "GOODWILL",              # 商誉
    "ACC_RECEIVABLE",        # 应收票据及应收账款
    "ACCT_PAYABLE",          # 应付账款
    "TOT_SHARE",             # 期末总股本
    "ST_BORROWING",          # 短期借款
    "LT_LOAN",               # 长期借款
)

INCOME_FIELDS = (
    "OPERA_REV",             # 营业收入
    "OPERA_PROFIT",          # 营业利润
    "LESS_OPERA_COST",       # 营业成本
    "NET_PRO_INCL_MIN_INT_INC",   # 净利润(含少数股东损益)
    "NET_PRO_EXCL_MIN_INT_INC",   # 净利润(不含少数股东损益)
    "BASIC_EPS",             # 基本每股收益
    "DILUTED_EPS",           # 稀释每股收益
    "INCOME_TAX",            # 所得税
    "LESS_SELLING_EXP",      # 销售费用
    "LESS_ADMIN_EXP",        # 管理费用
    "LESS_FIN_EXP",          # 财务费用
    "RD_EXP",                # 研发费用
    "EBIT",                  # 息税前利润
    "EBITDA",                # 息税折旧摊销前利润
    "LESS_ASSETS_IMPAIR_LOSS",  # 资产减值损失
)

CASH_FLOW_FIELDS = (
    "WS_OPERA_ACT",          # 经营活动产生的现金流量净额
    "NET_PROFIT",            # 净利润（现金流表附注）
)


def to_datetime_col(s: pd.Series) -> pd.Series:
    """日期列 → ``datetime64``，三种落盘形态都能吃。

    SDK 的日期列传进来可能是：

    * ``datetime64``（income / balance / cash_flow 这类）；
    * **epoch 整数**（``profit_notice`` / ``margin_detail`` / ``block_trading`` 等
      事件表的 ``ann_date`` / ``trade_date``，实测单位是**微秒**，如
      ``1_453_334_400_000_000`` = 2016-01-21）；
    * ``YYYYMMDD`` 整数/字符串（``calendar``、``alt_cninfo_holder``）。

    ``pd.to_datetime(int)`` 只按**纳秒**解释整数，于是微秒值落到 **1970-01-17**、
    YYYYMMDD 落到 1970-01-01。这个错**不抛异常**：所有事件日期都早于任何交易日，
    PIT 的 ``reindex(method="ffill")`` 就把每只股票的**最后一个值**一路前填到面板
    第一天 —— 整块面板变成"一个截面复制 N 遍"（2026-11-07 体检发现 31 块面板如此，
    其中 4 个还进了 920 模型的特征清单）。

    单位按**中位数**判（首值可能是 0/NaT，不可靠）：2026 年附近秒 1.8e9 /
    毫秒 1.8e12 / 微秒 1.8e15，三档相差三个数量级。
    """
    if s.empty:
        return pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    num = pd.to_numeric(s, errors="coerce")
    valid = s.notna()
    # 判据的分母必须是非空值：日期列常年带 NaN（没发过公告的股票），
    # 用整列比例会把 [1.45e15, NaN, NaN] 这种正常列误判成字符串列。
    if valid.any() and float(num[valid].notna().mean()) > 0.99:
        med = float(np.nanmedian(num.to_numpy(dtype="float64")))
        if med > 1e14:                       # epoch 微秒
            return pd.to_datetime(num, unit="us", errors="coerce")
        if med > 1e11:                       # epoch 毫秒
            return pd.to_datetime(num, unit="ms", errors="coerce")
        if med > 1e8:                        # epoch 秒
            return pd.to_datetime(num, unit="s", errors="coerce")
        return pd.to_datetime(num.astype("Int64").astype(str), format="%Y%m%d",
                              errors="coerce")

    txt = s.astype(str).str.strip()
    if float(txt.str.fullmatch(r"\d{8}").mean()) > 0.99:
        return pd.to_datetime(txt, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(txt, errors="coerce")


def build_pit_panel(
    report_df: pd.DataFrame,
    calendar: list[int] | pd.DatetimeIndex,
    field: str,
) -> pd.DataFrame:
    """把稀疏财报展开为按交易日的 PIT 面板。

    Args:
        report_df: 财报长表，需含 code / ann_date / field 三列。
        calendar: 交易日列表（int YYYYMMDD 或 Timestamp）。
        field: 要展开的字段名。
    Returns:
        DataFrame(index=交易日, columns=code)，值为截至该交易日最新已知财报字段值。
    """
    if field not in report_df.columns:
        raise KeyError(f"字段 {field} 不在财报表中，可用列: {list(report_df.columns)}")
    cal_list = list(calendar)
    # 兼容 int 型 YYYYMMDD 日历（get_calendar 返回的就是这种）：
    # 直接 pd.to_datetime(int_series) 会把整数当成"自 epoch 起的纳秒"，
    # 生成 1970 年的垃圾时间戳，导致 PIT 索引错乱、reindex 全 NaN。
    if cal_list and isinstance(cal_list[0], (int, np.integer)):
        cal = pd.to_datetime([str(c) for c in cal_list], format="%Y%m%d")
    else:
        cal = pd.to_datetime(pd.Series(cal_list))

    df = report_df[["code", "ann_date", field]].copy()
    df["ann_date"] = to_datetime_col(df["ann_date"])
    df = df.dropna(subset=["ann_date", "code"])
    df[field] = pd.to_numeric(df[field], errors="coerce")
    df = df.dropna(subset=[field])
    if df.empty:
        return pd.DataFrame(index=cal, columns=pd.Index([]))

    # 同一 (code, ann_date) 多条 → 保留 field 最大的（通常对应合并报表）
    df = df.sort_values(["code", "ann_date", field])
    df = df.drop_duplicates(subset=["code", "ann_date"], keep="last")

    cal_idx = pd.DatetimeIndex(cal).sort_values()
    panels = []
    for code, g in df.groupby("code"):
        s = g.set_index("ann_date")[field]
        # 去重索引（同一天多份已处理）
        s = s[~s.index.duplicated(keep="last")].sort_index()
        # reindex 到交易日，前向填充：当日只能看到已公告的财报
        s = s.reindex(cal_idx, method="ffill")
        panels.append(pd.Series(s.values, index=cal_idx, name=code))
    if not panels:
        return pd.DataFrame(index=cal_idx)
    return pd.concat(panels, axis=1)


def build_pit_panels(
    report_df: pd.DataFrame,
    calendar: list[int] | pd.DatetimeIndex,
    fields: list[str],
) -> dict[str, pd.DataFrame]:
    """批量展开多个字段，返回 {field: panel}。"""
    return {f: build_pit_panel(report_df, calendar, f) for f in fields}
