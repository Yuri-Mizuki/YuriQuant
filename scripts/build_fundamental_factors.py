"""
基本面因子构建与入库（价值/质量/成长/规模）
============================================

补齐因子库的横截面风格维度（当前库全是量价技术因子）。全部因子严格
point-in-time：财务值按公告日（ann_date）对齐交易日展开，利润表/现金流
做 TTM（滚动十二个月）与同比处理，无未来函数。

因子清单（32 个）
-----------------
规模  : ln_mktcap            对数市值 = ln(股本 × 收盘价)
价值  : ep_ttm / bp / sp_ttm 盈利收益率 / 账面市值比 / 营收市值比（TTM）
质量  : roe_ttm / roa_ttm / gross_margin / net_margin / accruals / leverage
成长  : rev_growth_yoy / np_growth_yoy  营收/净利同比增速

—— 2026-08-04 扩展：单季增速 / 周转率 / ROIC / 费用率 / 扣非（季频 12）——
   np_growth_sq_yoy / np_growth_sq_qoq   净利单季同比 / 环比
   rev_growth_sq_yoy / rev_growth_sq_qoq 营收单季同比 / 环比
   oppro_growth_sq_yoy / oppro_growth_sq_qoq  营业利润单季同比 / 环比
   asset_turnover_ttm   资产周转率 = 营收TTM / 总资产
   inv_turnover_ttm     存货周转率 = 营业成本TTM / 存货
   recv_turnover_ttm    应收周转率 = 营收TTM / (应收账款+应收票据)
   roic_ttm             资本回报率 = EBIT_TTM×(1-税率) / 投入资本
   fin_exp_ratio_ttm    财务费用率 = 财务费用TTM / 营收TTM
   np_ded_growth_ttm_yoy 扣非净利增速_TTM同比

—— 2026-08-04 扩展：日频估值/市值类（8 个，PIT 到交易日）——
   pcf_ttm              市现率TTM = 市值 / 经营现金流TTM
   pcf                  市现率   = 市值 / 当期经营现金流
   fcf_yield            自由现金流TTM / 总市值
   netcf_yield          净现金流TTM / 总市值
   peg                  PE_TTM / (净利TTM同比×100)
   float_mktcap         流通市值 = 流通股本 × 收盘价
   float_ratio          流通市值 / 总市值
   cfo_growth_ttm_yoy   经营现金流增速_TTM同比

—— 2026-08-04 晚：分红/股东类（7 个，需 dividend/share_holder/holder_num）——
   div_yield            股息率 = 最近一期已实施每股派息(税前) / 收盘价
   div_yield_ttm        股息率TTM = 过去365天已实施现金分红 / 总市值
   div_payout_ratio     股利支付率 = 每股分红×基准股本 / 对应报告期净利
   holder_num_zs        股东户数时序标准分（expanding zscore，披露日口径）
   inst_holder_cnt      持仓机构个数（十大股东中按性质筛机构，HOLDER_ENDDATE 口径）
   inst_holder_chg      持仓机构个数变化（当期-上期）
   holder_dispersion    十大股东持股占比分散度（各股东 HOLDER_PCT 的标准差）

TTM 口径：利润表为年初至今累计值，TTM(x) = 本期累计 + 上年年报 - 上年同期，
确保任意公告日看到的都是"滚动 12 个月"的盈利（避免 4 月后仍用全年累计）。
单季口径：Q 季单季值 = Q 季累计 - 上季度累计（Q1 为自身累计）。
分红口径：div_yield 用"实施公告日 <= t 的最新一期已实施派息"；div_yield_ttm
按派息日（payout_date）落入过去 365 天汇总（与星耀数智口径一致，无未来函数）。

用法
----
    python -m scripts.build_fundamental_factors --offline          # 读缓存（推荐）
    python -m scripts.build_fundamental_factors --mock             # mock 验证
    python -m scripts.build_fundamental_factors --offline --no-save  # 只算不入库
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.cache_helpers import load_daily, load_financial_tables  # noqa: E402
from data.financials import build_pit_panel  # noqa: E402
from research.factor_library import FactorLibrary  # noqa: E402
from scripts.cli_common import (  # noqa: E402
    add_build_args, make_data_context, print_no_save, record_experiment_safe,
    register_panels, returns_from_daily,
    setup_logging,
)

log = setup_logging("build_fundamental_factors")

# 财务字段（真实缓存已验证存在）
INCOME_FIELDS = {
    "OPERA_REV": "营业收入",
    "LESS_OPERA_COST": "营业成本",
    "NET_PRO_INCL_MIN_INT_INC": "净利润(含少数股东)",
}
BALANCE_FIELDS = {
    "TOTAL_ASSETS": "总资产",
    "TOT_SHARE_EQUITY_EXCL_MIN_INT": "股东权益(不含少数)",
    "TOT_SHARE": "期末总股本(股)",
}
CFO_FIELD = "NET_CASH_FLOWS_OPERA_ACT"   # 经营活动现金流量净额
FREE_CF_FIELD = "FREE_CASH_FLOW"         # 自由现金流
NET_CF_FIELD = "NET_INCR_CASH_AND_CASH_EQU"  # 现金及现金等价物净增加额

FACTOR_DEFS: dict[str, str] = {
    "ln_mktcap": "对数市值 ln(股本×收盘价)",
    "ep_ttm": "盈利收益率 EP = 净利TTM/市值",
    "bp": "账面市值比 BP = 股东权益/市值",
    "sp_ttm": "营收市值比 SP = 营收TTM/市值",
    "roe_ttm": "净资产收益率 ROE = 净利TTM/股东权益",
    "roa_ttm": "总资产收益率 ROA = 净利TTM/总资产",
    "gross_margin": "毛利率 = (营收-成本)TTM/营收TTM",
    "net_margin": "净利率 = 净利TTM/营收TTM",
    "accruals": "应计利润 = (净利TTM-经营现金流TTM)/总资产",
    "leverage": "财务杠杆 = 总资产/股东权益",
    "rev_growth_yoy": "营收同比增速",
    "np_growth_yoy": "净利同比增速",
    # ---- 2026-08-04 扩展 ----
    "np_growth_sq_yoy": "净利单季同比",
    "np_growth_sq_qoq": "净利单季环比",
    "rev_growth_sq_yoy": "营收单季同比",
    "rev_growth_sq_qoq": "营收单季环比",
    "oppro_growth_sq_yoy": "营业利润单季同比",
    "oppro_growth_sq_qoq": "营业利润单季环比",
    "asset_turnover_ttm": "资产周转率TTM = 营收TTM/总资产",
    "inv_turnover_ttm": "存货周转率TTM = 营业成本TTM/存货",
    "recv_turnover_ttm": "应收周转率TTM = 营收TTM/(应收+票据)",
    "roic_ttm": "资本回报率TTM = EBIT_TTM×(1-税率)/投入资本",
    "fin_exp_ratio_ttm": "财务费用率TTM = 财务费用TTM/营收TTM",
    "np_ded_growth_ttm_yoy": "扣非净利增速_TTM同比",
    "pcf_ttm": "市现率TTM = 市值/经营现金流TTM",
    "pcf": "市现率 = 市值/当期经营现金流",
    "fcf_yield": "自由现金流TTM/总市值",
    "netcf_yield": "净现金流TTM/总市值",
    "peg": "PEG = PE_TTM/(净利TTM同比×100)",
    "float_mktcap": "流通市值 = 流通股本×收盘价",
    "float_ratio": "流通市值/总市值",
    "cfo_growth_ttm_yoy": "经营现金流增速_TTM同比",

    # ---- 2026-08-04 晚：分红/股东类（需 dividend/share_holder/holder_num 接口）----
    "div_yield": "股息率 = 最近一期已实施每股派息/收盘价",
    "div_yield_ttm": "股息率TTM = 过去365天已实施现金分红/总市值",
    "div_payout_ratio": "股利支付率 = 每股分红×基准股本/净利",
    "holder_num_zs": "股东户数时序标准分（expanding zscore）",
    "inst_holder_cnt": "持仓机构个数（十大股东中机构股东数）",
    "inst_holder_chg": "持仓机构个数变化",
    "holder_dispersion": "十大股东持股占比分散度（比例标准差）",
}

# ---- TTM / 同比（长表维度，按 (code, report_period)）----

def _add_ttm_yoy(df: pd.DataFrame, field: str, ttm_col: str, yoy_col: str | None) -> pd.DataFrame:
    """给财报长表加 TTM 列（和可选同比列）。

    A 股利润表/现金流为年初至今累计值：
        TTM(x) = x(本期) + x(上年年报) - x(上年同期)
        YoY(x) = x(本期) / x(上年同期) - 1
    report_period 为季度末 Timestamp（3/6/9/12 月）。
    字段不存在时（如 mock 财务表只有单字段）原样返回并告警。
    """
    if field not in df.columns:
        log.warning("字段 %s 不在财报表中，跳过 TTM（可用: %s...）",
                    field, list(df.columns)[:6])
        return df
    d = df[["code", "ann_date", "report_period", field]].copy()
    d = d.dropna(subset=["report_period", field])
    d["year"] = d["report_period"].dt.year
    d["quarter"] = d["report_period"].dt.quarter
    d["_key"] = d["code"].astype(str) + "_" + d["year"].astype(str) + "_" + d["quarter"].astype(str)
    d["_prev_annual_key"] = d["code"].astype(str) + "_" + (d["year"] - 1).astype(str) + "_4"
    d["_prev_yoy_key"] = d["code"].astype(str) + "_" + (d["year"] - 1).astype(str) + "_" + d["quarter"].astype(str)

    # 按 code 取最新报告期值（同一 (code, period) 多报表类型时保留 ann_date 最新）
    d = d.sort_values(["code", "report_period", "ann_date"])
    d = d.drop_duplicates(subset=["code", "report_period"], keep="last")
    val_map = dict(zip(d["_key"], d[field]))

    def _lookup(key):
        v = val_map.get(key)
        return np.nan if v is None else float(v)

    d["_prev_annual"] = d["_prev_annual_key"].map(lambda k: _lookup(k))
    d["_prev_yoy"] = d["_prev_yoy_key"].map(lambda k: _lookup(k))
    d[ttm_col] = d[field] + d["_prev_annual"] - d["_prev_yoy"]
    if yoy_col:
        d[yoy_col] = d[field] / d["_prev_yoy"].replace(0.0, np.nan) - 1.0
    out = df.copy()
    keep = ["code", "ann_date", "report_period", ttm_col] + ([yoy_col] if yoy_col else [])
    extra = d[keep]
    # 同一 (code, ann_date) 可能对应多个报告期，必须带 report_period 一起 merge
    return out.merge(extra, on=["code", "ann_date", "report_period"], how="left")

def _add_single_quarter(df: pd.DataFrame, field: str, sq_col: str) -> pd.DataFrame:
    """利润表累计值 → 单季值（Q1=累计；Q2=半年报-一季报；Q3=三季报-半年报；Q4=年报-三季报）。

    report_period 为季度末 Timestamp（3/6/9/12 月）。单季值 = 本期累计 - 上季度累计
    （Q1 无上季累计，单季=累计本身）。字段缺失时原样返回。
    """
    if field not in df.columns:
        log.warning("字段 %s 不在财报表中，跳过单季拆解", field)
        return df
    d = df[["code", "ann_date", "report_period", field]].copy()
    d = d.dropna(subset=["report_period", field])
    d["year"] = d["report_period"].dt.year
    d["quarter"] = d["report_period"].dt.quarter
    d = d.sort_values(["code", "report_period", "ann_date"])
    d = d.drop_duplicates(subset=["code", "report_period"], keep="last")
    d["_key"] = d["code"].astype(str) + "_" + d["year"].astype(str) + "_" + d["quarter"].astype(str)
    # 上季度 key（Q1 的上季为上年 Q4）—— 在去重后的 d 上重算，避免索引错位
    prev_q = (d["quarter"] - 2) % 4 + 1          # 1->4, 2->1, 3->2, 4->3
    prev_y = d["year"] - (d["quarter"] == 1).astype(int)
    d["_prev_key"] = d["code"].astype(str) + "_" + prev_y.astype(str) + "_" + prev_q.astype(str)
    val_map = dict(zip(d["_key"], d[field]))
    d["_prev_cum"] = d["_prev_key"].map(lambda k: val_map.get(k, np.nan))
    # 单季 = 本期累计 - 上季度累计；Q1 的上季度是上年 Q4，减去后得到"本年 Q1 单季"
    # 但 Q1 累计本就是单季，故上季累计为 0。用 _prev_cum 判断：Q1 时上年 Q4 累计不应扣减
    # （否则 Q1=年报累计-上年Q4累计 是错误口径）。修正：仅当年内扣减。
    d["sq"] = d[field]
    same_year = (d["year"] == prev_y)
    # 当年 Q2+ 才扣上季累计；Q1 单季 = 累计本身（_prev_cum 是上年 Q4，不扣）
    d["sq"] = d["sq"].where(~same_year | (d["quarter"] == 1), d[field] - d["_prev_cum"])
    out = df.copy()
    keep = ["code", "ann_date", "report_period", "sq"]
    out = out.merge(d[keep].rename(columns={"sq": sq_col}),
                    on=["code", "ann_date", "report_period"], how="left")
    return out

def _add_sq_growth(df: pd.DataFrame, sq_col: str, yoy_col: str, qoq_col: str) -> pd.DataFrame:
    """单季同比（上年同季）与环比（上季）。已在长表含 sq_col。"""
    if sq_col not in df.columns:
        log.warning("字段 %s 不存在，跳过单季增速", sq_col)
        return df
    d = df[["code", "ann_date", "report_period", sq_col]].copy()
    d = d.dropna(subset=["report_period", sq_col])
    d["year"] = d["report_period"].dt.year
    d["quarter"] = d["report_period"].dt.quarter
    d["_key"] = d["code"].astype(str) + "_" + d["year"].astype(str) + "_" + d["quarter"].astype(str)
    d["_yoy_key"] = d["code"].astype(str) + "_" + (d["year"] - 1).astype(str) + "_" + d["quarter"].astype(str)
    prev_q = (d["quarter"] - 2) % 4 + 1
    prev_y = d["year"] - (d["quarter"] == 1).astype(int)
    d["_qoq_key"] = d["code"].astype(str) + "_" + prev_y.astype(str) + "_" + prev_q.astype(str)
    d = d.sort_values(["code", "report_period", "ann_date"])
    d = d.drop_duplicates(subset=["code", "report_period"], keep="last")
    val_map = dict(zip(d["_key"], d[sq_col]))
    d["_yoy_base"] = d["_yoy_key"].map(lambda k: val_map.get(k, np.nan))
    d["_qoq_base"] = d["_qoq_key"].map(lambda k: val_map.get(k, np.nan))
    d[yoy_col] = d[sq_col] / d["_yoy_base"].replace(0.0, np.nan) - 1.0
    d[qoq_col] = d[sq_col] / d["_qoq_base"].replace(0.0, np.nan) - 1.0
    out = df.copy()
    keep = ["code", "ann_date", "report_period", yoy_col, qoq_col]
    out = out.merge(d[keep], on=["code", "ann_date", "report_period"], how="left")
    return out

# ---- 主流程 ----

def load_data(cache, uni, index_code, begin, end):
    codes, cal, daily = load_daily(cache, uni, index_code, begin, end)
    fin = load_financial_tables(cache, codes)
    log.info("日线 %d 行 / 利润表 %d / 资产负债 %d / 现金流 %d / 股本 %d / 分红 %d / 十大股东 %d / 股东户数 %d",
             len(daily), len(fin["income"]), len(fin["balance_sheet"]), len(fin["cash_flow"]),
             len(fin["equity_structure"]), len(fin["dividend"]),
             len(fin["share_holder"]), len(fin["holder_num"]))
    return (codes, cal, daily, fin["income"], fin["balance_sheet"], fin["cash_flow"],
            fin["equity_structure"], fin["dividend"], fin["share_holder"], fin["holder_num"])

def _equity_pit(equity: pd.DataFrame, cal, field: str) -> pd.DataFrame:
    """股本结构按 change_date 前向填充到交易日（日频因子用）。"""
    if equity is None or equity.empty or field not in equity.columns:
        return pd.DataFrame(index=pd.DatetimeIndex(cal), columns=pd.Index([]))
    d = equity[["code", "change_date", field]].copy()
    d["change_date"] = pd.to_datetime(d["change_date"], errors="coerce")
    d = d.dropna(subset=["change_date", field])
    d[field] = pd.to_numeric(d[field], errors="coerce")
    d = d.dropna(subset=[field])
    d = d.sort_values(["code", "change_date"])
    d = d.drop_duplicates(subset=["code", "change_date"], keep="last")
    if d.empty:
        return pd.DataFrame(index=pd.DatetimeIndex(cal), columns=pd.Index([]))
    cal_idx = pd.DatetimeIndex(pd.to_datetime([str(c) for c in cal], format="%Y%m%d")).sort_values()
    panels = []
    for code, g in d.groupby("code"):
        s = g.set_index("change_date")[field]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s = s.reindex(cal_idx, method="ffill")
        panels.append(pd.Series(s.values, index=cal_idx, name=code))
    return pd.concat(panels, axis=1)

def _event_pit(event_df: pd.DataFrame, cal, date_col: str, field: str) -> pd.DataFrame:
    """通用事件表 PIT：按 date_col（公告日/披露日）前向填充到交易日。"""
    if event_df is None or event_df.empty or field not in event_df.columns:
        return pd.DataFrame(index=pd.DatetimeIndex(cal), columns=pd.Index([]))
    d = event_df[["code", date_col, field]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col, field])
    d[field] = pd.to_numeric(d[field], errors="coerce")
    d = d.dropna(subset=[field])
    d = d.sort_values(["code", date_col])
    d = d.drop_duplicates(subset=["code", date_col], keep="last")
    if d.empty:
        return pd.DataFrame(index=pd.DatetimeIndex(cal), columns=pd.Index([]))
    cal_idx = pd.DatetimeIndex(pd.to_datetime([str(c) for c in cal], format="%Y%m%d")).sort_values()
    panels = []
    for code, g in d.groupby("code"):
        s = g.set_index(date_col)[field]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s = s.reindex(cal_idx, method="ffill")
        panels.append(pd.Series(s.values, index=cal_idx, name=code))
    return pd.concat(panels, axis=1)

def _dividend_factors(dividend, cal, close_panel, cap_panel, net_pro_panel) -> dict[str, pd.DataFrame]:
    """分红类因子（按实施公告日 PIT；TTM 口径按派息日落在过去 365 天汇总）。

    div_yield       : 最新一期已实施每股派息(税前) / 收盘价
    div_yield_ttm   : 过去 365 天累计派息额 / 总市值
    div_payout_ratio: 每股分红×基准股本 / 对应报告期净利
    """
    empty = {k: pd.DataFrame(index=close_panel.index, columns=close_panel.columns)
             for k in ("div_yield", "div_yield_ttm", "div_payout_ratio")}
    if dividend is None or dividend.empty:
        return empty
    dd = dividend.copy()
    for col in ("ann_date", "record_date", "ex_date", "payout_date", "report_period"):
        if col in dd.columns:
            dd[col] = pd.to_datetime(dd[col], errors="coerce")
    if "cash_per_share_pre_tax" in dd.columns:
        dd["cash_per_share_pre_tax"] = pd.to_numeric(dd["cash_per_share_pre_tax"], errors="coerce")
    if "base_share" in dd.columns:
        dd["base_share"] = pd.to_numeric(dd["base_share"], errors="coerce")
    # 只保留已实施的分红（有派息日或实施公告日）
    impl = dd.dropna(subset=["payout_date", "ann_date"])

    # 1) 最新一期已实施每股派息（按 ann_date 实施公告日 PIT）
    latest_cash = _event_pit(impl, cal, "ann_date", "cash_per_share_pre_tax")
    # 2) TTM：派息日落在过去 365 天内的累计现金分红 / 市值
    cal_idx = pd.DatetimeIndex(pd.to_datetime([str(c) for c in cal], format="%Y%m%d")).sort_values()
    cum_payout = pd.DataFrame(index=cal_idx, columns=close_panel.columns).astype(float)
    cum_payout[:] = 0.0
    valid_payout = impl.dropna(subset=["payout_date", "cash_per_share_pre_tax", "base_share", "code"])
    for _, row in valid_payout.iterrows():
        pay_date = row["payout_date"]
        code = row["code"]
        if code not in cum_payout.columns:
            continue
        amt = float(row["cash_per_share_pre_tax"]) * float(row["base_share"]) * 1e4  # 元
        # 派息日之后 365 天内该日期的累计分红（简化：滚动窗口，事件稀疏可接受）
        win = cal_idx[(cal_idx >= pay_date) & (cal_idx <= pay_date + pd.Timedelta(days=365))]
        if len(win):
            cum_payout.loc[win, code] += amt
    # 3) 股利支付率：每股分红×基准股本(万股→股)×1e4(元) / 对应报告期净利
    #    按实施公告日 ann_date 做 PIT（事件型：实施后即生效，ffill 到后续交易日）。
    #    净利取该分红公告日（ann_date）已知的最新净利（PIT 面板在 ann_date 的值——
    #    2024 年报分红在 2025-06-20 实施时，当日 PIT 净利即 2024 年报净利）。
    payout_ratio = pd.DataFrame(index=close_panel.index, columns=close_panel.columns,
                                dtype=float)
    if "report_period" in impl.columns and net_pro_panel is not None:
        rp = impl.dropna(subset=["report_period", "cash_per_share_pre_tax", "base_share",
                                 "ann_date"])
        if not rp.empty:
            rp = rp.copy()
            rp["_total_cash"] = rp["cash_per_share_pre_tax"] * rp["base_share"] * 1e4
            rows_out = []
            for _, row in rp.iterrows():
                code = row["code"]
                if code not in net_pro_panel.columns:
                    continue
                ann = pd.Timestamp(row["ann_date"])
                np_series = net_pro_panel[code]
                # 分红公告日已知的净利：PIT 面板在 ann_date 或之前最近的值
                known = np_series[np_series.index <= ann]
                if known.empty:
                    continue
                np_val = float(known.iloc[-1])
                if np_val and np_val == np_val:
                    rows_out.append((ann, code, float(row["_total_cash"]) / np_val))
            if rows_out:
                payout_long = pd.DataFrame(rows_out, columns=["ann_date", "code", "ratio"])
                payout_ratio = _event_pit(payout_long, cal, "ann_date", "ratio").reindex(
                    index=close_panel.index, columns=close_panel.columns)
    # 面板对齐
    out = {
        "div_yield": latest_cash.reindex(index=close_panel.index, columns=close_panel.columns)
        / close_panel.replace(0.0, np.nan),
        "div_yield_ttm": cum_payout.reindex(index=close_panel.index, columns=close_panel.columns)
        / cap_panel.replace(0.0, np.nan),
        "div_payout_ratio": payout_ratio.reindex(index=close_panel.index, columns=close_panel.columns),
    }
    for k in out:
        out[k] = out[k].reindex(index=close_panel.index, columns=close_panel.columns)
    return out

def _holder_factors(share_holder, holder_num, cal, close_panel) -> dict[str, pd.DataFrame]:
    """股东类因子（十大股东 + 股东户数）。

    inst_holder_cnt   : 十大股东中机构股东个数（按 HOLDER_ENDDATE 披露期统计）
    inst_holder_chg   : 当期 - 上期
    holder_dispersion : 十大股东 HOLDER_PCT 的标准差（持股集中度反向）
    holder_num_zs     : 股东户数 expanding zscore（披露日口径，PIT 到交易日）
    """
    empty = {k: pd.DataFrame(index=close_panel.index, columns=close_panel.columns)
             for k in ("inst_holder_cnt", "inst_holder_chg", "holder_dispersion", "holder_num_zs")}
    if (share_holder is None or share_holder.empty) and (holder_num is None or holder_num.empty):
        return empty

    out: dict[str, pd.DataFrame] = {}

    # ---- 十大股东 ----
    if share_holder is not None and not share_holder.empty:
        sh = share_holder.copy()
        for col in ("ann_date", "holder_end_date"):
            if col in sh.columns:
                sh[col] = pd.to_datetime(sh[col], errors="coerce")
        if "holder_pct" in sh.columns:
            sh["holder_pct"] = pd.to_numeric(sh["holder_pct"], errors="coerce")
        sh = sh.dropna(subset=["code", "holder_end_date", "holder_pct"])

        # 机构识别：HOLDER_HOLDER_CATEGORY（股东性质）在星耀数智里
        # 未保留到归一化表，改用名称启发式（基金/保险/信托/券商/银行/社保/汇金等）
        inst_keywords = ("基金", "保险", "信托", "证券", "银行", "社保", "汇金",
                         "资产管理", "投资", "企业年金", "养老", "QFII", "外资")
        if "holder_name" in sh.columns:
            sh["is_inst"] = sh["holder_name"].astype(str).apply(
                lambda n: any(k in n for k in inst_keywords))
        else:
            sh["is_inst"] = False

        # 按 (code, holder_end_date) 聚合
        inst_cnt = sh.groupby(["code", "holder_end_date"])["is_inst"].sum()
        dispersion = sh.groupby(["code", "holder_end_date"])["holder_pct"].std()
        # 转面板：按披露期末日 PIT（公告日前向填充到交易日）
        inst_cnt_df = inst_cnt.reset_index()
        inst_cnt_df.columns = ["code", "holder_end_date", "cnt"]
        inst_cnt_panel = _event_pit(inst_cnt_df, cal, "holder_end_date", "cnt")
        dispersion_df = dispersion.reset_index()
        dispersion_df.columns = ["code", "holder_end_date", "disp"]
        disp_panel = _event_pit(dispersion_df, cal, "holder_end_date", "disp")
        # 变化 = 当期 - 上期（逐 code 差分）
        chg_panel = inst_cnt_panel.diff().reindex(
            index=close_panel.index, columns=close_panel.columns)
        out["inst_holder_cnt"] = inst_cnt_panel.reindex(
            index=close_panel.index, columns=close_panel.columns)
        out["inst_holder_chg"] = chg_panel
        out["holder_dispersion"] = disp_panel.reindex(
            index=close_panel.index, columns=close_panel.columns)

    # ---- 股东户数 ----
    if holder_num is not None and not holder_num.empty:
        hn = holder_num.copy()
        for col in ("ann_date", "holder_end_date"):
            if col in hn.columns:
                hn[col] = pd.to_datetime(hn[col], errors="coerce")
        if "holder_num" in hn.columns:
            hn["holder_num"] = pd.to_numeric(hn["holder_num"], errors="coerce")
        hn = hn.dropna(subset=["code", "ann_date", "holder_num"])
        # expanding zscore（按披露日，逐 code）
        rows = []
        for code, g in hn.groupby("code"):
            g = g.sort_values("ann_date")
            g["exp_mean"] = g["holder_num"].expanding().mean()
            g["exp_std"] = g["holder_num"].expanding().std()
            g["zs"] = (g["holder_num"] - g["exp_mean"]) / g["exp_std"].replace(0.0, np.nan)
            rows.append(g[["code", "ann_date", "zs"]])
        if rows:
            hn_zs = pd.concat(rows, ignore_index=True)
            out["holder_num_zs"] = _event_pit(hn_zs, cal, "ann_date", "zs").reindex(
                index=close_panel.index, columns=close_panel.columns)

    for k in empty:
        if k not in out:
            out[k] = empty[k]
    return out

def build_factor_panels(daily, cal, income, balance, cashflow,
                        equity: pd.DataFrame | None = None,
                        dividend: pd.DataFrame | None = None,
                        share_holder: pd.DataFrame | None = None,
                        holder_num: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """返回 {因子名: 原始面板(date×code)}。"""
    # 收盘价面板（日期×代码）
    d = daily.reset_index()
    d["date"] = d["date"].dt.normalize()
    close_panel = d.pivot(index="date", columns="code", values="close").sort_index()

    # 1) 财务长表加 TTM / 同比列
    inc = income.copy()
    inc = _add_ttm_yoy(inc, "OPERA_REV", "OPERA_REV_TTM", "REV_YOY")
    inc = _add_ttm_yoy(inc, "LESS_OPERA_COST", "LESS_OPERA_COST_TTM", None)
    inc = _add_ttm_yoy(inc, "NET_PRO_INCL_MIN_INT_INC", "NET_PRO_TTM", "NP_YOY")
    if "LESS_OPERA_COST_TTM" in inc.columns:
        inc["GROSS_PROFIT_TTM"] = inc["OPERA_REV_TTM"] - inc["LESS_OPERA_COST_TTM"]
    else:
        inc["GROSS_PROFIT_TTM"] = inc["OPERA_REV_TTM"]  # mock 无成本列，毛利≈营收
    cf = cashflow.copy()
    cf = _add_ttm_yoy(cf, CFO_FIELD, "CFO_TTM", "CFO_YOY")
    cf = _add_ttm_yoy(cf, FREE_CF_FIELD, "FREE_CF_TTM", None)
    cf = _add_ttm_yoy(cf, NET_CF_FIELD, "NET_CF_TTM", None)

    # 1b) 单季拆解 + 单季同比/环比
    inc = _add_single_quarter(inc, "NET_PRO_INCL_MIN_INT_INC", "NP_SQ")
    inc = _add_single_quarter(inc, "OPERA_REV", "REV_SQ")
    inc = _add_single_quarter(inc, "OPERA_PROFIT", "OPPROF_SQ")
    inc = _add_sq_growth(inc, "NP_SQ", "NP_SQ_YOY", "NP_SQ_QOQ")
    inc = _add_sq_growth(inc, "REV_SQ", "REV_SQ_YOY", "REV_SQ_QOQ")
    inc = _add_sq_growth(inc, "OPPROF_SQ", "OPPROF_SQ_YOY", "OPPROF_SQ_QOQ")
    # 扣非净利 TTM 同比
    inc = _add_ttm_yoy(inc, "NET_PRO_AFTER_DED_NR_GL", "NP_DED_TTM", "NP_DED_TTM_YOY")

    # 2) PIT 展开（按公告日对齐交易日，ffill）
    def _pit(report_df, field):
        if field not in report_df.columns:
            log.warning("字段 %s 不在表中（mock 数据字段有限），返回空面板", field)
            return pd.DataFrame(index=close_panel.index, columns=close_panel.columns)
        return build_pit_panel(report_df, cal, field)

    pit_income = {
        "OPERA_REV_TTM": _pit(inc, "OPERA_REV_TTM"),
        "NET_PRO_TTM": _pit(inc, "NET_PRO_TTM"),
        "GROSS_PROFIT_TTM": _pit(inc, "GROSS_PROFIT_TTM"),
        "REV_YOY": _pit(inc, "REV_YOY"),
        "NP_YOY": _pit(inc, "NP_YOY"),
        "LESS_OPERA_COST_TTM": _pit(inc, "LESS_OPERA_COST_TTM"),
        "NP_SQ_YOY": _pit(inc, "NP_SQ_YOY"),
        "NP_SQ_QOQ": _pit(inc, "NP_SQ_QOQ"),
        "REV_SQ_YOY": _pit(inc, "REV_SQ_YOY"),
        "REV_SQ_QOQ": _pit(inc, "REV_SQ_QOQ"),
        "OPPROF_SQ_YOY": _pit(inc, "OPPROF_SQ_YOY"),
        "OPPROF_SQ_QOQ": _pit(inc, "OPPROF_SQ_QOQ"),
        "NP_DED_TTM_YOY": _pit(inc, "NP_DED_TTM_YOY"),
    }
    pit_balance = {
        "TOTAL_ASSETS": _pit(balance, "TOTAL_ASSETS"),
        "EQUITY": _pit(balance, "TOT_SHARE_EQUITY_EXCL_MIN_INT"),
        "TOT_SHARE": _pit(balance, "TOT_SHARE"),
        "INV": _pit(balance, "INV"),
        "ACC_RECEIVABLE": _pit(balance, "ACC_RECEIVABLE"),
        "NOTES_RECEIVABLE": _pit(balance, "NOTES_RECEIVABLE"),
        "ST_BORROWING": _pit(balance, "ST_BORROWING"),
        "LT_LOAN": _pit(balance, "LT_LOAN"),
        "BONDS_PAYABLE": _pit(balance, "BONDS_PAYABLE"),
        "NONCUR_LIAB_DUE_WITHIN_1Y": _pit(balance, "NONCUR_LIAB_DUE_WITHIN_1Y"),
    }
    pit_cfo = {
        "CFO_TTM": _pit(cf, "CFO_TTM"),
        "CFO_YOY": _pit(cf, "CFO_YOY"),
        "FREE_CF_TTM": _pit(cf, "FREE_CF_TTM"),
        "NET_CF_TTM": _pit(cf, "NET_CF_TTM"),
    }
    # 统一索引/列
    all_pit = {**pit_income, **pit_balance, **pit_cfo}
    all_pit = {k: v.reindex(index=close_panel.index, columns=close_panel.columns)
               for k, v in all_pit.items()}

    # 3) 组合成因子
    cap = all_pit["TOT_SHARE"].astype(float) * close_panel.astype(float)  # 市值
    ln_cap = np.log(cap.clip(lower=1.0))
    eps = all_pit["NET_PRO_TTM"] / cap
    bps = all_pit["EQUITY"] / cap
    sps = all_pit["OPERA_REV_TTM"] / cap

    # 单季增速（直接 PIT 面板）
    sq_yoy = {k: all_pit[k] for k in ("NP_SQ_YOY", "NP_SQ_QOQ", "REV_SQ_YOY",
                                      "REV_SQ_QOQ", "OPPROF_SQ_YOY", "OPPROF_SQ_QOQ")}

    # 周转率 / ROIC / 费用率
    asset_turnover = all_pit["OPERA_REV_TTM"] / all_pit["TOTAL_ASSETS"].replace(0.0, np.nan)
    inv_turnover = all_pit["LESS_OPERA_COST_TTM"] / all_pit["INV"].replace(0.0, np.nan)
    recv = (all_pit["ACC_RECEIVABLE"].fillna(0.0) + all_pit["NOTES_RECEIVABLE"].fillna(0.0))
    recv_turnover = all_pit["OPERA_REV_TTM"] / recv.replace(0.0, np.nan)
    # 投入资本 = 股东权益 + 有息负债（短借+长借+应付债券+一年内到期非流动负债）
    debt = (all_pit["ST_BORROWING"].fillna(0.0) + all_pit["LT_LOAN"].fillna(0.0)
            + all_pit["BONDS_PAYABLE"].fillna(0.0) + all_pit["NONCUR_LIAB_DUE_WITHIN_1Y"].fillna(0.0))
    invest_cap = all_pit["EQUITY"] + debt
    # EBIT_TTM×(1-税率)/投入资本；税率=所得税/利润总额 缺省 25%
    ebit_ttm = _pit(inc, "EBIT") if "EBIT" in inc.columns else None
    # EBIT 是累计值，需 TTM（若存在）
    inc_ebit = _add_ttm_yoy(inc, "EBIT", "EBIT_TTM", None) if "EBIT" in inc.columns else inc
    ebit_ttm_panel = _pit(inc_ebit, "EBIT_TTM") if "EBIT_TTM" in inc_ebit.columns else None
    tax_rate = _pit(inc, "INCOME_TAX") / _pit(inc, "TOTAL_PROFIT").replace(0.0, np.nan)
    tax_rate = tax_rate.clip(0.0, 0.6).fillna(0.25)
    roic = (ebit_ttm_panel * (1 - tax_rate)) / invest_cap.replace(0.0, np.nan) if ebit_ttm_panel is not None else pd.DataFrame()
    fin_exp_ratio = all_pit["LESS_OPERA_COST_TTM"] * np.nan  # 占位，下方用财务费用
    if "LESS_FIN_EXP" in inc.columns:
        inc_fin = _add_ttm_yoy(inc, "LESS_FIN_EXP", "LESS_FIN_EXP_TTM", None)
        fin_exp_ratio = _pit(inc_fin, "LESS_FIN_EXP_TTM") / all_pit["OPERA_REV_TTM"].replace(0.0, np.nan)
    else:
        log.warning("LESS_FIN_EXP 不在表中，fin_exp_ratio_ttm 为空")

    # 日频估值/市值类
    cfo_now = _pit(cf, CFO_FIELD)   # 当期经营现金流
    pcf_ttm = cap / all_pit["CFO_TTM"].replace(0.0, np.nan)
    pcf = cap / cfo_now.replace(0.0, np.nan)
    fcf_yield = all_pit["FREE_CF_TTM"] / cap
    netcf_yield = all_pit["NET_CF_TTM"] / cap
    peg = (cap / all_pit["NET_PRO_TTM"].replace(0.0, np.nan)) / (all_pit["NP_YOY"] * 100)
    # 股本 PIT（流通股）
    float_share = _equity_pit(equity, cal, "float_share") if equity is not None else pd.DataFrame()
    float_share = float_share.reindex(index=close_panel.index, columns=close_panel.columns)
    float_mktcap = float_share.astype(float) * close_panel.astype(float)
    float_ratio = float_share / all_pit["TOT_SHARE"].replace(0.0, np.nan)

    # 分红/股东类（2026-08-04 晚）
    div_factors = _dividend_factors(dividend, cal, close_panel, cap, all_pit["NET_PRO_TTM"])
    holder_factors = _holder_factors(share_holder, holder_num, cal, close_panel)

    panels: dict[str, pd.DataFrame] = {
        "ln_mktcap": ln_cap,
        "ep_ttm": eps,
        "bp": bps,
        "sp_ttm": sps,
        "roe_ttm": all_pit["NET_PRO_TTM"] / all_pit["EQUITY"].replace(0.0, np.nan),
        "roa_ttm": all_pit["NET_PRO_TTM"] / all_pit["TOTAL_ASSETS"].replace(0.0, np.nan),
        "gross_margin": all_pit["GROSS_PROFIT_TTM"] / all_pit["OPERA_REV_TTM"].replace(0.0, np.nan),
        "net_margin": all_pit["NET_PRO_TTM"] / all_pit["OPERA_REV_TTM"].replace(0.0, np.nan),
        "accruals": (all_pit["NET_PRO_TTM"] - all_pit["CFO_TTM"]) / all_pit["TOTAL_ASSETS"].replace(0.0, np.nan),
        "leverage": all_pit["TOTAL_ASSETS"] / all_pit["EQUITY"].replace(0.0, np.nan),
        "rev_growth_yoy": all_pit["REV_YOY"],
        "np_growth_yoy": all_pit["NP_YOY"],
        # ---- 2026-08-04 扩展 ----
        "np_growth_sq_yoy": sq_yoy["NP_SQ_YOY"],
        "np_growth_sq_qoq": sq_yoy["NP_SQ_QOQ"],
        "rev_growth_sq_yoy": sq_yoy["REV_SQ_YOY"],
        "rev_growth_sq_qoq": sq_yoy["REV_SQ_QOQ"],
        "oppro_growth_sq_yoy": sq_yoy["OPPROF_SQ_YOY"],
        "oppro_growth_sq_qoq": sq_yoy["OPPROF_SQ_QOQ"],
        "asset_turnover_ttm": asset_turnover,
        "inv_turnover_ttm": inv_turnover,
        "recv_turnover_ttm": recv_turnover,
        "roic_ttm": roic,
        "fin_exp_ratio_ttm": fin_exp_ratio,
        "np_ded_growth_ttm_yoy": all_pit["NP_DED_TTM_YOY"],
        "pcf_ttm": pcf_ttm,
        "pcf": pcf,
        "fcf_yield": fcf_yield,
        "netcf_yield": netcf_yield,
        "peg": peg,
        "float_mktcap": float_mktcap,
        "float_ratio": float_ratio,
        "cfo_growth_ttm_yoy": all_pit["CFO_YOY"],
        # ---- 分红/股东类（2026-08-04 晚）----
        **div_factors,
        **holder_factors,
    }
    # 财务面板比价格面板可能多出日历外日期，截到 close 索引
    for name in panels:
        panels[name] = panels[name].reindex(index=close_panel.index, columns=close_panel.columns)
    return panels

def main():
    parser = argparse.ArgumentParser(description="基本面因子（价值/质量/成长/规模）构建入库")
    add_build_args(parser)
    args = parser.parse_args()

    cache, uni, begin, end, dataset = make_data_context(args)

    codes, cal, daily, income, balance, cashflow, equity, dividend, share_holder, holder_num = load_data(
        cache, uni, args.index, begin, end)
    if daily.empty:
        log.error("数据为空")
        sys.exit(1)

    log.info("计算基本面因子（%d 个）...", len(FACTOR_DEFS))
    panels = build_factor_panels(daily, cal, income, balance, cashflow, equity,
                                 dividend, share_holder, holder_num)

    if args.no_save:
        print_no_save(list(FACTOR_DEFS), panels)
        return

    lib = FactorLibrary(dataset=dataset)
    returns_panel = returns_from_daily(daily)

    log.info("入库到数据集: %s", dataset)
    register_panels(
        lib, panels, FACTOR_DEFS, returns_panel,
        source=f"fundamental:build_fundamental_factors:{begin}-{end}",
    )

    record_experiment_safe(
        kind="fundamental_factors",
        command=" ".join(sys.argv),
        params={"index": args.index, "begin": begin, "end": end, "dataset": dataset},
        fingerprint=cache.get_fingerprint(),
        result_path=str(lib.root),
        metrics={"n_factors": len([n for n in FACTOR_DEFS if n in panels])},
        note="基本面因子入库",
    )

    log.info("完成。数据集 %s 现有 %d 个因子", dataset, len(lib.list_all()))

if __name__ == "__main__":
    main()