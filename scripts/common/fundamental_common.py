"""财报长表的 TTM / 单季拆解公共函数（2026-09-11 收敛）。

此前 ``add_ttm_yoy`` / ``add_single_quarter`` 在 HS300 builder
（``scripts/factors/build_fundamental_factors.py``）与全A builder
（``scripts/builders/build_alla_fundamental_factors.py``）各持一份拷贝，
同名同职、实现漂移风险高。等价性探针（合成 2 码 × 8 季报长表，含缺失值）
证实两版**逐位一致**（max|Δ|=0）后收敛到本模块，真源取 factors 版
（含缺字段告警与完整 docstring）。

口径：
- 利润表/现金流为**年初至今累计值**：TTM = 本期 + 上年年报 - 上年同期；
- 单季 = 当年内本期累计 - 上季累计；Q1 单季 = 累计本身（上年 Q4 不扣减）；
- 同一 (code, report_period) 多报表类型保留 ann_date 最新一行。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.common.cli_common import setup_logging

log = setup_logging("fundamental_common")


def add_ttm_yoy(df: pd.DataFrame, field: str, ttm_col: str, yoy_col: str | None) -> pd.DataFrame:
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


def add_single_quarter(df: pd.DataFrame, field: str, sq_col: str) -> pd.DataFrame:
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
