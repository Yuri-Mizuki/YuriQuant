"""fundamental_common 单元测试（2026-09-11 随双实现收敛建立）。

背景：add_single_quarter / add_ttm_yoy 曾在 HS300 与全A 两个 builder 各持
一份拷贝（同名同职），收敛前用合成季报长表探针证实两版逐位等价
（max|Δ|=0）后合一。本文件把探针固化为回归测试，并覆盖两个口径要点：
Q1 单季 = 累计本身（上年 Q4 不扣减）、跨年 Q2 = 上季累计扣减。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.common.fundamental_common import add_single_quarter, add_ttm_yoy


def _quarterly_frame() -> pd.DataFrame:
    """2 码 × 8 季报的合成利润表长表（含缺失）。"""
    rng = np.random.default_rng(7)
    rows = []
    for code in ("000001.SZ", "600519.SH"):
        for y in (2023, 2024):
            for q in (1, 2, 3, 4):
                rows.append({
                    "code": code,
                    "ann_date": pd.Timestamp(f"{y}-{[4, 7, 10, 12][q - 1]}-15"),
                    "report_period": pd.Timestamp(f"{y}-{[3, 6, 9, 12][q - 1]}-30"),
                    "OPERA_REV": float(rng.uniform(1e8, 5e8)) * (1 + 0.1 * q),
                })
    df = pd.DataFrame(rows)
    df.loc[df.sample(5, random_state=1).index, "OPERA_REV"] = np.nan
    return df


def test_single_quarter_q1_equals_cumulative():
    """Q1 单季 = 累计本身（不扣上年 Q4）。"""
    df = _quarterly_frame()
    out = add_single_quarter(df, "OPERA_REV", "sq")
    q1 = out[out["report_period"].dt.month == 3].dropna(subset=["sq"])
    merged = q1.merge(df[["code", "report_period", "OPERA_REV"]],
                      on=["code", "report_period"], suffixes=("", "_cum"))
    assert np.allclose(merged["sq"], merged["OPERA_REV"], equal_nan=True)


def test_single_quarter_q2_deducts_previous_quarter():
    """同年 Q2 单季 = 半年累计 - Q1 累计。"""
    df = _quarterly_frame()
    out = add_single_quarter(df, "OPERA_REV", "sq")
    cum = df.set_index(["code", "report_period"])["OPERA_REV"]
    q2 = out[(out["report_period"].dt.month == 6)].set_index(["code", "report_period"])
    expected = q2.index.map(lambda k: cum.loc[k] - cum.loc[(k[0], pd.Timestamp(f"{k[1].year}-03-30"))])
    assert np.allclose(q2["sq"].values, np.asarray(expected, dtype=float), equal_nan=True)


def test_ttm_and_yoy_present():
    """TTM 列恒有值（累计口径自洽），yoy 列按参数生成。"""
    df = _quarterly_frame()
    out = add_ttm_yoy(df, "OPERA_REV", "ttm", "yoy")
    assert "ttm" in out.columns and "yoy" in out.columns
    base = out.dropna(subset=["ttm"])
    assert len(base) > 0 and base["yoy"].notna().any()
    # 无 yoy 列时不出该列
    out2 = add_ttm_yoy(df, "OPERA_REV", "ttm", None)
    assert "ttm" in out2.columns and "yoy" not in out2.columns


def test_missing_field_returns_original():
    """字段缺失时原样返回并告警（不抛异常）。"""
    df = _quarterly_frame()
    out = add_single_quarter(df, "NOT_EXIST", "sq")
    assert list(out.columns) == list(df.columns)
    out2 = add_ttm_yoy(df, "NOT_EXIST", "ttm", None)
    assert list(out2.columns) == list(df.columns)
