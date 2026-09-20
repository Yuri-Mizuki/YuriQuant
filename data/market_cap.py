"""
市值面板构建
============

InfoData.get_equity_structure 返回的是稀疏的股本变动事件表（一次变动一行，
不是每日行情），要构建可直接用于因子中性化的日频市值面板，需要把事件表
按代码前向填充到完整的交易日历上，再乘以对应日期的收盘价。

用法
----
    equity_structure = cache.get_equity_structure(codes)
    market_cap_panel = build_market_cap_panel(equity_structure, close_panel)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_shares_panel(
    equity_structure_df: pd.DataFrame,
    index: pd.Index,
    columns: pd.Index,
    share_field: str = "tot_share",
) -> pd.DataFrame:
    """股本变动事件表 → 日频股本面板（单位：**股**），逐日 PIT 前向填充。

    与 :func:`build_market_cap_panel` 的差别：本函数只出股本（不乘股价），
    且用 ``pivot + ffill`` 向量化实现 —— 全A 规模（5800 列 × 2500 日）下
    :func:`build_market_cap_panel` 的逐 code 循环过慢，而风格中性化只需要股本，
    不需要先算市值再取 log。公式与
    ``scripts/pipelines/rolling_grid_alla._shares_panel`` 等价（后者是回测基线
    ``_base/cov_size`` 的真源，改这里须同步核对）。
    """
    out = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    if equity_structure_df is None or equity_structure_df.empty:
        return out
    if share_field not in equity_structure_df.columns:
        return out
    df = equity_structure_df[equity_structure_df["code"].isin(columns)][
        ["code", "change_date", share_field]].copy()
    df["change_date"] = pd.to_datetime(df["change_date"])
    df = df.dropna(subset=[share_field]).drop_duplicates(
        subset=["code", "change_date"], keep="last")
    if df.empty:
        return out
    wide = df.pivot(index="change_date", columns="code",
                    values=share_field).sort_index()
    wide = wide.reindex(index=sorted(set(wide.index) | set(index))).ffill()
    wide = wide.reindex(columns=columns)
    common = index.intersection(wide.index)
    out.loc[common] = wide.loc[common].to_numpy()
    return out * 10000.0     # 万股 -> 股


def build_market_cap_panel(
    equity_structure_df: pd.DataFrame,
    close_panel: pd.DataFrame,
    share_field: str = "tot_share",
) -> pd.DataFrame:
    """构建日频市值面板（单位：元）。

    Args:
        equity_structure_df: DataCache.get_equity_structure 返回的长表，
            列含 code, change_date, tot_share, float_share（单位：万股）。
        close_panel: DataFrame(index=date, columns=code)，复权后收盘价。
        share_field: "tot_share"（总市值，默认）或 "float_share"（流通市值）。

    Returns:
        DataFrame(index=close_panel.index, columns=close_panel.columns)，
        市值单位为元。上市日之前的日期为 NaN（不做回填，这是正确行为——
        该公司当时确实没有可计量的股本）。
    """
    shares_panel = pd.DataFrame(
        index=close_panel.index, columns=close_panel.columns, dtype=float
    )

    if equity_structure_df is None or equity_structure_df.empty:
        return shares_panel * close_panel  # 全 NaN

    df = equity_structure_df.copy()
    df["change_date"] = pd.to_datetime(df["change_date"])

    for code in close_panel.columns:
        rows = df[df["code"] == code]
        if rows.empty or share_field not in rows.columns:
            continue
        series = (
            rows[["change_date", share_field]]
            .dropna(subset=[share_field])
            .drop_duplicates(subset="change_date", keep="last")
            .set_index("change_date")[share_field]
            .sort_index()
        )
        if series.empty:
            continue
        shares_panel[code] = series.reindex(close_panel.index, method="ffill")

    # 万股 -> 股，再乘以收盘价（元）得到市值（元）
    return shares_panel * 10000.0 * close_panel
