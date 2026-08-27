"""
可执行性掩码
============

把"这只股票这天能不能被回测引擎买卖"这件事显式建模出来，
避免向量化回测悄悄假设涨停能买、跌停能卖、停牌能成交。

用法
----
    mask = build_executable_mask(status_df, dates, codes, close_panel)
    result = bt.run(factor_panel, returns_panel, executable_mask=mask)
"""
from __future__ import annotations

import pandas as pd


def build_executable_mask(
    status_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    codes: pd.Index,
    close_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """构建 (date, code) 布尔矩阵：True=可执行（可开仓/调仓），False=不可执行。

    Args:
        status_df: DataCache.get_history_stock_status 返回的长表，
            列含 date, code, high_limited, low_limited, is_suspended（可选 is_st）。
            index 可以是 (date, code) 多索引，也可以是普通 RangeIndex。
        dates: 回测涉及的交易日索引。
        codes: 回测涉及的证券代码索引。
        close_panel: DataFrame(index=date, columns=code)，用于判断当日是否封板
            （close >= high_limited 视为涨停封板，close <= low_limited 视为跌停封板）。
            为 None 时跳过涨跌停封板判断，只按停牌过滤。

    Returns:
        DataFrame(index=dates, columns=codes)，dtype=bool，默认 True（无状态数据时不过滤）。
    """
    mask = pd.DataFrame(True, index=dates, columns=codes)

    if status_df is None or status_df.empty:
        return mask

    df = status_df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])

    # 停牌：直接不可执行
    if "is_suspended" in df.columns:
        susp_wide = df.pivot_table(
            index="date", columns="code", values="is_suspended", aggfunc="max"
        )
        susp_wide = susp_wide.reindex(index=dates, columns=codes).fillna(False).astype(bool)
        mask &= ~susp_wide

    # 涨跌停封板：用当日收盘价与涨跌停价比较（无分钟数据时的近似判断）
    if close_panel is not None and {"high_limited", "low_limited"}.issubset(df.columns):
        high_wide = df.pivot_table(index="date", columns="code", values="high_limited")
        low_wide = df.pivot_table(index="date", columns="code", values="low_limited")
        high_wide = high_wide.reindex(index=dates, columns=codes)
        low_wide = low_wide.reindex(index=dates, columns=codes)
        close_aligned = close_panel.reindex(index=dates, columns=codes)

        hit_up = (close_aligned >= high_wide) & high_wide.notna()
        hit_down = (close_aligned <= low_wide) & low_wide.notna()
        mask &= ~hit_up.fillna(False)
        mask &= ~hit_down.fillna(False)

    return mask


def build_directional_masks(
    status_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    codes: pd.Index,
    close_panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建**有方向的**可交易掩码 (buyable, sellable)，用于信号级约束校验。

    比 build_executable_mask 更细：涨停封板只是"买不进"（buyable=False），
    跌停封板只是"卖不出"（sellable=False），停牌则两者都禁。这样信号层能区分
    "想买但涨停"与"想卖但跌停"，分别标记 BLOCKED_BUY / BLOCKED_SELL。

    Args:
        status_df: get_history_stock_status 长表，列含 date, code, high_limited,
            low_limited, is_suspended。
        dates / codes: 交易日 / 代码索引。
        close_panel: (date, code) 收盘价面板，判涨停/跌停封板；None 时只按停牌过滤。
    Returns:
        (buyable, sellable)：均为 (date, code) 布尔面板，True=允许该方向。默认全 True。
    """
    buyable = pd.DataFrame(True, index=dates, columns=codes)
    sellable = pd.DataFrame(True, index=dates, columns=codes)

    if status_df is None or status_df.empty:
        return buyable, sellable

    df = status_df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])

    if "is_suspended" in df.columns:
        susp = df.pivot_table(index="date", columns="code", values="is_suspended", aggfunc="max")
        susp = susp.reindex(index=dates, columns=codes)
        susp = susp.astype(float).fillna(0.0).astype(bool)
        buyable &= ~susp
        sellable &= ~susp

    if close_panel is not None and {"high_limited", "low_limited"}.issubset(df.columns):
        hi = df.pivot_table(index="date", columns="code", values="high_limited")
        lo = df.pivot_table(index="date", columns="code", values="low_limited")
        high_wide = hi.reindex(index=dates, columns=codes)
        low_wide = lo.reindex(index=dates, columns=codes)
        close_aligned = close_panel.reindex(index=dates, columns=codes)

        hit_up = ((close_aligned >= high_wide) & high_wide.notna()).fillna(False)
        hit_down = ((close_aligned <= low_wide) & low_wide.notna()).fillna(False)
        buyable &= ~hit_up      # 涨停封板：买不进
        sellable &= ~hit_down   # 跌停封板：卖不出

    return buyable, sellable
