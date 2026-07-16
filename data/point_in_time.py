"""
Point-in-time 区间查找
======================

Universe.get_membership_mask（股票是否属于某指数）和
IndustryClassification.get_industry_panel（股票属于哪个行业）本质上是
同一个问题：给一张"code 在某个 [in_date, out_date) 区间内成立"的记录表，
按日展开成 (dates × codes) 矩阵。两者只是取值方式不同——前者只要"是否
被覆盖"的布尔值，后者要"被哪条记录覆盖"的具体值（同一天有多条区间重叠
时取最近开始的那条）。这里提供两者共用的底层实现。

用法
----
    # 布尔覆盖（如成分股归属）
    mask = lookup_intervals(df, dates, codes, code_col="con_code")

    # 取值（如行业代码，多区间重叠时取最近开始的）
    panel = lookup_intervals(df, dates, codes, code_col="code", value_col="industry_code")
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def lookup_intervals(
    records: pd.DataFrame,
    dates: pd.DatetimeIndex,
    codes: Sequence[str],
    code_col: str,
    in_col: str = "in_date",
    out_col: str = "out_date",
    value_col: str | None = None,
) -> pd.DataFrame:
    """按日展开 point-in-time 区间记录表。

    Args:
        records: 长表，至少含 code_col/in_col/out_col（value_col 存在时也含它）。
            in_col/out_col 可以是字符串或 Timestamp，out_col 为空（NaN/NaT）
            表示区间尚未结束（一直覆盖到最后）。
        dates: 目标交易日索引。
        codes: 目标证券代码列表。
        code_col: records 中标识证券代码的列名（不同调用方列名不一致，
            如 universe 用 "con_code"，industry 用 "code"）。
        value_col: 为 None 时返回 bool 矩阵（该 code 当天是否被任意区间覆盖）；
            指定列名时返回该列的值，多条区间重叠覆盖同一天时取 in_date
            最近（最大）的那条，没有任何区间覆盖时为 NaN。

    Returns:
        DataFrame(index=dates, columns=codes)。
    """
    codes = list(codes)
    fill_value = False if value_col is None else np.nan
    dtype = bool if value_col is None else object
    result = pd.DataFrame(fill_value, index=dates, columns=codes, dtype=dtype)

    if records.empty:
        return result

    df = records.copy()
    df[in_col] = pd.to_datetime(df[in_col], errors="coerce")
    df[out_col] = pd.to_datetime(df[out_col], errors="coerce")

    dates_arr = dates.values.reshape(-1, 1)  # (n_dates, 1)

    for code in codes:
        rows = df[df[code_col] == code]
        if rows.empty:
            continue
        # 按 in_date 降序：同一天被多条区间覆盖时，argmax 取到的首个 True
        # 就是最近开始的那条（value_col 场景下这是有意义的 tiebreak；
        # value_col=None 场景下排序对结果无影响，any() 与顺序无关）。
        rows = rows.sort_values(in_col, ascending=False).reset_index(drop=True)

        in_dates = rows[in_col].values.reshape(1, -1)   # (1, n_rows)
        out_dates = rows[out_col].values.reshape(1, -1)
        covered = (dates_arr >= in_dates) & (
            pd.isna(rows[out_col].values).reshape(1, -1) | (dates_arr < out_dates)
        )  # (n_dates, n_rows)

        if value_col is None:
            result[code] = covered.any(axis=1)
        else:
            has_match = covered.any(axis=1)
            first_idx = covered.argmax(axis=1)  # 首个 True（rows 已按 in_date 降序）
            result[code] = np.where(has_match, rows[value_col].values[first_idx], np.nan)

    return result
