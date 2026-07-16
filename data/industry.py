"""
行业分类（point-in-time）
=========================

把 DataCache.get_industry_classification 返回的"个股在某行业指数下的
纳入/剔除区间"长表，转成按日的行业归属面板，供因子行业中性化使用。

用法
----
    ind = IndustryClassification(cache, level=1)
    industry_panel = ind.get_industry_panel(codes, dates)
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from data.cache import DataCache
from data.point_in_time import lookup_intervals


class IndustryClassification:
    """行业分类管理器。"""

    def __init__(self, cache: DataCache, level: int = 1):
        self._cache = cache
        self._level = level

    def get_industry_panel(
        self,
        codes: Sequence[str],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """按日构建行业归属面板 (date, code) -> industry_code|NaN，point-in-time。

        与 Universe.get_membership_mask 共用同一套 point-in-time 区间查找
        实现（data.point_in_time.lookup_intervals），区别是这里要具体的行业
        代码而不是布尔值：同一天被多条区间覆盖时（分类改标的重叠边界情况），
        取最近开始的那条。没有任何区间覆盖的 (date, code) 为 NaN。
        """
        df = self._cache.get_industry_classification(self._level)
        codes = list(codes)
        return lookup_intervals(df, dates, codes, code_col="code", value_col="industry_code")
