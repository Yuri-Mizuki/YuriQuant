"""
股票池（Universe）管理
======================

功能：
1. 基于指数成分股构建股票池（沪深300/中证500等）。
2. 处理成分股动态变动（纳入/剔除），避免幸存者偏差。
3. 按交易日返回当时有效的成分股列表（point-in-time）。

用法
----
    uni = Universe(cache)
    hs300_codes = uni.get_constituent("000300.SH", 20240101)
    # 或便捷方法
    hs300_codes = uni.get_hs300(20240101)
"""
from __future__ import annotations

import pandas as pd

from config import Config
from data.cache import DataCache


class Universe:
    """股票池管理器。"""

    def __init__(self, cache: DataCache):
        self._cache = cache

    def get_constituent(self, index_code: str, date: int) -> list[str]:
        """获取指定日期的指数成分股列表（point-in-time）。

        逻辑：成分股的 in_date <= date 且 (out_date 为空 或 out_date > date)。
        """
        df = self._cache.get_index_constituent(index_code)
        if df.empty:
            return []

        # 日期格式归一化：SDK 返回的 in_date/out_date 可能是 str 或 Timestamp
        df = df.copy()
        df["in_date"] = pd.to_datetime(df["in_date"], errors="coerce")
        df["out_date"] = pd.to_datetime(df["out_date"], errors="coerce")
        ts = pd.Timestamp(str(date))

        mask = (df["in_date"] <= ts) & (df["out_date"].isna() | (df["out_date"] > ts))
        return df.loc[mask, "con_code"].unique().tolist()

    # ---- 便捷方法 ----
    def get_hs300(self, date: int) -> list[str]:
        return self.get_constituent("000300.SH", date)

    def get_zz500(self, date: int) -> list[str]:
        return self.get_constituent("000905.SH", date)

    def get_zz1000(self, date: int) -> list[str]:
        return self.get_constituent("000852.SH", date)

    def get_default(self, date: int) -> list[str]:
        """按 config 中 universe.default 返回。"""
        default = Config.universe().get("default", "hs300")
        mapping = {
            "hs300": self.get_hs300,
            "zz500": self.get_zz500,
            "zz1000": self.get_zz1000,
        }
        return mapping[default](date)

    def get_all_constituents(self, index_code: str) -> pd.DataFrame:
        """返回完整成分股变动表（不做日期过滤）。"""
        return self._cache.get_index_constituent(index_code)
