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
from data.point_in_time import lookup_intervals


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

    def get_all_a(self, date: int) -> list[str]:
        """全 A 池：本地缓存 daily_all_a 中截至该日出现过的全部代码。

        PIT 语义：上市前无行情 → 未出现在缓存 → 天然不属于该日池，
        等价于按"上市日期"过滤，无需额外接口。缓存中代码上限即全市场
        A 股（约 5400 只，daily_all_a 未拉取时回退 daily_hs300 只返回
        HS300 集合并给出警告）。
        """
        import warnings
        d = self._cache.read_daily("all_a")
        if d is None:
            warnings.warn(
                "全A池 daily_all_a 缓存不存在（尚未拉取），回退返回空池。"
                "请先 python -m scripts.update_data --pool all_a",
                stacklevel=2,
            )
            return []
        codes = d.index.get_level_values("code").unique().tolist()
        if not isinstance(date, (int, str)):
            date = int(pd.Timestamp(date).strftime("%Y%m%d"))
        return codes

    def get_default(self, date: int) -> list[str]:
        """按 config 中 universe.default 返回。"""
        default = Config.universe().get("default", "hs300")
        mapping = {
            "hs300": self.get_hs300,
            "zz500": self.get_zz500,
            "zz1000": self.get_zz1000,
            "all_a": self.get_all_a,
        }
        if default not in mapping:
            raise ValueError(f"未知股票池: {default}（可选 {list(mapping)}）")
        return mapping[default](date)

    def get_all_constituents(self, index_code: str) -> pd.DataFrame:
        """返回完整成分股变动表（不做日期过滤）。"""
        return self._cache.get_index_constituent(index_code)

    def get_membership_mask(
        self,
        index_code: str,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """按日构建成分股归属矩阵 (date, code) -> bool，point-in-time，避免幸存者偏差。

        与逐日调用 get_constituent 等价，但用一次性 broadcasting 比较代替
        O(天数) 次独立查询，实现见 data.point_in_time.lookup_intervals。

        Returns:
            DataFrame(index=dates, columns=当期出现过的全部 con_code), dtype=bool。
        """
        df = self._cache.get_index_constituent(index_code)
        if df.empty:
            return pd.DataFrame(False, index=dates, columns=[])

        codes = df["con_code"].dropna().unique().tolist()
        return lookup_intervals(df, dates, codes, code_col="con_code")
