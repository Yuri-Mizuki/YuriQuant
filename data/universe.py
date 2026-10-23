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
        """全 A 池：本地缓存 daily_all_a 中**该日处于上市区间**的代码。

        2026-10-19 审计修复：此前 ``date`` 只做类型转换后丢弃（返回"曾出现过的
        全部代码"），与 docstring 的 PIT 语义不符。现按首末行情日过滤：
        ``first_date <= date <= last_date``（末行近似退市/长期停牌边界）。

        2026-10-23 回归修复：晚间 update_data 以 ``target=今天`` 调本方法，而
        缓存数据止于上一交易日 ⇒ 右界 ``last >= 今天`` 恒假、返回**空池** →
        update 误判「本地无缓存」回退 API 清单 + 日K按 begin_date 走 4 年
        全量拉 → 超 1800s 预算被杀 → 数据永远落后一天（死循环；10-20~22
        三晚出榜故障根因）。修法：``t`` 钳制到缓存末日——数据滞后时按
        「截至已知数据」判断在市，PIT 语义不变（末行仍近似退市边界）。

        缓存中代码上限即全市场 A 股（约 5400 只；daily_all_a 未拉取时返回空池
        并给出警告）。
        """
        import warnings
        d = self._cache.read_daily("all_a")
        if d is None:
            warnings.warn(
                "全A池 daily_all_a 缓存不存在（尚未拉取），回退返回空池。"
                "请先 python -m scripts.ingest.update_data --pool all_a",
                stacklevel=2,
            )
            return []
        if len(d) == 0:
            return []
        # PIT 近似：首末行情日界定上市区间（末行 = 退市/最后交易日）
        dts = pd.DatetimeIndex(d.index.get_level_values("date"))
        cds = d.index.get_level_values("code")
        by_code = pd.Series(dts, index=cds)
        first = by_code.groupby(level=0).min()
        last = by_code.groupby(level=0).max()
        t = (pd.Timestamp(str(date)) if isinstance(date, (int, str))
             else pd.Timestamp(date))
        t = min(t, dts.max())     # 缓存滞后日右界钳制（见 docstring 回归修复）
        keep = (first <= t) & (last >= t)
        return first.index[keep].tolist()

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
