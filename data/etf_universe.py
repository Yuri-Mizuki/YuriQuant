"""
ETF 资产池（ETF Universe）
=========================

在 DataCache 之上封装 ETF 池与价格面板构建，供 ETF 轮动策略使用。

- 候选池：宽基 + 行业 ETF（MVP 精选 15 只，见 ``ETF_CANDIDATES``）。
- 价格面板：日频 close（可选后复权）与收益率，date × code 宽表，与项目
  既有因子/回测的 date × code 面板约定一致。
- 复权口径：动量收益用后复权价（raw close × backward_factor），避免 ETF
  分红除息在价格序列上造成的虚假回撤扭曲动量（对齐项目工程约定）。
"""
from __future__ import annotations

import pandas as pd

from config import Config
from data.cache import DataCache

# 候选池：code -> (名称, 类别)
ETF_CANDIDATES: dict[str, tuple[str, str]] = {
    # 宽基
    "510300.SH": ("沪深300ETF", "宽基"),
    "510050.SH": ("上证50ETF", "宽基"),
    "510500.SH": ("中证500ETF", "宽基"),
    "512100.SH": ("中证1000ETF", "宽基"),
    "159915.SZ": ("创业板ETF", "宽基"),
    "588000.SH": ("科创50ETF", "宽基"),
    # 行业 / 主题
    "512010.SH": ("医药ETF", "行业"),
    "512170.SH": ("医疗ETF", "行业"),
    "512690.SH": ("白酒ETF", "行业"),
    "512880.SH": ("证券ETF", "行业"),
    "512480.SH": ("半导体ETF", "行业"),
    "515030.SH": ("新能源车ETF", "行业"),
    "512400.SH": ("有色金属ETF", "行业"),
    "512660.SH": ("军工ETF", "行业"),
    "512200.SH": ("房地产ETF", "行业"),
}

ETF_TABLE = "etf"  # 缓存表名 → daily_{pool}.parquet


class EtfUniverse:
    """ETF 资产池管理器。"""

    def __init__(self, cache: DataCache):
        self._cache = cache

    def candidate_list(self) -> list[str]:
        return list(ETF_CANDIDATES.keys())

    def label(self, code: str) -> str:
        return ETF_CANDIDATES.get(code, (code, ""))[0]

    def category(self, code: str) -> str:
        return ETF_CANDIDATES.get(code, ("", ""))[1]

    def load_close(self, adjust: bool = True) -> pd.DataFrame:
        """返回 ETF 日频收盘价面板（date × code）。

        Args:
            adjust: True 返回后复权价（raw × backward_factor）；False 返回原始价。
        """
        d = self._cache.read_daily(ETF_TABLE)
        if d is None or d.empty:
            raise RuntimeError(
                "ETF 行情缓存为空，请先运行 python -m scripts.update_etf"
            )
        close = d["close"].unstack("code").sort_index()

        if adjust:
            bf_raw = self._cache.get_backward_factor(self.candidate_list())
            if not bf_raw.empty:
                bf = bf_raw.reindex(index=close.index, columns=close.columns)
                close = close * bf
            # 后复权因子缺失（如初始化失败）时退回原始价，动量口径仍可用
        return close

    def load_returns(self, adjust: bool = True, fill_method: str | None = None) -> pd.DataFrame:
        """返回 ETF 日频收益率面板（date × code）。"""
        close = self.load_close(adjust=adjust)
        return close.pct_change(fill_method=fill_method)

    def load_momentum(self, n: int = 20, adjust: bool = True) -> pd.DataFrame:
        """返回 N 日动量分面板 = 后复权收盘价 pct_change(n)（date × code）。"""
        close = self.load_close(adjust=adjust)
        return close.pct_change(n)