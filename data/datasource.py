"""
数据源抽象层
============

定义统一的 DataSource 接口，所有具体数据源（AmazingData / CSV / ...）
实现该接口。上层 cache / universe / factor / backtest 只依赖此接口，
切换数据源时只需改 config + 新增一个实现类，不动业务代码。

数据约定
--------
- 所有行情返回 multi-index DataFrame：(date, code) 索引，列统一为
  open / high / low / close / volume / amount。
- 所有日期统一为 pandas.Timestamp（内部用 int8 日期到 SDK 层转换）。
- 代码格式统一 "XXXXXX.SH/SZ/BJ"（6位+交易所后缀）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Iterable

import pandas as pd


class DataSource(ABC):
    """数据源抽象基类。"""

    # ---- 交易日历 ----
    @abstractmethod
    def get_calendar(self, begin: int = 20100101, end: int | None = None) -> list[int]:
        """返回交易日列表（8位整型 YYYYMMDD，升序）。"""

    # ---- 代码表 ----
    @abstractmethod
    def get_code_list(self, security_type: str = "EXTRA_STOCK_A") -> list[str]:
        """获取每日最新代码表。"""

    @abstractmethod
    def get_index_constituent(self, index_code: str) -> pd.DataFrame:
        """指数成分股。
        返回列: con_code, in_date, out_date。"""

    # ---- 行情 ----
    @abstractmethod
    def get_daily_kline(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        """日K线。返回 multi-index (date, code) DataFrame，
        列: open/high/low/close/volume/amount。"""

    # ---- 复权因子 ----
    @abstractmethod
    def get_adj_factor(self, code_list: Iterable[str]) -> pd.DataFrame:
        """单次复权因子。index=date, columns=code, values=adj_factor。"""

    # ---- 基础信息 ----
    @abstractmethod
    def get_code_info(self, security_type: str = "EXTRA_STOCK_A") -> pd.DataFrame:
        """每日最新证券信息。index=code, 列含 symbol/status/pre_close/
        high_limited/low_limited/price_tick。"""


# ---------------------------------------------------------------------------
# AmazingData 实现
# ---------------------------------------------------------------------------
class AmazingDataSource(DataSource):
    """银河证券 AmazingData SDK 封装。

    SDK 首次使用需 `pip install AmazingData-*.whl`（见开发手册 3.3）。
    实例化时自动 login，之后复用同一连接。
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._ad = None       # 模块句柄
        self._base = None     # BaseData 实例
        self._market = None   # MarketData 实例
        self._info = None     # InfoData 实例
        self._login(cfg)

    # ---- 登录 ----
    def _login(self, cfg: dict) -> None:
        import AmazingData as ad  # type: ignore

        self._ad = ad
        ad.login(
            username=cfg["username"],
            password=cfg["password"],
            host=cfg["host"],
            port=int(cfg["port"]),
        )
        self._base = ad.BaseData()
        calendar = self._base.get_calendar()
        self._market = ad.MarketData(calendar)
        self._info = ad.InfoData()

    # ---- 交易日历 ----
    def get_calendar(self, begin: int = 20100101, end: int | None = None) -> list[int]:
        cal = self._base.get_calendar()
        if end is None:
            return [d for d in cal if d >= begin]
        return [d for d in cal if begin <= d <= end]

    # ---- 代码表 ----
    def get_code_list(self, security_type: str = "EXTRA_STOCK_A") -> list[str]:
        return self._base.get_code_list(security_type=security_type)

    def get_index_constituent(self, index_code: str) -> pd.DataFrame:
        raw = self._info.get_index_constituent([index_code], is_local=False)
        df = raw[index_code]
        return df[["CON_CODE", "INDATE", "OUTDATE", "INDEX_NAME"]].rename(
            columns={"CON_CODE": "con_code", "INDATE": "in_date", "OUTDATE": "out_date"}
        )

    # ---- 日K线 ----
    def get_daily_kline(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        codes = list(code_list)
        kline_dict = self._market.query_kline(
            codes,
            begin_date=begin_date,
            end_date=end_date,
            period=self._ad.constant.Period.day.value,
        )
        frames = []
        for code, df in kline_dict.items():
            sub = df[["open", "high", "low", "close", "volume", "amount"]].copy()
            sub.insert(0, "code", code)
            frames.append(sub)
        if not frames:
            return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "volume", "amount"])
        out = pd.concat(frames)
        out.index.name = "date"
        out = out.reset_index().set_index(["date", "code"])
        return out.sort_index()

    # ---- 复权因子 ----
    def get_adj_factor(self, code_list: Iterable[str]) -> pd.DataFrame:
        cfg = self._cfg
        local_path = cfg.get("sdk_local_path", "e://data//sdk_cache//")
        # SDK 内部以 pickle 格式缓存到 local_path，首次 is_local=False 从服务端拉取并落地，
        # 之后 is_local=True 时直接读本地 pickle；两套缓存（SDK pickle / 自建 parquet）共存互补。
        return self._base.get_adj_factor(
            list(code_list), local_path=local_path, is_local=False
        )

    # ---- 基础信息 ----
    def get_code_info(self, security_type: str = "EXTRA_STOCK_A") -> pd.DataFrame:
        return self._base.get_code_info(security_type=security_type)


# ---------------------------------------------------------------------------
# CSV 备用数据源
# ---------------------------------------------------------------------------
class CSVDataSource(DataSource):
    """本地 CSV 目录数据源，目录结构: root/{table}/{code}.csv
    用于离线开发 / 无 SDK 凭证时 / 切换到第三方数据。"""

    def __init__(self, cfg: dict):
        self._root = Path(cfg["root"]) if cfg.get("root") else Path("data/csv")
        self._root.mkdir(parents=True, exist_ok=True)

    def get_calendar(self, begin: int = 20100101, end: int | None = None) -> list[int]:
        # 简化：从任意 K 线文件的日期列推断
        return []

    def get_code_list(self, security_type: str = "EXTRA_STOCK_A") -> list[str]:
        d = self._root / "daily"
        if not d.exists():
            return []
        return [f.stem for f in d.glob("*.csv")]

    def get_index_constituent(self, index_code: str) -> pd.DataFrame:
        p = self._root / f"index_constituent_{index_code}.csv"
        if p.exists():
            return pd.read_csv(p)
        return pd.DataFrame(columns=["con_code", "in_date", "out_date"])

    def get_daily_kline(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        frames = []
        for code in code_list:
            p = self._root / "daily" / f"{code}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p, parse_dates=["date"])
            df["code"] = code
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).set_index(["date", "code"])
        return out.sort_index()

    def get_adj_factor(self, code_list: Iterable[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def get_code_info(self, security_type: str = "EXTRA_STOCK_A") -> pd.DataFrame:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
from pathlib import Path

def create_datasource(cfg: dict | None = None) -> DataSource:
    """根据 config 创建数据源实例。"""
    from config import Config
    if cfg is None:
        cfg = Config.datasource()

    ds_type = cfg["type"]
    if ds_type == "amazing_data":
        return AmazingDataSource(cfg["amazing_data"])
    elif ds_type == "csv":
        return CSVDataSource(cfg["csv"])
    else:
        raise ValueError(f"未知数据源类型: {ds_type}")
