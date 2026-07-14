"""
本地 Parquet 缓存
=================

在 DataSource 之上加一层本地 Parquet 缓存，实现：
1. 增量更新：只从数据源拉取本地缺失的日期段。
2. 透明访问：上层调用与 DataSource 接口一致。
3. 离线研究：本地有数据时无需连接数据源。

存储布局
--------
cache_root/
├── daily.parquet          # 日K线 (date, code 多索引)
├── adj_factor.parquet    # 复权因子
├── calendar.parquet      # 交易日历
├── code_info.parquet     # 证券信息
├── index_constituent_000300SH.parquet  # 指数成分
└── _meta.json             # 各表最后更新日期
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import Config
from data.datasource import DataSource


class DataCache:
    """Parquet 本地缓存，封装增量更新逻辑。"""

    def __init__(self, ds: DataSource, cache_root: Path | str | None = None):
        self._ds = ds
        if cache_root is None:
            cache_root = Config.cache()["root"]
        # 统一为 OS 路径
        self._root = Path(str(cache_root).replace("//", "/"))
        self._root.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._root / "_meta.json"
        self._meta: dict = self._load_meta()

    # ---- 元数据 ----
    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            with open(self._meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_meta(self) -> None:
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2, ensure_ascii=False)

    def _get_last_date(self, table: str) -> int | None:
        v = self._meta.get(table, {}).get("last_date")
        return int(v) if v else None

    def _set_last_date(self, table: str, d: int) -> None:
        self._meta.setdefault(table, {})["last_date"] = d
        self._save_meta()

    # ---- 交易日历 ----
    def get_calendar(self, begin: int = 20100101, end: int | None = None) -> list[int]:
        p = self._root / "calendar.parquet"
        need_fetch = True
        if p.exists():
            df = pd.read_parquet(p)
            cal = sorted(df["date"].tolist())
            # 本地最新日期 >= end 时直接用本地
            if end is None or (cal and cal[-1] >= end):
                need_fetch = False
        if need_fetch:
            cal = self._ds.get_calendar(begin, end)
            pd.DataFrame({"date": cal}).to_parquet(p, compression="snappy")
            self._set_last_date("calendar", cal[-1] if cal else begin)
        else:
            cal = [d for d in cal if d >= begin and (end is None or d <= end)]
        return cal

    # ---- 日K线（增量更新核心）----
    def get_daily_kline(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        """获取日K线，本地缓存 + 增量补充。"""
        p = self._root / "daily.parquet"
        codes = list(code_list)

        # 1) 读本地
        local_df = pd.DataFrame()
        if p.exists():
            local_df = pd.read_parquet(p)
            # 过滤出本地已有但日期范围不足的代码
            if not local_df.empty:
                local_df = local_df.reset_index()
                # 只保留请求的 code
                local_df = local_df[local_df["code"].isin(codes)]
                local_df = local_df.set_index(["date", "code"]).sort_index()

        # 2) 判断需要增量的日期范围
        last = self._get_last_date("daily")
        fetch_begin = begin_date
        if last is not None:
            # 从 last 的下一个交易日开始拉
            fetch_begin = last + 1
        if fetch_begin > end_date:
            return local_df.xs(slice(None), level="code", drop_level=False) if not local_df.empty else local_df

        # 3) 从数据源拉增量
        new_df = self._ds.get_daily_kline(codes, fetch_begin, end_date)

        # 4. 合并、去重、落盘
        if not new_df.empty:
            combined = pd.concat([local_df, new_df])
            # 去重：保留最新
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            combined.to_parquet(p, compression="snappy")
            # 更新 meta
            max_date = combined.index.get_level_values("date").max()
            last_int = int(pd.Timestamp(max_date).strftime("%Y%m%d"))
            self._set_last_date("daily", last_int)
            return combined.xs(slice(None), level="code", drop_level=False).sort_index()
        else:
            return local_df

    # ---- 复权因子 ----
    def get_adj_factor(self, code_list: Iterable[str]) -> pd.DataFrame:
        p = self._root / "adj_factor.parquet"
        codes = list(code_list)
        local_df = pd.DataFrame()
        if p.exists():
            local_df = pd.read_parquet(p)
            # 只保留请求的 code（列）
            cols = [c for c in codes if c in local_df.columns]
            local_df = local_df[cols]

        # 增量：从数据源全量拉（SDK 内部也维护本地缓存）
        new_df = self._ds.get_adj_factor(codes)
        if not new_df.empty:
            combined = pd.concat([local_df, new_df], axis=1)
            combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
            combined.to_parquet(p, compression="snappy")
            return combined.sort_index()
        return local_df

    # ---- 指数成分 ----
    def get_index_constituent(self, index_code: str) -> pd.DataFrame:
        safe = index_code.replace(".", "")
        p = self._root / f"index_constituent_{safe}.parquet"
        if p.exists():
            return pd.read_parquet(p)
        df = self._ds.get_index_constituent(index_code)
        if not df.empty:
            df.to_parquet(p, compression="snappy")
        return df

    # ---- 证券信息 ----
    def get_code_info(self, security_type: str = "EXTRA_STOCK_A") -> pd.DataFrame:
        p = self._root / "code_info.parquet"
        # 每日信息直接覆盖（每日最新）
        df = self._ds.get_code_info(security_type)
        if not df.empty:
            df.to_parquet(p, compression="snappy")
        return df

    # ---- 代码表 ----
    def get_code_list(self, security_type: str = "EXTRA_STOCK_A") -> list[str]:
        return self._ds.get_code_list(security_type)

    @property
    def root(self) -> Path:
        return self._root
