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
├── adj_factor.parquet    # 单次复权因子
├── backward_factor.parquet  # 累积后复权因子
├── history_stock_status.parquet  # 历史涨跌停/停牌/ST (date, code 多索引)
├── industry_classification_level1.parquet  # 一级行业分类
├── equity_structure.parquet  # 股本结构变动事件表
├── calendar.parquet      # 交易日历
├── code_info.parquet     # 证券信息
├── index_constituent_000300SH.parquet  # 指数成分
└── _meta.json             # 各表最后更新日期
"""
from __future__ import annotations

import json
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

    # ---- 缓存模式：宽表全量刷新（index=date, columns=code）----
    def _refresh_wide_table(self, filename: str, codes: list[str], fetch_fn) -> pd.DataFrame:
        """本地按列过滤 + 从数据源全量拉取 + 按列去重落盘。

        用于 SDK 自身已维护增量缓存、调用方每次总是传整个 code_list 的场景
        （复权因子类接口），本地 parquet 只是这层再加一份离线可读的副本。
        """
        p = self._root / filename
        local_df = pd.DataFrame()
        if p.exists():
            local_df = pd.read_parquet(p)
            cols = [c for c in codes if c in local_df.columns]
            local_df = local_df[cols]

        new_df = fetch_fn(codes)
        if not new_df.empty:
            combined = pd.concat([local_df, new_df], axis=1)
            combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
            combined.to_parquet(p, compression="snappy")
            return combined.sort_index()
        return local_df

    # ---- 缓存模式：长表增量更新（(date, code) 多索引）----
    def _refresh_long_table(
        self,
        filename: str,
        table_name: str,
        codes: list[str],
        begin_date: int,
        end_date: int,
        fetch_fn,
    ) -> pd.DataFrame:
        """本地按 code 过滤 + 只从数据源拉取本地缺失的日期段 + 合并去重落盘。

        用于按 (date, code) 逐日记录、真正有"增量"概念的场景（日K线、
        历史涨跌停停牌状态等）。fetch_fn 签名为 (codes, begin_date, end_date)。
        """
        p = self._root / filename
        local_df = pd.DataFrame()
        if p.exists():
            local_df = pd.read_parquet(p)
            if not local_df.empty:
                local_df = local_df.reset_index()
                local_df = local_df[local_df["code"].isin(codes)]
                local_df = local_df.set_index(["date", "code"]).sort_index()

        last = self._get_last_date(table_name)
        fetch_begin = begin_date
        if last is not None:
            fetch_begin = last + 1
        if fetch_begin > end_date:
            return local_df

        new_df = fetch_fn(codes, fetch_begin, end_date)
        if not new_df.empty:
            if not isinstance(new_df.index, pd.MultiIndex):
                new_df = new_df.set_index(["date", "code"]).sort_index()
            combined = pd.concat([local_df, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            combined.to_parquet(p, compression="snappy")
            max_date = combined.index.get_level_values("date").max()
            last_int = int(pd.Timestamp(max_date).strftime("%Y%m%d"))
            self._set_last_date(table_name, last_int)
            return combined
        return local_df

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
        codes = list(code_list)
        return self._refresh_long_table(
            "daily.parquet", "daily", codes, begin_date, end_date, self._ds.get_daily_kline
        )

    # ---- 复权因子 ----
    def get_adj_factor(self, code_list: Iterable[str]) -> pd.DataFrame:
        codes = list(code_list)
        return self._refresh_wide_table("adj_factor.parquet", codes, self._ds.get_adj_factor)

    def get_backward_factor(self, code_list: Iterable[str]) -> pd.DataFrame:
        """累积后复权因子，缓存模式同 get_adj_factor（宽表全量刷新）。"""
        codes = list(code_list)
        return self._refresh_wide_table(
            "backward_factor.parquet", codes, self._ds.get_backward_factor
        )

    # ---- 历史涨跌停/停牌/ST ----
    def get_history_stock_status(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        """按日历史证券状态，增量更新模式同 get_daily_kline（长表 (date, code) 索引）。"""
        codes = list(code_list)
        return self._refresh_long_table(
            "history_stock_status.parquet", "history_stock_status",
            codes, begin_date, end_date, self._ds.get_history_stock_status,
        )

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

    # ---- 行业分类 ----
    def get_industry_classification(self, level: int = 1) -> pd.DataFrame:
        """行业分类表体量小，整表覆盖缓存（同 get_index_constituent）。"""
        p = self._root / f"industry_classification_level{level}.parquet"
        if p.exists():
            return pd.read_parquet(p)
        df = self._ds.get_industry_classification(level)
        if not df.empty:
            df.to_parquet(p, compression="snappy")
        return df

    # ---- 股本结构 ----
    def get_equity_structure(self, code_list: Iterable[str]) -> pd.DataFrame:
        """稀疏事件表，没有"增量"概念，整表覆盖缓存（同 get_code_info）。"""
        p = self._root / "equity_structure.parquet"
        codes = list(code_list)
        df = self._ds.get_equity_structure(codes)
        if not df.empty:
            df.to_parquet(p, compression="snappy")
            return df
        if p.exists():
            cached = pd.read_parquet(p)
            return cached[cached["code"].isin(codes)]
        return df

    # ---- 代码表 ----
    def get_code_list(self, security_type: str = "EXTRA_STOCK_A") -> list[str]:
        return self._ds.get_code_list(security_type)

    @property
    def root(self) -> Path:
        return self._root
