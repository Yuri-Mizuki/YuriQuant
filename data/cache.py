"""
本地 Parquet 缓存
=================

在 DataSource 之上加一层本地 Parquet 缓存，实现：
1. 增量更新：只从数据源拉取本地缺失的日期段。
2. 透明访问：上层调用与 DataSource 接口一致。
3. 离线研究：本地有数据时无需连接数据源。

存储布局（cache_root 默认 e:/data/parquet/，扁平存放，每表一个 parquet）
------------------------------------------------------------------------
行情类（长表增量更新，MultiIndex (time, code)，_meta.json 记水位）
    daily.parquet                     # 日K线 (date, code)，OHLCV+amount
    min{period}.parquet               # 分钟K线 (kline_time, code)，如 min5；按档位分文件
    adj_factor.parquet                # 单次复权因子（宽表 date×code，全量刷新）
    backward_factor.parquet           # 累积后复权因子（宽表 date×code，全量刷新）
状态类（长表增量更新，MultiIndex (date, code)，记水位）
    history_stock_status.parquet      # 历史涨跌停/停牌/ST/除权除息标记
财务类（稀疏报告期事件表，整表覆盖 + code 过滤，记 ann_date 水位）
    income.parquet                    # 利润表
    balance_sheet.parquet             # 资产负债表
    cash_flow.parquet                 # 现金流量表
参考类（稀疏事件表，整表覆盖，无增量水位）
    calendar.parquet                  # 交易日历（合并去重）
    index_constituent_{code}.parquet  # 指数成分（如 000300SH）
    industry_classification_level{N}.parquet  # 行业分类（申万，N=级别）
    equity_structure.parquet          # 股本结构变动事件
    dividend.parquet                  # 分红送转
    share_holder.parquet              # 十大股东
    holder_num.parquet                # 股东户数
    code_info.parquet                 # 证券信息（当前未拉取则不落盘）
元数据
    _meta.json                        # 各表增量水位（last_date）/ 最近数据日期

命名约定（2026-08-05 起，约束【后续新增】表；存量文件名保持不变）
    <域>_<表>[_参数].parquet
    域前缀：quote=行情 / fin=财务 / status=状态 / ref=参考事件 / meta=元数据
    例：quote_min15、fin_income、ref_index_constituent_000300SH、
        ref_industry_level1（参数后缀统一放末尾，与类型一致）。
    完整映射表与规范见 README「数据层缓存」章节。
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

    # ---- 数据指纹（P1，2026-08-03：实验结果绑定数据版本）----
    def get_fingerprint(self) -> str:
        """数据指纹：综合各缓存表的 (last_date, 文件大小, mtime) 的稳定 hash。

        用途：实验管理（``research.experiments``）把每次结果的 ``data_fingerprint``
        与数据版本绑定 —— 指纹相同 = 同一份数据，结果可比；指纹变化 = 数据
        更新过，旧结果需重新验证。轻量实现：不读全表，只取 meta + 文件 stat。
        """
        import hashlib
        parts: list[str] = []
        for table, info in sorted(self._meta.items()):
            parts.append(f"{table}:{info.get('last_date', '')}")
        for p in sorted(self._root.glob("*.parquet")):
            try:
                st = p.stat()
                parts.append(f"{p.stem}:{st.st_size}:{int(st.st_mtime)}")
            except OSError:
                continue
        h = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:12]
        return h

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
        time_col: str = "date",
        last_inclusive: bool = False,
        bars_per_day: int | None = None,
    ) -> pd.DataFrame:
        """本地按 code 过滤 + 只从数据源拉取本地缺失的日期段 + 合并去重落盘。

        用于按 (time_col, code) 逐条记录、真正有"增量"概念的场景（日K线、
        历史涨跌停停牌状态、分钟K线等）。fetch_fn 签名为 (codes, begin_date, end_date)。

        time_col: 索引时间列名，日频为 "date"（00:00 时间戳），分钟频为
            "kline_time"（含时分的完整 datetime）。过滤/写盘按日期边界统一处理。
        last_inclusive: 为 True（分钟频）时增量起点取 ``min(begin, last)``（而非
            last+1）——请求早于缓存的历史可回补、last 当天重拉可补全半拉缺口。
        bars_per_day: 分钟频传每日完整 bar 数（240//period）。当请求区间日期范围
            已被本地覆盖时，用它检测"半拉天"（某交易日 bar 数不足）——存在半拉
            则仍重拉补全，否则短路不访问数据源（离线可用）。

        重要：写盘合并的是**全量本地数据**（所有 code / 所有日期），仅在返回值
        上按 (codes, [begin_date, end_date]) 过滤。早期实现把过滤后的子集写回
        parquet，一次窄区间查询就会永久丢失其余 code / 日期。
        """
        p = self._root / filename
        local_full = pd.DataFrame()
        if p.exists():
            local_full = pd.read_parquet(p)

        # ---- 确定增量拉取起点 ----
        last = self._get_last_date(table_name)
        fetch_begin = begin_date
        if last is not None:
            if last_inclusive:
                covered = False
                if not local_full.empty:
                    # 用交易日对齐判断"请求区间是否已被本地覆盖"：begin/end 可能是
                    # 节假日（如 20250101 元旦），按自然日比较会把 20250102 的本地
                    # 首日误判为未覆盖，触发无谓的 SDK 全量重拉。
                    local_days = set(
                        pd.to_datetime(
                            local_full.index.get_level_values(time_col)
                        ).normalize().strftime("%Y%m%d")
                    )
                    try:
                        req_days = [str(d) for d in self.get_calendar(begin_date, end_date)]
                        missing = [d for d in req_days if d not in local_days]
                        # req_days 为空（请求区间无交易日，如纯节假日或数据源日历
                        # 范围外）时视为"未覆盖"：保守走补拉，避免空日历误判短路
                        covered = bool(req_days) and not missing
                    except (AttributeError, NotImplementedError):
                        # 数据源无交易日历接口（如只实现了行情的 mock/桩）：
                        # 无法确认覆盖，保守按"未覆盖"走补拉（多拉不丢数据）
                        covered = False
                    if covered and bars_per_day is not None and time_col == "kline_time":
                        # 半拉天检测：请求区间内某 (交易日, code) 的 bar 数不足完整数
                        ts = local_full.index.get_level_values(time_col)
                        per_day = local_full.groupby(
                            [ts.normalize(), local_full.index.get_level_values("code")]
                        ).size()
                        incomplete = per_day[per_day < bars_per_day]
                        req_begin = pd.Timestamp(str(begin_date))
                        req_end = pd.Timestamp(str(end_date))
                        if any(
                            (d >= req_begin) & (d <= req_end)
                            for d in incomplete.index.get_level_values(0).unique()
                        ):
                            covered = False
                if covered:
                    # 请求区间已被本地完整覆盖：短路不拉（离线可用）
                    fetch_begin = int(
                        (pd.Timestamp(str(end_date)) + pd.Timedelta(days=1)).strftime("%Y%m%d")
                    )
                else:
                    # 补历史（begin < local_begin）或补半拉（last 当天重拉去重）
                    fetch_begin = min(begin_date, last)
            else:
                # last 是 int YYYYMMDD，直接 +1 会得到 20240132 这样的非法日期，
                # 必须经 Timestamp 加一天再转回 int。
                fetch_begin = int(
                    (pd.Timestamp(str(last)) + pd.Timedelta(days=1)).strftime("%Y%m%d")
                )
                # 历史回补：请求 begin 早于本地最早日期时，增量起点取 min(begin, last)。
                # 否则只从 last+1 拉，2022-2024 这类更早的历史缺口永远不会被补上
                # （2026-08-04 补拉 2022-2025 实测发现：daily 仍从 2025 起）。
                if isinstance(local_full.index, pd.MultiIndex) and not local_full.empty:
                    local_min = local_full.index.get_level_values(time_col).min()
                    local_min_int = int(pd.Timestamp(local_min).strftime("%Y%m%d"))
                    if begin_date < local_min_int:
                        fetch_begin = min(begin_date, last)
        # 若请求的 code 中有本地完全没有的（新上市票），从 begin_date 全量拉取，
        # 否则全局 last_date 会跳过这些票上市以来的全部历史。
        if not local_full.empty:
            cached_codes = set(local_full.index.get_level_values("code").unique())
            if cached_codes and any(c not in cached_codes for c in codes):
                fetch_begin = begin_date
        if fetch_begin > end_date:
            return self._filter_long(local_full, codes, begin_date, end_date, time_col)

        new_df = fetch_fn(codes, fetch_begin, end_date)
        if not new_df.empty:
            if not isinstance(new_df.index, pd.MultiIndex):
                new_df = new_df.set_index([time_col, "code"]).sort_index()
            # 合并全量本地 + 新数据后写盘（不丢历史）
            combined = pd.concat([local_full, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            combined.to_parquet(p, compression="snappy")
            max_ts = combined.index.get_level_values(time_col).max()
            last_int = int(pd.Timestamp(max_ts).strftime("%Y%m%d"))
            self._set_last_date(table_name, last_int)
            return self._filter_long(combined, codes, begin_date, end_date, time_col)
        return self._filter_long(local_full, codes, begin_date, end_date, time_col)

    @staticmethod
    def _filter_long(
        df: pd.DataFrame, codes, begin_date: int, end_date: int, time_col: str = "date"
    ) -> pd.DataFrame:
        """按 (codes, [begin_date, end_date]) 过滤长表，仅用于返回值。

        分钟频 time_col="kline_time" 含日内时分，上界必须取 end_date+1 天
        （否则 end_date 当天除 00:00 外的全部 bar 都会被滤掉）。
        """
        if df.empty or not isinstance(df.index, pd.MultiIndex):
            return df
        df = df[df.index.get_level_values("code").isin(codes)]
        ts = df.index.get_level_values(time_col)
        if not pd.api.types.is_datetime64_any_dtype(ts):
            # 兼容 int / str 型 YYYYMMDD 日期（pd.to_datetime(int) 会被当纳秒→1970）
            ts = pd.to_datetime(ts.astype(str), format="%Y%m%d", errors="coerce")
        start = pd.Timestamp(str(begin_date))
        end = pd.Timestamp(str(end_date)) + pd.Timedelta(days=1)
        df = df.loc[(ts >= start) & (ts < end)]
        return df.sort_index()

    # ---- 交易日历 ----
    def get_calendar(self, begin: int = 20100101, end: int | None = None) -> list[int]:
        p = self._root / "calendar.parquet"
        existing: list[int] = []
        if p.exists():
            existing = sorted(pd.read_parquet(p)["date"].tolist())
        need_fetch = True
        if existing and (end is None or existing[-1] >= end):
            need_fetch = False
        if need_fetch:
            fetched = self._ds.get_calendar(begin, end)
            # 合并新旧日历去重后再写盘，避免窄区间查询覆盖丢失全部历史
            merged = sorted(set(existing) | set(fetched))
            pd.DataFrame({"date": merged}).to_parquet(p, compression="snappy")
            self._set_last_date("calendar", merged[-1] if merged else begin)
            cal = merged
        else:
            cal = existing
        return [d for d in cal if d >= begin and (end is None or d <= end)]

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

    # ---- 分钟K线（日内研究，2026-08-03 新增）----
    def get_minute_kline(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
        period: int = 5,
    ) -> pd.DataFrame:
        """获取分钟K线，本地缓存 + 增量补充。

        period: 分钟数 {1,3,5,10,15,30,60,120}，缓存文件按档位分开
        （min5.parquet / min15.parquet ...），互不串扰。

        分钟 bar 的时间列是含时分的 kline_time（跨交易日增量按天对齐），
        增量起点取 min(begin, last) 当天（last_inclusive=True）——即使某天
        只按日内时段部分拉取过，重拉 + 去重也能补全，不会永久缺半天。
        """
        codes = list(code_list)
        filename = f"min{period}.parquet"
        table = f"min{period}"
        return self._refresh_long_table(
            filename, table, codes, begin_date, end_date,
            lambda c, b, e: self._ds.get_minute_kline(c, b, e, period=period),
            time_col="kline_time",
            last_inclusive=True,
            bars_per_day=240 // period,
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

    # ---- 分红 / 十大股东 / 股东户数（稀疏事件表，整表覆盖缓存，同股本结构）----
    def get_dividend(self, code_list: Iterable[str]) -> pd.DataFrame:
        p = self._root / "dividend.parquet"
        codes = list(code_list)
        df = self._ds.get_dividend(codes)
        if not df.empty:
            df.to_parquet(p, compression="snappy")
            return df
        if p.exists():
            cached = pd.read_parquet(p)
            return cached[cached["code"].isin(codes)] if "code" in cached.columns else cached
        return df

    def get_share_holder(self, code_list: Iterable[str]) -> pd.DataFrame:
        p = self._root / "share_holder.parquet"
        codes = list(code_list)
        df = self._ds.get_share_holder(codes)
        if not df.empty:
            df.to_parquet(p, compression="snappy")
            return df
        if p.exists():
            cached = pd.read_parquet(p)
            return cached[cached["code"].isin(codes)] if "code" in cached.columns else cached
        return df

    def get_holder_num(self, code_list: Iterable[str]) -> pd.DataFrame:
        p = self._root / "holder_num.parquet"
        codes = list(code_list)
        df = self._ds.get_holder_num(codes)
        if not df.empty:
            df.to_parquet(p, compression="snappy")
            return df
        if p.exists():
            cached = pd.read_parquet(p)
            return cached[cached["code"].isin(codes)] if "code" in cached.columns else cached
        return df

    # ---- 财务报表（稀疏报告期表，整表覆盖 + code 过滤）----
    def _get_financial(self, filename: str, table_name: str,
                       codes: list[str], fetch_fn) -> pd.DataFrame:
        p = self._root / filename
        df = fetch_fn(codes)
        if not df.empty:
            df.to_parquet(p, compression="snappy")
            self._meta.setdefault(table_name, {})["last_date"] = (
                int(pd.Timestamp(df["ann_date"].max()).strftime("%Y%m%d"))
                if "ann_date" in df.columns and not df["ann_date"].isna().all() else 0
            )
            self._save_meta()
            return df
        if p.exists():
            cached = pd.read_parquet(p)
            return cached[cached["code"].isin(codes)] if "code" in cached.columns else cached
        return df

    def get_balance_sheet(self, code_list: Iterable[str],
                          begin_date: int | None = None, end_date: int | None = None) -> pd.DataFrame:
        codes = list(code_list)
        return self._get_financial(
            "balance_sheet.parquet", "balance_sheet", codes, self._ds.get_balance_sheet
        )

    def get_cash_flow(self, code_list: Iterable[str],
                      begin_date: int | None = None, end_date: int | None = None) -> pd.DataFrame:
        codes = list(code_list)
        return self._get_financial(
            "cash_flow.parquet", "cash_flow", codes, self._ds.get_cash_flow
        )

    def get_income(self, code_list: Iterable[str],
                   begin_date: int | None = None, end_date: int | None = None) -> pd.DataFrame:
        codes = list(code_list)
        return self._get_financial(
            "income.parquet", "income", codes, self._ds.get_income
        )

    # ---- 代码表 ----
    def get_code_list(self, security_type: str = "EXTRA_STOCK_A") -> list[str]:
        return self._ds.get_code_list(security_type)

    @property
    def root(self) -> Path:
        return self._root
