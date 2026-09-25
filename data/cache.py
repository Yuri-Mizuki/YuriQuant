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
    daily_{pool}.parquet             # 日K线（按池分文件，如 daily_hs300.parquet）(date, code)
    min{period}_{pool}.parquet       # 分钟K线 (kline_time, code)，如 min5_hs300；按档位+池分文件
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
    index_constituent_{code}.parquet  # 指数成分（如 000300_SH，点转下划线）
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

import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import Config
from data.datasource import DataSource

log = logging.getLogger("data.cache")


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

    def _get_refreshed_on(self, table: str) -> str:
        """表最近一次从数据源刷新的日期（YYYY-MM-DD）；无记录返回空串。"""
        return str(self._meta.get(table, {}).get("refreshed_on", ""))

    def _set_refreshed_on(self, table: str, day: str | None = None) -> None:
        self._meta.setdefault(table, {})["refreshed_on"] = (
            day or dt.date.today().isoformat()
        )
        self._save_meta()

    def _refresh_stale(self, table: str, max_age_days: int) -> bool:
        """判断整表覆盖型 PIT 表是否超过刷新间隔、需要回源。

        以 ``refreshed_on``（本地墙钟日期）为水位：成分/行业这类事件表没有
        自带的时间戳可比对，只能用"上次拉取距今多久"近似——超过 max_age_days
        或从未记录过刷新时间即视为过期。离线/无数据源环境下由调用方捕获异常。
        """
        last = self._get_refreshed_on(table)
        if not last:
            return True
        try:
            age = (dt.date.today() - dt.date.fromisoformat(last)).days
        except ValueError:
            return True
        return age >= max_age_days

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
    def _wide_table_last_date(self, p: Path) -> int | None:
        """宽表本地已覆盖的最后日期（YYYYMMDD）。

        只读 parquet 索引（``columns=[]``）、**不读数值列**——405MB 的宽表取
        索引仅约 0.3s。用作"是否需要回源"的短路判据（见 _refresh_wide_table）。
        """
        if not p.exists():
            return None
        try:
            idx = pd.read_parquet(p, columns=[]).index
        except Exception:
            return None
        if len(idx) == 0:
            return None
        try:
            return int(pd.Timestamp(idx.max()).strftime("%Y%m%d"))
        except (ValueError, TypeError):
            return None

    def _refresh_wide_table(
        self,
        filename: str,
        codes: list[str],
        fetch_fn,
        upto: int | None = None,
    ) -> pd.DataFrame:
        """本地列全保留 + 从数据源全量拉取 + 按列去重落盘。

        用于 SDK 自身已维护增量缓存、调用方每次总是传整个 code_list 的场景
        （复权因子类接口），本地 parquet 只是这层再加一份离线可读的副本。

        本地列**不**按本次请求的 codes 过滤：窄池请求（如每日增量更新只传
        当期成分并集）若过滤落盘，会永久丢弃历史成员的复权因子列，下次
        重建因子面板时这些股票的后复权价全变 NaN（幸存者偏差）。

        upto: 目标日期（YYYYMMDD）。本地表已覆盖该日期时短路返回本地数据、
            **完全不访问数据源**。复权因子的 SDK 接口没有日期参数（签名只有
            code_list/local_path/is_local），SDK 侧每次调用都是全量拉取并覆写
            自己的 h5（实测单次约 6 分钟、峰值内存约 2.8GB）——因此"本地已
            覆盖目标日"是唯一可用的短路条件：同日重复运行（重跑/补跑）不必
            重拉。不传（None）时保持原行为，总是回源。
        """
        p = self._root / filename
        local_df = pd.DataFrame()
        if p.exists():
            local_df = pd.read_parquet(p)

        if upto is not None:
            last = self._wide_table_last_date(p)
            if last is not None and last >= int(upto):
                cols = [c for c in codes if c in local_df.columns]
                return local_df[cols].sort_index() if cols else local_df

        new_df = fetch_fn(codes)
        if not new_df.empty:
            # 内存友好合并：只并 local 独有列。
            # 原实现 ``pd.concat([local, new], axis=1)`` 会在 5807 列上把两份
            # 405MB 宽表整份复制、再做一次布尔掩码全量拷贝（峰值约 2.8GB），是
            # 2026-09-16 两次静默死亡（疑似 OOM）的直接原因。列集合、列顺序与
            # 索引语义（union）均与原实现逐位等价——同名列以 new 为准，等价于
            # 原先的 duplicated(keep="last")（见 scripts/oneoff/
            # _probe_wide_table_merge_eq.py，14 组穷举输入 max|Δ|=0）。
            local_only = local_df.columns.difference(new_df.columns, sort=False)
            if not len(local_only) and bool(local_df.index.isin(new_df.index).all()):
                # 快路径：new 已覆盖 local 的全部列与行——零拷贝，不并表
                combined = new_df
            else:
                # local 有独有列、或独有日期：并表（索引取 union，与旧实现一致）
                combined = pd.concat([local_df[local_only], new_df], axis=1)
            combined.to_parquet(p, compression="snappy")
            local_df = combined
        cols = [c for c in codes if c in local_df.columns]
        return local_df[cols].sort_index() if cols else local_df

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
                    # 请求 begin 可能是节假日（如 20250101 元旦、20190101 元旦）——
                    # 用「请求区间首个交易日」与本地最早日期对齐比较，否则
                    # begin(1/1) < local_min(1/2) 会误判历史缺口，无谓触发 SDK
                    # 全量下载 + 写盘（2026-08-14 四窗口回测踩坑：PermissionError
                    # 写 e:\data\parquet 被锁/沙箱拦截）。
                    req_first = begin_date
                    try:
                        cal = self.get_calendar(begin_date, end_date)
                        if cal:
                            req_first = cal[0]
                    except (AttributeError, NotImplementedError):
                        pass
                    if req_first < local_min_int:
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
        # end=None 的语义是"覆盖到今天"（2026-09-08 修复：此前 None 直接短路
        # 不回源，日历被历史某次显式 end 调用封顶——如 20260902——之后所有
        # "更新到最新"的调用永远看不到新交易日）。
        end_eff = end if end is not None else int(
            pd.Timestamp.now().strftime("%Y%m%d"))
        need_fetch = True
        if existing and existing[-1] >= end_eff:
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
        return [d for d in cal if d >= begin and d <= end_eff]

    # ---- 日K线（增量更新核心）----
    def read_daily(self, pool: str | None = None) -> pd.DataFrame | None:
        """直接读日K线缓存文件（不做增量拉取），池名默认取 config。

        Returns:
            缓存 DataFrame 或 None（文件不存在）。
        """
        pool = pool or Config.universe().get("default", "hs300")
        p = self._root / f"daily_{pool}.parquet"
        if not p.exists():
            return None
        return pd.read_parquet(p)

    def read_minute_kline(self, pool: str | None = None, period: int = 5) -> pd.DataFrame | None:
        """直接读分钟K线缓存文件（不做增量拉取），池名默认取 config。

        供 data.intraday.MinutePanelStore.build 离线物化面板用——只读不拉，
        无 SDK 凭证也能工作（get_minute_kline 在缓存完整覆盖时也会短路，
        但 read 路径语义更明确、不会在节假日边界误触发回源）。
        """
        pool = pool or Config.universe().get("default", "hs300")
        p = self._root / f"min{period}_{pool}.parquet"
        if not p.exists():
            return None
        return pd.read_parquet(p)

    def get_daily_kline(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
        pool: str | None = None,
    ) -> pd.DataFrame:
        """获取日K线，本地缓存 + 增量补充。

        pool: 股票池（hs300/zz500/zz1000/all_a），缓存文件按池分文件
            （daily_hs300.parquet / daily_all_a.parquet ...），互不串扰。
            不传则取 config.universe.default（2026-08-26 池隔离扩展）。
        """
        codes = list(code_list)
        pool = pool or Config.universe().get("default", "hs300")
        filename = f"daily_{pool}.parquet"
        table = f"daily_{pool}"
        return self._refresh_long_table(
            filename, table, codes, begin_date, end_date, self._ds.get_daily_kline
        )

    # ---- 指数日K线（2026-08-24 新增：官方指数基准，如沪深300 000300.SH）----
    def get_index_daily(
        self,
        index_code: str,
        begin_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        """获取指数日K线，本地缓存 + 增量补充。

        与个股 K 线共用上层 query_kline（SDK 支持指数代码），但缓存独立存放
        （index_daily_{code}.parquet），避免与 daily.parquet（个股）混合。
        返回面板 index=(date, code)，含 OHLCV+amount。
        """
        safe = str(index_code).replace(".", "_")
        filename = f"index_daily_{safe}.parquet"
        table = f"index_daily_{safe}"
        return self._refresh_long_table(
            filename, table, [str(index_code)],
            begin_date, end_date, self._ds.get_daily_kline,
        )

    # ---- 分钟K线（日内研究，2026-08-03 新增）----
    def get_minute_kline(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
        period: int = 5,
        pool: str | None = None,
    ) -> pd.DataFrame:
        """获取分钟K线，本地缓存 + 增量补充。

        period: 分钟数 {1,3,5,10,15,30,60,120}，缓存文件按档位+池分开
        （min5_hs300.parquet / min5_all_a.parquet ...），互不串扰。

        分钟 bar 的时间列是含时分的 kline_time（跨交易日增量按天对齐），
        增量起点取 min(begin, last) 当天（last_inclusive=True）——即使某天
        只按日内时段部分拉取过，重拉 + 去重也能补全，不会永久缺半天。
        """
        codes = list(code_list)
        pool = pool or Config.universe().get("default", "hs300")
        filename = f"min{period}_{pool}.parquet"
        table = f"min{period}_{pool}"
        return self._refresh_long_table(
            filename, table, codes, begin_date, end_date,
            lambda c, b, e: self._ds.get_minute_kline(c, b, e, period=period),
            time_col="kline_time",
            last_inclusive=True,
            bars_per_day=240 // period,
        )

    # ---- 复权因子 ----
    def get_adj_factor(self, code_list: Iterable[str],
                       upto: int | None = None) -> pd.DataFrame:
        """单次复权因子宽表（index=date, columns=code）。

        upto: 目标日期 YYYYMMDD；本地已覆盖该日期则短路、不访问数据源
            （SDK 接口无日期参数，每次回源都是全量拉取，见 _refresh_wide_table）。
        """
        codes = list(code_list)
        return self._refresh_wide_table(
            "adj_factor.parquet", codes, self._ds.get_adj_factor, upto=upto
        )

    def get_backward_factor(self, code_list: Iterable[str],
                            upto: int | None = None) -> pd.DataFrame:
        """累积后复权因子，缓存模式同 get_adj_factor（宽表全量刷新）。

        upto 语义同 get_adj_factor：本地已覆盖目标日期则短路不回源。
        """
        codes = list(code_list)
        return self._refresh_wide_table(
            "backward_factor.parquet", codes, self._ds.get_backward_factor, upto=upto
        )

    # ---- 历史涨跌停/停牌/ST ----
    #: SDK 对大代码清单的单次状态查询会硬崩宿主进程且无 traceback
    #: （2026-08-28 实证：5550 只单查挂死，见 scripts/ingest/fetch_status_batched.py）。
    #: 缓存层统一分批 + 重试，调用方（update_data 等）无需各自实现。
    STATUS_BATCH = 200

    def _batched_status_fetch(self, codes: list[str], begin_date: int,
                              end_date: int) -> pd.DataFrame:
        import time as _time

        frames: list[pd.DataFrame] = []
        for i in range(0, len(codes), self.STATUS_BATCH):
            batch = codes[i:i + self.STATUS_BATCH]
            df = None
            for attempt in (1, 2, 3):
                try:
                    df = self._ds.get_history_stock_status(batch, begin_date, end_date)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    _time.sleep(5 * attempt)
            if df is not None and len(df):
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=0)
        return out.sort_index()

    def get_history_stock_status(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        """按日历史证券状态，增量更新模式同 get_daily_kline（长表 (date, code) 索引）。

        数据源调用分批（``STATUS_BATCH``）+ 重试——SDK 大清单单查会挂死；
        合并后仍走单次 ``_refresh_long_table`` 增量合并落盘。
        """
        codes = list(code_list)
        fetch = self._batched_status_fetch if len(codes) > self.STATUS_BATCH \
            else self._ds.get_history_stock_status
        return self._refresh_long_table(
            "history_stock_status.parquet", "history_stock_status",
            codes, begin_date, end_date, fetch,
        )

    # ---- 指数成分 ----
    # 成分是 PIT 事件表（in_date/out_date 全量历史），但**新纳入的成员/新指数
    # 只有回源才能拿到**——文件存在就直接读会让"本地首次拉取后成分集被冻结"。
    # 刷新策略：meta 记 refreshed_on（墙钟日期），超过 REFRESH_DAYS 回源整表
    # 合并去重（保留旧事件 + 追加新事件），离线数据源拉取失败时静默回退缓存。
    INDEX_CONSTITUENT_REFRESH_DAYS = 7

    def get_index_constituent(self, index_code: str) -> pd.DataFrame:
        safe = index_code.replace(".", "_")
        table = f"index_constituent_{safe}"
        return self._refresh_pit_event_table(
            f"{table}.parquet", table,
            fetch_fn=lambda: self._ds.get_index_constituent(index_code),
            merge_fn=self._merge_membership_events,
            max_age_days=self.INDEX_CONSTITUENT_REFRESH_DAYS,
        )

    @staticmethod
    def _merge_membership_events(cached: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        """成员关系事件表合并：按全部列去重（新旧重叠期的事件一致时仅保留一份）。"""
        if cached is None or cached.empty:
            return new
        combined = pd.concat([cached, new], ignore_index=True)
        return combined.drop_duplicates().reset_index(drop=True)

    # ---- 证券信息 ----
    def get_code_info(self, security_type: str = "EXTRA_STOCK_A") -> pd.DataFrame:
        p = self._root / "code_info.parquet"
        # 每日信息直接覆盖（每日最新）
        df = self._ds.get_code_info(security_type)
        if not df.empty:
            df.to_parquet(p, compression="snappy")
        return df

    # ---- 行业分类（PIT 事件表，刷新策略同指数成分）----
    def get_industry_classification(self, level: int = 1) -> pd.DataFrame:
        table = f"industry_classification_level{level}"
        return self._refresh_pit_event_table(
            f"{table}.parquet", table,
            fetch_fn=lambda: self._ds.get_industry_classification(level),
            merge_fn=self._merge_membership_events,
            max_age_days=self.INDEX_CONSTITUENT_REFRESH_DAYS,
        )

    def _refresh_pit_event_table(
        self,
        filename: str,
        table: str,
        fetch_fn,
        merge_fn,
        max_age_days: int,
    ) -> pd.DataFrame:
        """PIT 事件表通用读取：过期回源合并，失败/离线回退本地。

        - 本地无文件：必须回源（拿不到就返回空表，调用方按空处理）；
        - 本地有且未过期（refreshed_on 距今 < max_age_days）：直接读缓存，
          不访问数据源（离线研究可用）；
        - 本地有但过期：尝试回源；成功则与旧表**合并去重**落盘（事件表只增
          不删——旧行是历史真相，删了会造成 PIT 缺口），失败（无 SDK 凭证 /
          离线桩 / 网络故障）则回退本地旧表并打日志。
        """
        p = self._root / filename
        has_local = p.exists()
        need_fetch = (not has_local) or self._refresh_stale(table, max_age_days)
        if not need_fetch:
            return pd.read_parquet(p)

        try:
            df = fetch_fn()
        except Exception as e:  # noqa: BLE001
            if not has_local:
                raise
            log.warning("刷新 %s 失败（%s），回退本地缓存（%s）",
                        table, type(e).__name__, self._get_refreshed_on(table))
            return pd.read_parquet(p)

        if df.empty:
            if has_local:
                log.warning("%s 回源返回空表，保留本地缓存", table)
                return pd.read_parquet(p)
            return df

        if has_local:
            df = merge_fn(pd.read_parquet(p), df)
        df.to_parquet(p, compression="snappy")
        self._set_refreshed_on(table)
        return df

    # ---- 股本结构 ----
    def _get_sparse_table(self, filename: str, codes: list[str],
                          fetch_fn) -> pd.DataFrame:
        """稀疏事件表（股本/分红/十大股东/股东户数）：分批 + 每批重试 + 每批落盘。

        与 ``_get_financial`` 同因同治（见其注释）：整池一次性请求时，服务端
        偶发抖动会让整张表拉取失败、update_data 直接 exit；分批后单批抖动
        最多损失一批，已得批次不受影响，重跑即断点续拉。
        """
        p = self._root / filename
        parts, failed, aborted = self._fetch_batched(filename, codes, fetch_fn)
        if parts:
            if failed or aborted:
                log.warning("%s 有 %d 批未取到（熔断=%s），重跑本步骤可断点续拉",
                            filename, len(failed), aborted)
            return pd.concat(parts, ignore_index=True)
        if p.exists():
            cached = pd.read_parquet(p)
            return cached[cached["code"].isin(codes)] if "code" in cached.columns else cached
        return pd.DataFrame()

    def get_equity_structure(self, code_list: Iterable[str]) -> pd.DataFrame:
        """稀疏事件表，没有"增量"概念，整表覆盖缓存（同 get_code_info）。"""
        codes = list(code_list)
        return self._get_sparse_table(
            "equity_structure.parquet", codes, self._ds.get_equity_structure
        )

    # ---- 分红 / 十大股东 / 股东户数（稀疏事件表，整表覆盖缓存，同股本结构）----
    def get_dividend(self, code_list: Iterable[str]) -> pd.DataFrame:
        codes = list(code_list)
        return self._get_sparse_table(
            "dividend.parquet", codes, self._ds.get_dividend
        )

    def get_share_holder(self, code_list: Iterable[str]) -> pd.DataFrame:
        codes = list(code_list)
        return self._get_sparse_table(
            "share_holder.parquet", codes, self._ds.get_share_holder
        )

    def get_holder_num(self, code_list: Iterable[str]) -> pd.DataFrame:
        codes = list(code_list)
        return self._get_sparse_table(
            "holder_num.parquet", codes, self._ds.get_holder_num
        )

    # ---- 财务报表（稀疏报告期表，整表覆盖 + code 过滤）----
    def _merge_sparse_table(self, p: Path, codes: list[str], df: pd.DataFrame) -> pd.DataFrame:
        """稀疏事件表落盘：整表覆盖请求 code 的行，但**保留未请求 code 的本地行**。

        窄池请求（如每日增量更新只传当期成分并集）若直接整表覆盖，历史成员的
        财务/股东事件会被永久清掉，下次基本面因子构建即幸存者偏差。返回合并
        后的完整表（水位计算用），调用方返回值仍按请求 code 过滤。
        """
        if p.exists():
            cached = pd.read_parquet(p)
            if "code" in cached.columns:
                keep = cached[~cached["code"].isin(codes)]
                if len(keep):
                    df = pd.concat([keep, df], ignore_index=True)
        df.to_parquet(p, compression="snappy")
        return df

    # 财务三表（利润表/资产负债表/现金流量表）单次请求的代码批量。
    # 2026-09-22：原先整池一次性请求，服务端偶发
    #   error_code<301010> 查询语句异常：… ORA-00942: table or view does not exist
    # （同一时刻同池同接口重试即成功，判定为服务端瞬时抖动，非权限/表缺失）。
    # 一次性请求下，这种抖动会让**整张表**拉取失败、update_data 直接 exit 1，
    # 前面已跑完的行情/行业/股本全部作废。改为分批 + 每批独立重试 + 每批即落盘：
    # 单批抖动最多损失 100 只，且已得批次不受影响（断点续拉由本地 parquet 兜底）。
    _FINANCIAL_BATCH = 100
    _FINANCIAL_ATTEMPTS = 4
    # 连续多少批「重试耗尽仍全败」就判定该表当前不可用、放弃剩余批次。
    # 单批 4 次重试（退避 2/4/8s）已能滤掉瞬时抖动；**连续两批**全败基本只有
    # 「服务端该表本身不可用」一种解释（2026-09-22 实测：资产负债表整表
    # ORA-00942，而同池同口径 20 分钟前还全部成功——服务端在抖）。
    # 不熔断的话，全 A（5807 只 = 59 批）会在这张坏表上白烧约 2 小时。
    _FINANCIAL_CIRCUIT_BREAK = 2

    def _fetch_batched(self, filename: str, codes: list[str], fetch_fn):
        """分批 + 每批重试 + 每批落盘；返回 (已得 parts, 受限批次, 是否熔断)。

        ``parts`` 只含真正取到的批次；``failed`` 是重试耗尽仍失败的批次；
        熔断触发后剩余批次直接不再尝试（省掉在坏表上的等待）。
        """
        batches = [codes[i:i + self._FINANCIAL_BATCH]
                   for i in range(0, len(codes), self._FINANCIAL_BATCH)]
        parts: list[pd.DataFrame] = []
        failed: list[list[str]] = []
        consecutive_fail = 0
        aborted = False

        for bi, sub in enumerate(batches):
            part: pd.DataFrame | None = None
            for attempt in range(1, self._FINANCIAL_ATTEMPTS + 1):
                try:
                    part = fetch_fn(sub)
                    break
                except Exception as exc:  # noqa: BLE001 - 服务端偶发，重试后再定论
                    if attempt == self._FINANCIAL_ATTEMPTS:
                        log.warning("财务表 %s 批次 %d/%d（%d 只）%d 次重试仍失败，跳过: %s",
                                    filename, bi + 1, len(batches), len(sub), attempt, exc)
                        failed.append(sub)
                    else:
                        log.warning("财务表 %s 批次 %d/%d（%d 只）第 %d 次失败，%.0fs 后重试: %s",
                                    filename, bi + 1, len(batches), len(sub), attempt,
                                    2 ** attempt, exc)
                        time.sleep(2 ** attempt)
            if part is None or part.empty:
                consecutive_fail += 1
                if consecutive_fail >= self._FINANCIAL_CIRCUIT_BREAK:
                    rest = len(batches) - bi - 1
                    if rest:
                        log.warning("财务表 %s 连续 %d 批全败，判定该表当前不可用，"
                                    "放弃剩余 %d 批（共 %d 只）以省掉无效等待；"
                                    "服务端恢复后重跑本步骤即可断点续拉",
                                    filename, consecutive_fail, rest,
                                    sum(len(b) for b in batches[bi + 1:]))
                    aborted = True
                    break
                continue
            consecutive_fail = 0
            # 每批即合并落盘——后续批次全部失败也不丢已得数据
            self._merge_sparse_table(self._root / filename, sub, part)
            parts.append(part)
        return parts, failed, aborted

    def _get_financial(self, filename: str, table_name: str,
                       codes: list[str], fetch_fn) -> pd.DataFrame:
        p = self._root / filename
        parts, failed, aborted = self._fetch_batched(filename, codes, fetch_fn)

        if parts:
            if failed or aborted:
                log.warning("财务表 %s 有 %d 批未取到（熔断=%s），重跑本步骤可断点续拉",
                            table_name, len(failed), aborted)
            merged_full = pd.read_parquet(p)
            self._meta.setdefault(table_name, {})["last_date"] = (
                int(pd.Timestamp(merged_full["ann_date"].max()).strftime("%Y%m%d"))
                if "ann_date" in merged_full.columns and not merged_full["ann_date"].isna().all() else 0
            )
            self._save_meta()
            return pd.concat(parts, ignore_index=True)

        # 全部分批失败：回退到本地既有缓存，不阻断主流程
        if p.exists():
            cached = pd.read_parquet(p)
            return cached[cached["code"].isin(codes)] if "code" in cached.columns else cached
        return pd.DataFrame()

    def get_balance_sheet(self, code_list: Iterable[str],
                          begin_date: int | None = None,
                          end_date: int | None = None) -> pd.DataFrame:
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
