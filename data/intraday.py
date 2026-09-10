"""
内存映射稠密分钟面板存储
========================

为日内（分钟频）因子挖掘设计的数据层（参照国金 Alpha掘金 24 的工程方案：
分钟频数据计算面临内存瓶颈 → MemMap 按需访问 + 按年份/股票切片）。

为什么不用现有 parquet 长表直接算
--------------------------------
``min{period}_{pool}.parquet`` 是 (kline_time, code) 长表，取数/增量更新用它
没问题，但因子计算有两个痛点：
1. **内存**：all_a 的 5 分钟数据全量约数 GB～数十 GB，长表整读进内存做
   groupby 不可行；研究迭代时反复读取也慢。
2. **向量化**：长表上做"按 (date, code) 分组聚合"只能 groupby-apply 逐组
   Python 回调（``scripts/build_intraday_factors.py`` 的 14 因子即此实现，
   全量 hs300 约 5 分钟）；稠密数组 [day, bar, code] 上则是纯 numpy
   整块运算，快 1~2 个数量级。

存储布局（cache_root/intraday/ 下，按 池×档位 一个目录，按年分文件）
------------------------------------------------------------------
    intraday/min5_hs300/
        meta.json               # period/pool/codes(全量列并集)/bar_times/各年天数
        y2022/
            days.json           # 当年交易日（int YYYYMMDD，升序，= 数组第 0 维）
            open.npy            # [n_days, n_bars, n_codes] float32
            high/low/close/volume/amount.npy

- 每字段一个 ``.npy``，``np.load(mmap_mode="r")`` 按需页调入（MemMap），
  进程只映射不读取，切片读取才真正触发 IO。
- 按年分区：增量构建只追加新年份文件；并行计算按年切片互不干扰。
- codes 取**全部年份的并集**作为统一列（跨年拼接零对齐成本；代价是
  当年未上市的股票为全 NaN 列，float32 NaN 不占额外空间之外的成本）。
- bar_times 从数据推导（各 bar 的日内时刻表，如 5 分钟为 48 档），不硬编码
  ——不同数据源对 bar 打的是开始时刻（09:30）还是结束时刻（09:35）不完全
  一致，查表法天然兼容。
- 缺失 bar（停牌/半拉天）记 NaN；volume=0 是真实值，与 NaN 区分。

口径
----
- 价格为**未复权**分钟价（与 min parquet 缓存一致）。跨 bar 收益在日内
  无除权跳变问题；除权除息日的污染由上层特征层负责剔除（与
  build_intraday_factors 口径一致）。
- 本层不做 PIT 处理：分钟面板只描述"当天发生了什么"，特征是否 lookahead
  由特征定义决定（factor/intraday_features.py 全部只用当日信息）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from config import Config

log = logging.getLogger("data.intraday")

#: 面板字段（与 DataSource.get_minute_kline 返回列一致）
FIELDS = ("open", "high", "low", "close", "volume", "amount")

_META = "meta.json"
_DAYS = "days.json"


def _panel_root(cache_root: Path | str | None) -> Path:
    if cache_root is None:
        cache_root = Config.cache()["root"]
    return Path(str(cache_root).replace("//", "/")) / "intraday"


class MinutePanelStore:
    """按年分区、字段分文件的内存映射分钟面板。

    典型用法::

        store = MinutePanelStore.build(cache.read_minute_kline("hs300", 5), 5, "hs300")
        store = MinutePanelStore.open(5, "hs300")            # 之后直接打开
        blocks = store.iter_blocks(("close", "volume"))       # 流式（MemMap 友好）
        full = store.load(("close", "volume"), 20220101, 20261231)  # 全载（研究用）
    """

    def __init__(self, root: Path, period: int, pool: str):
        self.root = Path(root)
        self.period = int(period)
        self.pool = str(pool)
        meta_path = self.root / _META
        if not meta_path.exists():
            raise FileNotFoundError(
                f"分钟面板不存在: {meta_path}。先运行 "
                f"`python -m scripts.build_minute_panel --offline` 构建。"
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)
        # meta 自洽性：目录下有而 meta 没登记的年份（手工拷入/构建中断）不加载，
        # 避免"半写年份"被当成完整数据。
        self._years = [y for y in self._meta["years"] if (self.root / f"y{y}" / _DAYS).exists()]

    # ------------------------------------------------------------------ 构建入口
    @classmethod
    def build(
        cls,
        minute_df: pd.DataFrame,
        period: int,
        pool: str,
        cache_root: Path | str | None = None,
        float_dtype=np.float32,
    ) -> "MinutePanelStore":
        """把分钟长表 (kline_time, code) 稠密化为按年分区的面板并落盘。

        幂等：重复构建同一 (pool, period) 会整体重写（以传入长表为准），
        不做增量合并——增量水位由 parquet 缓存层负责，本层只是"物化视图"。
        """
        root = _panel_root(cache_root) / f"min{period}_{pool}"
        root.mkdir(parents=True, exist_ok=True)
        df = _validate_long(minute_df)

        codes = sorted(df.index.get_level_values("code").unique())
        code_ix = {c: i for i, c in enumerate(codes)}
        # bar 时刻表从数据推导（全局并集，跨年统一）
        tod = df.index.get_level_values("kline_time").time
        bar_keys = sorted({t.replace(second=0, microsecond=0) for t in tod})
        bar_ix = {t: i for i, t in enumerate(bar_keys)}
        n_bars = len(bar_keys)
        n_codes = len(codes)
        bar_times = [f"{t.hour:02d}{t.minute:02d}" for t in bar_keys]

        df = df.assign(
            _day=df.index.get_level_values("kline_time").normalize(),
            _tod=tod,
        )
        years = sorted(df["_day"].dt.year.unique())
        meta = {"period": period, "pool": pool, "codes": codes,
                "bar_times": bar_times, "fields": list(FIELDS), "years": []}

        for year in years:
            sub = df[df["_day"].dt.year == year]
            days = sorted(int(d.strftime("%Y%m%d")) for d in sub["_day"].unique())
            day_ix = {pd.Timestamp(str(d)).normalize(): i for i, d in enumerate(days)}

            # 高级索引一次性散列赋值，替代逐日逐码 groupby 循环
            rows_day = np.fromiter((day_ix[d] for d in sub["_day"]), dtype=np.int64, count=len(sub))
            rows_bar = np.fromiter((bar_ix[t] for t in sub["_tod"]), dtype=np.int64, count=len(sub))
            rows_code = np.fromiter(
                (code_ix[c] for c in sub.index.get_level_values("code")),
                dtype=np.int64, count=len(sub))

            ydir = root / f"y{year}"
            ydir.mkdir(exist_ok=True)
            with open(ydir / _DAYS, "w", encoding="utf-8") as f:
                json.dump(days, f)
            for field in FIELDS:
                arr = np.full((len(days), n_bars, n_codes), np.nan, dtype=float_dtype)
                arr[rows_day, rows_bar, rows_code] = sub[field].to_numpy(dtype=float_dtype)
                np.save(ydir / f"{field}.npy", arr)
            meta["years"].append(int(year))
            _log_coverage(year, days, arr, rows_code)

        with open(root / _META, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        log.info("分钟面板构建完成: %s（%d 年 / %d 码 / %d bar/日）",
                 root, len(years), n_codes, n_bars)
        return cls(root, period, pool)

    @classmethod
    def open(cls, period: int, pool: str,
             cache_root: Path | str | None = None) -> "MinutePanelStore":
        """打开已构建的面板（等价于直接构造，语义更明确的读取入口）。"""
        return cls(_panel_root(cache_root) / f"min{period}_{pool}", period, pool)

    # ------------------------------------------------------------------ 元信息
    @property
    def years(self) -> list[int]:
        return list(self._years)

    @property
    def codes(self) -> list[str]:
        return list(self._meta["codes"])

    @property
    def n_bars(self) -> int:
        return len(self._meta["bar_times"])

    @property
    def bar_times(self) -> list[str]:
        """各 bar 日内时刻（HHMM，如 '0935'），与数组第 1 维对齐。"""
        return list(self._meta["bar_times"])

    def days(self, year: int) -> list[int]:
        with open(self.root / f"y{year}" / _DAYS, encoding="utf-8") as f:
            return json.load(f)

    def all_days(self) -> list[int]:
        out: list[int] = []
        for y in self._years:
            out.extend(self.days(y))
        return out

    # ------------------------------------------------------------------ 读取
    def field(self, name: str, year: int, mmap: bool = True):
        """单字段单年的内存映射数组 [n_days, n_bars, n_codes]。"""
        if name not in FIELDS:
            raise ValueError(f"未知字段 {name}，可选 {FIELDS}")
        return np.load(self.root / f"y{year}" / f"{name}.npy",
                       mmap_mode="r" if mmap else None)

    def load(
        self,
        fields: Sequence[str],
        begin_date: int | None = None,
        end_date: int | None = None,
    ) -> dict[str, np.ndarray]:
        """跨年拼接加载为内存数组（研究用；数据量大时改用 iter_blocks 流式）。

        返回 {field: [D, B, C] ndarray}，D 为区间内全部交易日，列恒为全量 codes
        （未请求过滤——过滤索引交给上层，配合 ``codes_index`` 做列选择）。
        """
        begin = begin_date or 0
        end = end_date or 99991231
        slices: list[tuple[int, slice]] = []
        for y in self._years:
            days = [d for d in self.days(y) if begin <= d <= end]
            if days:
                # 同年内一定是连续区间（days 升序、无重复）
                i0, i1 = days[0], days[-1]
                all_days_y = self.days(y)
                slices.append((y, slice(all_days_y.index(i0), all_days_y.index(i1) + 1)))
        if not slices:
            n = self.n_bars
            return {f: np.empty((0, n, len(self.codes))) for f in fields}
        return {f: np.concatenate([self.field(f, y)[sl] for y, sl in slices], axis=0)
                for f in fields}

    def iter_blocks(
        self,
        fields: Sequence[str],
        block_days: int = 64,
        begin_date: int | None = None,
        end_date: int | None = None,
    ) -> Iterator[tuple[list[int], dict[str, np.ndarray]]]:
        """按 block_days 天分块流式迭代，块内字段为已载入内存的数组。

        MemMap 友好：一次只真实读取一个块的字节，全市场面板也不会撑爆内存
        （Alpha掘金 24 的"按年份和股票切片并行计算"即在此粒度上做）。
        """
        for f in fields:
            if f not in FIELDS:
                raise ValueError(f"未知字段 {f}，可选 {FIELDS}")
        begin = begin_date or 0
        end = end_date or 99991231
        for y in self._years:
            days_all = self.days(y)
            days = [d for d in days_all if begin <= d <= end]
            if not days:
                continue
            pos = {d: i for i, d in enumerate(days_all)}
            for i0 in range(0, len(days), block_days):
                chunk = days[i0:i0 + block_days]
                sl = slice(pos[chunk[0]], pos[chunk[-1]] + 1)
                yield chunk, {f: np.asarray(self.field(f, y)[sl]) for f in fields}

    def to_long(
        self,
        fields: Sequence[str] | None = None,
        begin_date: int | None = None,
        end_date: int | None = None,
    ) -> pd.DataFrame:
        """还原为 (kline_time, code) 长表（仅非 NaN 格子），用于对账/调试。"""
        fields = tuple(fields) if fields else FIELDS
        frames = []
        for chunk_days, block in self.iter_blocks(fields, block_days=256,
                                                  begin_date=begin_date, end_date=end_date):
            ts = pd.to_datetime([str(d) for d in chunk_days], format="%Y%m%d")
            # [D, B, C] → 长表：日×bar 展开为 kline_time
            bar_hm = self.bar_times
            times = pd.DatetimeIndex(
                [t.replace(hour=int(hm[:2]), minute=int(hm[2:])) for t in ts for hm in bar_hm])
            n_d, n_b, n_c = block[fields[0]].shape
            codes = np.tile(np.array(self.codes), n_d * n_b)
            idx = pd.MultiIndex.from_arrays([np.repeat(times, n_c), codes],
                                            names=["kline_time", "code"])
            data = {f: block[f].reshape(-1, n_c).ravel() for f in fields}
            df = pd.DataFrame(data, index=idx)
            frames.append(df.dropna(how="all"))
        if not frames:
            return pd.DataFrame(columns=list(fields))
        return pd.concat(frames).sort_index()

    def coverage_stats(self) -> pd.DataFrame:
        """逐年覆盖概览：天数/代码数/平均 bar 覆盖率/NaN 率（诊断半拉天）。"""
        rows = []
        n_bars, n_codes = self.n_bars, len(self.codes)
        for y in self._years:
            close = self.field("close", y)
            n_days = close.shape[0]
            valid = np.isfinite(close)
            # 每格至少 1 根 bar 的 (day, code) 占比 = "有行情"覆盖
            traded = valid.any(axis=1)
            rows.append({
                "year": y,
                "days": n_days,
                "codes": n_codes,
                "cell_coverage": float(valid.sum()) / close.size,
                "traded_coverage": float(traded.sum()) / (n_days * n_codes),
                "mean_bars": float(valid.sum(axis=1).mean()),
                "bars_full": n_bars,
            })
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------
def _validate_long(minute_df: pd.DataFrame) -> pd.DataFrame:
    """校验长表结构并按 (kline_time, code) 排序去重。"""
    if not isinstance(minute_df.index, pd.MultiIndex):
        raise ValueError("minute_df 需为 (kline_time, code) MultiIndex 长表")
    if list(minute_df.index.names) != ["kline_time", "code"]:
        raise ValueError(f"索引名应为 ['kline_time', 'code']，got {minute_df.index.names}")
    missing = [f for f in FIELDS if f not in minute_df.columns]
    if missing:
        raise ValueError(f"缺少字段 {missing}")
    ts = minute_df.index.get_level_values("kline_time")
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise ValueError("kline_time 需为 datetime（含日内时分）")
    df = minute_df.copy()
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def _log_coverage(year: int, days: list[int], last_field_arr: np.ndarray,
                  rows_code: np.ndarray) -> None:
    del last_field_arr, rows_code  # 覆盖率明细由 coverage_stats() 按需给出
    log.info("  y%d: %d 交易日落盘", year, len(days))
