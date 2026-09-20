"""另类数据 L2 结构化层：缓存 + 增量 + 统一入口
================================================

与 :class:`data.textmining.fetch.TextMiningCache` 同构（同一 parquet 缓存根
``Config.cache()["root"]`` = ``e:/data/parquet/``），但管的是**结构化事件表**
而非文本。

缓存表（``<root>/alt_*.parquet``）
----------------------------------
============================  ==============================  ============================
文件                           内容                            更新方式
============================  ==============================  ============================
``alt_holder_control.parquet``   实际控制人持股变动             全量一次（2010-12 起）
``alt_inner_trade.parquet``      内部人交易                     全量一次（滚动 20.5 月）
``alt_mgmt_hold.parquet``        高管增减持                     全量一次（滚动 1 年）
``alt_macro_calendar.parquet``   宏观日历 + 预期值              按日增量（断点续传）
``alt_news/news_YYYYMMDD.parquet`` 快照源（同花顺/新浪）         按日累积（**无历史可补**）
``alt_cls/cls_YYYYMMDD.parquet`` 财联社电报（**可回补历史**）     游标回补 + 前向增量
``_meta.json``                   增量水位（已抓日期 / 每日行数统计）
============================  ==============================  ============================

⚠️ **新闻有两套存储，别混用**（2026-09-20 实测后拆分）：

- ``alt_news/`` = **快照源**（同花顺 / 新浪），一次只给 20 条、**无历史可回补**
  ⇒ 只能高频轮询堆积，丢掉的时段拿不回来；
- ``alt_cls/`` = **财联社电报**，支持 ``last_time`` 往回翻页、**可回补到 ≥2015**，
  且自带 ``level``（加红等级）与 ``stock_list``（关联个股）⇒ **这是新闻研究的主表**。
  详见 :mod:`data.altdata.source_cls`。

增量策略（按接口特性分三类）
----------------------------
- **全量快照类**（前三个）：接口一次返回全部可回溯区间 ⇒ 策略为
  「重拉 → 与本地按关键列去重合并」。数据量小（5K–30K 行），无需日期水位。
- **单日查询类**（宏观日历）：接口无区间参数 ⇒ **必须按日循环**，
  用 ``_meta.json`` 的 ``macro_done_dates`` 做断点续传；并记录每日
  ``(rows, forecast_notna)`` 统计，用于**缺失日检测** —— 实测有整列
  「预期」缺失的交易日（2026-09-14 预期 0 条），必须能与「真无预期」区分。
- **游标翻页类**（财联社）：接口按 ``last_time`` 往回给 ⇒ 用
  ``cls_backfill_cursor``（unix 秒）做断点续传，按发布日分片落盘。

⚠️ PIT：本层写出的 ``ann_date`` 是**唯一合法对齐轴**。其中 ``holder_control``
与 ``inner_trade`` 的 ``ann_date`` 是**乐观可得日**（无公告日，回填变动日），
因子层须施加 ``source_akshare.SUGGESTED_LAG_TRADING_DAYS`` 的滞后。
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

from config import Config
from data.altdata import cninfo_holder as cninfo
from data.altdata import source_akshare as src
from data.altdata import source_cls as cls_src
from data.altdata import source_news as news_src

log = logging.getLogger(__name__)

#: 巨潮高管增减持可回补的最早年份（实测 2010 年增持 1 294 条）
CNINFO_HOLDER_MIN_YEAR = 2010

#: 各表的去重关键列（含 NaN 的列不能作全列 drop_duplicates，见 _merge）
_DEDUP_KEYS: dict[str, list[str]] = {
    "holder_control": ["code", "chg_date", "controller"],
    "inner_trade": ["name", "chg_date", "person", "chg_qty"],
    "mgmt_hold": ["code", "ann_date", "chg_date", "person", "chg_qty", "direction"],
    "macro_calendar": ["event_time", "region", "event"],
    "cninfo_holder": ["code", "ann_date", "chg_date", "direction", "person", "chg_qty"],
}

#: 表名 → 缓存文件名
_FILES: dict[str, str] = {
    "holder_control": "alt_holder_control.parquet",
    "inner_trade": "alt_inner_trade.parquet",
    "mgmt_hold": "alt_mgmt_hold.parquet",
    "macro_calendar": "alt_macro_calendar.parquet",
    "cninfo_holder": "alt_cninfo_holder.parquet",
}

#: 全量快照类表（可直接重拉合并）
_SNAPSHOT_TABLES = ("holder_control", "inner_trade", "mgmt_hold")

#: 缓存 schema 版本 —— **改列映射 / 符号约定 / 代码标准化 / 单位时必须 +1**。
#:
#: 为什么需要它（2026-09-20 实测事故）：修好 ``_code``（前缀式代码）与
#: ``signed_qty``（双重取反）后重抓，旧缓存**不会自动失效** —— 因为去重键里
#: 含 ``chg_qty``，符号一翻就判成「不同行」，于是新旧两版**并存**：
#: mgmt_hold 从 30317 涨到 47942，同一笔交易同时存在 ``chg_qty=-5000``（旧）
#: 与 ``+5000``（新）。这种错误不抛异常，只让因子在两个相反方向上互相抵消。
#: 版本号把「口径变更」变成可检测事件，而不是靠人记得清缓存。
SCHEMA_VERSION = 2   # v1 = 首轮冒烟（代码未标准化 / signed_qty 双重取反）


def _as_date(v: str | int | pd.Timestamp) -> pd.Timestamp:
    """``YYYYMMDD``（int / str，允许含横线）或 Timestamp → ``Timestamp``。

    ⚠️ 两个坑必须一起绕开（集中在这里，避免各处各写一份）：

    1. ``pd.Timestamp(v, format="%Y%m%d")`` **在 pandas 3.x 会 TypeError**
       （``__new__() got an unexpected keyword argument 'format'``，实测）；
    2. **int 绝不能直接进解析器** —— ``pd.to_datetime(20180101)`` 会把它当
       **纳秒时间戳** 得到 1970-01-01（项目既有 PIT 铁律：日历 int 必须先转 str）。
    """
    if isinstance(v, pd.Timestamp):
        return v
    return pd.to_datetime(str(v).replace("-", "")[:8], format="%Y%m%d")


def _ts_cst(v: str | int | pd.Timestamp) -> int:
    """``YYYYMMDD`` → **北京时区当日零点** 的 unix 秒。

    🚨 为什么不能写 ``_as_date(v).timestamp()``：``_as_date`` 返回的是 **naive**
    Timestamp，pandas 对它调 ``.timestamp()`` 会**按 UTC 解释**，得到的 epoch
    比「北京零点」**少 8 小时**（实测：两种写法差 28800 秒）。
    游标一旦沾上这 8 小时偏移，就别想再和接口的 ``last_time`` 对齐了。
    """
    return int(_as_date(v).tz_localize("Asia/Shanghai").timestamp())


class AltDataCache:
    """另类数据 parquet 缓存 + 增量拉取。"""

    def __init__(self, cache_root: Path | str | None = None):
        if cache_root is None:
            cache_root = Config.cache()["root"]
        self._root = Path(str(cache_root).replace("//", "/"))
        self._root.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._root / "_alt_meta.json"
        self._meta: dict = self._load_meta()

    # ---- meta ----
    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            with open(self._meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_meta(self) -> None:
        """原子 + 合并写 ``_alt_meta.json``。

        🚨 2026-09-20 事故（实测）：两个进程并发跑**不同** ``--tables`` 时，各自持有
        启动时加载的 ``_meta`` 内存副本；原先的「全量覆写」会**静默抹掉对方刚写入的
        字段** —— macro 回补刚写完 ``macro_done_dates``（2275 天水位），20:45 启动的
        cls 回补进程随后每次 ``_save_meta()`` 都把它整段抹掉 ⇒ 断点续传失效、下次
        全量重抓。修法 = **写前重读磁盘，按「磁盘为底、内存为准」合并**。

        .. warning::
           这消除的是「丢字段」，**不是**通用并发安全：同一字段被两进程并发写仍是
           「后者胜」。要绝对安全须加文件锁；当前约定是**一次把多表传给同一个进程**
           （``--tables a,b,c``），不要并发起两个进程各跑一批表。
        """
        if self._meta_path.exists():
            try:
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    disk = json.load(f)
                if isinstance(disk, dict):
                    self._meta = {**disk, **self._meta}
            except (OSError, json.JSONDecodeError):
                log.warning("[altdata] _alt_meta 读回失败，按内存版写入")
        tmp = self._meta_path.with_name(self._meta_path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self._meta_path)
        except OSError as e:
            # WinError 5：目标被其它进程的共享读句柄占用（实测此时**仍可写**）
            log.warning("[altdata] 原子替换 %s 失败(%s)，降级为直接覆写", tmp.name, e)
            with open(self._meta_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, indent=2, ensure_ascii=False)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        else:
            if tmp.exists():  # os.replace 成功后 tmp 不应存在，防御性清理
                try:
                    tmp.unlink()
                except OSError:
                    pass

    @property
    def root(self) -> Path:
        return self._root

    def path(self, table: str) -> Path:
        return self._root / _FILES[table]

    def load(self, table: str) -> pd.DataFrame:
        """读缓存表；**schema 版本不符时视为空**（触发重抓，不用脏数据）。"""
        p = self.path(table)
        if not p.exists():
            return pd.DataFrame()
        cached_v = self._meta.get("schema_versions", {}).get(table)
        if cached_v != SCHEMA_VERSION:
            log.warning("[altdata] %s 缓存 schema v%s != 当前 v%s ⇒ 判定失效，将重抓"
                        "（旧口径数据不与新口径混合）", table, cached_v, SCHEMA_VERSION)
            return pd.DataFrame()
        return pd.read_parquet(p)

    # ---- 通用合并 ----
    @staticmethod
    def _merge(local: pd.DataFrame, new: pd.DataFrame, table: str) -> pd.DataFrame:
        """按关键列去重合并（新数据优先），并按可得日排序。

        用 subset 而非全列 drop_duplicates：码/名等列可能为 NaN，
        全列去重会把 NaN 行全部判为「不重复」而无限膨胀。
        """
        if new is None or new.empty:
            return local
        frames = [f for f in (local, new) if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        keys = [k for k in _DEDUP_KEYS[table] if k in out.columns]
        if keys:
            # 🚨 不能直接 ``drop_duplicates(subset=keys)``：**NaN != NaN**，
            # 含 NaN 的键会被判为「不重复」而全部保留 —— 实测一次重抓就把
            # holder_control 从 5593 涨到 5870（多出的全是同一批 NaN 键行）。
            # 故先把键列 NaN 归一为哨兵字符串再判重。
            probe = out[keys].astype(object).fillna("__NA__").astype(str)
            out = out[~probe.duplicated(keep="last")]
        else:
            out = out.drop_duplicates(keep="last")
        sort_col = "ann_date" if "ann_date" in out.columns else "event_time"
        if sort_col in out.columns:
            out = out.sort_values(sort_col).reset_index(drop=True)
        return out

    def _write(self, table: str, df: pd.DataFrame) -> None:
        df.to_parquet(self.path(table), compression="snappy", index=False)
        # 记录写盘时的 schema 版本，供下次 load 校验
        self._meta.setdefault("schema_versions", {})[table] = SCHEMA_VERSION
        self._save_meta()

    # ------------------------------------------------------------------
    # 全量快照类
    # ------------------------------------------------------------------
    def _refresh_snapshot(self, table: str, fetch_fn, refresh: bool,
                          mode: str = "accumulate") -> pd.DataFrame:
        """重拉全量快照并按 ``mode`` 处理本地数据。

        ``mode`` 的选择**由接口特性决定，不是风格偏好**：

        - ``"replace"``：接口返回**完整历史**（如 holder_control 的 2010 起全量）
          ⇒ 直接覆盖。累积无信息增益，且会把接口侧的修正/下架数据永久留在本地。
        - ``"accumulate"``：接口只返回**滚动窗口**（inner_trade 20.5 月、
          mgmt_hold 1 年）⇒ 必须按关键列去重累积 —— **累积是突破滚动窗口、
          攒出更长历史的唯一手段**（每天抓一次，一年后就有 1 年+历史）。
        """
        local = self.load(table)
        if not refresh and not local.empty:
            return local
        new = fetch_fn()
        if new is None or new.empty:
            log.warning("[altdata] %s 本次抓取为空，保留本地 %d 行不做改动", table, len(local))
            return local
        if not local.empty and len(new) < 0.5 * len(local):
            log.warning("[altdata] %s 新抓 %d 行不足本地 %d 行的 50%% —— "
                        "接口可能只回了部分数据，请核查后再引用", table, len(new), len(local))
        if mode == "replace":
            self._write(table, new)
            log.info("[altdata] %s: 覆盖写 %d 行（原 %d）", table, len(new), len(local))
            return new
        merged = self._merge(local, new, table)
        self._write(table, merged)
        log.info("[altdata] %s: 本地 %d + 新拉 %d → %d 行（累积）",
                 table, len(local), len(new), len(merged))
        return merged

    def get_holder_control(self, refresh: bool = True) -> pd.DataFrame:
        """实际控制人持股变动（2010-12 起全量）。

        ⚠️ ``ann_date`` 为乐观可得日（接口无公告日），因子层须加 60 交易日滞后。
        """
        return self._refresh_snapshot("holder_control", src.fetch_holder_control,
                                      refresh, mode="replace")

    def get_inner_trade(self, refresh: bool = True) -> pd.DataFrame:
        """内部人交易（滚动 20.5 月）。

        ⚠️ ``ann_date`` 为乐观可得日，因子层须加 2 交易日滞后。
        """
        ntc = load_name_to_code(self._root)
        return self._refresh_snapshot(
            "inner_trade", lambda: src.fetch_inner_trade(name_to_code=ntc), refresh)

    def get_mgmt_hold(self, refresh: bool = True) -> pd.DataFrame:
        """高管增减持（滚动 1 年，含公告日 + 截止日 ⇒ PIT 质量最好）。"""
        return self._refresh_snapshot("mgmt_hold", src.fetch_mgmt_hold_change, refresh)

    # ------------------------------------------------------------------
    # 巨潮高管增减持**全历史**（P0 Step 1b，自建抓取器）
    # ------------------------------------------------------------------
    def get_cninfo_holder(
        self,
        begin: int | str = CNINFO_HOLDER_MIN_YEAR * 10000 + 101,
        end: int | str | None = None,
        *,
        directions: tuple[str, ...] = ("增持", "减持"),
        refresh: bool = True,
        sleep: float = 0.4,
        span_years: int = 2,
        verify: bool = True,
    ) -> pd.DataFrame:
        """回补/刷新 2010 起高管增减持全历史（:mod:`data.altdata.cninfo_holder`）。

        取数策略（每一层都对应一个实测到的接口怪癖，详见 cninfo_holder 模块头）：

        1. **stride-1 的 2 年年段**（``iter_overlap_windows``），不是逐年切
           —— 切窄会丢「变动在年内、公告日推后」的行（实测 2019 分季比全年少 2 886 条）；
        2. **撞 20000 行上限自动二分递归**（``total > len(records)`` 即判截断）；
        3. **完整性与幂等**：按关键列去重 + **逐年覆盖闸门**
           （``assert_no_empty_years``，2026-09-20 的非法日期二分事故让 3 个整年消失）。

        Parameters
        ----------
        begin / end : int | str
            ``YYYYMMDD``；``begin`` 缺省 = ``CNINFO_HOLDER_MIN_YEAR`` 年初。
        directions : tuple
            ``("增持", "减持")``。
        refresh : bool
            False 且本地有缓存 ⇒ 直接返回本地。
        sleep : float
            请求间隔（秒）—— 不给间隔会被服务端 RST。
        span_years / verify : int / bool
            窗口跨度 / 是否跑逐年覆盖闸门。
        """
        local = self.load("cninfo_holder")
        if not refresh and not local.empty:
            return local
        end = end if end is not None else int(pd.Timestamp.today().strftime("%Y%m%d"))
        y0 = int(str(begin)[:4])
        y1 = int(str(end)[:4])
        years = list(range(y0, y1 + 1))
        windows = list(cninfo.iter_overlap_windows(years, span=span_years))
        frames: list[pd.DataFrame] = []
        for d in directions:
            for wb, we in windows:
                try:
                    part = cninfo.fetch_holder_trades(wb, we, d, sleep=sleep)
                except Exception as e:  # noqa: BLE001
                    log.error("[altdata] cninfo_holder %s %s~%s 失败：%s: %s",
                              d, wb, we, type(e).__name__, str(e)[:160])
                    raise
                if not part.empty:
                    frames.append(part)
                log.info("[altdata] cninfo_holder %s %s~%s → %d 行", d, wb, we, len(part))
        if not frames:
            log.warning("[altdata] cninfo_holder 本次抓取为空，保留本地 %d 行", len(local))
            return local
        new = pd.concat(frames, ignore_index=True)
        # 去重：同一笔交易会被相邻窗口取到两次（这正是重叠策略的代价）
        keys = [k for k in _DEDUP_KEYS["cninfo_holder"] if k in new.columns]
        probe = new[keys].astype(object).fillna("__NA__").astype(str)
        new = new[~probe.duplicated(keep="first")].reset_index(drop=True)
        if verify:
            # 闸门：只对「完整年 + 当年已结束」检查，当年（进行中）不苛求
            full_years = [y for y in years if y < pd.Timestamp.today().year]
            cninfo.assert_no_empty_years(new, full_years, column="chg_date")
        merged = self._merge(local, new, "cninfo_holder")
        self._write("cninfo_holder", merged)
        self._meta["cninfo_holder_stats"] = {
            "rows": int(len(merged)),
            "ann_year_rows": {int(k): int(v) for k, v in
                              cninfo.year_coverage(merged)
                              .set_index("year")["chg_rows"].to_dict().items()},
            "windows": len(windows), "directions": list(directions),
        }
        self._save_meta()
        log.info("[altdata] cninfo_holder: 本地 %d + 新拉 %d → %d 行",
                 len(local), len(new), len(merged))
        return merged

    # ------------------------------------------------------------------
    # 宏观日历（单日查询 → 按日增量 + 断点续传 + 缺失日检测）
    # ------------------------------------------------------------------
    def get_macro_calendar(
        self,
        begin: int | str = 20180101,
        end: int | str | None = None,
        *,
        skip_weekend: bool = True,
        max_days: int | None = None,
        save_every: int = 50,
        progress: bool = True,
    ) -> pd.DataFrame:
        """按日回补/增量宏观日历（``macro_info_ws`` 是单日接口）。

        Parameters
        ----------
        begin / end : int | str
            ``YYYYMMDD``；``end`` 缺省为今天。
        skip_weekend : bool
            跳过周六周日（宏观数据发布极少在周末，省 ≈30% 请求）。默认 True。
        max_days : int | None
            本次最多抓多少天（限速/试跑用），None = 全部缺口。
        save_every : int
            每抓 N 天落盘一次（断点续传粒度）。
        progress : bool
            打印进度。

        Returns
        -------
        pd.DataFrame
            全量已缓存日历（不限于本次抓取区间）。
        """
        end = end if end is not None else pd.Timestamp.today().strftime("%Y%m%d")
        d0 = _as_date(begin)
        d1 = _as_date(end)
        all_days = pd.date_range(d0, d1, freq="D")
        if skip_weekend:
            all_days = all_days[all_days.dayofweek < 5]

        done: set[str] = set(self._meta.get("macro_done_dates", []))
        todo = [d for d in all_days if d.strftime("%Y%m%d") not in done]
        if max_days is not None:
            todo = todo[:max_days]

        local = self.load("macro_calendar")
        stats: dict = self._meta.get("macro_daily_stats", {})
        if progress:
            log.info("[altdata] macro_calendar: 区间 %d 天，已抓 %d，本次待抓 %d",
                     len(all_days), len(done), len(todo))

        buf: list[pd.DataFrame] = []
        for i, d in enumerate(todo, 1):
            ds = d.strftime("%Y%m%d")
            try:
                df = src.fetch_macro_calendar(ds)
            except Exception as e:  # noqa: BLE001
                # 单日失败不中断整轮回补（网络偶发），但**不写入 done** ⇒ 下次重试
                log.warning("[altdata] macro %s 失败: %s: %s", ds, type(e).__name__, str(e)[:80])
                continue
            n_fc = int(df["forecast"].notna().sum()) if not df.empty else 0
            stats[ds] = {"rows": int(len(df)), "forecast_notna": n_fc}
            if not df.empty:
                buf.append(df)
            done.add(ds)
            if i % save_every == 0:
                local = self._merge(local, pd.concat(buf, ignore_index=True), "macro_calendar")
                self._write("macro_calendar", local)
                self._meta["macro_done_dates"] = sorted(done)
                self._meta["macro_daily_stats"] = stats
                self._save_meta()
                buf = []
                if progress:
                    log.info("[altdata] macro %d/%d（已落盘，累计 %d 行）",
                             i, len(todo), len(local))

        if buf:
            local = self._merge(local, pd.concat(buf, ignore_index=True), "macro_calendar")
            self._write("macro_calendar", local)
        self._meta["macro_done_dates"] = sorted(done)
        self._meta["macro_daily_stats"] = stats
        self._save_meta()
        if progress:
            log.info("[altdata] macro_calendar 完成：累计 %d 行 / %d 个日期",
                     len(local), len(done))
        return local

    # ------------------------------------------------------------------
    # 新闻快讯存档（快照型源：无历史 ⇒ 只能按日累积 + 高频抓取）
    # ------------------------------------------------------------------
    @property
    def news_dir(self) -> Path:
        d = self._root / "alt_news"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def news_path(self, date: str | int | None = None) -> Path:
        ds = str(date if date is not None else pd.Timestamp.today().strftime("%Y%m%d"))
        ds = ds.replace("-", "")[:8]
        return self.news_dir / f"news_{ds}.parquet"

    def append_news(self, sources: tuple[str, ...] | None = None,
                    date: str | int | None = None) -> dict:
        """抓当前快照 → 追加到当日存档（幂等）。返回统计 dict。

        ⚠️ 快照深度只有 20 条/源 ⇒ **一次调用只覆盖最近几十分钟**。
        正式留档须高频调用（见脚本 ``--loop`` / ``--install-task``）：
        这是「无历史可回补」的直接后果，**丢掉的时段拿不回来**。

        注意本方法**不做 schema 校验**：新闻存档是追加型的原始文本，
        口径变更不该让已存档的历史失效。
        """
        srcs = tuple(sources) if sources else news_src.NEWS_SOURCES
        new = news_src.fetch_news_all(srcs)
        p = self.news_path(date)
        local = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=news_src.NEWS_COLS)
        before = len(local)
        if not new.empty:
            out = pd.concat([local, new], ignore_index=True)
            out["_key"] = news_src.dedup_key(out)
            out = out.drop_duplicates(subset="_key", keep="last").drop(columns="_key")
            out = out.sort_values("pub_time").reset_index(drop=True)
        else:
            out = local
        out.to_parquet(p, compression="snappy", index=False)
        stat = {"date": p.stem.replace("news_", ""), "before": before,
                "fetched": int(len(new)), "after": int(len(out)),
                "added": int(len(out) - before),
                "sources_ok": sorted(new["source"].unique().tolist()) if not new.empty else []}
        log.info("[altdata] news %s: %d + %d → %d（新增 %d，源 %s）",
                 stat["date"], before, stat["fetched"], stat["after"],
                 stat["added"], stat["sources_ok"])
        return stat

    def load_news(self, date: str | int | None = None) -> pd.DataFrame:
        p = self.news_path(date)
        return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=news_src.NEWS_COLS)

    def load_news_range(self, begin: str | int, end: str | int | None = None) -> pd.DataFrame:
        """合并读取日期区间内的按日存档（缺失日自动跳过 —— 存档有空洞属正常）。"""
        d0 = _as_date(begin)
        d1 = _as_date(end) if end else pd.Timestamp.today().normalize()
        frames, missing = [], []
        for d in pd.date_range(d0, d1, freq="D"):
            p = self.news_path(d.strftime("%Y%m%d"))
            if p.exists():
                frames.append(pd.read_parquet(p))
            else:
                missing.append(d.strftime("%Y%m%d"))
        if missing:
            log.info("[altdata] news 区间缺 %d 天存档（%s…）", len(missing), missing[:3])
        if not frames:
            return pd.DataFrame(columns=news_src.NEWS_COLS)
        return pd.concat(frames, ignore_index=True).sort_values("pub_time").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 财联社电报（**可回补历史** —— 见 source_cls 模块头的实测结论）
    # ------------------------------------------------------------------
    #
    # 为什么单独一套存储、不复用上面的 ``alt_news/`` 快照存档：
    # ``alt_news/`` 存的是**无历史源的最新快照**（同花顺/新浪，20 条/次，
    # 「丢了拿不回来」），只能按日堆积；而财联社**能按 last_time 往回翻**，
    # 需要的是「单表 + 游标水位」的模型，硬塞进按日快照模型反而丢了续传能力。
    @property
    def cls_dir(self) -> Path:
        d = self._root / "alt_cls"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cls_path(self, date: str | int | pd.Timestamp) -> Path:
        ds = str(date).replace("-", "")[:8]
        return self.cls_dir / f"cls_{ds}.parquet"

    def append_cls(self, df: pd.DataFrame) -> dict:
        """把一页/一批电报按**发布日分片**合并落盘（``news_id`` 幂等）。

        分片而非单表的原因：全量回补 ≈ 240 万行，单表每次断点落盘都要重写
        整个文件（GB 级）；按日分片时每次只重写当天那一片（≈580 行）。
        """
        if df is None or df.empty:
            return {"added": 0, "days": 0}
        out = df.copy()
        out["_d"] = pd.to_datetime(out["pub_time"]).dt.strftime("%Y%m%d")
        added, days = 0, 0
        for ds, part in out.groupby("_d", sort=True):
            part = part.drop(columns="_d")
            p = self.cls_path(ds)
            if p.exists():
                old = pd.read_parquet(p)
                before = len(old)
                merged = pd.concat([old, part], ignore_index=True)
                merged = merged.drop_duplicates(subset="news_id", keep="last")
                added += max(0, len(merged) - before)
            else:
                merged = part.drop_duplicates(subset="news_id", keep="last")
                added += len(merged)
            merged = merged.sort_values("pub_time").reset_index(drop=True)
            merged.to_parquet(p, compression="snappy", index=False)
            days += 1
        return {"added": int(added), "days": int(days)}

    def load_cls(self, begin: str | int | None = None,
                 end: str | int | None = None) -> pd.DataFrame:
        """合并读取区间内的电报分片（缺失日自动跳过）。"""
        files = sorted(self.cls_dir.glob("cls_*.parquet"))
        if begin is not None:
            b = str(begin).replace("-", "")[:8]
            files = [f for f in files if f.stem.replace("cls_", "") >= b]
        if end is not None:
            e = str(end).replace("-", "")[:8]
            files = [f for f in files if f.stem.replace("cls_", "") <= e]
        if not files:
            return pd.DataFrame(columns=cls_src.CLS_COLS)
        out = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        sort_col = "ctime" if "ctime" in out.columns else "pub_time"
        return out.drop_duplicates(subset="news_id").sort_values(sort_col).reset_index(drop=True)

    def cls_coverage(self) -> dict:
        """已存档覆盖范围（最早/最新时间 + 天数 + 条数）—— 用于快速判断缺口。"""
        files = sorted(self.cls_dir.glob("cls_*.parquet"))
        if not files:
            return {"days": 0, "items": 0, "first": None, "last": None}
        days = [f.stem.replace("cls_", "") for f in files]
        items = int(self._meta.get("cls_items", 0))
        return {"days": len(files), "items": items,
                "first": days[0], "last": days[-1],
                "oldest_ts": self._meta.get("cls_oldest_ts"),
                "newest_ts": self._meta.get("cls_newest_ts")}

    def backfill_cls(
        self,
        begin: int | str = cls_src.CLS_EARLIEST_DATE,
        end: int | str | None = None,
        *,
        rn: int = cls_src.CLS_MAX_RN,
        max_pages: int | None = None,
        flush_every: int = 20,
        sleep: float = 0.25,
        progress: bool = True,
        resume: bool = True,
    ) -> dict:
        """**往回翻页**回补历史电报，断点续传。

        行为
        ----
        - 游标从 ``end``（默认 = 续传水位，否则 = 现在）往回翻；
          每页取 ``min(ctime) - 1`` 作下一页游标。
        - 每 ``flush_every`` 页落盘一次 + 写水位（``cls_backfill_cursor``），
          进程被杀也只丢最后一小批。
        - 终止条件（三者任一）：游标越过 ``begin`` / 连续两页空 / 达到 ``max_pages``。
          **连续两页空**是「翻到端点尽头」的判据（实测 2013 起返回空）。

        ⚠️ 空页**不等于**「网络错误」—— ``fetch_cls_page`` 对网络异常抛异常，
        所以空页是可信的「到底了」信号（这个区分是回补能自动终止的前提）。

        Returns
        -------
        dict  本次统计：``pages`` / ``added`` / ``cursor`` / ``reached_begin``
        """
        end_ts = (_ts_cst(end) if end is not None else int(time.time())) + 86400
        begin_ts = _ts_cst(begin)
        meta_cur = self._meta.get("cls_backfill_cursor")
        cursor = int(meta_cur) if (resume and meta_cur) else end_ts

        sess = cls_src.requests.Session()
        pages = added = 0
        buf: list = []
        empty_streak = 0
        reached = False
        try:
            while True:
                if max_pages is not None and pages >= max_pages:
                    break
                if cursor <= begin_ts:
                    reached = True
                    break
                df = cls_src.fetch_cls_page(cursor, rn=rn, session=sess)
                if df.empty:
                    empty_streak += 1
                    if empty_streak >= 2:
                        reached = True
                        break
                    cursor -= 86400          # 空页可能只是「当天这个时点之前没发」
                    continue
                empty_streak = 0
                pages += 1
                buf.append(df)
                # 🚨 游标只用 ctime（unix 秒）—— 用 pub_time.timestamp() 会引入
                # 8 小时偏移并使游标单调递增（死循环），见模块头「时区铁律」。
                oldest = int(df["ctime"].min())
                newest = int(df["ctime"].max())
                self._meta["cls_newest_ts"] = max(int(self._meta.get("cls_newest_ts", 0)), newest)
                cur_old = int(self._meta.get("cls_oldest_ts", 10**12))
                self._meta["cls_oldest_ts"] = min(cur_old, oldest)
                cursor = oldest - 1

                if pages % flush_every == 0 or cursor <= begin_ts:
                    bat = pd.concat(buf, ignore_index=True)
                    st = self.append_cls(bat)
                    added += st["added"]
                    self._meta["cls_items"] = int(self._meta.get("cls_items", 0)) + st["added"]
                    self._meta["cls_backfill_cursor"] = cursor
                    self._save_meta()
                    buf = []
                    if progress:
                        log.info("[altdata] cls 回补 %d 页，累计新增 %d 条，游标 %s",
                                 pages, added,
                                 pd.Timestamp(cursor, unit="s", tz="Asia/Shanghai")
                                 .strftime("%Y-%m-%d %H:%M"))
                if sleep:
                    time.sleep(sleep)
            if buf:
                st = self.append_cls(pd.concat(buf, ignore_index=True))
                added += st["added"]
                self._meta["cls_items"] = int(self._meta.get("cls_items", 0)) + st["added"]
        finally:
            self._meta["cls_backfill_cursor"] = cursor
            self._save_meta()
        if reached:
            self._meta["cls_backfill_done"] = True
            self._save_meta()
        return {"pages": pages, "added": int(added), "cursor": cursor, "reached_begin": reached}

    def sync_cls_recent(
        self,
        *,
        rn: int = cls_src.CLS_MAX_RN,
        max_pages: int = 400,
        flush_every: int = 10,
        sleep: float = 0.25,
        progress: bool = False,
    ) -> dict:
        """**前向增量**：从「现在」往回翻，直到够上已存档的最新时点。

        依赖**连续性假设**：已存档区间是连续的（``backfill_cls`` 保证了这点）。
        故翻到 ``min(ctime) <= cls_newest_ts`` 即可停 —— 不必逐条查 ``news_id``
        是否已存在（那要读全部 2000+ 个分片，代价不可接受）。
        """
        newest = self._meta.get("cls_newest_ts")
        stop_ts = int(newest) - 3600 if newest else 0   # 留 1 小时重叠，防边界漏条
        cursor = int(time.time())
        sess = cls_src.requests.Session()
        pages = added = 0
        buf: list = []
        while pages < max_pages and cursor > stop_ts:
            df = cls_src.fetch_cls_page(cursor, rn=rn, session=sess)
            if df.empty:
                break
            pages += 1
            buf.append(df)
            oldest = int(df["ctime"].min())
            newest_b = int(df["ctime"].max())
            self._meta["cls_newest_ts"] = max(int(self._meta.get("cls_newest_ts", 0)), newest_b)
            self._meta["cls_oldest_ts"] = min(
                int(self._meta.get("cls_oldest_ts", 10**12)), oldest)
            cursor = oldest - 1
            if pages % flush_every == 0:
                st = self.append_cls(pd.concat(buf, ignore_index=True))
                added += st["added"]
                self._meta["cls_items"] = int(self._meta.get("cls_items", 0)) + st["added"]
                self._save_meta()
                buf = []
                if progress:
                    log.info("[altdata] cls 增量 %d 页，新增 %d 条", pages, added)
            if sleep:
                time.sleep(sleep)
        if buf:
            st = self.append_cls(pd.concat(buf, ignore_index=True))
            added += st["added"]
            self._meta["cls_items"] = int(self._meta.get("cls_items", 0)) + st["added"]
        self._save_meta()
        return {"pages": pages, "added": int(added), "cursor": cursor}

    # ------------------------------------------------------------------
    # 缺失日检测
    # ------------------------------------------------------------------
    def macro_missing_forecast_days(self) -> pd.DataFrame:
        """列出「有事件但『预期』整列缺失」的日期 —— 疑为数据漏洞而非真无预期。

        实测样例：2026-09-14 抓回 31 行但 forecast 全空、09-15 抓回 46 行仅 1 条。
        入库方据此决定是否二次补抓，**不得把缺失当成「无预期」参与 surprise 计算**。
        """
        stats: dict = self._meta.get("macro_daily_stats", {})
        rows = [{"date": k, **v} for k, v in sorted(stats.items())
                if v.get("rows", 0) > 0 and v.get("forecast_notna", 0) == 0]
        if not rows:
            return pd.DataFrame(columns=["date", "rows", "forecast_notna"])
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 名称 → 代码映射（补雪球内部交易缺失的 5.5% 代码）
# ---------------------------------------------------------------------------
def load_name_to_code(cache_root: Path | str | None = None) -> dict[str, str]:
    """从项目 ``code_info.parquet`` 读「中文简称 → 标准代码」映射。

    失败（文件缺失 / 列名变化）时返回空 dict —— 调用方按「无映射」降级，
    不抛异常（该映射只用于补全，不是必需路径）。
    """
    if cache_root is None:
        cache_root = Config.cache()["root"]
    root = Path(str(cache_root).replace("//", "/"))
    p = root / "code_info.parquet"
    if not p.exists():
        log.warning("[altdata] code_info.parquet 不存在，跳过名称→代码回补")
        return {}
    try:
        ci = pd.read_parquet(p)
        if "symbol" not in ci.columns:
            log.warning("[altdata] code_info 无 symbol 列（%s），跳过回补", list(ci.columns))
            return {}
        name = ci["symbol"].astype("string").str.strip()
        code = pd.Series(ci.index.astype(str), index=ci.index)
        m = (pd.DataFrame({"name": name, "code": code})
             .dropna(subset=["name"])
             .drop_duplicates(subset="name", keep="first"))
        return dict(zip(m["name"], m["code"]))
    except Exception as e:  # noqa: BLE001
        log.warning("[altdata] 读 code_info 失败: %s: %s", type(e).__name__, str(e)[:80])
        return {}


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def fetch_altdata(
    datasets: tuple[str, ...] = _SNAPSHOT_TABLES,
    *,
    cache_root: Path | str | None = None,
    refresh: bool = True,
    macro_begin: int | str | None = None,
    macro_end: int | str | None = None,
    macro_max_days: int | None = None,
    cls_begin: int | str | None = None,
    cls_max_pages: int | None = None,
) -> dict[str, pd.DataFrame]:
    """统一入口：抓取另类数据事件表，返回 ``{表名: DataFrame}``。

    Parameters
    ----------
    datasets : tuple
        要抓的表，取自 ``holder_control`` / ``inner_trade`` / ``mgmt_hold`` /
        ``cninfo_holder`` / ``macro_calendar`` / ``cls`` / ``news``。
    refresh : bool
        全量快照类是否重拉；False 时只用本地缓存。
    macro_begin / macro_end : int | str | None
        ``macro_calendar`` 的回补区间（缺省 2018-01-01 至今）。
    macro_max_days : int | None
        本次宏观日历最多抓多少天。
    cls_begin : int | str | None
        ``cls`` 回补的最早日期；为 None 时**只做前向增量**（不往回翻）。
    cls_max_pages : int | None
        本次 ``cls`` 最多抓多少页（限速/试跑用）。

    示例
    ----
    >>> from data.altdata import fetch_altdata
    >>> out = fetch_altdata(("holder_control", "mgmt_hold", "inner_trade"))
    >>> out["holder_control"].shape
    """
    cache = AltDataCache(cache_root)
    result: dict[str, pd.DataFrame] = {}
    for t in datasets:
        if t == "macro_calendar":
            result[t] = cache.get_macro_calendar(
                begin=macro_begin or 20180101, end=macro_end,
                max_days=macro_max_days)
        elif t == "holder_control":
            result[t] = cache.get_holder_control(refresh=refresh)
        elif t == "inner_trade":
            result[t] = cache.get_inner_trade(refresh=refresh)
        elif t == "mgmt_hold":
            result[t] = cache.get_mgmt_hold(refresh=refresh)
        elif t == "cninfo_holder":
            result[t] = cache.get_cninfo_holder(refresh=refresh)
        elif t == "cls":
            if cls_begin is not None:
                result[t] = cache.backfill_cls(
                    begin=cls_begin, max_pages=cls_max_pages)
            else:
                result[t] = cache.sync_cls_recent(max_pages=cls_max_pages or 400)
            result[t] = cache.load_cls()
        elif t == "news":
            cache.append_news()               # 抓取当前快照并追加（幂等）
            result[t] = cache.load_news()     # 返回当日全量存档
        else:
            raise ValueError(
                f"未知数据集 {t!r}；可用：{list(_FILES)} + ['cls', 'news']")
    return result
