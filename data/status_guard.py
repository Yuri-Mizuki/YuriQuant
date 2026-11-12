"""状态表断更守卫 —— 单一模块
================================

``history_stock_status.parquet``（涨跌停价 / 停牌 / ST / 除权标记）是**可交易性
判定的唯一真源**。它一旦断更，下游会静默降级，而且降级方向**不可见**：

- ``build_tradable_mask``：缺失 (date, code) 视为可交易（补 True）；
- ``signal_day_tradable``：状态表无信号日时退回「日线行情推断涨跌停」。

2026-11-10 事故正是这条链路：状态表止于 **10-22**，10-23 起剔除率从 3.4~4.8%
直接掉到 **0**，无人报警；定版出榜连续多日混进 23~24 只 ST。

本模块的职责只有一件事：**把「状态表够不够新」变成一个可判定、可报警的事实**，
供 ``update_data``（数据侧）与出榜链（消费侧）在入口处调用。

设计要点
--------

1. **判据用交易日差，不用自然日差**。状态表按交易日存，周五→周一的 3 个自然日
   不是断更；用工作日差会把每个周一都报成告警。这里用「最近交易日」归一化。
2. **阈值分两档**。``warn_days``（默认 1）→ 降级可用但需提示；``error_days``
   （默认 5，即一整周）→ 判定断更，ST/停牌判定已不可信。
3. **绝不因守卫自身失败而阻断主流程**。状态表不存在 / 索引不可读 / 日历不可用
   时，返回 ``severity="unknown"`` 并说明原因，调用方自行决定怎么处理。
   守卫是**诊断**，不是新的一道硬闸门。
4. **只读索引**。``columns=[]`` 读 85MB 表约 0.1s，可以在每次出榜时调用。
5. **日期新 ≠ 状态新**。补拉按批落盘、可被打断，合并后可能出现「最新日期很新、
   但只有少数标的更新过」。只判日期会给出**假 ok**，故同时校验最后一天的
   股票覆盖率（相对历史单日峰值）。这是 2026-11-10 恢复补拉时暴露的缺口。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from config import Config

log = logging.getLogger(__name__)

# 默认阈值（交易日）。1 天 = 当日状态尚未落盘（正常时序：出榜前才更新）；
# 5 天 = 一整周，此时 ST/停牌/涨跌停全部不可信。
DEFAULT_WARN_DAYS = 1
DEFAULT_ERROR_DAYS = 5
# 最后一天的股票覆盖率下限（相对历史单日峰值）。低于它说明「日期到位但标的不全」，
# 典型成因是分批补拉被打断后合并 —— 见 _coverage_on_last_day 的说明。
DEFAULT_COVERAGE_MIN_RATIO = 0.95


@dataclass
class StatusFreshness:
    """状态表新鲜度检查结果。"""

    #: ``ok`` / ``warn`` / ``error`` / ``unknown``
    severity: str
    #: 状态表最后一天（Timestamp），不可读时为 None
    last_date: pd.Timestamp | None = None
    #: 参照日（通常是要判定的信号日 / 目标交易日）
    asof: pd.Timestamp | None = None
    #: 缺失的交易日数（asof 与 last_date 之间的交易日个数，不含两端）
    missing_days: int = 0
    #: 人类可读说明
    message: str = ""
    #: 附加上下文（行数、路径等），便于日志排查
    detail: dict = field(default_factory=dict)

    @property
    def is_stale(self) -> bool:
        """是否已达到「认定断更」的严重度（``error``）。"""
        return self.severity == "error"

    def __bool__(self) -> bool:
        """``if freshness:`` 表示**健康**（ok 或 warn 以下），便于调用方短路。"""
        return self.severity in ("ok", "unknown")


def _status_path(cache_root: str | Path | None = None) -> Path:
    root = cache_root or str(Config.cache()["root"])
    return Path(root) / "history_stock_status.parquet"


def _last_status_date(path: Path) -> tuple[pd.Timestamp | None, int]:
    """只读索引，返回（最后日期, 行数）。任何异常都吞掉并返回 (None, 0)。"""
    if not path.exists():
        return None, 0
    try:
        idx = pd.read_parquet(path, columns=[]).index
        if len(idx) == 0:
            return None, 0
        dates = idx.get_level_values("date")
        return pd.Timestamp(dates.max()), int(len(idx))
    except Exception:  # noqa: BLE001 - 守卫不阻断主流程
        return None, 0


def _coverage_on_last_day(path: Path,
                          last: pd.Timestamp | None) -> tuple[int, int]:
    """返回（最后一天实际覆盖的股票数, 历史单日最大覆盖数）。

    为什么需要：只看「最后一天是什么日期」会漏掉**部分补拉**。补拉脚本按批落盘，
    中断后合并的结果是「600 只覆盖到 11-10、其余 4959 只停在 10-22」——
    此时 ``last_date`` 已是 11-10，纯日期判据会给出 **ok（假健康）**，
    而 89% 的股票仍在用旧状态，比直接报断更更危险。

    用「历史单日最大覆盖数」作分母而不是固定值：覆盖率本就随上市/退市变化，
    写死 5559 会在扩容后误报。取历史峰值意味着「相对自己最全的一天」看缺口。
    """
    if last is None or not path.exists():
        return 0, 0
    try:
        # 只读 index，不读列；先取最后一天的覆盖
        idx = pd.read_parquet(path, columns=[]).index
        dates = idx.get_level_values("date")
        counts = pd.Series(1, index=dates).groupby(level=0).size()
        if len(counts) == 0:
            return 0, 0
        peak = int(counts.max())
        cur = int(counts.get(pd.Timestamp(last), 0))
        return cur, peak
    except Exception:  # noqa: BLE001
        return 0, 0


def _trading_days_between(start: pd.Timestamp, end: pd.Timestamp,
                          cache_root: str | Path | None = None) -> int | None:
    """start 与 end 之间的交易日个数（**不含两端**）。

    优先用本地 ``calendar.parquet``。日历不可用时返回 None，调用方退化为
    工作日差 —— 后者对周末偏差为 0、对长假会高估，故仅作兜底。
    """
    if end <= start:
        return 0
    root = cache_root or str(Config.cache()["root"])
    cal_path = Path(root) / "calendar.parquet"
    if cal_path.exists():
        try:
            cal = pd.read_parquet(cal_path, columns=["date"])
            # 日历是 int YYYYMMDD；pd.to_datetime(int) 会落到 1970（项目铁律），
            # 必须 astype(str) + format。兼容已是 datetime 的旧版日历。
            raw = cal["date"]
            if pd.api.types.is_integer_dtype(raw):
                days = pd.to_datetime(raw.astype(str), format="%Y%m%d")
            else:
                days = pd.to_datetime(raw)
            mask = (days > start) & (days < end)
            return int(mask.sum())
        except Exception:  # noqa: BLE001
            pass
    try:
        bd = pd.bdate_range(start, end, inclusive="neither")
        return int(len(bd))
    except Exception:  # noqa: BLE001
        return None


def _normalize_date(value: pd.Timestamp | str | int) -> pd.Timestamp:
    """把 ``20261110`` / ``"20261110"`` / ``"2026-11-10"`` / Timestamp 统一成 Timestamp。

    注意 ``pd.Timestamp(20261110)`` 会被当成纳秒时间戳、落到 1970 年 ——
    这是项目里反复踩过的坑（见 MEMORY.md 的 PIT 铁律），必须显式拼字符串。
    """
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, int):
        s = str(value)
    else:
        s = str(value).strip()
    s = s.replace("-", "").replace("/", "")
    if len(s) >= 8 and s[:8].isdigit():
        return pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]}")
    return pd.Timestamp(value)


def check_status_freshness(
    asof: pd.Timestamp | str | int | None = None,
    *,
    warn_days: int = DEFAULT_WARN_DAYS,
    error_days: int = DEFAULT_ERROR_DAYS,
    coverage_min_ratio: float = DEFAULT_COVERAGE_MIN_RATIO,
    cache_root: str | Path | None = None,
    logger: logging.Logger | None = None,
    raise_on_error: bool = False,
) -> StatusFreshness:
    """检查状态表相对 ``asof`` 是否断更。

    典型调用点（两处，缺一不可）：

    - ``scripts/ingest/update_data.py``：拉取前置检查 —— 数据侧知道自己该补哪天；
    - ``scripts/pipelines/alla_daily_rank.py``：出榜前置检查 —— 消费侧在**开始
      选股之前**就知道 ST 判定不可信，从而不再默默降级。

    Args:
        asof: 参照日期。接受 ``Timestamp`` / ``"YYYY-MM-DD"`` / ``20261110`` 整数。
            None 时取状态表自身最后日期（只判「表是否存在且非空」，不判新鲜度）。
        warn_days: 缺失超过该交易日数 → ``warn``。
        error_days: 缺失超过该交易日数 → ``error``（判定断更）。
        coverage_min_ratio: 最后一天股票覆盖率下限（相对历史单日峰值）。低于它时
            即使日期是新的也降为 ``warn``，防「分批补拉未完成」的假 ok。
        cache_root: 缓存根，None 取 ``Config.cache()["root"]``。
        logger: 传入则在 ``warn``/``error`` 时打日志；None 用模块 logger。
        raise_on_error: 为 True 且判定 ``error`` 时抛 ``RuntimeError``。
            默认 False —— 守卫默认只诊断，不改变现有链路行为。

    Returns:
        :class:`StatusFreshness`。``severity`` 取值：
        ``ok`` / ``warn`` / ``error`` / ``unknown``。

    Examples:
        >>> r = check_status_freshness("2026-11-02")   # doctest: +SKIP
        >>> if r.is_stale:
        ...     print("ST 判定不可信:", r.message)
    """
    lg = logger or log
    path = _status_path(cache_root)
    last, n_rows = _last_status_date(path)
    detail = {"path": str(path), "rows": n_rows,
              "warn_days": warn_days, "error_days": error_days,
              "coverage_min_ratio": coverage_min_ratio}

    if last is None:
        res = StatusFreshness(
            severity="unknown", last_date=None, asof=None, missing_days=0,
            message=f"状态表不存在或索引不可读：{path.name}",
            detail=detail,
        )
        lg.warning("状态表守卫：%s", res.message)
        return res

    if asof is None:
        res = StatusFreshness(
            severity="ok", last_date=last, asof=last, missing_days=0,
            message=f"状态表最后一天 {last.date()}（仅确认非空）",
            detail=detail,
        )
        return res

    asof_ts = _normalize_date(asof)

    missing = _trading_days_between(last, asof_ts, cache_root)
    if missing is None:
        res = StatusFreshness(
            severity="unknown", last_date=last, asof=asof_ts, missing_days=0,
            message="无法计算交易日差（日历与工作日兜底均失败）",
            detail=detail,
        )
        lg.warning("状态表守卫：%s", res.message)
        return res

    detail["missing_trading_days"] = missing
    if missing <= warn_days:
        sev = "ok"
    elif missing <= error_days:
        sev = "warn"
    else:
        sev = "error"

    msg = (f"状态表最后一天 {last.date()}，参照日 {asof_ts.date()}，"
           f"缺 {missing} 个交易日")

    # ---- 覆盖率检查：防「部分补拉」造成的假 ok ----
    # 日期到位但只有一小撮股票更新过，等价于绝大多数标的状态仍是旧的。
    cur_cov, peak_cov = _coverage_on_last_day(path, last)
    ratio = (cur_cov / peak_cov) if peak_cov else 1.0
    detail["coverage_last_day"] = cur_cov
    detail["coverage_peak"] = peak_cov
    detail["coverage_ratio"] = round(ratio, 4)
    if peak_cov and ratio < coverage_min_ratio and sev == "ok":
        sev = "warn"
        msg += (f"；但最后一天仅覆盖 {cur_cov}/{peak_cov} 只"
                f"（{ratio:.1%} < {coverage_min_ratio:.0%}）"
                f"⇒ 多数标的状态仍停留在旧日期（疑似分批补拉未完成）")

    res = StatusFreshness(severity=sev, last_date=last, asof=asof_ts,
                          missing_days=missing, message=msg, detail=detail)

    if sev == "warn":
        lg.warning("🚨 状态表偏旧：%s（ST/停牌判定仍可用，但需尽快补拉）", msg)
    elif sev == "error":
        lg.error("🚨🚨 状态表断更：%s ⇒ ST/停牌/涨跌停判定**不可信**，"
                 "下游会静默降级。请先补拉："
                 "python -m scripts.ingest.fetch_status_batched --begin %s --end %s",
                 msg, last.strftime("%Y%m%d"), asof_ts.strftime("%Y%m%d"))
        if raise_on_error:
            raise RuntimeError(f"状态表断更：{msg}")
    return res


__all__ = [
    "StatusFreshness",
    "check_status_freshness",
    "DEFAULT_WARN_DAYS",
    "DEFAULT_ERROR_DAYS",
    "DEFAULT_COVERAGE_MIN_RATIO",
]