"""
离线/在线通用的缓存读取工具
============================

把散落在 scripts/ 各入口里的重复"从缓存加载数据"逻辑收敛到这里：

1. ``load_daily``            日历校验 + 成分股 + 日线（4 处 load_data 的公共开头）
2. ``load_backward_factor``  后复权因子（离线读 parquet / 在线走 cache，双分支）
3. ``load_financial_tables`` 财务/股东事件表（同上双分支，7 张表统一）
4. ``returns_from_cache``    从 daily.parquet 构造次日收益面板（因子库 IC 口径）

约定：offline 判定依据 cache._ds 是否为 data.offline.OfflineDataSource——
离线时直接读本地 parquet（避免整表覆盖型接口回调数据源抛错），在线时走
cache 接口。2026-08-05 统一（此前 select_stocks/synthesize_library 逐字
重复 returns_from_cache；build_*/intraday_analysis 各自实现 load_data；
walk_forward/backtest_two_periods/build_fundamental_factors 各自实现
_fin 离线读财务表）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.offline import OfflineDataSource


def _is_offline(cache) -> bool:
    return isinstance(getattr(cache, "_ds", None), OfflineDataSource)


def load_daily(cache, uni, index_code: str, begin: int, end: int | None):
    """日历校验 + 成分股 + 日K线。

    Returns:
        (codes, cal, daily): 股票池 / 交易日列表(YYYYMMDD int) / (date, code) 长表。
    """
    cal = cache.get_calendar(begin, end)
    if not cal:
        raise RuntimeError(f"交易日历为空（{begin}-{end}），请先更新数据")
    target = end or cal[-1]
    codes = uni.get_constituent(index_code, target)
    daily = cache.get_daily_kline(codes, begin, target)
    return codes, cal, daily


def load_backward_factor(cache, codes) -> pd.DataFrame:
    """累积后复权因子（date×code 宽表）。

    复权因子是"宽表全量刷新"型接口（_refresh_wide_table 每次回调数据源），
    offline 桩会报错 → 离线时直接读本地 parquet 并按股票池过滤。
    """
    if _is_offline(cache):
        p = Path(cache.root) / "backward_factor.parquet"
        if not p.exists():
            return pd.DataFrame()
        bf = pd.read_parquet(p)
        return bf[[c for c in codes if c in bf.columns]]
    return cache.get_backward_factor(codes)


# 财务/股东事件表清单（文件名 == cache 接口名）
_FINANCIAL_TABLES = (
    "income", "balance_sheet", "cash_flow", "equity_structure",
    "dividend", "share_holder", "holder_num",
)


def load_financial_tables(cache, codes) -> dict[str, pd.DataFrame]:
    """财务/股东事件表（稀疏长表）：{表名: DataFrame}。

    财务表缓存是"整表覆盖"模式（每次调用都回调数据源），offline 桩会报错；
    离线时直接读本地 parquet 并按股票池过滤。
    """
    if _is_offline(cache):
        out: dict[str, pd.DataFrame] = {}
        for name in _FINANCIAL_TABLES:
            p = Path(cache.root) / f"{name}.parquet"
            df = pd.read_parquet(p) if p.exists() else pd.DataFrame()
            out[name] = df[df["code"].isin(codes)] if "code" in df.columns else df
        return out
    getters = {
        "income": cache.get_income,
        "balance_sheet": cache.get_balance_sheet,
        "cash_flow": cache.get_cash_flow,
        "equity_structure": cache.get_equity_structure,
        "dividend": cache.get_dividend,
        "share_holder": cache.get_share_holder,
        "holder_num": cache.get_holder_num,
    }
    return {k: g(codes) for k, g in getters.items()}


def returns_from_cache(cache, begin: int, end: int) -> pd.DataFrame:
    """从日线缓存构造次日收益面板（与因子库 IC 口径一致）。"""
    cal = cache.get_calendar(begin, end)
    if not cal:
        raise RuntimeError("交易日历为空")
    d = pd.read_parquet(Path(cache.root) / "daily.parquet")
    close_w = d.reset_index().pivot(index="date", columns="code", values="close").sort_index()
    close_w = close_w.loc[str(begin): str(end)]
    return close_w.pct_change().shift(-1)
