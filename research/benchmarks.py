"""
基准序列模块（Benchmarks）
=========================

为回测/因子报告提供**统一的基准对照**：把"策略赚的是 alpha 还是 beta"一眼
看清楚。四种基准（均基于现有 close 面板，无额外数据依赖）：

- ``equal_weight`` ：全市场等权（**每日**再平衡）——研究最常见的朴素基准。
- ``buy_hold``     ：期初等权买入持有，权重随价格漂移（**全程**不调仓）。
- ``segmented``    ：**按调仓日重置等权**，段内权重随价漂移 —— 与月/周频策略同频，
                     是策略"超额"该用的基准。**做池子过滤时用它，别手写。**
- ``index``        ：指数收益（预留接口；当前数据层无指数行情，可传入外部序列）。

对照指标（``compare_to_benchmark``）：超额年化 / 信息比率 / 跟踪误差 /
相关性 / 简化 beta —— 与 ``backtest.metrics.calc_all_metrics`` 无缝合并。

口径：所有收益序列为日频，index=date。

🚨 **池化基准必须走 ``segmented_equal_weight_returns``**（2026-11-05 立规）
------------------------------------------------------------------
2026-11 的 "池子范围探针" 因框架缺"按调仓日重置 + 任意池"这一档，调用方
（``scripts/evaluation/pool_scope_probe.py``）手写了私有 ``_bench_returns``，
一次引入两个口径 bug，导致结论级数字两次反转：

1. **稀释**：权重按全部面板列归一，而 2018 年只有 3134/5801 列有价格
   ⇒ 42% 权重落在"收益记 0"的空列上，全A 等权 −29.5% 被算成 −18.4%。
2. **丢段**：从**窗口裁剪后**的价格算首日收益，窗口首日无前值 ⇒ 2024-01
   （微盘暴跌月）整段被跳过，基准从 4.5% 被抬到 28.9%。

本模块现在把这两件事**内建为防线**（不是靠文档提醒）：
- 一律在**完整** ``close_panel`` 上算收益，再裁输出索引；
- 段首有效列数异常偏少时**直接报错**，而不是静默用空列占权重。

⇒ 任何人要"某池子的平均表现"，传 ``investable=mask`` 即可，不要手写。
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from backtest.metrics import PERIODS_PER_YEAR

__all__ = ["equal_weight_returns", "buy_hold_returns",
           "segmented_equal_weight_returns", "monthly_first_trading_days",
           "benchmark_returns", "compare_to_benchmark", "with_benchmark_metrics"]


def monthly_first_trading_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每月**第一个**交易日（= 引擎 ``groupby(to_period('M')).first()`` 的调仓日）。"""
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).first().to_numpy())


def equal_weight_returns(close_panel: pd.DataFrame) -> pd.Series:
    """全市场等权日收益（每日再平衡，等权是默认行业/市值中性化的朴素对照）。

    对每只股票先算日收益（pct_change），再逐日取横截面均值。
    """
    return close_panel.pct_change(fill_method=None).mean(axis=1, skipna=True)


def buy_hold_returns(close_panel: pd.DataFrame) -> pd.Series:
    """期初等权买入持有：权重随价格漂移、期内不调仓（成本最低的持有基准）。

    组合净值 = Σ w_i × (close_i / close_i0)；收益 = 净值 pct_change。

    ⚠️ ``mean(skipna=True)`` 会在股票**陆续上市**时改变成分（新列从 NaN 变有值
    后自动计入等权）⇒ 严格说不等于"期初等权"。需要严格口径请用
    :func:`segmented_equal_weight_returns`（它对段首无数据的股票不分配权重）。
    """
    first = close_panel.iloc[0].replace(0, np.nan)
    nav_i = close_panel.div(first, axis=1)          # 各股归一净值
    port_nav = nav_i.mean(axis=1, skipna=True)      # 期初等权 → 权重随价格漂移
    return port_nav.pct_change(fill_method=None)


def segmented_equal_weight_returns(
    close_panel: pd.DataFrame,
    rebalance_dates: Sequence | None = None,
    index: pd.DatetimeIndex | None = None,
    *,
    investable: pd.DataFrame | None = None,
    min_valid_frac: float = 0.5,
) -> pd.Series:
    """**按调仓日重置等权、段内权重随价漂移**的日收益序列（池化基准唯一实现）。

    与月频/周频策略同频 —— 策略超额该拿它对标。

    Parameters
    ----------
    close_panel
        **完整**收盘价面板（date × code，建议后复权）。必须包含窗口首日**之前**
        的至少一个交易日，否则首段无前值（会 warning 并丢首日）。
        🚨 **不要**用 ``close_panel.where(mask)`` 表达池过滤：那样窗口前一日
        的价格也被抹掉 ⇒ 首日收益算不出来。请用 ``investable=``。
    rebalance_dates
        段起点（调仓日）。``None`` ⇒ 每月第一个交易日。传入的日期若不在
        ``close_panel.index`` 上，会自动取其后第一个可用交易日。
    index
        输出索引（回测窗口）。段仍在**完整**日期上切分，输出裁到此索引
        ⇒ 窗口首日恰为调仓日也**不会**丢段。
    investable
        ``date × code`` 的 bool 池归属矩阵（PIT）。非池内股票**不分配权重**。
        只在该段首日取值；后续日子里退出池子的股票不会被强制卖出
        （与"月内买入持有"语义一致）。
    min_valid_frac
        段首有效列数 / 该段应有权重列数。低于此比例视为"面板被裁剪"并报错。
        默认 0.5（正常停牌比例远低于此）。

    Returns
    -------
    pd.Series
        日收益（首段首日为 0，其余为 0 或实际值）。

    Notes
    -----
    段内权重**随价格漂移**（买入后不动），不是每日再平衡到等权 —— 后者会引入
    再平衡收益，对高波动小盘池影响可达每年数 pp。

    Examples
    --------
    >>> bench = segmented_equal_weight_returns(
    ...     close, index=dates, investable=membership)   # 同池等权基准
    """
    if close_panel.empty:
        return pd.Series(dtype=float)

    px = close_panel.replace([np.inf, -np.inf], np.nan)
    # 收益里的 inf 也要清：价格从 0 恢复会算出 inf，而零权重 × inf = NaN，
    # 会让 matmul 整段变 NaN（2026-11-05 实测 hs300 臂触发）。
    ret = (px.pct_change(fill_method=None)
             .replace([np.inf, -np.inf], np.nan))   # 完整面板上算，避免首日无前值

    # 面板首日无前值 ⇒ 该日收益本来就不可知，丢弃（不影响后续任何一段）。
    # 若该日**在窗口内**，说明调用方传的是裁剪面板 ⇒ 报警（窗口首月会丢）。
    # 若该日在窗口外（前值锚），丢它是预期行为，不吵。
    if len(ret) and ret.iloc[0].isna().all():
        if index is None or ret.index[0] in index:
            warnings.warn(
                f"close_panel 首日 {ret.index[0].date()} 无前值，已丢弃该日收益。"
                "若这是**窗口裁剪**的结果（而非数据绝对起点），请改传完整面板并用 "
                "`index=` 裁输出，否则窗口首月的收益会丢失。",
                RuntimeWarning, stacklevel=2)
        ret = ret.iloc[1:]

    dates = ret.index
    if rebalance_dates is None:
        starts = monthly_first_trading_days(dates)
    else:
        starts = pd.DatetimeIndex(sorted(set(rebalance_dates)))
        # 对齐到 >= 该日的第一个可用交易日
        pos = dates.searchsorted(starts, side="left")
        pos = np.clip(pos, 0, len(dates) - 1)
        starts = pd.DatetimeIndex(dates[pos])
        starts = pd.DatetimeIndex(sorted(set(starts)))

    if len(starts) == 0:
        return pd.Series(np.nan, index=index if index is not None else dates)
    if dates[0] not in starts:
        starts = pd.DatetimeIndex([dates[0], *starts])

    pool = None
    if investable is not None:
        pool = (investable.reindex(index=dates, columns=px.columns)
                .astype("boolean").fillna(False).astype(bool))

    seg_id = pd.Series(np.nan, index=dates)
    for k, d in enumerate(starts):
        seg_id.loc[d:] = k
    seg_id = seg_id.ffill()

    out = pd.Series(np.nan, index=dates)
    for _, sub in ret.groupby(seg_id):
        if sub.empty:
            continue
        first = sub.index[0]
        valid_start = sub.iloc[0].notna()
        n_start = int(valid_start.sum())

        if pool is not None:
            in_pool = pool.loc[first]
            n_expect = int(in_pool.sum())
            w_start = (in_pool & valid_start).to_numpy(dtype=float)
        else:
            n_expect = int(sub.notna().any(axis=0).sum())
            w_start = valid_start.to_numpy(dtype=float)

        # 防线：段首有效列数异常偏少 ⇒ 多半是传了窗口裁剪后的面板
        if n_expect > 0 and n_start < min_valid_frac * n_expect:
            raise ValueError(
                f"调仓日 {first.date()} 有效列数 {n_start} 远少于应有 "
                f"{n_expect} 列（阈值 {min_valid_frac:.0%}）。\n"
                "这通常意味着传入的 close_panel 被窗口裁剪过、窗口首日没有前值。"
                "请传**完整**面板（含窗口首日之前一日），本函数会在算完收益后按 "
                "`index=` 裁剪输出。")

        n = w_start.sum()
        if n == 0:
            continue
        w = w_start / n

        # 段内权重漂移：组合净值 = Σ w_i · (累计净值_i)
        # 停牌日（NaN）记 0 收益 ⇒ 该股净值冻结、隐含权重随他人涨跌被动变化。
        nav = (1.0 + sub.fillna(0.0)).cumprod()
        port_nav = nav.to_numpy() @ w                # w 已归一 ⇒ 即 Σ w_i·nav_i
        prev = np.concatenate([[1.0], port_nav[:-1]])
        out.loc[sub.index] = port_nav / prev - 1.0

    out = out.fillna(0.0)
    return out.reindex(index).fillna(0.0) if index is not None else out



def benchmark_returns(close_panel: pd.DataFrame, mode: str = "equal_weight",
                      **kwargs) -> pd.Series:
    """基准序列入口。

    mode: ``'segmented'``（按调仓日重置等权，**策略超额首选**）|
    ``'equal_weight'``（每日再平衡）| ``'buy_hold'``（全程不调仓）。

    ``segmented`` 额外接受 ``rebalance_dates`` / ``index``，见
    :func:`segmented_equal_weight_returns`。
    """
    if mode == "segmented":
        return segmented_equal_weight_returns(close_panel, **kwargs)
    if mode == "buy_hold":
        return buy_hold_returns(close_panel)
    return equal_weight_returns(close_panel)


def compare_to_benchmark(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict:
    """策略 vs 基准的对照指标（日频序列已对齐）。"""
    from backtest.metrics import annual_return

    a, b = daily_returns.align(benchmark_returns, join="inner")
    if len(a) < 2:
        return {"excess_annual": np.nan, "information_ratio": np.nan,
                "tracking_error": np.nan, "correlation": np.nan, "beta": np.nan}
    excess = a - b
    te = float(excess.std() * np.sqrt(periods_per_year))
    ir = float(excess.mean() * periods_per_year / te) if te > 0 else 0.0
    corr = float(a.corr(b)) if a.std() > 0 and b.std() > 0 else np.nan
    beta = float(a.cov(b) / b.var()) if b.var() > 0 else np.nan
    return {
        "excess_annual": float(annual_return(a, periods_per_year)
                               - annual_return(b, periods_per_year)),
        "information_ratio": ir,
        "tracking_error": te,
        "correlation": corr,
        "beta": beta,
    }


def with_benchmark_metrics(
    metrics: dict,
    daily_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict:
    """把基准对照指标并入 ``calc_all_metrics`` 的指标 dict（原 dict 不修改）。"""
    out = dict(metrics)
    out["benchmark"] = compare_to_benchmark(daily_returns, benchmark_returns, periods_per_year)
    return out
