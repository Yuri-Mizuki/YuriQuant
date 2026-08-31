"""公开经典因子集（alpha101 / gtja191）的公共面板算子。

两个因子集共用一套「date×code 面板」语义（与 factor/operators.py 一致），
本模块只补齐它们需要、而算子空间缺失或语义不同的部分：

- ``AlphaData``：数据命名空间（open/high/low/close/volume/amount/vwap）+
  派生字段缓存（returns、adv_n）；
- ``if_else``：三目（NaN 条件 → NaN，避免 warmup 期误走假分支）；
- ``sma_tdx``：通达信/GTJA 的递归 SMA(X,N,M)（ewm alpha=M/N）；
- ``sumif`` / ``count_`` / ``highday`` / ``lowday``：GTJA 公式专用；
- ``ind_neutralize``：行业中性化（按日组内去均值，行业面板缺省时恒等）；
- ``scale_wq``：WorldQuant scale(x) = x / Σ|x|（逐截面）。

价格口径约定：调用方传入**后复权** OHLC 与 vwap（vwap = amount/volume ×
后复权因子），returns = 复权 close 的 pct_change(fill_method=None)。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from factor.operators import (
    abs_,
    cs_rank,
    cs_scale_abs,
    log_,
    max_,
    min_,
    sign,
    ts_arg_max,
    ts_arg_min,
    ts_corr,
    ts_cov,
    ts_decay_linear,
    ts_delay,
    ts_delta,
    ts_max,
    ts_mean,
    ts_min,
    ts_product,
    ts_rank,
    ts_slope,
    ts_std,
    ts_sum,
    ts_wma,
)

log = logging.getLogger("alpha_base")

__all__ = [
    "AlphaData", "if_else", "sma_tdx", "sumif", "count_", "highday", "lowday",
    "ind_neutralize", "scale_wq", "lt", "w",
    "rank", "tsr", "corr", "cov", "delta", "delay", "stddev",
    "tsmax", "tsmin", "argmax", "argmin", "decay_linear", "product",
    "abs_", "sign", "log_", "power", "max_", "min_", "cs_rank", "ts_slope",
    "gt", "ge", "lt", "le", "eq", "and_", "or_",
]

# ---- 简写别名（公式书写用，语义与 factor/operators 完全一致）----
rank = cs_rank          # 截面百分位 [0,1]
tsr = ts_rank           # 时序百分位 [0,1]
corr = ts_corr
cov = ts_cov
delta = ts_delta
delay = ts_delay
stddev = ts_std
tsmax = ts_max
tsmin = ts_min
decay_linear = ts_decay_linear
product = ts_product
# ts_arg_max/min 归一化到 [0,1]（见 operators.py），alpha101 的 rank/decay
# 下游均为秩不变或线性缩放，IC（Spearman）不受影响；GTJA 的 HIGHDAY/LOWDAY
# 需还原为天数，见 highday/lowday。
argmax = ts_arg_max
argmin = ts_arg_min


def w(x: float) -> int:
    """论文公式中的小数窗口（如 12.6556）→ 整数窗口（四舍五入，≥1）。"""
    return max(1, int(round(x)))


def power(x: pd.DataFrame, n: float) -> pd.DataFrame:
    """幂（底数为负且指数非整数时 numpy 自然给出 NaN，与公开实现一致）。"""
    return x.pow(n)


# ---------------------------------------------------------------------------
# 三目 / 布尔
# ---------------------------------------------------------------------------
def _p(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, pd.Series):
        return pd.DataFrame(np.tile(x.values.reshape(-1, 1), (1, 1)), index=x.index)
    raise TypeError(f"期望 DataFrame/Series，得到 {type(x)}")


def _scalar_df(val: float, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(val, index=like.index, columns=like.columns)


def if_else(cond, a, b):
    """三目：cond>0 取 a 否则 b；cond 为 NaN 时结果 NaN。

    a/b 可为面板或标量（标量按 cond 形状广播）。
    """
    c = cond if isinstance(cond, pd.DataFrame) else _p(cond)
    if not isinstance(a, pd.DataFrame):
        a = _scalar_df(float(a), c)
    if not isinstance(b, pd.DataFrame):
        b = _scalar_df(float(b), c)
    out = pd.DataFrame(
        np.where(c.values > 0, a.values, b.values),
        index=c.index, columns=c.columns,
    )
    return out.where(c.notna())


def _bcast(x, like: pd.DataFrame):
    """标量 → 与 like 同形状面板；面板原样返回。"""
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(float(x), index=like.index, columns=like.columns)


def _cmp(op, a, b) -> pd.DataFrame:
    a_, b_ = _bcast(a, a if isinstance(a, pd.DataFrame) else b), _bcast(b, a if isinstance(a, pd.DataFrame) else b)
    out = op(a_, b_).astype(float)
    return out.where(a_.notna() & b_.notna())


def lt(a, b) -> pd.DataFrame:
    """a < b 的 0/1 面板（任一侧 NaN → NaN；一侧可为标量）。"""
    return _cmp(lambda x, y: x < y, a, b)


def gt(a, b) -> pd.DataFrame:
    """a > b 的 0/1 面板（任一侧 NaN → NaN；一侧可为标量）。"""
    return _cmp(lambda x, y: x > y, a, b)


def ge(a, b) -> pd.DataFrame:
    """a >= b 的 0/1 面板（任一侧 NaN → NaN；一侧可为标量）。"""
    return _cmp(lambda x, y: x >= y, a, b)


def le(a, b) -> pd.DataFrame:
    """a <= b 的 0/1 面板（任一侧 NaN → NaN；一侧可为标量）。"""
    return _cmp(lambda x, y: x <= y, a, b)


def eq(a, b) -> pd.DataFrame:
    """a == b 的 0/1 面板（任一侧 NaN → NaN；一侧可为标量）。"""
    return _cmp(lambda x, y: x == y, a, b)


def and_(a, b) -> pd.DataFrame:
    """两个 0/1 条件面板的逻辑与（NaN 传播）。"""
    a_, b_ = _bcast(a, b), _bcast(b, a)
    out = ((a_ > 0) & (b_ > 0)).astype(float)
    return out.where(a_.notna() & b_.notna())


def or_(a, b) -> pd.DataFrame:
    """两个 0/1 条件面板的逻辑或（NaN 传播）。"""
    a_, b_ = _bcast(a, b), _bcast(b, a)
    out = ((a_ > 0) | (b_ > 0)).astype(float)
    return out.where(a_.notna() & b_.notna())


# ---------------------------------------------------------------------------
# GTJA 专用算子
# ---------------------------------------------------------------------------
def sma_tdx(x: pd.DataFrame, n: int, m: int = 1) -> pd.DataFrame:
    """通达信/GTJA SMA(X,N,M)：Y = (M*X + (N-M)*Y')/N，等价 ewm(alpha=M/N)。"""
    alpha = m / n
    return x.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def sumif(x: pd.DataFrame, n: int, cond: pd.DataFrame) -> pd.DataFrame:
    """过去 n 日 x 在 cond>0 处的和（cond≤0 处计 0）。"""
    contrib = x.where(cond > 0, 0.0)
    return contrib.rolling(n, min_periods=n).sum()


def count_(cond: pd.DataFrame, n: int) -> pd.DataFrame:
    """过去 n 日 cond>0 成立的天数。"""
    return (cond > 0).astype(float).rolling(n, min_periods=n).sum()


def highday(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """过去 n 日最高点距当期的天数（0=当日即最高）。"""
    return (n - 1) * (1.0 - ts_arg_max(x, n))


def lowday(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """过去 n 日最低点距当期的天数（0=当日即最低）。"""
    return (n - 1) * (1.0 - ts_arg_min(x, n))


# ---------------------------------------------------------------------------
# 截面算子（WQ 语义）
# ---------------------------------------------------------------------------
def scale_wq(x: pd.DataFrame) -> pd.DataFrame:
    """WorldQuant scale(x)：x / Σ|x|，逐截面（Σ|x|=0 时置 NaN）。"""
    return cs_scale_abs(x)


def ind_neutralize(x: pd.DataFrame, industry: pd.DataFrame | None) -> pd.DataFrame:
    """行业中性化：按日行业组内去均值；industry 为 None 时恒等返回。

    industry 为 date×code → 行业代码面板；无行业归属（NaN）的股票归为
    同一"未知"组，避免被误去均值成 0。
    """
    if industry is None or industry.empty:
        return x
    ind = industry.reindex(index=x.index, columns=x.columns)
    parts = []
    for dt, row in x.iterrows():
        g = ind.loc[dt]
        vals = row.astype(float)
        grp = g.where(vals.notna())
        demeaned = vals - vals.groupby(grp).transform("mean")
        parts.append(demeaned)
    return pd.DataFrame(parts, index=x.index, columns=x.columns)


# ---------------------------------------------------------------------------
# 数据命名空间
# ---------------------------------------------------------------------------
class AlphaData:
    """alpha 公式的数据输入：基础面板 + 派生字段（returns / adv_n）懒加载缓存。

    Args:
        panels: {字段: date×code 面板}，需含 open/high/low/close/volume/amount，
               可选 vwap（缺省用 amount/volume）。
        industry: 行业面板（date×code → 行业代码），供 ind_neutralize。
    """

    def __init__(self, panels: dict[str, pd.DataFrame],
                 industry: pd.DataFrame | None = None):
        self.open = panels["open"]
        self.high = panels["high"]
        self.low = panels["low"]
        self.close = panels["close"]
        self.volume = panels["volume"]
        self.amount = panels["amount"]
        if "vwap" in panels:
            self.vwap = panels["vwap"]
        else:
            self.vwap = (panels["amount"] / panels["volume"]).replace(
                [np.inf, -np.inf], np.nan)
        # 复权 close 的日收益（fill_method=None，不虚构缺口收益）
        self.returns = self.close.pct_change(fill_method=None)
        self.industry = industry
        self._adv_cache: dict[int, pd.DataFrame] = {}

    def adv(self, n: int) -> pd.DataFrame:
        """过去 n 日平均成交额（dollar volume），缓存。"""
        n = int(n)
        if n not in self._adv_cache:
            self._adv_cache[n] = self.amount.rolling(n, min_periods=n).mean()
        return self._adv_cache[n]

    def ind(self, x: pd.DataFrame) -> pd.DataFrame:
        """行业中性化的便捷入口。"""
        return ind_neutralize(x, self.industry)


# ---------------------------------------------------------------------------
# 构建入口（build_alpha_factors / extend_factor_library 共用）
# ---------------------------------------------------------------------------
SET_LABELS = {
    "alpha101": "WorldQuant Alpha101 (Kakushadze 2016)",
    "alpha191": "GTJA Alpha191 短周期价量因子 (2017)",
    "alpha158": "Qlib Alpha158 量价特征集 (Microsoft, 2020)",
    "alpha360": "Qlib Alpha360 原始OHLCV序列 (Microsoft, 2020)",
}


def load_alpha_panels(cache, uni, index_code: str, warmup: int, end: int):
    """拉 warmup..end 的 PIT 日线并组装 AlphaData 输入面板（后复权）。

    Returns:
        (panels, industry, close_adj)：panels 含 open/high/low/close/volume/
        amount/vwap（后复权口径）；industry 为申万一级行业 PIT 面板（缺省 None）；
        close_adj 为复权 close（收益口径用）。
    """
    from data.cache_helpers import load_backward_factor, load_daily
    from data.industry import IndustryClassification

    codes, cal, daily = load_daily(cache, uni, index_code, warmup, end)
    bf = load_backward_factor(cache, codes)
    log.info("日线 %d 行 / 复权因子 %d 列", len(daily),
             bf.shape[1] if not bf.empty else 0)

    d = daily.reset_index()
    d["date"] = d["date"].dt.normalize()

    def _panel(col: str) -> pd.DataFrame:
        return d.pivot(index="date", columns="code", values=col).sort_index()

    o, h, l, c = _panel("open"), _panel("high"), _panel("low"), _panel("close")
    v, amt = _panel("volume"), _panel("amount")

    if not bf.empty:
        f = bf.reindex(index=c.index, columns=c.columns).ffill()
        for pnl in (o, h, l, c):
            pnl[:] = pnl.values * f.values
        # vwap = 均价 × 后复权因子（与价格同口径）
        vwap = (amt / v).replace([np.inf, -np.inf], np.nan) * f
    else:
        log.warning("无复权因子，使用原始价（除权日价量关系会有跳变）")
        vwap = (amt / v).replace([np.inf, -np.inf], np.nan)

    # 申万一级行业 PIT 面板（IndNeutralize 用；失败则恒等中性化）
    industry = None
    try:
        industry = IndustryClassification(cache, level=1).get_industry_panel(
            list(c.columns), c.index)
        if industry.isna().all().all():
            industry = None
    except Exception as e:  # noqa: BLE001
        log.warning("行业面板不可用（%s），IndNeutralize 退化为恒等", str(e)[:80])

    panels = {"open": o, "high": h, "low": l, "close": c,
              "volume": v, "amount": amt, "vwap": vwap}
    return panels, industry, c
