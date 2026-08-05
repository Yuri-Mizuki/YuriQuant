"""
算子空间
========

面板级因子算子，供「因子挖掘 / 遗传规划」复用。所有算子作用于
``date × code`` DataFrame（index=交易日期, columns=证券代码），返回同形状面板。

分三类（命名参照 AmazingData 手册算子 + WorldQuant alpha 约定）：

- 元素算子 (element-wise): abs / sign / log / add / sub / mul / div ...
- 时序算子 (ts_*): 沿时间轴(行)滚动，逐股票计算，窗口 N 为参数
- 截面算子 (cs_*): 沿截面(列)计算，逐交易日

设计要点
--------
- 形状保持：绝大多数算子 panel -> panel，便于在表达式树中任意嵌套。
- NaN 安全：滚动窗口 ``min_periods=window``（前若干行自然为 NaN）；
  除法 / 对数 / 开方对非法值返回 NaN（不抛异常）。
- 不含未来函数：所有时序算子只用历史窗口（REFX 类前视算子刻意不实现）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable


# ===========================================================================
# 内部工具
# ===========================================================================
def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """逐元素除法，分母为 0 返回 NaN（不产生 inf）。"""
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)


def _panel(x) -> pd.DataFrame:
    """确保返回 DataFrame（允许标量广播）。"""
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(x)


# ===========================================================================
# 元素算子 (element-wise, panel -> panel)
# ===========================================================================
def abs_(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()


def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)


def log_(x: pd.DataFrame) -> pd.DataFrame:
    """自然对数，非正返回 NaN。"""
    out = np.log(x.where(x > 0))
    return out


def log10(x: pd.DataFrame) -> pd.DataFrame:
    out = np.log10(x.where(x > 0))
    return out


def exp_(x: pd.DataFrame) -> pd.DataFrame:
    return _clean_inf(np.exp(x))


def sqrt_(x: pd.DataFrame) -> pd.DataFrame:
    return np.sqrt(x.where(x >= 0))


def power(x: pd.DataFrame, n: float) -> pd.DataFrame:
    return x.pow(n)


def reverse(x: pd.DataFrame) -> pd.DataFrame:
    """取相反数 (AmazingData REVERSE)。"""
    return -x


def add(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a + b


def sub(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a - b


def mul(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a * b


def div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return _safe_div(a, b)


def max_(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return np.maximum(a, b)


def min_(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return np.minimum(a, b)


# ===========================================================================
# 时序算子 (沿行滚动，逐股票；window N 为参数)
# ===========================================================================
def ts_ref(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """引用 N 周期前的值 (AmazingData REF / SHIFT)。"""
    return x.shift(n)


def ts_delay(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """同 ts_ref，语义别名。"""
    return x.shift(n)


def ts_delta(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """当期减 N 期前的值。"""
    return x - x.shift(n)


def ts_diff(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 期收益率: x / ts_ref(x, n) - 1。"""
    return _safe_div(x, x.shift(n)) - 1.0


def ts_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期简单移动平均 (AmazingData MA / MEAN)。"""
    return x.rolling(n, min_periods=n).mean()


def ts_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期求和 (AmazingData SUM)。"""
    return x.rolling(n, min_periods=n).sum()


def ts_std(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期样本标准差 (AmazingData STD)。"""
    return x.rolling(n, min_periods=n).std(ddof=1)


def ts_var(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).var(ddof=1)


def ts_max(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期最高值 (AmazingData HHV)。"""
    return x.rolling(n, min_periods=n).max()


def ts_min(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期最低值 (AmazingData LLV)。"""
    return x.rolling(n, min_periods=n).min()


def ts_arg_max(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期内最高值距当期的周期数 (AmazingData HHVBARS)。归一化到 [0,1]。"""
    raw = x.rolling(n, min_periods=n).apply(np.argmax, raw=True)
    return _safe_div(raw.astype(float), float(n - 1))


def ts_arg_min(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期内最低值距当期的周期数 (AmazingData LLVBARS)。归一化到 [0,1]。"""
    raw = x.rolling(n, min_periods=n).apply(np.argmin, raw=True)
    return _safe_div(raw.astype(float), float(n - 1))


def ts_rank(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """当期值在过去 N 期的分位排名 (pct, [0,1])。"""
    return x.rolling(n, min_periods=n).rank(pct=True)


def ts_median(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).median()


def ts_avedev(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期平均绝对偏差 (AmazingData AVEDEV)。"""
    return x.rolling(n, min_periods=n).apply(lambda v: np.abs(v - v.mean()).mean(), raw=True)


def ts_quantile(x: pd.DataFrame, n: int, q: float = 0.75) -> pd.DataFrame:
    """过去 N 期的 q 分位数 (AmazingData QUANTILE)。"""
    return x.rolling(n, min_periods=n).quantile(q)


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期相关系数 (AmazingData RELATE / COVAR 配套)。"""
    return x.rolling(n, min_periods=n).corr(y)


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期协方差 (AmazingData COVAR)。"""
    return x.rolling(n, min_periods=n).cov(y)


def ts_slope(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期线性回归斜率 (AmazingData SLOPE)。用 t=0..n-1 作自变量。"""
    t = np.arange(n, dtype=float)
    sum_t = t.sum()
    sum_t2 = (t * t).sum()
    sum_x = x.rolling(n, min_periods=n).sum()
    sum_tx = x.rolling(n, min_periods=n).apply(lambda v: np.dot(v, t), raw=True)
    denom = n * sum_t2 - sum_t * sum_t
    return _safe_div(n * sum_tx - sum_t * sum_x, denom)


def ts_skew(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).skew()


def ts_kurt(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).kurt()


def ts_ema(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期指数移动平均 (AmazingData EMA): Y = (X*2 + Y'*(N-1)) / (N+1)。"""
    alpha = 2.0 / (n + 1.0)
    return x.ewm(alpha=alpha, adjust=False, min_periods=n).mean()


def ts_wma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期加权移动平均 (AmazingData WMA): 近期权重更大。"""
    weights = np.arange(1, n + 1, dtype=float)

    def _w(v):
        return np.dot(v, weights) / weights.sum()

    return x.rolling(n, min_periods=n).apply(_w, raw=True)


def ts_decay_linear(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """线性衰减加权均值（等价于 WMA，alpha 挖掘常用命名）。"""
    return ts_wma(x, n)


def ts_count(x_cond: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期内满足条件(非0)的周期数 (AmazingData COUNT)。
    输入应为 0/1 布尔面板。"""
    return x_cond.rolling(n, min_periods=n).sum()


def ts_product(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期累乘 (AmazingData MULAR)。"""
    return x.rolling(n, min_periods=n).apply(np.prod, raw=True)


# ===========================================================================
# 截面算子 (沿列，逐交易日；panel -> panel)
# ===========================================================================
def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """截面百分位排名 [0,1] (AmazingData CSPCTRANK / CSRANK)。"""
    return x.rank(axis=1, pct=True)


def _clean_inf(df: pd.DataFrame) -> pd.DataFrame:
    """把 ±inf 替换成 NaN（截面除法分母可能为 0）。"""
    return df.replace([np.inf, -np.inf], np.nan)


def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    """截面 Z-score 标准化 (AmazingData CSZSCORE)。"""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    # 必须用 div(..., axis=0)：Series 按日期索引，默认会错对齐到列。
    return _clean_inf(x.sub(mean, axis=0).div(std, axis=0))


def cs_demean(x: pd.DataFrame) -> pd.DataFrame:
    """截面去均值 (AmazingData CSDEMEAN)。"""
    return x.sub(x.mean(axis=1), axis=0)


def cs_normalize(x: pd.DataFrame) -> pd.DataFrame:
    """截面 Min-Max 归一化到 [0,1] (AmazingData CSNORMALIZE)。"""
    lo = x.min(axis=1)
    hi = x.max(axis=1)
    return _clean_inf(x.sub(lo, axis=0).div((hi - lo), axis=0))


def cs_scale(x: pd.DataFrame) -> pd.DataFrame:
    """截面标准化为 L2 单位范数（截面平方和=1），常用于因子合成。"""
    norm = np.sqrt((x * x).sum(axis=1))
    return _clean_inf(x.div(norm, axis=0))


def cs_rank_normalize(x: pd.DataFrame) -> pd.DataFrame:
    """截面排名后 z-score：先 cs_rank 再 cs_zscore，挖掘中鲁棒性最好。"""
    return cs_zscore(cs_rank(x))


# ===========================================================================
# 算子注册表 —— 供候选生成 / 遗传规划枚举原语
# ===========================================================================
@dataclass
class OpSpec:
    """算子规格。

    Attributes:
        name: 算子名（表达式中的符号）。
        func: 实现函数。
        arity: 面板参数个数（1 或 2）。
        n_window: 整数窗口参数个数（0 或 1，如 ts_mean 需 1 个）。
        kind: element | ts | cs。
    """
    name: str
    func: Callable
    arity: int
    n_window: int = 0
    kind: str = "element"


# 元素算子：arity in {1,2}, 无窗口
ELEMENT_OPS: list[OpSpec] = [
    OpSpec("abs", abs_, 1, kind="element"),
    OpSpec("sign", sign, 1, kind="element"),
    OpSpec("log", log_, 1, kind="element"),
    OpSpec("log10", log10, 1, kind="element"),
    OpSpec("exp", exp_, 1, kind="element"),
    OpSpec("sqrt", sqrt_, 1, kind="element"),
    OpSpec("reverse", reverse, 1, kind="element"),
    OpSpec("add", add, 2, kind="element"),
    OpSpec("sub", sub, 2, kind="element"),
    OpSpec("mul", mul, 2, kind="element"),
    OpSpec("div", div, 2, kind="element"),
    OpSpec("max", max_, 2, kind="element"),
    OpSpec("min", min_, 2, kind="element"),
]

# 时序算子：arity=1 或 2，n_window=1
TS_OPS: list[OpSpec] = [
    OpSpec("ts_ref", ts_ref, 1, 1, "ts"),
    OpSpec("ts_delta", ts_delta, 1, 1, "ts"),
    OpSpec("ts_diff", ts_diff, 1, 1, "ts"),
    OpSpec("ts_mean", ts_mean, 1, 1, "ts"),
    OpSpec("ts_sum", ts_sum, 1, 1, "ts"),
    OpSpec("ts_std", ts_std, 1, 1, "ts"),
    OpSpec("ts_var", ts_var, 1, 1, "ts"),
    OpSpec("ts_max", ts_max, 1, 1, "ts"),
    OpSpec("ts_min", ts_min, 1, 1, "ts"),
    OpSpec("ts_arg_max", ts_arg_max, 1, 1, "ts"),
    OpSpec("ts_arg_min", ts_arg_min, 1, 1, "ts"),
    OpSpec("ts_rank", ts_rank, 1, 1, "ts"),
    OpSpec("ts_median", ts_median, 1, 1, "ts"),
    OpSpec("ts_avedev", ts_avedev, 1, 1, "ts"),
    OpSpec("ts_skew", ts_skew, 1, 1, "ts"),
    OpSpec("ts_kurt", ts_kurt, 1, 1, "ts"),
    OpSpec("ts_ema", ts_ema, 1, 1, "ts"),
    OpSpec("ts_wma", ts_wma, 1, 1, "ts"),
    OpSpec("ts_decay_linear", ts_decay_linear, 1, 1, "ts"),
    OpSpec("ts_count", ts_count, 1, 1, "ts"),
    OpSpec("ts_product", ts_product, 1, 1, "ts"),
    OpSpec("ts_corr", ts_corr, 2, 1, "ts"),
    OpSpec("ts_cov", ts_cov, 2, 1, "ts"),
    OpSpec("ts_slope", ts_slope, 1, 1, "ts"),
]

# 截面算子：arity=1 或 2，无窗口
CS_OPS: list[OpSpec] = [
    OpSpec("cs_rank", cs_rank, 1, kind="cs"),
    OpSpec("cs_zscore", cs_zscore, 1, kind="cs"),
    OpSpec("cs_demean", cs_demean, 1, kind="cs"),
    OpSpec("cs_normalize", cs_normalize, 1, kind="cs"),
    OpSpec("cs_scale", cs_scale, 1, kind="cs"),
    OpSpec("cs_rank_normalize", cs_rank_normalize, 1, kind="cs"),
]


def all_operators() -> list[OpSpec]:
    """返回全部算子规格（元素 + 时序 + 截面）。"""
    return ELEMENT_OPS + TS_OPS + CS_OPS


def op_registry() -> dict[str, OpSpec]:
    """name -> OpSpec 字典。"""
    return {op.name: op for op in all_operators()}


# 默认窗口候选（挖掘时枚举用）
DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)

# 默认原始特征字段（OHLCV；财务字段由 mining 模块按需追加）
DEFAULT_FEATURES: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")
