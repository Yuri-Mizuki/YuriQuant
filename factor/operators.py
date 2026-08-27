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


def inv(x: pd.DataFrame) -> pd.DataFrame:
    """取倒数 (华泰 gplearn 函数集 inv / WorldQuant 1/X)，0 与 ±inf → NaN。"""
    return _clean_inf(1.0 / x)


def sigmoid(x: pd.DataFrame) -> pd.DataFrame:
    """Logistic 变换 (华泰报告23新增)，映射到 (0,1)，捕捉非线性。"""
    with np.errstate(all="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


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


def greater(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """逐元素比较 a>b，返回 1/0（研报图表6 二元算子；NaN 参与比较得 0）。"""
    return (a > b).astype(float)


def less(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """逐元素比较 a<b，返回 1/0（研报图表6 二元算子；NaN 参与比较得 0）。"""
    return (a < b).astype(float)


def signed_power(x: pd.DataFrame, p: float) -> pd.DataFrame:
    """保号幂：sign(x)·|x|^p（研报 signed_power2/3 的通用形式）。"""
    return np.sign(x) * x.abs().pow(p)


def signed_power2(x: pd.DataFrame) -> pd.DataFrame:
    """保号平方（研报图表6 一元算子）。"""
    return signed_power(x, 2.0)


def signed_power3(x: pd.DataFrame) -> pd.DataFrame:
    """保号立方（研报图表6 一元算子）。"""
    return signed_power(x, 3.0)


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
    """N 周期相关系数 (AmazingData RELATE / COVAR 配套)。

    任一侧窗口内方差为 0（如截面 rank 序列短期恒定）时相关无定义；
    pandas 数值实现会吐 ±inf，这里统一清洗为 NaN，防止下游 rank/
    ts_rank 把 inf 当作最大值产生伪影。
    """
    return x.rolling(n, min_periods=n).corr(y).replace([np.inf, -np.inf], np.nan)


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期协方差 (AmazingData COVAR)。"""
    return x.rolling(n, min_periods=n).cov(y)


def ts_zscore(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期 z-score（华泰报告23新增 ts_zscore）：(X - ts_mean) / ts_std。"""
    mean = x.rolling(n, min_periods=n).mean()
    std = x.rolling(n, min_periods=n).std(ddof=1)
    return _safe_div(x - mean, std)


def ts_slope(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期线性回归斜率 (AmazingData SLOPE)。用 t=0..n-1 作自变量。"""
    t = np.arange(n, dtype=float)
    sum_t = t.sum()
    sum_t2 = (t * t).sum()
    sum_x = x.rolling(n, min_periods=n).sum()
    sum_tx = x.rolling(n, min_periods=n).apply(lambda v: np.dot(v, t), raw=True)
    denom = n * sum_t2 - sum_t * sum_t
    return _safe_div(n * sum_tx - sum_t * sum_x, denom)


def _ts_reg_stats(x: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """滚动回归 x~t（t=0..n-1）的 (斜率, 截距, R²)，向量化累积统计。"""
    t = np.arange(n, dtype=float)
    sum_t, sum_t2 = t.sum(), (t * t).sum()
    denom = n * sum_t2 - sum_t * sum_t
    sum_x = x.rolling(n, min_periods=n).sum()
    sum_tx = x.rolling(n, min_periods=n).apply(lambda v: np.dot(v, t), raw=True)
    sum_x2 = x.rolling(n, min_periods=n).apply(lambda v: np.dot(v, v), raw=True)
    slope = _safe_div(n * sum_tx - sum_t * sum_x, denom)
    intercept = _safe_div(sum_x - slope * sum_t, n)
    sst = sum_x2 - _safe_div(sum_x * sum_x, n)
    ssr = slope * (n * sum_tx - sum_t * sum_x)
    r2 = _safe_div(ssr, sst)
    return slope, intercept, r2


def ts_rsquare(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期线性回归 R²（研报图表6 一元时序算子，去趋势的确定性度量）。"""
    return _ts_reg_stats(x, n)[2]


def ts_residual(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期线性回归残差（最新点：x_t − ŷ_t，研报图表6 一元时序算子，去趋势）。"""
    slope, intercept, _ = _ts_reg_stats(x, n)
    return x - (intercept + slope * (n - 1))


def ts_pct_change(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 周期百分比变化：x/ref(x,n) − 1（研报图表6 一元时序算子）。"""
    return _safe_div(x, x.shift(n)) - 1.0


def ts_beta(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    """滚动回归 y~x 的斜率 β（研报图表6 二元时序算子；denom=0 → NaN）。"""
    sum_x = x.rolling(n, min_periods=n).sum()
    sum_y = y.rolling(n, min_periods=n).sum()
    sum_xy = (x * y).rolling(n, min_periods=n).sum()
    sum_x2 = (x * x).rolling(n, min_periods=n).sum()
    denom = n * sum_x2 - sum_x * sum_x
    return _safe_div(n * sum_xy - sum_x * sum_y, denom)


def ts_orth(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    """y 对 x 滚动回归残差（最新点）：y − (a + βx)，即正交化去 x（研报图表6）。"""
    beta = ts_beta(x, y, n)
    sum_x = x.rolling(n, min_periods=n).sum()
    sum_y = y.rolling(n, min_periods=n).sum()
    intercept = _safe_div(sum_y - beta * sum_x, n)
    return y - (intercept + beta * x)


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
    """截面 Z-score 标准化 (AmazingData CSZSCORE)。

    2026-08-17 收敛：复用 ``preprocessing.standardize_zscore``（同一逐日
    z-score 实现），消除 `operators` / `preprocessing` 两处复制。结果等价：
    正常行完全相同，std=0 的行两者均产生 NaN（``_clean_inf`` 兜底 ±inf）。
    """
    from factor.preprocessing import standardize_zscore as _z
    return _clean_inf(_z(x))


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


def cs_scale_abs(x: pd.DataFrame) -> pd.DataFrame:
    """截面尺度化 scale(X, a=1)（华泰 gplearn 函数集）：a*X / Σ|X|，逐截面。"""
    s = x.abs().sum(axis=1)
    return _clean_inf(x.div(s, axis=0))


def cs_winsorize(x: pd.DataFrame) -> pd.DataFrame:
    """截面 MAD 去极值（逐交易日独立，中位数 ±5×MAD，与华泰口径一致）。

    2026-08-17 收敛：复用 ``preprocessing.winsorize_mad``（其
    ``consistency_scale=False`` 即"不乘 1.4826"的华泰口径，等价于本函数），
    消除 `operators` / `preprocessing` 两处复制。
    """
    from factor.preprocessing import winsorize_mad as _w
    return _w(x, n_mad=5.0, consistency_scale=False)


def cs_truncate(x: pd.DataFrame) -> pd.DataFrame:
    """截面分位数截断（逐交易日独立，1%~99%）。"""
    lo = x.quantile(0.01, axis=1)
    hi = x.quantile(0.99, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


def rank_sub(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """排名差（华泰报告23 rank_sub）：cs_rank(a) - cs_rank(b)。"""
    return cs_rank(a) - cs_rank(b)


def rank_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """排名比（华泰报告23 rank_div）：cs_rank(a) / cs_rank(b)。
    cs_rank 用 pct=True，最小值为 1/n > 0，分母安全。"""
    return cs_rank(a) / cs_rank(b)


# ===========================================================================
# 技术指标算子（面板级，纯 pandas/numpy，2026-08-12 新增）
# 参考华泰报告26 CTA 函数集 + A 股实战代表：KAMA/AROONOSC/HT_DCPHASE/
# BOLL/OBV/RSI/ADX。全部无 SDK 依赖，date×code 向量化。
# ===========================================================================
def kama(x: pd.DataFrame, n: int, fast: int = 2, slow: int = 30) -> pd.DataFrame:
    """考夫曼自适应均线 (TA-Lib KAMA)。

    效率比 ER = |Δn| / Σ|Δ1|；平滑系数 SC = [ER·(2/(fast+1)−2/(slow+1)) + 2/(slow+1)]²；
    KAMA[t] = KAMA[t−1] + SC[t]·(X[t] − KAMA[t−1])（递归）。
    趋势强（ER 高）跟随快、震荡（ER 低）平滑慢 —— 窗口随市场自适应。
    """
    x = x.astype(float)
    er_num = (x - x.shift(n)).abs()
    er_den = x.diff().abs().rolling(n, min_periods=n).sum()
    er = _safe_div(er_num, er_den)
    sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
    vals, scv = x.values, sc.values
    out = np.full_like(vals, np.nan, dtype=float)
    for j in range(vals.shape[1]):
        prev = np.nan
        for i in range(vals.shape[0]):
            s = scv[i, j]
            if not np.isfinite(s):
                prev = np.nan
                continue
            xv = vals[i, j]
            if not np.isfinite(xv):
                prev = np.nan
                continue
            prev = xv if not np.isfinite(prev) else prev + s * (xv - prev)
            out[i, j] = prev
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def aroonosc(high: pd.DataFrame, low: pd.DataFrame, n: int) -> pd.DataFrame:
    """阿隆震荡 (TA-Lib AROONOSC)：新高新鲜度 − 新低新鲜度，[-100, 100]。

    aroon_up = 100·(n − 距 n 期最高价的天数)/n；aroon_down 对最低价。
    捕捉「新高/新低是多久前出现的」——时间维度的趋势新鲜度。
    """
    def _since_max(s):
        return float(n - 1 - int(np.nanargmax(s)))

    def _since_min(s):
        return float(n - 1 - int(np.nanargmin(s)))

    up = 100.0 * high.rolling(n, min_periods=n).apply(_since_max, raw=True) / n
    down = 100.0 * low.rolling(n, min_periods=n).apply(_since_min, raw=True) / n
    return up - down


def ht_dcphase(x: pd.DataFrame) -> pd.DataFrame:
    """希尔伯特变换瞬时相位（近似 TA-Lib HT_DCPHASE，无窗口）。

    用解析信号瞬时相位近似主导循环相位：phase = angle(hilbert(X))，[-180°, 180°]。
    表达价格处于周期的哪个位置（华泰报告26 用它在铁矿石/热轧卷板挖出信号）。
    依赖 scipy（项目已有）。
    """
    from scipy.signal import hilbert
    out = x.copy().astype(float)
    for col in x.columns:
        s = x[col].ffill().fillna(0.0).values
        out[col] = np.angle(hilbert(s), deg=True)
    return out


def boll_pctb(x: pd.DataFrame, n: int, k: float = 2.0) -> pd.DataFrame:
    """布林带 %B：价格在 [MA − k·σ, MA + k·σ] 带内的位置（TA-Lib 口径，k=2 默认）。

    **口径说明**：本函数为纯 pandas 向量化版（``rolling``）。AmazingData SDK 语义版在
    ``factor.technical_indicators.TechnicalIndicators.BOLL``（复刻平台 BOLL 指标）。
    两处口径不同（本处返回 %B 位置值，SDK 版返回 BOLL/UB/LB 三条带），非重复实现，勿合并。
    """
    mid = x.rolling(n, min_periods=n).mean()
    std = x.rolling(n, min_periods=n).std(ddof=0)
    return _safe_div(x - mid, k * std)


def obv(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """能量潮 (TA-Lib OBV)：Σ sign(Δclose)·volume —— 多空力量累积存量。

    **口径说明**：本函数为 TA-Lib 纯 pandas 版。AmazingData SDK 语义版在
    ``factor.technical_indicators.TechnicalIndicators.OBV``（平台 OBV + MAOBV）。
    两处接口/返回值不同（本处只返 OBV 一列，SDK 版返 OBV+MAOBV），非重复实现，勿合并。
    """
    flow = np.sign(close.diff()) * volume
    return flow.fillna(0.0).cumsum()


def rsi(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """相对强弱指标 (TA-Lib RSI，Wilder 平滑)：100 − 100/(1 + RS)，RS = 均涨/均跌。

    **口径说明**：本函数为 TA-Lib Wilder（指数平滑）口径，返单列。
    AmazingData SDK 语义版在 ``factor.technical_indicators.TechnicalIndicators.RSI``
    （SMA 平滑，三窗口 n1/n2/n3，返多列 dict）。两处平滑方式不同，非重复实现，勿合并。
    """
    diff = x.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    alpha = 1.0 / n
    avg_gain = gain.ewm(alpha=alpha, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=n, adjust=False).mean()
    rs = _safe_div(avg_gain, avg_loss)
    out = 100.0 - 100.0 / (1.0 + rs)
    # 边界：横盘（无涨无跌）→ 50；有涨无跌 → RS=∞ → 100（div0 的 NaN 落这里）
    out = out.where((avg_loss > 0) | (avg_gain > 0), 50.0)
    out = out.fillna(100.0)
    return out


def adx(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int) -> pd.DataFrame:
    """平均趋向指数 (TA-Lib ADX)：趋势强度（与方向无关，0~100）。

    TR/DM → +DI/−DI → DX → Wilder 平滑 ADX。ADX 高 = 强趋势。
    """
    pc = close.shift()
    idx, cols = high.index, high.columns
    tr = pd.DataFrame(np.maximum.reduce([high - low, (high - pc).abs(), (low - pc).abs()]),
                      index=idx, columns=cols)
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    alpha = 1.0 / n
    atr = pd.DataFrame(tr, index=idx, columns=cols).ewm(alpha=alpha, min_periods=n, adjust=False).mean()
    pdi = 100.0 * pd.DataFrame(plus_dm, index=idx, columns=cols).ewm(alpha=alpha, min_periods=n, adjust=False).mean() / atr
    mdi = 100.0 * pd.DataFrame(minus_dm, index=idx, columns=cols).ewm(alpha=alpha, min_periods=n, adjust=False).mean() / atr
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=alpha, min_periods=n, adjust=False).mean()


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
    OpSpec("inv", inv, 1, kind="element"),          # 华泰 gplearn 函数集
    OpSpec("sigmoid", sigmoid, 1, kind="element"),  # 华泰报告23 新增
    OpSpec("signed_power2", signed_power2, 1, kind="element"),  # 研报图表6
    OpSpec("signed_power3", signed_power3, 1, kind="element"),  # 研报图表6
    OpSpec("add", add, 2, kind="element"),
    OpSpec("sub", sub, 2, kind="element"),
    OpSpec("mul", mul, 2, kind="element"),
    OpSpec("div", div, 2, kind="element"),
    OpSpec("max", max_, 2, kind="element"),
    OpSpec("min", min_, 2, kind="element"),
    OpSpec("greater", greater, 2, kind="element"),  # 研报图表6 二元逻辑
    OpSpec("less", less, 2, kind="element"),        # 研报图表6 二元逻辑
]

# 时序算子：arity=1 或 2，n_window=1
TS_OPS: list[OpSpec] = [
    OpSpec("ts_ref", ts_ref, 1, 1, "ts"),
    OpSpec("ts_delay", ts_delay, 1, 1, "ts"),  # 华泰 gplearn 函数集 delay(X, d)
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
    OpSpec("ts_zscore", ts_zscore, 1, 1, "ts"),  # 华泰报告23 新增
    OpSpec("ts_corr", ts_corr, 2, 1, "ts"),
    OpSpec("ts_cov", ts_cov, 2, 1, "ts"),
    OpSpec("ts_beta", ts_beta, 2, 1, "ts"),          # 研报图表6 二元时序
    OpSpec("ts_orth", ts_orth, 2, 1, "ts"),          # 研报图表6 二元时序
    OpSpec("ts_slope", ts_slope, 1, 1, "ts"),
    OpSpec("ts_pct_change", ts_pct_change, 1, 1, "ts"),  # 研报图表6
    OpSpec("ts_rsquare", ts_rsquare, 1, 1, "ts"),        # 研报图表6
    OpSpec("ts_residual", ts_residual, 1, 1, "ts"),      # 研报图表6
    OpSpec("ts_quantile", ts_quantile, 1, 1, "ts"),      # 研报图表6（原实现补注册）
    # 技术指标（2026-08-12，参考华泰报告26）：单目+窗口，可进 exhaustive 与 GP
    OpSpec("kama", kama, 1, 1, "ts"),        # 考夫曼自适应均线
    OpSpec("rsi", rsi, 1, 1, "ts"),          # 相对强弱
    OpSpec("boll_pctb", boll_pctb, 1, 1, "ts"),  # 布林带 %B
]

# 技术指标算子（GP 专用：arity≥2 或多输入，exhaustive 自动跳过 arity≠1）
TECH_OPS: list[OpSpec] = [
    OpSpec("aroonosc", aroonosc, 2, 1, "ts"),   # 阿隆震荡（high, low, n）
    OpSpec("adx", adx, 3, 1, "ts"),             # 平均趋向指数（high, low, close, n）
    OpSpec("ht_dcphase", ht_dcphase, 1, 0, "ts"),  # 希尔伯特相位（无窗口，不接受窗口参数）
    OpSpec("obv", obv, 2, 0, "ts"),             # 能量潮（close, volume，无窗口）
]

# 截面算子：arity=1 或 2，无窗口
CS_OPS: list[OpSpec] = [
    OpSpec("cs_rank", cs_rank, 1, kind="cs"),
    OpSpec("cs_zscore", cs_zscore, 1, kind="cs"),
    OpSpec("cs_demean", cs_demean, 1, kind="cs"),
    OpSpec("cs_normalize", cs_normalize, 1, kind="cs"),
    OpSpec("cs_scale", cs_scale, 1, kind="cs"),
    OpSpec("cs_rank_normalize", cs_rank_normalize, 1, kind="cs"),
    OpSpec("cs_winsorize", cs_winsorize, 1, kind="cs"),  # 研报图表6
    OpSpec("cs_truncate", cs_truncate, 1, kind="cs"),    # 研报图表6
    OpSpec("scale", cs_scale_abs, 1, kind="cs"),  # 华泰 gplearn 函数集 scale(X, a=1)
    OpSpec("rank_sub", rank_sub, 2, kind="cs"),   # 华泰报告23 新增
    OpSpec("rank_div", rank_div, 2, kind="cs"),   # 华泰报告23 新增
]


def all_operators() -> list[OpSpec]:
    """返回全部算子规格（元素 + 时序 + 截面 + 技术指标）。"""
    return ELEMENT_OPS + TS_OPS + CS_OPS + TECH_OPS


def op_registry() -> dict[str, OpSpec]:
    """name -> OpSpec 字典。"""
    return {op.name: op for op in all_operators()}


# 默认窗口候选（挖掘时枚举用）
DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)

# 默认原始特征字段（OHLCV；财务字段由 mining 模块按需追加）
DEFAULT_FEATURES: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")
