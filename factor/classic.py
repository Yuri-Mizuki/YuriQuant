"""
经典日频因子类（classic factors）
=================================

常用日频因子的纯 pandas 实现，不依赖 SDK，离线可用。
每个因子继承 Factor，实现 calc 方法。

**命名区分（2026-08-17 收敛命名混淆）**：
本模块曾名为 ``factor/library.py``，易与持久化因子库 ``research/factor_library.py``
混淆。二者职责完全不同：
- 本模块 ``factor/classic.py``：**因子算法类**（Momentum/Reversal/... 的 calc 实现）。
- ``research/factor_library.py``：**持久化因子库**（FactorLibrary：注册/评估/入库/回测）。
本模块只被 ``factor/__init__.py`` 引用，经 ``from factor import Momentum`` 等使用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor.base import Factor
from stats import PERIODS_PER_YEAR


# ===========================================================================
# 动量类
# ===========================================================================
class Momentum(Factor):
    """N日收益率动量: close.shift(N) / close - 1 的过去 N 日收益。"""

    def __init__(self, n: int = 20):
        self.n = n
        self.name = f"momentum_{n}"

    def calc(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        return close.pct_change(self.n)


class Reversal(Factor):
    """N日反转: 过去 N 日收益取负（反转因子）。"""

    def __init__(self, n: int = 5):
        self.n = n
        self.name = f"reversal_{n}"

    def calc(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        return -close.pct_change(self.n)


# ===========================================================================
# 波动类
# ===========================================================================
class Volatility(Factor):
    """N日收益率标准差（年化）。"""

    def __init__(self, n: int = 20):
        self.n = n
        self.name = f"volatility_{n}"

    def calc(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        rets = close.pct_change()
        return rets.rolling(self.n).std() * np.sqrt(PERIODS_PER_YEAR)


class Amplitude(Factor):
    """N日振幅均值: (high-low)/close 的滚动均值。"""

    def __init__(self, n: int = 20):
        self.n = n
        self.name = f"amplitude_{n}"

    def calc(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        high = panel["high"]
        low = panel["low"]
        amp = (high - low) / close
        return amp.rolling(self.n).mean()


# ===========================================================================
# 流动性 / 量价类
# ===========================================================================
class Turnover(Factor):
    """N日平均成交额 / N日平均总市值（代理换手率）。

    无流通股本数据时用 amount 代理，做截面比较仍有区分度。
    """

    def __init__(self, n: int = 20):
        self.n = n
        self.name = f"turnover_{n}"

    def calc(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        amount = panel["amount"]
        return amount.rolling(self.n).mean()


class VolumeRatio(Factor):
    """量比: 当日成交量 / N日平均成交量。"""

    def __init__(self, n: int = 20):
        self.n = n
        self.name = f"vol_ratio_{n}"

    def calc(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = panel["volume"]
        return volume / volume.rolling(self.n).mean()


# ===========================================================================
# 价值类（需复权价格 + 成交额）
# ===========================================================================
class PriceMA(Factor):
    """价格偏离均线: close / MA(N) - 1。"""

    def __init__(self, n: int = 60):
        self.n = n
        self.name = f"price_ma_{n}"

    def calc(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        ma = close.rolling(self.n).mean()
        return close / ma - 1


# ===========================================================================
# 因子注册表
# ===========================================================================
ALL_FACTORS = {
    "momentum_20": lambda: Momentum(20),
    "reversal_5": lambda: Reversal(5),
    "volatility_20": lambda: Volatility(20),
    "amplitude_20": lambda: Amplitude(20),
    "turnover_20": lambda: Turnover(20),
    "vol_ratio_20": lambda: VolumeRatio(20),
    "price_ma_60": lambda: PriceMA(60),
}
