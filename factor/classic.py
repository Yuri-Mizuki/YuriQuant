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


# ===========================================================================
# 经典量价特征集（2026-08-31 从 scripts/e2e_common 下沉，e2e/实验脚本单一实现）
# ===========================================================================
def compute_classic_features(px: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """经典量价因子（动量/反转/波动/流动性/换手结构），截面标准化。

    Args:
        px: {open/high/low/close/volume/amount: date×code 面板}。
    Returns:
        {name: 已截面标准化面板}，共 12 个经典特征。
    """
    from factor.preprocessing import standardize_zscore

    close, open_ = px["close"], px["open"]
    amount = px["amount"]
    ret1 = close.pct_change(fill_method=None)
    feats = {
        "mom5": close.pct_change(5, fill_method=None),
        "mom10": close.pct_change(10, fill_method=None),
        "mom20": close.pct_change(20, fill_method=None),
        "mom60": close.pct_change(60, fill_method=None),
        "rev1": -ret1,
        "rev5": -close.pct_change(5, fill_method=None),
        "vol20": ret1.rolling(20).std(),
        "vol60": ret1.rolling(60).std(),
        "amihud20": (ret1.abs() / (amount + 1e-12)).rolling(20).mean(),
        "turn_trend": (px["volume"].rolling(5).mean()
                       / (px["volume"].rolling(60).mean() + 1e-12)),
        "gap": open_ / close.shift(1) - 1,
        "range20": (px["high"] - px["low"]).rolling(20).mean() / (close + 1e-12),
    }
    return {k: standardize_zscore(v) for k, v in feats.items()}


# ===========================================================================
# 银河 0608 特征族（时序截面三层预测报告附录表 12，2026-09-20 接入）
# ---------------------------------------------------------------------------
# 依据：reports/docs/research_notes/银河0608_研读_时序截面三层预测与QP口径.md。
# 三组特征 + 分组差异化预处理（研报 §2.4）：
# - G1 形态类：保留原始尺度与方向（不标准化）；
# - G1 动量/技术/相对强弱类：50 期滚动窗口内时序 Z-score（研报口径）；
# - G2 风险 / G3 量价：50 期滚动窗口内时序 Z-score（"相对自身历史的偏离"）。
# 与 compute_classic_features 的截面 Z-score 不同：银河族是**时序**标准化
# （每只股票相对自身历史），这是研报"分组差异化预处理"的核心。
# 频率口径：日线 mult=1（研报小时线 mult=4，公式同源）。
# ===========================================================================
def _rolling_zscore(panel: pd.DataFrame, win: int = 50) -> pd.DataFrame:
    """时序滚动 Z-score：每只股票相对自身过去 win 期的均值/标准差。"""
    mu = panel.rolling(win, min_periods=win // 2).mean()
    sd = panel.rolling(win, min_periods=win // 2).std()
    return (panel - mu) / (sd + 1e-8)


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Wilders RSI。"""
    diff = close.diff()
    up = diff.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-diff).clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False).mean()
    rs = up / (dn + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def _kdj_j(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
           n: int = 9) -> pd.DataFrame:
    """KDJ 的 J 值（9,3,3 参数）。"""
    llv = low.rolling(n, min_periods=1).min()
    hhv = high.rolling(n, min_periods=1).max()
    rsv = (close - llv) / (hhv - llv + 1e-12) * 100.0
    k = rsv.ewm(com=2.0, adjust=False).mean()
    d = k.ewm(com=2.0, adjust=False).mean()
    return 3.0 * k - 2.0 * d


def compute_galaxy_features(px: dict[str, pd.DataFrame],
                            index_close: pd.Series | pd.DataFrame,
                            win: int = 50) -> dict[str, pd.DataFrame]:
    """银河 0608 特征族（约 30 个，三组，分组差异化预处理）。

    Args:
        px: {open/high/low/close/volume/amount: date×code 面板}。
        index_close: 基准指数收盘（Series 或单列 DataFrame；excess/beta/idvol
            的基准——项目无指数权重面板时可用成分等权组合收盘作代理，
            调用方须在产出中注明该替代口径）。
        win: 滚动标准化窗口（研报 50 期）。

    Returns:
        {name: date×code 面板}；name 前缀 g1_/g2_/g3_ 标记特征组。
        均为**时序**口径（未经截面标准化）——下游如需截面可比自行 zscore。
    """
    close, open_, high, low = px["close"], px["open"], px["high"], px["low"]
    volume, amount = px["volume"], px["amount"]
    ret1 = close.pct_change(fill_method=None)
    idx = index_close if isinstance(index_close, pd.Series) \
        else index_close.iloc[:, 0]
    idx_ret = idx.pct_change(fill_method=None).reindex(close.index)

    # ---- G1 形态（原始尺度，不标准化）----
    g1_shape = {
        "g1_body": (close - open_) / close,
        "g1_range": (high - low) / close,
        "g1_upper_shadow": (high - np.maximum(open_, close)) / close,
        "g1_lower_shadow": (np.minimum(open_, close) - low) / close,
        "g1_body_ratio": (close - open_) / (high - low + 1e-12),
    }
    # ---- G1 收益/动量（50 期滚动 zscore）----
    g1_mom = {
        "g1_return": _rolling_zscore(ret1, win),
        "g1_mom_3": _rolling_zscore(ret1.rolling(3).sum(), win),
        "g1_mom_6": _rolling_zscore(ret1.rolling(6).sum(), win),
        "g1_mom_12": _rolling_zscore(ret1.rolling(12).sum(), win),
        "g1_mom_24": _rolling_zscore(ret1.rolling(24).sum(), win),
        "g1_mom_48": _rolling_zscore(ret1.rolling(48).sum(), win),
        "g1_mom_diff_6_24": _rolling_zscore(
            ret1.rolling(6).sum() - ret1.rolling(24).sum(), win),
        "g1_ma_gap_5": _rolling_zscore(np.log(close / (close.rolling(5).mean() + 1e-12)), win),
        "g1_ma_gap_20": _rolling_zscore(np.log(close / (close.rolling(20).mean() + 1e-12)), win),
        "g1_rsi_14": _rolling_zscore(_rsi(close, 14), win),
        "g1_kdj_j": _rolling_zscore(_kdj_j(high, low, close, 9), win),
    }
    # ---- G1 相对强弱（基准=传入指数口径）----
    excess_ret1 = ret1.sub(idx_ret, axis=0)
    g1_rel = {
        "g1_excess_ret_1": _rolling_zscore(excess_ret1, win),
        "g1_excess_mom_6": _rolling_zscore(excess_ret1.rolling(6).sum(), win),
        "g1_excess_mom_24": _rolling_zscore(excess_ret1.rolling(24).sum(), win),
    }
    betas = {}
    for n in (24, 80):
        cov = ret1.rolling(n).cov(idx_ret)
        var = idx_ret.rolling(n).var()
        betas[f"g1_beta_{n}"] = cov.div(var + 1e-12, axis=0)
    g1_rel.update(betas)

    # ---- G2 风险波动/路径（50 期滚动 zscore）----
    down_ret = ret1.clip(upper=0.0)          # 研报 min(return,0)：正收益记 0 非 NaN
    g2 = {
        "g2_ret_std_6": _rolling_zscore(ret1.rolling(6).std(), win),
        "g2_ret_std_24": _rolling_zscore(ret1.rolling(24).std(), win),
                "g2_downvol_24": _rolling_zscore(down_ret.rolling(24).std(), win),
        "g2_mdd_24": _rolling_zscore(
            close / close.rolling(24).max() - 1.0, win),
        "g2_ret_min_24": _rolling_zscore(ret1.rolling(24).min(), win),
        "g2_ret_max_24": _rolling_zscore(ret1.rolling(24).max(), win),
    }
    for n in (24, 80):
        beta_n = ret1.rolling(n).cov(idx_ret).div(
            idx_ret.rolling(n).var() + 1e-12, axis=0)
        resid = ret1.sub(beta_n.mul(idx_ret, axis=0))
        g2[f"g2_idvol_{n}"] = _rolling_zscore(resid.rolling(n).std(), win)

    # ---- G3 量价资金（50 期滚动 zscore）----
    g3 = {
        "g3_log_vol": _rolling_zscore(np.log(volume + 1.0), win),
        "g3_log_amount": _rolling_zscore(np.log(amount + 1.0), win),
        "g3_rv_6": _rolling_zscore(
            volume / (volume.rolling(6).mean() + 1e-12), win),
        "g3_rv_24": _rolling_zscore(
            volume / (volume.rolling(24).mean() + 1e-12), win),
        "g3_vpr_6": _rolling_zscore(ret1.rolling(6).corr(volume.pct_change(
            fill_method=None)), win),
        "g3_vpr_24": _rolling_zscore(ret1.rolling(24).corr(volume.pct_change(
            fill_method=None)), win),
        "g3_flow_24": _rolling_zscore(
            (amount * np.sign(ret1)).rolling(24).sum(), win),
    }

    out: dict[str, pd.DataFrame] = {}
    out.update(g1_shape)
    out.update(g1_mom)
    out.update(g1_rel)
    out.update(g2)
    out.update(g3)
    return out


def build_galaxy_labels(px: dict[str, pd.DataFrame],
                        index_close: pd.Series | pd.DataFrame,
                        *,
                        ret_win: int = 22,
                        mdd_win: int = 66) -> dict[str, pd.DataFrame]:
    """银河 0608 三标签（附录表 2）：alpha/sharpe（22 日）+ max_drawdown（66 日）。

    - ``label_alpha_{ret_win}``：窗口内相对基准的累计超额收益（截面 zscore）；
    - ``label_sharpe_{ret_win}``：窗口内**超额收益**的均值/标准差（截面 zscore）；
    - ``label_mdd_{mdd_win}``：窗口内价格最大回撤（**负值，越接近 0 越好**；
      截面 zscore）——与研报"取负绝对值使方向统一为越大越好"一致。

    全部为**前视面板**（date 日的值含 date 之后的信息），仅用于训练标签；
    防未来函数由调用方保证"训练段不越界"。
    """
    from factor.preprocessing import standardize_zscore

    close = px["close"]
    idx = index_close if isinstance(index_close, pd.Series) \
        else index_close.iloc[:, 0]
    idx_ret = idx.pct_change(fill_method=None).reindex(close.index)
    ret1 = close.pct_change(fill_method=None)
    ex = ret1.sub(idx_ret, axis=0)

    fwd_ex = ex.shift(-1).rolling(ret_win).sum().shift(-(ret_win - 1))
    fwd_ex_std = ex.shift(-1).rolling(ret_win).std().shift(-(ret_win - 1))
    fwd_mdd = (close.shift(-1)
               .rolling(mdd_win).max().shift(-(mdd_win - 1)))
    # 窗口内最低点相对窗口首日（=当前日）前收的回撤：min(close_future)/close - 1
    fut_min = close.shift(-1).rolling(mdd_win).min().shift(-(mdd_win - 1))
    fwd_mdd = fut_min / close - 1.0

    return {
        f"label_alpha_{ret_win}": standardize_zscore(fwd_ex),
        f"label_sharpe_{ret_win}": standardize_zscore(
            fwd_ex / (fwd_ex_std + 1e-12)),
        f"label_mdd_{mdd_win}": standardize_zscore(fwd_mdd),
    }
