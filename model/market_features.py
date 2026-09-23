"""市场状态特征 —— 指数点位派生（国金19 转译，2026-09-23）。

转译依据（国金19 研读笔记 §1.1 / §四补短板清单第 1 条）：
    「把三大宽基指数收益率/点位（或项目对应基准指数）作为市场状态特征
    注入 GBDT 面板——半天级改造，对'普涨/普跌 vs 结构分化'的行情区分
    有直接依据」。研报消融中**只有指数点位带来稳定增益**（时间/风格信息
    无稳定增益），故只做指数一族。

设计要点（与项目既有教训一一对应）：

1. **日级广播特征**：市场状态在同日对全市场同值，产出 date×code 面板。
   **不得进 FeatureStore 的截面 zscore 通道**——同日同值过截面标准化
   std=0 → 整列 NaN（与「winsorize_mad 整截面压成常数」同型坑）。
   量纲统一改用 **expanding z**（只用 t 日及以前统计量）。
2. **防前视**：全部特征只依赖指数 t 日及以前收盘（pct_change / rolling
   / expanding 天然回看）；测试含扰动未来值的因果锁。
3. **σ→0 / 零信息处置 NaN**：expanding std=0（恒定序列，如单边新高
   的 hi252≡0）置 NaN 而非除零常数/+1e-4——「防除零常数改写经济含义」
   教训（09-21）的直接应用；零信息特征诚实缺席。
4. **GBDT 单调不变**：expanding z 是逐点单调变换，对树模型无损；
   对 ridge 臂则保证特征量纲 ~O(1)，两臂共用同一套特征。

注入路径（scripts/pipelines/rolling_grid_alla.py::stage_predict）：
旁路并入训练/预测特征集——**不过 select 漏斗**（逐股 IC 对日级特征无
定义）、**不进 FeatureStore LRU**（无逐因子 parquet）、**不进
existence_mask**（幽灵股守卫保持因子口径，避免广播特征稀释阈值语义）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("model.market_features")

__all__ = [
    "build_market_state_features",
    "index_state_series",
    "FEATURES_PER_INDEX",
    "MARKET_FEAT_VERSION",
    "Z_MIN_PERIODS",
]

#: 特征集版本：改特征定义/增删特征时 +1（进入 pred 指纹，防静默混口径）。
MARKET_FEAT_VERSION = 1

#: expanding z 预热期（交易日）；不足置 NaN。2015 起的指数缓存到 2018 OOS
#: 首年早已出预热，对主实验窗口零影响。
Z_MIN_PERIODS = 120

#: 单只指数的特征清单（名称 → 构造闭式，全部回看型）：
#:   ret1d/ret5d/ret20d/ret60d  多周期动量（当日收益 ~ 季度动量）
#:   ma20r/ma60r                点位相对均线位置（趋势强度）
#:   vol20                      20 日已实现波动（风险状态）
#:   hi252/lo252                距 52 周高点/低点位置（≤0 / ≥0）
FEATURES_PER_INDEX = ("ret1d", "ret5d", "ret20d", "ret60d",
                      "ma20r", "ma60r", "vol20", "hi252", "lo252")

_MOM_HORIZONS = (5, 20, 60)
_MA_WINDOWS = (20, 60)
_VOL_WINDOW = 20
_HL_WINDOW = 252
_HL_MIN_PERIODS = 252   # 52 周高低点要求完整窗口（不足置 NaN，不缩窗）


def _expanding_z(s: pd.Series) -> pd.Series:
    """逐时点 expanding z（min_periods 前置 NaN；std=0 置 NaN）。"""
    mu = s.expanding(min_periods=Z_MIN_PERIODS).mean()
    sd = s.expanding(min_periods=Z_MIN_PERIODS).std()
    return (s - mu) / sd.where(sd > 0)


def index_state_series(close: pd.Series) -> dict[str, pd.Series]:
    """单只指数收盘 → 因果市场状态序列（全部只用 t 日及以前）。

    Args:
        close: 指数日收盘（index=date 升序；原始点位即可，比率/收益率
            类特征对点位刻度不变）。

    Returns:
        {特征名: Series}，名称即 ``FEATURES_PER_INDEX``（不带指数前缀）。
    """
    close = close.sort_index()
    close = close[close.notna()]
    ret = close.pct_change(fill_method=None)
    raw: dict[str, pd.Series] = {"ret1d": ret}
    for h in _MOM_HORIZONS:
        raw[f"ret{h}d"] = close.pct_change(h, fill_method=None)
    for w in _MA_WINDOWS:
        raw[f"ma{w}r"] = close / close.rolling(w).mean() - 1.0
    raw["vol20"] = ret.rolling(_VOL_WINDOW).std()
    hh = close.rolling(_HL_WINDOW, min_periods=_HL_MIN_PERIODS).max()
    ll = close.rolling(_HL_WINDOW, min_periods=_HL_MIN_PERIODS).min()
    raw["hi252"] = close / hh - 1.0
    raw["lo252"] = close / ll - 1.0
    assert set(raw) == set(FEATURES_PER_INDEX), "特征清单与实现脱节"
    return {k: _expanding_z(v) for k, v in raw.items()}


def build_market_state_features(
    index_closes: dict[str, pd.Series],
    dates: pd.DatetimeIndex,
    codes: pd.Index | list[str],
) -> dict[str, pd.DataFrame]:
    """指数收盘集 → date×code 广播特征面板（模型特征集可直接并入）。

    Args:
        index_closes: {tag: 指数收盘 Series}；tag 进入特征名
            （如 ``"000001_SH"``），**须已转安全字符**（无点号——LightGBM
            特征名含 ``.`` 会破坏 JSON/列名往返）。
        dates: 目标日期网格（个股面板 index，如 close_adj.index）。
        codes: 目标股票网格（个股面板 columns）。

    Returns:
        {f"mkt_{tag}_{feat}": date×code 面板}；同日全市场同值，
        指数缺失/预热不足的日期整行 NaN。float32（全A网格省内存）。
    """
    if not index_closes:
        raise ValueError("index_closes 为空")
    cols = pd.Index(codes)
    idx = pd.DatetimeIndex(dates)
    out: dict[str, pd.DataFrame] = {}
    for tag, close in index_closes.items():
        if "." in tag or "/" in tag:
            raise ValueError(f"tag {tag!r} 含非常规字符，须先转安全字符")
        for feat, s in index_state_series(close).items():
            v = s.reindex(idx).to_numpy(dtype="float64")
            panel = pd.DataFrame(
                np.tile(v[:, None], (1, len(cols))).astype("float32"),
                index=idx, columns=cols,
            )
            out[f"mkt_{tag}_{feat}"] = panel
    # 注：index_closes 是 dict，键天然唯一，无需再查重。
    log.info("市场状态特征: %d 指数 × %d 特征 = %d 面板（广播 %d 日 × %d 股）",
             len(index_closes), len(FEATURES_PER_INDEX), len(out), len(idx), len(cols))
    return out
