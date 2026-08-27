"""Qlib Alpha158 / Alpha360 公开因子集面板实现。

来源：Microsoft Qlib（https://github.com/microsoft/qlib），MIT License。
参考：qlib/contrib/data/loader.py 中 Alpha158DL / Alpha360DL。

两套因子集的设计差异：
- Alpha158：9 个 Kbar 形态因子 + 4 个价格相对因子 + 5×20=100 个滚动统计因子 = ~113 个
  （Qlib 默认 rolling windows=[5,10,20,30,60] 共 20 类 × 5 窗口 = 100，加上 kbar9 + price4 = 113）
  实际数量取决于 config：本文实现 Qlib 默认配置，含 kbar(9) + price(4) + rolling(20类×5窗口=100) = 113 个。
- Alpha360：60 日 OHLCV 原始价格序列归一化（÷最新 close/volume），6×60=360 个特征。
  纯原始特征，专为深度学习模型设计，不含任何加工因子。

与现有 alpha101/191 的区别：
- Alpha158 的特征更"轻加工"（多数是滚动 mean/std/rank 等基础统计），与 alpha101 的
  多算子嵌套公式互补；
- Alpha360 是"零加工"原始序列，适合做 LSTM/Transformer 的输入特征。

口径说明：
- 价格使用后复权 OHLC + vwap（与 alpha101/191 同口径，由 AlphaData 提供）；
- 所有除法分母加 1e-12 防零（与 Qlib 源码一致）；
- Qlib 的 Slope/Rsquare/Resi 对应本项目的 ts_slope/ts_rsquare/ts_residual；
- Qlib 的 IdxMax/IdxMin 对应 ts_arg_max/ts_arg_min（本项目已归一化到 [0,1]）。

用法：
    d = AlphaData(panels)
    panels158 = compute_alpha158(d)   # {name: 面板}
    panels360 = compute_alpha360(d)   # {name: 面板}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor.alpha_base import AlphaData
from factor.operators import (
    _safe_div,
    ts_arg_max,
    ts_arg_min,
    ts_corr,
    ts_max,
    ts_mean,
    ts_min,
    ts_quantile,
    ts_rank,
    ts_residual,
    ts_rsquare,
    ts_slope,
    ts_std,
    ts_sum,
)

__all__ = ["ALPHA158", "ALPHA360", "compute_alpha158", "compute_alpha360"]

_EPS = 1e-12

# Qlib 默认 rolling 窗口
_DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20, 30, 60)


# ===========================================================================
# Alpha158
# ===========================================================================
ALPHA158: dict[str, object] = {}


def _a158(name: str):
    def deco(fn):
        ALPHA158[name] = fn
        return fn
    return deco


# ---- Kbar 形态因子（9 个）----
# 公式直接来自 Qlib Alpha158DL.get_feature_config() kbar 段

@_a158("alpha158_KMID")
def _(d: AlphaData):
    """($close-$open)/$open — 收盘相对开盘的相对位置。"""
    return _safe_div(d.close - d.open, d.open)


@_a158("alpha158_KLEN")
def _(d: AlphaData):
    """($high-$low)/$open — K 线长度相对开盘价。"""
    return _safe_div(d.high - d.low, d.open)


@_a158("alpha158_KMID2")
def _(d: AlphaData):
    """($close-$open)/($high-$low) — 收盘在 K 线振幅中的位置。"""
    return _safe_div(d.close - d.open, d.high - d.low + _EPS)


@_a158("alpha158_KUP")
def _(d: AlphaData):
    """($high-Greater($open,$close))/$open — 上影线相对开盘价。"""
    upper_shadow = d.high - np.maximum(d.open, d.close)
    return _safe_div(upper_shadow, d.open)


@_a158("alpha158_KUP2")
def _(d: AlphaData):
    """($high-Greater($open,$close))/($high-$low) — 上影线占振幅比。"""
    upper_shadow = d.high - np.maximum(d.open, d.close)
    return _safe_div(upper_shadow, d.high - d.low + _EPS)


@_a158("alpha158_KLOW")
def _(d: AlphaData):
    """(Less($open,$close)-$low)/$open — 下影线相对开盘价。"""
    lower_shadow = np.minimum(d.open, d.close) - d.low
    return _safe_div(lower_shadow, d.open)


@_a158("alpha158_KLOW2")
def _(d: AlphaData):
    """(Less($open,$close)-$low)/($high-$low) — 下影线占振幅比。"""
    lower_shadow = np.minimum(d.open, d.close) - d.low
    return _safe_div(lower_shadow, d.high - d.low + _EPS)


@_a158("alpha158_KSFT")
def _(d: AlphaData):
    """(2*$close-$high-$low)/$open — 收盘偏离 K 线中点。"""
    return _safe_div(2 * d.close - d.high - d.low, d.open)


@_a158("alpha158_KSFT2")
def _(d: AlphaData):
    """(2*$close-$high-$low)/($high-$low) — 收盘偏离中点占振幅比。"""
    return _safe_div(2 * d.close - d.high - d.low, d.high - d.low + _EPS)


# ---- 价格相对因子（4 个，windows=[0] 的 OPEN/HIGH/LOW/VWAP）----

@_a158("alpha158_OPEN0")
def _(d: AlphaData):
    """$open/$close — 开盘价相对收盘价。"""
    return _safe_div(d.open, d.close)


@_a158("alpha158_HIGH0")
def _(d: AlphaData):
    """$high/$close — 最高价相对收盘价。"""
    return _safe_div(d.high, d.close)


@_a158("alpha158_LOW0")
def _(d: AlphaData):
    """$low/$close — 最低价相对收盘价。"""
    return _safe_div(d.low, d.close)


@_a158("alpha158_VWAP0")
def _(d: AlphaData):
    """$vwap/$close — VWAP 相对收盘价。"""
    return _safe_div(d.vwap, d.close)


# ---- Rolling 统计因子（20 类 × 5 窗口 = 100 个）----
# 与 Qlib 默认配置一致：windows=[5,10,20,30,60]，exclude=['RANK']，
# 但 Qlib Alpha158 handler 默认 rolling config 为空（{}），实际使用 handler 默认时
# rolling 部分不生成。此处实现完整 rolling 集（exclude RANK），供需要时使用。
# 注：Qlib Alpha158 handler 默认 get_feature_config 中 rolling={}，所以实际默认
# 只有 kbar(9)+price(4)=13 个。但 Alpha158DL 的完整 rolling 集是业界标准用法，
# 我们实现完整版（20 类 × 5 = 100），加上 kbar+price 共 113 个。

def _register_rolling():
    """注册 20 类 rolling 因子 × 5 窗口 = 100 个。"""
    windows = _DEFAULT_WINDOWS

    def _reg(name_suffix: str, fn_factory):
        for w in windows:
            name = f"alpha158_{name_suffix}{w}"
            ALPHA158[name] = fn_factory(w)

    # ROC: Ref($close, d)/$close — 过去 d 日收盘价 / 最新收盘价
    _reg("ROC", lambda w: lambda d: _safe_div(d.close.shift(w), d.close))

    # MA: Mean($close, d)/$close — d 日均值 / 最新收盘价
    _reg("MA", lambda w: lambda d: _safe_div(ts_mean(d.close, w), d.close))

    # STD: Std($close, d)/$close — d 日标准差 / 最新收盘价
    _reg("STD", lambda w: lambda d: _safe_div(ts_std(d.close, w), d.close))

    # BETA: Slope($close, d)/$close — 回归斜率 / 最新收盘价
    _reg("BETA", lambda w: lambda d: _safe_div(ts_slope(d.close, w), d.close))

    # RSQR: Rsquare($close, d) — 回归 R²
    _reg("RSQR", lambda w: lambda d: ts_rsquare(d.close, w))

    # RESI: Resi($close, d)/$close — 回归残差 / 最新收盘价
    _reg("RESI", lambda w: lambda d: _safe_div(ts_residual(d.close, w), d.close))

    # MAX: Max($high, d)/$close — d 日最高价 / 最新收盘价
    _reg("MAX", lambda w: lambda d: _safe_div(ts_max(d.high, w), d.close))

    # MIN: Min($low, d)/$close — d 日最低价 / 最新收盘价
    _reg("MIN", lambda w: lambda d: _safe_div(ts_min(d.low, w), d.close))

    # QTLU: Quantile($close, d, 0.8)/$close — 80% 分位 / 最新收盘价
    _reg("QTLU", lambda w: lambda d: _safe_div(ts_quantile(d.close, w, 0.8), d.close))

    # QTLD: Quantile($close, d, 0.2)/$close — 20% 分位 / 最新收盘价
    _reg("QTLD", lambda w: lambda d: _safe_div(ts_quantile(d.close, w, 0.2), d.close))

    # RANK: Rank($close, d) — 当期收盘价在过去 d 日的百分位
    # （Qlib 默认 exclude RANK，但本项目实现它——不排除任何因子）
    _reg("RANK", lambda w: lambda d: ts_rank(d.close, w))

    # RSV: ($close-Min($low,d))/(Max($high,d)-Min($low,d)) — 价格在区间中的位置
    def _rsv(w):
        def fn(d: AlphaData):
            lo = ts_min(d.low, w)
            hi = ts_max(d.high, w)
            return _safe_div(d.close - lo, hi - lo + _EPS)
        return fn
    _reg("RSV", _rsv)

    # IMAX: IdxMax($high, d)/d — 最高价距当期的归一化天数
    _reg("IMAX", lambda w: lambda d: ts_arg_max(d.high, w))

    # IMIN: IdxMin($low, d)/d — 最低价距当期的归一化天数
    _reg("IMIN", lambda w: lambda d: ts_arg_min(d.low, w))

    # IMXD: (IdxMax-IdxMin)/d — 最高最低价时间差
    def _imxd(w):
        def fn(d: AlphaData):
            return ts_arg_max(d.high, w) - ts_arg_min(d.low, w)
        return fn
    _reg("IMXD", _imxd)

    # CORR: Corr($close, Log($volume+1), d) — 量价相关性
    def _corr(w):
        def fn(d: AlphaData):
            log_vol = np.log(d.volume + 1)
            return ts_corr(d.close, log_vol, w)
        return fn
    _reg("CORR", _corr)

    # CORD: Corr($close/Ref($close,1), Log($volume/Ref($volume,1)+1), d) — 量价变化相关性
    def _cord(w):
        def fn(d: AlphaData):
            price_chg = _safe_div(d.close, d.close.shift(1)) - 1
            vol_chg = np.log(_safe_div(d.volume, d.volume.shift(1)) + 1)
            return ts_corr(price_chg, vol_chg, w)
        return fn
    _reg("CORD", _cord)

    # CNTP: Mean($close>Ref($close,1), d) — 过去 d 日上涨天数占比
    def _cntp(w):
        def fn(d: AlphaData):
            up = (d.close > d.close.shift(1)).astype(float)
            return ts_mean(up, w)
        return fn
    _reg("CNTP", _cntp)

    # CNTN: Mean($close<Ref($close,1), d) — 过去 d 日下跌天数占比
    def _cntn(w):
        def fn(d: AlphaData):
            down = (d.close < d.close.shift(1)).astype(float)
            return ts_mean(down, w)
        return fn
    _reg("CNTN", _cntn)

    # CNTD: CNTP - CNTN — 涨跌天数差
    def _cntd(w):
        def fn(d: AlphaData):
            up = (d.close > d.close.shift(1)).astype(float)
            down = (d.close < d.close.shift(1)).astype(float)
            return ts_mean(up, w) - ts_mean(down, w)
        return fn
    _reg("CNTD", _cntd)

    # SUMP: Sum(Greater($close-Ref($close,1),0), d)/(Sum(Abs($close-Ref($close,1)), d)) — 总涨幅占比
    def _sump(w):
        def fn(d: AlphaData):
            chg = d.close - d.close.shift(1)
            gain = chg.clip(lower=0.0)
            abs_chg = chg.abs()
            return _safe_div(ts_sum(gain, w), ts_sum(abs_chg, w) + _EPS)
        return fn
    _reg("SUMP", _sump)

    # SUMN: Sum(Greater(Ref($close,1)-$close,0), d)/(Sum(Abs(...))) — 总跌幅占比
    def _sumn(w):
        def fn(d: AlphaData):
            chg = d.close - d.close.shift(1)
            loss = (-chg).clip(lower=0.0)
            abs_chg = chg.abs()
            return _safe_div(ts_sum(loss, w), ts_sum(abs_chg, w) + _EPS)
        return fn
    _reg("SUMN", _sumn)

    # SUMD: (SumGain - SumLoss)/SumAbs — 涨跌净值占比
    def _sumd(w):
        def fn(d: AlphaData):
            chg = d.close - d.close.shift(1)
            gain = chg.clip(lower=0.0)
            loss = (-chg).clip(lower=0.0)
            abs_chg = chg.abs()
            return _safe_div(
                ts_sum(gain, w) - ts_sum(loss, w),
                ts_sum(abs_chg, w) + _EPS,
            )
        return fn
    _reg("SUMD", _sumd)

    # VMA: Mean($volume, d)/($volume) — 成交量均值 / 最新成交量
    _reg("VMA", lambda w: lambda d: _safe_div(ts_mean(d.volume, w), d.volume + _EPS))

    # VSTD: Std($volume, d)/($volume) — 成交量标准差 / 最新成交量
    _reg("VSTD", lambda w: lambda d: _safe_div(ts_std(d.volume, w), d.volume + _EPS))

    # WVMA: Std(Abs($close/Ref($close,1)-1)*$volume, d)/(Mean(Abs(...)*$volume, d)) — 加权价格波动率
    def _wvma(w):
        def fn(d: AlphaData):
            ret = (_safe_div(d.close, d.close.shift(1)) - 1).abs() * d.volume
            return _safe_div(ts_std(ret, w), ts_mean(ret, w) + _EPS)
        return fn
    _reg("WVMA", _wvma)

    # VSUMP: Sum(Greater($volume-Ref($volume,1),0), d)/(Sum(Abs(...))) — 成交量增加占比
    def _vsump(w):
        def fn(d: AlphaData):
            chg = d.volume - d.volume.shift(1)
            gain = chg.clip(lower=0.0)
            abs_chg = chg.abs()
            return _safe_div(ts_sum(gain, w), ts_sum(abs_chg, w) + _EPS)
        return fn
    _reg("VSUMP", _vsump)

    # VSUMN: Sum(Greater(Ref($volume,1)-$volume,0), d)/(Sum(Abs(...))) — 成交量减少占比
    def _vsumn(w):
        def fn(d: AlphaData):
            chg = d.volume - d.volume.shift(1)
            loss = (-chg).clip(lower=0.0)
            abs_chg = chg.abs()
            return _safe_div(ts_sum(loss, w), ts_sum(abs_chg, w) + _EPS)
        return fn
    _reg("VSUMN", _vsumn)

    # VSUMD: (VolGain - VolLoss)/VolAbs — 成交量净变化占比
    def _vsumd(w):
        def fn(d: AlphaData):
            chg = d.volume - d.volume.shift(1)
            gain = chg.clip(lower=0.0)
            loss = (-chg).clip(lower=0.0)
            abs_chg = chg.abs()
            return _safe_div(
                ts_sum(gain, w) - ts_sum(loss, w),
                ts_sum(abs_chg, w) + _EPS,
            )
        return fn
    _reg("VSUMD", _vsumd)


# 模块加载时注册 rolling 因子
_register_rolling()


def compute_alpha158(d: AlphaData, skip: bool = True) -> dict[str, pd.DataFrame]:
    """计算全部 Alpha158 因子面板。

    Args:
        d: AlphaData 数据命名空间。
        skip: 保留接口一致性，Alpha158 无跳过项。

    Returns:
        {因子名: date×code 面板}，±inf 统一替换为 NaN。
    """
    out: dict[str, pd.DataFrame] = {}
    for name, fn in ALPHA158.items():
        panel = fn(d)
        out[name] = panel.replace([np.inf, -np.inf], np.nan)
    return out


# ===========================================================================
# Alpha360
# ===========================================================================

ALPHA360: dict[str, object] = {}


def _a360(name: str):
    def deco(fn):
        ALPHA360[name] = fn
        return fn
    return deco


def _register_alpha360():
    """注册 Alpha360 的 360 个因子（6 字段 × 60 日）。"""

    def _close_ref(d: AlphaData, i: int):
        """Ref($close, i)/$close — i 日前收盘价 / 最新收盘价。"""
        return _safe_div(d.close.shift(i), d.close) if i > 0 else _safe_div(d.close, d.close)

    def _open_ref(d: AlphaData, i: int):
        return _safe_div(d.open.shift(i), d.close) if i > 0 else _safe_div(d.open, d.close)

    def _high_ref(d: AlphaData, i: int):
        return _safe_div(d.high.shift(i), d.close) if i > 0 else _safe_div(d.high, d.close)

    def _low_ref(d: AlphaData, i: int):
        return _safe_div(d.low.shift(i), d.close) if i > 0 else _safe_div(d.low, d.close)

    def _vwap_ref(d: AlphaData, i: int):
        return _safe_div(d.vwap.shift(i), d.close) if i > 0 else _safe_div(d.vwap, d.close)

    def _volume_ref(d: AlphaData, i: int):
        return _safe_div(d.volume.shift(i), d.volume + _EPS) if i > 0 else _safe_div(d.volume, d.volume + _EPS)

    # Qlib 顺序：CLOSE59..1,0 → OPEN59..1,0 → HIGH.. → LOW.. → VWAP.. → VOLUME..
    field_specs = [
        ("CLOSE", _close_ref),
        ("OPEN", _open_ref),
        ("HIGH", _high_ref),
        ("LOW", _low_ref),
        ("VWAP", _vwap_ref),
        ("VOLUME", _volume_ref),
    ]
    for field_name, ref_fn in field_specs:
        for i in range(59, -1, -1):
            name = f"alpha360_{field_name}{i}"
            # 用闭包捕获 i 的当前值
            ALPHA360[name] = lambda d, _fn=ref_fn, _i=i: _fn(d, _i)


_register_alpha360()


def compute_alpha360(d: AlphaData, skip: bool = True) -> dict[str, pd.DataFrame]:
    """计算全部 Alpha360 因子面板。

    Args:
        d: AlphaData 数据命名空间。
        skip: 保留接口一致性，Alpha360 无跳过项。

    Returns:
        {因子名: date×code 面板}，±inf 统一替换为 NaN。
    """
    out: dict[str, pd.DataFrame] = {}
    for name, fn in ALPHA360.items():
        panel = fn(d)
        out[name] = panel.replace([np.inf, -np.inf], np.nan)
    return out
