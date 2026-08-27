"""WorldQuant「101 Formulaic Alphas」面板实现（Kakushadze, arXiv:1601.00991）。

- 纯 pandas 面板实现（date×code），离线可用；复用 factor/operators 算子语义。
- 数据输入：open/high/low/close/volume/amount（后复权价，vwap=amount/volume×bf）、
  returns=复权 close 日收益、adv_n=n 日平均成交额。
- 论文小数窗口（如 12.6556）四舍五入取整（``w()``）。
- IndNeutralize 用一级行业面板近似（论文分 sector/industry/subindustry），
  行业面板缺省时恒等；alpha056 跳过（含 cap，市值口径未定）。
- 布尔比较类公式：061/075/079/095 输出 0/1；062/064/065/068/074/081/086/099
  再乘 -1 输出 0/-1（按论文转写，见 https://github.com/Harvey-Sun/World_Quant_Alphas）。
- alpha029 原文转写有括号歧义，按常见转写实现（见函数注释）。

用法：
    d = AlphaData(panels, industry=ind_panel)
    panels_out = compute_alpha101(d)      # {name: 面板}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor.alpha_base import (  # noqa: F401
    AlphaData, abs_, argmax, argmin, corr, cov, decay_linear, delay, delta,
    if_else, log_, lt, max_, min_, power, product, rank, scale_wq, sign,
    stddev, tsmax, tsmin, tsr, w,
)

__all__ = ["ALPHA101", "compute_alpha101", "SKIPPED_101"]

SKIPPED_101: dict[str, str] = {
    "alpha101_056": "公式含 cap（市值），项目市值口径未定，跳过",
}

ALPHA101: dict[str, object] = {}


def _a(name: str):
    def deco(fn):
        ALPHA101[name] = fn
        return fn
    return deco


def _sum(x, n):
    return x.rolling(n, min_periods=n).sum()


def _mean(x, n):
    return x.rolling(n, min_periods=n).mean()


# ---------------------------------------------------------------------------
# 001-020
# ---------------------------------------------------------------------------
@_a("alpha101_001")
def _(d: AlphaData):
    inner = if_else(d.returns < 0, stddev(d.returns, 20), d.close)
    return rank(argmax(power(inner, 2), 5)) - 0.5


@_a("alpha101_002")
def _(d: AlphaData):
    return -1 * corr(rank(delta(log_(d.volume), 2)), rank((d.close - d.open) / d.open), 6)


@_a("alpha101_003")
def _(d: AlphaData):
    return -1 * corr(rank(d.open), rank(d.volume), 10)


@_a("alpha101_004")
def _(d: AlphaData):
    return -1 * tsr(rank(d.low), 9)


@_a("alpha101_005")
def _(d: AlphaData):
    return rank(d.open - _mean(d.vwap, 10)) * (-1 * abs_(rank(d.close - d.vwap)))


@_a("alpha101_006")
def _(d: AlphaData):
    return -1 * corr(d.open, d.volume, 10)


@_a("alpha101_007")
def _(d: AlphaData):
    return if_else(
        d.adv(20) < d.volume,
        -1 * tsr(abs_(delta(d.close, 7)), 60) * sign(delta(d.close, 7)),
        -1.0,
    )


@_a("alpha101_008")
def _(d: AlphaData):
    x = _sum(d.open, 5) * _sum(d.returns, 5)
    return -1 * rank(x - delay(x, 10))


@_a("alpha101_009")
def _(d: AlphaData):
    dc = delta(d.close, 1)
    return if_else(0 < tsmin(dc, 5), dc, if_else(tsmax(dc, 5) < 0, dc, -1 * dc))


@_a("alpha101_010")
def _(d: AlphaData):
    dc = delta(d.close, 1)
    inner = if_else(0 < tsmin(dc, 4), dc, if_else(tsmax(dc, 4) < 0, dc, -1 * dc))
    return rank(inner)


@_a("alpha101_011")
def _(d: AlphaData):
    x = d.vwap - d.close
    return (rank(tsmax(x, 3)) + rank(tsmin(x, 3))) * rank(delta(d.volume, 3))


@_a("alpha101_012")
def _(d: AlphaData):
    return sign(delta(d.volume, 1)) * (-1 * delta(d.close, 1))


@_a("alpha101_013")
def _(d: AlphaData):
    return -1 * rank(cov(rank(d.close), rank(d.volume), 5))


@_a("alpha101_014")
def _(d: AlphaData):
    return (-1 * rank(delta(d.returns, 3))) * corr(d.open, d.volume, 10)


@_a("alpha101_015")
def _(d: AlphaData):
    return -1 * _sum(rank(corr(rank(d.high), rank(d.volume), 3)), 3)


@_a("alpha101_016")
def _(d: AlphaData):
    return -1 * rank(cov(rank(d.high), rank(d.volume), 5))


@_a("alpha101_017")
def _(d: AlphaData):
    return (
        (-1 * rank(tsr(d.close, 10)))
        * rank(delta(delta(d.close, 1), 1))
        * rank(tsr(d.volume / d.adv(20), 5))
    )


@_a("alpha101_018")
def _(d: AlphaData):
    return -1 * rank(
        stddev(abs_(d.close - d.open), 5) + (d.close - d.open) + corr(d.close, d.open, 10)
    )


@_a("alpha101_019")
def _(d: AlphaData):
    return (
        -1 * sign((d.close - delay(d.close, 7)) + delta(d.close, 7))
        * (1 + rank(1 + _sum(d.returns, 250)))
    )


@_a("alpha101_020")
def _(d: AlphaData):
    return (
        (-1 * rank(d.open - delay(d.high, 1)))
        * rank(d.open - delay(d.close, 1))
        * rank(d.open - delay(d.low, 1))
    )


# ---------------------------------------------------------------------------
# 021-040
# ---------------------------------------------------------------------------
@_a("alpha101_021")
def _(d: AlphaData):
    m8, s8, m2 = _mean(d.close, 8), stddev(d.close, 8), _mean(d.close, 2)
    return if_else(
        (m8 + s8) < m2, -1.0,
        if_else(m2 < (m8 - s8), 1.0, if_else(d.volume / d.adv(20) >= 1, 1.0, -1.0)),
    )


@_a("alpha101_022")
def _(d: AlphaData):
    return -1 * delta(corr(d.high, d.volume, 5), 5) * rank(stddev(d.close, 20))


@_a("alpha101_023")
def _(d: AlphaData):
    return if_else(_mean(d.high, 20) < d.high, -1 * delta(d.high, 2), 0.0)


@_a("alpha101_024")
def _(d: AlphaData):
    cond = delta(_mean(d.close, 100), 100) / delay(d.close, 100)
    return if_else(cond <= 0.05, -1 * (d.close - tsmin(d.close, 100)), -1 * delta(d.close, 3))


@_a("alpha101_025")
def _(d: AlphaData):
    return rank((-1 * d.returns) * d.adv(20) * d.vwap * (d.high - d.close))


@_a("alpha101_026")
def _(d: AlphaData):
    return -1 * tsmax(corr(tsr(d.volume, 5), tsr(d.high, 5), 5), 3)


@_a("alpha101_027")
def _(d: AlphaData):
    s = _sum(corr(rank(d.volume), rank(d.vwap), 6), 2) / 2.0
    return if_else(0.5 < rank(s), -1.0, 1.0)


@_a("alpha101_028")
def _(d: AlphaData):
    return scale_wq(corr(d.adv(20), d.low, 5) + (d.high + d.low) / 2 - d.close)


@_a("alpha101_029")
def _(d: AlphaData):
    # 原文转写存在括号歧义，按常见转写实现
    x = rank(rank(scale_wq(log_(_sum(tsmin(rank(rank(-1 * rank(delta(d.close - 1, 5)))), 2), 1)))))
    return tsmin(x, 5) + tsr(delay(-1 * d.returns, 6), 5)


@_a("alpha101_030")
def _(d: AlphaData):
    s = (
        sign(d.close - delay(d.close, 1))
        + sign(delay(d.close, 1) - delay(d.close, 2))
        + sign(delay(d.close, 2) - delay(d.close, 3))
    )
    return (1.0 - rank(s)) * _sum(d.volume, 5) / _sum(d.volume, 20)


@_a("alpha101_031")
def _(d: AlphaData):
    return (
        rank(rank(rank(decay_linear(-1 * rank(rank(delta(d.close, 10))), 10))))
        + rank(-1 * delta(d.close, 3))
        + sign(scale_wq(corr(d.adv(20), d.low, 12)))
    )


@_a("alpha101_032")
def _(d: AlphaData):
    return scale_wq(_mean(d.close, 7) - d.close) + 20 * scale_wq(
        corr(d.vwap, delay(d.close, 5), 230)
    )


@_a("alpha101_033")
def _(d: AlphaData):
    return rank(-1 * (1 - (d.open / d.close)) ** 1)


@_a("alpha101_034")
def _(d: AlphaData):
    return rank(
        1 - rank(stddev(d.returns, 2) / stddev(d.returns, 5)) + 1 - rank(delta(d.close, 1))
    )


@_a("alpha101_035")
def _(d: AlphaData):
    return (
        tsr(d.volume, 32)
        * (1 - tsr((d.close + d.high) - d.low, 16))
        * (1 - tsr(d.returns, 32))
    )


@_a("alpha101_036")
def _(d: AlphaData):
    return (
        2.21 * rank(corr((d.close - d.open), delay(d.volume, 1), 15))
        + 0.7 * rank(d.open - d.close)
        + 0.73 * rank(tsr(delay(-1 * d.returns, 6), 5))
        + rank(abs_(corr(d.vwap, d.adv(20), 6)))
        + 0.6 * rank((_mean(d.close, 200) - d.open) * (d.close - d.open))
    )


@_a("alpha101_037")
def _(d: AlphaData):
    return rank(corr(delay(d.open - d.close, 1), d.close, 200)) + rank(d.open - d.close)


@_a("alpha101_038")
def _(d: AlphaData):
    return (-1 * rank(tsr(d.close, 10))) * rank(d.close / d.open)


@_a("alpha101_039")
def _(d: AlphaData):
    return (
        -1 * rank(delta(d.close, 7) * (1 - rank(decay_linear(d.volume / d.adv(20), 9))))
        * (1 + rank(_sum(d.returns, 250)))
    )


@_a("alpha101_040")
def _(d: AlphaData):
    return (-1 * rank(stddev(d.high, 10))) * corr(d.high, d.volume, 10)


# ---------------------------------------------------------------------------
# 041-060
# ---------------------------------------------------------------------------
@_a("alpha101_041")
def _(d: AlphaData):
    return (d.high * d.low) ** 0.5 - d.vwap


@_a("alpha101_042")
def _(d: AlphaData):
    return rank(d.vwap - d.close) / rank(d.vwap + d.close)


@_a("alpha101_043")
def _(d: AlphaData):
    return tsr(d.volume / d.adv(20), 20) * tsr(-1 * delta(d.close, 7), 8)


@_a("alpha101_044")
def _(d: AlphaData):
    return -1 * corr(d.high, rank(d.volume), 5)


@_a("alpha101_045")
def _(d: AlphaData):
    return -1 * (
        rank(_mean(delay(d.close, 5), 20))
        * corr(d.close, d.volume, 2)
        * rank(corr(_sum(d.close, 5), _sum(d.close, 20), 2))
    )


@_a("alpha101_046")
def _(d: AlphaData):
    x = ((delay(d.close, 20) - delay(d.close, 10)) / 10) - ((delay(d.close, 10) - d.close) / 10)
    return if_else(0.25 < x, -1.0, if_else(x < 0, 1.0, -1 * (d.close - delay(d.close, 1))))


@_a("alpha101_047")
def _(d: AlphaData):
    return (
        (rank(1 / d.close) * d.volume) / d.adv(20)
        * (d.high * rank(d.high - d.close)) / (_mean(d.high, 5) / 5)
        - rank(d.vwap - delay(d.vwap, 5))
    )


@_a("alpha101_048")
def _(d: AlphaData):
    num = d.ind(
        corr(delta(d.close, 1), delta(delay(d.close, 1), 1), 250) * delta(d.close, 1) / d.close
    )
    return num / _sum((delta(d.close, 1) / delay(d.close, 1)) ** 2, 250)


@_a("alpha101_049")
def _(d: AlphaData):
    x = ((delay(d.close, 20) - delay(d.close, 10)) / 10) - ((delay(d.close, 10) - d.close) / 10)
    return if_else(x < -0.1, 1.0, -1 * (d.close - delay(d.close, 1)))


@_a("alpha101_050")
def _(d: AlphaData):
    return -1 * tsmax(rank(corr(rank(d.volume), rank(d.vwap), 5)), 5)


@_a("alpha101_051")
def _(d: AlphaData):
    x = ((delay(d.close, 20) - delay(d.close, 10)) / 10) - ((delay(d.close, 10) - d.close) / 10)
    return if_else(x < -0.05, 1.0, -1 * (d.close - delay(d.close, 1)))


@_a("alpha101_052")
def _(d: AlphaData):
    return (
        (-1 * tsmin(d.low, 5) + delay(tsmin(d.low, 5), 5))
        * rank((_sum(d.returns, 240) - _sum(d.returns, 20)) / 220)
        * tsr(d.volume, 5)
    )


@_a("alpha101_053")
def _(d: AlphaData):
    x = ((d.close - d.low) - (d.high - d.close)) / (d.close - d.low)
    return -1 * delta(x, 9)


@_a("alpha101_054")
def _(d: AlphaData):
    return (-1 * ((d.low - d.close) * (d.open ** 5))) / ((d.low - d.high) * (d.close ** 5))


@_a("alpha101_055")
def _(d: AlphaData):
    x = (d.close - tsmin(d.low, 12)) / (tsmax(d.high, 12) - tsmin(d.low, 12))
    return -1 * corr(rank(x), rank(d.volume), 6)


@_a("alpha101_057")
def _(d: AlphaData):
    return 0 - (1 * ((d.close - d.vwap) / decay_linear(rank(argmax(d.close, 30)), 2)))


@_a("alpha101_058")
def _(d: AlphaData):
    return -1 * tsr(
        decay_linear(corr(d.ind(d.vwap), d.volume, w(3.92795)), w(7.89291)), w(5.50322)
    )


@_a("alpha101_059")
def _(d: AlphaData):
    # (vwap*0.728317)+(vwap*(1-0.728317)) 数学上恒等于 vwap
    return -1 * tsr(
        decay_linear(corr(d.ind(d.vwap), d.volume, w(4.25197)), w(16.2289)), w(8.19648)
    )


@_a("alpha101_060")
def _(d: AlphaData):
    return 0 - (
        1 * (
            2 * scale_wq(
                rank(((d.close - d.low) - (d.high - d.close)) / (d.high - d.low) * d.volume)
            )
            - scale_wq(rank(argmax(d.close, 10)))
        )
    )


# ---------------------------------------------------------------------------
# 061-080
# ---------------------------------------------------------------------------
@_a("alpha101_061")
def _(d: AlphaData):
    return lt(
        rank(d.vwap - tsmin(d.vwap, w(16.1219))),
        rank(corr(d.vwap, d.adv(180), w(17.9282))),
    )


@_a("alpha101_062")
def _(d: AlphaData):
    # 论文转写即 rank(open)+rank(open)（非笔误修正，保持原样）
    inner = lt(rank(d.open) + rank(d.open), rank((d.high + d.low) / 2) + rank(d.high))
    return -1 * lt(
        rank(corr(d.vwap, _sum(d.adv(20), w(22.4101)), w(9.91009))),
        rank(inner),
    )


@_a("alpha101_063")
def _(d: AlphaData):
    a = rank(decay_linear(delta(d.ind(d.close), w(2.25164)), w(8.22237)))
    b = rank(decay_linear(
        corr(d.vwap * 0.318108 + d.open * (1 - 0.318108),
             _sum(d.adv(180), w(37.2467)), w(13.557)),
        w(12.2883),
    ))
    return -1 * (a - b)


@_a("alpha101_064")
def _(d: AlphaData):
    lhs = rank(corr(
        _sum(d.open * 0.178404 + d.low * (1 - 0.178404), w(12.7054)),
        _sum(d.adv(120), w(12.7054)), w(16.6208),
    ))
    rhs = rank(delta((d.high + d.low) / 2 * 0.178404 + d.vwap * (1 - 0.178404), w(3.69741)))
    return -1 * lt(lhs, rhs)


@_a("alpha101_065")
def _(d: AlphaData):
    lhs = rank(corr(
        d.open * 0.00817205 + d.vwap * (1 - 0.00817205),
        _sum(d.adv(60), w(8.6911)), w(6.40374),
    ))
    return -1 * lt(lhs, rank(d.open - tsmin(d.open, w(13.635))))


@_a("alpha101_066")
def _(d: AlphaData):
    # low*0.96633 + low*(1-0.96633) 恒等于 low
    a = rank(decay_linear(delta(d.vwap, w(3.51013)), w(7.23052)))
    b = tsr(
        decay_linear((d.low - d.vwap) / (d.open - (d.high + d.low) / 2), w(11.4157)),
        w(6.72611),
    )
    return -1 * (a + b)


@_a("alpha101_067")
def _(d: AlphaData):
    return -1 * power(
        rank(d.high - tsmin(d.high, w(2.14593))),
        rank(corr(d.ind(d.vwap), d.ind(d.adv(20)), w(6.02936))),
    )


@_a("alpha101_068")
def _(d: AlphaData):
    return -1 * lt(
        tsr(corr(rank(d.high), rank(d.adv(15)), w(8.91644)), w(13.9333)),
        rank(delta(d.close * 0.518371 + d.low * (1 - 0.518371), w(1.06157))),
    )


@_a("alpha101_069")
def _(d: AlphaData):
    return -1 * power(
        rank(tsmax(delta(d.ind(d.vwap), w(2.72412)), w(4.79344))),
        tsr(corr(d.close * 0.490655 + d.vwap * (1 - 0.490655), d.adv(20), w(4.92416)), w(9.0615)),
    )


@_a("alpha101_070")
def _(d: AlphaData):
    return -1 * power(
        rank(delta(d.vwap, w(1.29456))),
        tsr(corr(d.ind(d.close), d.adv(50), w(17.8256)), w(17.9171)),
    )


@_a("alpha101_071")
def _(d: AlphaData):
    a = tsr(
        decay_linear(corr(tsr(d.close, w(3.43976)), tsr(d.adv(180), w(12.0647)), w(18.0175)), w(4.20501)),
        w(15.6948),
    )
    b = tsr(decay_linear(power(rank((d.low + d.open) - (d.vwap + d.vwap)), 2), w(16.4662)), w(4.4388))
    return max_(a, b)


@_a("alpha101_072")
def _(d: AlphaData):
    return rank(decay_linear(corr((d.high + d.low) / 2, d.adv(40), w(8.93345)), w(10.1519))) / rank(
        decay_linear(corr(tsr(d.vwap, w(3.72469)), tsr(d.volume, w(18.5188)), w(6.86671)), w(2.95011))
    )


@_a("alpha101_073")
def _(d: AlphaData):
    a = rank(decay_linear(delta(d.vwap, w(4.72775)), w(2.91864)))
    px = d.open * 0.147155 + d.low * (1 - 0.147155)
    b = tsr(decay_linear(-1 * delta(px, w(2.03608)) / px, w(3.33829)), w(16.7411))
    return -1 * max_(a, b)


@_a("alpha101_074")
def _(d: AlphaData):
    lhs = rank(corr(d.close, _sum(d.adv(30), w(37.4843)), w(15.1365)))
    rhs = rank(corr(rank(d.high * 0.0261661 + d.vwap * (1 - 0.0261661)), rank(d.volume), w(11.4791)))
    return -1 * lt(lhs, rhs)


@_a("alpha101_075")
def _(d: AlphaData):
    return lt(
        rank(corr(d.vwap, d.volume, w(4.24304))),
        rank(corr(rank(d.low), rank(d.adv(50)), w(12.4413))),
    )


@_a("alpha101_076")
def _(d: AlphaData):
    a = rank(decay_linear(delta(d.vwap, w(1.24383)), w(11.8259)))
    b = tsr(
        decay_linear(tsr(corr(d.ind(d.low), d.adv(81), w(8.14941)), w(19.569)), w(17.1543)),
        w(19.383),
    )
    return -1 * max_(a, b)


@_a("alpha101_077")
def _(d: AlphaData):
    a = rank(decay_linear((d.high + d.low) / 2 + d.high - (d.vwap + d.high), w(20.0451)))
    b = rank(decay_linear(corr((d.high + d.low) / 2, d.adv(40), w(3.1614)), w(5.64125)))
    return min_(a, b)


@_a("alpha101_078")
def _(d: AlphaData):
    return power(
        rank(corr(
            _sum(d.low * 0.352233 + d.vwap * (1 - 0.352233), w(19.7428)),
            _sum(d.adv(40), w(19.7428)), w(6.83313),
        )),
        rank(corr(rank(d.vwap), rank(d.volume), w(5.77492))),
    )


@_a("alpha101_079")
def _(d: AlphaData):
    return lt(
        rank(delta(d.ind(d.close * 0.60733 + d.open * (1 - 0.60733)), w(1.23438))),
        rank(corr(tsr(d.vwap, w(3.60973)), tsr(d.adv(150), w(9.18637)), w(14.6644))),
    )


@_a("alpha101_080")
def _(d: AlphaData):
    return -1 * power(
        rank(sign(delta(d.ind(d.open * 0.868128 + d.high * (1 - 0.868128)), w(4.04545)))),
        tsr(corr(d.high, d.adv(10), w(5.11456)), w(5.53756)),
    )


# ---------------------------------------------------------------------------
# 081-101
# ---------------------------------------------------------------------------
@_a("alpha101_081")
def _(d: AlphaData):
    inner = rank(power(rank(corr(d.vwap, _sum(d.adv(10), w(49.6054)), w(8.47743))), 4))
    lhs = rank(log_(product(inner, w(14.9655))))
    rhs = rank(corr(rank(d.vwap), rank(d.volume), w(5.07914)))
    return -1 * lt(lhs, rhs)


@_a("alpha101_082")
def _(d: AlphaData):
    # open*0.634196 + open*(1-0.634196) 恒等于 open
    a = rank(decay_linear(delta(d.open, w(1.46063)), w(14.8717)))
    b = tsr(decay_linear(corr(d.ind(d.volume), d.open, w(17.4842)), w(6.92131)), w(13.4283))
    return -1 * min_(a, b)


@_a("alpha101_083")
def _(d: AlphaData):
    x = (d.high - d.low) / _mean(d.close, 5)
    return (rank(delay(x, 2)) * rank(rank(d.volume))) / (x / (d.vwap - d.close))


@_a("alpha101_084")
def _(d: AlphaData):
    # Ts_Rank ∈ (0,1] 非负，SignedPower 退化为普通幂
    return power(tsr(d.vwap - tsmax(d.vwap, w(15.3217)), w(20.7127)), delta(d.close, w(4.96796)))


@_a("alpha101_085")
def _(d: AlphaData):
    return power(
        rank(corr(d.high * 0.876703 + d.close * (1 - 0.876703), d.adv(30), w(9.61331))),
        rank(corr(tsr((d.high + d.low) / 2, w(3.70596)), tsr(d.volume, w(10.1595)), w(7.11408))),
    )


@_a("alpha101_086")
def _(d: AlphaData):
    return -1 * lt(
        tsr(corr(d.close, _sum(d.adv(20), w(14.7444)), w(6.00049)), w(20.4195)),
        rank((d.open + d.close) - (d.vwap + d.open)),
    )


@_a("alpha101_087")
def _(d: AlphaData):
    a = rank(decay_linear(
        delta(d.close * 0.369701 + d.vwap * (1 - 0.369701), w(1.91233)), w(2.65461),
    ))
    b = tsr(decay_linear(abs_(corr(d.ind(d.adv(81)), d.close, w(13.4132))), w(4.89768)), w(14.4535))
    return -1 * max_(a, b)


@_a("alpha101_088")
def _(d: AlphaData):
    a = rank(decay_linear(rank(d.open) + rank(d.low) - rank(d.high) - rank(d.close), w(8.06882)))
    b = tsr(
        decay_linear(corr(tsr(d.close, w(8.44728)), tsr(d.adv(60), w(20.6966)), w(8.01266)), w(6.65053)),
        w(2.61957),
    )
    return min_(a, b)


@_a("alpha101_089")
def _(d: AlphaData):
    # low*0.967285 + low*(1-0.967285) 恒等于 low
    a = tsr(decay_linear(corr(d.low, d.adv(10), w(6.94279)), w(5.51607)), w(3.79744))
    b = tsr(decay_linear(delta(d.ind(d.vwap), w(3.48158)), w(10.1466)), w(15.3012))
    return a - b


@_a("alpha101_090")
def _(d: AlphaData):
    return -1 * power(
        rank(d.close - tsmax(d.close, w(4.66719))),
        tsr(corr(d.ind(d.adv(40)), d.low, w(5.38375)), w(3.21856)),
    )


@_a("alpha101_091")
def _(d: AlphaData):
    a = tsr(
        decay_linear(decay_linear(corr(d.ind(d.close), d.volume, w(9.74928)), w(16.398)), w(3.83219)),
        w(4.8667),
    )
    b = rank(decay_linear(corr(d.vwap, d.adv(30), w(4.01303)), w(2.6809)))
    return -1 * (a - b)


@_a("alpha101_092")
def _(d: AlphaData):
    a = tsr(decay_linear(lt((d.high + d.low) / 2 + d.close, d.low + d.open), w(14.7221)), w(18.8683))
    b = tsr(decay_linear(corr(rank(d.low), rank(d.adv(30)), w(7.58555)), w(6.94024)), w(6.80584))
    return min_(a, b)


@_a("alpha101_093")
def _(d: AlphaData):
    return tsr(
        decay_linear(corr(d.ind(d.vwap), d.adv(81), w(17.4193)), w(19.848)), w(7.54455)
    ) / rank(decay_linear(
        delta(d.close * 0.524434 + d.vwap * (1 - 0.524434), w(2.77377)), w(16.2664)
    ))


@_a("alpha101_094")
def _(d: AlphaData):
    return -1 * power(
        rank(d.vwap - tsmin(d.vwap, w(11.5783))),
        tsr(corr(tsr(d.vwap, w(19.6462)), tsr(d.adv(60), w(4.02992)), w(18.0926)), w(2.70756)),
    )


@_a("alpha101_095")
def _(d: AlphaData):
    return lt(
        rank(d.open - tsmin(d.open, w(12.4105))),
        tsr(power(
            rank(corr(_sum((d.high + d.low) / 2, w(19.1351)),
                      _sum(d.adv(40), w(19.1351)), w(12.8742))),
            5,
        ), w(11.7584)),
    )


@_a("alpha101_096")
def _(d: AlphaData):
    a = tsr(decay_linear(corr(rank(d.vwap), rank(d.volume), w(3.83878)), w(4.16783)), w(8.38151))
    b = tsr(
        decay_linear(
            argmax(corr(tsr(d.close, w(7.45404)), tsr(d.adv(60), w(4.13242)), w(3.65459)), w(12.6556)),
            w(14.0365),
        ),
        w(13.4143),
    )
    return -1 * max_(a, b)


@_a("alpha101_097")
def _(d: AlphaData):
    a = rank(decay_linear(
        delta(d.ind(d.low * 0.721001 + d.vwap * (1 - 0.721001)), w(3.3705)), w(20.4523),
    ))
    b = tsr(
        decay_linear(tsr(corr(tsr(d.low, w(7.87871)), tsr(d.adv(60), w(17.255)), w(4.97547)), w(18.5925)), w(15.7152)),
        w(6.71659),
    )
    return -1 * (a - b)


@_a("alpha101_098")
def _(d: AlphaData):
    a = rank(decay_linear(corr(d.vwap, _sum(d.adv(5), w(26.4719)), w(4.58418)), w(7.18088)))
    b = rank(decay_linear(
        tsr(argmin(corr(rank(d.open), rank(d.adv(15)), w(20.8187)), w(8.62571)), w(6.95668)),
        w(8.07206),
    ))
    return a - b


@_a("alpha101_099")
def _(d: AlphaData):
    return -1 * lt(
        rank(corr(_sum((d.high + d.low) / 2, w(19.8975)), _sum(d.adv(60), w(19.8975)), w(8.8136))),
        rank(corr(d.low, d.volume, w(6.28259))),
    )


@_a("alpha101_100")
def _(d: AlphaData):
    # 双重 subindustry 中性化对组内去均值幂等，实现只做一次
    inner = rank((d.close - d.low - (d.high - d.close)) / (d.high - d.low) * d.volume)
    t1 = scale_wq(d.ind(inner))
    t2 = scale_wq(d.ind(corr(d.close, rank(d.adv(20)), 5) - rank(argmin(d.close, 30))))
    return 0 - (1 * ((1.5 * t1 - t2) * (d.volume / d.adv(20))))


@_a("alpha101_101")
def _(d: AlphaData):
    return (d.close - d.open) / ((d.high - d.low) + 0.001)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def compute_alpha101(d: AlphaData, skip: bool = True) -> dict[str, pd.DataFrame]:
    """计算全部 alpha101 因子面板。

    Args:
        d: AlphaData 数据命名空间。
        skip: True 时跳过 SKIPPED_101 列出的因子（缺省）。

    Returns:
        {因子名: date×code 面板}，±inf 统一替换为 NaN。
    """
    out: dict[str, pd.DataFrame] = {}
    for name, fn in ALPHA101.items():
        if skip and name in SKIPPED_101:
            continue
        panel = fn(d)
        out[name] = panel.replace([np.inf, -np.inf], np.nan)
    return out
