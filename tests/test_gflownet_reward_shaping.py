"""奖励塑形（研报之二十四）+ RRE 秩稳定性筛选测试。

覆盖：
- factor_long_stats / long_excess_ir：多头超额 IR 的构造与方向性
- build_barra_styles / max_style_corr：风格价差代理与时序相关
- composed_factor_reward：shaping 关闭时退回纯 |IC|；开启时单调正确
- rank_stability：稳定因子 ≈1、噪声因子 ≈0
- make_reward_fn：λ/μ 开关等价性与惩罚生效
- select_low_corr(min_autocorr)：RRE 门槛剔除高换手因子
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.gflownet.reward import (
    LONG_IR_LAMBDA,
    build_barra_styles,
    composed_factor_reward,
    factor_long_stats,
    long_excess_ir,
    make_reward_fn,
    max_style_corr,
    rank_stability,
)
from factor.gflownet.selection import select_low_corr

N_DAYS, N_CODES = 260, 30


def _make_panel(seed: int = 7, momentum_strength: float = 0.6):
    """构造带截面动量结构的面板（ momentum 因子有持续正 IC 与正多头 IR）。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=N_DAYS, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(N_CODES)]
    # 截面稳定的"质量"排序 + 时序缓慢漂移 → 动量风格主导收益横截面
    quality = rng.normal(0, 1, (1, N_CODES))
    rets = np.zeros((N_DAYS, N_CODES))
    for t in range(1, N_DAYS):
        common = rng.normal(0, 0.01)
        rets[t] = (momentum_strength * 0.001 * quality[0]
                   + common
                   + rng.normal(0, 0.01, N_CODES))
    close = pd.DataFrame(np.exp(np.cumsum(rets, axis=0)), idx, codes)
    volume = pd.DataFrame(rng.lognormal(12, 0.5, (N_DAYS, N_CODES)), idx, codes)
    return {"close": close, "volume": volume}, close.pct_change().shift(-1)


@pytest.fixture(scope="module")
def panel():
    return _make_panel()


def test_factor_long_stats_direction(panel):
    """动量排序因子的多头超额均值应为正、IR > 0。"""
    p, rets = panel
    fp = p["close"].pct_change(20)             # 动量因子（有真实信号）
    le, spread = factor_long_stats(fp, rets)
    assert long_excess_ir(le) > 0
    # 多头-空头价差方差大于多头-市场
    assert spread.var() > le.var()


def test_composed_reward_pure_ic_when_shaping_off(panel):
    """λ=0 且 μ=0 时退回纯 |IC| 口径（与旧版一致）。"""
    p, rets = panel
    fp = p["close"].pct_change(20)
    r_off = composed_factor_reward(fp.copy(), rets, long_ir_lambda=0.0,
                                   barra_mu=0.0)
    from factor.gflownet.reward import rank_ic_series
    expect = float(rank_ic_series(fp, rets).abs().mean())
    assert r_off == pytest.approx(expect)


def test_composed_reward_ir_bonus_positive(panel):
    """有效多头 IR 的因子，λ>0 奖励应高于 λ=0。"""
    p, rets = panel
    fp = p["close"].pct_change(20)
    r0 = composed_factor_reward(fp.copy(), rets, long_ir_lambda=0.0,
                                barra_mu=0.0)
    r1 = composed_factor_reward(fp.copy(), rets, long_ir_lambda=LONG_IR_LAMBDA,
                                barra_mu=0.0)
    assert r1 > r0


def test_composed_reward_barra_penalty(panel):
    """与动量风格完全同向的因子受 Barra 惩罚（μ>0 奖励更低）。"""
    p, rets = panel
    fp = p["close"].pct_change(20)             # 本身就是动量代理
    styles = build_barra_styles(p, rets)
    assert "momentum20" in styles
    sc = max_style_corr(factor_long_stats(fp, rets)[1], styles)
    assert sc > 0.8                            # 自我相关，理应接近 1
    r0 = composed_factor_reward(fp.copy(), rets, barra_mu=0.0)
    r1 = composed_factor_reward(fp.copy(), rets, barra_mu=0.5, styles=styles)
    assert r1 < r0


def test_rank_stability_extremes():
    """缓慢单调因子 autocorr≈1；每日独立重排的噪声因子 autocorr≈0。"""
    idx = pd.date_range("2023-01-01", periods=120, freq="B")
    codes = [f"{i:06d}.SH" for i in range(40)]
    slow = pd.DataFrame([np.arange(40) for _ in idx], idx, codes)   # 排名恒定
    assert rank_stability(slow) > 0.99
    rng = np.random.default_rng(3)
    noise = pd.DataFrame(rng.normal(size=(120, 40)), idx, codes)
    assert rank_stability(noise) < 0.15


def test_make_reward_fn_switch_equivalence(panel):
    """make_reward_fn 在 shaping 关闭时与纯 |IC| 一致；开启后不低于 base。"""
    p, rets = panel
    feats = ["open", "high", "low", "close", "volume"]
    # 用 evaluator 注入固定面板（ts_mean(close,10)），绕开 MDP 构造
    fp = p["close"].rolling(10).mean()
    from factor.gflownet.reward import rank_ic_series
    ic_abs = float(rank_ic_series(fp, rets).abs().mean())
    r_pure_v = make_reward_fn(p, rets, feats, long_ir_lambda=0.0, barra_mu=0.0,
                              evaluator=lambda f: fp)(None)
    r_shaped_v = make_reward_fn(p, rets, feats, long_ir_lambda=0.5, barra_mu=0.0,
                                evaluator=lambda f: fp)(None)
    assert r_pure_v == pytest.approx(max(ic_abs, 1e-4))
    assert r_shaped_v >= r_pure_v * 0.999     # 该因子 IR>0，shaping 只增不减


def test_select_low_corr_rre_gate():
    """min_autocorr>0 应剔除每日大换血的噪声因子，保留稳定信号因子。"""
    _, rets = _make_panel()
    idx = rets.index
    codes = rets.columns
    rng = np.random.default_rng(11)
    noise = {"f_noisy": pd.DataFrame(rng.normal(size=(len(idx), len(codes))), idx, codes)}
    stable = {"f_stable": rets.cumsum() * 0 +                # 恒定排序（稳定）
              np.tile(np.arange(len(codes)), (len(idx), 1))}
    def ev(formula):
        return (noise if "noisy" in formula else stable)[formula]
    samples = [("f_noisy", 0.9), ("f_stable", 0.8)]           # R 降序
    out0 = select_low_corr(samples, {}, [], threshold=0.4,
                           evaluator=ev, min_autocorr=0.0)
    assert [f for f, _ in out0] == ["f_noisy", "f_stable"]    # 默认不过滤
    out1 = select_low_corr(samples, {}, [], threshold=0.4,
                           evaluator=ev, min_autocorr=0.5)
    assert [f for f, _ in out1] == ["f_stable"]               # 噪声因子被 RRE 剔除
