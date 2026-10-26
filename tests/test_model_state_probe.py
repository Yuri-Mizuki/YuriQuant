"""模型自诊断仓位探针的口径守卫（scripts/evaluation/model_state_probe.py）。

锁死最容易静默翻车的四处：
  1. IC 口径 —— 完美信号 IC ≈ 1、纯噪声 ≈ 0（方向写反会静默给出负 IC）
  2. 前视 —— 扰动未来预测不得改变历史状态
  3. 锚点 —— ω ≡ 1 时净值逐位不变（Δ≠0 说明实现有 bug）
  4. 成本单调 —— 加大成本必须让净值变差
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluation.model_state_probe import (
    _apply_position_cost,
    annual_return,
    daily_rank_ic,
    gate_expanding_median,
    geo_sharpe,
    month_end_index,
    monthly_return,
    omega_daily,
    self_check,
)


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(13)
    n_days, n_stk = 320, 700
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    cols = [f"S{i:04d}" for i in range(n_stk)]
    ret = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_stk)), index=idx, columns=cols)
    mask = pd.DataFrame(True, index=idx, columns=cols)
    return ret, ret.shift(-1), mask


def test_self_check_passes():
    self_check()


def test_ic_direction_and_noise_floor(synth):
    ret, fwd1, mask = synth
    ic_perfect = daily_rank_ic(fwd1, fwd1, mask).dropna()
    assert ic_perfect.mean() > 0.95
    # 用**当日**收益（与次日收益独立）做噪声信号，IC 必须 ≈ 0
    noise = daily_rank_ic(ret, fwd1, mask).dropna()
    assert abs(noise.mean()) < 0.02


def test_state_is_causal(synth):
    """状态量必须 shift(1)：扰动 t 之后的预测，不得改变 t 及以前的 IC。"""
    ret, fwd1, mask = synth
    cutoff = ret.index[240]
    pert = ret.copy()
    pert.loc[pert.index > cutoff] *= 5.0
    a = daily_rank_ic(ret, fwd1, mask)
    b = daily_rank_ic(pert, fwd1, mask)
    pd.testing.assert_series_equal(a.loc[:cutoff], b.loc[:cutoff], check_names=False)


def test_anchor_is_exact(synth):
    ret, _, _ = synth
    p = ret.mean(axis=1)
    r = _apply_position_cost(p, pd.Series(1.0, index=p.index))
    assert annual_return(r) == pytest.approx(annual_return(p), abs=1e-12)
    assert geo_sharpe(r) == pytest.approx(geo_sharpe(p), abs=1e-12)


def test_cost_is_monotone(synth):
    ret, _, _ = synth
    p = ret.mean(axis=1)
    expo = pd.Series(
        np.random.default_rng(1).uniform(0.3, 1.0, len(p)), index=p.index)
    assert (annual_return(_apply_position_cost(p, expo, 0))
            >= annual_return(_apply_position_cost(p, expo, 10))
            >= annual_return(_apply_position_cost(p, expo, 50)))


def test_gate_uses_only_past(synth):
    """扩张中位门只能看历史：扰动未来状态，历史 ω 不得变化。"""
    ret, _, _ = synth
    st = ret.mean(axis=1).rolling(20, min_periods=10).mean().dropna()
    st_me = st.iloc[::20]
    om = gate_expanding_median(st_me, above=True)
    cutoff = st_me.index[8]
    pert = st_me.copy()
    pert.loc[pert.index > cutoff] += 1.0
    om2 = gate_expanding_median(pert, above=True)
    pd.testing.assert_series_equal(om.loc[:cutoff], om2.loc[:cutoff], check_names=False)


def test_monthly_helpers_are_consistent(synth):
    """月末索引 = 每月最后交易日；月收益 = 日收益复利。"""
    ret, _, _ = synth
    p = ret.mean(axis=1)
    me = month_end_index(p.index)
    assert me[0] == p.index[p.index.to_period("M") == p.index[0].to_period("M")][-1]
    mret = monthly_return(p)
    total_daily = float((1 + p).prod())
    total_month = float((1 + mret).prod())
    assert total_daily == pytest.approx(total_month, rel=1e-9)


def test_omega_maps_prev_month_end(synth):
    """月末 ω 只作用于**下一个月**：当月首日的 ω 等于上月末的值。"""
    ret, _, _ = synth
    p = ret.mean(axis=1)
    me = month_end_index(p.index)
    om_me = pd.Series(np.arange(len(me), dtype=float), index=me)
    daily = omega_daily(om_me, p.index)
    first_of_m2 = p.index[p.index.to_period("M") == me[1].to_period("M")][0]
    assert daily.loc[first_of_m2] == pytest.approx(om_me.iloc[0])
