"""大盘择时探针的口径守卫测试（scripts/evaluation/timing_probe.py）。

三条锁死最容易静默翻车的地方：
  1. 上界判据 —— oracle 低于满仓时必须被判为「无解」（这是本线的核心结论）
  2. 锚点臂 Δ 必须逐位为 0（Δ≠0 说明实现有 bug，输出数字全部不可信）
  3. 成本单调 —— 加大成本必须让净值变差（防成本项被写成加分项）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluation.timing_probe import (
    _apply_position_cost,
    annual_return,
    build_rules,
    consistency,
    geo_sharpe,
    max_drawdown,
    oracle_bounds,
    self_check,
)


@pytest.fixture(scope="module")
def synth():
    """合成底仓：正 alpha + beta=1 的市场 ⇒ beta 结构可解析验证。

    跨度覆盖 2018–2026（2600 个交易日），否则 ``2025+`` 一致性窗口为空。
    """
    rng = np.random.default_rng(7)
    idx = pd.date_range("2018-01-01", periods=2600, freq="B")
    m = pd.Series(rng.normal(0.0002, 0.011, 2600), index=idx)   # 市场上行
    p = m + 0.0003 + rng.normal(0, 0.004, 2600)                 # 组合 = 市场 + alpha
    return p, m


def test_self_check_passes():
    """脚本自带的前视 / 成本 / Sharpe 口径 / 基准重建四条守卫。"""
    self_check()


def test_anchor_arm_delta_is_exactly_zero(synth):
    """锚点（ω≡1）下所有 Δ 必须逐位为 0，否则实现有 bug。"""
    p, _ = synth
    expo = pd.Series(1.0, index=p.index)
    r = _apply_position_cost(p, expo)
    assert annual_return(r) == pytest.approx(annual_return(p), abs=1e-12)
    assert geo_sharpe(r) == pytest.approx(geo_sharpe(p), abs=1e-12)
    assert max_drawdown(r) == pytest.approx(max_drawdown(p), abs=1e-12)


def test_cost_is_monotone():
    """成本 bp 越大净值越差 —— 成本项不得被写成收益项。"""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=600, freq="B")
    p = pd.Series(rng.normal(0.0005, 0.010, 600), index=idx)
    expo = pd.Series(rng.uniform(0.3, 1.0, 600), index=idx)
    a0 = annual_return(_apply_position_cost(p, expo, 0))
    a10 = annual_return(_apply_position_cost(p, expo, 10))
    a50 = annual_return(_apply_position_cost(p, expo, 50))
    assert a0 >= a10 >= a50


def test_oracle_lookup_is_label_based(synth):
    """上界表取值必须按标签匹配，不能依赖行序（加行即静默取错）。"""
    p, m = synth
    orc = oracle_bounds(p, m).set_index("oracle")
    assert "日频" in str(orc.index[0])
    # 标签匹配能稳定取到日频行，且不与月频行混淆
    d = orc.loc[[i for i in orc.index if str(i).startswith("日频")], "annual"].iloc[0]
    mo = orc.loc[[i for i in orc.index if str(i).startswith("月频")], "annual"].iloc[0]
    assert not np.isclose(d, mo)


def test_rules_are_shifted_and_causal():
    """所有规则必须 shift(1) ⇒ 扰动未来值不应改变历史信号。"""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2020-01-01", periods=800, freq="B")
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800))), index=idx)
    mret = px.pct_change().fillna(0.0)
    rules = build_rules(px, mret)
    assert rules, "build_rules 返回空"

    cutoff = idx[600]
    perturbed = px.copy()
    perturbed.loc[perturbed.index > cutoff] *= 3.0     # 只扰动未来
    m2 = perturbed.pct_change().fillna(0.0)
    rules2 = build_rules(perturbed, m2)
    for k in rules:
        a = rules[k].loc[:cutoff]
        b = rules2[k].loc[:cutoff]
        pd.testing.assert_series_equal(a, b, check_names=False)


def test_consistency_window_shapes(synth):
    """一致性检验返回与窗口定义等长的元组。"""
    p, _ = synth
    from scripts.evaluation.timing_probe import CONSISTENCY_WINDOWS
    ds = consistency(p, p, p)
    assert len(ds) == len(CONSISTENCY_WINDOWS)
    assert all(x == pytest.approx(0.0, abs=1e-12) for x in ds)  # 自身对自身 Δ=0
