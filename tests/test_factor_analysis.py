"""因子分析回归测试：quantile_backtest 与 IC/主回测口径对齐（2026-08-03 修复）。

旧实现用 factor[t-1] 赚 returns_panel[t]，因子整体晚一天生效，分层单调性
检验系统性丢失一天信号。修复后 factor[t] ↔ returns_panel[t]，与 IC 同口径。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.factor_analysis import calc_ic_series, quantile_backtest


def _strong_signal_data(n_days=180, n_codes=40, seed=3):
    """factor[t] = returns_panel[t] + 小噪声 → 强正信号，且只对当日未来收益有效。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"c{i:02d}" for i in range(n_codes)]
    rets = pd.DataFrame(rng.normal(0.0002, 0.01, (n_days, n_codes)), idx, codes)
    future = rets.shift(-1)                          # 未来一期收益（主口径）
    factor = future + rng.normal(0, 0.03, (n_days, n_codes))
    factor.iloc[-1] = np.nan
    return factor, future


def test_quantile_backtest_first_day_carries_signal():
    """修复验证：第一天（i=0）起分层就有信号 —— 当日因子赚当日收益。

    旧实现 `if i == 0: continue` + factor[t-1]，首日无信号、整体晚一天。
    """
    factor, future = _strong_signal_data()
    ic = calc_ic_series(factor, future).dropna()
    assert ic.mean() > 0.2                       # 构造的强信号

    layers = quantile_backtest(factor, future, n_quantiles=5)
    # cumprod 第一行 = 1 + 首日收益；未来收益面板最后一行 NaN → 末行无效，用 iloc[-2]
    first_diff = layers["Q5"].iloc[0] - layers["Q1"].iloc[0]
    assert first_diff > 0, "首日应有信号（与 IC 同向）"
    last_diff = layers["Q5"].iloc[-2] - layers["Q1"].iloc[-2]
    assert last_diff > 0, "分层单调性应与 IC 同向"


def test_quantile_backtest_monotonic_with_ic_sign():
    """分层 Q5-Q1 累计与 IC 同号（修复后口径一致时才成立）。"""
    factor, future = _strong_signal_data(seed=4)
    ic_mean = calc_ic_series(factor, future).dropna().mean()
    layers = quantile_backtest(factor, future, n_quantiles=5)
    q5q1 = layers["Q5"].iloc[-2] - layers["Q1"].iloc[-2]
    assert np.sign(q5q1) == np.sign(ic_mean)


def test_quantile_backtest_na_last_row():
    """未来收益面板末行无未来 → 该日因子无有效观测，跳过且不报错（保持前值）。"""
    factor, future = _strong_signal_data()
    layers = quantile_backtest(factor, future, n_quantiles=5)
    # 末行因子全 NaN（future 最后一行无未来）→ 该行保持 0 收益 → 净值与前一行相同
    assert np.allclose(layers.iloc[-1].values, layers.iloc[-2].values)
