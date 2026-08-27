"""GFlowNet Phase 1（对齐研报）测试。

覆盖：51 算子全覆盖回归保护、市值中性化向量化与含截距 lstsq 等价、低相关筛选。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.gflownet.env import FactorMDP
from factor.gflownet.reward import neutralize_market_cap
from factor.gflownet.selection import select_low_corr

# 研报图表6 的 51 算子全集
REPORT_OPS = [
    "abs", "neg", "sign", "log", "inv", "sqrt", "signed_power2", "signed_power3",
    "ts_mean", "ts_std", "ts_max", "ts_min", "ts_rank", "ts_skew", "ts_kurt",
    "ts_median", "ts_delay", "ts_delta", "ts_pct_change", "ts_sum", "ts_argmax",
    "ts_argmin", "ts_decay_linear", "ts_var", "ts_mad", "ts_count", "ts_ema",
    "ts_wma", "ts_slope", "ts_rsquare", "ts_residual", "ts_quantile",
    "add", "sub", "mul", "div", "max2", "min2", "greater", "less",
    "ts_corr", "ts_cov", "ts_beta", "ts_orth",
    "cs_rank", "cs_zscore", "cs_demean", "cs_scale", "cs_normalize",
    "cs_winsorize", "cs_truncate",
]


@pytest.fixture
def small_panel():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=80, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(12)]
    close = pd.DataFrame(rng.normal(0, 1, (80, 12)), idx, codes)
    mc = pd.DataFrame(rng.lognormal(16, 1.2, (80, 12)), idx, codes)
    return {"close": close}, close, mc


def test_report_51_ops_covered():
    """研报 51 算子全集可被 FactorMDP 实例化（别名映射后无 KeyError）。"""
    mdp = FactorMDP(REPORT_OPS, (5, 10, 20), ["close", "volume"])
    assert mdp.n_op == 51


def test_neutralize_market_cap_equals_lstsq(small_panel):
    """向量化市值中性化 == 逐日含截距 lstsq 残差（浮点精度）。"""
    _, f, mc = small_panel
    b = neutralize_market_cap(f, mc)
    out = pd.DataFrame(np.nan, index=f.index, columns=f.columns)
    for d in f.index:
        y = f.loc[d].to_numpy(dtype=float)
        x = np.log(mc.loc[d].to_numpy(dtype=float))
        m = np.isfinite(y) & np.isfinite(x)
        X = np.column_stack([np.ones(m.sum()), x[m]])
        beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
        out.loc[d, f.columns[m]] = y[m] - X @ beta
    both = b.notna() & out.notna()
    assert both.sum().sum() > 0
    assert float((b[both] - out[both]).abs().max().max()) < 1e-10


def test_neutralize_market_cap_nan_safe(small_panel):
    """因子带 NaN 时中性化不崩且有效样本正确。"""
    _, f, mc = small_panel
    f2 = f.where(np.random.default_rng(1).random(f.shape) > 0.1)
    b = neutralize_market_cap(f2, mc)
    assert int(b.notna().sum().sum()) == int(f2.notna().sum().sum())


def test_select_low_corr_filters(small_panel):
    """低相关筛选：强相关公式被剔除，弱相关保留。"""
    panel, close, mc = small_panel
    feat = {"close": close, "close2": close * 2}  # 两特征完全共线
    samples = [("close", 0.2), ("close2", 0.19), ("close", 0.18)]  # 后两个与第一个相关=1
    selected = select_low_corr(samples, feat, ["close", "close2"], threshold=0.4)
    # close2 与 close 相关 1 应被剔除；重复 close 也应被剔除
    assert len(selected) == 1
    assert selected[0][0] == "close"
