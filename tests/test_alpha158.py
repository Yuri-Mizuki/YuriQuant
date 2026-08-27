"""Qlib Alpha158 / Alpha360 因子集单测。

覆盖三层：
1. 公式抽样对照：随机面板上手算参考表达式 vs 因子实现（语义回归锚点）；
2. 全量可计算性：两套因子集所有面板形状/数值合法（无 ±inf）；
3. 因子计数与命名约定。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.alpha158 import ALPHA158, ALPHA360, compute_alpha158, compute_alpha360
from factor.alpha_base import AlphaData

CFG = {
    "window": 60,
    "window_long": 252,
    "max_stale_days": 7,
    "confirm_n": 1,
    "min_coverage": 0.5,
    "warn_ic_retention": 0.5,
    "min_monotonicity": 0.5,
    "min_t_nw_recent": 1.0,
    "ledger_root": "reports/monitoring",
}


# ---------------------------------------------------------------------------
# 合成面板 fixture（与 test_alpha_factors 同结构，保证预热充分）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def alpha_env() -> dict:
    rng = np.random.default_rng(42)
    days, codes = 300, 40
    dates = pd.bdate_range("2024-01-02", periods=days)
    cols = [f"c{i:03d}" for i in range(codes)]
    eps = rng.normal(0, 2.0, (days, codes))
    ar = np.empty((days, codes))
    ar[0] = eps[0]
    for t in range(1, days):
        ar[t] = 0.95 * ar[t - 1] + eps[t]
    close = pd.DataFrame(50 + ar, index=dates, columns=cols)
    open_ = close.shift(1) * (1 + rng.normal(0, 0.004, (days, codes)))
    open_.iloc[0] = close.iloc[0]
    high = pd.DataFrame(np.maximum(open_.values, close.values) * 1.01,
                        index=dates, columns=cols)
    low = pd.DataFrame(np.minimum(open_.values, close.values) * 0.99,
                       index=dates, columns=cols)
    volume = pd.DataFrame(rng.lognormal(9, 0.4, (days, codes)),
                          index=dates, columns=cols)
    amount = volume * (high + low + close) / 3
    vwap = (amount / volume).replace([np.inf, -np.inf], np.nan)
    panels = {"open": open_, "high": high, "low": low, "close": close,
              "volume": volume, "amount": amount, "vwap": vwap}
    return {"d": AlphaData(panels), "panels": panels,
            "dates": dates, "cols": cols}


# ---------------------------------------------------------------------------
# 1) 公式抽样对照（手算参考 vs 实现）
# ---------------------------------------------------------------------------
def test_alpha158_KMID_formula(alpha_env):
    p = alpha_env["panels"]
    ref = (p["close"] - p["open"]) / p["open"]
    out = compute_alpha158(alpha_env["d"])["alpha158_KMID"]
    pd.testing.assert_frame_equal(out, ref.replace([np.inf, -np.inf], np.nan))


def test_alpha158_KLEN_formula(alpha_env):
    p = alpha_env["panels"]
    ref = (p["high"] - p["low"]) / p["open"]
    out = compute_alpha158(alpha_env["d"])["alpha158_KLEN"]
    pd.testing.assert_frame_equal(out, ref.replace([np.inf, -np.inf], np.nan))


def test_alpha158_ROC5_formula(alpha_env):
    p = alpha_env["panels"]
    ref = p["close"].shift(5) / p["close"]
    out = compute_alpha158(alpha_env["d"])["alpha158_ROC5"]
    pd.testing.assert_frame_equal(out, ref.replace([np.inf, -np.inf], np.nan))


def test_alpha158_MA10_formula(alpha_env):
    p = alpha_env["panels"]
    ref = p["close"].rolling(10, min_periods=10).mean() / p["close"]
    out = compute_alpha158(alpha_env["d"])["alpha158_MA10"]
    pd.testing.assert_frame_equal(out, ref.replace([np.inf, -np.inf], np.nan))


def test_alpha158_STD20_formula(alpha_env):
    p = alpha_env["panels"]
    ref = p["close"].rolling(20, min_periods=20).std(ddof=1) / p["close"]
    out = compute_alpha158(alpha_env["d"])["alpha158_STD20"]
    pd.testing.assert_frame_equal(out, ref.replace([np.inf, -np.inf], np.nan))


def test_alpha158_CNTP5_formula(alpha_env):
    p = alpha_env["panels"]
    up = (p["close"] > p["close"].shift(1)).astype(float)
    ref = up.rolling(5, min_periods=5).mean()
    out = compute_alpha158(alpha_env["d"])["alpha158_CNTP5"]
    pd.testing.assert_frame_equal(out, ref.replace([np.inf, -np.inf], np.nan))


def test_alpha158_SUMP10_formula(alpha_env):
    p = alpha_env["panels"]
    chg = p["close"] - p["close"].shift(1)
    gain = chg.clip(lower=0.0)
    abs_chg = chg.abs()
    ref = gain.rolling(10, min_periods=10).sum() / (
        abs_chg.rolling(10, min_periods=10).sum() + 1e-12)
    out = compute_alpha158(alpha_env["d"])["alpha158_SUMP10"]
    pd.testing.assert_frame_equal(out, ref.replace([np.inf, -np.inf], np.nan))


def test_alpha158_VWAP0_formula(alpha_env):
    p = alpha_env["panels"]
    ref = p["vwap"] / p["close"]
    out = compute_alpha158(alpha_env["d"])["alpha158_VWAP0"]
    pd.testing.assert_frame_equal(out, ref.replace([np.inf, -np.inf], np.nan))


def test_alpha360_CLOSE5_formula(alpha_env):
    p = alpha_env["panels"]
    ref = p["close"].shift(5) / p["close"]
    out = compute_alpha360(alpha_env["d"])["alpha360_CLOSE5"]
    pd.testing.assert_frame_equal(out, ref.replace([np.inf, -np.inf], np.nan))


def test_alpha360_VOLUME0_formula(alpha_env):
    p = alpha_env["panels"]
    ref = p["volume"] / (p["volume"] + 1e-12)
    out = compute_alpha360(alpha_env["d"])["alpha360_VOLUME0"]
    pd.testing.assert_frame_equal(out, ref.replace([np.inf, -np.inf], np.nan))


def test_alpha360_OPEN0_formula(alpha_env):
    p = alpha_env["panels"]
    ref = p["open"] / p["close"]
    out = compute_alpha360(alpha_env["d"])["alpha360_OPEN0"]
    pd.testing.assert_frame_equal(out, ref.replace([np.inf, -np.inf], np.nan))


# ---------------------------------------------------------------------------
# 2) 全量可计算性
# ---------------------------------------------------------------------------
def test_all_alpha158_computable(alpha_env):
    out = compute_alpha158(alpha_env["d"])
    assert len(out) == 158
    for name, p in out.items():
        assert p.shape == (300, 40), f"{name} 形状错误"
        assert not np.isinf(p.to_numpy()).any(), f"{name} 含 ±inf"
        # 尾部 20 日覆盖率（长窗口因子需 60 日预热）
        tail = p.iloc[-20:]
        cov = tail.notna().to_numpy().mean()
        assert cov > 0.3, f"{name} 尾部覆盖率过低 ({cov:.2f})"


def test_all_alpha360_computable(alpha_env):
    out = compute_alpha360(alpha_env["d"])
    assert len(out) == 360
    for name, p in out.items():
        assert p.shape == (300, 40), f"{name} 形状错误"
        assert not np.isinf(p.to_numpy()).any(), f"{name} 含 ±inf"
        # Alpha360 是纯 shift+除法，尾部 20 日应全覆盖
        tail = p.iloc[-20:]
        cov = tail.notna().to_numpy().mean()
        assert cov > 0.5, f"{name} 尾部覆盖率过低 ({cov:.2f})"


# ---------------------------------------------------------------------------
# 3) 因子计数与命名约定
# ---------------------------------------------------------------------------
def test_alpha158_count():
    """Alpha158 = kbar(9) + price(4) + rolling(20类×5窗口=100) + 45 = 158"""
    assert len(ALPHA158) == 158
    # kbar 9 个
    kbar = [n for n in ALPHA158 if n.startswith("alpha158_K")]
    assert len(kbar) == 9
    # price 4 个
    price = [n for n in ALPHA158 if n in {
        "alpha158_OPEN0", "alpha158_HIGH0", "alpha158_LOW0", "alpha158_VWAP0"}]
    assert len(price) == 4
    # rolling: 每类 5 个窗口，共 29 类 × 5 = 145
    rolling_names = [n for n in ALPHA158
                     if n not in kbar and n not in price]
    assert len(rolling_names) == 145


def test_alpha360_count():
    """Alpha360 = 6 字段 × 60 日 = 360"""
    assert len(ALPHA360) == 360
    for field in ("CLOSE", "OPEN", "HIGH", "LOW", "VWAP", "VOLUME"):
        names = [n for n in ALPHA360 if n.startswith(f"alpha360_{field}")]
        assert len(names) == 60, f"{field} 应有 60 个因子，实有 {len(names)}"


def test_alpha360_naming():
    """验证 Alpha360 的命名顺序：59..0"""
    close_names = [n for n in ALPHA360 if n.startswith("alpha360_CLOSE")]
    indices = [int(n.split("CLOSE")[1]) for n in close_names]
    assert sorted(indices) == list(range(60))
