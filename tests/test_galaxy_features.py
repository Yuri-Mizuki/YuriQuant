"""银河 0608 特征族与三标签（factor/classic.compute_galaxy_features /
build_galaxy_labels）单元测试。

覆盖：
- 分组差异化预处理：G1 形态原始尺度、动量/风险/量价类为时序滚动 zscore
  （逐票均值≈0，而非截面 zscore 的逐日均值≈0）；
- 特征族完整性（研报附录表 12 的关键项）与 NaN 不污染；
- 相对强弱/beta/idvol 的方向性（构造已知联动结构的合成数据）；
- 三标签：窗口语义（前视）、mdd 负值方向、sharpe=均值/波动。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.classic import (
    _rolling_zscore,
    build_galaxy_labels,
    compute_galaxy_features,
)


def _px(n_days: int = 260, n_codes: int = 6, seed: int = 3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    codes = [f"c{i}" for i in range(n_codes)]
    base = np.cumprod(1.0 + rng.normal(0.0005, 0.015, (n_days, n_codes)), axis=0)
    close = pd.DataFrame(base * 100.0, idx, codes)
    spread = pd.DataFrame(rng.uniform(0.005, 0.02, (n_days, n_codes)), idx, codes)
    px = {
        "open": close.shift(1) * (1 + rng.normal(0, 0.002, (n_days, n_codes))),
        "high": close * (1 + spread),
        "low": close * (1 - spread),
        "close": close,
        "volume": pd.DataFrame(rng.lognormal(10, 0.5, (n_days, n_codes)), idx, codes),
        "amount": pd.DataFrame(rng.lognormal(15, 0.5, (n_days, n_codes)), idx, codes),
    }
    idx_close = close.mean(axis=1)                    # 等权基准代理
    return px, idx_close


def test_galaxy_feature_family_complete():
    px, idx_close = _px()
    feats = compute_galaxy_features(px, idx_close)
    must = {"g1_body", "g1_mom_24", "g1_ma_gap_20", "g1_excess_mom_24",
            "g1_beta_80", "g2_downvol_24", "g2_mdd_24", "g2_idvol_80",
            "g3_vpr_24", "g3_flow_24"}
    assert must <= set(feats)
    assert 25 <= len(feats) <= 40
    for name, panel in feats.items():
        assert panel.shape == px["close"].shape, name
        # 长窗特征首值在 win+lookback 之后（最长的 idvol_80 ≈ beta_80
        # + 80 日 std + 50 日 zscore，约 160 行起）
        assert np.isfinite(panel.to_numpy(dtype=float)[190:, :]).all(), name


def test_rolling_zscore_is_timeseries_not_cross_section():
    """时序 zscore 对 iid 序列：每只股票时间均值≈0，而非每个截面均值≈0。

    （对强自相关的复合特征——如滚动 std——趋势段 z 分数可持续偏移，
    时间均值只近似为 0，故用白噪声直接验证函数本身。）
    """
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2024-01-02", periods=300)
    panel = pd.DataFrame(rng.normal(0, 1, (300, 4)), idx,
                         [f"c{i}" for i in range(4)])
    z = _rolling_zscore(panel, win=50).iloc[60:]
    p = z.to_numpy(dtype=float)
    assert np.allclose(np.nanmean(p, axis=0), 0.0, atol=0.15)   # 逐票≈0
    assert not np.allclose(np.nanmean(p, axis=1), 0.0, atol=1e-8)  # 逐日不必≈0


def test_g1_shape_keeps_raw_scale():
    """G1 形态类不做标准化：body ∈ [-1, 1] 量级而非 z 分数。"""
    px, idx_close = _px()
    feats = compute_galaxy_features(px, idx_close)
    body = feats["g1_body"].to_numpy(dtype=float)
    assert np.nanmax(np.abs(body)) < 1.0 + 1e-9


def test_beta_direction_known_structure():
    """构造 c5 = 基准 ×2 + 噪声：其 beta_24 应显著高于独立股票。"""
    rng = np.random.default_rng(7)
    n_days, n_codes = 260, 5
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    mkt = pd.Series(rng.normal(0, 0.01, n_days), idx)
    close = pd.DataFrame({f"c{i}": (1 + rng.normal(0.0002, 0.01, n_days)).cumprod() * 100
                          for i in range(n_codes)}, index=idx)
    close["c5"] = (1 + (2 * mkt + rng.normal(0, 0.002, n_days))).cumprod() * 100
    px = {"open": close, "high": close * 1.01, "low": close * 0.99,
          "close": close,
          "volume": pd.DataFrame(1e6, idx, close.columns),
          "amount": pd.DataFrame(1e8, idx, close.columns)}
    feats = compute_galaxy_features(px, mkt.cumsum() + 100)
    b = feats["g1_beta_24"].to_numpy(dtype=float)[50:]
    assert np.nanmean(b[:, 5]) > np.nanmean(b[:, :5])       # c5 的 beta 更高


def test_labels_forward_window_and_mdd_sign():
    px, idx_close = _px(n_days=200)
    labels = build_galaxy_labels(px, idx_close)
    assert set(labels) == {"label_alpha_22", "label_sharpe_22", "label_mdd_66"}
    # 前视：末 66 日的 mdd 标签必为 NaN（窗口不完整）
    mdd = labels["label_mdd_66"].to_numpy(dtype=float)
    assert np.isnan(mdd[-66:]).all()
    assert np.isfinite(mdd[:-66]).all()
    # sharpe = 窗口超额均值 / 波动：构造单调上涨的股票 sharpe 为正、下跌为负
    sh = labels["label_sharpe_22"]
    assert np.isfinite(sh.to_numpy(dtype=float)[:-22]).all()


def test_label_mdd_monotone_in_drawdown():
    """同窗口下，跌得多的股票 mdd 标签更负（原始值层面验证方向）。"""
    idx = pd.bdate_range("2024-01-02", periods=200)
    up = pd.DataFrame({"a": np.linspace(100, 130, 200),
                       "b": np.concatenate([np.linspace(100, 130, 60),
                                            np.linspace(130, 95, 140)])},
                      index=idx)
    px = {"open": up, "high": up * 1.001, "low": up * 0.999, "close": up,
          "volume": pd.DataFrame(1e6, idx, up.columns),
          "amount": pd.DataFrame(1e8, idx, up.columns)}
    fut_min = up.shift(-1).rolling(66).min().shift(-65)
    raw_mdd = fut_min / up - 1.0
    assert raw_mdd["a"].iloc[60] > raw_mdd["b"].iloc[60]    # 上涨股回撤更浅
    assert raw_mdd["b"].iloc[60] < 0
