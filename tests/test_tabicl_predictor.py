"""TabICLPredictor 单测（tabicl 未安装时自动跳过）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

tabicl = pytest.importorskip("tabicl", reason="tabicl 未安装")

from model.predictor import PREDICTORS, TabICLPredictor  # noqa: E402


def _panels(days: int = 40, codes: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    cols = [f"c{i:03d}" for i in range(codes)]
    feats = {
        "f1": pd.DataFrame(rng.normal(size=(days, codes)), index=dates, columns=cols),
        "f2": pd.DataFrame(rng.normal(size=(days, codes)), index=dates, columns=cols),
    }
    # y 依赖 f1：模型应能恢复该结构
    y = pd.DataFrame(
        0.8 * feats["f1"].values + rng.normal(scale=0.2, size=(days, codes)),
        index=dates, columns=cols)
    return feats, y, dates, cols


def test_registered_in_predictors():
    assert PREDICTORS["tabicl"] is TabICLPredictor


def test_fit_predict_recovers_signal():
    feats, y, dates, cols = _panels()
    m = TabICLPredictor(max_context_samples=600, n_estimators=1, chunk_size=400)
    m.fit(feats, y)
    pred = m.predict(feats)
    assert pred.shape == (len(dates), len(cols))
    # 预测与真实标签的日截面相关应为正（信号可恢复）
    from research.factor_analysis import calc_ic_series
    ic = calc_ic_series(pred, y).dropna()
    assert float(ic.mean()) > 0.3
    assert m.n_samples_ <= 600


def test_context_truncation_keeps_recent():
    """context 截断应保留最近样本（ICL 语义：时间上贴近 query）。"""
    feats, y, _, _ = _panels(days=60, codes=30)
    m = TabICLPredictor(max_context_samples=500, n_estimators=1)
    m.fit(feats, y)
    # 60 日 × 30 股 = 1800 有效样本 > 500 → n_samples_ = 500
    assert m.n_samples_ == 500


def test_nan_handling():
    """fit 丢 any-NaN 行；predict 的 NaN 特征填 0（截面均值）不产出 NaN。"""
    feats, y, dates, cols = _panels(days=35, codes=30)
    feats["f1"].iloc[:5, :10] = np.nan          # 前置 NaN（含前置期）
    y.iloc[3, 5] = np.nan                        # 标签单点 NaN
    m = TabICLPredictor(max_context_samples=800, n_estimators=1)
    m.fit(feats, y)
    feats_q = {k: v.copy() for k, v in feats.items()}
    feats_q["f2"].iloc[-1, 0] = np.nan           # query 含 NaN
    pred = m.predict(feats_q)
    assert not pred.iloc[-1].isna().any()
