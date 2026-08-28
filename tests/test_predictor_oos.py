"""model.predictor.fit_predict_oos 走统一 CV 调度器的接线冒烟测试。

验证：默认 forward（expanding 前推）与 purged 切换都能产出 OOS 面板，
且 forward 首段训练区为 NaN（无未来预测）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.predictor import RidgePredictor, fit_predict_oos


@pytest.fixture
def panel():
    """80 交易日 × 8 代码的小面板（带可学习信号）。"""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-02", periods=80, freq="B")
    codes = [f"c{i:02d}" for i in range(8)]
    n = len(dates)
    t = np.arange(n)
    # 特征1：带时间趋势（ridge 可拟合）、特征2：纯噪声
    f1 = pd.DataFrame(np.tile(t[:, None], (1, 8)) + rng.normal(
        0, 1, (n, 8)), index=dates, columns=codes)
    f2 = pd.DataFrame(rng.normal(0, 1, (n, 8)), index=dates, columns=codes)
    # 标签：滞后后一期特征 + 噪声（保证可学习且无泄漏）
    lab = pd.DataFrame(np.roll(f1.values, -1, axis=0), index=dates, columns=codes)
    lab.iloc[-1] = np.nan
    return {"x1": f1, "x2": f2}, lab


def test_forward_default_produces_oos(panel):
    feats, labels = panel
    out = fit_predict_oos(RidgePredictor, feats, labels,
                          n_splits=3, embargo_days=2, min_train_days=10)
    # 首段训练区应为 NaN，其余有预测
    assert out.isna().any().any()
    assert out.notna().any().any()
    # 有 NaN 的日期应全部在面板前半段
    nan_days = out.index[out.isna().all(axis=1)]
    has_pred_days = out.index[out.notna().all(axis=1)]
    assert len(nan_days) > 0 and len(has_pred_days) > 0
    assert nan_days.max() < has_pred_days.min(), \
        "forward 首段（训练区）应无预测，且早于后续有预测的测试段"


def test_purged_switches_and_covers(panel):
    feats, labels = panel
    out = fit_predict_oos(RidgePredictor, feats, labels,
                          n_splits=3, embargo_days=2, min_train_days=10,
                          cv_method="purged")
    assert out.notna().any().any()
    assert len(out[out.notna().all(axis=1)]) > 0


def test_forward_and_purged_both_valid(panel):
    """两种切分都能跑通（不抛异常），且输出面板同网格。"""
    feats, labels = panel
    a = fit_predict_oos(RidgePredictor, feats, labels, n_splits=3,
                        embargo_days=2, min_train_days=10, cv_method="forward")
    b = fit_predict_oos(RidgePredictor, feats, labels, n_splits=3,
                        embargo_days=2, min_train_days=10, cv_method="purged")
    assert a.shape == b.shape
    assert a.index.equals(b.index)
    assert a.columns.equals(b.columns)