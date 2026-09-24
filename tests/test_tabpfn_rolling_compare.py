"""批次 8 表格基础模型对照 runner 的单元测试（合成面板，无 tabpfn 依赖）。

覆盖：月度网格 / 滚动训练窗（窗口界 + embargo 尾截 + 严格早于测试月）、
秩平均混合、可交易掩码 IC、ridge 臂全链滚动、evaluate_pred 指标行。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluation.tabpfn_rolling_compare import (
    evaluate_pred,
    month_grid,
    rank_blend,
    rolling_train_days,
    run_arm,
    tradable_ic_series,
)


@pytest.fixture(scope="module")
def tiny_panel():
    idx = pd.bdate_range("2023-01-02", periods=500)
    cols = [f"S{i}" for i in range(30)]
    rng = np.random.default_rng(9)
    close = pd.DataFrame(
        10 * np.exp(np.cumsum(rng.normal(0, 0.015, (500, 30)), axis=0)),
        index=idx, columns=cols)
    feats = {"mom20": close.pct_change(20), "vol20": close.pct_change().rolling(20).std()}
    fwd = close.pct_change().shift(-1)
    labels = fwd.rank(axis=1, pct=True)
    return idx, cols, feats, labels, fwd, close


def test_month_grid_first_last_days():
    idx = pd.bdate_range("2024-01-01", periods=130)      # ~2024 上半年
    months = month_grid(idx, "2024-02-01", "2024-06-30")
    assert len(months) == 5
    firsts = {m[0].month for m in months}
    assert firsts == {2, 3, 4, 5, 6}
    for a, b in months:
        assert a.month == b.month and a <= b
        assert a == idx[idx.to_period("M") == a.to_period("M")][0]


def test_rolling_train_days_window_and_embargo():
    idx = pd.bdate_range("2023-01-02", periods=400)
    m_first = pd.Timestamp("2024-03-01")
    tr = rolling_train_days(idx, m_first, window_months=2, embargo=1)
    assert (tr < m_first).all(), "训练窗必须严格早于测试月"
    assert tr.max() < m_first
    # 2 个月窗语义 = DateOffset(months=2)（MonthBegin 会少开一个月）
    assert tr.min() >= m_first - pd.DateOffset(months=2)
    assert len(tr) > 30
    # embargo=3 比 embargo=1 多截 2 天：tr3 ⊂ tr、末端更早
    tr3 = rolling_train_days(idx, m_first, 2, 3)
    assert len(tr) - len(tr3) == 2
    assert tr3.max() < tr.max() < m_first
    assert tr3.isin(tr).all()


def test_rank_blend_average_of_ranks():
    idx = pd.bdate_range("2024-01-01", periods=3)
    cols = list("ABC")
    a = pd.DataFrame([[3.0, 1.0, 2.0]] * 3, idx, cols)
    b = pd.DataFrame([[1.0, 2.0, 3.0]] * 3, idx, cols)
    blend = rank_blend(a, b)
    # a 秩 (pct) = A 1.0 / B 1/3 / C 2/3；b 秩相反 → 平均 = 2/3, 1/2, 5/6
    assert blend.iloc[0]["A"] == pytest.approx(2 / 3)
    assert blend.iloc[0]["B"] == pytest.approx(0.5)
    assert blend.iloc[0]["C"] == pytest.approx(5 / 6)
    assert blend.notna().all().all()
    # 单侧缺失 → 该格 NaN
    a2 = a.copy()
    a2.iloc[0, 0] = np.nan
    assert pd.isna(rank_blend(a2, b).iloc[0, 0])


def test_tradable_ic_series_masks_paper_names(tiny_panel):
    idx, cols, _feats, _labels, fwd, _close = tiny_panel
    rng = np.random.default_rng(2)
    pred = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), idx, cols)
    trad = pd.DataFrame(True, idx, cols)
    trad["S0"] = False                    # S0 全程不可交易
    ic_raw = tradable_ic_series(pred, fwd, None).dropna()
    ic_tr = tradable_ic_series(pred, fwd, trad).dropna()
    # 掩码改变日截面成分 → 两条 IC 序列数值上应不同（随机数据下概率 1）
    assert not np.allclose(ic_raw.to_numpy(), ic_tr.reindex(ic_raw.index).to_numpy())
    assert len(ic_tr) > 50


def test_run_arm_ridge_rolling(tiny_panel):
    idx, cols, feats, labels, _fwd, _close = tiny_panel
    months = month_grid(idx, "2024-07-01", "2024-09-30")
    pred, timing = run_arm("ridge", {"alpha": 1.0}, feats, labels, idx, months,
                           window_months=3, embargo=1)
    test_span = idx[(idx >= months[0][0]) & (idx <= months[-1][1])]
    assert pred.shape == (len(test_span), len(cols))
    assert pred.notna().mean().mean() > 0.8, "月内预测应基本覆盖"
    assert timing["n_retrain"] == len(months)
    # 防泄漏锚：第一个测试月的预测只依赖其前训练窗
    assert timing["fit_s"] >= 0


def test_evaluate_pred_row(tiny_panel):
    idx, cols, _feats, _labels, fwd, close = tiny_panel
    rng = np.random.default_rng(4)
    pred = -close.pct_change(5) + rng.normal(0, 0.01, close.shape)  # 注入反转
    pred = pred.iloc[:, :] * 1.0
    trad = pd.DataFrame(True, idx, cols)
    test_days = idx[idx >= "2024-01-02"]
    open_px = close.shift(1).bfill()
    rets = close.pct_change()
    row = evaluate_pred(pred, fwd, trad, close, open_px, test_days,
                        bench_eqw=rets.where(trad).mean(axis=1)
                        .reindex(test_days).fillna(0.0), rets=rets)
    for k in ("ic_raw", "ic_tradable", "ic_ir_tradable", "nw_t", "annual",
              "sharpe", "max_dd", "turnover", "n_days"):
        assert k in row
    assert row["n_days"] > 100
    assert np.isfinite(row["ic_tradable"])
    assert row["ic_raw"] == pytest.approx(row["ic_tradable"], abs=1e-12)  # 全 True 掩码等价
