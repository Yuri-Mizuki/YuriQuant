"""因子挖掘摘要表（Alphalens 式）与 IR 排序的回归测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor.mining import (
    _benjamini_hochberg, evaluate_candidates, generate_candidates,
    rolling_evaluate_candidates,
)


def _mock_panel(n_days=150, n_codes=20, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    phi = 0.25
    rets = np.zeros((n_days, n_codes))
    for t in range(1, n_days):
        rets[t] = phi * rets[t - 1] + rng.normal(0, 0.02, n_codes)
    base = 10.0 + rng.uniform(0, 50, n_codes)
    close = pd.DataFrame(base * np.exp(np.cumsum(rets, axis=0)), idx, codes)
    volume = pd.DataFrame(rng.lognormal(12, 0.5, (n_days, n_codes)), idx, codes)
    panel = {"close": close, "volume": volume}
    returns = close.pct_change().shift(-1)
    return panel, returns


def _cands():
    return generate_candidates(features=["close", "volume"], windows=(5,), depth=1)


def test_evaluate_candidates_standard_columns():
    panel, returns = _mock_panel()
    df = evaluate_candidates(_cands(), panel, returns, detail_n=5)
    assert not df.empty
    for col in ("ic_mean", "ic_std", "ir", "ic_win_rate",
                "ic_decay5", "ic_decay10", "autocorr",
                "t_stat", "p_value", "significant", "n"):
        assert col in df.columns
    assert df["ic_decay5"].notna().sum() >= 1
    assert df["autocorr"].notna().sum() >= 1


def test_evaluate_candidates_sorted_by_abs_ir():
    panel, returns = _mock_panel()
    df = evaluate_candidates(_cands(), panel, returns, sort_by="ir", detail_n=5)
    ir_abs = df["ir"].abs().values
    assert np.all(np.diff(ir_abs) <= 1e-9)  # 非增
    df_t = evaluate_candidates(_cands(), panel, returns, sort_by="t", detail_n=5)
    assert df.iloc[0]["name"] == df_t.iloc[0]["name"]


def test_detail_columns_nan_beyond_detail_n():
    panel, returns = _mock_panel()
    df = evaluate_candidates(_cands(), panel, returns, detail_n=3)
    assert df["autocorr"].iloc[3:].isna().all()
    assert df["autocorr"].iloc[:3].notna().all()


def test_significant_based_on_nw_p_when_robust():
    """FDR 决策列：robust=True 时基于 p_value_nw（2026-08-03 修复）。

    旧实现 significant 基于 OLS p（自相关 IC 下虚高标伪显著）；
    修复后默认基于 NW 稳健 p，缺失时回退 OLS p。
    """
    panel, returns = _mock_panel()
    df = evaluate_candidates(_cands(), panel, returns, detail_n=3)
    expected = _benjamini_hochberg(df["p_value_nw"].fillna(df["p_value"]).values, 0.05)
    assert df["significant"].tolist() == expected.tolist()
    # robust=False 时回退 OLS p
    df2 = evaluate_candidates(_cands(), panel, returns, detail_n=3, robust=False)
    expected2 = _benjamini_hochberg(df2["p_value"].values, 0.05)
    assert df2["significant"].tolist() == expected2.tolist()
    # NW 更保守：自相关 IC 下显著的因子数不增
    assert df["significant"].sum() <= df2["significant"].sum()


def test_evaluate_candidates_parallel_matches_serial():
    """并行评估（2026-08-03）：n_jobs>1 与串行结果等价（IC/IR/t 全列一致）。"""
    panel, returns = _mock_panel()
    cands = _cands()
    df_serial = evaluate_candidates(cands, panel, returns, detail_n=5, n_jobs=1)
    df_par = evaluate_candidates(cands, panel, returns, detail_n=5, n_jobs=2)
    assert len(df_serial) == len(df_par)
    cols = ["ic_mean", "ic_std", "ir", "ic_win_rate", "t_stat", "p_value",
            "t_stat_nw", "p_value_nw", "n"]
    pd.testing.assert_frame_equal(
        df_serial[["name"] + cols].reset_index(drop=True),
        df_par[["name"] + cols].reset_index(drop=True),
        atol=1e-12,
    )


# ===========================================================================
# 滚动复核挖因子（walk-forward mining）
# ===========================================================================
def _strong_panel(n_days=220, n_codes=24, seed=3):
    """带真实持久动量信号的 mock：跨折应能识别出稳定正 IC 因子。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    # 固定强度向量 → 动量持续性
    strength = rng.normal(0, 1, n_codes)
    noise = rng.normal(0, 0.2, (n_days, n_codes))
    momentum = strength[None, :] + 0.2 * noise
    rets = momentum + rng.normal(0, 0.5, (n_days, n_codes))
    close = pd.DataFrame(10.0 * np.exp(np.cumsum(rets, axis=0)), idx, codes)
    volume = pd.DataFrame(rng.lognormal(12, 0.5, (n_days, n_codes)), idx, codes)
    panel = {"close": close, "volume": volume}
    returns = close.pct_change().shift(-1)
    return panel, returns


def test_rolling_returns_expected_columns():
    panel, returns = _strong_panel()
    cands = generate_candidates(features=["close", "volume"], windows=(5,), depth=1)
    df = rolling_evaluate_candidates(
        cands, panel, returns, n_splits=4, embargo_days=3,
        top_train=50, min_folds=2, robust=False,
    )
    assert not df.empty
    for col in ("name", "n_folds", "ic_mean", "ic_std", "ir_fold",
                "win_rate_fold", "consistent_frac", "significant_frac",
                "direction", "mean_train_ir", "rolling_significant"):
        assert col in df.columns
    # 至少 1 个候选进入过 test 复核
    assert df["n_folds"].max() >= 2
    # 排序主轴 |ir_fold| 非增
    ir = df["ir_fold"].abs().values
    assert np.all(np.diff(ir[~np.isnan(ir)]) <= 1e-9)


def test_rolling_strong_signal_flagged_significant():
    """强持久信号下，应至少有 1 个因子被标 rolling_significant。"""
    panel, returns = _strong_panel()
    cands = generate_candidates(features=["close", "volume"], windows=(5, 10), depth=1)
    df = rolling_evaluate_candidates(
        cands, panel, returns, n_splits=4, embargo_days=3,
        top_train=50, min_consistent_frac=0.5, min_sig_frac=0.2,
        min_folds=2, robust=False,
    )
    assert df["rolling_significant"].sum() >= 1


def test_rolling_no_cross_fold_never_flags():
    """若候选从未进入任何 test 复核（n_folds=0），rolling_significant 恒为 False。"""
    panel, returns = _mock_panel(n_days=60, n_codes=10, seed=7)
    cands = generate_candidates(features=["close", "volume"], windows=(5,), depth=1)
    # 折数多、test 段极小 → 多数候选过不了 min_obs，n_folds=0 的不应标 True
    df = rolling_evaluate_candidates(
        cands, panel, returns, n_splits=3, embargo_days=0,
        top_train=5, min_folds=2, robust=False,
    )
    assert not ((df["n_folds"] == 0) & df["rolling_significant"]).any()


def test_rolling_embargo_no_train_future():
    """滚动切分不可泄漏：训练段 max 必须 < test 段 min。"""
    panel, returns = _mock_panel(n_days=120, n_codes=15, seed=11)
    generate_candidates(features=["close"], windows=(5,), depth=1)
    from factor.cv import forward_folds
    folds = forward_folds(returns.index, 4, embargo_days=5)
    for f in folds:
        assert f.train_days.max() < f.test_days.min()
        assert len(set(f.train_days) & set(f.test_days)) == 0
