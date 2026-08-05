"""因子挖掘摘要表（Alphalens 式）与 IR 排序的回归测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.mining import _benjamini_hochberg, evaluate_candidates, generate_candidates


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
