"""残差 IC 探针的数学口径测试（2026-10-19）。

锁三件事：
1. `resid_rank` 与 `text_factor_probe.residual_panel` 逐位一致；
2. 自实现的 raw IC 与 `stats.ic.calc_ic_series` 逐位一致（掩码口径相同）；
3. 残差目标真的能把「主信号自身」剔掉：与主信号同向的因子 `both_ic≈0`，
   只与残差同向的因子 `both_ic>0`。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluation.residual_ic_probe import one_factor, resid_rank
from scripts.evaluation.text_factor_probe import residual_panel
from stats.ic import calc_ic_series


def _panel(rng, n_days=100, n_codes=40):
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    cols = [f"S{i:03d}" for i in range(n_codes)]
    return pd.DataFrame(rng.normal(size=(n_days, n_codes)), index=idx,
                        columns=cols)


def testresid_rank_matches_reference():
    rng = np.random.default_rng(7)
    sig = _panel(rng)
    main = _panel(rng)
    # 制造散点缺失，覆盖 v=False 的分支
    sig.iloc[0, 0] = np.nan
    main.iloc[1, 1] = np.nan
    a = resid_rank(sig.rank(axis=1, pct=True), main.rank(axis=1, pct=True))
    b = residual_panel(sig.rank(axis=1, pct=True), main.rank(axis=1, pct=True))
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12,
                                 atol=1e-12)


def test_raw_ic_matches_calc_ic_series():
    rng = np.random.default_rng(11)
    x = _panel(rng)
    fwd = _panel(rng)
    mask = pd.DataFrame(True, index=x.index, columns=x.columns)
    mask.iloc[0, :5] = False
    row = one_factor(x, fwd, mask, x.rank(axis=1, pct=True), x)
    assert row is not None
    ref = calc_ic_series(x, fwd, tradable=mask).mean()
    assert row["raw_ic"] == pytest.approx(ref, abs=1e-12)


def test_both_ic_isolates_main_from_residual():
    rng = np.random.default_rng(23)
    n_days, n_codes = 120, 60
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    cols = [f"S{i:03d}" for i in range(n_codes)]
    main = pd.DataFrame(rng.normal(size=(n_days, n_codes)), index=idx,
                        columns=cols)
    eps = pd.DataFrame(rng.normal(size=(n_days, n_codes)), index=idx,
                       columns=cols)
    fwd = 2.0 * main + eps
    # 只与主信号同向：残差化后应≈0
    x_main = main + 0.05 * _panel(rng, n_days, n_codes)
    # 只与残差同向：both_ic 应显著为正
    x_resid = eps + 0.05 * _panel(rng, n_days, n_codes)
    mask = pd.DataFrame(True, index=idx, columns=cols)
    mrank = main.rank(axis=1, pct=True)
    r_main = one_factor(x_main, fwd, mask, mrank, main)
    r_resid = one_factor(x_resid, fwd, mask, mrank, main)
    assert r_main is not None and r_resid is not None
    # 残差化后主信号成分应基本消失（rank 残差不是精确投影，留 ~7% 属正常）
    assert abs(r_main["both_ic"]) < 0.15 * abs(r_main["raw_ic"])
    assert r_resid["both_ic"] > 0.1
    assert r_resid["both_nw_t"] > 5
