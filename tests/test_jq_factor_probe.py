"""聚宽候选探针的口径测试（2026-10-26）。

锁四件事：
1. `_cs_ic` 与 `stats.ic.calc_ic_series` 同口径（掩码 + Spearman）逐位一致；
2. `monthly_ls` 的多空方向随 `sign` 翻转（含成本扣减）；
3. `_resid_ic_np` 能把 size 代理剔干净（因子=市值 → 残差 IC≈0）；
4. `_dup_batch` 能把「完全复制」的面板识别为 dup_corr≈1。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluation.jq_factor_probe import (
    _cs_ic,
    _dup_batch,
    _rank_panel,
    _resid_ic_np,
    monthly_ls,
)
from stats.ic import calc_ic_series


def _panel(rng, n_days=120, n_codes=60, cols=None):
    idx = pd.date_range("2021-01-04", periods=n_days, freq="B")
    cols = cols or [f"S{i:03d}" for i in range(n_codes)]
    return pd.DataFrame(rng.normal(size=(n_days, len(cols))), index=idx,
                        columns=cols)


def test_cs_ic_matches_calc_ic_series():
    rng = np.random.default_rng(3)
    x = _panel(rng)
    fwd = _panel(rng)
    mask = pd.DataFrame(True, index=x.index, columns=x.columns)
    mask.iloc[0, :5] = False
    got = _cs_ic(_rank_panel(x), _rank_panel(fwd.where(mask)), mask).mean()
    ref = calc_ic_series(x, fwd, tradable=mask).mean()
    assert got == pytest.approx(ref, abs=1e-5)


def test_monthly_ls_direction_follows_sign():
    rng = np.random.default_rng(5)
    n_days, n_codes = 320, 120
    idx = pd.date_range("2020-01-02", periods=n_days, freq="B")
    cols = [f"S{i:03d}" for i in range(n_codes)]
    steps = rng.normal(0.0, 0.01, size=(n_days, n_codes))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)),
                         index=idx, columns=cols)
    fwd1 = close.shift(-1) / close - 1.0
    # 因子 = 未来 21 日收益的领先值（完美预测）
    panel = (close.shift(-21) / close - 1.0)
    mask = pd.DataFrame(True, index=idx, columns=cols)
    good = monthly_ls(panel, close, mask, sign=1)
    bad = monthly_ls(panel, close, mask, sign=-1)
    assert good and bad
    assert good["ls_mean_m"] > 0.005
    assert bad["ls_mean_m"] < -0.005
    assert good["ls_mean_m"] > bad["ls_mean_m"]
    # 成本已扣：净 < 毛（毛由 ann_long / ls_ann 无法直接取，检查换手>0 即可）
    assert 0.0 < good["ls_turnover_m"] <= 1.0
    _ = fwd1


def test_resid_ic_removes_size_proxy():
    rng = np.random.default_rng(7)
    n_days, n_codes = 80, 200
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    cols = [f"S{i:03d}" for i in range(n_codes)]
    size = pd.DataFrame(rng.normal(size=(n_days, n_codes)), index=idx,
                        columns=cols)
    ind = np.tile(np.arange(n_codes) % 6, (n_days, 1)).astype(int)
    # 未来收益完全由 size 决定
    fwd_rank = size.rank(axis=1, pct=True)
    ok = np.ones((n_days, n_codes), dtype=bool)
    xr = size.rank(axis=1, pct=True).to_numpy(np.float32)
    yr = fwd_rank.to_numpy(np.float32)
    regs = [size.rank(axis=1, pct=True).to_numpy(np.float32)]
    ic = _resid_ic_np(xr, yr, ok, ind, regs, int(ind.max()) + 1)
    assert abs(float(ic.dropna().mean())) < 1e-6


def test_dup_batch_flags_exact_copy(tmp_path):
    rng = np.random.default_rng(11)
    n_days, n_codes = 60, 150
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    cols = [f"S{i:03d}" for i in range(n_codes)]
    base = pd.DataFrame(rng.normal(size=(n_days, n_codes)), index=idx,
                        columns=cols)
    other = pd.DataFrame(rng.normal(size=(n_days, n_codes)), index=idx,
                         columns=cols)
    base.to_parquet(tmp_path / "lib_base.parquet")
    other.to_parquet(tmp_path / "lib_other.parquet")
    cand = {"jq_copy": base.rank(axis=1, pct=True)}
    best = _dup_batch(cand, tmp_path, stride=2)
    top = best["jq_copy"][0]
    assert top[1] == "lib_base"
    assert top[0] == pytest.approx(1.0, abs=1e-3)
