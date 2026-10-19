"""A1/A2 增量方向新探针的口径测试（2026-10-19）。

覆盖：
1. `residual_overlay_probe.penalty_mean`：多因子均值语义（部分缺失取可得
   均值、全缺 NaN）、安慰剂时间平移保留截面结构；
2. `residual_overlay_probe.walkforward_sets`：逐年重选只用历史年份列
   （无选择前视）、start_year/min_hist 门槛；
3. `select_horizon_features.quality_window`：fwd 不穿透选择窗（防前视）；
4. `horizon_cross_eval.ic_stats`：NW t 与 `stats.robust_stats.nw_tstat` 同源、
   空序列安全。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluation.horizon_cross_eval import ic_stats
from scripts.evaluation.residual_overlay_probe import (
    penalty_mean,
    walkforward_sets,
)
from scripts.evaluation.select_horizon_features import quality_window
from stats.robust_stats import nw_tstat


def _panel(rng, n_days=60, n_codes=20, seed_offset=0):
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    cols = [f"S{i:03d}" for i in range(n_codes)]
    return pd.DataFrame(rng.normal(size=(n_days, n_codes)) + seed_offset,
                        index=idx, columns=cols)


class TestPenaltyMean:
    def _panels_dir(self, tmp_path, panels):
        for name, x in panels.items():
            x.to_parquet(tmp_path / f"{name}.parquet")
        return tmp_path

    def test_mean_of_ranked_factors(self, tmp_path):
        rng = np.random.default_rng(3)
        a, b = _panel(rng), _panel(rng)
        pdir = self._panels_dir(tmp_path, {"fa": a, "fb": b})
        pen = penalty_mean(["fa", "fb"], pdir, a.index, a.columns)
        expected = ((a.rank(axis=1, pct=True) + b.rank(axis=1, pct=True)) / 2)
        pd.testing.assert_frame_equal(pen, expected.astype(np.float32),
                                      check_exact=False, rtol=1e-6)

    def test_partial_missing_uses_available(self, tmp_path):
        rng = np.random.default_rng(5)
        a = _panel(rng)
        b = a.copy()
        b.iloc[:, :5] = np.nan          # fb 前 5 列缺失 → 只用 fa
        pdir = self._panels_dir(tmp_path, {"fa": a, "fb": b})
        pen = penalty_mean(["fa", "fb"], pdir, a.index, a.columns)
        ar = a.rank(axis=1, pct=True)
        pd.testing.assert_frame_equal(
            pen.iloc[:, :5], ar.iloc[:, :5].astype(np.float32),
            check_exact=False, rtol=1e-6)
        assert pen.iloc[:, 5:].notna().all().all()

    def test_all_missing_is_nan(self, tmp_path):
        rng = np.random.default_rng(7)
        a = _panel(rng)
        b = a.copy()
        b[:] = np.nan                   # fb 全缺 → penalty 只剩 fa
        pdir = self._panels_dir(tmp_path, {"fa": a, "fb": b})
        pen = penalty_mean(["fa", "fb"], pdir, a.index, a.columns)
        expected = a.rank(axis=1, pct=True).astype(np.float32)
        assert pen.notna().all().all()
        pd.testing.assert_frame_equal(pen, expected, check_exact=False,
                                      rtol=1e-6)

    def test_placebo_shift_preserves_daily_cross_section(self, tmp_path):
        """时间平移后：第 t 日的截面 = 原面板第 (t−shift) 日的截面（循环）。"""
        rng = np.random.default_rng(11)
        a = _panel(rng)
        pdir = self._panels_dir(tmp_path, {"fa": a})
        shift = 10
        pen = penalty_mean(["fa"], pdir, a.index, a.columns, shift=shift)
        rolled = pd.DataFrame(np.roll(a.to_numpy(), shift, axis=0),
                              index=a.index, columns=a.columns)
        expected = rolled.rank(axis=1, pct=True).astype(np.float32)
        pd.testing.assert_frame_equal(pen, expected, check_exact=False,
                                      rtol=1e-6)


class TestQualityWindow:
    def test_fwd_does_not_cross_year_end(self):
        idx = pd.bdate_range("2020-01-02", "2023-12-29")
        h, qw = 60, 500
        q = quality_window(idx, 2023, h, qw)
        pos = {d: i for i, d in enumerate(idx)}
        # 窗口最后一天的 fwd 终点（+h 交易日）仍在 2023 年内
        assert idx[pos[q[-1]] + h] <= pd.Timestamp("2023-12-31")
        assert len(q) == qw
        assert (q <= pd.Timestamp("2023-12-31")).all()

    def test_insufficient_days_raises(self):
        idx = pd.bdate_range("2023-11-01", "2023-12-29")
        with pytest.raises(ValueError):
            quality_window(idx, 2023, 60, 500)


class TestCrossEvalStats:
    def test_nw_t_matches_reference(self):
        rng = np.random.default_rng(13)
        s = pd.Series(rng.normal(0.01, 0.05, 300))
        st = ic_stats(s)
        t_ref, _se, _lag = nw_tstat(s.to_numpy(dtype=float))
        assert st["ic_mean"] == pytest.approx(float(s.mean()))
        assert st["n"] == 300
        assert st["ic_nw_t"] == pytest.approx(float(t_ref))

    def test_degenerate_series(self):
        st = ic_stats(pd.Series(dtype=float))
        assert st["n"] == 0
        assert np.isnan(st["ic_mean"])


class TestWalkforwardSets:
    @staticmethod
    def _csv(tmp_path, rows):
        fp = tmp_path / "res.csv"
        pd.DataFrame(rows).to_csv(fp, index=False)
        return fp

    def test_uses_only_prior_years(self, tmp_path):
        # A：2018 极负、2019 转正；B：2018 正、2019 极负
        # 2019 年初选因子只看 ic_2018 → A；2020 年初看 2018+2019 → B
        rows = [
            {"name": "A", "both_nw_t": -1.0, "cov": 0.6,
             "ic_2018": -0.05, "ic_2019": +0.05, "ic_2020": 0.0},
            {"name": "B", "both_nw_t": -1.0, "cov": 0.6,
             "ic_2018": +0.01, "ic_2019": -0.06, "ic_2020": 0.0},
        ]
        sets = walkforward_sets(self._csv(tmp_path, rows), [1],
                                start_year=2019, min_hist=1)
        assert sets[2019][1] == ["A"]
        assert sets[2020][1] == ["B"]

    def test_negative_pool_only_and_gates(self, tmp_path):
        rows = [
            {"name": "pos", "both_nw_t": +5.0, "cov": 0.9,
             "ic_2018": -0.05, "ic_2019": -0.05, "ic_2020": 0.0},
            {"name": "lowcov", "both_nw_t": -5.0, "cov": 0.1,
             "ic_2018": -0.05, "ic_2019": -0.05, "ic_2020": 0.0},
            {"name": "ok", "both_nw_t": -2.0, "cov": 0.5,
             "ic_2018": -0.03, "ic_2019": -0.03, "ic_2020": 0.0},
        ]
        sets = walkforward_sets(self._csv(tmp_path, rows), [1, 2],
                                start_year=2020, min_hist=2)
        assert sets[2020][1] == ["ok"] and sets[2020][2] == ["ok"]
        # min_hist=2 时 2019（只有 2018 一年历史）无段
        assert 2019 not in sets
