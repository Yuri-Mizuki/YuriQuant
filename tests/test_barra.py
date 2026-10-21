"""Barra CNE6（简化版）截面回归链路单元测试（合成数据精确答案）。

覆盖 :mod:`research.barra` 四组核心件：
1. :func:`winsorize_cs` / :func:`standardize_exposures` —— 截面预处理；
2. :func:`cross_sectional_regression` —— WLS 截面回归（精确答案：构造已知
   因子收益的数据，验证回归恢复精度；含行业哑变量共线、cap=0 剔除、样本下限）;
3. :func:`factor_covariance` / :func:`specific_risk` / :func:`assemble_sigma`
   —— 时序层与 Σ 合成（半正定性、对角块、X·F·X' 结构）；
4. :func:`run_factor_returns` —— 端到端小样本主循环（收益 shift 口径）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.barra import (
    assemble_sigma,
    cross_sectional_regression,
    factor_covariance,
    specific_risk,
    standardize_exposures,
    winsorize_cs,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. 截面预处理
# ---------------------------------------------------------------------------
class TestWinsorize:
    def test_clips_extremes(self):
        s = pd.Series(RNG.normal(0, 1.0, 400))
        out = winsorize_cs(s)
        med, mad = s.median(), (s - s.median()).abs().median()
        # 去极值后应落在 med ± 5·MAD 内（原分布 400 点大概率越界）
        assert out.max() <= med + 5 * mad + 1e-12
        assert out.min() >= med - 5 * mad - 1e-12

    def test_mad_zero_returns_original(self):
        # 过半同值 → MAD=0 → 不做 clip（与 neutralize 章节同陷阱：别压常数）
        s = pd.Series([2.0] * 40 + [5.0, -3.0])
        out = winsorize_cs(s)
        assert out.equals(s)

    def test_constant_series_unchanged(self):
        s = pd.Series([3.0] * 100)
        assert winsorize_cs(s).equals(s)


class TestStandardize:
    def test_capweighted_mean_eq_sd(self):
        # 市值加权均值：构造两组对称点 + 不对称权重，验证 mu ≠ 等权均值
        idx = [f"c{i}" for i in range(400)]
        s = pd.Series(np.r_[np.full(200, -1.0), np.full(200, 1.0)], index=idx)
        cap = pd.Series(np.r_[np.full(200, 1.0), np.full(200, 9.0)], index=idx)
        out = standardize_exposures(s, cap, "beta")
        # 加权均值 = (200*-1*1 + 200*1*9)/(200*1+200*9) = 0.8；sd = s.std()（ddof=1）
        mu_w = (200 * -1 * 1 + 200 * 1 * 9) / (200 * 1 + 200 * 9)
        sd = s.std()
        assert out.iloc[0] == pytest.approx((-1.0 - mu_w) / sd)
        assert out.iloc[-1] == pytest.approx((1.0 - mu_w) / sd)
        # 加权均值 0.8 ≠ 等权均值 0 → 低组标准化后应比 -1 更负
        assert out.iloc[0] < -1.0

    def test_cap_native_uses_equal_weight(self):
        # lncap 是市值函数 → 均值用等权，防自反
        idx = [f"c{i}" for i in range(400)]
        s = pd.Series(np.r_[np.full(200, -1.0), np.full(200, 1.0)], index=idx)
        cap = pd.Series(np.r_[np.full(200, 1.0), np.full(200, 9.0)], index=idx)
        out = standardize_exposures(s, cap, "lncap")
        sd = s.std()
        assert out.iloc[0] == pytest.approx(-1.0 / sd)   # 等权均值 = 0
        assert out.iloc[-1] == pytest.approx(1.0 / sd)

    def test_small_sample_returns_nan(self):
        s = pd.Series(np.arange(20, dtype=float))
        cap = pd.Series(np.ones(20))
        out = standardize_exposures(s, cap, "beta")
        assert out.isna().all()

    def test_mad_winsorize_applied(self):
        # 极端值应被 clip 后标准化（而非直接溢出）
        idx = [f"c{i}" for i in range(400)]
        s = pd.Series(RNG.normal(0, 1, 400), index=idx)
        s.iloc[0] = 1000.0
        cap = pd.Series(np.ones(400), index=idx)
        out = standardize_exposures(s, cap, "beta")
        assert out.abs().max() < 10.0


# ---------------------------------------------------------------------------
# 2. WLS 截面回归（精确答案）
# ---------------------------------------------------------------------------
def _make_known_answer(n: int = 500, seed: int = 7):
    """构造 r = f_ind·D + f_sty·sty + u 的精确答案数据。

    行业 3 个，风格 2 个；权重 = sqrt(cap)。噪声很小 → 回归应高精度恢复。
    """
    rng = np.random.default_rng(seed)
    codes = [f"{600000 + i:06d}" for i in range(n)]
    ind = pd.Series(rng.choice(["A", "B", "C"], n), index=codes)
    cap = pd.Series(rng.uniform(1e9, 1e10, n), index=codes)
    sty1 = pd.Series(rng.normal(0, 1, n), index=codes)
    sty2 = pd.Series(rng.normal(0, 1, n), index=codes)

    true_f_ind = pd.Series({"A": 0.002, "B": -0.001, "C": 0.004})
    true_f_sty = pd.Series({"sty1": 0.0015, "sty2": -0.0012})
    u = pd.Series(rng.normal(0, 5e-4, n), index=codes)

    r = pd.Series(0.0, index=codes)
    for i in codes:
        r[i] += true_f_ind[ind[i]]
    r += true_f_sty["sty1"] * sty1 + true_f_sty["sty2"] * sty2 + u
    return r, {"sty1": sty1, "sty2": sty2}, ind, cap, true_f_ind, true_f_sty


class TestCrossSectionalRegression:
    def test_recovers_known_answers(self):
        r, stys, ind, cap, f_ind, f_sty = _make_known_answer()
        res = cross_sectional_regression(r, stys, ind, cap)
        assert res, "回归不应为空"
        fr = res["factor_returns"]
        assert res["n"] == 500
        assert res["r2"] > 0.9
        for name, val in f_ind.items():
            assert fr[f"ind:{name}"] == pytest.approx(val, abs=1e-4)
        for name, val in f_sty.items():
            assert fr[f"sty:{name}"] == pytest.approx(val, abs=2e-4)

    def test_wls_weighting_pulls_toward_large_caps(self):
        # 大市值个股的特异性收益被右移 → WLS 因子收益应比等权 OLS 更接近大票答案
        r, stys, ind, cap, _, _ = _make_known_answer()
        big = cap.idxmax()
        r[big] += 0.05                      # 大票特异性 +5%
        res = cross_sectional_regression(r, stys, ind, cap)
        fr_wls = res["factor_returns"]
        # 等权 OLS 同数据对照（权重全 1）
        res_ols = cross_sectional_regression(
            r, stys, ind, pd.Series(1.0, index=cap.index))
        fr_ols = res_ols["factor_returns"]
        ind_big = f"ind:{ind[big]}"
        # 大票所在行业因子：WLS 应比等权 OLS 更看重该大票 → 更高
        assert fr_wls[ind_big] > fr_ols[ind_big]

    def test_zero_cap_excluded(self):
        r, stys, ind, cap, _, _ = _make_known_answer()
        cap_zero = cap.copy()
        cap_zero.iloc[0] = 0.0
        res = cross_sectional_regression(r, stys, ind, cap_zero)
        assert res and res["n"] == 499

    def test_sample_floor(self):
        r, stys, ind, cap, _, _ = _make_known_answer(n=80)
        assert cross_sectional_regression(r, stys, ind, cap) == {}

    def test_missing_exposure_dropped(self):
        r, stys, ind, cap, _, _ = _make_known_answer()
        stys_nan = {k: v.copy() for k, v in stys.items()}
        stys_nan["sty1"].iloc[0] = np.nan
        res = cross_sectional_regression(r, stys_nan, ind, cap)
        assert res and res["n"] == 499

    def test_r2_and_tstats_present(self):
        r, stys, ind, cap, _, _ = _make_known_answer(seed=11)
        res = cross_sectional_regression(r, stys, ind, cap)
        assert 0.0 < res["r2"] <= 1.0
        assert res["adj_r2"] <= res["r2"] + 1e-12
        t = res["t_stats"]
        # 真实因子收益显著非零
        assert abs(t["sty:sty1"]) > 2.0
        assert abs(t["sty:sty2"]) > 2.0


# ---------------------------------------------------------------------------
# 3. 时序层与 Σ
# ---------------------------------------------------------------------------
class TestFactorCovariance:
    def test_psd_and_symmetric(self):
        rng = np.random.default_rng(3)
        fr = pd.DataFrame(rng.normal(0, 0.01, (300, 4)),
                          columns=["a", "b", "c", "d"])
        cov = factor_covariance(fr, halflife=60)
        assert np.allclose(cov, cov.T)
        vals = np.linalg.eigvalsh(cov.values)
        assert vals.min() > -1e-12

    def test_nw_inflation_positive_autocorr(self):
        # 正自相关的因子序列 → NW 期限调整后方差应 ≥ 原始 EWMA 方差
        rng = np.random.default_rng(5)
        e = rng.normal(0, 0.01, 400)
        x = np.zeros(400)
        for t in range(1, 400):
            x[t] = 0.5 * x[t - 1] + e[t]
        fr = pd.DataFrame({"a": x})
        cov = factor_covariance(fr, halflife=200, max_lag=3)
        var_raw = fr["a"].var()
        assert cov.loc["a", "a"] >= var_raw * 0.8

    def test_nan_columns_tolerated(self):
        fr = pd.DataFrame({"a": np.r_[RNG.normal(0, .01, 200), [np.nan] * 50],
                           "b": RNG.normal(0, .01, 250)})
        cov = factor_covariance(fr, halflife=100)
        assert cov.shape == (2, 2)


class TestSigma:
    def test_assemble_structure(self):
        # 3 股 2 行业 1 风格：手工算 X·F·X' + D
        codes = ["c1", "c2", "c3"]
        exposure_row = pd.Series({"c1": 0.5, "c2": -0.5, "c3": 1.0})
        industry_row = pd.Series({"c1": "A", "c2": "A", "c3": "B"})
        f_names = ["ind:A", "ind:B", "sty:vl"]
        f_cov = pd.DataFrame(np.diag([4e-5, 9e-5, 1e-4]), index=f_names,
                             columns=f_names)
        spec_var = pd.Series({"c1": 1e-4, "c2": 2e-4, "c3": 3e-4})

        sigma = assemble_sigma(exposure_row, industry_row, f_cov, spec_var)
        assert list(sigma.index) == codes
        # c1/c2 同行业同风格无关 → 协方差 = F_AA + x1*x2*F_vl
        expect_12 = 4e-5 + 0.5 * (-0.5) * 1e-4
        assert sigma.loc["c1", "c2"] == pytest.approx(expect_12)
        # 对角 = 行业方差 + 风格贡献 + 特异性
        expect_11 = 4e-5 + 0.25 * 1e-4 + 1e-4
        assert sigma.loc["c1", "c1"] == pytest.approx(expect_11)

    def test_diag_dominant_with_spec(self):
        codes = [f"c{i}" for i in range(50)]
        exposure_row = pd.Series(RNG.normal(0, 1, 50), index=codes)
        industry_row = pd.Series(RNG.choice(["A", "B"], 50), index=codes)
        f_names = ["ind:A", "ind:B", "sty:vl"]
        f_cov = pd.DataFrame(np.eye(3) * 1e-5, index=f_names, columns=f_names)
        spec_var = pd.Series(np.full(50, 1e-3), index=codes)
        sigma = assemble_sigma(exposure_row, industry_row, f_cov, spec_var)
        assert (np.diag(sigma) > 0).all()
        # PSD
        vals = np.linalg.eigvalsh(sigma.values)
        assert vals.min() > -1e-15

    def test_specific_risk_shape_and_positivity(self):
        rng = np.random.default_rng(9)
        resid = pd.DataFrame(rng.normal(0, 0.02, (300, 5)),
                             columns=[f"c{i}" for i in range(5)])
        sv = specific_risk(resid, halflife=60)
        assert sv.shape == resid.shape
        # min_periods=30：暖机后全为非负有限值
        warm = sv.iloc[30:]
        assert (warm >= 0).all().all()
        assert not warm.isna().any().any()
        # 逐股收敛到近似方差量级（|u| ~ 0.02 → u² ~ 4e-4）
        assert (warm.abs() < 1e-2).all().all()


# ---------------------------------------------------------------------------
# 4. 端到端主循环（小样本）
# ---------------------------------------------------------------------------
class TestRunFactorReturns:
    def test_end_to_end_shift_alignment(self, monkeypatch, tmp_path):
        """收益口径 = T 信号 → T→T+1 收益：改 T 日暴露，T 日因子收益应变化。

        构造 30 日 × 150 股（>回归 n≥100 下限），第 20 日在 close 里注入
        「高 vl 股票次日大涨」的脉冲 → 第 20 日 sty:vl 因子收益应显著为正。
        """
        import research.barra as rb

        rng = np.random.default_rng(17)
        dates = pd.bdate_range("2024-01-01", periods=30)
        codes = [f"{600000 + i:06d}" for i in range(150)]
        close = pd.DataFrame(
            10 * np.exp(np.cumsum(rng.normal(0, 0.02, (30, 150)), axis=0)),
            index=dates, columns=codes)
        lncap = pd.DataFrame(rng.uniform(20, 24, (30, 150)), index=dates,
                             columns=codes)
        vl = pd.DataFrame(rng.normal(0, 1, (30, 150)), index=dates,
                          columns=codes)
        ind = pd.DataFrame(
            rng.choice(["A", "B"], (30, 150)), index=dates, columns=codes)
        cap = np.exp(lncap)

        # 脉冲：第 20 日高 vl 股票的 T→T+1 收益整体 +3%（直接改 close）；
        # vl 面板同步加同值脉冲，让第 20 日截面 vl 暴露与 boost 对齐
        boost = (vl.iloc[20] > vl.iloc[20].median()).astype(float)
        vl.iloc[20] = vl.iloc[20] + 5.0
        close.iloc[21] = close.iloc[20].values * (1.0 + 0.03 * boost.values)

        def fake_load():
            return {"exposures": {"lncap": lncap, "vl": vl},
                    "close": close, "cap": cap, "industry": ind}

        monkeypatch.setattr(rb, "load_barra_inputs", fake_load)
        inputs = rb.load_barra_inputs()
        fr = rb.run_factor_returns(inputs, min_names=100)
        assert not fr.empty
        assert all(c.startswith(("ind:", "sty:")) for c in fr.columns)
        assert "sty:vl" in fr.columns
        # 脉冲日的 sty:vl 因子收益应显著为正且压过邻域噪声
        d20 = dates[20]
        neighbors = fr["sty:vl"].drop(d20).abs().max()
        assert fr.loc[d20, "sty:vl"] > max(0.003, 1.5 * neighbors)

    def test_min_names_skips_thin_days(self, monkeypatch):
        import research.barra as rb
        rng = np.random.default_rng(23)
        dates = pd.bdate_range("2024-01-01", periods=10)
        codes = [f"{600000 + i:06d}" for i in range(150)]
        close = pd.DataFrame(
            10 * np.exp(np.cumsum(rng.normal(0, 0.01, (10, 150)), axis=0)),
            index=dates, columns=codes)
        exp_panel = pd.DataFrame(rng.normal(0, 1, (10, 150)), index=dates,
                                 columns=codes)
        cap = pd.DataFrame(np.full((10, 150), 1e9), index=dates, columns=codes)
        ind = pd.DataFrame(np.full((10, 150), "A"), index=dates, columns=codes)

        inputs = {"exposures": {"vl": exp_panel}, "close": close, "cap": cap,
                  "industry": ind}
        fr = rb.run_factor_returns(inputs, min_names=100)
        # 末日无 T+1 收益 → 不入表
        assert dates[-1] not in fr.index


# ---------------------------------------------------------------------------
# 5. 第二期：多风格 Σ / 贝叶斯压缩 / VRA / Bias
# ---------------------------------------------------------------------------
class TestMultiStyleSigma:
    def test_each_style_gets_its_own_exposure(self):
        """多风格时每个 sty 列用各自暴露（10-21 修：旧版全赋同一个 Series）。"""
        codes = ["c1", "c2", "c3"]
        sty = {
            "vl": pd.Series({"c1": 1.0, "c2": 0.0, "c3": 0.0}),
            "mom": pd.Series({"c1": 0.0, "c2": 1.0, "c3": 0.0}),
        }
        ind_row = pd.Series({"c1": "A", "c2": "A", "c3": "A"})
        names = ["ind:A", "sty:vl", "sty:mom"]
        # 行业方差 0（纯风格）；vl 方差 1、mom 方差 4
        f_cov = pd.DataFrame(np.diag([0.0, 1.0, 4.0]), index=names, columns=names)
        spec = pd.Series(0.0, index=codes)

        sigma = assemble_sigma(sty, ind_row, f_cov, spec)
        assert sigma.loc["c1", "c1"] == pytest.approx(1.0)
        assert sigma.loc["c2", "c2"] == pytest.approx(4.0)
        assert sigma.loc["c3", "c3"] == pytest.approx(0.0)
        # 协方差项：c1 与 c2 无共享因子 → 0（旧 bug 会算出非零）
        assert sigma.loc["c1", "c2"] == pytest.approx(0.0)

    def test_missing_style_raises(self):
        names = ["ind:A", "sty:vl"]
        f_cov = pd.DataFrame(np.eye(2) * 1e-4, index=names, columns=names)
        with pytest.raises(KeyError):
            assemble_sigma({"mom": pd.Series({"c1": 1.0})},
                           pd.Series({"c1": "A"}), f_cov,
                           pd.Series({"c1": 1e-4}))


class TestBayesShrink:
    def _fr(self, n=600, k=6, seed=5):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2020-01-01", periods=n)
        names = [f"f{i}" for i in range(k)]
        # 共同因子 + 特异噪声 → 样本相关有噪声，压缩应把它拉平
        common = rng.normal(0, 0.01, (n, 1))
        X = 0.6 * common + rng.normal(0, 0.008, (n, k))
        return pd.DataFrame(X, index=dates, columns=names)

    def test_psd_diagonal_kept_and_delta_in_range(self):
        from research.barra import bayes_shrink_fcov, factor_covariance
        fr = self._fr()
        base = factor_covariance(fr)
        shrunk, delta = bayes_shrink_fcov(fr, base)
        assert 0.05 <= delta <= 0.95
        # 对角（各因子方差）保持不变——压缩只动相关结构
        np.testing.assert_allclose(np.diag(shrunk.values), np.diag(base.values),
                                   rtol=1e-6)
        vals = np.linalg.eigvalsh(shrunk.values)
        assert vals.min() > -1e-12
        np.testing.assert_allclose(shrunk.values, shrunk.values.T, atol=1e-14)

    def test_shrinks_offdiagonal_spread(self):
        """压缩后非对角相关的离散度应下降（向 ρ̄ 靠拢）。"""
        from research.barra import bayes_shrink_fcov
        fr = self._fr(n=120, k=6)          # 短样本 → delta 大
        base = factor_covariance(fr, halflife=60)
        shrunk, _ = bayes_shrink_fcov(fr, base)
        d0 = np.sqrt(np.diag(base.values))
        c0 = base.values / np.outer(d0, d0)
        d1 = np.sqrt(np.diag(shrunk.values))
        c1 = shrunk.values / np.outer(d1, d1)
        off = ~np.eye(len(c0), dtype=bool)
        assert c1[off].std() < c0[off].std()

    def test_fixed_delta_respected(self):
        from research.barra import bayes_shrink_fcov
        fr = self._fr(n=300)
        _, delta = bayes_shrink_fcov(fr, delta=0.3)
        assert delta == pytest.approx(0.3)


class TestVolRegimeAdjust:
    def test_scale_reflects_vol_jump(self):
        """近期波动突然放大 → 特异 scale > 1（上钳 2.0）。"""
        from research.barra import vol_regime_adjust
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2020-01-01", periods=400)
        codes = [f"c{i}" for i in range(30)]
        resid = pd.DataFrame(rng.normal(0, 0.01, (400, 30)),
                             index=dates, columns=codes)
        resid.iloc[-60:] *= 6.0            # 近 60 日波动 ×6
        names = ["ind:A", "sty:vl"]
        f_cov = pd.DataFrame(np.eye(2) * 1e-4, index=names, columns=names)
        fr = pd.DataFrame(rng.normal(0, 0.002, (400, 2)), index=dates,
                          columns=names)
        out, fs, ss = vol_regime_adjust(f_cov, fr, resid)
        assert ss == pytest.approx(2.0)    # 6 倍 → 钳到 2.0
        np.testing.assert_allclose(out.values, f_cov.values * fs ** 2)

    def test_no_history_keeps_identity(self):
        from research.barra import vol_regime_adjust
        names = ["ind:A"]
        f_cov = pd.DataFrame(np.eye(1) * 1e-4, index=names, columns=names)
        out, fs, ss = vol_regime_adjust(f_cov)
        assert (fs, ss) == (1.0, 1.0)
        np.testing.assert_allclose(out.values, f_cov.values)


class TestRiskBias:
    def test_bias_is_one_when_sigma_matches_realized(self):
        from research.barra import risk_bias_stat
        rng = np.random.default_rng(3)
        codes = [f"c{i}" for i in range(400)]
        h = 20
        true_sd = 0.02
        # 构造 Σ = diag(true_sd²)，未来收益 ~ N(0, true_sd²·h)
        sigma = pd.DataFrame(np.diag(np.full(len(codes), true_sd ** 2)),
                             index=codes, columns=codes)
        fut = pd.Series(rng.normal(0, true_sd * np.sqrt(h), len(codes)),
                        index=codes)
        st = risk_bias_stat(sigma, fut, horizon=h)
        assert 0.9 < st["bias"] < 1.1
        assert abs(st["mean_z"]) < 0.1

    def test_underestimate_raises_bias(self):
        from research.barra import risk_bias_stat
        codes = [f"c{i}" for i in range(400)]
        h = 20
        sigma = pd.DataFrame(np.diag(np.full(len(codes), (0.01) ** 2)),
                             index=codes, columns=codes)
        rng = np.random.default_rng(4)
        fut = pd.Series(rng.normal(0, 0.04 * np.sqrt(h), len(codes)), index=codes)
        st = risk_bias_stat(sigma, fut, horizon=h)
        assert st["bias"] > 2.0            # 低估 4 倍波动 → bias ≈ 4
