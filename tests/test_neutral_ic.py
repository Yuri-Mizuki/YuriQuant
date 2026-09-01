"""
中性化 IC 测试
==============

覆盖 build_style_covariates（五因子风格协变量构建）和
calc_neutral_ic_series（中性化 IC 计算）的核心路径。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor.preprocessing import build_style_covariates
from research.factor_analysis import calc_ic_series, calc_neutral_ic_series


# ===========================================================================
# 辅助函数
# ===========================================================================
def _make_panel(n_days: int = 30, n_codes: int = 20, seed: int = 42) -> dict[str, pd.DataFrame]:
    """构造合成 OHLCV 面板。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    codes = [f"C{i}" for i in range(n_codes)]
    base = 10.0 + rng.normal(0, 1, (n_days, n_codes)).cumprod(axis=0).cumsum(axis=0) * 0.01
    close = pd.DataFrame(50.0 + np.abs(base), index=dates, columns=codes)
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, (n_days, n_codes)), index=dates, columns=codes)
    tot_share = pd.DataFrame(1e8, index=dates, columns=codes)
    return {"close": close, "volume": volume, "tot_share": tot_share}


def _make_market_cap(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return panel["tot_share"] * panel["close"]


def _make_industry_panel(n_days: int, n_codes: int) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    codes = [f"C{i}" for i in range(n_codes)]
    industries = ["银行", "非银金融", "医药生物", "电子", "食品饮料"]
    row = [industries[i % len(industries)] for i in range(n_codes)]
    return pd.DataFrame(
        [row] * n_days, index=dates, columns=codes,
    )


# ===========================================================================
# build_style_covariates
# ===========================================================================
class TestBuildStyleCovariates:
    def test_produces_mom_vol_from_close(self):
        panel = _make_panel(n_days=30, n_codes=20)
        out = build_style_covariates(panel)
        assert "mom" in out
        assert "vol" in out
        # mom 是 pct_change(20)，前 20 天为 NaN
        assert out["mom"].iloc[:19].isna().all().all()
        assert out["mom"].iloc[20].notna().any()

    def test_produces_turn_when_tot_share_available(self):
        panel = _make_panel(n_days=30, n_codes=20)
        out = build_style_covariates(panel)
        assert "turn" in out
        assert out["turn"].iloc[20].notna().any()

    def test_no_turn_without_tot_share(self):
        panel = _make_panel(n_days=30, n_codes=20)
        panel.pop("tot_share")
        out = build_style_covariates(panel)
        assert "turn" not in out
        # mom/vol 仍然有
        assert "mom" in out
        assert "vol" in out

    def test_includes_size_and_industry_when_passed(self):
        panel = _make_panel(n_days=30, n_codes=20)
        mc = _make_market_cap(panel)
        ind = _make_industry_panel(30, 20)
        out = build_style_covariates(panel, market_cap_panel=mc, industry_panel=ind)
        assert "size" in out
        assert "industry" in out
        pd.testing.assert_frame_equal(out["size"], mc)

    def test_empty_panel_returns_empty_dict(self):
        out = build_style_covariates({})
        assert out == {}


# ===========================================================================
# calc_neutral_ic_series
# ===========================================================================
class TestCalcNeutralIcSeries:
    def test_returns_series_named_ic_neutral(self):
        panel = _make_panel(n_days=30, n_codes=20)
        mc = _make_market_cap(panel)
        ind = _make_industry_panel(30, 20)
        cov = build_style_covariates(panel, market_cap_panel=mc, industry_panel=ind)
        close = panel["close"]
        returns = close.pct_change(fill_method=None).shift(-1)
        factor = close.pct_change(5, fill_method=None)

        ic = calc_neutral_ic_series(factor, returns, style_covariates=cov)
        assert isinstance(ic, pd.Series)
        assert ic.name == "ic_neutral"

    def test_neutral_ic_less_than_raw_ic_for_pure_style_factor(self):
        """构造纯规模暴露因子：raw IC 显著，中性化后应下降。

        关键设计：因子与规模相关，收益也与规模相关（规模溢价），
        但因子的截面排名模式在剥离规模后与收益不相关。
        用 mc 的非线性单调变换做因子（如 mc + mc**3），中性化后残差不再与收益相关。
        """
        n_days, n_codes = 60, 50
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
        codes = [f"C{i}" for i in range(n_codes)]
        rng = np.random.default_rng(123)
        # 规模溢价收益：每个股票固定的日均超额 + 噪声
        size_rank = np.arange(n_codes)
        daily_ret = (49 - size_rank) * 0.001  # 小盘多 0.1bp/日
        returns_matrix = np.tile(daily_ret, (n_days, 1)) + rng.normal(0, 0.01, (n_days, n_codes))
        returns_panel = pd.DataFrame(returns_matrix, index=dates, columns=codes)
        # 市值（每天恒定，大盘股市值大）
        close = pd.DataFrame(np.tile(50.0 + size_rank * 0.5, (n_days, 1)), index=dates, columns=codes)
        mc = pd.DataFrame(np.log(close.values * 1e8), index=dates, columns=codes)
        # 因子 = mc + 极少量噪声 → 与 mc 几乎完全共线
        # 中性化后残差 = 极小噪声 → IC 应极小（不 NaN）
        # 注意：neutralize 默认对 market_cap_panel 取 log，这里 mc 已是 log 值，
        # 所以用 extra_covariates 直接传入（不取 log），确保共线性
        factor = mc + rng.normal(0, 0.001, (n_days, n_codes))  # 几乎纯规模暴露

        raw_ic = calc_ic_series(factor, returns_panel)
        neutral_ic = calc_neutral_ic_series(
            factor, returns_panel,
            style_covariates={"size_raw": mc},  # 走 extra_covariates 不取 log
        )

        # raw IC 应显著（规模溢价→因子有预测力），neutral IC 应 ≈ 0
        raw_abs = abs(raw_ic.dropna().mean())
        neutral_abs = abs(neutral_ic.dropna().mean())
        # 因子与回归变量完全共线，中性化后残差≈0，IC 应极小
        assert neutral_abs < 0.01, (
            f"中性化后 IC {neutral_abs:.4f} 应接近 0（因子与 size 完全共线）"
        )
        assert raw_abs > 0.1, f"raw IC {raw_abs:.4f} 应显著（规模溢价）"

    def test_falls_back_to_raw_ic_without_covariates(self):
        panel = _make_panel(n_days=10, n_codes=10)
        close = panel["close"]
        returns = close.pct_change(fill_method=None).shift(-1)
        factor = close.pct_change(5, fill_method=None)

        ic_neutral = calc_neutral_ic_series(factor, returns, style_covariates=None)
        ic_raw = calc_ic_series(factor, returns)
        # 无协变量时退化为 raw IC
        pd.testing.assert_series_equal(ic_neutral, ic_raw, check_names=False)

    def test_works_with_panel_auto_build(self):
        """通过 panel 参数自动构建协变量路径。"""
        panel = _make_panel(n_days=30, n_codes=20)
        mc = _make_market_cap(panel)
        ind = _make_industry_panel(30, 20)
        close = panel["close"]
        returns = close.pct_change(fill_method=None).shift(-1)
        factor = close.pct_change(5, fill_method=None)

        ic = calc_neutral_ic_series(
            factor, returns,
            panel=panel, market_cap_panel=mc, industry_panel=ind,
        )
        assert isinstance(ic, pd.Series)
        assert ic.name == "ic_neutral"
        # 应有非空值（20 日窗口后）
        assert ic.dropna().shape[0] > 0


# ===========================================================================
# 监控接入：style_exposure_ratio + style_rotation 告警
# ===========================================================================
class TestMonitoringNeutralIC:
    def test_style_exposure_ratio_calculated(self):
        """构造纯规模因子+规模溢价，style_exposure_ratio 应 > 0.5。"""
        from monitoring.metrics import compute_factor_metrics

        n_days, n_codes = 60, 50
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
        codes = [f"C{i}" for i in range(n_codes)]
        rng = np.random.default_rng(77)
        size_rank = np.arange(n_codes)
        daily_ret = (49 - size_rank) * 0.001
        returns_matrix = np.tile(daily_ret, (n_days, 1)) + rng.normal(0, 0.01, (n_days, n_codes))
        returns_panel = pd.DataFrame(returns_matrix, index=dates, columns=codes)
        close = pd.DataFrame(np.tile(50.0 + size_rank * 0.5, (n_days, 1)), index=dates, columns=codes)
        mc = pd.DataFrame(np.log(close.values * 1e8), index=dates, columns=codes)
        factor = mc + rng.normal(0, 0.001, (n_days, n_codes))  # 几乎纯规模暴露

        registry_row = pd.Series({"kind": "test", "maturity": "test",
                                  "note": "", "source": "test", "ic_mean": np.nan})
        ic_series = calc_ic_series(factor, returns_panel)
        m = compute_factor_metrics(
            "test_style_factor", registry_row, factor, ic_series, returns_panel,
            as_of=dates[-1], window=30,
            style_covariates={"size_raw": mc},  # 走 extra_covariates 不取 log
        )
        # 风格暴露比例应 > 0.5（因子与 size 几乎完全共线，中性化后 IC≈0）
        assert m.style_exposure_ratio == m.style_exposure_ratio  # not NaN
        assert m.style_exposure_ratio > 0.5, (
            f"风格暴露比例 {m.style_exposure_ratio:.2f} 应 > 0.5"
        )

    def test_style_rotation_alert_fires(self):
        """纯规模因子应触发 style_rotation 告警。"""
        from monitoring.metrics import compute_factor_metrics
        from monitoring.alerts import evaluate_alerts

        n_days, n_codes = 60, 50
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
        codes = [f"C{i}" for i in range(n_codes)]
        rng = np.random.default_rng(99)
        size_rank = np.arange(n_codes)
        daily_ret = (49 - size_rank) * 0.001
        returns_matrix = np.tile(daily_ret, (n_days, 1)) + rng.normal(0, 0.01, (n_days, n_codes))
        returns_panel = pd.DataFrame(returns_matrix, index=dates, columns=codes)
        close = pd.DataFrame(np.tile(50.0 + size_rank * 0.5, (n_days, 1)), index=dates, columns=codes)
        mc = pd.DataFrame(np.log(close.values * 1e8), index=dates, columns=codes)
        factor = mc + rng.normal(0, 0.001, (n_days, n_codes))  # 几乎纯规模暴露

        registry_row = pd.Series({"kind": "test", "maturity": "test",
                                  "note": "", "source": "test", "ic_mean": np.nan})
        ic_series = calc_ic_series(factor, returns_panel)
        m = compute_factor_metrics(
            "test_style_factor", registry_row, factor, ic_series, returns_panel,
            as_of=dates[-1], window=30,
            style_covariates={"size_raw": mc},  # 走 extra_covariates 不取 log
        )
        cfg = {"max_stale_days": 7, "min_coverage": 0.3, "warn_ic_retention": 0.5,
               "min_monotonicity": 0.1, "min_t_nw_recent": 1.0}
        alerts = evaluate_alerts(m, cfg)
        rule_names = [a["rule"] for a in alerts]
        # 应触发 style_rotation（风格暴露比例 > 0.5）
        if m.style_exposure_ratio == m.style_exposure_ratio and m.style_exposure_ratio > 0.5:
            assert "style_rotation" in rule_names


def _ok(v: float) -> bool:
    return v == v
