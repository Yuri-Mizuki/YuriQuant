"""
因子预处理测试
==============

覆盖去极值、标准化、中性化（含小样本降级、NaN 处理），以及组合入口
preprocess_factor 在 mock 模式（无市值/行业数据）下的优雅退化。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.preprocessing import (
    neutralize,
    preprocess_factor,
    standardize_rank,
    standardize_zscore,
    winsorize_mad,
    winsorize_quantile,
)


# ===========================================================================
# 去极值
# ===========================================================================
def test_winsorize_mad_clips_planted_outlier():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    codes = [f"C{i}" for i in range(20)]
    values = rng.normal(0, 1, 20)
    values[0] = 100.0  # 植入极端值
    panel = pd.DataFrame([values], index=dates, columns=codes)

    out = winsorize_mad(panel, n_mad=3.0)

    assert out.iloc[0, 0] != 100.0
    median = panel.iloc[0].median()
    mad = (panel.iloc[0] - median).abs().median()
    upper = median + 3.0 * 1.4826 * mad
    assert out.iloc[0, 0] == pytest.approx(upper)
    # 其余值不受影响
    assert (out.iloc[0, 1:] == panel.iloc[0, 1:]).all()


def test_winsorize_quantile_clips_planted_outlier():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    codes = [f"C{i}" for i in range(50)]
    values = rng.normal(0, 1, 50)
    values[0] = 100.0
    panel = pd.DataFrame([values], index=dates, columns=codes)

    out = winsorize_quantile(panel, lower=0.01, upper=0.99)

    lower_bound = panel.iloc[0].quantile(0.01)
    upper_bound = panel.iloc[0].quantile(0.99)
    assert out.iloc[0, 0] == pytest.approx(upper_bound)
    # 落在边界内的值不受影响，边界外的值应被压到边界上
    within_bound = (panel.iloc[0] >= lower_bound) & (panel.iloc[0] <= upper_bound)
    assert (out.iloc[0][within_bound] == panel.iloc[0][within_bound]).all()
    assert (out.iloc[0] >= lower_bound).all() and (out.iloc[0] <= upper_bound).all()


# ===========================================================================
# 标准化
# ===========================================================================
def test_standardize_zscore_mean_zero_std_one():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    codes = [f"C{i}" for i in range(30)]
    panel = pd.DataFrame(rng.normal(5, 2, (5, 30)), index=dates, columns=codes)

    out = standardize_zscore(panel)

    for d in dates:
        assert out.loc[d].mean() == pytest.approx(0.0, abs=1e-8)
        assert out.loc[d].std() == pytest.approx(1.0, abs=1e-6)


def test_standardize_rank_bounded_and_monotonic():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    codes = [f"C{i}" for i in range(10)]
    values = list(range(10))
    panel = pd.DataFrame([values], index=dates, columns=codes)

    out = standardize_rank(panel)

    assert (out.iloc[0] >= 0).all() and (out.iloc[0] <= 1).all()
    # 保持原始大小顺序
    assert (out.iloc[0].values == out.iloc[0].values[np.argsort(np.argsort(values))]).all()


# ===========================================================================
# 中性化
# ===========================================================================
def _synthetic_market_cap(dates, codes, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.uniform(1e9, 1e11, len(codes))
    return pd.DataFrame([base] * len(dates), index=dates, columns=codes)


def test_neutralize_zeros_out_size_correlation():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    codes = [f"C{i}" for i in range(20)]
    market_cap = _synthetic_market_cap(dates, codes)
    factor_panel = np.log(market_cap)  # 因子完全等于 log(市值)，无噪声

    out = neutralize(factor_panel, market_cap_panel=market_cap)

    # 因子和回归变量完全共线（y = size_x），完美拟合下残差应精确为 0——
    # 这时对残差（零方差向量）算相关系数是 0/0 未定义，因此直接断言残差本身，
    # 而不是断言残差与市值的相关系数。
    for d in dates:
        assert out.loc[d].abs().max() < 1e-8


def test_neutralize_removes_industry_mean_offset():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    codes = [f"C{i}" for i in range(20)]
    rng = np.random.default_rng(4)

    industry = pd.Series(["A"] * 10 + ["B"] * 10, index=codes)
    base = rng.normal(0, 1, 20)
    offset = np.array([5.0] * 10 + [-5.0] * 10)
    factor_values = base + offset
    factor_panel = pd.DataFrame([factor_values], index=dates, columns=codes)
    industry_panel = pd.DataFrame([industry.values], index=dates, columns=codes)
    market_cap = _synthetic_market_cap(dates, codes)  # 常数市值：无 size 效应

    out = neutralize(factor_panel, market_cap_panel=market_cap, industry_panel=industry_panel)

    group_a_mean = out.loc[dates[0], industry[industry == "A"].index].mean()
    group_b_mean = out.loc[dates[0], industry[industry == "B"].index].mean()
    assert group_a_mean == pytest.approx(0.0, abs=1e-6)
    assert group_b_mean == pytest.approx(0.0, abs=1e-6)


def test_neutralize_handles_nan_industry():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    codes = [f"C{i}" for i in range(20)]
    rng = np.random.default_rng(5)

    industry_values = ["A"] * 8 + ["B"] * 8 + [np.nan] * 4
    factor_panel = pd.DataFrame([rng.normal(0, 1, 20)], index=dates, columns=codes)
    industry_panel = pd.DataFrame([industry_values], index=dates, columns=codes)
    market_cap = _synthetic_market_cap(dates, codes)

    out = neutralize(factor_panel, market_cap_panel=market_cap, industry_panel=industry_panel)

    nan_codes = codes[16:]
    finite_codes = codes[:16]
    assert out.loc[dates[0], nan_codes].isna().all()
    assert out.loc[dates[0], finite_codes].notna().all()


def test_neutralize_small_cross_section_falls_back_to_size_only():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    codes = [f"C{i}" for i in range(5)]
    rng = np.random.default_rng(6)
    # 5 个样本、5 个不同行业类别（每个样本各自一个行业），完整模型
    # （1 个市值系数 + 5 个行业哑变量）必然会因样本不足被降级为只用市值
    industry_panel = pd.DataFrame([[f"IND{i}" for i in range(5)]], index=dates, columns=codes)
    market_cap = _synthetic_market_cap(dates, codes)
    factor_panel = pd.DataFrame([rng.normal(0, 1, 5)], index=dates, columns=codes)

    out = neutralize(factor_panel, market_cap_panel=market_cap, industry_panel=industry_panel)

    # 不应报错，且残差不应退化成全部趋近于 0（那意味着被过拟合抹平）
    assert out.loc[dates[0]].notna().all()
    assert out.loc[dates[0]].abs().sum() > 1e-6


def test_neutralize_insufficient_data_returns_nan_for_day():
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    codes = [f"C{i}" for i in range(5)]
    factor_panel = pd.DataFrame(
        [[1.0, np.nan, np.nan, np.nan, np.nan], [1.0, 2.0, 3.0, 4.0, 5.0]],
        index=dates, columns=codes,
    )
    market_cap = _synthetic_market_cap(dates, codes)

    out = neutralize(factor_panel, market_cap_panel=market_cap)

    # 第一天只有 1 个有效样本（< min_samples_size_only=2），全部 NaN
    assert out.loc[dates[0]].isna().all()
    # 第二天样本充足，应有有限值
    assert out.loc[dates[1]].notna().all()


def test_neutralize_noop_without_any_panels():
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    codes = [f"C{i}" for i in range(5)]
    factor_panel = pd.DataFrame(
        np.random.default_rng(7).normal(0, 1, (2, 5)), index=dates, columns=codes
    )

    out = neutralize(factor_panel)

    pd.testing.assert_frame_equal(out, factor_panel)


# ===========================================================================
# 组合入口
# ===========================================================================
def test_preprocess_factor_pipeline_order():
    rng = np.random.default_rng(8)
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    codes = [f"C{i}" for i in range(30)]
    factor_panel = pd.DataFrame(rng.normal(5, 2, (3, 30)), index=dates, columns=codes)
    market_cap = _synthetic_market_cap(dates, codes)
    industry_panel = pd.DataFrame(
        [["A", "B"] * 15] * 3, index=dates, columns=codes
    )

    out = preprocess_factor(
        factor_panel, market_cap_panel=market_cap, industry_panel=industry_panel,
        standardize="zscore",
    )

    for d in dates:
        assert out.loc[d].mean() == pytest.approx(0.0, abs=1e-6)
        assert out.loc[d].std() == pytest.approx(1.0, abs=1e-6)


def test_preprocess_factor_mock_mode_no_panels():
    rng = np.random.default_rng(9)
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    codes = [f"C{i}" for i in range(20)]
    factor_panel = pd.DataFrame(rng.normal(5, 2, (2, 20)), index=dates, columns=codes)

    out = preprocess_factor(factor_panel)  # 不传市值/行业，默认 flags 保持 True

    for d in dates:
        assert out.loc[d].mean() == pytest.approx(0.0, abs=1e-6)
        assert out.loc[d].std() == pytest.approx(1.0, abs=1e-6)
