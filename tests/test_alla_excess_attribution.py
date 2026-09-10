"""alla_excess_attribution 的归因链路测试（mock 面板，不依赖真实回测产物）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.alla_excess_attribution import (
    compute_attribution,
    equal_weight_bench,
    industry_mapping,
)

N_DAYS = 300
N_CODES = 20
CODES = [f"{600000 + i}.SH" for i in range(N_CODES)]
DATES = pd.bdate_range("2024-01-02", periods=N_DAYS)


@pytest.fixture(scope="module")
def mock_world():
    rng = np.random.default_rng(7)
    rets = pd.DataFrame(rng.normal(0.0005, 0.015, (N_DAYS, N_CODES)),
                        index=DATES, columns=CODES)
    close = (1 + rets.fillna(0)).cumprod() * 10.0

    # 组合权重：每月初等权持有前 5 只（其余为 0）
    w = pd.DataFrame(0.0, index=DATES, columns=CODES)
    month_starts = DATES.to_series().groupby(DATES.to_period("M")).first()
    for d in month_starts:
        w.loc[d, CODES[:5]] = 0.2
    weights = w.replace(0.0, np.nan).ffill().fillna(0.0)  # 引擎口径：调仓日之外 ffill

    # 组合日收益 = 权重组合收益 + 每日 5bp 纯 α
    port_ret = (weights * rets).sum(axis=1) + 0.0005
    bench_idx = pd.Series(rng.normal(0.0002, 0.010, N_DAYS), index=DATES)
    bench_eqw = rets.mean(axis=1)

    ind_by_code = ["bank" if i % 2 == 0 else "tech" for i in range(N_CODES)]
    cov_industry = pd.DataFrame(
        np.tile(ind_by_code, (N_DAYS, 1)), index=DATES, columns=CODES)

    base = {
        "close": close,
        "cov": {"industry": cov_industry},
        "bench_index": bench_idx,
        "bench_eqw": bench_eqw,
    }
    return {"rets": rets, "weights": weights, "base": base,
            "dr": port_ret, "bench_idx": bench_idx}


# ---------------------------------------------------------------------------
def test_equal_weight_bench_rows_sum_to_one(mock_world):
    rets = mock_world["rets"]
    bw = equal_weight_bench(rets)
    row_sum = bw.sum(axis=1)
    assert np.allclose(row_sum.values, 1.0)
    # 与"可用收益均值"口径一致：Σ w·r = mean(r)
    weighted = (bw * rets.fillna(0.0)).sum(axis=1)
    assert np.allclose(weighted.values, rets.mean(axis=1).values, atol=1e-12)


def test_industry_mapping_takes_latest_nonnull():
    cov = pd.DataFrame(
        {"A": ["x", "x", "y"], "B": ["z", np.nan, np.nan]},
        index=pd.date_range("2024-01-01", periods=3))
    m = industry_mapping(cov)
    assert m["A"] == "y"       # ffill 后取最近值
    assert m["B"] == "z"       # 缺失处保留历史值


def test_compute_attribution_end_to_end(mock_world):
    out = compute_attribution(
        mock_world["dr"], mock_world["weights"],
        mock_world["base"], mock_world["base"]["close"].index)

    ab = out["ab_df"]
    assert list(ab.index) == ["vs_上证指数", "vs_全A等权"]
    for col in ("alpha_annual", "alpha_t_nw", "alpha_p_nw", "beta",
                "beta_t_nw", "r2", "n"):
        assert col in ab.columns
    # 注入每日 +5bp α：α 估计 = 组合日均(去掉市场项) + 0.0005
    port_daily_mean = mock_world["dr"].mean()
    expected_annual = port_daily_mean * 252
    assert ab["alpha_annual"].iloc[0] == pytest.approx(expected_annual, rel=0.25)
    assert ab["alpha_t_nw"].iloc[0] > 1.96

    br_df, br_summary = out["br_df"], out["br_summary"]
    assert "TOTAL" in br_df.index
    # Carino 链接下效应合计 ≈ 主动收益；算术口径重建误差应≈0
    assert abs(br_summary["recon_error"]) < 1e-8
    effect_sum = (br_summary["allocation"] + br_summary["selection"]
                  + br_summary["interaction"])
    assert effect_sum == pytest.approx(br_summary["active_return"], rel=0.05)

    cmp_summary = out["cmp_summary"]
    assert cmp_summary["excess_idx"] == cmp_summary["excess_idx"]  # 非 NaN
