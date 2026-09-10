"""AI 41 情感因子构建的纯函数单测（build_senti_factors / build_senti_scores）。"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.textmining.build_senti_factors import (
    RATING_MAP,
    aggregate_window,
    residual_factor,
    winsor_zscore,
)
from scripts.textmining.build_senti_scores import clean_text


def test_clean_text_strips_risk_and_escapes():
    t = "业绩超预期\\r###第二段\\r\\n观点积极。风险提示：市场波动。"
    out = clean_text("标题", t)
    assert "风险提示" not in out
    assert "第二段" in out
    assert out.startswith("标题")
    assert "\r" not in out and "\n" not in out


def test_rating_map_matches_ai41_score_id():
    assert RATING_MAP["买入"] == 7
    assert RATING_MAP["增持"] == 5
    assert RATING_MAP["中性"] == 3
    assert RATING_MAP["减持"] == 2
    assert RATING_MAP["卖出"] == 1


def test_winsor_zscore_clips_extreme_and_standardizes():
    s = pd.Series([0.1, 0.2, 0.15, 0.18, 0.12, 1000.0])
    z = winsor_zscore(s)
    assert z.mean() == pytest.approx(0, abs=1e-9)
    assert z.max() < 5  # 极端值被压回 5×MAD 边界内


def test_aggregate_window_linear_decay():
    daily = pd.DataFrame({
        "code": ["A"] * 3,
        "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-02-01"]),
        "v": [1.0, 0.0, 0.5],
    })
    ends = [pd.Timestamp("2021-02-05")]
    # 90 日窗口内全部命中：lag = 35, 34, 4 → w = 55/90, 56/90, 86/90
    out = aggregate_window(daily, ends, "v", how="mean")
    w = pd.Series([55, 56, 86], dtype=float) / 90
    expect = (w * [1.0, 0.0, 0.5]).sum() / w.sum()
    assert out.loc[0, "factor"] == pytest.approx(expect)

    out_sum = aggregate_window(daily, ends, "v", how="sum")
    assert out_sum.loc[0, "factor"] == pytest.approx((w * [1.0, 0.0, 0.5]).sum())


def test_aggregate_window_excludes_stale_rows():
    daily = pd.DataFrame({
        "code": ["A", "A"],
        "date": pd.to_datetime(["2020-01-01", "2021-02-01"]),
        "v": [0.9, 0.1],
    })
    ends = [pd.Timestamp("2021-02-05")]
    out = aggregate_window(daily, ends, "v", how="mean")
    assert len(out) == 1
    assert out.loc[0, "factor"] == pytest.approx(0.1)  # 2020-01-01 在 90 日窗外


def test_residual_factor_orthogonal_to_controls():
    rng = pd.Series([0.1, -0.2, 0.3, -0.1, 0.25, -0.35])
    dates = pd.Timestamp("2021-01-29")
    codes = [f"C{i}" for i in range(len(rng))]
    target = pd.DataFrame({"date": dates, "code": codes, "factor": winsor_zscore(rng)})
    ctrl = pd.DataFrame({"date": dates, "code": codes,
                         "factor": winsor_zscore(rng * 2 + 0.05)})
    res = residual_factor(target, [ctrl])
    # y 与 x 完全线性相关（标准化不改变线性关系）→ 残差≈0
    assert res["factor"].abs().max() == pytest.approx(0, abs=1e-8)
