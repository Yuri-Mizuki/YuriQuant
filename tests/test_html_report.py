"""
交互式 HTML 报告测试
====================
覆盖 research/html_report.py：
- 单因子报告（无对比表、有 canvas）
- 多因子报告（对比排序表 + 标签页）
- render_sortable_table（可排序表、百分比/正负着色）
- 指标格式化（NaN/百分比/数值）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.costs import ShortCostModel
from backtest.engine import VectorBacktest
from research.factor_analysis import factor_summary
from research.html_report import (
    _fmt,
    _month_cell_style,
    generate_html_report,
    render_sortable_table,
)
from strategy.examples import TopKLongShort


def _mock_results():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-02", periods=60, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(20)]
    close = pd.DataFrame(
        10 * np.exp(np.cumsum(rng.normal(0, 0.02, (60, 20)), axis=0)), dates, codes
    )
    ret = close.pct_change().shift(-1)
    fps = {
        "rev_5d": close.rolling(5).mean() / close - 1,
        "mom_10d": -close.pct_change(10),
    }
    results, summaries = {}, {}
    for name, f in fps.items():
        res = VectorBacktest(TopKLongShort(k=5), "W",
                             short_costs=ShortCostModel(borrow_rate=0.08)).run(
            f, ret, check_convention=False)  # 库内 shift(-1) 贴标签口径
        results[name] = res
        summaries[name] = factor_summary(f, ret)
    return results, summaries


def test_generate_html_report_single(tmp_path):
    results, summaries = _mock_results()
    out = generate_html_report(
        {"rev_5d": results["rev_5d"]}, {"rev_5d": summaries["rev_5d"]},
        tmp_path / "r.html", title="T", meta="m",
    )
    html = out.read_text(encoding="utf-8")
    assert "chart.js" in html and "<canvas" in html
    assert "rev_5d" in html
    assert 'id="cmp-body"' not in html  # 单因子无对比表（JS 模板含函数引用，故只查 HTML 块）
    assert html.count("</script>") >= 3


def test_generate_html_report_multi_has_comparison(tmp_path):
    results, summaries = _mock_results()
    out = generate_html_report(results, summaries, tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert "cmp-head" in html and "cmp-body" in html and "cmp-equity" in html
    assert "data-target" in html  # 标签页
    assert "月度收益" in html


def test_render_sortable_table(tmp_path):
    df = pd.DataFrame({
        "name": ["a", "b"],
        "sharpe": [1.2, -0.5],
        "annual_return": [0.08, -0.03],
    })
    out = render_sortable_table(df, tmp_path / "t.html", title="表",
                                pct_cols=["annual_return"],
                                color_cols=["sharpe", "annual_return"])
    html = out.read_text(encoding="utf-8")
    assert "点击表头排序" in html
    assert "1.2000" in html      # sharpe 数值
    assert "8.00%" in html       # 0.08 → 8.00%
    assert "class='neg'" in html  # 负值跌绿
    assert "class='pos'" in html  # 正值涨红


def test_metric_formatting():
    assert _fmt(float("nan"), "sharpe") == "-"
    assert _fmt(0.08, "annual_return") == "8.00%"
    assert _fmt(1.23456, "sharpe") == "1.2346"
    assert _fmt(5000, "borrow_fee_total") == "5,000"
    assert "rgba" in _month_cell_style(0.05)
    assert "rgba" in _month_cell_style(-0.05)


def test_monthly_returns_no_double_add_one():
    """回归：月度复利只能加一次 1。

    早期实现 (1+r).resample('ME').apply((1+x).prod()-1) 双重加 1，
    一个月收益 ≈ (2+r)^21 ≈ 2^n 天文数字（+838860700% 之类）。
    """
    from research.html_report import _monthly_table

    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    dr = pd.Series(0.01, index=dates)  # 恒定日收益 1%
    mt = _monthly_table(dr)
    # 每月约 20 个交易日 → 月度 ≈ (1.01)^20 - 1 ≈ 0.22，绝不可能是百万级
    for y, months in mt.items():
        for m, v in months.items():
            assert abs(v) < 1, f"月度收益异常: {y}-{m} = {v}"
            assert v == pytest.approx(1.01 ** 20 - 1, rel=0.2)
    # 全年复利 ≈ (1.01)^40 - 1 ≈ 0.49
    from research.html_report import _monthly_html
    h = _monthly_html(mt)
    assert "0.49" in h or "48.9" in h or "+48" in h  # 全年显示约 +48.9%
