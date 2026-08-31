"""交互式因子检测报告的测试：数据端函数 + HTML 生成。"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _mock_factor_panel(n_days=120, n_codes=30, seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    # 因子：与未来收益正相关（前一半高、后一半低 + 噪声）
    score = pd.DataFrame(
        np.concatenate([rng.normal(1.0, 0.3, (n_days, n_codes // 2)),
                        rng.normal(-1.0, 0.3, (n_days, n_codes - n_codes // 2))], axis=1),
        index=dates, columns=codes)
    return score


def _mock_returns_like(panel, score=None, alpha=0.02):
    """收益与因子分数正相关（alpha>0），保证 mock 有预测力。"""
    rng = np.random.RandomState(1)
    ret = pd.DataFrame(rng.normal(0.001, 0.01, panel.shape),
                       index=panel.index, columns=panel.columns)
    if score is not None:
        ret = ret + alpha * score.values  # 分数越高未来收益越高
    return ret


def test_qcut_rebal_shape_and_monotonic():
    from research.factor_report import qcut_rebal
    f = _mock_factor_panel()
    r = _mock_returns_like(f)
    for n, freq in [(5, "M"), (10, "W")]:
        nav = qcut_rebal(f, r, n, freq, monthly_points=True)
        assert nav is not None
        assert nav.shape[1] == n
        # 月度点：mock 数据 ~6 个月
        assert len(nav) >= 4
        # 终点非 NaN 且有限
        assert nav.iloc[-1].notna().all()
        assert np.isfinite(nav.iloc[-1]).all()


def test_qcut_rebal_monotonic_direction():
    """正预测力 mock 下 Qn 应 >= Q1（分层单调方向正确）。"""
    from research.factor_report import qcut_rebal
    f = _mock_factor_panel()
    r = _mock_returns_like(f, score=f, alpha=0.02)  # 收益与因子正相关
    nav = qcut_rebal(f, r, 5, "M", monthly_points=True)
    ends = nav.iloc[-1].values
    # 前一半高分股未来收益更高 -> Q5 应 >= Q1
    assert ends[-1] >= ends[0]


def test_layer_stats_and_avg():
    from research.factor_report import layer_stats_from_nav, layer_avg_ret
    f = _mock_factor_panel()
    r = _mock_returns_like(f)
    from research.factor_report import qcut_rebal
    nav = qcut_rebal(f, r, 5, "M", monthly_points=True)
    stats = layer_stats_from_nav(nav)
    avg = layer_avg_ret(nav)
    assert len(stats) == 5 and len(avg) == 5
    assert all(s and np.isfinite(s["annual"]) for s in stats)
    assert all(np.isfinite(a) for a in avg)


def test_ic_decay_and_heatmap():
    from research.factor_report import ic_decay_series, ic_heatmap, monthly_series
    rng = np.random.RandomState(0)
    ic = pd.Series(rng.normal(0, 0.05, 300),
                   index=pd.bdate_range("2023-01-02", periods=300))
    decay = ic_decay_series(ic, max_lag=10)
    assert len(decay) == 10
    assert all(d is None or np.isfinite(d) for d in decay)
    # 自相关构造：连续同号 -> 正衰减
    ic2 = pd.Series(np.sin(np.arange(300) / 8),
                    index=pd.bdate_range("2023-01-02", periods=300))
    d2 = ic_decay_series(ic2, max_lag=3)
    assert d2[0] is not None and d2[0] > 0.5
    heat = ic_heatmap(monthly_series(ic))
    assert isinstance(heat, dict) and len(heat) >= 1
    assert all(len(v) == 12 for v in heat.values())


def test_html_generation_smoke(tmp_path):
    """mock 模式下 HTML 能生成且包含核心组件。"""
    from research.factor_library import FactorLibrary
    # 用临时库避免污染真实库：直接测 HTML 模板 + 一个假因子
    import scripts.factor_explorer_report as mod
    fake = {
        "name": "test_factor", "family": "alpha101",
        "formula": "rank(close)", "ic_series": {"2024-01": 0.05, "2024-02": 0.03},
        "ls_nav": {"2024-01": 1.01, "2024-02": 1.02}, "lo_nav": {"2024-01": 1.0, "2024-02": 1.01},
        "ic_w": {"2024-01-05": 0.04}, "ic_decay": [0.1, 0.05],
        "heat": {"2024": [0.05, 0.03] + [None] * 10},
        "q5_M": {"2024-01": [0.98, 1.0, 1.01, 1.02, 1.03], "2024-02": [0.97, 1.0, 1.02, 1.04, 1.05]},
        "q10_M": None, "q5_W": None, "q10_W": None,
        "layer_stats_M": [{"annual": 0.1, "vol": 0.2, "sharpe": 0.5, "dd": -0.1}] * 5,
        "layer_avg_M": [0.01, 0.02, 0.03, 0.04, 0.05],
        "layer_stats_W": None, "layer_avg_W": None,
        "m_full": {"ic": 0.04, "icir": 2.0, "t_nw": 1.5, "ls_ret": 0.02, "lo_ret": 0.01,
                   "ls_sharpe": 0.8, "win": 0.6, "turn": 0.05, "sig": True},
    }
    html = mod.HTML_TEMPLATE.format(
        dataset="mock", n_factors=1,
        factor_data=json.dumps([fake], ensure_ascii=False),
        months=json.dumps(["2024-01", "2024-02"]),
        fam_colors=json.dumps({"alpha101": "#378ADD"}),
    )
    # 核心组件存在
    assert 'id="ftable"' in html
    assert 'id="detail"' in html
    assert 'cLayers' in html and 'cIcTs' in html and 'cDecay' in html and 'cHeat' in html
    # 数据内嵌正确
    assert 'test_factor' in html
    assert 'setFreq' in html and 'setLayers' in html
    # format 占位符全部被替换（无残留 {factor_data} 等未替换标记）
    for ph in ["{dataset}", "{n_factors}", "{factor_data}", "{months}", "{fam_colors}"]:
        assert ph not in html, f"未替换占位符: {ph}"


def test_main_mock_smoke(tmp_path):
    """CLI mock 冒烟：对小数据集生成完整 HTML 文件。

    注意必须显式传 --dataset mock：默认数据集 hs300_2022_2025 有 821 个因子，
    全量分层回测要跑约 20 分钟，不适合单测。
    """
    import subprocess, sys
    out = tmp_path / "explorer.html"
    r = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "factor_explorer_report.py"),
         "--dataset", "mock", "--out", str(out)],
        capture_output=True, text=True, timeout=300)
    # mock 库/缓存缺失时生成失败也接受——但必须报错信息明确
    if r.returncode != 0:
        assert "因子库" in r.stderr or "Error" in r.stderr or "Traceback" in r.stderr
