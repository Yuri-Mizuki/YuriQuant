"""
投资收益报告测试（investment_report.py）。

覆盖：
- factor_test：模型预测面板作因子的检验结构（双口径 IC/分层净值/月度 IC）
- load_index_returns：沪深300指数缓存读取
- CLI mock 端到端：产出文件齐全 + HTML 含内嵌图
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------
def _mock_pred_and_fwd():
    """构造带预测力的 mock：因子面板 + 价格面板（与因子同向收益）+ 未来收益。"""
    rng = np.random.RandomState(7)
    dates = pd.bdate_range("2024-01-01", periods=120)
    codes = [f"{600000+i:06d}.SH" for i in range(40)]
    # 因子：前 20 只高、后 20 只低（带噪声）
    score = pd.DataFrame(
        np.concatenate([rng.normal(1.0, 0.3, (120, 20)), rng.normal(-1.0, 0.3, (120, 20))],
                       axis=1),
        index=dates, columns=codes)
    # 日收益与因子同向（预测力）+ 噪声（真实量级：日波动 ~1.2%）
    ret_d = pd.DataFrame(score.values * 0.008 + rng.normal(0, 0.012, (120, 40)),
                         index=dates, columns=codes)
    # 价格面板（从日收益累乘）
    close = pd.DataFrame(100 * (1 + ret_d).cumprod(), index=dates, columns=codes)
    # 未来 5 日累计收益（IC 口径）
    fwd = close.pct_change(5, fill_method=None).shift(-5)
    # 稀疏预测：仅月初
    sparse = score.copy()
    sparse.loc[~sparse.index.isin(score.index[::21])] = np.nan
    return sparse, fwd, close


def test_factor_test_structure(tmp_path):
    from scripts.investment_report import factor_test
    sparse, fwd, close = _mock_pred_and_fwd()
    out = factor_test(sparse, fwd, close, tmp_path, "mock")
    assert set(out) == {"sum_sparse", "sum_hold", "layer_nav", "ic_hold", "monthly_ic"}
    for key in ("ic_mean", "ic_std", "ir", "t_stat", "t_stat_nw", "ic_win_rate", "n"):
        assert key in out["sum_sparse"] and key in out["sum_hold"]
    # 持仓口径应有预测力：IC > 0
    assert out["sum_hold"]["ic_mean"] > 0.01
    # 分层净值 5 组
    assert list(out["layer_nav"].columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert (tmp_path / "factor_summary_mock.csv").exists()
    assert (tmp_path / "layer_nav_mock.csv").exists()


def test_layer_nav_not_exploding():
    """回归：分层净值不得因 horizon 累计收益按日累乘而重叠放大（曾出现 Q1=6.56 假象）。"""
    from scripts.investment_report import factor_test
    sparse, fwd, close = _mock_pred_and_fwd()
    out = factor_test(sparse, fwd, close, Path("."), "explode_check")
    nav = out["layer_nav"].dropna(how="all")
    # 每条分层净值终点应处于合理量级（< 3），而非重叠放大的 6+/93
    for c in nav.columns:
        assert nav[c].iloc[-1] < 3.0, f"{c} 终点 {nav[c].iloc[-1]:.3f} 疑似重叠放大"
    # 预测力 mock 下 Q5 应 >= Q1（方向正确）
    assert nav["Q5"].iloc[-1] >= nav["Q1"].iloc[-1] - 0.2
    # 网格应从因子有效起始开始（无 1.0 平线前缀）
    first = nav[nav.ne(1.0).any(axis=1)].index
    assert first[0] == nav.index[0]


def test_load_index_returns_from_cache():
    from scripts.investment_report import load_index_returns
    ret = load_index_returns("000300.SH", 20240101, 20241231, real=False)
    if ret is None:
        pytest.skip("本机无沪深300指数缓存")
    assert len(ret.dropna()) > 100
    assert abs(ret.mean()) < 0.05  # 日收益量级合理


def test_plot_images():
    from scripts.investment_report import plot_layers, plot_monthly_ic
    idx = pd.bdate_range("2024-01-01", periods=60)
    layer = pd.DataFrame({
        "Q1": np.cumprod(1 + 0.001 * np.arange(1, 61)),
        "Q2": np.cumprod(1 + 0.002 * np.arange(1, 61)),
        "Q3": np.cumprod(1 + 0.003 * np.arange(1, 61)),
        "Q4": np.cumprod(1 + 0.004 * np.arange(1, 61)),
        "Q5": np.cumprod(1 + 0.005 * np.arange(1, 61)),
    }, index=idx)
    b64 = plot_layers(layer)
    assert b64.startswith("iVBOR")  # PNG base64 头
    mic = pd.Series([0.02, -0.01, 0.03, 0.01], index=pd.period_range("2024-01", periods=4, freq="M"))
    b642 = plot_monthly_ic(mic)
    assert b642.startswith("iVBOR")


# ---------------------------------------------------------------------------
# CLI 端到端（mock）
# ---------------------------------------------------------------------------
def test_investment_report_mock(tmp_path):
    out_dir = tmp_path / "ir"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "investment_report.py"),
        "--top", "20", "--model", "ridge", "--skip-rp",
        "--n-days", "400", "--n-codes", "30", "--seed", "1",
        "--out", str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=600, cwd=str(ROOT))
    assert result.returncode == 0, f"脚本失败:\n{result.stderr}"
    for f in ("investment_report.html", "investment_summary.csv",
              "equity_curve.csv", "monthly_returns.csv",
              "factor_summary_model_pred.csv", "layer_nav_model_pred.csv",
              "monthly_ic_model_pred.csv", "walk_forward_predictions.csv",
              "investment_meta.json"):
        assert (out_dir / f).exists(), f"缺少 {f}"
    html = (out_dir / "investment_report.html").read_text(encoding="utf-8")
    imgs = re.findall(r"data:image/png;base64,", html)
    assert len(imgs) >= 3, f"HTML 内嵌图不足: {len(imgs)}"
    meta = json.load(open(out_dir / "investment_meta.json", encoding="utf-8"))
    assert meta["n_rebalance"] >= 5
    summary = pd.read_csv(out_dir / "investment_summary.csv", index_col=0)
    assert "等权top20" in summary.columns
