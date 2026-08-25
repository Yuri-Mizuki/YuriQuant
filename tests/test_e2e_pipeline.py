"""
端到端选股工作流测试（e2e_common / e2e_stock_picks / e2e_backtest）。

覆盖：
- e2e_common：经典因子形状、标签构建、mock 数据形状
- 因子库加载排除 model:*（stale-date bug 回归：预测日必须到数据末端）
- _enforce_caps 约束后处理（w<=cap 且 sum=1）
- e2e_stock_picks --mock 端到端产出文件齐全
- e2e_backtest --mock 端到端产出 summary/净值/月度表
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 共享模块单元测试
# ---------------------------------------------------------------------------
def test_classic_features_shape():
    from scripts.e2e_common import compute_classic_features, load_mock_data
    px = load_mock_data(n_days=200, n_codes=30, seed=0)
    feats = compute_classic_features(px)
    assert set(feats) == {
        "mom5", "mom10", "mom20", "mom60", "rev1", "rev5",
        "vol20", "vol60", "amihud20", "turn_trend", "gap", "range20"}
    for name, p in feats.items():
        assert p.shape == (200, 30)
        assert not p.isna().all().all()  # 非全 NaN


def test_mock_data_shape():
    from scripts.e2e_common import load_mock_data
    px = load_mock_data(n_days=100, n_codes=20, seed=3)
    assert px["close"].shape == (100, 20)
    assert px["high"].ge(px["low"]).all().all()  # high >= low


def test_build_labels_horizon():
    from scripts.e2e_common import build_labels
    from scripts.e2e_common import load_mock_data
    px = load_mock_data(n_days=120, n_codes=10, seed=0)
    labels, fwd = build_labels(px["close"], horizon=5)
    # 尾部 horizon 日标签为 NaN（无未来窗口）
    assert labels.iloc[-5:].isna().all().all()


def test_load_library_factors_excludes_model():
    """stale-date 回归：排除 model:* 因子，面板末端必须到数据末端而非 2025-12-31。"""
    try:
        from scripts.e2e_common import load_library_factors
    except Exception:
        pytest.skip("因子库不可用")
    feats = load_library_factors(exclude_model=True)
    if not feats:
        pytest.skip("库内无 significant 因子")
    names = list(feats)
    assert not any(n.startswith("model:") for n in names)
    # 日期末端应晚于 model:* 面板的截断点（2025-12-31）
    last = max(p.index[-1] for p in feats.values())
    assert last > pd.Timestamp("2025-12-31"), (
        f"存在滞后面板导致预测日截断: 最晚 {last.date()}")


def test_enforce_caps_constraints():
    from scripts.e2e_backtest import _enforce_caps
    w = pd.Series({"a": 0.5, "b": 0.3, "c": 0.2, "d": 0.0})
    out = _enforce_caps(w, cap=0.3)
    assert abs(out.sum() - 1.0) < 1e-9
    assert out.max() <= 0.3 + 1e-9
    assert out.min() >= 0


# ---------------------------------------------------------------------------
# CLI 端到端（mock）
# ---------------------------------------------------------------------------
def test_e2e_stock_picks_mock(tmp_path):
    out_dir = tmp_path / "picks"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "e2e_stock_picks.py"),
        "--top", "10", "--model", "ridge", "--seed", "0",
        "--n-days", "300", "--n-codes", "30",
        "--out", str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=300, cwd=str(ROOT))
    assert result.returncode == 0, f"脚本失败:\n{result.stderr}"
    files = list(out_dir.glob("picks_*.csv")) + list(out_dir.glob("picks_*.txt"))
    assert len(files) >= 2, f"缺少选股产出: {list(out_dir.glob('*'))}"
    meta = json.load(open(out_dir / "pipeline_log.json", encoding="utf-8"))
    assert meta["n_picks"] >= 1
    assert 0.9 <= meta["total_weight"] <= 1.1
    picks = pd.read_csv(out_dir / f"picks_{meta['predict_date'].replace('-', '')}.csv")
    assert len(picks) == meta["n_picks"]


def test_e2e_backtest_mock(tmp_path):
    out_dir = tmp_path / "bt"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "e2e_backtest.py"),
        "--top", "20", "--model", "ridge", "--skip-rp",
        "--n-days", "400", "--n-codes", "30", "--seed", "1",
        "--out", str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=600, cwd=str(ROOT))
    assert result.returncode == 0, f"脚本失败:\n{result.stderr}"
    for f in ("backtest_summary.csv", "monthly_returns.csv", "equity_curve.csv",
              "walk_forward_predictions.csv", "backtest_meta.json"):
        assert (out_dir / f).exists(), f"缺少 {f}"
    meta = json.load(open(out_dir / "backtest_meta.json", encoding="utf-8"))
    assert meta["n_rebalance"] >= 5
    summary = pd.read_csv(out_dir / "backtest_summary.csv", index_col=0)
    assert "等权top20" in summary.columns
    assert "全池等权基准" in summary.columns
