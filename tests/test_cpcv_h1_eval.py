"""
CPCV h=1 无偏评估测试。

覆盖：
- mock 数据下 15 路径全部产出 IC（无 NaN）
- 路径间 t-test 函数正确
- horizon 对比输出格式正确
- embargo 随 horizon 变化
- 输出文件齐全（path_ic.csv / summary.csv / horizon_compare.csv / json）
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


def _run_mock_cli(tmp_path, horizons="1,5", extra_args=None):
    """运行 cpcv_h1_eval.py --mock，输出到 tmp_path。"""
    out_dir = tmp_path / "cpcv_h1"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "cpcv_h1_eval.py"),
        "--mock", "--horizons", horizons,
        "--out", str(out_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT),
    )
    assert result.returncode == 0, f"脚本失败:\n{result.stderr}"
    return out_dir


def test_cpcv_h1_mock_runs_and_produces_all_paths(tmp_path):
    """mock 模式下 15 路径全部产出 IC（无 NaN）。"""
    out_dir = _run_mock_cli(tmp_path, horizons="1")
    path_ic = pd.read_csv(out_dir / "path_ic.csv")
    assert len(path_ic) == 15, f"应有 15 条路径，实际 {len(path_ic)}"
    assert path_ic["ic_mean"].notna().all(), "有路径 IC 为 NaN"
    assert path_ic["horizon"].iloc[0] == 1
    assert (path_ic["n_train"] > 0).all(), "训练天数应 > 0"
    assert (path_ic["n_test"] > 0).all(), "测试天数应 > 0"


def test_cpcv_h1_summary_has_ttest(tmp_path):
    """summary.csv 含 t-test 统计量。"""
    out_dir = _run_mock_cli(tmp_path, horizons="1")
    summary = pd.read_csv(out_dir / "summary.csv")
    assert len(summary) == 1
    assert "ttest_stat" in summary.columns
    assert "ttest_p" in summary.columns
    assert "pct_positive" in summary.columns
    assert not pd.isna(summary["ttest_stat"].iloc[0]), "t-test 统计量为 NaN"
    assert not pd.isna(summary["ttest_p"].iloc[0]), "t-test p 值为 NaN"


def test_horizon_compare_format(tmp_path):
    """horizon 对比表格式正确。"""
    out_dir = _run_mock_cli(tmp_path, horizons="1,5")
    compare = pd.read_csv(out_dir / "horizon_compare.csv")
    assert len(compare) == 2, f"应有 2 个 horizon，实际 {len(compare)}"
    assert set(compare["horizon"]) == {1, 5}
    for col in ["cpcv_ic_mean", "cpcv_ic_median", "n_positive", "ttest_p"]:
        assert col in compare.columns, f"缺少列 {col}"


def test_embargo_scales_with_horizon(tmp_path):
    """embargo 随 horizon 变化（h=1 embargo=1，h=5 embargo=5）。"""
    out_dir = _run_mock_cli(tmp_path, horizons="1,5")
    path_ic = pd.read_csv(out_dir / "path_ic.csv")
    # h=5 的 embargo 更大，训练段应更少
    h1_train = path_ic[path_ic["horizon"] == 1]["n_train"].mean()
    h5_train = path_ic[path_ic["horizon"] == 5]["n_train"].mean()
    assert h5_train <= h1_train, (
        f"h=5 训练天数 {h5_train} 应 <= h=1 {h1_train}（embargo 更大）"
    )


def test_json_output_complete(tmp_path):
    """JSON 输出完整。"""
    out_dir = _run_mock_cli(tmp_path, horizons="1,5")
    with open(out_dir / "cpcv_h1_report.json", encoding="utf-8") as f:
        data = json.load(f)
    assert "config" in data
    assert "summary" in data
    assert "path_ic" in data
    assert data["config"]["horizons"] == [1, 5]
    assert len(data["summary"]) == 2
    assert len(data["path_ic"]) == 30  # 15 paths × 2 horizons


def test_ridge_predictor(tmp_path):
    """ridge 预测器可跑通。"""
    out_dir = _run_mock_cli(tmp_path, horizons="1", extra_args=["--predictor", "ridge"])
    summary = pd.read_csv(out_dir / "summary.csv")
    assert summary["predictor"].iloc[0] == "ridge"
    assert not pd.isna(summary["cpcv_ic_mean"].iloc[0]), "ridge IC 为 NaN"
