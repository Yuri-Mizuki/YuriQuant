"""中性化 builder 去重后的行为回归（2026-10-19）。

两个 90% 重复的脚本（panels_neu / panels_neu_fundind）现共用
``scripts.builders.common.run_neutralization``；本文件用合成面板锁死：
① 标准口径产出文件 + stats jsonl；② fundind 的 industry_only 分支可运行。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.builders import common


def _make_case(tmp_path):
    ds = tmp_path / "ds"
    (ds / "panels").mkdir(parents=True)
    base = tmp_path / "_base"
    base.mkdir()
    idx = pd.bdate_range("2024-01-02", periods=30)
    cols = [f"c{i:02d}" for i in range(20)]
    rng = np.random.default_rng(7)
    for name in ("f1", "f2"):
        p = pd.DataFrame(rng.normal(size=(30, 20)), index=idx, columns=cols)
        p.to_parquet(ds / "panels" / f"{name}.parquet")
    mc = pd.DataFrame(rng.uniform(1e9, 1e10, (30, 20)), index=idx, columns=cols)
    mc.to_parquet(base / "market_cap.parquet")
    ind = pd.DataFrame(
        np.tile(np.array(["银行", "医药", "电子", "机械"]), (30, 5)),
        index=idx, columns=cols)
    ind.to_parquet(base / "cov_industry.parquet")
    pd.DataFrame({"name": ["f1", "f2"], "coverage": [0.9, 0.9]}).to_csv(
        ds / "registry.csv", index=False)
    return ds, base


def test_run_neutralization_standard(tmp_path, monkeypatch):
    ds, base = _make_case(tmp_path)
    monkeypatch.setattr(common, "_NEU_BASE_DIR", base)
    n = common.run_neutralization(ds, "panels_neu", workers=1, force=True,
                                  set_name="neutralized", label="test")
    assert n == 2
    for name in ("f1", "f2"):
        p = pd.read_parquet(ds / "panels_neu" / f"{name}.parquet")
        assert p.shape == (30, 20)
        assert p.notna().any().any()
    rows = [json.loads(line) for line in
            (ds / "factor_stats_neutralized.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    assert {r["name"] for r in rows} == {"f1", "f2"}
    assert all(r["coverage"] > 0 for r in rows)


def test_run_neutralization_industry_only(tmp_path, monkeypatch):
    ds, base = _make_case(tmp_path)
    monkeypatch.setattr(common, "_NEU_BASE_DIR", base)
    n = common.run_neutralization(
        ds, "panels_neu_fundind", workers=1, force=True,
        set_name="fundind", label="test", industry_only=frozenset({"f1"}))
    assert n == 2
    p = pd.read_parquet(ds / "panels_neu_fundind" / "f1.parquet")
    assert p.notna().any().any()
