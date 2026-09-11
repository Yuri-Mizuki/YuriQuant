"""rolling_grid_alla 实验脚本的核心纯函数单测（不依赖全A数据缓存）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pd.Timestamp("2018-01-01")


def _make_base(tmp_path):
    """最小 _base 目录：close/mask/基准（float32 小面板）。"""
    base = tmp_path / "_base"
    base.mkdir(parents=True)
    idx = pd.bdate_range("2018-01-01", "2019-12-31")
    cols = [f"c{i}" for i in range(6)]
    close = pd.DataFrame(
        100 + np.arange(len(idx))[:, None] * 0.1 + np.zeros((1, len(cols))),
        index=idx, columns=cols, dtype=np.float32)
    close.iloc[:3, 0] = np.nan          # 首日 pct_change NaN（引擎口径）
    close.to_parquet(base / "close_adj.parquet")
    pd.DataFrame(True, index=idx, columns=cols).to_parquet(base / "tradable_mask.parquet")
    ret = close.pct_change(fill_method=None).mean(axis=1).fillna(0)
    ret.to_frame("ret").to_parquet(base / "bench_index.parquet")
    ret.to_frame("ret").to_parquet(base / "bench_eqw.parquet")
    pd.DataFrame(np.nan, index=idx, columns=cols, dtype=np.float32).to_parquet(
        base / "market_cap.parquet")
    return base


def test_yearly_metrics_basic():
    from scripts.pipelines.rolling_grid_alla import _yearly_metrics
    idx = pd.bdate_range("2019-01-01", "2020-12-31")
    rng = np.random.default_rng(0)
    dr = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    bench = pd.Series(0.0, index=idx)
    ym = _yearly_metrics(dr, bench)
    assert set(ym) == {2019, 2020}
    for y, m in ym.items():
        sub = dr[dr.index.year == y]
        assert m["excess"] == pytest.approx(m["annual"] - 0.0, abs=1e-9)
        assert m["max_dd"] >= 0      # 正幅度（同 backtest.metrics 约定）
        # 年化 = 复利年化
        expect = (1 + sub).prod() ** (252 / len(sub)) - 1
        assert m["annual"] == pytest.approx(expect, rel=1e-9)


def test_rebalance_days_validated():
    from scripts.pipelines.rolling_grid_alla import _rebalance_days_validated
    idx = pd.bdate_range("2018-01-01", "2018-06-30")
    # 2M：每 2 个月首个交易日；末日 6/1 的区间（6/1->末尾）跨度 21 >= 1 保留
    rbd = _rebalance_days_validated(idx, 1, "2M")
    months = sorted({pd.Timestamp(d).month for d in rbd})
    assert months == [1, 3, 5]
    for d in rbd:
        same_month = idx[(idx.month == d.month) & (idx.year == d.year)]
        assert d == same_month[0]
    # 跨度守卫：horizon 超过末段跨度时，末段调仓日被剔除
    # 2018-06-01 到序列末尾共 21 个交易日位置（跨度 20），h=20 保留、h=25 剔除
    rbd20 = _rebalance_days_validated(idx, 20, "M")
    rbd25 = _rebalance_days_validated(idx, 25, "M")
    assert pd.Timestamp("2018-06-01") not in rbd25
    assert pd.Timestamp("2018-06-01") in rbd20


def test_select_features_dedup_and_coverage(tmp_path, monkeypatch):
    from scripts.pipelines import rolling_grid_alla as R

    days = pd.bdate_range("2016-07-01", "2018-12-31")
    codes = [f"c{i}" for i in range(8)]
    rng = np.random.default_rng(1)
    # f_good 有真实 IC，f_dup 与其完全相关，f_lowcov 覆盖不足
    signal = pd.DataFrame(rng.normal(size=(len(days), len(codes))), index=days,
                          columns=codes)
    fwd = -0.3 * signal + rng.normal(scale=0.1, size=(len(days), len(codes)))
    ic_series = {
        "f_good": (fwd * signal).mean(axis=1) / 100,
        "f_mid": (fwd * (0.5 * signal)).mean(axis=1) / 100,
        "f_lowcov": (fwd * signal).mean(axis=1) / 100,
        "f_weak": pd.Series(rng.normal(scale=1e-5, size=len(days)), index=days),
    }
    ic_cache = pd.DataFrame(ic_series)

    panels_dir = tmp_path / "panels"
    panels_dir.mkdir()
    # f_weak 与 f_good 面板独立（否则会被相关去冗余正确剔除，测不到质量排序）
    weak_panel = pd.DataFrame(rng.normal(size=(len(days), len(codes))),
                              index=days, columns=codes)
    for n in ("f_good", "f_mid", "f_lowcov", "f_weak"):
        if n == "f_lowcov":
            p = signal.where(
                pd.DataFrame(rng.random((len(days), len(codes))) > 0.7,
                             index=days, columns=codes))
        elif n == "f_weak":
            p = weak_panel
        else:
            p = signal.copy()
        p.astype(np.float32).to_parquet(panels_dir / f"{n}.parquet")
    registry = pd.DataFrame({
        "name": ["f_good", "f_mid", "f_lowcov", "f_weak"],
        "coverage": [1.0, 1.0, 0.2, 1.0]})

    store = R.FeatureStore(panels_dir)
    feats = R.select_features_for_year(2018, 1, ic_cache, registry, store, days)
    assert "f_good" in feats
    assert "f_lowcov" not in feats        # 覆盖率过滤
    assert feats.index("f_good") < feats.index("f_weak")  # 质量降序优先


def test_res_metrics_matches_manual():
    from scripts.pipelines.rolling_grid_alla import res_metrics
    idx = pd.bdate_range("2019-01-01", periods=252)
    rng = np.random.default_rng(2)
    dr = pd.Series(rng.normal(0.001, 0.012, len(idx)), index=idx)
    bench = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    to = pd.Series([0.3] * 12, index=idx[::21])
    m = res_metrics(dr, bench, to)
    assert m["turnover"] == pytest.approx(0.3)
    expect_ann = (1 + dr).prod() ** (252 / len(dr)) - 1
    assert m["annual"] == pytest.approx(expect_ann, rel=1e-9)
    assert m["max_dd"] > 0


def test_load_base_guard(tmp_path, monkeypatch):
    from scripts.pipelines import rolling_grid_alla as R
    monkeypatch.setattr(R, "OUT", tmp_path)
    with pytest.raises(FileNotFoundError, match="先跑 --stage prep"):
        R.load_base()
    _make_base(tmp_path)
    base = R.load_base()
    assert set(base) >= {"close", "mask", "bench_index", "bench_eqw"}
    assert base["close"].index[0] == pd.Timestamp("2018-01-01")
