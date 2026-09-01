"""Alpha101 / GTJA Alpha191 公开因子集单测。

覆盖四层：
1. 公式抽样对照：随机面板上手算参考表达式 vs 因子实现（语义回归锚点）；
2. 全量可计算性：两套因子集所有面板形状/数值合法（无 ±inf）；
3. 去重与跳过清单：与 alpha101 完全重复的 alpha191 因子未注册；
4. 构建入库（mock）与监控 source 分组：registry 的 source 前缀、
   snapshots.csv 的 source 列、报告的来源分组对比章节。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor.alpha101 import ALPHA101, SKIPPED_101, compute_alpha101
from factor.alpha191 import ALPHA191, SKIPPED_191, compute_alpha191
from factor.alpha_base import AlphaData
from factor.operators import cs_rank

CFG = {
    "window": 60,
    "window_long": 252,
    "max_stale_days": 7,
    "confirm_n": 1,
    "min_coverage": 0.5,
    "warn_ic_retention": 0.5,
    "min_monotonicity": 0.8,
    "min_t_nw_recent": 1.0,
    "ledger_root": "reports/monitoring",
}


# ---------------------------------------------------------------------------
# 合成面板 fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def alpha_env() -> dict:
    rng = np.random.default_rng(42)
    # > 最大回看 250 日（尾部充分预热）；40 只股票保证截面 rank 足够细，
    # rank-corr / MAX(A,B) 类因子不因 rank 短期恒定（方差 0）而大面积 NaN
    days, codes = 300, 40
    dates = pd.bdate_range("2024-01-02", periods=days)
    cols = [f"c{i:03d}" for i in range(codes)]
    # AR(1) 均值回复价格：截面排名随时间洗牌，rank-corr 类因子窗口内
    # 不至于方差恒 0（相关无定义 → NaN）
    eps = rng.normal(0, 2.0, (days, codes))
    ar = np.empty((days, codes))
    ar[0] = eps[0]
    for t in range(1, days):
        ar[t] = 0.95 * ar[t - 1] + eps[t]
    close = pd.DataFrame(50 + ar, index=dates, columns=cols)
    open_ = close.shift(1) * (1 + rng.normal(0, 0.004, (days, codes)))
    open_.iloc[0] = close.iloc[0]
    high = pd.DataFrame(np.maximum(open_.values, close.values) * 1.01,
                        index=dates, columns=cols)
    low = pd.DataFrame(np.minimum(open_.values, close.values) * 0.99,
                       index=dates, columns=cols)
    volume = pd.DataFrame(rng.lognormal(9, 0.4, (days, codes)),
                          index=dates, columns=cols)
    amount = volume * (high + low + close) / 3
    vwap = (amount / volume).replace([np.inf, -np.inf], np.nan)
    panels = {"open": open_, "high": high, "low": low, "close": close,
              "volume": volume, "amount": amount, "vwap": vwap}
    return {"d": AlphaData(panels), "panels": panels,
            "dates": dates, "cols": cols}


# ---------------------------------------------------------------------------
# 1) 公式抽样对照（手算参考 vs 实现）
# ---------------------------------------------------------------------------
def test_alpha101_006_formula(alpha_env):
    p = alpha_env["panels"]
    ref = -1 * p["open"].rolling(10, min_periods=10).corr(p["volume"])
    out = compute_alpha101(alpha_env["d"])["alpha101_006"]
    pd.testing.assert_frame_equal(out, ref)


def test_alpha101_012_formula(alpha_env):
    p = alpha_env["panels"]
    dv = p["volume"] - p["volume"].shift(1)
    dc = p["close"] - p["close"].shift(1)
    ref = np.sign(dv) * (-1 * dc)
    out = compute_alpha101(alpha_env["d"])["alpha101_012"]
    pd.testing.assert_frame_equal(out, ref)


def test_alpha191_002_formula(alpha_env):
    p = alpha_env["panels"]
    inner = ((p["close"] - p["low"]) - (p["high"] - p["close"])) / (p["high"] - p["low"])
    ref = -1 * (inner - inner.shift(1))
    out = compute_alpha191(alpha_env["d"])["alpha191_002"]
    pd.testing.assert_frame_equal(out, ref)


def test_alpha191_006_formula(alpha_env):
    p = alpha_env["panels"]
    x = p["open"] * 0.85 + p["high"] * 0.15
    ref = cs_rank(np.sign(x - x.shift(4))) * -1
    out = compute_alpha191(alpha_env["d"])["alpha191_006"]
    pd.testing.assert_frame_equal(out, ref)


def test_alpha191_012_formula(alpha_env):
    p = alpha_env["panels"]
    ref = (cs_rank(p["open"] - p["vwap"].rolling(10, min_periods=10).mean())
           * -1 * cs_rank((p["close"] - p["vwap"]).abs()))
    out = compute_alpha191(alpha_env["d"])["alpha191_012"]
    pd.testing.assert_frame_equal(out, ref)


# ---------------------------------------------------------------------------
# 2) 全量可计算性
# ---------------------------------------------------------------------------
# 结构性低覆盖因子（非实现 bug）：公式含 corr(rank/tsrank(慢变量), …, 短窗)
# —— 慢变量（60 日均量等）的截面/时序 rank 在短窗内常恒定（或单调漂移）
# → 方差 0 → 相关无定义 → NaN 经 decay/tsmax/tsrank 链式传播。
# 真实数据同样如此，仅要求"部分股票可算"证明链路未死。
LOW_COVERAGE_OK = {"alpha101_096", "alpha101_097", "alpha191_064", "alpha191_138"}


@pytest.mark.parametrize("fn,registry,skipped,tag", [
    (compute_alpha101, ALPHA101, SKIPPED_101, "alpha101"),
    (compute_alpha191, ALPHA191, SKIPPED_191, "alpha191"),
])
def test_all_factors_computable(alpha_env, fn, registry, skipped, tag):
    out = fn(alpha_env["d"], skip=True)
    assert set(out) == set(registry) - set(skipped)
    for name, p in out.items():
        assert p.shape == (300, 40), f"{name} 形状错误"
        assert not np.isinf(p.to_numpy()).any(), f"{name} 含 ±inf"
        tail = p.iloc[-20:]  # 长回看因子仅尾部充分预热
        cov = tail.notna().to_numpy().mean()
        bar = 0.02 if name in LOW_COVERAGE_OK else 0.5
        assert cov > bar, f"{name} 尾部覆盖率过低 ({cov:.2f})"


# ---------------------------------------------------------------------------
# 3) 去重与跳过清单
# ---------------------------------------------------------------------------
def test_alpha191_dedup_skips():
    # 与 alpha101 完全重复的 3 个因子（032/040/139）不得注册
    dups = [k for k in SKIPPED_191 if "完全相同" in SKIPPED_191[k]]
    assert set(dups) == {"alpha191_032", "alpha191_040", "alpha191_139"}
    assert not (set(SKIPPED_191) & set(ALPHA191))
    # 191 - 11 跳过 = 180 注册；101 - 1 跳过 = 100 注册
    assert len(ALPHA191) == 180
    assert len(ALPHA101) == 100
    # 对应的 alpha101 侧因子确实存在
    for n in (32, 40):
        assert f"alpha101_{n:03d}" in ALPHA101
    assert "alpha101_006" in ALPHA101  # 139 与 006 同式


def test_alpha101_skipped_documented():
    assert set(SKIPPED_101) == {"alpha101_056"}
    assert "alpha101_056" not in set(ALPHA101)


# ---------------------------------------------------------------------------
# 4) 构建入库（mock 数据，小子集）+ 监控 source 分组
# ---------------------------------------------------------------------------
def test_build_alpha_factors_mock_register(tmp_path: Path):
    from data.cache import DataCache
    from data.universe import Universe
    from factor.alpha_base import load_alpha_panels
    from research.factor_library import FactorLibrary
    from scripts.cli_common import register_panels
    from tests.conftest import MockDataSource

    cache = DataCache(MockDataSource(), cache_root=str(tmp_path / "cache"))
    uni = Universe(cache)
    panels_px, industry, close_adj = load_alpha_panels(
        cache, uni, "000300.SH", 20220101, 20241231)
    d = AlphaData(panels_px, industry=industry)
    assert panels_px["close"].shape[0] > 400

    b0, e0 = pd.Timestamp("2023-01-03"), pd.Timestamp("2024-12-31")
    out = {**compute_alpha101(d), **compute_alpha191(d)}
    defs = {k: f"测试 {k}" for k in out}
    panels = {k: p.loc[b0:e0] for k, p in out.items()}
    returns = close_adj.pct_change().shift(-1).loc[b0:e0]

    lib = FactorLibrary(root=tmp_path / "flib", dataset="alpha_mock")
    names = ["alpha101_006", "alpha101_012", "alpha191_002", "alpha191_006"]
    rows = register_panels(
        lib, panels, defs, returns,
        source="alpha101:build_alpha_factors:test", names=names)
    # names 中含 alpha191_，但 source 是统一前缀 —— 构建脚本按 set 分批调用，
    # 这里验证单批调用本身；再按 alpha191 前缀补一批
    rows += register_panels(
        lib, panels, defs, returns,
        source="alpha191:build_alpha_factors:test",
        names=["alpha191_002", "alpha191_006"])
    assert len(rows) >= 5

    reg = lib.list_all()
    assert set(reg["name"]) == set(names)
    src = dict(zip(reg["name"], reg["source"]))
    assert src["alpha101_006"].startswith("alpha101:")
    assert src["alpha191_002"].startswith("alpha191:")
    # 入库面板已 zscore：截面均值≈0
    p = lib.get_panel("alpha101_006")
    tail = p.dropna(how="all")
    assert abs(float(tail.iloc[-1].mean())) < 1e-8


def test_monitoring_source_groups(tmp_path: Path):
    from research.factor_library import FactorLibrary

    rng = np.random.default_rng(3)
    days, codes = 170, 25
    dates = pd.bdate_range("2025-01-01", periods=days)
    cols = [f"c{i:03d}" for i in range(codes)]
    close = pd.DataFrame(
        100 + rng.normal(0, 1, (days, codes)).cumsum(axis=0), index=dates, columns=cols
    )
    rets = close.pct_change(fill_method=None).shift(-1)

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    long_close = close.stack().rename("close")
    long_close.index = long_close.index.set_names(["date", "code"])
    long_close.to_frame().to_parquet(cache_root / "daily_hs300.parquet")

    lib_root = tmp_path / "flib"
    lib = FactorLibrary(root=lib_root, dataset="srcds")
    good = rets.rank(pct=True, axis=1) - 0.5
    lib.register("gp_alpha", good, rets, kind="raw", formula="g",
                 source="gp:mine_factors")
    lib.register("alpha101_006", good * 0.9, rets, kind="raw", formula="a6",
                 source="alpha101:build_alpha_factors:2022-2025")
    lib.register("alpha191_002", good * 0.8, rets, kind="raw", formula="a2",
                 source="alpha191:build_alpha_factors:2022-2025")

    from monitoring.runner import run_monitoring

    summary = run_monitoring(
        dataset="srcds",
        factor_root=lib_root,
        cache_root=cache_root,
        ledger_root=tmp_path / "ledger",
        cfg=CFG,
        record=False,
    )
    assert summary["n_factors"] == 3

    snap = pd.read_csv(tmp_path / "ledger" / "snapshots.csv")
    assert "source" in snap.columns
    by_name = snap.set_index("name")["source"]
    assert by_name.loc["gp_alpha"] == "gp:mine_factors"
    assert by_name.loc["alpha101_006"].startswith("alpha101:")

    html = (tmp_path / "ledger" / "monitor_report.html").read_text(encoding="utf-8")
    assert "来源组" in html
    for g in ("gp", "alpha101", "alpha191"):
        assert f">{g}<" in html, f"报告缺少来源组 {g}"
    assert "中位|12个月IC|" in html


def test_source_group_property():
    from monitoring.metrics import MonitorMetrics

    m = MonitorMetrics(name="x", source="alpha101:build_alpha_factors:2022-2025")
    assert m.source_group == "alpha101"
    m2 = MonitorMetrics(name="y")
    assert m2.source_group == "(未标注)"
    assert MonitorMetrics(name="z", source="gp:mine").source_group == "gp"
