"""统一评估口径（``scripts/evaluation/caliber.py`` + ``config/baselines.yaml``）回归。

2026-10-13 建立。锁死三件事：
1. 定版登记表的 schema 与不变量（有且仅有 1 条 production；不可复现须给重建命令）；
2. 口径原语的口径方向（几何 ≤ 算术 Sharpe；指纹确定且敏感；调仓日 = 每月首个交易日）；
3. **协方差暖机** —— ``ret_hist`` 必须比回测窗口长（防「窗口起点=数据起点」静默空仓）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 登记表
# ---------------------------------------------------------------------------
def test_registry_schema_and_invariants():
    from scripts.evaluation.caliber import SIGNALS, WINDOWS, load_registry

    reg = load_registry()
    bl = reg.get("baselines") or []
    assert bl, "定版登记表不得为空"

    ids = [b["id"] for b in bl]
    assert len(ids) == len(set(ids)), "baseline id 必须唯一"
    for b in bl:
        assert b["status"] in {"production", "working", "superseded"}, b["id"]
        assert b["window"] in WINDOWS, b["id"]
        assert b["signal_key"] in SIGNALS, b["id"]
        assert isinstance(b["reproducible_local"], bool), b["id"]
        if not b["reproducible_local"]:
            assert b.get("regen_cmd"), f"{b['id']} 非本机可复现则必须给 regen_cmd"

    prod = [b for b in bl if b["status"] == "production"]
    assert len(prod) == 1, "有且仅有一条 production 定版"


def test_registry_conventions_match_code():
    from scripts.evaluation.caliber import DEFAULT_WINDOW, WINDOWS, load_registry

    conv = (load_registry().get("conventions") or {})
    assert conv.get("windows", {}).get("prod") == WINDOWS["prod"]
    assert DEFAULT_WINDOW == "prod"
    assert conv.get("sharpe") == "geometric"


# ---------------------------------------------------------------------------
# 口径原语
# ---------------------------------------------------------------------------
def test_rebalance_dates_is_first_trading_day_of_month():
    from scripts.evaluation.caliber import rebalance_dates

    idx = pd.bdate_range("2020-01-01", periods=400)
    reb = rebalance_dates(idx)
    assert len(reb) > 0
    for t in reb:
        same_month = idx[idx.to_period("M") == t.to_period("M")]
        assert t == same_month[0]


def test_fingerprint_deterministic_and_sensitive():
    from scripts.evaluation.caliber import fingerprint_panel

    p = pd.DataFrame(np.arange(12.0).reshape(4, 3),
                     index=pd.bdate_range("2020-01-01", periods=4))
    assert fingerprint_panel(p) == fingerprint_panel(p.copy())
    p2 = p.copy()
    p2.iloc[0, 0] += 1.0
    assert fingerprint_panel(p2) != fingerprint_panel(p)


def test_sharpe_pair_geometric_le_arithmetic():
    from backtest.metrics import sharpe_ratio
    from scripts.evaluation.caliber import sharpe_pair

    rng = np.random.default_rng(0)
    dr = pd.Series(rng.normal(0.0008, 0.012, 800),
                   index=pd.bdate_range("2020-01-02", periods=800))
    out = sharpe_pair(dr)
    assert out["sharpe"] == pytest.approx(sharpe_ratio(dr))
    assert out["sharpe_arith"] == pytest.approx(dr.mean() / dr.std() * np.sqrt(252.0))
    assert out["sharpe_arith"] > out["sharpe"]        # 几何 ≤ 算术（波动拖累）


def test_sharpe_pair_degenerate_is_nan():
    from scripts.evaluation.caliber import sharpe_pair

    dr = pd.Series([0.0] * 40, index=pd.bdate_range("2021-01-04", periods=40))
    out = sharpe_pair(dr)
    assert np.isnan(out["sharpe"]) and np.isnan(out["sharpe_arith"])


def test_cost_profile_kinds():
    from backtest.costs import ImpactCostModel, TransactionCosts
    from scripts.evaluation.caliber import cost_profile

    imp = cost_profile("impact")
    assert isinstance(imp, TransactionCosts)
    assert isinstance(imp.impact, ImpactCostModel)
    assert imp.slippage_bp == 0.0                    # 冲击替代固定滑点
    flat = cost_profile("flat")
    assert flat.impact is None and flat.slippage_bp > 0
    with pytest.raises(KeyError):
        cost_profile("nope")


# ---------------------------------------------------------------------------
# 与旧探针原语的一致性（防重构口径漂移）
# ---------------------------------------------------------------------------
def test_probe_reexports_match_caliber():
    from scripts.evaluation import caliber
    from scripts.evaluation import cost_penalty_probe as probe

    assert probe._sharpe_pair is caliber.sharpe_pair
    assert probe._rebalance_dates is caliber.rebalance_dates
    rng = np.random.default_rng(3)
    dr = pd.Series(rng.normal(0.0005, 0.01, 600),
                   index=pd.bdate_range("2021-01-04", periods=600))
    assert probe._sharpe_pair(dr) == caliber.sharpe_pair(dr)


# ---------------------------------------------------------------------------
# 协方差暖机（集成；无本地基础面板时跳过）
# ---------------------------------------------------------------------------
def _has_data() -> bool:
    try:
        from scripts.evaluation.caliber import BASE_DIR
    except Exception:
        return False
    return (BASE_DIR / "close_adj.parquet").exists()


@pytest.mark.skipif(not _has_data(), reason="无本地基础面板（reports/alla_rolling/_base）")
def test_load_context_warmup_extends_before_window():
    """``ret_hist``（Σ 暖机）必须**长于**回测窗口。

    2026-10-13 实测缺陷：探针把收益面板截断到回测窗口 ⇒ 窗口最早 4 个调仓日
    历史 < min_periods ⇒ ``rolling_covariance`` 返回 None ⇒ 组合静默空仓
    （压低 QP 约 2.7pp/年）。本测试锁死「暖机面板不截断」。
    """
    from scripts.evaluation.caliber import load_context

    ctx = load_context(signal="ens_h1h5", window="prod", begin=20240101, end=20241231)
    assert ctx.ret_hist.index.min() < ctx.pred.index.min()
    assert len(ctx.ret_hist) > len(ctx.pred)
    # 窗口内 fwd 与 ret_hist 是**两个**面板：fwd 从窗口起点算、ret_hist 更早
    assert ctx.fwd.index.equals(ctx.pred.index)


# ---------------------------------------------------------------------------
# 口径组合（2026-10-14：生产口径 vs 研究压力口径）
# ---------------------------------------------------------------------------
def test_profiles_are_explicit_and_production_is_default():
    from scripts.evaluation.caliber import DEFAULT_PROFILE, PROFILES

    assert DEFAULT_PROFILE == "production"
    p = PROFILES["production"]
    assert (p.cost, p.execution, p.capital) == ("production", "open", 1.0e6)
    f = PROFILES["fair"]
    assert (f.cost, f.execution, f.capital) == ("impact", "close", 1.0e8)
    assert p.execution != f.execution and p.capital != f.capital


def test_cost_profile_production_matches_default_costs():
    """``cost_profile("production")`` 必须与生产真源 ``default_costs()`` 逐字同源。"""
    from backtest.costs import default_costs
    from scripts.evaluation.caliber import cost_profile

    a, b = cost_profile("production"), default_costs()
    assert a.slippage_bp == b.slippage_bp
    assert a.commission_rate == b.commission_rate
    assert a.stamp_duty == b.stamp_duty
    assert (a.impact is None) == (b.impact is None)


def test_next_day_rebalance_is_trading_day_after_month_first():
    from scripts.evaluation.caliber import next_day_rebalance

    idx = pd.bdate_range("2020-01-01", periods=200)
    rb = next_day_rebalance(idx)
    assert len(rb) > 0
    pos = {d: i for i, d in enumerate(idx)}
    for t in rb:
        assert t in pos
        prev = idx[pos[t] - 1]
        same_month = idx[idx.to_period("M") == prev.to_period("M")]
        assert prev == same_month[0]          # 前一天是当月首个交易日


def test_execution_open_without_fill_raises():
    """未加载成交价面板却要求 T+1 执行 → 必须明确报错（而非静默用 close）。"""
    from scripts.evaluation.caliber import (
        CaliberContext,
        cost_profile,
        evaluate_strategy,
    )

    ctx = CaliberContext(window="prod", signal="x", begin="20200101",
                         pred=pd.DataFrame(), close=pd.DataFrame(),
                         fwd=pd.DataFrame(), mask=pd.DataFrame(),
                         amt=pd.DataFrame(), vol60=pd.DataFrame(),
                         ret_hist=pd.DataFrame(), fingerprint="t",
                         execution="open", fill=None)

    class _Noop:
        def target_weights(self, *a, **k):
            return pd.Series(dtype=float)

    with pytest.raises(ValueError, match="ctx.fill"):
        evaluate_strategy(_Noop(), ctx, cost_profile("production"), 1e6)
