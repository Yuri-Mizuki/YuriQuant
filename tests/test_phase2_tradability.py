"""stage2 Phase 2 可交易掩码注入的单元测试（TODO §三 工程债，2026-09-24）。

覆盖：
- ``build_tradable_mask`` 的 ``execution_lag=0`` 分支（T 日状态口径：
  决策日收盘调仓的 env 用）——与默认 T+1 口径在"状态日错位"时必须不同；
- ``run_portfolio_phase2.make_env`` 的掩码合成：成分 ∩ T 日可交易，
  预注入 ``_TRAD_CACHE`` 隔离真实数据面（合成面板，无数据依赖）；
- 整行不可交易的防御回退（成分掩码兜底，不炸训练）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _status_parquet(tmp_path, rows: dict) -> None:
    recs = []
    for (d, c), (hi, lo, st, susp) in rows.items():
        recs.append({"date": d, "code": c, "pre_close": 10.0,
                     "high_limited": hi, "low_limited": lo,
                     "is_st": st, "is_suspended": susp,
                     "is_ex_dividend": False, "is_ex_rights": False})
    df = pd.DataFrame(recs).set_index(["date", "code"])
    df.to_parquet(tmp_path / "history_stock_status.parquet")


def test_execution_lag_zero_uses_t_day_state(tmp_path):
    """execution_lag=0：T 日封板/停牌剔除；T+1 状态不再影响 T 行。"""
    from data.tradability import build_tradable_mask

    idx = pd.date_range("2023-01-02", periods=4, freq="B")
    cols = ["A", "B"]
    close = pd.DataFrame(10.0, idx, cols)      # bwd=None：close 即原始价
    d1, d2 = idx[0], idx[1]
    _status_parquet(tmp_path, {
        # A：d1 封涨停（close 10 ≥ high_limited 10）；d2 正常
        (d1, "A"): (10.0, 9.0, False, False),
        (d1, "B"): (11.0, 9.0, False, False),
        # B：d2 停牌（T+1 语义下会污染 d1 行；T 日语义不应影响）
        (d2, "A"): (11.0, 9.0, False, False),
        (d2, "B"): (11.0, 9.0, False, True),
    })
    t0 = build_tradable_mask(close, bwd=None, cache_root=str(tmp_path),
                             execution_lag=0)
    assert not t0.loc[d1, "A"], "T 日封涨停应剔除（lag=0）"
    assert t0.loc[d1, "B"], "B 的 d2 停牌不影响 d1 行（lag=0）"
    # 对照：默认 lag=1 下 d1 行看 d2 状态 → B 被剔除
    t1 = build_tradable_mask(close, bwd=None, cache_root=str(tmp_path))
    assert not t1.loc[d1, "B"], "T+1 口径下 d2 停牌应污染 d1 行（lag=1）"
    assert t1.loc[d1, "A"], "T+1 口径下 A 的 d1 封板不影响 d1 行"


def _tiny_env_parts():
    """合成 px/mask：60 个交易日 × 5 码，每月（10 个交易日）一个决策日。"""
    idx = pd.bdate_range("2023-01-02", periods=60)
    cols = [f"S{i}" for i in range(5)]
    rng = np.random.default_rng(5)
    px = {"close": pd.DataFrame(
        10 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, 5)), axis=0)),
        index=idx, columns=cols)}
    member = pd.DataFrame(True, idx, cols)
    member.iloc[:20, 2] = False                 # S2 前两月非成分
    return px, member, cols, idx


def test_make_env_combines_membership_and_tradability(monkeypatch):
    """make_env 掩码 = 成分 ∩ T 日可交易（预注入缓存隔离真实状态表）。"""
    import scripts.factors.run_portfolio_phase2 as ph2

    px, member, cols, idx = _tiny_env_parts()
    decision_dates = list(idx[::10])
    # 合成可交易掩码：S3 全程不可交易（模拟持续 ST/停牌）
    fake_trad = pd.DataFrame(True, idx, cols)
    fake_trad["S3"] = False
    monkeypatch.setitem(ph2._TRAD_CACHE, id(px["close"]), fake_trad)

    f = pd.Series(0.0, index=pd.MultiIndex.from_product(
        [decision_dates, cols], names=["date", "code"]))
    env, bwm = ph2.make_env(px, member, cols, f, f.copy(), decision_dates,
                            dict(), seed=0)
    assert env is not None
    m = env.tradable_masks
    # S3 成分内但不可交易 → False；S2 非成分期 False、入成分期 True
    assert not m["S3"].any()
    assert not m.iloc[:2][ "S2"].all()
    assert m.iloc[3:]["S2"].all()
    assert m["S0"].all() and m["S1"].all()
    # 基准权重不受可交易掩码影响（基准仍按成分归一）
    assert bwm["S3"].sum() > 0


def test_make_env_all_false_row_falls_back_to_membership(monkeypatch):
    """整行不可交易 → 退回成分掩码（防 map_actions_to_weights 全 False 异常）。"""
    import scripts.factors.run_portfolio_phase2 as ph2

    px, member, cols, idx = _tiny_env_parts()
    decision_dates = list(idx[::10])
    fake_trad = pd.DataFrame(True, idx, cols)
    fake_trad.iloc[2] = False                   # 第 3 个决策日全线不可交易
    monkeypatch.setitem(ph2._TRAD_CACHE, id(px["close"]), fake_trad)
    f = pd.Series(0.0, index=pd.MultiIndex.from_product(
        [decision_dates, cols], names=["date", "code"]))
    env, _ = ph2.make_env(px, member, cols, f, f.copy(), decision_dates,
                          dict(), seed=0)
    row = env.tradable_masks.iloc[2]
    assert row.equals(pd.Series(True, index=cols)), "全 False 行应退回成分掩码"
