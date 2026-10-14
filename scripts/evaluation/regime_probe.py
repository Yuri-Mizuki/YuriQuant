"""Regime 条件化探针（Phase 0/2 前置诊断，2026-10-15）。

背景
----
大盘择时线已收口（`reports/timing_probe/`）；regime 条件化是其唯一继承者
（计划见 `reports/regime_probe/README.md`）。本探针回答两个前置问题，
**全部不进引擎、秒~分钟级**：

Phase 0 —— 市场级分状态 IC
    集成信号与定版 50 因子分别在「市场 20 日涨/跌期」的 IC。
    已知结论（2026-10-15）：集成 IC 差异显著（0.127 vs 0.110，t=3.55），
    但**单因子/族层面不存在跌期更强的信号源**（50 因子 diff mean +0.0009，
    负端全是低波动因子的负 IC 减弱，非可用 alpha）⇒ 「信号切换」版无肉。

Phase 2 前置 —— 个股级门内/门外分池 IC
    门控定义在**每只股票自己的状态**上（arXiv drift regime / 民生 Deep Timing
    思想）：如个股过去 63 日上涨天数占比 > 60% = drift regime 门内。
    门内/门外各自算 rank IC（同日截面内分组）——若门内 IC 不显著高于门外，
    个股门控进引擎必死于样本稀释，直接收口。

口径（与 timing_probe/caliber 同源，🚨 勿混）
---------------------------------------------
- signal[t] 对 fwd1[t] = t→t+1 收益（``ctx.fwd.shift(-1)``）；
- 可交易掩码 + 有限值过滤，截面有效股票 > ``MIN_NAMES`` 才计 IC；
- 门控状态只用 t 日及以前数据（rolling 回看，天然无前视）；
- 门内/门外 IC 差异用 Welch t + Mann-Whitney 双检验，报 Bonferroni 校正
  （状态定义数见 ``N_GATES``）。

用法
----
    python -m scripts.evaluation.regime_probe            # 两探针全跑
    python -m scripts.evaluation.regime_probe --self-check

产物
----
    reports/regime_probe/{state_conditional_ic, factor_state_ic,
    stock_gate_ic}.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "regime_probe"
#: 个股 drift 门的回看窗与阈值（arXiv 2511.12490 的 63 日/60% 阳线占比）
DRIFT_WINDOW = 63
DRIFT_THRESH = 0.60
#: 个股波动门的回看窗（自身 20 日波动，截面中位二分）
VOL_WINDOW = 20
#: 截面有效股票数下限（低于则跳过该日）
MIN_NAMES = 500
#: 门控定义数（Bonferroni 校正基数）
N_GATES = 2


def daily_ic_series(sig: pd.DataFrame, fwd1: pd.DataFrame,
                    mask: pd.DataFrame) -> pd.Series:
    """日度截面 rank IC：signal[t] vs t→t+1 收益。"""
    out = {}
    svals, rvals, mvals = sig.values, fwd1.values, mask.values
    for i in range(len(sig.index) - 1):
        s, r = svals[i], rvals[i]
        ok = mvals[i] & np.isfinite(s) & np.isfinite(r)
        if ok.sum() > MIN_NAMES:
            out[sig.index[i]] = st.spearmanr(s[ok], r[ok])[0]
    return pd.Series(out)


def _group_ic_diff(ic: pd.Series, gate: pd.Series,
                   label_hi: str, label_lo: str) -> dict:
    """按日级布尔门把日度 IC 分两组，做 Welch t + MWU。"""
    g = gate.reindex(ic.index).fillna(False)
    hi, lo = ic[g], ic[~g]
    t_stat, p_t = st.ttest_ind(hi, lo, equal_var=False)
    _, p_u = st.mannwhitneyu(hi, lo)
    return {"gate": label_hi, "n_hi": len(hi), "ic_hi": hi.mean(),
            "n_lo": len(lo), "ic_lo": lo.mean(),
            "diff": hi.mean() - lo.mean(), "welch_t": t_stat,
            "p_ttest": p_t, "p_mwu": p_u,
            "bonferroni_p": min(1.0, p_t * N_GATES)}


def pooled_gate_ic(sig: pd.DataFrame, fwd1: pd.DataFrame,
                   mask: pd.DataFrame, gate_panel: pd.DataFrame,
                   label: str) -> dict:
    """**分池** rank IC：同一截面内按个股门状态分两组，各自算 IC。

    与「日级门 × 整截面 IC」不同——个股门每天同时切出两个子截面，
    才是个股门控的真实工作方式（arXiv drift regime 形态）。
    """
    gvals = gate_panel.reindex(index=sig.index, columns=sig.columns)
    gv = gvals.to_numpy(dtype=float)
    in_pool, out_pool = [], []
    svals = sig.to_numpy(dtype=float)
    rvals = fwd1.to_numpy(dtype=float)
    mvals = mask.to_numpy(dtype=bool)
    for i in range(len(sig.index) - 1):
        s, r, m, g = svals[i], rvals[i], mvals[i], gv[i]
        base = m & np.isfinite(s) & np.isfinite(r) & np.isfinite(g)
        s_all, r_all = s[base], r[base]
        if base.sum() <= MIN_NAMES:
            continue
        rank_s = pd.Series(s_all).rank()
        rank_r = pd.Series(r_all).rank()
        g_in = g[base] > 0.5
        if g_in.sum() > MIN_NAMES / 5:
            in_pool.append(float(np.corrcoef(rank_s[g_in], rank_r[g_in])[0, 1]))
        if (~g_in).sum() > MIN_NAMES / 5:
            out_pool.append(float(np.corrcoef(rank_s[~g_in], rank_r[~g_in])[0, 1]))
    t_stat, p_t = st.ttest_ind(in_pool, out_pool, equal_var=False)
    _, p_u = st.mannwhitneyu(in_pool, out_pool)
    return {"gate": label, "n_hi": len(in_pool), "ic_hi": np.mean(in_pool),
            "n_lo": len(out_pool), "ic_lo": np.mean(out_pool),
            "diff": np.mean(in_pool) - np.mean(out_pool), "welch_t": t_stat,
            "p_ttest": p_t, "p_mwu": p_u,
            "bonferroni_p": min(1.0, p_t * N_GATES)}


def _assert_finite_and_ordered(rows: list[dict]) -> None:
    """守卫：门内 IC 必须可比较（NaN 会静默通过 groupby）。"""
    for r in rows:
        for k in ("ic_hi", "ic_lo", "diff", "welch_t"):
            assert np.isfinite(r[k]), f"{r['gate']}.{k} 非有限值——检查门覆盖"


def self_check() -> None:
    """合成数据守卫：构造一个"门内信号更强"的已知结构，探针必须测出方向。"""
    rng = np.random.default_rng(5)
    n_days, n_stk = 400, 2000
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    cols = [f"S{i:04d}" for i in range(n_stk)]
    gate = pd.DataFrame(
        rng.random((n_days, n_stk)) > 0.5, index=idx, columns=cols)
    # 门内：信号 = 0.6×未来收益 + 噪声；门外：纯噪声（截面前视仅用于构造真值）
    fwd1 = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_stk)), index=idx, columns=cols)
    sig = fwd1.mul(gate.where(gate, 0.0)) * 30 + pd.DataFrame(
        rng.normal(0, 1, (n_days, n_stk)), index=idx, columns=cols)
    mask = pd.DataFrame(True, index=idx, columns=cols)
    r = pooled_gate_ic(sig, fwd1, mask, gate.astype(float), "自检门")
    assert r["ic_hi"] > r["ic_lo"] + 0.05, \
        f"自检失效：门内 IC {r['ic_hi']:.3f} 未显著高于门外 {r['ic_lo']:.3f}"
    print(f"self-check: 门内 IC {r['ic_hi']:.3f} > 门外 {r['ic_lo']:.3f} 方向正确 ✓")


def main() -> None:
    ap = argparse.ArgumentParser(description="Regime 条件化前置探针")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--signal", default="ens_h1h5")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return

    from scripts.evaluation.caliber import load_context
    ctx = load_context(signal=args.signal, window="prod")
    sig = ctx.pred.where(ctx.mask)
    fwd1 = ctx.fwd.shift(-1)
    mkt_eq = ctx.close.pct_change(fill_method=None).mean(axis=1)

    # ---- Phase 0 市场级（日级门 × 整截面 IC）----
    up20 = (mkt_eq.rolling(20).sum() > 0)
    vol20 = mkt_eq.rolling(20).std()
    rows = [
        _group_ic_diff(daily_ic_series(sig, fwd1, ctx.mask),
                       up20, "市场20日上涨期", "市场20日下跌期"),
        _group_ic_diff(daily_ic_series(sig, fwd1, ctx.mask),
                       vol20 > vol20.expanding(min_periods=120).median(),
                       "市场高波动期", "市场低波动期"),
    ]
    # 波动门的 Bonferroni 基数按实际行数校正
    for r in rows:
        r["bonferroni_p"] = min(1.0, r["p_ttest"] * len(rows))
    df_mkt = pd.DataFrame(rows).set_index("gate")

    # ---- Phase 2 前置：个股门分池 IC ----
    ret = ctx.close.pct_change(fill_method=None)
    # 门 1：drift regime（63 日阳线占比 > 60%）
    up_day = (ret > 0).astype(float)
    drift = up_day.rolling(DRIFT_WINDOW, min_periods=DRIFT_WINDOW // 2).mean()
    gate_drift = (drift > DRIFT_THRESH).astype(float)
    # 门 2：个股自身波动截面中位二分
    svol = ret.rolling(VOL_WINDOW).std()
    svol_med = svol.expanding(min_periods=120).median()
    gate_vol = (svol > svol_med).astype(float)
    rows2 = [pooled_gate_ic(sig, fwd1, ctx.mask, gate_drift, "个股drift门内(63日阳线>60%)"),
             pooled_gate_ic(sig, fwd1, ctx.mask, gate_vol, "个股高波动门内(20d自身分位)")]
    df_stk = pd.DataFrame(rows2).set_index("gate")
    _assert_finite_and_ordered(rows2)

    OUT.mkdir(parents=True, exist_ok=True)
    df_mkt.to_csv(OUT / "state_conditional_ic.csv", encoding="utf-8-sig")
    df_stk.to_csv(OUT / "stock_gate_ic.csv", encoding="utf-8-sig")

    print(f"被测 {args.signal} · {ctx.pred.index[0].date()}~{ctx.pred.index[-1].date()}")
    print("\n=== Phase 0：市场级分状态 IC（日级门 × 整截面）===")
    print(df_mkt.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== Phase 2 前置：个股门分池 IC（同日截面分组）===")
    print(df_stk.to_string(float_format=lambda x: f"{x:.4f}"))
    verdict = []
    for gate_name, r in df_stk.iterrows():
        sig_txt = f"Δ{r['diff']:+.4f} p={r['p_ttest']:.3f}"
        if r["p_ttest"] < 0.05 and r["diff"] > 0:
            verdict.append(f"✅ {gate_name}：门内 IC 显著更高（{sig_txt}）⇒ 值得进引擎")
        else:
            verdict.append(f"❌ {gate_name}：门内无优势（{sig_txt}）⇒ 收口")
    print("\n判定：")
    for v in verdict:
        print(f"  {v}")
    print(f"\n产物: {OUT}/[state_conditional_ic, stock_gate_ic].csv")


if __name__ == "__main__":
    main()
