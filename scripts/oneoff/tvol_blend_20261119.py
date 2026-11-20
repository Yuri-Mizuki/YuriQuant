"""一次性探针（2026-11-19）：monthly 的 tvol 扫描 + monthly/balanced 两档混合。

两个问题，都只出数不动参数：

A. ``target_vol`` 0.10 / 0.12 / 0.15 / 0.18 × 月内 5 相位 → 相位平均 Calmar。
   历史结论见 ``reports/etf_contest5/README.md`` §5.3（45 只池、脚本已回收），
   本次在 50 只正常口径上重跑，判断 0.15 是否仍是「无理由的居中点」。

B. monthly 与 balanced 的日收益相关性，以及 ``w·monthly + (1−w)·balanced``
   的净值指标。若相关 < 0.7，说明两档不是同一个风险来源，混合可能优于单档。

⚠️ 混合按**日频再平衡**计算，未计两腿之间的调仓成本（月频档换手本来就低，
实际代价很小），因此 B 的混合指标是**上界**。

用法（系统 Python 3.12）::

    python -m scripts.oneoff.tvol_blend_20261119

产物：``reports/etf_champ/tvol_scan_50pool.csv``、``reports/etf_champ/blend_monthly_balanced.csv``
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.portfolio.etf_champ_phase_matrix import (
    build_env,
    month_phase_rebalance_days,
    run_champ,
)
from scripts.portfolio.etf_champ_signals import PROFILES
from scripts.portfolio.etf_weight_probe import run

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/etf_champ"

TVOLS = (0.10, 0.12, 0.15, 0.18)
PHASES = (1, 2, 3, 4, 5)
BLEND_W = (0.3, 0.4, 0.5, 0.6)  # monthly 的权重


def metrics(eq: pd.Series) -> dict:
    r = eq.pct_change().dropna()
    n = len(r)
    ann = float((1.0 + r).prod() ** (252.0 / n) - 1.0)
    sd = float(r.std(ddof=1))
    sr = float(r.mean() / sd * np.sqrt(252.0)) if sd > 0 else float("nan")
    cum = (1.0 + r).cumprod()
    mdd = float((1.0 - cum / cum.cummax()).max())
    return {"ann": ann, "sharpe": sr, "mdd": mdd,
            "calmar": ann / mdd if mdd > 0 else float("nan")}


def task_a(env, rb_map) -> pd.DataFrame:
    rows: list[dict] = []
    base = dict(PROFILES["monthly"])
    for tv in TVOLS:
        spec = {**base, "target_vol": tv}
        for k in PHASES:
            t0 = time.time()
            _r0, r1 = run_champ(env, spec, rb_map[k])
            m = metrics(r1["equity"])
            rows.append({"tvol": tv, "phase": k, **m,
                         "turnover": r1["avg_turnover"],
                         "secs": round(time.time() - t0, 1)})
            print(f"  tvol={tv:.2f} k={k}  年化 {m['ann']:>7.2%}  "
                  f"SR {m['sharpe']:>5.2f}  MDD {m['mdd']:>7.2%}  "
                  f"Calmar {m['calmar']:>5.2f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "tvol_scan_50pool.csv", index=False, encoding="utf-8-sig")

    agg = d.groupby("tvol").agg(
        ann=("ann", "mean"), sr=("sharpe", "mean"), mdd=("mdd", "mean"),
        mdd_worst=("mdd", "max"), sr_min=("sharpe", "min"),
        turnover=("turnover", "mean"),
    ).reset_index()
    agg["ann_range"] = d.groupby("tvol")["ann"].apply(lambda x: x.max() - x.min()).values
    agg["calmar"] = agg["ann"] / agg["mdd"]
    print("\n== A. tvol 相位平均（50 只池）==")
    print(f"  {'tvol':>5}{'年化均':>9}{'SR均':>7}{'SR最差':>8}{'MDD均':>8}"
          f"{'MDD最差':>9}{'Calmar':>8}{'换手':>8}{'极差':>7}")
    for _, x in agg.iterrows():
        print(f"  {x['tvol']:>5.2f}{x['ann']:>9.2%}{x['sr']:>7.2f}{x['sr_min']:>8.2f}"
              f"{x['mdd']:>8.2%}{x['mdd_worst']:>9.2%}{x['calmar']:>8.2f}"
              f"{x['turnover']:>8.1%}{x['ann_range'] * 100:>6.1f}pp")
    return d


def task_b(env, rb_map) -> tuple[pd.DataFrame, pd.Series]:
    spec_m = dict(PROFILES["monthly"])
    spec_b = dict(PROFILES["balanced"])
    _r0m, rm = run_champ(env, spec_m, rb_map[1])
    _r0b, rb = run(env, spec_b)
    rm_eq, rb_eq = rm["equity"], rb["equity"]
    ret = pd.DataFrame({"monthly": rm_eq.pct_change(),
                        "balanced": rb_eq.pct_change()}).dropna()
    corr = float(ret["monthly"].corr(ret["balanced"]))
    print(f"\n== B. 日收益相关 = {corr:.3f}"
          f"（月频档 {len(rm_eq)} 日 / 均衡档 {len(rb_eq)} 日，重叠 {len(ret)} 日）==")

    rows = []
    for name, eq in (("monthly", rm_eq), ("balanced", rb_eq)):
        rows.append({"mix": name, **metrics(eq)})
    for w in BLEND_W:
        r = w * ret["monthly"] + (1.0 - w) * ret["balanced"]
        eq = (1.0 + r).cumprod()
        rows.append({"mix": f"monthly {w:.0%}", **metrics(eq)})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "blend_monthly_balanced.csv", index=False, encoding="utf-8-sig")

    print(f"  {'组合':<14}{'年化':>9}{'SR':>7}{'MDD':>9}{'Calmar':>8}")
    for _, x in d.iterrows():
        print(f"  {x['mix']:<14}{x['ann']:>9.2%}{x['sharpe']:>7.2f}"
              f"{x['mdd']:>9.2%}{x['calmar']:>8.2f}")
    return d, rb_eq


def task_c(env, rb_map, rb_eq: pd.Series) -> pd.DataFrame:
    """混合的**相位平均**口径：monthly 取 k=1..5，balanced 固定自身日历。

    单相位（k=1）的混合读数含 monthly 的 2.55pp 相位红利，不能直接引用；
    本表把每个 k 的混合净值都算一遍，再对指标取均值。
    """
    spec_m = dict(PROFILES["monthly"])
    rows = []
    for k in PHASES:
        _r0, rm = run_champ(env, spec_m, rb_map[k])
        ret = pd.DataFrame({"m": rm["equity"].pct_change(),
                            "b": rb_eq.pct_change()}).dropna()
        for w in (0.5, 0.6):
            r = w * ret["m"] + (1.0 - w) * ret["b"]
            rows.append({"w_monthly": w, "phase": k,
                         **metrics((1.0 + r).cumprod())})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "blend_monthly_balanced_phase.csv", index=False,
             encoding="utf-8-sig")
    agg = d.groupby("w_monthly").agg(
        ann=("ann", "mean"), sharpe=("sharpe", "mean"),
        mdd=("mdd", "mean"), mdd_worst=("mdd", "max"),
        ann_min=("ann", "min"), ann_max=("ann", "max"),
    ).reset_index()
    agg["calmar"] = agg["ann"] / agg["mdd"]
    print("\n== C. 混合 · 相位平均（monthly 换 5 个相位，balanced 不动）==")
    print(f"  {'monthly 权重':<12}{'年化均':>9}{'SR均':>7}{'MDD均':>8}"
          f"{'MDD最差':>9}{'Calmar':>8}{'年化区间':>18}")
    for _, x in agg.iterrows():
        print(f"  {x['w_monthly']:<12.0%}{x['ann']:>9.2%}{x['sharpe']:>7.2f}"
              f"{x['mdd']:>8.2%}{x['mdd_worst']:>9.2%}{x['calmar']:>8.2f}"
              f"{x['ann_min']:>8.2%}~{x['ann_max']:>9.2%}")
    return d


def _ann_seg(eq: pd.Series, lo: str, hi: str) -> float:
    s = eq.loc[lo:hi]
    if len(s) < 30:
        return float("nan")
    r = s.pct_change().dropna()
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)


def task_d(env, rb_map) -> pd.DataFrame:
    """防御腿内部构成 A/B：bond / gold / mix / mix30 / mix70 × 月内 5 相位。

    判定看三点：① Calmar 排序；② 差异是否超出相位噪声（ann_range）；
    ③ 前段（2020–2023）与后段（2024–2026）是否同向 —— 黄金单边行情会让
    ``gold`` / ``mix30`` 在后段虚高，若前段不支持则视为机会性结果。
    """
    from scripts.portfolio.etf_rotation_panels import build_defense, level_from_returns

    base_def = build_defense(env["close"])
    spec = dict(PROFILES["monthly"])
    rows: list[dict] = []
    for key, ret in base_def.items():
        ex = pd.DataFrame({"DEF": level_from_returns(ret)})
        env["extra_def"] = ex
        env["close_def"] = env["close"].join(ex)
        env["open_def"] = env["open_adj"].join(ex)
        for k in PHASES:
            _r0, r1 = run_champ(env, spec, rb_map[k])
            m = metrics(r1["equity"])
            rows.append({"def": key, "phase": k, **m,
                         "ann_early": _ann_seg(r1["equity"], "2020-01-01", "2023-12-31"),
                         "ann_late": _ann_seg(r1["equity"], "2024-01-01", "2026-12-31"),
                         "turnover": r1["avg_turnover"]})
            print(f"  def={key:<5} k={k}  年化 {m['ann']:>7.2%}  SR {m['sharpe']:>5.2f}  "
                  f"MDD {m['mdd']:>7.2%}  Calmar {m['calmar']:>5.2f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "defense_mix_scan.csv", index=False, encoding="utf-8-sig")
    agg = d.groupby("def").agg(
        ann=("ann", "mean"), sr=("sharpe", "mean"), mdd=("mdd", "mean"),
        mdd_worst=("mdd", "max"), sr_min=("sharpe", "min"),
        ann_range=("ann", lambda x: x.max() - x.min()),
        ann_early=("ann_early", "mean"), ann_late=("ann_late", "mean"),
    ).reindex(["bond", "gold", "mix", "mix30", "mix70"]).reset_index()
    agg["calmar"] = agg["ann"] / agg["mdd"]
    print("\n== D. 防御腿构成 · 相位平均 ==")
    print(f"  {'腿':<7}{'年化均':>9}{'SR均':>7}{'MDD均':>8}{'MDD最差':>9}"
          f"{'Calmar':>8}{'极差':>7}{'前段':>9}{'后段':>9}")
    for _, x in agg.iterrows():
        print(f"  {x['def']:<7}{x['ann']:>9.2%}{x['sr']:>7.2f}{x['mdd']:>8.2%}"
              f"{x['mdd_worst']:>9.2%}{x['calmar']:>8.2f}"
              f"{x['ann_range'] * 100:>6.1f}pp{x['ann_early']:>9.2%}{x['ann_late']:>9.2%}")
    return d


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-a", action="store_true", help="跳过 tvol 扫描（已跑过时）")
    ap.add_argument("--skip-b", action="store_true", help="跳过混合两表")
    ap.add_argument("--only-d", action="store_true", help="只跑防御腿 A/B")
    ap.add_argument("--only-b", action="store_true", help="只跑两档混合（A/D 跳过）")
    args = ap.parse_args()

    print("加载环境（全量 ETF 面板 + DEF 合成腿，约 30–90 秒）…", flush=True)
    t0 = time.time()
    env = build_env()
    rb_map = month_phase_rebalance_days(env["close"].index)
    print(f"环境就绪 {time.time() - t0:.0f}s\n", flush=True)
    if args.only_d:
        task_d(env, rb_map)
        print(f"\n明细已落盘 {OUT}")
        return 0
    if not args.skip_a:
        task_a(env, rb_map)
    if not args.skip_b:
        _d, rb_eq = task_b(env, rb_map)
        task_c(env, rb_map, rb_eq)
    if not args.only_b:
        task_d(env, rb_map)
    print(f"\n明细已落盘 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
