"""权重层质疑核查（2026-10-29）
=================================

针对外部质疑逐条取证，全部用**修复后口径**（T+1 open + PIT 修正）：

1. ``--mode conc``：权重集中度（最大单只分布 / HHI / 有效只数 / 逃过去重的相关对）
2. ``--mode ret``：收益集中度（去掉最好 N 周 + 时间聚集性 + 基准对照）
3. ``--mode maxw``：``max_weight`` 上限扫描（修复后口径重测）
4. ``--mode rule``：换权重规则 12 档对照（含 GMV / MDP / 风险预算 / 收缩 HRP）
5. ``--mode wf``：🚨 **滚动样本外选规则**（决定「全样本最优」是否可事前识别）
6. ``--mode phase``：周内相位（第 1~5 个交易日调仓）年化极差
7. ``--mode loo``：逐只剔除（leave-one-out）看是否单只驱动

用法::

    python -m scripts.portfolio.etf_weight_probe --mode conc,ret,maxw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backtest.engine import build_execution_split  # noqa: E402
from scripts.portfolio.etf_contest5_eval import (  # noqa: E402
    BEGIN,
    EXCLUDED_5,
    two_month_windows,
    yearly,
)
from scripts.portfolio.etf_contest_spec import FINAL  # noqa: E402
from scripts.portfolio.etf_rotation_panels import build_extra_panels  # noqa: E402
from scripts.portfolio.etf_rotation_probe import (  # noqa: E402
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)


def load_env():
    from data.cache import DataCache
    from data.datasource import create_datasource

    cache = DataCache(create_datasource())
    bundle = load_bundle(cache, WARMUP_BEGIN, None)
    _, close, amount, bench = bundle
    raw = cache.read_daily("etf")
    rc = raw["close"].unstack("code").reindex(index=close.index, columns=close.columns)
    ro = raw["open"].unstack("code").reindex(index=close.index, columns=close.columns)
    open_adj = (ro * (close / rc)).astype(np.float32)
    panels = build_extra_panels(close, build_panels(close, amount), rc)
    for x in EXCLUDED_5:
        panels["mask"].loc[:, x] = False
    mfs = pd.DatetimeIndex(
        pd.Series(close.index, index=close.index)
        .groupby(close.index.to_period("M")).first().to_numpy())
    return dict(bundle=bundle, panels=panels, close=close, amount=amount,
                bench=bench, open_adj=open_adj, mfs=mfs)


def run(env, spec: dict, rebalance_days=None, begin=BEGIN):
    """跑 close 口径取调仓日，再跑 T+1 open 口径（与 eval 同源）。"""
    r0 = run_variant(env["bundle"], env["panels"], spec, begin, None,
                     rebalance_days=rebalance_days)
    spl = build_execution_split(env["open_adj"], env["close"],
                                set(pd.DatetimeIndex(r0["rebalance_dates"])))
    r1 = run_variant(env["bundle"], env["panels"], spec, begin, None,
                     rebalance_days=rebalance_days,
                     execution_split=spl, execution_shift=True)
    return r0, r1


def hold_matrix(w: pd.DataFrame) -> pd.DataFrame:
    """只留多头腿（剔 CASH/DEF），并按调仓日取快照。"""
    held = [c for c in w.columns if c not in ("CASH", "DEF")]
    live = w[w.abs().sum(axis=1) > 1e-9]
    return live[held]


# ---------------------------------------------------------------- 1. 集中度
def mode_conc(env, spec):
    _, r1 = run(env, spec)
    hw = hold_matrix(r1["weights"])
    mw = hw.max(axis=1)
    hhi = (hw ** 2).sum(axis=1)
    n_eff = 1.0 / hhi.replace(0, np.nan)
    n_hold = (hw.abs() > 1e-9).sum(axis=1)
    print("\n== 1. 权重集中度（每日快照，T+1 open 口径）==")
    print(f"  样本天数 {len(hw)}  调仓日 {r1['n_rebalance']}")
    print(f"  最大单只权重：中位 {mw.median():.1%}  p90 {mw.quantile(.9):.1%}  "
          f"最大 {mw.max():.1%}")
    print(f"  有效只数 1/Σw²：中位 {n_eff.median():.2f}  p90 {n_eff.quantile(.9):.2f}")
    print(f"  持仓只数：中位 {n_hold.median():.0f}  均值 {n_hold.mean():.2f}")
    print("  单只权重超阈值的天数占比：", end="")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        print(f"  >{t:.0%}:{(mw > t).mean():.1%}", end="")
    print()

    # 逃过 dedup 的相关对：在调仓日算持仓两两 120 日相关
    print("\n== 1b. 持仓内部相关（120 日，与 _dedup 同窗）==")
    rib = r1["rebalance_dates"]
    rets = env["close"].pct_change(fill_method=None)
    pairs = []
    for d in rib:
        cols = [c for c in hw.columns if d in hw.index and hw.at[d, c] > 1e-9]
        if len(cols) < 2:
            continue
        win = rets.loc[:d].tail(120)[cols].dropna(how="all")
        if len(win) < 40:
            continue
        c = win.corr(min_periods=40)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                v = c.at[a, b]
                if pd.notna(v):
                    pairs.append({"date": d, "a": a, "b": b, "corr": float(v)})
    p = pd.DataFrame(pairs)
    print(f"  样本 {len(p)} 对；中位相关 {p['corr'].median():.3f}  "
          f"p90 {p['corr'].quantile(.9):.3f}  最大 {p['corr'].max():.3f}")
    print(f"  ≥0.90 应对去重却仍同时持有：{int((p['corr'] >= 0.90).sum())} 对 "
          f"（占 {(p['corr'] >= 0.90).mean():.1%}）")
    near = p[(p["corr"] >= 0.85) & (p["corr"] < 0.90)].sort_values("corr")
    if len(near):
        print(f"  0.85~0.90 灰区 {len(near)} 对，最近 5 条：")
        for _, x in near.head(5).iterrows():
            print(f"    {x['date']:%Y-%m-%d}  {x['a']} vs {x['b']}  ρ={x['corr']:.4f}")
    # 🚨 最关键的一条：真实**风险贡献**（不是权重）。名为「风险平价」，若风险
    # 贡献也不均，则这个命名是误导。RC_i = w_i (Σw)_i / (w'Σw)，等风险贡献 = 1/n。
    rets60 = env["close"].pct_change(fill_method=None)
    rcs = []
    for d in rib:
        if d not in hw.index:
            continue
        row = hw.loc[d]
        row = row[row > 1e-9]
        cols = list(row.index)
        if len(cols) < 2:
            continue
        cov = rets60.loc[:d].tail(60)[cols].dropna(how="all").cov().to_numpy() * 252
        wv = row.reindex(cols).to_numpy()
        wv = wv / wv.sum()
        port = float(wv @ cov @ wv)
        if port <= 0:
            continue
        rc = wv * (cov @ wv) / port
        rcs.append({"date": d, "max_w": float(wv.max()),
                    "max_rc": float(rc.max()), "gap": float(rc.max() - rc.min())})
    rc = pd.DataFrame(rcs)
    if len(rc):
        print("\n== 1c. 真实风险贡献 RC（60 日协方差，与 _risk_parity 同窗）==")
        print(f"  最大单只权重    中位 {rc['max_w'].median():.1%}")
        print(f"  最大单只风险贡献 中位 {rc['max_rc'].median():.1%}  "
              f"p90 {rc['max_rc'].quantile(.9):.1%}")
        print(f"  风险贡献极差(max−min) 中位 {rc['gap'].median():.1%}")
        print(f"  「等风险贡献」应为每只 {1 / 3:.1%} ⇒ 实测最大单只 "
              f"{rc['max_rc'].median():.0%} ⇒ **不是风险平价**")

    if len(p):
        top = p.sort_values("corr", ascending=False).head(3)
        for _, x in top.iterrows():
            print(f"    最高：{x['date']:%Y-%m-%d}  {x['a']} vs {x['b']}  ρ={x['corr']:.4f}")
    return dict(maxw_median=float(mw.median()), maxw_p90=float(mw.quantile(.9)),
                maxw_max=float(mw.max()), n_eff_median=float(n_eff.median()),
                corr_median=float(p["corr"].median()) if len(p) else float("nan"))


# ---------------------------------------------------------------- 2. 收益集中度
def _drop_best_weeks(eq: pd.Series, ns=(0, 5, 10, 20, 30)) -> pd.DataFrame:
    r = eq.pct_change().dropna()
    wk = (1.0 + r).groupby(r.index.to_period("W")).prod() - 1.0
    rows = []
    for n in ns:
        x = wk.copy()
        if n:
            x = x.drop(x.nlargest(n).index)
        rows.append({"n_drop": n, "total": float((1.0 + x).prod() - 1.0),
                     "n_weeks": len(x)})
    return pd.DataFrame(rows)


def mode_ret(env, spec):
    _, r1 = run(env, spec)
    eq = r1["equity"]
    print("\n== 2. 收益集中度：去掉最好的 N 周 ==")
    t = _drop_best_weeks(eq)
    base = t.loc[0, "total"]
    print(f"  全样本 {t.loc[0,'n_weeks']} 周，总收益 {base:.0%}")
    for _, x in t.iterrows():
        if x["n_drop"] == 0:
            continue
        print(f"  去掉最好 {int(x['n_drop']):>2} 周（{x['n_drop']/t.loc[0,'n_weeks']:.1%} 时间）"
              f" ⇒ {x['total']:>7.0%}   （保留 {x['total']/base:.0%}）")

    # 基准对照：同池等权 buy&hold
    r_ew = env["close"].pct_change(fill_method=None).mean(axis=1)
    eq_ew = (1.0 + r_ew.fillna(0.0)).cumprod()
    print("\n  基准对照（同池 45 只等权持有）：")
    t2 = _drop_best_weeks(eq_ew)
    print(f"    全样本总收益 {t2.loc[0,'total']:.0%}")
    for _, x in t2.iterrows():
        if x["n_drop"] == 0:
            continue
        print(f"    去掉最好 {int(x['n_drop']):>2} 周 ⇒ {x['total']:>7.0%}"
              f"   （保留 {x['total']/t2.loc[0,'total']:.0%}）")

    # 时间聚集性：最好 20 周落在哪
    r = eq.pct_change().dropna()
    wk = (1.0 + r).groupby(r.index.to_period("W")).prod() - 1.0
    best20 = wk.nlargest(20)
    print("\n== 2b. 最好 20 周的时间分布（判断是「某几次行情」还是「各年都有」）==")
    by_year = best20.groupby(best20.index.year).size()
    print("  按年计数：", dict(by_year))
    print("  前 8 名：")
    for d, v in best20.head(8).items():
        print(f"    {d}  {v:+.2%}")
    # 相邻周合并（±2 周内视作同一波）
    idx = sorted(best20.index)
    clusters, cur = [], [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if (b.start_time - a.end_time).days <= 21:
            cur.append(b)
        else:
            clusters.append(cur)
            cur = [b]
    clusters.append(cur)
    print(f"  合并相邻（±3 周）后共 {len(clusters)} 波，"
          f"最大一波 {max(len(c) for c in clusters)} 周")
    return dict(total=float(base))


# ---------------------------------------------------------------- 3. max_weight
def mode_maxw(env, spec):
    print("\n== 3. max_weight 上限扫描（修复后口径：T+1 open + PIT）==")
    print(f"  {'上限':<8}{'年化':>9}{'SR':>7}{'MDD':>9}{'2月均值':>10}"
          f"{'中位':>9}{'p10':>9}{'最差':>9}{'最大单只':>9}")
    rows = []
    for mwv in (None, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.34):
        s = dict(spec)
        s["max_weight"] = mwv
        r0, r1 = run(env, s)
        hw = hold_matrix(r1["weights"])
        wt = two_month_windows(r1["equity"], env["mfs"])
        lab = "无上限" if mwv is None else f"{mwv:.2f}"
        print(f"  {lab:<8}{r1['annual_return']:>9.2%}{r1['sharpe']:>7.2f}"
              f"{r1['max_drawdown']:>9.2%}{wt.mean():>+10.2%}{wt.median():>+9.2%}"
              f"{wt.quantile(.1):>+9.2%}{wt.min():>+9.2%}"
              f"{hw.max(axis=1).median():>9.1%}")
        rows.append({"max_weight": mwv, "ann": r1["annual_return"],
                     "sharpe": r1["sharpe"], "mdd": r1["max_drawdown"],
                     "w_mean": wt.mean(), "w_p10": wt.quantile(.1),
                     "w_worst": wt.min()})
    return pd.DataFrame(rows)


# ------------------------------------------------- 3b. 权重规则对照（P0 直接检验）
#: 全部候选权重规则。前 5 个是历史档（equal/inverse_vol/rank_decay 不看协方差），
#: 后 7 个是 2026-10-29 新增的「有理论依据但没测过」的规则。
ALL_RULES = ("hrp", "erc", "inv_var", "inverse_vol", "equal", "rank_decay",
             "min_var", "max_div", "risk_budget",
             "hrp_shrunk", "hrp_avgcorr", "hrp_ew", "hrp_anticorr")


def mode_rule(env, spec, rules=None):
    """同一选标，换权重规则（默认 12 档）。

    若 HRP 相对真风控规则（erc / 逆方差 / 等权 / GMV / MDP）有大幅超出，则超出
    部分就是「押最不合群那只」的赌注收益，不是风险平价的收益。
    """
    rules = list(rules) if rules else list(ALL_RULES)
    print("\n== 3b. 同一选标下换权重规则（修复后口径，T+1 open）==")
    print(f"  {'规则':<14}{'年化':>9}{'SR':>7}{'MDD':>9}{'2月均值':>10}"
          f"{'p10':>9}{'最差':>9}{'最大单只':>9}{'有效只数':>9}{'ΔvsHRP':>9}")
    rows = []
    for wt in rules:
        s = dict(spec)
        s["weighting"] = wt
        _r0, r1 = run(env, s)
        hw = hold_matrix(r1["weights"])
        mw = hw.max(axis=1)
        neff = 1.0 / (hw ** 2).sum(axis=1).replace(0, np.nan)
        w2 = two_month_windows(r1["equity"], env["mfs"])
        rows.append({"rule": wt, "ann": r1["annual_return"],
                     "sharpe": r1["sharpe"], "mdd": r1["max_drawdown"],
                     "w_mean": w2.mean(), "w_p10": w2.quantile(.1),
                     "w_worst": w2.min(), "maxw": float(mw.median()),
                     "n_eff": float(neff.median()),
                     "equity": r1["equity"]})
    d = pd.DataFrame(rows)
    hrp = float(d.loc[d.rule == "hrp", "ann"].iloc[0]) if "hrp" in set(rules) else float("nan")
    for _, x in d.iterrows():
        print(f"  {x['rule']:<14}{x['ann']:>9.2%}{x['sharpe']:>7.2f}"
              f"{x['mdd']:>9.2%}{x['w_mean']:>+10.2%}"
              f"{x['w_p10']:>+9.2%}{x['w_worst']:>+9.2%}"
              f"{x['maxw']:>9.1%}{x['n_eff']:>9.2f}"
              f"{(x['ann'] - hrp) * 100:>+8.2f}pp")
    print("\n  ⚠️ 全样本最优 = 12 档里挑出来的，含选择偏差 ⇒ 必须看 --mode wf 的样本外结果")
    for r in ("erc", "inv_var", "equal", "min_var", "max_div"):
        if r in set(rules):
            v = float(d.loc[d.rule == r, "ann"].iloc[0])
            print(f"  HRP − {r:<10} = {(hrp - v) * 100:>+7.2f}pp")

    # 分年度：判断优势是「每年都有」还是「某一两年」
    keep = [c for c in ("hrp", "erc", "min_var", "max_div", "risk_budget", "equal")
            if c in set(rules)]
    print("\n  分年度年化（同一选标，只换权重）：")
    yy = {k: yearly(v) for k, v in
          ((x["rule"], x["equity"]) for _, x in d.iterrows() if x["rule"] in keep)}
    yr = pd.DataFrame(yy)[keep]
    print("    " + f"{'年份':<8}" + "".join(f"{c:>12}" for c in yr.columns))
    for y, row in yr.iterrows():
        print(f"    {y:<8}" + "".join(f"{row[c]:>11.2%}" for c in yr.columns)
              + (f"{row['hrp'] - row['erc']:>+12.2%}" if "hrp" in keep and "erc" in keep else ""))
    d = d.drop(columns=["equity"])
    d.attrs["yearly"] = yr
    yr.to_csv("reports/etf_contest5/weight_probe_yearly.csv")
    return d


def nonoverlap_periods(eq: pd.Series, mfs: pd.DatetimeIndex,
                       step: int = 2) -> pd.Series:
    """非重叠 ``step`` 月期收益（索引 = 期初）。与 ``two_month_windows`` 的区别：
    后者每月滚动 ⇒ 相邻窗口共享 1 个月收益，**不能用于 walk-forward 拼装**
    （重叠期的收益会被重复计入，且训练/测试不再独立）。
    """
    out: dict[pd.Timestamp, float] = {}
    for i in range(0, len(mfs) - step, step):
        t0, t1 = mfs[i], mfs[i + step]
        if t0 not in eq.index or t1 not in eq.index:
            continue
        if len(eq.loc[t0:t1]) < 20:
            continue
        out[t0] = float(eq.loc[t1] / eq.loc[t0] - 1.0)
    return pd.Series(out)


def _ann_from_periods(rets: pd.Series, months_per_period: int = 2) -> float:
    """由非重叠期收益复利还原年化。"""
    if len(rets) == 0:
        return float("nan")
    tot = float(np.prod(1.0 + rets.to_numpy()))
    years = len(rets) * months_per_period / 12.0
    if years <= 0 or tot <= 0:
        return float("nan")
    return tot ** (1.0 / years) - 1.0


def mode_wf(env, spec, rules=None, min_train: int = 8, metric: str = "sharpe"):
    """🚨 决定性检验：**滚动样本外**选权重规则。

    做法：对每条规则跑全样本 → 取**非重叠** 2 月期收益（41 期）→ 对第 i 期，
    只用前 i 期的历史表现（``metric`` = sharpe / mean）选一条规则，用它的第 i 期
    收益 → 拼成「自适应选规则」的净值。

    要回答的问题：**HRP 的全样本第一，是可以在事前被选出来的吗？**
    - 若 WF 组合 ≈ 固定 HRP ⇒ 规则选择不花成本，HRP 可信；
    - 若 WF 明显更差 ⇒ 「12 档挑最好」这个动作本身在毁价值，30.63% 是事后挑的；
    - 若 WF 选出来的多数不是 HRP ⇒ HRP 的优势在事前不可识别。
    """
    rules = list(rules) if rules else list(ALL_RULES)
    print(f"\n== 3e. 滚动样本外选规则（walk-forward，metric={metric}，"
          f"min_train={min_train} 期）==")
    pr = {}
    for wt in rules:
        s = dict(spec)
        s["weighting"] = wt
        _r0, r1 = run(env, s)
        pr[wt] = nonoverlap_periods(r1["equity"], env["mfs"])
    P = pd.DataFrame(pr).dropna(how="any")
    print(f"  非重叠期 {len(P)} 个（步长 2 月）")
    if len(P) <= min_train + 2:
        print("  期数不足，跳过")
        return pd.DataFrame()

    def _score(hist: pd.DataFrame) -> pd.Series:
        if metric == "mean":
            return hist.mean()
        sd = hist.std(ddof=1).replace(0.0, np.nan)
        return hist.mean() / sd

    picks, wf = [], []
    for i in range(min_train, len(P)):
        sc = _score(P.iloc[: i]).dropna()
        if sc.empty:
            continue
        best = str(sc.idxmax())
        picks.append({"date": P.index[i], "rule": best, "ret": float(P.iloc[i][best])})
        wf.append(float(P.iloc[i][best]))
    wf_s = pd.Series(wf)
    print(f"  样本外 {len(wf_s)} 期")
    print(f"  {'方案':<22}{'年化':>9}{'期均值':>10}{'期标准差':>10}{'期SR':>8}")
    rows = []
    for wt in rules:
        r = P[wt].iloc[min_train:]
        rows.append({"scheme": f"固定 {wt}", "ann": _ann_from_periods(r),
                     "mean": r.mean(), "sd": r.std(ddof=1),
                     "sr": r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan,
                     "sharpe": np.nan})
    rows.append({"scheme": "🚨 WF 自适应", "ann": _ann_from_periods(wf_s),
                 "mean": wf_s.mean(), "sd": wf_s.std(ddof=1),
                 "sr": wf_s.mean() / wf_s.std(ddof=1),
                 "sharpe": np.nan})
    # 事后最优（上界，作弊参照）
    # 🚨 按**列**求均值（axis=0）= 每条规则的样本外平均；写成 axis=1 会得到
    # 「每期的横截面均值」，idxmax 返回的是时间戳而不是规则名。
    best_post = P.iloc[min_train:].mean().idxmax()
    rows.append({"scheme": f"事后最优（{best_post}，作弊）",
                 "ann": _ann_from_periods(P[best_post].iloc[min_train:]),
                 "mean": P[best_post].iloc[min_train:].mean(),
                 "sd": P[best_post].iloc[min_train:].std(ddof=1),
                 "sr": np.nan, "sharpe": np.nan})
    d = pd.DataFrame(rows)
    for _, x in d.iterrows():
        print(f"  {x['scheme']:<22}{x['ann']:>9.2%}{x['mean']:>+10.2%}"
              f"{x['sd']:>10.2%}{x['sr']:>8.2f}")
    pk = pd.DataFrame(picks)
    hrp_share = (pk["rule"] == "hrp").mean() if len(pk) else float("nan")
    print(f"\n  WF 选中 hrp 的比例 {hrp_share:.1%}；各规则被选次数：")
    print("    " + str(pk["rule"].value_counts().to_dict()))
    wf_ann = _ann_from_periods(wf_s)
    for ref in ("hrp", "erc"):
        if ref in P.columns:
            gap = wf_ann - _ann_from_periods(P[ref].iloc[min_train:])
            print(f"  WF − 固定{ref:<6} = {gap * 100:+.2f}pp")

    # 逐年：当年最优规则（看 HRP 是不是稳定第一）
    print("\n  逐年最优规则（用**当年**全样本排名，事后口径，看稳定性）：")
    idx = P.index
    for y in sorted({d_.year for d_ in idx}):
        sub = P[[d_.year == y for d_ in idx]]
        if len(sub) < 3:
            continue
        sc = sub.mean().sort_values(ascending=False)
        print(f"    {y}  n={len(sub):<2} 最优 {sc.index[0]:<12}{sc.iloc[0]:+.2%}/期"
              f"   最差 {sc.index[-1]:<12}{sc.iloc[-1]:+.2%}/期"
              f"   hrp 名次 {list(sc.index).index('hrp') + 1}/{len(sc)}")
    pk.to_csv("reports/etf_contest5/weight_probe_wf_picks.csv", index=False)
    P.to_csv("reports/etf_contest5/weight_probe_periods.csv")
    return d


def _run_cls(env, spec: dict, cls):
    """``run()`` 的变体：允许替换策略类（安慰剂用）。口径完全一致。"""
    r0 = run_variant(env["bundle"], env["panels"], spec, BEGIN, None,
                     strategy_cls=cls)
    spl = build_execution_split(env["open_adj"], env["close"],
                                set(pd.DatetimeIndex(r0["rebalance_dates"])))
    r1 = run_variant(env["bundle"], env["panels"], spec, BEGIN, None,
                     execution_split=spl, execution_shift=True,
                     strategy_cls=cls)
    return r0, r1


def _rand_single_cls(seed: int):
    """「随机押一只」策略类：选标与定版一致，权重 = 随机挑一只给 100%。

    ⚠️ 这个对照**不干净**：100% 单押的波动远高于 HRP 的 79%，几何收益会被
    波动拖累压低。真正的干净对照是下面的 ``_perm_weight_cls``。
    """
    from strategy.etf_rotation import EtfRotationAdvanced

    class _RandSingle(EtfRotationAdvanced):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._rng = np.random.default_rng(seed)

        def _weights(self, factor_values, date):
            w = super()._weights(factor_values, date)
            live = [c for c in w.index
                    if c != self.cash_code and float(w.get(c, 0.0)) > 1e-12]
            if len(live) <= 1:
                return w
            return pd.Series({str(self._rng.choice(live)): 1.0})

    return _RandSingle


def _perm_weight_cls(seed: int):
    """🚨 **干净安慰剂**：把 HRP 算出的权重向量**随机置换**到选中的几只上。

    集中度、波动率、换手结构全部保持不变，唯一被消除的是「把最多的钱压在
    哪一只」这个选择。若置换版与真 HRP 无差异 ⇒ HRP 的「挑」没有信息，
    收益全部来自集中度本身。
    """
    from strategy.etf_rotation import EtfRotationAdvanced

    class _PermWeight(EtfRotationAdvanced):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._rng = np.random.default_rng(seed)

        def _weights(self, factor_values, date):
            w = super()._weights(factor_values, date)
            live = [c for c in w.index
                    if c != self.cash_code and float(w.get(c, 0.0)) > 1e-12]
            if len(live) <= 1:
                return w
            vals = w[live].to_numpy().copy()
            self._rng.shuffle(vals)
            out = pd.Series(vals, index=live)
            if self.cash_code and self.cash_code in w.index:
                out[self.cash_code] = float(w[self.cash_code])
            return out

    return _PermWeight


def mode_placebo(env, spec, n_seed: int = 20, rules=None):
    """🚨 零信息对照：**HRP 的「挑」到底有没有信息**。

    HRP 在 n=3 时给 ~0.79/0.10/0.10，已接近单押。必须把两件事拆开：
    ① **集中**（把 80% 压到一只上）本身贡献多少；
    ② **「把最多的钱压在哪一只」**贡献多少。

    两个对照：
    - ``perm``（**干净**）：把 HRP 的权重向量随机置换到选中的几只上 ——
      集中度/波动/换手结构完全不变，只去掉「挑哪只」；
    - ``single``（不干净）：随机挑一只 100% —— 集中度更高，波动拖累不同，
      只能当参考。
    """
    print(f"\n== 3f. 零信息对照（n_seed={n_seed}）==")
    _r0, hrp = run(env, spec)
    print(f"  HRP（定版）年化 {hrp['annual_return']:.2%}  SR {hrp['sharpe']:.2f}  "
          f"MDD {hrp['max_drawdown']:.2%}  ← 对照基准")
    out = {}
    # 方向镜像：同样集中度、同样换手结构，只是压给「最相关」那只
    s_inv = dict(spec)
    s_inv["weighting"] = "hrp_anticorr"
    _b, r_inv = run(env, s_inv)
    w2i = two_month_windows(r_inv["equity"], env["mfs"])
    print("\n  [anticorr] 压最相关那只（方向镜像，同换手结构）")
    print(f"    年化 {r_inv['annual_return']:.2%}  SR {r_inv['sharpe']:.2f}  "
          f"MDD {r_inv['max_drawdown']:.2%}  2月均值 {w2i.mean():+.2%}")
    print(f"    HRP − anticorr = {(hrp['annual_return'] - r_inv['annual_return']) * 100:+.2f}pp"
          f"（≈0 ⇒ 方向无信息，只是集中度；>0 ⇒ 「压最不合群」有信息）")
    out["anticorr"] = pd.DataFrame([{
        "kind": "anticorr", "seed": -1, "ann": r_inv["annual_return"],
        "sharpe": r_inv["sharpe"], "mdd": r_inv["max_drawdown"],
        "w_mean": w2i.mean()}])
    for kind, fn, lab in ((("perm"), _perm_weight_cls, "权重置换（干净）"),
                          (("single"), _rand_single_cls, "随机押一只（不干净，仅供参考）")):
        res = []
        for sd in range(n_seed):
            _a, r1 = _run_cls(env, spec, fn(sd))
            w2 = two_month_windows(r1["equity"], env["mfs"])
            res.append({"seed": sd, "ann": r1["annual_return"],
                        "sharpe": r1["sharpe"], "mdd": r1["max_drawdown"],
                        "w_mean": w2.mean()})
        d = pd.DataFrame(res)
        a = d["ann"]
        pct = float((a < hrp["annual_return"]).mean())
        print(f"\n  [{kind}] {lab}")
        print(f"    年化 min {a.min():.2%} / p25 {a.quantile(.25):.2%} / "
              f"**中位 {a.median():.2%}** / p75 {a.quantile(.75):.2%} / max {a.max():.2%}")
        print(f"    均值 {a.mean():.2%}（SR 中位 {d['sharpe'].median():.2f}，"
              f"MDD 中位 {d['mdd'].median():.2%}）")
        print(f"    HRP {hrp['annual_return']:.2%} 位于该分布 **{pct:.0%} 分位**"
              f"；差 {(hrp['annual_return'] - a.mean()) * 100:+.2f}pp"
              f"（≈50 分位/差≈0 ⇒ 挑选无信息）")
        d.insert(0, "kind", kind)
        out[kind] = d
    d = pd.concat(out.values(), ignore_index=True)
    d.to_csv("reports/etf_contest5/weight_probe_placebo.csv", index=False)
    return d


def mode_covwin(env, spec, windows=(30, 60, 120, 250)):
    """协方差窗长敏感性：60 日这个数是拍的，不是测的。"""
    print("\n== 3g. 协方差窗长敏感性（HRP）==")
    print(f"  {'窗口':<10}{'年化':>9}{'SR':>7}{'MDD':>9}{'2月均值':>10}{'最大单只':>9}")
    rows = []
    for cw in windows:
        s = dict(spec)
        s["cov_window"] = int(cw)
        _r0, r1 = run(env, s)
        hw = hold_matrix(r1["weights"])
        w2 = two_month_windows(r1["equity"], env["mfs"])
        print(f"  {cw:<10}{r1['annual_return']:>9.2%}{r1['sharpe']:>7.2f}"
              f"{r1['max_drawdown']:>9.2%}{w2.mean():>+10.2%}"
              f"{hw.max(axis=1).median():>9.1%}")
        rows.append({"cov_window": cw, "ann": r1["annual_return"],
                     "sharpe": r1["sharpe"], "mdd": r1["max_drawdown"],
                     "w_mean": w2.mean(),
                     "maxw": float(hw.max(axis=1).median())})
    d = pd.DataFrame(rows)
    print(f"  极差 {(d['ann'].max() - d['ann'].min()) * 100:.2f}pp"
          f"（窗长选哪档本身就是个自由度）")
    return d


def mode_focus(env, spec, topn: int = 10):
    """HRP 到底押在**什么资产**上（回答「它是不是在押某个赛道」）。

    输出每个被押标的的出现次数、占比，以及它在**全池内的平均相关**
    （越低越「独特」）和它的全期年化（押对了没有）。
    """
    print("\n== 3h. HRP 押注落在哪些标的 ==")
    _r0, r1 = run(env, spec)
    hw = hold_matrix(r1["weights"])
    rows = []
    for d in r1["rebalance_dates"]:
        if d not in hw.index:
            continue
        row = hw.loc[d]
        row = row[row > 1e-9]
        if row.empty:
            continue
        rows.append({"date": d, "top": str(row.idxmax()), "w": float(row.max())})
    f = pd.DataFrame(rows)
    rets = env["close"].pct_change(fill_method=None)
    corr = rets.corr(min_periods=120)
    np.fill_diagonal(corr.values, np.nan)
    mc = corr.mean()                       # 与全池其余标的的平均相关
    ann = (1.0 + rets).prod() ** (252.0 / max(len(rets), 1)) - 1.0
    cnt = f["top"].value_counts()
    tot = len(f)
    print(f"  调仓日 {tot}；被押过 {len(cnt)} 只（池内 45 只）")
    print(f"  {'代码':<12}{'次数':>6}{'占比':>8}{'池内平均相关':>13}{'全期年化':>10}")
    for code, n in cnt.head(topn).items():
        print(f"  {code:<12}{n:>6}{n / tot:>8.1%}"
              f"{mc.get(code, float('nan')):>13.3f}"
              f"{ann.get(code, float('nan')):>10.2%}")
    print(f"  前 {topn} 名合计占 {cnt.head(topn).sum() / tot:.1%}")
    print(f"  被押标的的池内平均相关 中位 {mc[cnt.index].median():.3f}"
          f" vs 全池 中位 {mc.median():.3f}")
    print(f"  被押标的的全期年化 中位 {ann[cnt.index].median():.2%}"
          f" vs 全池 中位 {ann.median():.2%}")
    d = pd.DataFrame({"code": cnt.index, "n": cnt.to_numpy()})
    d["share"] = d["n"] / tot
    d["mean_corr"] = d["code"].map(mc)
    d["ann"] = d["code"].map(ann)
    return d


def mode_struct(env, spec):
    """诊断：HRP 押中的那只，是不是就是动量第一名？

    若「最大权重标的」几乎总是 rank1，则 HRP 的真实效果是把 Top3 退回 Top1；
    再跑 top_k=1/2/3 等权对照，看 30.63% 能否被「Top1 单押」解释。
    """
    r0, r1 = run(env, spec)
    sig = spec["signal"]
    fac = env["panels"][sig]
    mask = env["panels"]["mask"]
    fac = fac.where(mask)
    hw = hold_matrix(r1["weights"])
    ranks = []
    for d in r1["rebalance_dates"]:
        if d not in hw.index or d not in fac.index:
            continue
        row = hw.loc[d]
        row = row[row > 1e-9]
        if row.empty:
            continue
        top = row.idxmax()
        f = fac.loc[d].dropna()
        f = f[[c for c in f.index if c in hw.columns]]
        if top not in f.index or len(f) < 2:
            continue
        order = list(f.sort_values(ascending=False).index)
        ranks.append({"date": d, "top": top, "rank": order.index(top) + 1,
                      "w": float(row.max())})
    rk = pd.DataFrame(ranks)
    print("\n== 3c. 最大权重标的的动量排名 ==")
    print(f"  调仓日 {len(rk)}；rank=1 占比 {(rk['rank'] == 1).mean():.1%}"
          f"，rank≤2 占比 {(rk['rank'] <= 2).mean():.1%}")
    print("  分布：", rk["rank"].value_counts().sort_index().to_dict())

    print("\n== 3c-2. top_k 等权对照（看 30.63% 是否可被 Top1 单押解释）==")
    print(f"  {'档位':<16}{'年化':>9}{'SR':>7}{'MDD':>9}{'2月均值':>10}{'最大单只':>9}")
    for tk in (1, 2, 3):
        for wt_rule in ("equal", "hrp"):
            if tk == 1 and wt_rule == "hrp":
                continue          # n=1 时 HRP 恒 100%，与等权同
            s = dict(spec)
            s["top_k"] = tk
            s["weighting"] = wt_rule
            _, r = run(env, s)
            h = hold_matrix(r["weights"])
            w2 = two_month_windows(r["equity"], env["mfs"])
            print(f"  top{tk}+{wt_rule:<10}{r['annual_return']:>9.2%}{r['sharpe']:>7.2f}"
                  f"{r['max_drawdown']:>9.2%}{w2.mean():>+10.2%}"
                  f"{h.max(axis=1).median():>9.1%}")
    return rk


# ------------------------------------------------- 3d. 归因：押注有信息吗
def mode_attr(env, spec):
    """检验「押最不合群那只」是否含预测力。

    对每个调仓日取实际持仓，记录每只的动量排名、与另两只的平均 120 日相关、
    以及持有一期的前瞻收益。若「最少相关者」的前瞻收益显著高于三只均值，
    则 HRP 的集中不是纯赌注；若相等，则 13.55pp 只能归因于集中本身（运气/杠杆化）。
    """
    _, r1 = run(env, spec)
    reb = list(r1["rebalance_dates"])
    hw = hold_matrix(r1["weights"])
    close = env["close"]
    rets = close.pct_change(fill_method=None)
    fac = env["panels"][spec["signal"]].where(env["panels"]["mask"])
    rows = []
    for i, E in enumerate(reb[:-1]):
        E2 = reb[i + 1]
        if E not in hw.index or E2 not in close.index or E not in fac.index:
            continue
        row = hw.loc[E]
        row = row[row > 1e-9]
        cols = list(row.index)
        if len(cols) < 2:
            continue
        win = rets.loc[:E].tail(120)[cols].dropna(how="all")
        if len(win) < 40:
            continue
        cm = win.corr(min_periods=40)
        f = fac.loc[E].dropna()
        order = [c for c in f.sort_values(ascending=False).index if c in cols]
        for c in cols:
            others = [x for x in cols if x != c]
            rc = [cm.at[c, x] for x in others if pd.notna(cm.at[c, x])]
            if not rc or pd.isna(close.at[E2, c]) or pd.isna(close.at[E, c]):
                continue
            rows.append({"date": E, "code": c, "w": float(row[c]),
                         "rank": order.index(c) + 1 if c in order else 99,
                         "mean_corr": float(np.mean(rc)),
                         "fwd": float(close.at[E2, c] / close.at[E, c] - 1.0)})
    d = pd.DataFrame(rows)
    g = d.groupby("date")
    print(f"\n== 3d. 押注归因（{d['date'].nunique()} 期，{len(d)} 个「标的×期」观测）==")
    # 每期各统计量
    per = g.apply(lambda x: pd.Series({
        "fwd_mean3": x["fwd"].mean(),
        "fwd_maxw": x.loc[x["w"].idxmax(), "fwd"],
        "fwd_mincorr": x.loc[x["mean_corr"].idxmin(), "fwd"],
        "fwd_maxcorr": x.loc[x["mean_corr"].idxmax(), "fwd"],
        "fwd_rank1": x.loc[x["rank"].idxmin(), "fwd"],
        "w_max": x["w"].max(),
        "corr_spread": x["mean_corr"].max() - x["mean_corr"].min(),
    }), include_groups=False)
    print(f"  {'选择方式':<16}{'期均前瞻收益':>13}{'相对三只均值':>13}{'胜率':>9}{'t':>7}")
    for k, lab in (("fwd_maxw", "HRP 实际押注"), ("fwd_mincorr", "押最不相关"),
                   ("fwd_maxcorr", "押最相关"), ("fwd_rank1", "押动量第一")):
        v = per[k] - per["fwd_mean3"]
        t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if v.std(ddof=1) > 0 else np.nan
        print(f"  {lab:<16}{per[k].mean():>+13.2%}{v.mean():>+13.2%}"
              f"{(v > 0).mean():>9.1%}{t:>7.2f}")
    print(f"  三只等权基准期均 {per['fwd_mean3'].mean():+.2%}")
    # 按平均相关分档
    d["corr_q"] = pd.qcut(d["mean_corr"], 3, labels=["低相关", "中", "高相关"])
    print("\n  按「与另两只平均相关」分档的前瞻收益：")
    for q, x in d.groupby("corr_q", observed=True):
        print(f"    {q:<6} n={len(x):<5} 平均ρ={x['mean_corr'].mean():.3f}  "
              f"期均前瞻 {x['fwd'].mean():+.2%}  中位 {x['fwd'].median():+.2%}")
    print("\n  按动量排名分档的前瞻收益：")
    for k, x in d.groupby(d["rank"].clip(upper=4)):
        lab = f"rank{k}" if k < 4 else "rank4+"
        print(f"    {lab:<7} n={len(x):<5} 期均前瞻 {x['fwd'].mean():+.2%}  "
              f"中位 {x['fwd'].median():+.2%}")
    return d


# ---------------------------------------------------------------- 4. 相位
def mode_phase(env, spec):
    print("\n== 4. 周内相位（第 k 个交易日调仓，修复后口径）==")
    days = list(env["close"].index)
    per = pd.Series(days, index=pd.DatetimeIndex(days)).groupby(
        pd.DatetimeIndex(days).to_period("W"))
    print(f"  {'相位':<8}{'年化':>9}{'SR':>7}{'MDD':>9}{'2月均值':>10}")
    out = []
    for k in range(5):
        rb = set()
        for _, g in per:
            v = list(g)
            if len(v) > k:
                rb.add(v[k])
        r0, r1 = run(env, spec, rebalance_days=rb)
        wt = two_month_windows(r1["equity"], env["mfs"])
        print(f"  k={k+1:<6}{r1['annual_return']:>9.2%}{r1['sharpe']:>7.2f}"
              f"{r1['max_drawdown']:>9.2%}{wt.mean():>+10.2%}")
        out.append({"phase": k + 1, "ann": r1["annual_return"],
                    "sharpe": r1["sharpe"], "w_mean": wt.mean()})
    d = pd.DataFrame(out)
    print(f"  极差 {(d['ann'].max() - d['ann'].min()) * 100:.2f}pp（{d['ann'].min():.2%} ~ "
          f"{d['ann'].max():.2%}），均值 {d['ann'].mean():.2%}")
    z = (d["ann"].max() - d["ann"].mean()) / d["ann"].std(ddof=1)
    print(f"  最高相位相对均值 z = {z:.2f}（n=5，自由度极低）")
    return d


# ---------------------------------------------------------------- 5. LOO
def mode_loo(env, spec):
    _, base = run(env, spec)
    hw = hold_matrix(base["weights"])
    appear = (hw > 1e-9).any(axis=0)
    codes = [c for c in hw.columns if appear[c]]
    print(f"\n== 5. leave-one-out（{len(codes)} 只曾入选）==")
    print(f"  基线年化 {base['annual_return']:.2%}")
    rows = []
    for c in codes:
        m = env["panels"]["mask"].copy()
        m.loc[:, c] = False
        env["panels"]["mask"] = m
        try:
            _, r = run(env, spec)
            rows.append({"code": c, "ann": r["annual_return"],
                         "d_ann": r["annual_return"] - base["annual_return"]})
        except Exception as e:  # noqa: BLE001
            print(f"    {c} 失败：{e}")
        finally:
            env["panels"]["mask"].loc[:, c] = True
    d = pd.DataFrame(rows).sort_values("d_ann")
    print(f"  {'代码':<14}{'年化':>9}{'Δ年化':>10}")
    for _, x in d.iterrows():
        print(f"  {x['code']:<14}{x['ann']:>9.2%}{x['d_ann']:>+10.2%}")
    print(f"  正贡献（剔除后变差）只数 {(d['d_ann'] < 0).sum()} / {len(d)}；"
          f"最大单一依赖 {-d['d_ann'].min() * 100:.2f}pp")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="权重层质疑核查")
    ap.add_argument("--mode", default="conc,ret,maxw",
                    help="conc,ret,maxw,rule,struct,attr,phase,loo,wf,placebo,covwin,focus")
    ap.add_argument("--rules", default="",
                    help="逗号分隔的权重规则（默认 12 档全跑，见 ALL_RULES）")
    ap.add_argument("--min-train", type=int, default=8,
                    help="walk-forward 最少训练期数（默认 8 期 = 16 个月）")
    ap.add_argument("--wf-metric", default="sharpe", choices=["sharpe", "mean"],
                    help="walk-forward 选规则用的指标")
    ap.add_argument("--placebo", type=int, default=20,
                    help="随机单押安慰剂的 seed 数")
    ap.add_argument("--csv-dir", default="reports/etf_contest5")
    args = ap.parse_args(argv)
    modes = [s.strip() for s in args.mode.split(",")]
    env = load_env()
    spec = dict(FINAL)
    print(f"窗口 {BEGIN}→末日；spec = FINAL（三锋）")
    out = {}
    if "conc" in modes:
        out["conc"] = mode_conc(env, spec)
    if "ret" in modes:
        out["ret"] = mode_ret(env, spec)
    if "maxw" in modes:
        out["maxw"] = mode_maxw(env, spec)
    rules = [x.strip() for x in args.rules.split(",") if x.strip()] or None
    if "rule" in modes:
        out["rule"] = mode_rule(env, spec, rules=rules)
    if "struct" in modes:
        out["struct"] = mode_struct(env, spec)
    if "attr" in modes:
        out["attr"] = mode_attr(env, spec)
    if "phase" in modes:
        out["phase"] = mode_phase(env, spec)
    if "loo" in modes:
        out["loo"] = mode_loo(env, spec)
    if "wf" in modes:
        out["wf"] = mode_wf(env, spec, rules=rules,
                            min_train=args.min_train, metric=args.wf_metric)
    if "placebo" in modes:
        out["placebo"] = mode_placebo(env, spec, n_seed=args.placebo, rules=rules)
    if "covwin" in modes:
        out["covwin"] = mode_covwin(env, spec)
    if "focus" in modes:
        out["focus"] = mode_focus(env, spec)
    d = Path(args.csv_dir)
    d.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(d / f"weight_probe_{k}.csv", index=False, encoding="utf-8-sig")
            print(f"落盘 → {d / f'weight_probe_{k}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
