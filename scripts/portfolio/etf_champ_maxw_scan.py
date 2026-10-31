"""ETF 磐石（``champ``）· 单只权重上限 ``max_weight`` 扫描（相位平均口径）。

## 为什么需要这个脚本（§5.1.3 的直接后续）

``reports/etf_contest5/README.md`` §5.1.3 查明：磐石对外口径「单只中位 17.7%」是
**分布中位数，不是典型仓位**。``inverse_vol`` 把国债 ETF（波动 ~2%）在入选 Top5 时
按 1/vol 顶到 **52.4% 中位**（``min_var`` 为 84.8%），29.6% 的交易日首位就是国债 ETF。
⇒ 磐石的低回撤机制是**利率暴露**，不是分散化；主要失效模式是「股债同跌」而非个股暴雷。

本脚本测那个唯一能直接改这件事的旋钮：给多头腿加 ``max_weight`` 上限。

## 关键结构（决定了本实验的语义）

1. ``_cap_weight`` 在 ``_weights`` 里按此顺序执行：
   ``多头腿归一化到 1 → cap + 按比例回补 → 乘 target_vol 缩放 → 余量给 cash_code``。
   因此 cap 只约束**多头腿内部**的比例。
2. 磐石的 ``defense="mix"`` ⇒ ``cash_code == "DEF"``，DEF 权重 = ``1 − gross``。
   ``gross`` 由 ``target_vol / vol.mean()`` 决定，**与权重规则无关**
   ⇒ cap 不改变总敞口，**它不是降杠杆**，是同一敞口下的风险再分配
   （与 §4.7.4「降杠杆支配」区分开的必要条件）。
3. 🚨 ``_cap_weight`` 有数学可行性保护：``cap * 腿数 < 1`` 时**静默返回不 cap**。
   对 Top5 意味着 cap 下限就是 **0.20（恰等权）**，低于它无效。有效候选少于 5 只的
   调仓日（去重后常见）同样会静默失效 ⇒ 本脚本额外输出 **cap 实际生效日占比**，
   否则「读数没动」会被误读成「cap 没用」。

## 判别量

- 相位平均 ``ann / sharpe / mdd / calmar``（长期实盘口径，与 §5.1.1 同源可对照）
- 集中度形状 ``maxw_p50/p75/p90/mean``、``>40%`` / ``>60%`` 占比、国债首位占比
- ``cap_on``：capped 多头腿与无 cap 版不同的交易日占比（生效性自检）
- 判否三件套（复用 ``etf_champ_phase_robust`` 口径）：月度配对 t、12 个月滚动最差回撤、
  换手增量

## 用法

    python -m scripts.portfolio.etf_champ_maxw_scan
    python -m scripts.portfolio.etf_champ_maxw_scan --caps 0,0.30,0.20
    python -m scripts.portfolio.etf_champ_maxw_scan --from-csv   # 只重算汇总

产物：``reports/etf_champ/maxw_scan.csv`` + 终端汇总表。
环境：**系统 Python 3.12**（``.venv`` 缺 ``AmazingData``，跑不了数据源）。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.portfolio.etf_champ_phase_matrix import (
    build_env,
    month_phase_rebalance_days,
    run_champ,
)
from scripts.portfolio.etf_champ_phase_robust import _monthly, _roll_mdd
from scripts.portfolio.etf_champ_signals import PROFILES

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "reports/etf_champ/maxw_scan.csv"

PHASES = (1, 2, 3, 4, 5)

#: cap 档位。0 = 定版无上限（对照）。0.20 是 Top5 的数学下限（= 等权）。
DEFAULT_CAPS = (0.0, 0.40, 0.35, 0.30, 0.25, 0.20)

#: 国债 ETF（用于判定「首位持仓是不是利率押注」，与 §5.1.3 同一集合）。
TREAS = {"511010.SH", "511260.SH", "511880.SH", "511990.SH"}

#: §5.1.1 相位平均读数（无 cap 定版），用于自检对齐。
PA_REF = {"ann": 0.1062, "sharpe": 0.87, "mdd": 0.1620, "calmar": 0.66,
          "maxw": 0.177}


def label(cap: float) -> str:
    """档位标签：``iv`` = 定版无上限，``iv@0.30`` = 加 30% 上限。"""
    return "iv" if cap <= 0 else f"iv@{cap:.2f}"


def _leg(w: pd.DataFrame) -> pd.DataFrame:
    """取多头腿（剔除 CASH / DEF），只保留真正持仓的日子。"""
    held = [c for c in w.columns if c not in ("CASH", "DEF")]
    live = w[w.abs().sum(axis=1) > 1e-9][held]
    return live


def risk_profile(live: pd.DataFrame, close: pd.DataFrame,
                 window: int = 60) -> tuple[float, float]:
    """返回 ``(多头腿年化波动中位, 国债在场时其风险贡献份额中位)``。

    这是本实验的关键解释量。``inverse_vol`` 让国债 ETF 拿到 50% 以上**资金**
    权重，但它的波动只有股票 ETF 的 ~1/10 ⇒ 其**风险**份额远低于资金份额。
    若该值显著小于资金份额，则 cap（按资金比例截断）砍掉的是「低风险高资金」
    资产 ⇒ 净效果是抬高组合风险，与「降风险」的初衷相反。
    波动中位数直接检验这一点，不依赖 MCTR 分解（后者在低相关资产上数值敏感）。
    """
    rets = close.pct_change(fill_method=None)
    uniq = live.drop_duplicates()          # 权重在两次调仓间恒定 ⇒ 只算唯一状态
    vals: list[float] = []
    vols: list[float] = []
    for d, row in uniq.iterrows():
        codes = [c for c in row.index if row[c] > 1e-9]
        tset = [c for c in codes if c in TREAS]
        if not tset or len(codes) < 2:
            continue
        sub = rets.loc[:d].tail(window).reindex(columns=codes).dropna()
        if len(sub) < 20:
            continue
        cov = sub.cov().to_numpy(dtype=float)
        w = row[codes].to_numpy(dtype=float)
        tot = w.sum()
        if tot <= 0:
            continue
        w = w / tot
        var = float(w @ cov @ w)
        if var <= 0:
            continue
        vols.append(float(np.sqrt(var * 252.0)))
        if tset:
            mctr = w * (cov @ w) / var      # Σ 份额 = 1
            vals.append(float(mctr[[codes.index(c) for c in tset]].sum()))
    v = float(np.median(vols)) if vols else float("nan")
    rc = float(np.median(vals)) if vals else float("nan")
    return v, rc


def _conc_stats(live: pd.DataFrame, base_live: pd.DataFrame | None,
                close: pd.DataFrame | None = None) -> dict:
    """集中度形状 + cap 生效日占比。"""
    if not len(live):
        return {}
    mw = live.max(axis=1)
    top = live.idxmax(axis=1)
    ist = top.isin(TREAS)
    lv, rc = risk_profile(live, close) if close is not None else (float("nan"),) * 2
    out = {
        "maxw_p50": float(mw.median()),
        "maxw_p75": float(mw.quantile(0.75)),
        "maxw_p90": float(mw.quantile(0.90)),
        "maxw_mean": float(mw.mean()),
        "share_gt40": float((mw > 0.40).mean()),
        "share_gt60": float((mw > 0.60).mean()),
        "treas_share": float(ist.mean()),
        "treas_maxw": float(mw[ist].median()) if ist.any() else float("nan"),
        "neff": float((1.0 / (live ** 2).sum(axis=1).replace(0, np.nan)).median()),
        "n_hold": float((live > 1e-9).sum(axis=1).median()),
        "gross": float(live.sum(axis=1).median()),
        "leg_vol": lv,
        "treas_rc": rc,
        "cap_on": float("nan"),
    }
    if base_live is not None and len(base_live):
        j = live.index.intersection(base_live.index)
        diff = (live.loc[j] - base_live.loc[j].reindex(columns=live.columns)
                .fillna(0.0)).abs().max(axis=1)
        out["cap_on"] = float((diff > 1e-6).mean())
    return out


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    """相位平均汇总。``mdd`` 是**正的深度** ⇒ 最差相位取 max。"""
    agg = d.groupby("cap_lbl", sort=False).agg(
        cap=("cap", "first"),
        ann_mean=("ann", "mean"), ann_min=("ann", "min"), ann_max=("ann", "max"),
        sr_mean=("sharpe", "mean"), sr_min=("sharpe", "min"),
        mdd_mean=("mdd", "mean"), mdd_worst=("mdd", "max"),
        turnover=("turnover", "mean"),
        maxw_p50=("maxw_p50", "median"), maxw_p75=("maxw_p75", "median"),
        maxw_p90=("maxw_p90", "median"), maxw_mean=("maxw_mean", "mean"),
        share_gt40=("share_gt40", "mean"), share_gt60=("share_gt60", "mean"),
        treas_share=("treas_share", "mean"), treas_maxw=("treas_maxw", "median"),
        treas_rc=("treas_rc", "median"),
        leg_vol=("leg_vol", "median"),
        cap_on=("cap_on", "mean"), neff=("neff", "median"),
    ).reset_index()
    agg["ann_range"] = (agg["ann_max"] - agg["ann_min"]) * 100
    agg["phase_luck"] = (agg["ann_max"] - agg["ann_mean"]) * 100
    agg["calmar"] = agg["ann_mean"] / agg["mdd_mean"].abs()
    return agg.sort_values("cap").reset_index(drop=True)


def print_summary(agg: pd.DataFrame, n_phases: int | None = None) -> None:
    print("\n== 磐石 · max_weight 扫描（相位平均，T+1 open，2020 起）==")
    print(f"  {'档位':<10}{'年化均':>8}{'区间':>16}{'极差':>7}{'SR均':>7}"
          f"{'MDD均':>8}{'MDD最差':>9}{'Calmar':>8}{'换手':>7}")
    for _, x in agg.iterrows():
        print(f"  {x['cap_lbl']:<10}{x['ann_mean']:>8.2%}"
              f"{x['ann_min']:>7.2%}~{x['ann_max']:>7.2%}"
              f"{x['ann_range']:>6.1f}pp{x['sr_mean']:>7.2f}"
              f"{x['mdd_mean']:>8.2%}{x['mdd_worst']:>9.2%}"
              f"{x['calmar']:>8.2f}{x['turnover']:>7.1%}")

    print("\n== 集中度形状（是否压平了 §5.1.3 的双峰尾巴）==")
    print(f"  {'档位':<10}{'p50':>7}{'p75':>7}{'p90':>7}{'均值':>7}"
          f"{'>40%':>8}{'>60%':>7}{'国债首位':>9}{'其时中位':>9}"
          f"{'国债风险份额':>13}{'多头波动':>9}{'有效只':>7}{'cap生效':>8}")
    for _, x in agg.iterrows():
        on = "" if not np.isfinite(x["cap_on"]) else f"{x['cap_on']:>8.1%}"
        print(f"  {x['cap_lbl']:<10}{x['maxw_p50']:>7.1%}{x['maxw_p75']:>7.1%}"
              f"{x['maxw_p90']:>7.1%}{x['maxw_mean']:>7.1%}"
              f"{x['share_gt40']:>8.1%}{x['share_gt60']:>7.1%}"
              f"{x['treas_share']:>9.1%}{x['treas_maxw']:>9.1%}"
              f"{x['treas_rc']:>13.1%}{x['leg_vol']:>9.1%}"
              f"{x['neff']:>7.1f}{on}")

    base = agg[agg.cap <= 0]
    if not len(base):
        return
    b = base.iloc[0]
    print(f"\n  定版（无上限）：年化 {b['ann_mean']:.2%} / SR {b['sr_mean']:.2f} / "
          f"MDD {b['mdd_mean']:.2%} / Calmar {b['calmar']:.2f}")
    if n_phases == 5:
        print(f"  自检（应≈§5.1.1）：年化 {PA_REF['ann']:.2%} / SR {PA_REF['sharpe']:.2f} / "
              f"MDD {PA_REF['mdd']:.2%} / Calmar {PA_REF['calmar']:.2f}")
    print(f"\n  {'档位':<10}{'Δ年化':>9}{'ΔMDD':>9}{'ΔCalmar':>10}{'Δ换手':>8}")
    for _, x in agg[agg.cap > 0].iterrows():
        print(f"  {x['cap_lbl']:<10}{(x['ann_mean'] - b['ann_mean']) * 100:>+8.2f}pp"
              f"{(x['mdd_mean'] - b['mdd_mean']) * 100:>+8.2f}pp"
              f"{x['calmar'] - b['calmar']:>+10.2f}"
              f"{(x['turnover'] - b['turnover']) * 100:>+7.1f}pp")


def print_judge(eq: dict, labels: list[str], base: str,
                phases: tuple[int, ...] = PHASES) -> None:
    """判否三件套：月度配对 t + 12 个月滚动最差回撤。"""
    print("\n== ① 非重叠月度配对 t（vs 定版无上限，同相位）==")
    print(f"  {'档位':<10}{'k':>3}{'Δ月均值':>10}{'t':>7}{'胜率':>8}")
    for lb in labels:
        if lb == base:
            continue
        for k in phases:
            a, b = _monthly(eq[(lb, k)]), _monthly(eq[(base, k)])
            j = a.index.intersection(b.index)
            dv = (a[j] - b[j]).dropna()
            if len(dv) < 10:
                continue
            t = dv.mean() / dv.std(ddof=1) * np.sqrt(len(dv))
            print(f"  {lb:<10}{k:>3}{dv.mean() * 100:>+9.2f}%{t:>7.2f}"
                  f"{(dv > 0).mean():>8.1%}")

    print("\n== ② 12 个月滚动窗口最差回撤（每 ~21 交易日一个起点）==")
    print(f"  {'档位':<10}{'k':>3}{'中位':>8}{'p90':>8}{'最差':>8}{'Δ中位vs定版':>13}")
    for k in phases:
        ref = _roll_mdd(eq[(base, k)])
        for lb in labels:
            r = _roll_mdd(eq[(lb, k)])
            j = r.index.intersection(ref.index)
            dm = r[j].median() - ref[j].median()
            print(f"  {lb:<10}{k:>3}{r.median():>8.2%}{r.quantile(0.9):>8.2%}"
                  f"{r.max():>8.2%}"
                  f"{'' if lb == base else f'{dm * 100:>+12.2f}pp'}")
        print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--caps", default=",".join(str(c) for c in DEFAULT_CAPS),
                    help="逗号分隔的 max_weight 档位，0 = 无上限")
    ap.add_argument("--phases", default=",".join(str(p) for p in PHASES))
    ap.add_argument("--csv", default=str(OUT_CSV))
    ap.add_argument("--from-csv", action="store_true")
    args = ap.parse_args(argv)

    if args.from_csv:
        d = pd.read_csv(Path(args.csv), encoding="utf-8-sig")
        agg = summarize(d)
        print_summary(agg, int(d["phase"].nunique()))
        print("\n（--from-csv 不重跑回测 ⇒ 配对 t / 滚动回撤不可得）")
        return 0

    caps = [float(c) for c in args.caps.split(",") if c.strip()]
    phases = [int(p) for p in args.phases.split(",") if p.strip()]
    labels = [label(c) for c in caps]

    print("加载环境（全量 ETF 面板 + DEF 合成腿，约 1-2 分钟）…", flush=True)
    t0 = time.time()
    env = build_env()
    rb_map = month_phase_rebalance_days(env["close"].index)
    print(f"环境就绪 {time.time() - t0:.0f}s；{len(caps)} 档 cap × {len(phases)} 相位 "
          f"= {len(caps) * len(phases)} 次回测（每次 close/open 两遍）", flush=True)

    base_spec = dict(PROFILES["champ"])
    base_spec["weighting"] = "inverse_vol"

    rows: list[dict] = []
    eq: dict[tuple[str, int], pd.Series] = {}
    leg_by_key: dict[tuple[str, int], pd.DataFrame] = {}

    # 先跑无 cap 版：既是主表第一行，也是各 cap 档「生效性」的比较基准
    lb0 = label(0.0)
    for k in phases:
        t1 = time.time()
        _r0, r1 = run_champ(env, base_spec, rb_map[k])
        eq[(lb0, k)] = r1["equity"].dropna()
        leg_by_key[(lb0, k)] = _leg(r1["weights"])
        row = {"cap": 0.0, "cap_lbl": lb0, "phase": k,
               "ann": r1["annual_return"], "sharpe": r1["sharpe"],
               "mdd": r1["max_drawdown"], "turnover": r1["avg_turnover"],
               "n_rebalance": r1.get("n_rebalance", 0)}
        row.update(_conc_stats(leg_by_key[(lb0, k)], None, env["close"]))
        rows.append(row)
        print(f"  {lb0:<10} k={k}  年化 {r1['annual_return']:>7.2%}  "
              f"SR {r1['sharpe']:>5.2f}  MDD {r1['max_drawdown']:>7.2%}  "
              f"单只p75 {row['maxw_p75']:>5.1%}  ({time.time() - t1:.0f}s)",
              flush=True)

    for cap in caps:
        if cap <= 0:
            continue
        lb = label(cap)
        spec = dict(base_spec)
        spec["max_weight"] = cap
        for k in phases:
            t1 = time.time()
            _r0, r1 = run_champ(env, spec, rb_map[k])
            eq[(lb, k)] = r1["equity"].dropna()
            leg_by_key[(lb, k)] = _leg(r1["weights"])
            row = {"cap": cap, "cap_lbl": lb, "phase": k,
                   "ann": r1["annual_return"], "sharpe": r1["sharpe"],
                   "mdd": r1["max_drawdown"], "turnover": r1["avg_turnover"],
                   "n_rebalance": r1.get("n_rebalance", 0)}
            row.update(_conc_stats(leg_by_key[(lb, k)], leg_by_key[(lb0, k)],
                              env["close"]))
            rows.append(row)
            print(f"  {lb:<10} k={k}  年化 {r1['annual_return']:>7.2%}  "
                  f"SR {r1['sharpe']:>5.2f}  MDD {r1['max_drawdown']:>7.2%}  "
                  f"单只p75 {row['maxw_p75']:>5.1%}  "
                  f"cap生效 {row.get('cap_on', float('nan')):>5.1%}  "
                  f"({time.time() - t1:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False, encoding="utf-8-sig")

    agg = summarize(d)
    print_summary(agg, int(d["phase"].nunique()))
    print_judge(eq, labels, label(0.0), tuple(phases))
    print(f"\n明细已落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
