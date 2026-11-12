"""行业去重：逻辑检验 + 不去重的效果对比（五臂权衡）。

回答 2026-11-10 的追问：「从逻辑上讲行业去重有什么优点吗，不去重的效果是怎么样」

为什么要单独做这个
------------------
`focus_micro` 提出了去重，`rtl_rebalance_timing/VARIANTS.md` 复核了增益幅度（+6.48 → +1.39pp），
**但两者的解释都是「把后面的跟风股换成其他行业的最强股」—— 这句话是错的**。
定版 `fill_to_k=False` ⇒ 被行业挡掉的票**直接踢掉、不往下补**，持仓 8→5.83 只。
本脚本把「三段常见逻辑」逐条实测，并把「不去重」的几种替代做法一起量化。

三段逻辑与对应检验
------------------
1. 「信号会扎堆行业」→ 统计不去重时 Top8 的行业分布（§机制 A）
2. 「换成其他行业最强股」→ 复现策略真实选股，
   对比「保留」vs「踢掉」vs「往下补第 9~16 名」（§机制 C/D）
3. 「同行业第 2 名比第 1 名差」→ 按 **pred 分数**在行业内排名，看相对行业超额（§机制 B）
   🚨 排序键必须是信号，不能是事后收益 —— 后者会得到 t=18 的前视假阳性（实测过）。

五臂（不去重的替代做法）
------------------------
  P0 不去重 k8           纯 Top8
  P1 去重·踢掉（定版）    Top8 内行业去重，不补 ⇒ 实持 5.83
  P2 去重·补到 8 只       同样去重，但往下补满 k（`fill_to_k=True` + `min_sectors=k`）
  P3 每行业 2 只          放宽到 max_per_sector=2
  P4 不去重 k6            持仓数匹配对照（实持 5.96 ≈ P1 的 5.83）

输出 ``reports/rtl_rebalance_timing/``
--------------------------------------
  sector_dedup_mechanism.csv  机制检验（扎堆分布 / 行业内名次 / 保留vs踢掉vs补位 / 集中度分组）
  sector_dedup_arms.csv       五臂 × 21 相位汇总
  sector_dedup_yearly.csv     五臂逐年年化

用法::

    python -m scripts.evaluation.sector_dedup_probe
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import scripts.evaluation.caliber as cal
from scripts.evaluation.rtl_variant_probe import OUT
from scripts.evaluation.solo_freq_probe import (
    CAPITAL,
    END,
    LOT,
    industry_panel,
    price_panel,
    stats,
)
from strategy.cross_section import BufferedTopKLongOnly, SectorCappedBufferedTopKLongOnly

N_PHASE = 21
K = 8


def _hold_pairs(pred: pd.DataFrame):
    """返回（信号日 t, 买入日 T+1, 卖出日 T'+1）三元组。"""
    rb = cal.rebalance_dates(pred.index, freq="M", phase=0)
    pos = {d: i for i, d in enumerate(pred.index)}
    out = []
    for a, b in zip(rb, rb[1:]):
        if pos[a] + 1 >= len(pred.index) or pos[b] + 1 >= len(pred.index):
            continue
        out.append((a, pred.index[pos[a] + 1], pred.index[pos[b] + 1]))
    return out


def _dedup(top: list, sec: pd.Series) -> tuple[list, list]:
    """复现 `max_per_sector=1` + `fill_to_k=False`：行业已满则跳过，不往下补。"""
    seen, keep, drop = set(), [], []
    for c in top:
        g = sec.get(c)
        key = "__NA__" if g is None or pd.isna(g) else str(g)
        if key in seen:
            drop.append(c)
        else:
            seen.add(key)
            keep.append(c)
    return keep, drop


def _t(a) -> float:
    a = np.asarray(a, dtype=float)
    return float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 1 else np.nan


def mechanism(ctx, ind: pd.DataFrame) -> pd.DataFrame:
    """§机制 A/B/C/D/E：全部用信号日 pred 排序，无前视。"""
    pred, mask, fill = ctx.pred, ctx.mask, ctx.fill
    pairs = _hold_pairs(pred)
    rows: list[dict] = []
    rk1, rk2, rk3, rk4 = [], [], [], []       # 行业内 pred 名次的相对行业超额
    kept_r, drop_r, tail_r = [], [], []
    n_ind, max_w, hhi_l, port_r, crowd = [], [], [], [], []
    per_dup_gap, per_kd_vs_tail = [], []

    for t, buy, sell in pairs:
        s = pred.loc[t].where(mask.loc[t]).dropna()
        if len(s) < 50:
            continue
        ret = fill.loc[sell] / fill.loc[buy] - 1
        top8 = list(s.nlargest(K).index)
        sec8 = ind.loc[t].reindex(top8)

        # A. 扎堆分布
        vc = sec8.value_counts(dropna=False)
        n_ind.append(sec8.nunique())
        max_w.append(float(vc.iloc[0]) / K)

        keep, drop = _dedup(top8, sec8)
        rank = s.rank(ascending=False, method="first")
        tail = [c for c in s.index if K + 1 <= rank[c] <= K + len(drop)]
        rk = ret.reindex(keep).dropna()
        rd = ret.reindex(drop).dropna()
        rt = ret.reindex(tail).dropna()
        kept_r.extend(rk.tolist())
        drop_r.extend(rd.tolist())
        tail_r.extend(rt.tolist())
        if len(rd) > 0 and len(rt) > 0:
            per_kd_vs_tail.append(rd.mean() - rt.mean())
        if len(rd) > 0 and len(rk) > 0:
            per_dup_gap.append(rk.mean() - rd.mean())

        # E. 行业权重集中度（全部 Top8，含重复）
        w = pd.Series(1.0 / K, index=top8)
        iw = w.groupby(sec8).sum()
        hhi_l.append(float((iw**2).sum()))
        pr = ret.reindex(top8).dropna()
        port_r.append(float(pr.mean()))
        crowd.append(float(ret.reindex(keep).groupby(
            sec8.reindex(rk.index)).mean().max() - pr.mean())
            if len(rk) else np.nan)

        # B. 行业内按 pred 分数的名次 → 相对行业超额（Top30 内）
        top30 = list(s.nlargest(30).index)
        r30 = ret.reindex(top30).dropna()
        sec30 = ind.loc[t].reindex(r30.index)
        d = pd.DataFrame({"pred": s.reindex(r30.index), "ret": r30, "sec": sec30}).dropna()
        d["rk"] = d.groupby("sec")["pred"].rank(ascending=False, method="first")
        d["ex"] = d["ret"] - d.groupby("sec")["ret"].transform("mean")
        for k_, lst in ((1, rk1), (2, rk2), (3, rk3)):
            lst.extend(d.loc[d["rk"] == k_, "ex"].tolist())
        rk4.extend(d.loc[d["rk"] >= 4, "ex"].tolist())

    def push(name: str, x, unit: str, note: str = "") -> None:
        a = np.asarray(x, dtype=float)
        a = a[~np.isnan(a)]
        rows.append({"test": name, "n": len(a), "mean": a.mean(),
                     "t": _t(a), "unit": unit, "note": note})

    push("A_不同行业数(均值)", n_ind, "个")
    push("A_最大行业权重", max_w, "比例")
    push("C_保留票收益", kept_r, "月")
    push("C_踢掉票收益", drop_r, "月")
    push("C_往下补(9~16名)收益", tail_r, "月")
    push("C_逐期配对_重复票减补位票", per_kd_vs_tail, "月")
    push("C_逐期配对_保留减踢掉", per_dup_gap, "月")
    push("B_行业内第1名_相对行业超额", rk1, "月")
    push("B_行业内第2名_相对行业超额", rk2, "月")
    push("B_行业内第3名_相对行业超额", rk3, "月")
    push("B_行业内第4名及以后_相对行业超额", rk4, "月")
    push("E_HHI", hhi_l, "值")
    push("E_组合月收益(全部Top8)", port_r, "月")
    push("E_扎堆行业超额", crowd, "月")

    # E 补充：按最大行业权重三分位分组的组合收益 + 扎堆行业超额
    # 分组方式固定为 qcut 三等分（不用硬阈值 25%/45%，避免事后挑门槛）
    A = pd.DataFrame({"max_w": max_w, "port_r": port_r, "crowd": crowd}).dropna()
    A["q"] = pd.qcut(A["max_w"], 3, labels=["低集中", "中", "高集中"])
    for q, sub in A.groupby("q", observed=True):
        edges = f"{sub['max_w'].min():.0%}~{sub['max_w'].max():.0%}"
        rows.append({"test": f"E_组合月收益_{q}", "n": len(sub),
                     "mean": sub["port_r"].mean(),
                     "t": _t(sub["port_r"]), "unit": "月", "note": edges})
        rows.append({"test": f"E_扎堆行业超额_{q}", "n": len(sub),
                     "mean": sub["crowd"].mean(),
                     "t": _t(sub["crowd"]), "unit": "月", "note": edges})
    return pd.DataFrame(rows)


def arms(ctx, ind, px) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """五臂 × 21 相位，返回（汇总, 逐年, 逐相位明细）。"""
    cost = cal.cost_profile("production")
    spec = [
        ("P0_不去重_k8", lambda: BufferedTopKLongOnly(k=K, exit_mult=1.0)),
        ("P1_去重踢掉（定版）", lambda: SectorCappedBufferedTopKLongOnly(
            ind, k=K, exit_mult=1.0, max_per_sector=1)),
        ("P2_去重补到8", lambda: SectorCappedBufferedTopKLongOnly(
            ind, k=K, exit_mult=1.0, max_per_sector=1, fill_to_k=True, min_sectors=K)),
        ("P3_每行业2只", lambda: SectorCappedBufferedTopKLongOnly(
            ind, k=K, exit_mult=1.0, max_per_sector=2, min_sectors=4)),
        ("P4_不去重_k6", lambda: BufferedTopKLongOnly(k=6, exit_mult=1.0)),
    ]
    summ, yearly, detail = [], {}, []
    for name, f in spec:
        t0 = time.time()
        ann, sr, nh, ny = [], [], [], {}
        for p in range(N_PHASE):
            bt = cal.VectorBacktest(strategy=f(), rebalance_freq="M",
                                    initial_capital=CAPITAL, costs=cost)
            res = cal.run_once(bt, ctx, "open", freq="M", phase=p,
                               price_panel=px, lot_size=LOT)
            st = stats(res, len(cal.rebalance_dates(ctx.pred.index, "M", p)))
            ann.append(st["ann"])
            sr.append(st["sharpe_arith"])
            nh.append(st["nhold_mean"])
            detail.append({"arm": name, "phase": p, "ann": st["ann"],
                           "sr": st["sharpe_arith"], "dd": st["max_dd"],
                           "nhold": st["nhold_mean"]})
            dr = res.daily_returns
            for y, sub in dr.groupby(dr.index.year):
                ny.setdefault(int(y), []).append(float((1 + sub).prod() - 1))
        yearly[name] = {y: float(np.mean(v)) for y, v in ny.items()}
        a = np.array(ann)
        summ.append({"arm": name, "ann_mean": a.mean(), "ann_phase0": ann[0],
                     "ann_min": a.min(), "ann_max": a.max(),
                     "ann_std": a.std(ddof=1), "sr_mean": float(np.mean(sr)),
                     "nhold": float(np.mean(nh))})
        print(f"  {name:<18} 均值 {a.mean() * 100:6.2f}%  phase0 {ann[0] * 100:6.2f}%  "
              f"min {a.min() * 100:6.2f}%  SR {np.mean(sr):.3f}  "
              f"实持 {np.mean(nh):4.2f}  [{time.time() - t0:.0f}s]", flush=True)

    D = pd.DataFrame(detail)
    W = D.pivot(index="phase", columns="arm", values="ann")
    print("\n=== 逐相位配对差 ===", flush=True)
    pairs = [("P1_去重踢掉（定版）", "P0_不去重_k8", "去重·踢掉 vs 不去重"),
             ("P2_去重补到8", "P0_不去重_k8", "去重·补到8 vs 不去重"),
             ("P1_去重踢掉（定版）", "P2_去重补到8", "少持2只 vs 补满8只")]
    prow = []
    for a, b, note in pairs:
        d = (W[a] - W[b]).dropna()
        se = d.std(ddof=1) / np.sqrt(len(d))
        sr_diff = (D.pivot(index="phase", columns="arm", values="sr")[a]
                   - D.pivot(index="phase", columns="arm", values="sr")[b])
        prow.append({"a": a, "b": b, "note": note, "n": len(d),
                     "diff_mean": d.mean(), "diff_se": se, "t_nominal": d.mean() / se,
                     "n_pos": int((d > 0).sum()), "sr_diff": sr_diff.mean()})
        print(f"  {note:<26} {a} − {b}: {d.mean() * 100:+6.2f}pp ±{se * 100:4.2f} "
              f"(名义 t={d.mean() / se:4.1f})  正 {int((d > 0).sum())}/{len(d)}  "
              f"SR差 {sr_diff.mean():+.4f}", flush=True)
    pd.DataFrame(prow).to_csv(OUT / "sector_dedup_pairs.csv", index=False,
                              encoding="utf-8-sig")
    return pd.DataFrame(summ), pd.DataFrame(yearly).T, D


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = cal.load_context("gbdt_mid13", window="prod", end=END, execution="open")
    ind = industry_panel(ctx)
    px = price_panel(ctx)
    print(f"窗口 {ctx.pred.index[0].date()}..{ctx.pred.index[-1].date()} | "
          f"资金 {CAPITAL:,.0f} | 整手 {LOT} | k={K}", flush=True)

    print("\n=== 机制检验（全部按 pred 排序，无前视）===", flush=True)
    M = mechanism(ctx, ind)
    M.to_csv(OUT / "sector_dedup_mechanism.csv", index=False, encoding="utf-8-sig")
    for _, r in M.iterrows():
        u = "%" if r["unit"] in ("月", "比例") else ""
        print(f"  {r['test']:<28} n={int(r['n']):5d}  "
              f"{r['mean'] * 100:+.3f}{u}  t={r['t']:6.2f}", flush=True)

    if "--skip-arms" in sys.argv:      # 只重算机制段（约 25s），不重跑 5 臂
        print(f"\n产物 → {OUT}（仅机制段）", flush=True)
        return

    print("\n=== 五臂 × 21 相位 ===", flush=True)
    S, Y, D = arms(ctx, ind, px)
    S.to_csv(OUT / "sector_dedup_arms.csv", index=False, encoding="utf-8-sig")
    Y.to_csv(OUT / "sector_dedup_yearly.csv", encoding="utf-8-sig")
    D.to_csv(OUT / "sector_dedup_phase.csv", index=False, encoding="utf-8-sig")
    print("\n=== 逐年（21 相位均值）===", flush=True)
    print((Y * 100).round(2).to_string(), flush=True)
    print(f"\n产物 → {OUT}", flush=True)


if __name__ == "__main__":
    main()