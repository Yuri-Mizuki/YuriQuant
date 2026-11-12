"""`rtl_variant_probe` 的补充：换年度轴复核行业去重，并量化分腿能降多少 RTL。

回答 2026-11-10 的「哪个 RTL 比较小」——把「RTL 有多大」推进到「**能降到多小、代价多少**」。

为什么需要补这一层（两个都必须）
--------------------------------
1. **21 个相位不是 21 个独立样本**。它们是同一批股票、同一信号、只把换仓日错开几天的
   近似重复实验 —— 实测相位间日收益相关 **r̄ ≈ 0.82** ⇒ 有效样本量只有 **≈1.2**。
   拿「相位间离散」当标准误会把 t 值放大 4 倍。故主表的 t 只能当**排序**用，
   显著性必须换**年度轴**复核（年份间的共同因子弱得多）。
2. **分腿（tranching）的 RTL 不能按「单腿相对叠加」测**。腿数少时叠加组合本身就落在
   那两条腿附近，单腿离均值近 ⇒ 算出来的 RTL 反而更小，得出「腿越少 RTL 越小」的
   荒谬结论。文献问的是「**叠加组合本身的收益有多依赖抽到哪几天**」⇒ 必须测
   **对 N 腿子集的抽样分散度**（本节改为随机抽 300 个子集，比 21 选 N 全枚举省时）。

输出 ``reports/rtl_rebalance_timing/``
--------------------------------------
  extra_yearly.csv         4 臂 × 年度年化（21 相位均值）
  extra_yearly_paired.csv  年度配对差 + t
  extra_tranching.csv      N 腿子集抽样 → 组合年化的分散度（= 该方案的 RTL）
  extra_corr.csv           相位间日收益相关性矩阵

用法::

    python -m scripts.evaluation.rtl_variant_extra
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import scripts.evaluation.caliber as cal
from scripts.evaluation.rtl_variant_probe import ARMS, OUT, annualized, build_arm
from scripts.evaluation.solo_freq_probe import (
    CAPITAL,
    END,
    LOT,
    industry_panel,
    price_panel,
)

N_PHASE = 21
#: 只跑复核结论必需的臂（21 相位 × 4 臂 ≈ 100s）
KEEP = ["plain_k8_b1.0", "plain_k6_b1.0", "m1_k8_b1.0", "m1_k8_b2.0"]
#: 分腿测试用哪条臂（最佳候选）
TRANCH_ARM = "m1_k8_b2.0"
#: 每个腿数的子集抽样数（21 选 5 = 20349，抽样即可，误差 < 0.1pp）
N_DRAW = 300
SEED = 20261110


def _yearly(dr: pd.Series) -> dict[int, float]:
    return {int(y): float((1 + s).prod() - 1) for y, s in dr.groupby(dr.index.year)}


def _tranch_rtl(daily: dict[int, pd.Series], n_legs: int, rng) -> dict:
    """N 腿等权叠加的 **RTL = 不同子集上该组合年化的标准差**。

    这才是文献语义：组合收益对「抽到哪几天」有多敏感。N 增大时子集成员趋同
    ⇒ 分散应趋 0（= 文献说的 1/N 收敛）。
    """
    phases = sorted(daily)
    R = pd.DataFrame({p: daily[p] for p in phases})
    subs = list(itertools.combinations(phases, n_legs))
    if len(subs) > N_DRAW:
        idx = rng.choice(len(subs), size=N_DRAW, replace=False)
        subs = [subs[i] for i in idx]
    anns = [annualized(R[list(s)].mean(axis=1)) for s in subs]
    full = R.mean(axis=1)
    return {"n_legs": n_legs, "n_subset": len(subs),
            "rtl_subset_std": float(np.std(anns, ddof=1)),
            "ann_mean": float(np.mean(anns)), "ann_min": float(np.min(anns)),
            "ann_max": float(np.max(anns)),
            "ann_range": float(np.max(anns) - np.min(anns)),
            "ann_21legs": annualized(full)}


def main() -> None:
    ctx = cal.load_context("gbdt_mid13", window="prod", end=END, execution="open")
    cost = cal.cost_profile("production")
    px = price_panel(ctx)
    ind = industry_panel(ctx)
    arms = [a for a in ARMS if a[0] in KEEP]
    print(f"窗口 {ctx.pred.index[0].date()}..{ctx.pred.index[-1].date()} | "
          f"臂 {[a[0] for a in arms]} × {N_PHASE} 相位", flush=True)

    daily: dict[str, dict[int, pd.Series]] = {}
    for arm in arms:
        name = arm[0]
        daily[name] = {}
        for p in range(N_PHASE):
            bt = cal.VectorBacktest(strategy=build_arm(arm, ind), rebalance_freq="M",
                                    initial_capital=CAPITAL, costs=cost)
            res = cal.run_once(bt, ctx, "open", freq="M", phase=p,
                               price_panel=px, lot_size=LOT)
            daily[name][p] = res.daily_returns
        print(f"  {name} 完成", flush=True)

    # ---- 1) 年度轴：去重是否在多数年份为正 ----
    Y = pd.DataFrame({name: pd.DataFrame({p: _yearly(dr) for p, dr in d.items()}).T.mean()
                      for name, d in daily.items()})
    Y.to_csv(OUT / "extra_yearly.csv", encoding="utf-8-sig")
    print("\n=== 年度年化（21 相位均值，消掉相位噪声）===", flush=True)
    print((Y * 100).round(2).to_string(), flush=True)
    print("\n=== 年度配对差（行业去重 − 对照）===", flush=True)
    rows = []
    for a, b in [("m1_k8_b1.0", "plain_k8_b1.0"), ("m1_k8_b1.0", "plain_k6_b1.0"),
                 ("m1_k8_b2.0", "plain_k8_b1.0")]:
        d = (Y[a] - Y[b]) * 100
        se = d.std(ddof=1) / np.sqrt(len(d))
        rows.append({"a": a, "b": b, **{f"y{k}": v for k, v in d.items()},
                     "mean": d.mean(), "se": se, "t": d.mean() / se,
                     "n_pos": int((d > 0).sum())})
        print(f"  {a} − {b:<14} 均值 {d.mean():+6.2f}pp ± {se:4.2f} (t={d.mean() / se:5.2f})"
              f"  正 {int((d > 0).sum())}/{len(d)}  | "
              + " ".join(f"{int(k)}:{v:+.1f}" for k, v in d.items()), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "extra_yearly_paired.csv", index=False,
                              encoding="utf-8-sig")

    # ---- 2) 分腿轴：子集抽样测「组合自身」的 RTL ----
    rng = np.random.default_rng(SEED)
    print(f"\n=== 分腿（{TRANCH_ARM}）：N 腿叠加组合的年化分散 ===", flush=True)
    trows = [_tranch_rtl(daily[TRANCH_ARM], n, rng) for n in (1, 2, 3, 5, 8, 12, 21)]
    T = pd.DataFrame(trows)
    T.to_csv(OUT / "extra_tranching.csv", index=False, encoding="utf-8-sig")
    for r in trows:
        print(f"  {r['n_legs']:2d} 腿（抽 {r['n_subset']:3d} 组）："
              f"RTL {r['rtl_subset_std'] * 100:5.2f}%  "
              f"年化 {r['ann_mean'] * 100:6.2f}%（{r['ann_min'] * 100:6.2f}"
              f"~{r['ann_max'] * 100:6.2f}）  21 腿叠加 {r['ann_21legs'] * 100:6.2f}%",
              flush=True)

    # ---- 3) 相位间相关性 ⇒ 有效样本量 ----
    C = pd.DataFrame(daily[TRANCH_ARM]).corr()
    C.to_csv(OUT / "extra_corr.csv", encoding="utf-8-sig")
    off = C.to_numpy()[np.triu_indices_from(C.to_numpy(), k=1)]
    rbar = float(np.mean(off))
    n_eff = N_PHASE / (1 + (N_PHASE - 1) * rbar)
    print(f"\n=== 相位间日收益相关性 ===\n  均值 r̄={rbar:.3f}  "
          f"最小 {off.min():.3f}  最大 {off.max():.3f}", flush=True)
    print(f"  ⇒ 有效样本量 n_eff = N/(1+(N−1)r̄) = **{n_eff:.2f}**（名义 21）"
          f" ⇒ 相位轴的 t 应乘 {np.sqrt(n_eff / N_PHASE):.2f}", flush=True)
    print(f"\n产物 → {OUT}", flush=True)


if __name__ == "__main__":
    main()