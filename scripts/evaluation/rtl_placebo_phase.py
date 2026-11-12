"""行业去重安慰剂：跨相位重跑（原安慰剂只在 phase=0，无法排除相位选择偏差）。

为什么要重跑
------------
``focus_micro`` 的安慰剂（随机打乱行业标签 10 次，真实 38.3% vs 随机 28.90%±2.73pp）
**只在 phase=0 跑过**。它证明的是「真实行业分组优于随机分组」，但**不能**排除
「这个优势本身是 phase=0 这个签带来的」—— 因为 ``rtl_variant_probe`` 已实测：
同一个去重效应在 21 相位上的均值只有 **+1.39pp**（phase=0 单点 +6.48pp）。

本脚本把安慰剂扩展到多相位：每个随机标签抽签跑 21 个相位，与真实标签的
21 相位逐相位配对。若真实标签的配对优势在多相位下仍显著，则「行业分组含真实信息」
成立；若优势集中在 phase=0，则原安慰剂的 100% 分位只是相位运气。

口径
----
与 ``rtl_variant_probe`` 完全一致：k=8 / exit_mult=1.0 / max_per_sector=1 /
10 万整手 / T+1 open / 36bp 往返 / 信号 gbdt_mid13 / 窗口 2019-01-02..2026-09-07。
随机标签 **每行内 shuffle**（保留当日行业分布，只打断「哪只股票属于哪个行业」）。

输出 ``reports/rtl_rebalance_timing/``
--------------------------------------
  placebo_phase.csv   每次抽签 × 每相位的年化（真实 + N 次随机）
  placebo_summary.csv 逐相位分位（真实值在随机分布里的位置）

用法::

    python -m scripts.evaluation.rtl_placebo_phase --draws 10
"""
from __future__ import annotations

import argparse

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
)
from strategy.cross_section import SectorCappedBufferedTopKLongOnly

N_PHASE = 21
K, BUF, MPS = 8, 1.0, 1


def _ann(ctx, cost, px, ind, phase: int) -> float:
    st = SectorCappedBufferedTopKLongOnly(ind, k=K, exit_mult=BUF, max_per_sector=MPS)
    bt = cal.VectorBacktest(strategy=st, rebalance_freq="M",
                            initial_capital=CAPITAL, costs=cost)
    res = cal.run_once(bt, ctx, "open", freq="M", phase=phase,
                       price_panel=px, lot_size=LOT)
    return float(cal._metrics(res, CAPITAL)["ann"])


def main(draws: int, phases: list[int]) -> None:
    ctx = cal.load_context("gbdt_mid13", window="prod", end=END, execution="open")
    cost = cal.cost_profile("production")
    px = price_panel(ctx)
    ind = industry_panel(ctx)
    print(f"真实标签 + {draws} 次随机标签 × {len(phases)} 相位", flush=True)

    rows = []
    for p in phases:
        rows.append({"draw": "real", "phase": p, "ann": _ann(ctx, cost, px, ind, p)})
    real = pd.DataFrame(rows)
    print(f"  real 完成（21 相位均值 {real['ann'].mean() * 100:.2f}%）", flush=True)

    rng = np.random.default_rng(20261110)
    arr = ind.to_numpy(dtype=object)
    plc_rows = []
    for i in range(draws):
        shuf = arr.copy()
        for r_i in range(shuf.shape[0]):
            rng.shuffle(shuf[r_i])
        ind_s = pd.DataFrame(shuf, index=ind.index, columns=ind.columns)
        for p in phases:
            plc_rows.append({"draw": i, "phase": p,
                             "ann": _ann(ctx, cost, px, ind_s, p)})
        sub = [r["ann"] for r in plc_rows if r["draw"] == i]
        print(f"  随机 {i + 1:>2}/{draws} 完成  21 相位均值 {np.mean(sub) * 100:6.2f}%",
              flush=True)

    plc = pd.DataFrame(plc_rows)
    allr = pd.concat([real, plc], ignore_index=True)
    allr.to_csv(OUT / "placebo_phase.csv", index=False, encoding="utf-8-sig")

    print("\n=== 逐相位：真实值在随机分布中的位置 ===", flush=True)
    srows = []
    for p in phases:
        r = float(real.loc[real.phase == p, "ann"].iloc[0])
        s = plc.loc[plc.phase == p, "ann"]
        pct = float((s < r).mean() * 100)
        srows.append({"phase": p, "real": r, "rand_mean": s.mean(),
                      "rand_std": s.std(ddof=1), "diff": r - s.mean(),
                      "pctile": pct, "rand_max": s.max()})
        print(f"  phase={p:<2} 真实 {r * 100:6.2f}%  随机 {s.mean() * 100:6.2f}%"
              f"±{s.std(ddof=1) * 100:4.2f}  差 {(r - s.mean()) * 100:+6.2f}pp  "
              f"分位 {pct:3.0f}%", flush=True)
    S = pd.DataFrame(srows)
    S.to_csv(OUT / "placebo_summary.csv", index=False, encoding="utf-8-sig")

    # 汇总：随机均值在 21 相位上的分散（= 纯噪声能造出多大的「相位效应」）
    rnd_phase_mean = plc.groupby("phase")["ann"].mean()
    print("\n=== 汇总 ===", flush=True)
    print(f"  真实：21 相位均值 {real['ann'].mean() * 100:6.2f}%  "
          f"极差 {(real['ann'].max() - real['ann'].min()) * 100:5.2f}pp  "
          f"phase0 {real.loc[real.phase == 0, 'ann'].iloc[0] * 100:.2f}%", flush=True)
    print(f"  随机：21 相位均值 {rnd_phase_mean.mean() * 100:6.2f}%  "
          f"极差 {(rnd_phase_mean.max() - rnd_phase_mean.min()) * 100:5.2f}pp  "
          f"phase0 {rnd_phase_mean.loc[0] * 100:.2f}%", flush=True)
    # 逐抽签的「21 相位均值」，看真实值在抽签分布里的分位
    draw_means = plc.groupby("draw")["ann"].mean()
    r = real["ann"].mean()
    print(f"  真实 21 相位均值 {r * 100:.2f}% vs 随机抽签均值分布 "
          f"{draw_means.mean() * 100:.2f}%±{draw_means.std(ddof=1) * 100:.2f}  "
          f"⇒ 分位 {float((draw_means < r).mean() * 100):.0f}%  "
          f"({int((draw_means < r).sum())}/{len(draw_means)})", flush=True)
    print(f"  分位 100% 且抽签数 {draws} ⇒ 置换 p ≤ {1 / (draws + 1):.3f}", flush=True)
    print(f"\n产物 → {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--phases", default="")
    a = ap.parse_args()
    main(a.draws, [int(x) for x in a.phases.split(",")] if a.phases
         else list(range(N_PHASE)))