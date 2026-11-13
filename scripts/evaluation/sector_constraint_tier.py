"""行业约束（每申万一级 ≤N 只）放在**哪个档位**才值钱 —— 七臂 × 相位。

背景（2026-11-13 追问）
----------------------
`rtl_phase_tiers` 量出三档 21 相位均值几乎打平（solo 25.13% vs focus 24.75%），
而 `tier_sector_concentration` 量出：
- raw8（solo 不约束）最大行业权重 **31.1%**、p90 **50%**、max **62.5%**
- dedup8（solo 定版，踢掉不补）**17.4%**，但个股 HHI 从 0.125 → 0.174（+40%）
- fill8（去重 + 补到 8 个行业）**12.5%**，个股 HHI 仍 0.125 ⇒ 两头都好
- raw15（focus 不约束）**25.2%**、p90 **40%**、max **60%** ⇒ **行业集中度比 solo 定版更差**

⇒ 「focus 持仓多所以行业自然分散」是**错的**。那行业约束该怎么配？
本脚本把「K」与「行业约束形态」拆成两个轴，在**同资金档内**配对：
- solo 组（K8）：定版 / 去重补位（P2）
- focus 组（K15）：定版 / 去重踢掉 / 去重补到 15 行业

判据：先看 phase 0 方向（几十秒），有信号再 `--phases 0-20` 全扫。
`base` 臂必须逐位复现 `yq_core` 登记值，否则臂间差不成立。

用法::

    python -m scripts.evaluation.sector_constraint_tier              # phase 0 冒烟
    python -m scripts.evaluation.sector_constraint_tier --phases 0-20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

import scripts.evaluation.caliber as cal
from scripts.evaluation import yq_core_report as ycr

OUT = Path("reports/sector_constraint_tier")


def industry_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    ind = pd.read_parquet(cal.BASE_DIR / "industry.parquet")
    return ind.reindex(index=ctx.pred.index, columns=ctx.pred.columns)


#: (臂名, 资金档来源, 构造 kwargs for SectorCappedBufferedTopKLongOnly)
#: ``None`` 的 kwargs 表示「用 focus 定版的 BufferedTopKLongOnly(15, 2.0)」。
ARMS = (
    ("solo_base",   "yqcore-solo",   dict(k=8, exit_mult=1.0, max_per_sector=1)),
    ("solo_fill8",  "yqcore-solo",   dict(k=8, exit_mult=1.0, max_per_sector=1,
                                          min_sectors=8, fill_to_k=True)),
    ("focus_base",  "yqcore-focus",  None),
    ("focus_dedup", "yqcore-focus",  dict(k=15, exit_mult=2.0, max_per_sector=1)),
    ("focus_fill15", "yqcore-focus", dict(k=15, exit_mult=2.0, max_per_sector=1,
                                          min_sectors=15, fill_to_k=True)),
    ("focus_fill12", "yqcore-focus", dict(k=15, exit_mult=2.0, max_per_sector=1,
                                          min_sectors=12, fill_to_k=True)),
    ("focus_sec2",  "yqcore-focus",  dict(k=15, exit_mult=2.0, max_per_sector=2)),
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="0")
    ap.add_argument("--arms", default=",".join(a[0] for a in ARMS))
    a = ap.parse_args(argv)
    if "-" in a.phases:
        lo, hi = a.phases.split("-")
        phases = list(range(int(lo), int(hi) + 1))
    else:
        phases = [int(x) for x in a.phases.split(",")]
    want = {x.strip() for x in a.arms.split(",") if x.strip()}

    from strategy.examples import BufferedTopKLongOnly, SectorCappedBufferedTopKLongOnly

    ctx = cal.load_context(signal="gbdt_mid13", window="prod", end=20260907,
                           execution="open")
    px = ycr.price_panel(ctx)
    ind = industry_panel(ctx)
    cost = cal.cost_profile(ycr.PROFILE)

    rows = []
    for arm, tier, kw in ARMS:
        if arm not in want:
            continue
        capital = float(ycr.NATIVE_CAPITAL[tier])
        for p in phases:
            t0 = time.time()
            if kw is None:
                strat = BufferedTopKLongOnly(k=15, exit_mult=2.0)
            else:
                strat = SectorCappedBufferedTopKLongOnly(sectors=ind, **kw)
            bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                    initial_capital=capital, costs=cost)
            res = cal.run_once(bt, ctx, "open", freq="M", phase=p,
                               price_panel=px, lot_size=ycr.LOT)
            m = cal.metrics_row(res, capital)
            rows.append({"tier": tier, "arm": arm, "phase": p, "capital": capital,
                         "ann": m["ann"], "sr": m["sharpe_arith"],
                         "max_dd": m["max_dd"], "turnover": m["turnover"],
                         "cost_bp": m["cost_bp"], "n_hold": m["n_active"]})
            print(f"  [{arm:<12}] p={p:<3} ann={m['ann'] * 100:7.2f}% "
                  f"SR={m['sharpe_arith']:6.3f} 持仓={m['n_active']:6.1f} "
                  f"换手={m['turnover']:5.3f} ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"arms_p{'x'.join(map(str, phases[:1]))}"
                     f"{'' if len(phases) == 1 else 'N'}.csv",
              index=False, encoding="utf-8-sig")
    if len(phases) > 1:
        sm = df.groupby("arm")[["ann", "sr", "max_dd", "turnover", "n_hold"]].mean()
        sm["ann_std"] = df.groupby("arm")["ann"].std(ddof=1)
        print("\n=== 21 相位均值 ===")
        print(sm.round(4).to_string())
        sm.to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
