"""k8 缓冲带 `exit_mult` 在**生产口径 + 21 相位**下的复核（2026-11-25）。

## 为什么要重做

`reports/buffer_exit_scan/`（2026-11-17）已判负，但那份实验有三个与本档不同的地方：

1. 它是**纯 ``BufferedTopKLongOnly``**（无行业去重），不是本档的
   ``SectorCappedBufferedTopKLongOnly``。行业去重把可选池压到「每行业 1 只」，
   buffer 的留人空间与纯 TopK 不同 —— 旧结论不能直接搬。
2. 它只跑 **phase 0 单点**。本档 6 只持仓、换手 92%，单点噪声极大
   （`reports/yq_core/README.md`：掩码 off-by-one 就让 ann 动 2pp）。
3. 它跑在掩码顺序修正**之前**的旧口径上。

本脚本只做一件事：在**现行登记口径**下，用 **21 相位均值**判断 `exit_mult` 是否还有余量。

## 口径

production（27bp 时变 / T+1 open / 1e5 / 整手 100 / 月频），信号 `gbdt_mid13`，
策略取 `get_tier("yqcore-focus-k8")` 的登记参数（k8 / 每行业≤1 / min_sectors 3），
**只覆盖 `exit_mult`**。

用法::

    python -m scripts.portfolio.k8_buffer_scan --grid 1.0,2.0,3.0,5.0

产物：``reports/k8_buffer_scan/{scan.csv,phase_summary.csv,paired.csv,daily_*.csv}``。
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import caliber as cal  # noqa: E402
from scripts.portfolio.microcap_ref import LOT, lean_ctx, panel_for, price_panel  # noqa: E402
from strategy.registry import get_tier  # noqa: E402

OUT = ROOT / "reports" / "k8_buffer_scan"
TIER = "yqcore-focus-k8"
CAPITAL = 1.0e5
BASELINE = 1.0


def run_one(ctx, pp, ind, exit_mult: float, phase: int) -> tuple[dict, pd.Series]:
    strat = get_tier(TIER).build(sectors=ind, exit_mult=float(exit_mult))
    bt = cal.VectorBacktest(
        strategy=strat, rebalance_freq="M", initial_capital=CAPITAL,
        costs=cal.cost_profile("production"),
    )
    res = cal.run_once(bt, ctx, "open", signal=ctx.pred, price_panel=pp,
                       lot_size=LOT, phase=phase)
    m = cal.metrics_row(res, CAPITAL)
    n_hold = (res.weights_history > 1e-12).sum(axis=1)
    rec = {
        "exit_mult": float(exit_mult),
        "phase": int(phase),
        "ann": m["ann"],
        "sharpe": m["sharpe"],
        "max_dd": m["max_dd"],
        "calmar": m["calmar"],
        "turnover": m["turnover"],
        "cost_bp": m["cost_bp"],
        "n_hold": float(n_hold.mean()),
    }
    return rec, res.daily_returns.copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="1.0,2.0,3.0,5.0")
    ap.add_argument("--phases", type=int, default=21)
    ap.add_argument("--tag", default="", help="输出子目录（分年复核等分化运行用）")
    a = ap.parse_args()

    grid = [float(x) for x in a.grid.split(",")]
    out = OUT / a.tag if a.tag else OUT
    out.mkdir(parents=True, exist_ok=True)

    ctx = lean_ctx()
    pp = price_panel(ctx)
    ind = panel_for("industry", ctx)
    print(f"上下文就绪：{len(ctx.pred)} 日 × {ctx.pred.shape[1]} 只；"
          f"网格 {grid} × {a.phases} 相位")

    rows, daily = [], {}
    for em in grid:
        for ph in range(a.phases):
            rec, dr = run_one(ctx, pp, ind, em, ph)
            rows.append(rec)
            daily[(em, ph)] = dr
            print(f"  x{em:<4} phase {ph:2d}  ann {rec['ann']:7.2%}  "
                  f"SR {rec['sharpe']:6.3f}  MDD {rec['max_dd']:7.2%}  "
                  f"换手 {rec['turnover']:6.2%}  持仓 {rec['n_hold']:.1f}")
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out / "scan.csv", index=False)

    # 21 相位均值（可期望值口径）
    summ = (df.groupby("exit_mult")
              .agg(ann=("ann", "mean"), ann_sd=("ann", "std"),
                   sharpe=("sharpe", "mean"), sharpe_sd=("sharpe", "std"),
                   max_dd=("max_dd", "mean"), calmar=("calmar", "mean"),
                   turnover=("turnover", "mean"), cost_bp=("cost_bp", "mean"),
                   n_hold=("n_hold", "mean"), n_phase=("phase", "size"))
              .reset_index())
    summ.to_csv(out / "phase_summary.csv", index=False)

    # 与基线的比较：**按相位配对**，不以日频拼样本
    # 🚨 21 个相位不独立（同信号、同策略，只差调仓日起点）。把 21×1864 天拼起来算 t
    # 会把 n 放大 21 倍、t 虚高 √21 倍。正确口径是「每个相位一个差」再对 21 个值做 t。
    wide = df.pivot(index="phase", columns="exit_mult", values="ann")
    wide_sr = df.pivot(index="phase", columns="exit_mult", values="sharpe")
    wide_to = df.pivot(index="phase", columns="exit_mult", values="turnover")
    paired = []
    for em in grid:
        if em == BASELINE:
            continue
        d = (wide[em] - wide[BASELINE]).dropna()
        dsr = (wide_sr[em] - wide_sr[BASELINE]).dropna()
        dt = (wide_to[em] - wide_to[BASELINE]).dropna()
        n = len(d)
        t = d.mean() / d.std() * np.sqrt(n) if d.std() > 0 else np.nan
        # 日频逐笔差（仅描述，不作显著性判据）
        diffs = []
        for ph in range(a.phases):
            j = pd.concat([daily[(BASELINE, ph)].rename("b"),
                           daily[(em, ph)].rename("t")], axis=1).dropna()
            diffs.append(j["t"] - j["b"])
        dd = pd.concat(diffs)
        paired.append({
            "exit_mult": em,
            "d_ann_pp": float(d.mean() * 100),
            "d_ann_sd_pp": float(d.std() * 100),
            "t_ann": float(t),
            "n_phase": int(n),
            "n_phase_win": int((d > 0).sum()),
            "d_sharpe": float(dsr.mean()),
            "d_turnover_pp": float(dt.mean() * 100),
            "d_daily_bp": float(dd.mean() * 1e4),
        })
    pdf = pd.DataFrame(paired)
    pdf.to_csv(out / "paired.csv", index=False)

    # 分年（每个相位各自分年，再按 exit_mult 平均）—— 判「是不是近期运气」
    yr_rows = []
    for (em, ph), dr in daily.items():
        g = (1 + dr.dropna()).groupby(dr.dropna().index.year).prod() - 1
        for y, v in g.items():
            yr_rows.append({"exit_mult": em, "phase": ph, "year": int(y), "ret": float(v)})
    ydf = pd.DataFrame(yr_rows)
    ysum = (ydf.pivot_table(index="year", columns="exit_mult", values="ret",
                            aggfunc="mean"))
    ysum.to_csv(out / "yearly_mean.csv")

    print("\n=== 21 相位均值")
    print(summ.round(4).to_string(index=False))
    print("\n=== 与 x1.0 的逐相位配对（日收益差）")
    print(pdf.round(4).to_string(index=False))
    print("\n=== 分年（21 相位均值）")
    print(ysum.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
