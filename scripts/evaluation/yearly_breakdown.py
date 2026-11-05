"""分年拆解 —— 策略 / 基准 / IC 的逐年表现，用于解释某一年为什么异常。

动机（2026-11-05）：`pool_scope_probe` 显示 2018 年单年 −25.76%、超额 −8.0%，
一个 2018 就把 2018+ 的年化从 23.6% 压到 16.4%。本脚本逐年拆开：
年化 / Sharpe / MDD / 同池等权基准 / 超额 / rank IC / ICIR / 反向臂。

判读要点
--------
* **IC 正常但超额为负** ⇒ 问题在多头端（A 股不可做空，IC 的空头端拿不到）。
  用 ``rev_ret``（信号取负 → 买最低分 15%）对照：若反向臂反而赚钱，说明当年
  信号方向在多头端是反的。
* **IC 也塌了** ⇒ 信号本身当年失效（环境漂移），与组合构建无关。
* 单年 Sharpe 噪声很大（一年才 240 个点），只作方向参考。

用法
----
    python -m scripts.evaluation.yearly_breakdown --begin 20180101 --pool all_a
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import VectorBacktest  # noqa: E402
from backtest.metrics import (  # noqa: E402
    annual_return,
    max_drawdown,
    sharpe_ratio,
)
from scripts.evaluation.caliber import (  # noqa: E402
    cost_profile,
    get_profile,
    load_context,
    run_once,
)
from scripts.evaluation.pool_scope_probe import (  # noqa: E402
    POOLS,
    _pool_bench,
    _pool_mask,
)
from strategy.cross_section import BufferedTopFracLongOnly  # noqa: E402


def _ic_by_year(pred: pd.DataFrame, fwd: pd.DataFrame,
                mask: pd.DataFrame) -> pd.DataFrame:
    """逐日 rank IC，按年聚合。``fwd`` 须已 shift(-1)（信号 T vs 收益 T+1）。"""
    p, r = pred.where(mask), fwd.where(mask)
    ar, br = p.rank(axis=1), r.rank(axis=1)
    am = ar.sub(ar.mean(axis=1), axis=0)
    bm = br.sub(br.mean(axis=1), axis=0)
    num = (am * bm).sum(axis=1)
    den = np.sqrt((am ** 2).sum(axis=1) * (bm ** 2).sum(axis=1))
    ic = (num / den.replace(0, np.nan))
    ic = ic.where((p.notna() & r.notna()).sum(axis=1) >= 30).dropna()
    rows = []
    for y, sub in ic.groupby(ic.index.year):
        rows.append({
            "year": y, "ic": float(sub.mean()),
            "icir": float(sub.mean() / sub.std() * np.sqrt(252)),
            "ic_pos": float((sub > 0).mean()), "ic_days": int(len(sub)),
        })
    return pd.DataFrame(rows).set_index("year")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="ens_h1h5")
    ap.add_argument("--pool", default="all_a")
    ap.add_argument("--begin", type=int, default=20180101)
    ap.add_argument("--end", type=int, default=20260907)
    ap.add_argument("--out", default="reports/pool_scope_probe/yearly_all_a.csv")
    args = ap.parse_args()

    prof = get_profile("production")
    ctx = load_context(signal=args.signal, begin=args.begin, end=args.end,
                       execution=prof.execution)
    cost = cost_profile(prof.cost)

    if args.pool == "all_a":
        mask = pd.DataFrame(True, index=ctx.pred.index, columns=ctx.pred.columns)
    else:
        mask = _pool_mask(POOLS[args.pool], ctx.pred.index, ctx.pred.columns)
    bj = [c for c in ctx.pred.columns if c.endswith(".BJ")]
    sig = ctx.pred.where(mask)
    if bj:
        sig = sig.copy()
        sig.loc[:, bj] = np.nan
        mask = mask.copy()
        mask.loc[:, bj] = False

    def _run(panel: pd.DataFrame) -> pd.Series:
        bt = VectorBacktest(strategy=BufferedTopFracLongOnly(0.15, 0.40),
                            rebalance_freq="M", initial_capital=prof.capital,
                            costs=cost)
        return run_once(bt, ctx, prof.execution, signal=panel).daily_returns

    dr_long = _run(sig)                 # 做多高分 15%
    dr_rev = _run(-sig)                 # 做多低分 15%（反向臂）

    # 完整面板（只对齐列，不裁 index）—— 裁 index 会让窗口首日无前值
    close_full = pd.read_parquet(
        ROOT / "reports/alla_rolling/_base/close_adj.parquet"
    ).reindex(columns=ctx.pred.columns)
    fwd = close_full.pct_change(fill_method=None).shift(-1).reindex(ctx.pred.index)
    ic = _ic_by_year(sig, fwd, mask)
    bench = _pool_bench(mask, ctx.pred.index)

    rows = []
    for y in sorted(dr_long.index.year.unique()):
        a = dr_long[dr_long.index.year == y]
        b = dr_rev[dr_rev.index.year == y]
        c = bench[bench.index.year == y]
        rows.append({
            "year": y, "days": len(a),
            "ann": float(annual_return(a)), "sharpe": float(sharpe_ratio(a)),
            "mdd": float(max_drawdown(a)),
            "bench_ann": float(annual_return(c)),
            "bench_sharpe": float(sharpe_ratio(c)),
            "excess": float(annual_return(a)) - float(annual_return(c)),
            "rev_ann": float(annual_return(b)),
            "ic": ic.loc[y, "ic"] if y in ic.index else np.nan,
            "icir": ic.loc[y, "icir"] if y in ic.index else np.nan,
            "ic_pos": ic.loc[y, "ic_pos"] if y in ic.index else np.nan,
        })
    df = pd.DataFrame(rows).set_index("year")

    pd.set_option("display.width", 200)
    show = df.copy()
    for c in ["ann", "mdd", "bench_ann", "excess", "rev_ann", "ic_pos"]:
        show[c] = (show[c] * 100).round(1)
    for c in ["sharpe", "bench_sharpe", "ic", "icir"]:
        show[c] = show[c].round(3)
    print(f"信号 {args.signal} · 池 {args.pool} · {prof.name} 口径 · "
          f"{ctx.pred.index.min().date()}..{ctx.pred.index.max().date()}")
    print(show.to_string())
    print("\n列说明：ann/bench_ann/excess/rev_ann 为 %；excess = ann − bench_ann；"
          "rev_ann = 买最低分 15% 的收益（对照臂）")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"已写 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
