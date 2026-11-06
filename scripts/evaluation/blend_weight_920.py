"""定版信号「快慢融合权重」λ 在**正确底座（h1020）**上的补齐扫描。

为什么要有这个脚本
------------------
定版信号 = ``(1−λ)·rank(h1020) + λ·rank(slow)``，λ=0.5。但历史记录里 λ 只被扫过
三个点（0.3 / 0.5 / 0.7），而且**全部落在 ``ens_h1h5`` 底座上**
（``fundamental_blend.py`` 的 ``blend_s*`` 臂）；落到 h1020 底座的只有一个硬编码的
``h1020_s0.5``（脚本里取 ``--lambdas`` 的第 2 项）。也就是说：

* 真正投产的那个组合，λ 只测过 **1 个点**；
* 在测过三个点的那个底座上，λ 的排序还随数据态翻转
  （``fundamental_blend_920v2``：s0.7 > s0.5 > s0.3；``..._fundind_full``：s0.3 > s0.5 > s0.7）。

本脚本把 λ 网格搬到定版底座上补齐，production 口径（T+1 open / buffer(15,40) /
月频 / 固定 10bp / 1e6）。λ=0.5 臂必须**逐位复现** ``buffer_arm_920 --variant defv``
的锚（16.31% / 0.738），否则口径有问题。

用法：
    python -m scripts.evaluation.blend_weight_920
    python -m scripts.evaluation.blend_weight_920 --lams 0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("blend_weight_920")

#: 锚：λ=0.5 必须复现的值（buffer_arm_920 --variant defv，2026-10-21 复算）
ANCHOR = {"annual": 0.1631, "sharpe": 0.738}
TOL = {"annual": 0.0015, "sharpe": 0.006}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="定版信号 λ 网格（h1020 底座）")
    ap.add_argument("--pred-dir", default=str(ROOT / "reports" / "alla_rolling_ortho920fundind"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "blend_weight_920"))
    ap.add_argument("--lams", default="0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0")
    ap.add_argument("--entry", type=float, default=0.15)
    ap.add_argument("--exit-frac", type=float, default=0.40)
    ap.add_argument("--ic-months", type=int, default=24)
    ap.add_argument("--embargo-months", type=int, default=2)
    ap.add_argument("--min-cov", type=float, default=0.5)
    ap.add_argument("--min-months", type=int, default=12)
    args = ap.parse_args(argv)

    from backtest.costs import default_costs
    from backtest.engine import VectorBacktest, build_execution_split
    from scripts.common import alla_common as AC
    from scripts.evaluation.fundamental_blend import build_slow_panel
    from scripts.evaluation.horizon_mix import load
    from strategy.examples import BufferedTopFracLongOnly

    pred_dir, dest = Path(args.pred_dir), Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    h1, h5 = load(pred_dir, "gbdt__h1"), load(pred_dir, "gbdt__h5")
    h10, h20 = load(pred_dir, "gbdt__h10"), load(pred_dir, "gbdt__h20")
    fast = AC.rank_average([h1, h5, h10, h20])

    base = AC.load_base()
    slow, _slow_max, _w = build_slow_panel(
        base, fast.index, args.ic_months, args.embargo_months,
        args.min_cov, args.min_months, log=print)
    log.info("slow 面板 %d 日；开始 λ 网格", slow.shape[0])

    fr = fast.rank(axis=1, pct=True)
    sr = slow.rank(axis=1, pct=True)

    close = base["close"]
    mask = base["mask"].astype(bool)
    costs = default_costs()
    fwd = close.pct_change(fill_method=None).reindex(close.index)
    oos_days = fast.index
    fwd = fwd.loc[oos_days]
    mask_oos = mask.reindex(index=oos_days, columns=close.columns).fillna(True)
    bench_idx = base["bench_index"].reindex(oos_days).fillna(0.0)
    bench_eqw = base["bench_eqw"].reindex(oos_days).fillna(0.0)

    fp = pred_dir / "_base" / "open_adj.parquet"
    if not fp.exists():
        raise SystemExit(f"{fp} 缺失：先跑 rolling_grid_alla --stage prep")
    fill = pd.read_parquet(fp)
    s_ = pd.Series(oos_days, index=oos_days)
    firsts = s_.groupby(s_.index.to_period("M")).first()
    pos_ = {d: i for i, d in enumerate(oos_days)}
    rb_signal = {t for t in firsts if pos_[t] + 1 < len(oos_days)}
    rb_exec = {oos_days[pos_[t] + 1] for t in rb_signal}
    exec_split = build_execution_split(fill, close, rb_signal)

    rows = []
    for lam in [float(x) for x in args.lams.split(",")]:
        sig = (1.0 - lam) * fr + lam * sr
        sig_in = sig.reindex(index=oos_days, columns=close.columns)
        strat = BufferedTopFracLongOnly(args.entry, args.exit_frac)
        bt = VectorBacktest(strategy=strat, rebalance_freq="M",
                            initial_capital=1_000_000.0, costs=costs)
        res = bt.run(sig_in.shift(1).where(mask_oos), fwd, horizon=1,
                     rebalance_days=rb_exec, execution_split=exec_split)
        m = AC.res_metrics(res.daily_returns, bench_idx, res.turnover_series)
        dr = res.daily_returns.dropna()
        row = {"lam": lam, "annual": m["annual"], "sharpe": m["sharpe"],
               "ir": m["ir"], "max_dd": m["max_dd"],
               "excess_idx": m["excess"], "turnover": m["turnover"]}
        for y, sub in dr.groupby(dr.index.year):
            row[f"ret_{y}"] = float((1 + sub).prod() - 1)
        rows.append(row)
        log.info("λ=%.2f | 年化=%.2f%% Sharpe=%.3f 回撤=%.1f%% 换手=%.1f%%",
                 lam, m["annual"] * 100, m["sharpe"], m["max_dd"] * 100,
                 m["turnover"] * 100)

    table = pd.DataFrame(rows)
    table.to_csv(dest / "lam_grid.csv", index=False, encoding="utf-8-sig")

    # 锚自检：λ=0.5 必须复现 buffer_arm_920 --variant defv
    hit = table[np.isclose(table["lam"], 0.5)]
    if not hit.empty:
        a = float(hit["annual"].iloc[0])
        s = float(hit["sharpe"].iloc[0])
        ok = abs(a - ANCHOR["annual"]) <= TOL["annual"] and \
            abs(s - ANCHOR["sharpe"]) <= TOL["sharpe"]
        log.info("锚自检 λ=0.5：%.2f%% / %.3f vs 登记 %.2f%% / %.3f → %s",
                 a * 100, s, ANCHOR["annual"] * 100, ANCHOR["sharpe"],
                 "通过" if ok else "❌ 不符，先查口径")
        if not ok:
            raise SystemExit("λ=0.5 未复现锚点，网格结果不可用")

    print(table[["lam", "annual", "sharpe", "max_dd", "turnover"]].round(4).to_string(index=False))
    print(f"完成 {time.time() - t0:.0f}s → {dest / 'lam_grid.csv'}")


if __name__ == "__main__":
    main()
