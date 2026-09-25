"""批次 1.4：定版信号（h1020）的 buffer(20/30) 臂 vs naive top20。

GOOD_MACHINE_TASKS 批次 1.4：对 1.3 胜出信号加 ``BufferedTopFracLongOnly``
(entry 0.20 / exit 0.30) 对照（882 口径实测 buffer 年化 25.32%/换手 53.9% 优于
naive 24.50%/66.1%）。信号构造与执行口径逐行对齐 ``horizon_mix.py``
（ens_h1h5h10h20 秩平均、T+1 open 可执行、mask 预掩码）。

用法:
    python -m scripts.evaluation.buffer_arm_920 --pred-dir reports/alla_rolling_ortho920tl \
        --out reports/buffer_arm_920
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("buffer_arm_920")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho920tl")
    ap.add_argument("--out", default="reports/buffer_arm_920")
    ap.add_argument("--entry", type=float, default=0.20)
    ap.add_argument("--exit", dest="exit_frac", type=float, default=0.30)
    ap.add_argument("--execution", default="open", choices=["close", "open", "vwap"])
    args = ap.parse_args(argv)

    from backtest.costs import default_costs
    from backtest.engine import VectorBacktest, build_execution_split
    from scripts.evaluation.horizon_mix import load
    from scripts.pipelines import rolling_grid_alla as RG
    from strategy.examples import BufferedTopFracLongOnly, TopFracLongOnly

    pred_dir, dest = Path(args.pred_dir), Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    h1, h5 = load(pred_dir, "gbdt__h1"), load(pred_dir, "gbdt__h5")
    h10, h20 = load(pred_dir, "gbdt__h10"), load(pred_dir, "gbdt__h20")
    sig = RG._rank_average([h1, h5, h10, h20])

    base = RG.load_base()
    close = base["close"]
    mask = base["mask"].astype(bool)
    costs = default_costs()
    fwd = close.pct_change(fill_method=None).reindex(close.index)
    oos_days = sig.index
    fwd = fwd.loc[oos_days]
    mask_oos = mask.reindex(index=oos_days, columns=close.columns).fillna(True)
    bench_idx = base["bench_index"].reindex(oos_days).fillna(0.0)
    bench_eqw = base["bench_eqw"].reindex(oos_days).fillna(0.0)

    exec_split, rb_exec = None, None
    if args.execution != "close":
        fp = pred_dir / "_base" / f"{args.execution}_adj.parquet"
        if not fp.exists():
            raise SystemExit(f"{fp} 缺失：先跑 rolling_grid_alla --stage prep")
        fill = pd.read_parquet(fp)
        s_ = pd.Series(oos_days, index=oos_days)
        firsts = s_.groupby(s_.index.to_period("M")).first()
        pos_ = {d: i for i, d in enumerate(oos_days)}
        rb_exec = {oos_days[pos_[t] + 1] for t in firsts
                   if pos_[t] + 1 < len(oos_days)}
        exec_split = build_execution_split(fill, close, rb_exec)

    sig_in = sig.reindex(index=oos_days, columns=close.columns)
    variants = [
        ("naive_top20", lambda: TopFracLongOnly(frac=args.entry, weight_mode="equal")),
        (f"buffer_{int(args.entry * 100)}_{int(args.exit_frac * 100)}",
         lambda: BufferedTopFracLongOnly(args.entry, args.exit_frac)),
    ]
    rows = []
    for name, make_strat in variants:
        bt = VectorBacktest(strategy=make_strat(), rebalance_freq="M",
                            initial_capital=1_000_000.0, costs=costs)
        if exec_split is not None:
            res = bt.run(sig_in.shift(1).where(mask_oos), fwd, horizon=1,
                         rebalance_days=rb_exec, execution_split=exec_split)
        else:
            res = bt.run(sig_in, fwd, executable_mask=mask_oos, horizon=1,
                         rebalance_days=None)
        m = RG.res_metrics(res.daily_returns, bench_idx, res.turnover_series)
        rows.append({"variant": name, "annual": m["annual"],
                     "excess_idx": m["excess"],
                     "excess_eqw": m["annual"] - RG._ann(bench_eqw),
                     "sharpe": m["sharpe"], "ir": m["ir"], "max_dd": m["max_dd"],
                     "turnover": m["turnover"], "n_days": len(res.daily_returns.dropna())})
        log.info("%s | 年化=%.2f%% 超额(指)=%+.2fpp 换手=%.1f%%",
                 name, m["annual"] * 100, m["excess"] * 100, m["turnover"] * 100)

    table = pd.DataFrame(rows)
    table.to_csv(dest / "buffer_arm.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print(f"完成 {time.time() - t0:.0f}s → {dest}")


if __name__ == "__main__":
    main()
