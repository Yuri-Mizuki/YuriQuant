"""定版信号上的「目标波动」仓位 overlay 复验（风控层，production 口径）。

为什么要有这个脚本
------------------
项目对「组合层风控」的结论来自 ``reports/risk_budget/``（2026-10-10）与
``timing_probe``（10-14）——**波动率目标化 Sharpe 0.633→0.581，判负**。但那两处的
被测净值是 ``882 ortho · ens_h1h5 · M · raw · f0.10 · 满仓无 buffer``
（见 ``timing_probe.DEFAULT_EQUITY`` 注释），**不是**现在的定版
``h1020⊕s0.5 + buffer(15/40) + T+1 open``。底座换了三代（882 → 920 v2 → fundind →
buffer），目标波动这条结论从未在新底座上复算。

本脚本把 overlay 搬到定版上：``w_t = ω_t · w_buffer(15/40)``，``ω_t`` 只由**调仓日
之前的**市场已实现波动决定（PIT，``shift(1)``），余额进现金。臂 = 目标波动
{10%, 12%, 15%, 20%} × 下限 {0.2, 0.5} + 满仓锚。锚（满仓）必须逐位复现
``buffer_arm_920 --variant defv`` 的 16.31% / 0.738。

判据（先验固定）：overlay 臂要「不降年化超过 1pp」且「Sharpe 高于锚 +0.03」才算有
价值——只压回撤不提升 Sharpe 的，按 ``position_overlay_grid`` 的既有口径归档。

用法：
    python -m scripts.evaluation.vol_target_920
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

log = setup_logging("vol_target_920")

ANCHOR = {"annual": 0.1631, "sharpe": 0.738}
TOL = {"annual": 0.0015, "sharpe": 0.006}


class ScaledStrategy:
    """把内层策略的权重整体乘一个仓位乘数 ω_t（余额=现金）。

    内层 ``BufferedTopFracLongOnly`` 是有状态的（缓冲带依赖上期持仓），乘法发生在
    ``get_weights`` **之后**，不影响其排名/缓冲判定。
    """

    def __init__(self, inner, omega: pd.Series, name: str):
        self.inner = inner
        self.omega = omega
        self.name = name

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        w = self.inner.get_weights(factor_values)
        if len(w) == 0:
            return w
        om = float(self.omega.get(factor_values.name, np.nan))
        if not np.isfinite(om):
            om = 1.0
        return w * om

    def get_weights_at(self, date, factor_values: pd.Series) -> pd.Series:
        return self.get_weights(factor_values)


def realized_vol(bench: pd.Series, window: int = 20) -> pd.Series:
    """年化已实现波动（只用当日及以前），再 ``shift(1)`` 供下一日决策使用。"""
    rv = bench.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252.0)
    return rv.shift(1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="定版信号的目标波动 overlay 复验")
    ap.add_argument("--pred-dir", default=str(ROOT / "reports" / "alla_rolling_ortho920fundind"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "vol_target_920"))
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--targets", default="0.10,0.12,0.15,0.20")
    ap.add_argument("--floors", default="0.2,0.5")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--consts", default="",
                    help="恒定仓位对照（逗号分隔），用来把「长期半仓」的水平效应"
                         "与择时的信息效应拆开：若 const_X 的 Sharpe ≈ vtX 的 Sharpe，"
                         "则该 vt 臂的优势全部来自降仓、不含择时 alpha")
    ap.add_argument("--entry", type=float, default=0.15)
    ap.add_argument("--exit-frac", type=float, default=0.40)
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
    slow, _m, _w = build_slow_panel(base, fast.index, 24, 2, 0.5, 12, log=print)
    sig = (1.0 - args.lam) * fast.rank(axis=1, pct=True) \
        + args.lam * slow.rank(axis=1, pct=True)

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

    rv = realized_vol(bench_eqw.reindex(oos_days), args.window)
    log.info("市场已实现波动（%d日）：中位 %.1f%%", args.window,
             float(rv.dropna().median()) * 100)

    arms: list[tuple[str, pd.Series | None]] = [("anchor_full", None)]
    for tgt in [float(x) for x in args.targets.split(",") if x.strip()]:
        for fl in [float(x) for x in args.floors.split(",") if x.strip()]:
            om = (tgt / rv).clip(fl, 1.0)
            arms.append((f"vt{int(tgt * 100)}_fl{fl:.1f}", om))
    for cv in [float(x) for x in args.consts.split(",") if x.strip()]:
        arms.append((f"const_{cv:.2f}", pd.Series(cv, index=oos_days)))

    sig_in = sig.reindex(index=oos_days, columns=close.columns)
    rows = []
    for name, om in arms:
        inner = BufferedTopFracLongOnly(args.entry, args.exit_frac)
        strat = inner if om is None else ScaledStrategy(inner, om, name)
        bt = VectorBacktest(strategy=strat, rebalance_freq="M",
                            initial_capital=1_000_000.0, costs=costs)
        res = bt.run(sig_in.shift(1).where(mask_oos), fwd, horizon=1,
                     rebalance_days=rb_exec, execution_split=exec_split)
        m = AC.res_metrics(res.daily_returns, bench_idx, res.turnover_series)
        dr = res.daily_returns.dropna()
        row = {"arm": name, "annual": m["annual"], "sharpe": m["sharpe"],
               "ir": m["ir"], "max_dd": m["max_dd"],
               "excess_idx": m["excess"], "turnover": m["turnover"],
               "avg_exposure": float(om.reindex(oos_days).mean()) if om is not None else 1.0}
        for y, sub in dr.groupby(dr.index.year):
            row[f"ret_{y}"] = float((1 + sub).prod() - 1)
        rows.append(row)
        log.info("%-14s 年化=%.2f%% Sharpe=%.3f 回撤=%.1f%% 换手=%.1f%% 均仓=%.2f",
                 name, m["annual"] * 100, m["sharpe"], m["max_dd"] * 100,
                 m["turnover"] * 100, row["avg_exposure"])

    table = pd.DataFrame(rows)
    table.to_csv(dest / "vol_target.csv", index=False, encoding="utf-8-sig")

    a = table.iloc[0]
    ok = abs(a["annual"] - ANCHOR["annual"]) <= TOL["annual"] and \
        abs(a["sharpe"] - ANCHOR["sharpe"]) <= TOL["sharpe"]
    log.info("锚自检：%.2f%% / %.3f vs 登记 %.2f%% / %.3f → %s",
             a["annual"] * 100, a["sharpe"], ANCHOR["annual"] * 100, ANCHOR["sharpe"],
             "通过" if ok else "❌ 不符，先查口径")
    if not ok:
        raise SystemExit("满仓锚未复现，overlay 结果不可比")

    print(table[["arm", "annual", "sharpe", "max_dd", "turnover", "avg_exposure"]]
          .round(4).to_string(index=False))
    print(f"完成 {time.time() - t0:.0f}s → {dest / 'vol_target.csv'}")


if __name__ == "__main__":
    main()
