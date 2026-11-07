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


def build_ctx(pred_dir: Path, *, ic_months: int = 24, embargo_months: int = 2,
              min_cov: float = 0.5, min_months: int = 12):
    """装配一次网格所需的重面板（快信号、慢信号、收益、执行切分）。

    抽出来是为了让裁决脚本复用**同一份**回测装配——两脚本各写一遍迟早漂移，
    而这里的每个参数（T+1 open / buffer / 月频）都是口径本身。
    """
    from backtest.engine import build_execution_split
    from scripts.common import alla_common as AC
    from scripts.evaluation.fundamental_blend import build_slow_panel
    from scripts.evaluation.horizon_mix import load

    h1, h5 = load(pred_dir, "gbdt__h1"), load(pred_dir, "gbdt__h5")
    h10, h20 = load(pred_dir, "gbdt__h10"), load(pred_dir, "gbdt__h20")
    fast = AC.rank_average([h1, h5, h10, h20])

    base = AC.load_base()
    slow, _slow_max, _w = build_slow_panel(
        base, fast.index, ic_months, embargo_months, min_cov, min_months, log=log.info)
    log.info("slow 面板 %d 日", slow.shape[0])

    close = base["close"]
    oos_days = fast.index
    fwd = close.pct_change(fill_method=None).reindex(close.index).loc[oos_days]
    mask_oos = base["mask"].astype(bool).reindex(index=oos_days, columns=close.columns).fillna(True)

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

    return {
        "fr": fast.rank(axis=1, pct=True),
        "sr": slow.rank(axis=1, pct=True),
        "close": close,
        "fwd": fwd, "mask_oos": mask_oos, "oos_days": oos_days,
        "bench_idx": base["bench_index"].reindex(oos_days).fillna(0.0),
        "rb_exec": rb_exec, "exec_split": exec_split,
    }


def run_arms(ctx: dict, lams, *, entry: float = 0.15, exit_frac: float = 0.40,
             cost_mult: float = 1.0) -> tuple[pd.DataFrame, dict[float, pd.Series]]:
    """在 ``build_ctx`` 的底座上扫一批 λ，返回（指标表, {λ: 日收益}）。

    ``cost_mult`` 只缩放费率（佣金/印花/滑点），用于「成本抬升后 λ 的排序会不会翻」
    这类稳健性检查；佣金最低 5 元不缩放（它是撮合规则，不是费率）。
    """
    from backtest.costs import default_costs
    from backtest.engine import VectorBacktest
    from scripts.common import alla_common as AC
    from strategy.examples import BufferedTopFracLongOnly

    costs = default_costs()
    if cost_mult != 1.0:
        costs = type(costs)(commission_rate=costs.commission_rate * cost_mult,
                            commission_min=costs.commission_min,
                            stamp_duty=costs.stamp_duty * cost_mult,
                            slippage_bp=costs.slippage_bp * cost_mult,
                            impact=costs.impact)

    rows, daily = [], {}
    for lam in [float(x) for x in lams]:
        sig = (1.0 - lam) * ctx["fr"] + lam * ctx["sr"]
        sig_in = sig.reindex(index=ctx["oos_days"], columns=ctx["close"].columns)
        bt = VectorBacktest(strategy=BufferedTopFracLongOnly(entry, exit_frac),
                            rebalance_freq="M", initial_capital=1_000_000.0, costs=costs)
        res = bt.run(sig_in.shift(1).where(ctx["mask_oos"]), ctx["fwd"], horizon=1,
                     rebalance_days=ctx["rb_exec"], execution_split=ctx["exec_split"])
        m = AC.res_metrics(res.daily_returns, ctx["bench_idx"], res.turnover_series)
        dr = res.daily_returns.dropna()
        daily[lam] = dr
        row = {"lam": lam, "annual": m["annual"], "sharpe": m["sharpe"],
               "ir": m["ir"], "max_dd": m["max_dd"],
               "excess_idx": m["excess"], "turnover": m["turnover"]}
        for y, sub in dr.groupby(dr.index.year):
            row[f"ret_{y}"] = float((1 + sub).prod() - 1)
        rows.append(row)
        log.info("λ=%.2f | 年化=%.2f%% Sharpe=%.3f 回撤=%.1f%% 换手=%.1f%%",
                 lam, m["annual"] * 100, m["sharpe"], m["max_dd"] * 100,
                 m["turnover"] * 100)
    return pd.DataFrame(rows), daily


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

    pred_dir, dest = Path(args.pred_dir), Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ctx = build_ctx(pred_dir, ic_months=args.ic_months,
                    embargo_months=args.embargo_months,
                    min_cov=args.min_cov, min_months=args.min_months)
    table, _daily = run_arms(ctx, args.lams.split(","), entry=args.entry,
                             exit_frac=args.exit_frac)
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
