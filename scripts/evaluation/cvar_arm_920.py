"""批次 12⑤：CVaR 约束组合臂 vs Top10% 等权 vs buffer 形态（production 口径）。

GOOD_MACHINE_TASKS 批次 12⑤：``solve_portfolio`` 的 CVaR_α 约束
（RU 线性化 / OSQP max_iter=200k / eps 1e-8，``optimize/solver.py`` 已入库带
6 测试）对照 Top10% 等权与 buffer 形态，production 口径对拍（T+1 open、
固定 10bp、1e6 资金、buffer_arm_920 同 harness）。基座 = 920 定版信号
（defv = h1020 秩平均 ⊕ 慢信号 s0.5，``reports/frontier_audit_920_reval/
defv_sig__h1.parquet``）。

臂设计（同一信号、同一调仓日）：
  - naive_top15 / buffer_15_40：production 锚（与 buffer_arm_920 defv 对照臂
    逐位一致，兼作本脚本口径自检）；
  - top10_ew：任务书 Top10% 等权（QP 臂的同宇宙等权对照）；
  - cvar_cX：top10% 宇宙内 mvo + CVaR 约束（滚动 LW/收缩 Σ、rank pct α、
    alpha_scale="auto"、个股上限 ``--max-weight``、日频场景 = 调仓日前
    ``--window`` 日收益）；c 档位网格 ``--cvar-limits``；
  - cvar_cX_tn：同上 + 单边换手硬约束 ``--max-turnover``（QP 控换手示范臂）。

用法:
    python -m scripts.evaluation.cvar_arm_920 \
        --sig reports/frontier_audit_920_reval/defv_sig__h1.parquet \
        --out reports/cvar_arm_920
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
from strategy.base import Strategy  # noqa: E402

log = setup_logging("cvar_arm_920")


class CVaRTopFracLongOnly(Strategy):
    """top-frac 宇宙内 CVaR 约束 mvo（月频调仓、日频场景历史模拟）。

    防前视：场景收益与 Σ 只用**严格早于**调仓执行日的 ``window`` 日收盘收益；
    求解失败 / 样本不足时降级为宇宙内等权（留痕计数）。
    """

    def __init__(self, close: pd.DataFrame, frac: float, window: int,
                 cvar_alpha: float, cvar_limit: float, max_weight: float,
                 max_turnover: float | None = None, min_periods: int = 60):
        self.close = close
        self.frac = float(frac)
        self.window = int(window)
        self.cvar_alpha = float(cvar_alpha)
        self.cvar_limit = float(cvar_limit)
        self.max_weight = float(max_weight)
        self.max_turnover = max_turnover
        self.min_periods = int(min_periods)
        self.name = (f"cvar_a{cvar_alpha:.2f}_c{cvar_limit:.3f}"
                     + (f"_tn{max_turnover:.2f}" if max_turnover else ""))
        self._prev_w: pd.Series | None = None
        self.n_fallback = 0
        self.n_solve = 0

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        return self.get_weights_at(factor_values.name, factor_values)

    def get_weights_at(self, date: pd.Timestamp,
                       factor_values: pd.Series) -> pd.Series:
        from optimize.solver import estimate_covariance, solve_portfolio

        vals = factor_values.dropna()
        if len(vals) == 0:
            return pd.Series(dtype=float)
        n_target = max(int(round(self.frac * len(vals))), 1)
        univ = vals.nlargest(n_target).index

        hist = self.close.reindex(columns=univ)
        hist = hist.loc[hist.index < date].tail(self.window + 1)
        rets = hist.pct_change(fill_method=None).iloc[1:]
        # 大截面逐日全非 NaN 几乎不可能（停牌洞）→ 列覆盖 ≥80% 过滤 +
        # 残余洞保守零填充，否则 usable 不足 min_periods 触发大面积等权降级
        cov_ok = rets.notna().mean(axis=0) >= 0.8
        rets = rets.loc[:, cov_ok].fillna(0.0)
        usable = rets
        if len(usable) < self.min_periods or usable.shape[1] < 2:
            self.n_fallback += 1
            w = pd.Series(1.0 / len(univ), index=univ)
            self._prev_w = w
            return w.reindex(factor_values.index).fillna(0.0)

        try:
            Sigma = estimate_covariance(usable)
            alpha_s = vals.rank(pct=True).reindex(usable.columns).dropna()
            w = solve_portfolio(
                alpha=alpha_s, Sigma=Sigma, method="mvo",
                alpha_scale="auto", risk_aversion=1.0,
                max_weight=self.max_weight,
                scenario_returns=usable, cvar_limit=self.cvar_limit,
                cvar_alpha=self.cvar_alpha,
                prev_weights=self._prev_w.reindex(alpha_s.index)
                if self._prev_w is not None else None,
                max_turnover=self.max_turnover,
            )
            if w.abs().sum() <= 1e-8:
                raise RuntimeError("退化零解")
            self.n_solve += 1
        except Exception as exc:  # 求解失败 → 等权降级（留痕）
            self.n_fallback += 1
            log.debug("cvar solve 降级 %s: %s", date.date(), exc)
            w = pd.Series(1.0 / len(univ), index=univ)
        w = w.reindex(factor_values.index).fillna(0.0)
        self._prev_w = w[w != 0]
        return w


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sig", default="reports/frontier_audit_920_reval/"
                                     "defv_sig__h1.parquet",
                    help="定版信号面板（defv）路径")
    ap.add_argument("--out", default="reports/cvar_arm_920")
    ap.add_argument("--entry", type=float, default=0.15,
                    help="production 锚臂的 entry（naive/buffer）")
    ap.add_argument("--exit", dest="exit_frac", type=float, default=0.40)
    ap.add_argument("--frac-qp", type=float, default=0.10,
                    help="QP 臂宇宙比例（任务书 Top10%）")
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--cvar-alpha", type=float, default=0.95)
    ap.add_argument("--cvar-limits", default="0.03,0.05")
    ap.add_argument("--max-weight", type=float, default=0.05)
    ap.add_argument("--max-turnover", type=float, default=0.35)
    ap.add_argument("--execution", default="open", choices=["close", "open", "vwap"])
    ap.add_argument("--freq", default="M", choices=["M", "W"])
    args = ap.parse_args(argv)

    from backtest.costs import default_costs
    from backtest.engine import VectorBacktest, build_execution_split
    from scripts.common import alla_common as AC
    from strategy.examples import BufferedTopFracLongOnly, TopFracLongOnly

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sig = pd.read_parquet(args.sig)
    base = AC.load_base()
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
        pred_dir = Path("reports/alla_rolling_ortho920fundind")
        fp = pred_dir / "_base" / f"{args.execution}_adj.parquet"
        if not fp.exists():
            raise SystemExit(f"{fp} 缺失：先跑 rolling_grid_alla --stage prep")
        fill = pd.read_parquet(fp)
        s_ = pd.Series(oos_days, index=oos_days)
        firsts = s_.groupby(s_.index.to_period("M")).first()
        pos_ = {d: i for i, d in enumerate(oos_days)}
        rb_signal = {t for t in firsts if pos_[t] + 1 < len(oos_days)}
        rb_exec = {oos_days[pos_[t] + 1] for t in rb_signal}
        exec_split = build_execution_split(fill, close, rb_signal)

    sig_in = sig.reindex(index=oos_days, columns=close.columns)
    close_hist = close.reindex(columns=close.columns)

    cvars = [float(x) for x in args.cvar_limits.split(",")]
    arms: list[tuple[str, Strategy]] = [
        ("naive_top15", TopFracLongOnly(frac=args.entry, weight_mode="equal")),
        (f"buffer_{int(args.entry*100)}_{int(args.exit_frac*100)}",
         BufferedTopFracLongOnly(args.entry, args.exit_frac)),
        ("top10_ew", TopFracLongOnly(frac=args.frac_qp, weight_mode="equal")),
    ]
    for c in cvars:
        arms.append((f"cvar_c{c:.3f}",
                     CVaRTopFracLongOnly(close_hist, args.frac_qp, args.window,
                                         args.cvar_alpha, c, args.max_weight)))
    arms.append((f"cvar_c{cvars[-1]:.3f}_tn{args.max_turnover:.2f}",
                 CVaRTopFracLongOnly(close_hist, args.frac_qp, args.window,
                                     args.cvar_alpha, cvars[-1], args.max_weight,
                                     max_turnover=args.max_turnover)))

    rows = []
    for name, strat in arms:
        bt = VectorBacktest(strategy=strat, rebalance_freq=args.freq,
                            initial_capital=1_000_000.0, costs=costs)
        if exec_split is not None:
            res = bt.run(sig_in.shift(1).where(mask_oos), fwd, horizon=1,
                         rebalance_days=rb_exec, execution_split=exec_split)
        else:
            res = bt.run(sig_in, fwd, executable_mask=mask_oos, horizon=1,
                         rebalance_days=None)
        m = AC.res_metrics(res.daily_returns, bench_idx, res.turnover_series)
        extra = {}
        if hasattr(strat, "n_solve"):
            extra = {"n_solve": strat.n_solve, "n_fallback": strat.n_fallback}
        rows.append({"variant": name, "annual": m["annual"],
                     "excess_idx": m["excess"],
                     "excess_eqw": m["annual"] - AC.ann(bench_eqw),
                     "sharpe": m["sharpe"], "ir": m["ir"], "max_dd": m["max_dd"],
                     "turnover": m["turnover"],
                     "n_days": len(res.daily_returns.dropna()), **extra})
        log.info("%s | 年化=%.2f%% 超额(指)=%+.2fpp Sharpe=%.3f 换手=%.1f%% %s",
                 name, m["annual"] * 100, m["excess"] * 100, m["sharpe"],
                 m["turnover"] * 100,
                 extra if extra else "")

    table = pd.DataFrame(rows)
    table.to_csv(dest / "cvar_arm.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print(f"完成 {time.time() - t0:.0f}s → {dest}")


if __name__ == "__main__":
    main()
