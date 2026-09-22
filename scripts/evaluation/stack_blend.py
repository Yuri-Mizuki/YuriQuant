"""E6 stacking：滚动元学习器**学**融合权重（对齐 CFA ensemble / MSIF-OEM 两阶段
框架的简化版——融合层从手工 λ 网格升级为数据驱动）。

原料：滚动网格的全量 OOS 预测面板本身就是 out-of-fold 预测，可直接做元学习：
  - 成员（7 个）：gbdt h1/h5/h10/h20 + ranker h1 + ridge h1 + 基本面慢信号
    （复用 fundamental_blend.build_slow_panel 的 ICIR 版）
  - 元模型：秩空间线性凸组合（lstsq → 负权截零 → 归一化；不加非常数项，
    防元过拟合）
  - walk-forward：逐月重训，训练集 = 该月前 {embargo_months} 个月末之前的
    周采样日（目标 = fwd20 截面秩，慢信号与月频持仓同视野），滚动窗
    {window_months} 个月；历史不足 {min_months} 个月退化为等权（=h1020 基线）
  - 产出：stack_fast4（4 个 gbdt horizon 的学权组合，对照 h1020 等权）
    与 stack_all7（含慢信号/两模型）

用法：
    python -m scripts.evaluation.stack_blend \
        --pred-dir reports/alla_rolling_ortho --out reports/stack_blend
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.pipelines import rolling_grid_alla as RG
from scripts.evaluation.fundamental_blend import build_slow_panel
from backtest.costs import default_costs
from backtest.engine import VectorBacktest, build_execution_split
from strategy.examples import TopFracLongOnly


def main():
    ap = argparse.ArgumentParser(description="E6 stacking 学权融合")
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho")
    ap.add_argument("--out", default="reports/stack_blend")
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--execution", default="open", choices=["close", "open", "vwap"])
    ap.add_argument("--window-months", type=int, default=24)
    ap.add_argument("--embargo-months", type=int, default=2)
    ap.add_argument("--min-months", type=int, default=12)
    ap.add_argument("--sample-every", type=int, default=5,
                    help="训练样本日抽样间隔（交易日）")
    args = ap.parse_args()
    pred_dir, dest = Path(args.pred_dir), Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    def load(n):
        return pd.read_parquet(pred_dir / "pred" / f"{n}.parquet")

    base = RG.load_base()
    close = base["close"]
    mask = base["mask"].astype(bool)
    costs = default_costs()
    oos_days = load("gbdt__h1").index
    fwd = close.pct_change(fill_method=None).reindex(close.index).loc[oos_days]
    mask_oos = mask.reindex(index=oos_days, columns=close.columns).fillna(True)
    bench_idx = base["bench_index"].reindex(oos_days).fillna(0.0)
    bench_eqw = base["bench_eqw"].reindex(oos_days).fillna(0.0)

    # ---- 成员秩面板 ----
    slow, _slow_max, _w = build_slow_panel(
        base, oos_days, 24, args.embargo_months, 0.5, args.min_months)
    members = {
        "gbdt_h1": load("gbdt__h1"), "gbdt_h5": load("gbdt__h5"),
        "gbdt_h10": load("gbdt__h10"), "gbdt_h20": load("gbdt__h20"),
        "ranker_h1": load("ranker__h1"), "ridge_h1": load("ridge__h1"),
        "slow_icir": slow,
    }
    mnames = list(members)
    ranks = {k: v.reindex(index=oos_days, columns=close.columns)
             .rank(axis=1, pct=True).astype(np.float32) for k, v in members.items()}
    print(f"[load] 7 成员秩面板就绪 {time.time() - t0:.0f}s", flush=True)

    # ---- 逐月 walk-forward 元学习 ----
    train_days = oos_days[::args.sample_every]
    fwd20 = close.pct_change(20, fill_method=None).shift(-20)
    y_train = fwd20.reindex(train_days).rank(axis=1, pct=True)
    X_days = train_days  # 保持 DatetimeIndex（元素为 Timestamp，支持 to_period）
    months_all = pd.PeriodIndex(oos_days, freq="M").unique()
    w_rows = []
    for m in months_all:
        window = [mm for mm in months_all if mm <= m - args.embargo_months]
        window = window[-args.window_months:]
        wset = set(window)
        td_mask = np.array([d.to_period("M") in wset for d in X_days])
        n_m = int(td_mask.sum())
        if n_m >= args.min_months:
            Xd = X_days[td_mask]
            X = np.column_stack([
                ranks[k].loc[Xd].to_numpy().reshape(-1) for k in mnames])
            y = y_train.loc[Xd].to_numpy().reshape(-1)
            ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            if ok.sum() > 10_000:
                coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
                coef = np.clip(coef, 0.0, None)
                w = coef / coef.sum() if coef.sum() > 0 else None
            else:
                w = None
        else:
            w = None
        if w is None:
            w = np.full(len(mnames), 1.0 / len(mnames))
        w_rows.append({"signal_month": str(m), **{k: round(float(x), 4)
                       for k, x in zip(mnames, w)}})
    wmat = pd.DataFrame(w_rows).set_index("signal_month")
    wmat.to_csv(dest / "stack_weights.csv", encoding="utf-8-sig")
    print(f"[stack] 元学习完成 {time.time() - t0:.0f}s；"
          f"末月权重 {wmat.iloc[-1].to_dict()}", flush=True)

    # ---- 权重应用到日频面板 ----
    month_of_day = pd.PeriodIndex(oos_days, freq="M")
    wday = wmat.loc[[str(m) for m in month_of_day]]  # (T, 7)
    arr = np.stack([ranks[k].to_numpy() for k in mnames])  # (7, T, N)
    Wd = wday[mnames].to_numpy().T.astype(np.float32)[:, :, None]
    finite = np.isfinite(arr)

    def combine(sel, weights_override=None):
        a = arr[sel]
        w = (Wd[sel] if weights_override is None
             else weights_override.astype(np.float32)[:, :, None])
        finite_sel = finite[sel]
        num = np.nansum(np.where(finite_sel, a * w, 0.0), axis=0)
        den = (finite_sel & (w > 0)).sum(axis=0)
        return pd.DataFrame(np.where(den > 0, num / den, np.nan),
                            index=oos_days, columns=close.columns)

    h1020 = RG._rank_average([members["gbdt_h1"], members["gbdt_h5"],
                              members["gbdt_h10"], members["gbdt_h20"]]) \
        .reindex(index=oos_days, columns=close.columns).rank(axis=1, pct=True)
    variants = {
        # 基线：等权秩平均（RG._rank_average，与生产 ens 同源）
        "baseline_ens_h1h5": RG._rank_average(
            [members["gbdt_h1"], members["gbdt_h5"]])
            .reindex(index=oos_days, columns=close.columns).rank(axis=1, pct=True),
        "h1020_baseline": h1020,
        "stack_fast4": combine([0, 1, 2, 3]),   # 4-horizon 学权（对照 h1020 等权）
        "stack_all7": combine(list(range(7))),  # +ranker/ridge/慢信号 学权
        "h1020_s0.5": 0.5 * h1020 + 0.5 * ranks["slow_icir"],  # E3 参照
    }

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
        print(f"[exec] 执行价口径: {args.execution}（fill={fp.name}，{len(rb_exec)} 个）",
              flush=True)

    sfx = {"close": "_close", "open": ""}.get(args.execution, f"_{args.execution}")
    rows_overall, rows_yearly = [], []
    for name, sig in variants.items():
        strat = TopFracLongOnly(frac=args.frac, weight_mode="equal")
        bt = VectorBacktest(strategy=strat, rebalance_freq="M",
                            initial_capital=1_000_000.0, costs=costs)
        if exec_split is not None:
            res = bt.run(sig.shift(1).where(mask_oos), fwd, horizon=1,
                         rebalance_days=rb_exec, execution_split=exec_split)
        else:
            res = bt.run(sig, fwd, executable_mask=mask_oos, horizon=1,
                         rebalance_days=None)
        dr = res.daily_returns
        m = RG.res_metrics(dr, bench_idx, res.turnover_series)
        row = {"run_id": name, "annual": m["annual"], "excess_idx": m["excess"],
               "excess_eqw": m["annual"] - RG._ann(bench_eqw),
               "sharpe": m["sharpe"], "max_dd": m["max_dd"],
               "turnover": m["turnover"], "n_days": len(dr.dropna())}
        rows_overall.append(row)
        rows_yearly.extend(RG._yearly_rows(row, dr, bench_idx, res.turnover_series))
        print(f"[{name}] 年化={row['annual'] * 100:.2f}% "
              f"超额(指)={row['excess_idx'] * 100:+.2f}pp "
              f"超额(等权)={row['excess_eqw'] * 100:+.2f}pp "
              f"Sharpe={row['sharpe']:.2f} 回撤={row['max_dd'] * 100:.1f}% "
              f"换手={row['turnover'] * 100:.1f}%", flush=True)

    pd.DataFrame(rows_overall).to_csv(dest / f"metrics_stack_blend{sfx}.csv",
                                      index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_yearly).to_csv(dest / f"metrics_yearly{sfx}.csv",
                                     index=False, encoding="utf-8-sig")
    print(f"完成 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
