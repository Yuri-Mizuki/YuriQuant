"""多模型×多标签预测合成对照（Task ⑥ / AI29·国金19 语义）。

问题：预测层合成时，"多模型族"与"多 horizon 标签"两个维度各贡献多少？
  - 成员 12 个 = 3 模型族（gbdt / ranker / ridge）× 4 horizon（h1/h5/h10/h20）。
    各 horizon 的训练目标本身不同（h1 短视 → h20 长视），
    即 AI29"多标签"维度的现有 pred 等价物；ir/calmar 另类标签需重训（挂 920）。
  - A 等权组（零学习，RG._rank_average 与生产同源）：
      solo 单成员（模型基本盘）→ horizon 内三模型等权 → 族内四 horizon 等权
      → 全 12 等权，逐级看增益来自哪个维度。
  - B 学权组：walk-forward 秩空间 lstsq（与 stack_blend 完全同参：
    window 24 / embargo 2 / min 12 / sample 5 / fwd20 目标），
    stack_all12 vs 等权，判断"学权"在 12 成员上是否还有增量。
口径：close 成交（09-14 终审）+ executable_mask + 月频 Top10% 等权，
收益面板未 shift pct_change(fill_method=None)。

用法：
    python -m scripts.evaluation.member_blend12 \
        --pred-dir reports/alla_rolling_ortho --out reports/member_blend12
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import default_costs
from backtest.engine import VectorBacktest
from scripts.pipelines import rolling_grid_alla as RG
from strategy.examples import TopFracLongOnly

MODELS = ("gbdt", "ranker", "ridge")
HORIZONS = ("h1", "h5", "h10", "h20")


def main():
    ap = argparse.ArgumentParser(description="多模型×多标签 12 成员合成对照")
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho")
    ap.add_argument("--out", default="reports/member_blend12")
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--window-months", type=int, default=24)
    ap.add_argument("--embargo-months", type=int, default=2)
    ap.add_argument("--min-months", type=int, default=12)
    ap.add_argument("--sample-every", type=int, default=5)
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

    # ---- 12 成员：原面板（等权组用）+ 秩面板（学权组用）----
    members = {f"{m}__{h}": load(f"{m}__{h}")
               for m in MODELS for h in HORIZONS}
    ranks = {k: v.reindex(index=oos_days, columns=close.columns)
             .rank(axis=1, pct=True).astype(np.float32)
             for k, v in members.items()}
    print(f"[load] 12 成员就绪 {time.time() - t0:.0f}s", flush=True)

    # ---- A 等权组变体（_rank_average 与生产 ens 同口径）----
    def rp(name):  # 单成员信号：截面 pct 秩
        return ranks[name]

    def eq(names):
        return RG._rank_average([members[n] for n in names]) \
            .reindex(index=oos_days, columns=close.columns) \
            .rank(axis=1, pct=True)

    variants = {
        # 单成员基本盘（h1 代表短视标签）
        "solo_gbdt_h1": rp("gbdt__h1"),
        "solo_ranker_h1": rp("ranker__h1"),
        "solo_ridge_h1": rp("ridge__h1"),
        # 生产基线
        "baseline_ens_h1h5": eq(["gbdt__h1", "gbdt__h5"]),
        "h1020_baseline": eq(["gbdt__h1", "gbdt__h5",
                              "gbdt__h10", "gbdt__h20"]),
        # horizon 内跨模型等权（多模型维度的纯贡献）
        "eq_h1_3": eq(["gbdt__h1", "ranker__h1", "ridge__h1"]),
        "eq_h5_3": eq(["gbdt__h5", "ranker__h5", "ridge__h5"]),
        "eq_h10_3": eq(["gbdt__h10", "ranker__h10", "ridge__h10"]),
        "eq_h20_3": eq(["gbdt__h20", "ranker__h20", "ridge__h20"]),
        # 模型族内跨 horizon 等权（多标签维度的纯贡献；gbdt4 == h1020）
        "eq_ranker4": eq([f"ranker__{h}" for h in HORIZONS]),
        "eq_ridge4": eq([f"ridge__{h}" for h in HORIZONS]),
        # 全 12 等权（两维度全开）
        "eq_all12": eq(list(members)),
    }

    # ---- B 学权组：walk-forward 元学习（与 stack_blend 同参同目标）----
    train_days = oos_days[::args.sample_every]
    fwd20 = close.pct_change(20, fill_method=None).shift(-20)
    y_train = fwd20.reindex(train_days).rank(axis=1, pct=True)
    mnames = list(members)
    months_all = pd.PeriodIndex(oos_days, freq="M").unique()
    w_rows = []
    for m in months_all:
        window = [mm for mm in months_all if mm <= m - args.embargo_months]
        window = window[-args.window_months:]
        wset = set(window)
        td_mask = np.array([d.to_period("M") in wset for d in train_days])
        n_m = int(td_mask.sum())
        if n_m >= args.min_months:
            Xd = train_days[td_mask]
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
    wmat.to_csv(dest / "stack_weights12.csv", encoding="utf-8-sig")
    print(f"[stack] 元学习完成 {time.time() - t0:.0f}s；"
          f"末月权重 {wmat.iloc[-1].to_dict()}", flush=True)

    month_of_day = pd.PeriodIndex(oos_days, freq="M")
    wday = wmat.loc[[str(m) for m in month_of_day]]
    arr = np.stack([ranks[k].to_numpy() for k in mnames])  # (12, T, N)
    Wd = wday[mnames].to_numpy().T.astype(np.float32)[:, :, None]
    finite = np.isfinite(arr)

    a12 = arr
    w12 = Wd
    fin12 = finite
    num = np.nansum(np.where(fin12, a12 * w12, 0.0), axis=0)
    den = (fin12 & (w12 > 0)).sum(axis=0)
    stack_sig = pd.DataFrame(np.where(den > 0, num / den, np.nan),
                             index=oos_days, columns=close.columns)
    variants["stack_all12"] = stack_sig

    # ---- 统一回测：close 成交 + executable_mask，月频 Top10% ----
    rows_overall, rows_yearly = [], []
    for name, sig in variants.items():
        strat = TopFracLongOnly(frac=args.frac, weight_mode="equal")
        bt = VectorBacktest(strategy=strat, rebalance_freq="M",
                            initial_capital=1_000_000.0, costs=costs)
        res = bt.run(sig, fwd, executable_mask=mask_oos, horizon=1,
                     rebalance_days=None)
        dr = res.daily_returns
        m = RG.res_metrics(dr, bench_idx, res.turnover_series)
        row = {"run_id": name, "annual": m["annual"], "excess_idx": m["excess"],
               "excess_eqw": m["annual"] - RG._ann(bench_eqw),
               "sharpe": m["sharpe"], "max_dd": m["max_dd"],
               "turnover": m["turnover"], "n_days": len(dr.dropna())}
        rows_overall.append(row)
        rows_yearly.extend(RG._yearly_rows(row, dr, bench_idx,
                                           res.turnover_series))
        print(f"[{name}] 年化={row['annual'] * 100:.2f}% "
              f"超额(指)={row['excess_idx'] * 100:+.2f}pp "
              f"超额(等权)={row['excess_eqw'] * 100:+.2f}pp "
              f"Sharpe={row['sharpe']:.2f} 回撤={row['max_dd'] * 100:.1f}% "
              f"换手={row['turnover'] * 100:.1f}%", flush=True)

    pd.DataFrame(rows_overall).to_csv(dest / "metrics_member_blend12.csv",
                                      index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_yearly).to_csv(dest / "metrics_yearly12.csv",
                                     index=False, encoding="utf-8-sig")
    print(f"完成 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
