"""E3 基本面慢信号 ⊕ 快信号双层融合（信达式分域建模 + 预测层 λ 加权）。

背景（RESEARCH_TODO §一 基本面诊断）：基本面弱因期限错配——它对 h20 的 |IC|
弹性 +14% 而量价族为负，且基本面信息偏线性（信达《深度学习揭秘之一：量价与
基本面结合》的结论）。本实验把基本面族（B 族财报 + A 族股东，与保留席位同族）
做成**滚动 ICIR 加权的线性慢信号**（月频更新、 PIT 安全：权重只用 embargo 后
的月末 IC），与既有 gbdt h1/h5 快信号在预测层秩空间加权融合，λ 扫描。

慢信号构造：
  - 因子集 = FUNDAMENTAL_FAMILY_SETS（56 B 族 + 7 A 族，panels_neu 口径）
  - IC 采样 = 月末交易日，spearman(z_i, fwd20)（慢信号配长视野标签）
  - 权重 = trailing {ic_months} 月 ICIR（clip [0,3]，负 ICIR 出局），embargo
    {embargo_months} 个月末；历史不足 {min_months} 个月时退化为等权
  - 合成 = 月初首个交易日的截面加权 nanmean；日频面板 ffill 展开

用法：
    python -m scripts.evaluation.fundamental_blend \
        --pred-dir reports/alla_rolling_ortho --out reports/fundamental_blend
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.pipelines import rolling_grid_alla as RG
from backtest.costs import default_costs
from backtest.engine import VectorBacktest, build_execution_split
from stats.ic import calc_ic_series
from strategy.examples import TopFracLongOnly


def month_firsts(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(days, index=days)
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).first())


def month_lasts(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(days, index=days)
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).last())


def load_family_slices(names: set, dates: pd.DatetimeIndex, root: Path) -> dict:
    """逐个加载 panels_neu 因子面板并只保留 needed 日期（控制内存峰值）。"""
    out = {}
    for n in sorted(names):
        p = root / "panels_neu" / f"{n}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        out[n] = df.reindex(index=dates).astype(np.float32)
        del df
    return out


def build_slow_panel(base: dict, oos_days: pd.DatetimeIndex, ic_months: int,
                     embargo_months: int, min_cov: float, min_months: int,
                     extra_family: set | None = None,
                     log=print) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """返回 (ICIR 慢信号面板, max-ICIR 慢信号面板, 末月权重表)。

    max-ICIR（华泰 6 法对比最优）：w ∝ Σ_s⁻¹μ，μ/Σ = trailing 月度 IC 的均值/
    协方差；24 样本估 n≈50 的协方差严重病态 → 收缩 Σ_s = 0.5Σ + 0.5diag(Σ) +
    1e-6 I，非负解经 active-set 迭代 2 轮。
    """
    close = base["close"]
    all_days = close.index
    t0_days = month_firsts(oos_days)
    ic_days_all = month_lasts(all_days)
    ic_days_all = ic_days_all[ic_days_all >= t0_days[0] - pd.Timedelta(days=500)]
    fwd20 = close.pct_change(20, fill_method=None).shift(-20)

    needed = pd.DatetimeIndex(sorted(set(t0_days) | set(ic_days_all)))
    family = set(RG.FUNDAMENTAL_FAMILY_SETS) | set(extra_family or ())
    panels = load_family_slices(family, needed, RG.ds_root())
    names = sorted(panels)
    log(f"[slow] 家族因子 {len(family)} 个、面板就绪 "
        f"{len(names)} 个（缺失 {sorted(family - set(names))}）")

    # 月末 IC（spearman = 秩的 pearson），覆盖率不足的月份置 NaN
    fwd_r = fwd20.reindex(ic_days_all).rank(axis=1, pct=True)
    ic_tab = {}
    for n in names:
        z = panels[n].loc[ic_days_all]
        cov = z.notna().mean(axis=1)
        ic = z.rank(axis=1, pct=True).corrwith(fwd_r, axis=1)
        ic_tab[n] = ic.where(cov >= min_cov)
    ic_tab = pd.DataFrame(ic_tab)  # index=月末日, columns=因子

    # 逐信号月：trailing ICIR / max-ICIR 权重（embargo）→ 截面加权 nanmean 合成
    comp_rows, comp_max_rows, weight_rows = {}, {}, []

    def _nanmean(w: pd.Series, t0) -> pd.Series:
        Zt = pd.concat({n: panels[n].loc[t0] for n in w.index}, axis=1)
        present = Zt.notna().mul(w, axis=1)
        return (Zt.mul(w, axis=1)).sum(axis=1) / present.sum(axis=1)

    for t0, m in zip(t0_days, pd.PeriodIndex(t0_days, freq="M")):
        window = ic_tab[(ic_tab.index.to_period("M") <= m - embargo_months)].tail(ic_months)
        mu, sd = window.mean(), window.std()
        icir = (mu / sd.replace(0.0, np.nan)).clip(0.0, 3.0).dropna()
        if len(window) >= min_months and icir.sum() > 0:
            w = icir / icir.sum()
        else:  # 历史不足：等权（覆盖率达标且当月有值的因子）
            ok = window.notna().sum() >= max(6, len(window) // 2)
            w = pd.Series(1.0, index=ok[ok].index)
            w = w / w.sum()
        comp_rows[t0] = _nanmean(w, t0)

        w_max = w
        cols = window.columns[window.notna().sum() >= max(6, int(len(window) * 0.6))]
        if len(window) >= min_months and len(cols) >= 5:
            W = window[cols].fillna(window[cols].mean())
            mu_v = W.mean().to_numpy()
            Sig = np.cov(W.to_numpy(), rowvar=False)
            Sig_s = 0.5 * Sig + 0.5 * np.diag(np.diag(Sig)) \
                + 1e-6 * np.eye(len(cols))
            try:
                wv = np.clip(np.linalg.solve(Sig_s, mu_v), 0.0, None)
                for _ in range(2):  # active-set 迭代逼近非负最优
                    act = wv > 0
                    if act.sum() < 2:
                        break
                    sub = Sig_s[np.ix_(act, act)]
                    sol = np.clip(np.linalg.solve(sub, mu_v[act]), 0.0, None)
                    if sol.sum() <= 0:
                        break
                    wv[:] = 0.0
                    wv[act] = sol / sol.sum()
                w_max = (wv / wv.sum()) if wv.sum() > 0 else w
                w_max = pd.Series(w_max, index=cols)
            except np.linalg.LinAlgError:
                pass
        comp_max_rows[t0] = _nanmean(w_max, t0)
        weight_rows.append({"signal_month": str(m), "n_factors": len(w),
                            **{k: round(float(v), 4) for k, v in
                               w.sort_values(ascending=False).head(8).items()}})

    slow = pd.DataFrame(comp_rows).T.reindex(columns=close.columns) \
        .reindex(oos_days).ffill()
    slow_max = pd.DataFrame(comp_max_rows).T.reindex(columns=close.columns) \
        .reindex(oos_days).ffill()
    weights = pd.DataFrame(weight_rows)
    log(f"[slow] 慢信号面板 {slow.shape}，{len(t0_days)} 个信号月；"
        f"末月 Top ICIR 权重: {weights.iloc[-1, 2:].dropna().astype(float).round(3).to_dict()}")
    return slow, slow_max, weights


def main():
    ap = argparse.ArgumentParser(description="E3 基本面慢信号⊕快信号双层融合")
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho")
    ap.add_argument("--out", default="reports/fundamental_blend")
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--execution", default="open", choices=["close", "open", "vwap"],
                    help="成交价口径（默认 open=T+1 可执行定版；close=乐观上限对照）")
    ap.add_argument("--lambdas", default="0.3,0.5,0.7",
                    help="快信号上叠加慢信号的权重扫描")
    ap.add_argument("--ic-months", type=int, default=24)
    ap.add_argument("--embargo-months", type=int, default=2)
    ap.add_argument("--min-cov", type=float, default=0.5)
    ap.add_argument("--min-months", type=int, default=12)
    ap.add_argument("--include-holder-dyn", action="store_true",
                    help="慢信号家族并入 holder_dyn 增减持/高管持股 9 因子"
                         "（P5 盘点：mgmt_netbuy 系 h20 IC 0.012~0.013）")
    args = ap.parse_args()
    pred_dir, dest = Path(args.pred_dir), Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    h1 = pd.read_parquet(pred_dir / "pred" / "gbdt__h1.parquet")
    h5 = pd.read_parquet(pred_dir / "pred" / "gbdt__h5.parquet")
    h10 = pd.read_parquet(pred_dir / "pred" / "gbdt__h10.parquet")
    h20 = pd.read_parquet(pred_dir / "pred" / "gbdt__h20.parquet")
    print(f"[load] 预测面板就绪 {time.time() - t0:.0f}s", flush=True)

    base = RG.load_base()
    close = base["close"]
    mask = base["mask"].astype(bool)
    costs = default_costs()
    oos_days = h1.index
    fwd = close.pct_change(fill_method=None).reindex(close.index).loc[oos_days]
    mask_oos = mask.reindex(index=oos_days, columns=close.columns).fillna(True)
    bench_idx = base["bench_index"].reindex(oos_days).fillna(0.0)
    bench_eqw = base["bench_eqw"].reindex(oos_days).fillna(0.0)

    hold_dyn = {
        "ctrl_change_cnt_60d", "ctrl_hold_ratio", "inner_netbuy_20d",
        "inner_netbuy_60d", "inner_netbuy_cnt_60d", "mgmt_netbuy_20d",
        "mgmt_netbuy_60d", "mgmt_netbuy_cnt_20d", "mgmt_sell_ratio_60d",
    }
    slow, slow_max, weights = build_slow_panel(
        base, oos_days, args.ic_months, args.embargo_months,
        args.min_cov, args.min_months,
        extra_family=(hold_dyn if args.include_holder_dyn else None))
    weights.to_csv(dest / "slow_weights.csv", index=False, encoding="utf-8-sig")

    fast = RG._rank_average([h1, h5]).reindex(index=oos_days,
                                              columns=close.columns)
    fast1020 = RG._rank_average([h1, h5, h10, h20]).reindex(index=oos_days,
                                                            columns=close.columns)
    fast_r, fast1020_r = fast.rank(axis=1, pct=True), fast1020.rank(axis=1, pct=True)
    slow_r, slow_max_r = slow.rank(axis=1, pct=True), slow_max.rank(axis=1, pct=True)

    variants = {"baseline_ens_h1h5": fast_r, "slow_only": slow_r,
                "slow_only_max": slow_max_r, "h1020_baseline": fast1020_r}
    for lam in [float(x) for x in args.lambdas.split(",")]:
        variants[f"blend_s{lam:.1f}"] = (1 - lam) * fast_r + lam * slow_r
        variants[f"blendmax_s{lam:.1f}"] = (1 - lam) * fast_r + lam * slow_max_r
    variants[f"h1020_s{float(args.lambdas.split(',')[1]):.1f}"] = (
        0.5 * fast1020_r + float(args.lambdas.split(",")[1]) * slow_r)

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
            # 信号预掩码：引擎 executable_mask 会重归一多头、抹掉执行价分段
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

    pd.DataFrame(rows_overall).to_csv(dest / f"metrics_fundamental_blend{sfx}.csv",
                                      index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_yearly).to_csv(dest / f"metrics_yearly{sfx}.csv",
                                     index=False, encoding="utf-8-sig")
    print(f"完成 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
