"""E1'' horizon 混合消融：把慢 horizon OOS 预测混入 ens_h1h5 的对照实验。

背景（RESEARCH_TODO §一 基本面诊断，2026-09-22）：分族 IC-horizon 弹性证明
基本面/股东族 |IC| 随 horizon 上升（×1.14）而量价族全负；月频调仓组合下
把 gbdt__h10/h20 混入集成，年化 +1.2~1.4pp、换手 −4~15pp（882 口径实测，
产物 reports/horizon_mix/）。本脚本即该实验的固化版：920 口径基线重跑后
对新 pred 目录再跑一次即可复核定版。

口径与 rolling_grid_alla.stage_ensemble 同构（raw 信号 / M 月频 / equal 等权 /
frac 默认 0.10 / 结算 horizon=1），成交价口径默认 open=T+1 可执行定版
（09-22 换主；close=乐观上限对照，产物加 _close 后缀）。
变体：baseline_ens_h1h5、ens_h1h5h20、ens_h1h5h10h20、blend_h20_l25、
blend_h20_l40、gbdt_h20_only。

用法：
    python -m scripts.evaluation.horizon_mix \
        --pred-dir reports/alla_rolling_ortho --out reports/horizon_mix
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


def load(pred_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(pred_dir / "pred" / f"{name}.parquet")


def weighted_rank_average(panels: list, weights: list) -> pd.DataFrame:
    """与 RG._rank_average 同构的加权版（逐日截面百分位秩的加权和）。"""
    out = pd.DataFrame(np.nan, index=panels[0].index, columns=panels[0].columns)
    wv = np.asarray(weights, dtype=float)
    for d in panels[0].index:
        rows = []
        for p in panels:
            if d not in p.index:
                continue
            row = p.loc[d]
            r = row.rank(pct=True)
            rows.append(r[~row.isna()])
        if len(rows) < 2:
            continue
        valid = rows[0].index
        for r in rows[1:]:
            valid = valid.intersection(r.index)
        if len(valid) < 1:
            continue
        sub = [r.reindex(valid) for r in rows]
        ok = pd.concat(sub, axis=1).dropna(axis=0)
        w = wv[: len(rows)]
        if len(ok) < 30:
            out.loc[d, valid] = (pd.concat(sub, axis=1) * w).sum(axis=1) / w.sum()
            continue
        out.loc[d, ok.index] = ((ok * w).sum(axis=1) / w.sum()).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description="horizon 混合消融（E1''）")
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho",
                    help="滚动实验产物目录（含 pred/gbdt__h*.parquet）")
    ap.add_argument("--out", default="reports/horizon_mix",
                    help="本实验产物目录")
    ap.add_argument("--frac", type=float, default=0.10, help="Top 分位")
    ap.add_argument("--execution", default="open", choices=["close", "open", "vwap"],
                    help="成交价口径（默认 open=T+1 可执行定版；close=乐观上限对照）")
    args = ap.parse_args()
    pred_dir, dest = Path(args.pred_dir), Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    h1, h5 = load(pred_dir, "gbdt__h1"), load(pred_dir, "gbdt__h5")
    h10, h20 = load(pred_dir, "gbdt__h10"), load(pred_dir, "gbdt__h20")
    variants = {
        "baseline_ens_h1h5": RG._rank_average([h1, h5]),
        "ens_h1h5h20": RG._rank_average([h1, h5, h20]),
        "ens_h1h5h10h20": RG._rank_average([h1, h5, h10, h20]),
        "blend_h20_l25": weighted_rank_average([h1, h5, h20], [3, 3, 2]),
        "blend_h20_l40": weighted_rank_average([h1, h5, h20], [3, 3, 4]),
        "gbdt_h20_only": h20,
    }
    print(f"[load] 预测面板就绪 {time.time() - t0:.0f}s", flush=True)

    base = RG.load_base()
    close = base["close"]
    mask = base["mask"].astype(bool)
    costs = default_costs()
    fwd = close.pct_change(fill_method=None).reindex(close.index)
    oos_days = variants["baseline_ens_h1h5"].index
    fwd = fwd.loc[oos_days]
    mask_oos = mask.reindex(index=oos_days, columns=close.columns).fillna(True)
    bench_idx = base["bench_index"].reindex(oos_days).fillna(0.0)
    bench_eqw = base["bench_eqw"].reindex(oos_days).fillna(0.0)

    exec_split = None
    rb_exec = None
    if args.execution != "close":
        fp = pred_dir / "_base" / f"{args.execution}_adj.parquet"
        if not fp.exists():
            raise SystemExit(f"{fp} 缺失：先跑 rolling_grid_alla --stage prep 重建基础面板")
        fill = pd.read_parquet(fp)
        s_ = pd.Series(oos_days, index=oos_days)
        firsts = s_.groupby(s_.index.to_period("M")).first()
        pos_ = {d: i for i, d in enumerate(oos_days)}
        rb_exec = {oos_days[pos_[t] + 1] for t in firsts
                   if pos_[t] + 1 < len(oos_days)}
        exec_split = build_execution_split(fill, close, rb_exec)
        print(f"[exec] 执行价口径: {args.execution}（fill={fp.name}，"
              f"调仓日=T 次一交易日，{len(rb_exec)} 个）", flush=True)

    rows_overall, rows_yearly = [], []
    for name, sig in variants.items():
        sig = sig.reindex(index=oos_days, columns=close.columns)
        oos_ic = float(calc_ic_series(sig, fwd.shift(-1)).mean())
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
        row = {
            "run_id": name, "oos_ic": oos_ic,
            "annual": m["annual"], "excess_idx": m["excess"],
            "excess_eqw": m["annual"] - RG._ann(bench_eqw),
            "sharpe": m["sharpe"], "ir": m["ir"], "max_dd": m["max_dd"],
            "turnover": m["turnover"], "beta": m["beta"],
            "n_days": len(dr.dropna()),
        }
        rows_overall.append(row)
        rows_yearly.extend(RG._yearly_rows(row, dr, bench_idx, res.turnover_series))
        print(f"[{name}] IC={oos_ic:.4f} 年化={row['annual'] * 100:.2f}% "
              f"超额(指)={row['excess_idx'] * 100:+.2f}pp "
              f"超额(等权)={row['excess_eqw'] * 100:+.2f}pp "
              f"Sharpe={row['sharpe']:.2f} 回撤={row['max_dd'] * 100:.1f}% "
              f"换手={row['turnover'] * 100:.1f}%", flush=True)

    sfx = {"close": "_close", "open": ""}.get(args.execution, f"_{args.execution}")
    pd.DataFrame(rows_overall).to_csv(dest / f"metrics_horizon_mix{sfx}.csv",
                                      index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_yearly).to_csv(dest / f"metrics_yearly{sfx}.csv",
                                     index=False, encoding="utf-8-sig")
    print(f"完成 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
