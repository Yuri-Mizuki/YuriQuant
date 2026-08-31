"""
诊断：模型预测 vs 单因子 —— 为什么模型看起来"烂"？
==================================================

用户质疑：输入的单因子可能都比模型预测强。本脚本用同一把尺子对比：

1. 特征选择：valid 段 |IC| 排序取 top-K 单因子（含经典 + 因子库）
2. 回测：每个候选（单因子 + 模型预测）做月频 top-50 等权 walk-forward 回测
   （单因子无训练，直接按因子值取 top；模型用 investment_report 的预测面板）
3. 指标：调仓日 IC（5日）/ 回测总收益 / Sharpe / 超额 vs 沪深300指数
4. 相关性：模型预测与 top 单因子的截面相关（模型是否在学这些因子）

用法：
    python scripts/diagnose_factor_vs_model.py --real
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cli_common import add_real_mock_args, setup_logging  # noqa: E402

from scripts.e2e_common import HORIZON, drop_stale_factors, load_daily_data  # noqa: E402
from factor.classic import compute_classic_features  # noqa: E402
from model.labels import build_labels  # noqa: E402
from scripts.e2e_backtest import (  # noqa: E402
    run_equal_weight_backtest, perf_stats,
)
from data.cache_helpers import load_index_returns  # noqa: E402


log = setup_logging("diagnose")

BT_START = "2024-01-01"
TOP_N = 50


def run(args) -> None:
    # ---- 1. 数据 + 特征池 ----
    px, lib_feats = load_daily_data(begin=20220101)
    classic = compute_classic_features(px)
    all_feats = {**classic, **lib_feats}
    all_feats = drop_stale_factors(all_feats, px["close"].index[-1])
    log.info("特征池: %d（经典 %d + 因子库 %d）", len(all_feats), len(classic), len(lib_feats))

    close = px["close"]
    returns = close.pct_change(fill_method=None)
    labels, fwd = build_labels(close, horizon=HORIZON)

    # ---- 2. valid 段 |IC| 质量分（与选择漏斗同口径）----
    from research.factor_analysis import calc_ic_series
    all_days = close.index
    sel_days = all_days[all_days < pd.Timestamp(BT_START)]
    fwd_sel = fwd.reindex(index=sel_days)
    q = {}
    for nm, p in all_feats.items():
        try:
            ic = calc_ic_series(p.reindex(index=sel_days), fwd_sel).dropna()
            if len(ic) >= 10:
                q[nm] = abs(float(ic.mean()))
        except Exception:
            pass
    quality = pd.Series(q).sort_values(ascending=False)
    log.info("valid 段 top10 因子: %s",
             {k: round(v, 4) for k, v in quality.head(10).items()})

    # ---- 3. 回测区间/调仓日 ----
    common = None
    for f in all_feats.values():
        d = f.dropna(how="all").index
        common = d if common is None else common.intersection(d)
    bt_days = common[common >= pd.Timestamp(BT_START)]
    s = pd.Series(bt_days, index=bt_days)
    reb_days = list(s.groupby(s.index.to_period("M")).first())
    log.info("回测: %s ~ %s, 调仓 %d 次", bt_days[0].date(), bt_days[-1].date(), len(reb_days))

    # 指数基准
    bench_ret = load_index_returns("000300.SH", int(bt_days[0].strftime("%Y%m%d")),
                                   int(bt_days[-1].strftime("%Y%m%d")), real=args.real)
    bench_ret = bench_ret.reindex(bt_days) if bench_ret is not None else None

    # ---- 4. 候选集：top 单因子 + 模型预测 ----
    cands = {}
    for nm in quality.head(args.k).index:
        cands[f"单因子:{nm[:24]}"] = all_feats[nm]
    model_path = Path(args.model_predictions)
    if model_path.exists():
        mp = pd.read_csv(model_path, index_col=0, parse_dates=True)
        cands["模型预测"] = mp
        log.info("载入模型预测: %s (%d 调仓日)", model_path.name, len(mp))
    else:
        log.warning("模型预测面板不存在: %s", model_path)

    # ---- 5. 逐候选回测 + IC ----
    rows = []
    corr_rows = []
    for name, panel in cands.items():
        # 调仓日因子值
        reb_panel = panel.reindex(index=reb_days)
        # 调仓日 5日 IC
        ic = calc_ic_series(reb_panel, fwd.reindex(index=reb_days)).dropna()
        ic_mean = float(ic.mean()) if len(ic) else np.nan
        # 月频 top-N 等权回测
        try:
            result = run_equal_weight_backtest(reb_panel, returns, bt_days, TOP_N)
            st = perf_stats(result.daily_returns, name)
            extra = {}
            if bench_ret is not None:
                df = pd.DataFrame({"s": result.daily_returns, "b": bench_ret}).dropna()
                ex = df["s"] - df["b"]
                extra["excess"] = float((1 + ex).prod() - 1)
            rows.append({
                "候选": name,
                "调仓IC(5日)": ic_mean,
                "总收益": st["total_return"],
                "Sharpe": st["sharpe"],
                "最大回撤": st["max_drawdown"],
                "超额vs指数": extra.get("excess", np.nan),
            })
        except Exception as e:
            log.warning("%s 回测失败: %s", name, e)
            rows.append({"候选": name, "调仓IC(5日)": ic_mean})
        # 模型与 top 因子截面相关
        if name == "模型预测":
            for nm2 in quality.head(args.k).index:
                f2 = all_feats[nm2].reindex(index=reb_days)
                c = reb_panel.corrwith(f2, axis=1, method="spearman").dropna()
                if len(c):
                    corr_rows.append({"与因子": nm2[:24], "截面相关": float(c.mean())})

    table = pd.DataFrame(rows).sort_values("调仓IC(5日)", ascending=False)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n===== 候选对比（月频 top-%d 等权, %s ~ %s）=====" %
          (TOP_N, bt_days[0].date(), bt_days[-1].date()))
    print(table.to_string())
    if bench_ret is not None:
        print(f"\n沪深300指数同期总收益: {(1+bench_ret).prod()-1:.4f}")
    if corr_rows:
        print("\n===== 模型预测与 top 因子的截面相关 =====")
        print(pd.DataFrame(corr_rows).sort_values("截面相关", ascending=False).to_string())

    table.to_csv(Path(args.out) / "diagnose_compare.csv", index=False, encoding="utf-8-sig")
    if corr_rows:
        pd.DataFrame(corr_rows).to_csv(Path(args.out) / "diagnose_corr.csv",
                                       index=False, encoding="utf-8-sig")
    log.info("诊断完成 → %s", args.out)


def main() -> None:
    ap = argparse.ArgumentParser(description="模型 vs 单因子诊断")
    add_real_mock_args(ap)
    ap.add_argument("--k", type=int, default=8, help="参与对比的单因子个数")
    ap.add_argument("--model-predictions",
                    default="reports/investment_report/walk_forward_predictions.csv")
    ap.add_argument("--out", default="reports/diagnose")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
