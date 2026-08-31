"""
端到端优化实验：调仓频率 × 风格中性化
======================================

回答：周频调仓配 h=5 是否减少信号衰减？预测分数做五因子中性化
（市值/行业/动量/波动/换手残差）是否提升纯 alpha？

设计（同口径对比，全部 top-N 等权，费后）：
    A. 月频调仓（M）       —— 基线（现有）
    B. 月频 + 中性化        —— 风格剥离后纯 alpha
    C. 周频调仓（W）       —— 信号衰减减少（h=5 配 5 交易日）
    D. 周频 + 中性化        —— 组合拳

中性化：对 walk-forward 预测面板逐日做截面回归
    pred ~ size + industry + mom20 + vol20 + turn20
取残差作为组合信号（factor.preprocessing.neutralize，华泰五因子口径）。

用法：
    python scripts/optimize_e2e.py --real --top 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cli_common import add_real_mock_args, setup_logging  # noqa: E402

from data.cache_helpers import load_index_returns  # noqa: E402
from factor.classic import compute_classic_features  # noqa: E402
from model.labels import build_label_pair  # noqa: E402
from scripts.e2e_backtest import (  # noqa: E402
    perf_stats,
    run_equal_weight_backtest,
    walk_forward_predictions,
)
from scripts.e2e_common import (  # noqa: E402
    HORIZON,
    build_neutral_covariates,
    drop_stale_factors,
    load_daily_data,
    neutralize_predictions,
    select_features,
)


log = setup_logging("optimize_e2e")

BT_START = "2024-01-01"


def build_neutral_covariates_local(px: dict, close: pd.DataFrame, real: bool):
    """兼容入口：直接复用 e2e_common.build_neutral_covariates。"""
    return build_neutral_covariates(px, close, real)


def neutralize_predictions_local(pred: pd.DataFrame, mc, ind, extra) -> pd.DataFrame:
    """兼容入口：直接复用 e2e_common.neutralize_predictions。"""
    return neutralize_predictions(pred, mc, ind, extra)


def run(args) -> None:
    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(" 端到端优化实验：调仓频率 × 风格中性化")
    log.info("=" * 60)

    # ---- 数据/因子 ----
    px, lib_feats = load_daily_data(begin=20220101)
    classic = compute_classic_features(px)
    all_feats = {**classic, **lib_feats}
    all_feats = drop_stale_factors(all_feats, px["close"].index[-1])
    close = px["close"]
    returns = close.pct_change(fill_method=None)
    labels, fwd = build_label_pair(close, horizon=HORIZON)

    # 特征选择（回测前窗口）
    all_days = close.index
    sel_days = all_days[all_days < pd.Timestamp(BT_START)]
    feats, _ = select_features(all_feats, fwd, sel_days, max_features=args.max_features)
    log.info("特征选择: %d -> %d", len(all_feats), len(feats))

    # 公共网格
    common = None
    for f in feats.values():
        d = f.dropna(how="all").index
        common = d if common is None else common.intersection(d)
    common = common.intersection(labels.dropna(how="all").index)
    bt_days = common[common >= pd.Timestamp(BT_START)]

    # ---- 中性化协变量 ----
    mc_panel, ind_panel, extra = build_neutral_covariates(px, close, args.real)

    # ---- walk-forward 预测（W 重训 / M 复用）----
    predictions: dict[str, pd.DataFrame] = {}
    for freq in ("M", "W"):
        if freq == "M" and Path(args.m_predictions).exists() and not args.force_retrain:
            pred = pd.read_csv(args.m_predictions, index_col=0, parse_dates=True)
            log.info("复用月频预测: %s (%d 调仓日)", args.m_predictions, len(pred))
        else:
            s = pd.Series(bt_days, index=bt_days)
            reb_days = list(s.groupby(s.index.to_period(freq)).first())
            log.info("walk-forward 训练（%s 频, %d 次调仓）...", freq, len(reb_days))
            pred = walk_forward_predictions(feats, labels, reb_days, common, model=args.model)
            if freq == "M":
                pred.to_csv(out_dir / f"walk_forward_{freq}.csv", encoding="utf-8-sig")
        predictions[freq] = pred

    # ---- 指数基准 ----
    bench = load_index_returns("000300.SH", int(bt_days[0].strftime("%Y%m%d")),
                               int(bt_days[-1].strftime("%Y%m%d")), real=args.real)
    bench = bench.reindex(bt_days) if bench is not None else None
    bench_total = float((1 + bench).prod() - 1) if bench is not None else float("nan")

    # ---- 四组回测 ----
    rows = []
    for freq, pred in predictions.items():
        for tag, p in (("原分数", pred),
                       ("中性化", neutralize_predictions(pred, mc_panel, ind_panel, extra))):
            # 对齐到完整日历（walk_forward 输出只有调仓日）
            reb = p.index
            r = run_equal_weight_backtest(p, returns, bt_days, args.top)
            st = perf_stats(r.daily_returns, f"{freq}+{tag}")
            ex = float("nan")
            if bench is not None:
                df = pd.DataFrame({"s": r.daily_returns, "b": bench}).dropna()
                ex = float((1 + (df["s"] - df["b"])).prod() - 1)
            rows.append({
                "组": f"{freq} 调仓 + {tag}", "调仓次数": len(reb),
                "总收益": st["total_return"], "Sharpe": st["sharpe"],
                "最大回撤": st["max_drawdown"], "超额vs指数": ex,
            })
            log.info("%s: 收益 %.4f | Sharpe %.2f | 超额 %+.4f",
                     rows[-1]["组"], st["total_return"], st["sharpe"], ex)

    table = pd.DataFrame(rows).set_index("组")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n===== 优化实验对比（top-%d 等权, %s ~ %s, 费后）=====" %
          (args.top, bt_days[0].date(), bt_days[-1].date()))
    if bench is not None:
        print(f"沪深300指数同期: {bench_total:.4f}")
    print(table.to_string())
    table.to_csv(out_dir / "optimize_compare.csv", encoding="utf-8-sig")
    log.info("完成（%.1f 分钟）→ %s", (time.time() - t0) / 60, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="端到端优化实验")
    add_real_mock_args(ap)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--model", default="gbdt", choices=["gbdt", "ridge"])
    ap.add_argument("--max-features", type=int, default=30)
    ap.add_argument("--m-predictions",
                    default="reports/investment_report/walk_forward_predictions.csv",
                    help="复用既有月频预测面板（避免重复重训）")
    ap.add_argument("--force-retrain", action="store_true", help="强制重训月频模型")
    ap.add_argument("--out", default="reports/optimize_e2e")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
