"""风格错配解法对照实验：训练窗口 × 标签中性化。

背景：模型 2022-2023（熊市）expanding 训练，2024-2026（牛市）预测，学到低波偏好
导致 β 中性 alpha 为负（-4.9%），而 range20 高波风格 alpha 正（+10.2%）。

解法候选：
1. 基线：expanding window + rank 标签（现状）
2. 滚动窗口：只取最近 N 日训练（跟上市场状态）
3. 标签中性化：训练目标 = 未来收益对五因子回归残差（学纯 alpha 而非风格）
4. 滚动窗口 + 标签中性化 组合

评估：β 中性（滚动对冲）后的 alpha + 多头总收益 + Sharpe。
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("style_mismatch")

from scripts.e2e_common import (  # noqa: E402
    HORIZON, GBDT_PARAMS, build_labels, build_neutral_covariates,
    compute_classic_features, drop_stale_factors, load_daily_data,
    neutralize_predictions, select_features,
)
from scripts.e2e_backtest import run_equal_weight_backtest, perf_stats  # noqa: E402
from scripts.investment_report import load_index_returns  # noqa: E402

BT_START = "2024-01-01"
TOP_N = 50


def walk_forward_variant(feats, labels, reb_days, common_days, model="gbdt",
                         window=None, neutral_label=None):
    """walk-forward 变体：支持滚动窗口 + 标签中性化。

    window=None -> expanding；window=N -> 最近 N 日。
    neutral_label: dict 面板（逐日五因子残差标签），非 None 时用残差标签训练。
    """
    from model.predictor import LGBMPredictor, RidgePredictor

    lab = neutral_label if neutral_label is not None else labels
    rows = {}
    for t in reb_days:
        tr = common_days[common_days <= t][:-HORIZON]
        if len(tr) < 120:
            continue
        if window is not None:
            tr = tr[-window:]
            if len(tr) < 120:
                continue
        if model == "gbdt":
            p = LGBMPredictor(**GBDT_PARAMS)
        else:
            p = RidgePredictor(alpha=1.0)
        p.fit({k: v.loc[tr] for k, v in feats.items()}, lab.loc[tr])
        pred = p.predict({k: v.loc[[t]] for k, v in feats.items()})
        rows[t] = pred.iloc[0]
    return pd.DataFrame(rows).T


def market_neutral_alpha(daily_ret, bench, window=120):
    df = pd.DataFrame({"p": daily_ret, "b": bench}).dropna()
    rb = df["b"].rolling(window, min_periods=30).cov(df["p"]) / df["b"].rolling(window, min_periods=30).var()
    mn = df["p"] - rb * df["b"]
    mn = mn.dropna()
    if len(mn) < 30:
        return float("nan"), float("nan")
    ret = (1 + mn).prod() - 1
    sharpe = mn.mean() / mn.std() * np.sqrt(244) if mn.std() > 0 else 0.0
    return ret, sharpe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--out", default="reports/style_mismatch")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    px, lib_feats = load_daily_data(begin=20220101)
    classic = compute_classic_features(px)
    all_feats = {**classic, **lib_feats}
    all_feats = drop_stale_factors(all_feats, px["close"].index[-1])
    close = px["close"]
    returns = close.pct_change(fill_method=None)
    labels, fwd = build_labels(close, horizon=HORIZON)
    mc, ind, extra = build_neutral_covariates(px, close, True)

    common = None
    for f in all_feats.values():
        d = f.dropna(how="all").index
        common = d if common is None else common.intersection(d)
    common = common.intersection(labels.dropna(how="all").index)
    bt_days = common[common >= pd.Timestamp(BT_START)]
    s = pd.Series(bt_days, index=bt_days)
    reb_days = list(s.groupby(s.index.to_period("M")).first())
    log.info("回测 %s ~ %s, %d 次调仓", bt_days[0].date(), bt_days[-1].date(), len(reb_days))

    # 特征选择（回测前窗口，固定）
    sel_days = close.index[close.index < pd.Timestamp(BT_START)]
    feats, _ = select_features(all_feats, fwd, sel_days, max_features=30)
    log.info("特征选择: %d -> %d", len(all_feats), len(feats))

    # 标签中性化：fwd 对五因子逐日回归取残差 = 纯 alpha 训练目标
    label_neu = neutralize_predictions(fwd, mc, ind, extra)
    # rank 变换残差标签（与 rank 标签同口径）
    from model.labels import build_labels as _build_labels
    rank_neu, _ = _build_labels(close_panel=label_neu, horizon=HORIZON, mode="rank")
    rank_neu = rank_neu.reindex(common)

    bench = load_index_returns("000300.SH", 20240101, 20260821, real=True).reindex(bt_days)
    mkt = (1 + bench).prod() - 1

    variants = {
        "基线_expanding_rank": dict(window=None, neutral_label=None),
        "滚动1年_rank": dict(window=250, neutral_label=None),
        "滚动2年_rank": dict(window=500, neutral_label=None),
        "expanding_标签中性化": dict(window=None, neutral_label=rank_neu),
        "滚动1年_标签中性化": dict(window=250, neutral_label=rank_neu),
    }

    rows = []
    for name, kw in variants.items():
        pred = walk_forward_variant(feats, labels, reb_days, common, model="gbdt", **kw)
        # 预测分数再中性化（与报告一致）
        pred_neu = neutralize_predictions(pred, mc, ind, extra)
        r = run_equal_weight_backtest(pred_neu, returns, bt_days, args.top)
        st = perf_stats(r.daily_returns, name)
        alpha_mn, sh_mn = market_neutral_alpha(r.daily_returns, bench)
        excess = (1 + (r.daily_returns.dropna() - bench.reindex(r.daily_returns.index)).dropna()).prod() - 1 \
            if len(bench) else float("nan")
        rows.append({
            "方案": name,
            "多头收益": round(st["total_return"], 4),
            "Sharpe": round(st["sharpe"], 3),
            "β中性alpha": round(alpha_mn, 4) if not np.isnan(alpha_mn) else None,
            "β中性Sharpe": round(sh_mn, 3) if not np.isnan(sh_mn) else None,
            "超额vs指数": round(excess, 4),
        })
        log.info("  %s 完成", name)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "style_mismatch_compare.csv", index=False, encoding="utf-8-sig")
    print("\n=== 风格错配解法对照（月频 top-%d 等权，费后，指数 %+.1f%%）===" % (args.top, mkt * 100))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
