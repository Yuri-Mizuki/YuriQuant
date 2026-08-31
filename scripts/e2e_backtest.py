"""
端到端选股策略 walk-forward 回测
==================================

回答：按月（或 W/D）调仓，用「因子筛选 → 模型 walk-forward 预测 → top-N 组合」
投资到当前收益如何。与 e2e_stock_picks 共享数据/因子/选择逻辑（e2e_common）。

协议（防前视纪律）：
1. 特征：经典量价 12 + 因子库 significant（排除 model:*）
2. 特征选择：只用回测前窗口（默认 2022~2023）——干净 OOS
3. Walk-forward：每个调仓日 t 重训 GBDT（expanding window + embargo=5）→ 预测截面
4. 组合：等权 top-N（纯信号检验）+ risk_parity top-N（SCS 解做 _enforce_caps 后处理）
5. 成本：佣金 0.01% + 印花税 0.1%(卖出) + 滑点 5bp（VectorBacktest 默认，已扣除）
6. 基准：股票池等权日收益（日度再平衡，与策略月频调仓有口径差异）

用法：
    python scripts/e2e_backtest.py --real --top 50
    python scripts/e2e_backtest.py --real --top 50 --freq W
    python scripts/e2e_backtest.py --mock --top 20 --model ridge   # 测试/演示
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cli_common import add_real_mock_args, setup_logging  # noqa: E402

from backtest.metrics import PERIODS_PER_YEAR  # noqa: E402
from data.mock import load_mock_data  # noqa: E402
from factor.classic import compute_classic_features  # noqa: E402
from model.labels import build_label_pair  # noqa: E402
from scripts.e2e_common import (  # noqa: E402
    GBDT_PARAMS,
    HORIZON,
    drop_stale_factors,
    load_daily_data,
    select_features,
)


log = setup_logging("e2e_backtest")

OUT_DIR = Path("reports/e2e_backtest")
BT_START = "2024-01-01"


# ---------------------------------------------------------------------------
# Walk-forward 预测
# ---------------------------------------------------------------------------
def walk_forward_predictions(feats, labels, reb_days, common_days, model="gbdt",
                             window: int | None = None):
    """每个调仓日重训模型并预测当期截面。

    window=None -> expanding（全历史）；window=N -> 滚动最近 N 日训练。
    滚动窗口解决训练/预测市场状态错配（熊市训练->牛市预测时模型学到过时的
    低波偏好；实测 2024-01~2026-08 月频 top-50 五因子中性化后滚动2年
    beta-neutral alpha 从 -8.6% 转 +5.5%，见 scripts/experiment_style_mismatch.py）。
    """
    from model.predictor import LGBMPredictor, RidgePredictor

    rows = {}
    for t in reb_days:
        tr = common_days[common_days <= t][:-HORIZON]
        if len(tr) < 120:
            log.warning("跳过 %s：训练段不足", t.date())
            continue
        if window is not None:
            tr = tr[-window:]
            if len(tr) < 120:
                log.warning("跳过 %s：滚动窗口内训练段不足", t.date())
                continue
        if model == "gbdt":
            p = LGBMPredictor(**GBDT_PARAMS)
        else:
            p = RidgePredictor(alpha=1.0)
        p.fit({k: v.loc[tr] for k, v in feats.items()}, labels.loc[tr])
        pred = p.predict({k: v.loc[[t]] for k, v in feats.items()})
        rows[t] = pred.iloc[0]
        log.info("  %s: 训练 %d 日 -> 预测 %d 股", t.date(), len(tr), pred.iloc[0].notna().sum())
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# 组合
# ---------------------------------------------------------------------------
def _enforce_caps(w: pd.Series, cap: float) -> pd.Series:
    """约束后处理：SCS 对数障碍解可能违反 max_weight/budget（已知数值坑）。

    用带上下限的 L2 投影（water-filling）把权重投影到
    {0 <= w <= cap, sum(w)=1}：二分 λ 求 sum(min(cap, max(0, w0-lambda))) = 1。
    数学上保证 sum=1 且 w<=cap（不可行时返回 sum<1 的封顶解）。
    """
    w0 = w.clip(lower=0.0).values.astype(float)
    names = w.index
    if w0.sum() <= 0:
        return pd.Series(0.0, index=names)
    # s0 = 在 λ=0 时的和；<1 需 λ<0（抬升小权重），>1 需 λ>0（压低大权重）
    s0 = float(np.minimum(cap, w0).sum())
    lo, hi = (-cap, 0.0) if s0 < 1.0 else (0.0, float(w0.max()))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = float(np.minimum(cap, np.maximum(0.0, w0 - mid)).sum())
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    out = np.minimum(cap, np.maximum(0.0, w0 - lam))
    s = float(out.sum())
    # 数值收尾：sum 略小于 1 且无权重触及 cap 时，按比例补足
    if abs(s - 1.0) > 1e-9 and s > 1e-12 and out.max() < cap - 1e-12:
        out *= 1.0 / s
    return pd.Series(out, index=names)


def _run_backtest(target_df, pred_reb, returns, full_dates, label):
    """预计算目标权重 -> VectorBacktest 记账（含成本）。"""
    from backtest.engine import VectorBacktest
    from optimize.multi_period import PrecomputedWeightsStrategy

    target_df = target_df.reindex(columns=returns.columns)
    pred_panel = pred_reb.reindex(full_dates)
    strat = PrecomputedWeightsStrategy(target_df)
    bt = VectorBacktest(strat, rebalance_freq="M")
    result = bt.run(pred_panel, returns.loc[full_dates[0]:])
    result.target_label = label  # type: ignore[attr-defined]
    return result


def run_equal_weight_backtest(pred_reb, returns, full_dates, top_n):
    target = {}
    for t, row in pred_reb.iterrows():
        scores = row.dropna().sort_values(ascending=False)
        top = scores.head(top_n).index
        target[t] = pd.Series(1.0 / top_n, index=top)
    return _run_backtest(pd.DataFrame(target).T, pred_reb, returns, full_dates,
                         f"等权top{top_n}")


def run_risk_parity_backtest(pred_reb, returns, full_dates, top_n, max_weight):
    """risk_parity top-N：逐调仓日只在 top-N 列上解 QP（全池 dropna(how="any")
    会因停牌 NaN 清空协方差窗口 -> 全空仓，已知数值坑）。SCS 解做 cap 后处理。"""
    from optimize.solver import optimize_weights_qp

    target_rows = {}
    for t, row in pred_reb.iterrows():
        scores = row.dropna().sort_values(ascending=False)
        top = list(scores.head(top_n).index)
        f_sub = scores.head(top_n).to_frame().T
        f_sub.index = [t]
        r_sub = returns[top].loc[:t]
        try:
            w = optimize_weights_qp(
                f_sub, r_sub, method="risk_parity",
                max_weight=max_weight, window=120, min_periods=30)
            ww = w.iloc[-1] if len(w) else pd.Series(dtype=float)
            if len(ww) == 0 or ww.sum() <= 0:
                raise RuntimeError("空解")
            ww = _enforce_caps(ww, max_weight)
        except Exception as e:
            log.warning("  risk_parity %s 失败，降级等权: %s", t.date(), e)
            ww = pd.Series(1.0 / len(top), index=top)
        target_rows[t] = ww
    return _run_backtest(pd.DataFrame(target_rows).T, pred_reb, returns, full_dates,
                         f"风险平价top{top_n}")


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def perf_stats(daily_ret: pd.Series, label: str) -> dict:
    ret = daily_ret.dropna()
    if len(ret) == 0:
        return {"label": label}
    n_years = len(ret) / PERIODS_PER_YEAR
    total = (1 + ret).prod() - 1
    annual = (1 + total) ** (1 / max(n_years, 1e-9)) - 1 if n_years > 0.3 else total
    vol = ret.std() * (PERIODS_PER_YEAR ** 0.5)
    sharpe = annual / vol if vol > 0 else float("nan")
    eq = (1 + ret).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    monthly = ret.groupby(ret.index.to_period("M")).apply(lambda x: (1 + x).prod() - 1)
    return {
        "label": label, "total_return": float(total), "annual_return": float(annual),
        "annual_vol": float(vol), "sharpe": float(sharpe), "max_drawdown": float(dd),
        "win_rate_monthly": float((monthly > 0).mean()), "n_months": int(len(monthly)),
        "monthly": monthly, "equity": eq, "daily": ret,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log.info("=" * 60)
    log.info(" 端到端选股策略 walk-forward 回测")
    log.info(" 调仓频率: %s | top-%d | 模型: %s", args.freq, args.top, args.model)
    log.info("=" * 60)

    # ---- 1. 数据 + 因子 ----
    if args.real:
        px, lib_feats = load_daily_data(begin=20220101)
    else:
        px = load_mock_data(n_days=args.n_days, n_codes=args.n_codes, seed=args.seed)
        lib_feats = {}
    classic = compute_classic_features(px)
    all_feats = {**classic, **lib_feats}
    log.info("因子总池: %d（经典 %d + 因子库 %d）",
             len(all_feats), len(classic), len(lib_feats))

    close = px["close"]
    # 面板新鲜度守卫：剔除滞后面板（否则回测区间末端被拖短）
    all_feats = drop_stale_factors(all_feats, close.index[-1])

    returns = close.pct_change(fill_method=None)
    labels, fwd = build_label_pair(close, horizon=HORIZON)

    # ---- 2. 特征选择（只用回测前窗口，防前视）----
    all_days = close.index
    bt_start = pd.Timestamp(args.bt_start)
    sel_days = all_days[all_days < bt_start]
    if len(sel_days) < 60:
        # 短数据（如 mock）：前 50% 做选择，后 50% 回测
        split = len(all_days) // 2
        sel_days = all_days[:split]
        bt_start_actual = all_days[split]
        log.info("数据不足 %s，自动切分: 选择 %s~%s, 回测 %s 起",
                 args.bt_start, sel_days[0].date(), sel_days[-1].date(),
                 bt_start_actual.date())
    else:
        bt_start_actual = bt_start
    feats, quality = select_features(all_feats, fwd, sel_days, max_features=args.max_features)
    log.info("特征选择（%s ~ %s）: %d -> %d", sel_days[0].date(), sel_days[-1].date(),
             len(all_feats), len(feats))
    pd.DataFrame({"valid_abs_ic": [quality.get(k, float("nan")) for k in feats]}) \
        .to_csv(out_dir / "selected_features.csv", encoding="utf-8-sig")

    # ---- 3. 公共日期网格 + 调仓日 ----
    common = None
    for f in feats.values():
        d = f.dropna(how="all").index
        common = d if common is None else common.intersection(d)
    common = common.intersection(labels.dropna(how="all").index)
    log.info("公共日期: %d 日 (%s ~ %s)", len(common), common[0].date(), common[-1].date())

    bt_days = common[common >= bt_start_actual]
    s = pd.Series(bt_days, index=bt_days)
    reb_days = list(s.groupby(s.index.to_period(args.freq)).first())
    log.info("回测区间: %s ~ %s (%d 日), 调仓 %d 次",
             bt_days[0].date(), bt_days[-1].date(), len(bt_days), len(reb_days))

    # ---- 4. Walk-forward 预测 ----
    log.info("Walk-forward 训练预测中（每次重训 %s, window=%s）...", args.model, args.train_window)
    pred_reb = walk_forward_predictions(feats, labels, reb_days, common, model=args.model,
                                        window=args.train_window)
    pred_reb.to_csv(out_dir / "walk_forward_predictions.csv", encoding="utf-8-sig")
    log.info("预测完成: %d 个调仓日 x %d 股", len(pred_reb), pred_reb.shape[1])

    bt_returns = returns.loc[bt_days[0]:bt_days[-1]]

    # ---- 5a. 等权 top-N ----
    log.info("回测 A：等权 top-%d ...", args.top)
    result_eq = run_equal_weight_backtest(pred_reb, returns, bt_days, args.top)
    stats_eq = perf_stats(result_eq.daily_returns, f"等权top{args.top}")

    # ---- 5b. risk_parity top-N ----
    stats_rp = None
    if not args.skip_rp:
        try:
            import cvxpy  # noqa: F401
            log.info("回测 B：risk_parity top-%d ...", args.top)
            result_rp = run_risk_parity_backtest(pred_reb, returns, bt_days, args.top,
                                                 args.max_weight)
            stats_rp = perf_stats(result_rp.daily_returns, f"风险平价top{args.top}")
        except ImportError:
            log.warning("cvxpy 不可用，跳过 risk_parity（mock 环境）")
        except Exception as e:
            log.warning("risk_parity 回测失败: %s", e)

    # ---- 6. 基准 ----
    stats_bm = perf_stats(bt_returns.mean(axis=1), "全池等权基准")

    # ---- 7. 汇总 ----
    rows = []
    for st in [stats_eq, stats_rp, stats_bm]:
        if st is None:
            continue
        rows.append({k: v for k, v in st.items() if k not in ("monthly", "equity", "daily")})
    summary = pd.DataFrame(rows).set_index("label").T
    print("\n===== 收益对照表（%s ~ %s, %s 调仓）=====" %
          (bt_days[0].date(), bt_days[-1].date(), args.freq))
    print(summary.to_string())
    summary.to_csv(out_dir / "backtest_summary.csv", encoding="utf-8-sig")

    monthly = pd.DataFrame({st["label"]: st.get("monthly", pd.Series(dtype=float))
                            for st in [stats_eq, stats_rp, stats_bm] if st is not None})
    monthly.to_csv(out_dir / "monthly_returns.csv", encoding="utf-8-sig")
    eq_df = pd.DataFrame({st["label"]: st.get("equity", pd.Series(dtype=float))
                          for st in [stats_eq, stats_rp, stats_bm] if st is not None})
    eq_df.to_csv(out_dir / "equity_curve.csv", encoding="utf-8-sig")

    extra = {}
    for st, res in (("eq", result_eq), ("rp", locals().get("result_rp"))):
        if res is None:
            continue
        try:
            extra[f"avg_turnover_{st}"] = float(res.turnover_series.mean())
            extra[f"total_cost_{st}"] = float(res.cost_series.sum())
        except Exception:
            pass

    meta = {
        "freq": args.freq, "top_n": args.top,
        "bt_start": str(bt_days[0].date()), "bt_end": str(bt_days[-1].date()),
        "n_rebalance": len(pred_reb), "n_features": len(feats),
        "model": args.model, "horizon": HORIZON,
        "summary": {r["label"]: {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in r.items()} for r in rows},
        "extra": {k: round(v, 6) for k, v in extra.items()},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(out_dir / "backtest_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info("=" * 60)
    log.info(" 回测完成（%.1f 分钟）, 输出: %s", (time.time() - t0) / 60, out_dir)
    log.info("=" * 60)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="端到端选股 walk-forward 回测")
    add_real_mock_args(ap)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--freq", default="M", choices=["D", "W", "M"])
    ap.add_argument("--model", default="gbdt", choices=["gbdt", "ridge"])
    ap.add_argument("--train-window", type=int, default=None,
                    help="滚动训练窗口（交易日数）；None=expanding 全历史")
    ap.add_argument("--max-weight", type=float, default=0.05)
    ap.add_argument("--max-features", type=int, default=30)
    ap.add_argument("--skip-rp", action="store_true", help="跳过 risk_parity（无 cvxpy 环境）")
    ap.add_argument("--bt-start", default="2024-01-01", help="回测起点（默认 2024-01-01）")
    ap.add_argument("--n-days", type=int, default=500)
    ap.add_argument("--n-codes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
