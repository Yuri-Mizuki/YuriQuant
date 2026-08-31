"""
端到端选股流水线 —— 因子筛选 → 模型预测 → 组合优化 → 信号报告
============================================================

把数据/因子/选择逻辑与 e2e_backtest 共享（scripts/e2e_common.py），
一条命令跑通选股信号：

  1. 数据：日线缓存对齐到因子库股票池（~420 股）
  2. 因子：经典量价 + 因子库 significant（排除 model:*，保证预测日到数据末端）
  3. 选择：build_feature_set 三级漏斗（覆盖率→去冗余→质量排序）
  4. 模型：全历史训练 GBDT / ridge，预测最新截面
  5. 组合：等权 / risk_parity / MVO → 目标权重
  6. 信号：可执行清单 + 报告

用法:
    python scripts/e2e_stock_picks.py --mock --top 30
    python scripts/e2e_stock_picks.py --real --top 30
    python scripts/e2e_stock_picks.py --real --top 50 --portfolio risk_parity
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mock import load_mock_data  # noqa: E402
from factor.classic import compute_classic_features  # noqa: E402
from model.labels import build_label_pair  # noqa: E402
from scripts.e2e_common import (  # noqa: E402
    HORIZON,
    RIDGE_ALPHA,
    drop_stale_factors,
    load_daily_data,
    select_features,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("e2e_picks")

OUT_DIR = Path("reports/e2e_picks")


# ---------------------------------------------------------------------------
# Step 3: 模型
# ---------------------------------------------------------------------------
def train_and_predict(
    feats: dict,
    labels: pd.DataFrame,
    predict_date: pd.Timestamp,
    model: str = "gbdt",
) -> pd.Series:
    """用 predict_date 之前全部数据训练，预测 predict_date 截面（embargo=horizon）。

    预测日与训练网格解耦：预测日只依赖特征面板末端（去掉滞后面板后应到
    数据末端）；训练网格才与标签有效日期相交。
    """
    # 特征面板公共日期（用于确定预测日）
    feat_cal = None
    for f in feats.values():
        d = f.dropna(how="all").index
        feat_cal = d if feat_cal is None else feat_cal.intersection(d)
    if predict_date not in feat_cal:
        earlier = feat_cal[feat_cal <= predict_date]
        if len(earlier) == 0:
            raise ValueError(f"predict_date {predict_date} 无可用数据")
        predict_date = earlier[-1]
    log.info("特征网格: %d 日, 预测日=%s", len(feat_cal), predict_date.date())

    # 训练网格：特征 ∩ 标签有效日期
    train_cal = feat_cal.intersection(labels.dropna(how="all").index)
    train_end = predict_date - pd.Timedelta(days=HORIZON)
    tr_days = train_cal[train_cal <= train_end]
    if len(tr_days) < 60:
        raise ValueError(f"训练段不足 60 日: {len(tr_days)}")

    feat_train = {k: v.loc[tr_days] for k, v in feats.items()}
    label_train = labels.loc[tr_days]
    log.info("训练: %d 日 (%s ~ %s) → 预测 %s",
             len(tr_days), tr_days[0].date(), tr_days[-1].date(), predict_date.date())

    if model == "gbdt":
        try:
            from model.predictor import LGBMPredictor
            from scripts.e2e_common import GBDT_PARAMS
            p = LGBMPredictor(**GBDT_PARAMS)
        except ImportError:
            log.warning("lightgbm 不可用，降级 ridge")
            from model.predictor import RidgePredictor
            p = RidgePredictor(alpha=RIDGE_ALPHA)
    else:
        from model.predictor import RidgePredictor
        p = RidgePredictor(alpha=RIDGE_ALPHA)

    p.fit(feat_train, label_train)

    feat_pred = {k: v.loc[[predict_date]] for k, v in feats.items()}
    pred = p.predict(feat_pred)
    return pred.iloc[0] if len(pred) > 0 else pd.Series(dtype="float64")


# ---------------------------------------------------------------------------
# Step 4: 组合优化
# ---------------------------------------------------------------------------
def build_portfolio(
    scores: pd.Series,
    returns_panel: pd.DataFrame,
    predict_date: pd.Timestamp,
    method: str = "risk_parity",
    top_k: int = 30,
    max_weight: float = 0.1,
    risk_aversion: float = 5.0,
) -> pd.Series:
    """根据预测分数构建组合权重。

    - equal: top-k 等权
    - risk_parity / mvo: 只在 top-k 列上解 QP（全池 dropna(how="any") 会因
      停牌 NaN 清空协方差窗口 → 全空仓，已知数值坑）
    """
    scores_clean = scores.dropna().sort_values(ascending=False)
    top_codes = scores_clean.head(top_k).index.tolist()
    log.info("Top %d 候选: %s ... %s", top_k, top_codes[0], top_codes[-1])

    if method == "equal":
        return pd.Series(1.0 / top_k, index=top_codes)

    try:
        from optimize.solver import optimize_weights_qp
    except ImportError:
        log.warning("cvxpy 不可用，降级等权")
        return pd.Series(1.0 / top_k, index=top_codes)

    sub_ret = returns_panel[top_codes].loc[:predict_date]
    factor_sub = scores_clean.head(top_k).to_frame().T
    factor_sub.index = [predict_date]
    try:
        w = optimize_weights_qp(
            factor_sub, sub_ret, method=method if method != "risk_parity" else "risk_parity",
            max_weight=max_weight, window=120, min_periods=30,
            risk_aversion=risk_aversion)
        ww = w.iloc[-1].dropna() if len(w) else pd.Series(dtype=float)
        if len(ww) == 0 or ww.sum() <= 0:
            raise RuntimeError("空解")
        return ww
    except Exception as e:
        log.warning("%s 求解失败，降级等权: %s", method, e)
        return pd.Series(1.0 / top_k, index=top_codes)


# ---------------------------------------------------------------------------
# Step 5: 信号报告
# ---------------------------------------------------------------------------
def generate_report(
    scores: pd.Series,
    weights: pd.Series,
    predict_date: pd.Timestamp,
    out_dir: Path,
    n_features: int = 0,
    n_total_factors: int = 0,
    method: str = "risk_parity",
    max_weight: float = 0.1,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = predict_date.strftime("%Y%m%d")

    df = pd.DataFrame({"score": scores.dropna().sort_values(ascending=False)})
    df["weight"] = weights.reindex(df.index).fillna(0)
    df = df[df["weight"] > 1e-4].copy()
    df["rank"] = range(1, len(df) + 1)
    df = df[["rank", "score", "weight"]]

    csv_path = out_dir / f"picks_{date_str}.csv"
    df.to_csv(csv_path, encoding="utf-8-sig")
    pred_path = out_dir / f"model_prediction_{date_str}.csv"
    scores.dropna().sort_values(ascending=False).to_csv(
        pred_path, header=["score"], encoding="utf-8-sig")
    w_path = out_dir / f"portfolio_weights_{date_str}.csv"
    weights.to_csv(w_path, header=["weight"], encoding="utf-8-sig")

    txt_path = out_dir / f"picks_{date_str}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(" YuriQuant 端到端选股报告\n")
        f.write(f" 信号日期: {predict_date.date()}\n")
        f.write(f" 选股数量: {len(df)}\n")
        f.write(f" 因子筛选: {n_total_factors} 候选 -> {n_features} 入选\n")
        f.write(f" 组合方法: {method} (max_weight={max_weight})\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Rank':>4}  {'Code':<12} {'Score':>10} {'Weight':>10}\n")
        f.write("-" * 60 + "\n")
        for i, (code, row) in enumerate(df.iterrows(), 1):
            f.write(f"{i:>4}  {code:<12} {row['score']:>10.4f} {row['weight']:>10.4f}\n")
        f.write("=" * 60 + "\n")
        f.write("\n 注意事项:\n")
        f.write("  - 本清单由模型预测生成，非投资建议\n")
        f.write("  - 因子经 build_feature_set 三级漏斗筛选（覆盖率->去冗余->质量排序）\n")
        f.write(f"  - 组合权重经 {method} 优化（Ledoit-Wolf Σ）\n")
        f.write("  - 实际交易需考虑涨跌停/停牌/流动性约束\n")

    meta = {
        "predict_date": str(predict_date.date()),
        "n_picks": len(df),
        "n_features": n_features,
        "n_total_factors": n_total_factors,
        "top_score": float(df["score"].iloc[0]) if len(df) else None,
        "total_weight": float(df["weight"].sum()),
        "max_weight": float(df["weight"].max()),
        "min_weight": float(df["weight"].min()),
        "weight_std": float(df["weight"].std()) if len(df) > 1 else 0,
        "method": method,
        "max_weight_cap": max_weight,
        "files": {"picks_csv": str(csv_path), "picks_txt": str(txt_path),
                  "predictions": str(pred_path), "weights": str(w_path)},
    }
    with open(out_dir / "pipeline_log.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(" 端到端选股流水线启动")
    log.info(" 模式: %s", "真实" if args.real else "mock")
    log.info("=" * 60)

    # Step 1: 数据
    if args.real:
        px, lib_feats = load_daily_data(begin=args.begin)
    else:
        px = load_mock_data(n_days=args.n_days, n_codes=args.n_codes, seed=args.seed)
        lib_feats = {}

    # Step 2: 因子 + 选择
    classic = compute_classic_features(px)
    all_feats = {**classic, **lib_feats}
    log.info("因子总池: %d 个（经典 %d + 因子库 %d）",
             len(all_feats), len(classic), len(lib_feats))
    # 面板新鲜度守卫：剔除滞后面板（否则单个失效因子拖短预测日）
    all_feats = drop_stale_factors(all_feats, px["close"].index[-1])
    log.info("新鲜度过滤后: %d 个", len(all_feats))

    labels, fwd = build_label_pair(px["close"], horizon=HORIZON)
    last_feat_day = None
    for f in all_feats.values():
        d = f.dropna(how="all").index[-1] if len(f) > 0 else None
        if d is not None:
            last_feat_day = d if last_feat_day is None else min(last_feat_day, d)
    if last_feat_day is None:
        last_feat_day = px["close"].index[-1]
    log.info("最新可预测日: %s", last_feat_day.date())

    train_end = last_feat_day - pd.Timedelta(days=HORIZON)
    all_days = fwd.dropna(how="all").index
    valid_days = all_days[all_days <= train_end]
    if len(valid_days) < 60:
        valid_days = all_days

    feats = select_features(all_feats, fwd, valid_days, max_features=args.max_features)[0]
    log.info("筛选后因子: %d 个", len(feats))

    # Step 3: 模型预测
    scores = train_and_predict(feats, labels, last_feat_day, model=args.model).dropna()
    log.info("预测截面: %d 股有分数", len(scores))
    log.info("Top 5 预测: %s",
             {k: round(float(v), 4) for k, v in scores.head(5).items()})

    # Step 4: 组合
    returns_panel = px["close"].pct_change(fill_method=None)
    weights = build_portfolio(
        scores, returns_panel, last_feat_day,
        method=args.portfolio, top_k=args.top, max_weight=args.max_weight,
        risk_aversion=args.risk_aversion)
    log.info("组合权重: %d 只, 总权重=%.4f", len(weights), weights.sum())

    # Step 5: 报告
    meta = generate_report(
        scores, weights, last_feat_day, out_dir,
        n_features=len(feats), n_total_factors=len(all_feats),
        method=args.portfolio, max_weight=args.max_weight)
    log.info("报告已生成: %s", out_dir / f"picks_{meta['predict_date'].replace('-','')}.txt")

    log.info("=" * 60)
    log.info(" 完成: %d 候选 -> %d 因子 -> %d 选股", len(all_feats), len(feats), meta["n_picks"])
    log.info("=" * 60)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="端到端选股流水线")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--begin", type=int, default=20190101)
    ap.add_argument("--model", default="gbdt", choices=["gbdt", "ridge"])
    ap.add_argument("--portfolio", default="risk_parity",
                    choices=["mvo", "risk_parity", "equal"])
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--max-weight", type=float, default=0.05)
    ap.add_argument("--risk-aversion", type=float, default=5.0)
    ap.add_argument("--max-features", type=int, default=30)
    ap.add_argument("--n-days", type=int, default=500)
    ap.add_argument("--n-codes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
