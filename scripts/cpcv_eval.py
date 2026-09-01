"""CPCV 多路径评估 —— 挖因子方法选型决策工具。

把全历史切 N 组，取 k 组当测试集的所有组合 → φ=C(N,k) 条 OOS 路径，
每条路径独立 fit→predict→IC。产出 IC 分布而非单点，回答：
- GP / 暴力枚举 / GFlowNet 谁的 OOS IC 分布更优？
- 单一路径评估的排名在多路径下是否翻转？
- 某方法的高 IC 是实力还是运气（路径间方差）？

经典配置 N=6, k=2 → 15 条路径。

用法：
    python scripts/cpcv_eval.py                          # mock 快速验证
    python scripts/cpcv_eval.py --real                   # 真实 HS300
    python scripts/cpcv_eval.py --real --n-groups 6 --k 2
    python scripts/cpcv_eval.py --real --methods gp,exhaustive

产出：
    reports/cpcv_eval/path_ic.csv         每条路径 × 每方法的 IC
    reports/cpcv_eval/summary.csv         每方法的 IC 分布摘要
    reports/cpcv_eval/path_ic.html        IC 分布箱线图
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cli_common import add_real_mock_args, setup_logging  # noqa: E402

log = setup_logging("cpcv_eval")

OUT_DIR = Path("reports") / "cpcv_eval"

from factor.cv import CPCVPath, cpcv  # noqa: E402
from research.factor_analysis import calc_ic_series  # noqa: E402


def _eval_path_ic(
    predictor_cls,
    features: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    path: CPCVPath,
    fwd_returns: pd.DataFrame,
    min_train_days: int = 120,
    **predictor_params,
) -> float:
    """单条 CPCV 路径：fit(train) → predict(test) → IC(test)。

    Returns:
        该路径 test 段的 rank IC 均值（NaN 如果训练数据不足）。
    """
    tr = path.train_days
    te = path.test_days
    if len(tr) < min_train_days or len(te) == 0:
        return np.nan
    model = predictor_cls(**predictor_params)
    model.fit(
        {k: v.loc[tr] for k, v in features.items()},
        labels.loc[tr],
    )
    pred = model.predict({k: v.loc[te] for k, v in features.items()})
    # IC = pred vs fwd_returns 在 test 段
    ic = calc_ic_series(pred, fwd_returns.loc[te]).dropna()
    if len(ic) < 5:
        return np.nan
    return float(ic.mean())


def _build_mock_data():
    """mock 数据：注入 AR(1) 动量信号，500 日 × 50 股。"""
    rng = np.random.default_rng(0)
    n_days, n_codes = 500, 50
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]

    phi = 0.25
    rets = np.zeros((n_days, n_codes))
    for t in range(1, n_days):
        rets[t] = phi * rets[t - 1] + rng.normal(0, 0.02, n_codes)
    close = pd.DataFrame(10.0 * np.exp(np.cumsum(rets, axis=0)), idx, codes)
    volume = pd.DataFrame(rng.lognormal(12, 0.5, (n_days, n_codes)), idx, codes)
    returns_panel = close.pct_change().shift(-1)

    # 简单量价因子作为 ML 特征
    from factor.preprocessing import standardize_zscore
    feats = {
        "mom_5": standardize_zscore(close.pct_change(5)),
        "mom_10": standardize_zscore(close.pct_change(10)),
        "mom_20": standardize_zscore(close.pct_change(20)),
        "vol_5": standardize_zscore(close.pct_change().rolling(5).std()),
        "vol_20": standardize_zscore(close.pct_change().rolling(20).std()),
        "turn_5": standardize_zscore((volume / volume.rolling(5).mean())),
    }
    # 对齐
    feats = {k: v.reindex(index=close.index, columns=close.columns) for k, v in feats.items()}
    return feats, returns_panel, returns_panel


def _build_real_data():
    """真实 HS300 数据：PIT 面板 + 经典因子。"""
    from data.cache_helpers import load_pit_panels
    from factor.classic import compute_classic_features
    from model.labels import build_labels, forward_returns

    begin = 20190102
    px = load_pit_panels(begin, None)
    classic = compute_classic_features(px)
    labels, _ = build_labels(px["close"], horizon=1, mode="rank")
    fwd = forward_returns(px["close"], horizon=1)

    # 特征选择：用全段 valid 段 IC 排序选 top 12（单一实现 e2e_common.select_features）
    va_days = labels.index[(labels.index > pd.Timestamp("2024-01-01"))
                           & (labels.index <= pd.Timestamp("2024-12-31"))]
    from scripts.e2e_common import select_features
    feats, _quality = select_features(classic, fwd, quality_days=va_days,
                                      panel_days=fwd.index, max_features=12)
    return feats, labels, fwd


def main():
    parser = argparse.ArgumentParser(description="CPCV 多路径评估")
    add_real_mock_args(parser, real_help="真实 HS300 数据（默认 mock）")
    parser.add_argument("--n-groups", type=int, default=6, help="CPCV 组数（默认6）")
    parser.add_argument("--k", type=int, default=2, help="每路径测试组数（默认2）")
    parser.add_argument("--embargo-days", type=int, default=5, help="embargo 天数")
    parser.add_argument("--min-train-days", type=int, default=120, help="最少训练天数")
    parser.add_argument("--methods", default="ridge,gbdt",
                        help="评估方法，逗号分隔（ridge,gbdt,tabicl）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 数据 ----
    if args.real:
        log.info("加载真实 HS300 数据 ...")
        feats, labels, fwd = _build_real_data()
    else:
        log.info("使用 mock 数据（AR(1) 信号注入）...")
        feats, labels, fwd = _build_mock_data()

    all_days = labels.index
    log.info("全区间 %d 日 × %d 特征", len(all_days), len(feats))

    # ---- CPCV 路径 ----
    paths = cpcv(all_days, n_groups=args.n_groups, k=args.k,
                 embargo_days=args.embargo_days)
    log.info("CPCV: N=%d k=%d → %d 条路径", args.n_groups, args.k, len(paths))

    # ---- 评估方法 ----
    from model.predictor import LGBMPredictor, RidgePredictor
    method_map = {
        "ridge": (RidgePredictor, {"alpha": 1.0}),
        "gbdt": (LGBMPredictor, {"learning_rate": 0.03, "num_leaves": 31, "seed": 42}),
    }
    try:
        from model.predictor import TabICLPredictor
        method_map["tabicl"] = (TabICLPredictor, {"max_context_samples": 6000})
    except ImportError:
        log.info("TabICL 不可用，跳过")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in method_map:
            log.warning("未知方法 %s，跳过", m)
            methods.remove(m)

    # ---- 逐路径 × 逐方法 算 IC ----
    rows = []
    for path in paths:
        row = {"path_id": path.path_id, "test_groups": str(path.test_groups),
               "n_train": len(path.train_days), "n_test": len(path.test_days)}
        for m in methods:
            cls, params = method_map[m]
            ic = _eval_path_ic(cls, feats, labels, path, fwd,
                               min_train_days=args.min_train_days, **params)
            row[f"ic_{m}"] = ic
            log.info("  路径 %2d %s → IC %.4f", path.path_id, m, ic)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "path_ic.csv", index=False)
    log.info("路径 IC 已保存: %s", OUT_DIR / "path_ic.csv")

    # ---- 分布摘要 ----
    summary_rows = []
    for m in methods:
        col = f"ic_{m}"
        vals = df[col].dropna()
        summary_rows.append({
            "method": m,
            "n_paths": len(vals),
            "ic_mean": float(vals.mean()) if len(vals) else np.nan,
            "ic_std": float(vals.std()) if len(vals) else np.nan,
            "ic_min": float(vals.min()) if len(vals) else np.nan,
            "ic_max": float(vals.max()) if len(vals) else np.nan,
            "n_positive": int((vals > 0).sum()) if len(vals) else 0,
            "n_negative": int((vals <= 0).sum()) if len(vals) else 0,
            "pct_positive": float((vals > 0).mean()) if len(vals) else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    print("\n===== CPCV 多路径 IC 分布摘要 =====")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.4f}"):
        print(summary.to_string(index=False))

    # ---- 单路径排名 vs 多路径分布 ----
    if len(methods) >= 2:
        # 单路径 = 第一条路径
        single = df.iloc[0]
        single_rank = sorted(methods, key=lambda m: -single[f"ic_{m}"])
        # 多路径 = 均值排名
        multi_rank = sorted(methods, key=lambda m: -summary.loc[
            summary["method"] == m, "ic_mean"].values[0])
        if single_rank != multi_rank:
            print(f"\n⚠️ 排名翻转！单路径: {single_rank} → 多路径: {multi_rank}")
        else:
            print(f"\n✅ 排名稳定：单路径 = 多路径 = {multi_rank}")

    # ---- JSON 供报告引用 ----
    json_out = {
        "config": {"n_groups": args.n_groups, "k": args.k,
                   "embargo_days": args.embargo_days, "n_paths": len(paths)},
        "summary": summary.to_dict(orient="records"),
        "path_ic": df.to_dict(orient="records"),
    }
    (OUT_DIR / "cpcv_report_data.json").write_text(
        json.dumps(json_out, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    log.info("完成，产出: %s", OUT_DIR)


if __name__ == "__main__":
    main()
