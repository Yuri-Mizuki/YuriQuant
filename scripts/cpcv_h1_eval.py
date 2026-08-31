"""
h=1 模型 CPCV 无偏评估
=======================

对 h=1 ML 合成模型做 CPCV（Combinatorial Purged Cross-Validation）多路径
无偏评估，消除 horizon 选择偏差。

背景：ml_synthesis_experiment 的 h=1 test IC 0.039-0.041（NW-t 2.5-2.9）
存在数据窥探——horizon=1 是在查看 test 结果后选择的（valid 段会选 h=20）。
CPCV 通过 φ=C(N,k) 条 OOS 路径产出 IC 分布而非单点，回答：
  - 0.04 IC 是实力还是运气？
  - 消除 horizon 选择偏差后 h=1 是否仍显著？

协议：
  1. 固定 h=1 已定型的 10 特征 + gbdt 超参（不再重新选择/调参）
  2. CPCV N=6, k=2 → 15 条路径，embargo=horizon
  3. 逐路径 fit→predict→IC，聚合为分布
  4. 路径间 t-test 显著性
  5. 对 h=5/h=20 同协议对比（固定同一批 10 特征，只换 horizon/labels/embargo）

判定标准：
  - 路径间 t-test p < 0.05 且正路径比例 > 10/15 → h=1 信号真实
  - CPCV 均值显著低于单路径 test IC 0.04 → 确认原 0.04 是乐观有偏
  - h=1 分布不显著优于 h=5/h=20 → horizon 选择偏差是主要来源

用法：
    python scripts/cpcv_h1_eval.py --mock                 # mock 验证管线
    python scripts/cpcv_h1_eval.py --real                 # 真实 HS300 h=1
    python scripts/cpcv_h1_eval.py --real --horizons 1,5  # 只比 h=1 vs h=5
    python scripts/cpcv_h1_eval.py --real --n-groups 6 --k 2

产出（reports/cpcv_h1/）：
    path_ic.csv          每条路径 × 每个 horizon 的 IC
    summary.csv          每个 horizon 的 IC 分布摘要 + t-test
    horizon_compare.csv  horizon 对比表
    cpcv_h1_report.json  完整数据
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

log = setup_logging("cpcv_h1_eval")

OUT_DIR = Path("reports") / "cpcv_h1"

from scipy import stats as sp_stats  # noqa: E402

from factor.cv import CPCVPath, cpcv  # noqa: E402
from model.labels import build_labels, forward_returns  # noqa: E402
from research.factor_analysis import calc_ic_series  # noqa: E402

# ===========================================================================
# h=1 固定配置（已定型，不重新选择）
# ===========================================================================
H1_FEATURES = [
    "gap",
    "gp::cs_normalize(ts_avedev_10(close))",
    "gp::ts_max_20(amount)",
    "mom10",
    "mom20",
    "mom5",
    "mom60",
    "rev1",
    "turn_trend",
    "vol60",
]

H1_GBDT_PARAMS = {
    "learning_rate": 0.01,
    "num_leaves": 15,
    "min_child_samples": 100,
    "n_estimators": 200,
    "seed": 42,
}

H1_RIDGE_PARAMS = {"alpha": 30.0}


# ===========================================================================
# 数据加载
# ===========================================================================
def _build_mock_data(horizon: int = 1):
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

    from factor.preprocessing import standardize_zscore
    ret1 = close.pct_change(fill_method=None)
    feats = {
        "gap": standardize_zscore(close.shift(1) / close - 1),  # proxy
        "mom5": standardize_zscore(close.pct_change(5, fill_method=None)),
        "mom10": standardize_zscore(close.pct_change(10, fill_method=None)),
        "mom20": standardize_zscore(close.pct_change(20, fill_method=None)),
        "mom60": standardize_zscore(close.pct_change(60, fill_method=None)),
        "rev1": standardize_zscore(-ret1),
        "turn_trend": standardize_zscore(volume.rolling(5).mean() / (volume.rolling(60).mean() + 1e-12)),
        "vol60": standardize_zscore(ret1.rolling(60).std()),
        # mock 简化：GP 因子用代理
        "gp::cs_normalize(ts_avedev_10(close))": standardize_zscore(ret1.rolling(10).mean()),
        "gp::ts_max_20(amount)": standardize_zscore((close * volume).rolling(20).max()),
    }
    feats = {k: v.reindex(index=close.index, columns=close.columns) for k, v in feats.items()}

    labels, _ = build_labels(close, horizon=horizon, mode="rank")
    fwd = forward_returns(close, horizon=horizon)
    return feats, labels, fwd


def _build_real_data(horizon: int = 1):
    """真实 HS300 数据：复用 ml_synthesis_experiment 的数据加载 + 固定 10 特征。"""
    from factor.classic import compute_classic_features
    from scripts.ml_synthesis_experiment import DATASET, _px_panels

    px = _px_panels()
    classic = compute_classic_features(px)

    # GP 因子从因子库加载
    from research.factor_library import FactorLibrary
    lib = FactorLibrary(dataset=DATASET)
    gp_feats = lib.load_library_features()

    # 合并：经典 + GP，只取 h=1 定型的 10 个
    all_feats = {**classic, **gp_feats}
    feats = {}
    for name in H1_FEATURES:
        if name in all_feats:
            panel = all_feats[name]
            # 对齐到 close 网格
            feats[name] = panel.reindex(index=px["close"].index, columns=px["close"].columns)
        else:
            log.warning("特征 %s 未找到，跳过", name)

    labels, _ = build_labels(px["close"], horizon=horizon, mode="rank")
    fwd = forward_returns(px["close"], horizon=horizon)
    return feats, labels, fwd


# ===========================================================================
# 单路径评估
# ===========================================================================
def _eval_path(
    predictor_cls,
    features: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    path: CPCVPath,
    fwd_returns: pd.DataFrame,
    min_train_days: int = 120,
    **predictor_params,
) -> dict:
    """单条 CPCV 路径：fit(train) → predict(test) → IC(test)。

    Returns:
        dict: ic_mean, ic_std, ic_ir, ic_t_nw, ic_p_nw, n_days, ic_win_rate
    """
    from research.robust_stats import nw_tstat

    tr = path.train_days
    te = path.test_days
    if len(tr) < min_train_days or len(te) == 0:
        return {"ic_mean": np.nan, "n_days": 0}

    model = predictor_cls(**predictor_params)
    model.fit(
        {k: v.loc[tr] for k, v in features.items()},
        labels.loc[tr],
    )
    pred = model.predict({k: v.loc[te] for k, v in features.items()})
    ic = calc_ic_series(pred, fwd_returns.loc[te]).dropna()
    if len(ic) < 5:
        return {"ic_mean": np.nan, "n_days": len(ic)}

    t_nw, _, _ = nw_tstat(ic.values) if len(ic) > 1 else (0.0, 0.0, 0)
    p_nw = 2 * (1 - sp_stats.t.cdf(abs(t_nw), df=max(len(ic) - 1, 1)))

    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()) if len(ic) > 1 else np.nan,
        "ic_ir": float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() > 0 else np.nan,
        "ic_t_nw": float(t_nw),
        "ic_p_nw": float(p_nw),
        "n_days": int(len(ic)),
        "ic_win_rate": float((ic > 0).mean()),
    }


# ===========================================================================
# 路径间显著性检验
# ===========================================================================
def _path_significance(path_ics: np.ndarray) -> dict:
    """对路径间 IC 均值做单样本 t-test（H0: mean IC = 0）。"""
    valid = path_ics[~np.isnan(path_ics)]
    if len(valid) < 3:
        return {"ttest_stat": np.nan, "ttest_p": np.nan, "note": "路径不足"}

    t_stat, p_val = sp_stats.ttest_1samp(valid, 0.0)
    return {
        "ttest_stat": float(t_stat),
        "ttest_p": float(p_val),
        "n_valid_paths": int(len(valid)),
        "n_positive": int((valid > 0).sum()),
        "pct_positive": float((valid > 0).mean()),
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="h=1 模型 CPCV 无偏评估")
    add_real_mock_args(parser, real_help="真实 HS300 数据", mock_help="用 mock 数据")
    parser.add_argument("--horizons", type=str, default="1,5,20",
                        help="对比的 horizon 列表，逗号分隔（默认 1,5,20）")
    parser.add_argument("--n-groups", type=int, default=6, help="CPCV 组数")
    parser.add_argument("--k", type=int, default=2, help="每路径测试组数")
    parser.add_argument("--min-train-days", type=int, default=120)
    parser.add_argument("--predictor", default="gbdt", choices=["gbdt", "ridge"],
                        help="预测器（默认 gbdt）")
    parser.add_argument("--out", type=str, default=None,
                        help="输出目录（默认 reports/cpcv_h1）")
    args = parser.parse_args()

    global OUT_DIR
    if args.out:
        OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    from model.predictor import LGBMPredictor, RidgePredictor

    if args.predictor == "gbdt":
        predictor_cls = LGBMPredictor
        predictor_params = H1_GBDT_PARAMS
    else:
        predictor_cls = RidgePredictor
        predictor_params = H1_RIDGE_PARAMS

    all_path_rows = []
    all_summary_rows = []

    for H in horizons:
        log.info("=" * 60)
        log.info("Horizon = %d (embargo = %d)", H, H)
        log.info("=" * 60)

        # 加载数据
        if args.real:
            feats, labels, fwd = _build_real_data(horizon=H)
        else:
            feats, labels, fwd = _build_mock_data(horizon=H)

        all_days = labels.index
        log.info("全区间 %d 日 × %d 特征", len(all_days), len(feats))

        # CPCV 切分
        paths = cpcv(all_days, n_groups=args.n_groups, k=args.k,
                     embargo_days=H)  # embargo = horizon
        log.info("CPCV: N=%d k=%d → %d 条路径，embargo=%d",
                 args.n_groups, args.k, len(paths), H)

        # 逐路径评估
        path_ics = []
        for path in paths:
            result = _eval_path(
                predictor_cls, feats, labels, path, fwd,
                min_train_days=args.min_train_days, **predictor_params,
            )
            ic_val = result.get("ic_mean", np.nan)
            path_ics.append(ic_val)
            row = {
                "horizon": H,
                "path_id": path.path_id,
                "test_groups": str(path.test_groups),
                "n_train": len(path.train_days),
                "n_test": len(path.test_days),
                "ic_mean": ic_val,
                "ic_std": result.get("ic_std", np.nan),
                "ic_ir": result.get("ic_ir", np.nan),
                "ic_t_nw": result.get("ic_t_nw", np.nan),
                "ic_p_nw": result.get("ic_p_nw", np.nan),
                "n_days": result.get("n_days", 0),
                "ic_win_rate": result.get("ic_win_rate", np.nan),
            }
            all_path_rows.append(row)
            log.info("  路径 %2d → IC %.4f (t_nw=%.2f, n=%d)",
                     path.path_id, ic_val, result.get("ic_t_nw", 0),
                     result.get("n_days", 0))

        # 路径间显著性
        sig = _path_significance(np.array(path_ics))

        # 分布摘要
        valid_ics = np.array([ic for ic in path_ics if not np.isnan(ic)])
        summary_row = {
            "horizon": H,
            "predictor": args.predictor,
            "n_paths": len(paths),
            "n_valid": len(valid_ics),
            "cpcv_ic_mean": float(valid_ics.mean()) if len(valid_ics) else np.nan,
            "cpcv_ic_std": float(valid_ics.std()) if len(valid_ics) > 1 else np.nan,
            "cpcv_ic_median": float(np.median(valid_ics)) if len(valid_ics) else np.nan,
            "cpcv_ic_min": float(valid_ics.min()) if len(valid_ics) else np.nan,
            "cpcv_ic_max": float(valid_ics.max()) if len(valid_ics) else np.nan,
            "n_positive": int((valid_ics > 0).sum()) if len(valid_ics) else 0,
            "pct_positive": float((valid_ics > 0).mean()) if len(valid_ics) else np.nan,
            "ttest_stat": sig.get("ttest_stat", np.nan),
            "ttest_p": sig.get("ttest_p", np.nan),
        }
        all_summary_rows.append(summary_row)

        log.info("--- Horizon %d CPCV 分布 ---", H)
        log.info("  IC 均值:   %.4f", summary_row["cpcv_ic_mean"])
        log.info("  IC 中位数: %.4f", summary_row["cpcv_ic_median"])
        log.info("  IC std:   %.4f", summary_row["cpcv_ic_std"])
        log.info("  正路径:   %d/%d (%.1f%%)",
                 summary_row["n_positive"], summary_row["n_valid"],
                 summary_row["pct_positive"] * 100 if not np.isnan(summary_row["pct_positive"]) else 0)
        log.info("  t-test:   t=%.2f, p=%.4f",
                 summary_row["ttest_stat"], summary_row["ttest_p"])

    # 保存
    path_df = pd.DataFrame(all_path_rows)
    path_df.to_csv(OUT_DIR / "path_ic.csv", index=False)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

    # horizon 对比表
    compare_df = summary_df[["horizon", "cpcv_ic_mean", "cpcv_ic_median",
                             "cpcv_ic_std", "n_positive", "pct_positive",
                             "ttest_stat", "ttest_p"]].copy()
    compare_df.to_csv(OUT_DIR / "horizon_compare.csv", index=False)

    # JSON
    json_out = {
        "config": {
            "horizons": horizons,
            "n_groups": args.n_groups,
            "k": args.k,
            "predictor": args.predictor,
            "predictor_params": predictor_params,
            "features": H1_FEATURES,
            "mock": not args.real,
        },
        "summary": summary_df.to_dict(orient="records"),
        "path_ic": path_df.to_dict(orient="records"),
    }
    with open(OUT_DIR / "cpcv_h1_report.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2, default=str)

    # 控制台摘要
    print(f"\n{'='*70}")
    print(f"CPCV 无偏评估摘要（predictor={args.predictor}, N={args.n_groups} k={args.k}）")
    print(f"{'='*70}")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.4f}"):
        print("\n--- Horizon 对比 ---")
        print(compare_df.to_string(index=False))

    # 判定结论
    h1_row = summary_df[summary_df["horizon"] == 1]
    if not h1_row.empty:
        h1 = h1_row.iloc[0]
        print("\n--- h=1 判定 ---")
        print(f"  CPCV IC 均值:      {h1['cpcv_ic_mean']:.4f}")
        print("  原单路径 test IC:  ~0.040（乐观有偏估计）")
        if not np.isnan(h1["ttest_p"]):
            if h1["ttest_p"] < 0.05 and h1["pct_positive"] > 10/15:
                print(f"  ✅ t-test p={h1['ttest_p']:.4f} < 0.05，正路径 {h1['n_positive']}/{h1['n_valid']} → 信号真实")
            else:
                print(f"  ⚠️ t-test p={h1['ttest_p']:.4f}，正路径 {h1['n_positive']}/{h1['n_valid']} → 信号不稳健")

        if len(horizons) > 1:
            others = summary_df[summary_df["horizon"] != 1]
            if not others.empty:
                h1_mean = h1["cpcv_ic_mean"]
                other_means = others["cpcv_ic_mean"].dropna()
                if len(other_means) and not np.isnan(h1_mean):
                    if h1_mean > other_means.max():
                        print(f"  ✅ h=1 CPCV 均值 {h1_mean:.4f} > 其他 horizon 最大值 {other_means.max():.4f}")
                    else:
                        print(f"  ⚠️ h=1 CPCV 均值 {h1_mean:.4f} 不显著优于其他 horizon → horizon 选择偏差确认")
    print(f"{'='*70}")
    log.info("产出: %s", OUT_DIR)


if __name__ == "__main__":
    main()
