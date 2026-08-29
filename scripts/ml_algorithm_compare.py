"""
长时段算法对比实验 —— 2019-2026 HS300，ridge / gbdt / TabICL + 基线
====================================================================

目的：在**更长时间范围**上对比因子合成算法，重点检验 TabICL（表格基础
模型，in-context learning，主打小样本强泛化）相对传统 ridge / LightGBM
是否有样本外优势。

实验契约（本脚本头部冻结，L3 扩展声明——begin 早于 config L2 的 20220101，
train_end/valid_end 与 config discipline 一致）：

1. 数据：HS300 PIT 并集池日线，2019-01-02 ~ 数据末端（2026-08）。
   ~1850 交易日（此前 ml_synthesis 实验仅用 2022-2025 共 969 日）。
2. 特征：12 个经典量价因子（动量/反转/波动/流动性/换手/振幅），**全期
   一致可得**。GP 因子库面板仅覆盖 2022-2025，为保证跨期一致不采用。
   选择漏斗只在 dev 段（train+valid）上做，test 不参与。
3. 切分：train 2019-2023（~1211 日）｜ valid 2024（242 日，调参+质量分）
   ｜ test 2025-01-01 ~ 2026-08（~400 日，跨 2025 风格反转 + 2026 全新数据）。
4. 标签：h=1 截面 rank（沿用 ml_synthesis 实验确认的最优视野；本实验
   目的是**同标签下算法横向对比**，h=1 的选择偏差披露见 ml_synthesis
   报告 7.2，不影响算法间相对排序）。
5. 调参：train 拟合 → valid rank IC 选优。
   ridge: alpha 网格；gbdt: lr×leaves 小网格；
   tabicl: context 样本量 {3000, 6000, 10000}（ICL 的关键超参——
   context 越大越接近「用全部历史」，越小越体现「小样本强泛化」）。
6. 上线：walk-forward，test 按季度 8 折滚动再训练，embargo = h = 1 日。
7. 基线：equal_weight / ic_weighted（valid 段定权重）。
8. 评价：test 段 rank IC / ICIR / NW-t / 月度 IC / 分年 IC（2025 vs 2026
   分开看——2026 是模型从未见过的时段）。

用法：
    python scripts/ml_algorithm_compare.py                # 完整实验（~25 分钟）
    python scripts/ml_algorithm_compare.py --skip-tune    # 复用调参缓存
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ml_algorithm_compare")

BEGIN = 20190102
TRAIN_END = "2023-12-31"
VALID_END = "2024-12-31"
HORIZON = 1
N_FOLDS = 8
OUT_DIR = Path("reports/ml_algorithm_compare")

# 复用 ml_synthesis_experiment 的口径（纯函数：经典因子 / 评价 / valid 调参）
from scripts.e2e_common import compute_classic_features
from scripts.ml_synthesis_experiment import (  # noqa: E402
    _eval_row,
    _fit_predict_valid,
    _monthly_ic,
)


# ---------------------------------------------------------------------------
# 数据：PIT 量价面板（2019-2026）
# ---------------------------------------------------------------------------
def _px_panels(begin: int, end: int | None) -> dict[str, pd.DataFrame]:
    """日线缓存 → PIT mask 后的 OHLCV 宽表（date×code）。"""
    from data.cache import DataCache
    from data.cache_helpers import load_daily
    from data.offline import OfflineDataSource
    from data.universe import Universe

    cache = DataCache(OfflineDataSource())
    uni = Universe(cache)
    _codes, _cal, daily = load_daily(cache, uni, "000300.SH", begin, end)
    out = {}
    for col in ("open", "high", "low", "close", "volume", "amount"):
        out[col] = daily[col].unstack("code").sort_index()
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(skip_tune: bool = False, do_register: bool = True) -> None:
    from model.labels import build_labels, forward_returns
    from model.predictor import LGBMPredictor, RidgePredictor, TabICLPredictor
    from scripts.walk_forward_model import rolling_oos

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. 数据与特征 ----------------
    px = _px_panels(BEGIN, None)
    all_days = px["close"].index
    log.info("PIT 量价面板: %d 日（%s ~ %s）× %d 股",
             len(all_days), all_days[0].date(), all_days[-1].date(),
             len(px["close"].columns))
    classic = compute_classic_features(px)
    feats_all = classic

    # ---------------- 2. 三段切分 ----------------
    tr_days = all_days[all_days <= pd.Timestamp(TRAIN_END)]
    va_days = all_days[(all_days > pd.Timestamp(TRAIN_END)) &
                       (all_days <= pd.Timestamp(VALID_END))]
    te_days = all_days[all_days > pd.Timestamp(VALID_END)]
    dev_days = all_days[all_days <= pd.Timestamp(VALID_END)]
    log.info("三段切分: train %d 日 | valid %d 日 | test %d 日（%s ~ %s）| embargo=%dd",
             len(tr_days), len(va_days), len(te_days),
             te_days[0].date(), te_days[-1].date(), HORIZON)

    # ---------------- 3. 标签 ----------------
    labels, _ = build_labels(px["close"], horizon=HORIZON, mode="rank")
    fwd = forward_returns(px["close"], horizon=HORIZON)

    # ---------------- 4. 特征漏斗（dev 段；单一实现 e2e_common.select_features） ----------------
    from scripts.e2e_common import select_features
    feats, quality = select_features(feats_all, fwd, quality_days=va_days,
                                     panel_days=dev_days, max_features=12)
    if quality is not None:
        log.info("valid 段特征质量分（|IC|）: %s",
                 {k: round(v, 4) for k, v in quality.items()})
    pd.Series({k: (quality.get(k, np.nan) if quality is not None else np.nan)
               for k in feats}, name="valid_abs_ic") \
        .to_csv(OUT_DIR / "selected_features.csv")

    # ---------------- 5. 调参（train → valid） ----------------
    tune_path = OUT_DIR / "tune_results.csv"
    if skip_tune and tune_path.exists():
        tune_df = pd.read_csv(tune_path)
        log.info("复用调参缓存（%d 组）", len(tune_df))
    else:
        rows = []
        log.info("调参 ridge ...")
        for a in (1.0, 10.0, 30.0, 100.0, 300.0):
            ic = _fit_predict_valid(
                lambda a=a: RidgePredictor(alpha=a),
                {k: v.loc[tr_days] for k, v in feats.items()}, labels.loc[tr_days],
                {k: v.loc[va_days] for k, v in feats.items()}, labels.loc[va_days])
            rows.append({"model": "ridge", "param": "alpha", "value": a, "valid_ic": ic})
            log.info("  ridge alpha=%-6g valid IC=%.4f", a, ic)

        log.info("调参 gbdt ...")
        for lr, leaves in ((0.03, 31), (0.05, 31), (0.05, 63), (0.03, 63)):
            ic = _fit_predict_valid(
                lambda lr=lr, leaves=leaves: LGBMPredictor(
                    learning_rate=lr, num_leaves=leaves, seed=42),
                {k: v.loc[tr_days] for k, v in feats.items()}, labels.loc[tr_days],
                {k: v.loc[va_days] for k, v in feats.items()}, labels.loc[va_days])
            rows.append({"model": "gbdt", "param": f"lr={lr},leaves={leaves}",
                         "value": leaves, "valid_ic": ic})
            log.info("  gbdt lr=%s leaves=%s valid IC=%.4f", lr, leaves, ic)

        log.info("调参 tabicl（context 样本量；n_estimators=2）...")
        for ctx in (3000, 6000, 10000):
            ic = _fit_predict_valid(
                lambda ctx=ctx: TabICLPredictor(
                    max_context_samples=ctx, n_estimators=2),
                {k: v.loc[tr_days] for k, v in feats.items()}, labels.loc[tr_days],
                {k: v.loc[va_days] for k, v in feats.items()}, labels.loc[va_days])
            rows.append({"model": "tabicl", "param": "context", "value": ctx, "valid_ic": ic})
            log.info("  tabicl context=%-6d valid IC=%.4f", ctx, ic)

        tune_df = pd.DataFrame(rows)
        tune_df.to_csv(tune_path, index=False)
    tune_df.to_csv(tune_path, index=False)

    best = {}
    for mdl in ("ridge", "gbdt", "tabicl"):
        sub = tune_df[tune_df["model"] == mdl]
        r = sub.loc[sub["valid_ic"].idxmax()]
        best[mdl] = r
        log.info("最优 %s: %s=%s (valid IC=%.4f)", mdl, r["param"], r["value"], r["valid_ic"])

    # ---------------- 6. walk-forward（test 8 折滚动再训练） ----------------
    preds: dict[str, pd.DataFrame] = {}
    log.info("walk-forward: ridge ...")
    preds["ridge_wf"] = rolling_oos(
        RidgePredictor, feats, labels, te_days, all_days,
        n_folds=N_FOLDS, embargo_days=HORIZON, min_train_days=120,
        alpha=float(best["ridge"]["value"]))

    log.info("walk-forward: gbdt ...")
    lr, leaves = str(best["gbdt"]["param"]).replace("lr=", "").replace(",leaves=", ",").split(",")
    preds["gbdt_wf"] = rolling_oos(
        LGBMPredictor, feats, labels, te_days, all_days,
        n_folds=N_FOLDS, embargo_days=HORIZON, min_train_days=120,
        learning_rate=float(lr), num_leaves=int(leaves), seed=42)

    log.info("walk-forward: tabicl（ICL，每折 context=折前最近样本）...")
    t_tab = time.time()
    preds["tabicl_wf"] = rolling_oos(
        TabICLPredictor, feats, labels, te_days, all_days,
        n_folds=N_FOLDS, embargo_days=HORIZON, min_train_days=120,
        max_context_samples=int(best["tabicl"]["value"]), n_estimators=2)
    log.info("tabicl walk-forward 耗时 %.1f 分钟", (time.time() - t_tab) / 60)

    # ---------------- 7. 基线 ----------------
    from factor.preprocessing import standardize_zscore
    from factor.synthesis import CompositeInput, rebuild_train_weights, synthesize_ic_weighted
    comps = list(feats.values())
    preds["baseline_equal"] = standardize_zscore(
        sum(c.fillna(0.0) for c in comps) / len(comps)).loc[te_days]
    ci = [CompositeInput(name=n, panel=p) for n, p in feats.items()]
    ci = rebuild_train_weights(ci, fwd, va_days)
    preds["baseline_icw"] = synthesize_ic_weighted(ci).loc[te_days]

    # ---------------- 8. 评价 ----------------
    rows = [_eval_row(tag, p, fwd, te_days) for tag, p in preds.items()]
    table = pd.DataFrame(rows)
    print("\n===== test 段（2025-01 ~ 2026-08，~400 日）OOS 对照表 =====")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.4f}"):
        print(table.to_string(index=False))
    table.to_csv(OUT_DIR / "oos_results.csv", index=False)

    monthly = pd.DataFrame({tag: _monthly_ic(p, fwd, te_days) for tag, p in preds.items()})
    monthly.to_csv(OUT_DIR / "monthly_ic.csv")

    # 分年 IC（2025 vs 2026：2026 是模型完全未见过的年份）
    yearly = {}
    for tag, p in preds.items():
        ic = pd.Series(dtype=float)
        for yr in sorted({d.year for d in te_days}):
            days = te_days[te_days.year == yr]
            sub = _eval_row(f"{tag}_{yr}", p, fwd, days)
            yearly[f"{tag}|{yr}"] = sub
    yearly_df = pd.DataFrame(yearly).T
    yearly_df.to_csv(OUT_DIR / "yearly_ic.csv")
    print("\n===== 分年 rank IC（2026 = 全新未见时段）=====")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.4f}"):
        print(yearly_df[["name", "ic_mean", "ic_ir", "ic_t_nw", "ic_win_rate", "n_days"]]
              .to_string(index=False))

    for tag in ("ridge_wf", "gbdt_wf", "tabicl_wf"):
        preds[tag].to_parquet(OUT_DIR / f"pred_{tag}.parquet")

    summary = {
        "begin": str(all_days[0].date()), "test_begin": str(te_days[0].date()),
        "test_end": str(te_days[-1].date()),
        "horizon": HORIZON, "n_folds": N_FOLDS,
        "n_train_days": int(len(tr_days)), "n_valid_days": int(len(va_days)),
        "n_test_days": int(len(te_days)),
        "features": sorted(feats),
        "best": {m: {"param": str(r["param"]), "value": float(r["value"]),
                     "valid_ic": float(r["valid_ic"])} for m, r in best.items()},
        "oos": {r["name"]: {k: (None if v != v else v) for k, v in r.items() if k != "name"}
                for r in rows},
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- 9. 注册 + 实验记录 ----------------
    if do_register:
        from model.registry import ModelRegistry
        reg = ModelRegistry(Path("reports") / "models")
        for tag in ("ridge_wf", "gbdt_wf", "tabicl_wf"):
            r = next(x for x in rows if x["name"] == tag)
            mdl = tag.split("_")[0]
            reg.register(
                name=f"{mdl}_wf_longrange_h{HORIZON}", kind="predictor",
                spec={"model": mdl, "horizon": HORIZON, "train_begin": str(all_days[0].date()),
                      "train_end": TRAIN_END, "valid_end": VALID_END,
                      "n_folds": N_FOLDS, "best": {str(best[mdl]["param"]):
                                                   float(best[mdl]["value"])},
                      "features": sorted(feats)},
                fingerprint=f"hs300_2019_2026:h{HORIZON}",
                train_begin=str(all_days[0].date()), train_end=TRAIN_END,
                metrics={"ic_mean": r["ic_mean"], "ic_ir": r["ic_ir"],
                         "ic_t_nw": r["ic_t_nw"]},
                parents=sorted(feats),
                note="长时段算法对比 2019-2026，test=2025-01~2026-08 walk-forward 8 折")

    from research.experiments import record_experiment
    record_experiment(
        kind="ml_algorithm_compare",
        command="ml_algorithm_compare",
        params={"begin": BEGIN, "train_end": TRAIN_END, "valid_end": VALID_END,
                "horizon": HORIZON, "n_folds": N_FOLDS,
                "features": sorted(feats),
                "best": {m: {str(r["param"]): float(r["value"])} for m, r in best.items()}},
        result_path=str(OUT_DIR / "oos_results.csv"),
        metrics={r["name"]: round(float(r["ic_mean"]), 4) for r in rows},
        note="长时段算法对比：ridge/gbdt/tabicl + 基线，test 2025-01~2026-08 (~400日)")

    log.info("实验完成，耗时 %.1f 分钟，产物: %s", (time.time() - t0) / 60, OUT_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="长时段算法对比实验（2019-2026）")
    ap.add_argument("--skip-tune", action="store_true", help="复用已存调参结果")
    ap.add_argument("--no-register", action="store_true", help="跳过模型注册")
    args = ap.parse_args()
    main(skip_tune=args.skip_tune, do_register=not args.no_register)
