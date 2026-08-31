"""
ML 因子合成实验 —— HS300 2022-2025，L2 三段纪律 + valid 调参 + 冻结 test
==========================================================================

实验协议（对齐公开研究最佳实践：purged CV / 嵌套调参 / 截面 rank 标签）：

1. 特征：因子库 GP 因子（hs300_2022_2025）+ 缓存量价派生经典因子，
   合计 ~31 候选；选择漏斗只在定型期（train+valid）上做，test 不参与。
2. 标签：截面 rank（build_labels horizon=5, mode=rank）；
   embargo = horizon = 5 日（purge 覆盖标签前视，测试段起点剔除）。
3. 切分（L2 冻结日历，config discipline 同源）：
      train  2022-2023（484 日）—— 拟合
      valid  2024（242 日）—— 超参搜索 / 特征质量分 / 早停判断
      test   2025（243 日）—— 冻结，一次性评估（报告后不得回头改）
4. 调参：train 拟合 → valid rank IC 选优（嵌套思想：外层 test 不参与任何选择）。
      ridge: alpha 网格；gbdt: 小学习率 × 浅树 × 强 min_child_samples 网格。
5. 上线模式（对照「一次重训 vs 滚动再训练」）：
      holdout      train+valid 一次拟合 → 预测 test 全段（减 5 日 embargo）
      walk_forward test 按 4 折滚动，每折用折前全部历史 − embargo 再拟合
6. 基线（蓝图验收纪律）：equal_weight / ic_weighted（valid 段定权重）。
7. 评价：test 段 rank IC 均值 / ICIR / Newey-West t / 月度 IC 稳定性 /
   分层多空（quantile_backtest）；结果落 CSV + ModelRegistry 注册 +
   experiments 记录。

用法：
    python scripts/ml_synthesis_experiment.py                 # 完整实验
    python scripts/ml_synthesis_experiment.py --skip-tune     # 复用已存调参结果
"""
from __future__ import annotations

import argparse
import itertools
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
log = logging.getLogger("ml_synthesis_experiment")

# 经典量价 12 因子：单一实现在 factor.classic.compute_classic_features
# （2026-08-29 收敛逐字副本；保留本别名兼容历史 import 与内部调用）
from factor.classic import compute_classic_features as _classic_features  # noqa: E402

DATASET = "hs300_2022_2025"
HORIZON = 5
EMBARGO = HORIZON
TRAIN_END = "2023-12-31"
VALID_END = "2024-12-31"
MAX_FEATURES = 20


def run(horizon: int = HORIZON, out_dir: str | None = None,
        skip_tune: bool = False, do_register: bool = True) -> None:
    """主流程（供 CLI 与二次导入复用；horizon 可覆盖默认 5 日）。"""
    H = int(horizon)
    out_dir = Path(out_dir or f"reports/ml_synthesis_h{H}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _main_impl(H, out_dir, skip_tune, do_register)


# ---------------------------------------------------------------------------
# 数据：量价面板（离线缓存） + 因子库 GP 因子
# ---------------------------------------------------------------------------
def _px_panels() -> dict[str, pd.DataFrame]:
    """从日线缓存读 OHLCV，对齐到因子库网格（300 股 × 969 日）。"""
    from data.cache import DataCache
    from data.offline import OfflineDataSource

    cache = DataCache(OfflineDataSource())
    d = pd.read_parquet(Path(cache.root) / "daily_hs300.parquet")

    from research.factor_library import FactorLibrary
    grid = next(iter(FactorLibrary(dataset=DATASET).load_library_features().values()))
    codes, dates = grid.columns, grid.index

    out = {}
    for col in ("open", "high", "low", "close", "volume", "amount"):
        w = (d.reset_index()
             .pivot(index="date", columns="code", values=col).sort_index()
             .reindex(index=dates, columns=codes))
        out[col] = w
    return out


# ---------------------------------------------------------------------------
# 评价工具
# ---------------------------------------------------------------------------
def _rank_ic(pred: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    from research.factor_analysis import calc_ic_series
    return calc_ic_series(pred, target)


def _eval_row(tag: str, pred: pd.DataFrame, fwd: pd.DataFrame,
              days: pd.DatetimeIndex) -> dict:
    from model.evaluation import evaluate_model
    from research.robust_stats import nw_tstat

    pred = pred.loc[days]
    tgt = fwd.loc[days]
    ic = _rank_ic(pred, tgt).dropna()
    t_nw, _, _ = nw_tstat(ic.values) if len(ic) > 1 else (0.0, 0.0, 0)
    from scipy import stats as st
    p = 2 * (1 - st.t.cdf(abs(t_nw), df=max(len(ic) - 1, 1)))
    ev = evaluate_model(pred, tgt)
    qb = ev["quantile_backtest"]
    ls = None
    if qb is not None and len(qb) > 1:
        # Q5(预测最高组) - Q1(最低组) 的日均收益（组内日收益 = 净值日差）
        ret = qb.diff().dropna(how="all")
        if "Q5" in ret.columns and "Q1" in ret.columns:
            ls = float((ret["Q5"] - ret["Q1"]).mean())
    return {
        "name": tag, "ic_mean": float(ic.mean()),
        "ic_ir": float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() > 0 else np.nan,
        "ic_t_nw": float(t_nw), "ic_p_nw": float(p),
        "ic_win_rate": float((ic > 0).mean()),
        "n_days": int(len(ic)),
        "ls_daily": ls,
    }


def _monthly_ic(pred: pd.DataFrame, fwd: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    ic = _rank_ic(pred.loc[days], fwd.loc[days]).dropna()
    return ic.groupby(ic.index.to_period("M")).mean()


# ---------------------------------------------------------------------------
# 调参：train 拟合 → valid rank IC 选优
# ---------------------------------------------------------------------------
def _fit_predict_valid(predictor, feats_tr, labels_tr, feats_va, labels_va) -> float:
    p = predictor()
    p.fit(feats_tr, labels_tr)
    pred = p.predict(feats_va)
    ic = _rank_ic(pred, labels_va).dropna()
    return float(ic.mean()) if len(ic) else np.nan


def tune_ridge(feats, labels, tr_days, va_days, alphas) -> pd.DataFrame:
    from model.predictor import RidgePredictor

    rows = []
    for a in alphas:
        ic = _fit_predict_valid(
            lambda a=a: RidgePredictor(alpha=a),
            {k: v.loc[tr_days] for k, v in feats.items()}, labels.loc[tr_days],
            {k: v.loc[va_days] for k, v in feats.items()}, labels.loc[va_days])
        rows.append({"model": "ridge", "alpha": a, "valid_ic": ic})
        log.info("  ridge alpha=%-6g valid IC=%.4f", a, ic)
    return pd.DataFrame(rows)


def tune_gbdt(feats, labels, tr_days, va_days, grid: dict) -> pd.DataFrame:
    from model.predictor import LGBMPredictor

    keys = list(grid)
    rows = []
    combos = list(itertools.product(*grid.values()))
    for i, combo in enumerate(combos, 1):
        kw = dict(zip(keys, combo))
        ic = _fit_predict_valid(
            lambda kw=kw: LGBMPredictor(**kw, seed=42),
            {k: v.loc[tr_days] for k, v in feats.items()}, labels.loc[tr_days],
            {k: v.loc[va_days] for k, v in feats.items()}, labels.loc[va_days])
        row = {"model": "gbdt", **kw, "valid_ic": ic}
        rows.append(row)
        log.info("  gbdt %d/%d %s valid IC=%.4f", i, len(combos),
                 {k: v for k, v in kw.items()}, ic)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 上线模式
# ---------------------------------------------------------------------------
def holdout_predict(predictor, feats, labels, tr_va_days, te_days, embargo: int):
    """train+valid 一次拟合；test 段起点向前剥 embargo 日（标签前视隔离）。"""
    fit_days = tr_va_days[:-embargo] if embargo > 0 else tr_va_days
    p = predictor()
    p.fit({k: v.loc[fit_days] for k, v in feats.items()}, labels.loc[fit_days])
    return p.predict({k: v.loc[te_days] for k, v in feats.items()}), fit_days


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _main_impl(H: int, out_dir: Path, skip_tune: bool, do_register: bool):
    from model.labels import build_labels, forward_returns
    from model.predictor import RidgePredictor, rolling_oos
    from research.factor_library import FactorLibrary

    t0 = time.time()

    # ---------------- 1. 特征 ----------------
    lib = FactorLibrary(dataset=DATASET)
    # 只取 GP 挖掘因子：库内另有 alpha101/alpha191 公开基准等其它 source，
    # 特征池保持历史口径（避免入库新来源后悄悄改变实验输入集）
    reg = lib.list_all()
    gp_names = {r["name"] for _, r in reg.iterrows()
                if str(r.get("source") or "").startswith("gp:")}
    gp_feats = {k: v for k, v in lib.load_library_features().items()
                if k in gp_names}
    px = _px_panels()
    classic = _classic_features(px)
    feats_all = {**{f"gp::{k}": v for k, v in gp_feats.items()}, **classic}
    all_days = px["close"].index
    log.info("特征候选 %d 个（GP %d + 经典 %d）| %d 日 × %d 股",
             len(feats_all), len(gp_feats), len(classic),
             len(all_days), len(px["close"].columns))

    # ---------------- 2. 三段切分（L2 冻结日历） ----------------
    tr_days = all_days[all_days <= pd.Timestamp(TRAIN_END)]
    va_days = all_days[(all_days > pd.Timestamp(TRAIN_END)) &
                       (all_days <= pd.Timestamp(VALID_END))]
    te_days = all_days[all_days > pd.Timestamp(VALID_END)]
    dev_days = all_days[all_days <= pd.Timestamp(VALID_END)]
    log.info("三段切分: train %d 日 (%s~%s) | valid %d 日 | test %d 日 (%s~%s) | embargo=%dd",
             len(tr_days), tr_days[0].date(), tr_days[-1].date(),
             len(va_days), len(te_days), te_days[0].date(), te_days[-1].date(), H)

    # ---------------- 3. 标签 ----------------
    labels, _ = build_labels(px["close"], horizon=H, mode="rank")
    fwd = forward_returns(px["close"], horizon=H)

    # ---------------- 4. 特征选择（只在 dev 段；单一实现 e2e_common.select_features） ----------------
    from scripts.e2e_common import select_features
    feats, quality = select_features(feats_all, fwd, quality_days=va_days,
                                     panel_days=dev_days, max_features=MAX_FEATURES)
    selected = sorted(feats)
    if quality is not None:
        log.info("valid 段质量分 Top10: %s",
                 {k: round(v, 4) for k, v in quality.head(10).items()})
    pd.Series({k: (quality.get(k, np.nan) if quality is not None else np.nan)
               for k in selected}, name="valid_abs_ic") \
        .to_csv(out_dir / "selected_features.csv")

    # ---------------- 5. 调参（train → valid） ----------------
    tune_path = out_dir / "tune_results.csv"
    if skip_tune and tune_path.exists():
        tune_df = pd.read_csv(tune_path)
        log.info("复用调参缓存 %s（%d 组）", tune_path, len(tune_df))
    else:
        log.info("调参: ridge alpha 网格")
        ridge_df = tune_ridge(feats, labels, tr_days, va_days,
                              alphas=[0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
        log.info("调参: gbdt 网格（小学习率 × 浅树 × 强正则）")
        gbdt_df = tune_gbdt(feats, labels, tr_days, va_days, grid={
            "learning_rate": [0.01, 0.05],
            "num_leaves": [15, 31],
            "min_child_samples": [100, 200],
            "n_estimators": [200, 400],
        })
        tune_df = pd.concat([ridge_df, gbdt_df], ignore_index=True)
        tune_df.to_csv(tune_path, index=False)

    best_ridge = (tune_df[tune_df["model"] == "ridge"]
                  .sort_values("valid_ic", ascending=False).iloc[0])
    best_gbdt = (tune_df[tune_df["model"] == "gbdt"]
                 .sort_values("valid_ic", ascending=False).iloc[0])
    log.info("最优 ridge: alpha=%s valid IC=%.4f", best_ridge["alpha"], best_ridge["valid_ic"])
    log.info("最优 gbdt: lr=%s leaves=%s min_child=%s n_est=%s valid IC=%.4f",
             best_gbdt["learning_rate"], best_gbdt["num_leaves"],
             best_gbdt["min_child_samples"], best_gbdt["n_estimators"],
             best_gbdt["valid_ic"])

    def ridge_best():
        return RidgePredictor(alpha=float(best_ridge["alpha"]))

    def gbdt_best():
        from model.predictor import LGBMPredictor
        return LGBMPredictor(
            learning_rate=float(best_gbdt["learning_rate"]),
            num_leaves=int(best_gbdt["num_leaves"]),
            min_child_samples=int(best_gbdt["min_child_samples"]),
            n_estimators=int(best_gbdt["n_estimators"]), seed=42)

    # ---------------- 6. 上线：holdout + walk-forward ----------------
    preds: dict[str, pd.DataFrame] = {}

    pred, fit_days = holdout_predict(ridge_best, feats, labels, dev_days, te_days, H)
    preds["ridge_holdout"] = pred
    log.info("holdout ridge: 拟合段 %s~%s（%d 日，尾剥 %d 日）→ 预测 %d 日",
             fit_days[0].date(), fit_days[-1].date(), len(fit_days), H, len(te_days))

    pred, fit_days = holdout_predict(gbdt_best, feats, labels, dev_days, te_days, H)
    preds["gbdt_holdout"] = pred
    log.info("holdout gbdt: 拟合段 %s~%s（%d 日）→ 预测 %d 日",
             fit_days[0].date(), fit_days[-1].date(), len(fit_days), len(te_days))

    from model.predictor import LGBMPredictor
    preds["ridge_wf"] = rolling_oos(
        RidgePredictor, feats, labels, te_days, all_days,
        n_folds=4, embargo_days=H, min_train_days=120,
        alpha=float(best_ridge["alpha"]))
    preds["gbdt_wf"] = rolling_oos(
        LGBMPredictor, feats, labels, te_days, all_days,
        n_folds=4, embargo_days=H, min_train_days=120,
        learning_rate=float(best_gbdt["learning_rate"]),
        num_leaves=int(best_gbdt["num_leaves"]),
        min_child_samples=int(best_gbdt["min_child_samples"]),
        n_estimators=int(best_gbdt["n_estimators"]), seed=42)

    # ---------------- 7. 基线 ----------------
    from factor.preprocessing import standardize_zscore
    comps = list(feats.values())
    preds["baseline_equal"] = standardize_zscore(
        sum(c.fillna(0.0) for c in comps) / len(comps)).loc[te_days]

    from factor.synthesis import CompositeInput, rebuild_train_weights, synthesize_ic_weighted
    ci = [CompositeInput(name=n, panel=p) for n, p in feats.items()]
    ci = rebuild_train_weights(ci, fwd, va_days)
    preds["baseline_icw"] = synthesize_ic_weighted(ci).loc[te_days]

    # ---------------- 8. 评价 ----------------
    rows = []
    for tag, pred in preds.items():
        rows.append(_eval_row(tag, pred, fwd, te_days))
    table = pd.DataFrame(rows)
    print("\n===== test 段（2025 冻结）OOS 对照表 =====")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.4f}"):
        print(table.to_string(index=False))
    table.to_csv(out_dir / "oos_results.csv", index=False)

    # 月度 IC 稳定性（test 段按月）
    monthly = pd.DataFrame({tag: _monthly_ic(p, fwd, te_days)
                            for tag, p in preds.items()})
    monthly.to_csv(out_dir / "monthly_ic.csv")
    print("\n===== test 段月度 rank IC =====")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.3f}"):
        print(monthly.to_string())

    # 分层回测（最优模型 vs 基线）
    from model.evaluation import evaluate_model
    layer = {}
    for tag in ("gbdt_holdout", "ridge_holdout", "baseline_icw", "baseline_equal"):
        if tag in preds:
            ev = evaluate_model(preds[tag].loc[te_days], fwd.loc[te_days])
            qb = ev["quantile_backtest"]
            if qb is not None:
                layer[tag] = qb.mean(axis=0) if isinstance(qb, pd.DataFrame) else None
    if layer:
        pd.DataFrame(layer).to_csv(out_dir / "quantile_daily_returns.csv")

    # 预测面板存档（供报告画图/复检）
    for tag in ("gbdt_holdout", "ridge_holdout"):
        preds[tag].to_parquet(out_dir / f"pred_{tag}.parquet")

    # ---------------- 9. 注册 + 实验记录 ----------------
    if do_register:
        from model.registry import ModelRegistry
        reg = ModelRegistry(Path("reports") / "models")
        fingerprint = f"{DATASET}:h{H}"
        for tag in ("ridge_holdout", "gbdt_holdout"):
            r = next(x for x in rows if x["name"] == tag)
            best = best_ridge if tag.startswith("ridge") else best_gbdt
            params = ({"alpha": float(best["alpha"])} if tag.startswith("ridge")
                      else {"learning_rate": float(best["learning_rate"]),
                            "num_leaves": int(best["num_leaves"]),
                            "min_child_samples": int(best["min_child_samples"]),
                            "n_estimators": int(best["n_estimators"]), "seed": 42})
            reg.register(
                name=f"{tag}_h{H}", kind="predictor",
                spec={"method": tag.split("_")[0], "features": selected,
                      "horizon": H, "target_mode": "rank",
                      "split": {"train": "2022-2023", "valid": "2024", "test": "2025",
                                "embargo_days": H},
                      "feature_selection": {"min_coverage": 0.5, "dedup_corr": 0.7,
                                            "max_features": MAX_FEATURES},
                      "predictor_params": params},
                fingerprint=fingerprint,
                train_begin=int(tr_days[0].strftime("%Y%m%d")),
                train_end=int(te_days[-1].strftime("%Y%m%d")),
                metrics={"ic_mean": r["ic_mean"], "ic_ir": r["ic_ir"],
                         "ic_t_nw": r["ic_t_nw"]},
                parents=selected,
                note="ml_synthesis_experiment holdout, test=2025 frozen")
        log.info("已注册 ridge/gbdt holdout 模型（账本 reports/models）")

        try:
            from research.experiments import record_experiment

            def _plain(s):
                return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in s.to_dict().items()}

            record_experiment(
                kind="ml_synthesis_experiment",
                command=" ".join(sys.argv),
                params={"dataset": DATASET, "horizon": H, "mode": "rank",
                        "split": "2022-2023/2024/2025", "embargo": H,
                        "n_features": len(selected), "features": selected,
                        "best_ridge": _plain(best_ridge),
                        "best_gbdt": _plain(best_gbdt)},
                data_fingerprint=fingerprint,
                result_path=str(out_dir / "oos_results.csv"),
                metrics={r["name"]: {"ic_mean": r["ic_mean"], "ic_ir": r["ic_ir"],
                                     "ic_t_nw": r["ic_t_nw"]} for r in rows},
            )
        except Exception as exc:
            log.warning("实验记录失败（不影响结果）: %s", exc)

    # 调参曲线快照（报告用）
    summary = {
        "n_candidates": len(feats_all), "n_selected": len(selected),
        "selected": selected,
        "train_days": len(tr_days), "valid_days": len(va_days), "test_days": len(te_days),
        "embargo": H, "horizon": H,
        "best_ridge": {"alpha": float(best_ridge["alpha"]),
                       "valid_ic": float(best_ridge["valid_ic"])},
        "best_gbdt": {"learning_rate": float(best_gbdt["learning_rate"]),
                      "num_leaves": int(best_gbdt["num_leaves"]),
                      "min_child_samples": int(best_gbdt["min_child_samples"]),
                      "n_estimators": int(best_gbdt["n_estimators"]),
                      "valid_ic": float(best_gbdt["valid_ic"])},
        "oos": {r["name"]: {"ic_mean": r["ic_mean"], "ic_ir": r["ic_ir"],
                            "ic_t_nw": r["ic_t_nw"], "ic_p_nw": r["ic_p_nw"],
                            "ic_win_rate": r["ic_win_rate"]}
                for r in rows},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"\n产物目录: {out_dir}")
    print(f"耗时: {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description="ML 因子合成实验（HS300 三段纪律）")
    ap.add_argument("--horizon", type=int, default=HORIZON,
                    help="预测 horizon（日）；embargo 自动等于 horizon")
    ap.add_argument("--out-dir", default=None, help="默认 reports/ml_synthesis_h{horizon}")
    ap.add_argument("--skip-tune", action="store_true", help="跳过调参，复用缓存结果")
    ap.add_argument("--no-register", action="store_true", help="不写 ModelRegistry/experiments")
    args = ap.parse_args()
    run(horizon=args.horizon, out_dir=args.out_dir, skip_tune=args.skip_tune,
        do_register=not args.no_register)


if __name__ == "__main__":
    main()
