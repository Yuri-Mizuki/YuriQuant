"""
模型层滚动训练 —— Predictor 的 walk-forward 编排
==================================================

模型层蓝图 Step 6（reports/yuriquant_model_layer_design）：把 walk_forward.py
的三段纪律升级为模型级滚动训练。

流程（纪律：L2 段落契约 = config discipline 冻结日历）：
    1. 特征：--mock 用注入信号的 mock 因子；--real 用因子库面板（或 --raw-features
       原始量价）。
    2. 标签：build_labels(close, horizon, mode) + embargo=horizon。
    3. 定型期（train+valid）：特征选择（覆盖率/去冗余，质量分=valid 段 |IC|）
       ——只用定型期数据，test 段不参与任何选择。
    4. 上线期（test）：按 --n-folds 折滚动再训练（每折用折前全部历史，
       去掉 horizon 日隔离带），OOS 预测拼接。
    5. 双基线对照（蓝图验收纪律）：等权合成 / IC 加权合成（valid 段定权重）。
    6. 出口：evaluate_model 统一评价 + ModelRegistry 训练即注册
       （mock 落 reports/models_mock，不污染真实账本）；
       --save-library 时经 serving.register_model_as_factor 回写因子库。

用法：
    python scripts/walk_forward_model.py --mock                     # 快速验证（无 SDK）
    python scripts/walk_forward_model.py --mock --methods ridge,gbdt --save-library
    python scripts/walk_forward_model.py --real                     # 因子库特征（2022-2025）
    python scripts/walk_forward_model.py --real --raw-features      # 原始量价特征
"""
from __future__ import annotations

import argparse
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
log = logging.getLogger("walk_forward_model")


# ---------------------------------------------------------------------------
# 特征来源
# ---------------------------------------------------------------------------
def _mock_feature_panels(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """mock 因子特征（复用 mine_factors 的 AR(1) 动量信号面板派生）。"""
    from factor.preprocessing import standardize_zscore

    close, open_ = panel["close"], panel["open"]
    amount = panel["amount"]
    ret1 = close.pct_change()
    feats = {
        "mom5": close.pct_change(5),
        "mom10": close.pct_change(10),
        "mom20": close.pct_change(20),
        "rev1": -ret1,
        "vol20": ret1.rolling(20).std(),
        "amihud20": (ret1.abs() / (amount + 1e-12)).rolling(20).mean(),
        "gap": open_ / close.shift(1) - 1,
    }
    return {k: standardize_zscore(v) for k, v in feats.items()}


def _derive_library_dataset(begin: int, end: int) -> str:
    """真实模式因子库数据集名（对齐 mine_factors._derive_dataset 约定）。"""
    from config import Config
    idx = Config.get()["universe"]["index_code"].split(".")[0]
    yr_b, yr_e = str(begin)[:4], str(end)[:4]
    yr = yr_b if yr_b == yr_e else f"{yr_b}_{yr_e}"
    return f"{idx}_{yr}"


def _load_real_features(args, begin: int, end: int
                        ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """真实模式特征：因子库面板（默认）或原始量价（--raw-features）。"""
    from config import Config
    from data.cache_helpers import build_panel
    from factor.preprocessing import standardize_zscore

    cfg = Config.get()
    panel_dict, _rets = build_panel(cfg, begin, end, retry=True)
    close = panel_dict["close"]

    if args.raw_features:
        feats = {
            "open": panel_dict["open"], "high": panel_dict["high"],
            "low": panel_dict["low"], "close": close,
            "volume": panel_dict["volume"], "amount": panel_dict["amount"],
            "vwap": panel_dict["amount"] / panel_dict["volume"].replace(0, np.nan),
        }
        feats = {k: standardize_zscore(v) for k, v in feats.items()}
        log.info("原始量价特征 %d 个（已截面标准化）", len(feats))
    else:
        from research.factor_library import FactorLibrary
        lib = FactorLibrary(dataset=args.library_dataset)
        feats = lib.load_library_features()
        if not feats:
            raise RuntimeError(
                f"因子库数据集 {args.library_dataset!r} 为空——先跑 mine_factors --save-library "
                "或改用 --raw-features")
        log.info("因子库特征 %d 个（数据集 %s）", len(feats), args.library_dataset)
    return feats, close


# ---------------------------------------------------------------------------
# 滚动训练（上线期）
# ---------------------------------------------------------------------------
def rolling_oos(
    predictor_cls,
    features: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    test_days: pd.DatetimeIndex,
    all_days: pd.DatetimeIndex,
    n_folds: int,
    embargo_days: int,
    min_train_days: int,
    **predictor_params,
) -> pd.DataFrame:
    """test 段按日历等分折；每折用「折前全部历史 − embargo 隔离带」训练，预测折内。

    OOS 拼接纪律：折间日期不交叉（每天只被其之后的数据预测过）；训练段尾部
    剔除 embargo_days 天（horizon 标签前视隔离带）。
    """
    fold_days = np.array_split(test_days.to_numpy(), n_folds)
    codes = labels.columns
    out = pd.DataFrame(np.nan, index=test_days, columns=codes)
    for i, fd in enumerate(fold_days, 1):
        fold = pd.DatetimeIndex(fd)
        cut = fold[0]
        tr_days = all_days[all_days < cut]
        if embargo_days > 0:
            tr_days = tr_days[:-embargo_days]
        if len(tr_days) < min_train_days:
            log.warning("折 %d 训练段不足（%d 日 < %d），跳过", i, len(tr_days), min_train_days)
            continue
        p = predictor_cls(**predictor_params)
        p.fit({k: v.loc[tr_days] for k, v in features.items()}, labels.loc[tr_days])
        pred = p.predict({k: v.loc[fold] for k, v in features.items()})
        out.loc[fold] = pred.reindex(index=fold, columns=codes)
        log.info("滚动折 %d/%d: train %s~%s (%d 日, 去 %d 日隔离带) -> 预测 %s~%s (%d 日)",
                 i, n_folds, tr_days[0].date(), tr_days[-1].date(), len(tr_days),
                 embargo_days, fold[0].date(), fold[-1].date(), len(fold))
    if not out.notna().any().any():
        raise RuntimeError("滚动训练无任何 OOS 预测产出")
    return out


# ---------------------------------------------------------------------------
# 基线（蓝图验收纪律：等权 / IC 加权，valid 段定权重）
# ---------------------------------------------------------------------------
def _equal_weight_panel(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from factor.preprocessing import standardize_zscore
    comps = list(features.values())
    eq = sum(c.fillna(0.0) for c in comps) / len(comps)
    return standardize_zscore(eq)


def _ic_weighted_panel(features: dict[str, pd.DataFrame],
                       fwd: pd.DataFrame, valid_days: pd.DatetimeIndex) -> pd.DataFrame:
    from factor.synthesis import CompositeInput, rebuild_train_weights, synthesize_ic_weighted

    comps = [CompositeInput(name=n, panel=p) for n, p in features.items()]
    comps = rebuild_train_weights(comps, fwd, valid_days)
    return synthesize_ic_weighted(comps)


# ---------------------------------------------------------------------------
# 评价行
# ---------------------------------------------------------------------------
def _eval_row(tag: str, panel: pd.DataFrame, target: pd.DataFrame) -> dict:
    from model.evaluation import evaluate_model
    ev = evaluate_model(panel, target)
    return {
        "name": tag,
        "ic_mean": ev["ic_mean"], "ic_ir": ev["ic_ir"],
        "ic_t_nw": ev["ic_t_nw"], "ic_p_nw": ev["ic_p_nw"],
        "n_days": int(ev["ic_series"].notna().sum()),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="模型层滚动训练（Predictor walk-forward）")
    ap.add_argument("--mock", action="store_true", help="mock 数据（默认）")
    ap.add_argument("--real", action="store_true", help="真实数据（SDK/缓存）")
    ap.add_argument("--begin", type=int, default=None,
                    help="真实模式起始日（默认 discipline.begin）")
    ap.add_argument("--end", type=int, default=None, help="真实模式结束日（默认至今）")
    ap.add_argument("--horizon", type=int, default=5, help="预测视野（交易日）")
    ap.add_argument("--mode", default="rank", choices=["rank", "zscore", "raw"],
                    help="标签变换（rank 与 rank IC 评价口径对齐，推荐）")
    ap.add_argument("--n-folds", type=int, default=4, help="上线期滚动折数（如 4=季度）")
    ap.add_argument("--methods", default="gbdt", help="预测器列表（逗号分隔：ridge,gbdt）")
    ap.add_argument("--n-estimators", type=int, default=300, help="gbdt 树数")
    ap.add_argument("--min-train-days", type=int, default=120, help="滚动折最小训练天数")
    ap.add_argument("--dedup-corr", type=float, default=0.7, help="特征去冗余相关阈值")
    ap.add_argument("--max-features", type=int, default=50, help="特征数上限")
    ap.add_argument("--min-coverage", type=float, default=0.5, help="特征覆盖率下限")
    ap.add_argument("--seed", type=int, default=0)
    # mock 专用
    ap.add_argument("--mock-days", type=int, default=500)
    ap.add_argument("--mock-codes", type=int, default=60)
    # 真实专用
    ap.add_argument("--library-dataset", default=None, help="因子库数据集（默认自动推导）")
    ap.add_argument("--raw-features", action="store_true", help="用原始量价特征（不用因子库）")
    # 出口
    ap.add_argument("--save-library", action="store_true",
                    help="模型 OOS 面板回写因子库（联动②血缘回写）")
    ap.add_argument("--registry-root", default=None,
                    help="模型账本目录（默认：real→reports/models，mock→reports/models_mock）")
    args = ap.parse_args()

    from model.features import build_feature_set
    from model.labels import build_labels, forward_returns
    from model.predictor import PREDICTORS
    from model.registry import ModelRegistry
    from model.serving import register_model_as_factor

    t0 = time.time()
    real = args.real and not args.mock
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in PREDICTORS:
            raise SystemExit(f"未知预测器 {m!r}，可选: {sorted(PREDICTORS)}")

    # ---------------- 1. 数据与特征 ----------------
    if real:
        from config import Config
        disc = Config.discipline()
        begin = args.begin or disc["begin"]
        end = args.end or 20261231
        if args.library_dataset is None:
            args.library_dataset = _derive_library_dataset(begin, end)
        feats, close = _load_real_features(args, begin, end)
        all_days = close.index
        valid_days = all_days[(all_days > pd.Timestamp(str(disc["train_end"]))) &
                              (all_days <= pd.Timestamp(str(disc["valid_end"])))]
        test_days = all_days[all_days > pd.Timestamp(str(disc["valid_end"]))]
        fingerprint = _real_fingerprint()
        tag = f"real_{args.library_dataset}"
    else:
        from scripts.mine_factors import gen_mock_panel_with_signal
        panel = gen_mock_panel_with_signal(n_days=args.mock_days,
                                           n_codes=args.mock_codes, seed=args.seed)
        close = panel["close"]
        feats = _mock_feature_panels(panel)
        all_days = close.index
        n = len(all_days)
        i_train, i_valid = int(n * 0.6), int(n * 0.8)
        valid_days = all_days[i_train:i_valid]
        test_days = all_days[i_valid:]
        fingerprint = f"mock:seed{args.seed}:days{args.mock_days}:codes{args.mock_codes}"
        tag = "mock"
    if len(test_days) < args.n_folds * 5:
        raise SystemExit(f"test 段太短（{len(test_days)} 日），不足以切 {args.n_folds} 折")

    dev_days = all_days[:len(all_days) - len(test_days)]   # 定型期 = train + valid
    log.info("[%s] 特征 %d 个 | 定型期 %s~%s（含 valid %d 日）| test %d 日 | fingerprint=%s",
             tag, len(feats), all_days[0].date(), dev_days[-1].date(),
             len(valid_days), len(test_days), fingerprint)

    # ---------------- 2. 标签 ----------------
    labels, embargo = build_labels(close, horizon=args.horizon, mode=args.mode)
    fwd = forward_returns(close, horizon=args.horizon)
    log.info("标签: horizon=%dd mode=%s embargo=%dd（隔离带=标签前视长度）",
             args.horizon, args.mode, embargo)

    # ---------------- 3. 定型期特征选择（test 不参与） ----------------
    from research.factor_analysis import calc_ic_series
    quality = None
    try:
        q = {}
        for nm, p in feats.items():
            ic = calc_ic_series(p.loc[valid_days], fwd.loc[valid_days]).dropna()
            if len(ic) >= 10:
                q[nm] = abs(float(ic.mean()))
        if q:
            quality = pd.Series(q)
            log.info("valid 段质量分（|IC|）: %s",
                     {k: round(v, 4) for k, v in quality.sort_values(ascending=False).items()})
    except Exception as exc:
        log.warning("质量分计算失败（按独立性去冗余）: %s", exc)

    feats_dev = build_feature_set(
        {k: v.loc[dev_days] for k, v in feats.items()},
        min_coverage=args.min_coverage, dedup_corr=args.dedup_corr,
        max_features=args.max_features, quality=quality,
    )
    selected = sorted(feats_dev.keys())
    feats_sel = {k: feats[k] for k in selected}    # 全时段面板（test 段只做预测）
    log.info("定型期特征选择: %d -> %d -> %s", len(feats), len(feats_dev), selected)

    # ---------------- 4. 滚动训练 + 双基线 ----------------
    rows = []
    panels: dict[str, pd.DataFrame] = {}
    for m in methods:
        params: dict = {}
        if m == "gbdt":
            params = {"seed": args.seed, "n_estimators": args.n_estimators}
        pred = rolling_oos(
            PREDICTORS[m], feats_sel, labels, test_days, all_days,
            n_folds=args.n_folds, embargo_days=embargo,
            min_train_days=args.min_train_days, **params,
        )
        panels[m] = pred
        rows.append(_eval_row(f"model:{m}", pred, fwd.loc[test_days]))

    baseline_eq = _equal_weight_panel(feats_sel).loc[test_days]
    baseline_ic = _ic_weighted_panel(feats_sel, fwd, valid_days).loc[test_days]
    rows.append(_eval_row("baseline:equal_weight", baseline_eq, fwd.loc[test_days]))
    rows.append(_eval_row("baseline:ic_weighted", baseline_ic, fwd.loc[test_days]))

    table = pd.DataFrame(rows)
    print(f"\n===== OOS 对照表（test 段 {test_days[0].date()} ~ {test_days[-1].date()}，"
          f"horizon={args.horizon}d，目标=horizon 前瞻收益）=====")
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.4f}"):
        print(table.to_string(index=False))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = Path("reports") / f"model_walk_forward_{tag}_h{args.horizon}_{stamp}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    log.info("对照表已保存: %s", out_csv)

    # ---------------- 5. 训练即注册（ModelRegistry） ----------------
    reg_root = Path(args.registry_root) if args.registry_root else \
        (Path("reports") / "models" if real else Path("reports") / "models_mock")
    reg = ModelRegistry(reg_root)
    model_ids: dict[str, str] = {}
    for m in methods:
        r = next(x for x in rows if x["name"] == f"model:{m}")
        spec = {
            "method": m, "features": selected,
            "horizon": args.horizon, "target_mode": args.mode,
            "rolling": {"n_folds": args.n_folds, "embargo_days": embargo,
                        "min_train_days": args.min_train_days},
            "feature_selection": {"dedup_corr": args.dedup_corr,
                                  "min_coverage": args.min_coverage,
                                  "max_features": args.max_features},
            "predictor_params": {"seed": args.seed, "n_estimators": args.n_estimators}
                                if m == "gbdt" else {},
        }
        model_ids[m] = reg.register(
            name=f"{m}_h{args.horizon}", kind="predictor", spec=spec,
            fingerprint=fingerprint,
            train_begin=int(all_days[0].strftime("%Y%m%d")),
            train_end=int(test_days[-1].strftime("%Y%m%d")),
            metrics={"ic_mean": r["ic_mean"], "ic_ir": r["ic_ir"], "ic_t_nw": r["ic_t_nw"]},
            parents=selected,
            note=(f"rolling walk-forward {args.n_folds} folds, "
                  f"test {test_days[0].date()}~{test_days[-1].date()}"),
        )
        log.info("注册模型 %s_h%d -> model_id=%s（账本 %s）",
                 m, args.horizon, model_ids[m], reg_root)

    # ---------------- 6. 回写因子库（联动②） ----------------
    if args.save_library:
        dataset = None if real else "mock"
        for m in methods:
            name = f"model:{m}_h{args.horizon}"
            row = register_model_as_factor(
                name=name, pred_panel=panels[m], returns_panel=fwd,
                parents=selected, dataset=dataset, model_id=model_ids[m],
                horizon=args.horizon, oos=True,
                note=f"{tag} rolling walk-forward",
            )
            log.info("回写因子库 %s（dataset=%s, ic_mean=%.4f）",
                     name, dataset, row.get("ic_mean", float("nan")))

    # ---------------- 7. 实验记录 ----------------
    try:
        import sys as _sys

        from research.experiments import record_experiment
        record_experiment(
            kind="model_walk_forward",
            command=" ".join(_sys.argv),
            params={"real": real, "horizon": args.horizon, "mode": args.mode,
                    "methods": methods, "n_folds": args.n_folds,
                    "n_features": len(selected), "features": selected,
                    "library_dataset": args.library_dataset if real else "mock"},
            data_fingerprint=fingerprint,
            result_path=str(out_csv),
            metrics={r["name"]: {"ic_mean": r["ic_mean"], "ic_ir": r["ic_ir"],
                                 "ic_t_nw": r["ic_t_nw"]} for r in rows},
        )
    except Exception as exc:
        log.warning("实验记录写入失败（不影响结果）: %s", exc)

    print(f"\n耗时: {time.time() - t0:.0f}s")


def _real_fingerprint() -> str:
    try:
        from data.cache import DataCache
        from data.datasource import create_datasource
        return DataCache(create_datasource()).get_fingerprint()
    except Exception:
        return "real:unavailable"


if __name__ == "__main__":
    main()
