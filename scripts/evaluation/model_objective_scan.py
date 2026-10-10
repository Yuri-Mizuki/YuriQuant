"""P2 —— 模型层「目标函数 / 样本权重」扫描。

背景
----
P2 超参扫描已收口（13 配置全在 seed 噪声地板 7.6e-4 内），但那一轮只扫了**容量类**超参
（n_estimators / learning_rate / num_leaves / max_depth）。模型层还有两个维度从未碰过：

1. **目标函数**：项目全部用 ``LGBMRegressor`` 的默认 L2 损失（外加一个 LambdaRank 变体）。
   从未试过 Huber / L1 / Quantile —— 这些对**收益分布的厚尾**理论上更稳健。
2. **样本权重**：全部样本等权。从未试过时间衰减（近期样本更重要）、波动率倒数
   （低波样本信噪比高）、幅度加权（只学大偏离）—— 即 `LGBMDataset` 的 weight 维度。

本脚本在**同一特征、同一标签口径、同一 CV** 下只改这两维，判据同 P2：**超过 seed 噪声地板**。

臂
--
- ``l2_rank``        : L2 回归（基线，= 项目默认）
- ``l2_zscore``      : 标签改截面 z-score（而非 rank）
- ``huber_rank``     : Huber 损失（厚尾稳健）
- ``l1_rank``        : MAE
- ``quant50_rank``   : 中位数分位损失
- ``ranker_bin2``    : LambdaRank（截面二分 pairwise）
- ``l2_wdecay``      : L2 + 时间半衰期样本权重
- ``l2_wvolinv``     : L2 + 波动率倒数权重
- ``l2_wabs``        : L2 + |标签偏离中位|权重（只学大偏离）
- ``l2_wday``        : L2 + 每日等权（消除每日在册股票数差异）

用法
----
    python scripts/evaluation/model_objective_scan.py --smoke
    python scripts/evaluation/model_objective_scan.py --begin 20180101 --end 20260831
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.seq_predictor_compare import _build_px_panels, _cs_z  # noqa: E402

log = logging.getLogger("model_objective_scan")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s")

OUT_DIR = ROOT / "reports" / "p2_objective"

ARMS = {
    "l2_rank":         dict(label="rank",   obj={}, w=None),
    "l2_zscore":       dict(label="zscore", obj={}, w=None),
    # 注意：huber 默认 alpha=0.9，而 rank 标签 ∈[0,1] ⇒ 残差恒 < alpha ⇒ 退化为 L2
    # （实测逐位相同）。必须显式压小 alpha 才是真正的厚尾稳健损失。
    "huber_rank_a05":  dict(label="rank",   obj={"objective": "huber", "alpha": 0.05}, w=None),
    "l1_rank":         dict(label="rank",   obj={"objective": "regression_l1"}, w=None),
    "quant50_rank":    dict(label="rank",   obj={"objective": "quantile", "alpha": 0.5}, w=None),
    "l2_wdecay":       dict(label="rank",   obj={}, w="decay"),
    "l2_wvolinv":      dict(label="rank",   obj={}, w="volinv"),
    "l2_wabs":         dict(label="rank",   obj={}, w="abs"),
    "l2_wday":         dict(label="rank",   obj={}, w="day"),
    "ranker_bin2":     dict(label="rank2",  obj={}, w=None),
}


def build_features(index_code: str, warmup: int, end: int, n_factors: int):
    """alpha158 工程化特征（截面 z-score）+ 复权 close。"""
    from factor.alpha158 import compute_alpha158
    from factor.alpha_base import AlphaData

    panels, close_adj = _build_px_panels(index_code, warmup, end)
    d = AlphaData(panels)
    t0 = time.time()
    p158 = compute_alpha158(d)
    names = sorted(p158)
    if n_factors and n_factors < len(names):
        names = names[:n_factors]
    dates, codes = close_adj.index, close_adj.columns
    flat = np.full((len(dates), len(codes), len(names)), np.nan, dtype=np.float32)
    for j, name in enumerate(names):
        v = _cs_z(p158[name]).reindex(index=dates, columns=codes).values
        flat[:, :, j] = v.astype(np.float32)
    # 60 日已实现波动（权重用）
    ret = close_adj.pct_change(fill_method=None)
    vol = ret.rolling(60, min_periods=20).std()
    log.info("[feat] %d 特征 (%.0fs) vol_nan=%.2f", len(names), time.time() - t0,
             float(vol.isna().mean().mean()))
    return flat, close_adj, vol, dates, codes


def _weights(kind: str, y: np.ndarray, day_pos: np.ndarray, vol: np.ndarray) -> np.ndarray | None:
    if kind is None:
        return None
    if kind == "decay":                        # 半衰期 500 交易日
        age = day_pos.max() - day_pos
        w = np.power(0.5, age / 500.0)
    elif kind == "volinv":
        w = 1.0 / np.where(vol > 0, vol, np.nan)
    elif kind == "abs":
        w = np.abs(y - np.nanmedian(y))
    elif kind == "day":
        w = np.ones_like(y)
        for d in np.unique(day_pos):
            m = day_pos == d
            w[m] = 1.0 / max(1, int(m.sum()))
    else:
        raise ValueError(kind)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    m = w.mean()
    return w / m if m > 0 else None    # 归一化到均值 1，避免改变有效学习率


def run(args):
    from factor.cv import forward_roll_folds
    from model.evaluation import evaluate_model
    from model.labels import build_labels, forward_returns

    flat, close_adj, vol, dates, codes = build_features(
        args.index, args.warmup, args.end, args.n_factors)

    b0, e0 = pd.Timestamp(str(args.begin)), pd.Timestamp(str(args.end))
    keep = (dates >= b0) & (dates <= e0)
    flat, dates = flat[keep], dates[keep]
    close_adj, vol = close_adj.loc[dates], vol.loc[dates]
    log.info("[slice] %s~%s %d 日 × %d 股", dates[0].date(), dates[-1].date(),
             len(dates), len(codes))

    labels_by_mode = {}
    for mode in {"rank", "zscore"}:
        labels_by_mode[mode], embargo = build_labels(close_adj, horizon=args.horizon, mode=mode)
    fwd = forward_returns(close_adj, horizon=args.horizon)
    log.info("[label] horizon=%dd embargo=%dd", args.horizon, embargo)

    n, n_codes = len(dates), len(codes)
    n_test = max(120, int(n * args.test_frac))
    test_days = dates[-n_test:]
    folds = forward_roll_folds(dates, test_days, args.n_folds, embargo)
    pos = {d: i for i, d in enumerate(dates)}
    valid_2d = close_adj.notna().values
    vol_flat = np.nan_to_num(vol.values.ravel(), nan=0.0)
    flat_flat = flat.reshape(-1, flat.shape[-1])

    preds = {k: pd.DataFrame(np.nan, index=test_days, columns=codes) for k in ARMS}
    for fi, fold in enumerate(folds, 1):
        tr_days, te_days = fold.train_days, fold.test_days
        if len(tr_days) < args.min_train_days or len(te_days) == 0:
            continue
        tri = np.array([pos[d] for d in tr_days])
        tei = np.array([pos[d] for d in te_days])
        rows_tr_all = (tri[:, None] * n_codes + np.arange(n_codes)[None, :])
        rows_te = (tei[:, None] * n_codes + np.arange(n_codes)[None, :]).ravel()
        valid_tr = valid_2d[tri]
        log.info("[fold %d/%d] train %s~%s (%d 日, 有效 %d) -> test %s~%s",
                 fi, len(folds), tr_days[0].date(), tr_days[-1].date(), len(tr_days),
                 int(valid_tr.sum()), te_days[0].date(), te_days[-1].date())
        Xte = flat_flat[rows_te]

        for name, spec in ARMS.items():
            mode = spec["label"]
            y_mode = "rank" if mode == "rank2" else mode
            lab = labels_by_mode[y_mode].loc[tr_days].values.ravel().astype(np.float32)
            ok = valid_tr.ravel() & ~np.isnan(lab)
            rows_tr = rows_tr_all.ravel()[ok]
            Xtr, yv = flat_flat[rows_tr], lab[ok]
            day_pos = (rows_tr // n_codes).astype(np.int64)
            p = _fit_arm(name, spec, Xtr, yv, Xte, day_pos, vol_flat[rows_tr], tri, n_codes, args)
            preds[name].loc[te_days] = p.reshape(len(te_days), n_codes)
        del Xte

    rows = []
    for name, pred in preds.items():
        ev = evaluate_model(_cs_z(pred), fwd.loc[pred.index])
        rows.append({"arm": name, "ic_mean": ev["ic_mean"], "ic_ir": ev["ic_ir"],
                     "ic_t_nw": ev["ic_t_nw"]})
    return pd.DataFrame(rows).sort_values("ic_mean", ascending=False), preds, fwd, test_days


def _fit_arm(name, spec, Xtr, yv, Xte, day_pos, vol_tr, tri, n_codes, args):
    from lightgbm import LGBMRanker, LGBMRegressor

    obj = dict(spec["obj"])
    w = _weights(spec["w"], yv, day_pos, vol_tr)
    base = dict(n_estimators=args.n_estimators, learning_rate=0.03, num_leaves=15,
                min_child_samples=50, n_jobs=args.threads, random_state=0, verbose=-1)

    if name == "ranker_bin2":
        day_uniq = np.unique(day_pos)
        grp = np.array([int((day_pos == d).sum()) for d in day_uniq])
        med = np.median(yv)
        y_int = (yv > med).astype(int)
        m = LGBMRanker(objective="rank_xendcg", **base)
        m.fit(Xtr, y_int, group=grp)
        return m.predict(Xte)

    m = LGBMRegressor(**base, **obj)
    if w is None:
        m.fit(Xtr, yv)
    else:
        m.fit(Xtr, yv, sample_weight=w)
    return m.predict(Xte)


def main():
    ap = argparse.ArgumentParser(description="P2 目标函数/样本权重扫描")
    ap.add_argument("--index", default="000300.SH")
    ap.add_argument("--begin", type=int, default=20180101)
    ap.add_argument("--end", type=int, default=20260831)
    ap.add_argument("--warmup", type=int, default=20160601)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--n-factors", type=int, default=0, help="0 = 全部 alpha158")
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--min-train-days", type=int, default=240)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-tag", default="hs300_h5")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.begin, args.end, args.warmup = 20230601, 20260831, 20220101
        args.n_folds, args.n_estimators, args.out_tag = 1, 60, "smoke"

    t0 = time.time()
    table, preds, fwd, test_days = run(args)
    print(f"\n===== P2 目标函数/权重对照（test {test_days[0].date()}~{test_days[-1].date()}，"
          f"h={args.horizon}）=====")
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.4f}"):
        print(table.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / f"arms_{args.out_tag}.csv", index=False)
    for name, pred in preds.items():
        _cs_z(pred).to_parquet(OUT_DIR / f"pred_{name}_{args.out_tag}.parquet")
    meta = dict(vars(args))
    meta["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / f"meta_{args.out_tag}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("产物 -> %s（耗时 %.0fs）", OUT_DIR, time.time() - t0)


if __name__ == "__main__":
    main()
