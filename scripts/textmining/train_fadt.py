"""
FADT 训练与因子构建（对齐华泰 AI 57 文本FADT选股）
==================================================

与 AI 51（SUE.txt）的参数差异（AI 57 图表 17 基准模型）：
1. 词域：标题 top200 + 摘要 top1000（AI 51 是 100+500）；
2. 滚动：样本内 12 个月 + 样本外 12 个月（AI 51 是 24+12）；
3. 因子：log-odds(涨)-log-odds(跌)，月末回溯 3 个月，**个股全部调整研报
   得分直接求均值**（研报："追溯过去 3 个月的全部分析师盈利预测调整样本，
   分别计算出文本得分，最后求均值"）——**无 0.95^(T-t) 指数衰减**
   （AI 51 有衰减，AI 57 无）。

用法：
    python -m scripts.textmining.train_fadt --model xgb --pool zz1000
    python -m scripts.textmining.train_fadt --model logit --pool hs300

产出：
    reports/textmining/fadt_factor_{model}_{pool}.parquet
    reports/textmining/fadt_train_{model}_{pool}.log
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.textmining.train_sue_txt import (
    LOGIT_LAMBDA,
    SUEVectorizer,
    _auc_ovr,
    _sue0_from_model,
    make_labels,
    train_logit,
    tokenize_summary,
    tokenize_title,
)

OUT_DIR = Path(r"E:\YuriQuant\reports\textmining")
# AI 57 基准参数
TITLE_TOP = 200
SUMMARY_TOP = 1000
TRAIN_MONTHS = 12   # AI 57：样本内 12 个月
TEST_MONTHS = 12
LOOKBACK_MONTHS = 3  # 因子回溯 1 季度

log = logging.getLogger("fadt")

# FADT 样本量约为预告场景 2-9 倍，XGBoost 网格减半控制训练时长
# （lr 3→2 / subsample 4→2，保留 depth 3/5 覆盖复杂度）
FADT_XGB_GRID = [
    {"learning_rate": lr, "max_depth": md, "subsample": ss}
    for lr in (0.05, 0.075)
    for md in (3, 5)
    for ss in (0.85, 0.95)
]


def train_fadt_xgb(X, y, groups=None, seed=42):
    import xgboost as xgb
    best, best_auc = None, -1.0
    for p in FADT_XGB_GRID:
        m = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3,
            learning_rate=p["learning_rate"], max_depth=p["max_depth"],
            subsample=p["subsample"], n_estimators=200,
            random_state=seed, n_jobs=1, eval_metric="mlogloss")
        a = _auc_ovr(m, X, y, groups=groups)
        if a > best_auc:
            best_auc, best = a, (p, m)
    return best[1], best_auc


def build_factor_from_pred(pred: pd.DataFrame, model_name: str,
                           pool: str) -> pd.DataFrame:
    """月末截面回溯 3 个月，个股全部样本 log-odds 差均值（无衰减，AI 57 口径）。"""
    pred = pred.copy()
    pred["event_date"] = pd.to_datetime(pred["event_date"]).dt.normalize()
    months = sorted(set(pd.Timestamp(d.year, d.month, 1) for d in pred["event_date"]))
    rows = []
    for m in months:
        lb = m - pd.DateOffset(months=LOOKBACK_MONTHS)
        sub = pred[(pred["event_date"] >= lb) & (pred["event_date"] <= m)]
        if sub.empty:
            continue
        # AI 57：回溯期内该股全部调整研报的文本得分直接求均值（无衰减）
        g = sub.groupby("code").agg(
            factor=("sue0", "mean"),
            n_report=("sue0", "size"),
            n_event=("event_date", "nunique")).reset_index()
        g["date"] = m + pd.offsets.MonthEnd(0)
        rows.append(g)
    f = pd.concat(rows, ignore_index=True).set_index(["date", "code"]).sort_index()
    f.to_parquet(OUT_DIR / f"fadt_factor_{model_name}_{pool}.parquet",
                 compression="snappy")
    return f


def run(begin: int = 20190101, end: int = 20261231,
        model_name: str = "xgb", force: bool = False,
        pool: str = "zz1000") -> pd.DataFrame:
    sample_path = OUT_DIR / f"fadt_samples_{pool}.parquet"
    log.info("加载样本: %s", sample_path)
    samples = pd.read_parquet(sample_path)
    samples["event_date"] = pd.to_datetime(samples["event_date"]).dt.normalize()
    samples = samples[(samples["event_date"] >= pd.Timestamp(str(begin))) &
                      (samples["event_date"] <= pd.Timestamp(str(end)))]
    log.info("样本 %d 行 / %d 股票", len(samples), samples["code"].nunique())

    # 分词（带缓存，AI 57 词域参数）
    tok_path = OUT_DIR / f"fadt_samples_tokenized_{pool}.parquet"
    if tok_path.exists() and not force:
        tok = pd.read_parquet(tok_path)
    else:
        tok = samples.copy()
        tok["title_tok"] = tok["title"].map(tokenize_title)
        tok["summary_tok"] = tok["summary"].map(tokenize_summary)
        tok.to_parquet(tok_path, compression="snappy")
        log.info("分词完成，已缓存 %d 行", len(tok))
    if len(tok) == len(samples):
        samples["title_tok"] = tok["title_tok"].values
        samples["summary_tok"] = tok["summary_tok"].values
    else:
        samples = samples.merge(
            tok[["code", "event_date", "title_tok", "summary_tok"]],
            on=["code", "event_date"], how="left")

    # 滚动训练（AI 57：12+12）
    test_start = pd.Timestamp("20210101")
    all_pred: list[pd.DataFrame] = []
    round_no = 0
    while True:
        tr_end = test_start - pd.Timedelta(days=1)
        tr_start = tr_end - pd.DateOffset(months=TRAIN_MONTHS) + pd.Timedelta(days=1)
        te_end = test_start + pd.DateOffset(months=TEST_MONTHS) - pd.Timedelta(days=1)
        tr = samples[(samples["event_date"] >= tr_start) &
                     (samples["event_date"] <= tr_end)].copy()
        te = samples[(samples["event_date"] >= test_start) &
                     (samples["event_date"] <= te_end)].copy()
        if len(te) == 0:
            break
        round_no += 1
        log.info("轮 %d: 训练 [%s~%s] %d 行 / 测试 [%s~%s] %d 行",
                 round_no, tr_start.date(), tr_end.date(), len(tr),
                 test_start.date(), te_end.date(), len(te))
        if len(tr) < 100:
            log.warning("训练样本过少(%d)，跳过", len(tr))
            test_start = te_end + pd.Timedelta(days=1)
            continue

        # 样本内打标签 + 特征（AI 57 词域 200/1000）
        tr = tr.dropna(subset=["ar", "title_tok"]).copy()
        tr["label"] = make_labels(tr["ar"]) + 1  # -1/0/1 -> 0/1/2
        vec = SUEVectorizer(title_top=TITLE_TOP, summary_top=SUMMARY_TOP)
        vec.fit(tr["title_tok"], tr["summary_tok"])
        X_tr = vec.transform(tr["title_tok"], tr["summary_tok"])
        y_tr = tr["label"].values
        # FADT 研报即事件：event_date=研报发布日，按 (code, event_date) 分组
        groups_tr = tr[["code", "event_date"]].astype(str).agg("|".join, axis=1).values

        if model_name == "logit":
            model, auc = train_logit(X_tr, y_tr, groups=groups_tr)
            _, auc_leak = train_logit(X_tr, y_tr, groups=None)
        else:
            model, auc = train_fadt_xgb(X_tr, y_tr, groups=groups_tr)
            _, auc_leak = train_fadt_xgb(X_tr, y_tr, groups=None)
        log.info("  最佳模型 CV AUC(grouped)=%.4f | AUC(leak)=%.4f | Δ=%.4f",
                 auc, auc_leak, auc_leak - auc)
        joblib.dump(model, OUT_DIR / f"fadt_model_{model_name}_{pool}_r{round_no}.joblib")

        te = te.dropna(subset=["ar", "title_tok"]).copy()
        X_te = vec.transform(te["title_tok"], te["summary_tok"])
        te["sue0"] = _sue0_from_model(model, X_te)
        all_pred.append(te[["code", "event_date", "sue0"]])

        test_start = te_end + pd.Timedelta(days=1)
        if test_start > pd.Timestamp(str(end)):
            break

    if not all_pred:
        log.error("无测试样本")
        return pd.DataFrame()
    pred = pd.concat(all_pred, ignore_index=True)
    factor = build_factor_from_pred(pred, model_name, pool)
    log.info("因子面板: %d 行, 覆盖 %d 只",
             len(factor), factor.index.get_level_values("code").nunique())
    return factor


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--begin", type=int, default=20190101)
    ap.add_argument("--end", type=int, default=20261231)
    ap.add_argument("--model", default="xgb", choices=["xgb", "logit"])
    ap.add_argument("--pool", default="zz1000", choices=["hs300", "zz1000"])
    ap.add_argument("--force-tokenize", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(
                                      OUT_DIR / f"fadt_train_{args.model}_{args.pool}.log",
                                      encoding="utf-8")])
    run(args.begin, args.end, args.model, args.force_tokenize, pool=args.pool)
