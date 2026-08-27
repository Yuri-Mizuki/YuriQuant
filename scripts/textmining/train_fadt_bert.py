"""
BERT 版 FADT 训练（对齐华泰 AI 63 文本表示升级）
================================================

研报 AI 63 方案：
1. FinBERT 微调（我们用已微调中文金融 FinBERT 近似，见 encode_fadt_bert.py）
2. 研报 → CLS 层 768 维编码（替代词频向量）
3. XGBoost 二次训练：标签不变（T-1~T+1 三分类），特征 = CLS 编码
4. 因子 = log-odds(涨) - log-odds(跌)，月末回溯 3 个月个股均值（无衰减）

对比词频版（train_fadt.py）仅特征不同，滚动/标签/因子构建完全一致。

用法：
    python -m scripts.textmining.train_fadt_bert --task fadt --model xgb --pool zz1000
    python -m scripts.textmining.train_fadt_bert --task sue --model xgb --pool hs300

产出：
    reports/textmining/{task}_factor_bert_{model}_{pool}.parquet
    reports/textmining/{task}_train_bert_{model}_{pool}.log
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.textmining.train_fadt import (
    TRAIN_MONTHS, TEST_MONTHS, LOOKBACK_MONTHS,
    build_factor_from_pred,
)
from scripts.textmining.train_sue_txt import (
    _auc_ovr, _sue0_from_model, make_labels,
)

OUT_DIR = Path(r"E:\YuriQuant\reports\textmining")
log = logging.getLogger("fadt_bert")

# 与词频版同款网格（研报 AI 63 图表29 学习率 [0.025,0.05,0.075,0.1] × depth
# [3,5,7] × subsample [0.8,0.85,0.9,0.95]；我们沿用 train_fadt 精简网格）
# BERT 版 768 维特征训练显著慢于词频版（实测 732 行 × 80 次拟合 >30 分钟），
# 验证性实验缩小网格：SUE 用 2 组；FADT 样本大（2.7 万行）进一步缩至 1 组
# lr=0.05 × depth=3 × subsample=0.9（SUE 实测该组附近 AUC 最优）
BERT_XGB_GRID = [
    {"learning_rate": 0.05, "max_depth": 3, "subsample": 0.9},
    {"learning_rate": 0.075, "max_depth": 3, "subsample": 0.9},
]


def _load_cls(task: str, pool: str) -> pd.DataFrame:
    """加载 CLS 编码（row_idx, code, event_date, cls_0..cls_767）。"""
    p = OUT_DIR / f"{task}_cls_{pool}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} 不存在，先跑 encode_fadt_bert")
    return pd.read_parquet(p)


def train_bert_xgb(X, y, groups=None, seed=42):
    import xgboost as xgb
    best, best_auc = None, -1.0
    for p in BERT_XGB_GRID:
        m = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3,
            learning_rate=p["learning_rate"], max_depth=p["max_depth"],
            subsample=p["subsample"], n_estimators=200,
            random_state=seed, n_jobs=1, eval_metric="mlogloss")
        a = _auc_ovr(m, X, y, groups=groups)
        if a > best_auc:
            best_auc, best = a, (p, m)
    return best[1], best_auc


def run(task: str = "fadt", model_name: str = "xgb",
        pool: str = "zz1000", begin: int = 20190101,
        end: int = 20261231) -> pd.DataFrame:
    sample_path = OUT_DIR / (f"{task}_samples_{pool}.parquet" if task == "fadt"
                             else f"sue_txt_samples_{pool}.parquet")
    samples = pd.read_parquet(sample_path)
    samples["event_date"] = pd.to_datetime(samples["event_date"]).dt.normalize()
    samples = samples[(samples["event_date"] >= pd.Timestamp(str(begin))) &
                      (samples["event_date"] <= pd.Timestamp(str(end)))]
    samples = samples.reset_index(names="row_idx")
    log.info("样本 %d 行 / %d 只", len(samples), samples["code"].nunique())

    cls = _load_cls(task, pool)
    # 按原始行索引对齐（encode 保留 row_idx）
    samples = samples.merge(cls, on=["row_idx", "code", "event_date"], how="left")
    cls_cols = [c for c in samples.columns if c.startswith("cls_")]
    log.info("CLS 特征 %d 维, 覆盖 %d 行", len(cls_cols),
             samples[cls_cols[0]].notna().sum())

    # 滚动训练（12+12，与词频版一致）
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

        tr = tr.dropna(subset=["ar", cls_cols[0]]).copy()
        tr["label"] = make_labels(tr["ar"]) + 1
        X_tr = tr[cls_cols].values.astype(np.float32)
        y_tr = tr["label"].values
        groups_tr = tr[["code", "event_date"]].astype(str).agg("|".join, axis=1).values

        if model_name == "logit":
            from scripts.textmining.train_sue_txt import train_logit
            model, auc = train_logit(X_tr, y_tr, groups=groups_tr)
            _, auc_leak = train_logit(X_tr, y_tr, groups=None)
        else:
            model, auc = train_bert_xgb(X_tr, y_tr, groups=groups_tr)
            _, auc_leak = train_bert_xgb(X_tr, y_tr, groups=None)
        log.info("  最佳模型 CV AUC(grouped)=%.4f | AUC(leak)=%.4f | Δ=%.4f",
                 auc, auc_leak, auc_leak - auc)
        joblib.dump(model, OUT_DIR / f"{task}_bert_model_{model_name}_{pool}_r{round_no}.joblib")

        te = te.dropna(subset=["ar", cls_cols[0]]).copy()
        X_te = te[cls_cols].values.astype(np.float32)
        te["sue0"] = _sue0_from_model(model, X_te)
        all_pred.append(te[["code", "event_date", "sue0"]])

        test_start = te_end + pd.Timedelta(days=1)
        if test_start > pd.Timestamp(str(end)):
            break

    if not all_pred:
        log.error("无测试样本")
        return pd.DataFrame()
    pred = pd.concat(all_pred, ignore_index=True)
    # build_factor_from_pred 硬编码 fadt_ 前缀，且 model_name 传 "bert_xgb"
    # 会存成 fadt_factor_bert_xgb_{pool}.parquet；这里统一改为 task 前缀保存
    factor = build_factor_from_pred(pred, f"bert_{model_name}", pool)
    out = OUT_DIR / f"{task}_factor_bert_{model_name}_{pool}.parquet"
    factor.to_parquet(out, compression="snappy")
    # 清理 build_factor_from_pred 误存的 fadt_ 前缀文件（存在则覆盖为正确名）
    wrong = OUT_DIR / f"fadt_factor_bert_{model_name}_{pool}.parquet"
    if wrong.exists() and wrong.resolve() != out.resolve():
        wrong.unlink(missing_ok=True)
    log.info("因子面板: %d 行, 覆盖 %d 只 → %s",
             len(factor), factor.index.get_level_values("code").nunique(), out)
    return factor


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="fadt", choices=["sue", "fadt"])
    ap.add_argument("--model", default="xgb", choices=["xgb", "logit"])
    ap.add_argument("--pool", default="zz1000", choices=["hs300", "zz1000"])
    ap.add_argument("--begin", type=int, default=20190101)
    ap.add_argument("--end", type=int, default=20261231)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(
                                      OUT_DIR / f"{args.task}_train_bert_{args.model}_{args.pool}.log",
                                      encoding="utf-8")])
    run(args.task, args.model, args.pool, args.begin, args.end)
