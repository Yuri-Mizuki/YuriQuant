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
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.textmining.train_fadt import (
    SUMMARY_TOP,
    TEST_MONTHS,
    TITLE_TOP,
    TRAIN_MONTHS,
    build_factor_from_pred,
)
from scripts.textmining.train_sue_txt import (
    _auc_ovr,
    _sue0_from_model,
    make_labels,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cli_common import setup_logging  # noqa: E402
from scripts.textmining._paths import Out  # noqa: E402

OUT_DIR = Out("fadt")
log = setup_logging("fadt_bert")

# 与词频版同款网格（研报 AI 63 图表29 学习率 [0.025,0.05,0.075,0.1] × depth
# [3,5,7] × subsample [0.8,0.85,0.9,0.95]；我们沿用 train_fadt 精简网格）
# BERT 版 768 维特征训练显著慢于词频版（实测 732 行 × 80 次拟合 >30 分钟），
# 验证性实验缩小网格：SUE 用 2 组；FADT 样本大（2.7 万行）进一步缩至 1 组
# lr=0.05 × depth=3 × subsample=0.9（SUE 实测该组附近 AUC 最优）
BERT_XGB_GRID = [
    {"learning_rate": 0.05, "max_depth": 3, "subsample": 0.9},
    {"learning_rate": 0.075, "max_depth": 3, "subsample": 0.9},
]


def _load_cls(task: str, pool: str,
              filename: str | None = None) -> pd.DataFrame:
    """加载 CLS 编码（row_idx, code, event_date, cls_0..cls_767）。

    filename 可指向消融变体编码（分段/ pooler），列名仍为 cls_*。
    """
    p = OUT_DIR / (filename or f"{task}_cls_{pool}.parquet")
    if not p.exists():
        raise FileNotFoundError(f"{p} 不存在，先跑对应 encode 脚本")
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
        end: int = 20261231,
        cls_filename: str | None = None,
        variant: str | None = None,
        concat_wordfreq: bool = False) -> pd.DataFrame:
    """训练 + 因子构建。

    消融扩展（AI 63 五组扩展测试的本地复现，2026-09-09）：
    - cls_filename: 编码文件名（默认 {task}_cls_{pool}.parquet），可指向
      分段编码 fadt_cls_seg_{pool}.parquet 或 pooler 编码 fadt_cls_pooler_{pool}.parquet
    - variant: 变体标签，决定输出因子名 {task}_factor_{variant}_{pool}.parquet
      （默认 bert_{model_name}，与历史产出命名一致）
    - concat_wordfreq: CLS 编码后拼接词频特征（AI 63 扩展测试 4：CLS+词频 concat）
    """
    label = variant or f"bert_{model_name}"
    sample_path = OUT_DIR / (f"{task}_samples_{pool}.parquet" if task == "fadt"
                             else f"sue_txt_samples_{pool}.parquet")
    samples = pd.read_parquet(sample_path)
    samples["event_date"] = pd.to_datetime(samples["event_date"]).dt.normalize()
    samples = samples[(samples["event_date"] >= pd.Timestamp(str(begin))) &
                      (samples["event_date"] <= pd.Timestamp(str(end)))]
    samples = samples.reset_index(names="row_idx")
    log.info("样本 %d 行 / %d 只", len(samples), samples["code"].nunique())

    cls = _load_cls(task, pool, filename=cls_filename)
    # 按原始行索引对齐（encode 保留 row_idx）
    samples = samples.merge(cls, on=["row_idx", "code", "event_date"], how="left")
    cls_cols = [c for c in samples.columns if c.startswith("cls_")]
    log.info("CLS 特征 %d 维, 覆盖 %d 行", len(cls_cols),
             samples[cls_cols[0]].notna().sum())

    if concat_wordfreq:
        # 词域特征（AI 57 词域 200/1000，与 train_fadt 一致；带分词缓存）
        from scripts.textmining.train_sue_txt import (
            SUEVectorizer,
            tokenize_summary,
            tokenize_title,
        )
        tok_path = OUT_DIR / f"fadt_samples_tokenized_{pool}.parquet"
        if tok_path.exists():
            tok = pd.read_parquet(tok_path)
            samples = samples.drop(columns=["title_tok", "summary_tok"],
                                   errors="ignore").merge(
                tok[["row_idx", "title_tok", "summary_tok"]]
                if "row_idx" in tok.columns
                else tok[["code", "event_date", "title_tok", "summary_tok"]],
                on=["row_idx"] if "row_idx" in tok.columns
                else ["code", "event_date"], how="left")
        else:
            samples["title_tok"] = samples["title"].map(tokenize_title)
            samples["summary_tok"] = samples["summary"].map(tokenize_summary)
            samples[["row_idx", "code", "event_date", "title_tok",
                     "summary_tok"]].to_parquet(tok_path, compression="snappy")
        from scipy.sparse import csr_matrix
        from scipy.sparse import hstack as sp_hstack

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

        if concat_wordfreq:
            vec = SUEVectorizer(title_top=TITLE_TOP, summary_top=SUMMARY_TOP)
            vec.fit(tr["title_tok"], tr["summary_tok"])
            wf_tr = vec.transform(tr["title_tok"], tr["summary_tok"])
            X_tr = sp_hstack([csr_matrix(X_tr), wf_tr]).tocsr()

        if model_name == "logit":
            from scripts.textmining.train_sue_txt import train_logit
            model, auc = train_logit(X_tr, y_tr, groups=groups_tr)
            _, auc_leak = train_logit(X_tr, y_tr, groups=None)
        else:
            model, auc = train_bert_xgb(X_tr, y_tr, groups=groups_tr)
            _, auc_leak = train_bert_xgb(X_tr, y_tr, groups=None)
        log.info("  最佳模型 CV AUC(grouped)=%.4f | AUC(leak)=%.4f | Δ=%.4f",
                 auc, auc_leak, auc_leak - auc)
        joblib.dump(model, OUT_DIR / f"{task}_{label}_model_{pool}_r{round_no}.joblib")

        te = te.dropna(subset=["ar", cls_cols[0]]).copy()
        X_te = te[cls_cols].values.astype(np.float32)
        if concat_wordfreq:
            X_te = sp_hstack([csr_matrix(X_te),
                              vec.transform(te["title_tok"],
                                            te["summary_tok"])]).tocsr()
        te["sue0"] = _sue0_from_model(model, X_te)
        all_pred.append(te[["code", "event_date", "sue0"]])

        test_start = te_end + pd.Timedelta(days=1)
        if test_start > pd.Timestamp(str(end)):
            break

    if not all_pred:
        log.error("无测试样本")
        return pd.DataFrame()
    pred = pd.concat(all_pred, ignore_index=True)
    # build_factor_from_pred 硬编码 fadt_ 前缀；sue 任务需改回 task 前缀
    factor = build_factor_from_pred(pred, label, pool)
    out = OUT_DIR / f"{task}_factor_{label}_{pool}.parquet"
    factor.to_parquet(out, compression="snappy")
    # 清理 build_factor_from_pred 误存的 fadt_ 前缀文件（存在则覆盖为正确名）
    wrong = OUT_DIR / f"fadt_factor_{label}_{pool}.parquet"
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
    ap.add_argument("--cls-file", default=None,
                    help="编码文件名（默认 {task}_cls_{pool}.parquet），"
                         "可指向分段/pooler 消融编码")
    ap.add_argument("--variant", default=None,
                    help="变体标签（决定输出因子名），默认 bert_{model}")
    ap.add_argument("--concat-wordfreq", action="store_true",
                    help="CLS 编码拼接词频特征（AI 63 扩展测试 4）")
    args = ap.parse_args()

    log_tag = args.variant or f"bert_{args.model}"
    setup_logging("fadt_bert",
                  file=OUT_DIR / f"{args.task}_train_{log_tag}_{args.pool}.log")
    run(args.task, args.model, args.pool, args.begin, args.end,
        cls_filename=args.cls_file, variant=args.variant,
        concat_wordfreq=args.concat_wordfreq)
