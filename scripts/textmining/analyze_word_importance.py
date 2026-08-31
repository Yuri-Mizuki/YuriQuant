"""
词重要性分析（对齐华泰 AI 51/AI 57 模型可解释性）
====================================================

研报方法（AI 51 §模型可解释性分析/单词重要性，参考 Yano 2012）：
    词重要性 I_w(x) = (β^x_上涨 - β^x_下跌) × (1/N) Σ_i c_ix
- β^x_上涨 / β^x_下跌：**最后一个训练期** OvR 逻辑回归中词 x 在"上涨"
  与"下跌"二分类模型的回归系数（系数差反映词对结果的影响方向与力度）；
- c_ix：第 i 条样本中词 x 的**对数词频**（log(1+count)，与训练特征一致）；
- 选取最后一个训练期作为示例（研报口径）。

输出（对齐研报图表 25/26/27/28）：
1. Top15 正向词 + Top15 负向词（按单词重要性排序，含 log 词频均值/β差/重要性）
2. Top15 系数差最大的正向/负向词
3. Top30 词频最高的关键词

用法：
    python -m scripts.textmining.analyze_word_importance --pool zz1000 --task sue
    python -m scripts.textmining.analyze_word_importance --pool zz1000 --task fadt
    python -m scripts.textmining.analyze_word_importance --pool hs300 --round 6
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.textmining.train_sue_txt import SUEVectorizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cli_common import setup_logging  # noqa: E402

OUT_DIR = ROOT / "reports" / "textmining"
log = setup_logging("word_importance")

# 任务参数：sue(AI51: 词域100/500, 训练窗24月) vs fadt(AI57: 200/1000, 12月)
TASK_CFG = {
    "sue": {"prefix": "sue_txt", "title_top": 100, "summary_top": 500,
            "train_months": 24},
    "fadt": {"prefix": "fadt", "title_top": 200, "summary_top": 1000,
             "train_months": 12},
}


def _find_last_round(pool: str, model_name: str, prefix: str) -> tuple[int, Path]:
    files = glob.glob(str(OUT_DIR / f"{prefix}_model_{model_name}_{pool}_r*.joblib"))
    rounds = []
    for f in files:
        m = re.search(r"_r(\d+)\.joblib$", f)
        if m:
            rounds.append((int(m.group(1)), Path(f)))
    if not rounds:
        raise FileNotFoundError(f"{pool} 无 {prefix} {model_name} 模型，先跑 train")
    return max(rounds, key=lambda x: x[0])


def _train_window(test_start: pd.Timestamp, train_months: int
                  ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """滚动训练窗口 = 测试窗口前 train_months 个月。"""
    tr_end = test_start - pd.Timedelta(days=1)
    tr_start = tr_end - pd.DateOffset(months=train_months) + pd.Timedelta(days=1)
    return tr_start, tr_end


def analyze(pool: str = "hs300", model_name: str = "logit",
            round_no: int | None = None, task: str = "sue") -> pd.DataFrame:
    """输出词重要性表（词 / log词频均值 / β差 / 重要性）。"""
    cfg = TASK_CFG[task]
    prefix, t_top, s_top, tr_months = (cfg["prefix"], cfg["title_top"],
                                       cfg["summary_top"], cfg["train_months"])
    r_no, model_path = _find_last_round(pool, model_name, prefix)
    if round_no is not None:
        r_no = round_no
        model_path = OUT_DIR / f"{prefix}_model_{model_name}_{pool}_r{r_no}.joblib"
        assert model_path.exists(), f"{model_path} 不存在"
    model = joblib.load(model_path)
    log.info("模型: %s (轮 %d)", model_path.name, r_no)

    # 加载分词后样本，取最后一个训练期
    tok = pd.read_parquet(OUT_DIR / f"{prefix}_samples_tokenized_{pool}.parquet")
    tok["event_date"] = pd.to_datetime(tok["event_date"]).dt.normalize()
    # 轮 r 的测试窗口起点：2021-01-01 + (r-1)*12 月（train_sue_txt/fadt 固定口径）
    test_start = pd.Timestamp("20210101") + pd.DateOffset(months=(r_no - 1) * 12)
    tr_start, tr_end = _train_window(test_start, tr_months)
    tr = tok[(tok["event_date"] >= tr_start) & (tok["event_date"] <= tr_end)].copy()
    tr = tr.dropna(subset=["title_tok"])
    log.info("训练期 [%s ~ %s] 样本 %d 行", tr_start.date(), tr_end.date(), len(tr))
    if len(tr) < 50:
        raise RuntimeError("训练期样本过少")

    # 重训 vectorizer（与训练同参数，CountVectorizer 词典顺序确定性一致）
    vec = SUEVectorizer(title_top=t_top, summary_top=s_top)
    vec.fit(tr["title_tok"], tr["summary_tok"])
    X = vec.transform(tr["title_tok"], tr["summary_tok"])  # log(1+x) 词频矩阵
    avg_logfreq = X.mean(axis=0)  # (1/N) Σ c_ix

    # OvR 系数：coef_ 形状 (n_classes, n_features)，类别顺序 = model.classes_
    cls = list(model.classes_)
    b_up = model.coef_[cls.index(2)]
    b_dn = model.coef_[cls.index(0)]
    beta_diff = b_up - b_dn  # β(上涨) - β(下跌)

    words = np.concatenate([vec.title_vec.get_feature_names_out(),
                            vec.summary_vec.get_feature_names_out()])
    assert len(words) == len(beta_diff), (len(words), len(beta_diff))

    imp = pd.DataFrame({
        "word": words,
        "avg_logfreq": avg_logfreq,
        "beta_diff": beta_diff,
        "importance": beta_diff * avg_logfreq,
    })
    imp = imp[imp["word"] != ""].sort_values("importance", ascending=False)
    return imp


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float, np.floating)) else str(x)


def main(pool: str = "hs300", round_no: int | None = None,
         task: str = "sue") -> None:
    imp = analyze(pool, "logit", round_no, task=task)
    out_lines = []
    out_lines.append(f"===== 单词重要性分析（task={task}，pool={pool}）=====")
    out_lines.append(f"模型: 最后训练期 OvR Logistic | 词域: "
                     f"标题{TASK_CFG[task]['title_top']}+摘要{TASK_CFG[task]['summary_top']}")
    out_lines.append("")

    def block(title: str, df: pd.DataFrame) -> None:
        out_lines.append(f"--- {title} ---")
        out_lines.append(f"{'词':<8}{'log词频均值':>12}{'β差(涨-跌)':>14}{'重要性':>12}")
        for _, r in df.iterrows():
            out_lines.append(f"{r['word']:<10}{_fmt(r['avg_logfreq']):>12}"
                             f"{_fmt(r['beta_diff']):>14}{_fmt(r['importance']):>12}")
        out_lines.append("")

    block("Top15 正向词（按重要性）", imp.head(15))
    block("Top15 负向词（按重要性）", imp.tail(15).iloc[::-1])
    block("Top15 系数差最大的正向词", imp.sort_values("beta_diff", ascending=False).head(15))
    block("Top15 系数差最大的负向词", imp.sort_values("beta_diff").head(15))
    block("Top30 词频最高的词", imp.sort_values("avg_logfreq", ascending=False).head(30))

    text = "\n".join(out_lines)
    prefix = TASK_CFG[task]["prefix"]
    out = OUT_DIR / f"{prefix}_word_importance_{pool}.txt"
    out.write_text(text, encoding="utf-8")
    print(text)
    imp.to_parquet(OUT_DIR / f"{prefix}_word_importance_{pool}.parquet",
                   compression="snappy")
    log.info("已输出 %s", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="hs300", choices=["hs300", "zz1000"])
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--task", default="sue", choices=["sue", "fadt"])
    args = ap.parse_args()
    main(args.pool, args.round, args.task)
