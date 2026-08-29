"""
SUE.txt 训练与因子构建（对齐华泰 AI 51 文本PEAD）
===================================================

流程（严格对齐研报《人工智能51：文本PEAD选股策略》20220107）：
1. 分词：Jieba 词性标注，保留 普通名词(n)/专有名词(nz)/动词(v)/副动词(vd)/
   动名词(vn)/形容词(a)/副词(d)。
2. 特征：CountVectorizer——样本内标题出现频率最高 100 词 + 摘要最高 500 词，
   拼接两个词频矩阵 → log(1+x)。样本外用样本内词典（防前视）。
3. 标签：样本内 AR 三等分（前30%上涨 y=1 / 30-70%震荡 y=0 / 后30%下跌 y=-1）。
4. 模型：Logistic（弹性网络 l1_ratio=0.5，λ 网格搜索 5 折 CV 按 AUC）+
   XGBoost（learning_rate/max_depth/subsample 网格搜索 5 折 CV 按 AUC）。
5. 滚动：样本内过去 24 个月，样本外未来 12 个月，逐轮推进。
6. 因子：log-odds(上涨)-log-odds(下跌) = SUE0；月末截面回溯过去 3 个月事件，
   按 0.95^(T-t) 指数衰减（T=月末截面，t=事件日）；同股同事件多篇研报取均值。

用法：
    python -m scripts.textmining.train_sue_txt --model xgb
    python -m scripts.textmining.train_sue_txt --model logit

产出：
    reports/textmining/sue_txt_factor_{model}.parquet  （因子面板 date×code）
    reports/textmining/sue_txt_train_{model}.log
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

from scripts.textmining.build_sue_txt_samples import _load_daily, _to_naive

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "textmining"
SAMPLE_PATH = OUT_DIR / "sue_txt_samples.parquet"

# AI 51：保留词性（普通名词/专有名词/动词/副动词/动名词/形容词/副词）
KEEP_POS = {"n", "nz", "v", "vd", "vn", "a", "d"}
# 标题 top100 词 + 摘要 top500 词
TITLE_TOP = 100
SUMMARY_TOP = 500
# 滚动窗口：样本内 24 个月 / 样本外 12 个月
TRAIN_MONTHS = 24
TEST_MONTHS = 12
# 因子：回溯 3 个月（1 季度）/ 衰减系数 0.95
LOOKBACK_MONTHS = 3
DECAY = 0.95

log = logging.getLogger("sue_txt")


# ---------- 分词 ----------
def _jieba_pos_filter(text: str) -> list[str]:
    import jieba.posseg as pseg
    return [w for w, p in pseg.cut(text) if p in KEEP_POS and len(w) > 1]


def tokenize_title(text: str) -> str:
    return " ".join(_jieba_pos_filter(str(text)))


def tokenize_summary(text: str) -> str:
    return " ".join(_jieba_pos_filter(str(text)))


# ---------- 特征 ----------
class SUEVectorizer:
    """标题 top_N + 摘要 top_M 词频 → log(1+x)，样本内 fit / 样本外 transform。

    AI 51 默认标题 top100 + 摘要 top500；AI 57（FADT）用标题 top200 + 摘要
    top1000。参数化后两场景共用本类。
    """

    def __init__(self, title_top: int = TITLE_TOP, summary_top: int = SUMMARY_TOP):
        self.title_vec: CountVectorizer | None = None
        self.summary_vec: CountVectorizer | None = None
        self.title_top = title_top
        self.summary_top = summary_top

    def fit(self, titles: pd.Series, summaries: pd.Series) -> None:
        self.title_vec = CountVectorizer(tokenizer=lambda s: s.split(),
                                         lowercase=False, max_features=self.title_top)
        self.summary_vec = CountVectorizer(tokenizer=lambda s: s.split(),
                                           lowercase=False, max_features=self.summary_top)
        self.title_vec.fit(titles)
        self.summary_vec.fit(summaries)

    def transform(self, titles: pd.Series, summaries: pd.Series) -> np.ndarray:
        Xt = self.title_vec.transform(titles).toarray()
        Xs = self.summary_vec.transform(summaries).toarray()
        return np.log1p(np.hstack([Xt, Xs]))


# ---------- 标签 ----------
def make_labels(ar: pd.Series) -> np.ndarray:
    q30, q70 = ar.quantile([0.3, 0.7])
    return ar.apply(lambda x: 1 if x >= q70 else (-1 if x <= q30 else 0)).values


# 标签 horizon：默认 "t01"（T-1~T+1 两日 AR，对齐 AI 51）；"t15" 用 T+1~T+5
# 累计超额做标签——标签与信号效应窗口对齐（ablation 发现 ar 在 T+1 最强，
# 用 T+1~T+5 标签可能让模型学到更有泛化力的模式）
LABEL_HORIZON_DEFAULT = "t01"
BENCH = "000905.SH"


def _compute_fwd_excess(samples: pd.DataFrame, daily: pd.DataFrame,
                        k_max: int = 5) -> pd.Series:
    """计算每个事件的 T+1~T+k_max 累计超额收益（相对中证500）。

    用于 label_horizon="t15" 时的标签：替代 T-1~T+1 两日 AR。
    """
    by_code = {}
    for code, sub in daily.reset_index().groupby("code"):
        sub = sub.copy()
        sub["date"] = _to_naive(sub["date"])
        by_code[code] = sub.sort_values("date")[["date", "close"]].reset_index(drop=True)

    def _close_at(code: str, d: pd.Timestamp, shift: int) -> float | None:
        tbl = by_code.get(code)
        if tbl is None or tbl.empty:
            return None
        dates = tbl["date"].values
        pos = int(pd.Index(dates).searchsorted(pd.Timestamp(d).to_datetime64(),
                                               side="right")) - 1
        if pos < 0:
            return None
        j = pos + shift
        if 0 <= j < len(dates):
            return float(tbl["close"].iloc[j])
        return None

    vals = []
    for _, r in samples.iterrows():
        code = r["code"]
        d = pd.Timestamp(r["event_date"])
        p0 = _close_at(code, d, 0)
        b0 = _close_at(BENCH, d, 0)
        pk = _close_at(code, d, k_max)
        bk = _close_at(BENCH, d, k_max)
        if None in (p0, b0, pk, bk):
            vals.append(np.nan)
        else:
            vals.append((pk / p0 - 1) - (bk / b0 - 1))
    return pd.Series(vals, index=samples.index, name="fwd_excess")


# ---------- 模型 ----------
# AI 51 用弹性网络（elasticnet，saga）。实测 saga 在 600 维×3000 样本下单折
# ~11s，8λ×5 折×6 轮×2 池不可行；改用 liblinear + L1（弹性网络在 l1_ratio=1
# 的特例，同属"正则化逻辑回归"），速度快 50 倍，稀疏系数对词重要性更可解释。
LOGIT_LAMBDA = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 1e-2]
XGB_PARAM_GRID = [
    {"learning_rate": lr, "max_depth": md, "subsample": ss}
    for lr in (0.025, 0.05, 0.075)
    for md in (3, 5)
    for ss in (0.8, 0.85, 0.9, 0.95)
]


def _auc_ovr(model, X, y, groups=None, cv=5, seed=42):
    """5 折 CV 平均 AUC（OvR 多分类：one-vs-rest 平均）。

    事件级 grouped CV（修正行级泄漏）：
    - groups 不为 None 时用 StratifiedGroupKFold，保证同一 (code, event_date)
      的多篇研报同属一折——同一事件的多篇研报文本高度相似（同一公告的
      不同机构点评），若分散到训练/验证两折会造成 AUC 虚高（行级泄漏）。
    - groups 为 None 时退回 StratifiedKFold（保留旧行为用于对比）。
    """
    if groups is not None:
        # 去重 group id 为连续整数（StratifiedGroupKFold 要求）
        _, gid = np.unique(groups, return_inverse=True)
        kf = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
        splitter = kf.split(X, y, gid)
    else:
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        splitter = kf.split(X, y)
    aucs = []
    for tr, va in splitter:
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[va])
        # OvR AUC：每类 one-vs-rest
        for c in np.unique(y):
            yc = (y[va] == c).astype(int)
            pc = p[:, list(model.classes_).index(c)]
            try:
                aucs.append(roc_auc_score(yc, pc))
            except ValueError:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


def train_logit(X, y, groups=None, seed=42):
    best, best_auc = None, -1.0
    for lam in LOGIT_LAMBDA:
        # AI 51：OvR 多分类（"做二元逻辑回归，得到第 K 类的分类模型，其他类别同理"）
        # max_iter=200 + tol=1e-2：liblinear 小 λ（大 C）下收敛极慢（实测可卡死数
        # 十分钟），收紧容差牺牲极小 AUC 换取可行性（CV AUC 影响 <0.005）。
        m = LogisticRegression(C=1.0 / lam, penalty="l1", solver="liblinear",
                               max_iter=200, tol=1e-2, random_state=seed,
                               multi_class="ovr")
        a = _auc_ovr(m, X, y, groups=groups)
        if a > best_auc:
            best_auc, best = a, (lam, m)
    return best[1], best_auc


def train_xgb(X, y, groups=None, seed=42):
    best, best_auc = None, -1.0
    for p in XGB_PARAM_GRID:
        m = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3,
            learning_rate=p["learning_rate"], max_depth=p["max_depth"],
            subsample=p["subsample"], n_estimators=300,
            random_state=seed, n_jobs=1, eval_metric="mlogloss")
        a = _auc_ovr(m, X, y, groups=groups)
        if a > best_auc:
            best_auc, best = a, (p, m)
    return best[1], best_auc


# ---------- 滚动训练 + 因子 ----------
def run(begin: int = 20190101, end: int = 20261231,
        model_name: str = "xgb", force: bool = False,
        pool: str = "hs300",
        label_horizon: str = LABEL_HORIZON_DEFAULT) -> pd.DataFrame:
    sample_path = OUT_DIR / f"sue_txt_samples_{pool}.parquet"
    log.info("加载样本: %s", sample_path)
    samples = pd.read_parquet(sample_path)
    samples["event_date"] = pd.to_datetime(samples["event_date"]).dt.normalize()
    samples["report_date"] = pd.to_datetime(samples["report_date"]).dt.normalize()
    samples = samples[(samples["event_date"] >= pd.Timestamp(str(begin))) &
                      (samples["event_date"] <= pd.Timestamp(str(end)))]
    log.info("样本 %d 行 / %d 事件", len(samples),
             samples.groupby(["code", "event_date"]).ngroups)

    # 分词（带缓存）
    tok_path = OUT_DIR / f"sue_txt_samples_tokenized_{pool}.parquet"
    if tok_path.exists() and not force:
        tok = pd.read_parquet(tok_path)
    else:
        tok = samples.copy()
        tok["title_tok"] = tok["title"].map(tokenize_title)
        tok["summary_tok"] = tok["summary"].map(tokenize_summary)
        tok.to_parquet(tok_path, compression="snappy")
        log.info("分词完成，已缓存 %d 行", len(tok))
    # 对齐（分词可能因列差异错位，按行号取）
    if len(tok) == len(samples):
        samples["title_tok"] = tok["title_tok"].values
        samples["summary_tok"] = tok["summary_tok"].values
    else:
        samples = samples.merge(
            tok[["code", "event_date", "report_date", "title_tok", "summary_tok"]],
            on=["code", "event_date", "report_date"], how="left")

    # 滚动训练
    # 标签 horizon 对齐：t01 = T-1~T+1 两日 AR（AI 51 原版）；t15 = T+1~T+5
    # 累计超额（与信号效应窗口对齐）。ablation 发现 ar 在 T+1 最强，用 t15
    # 标签可能让模型学到更有泛化力的模式（grouped CV 之前≈随机）。
    daily_cache = None
    if label_horizon == "t15":
        daily_cache = _load_daily(None, [], begin, end)
        log.info("标签 horizon=t15（T+1~T+5 累计超额），日线已加载")

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
        if len(tr) < 50:
            log.warning("训练样本过少(%d)，跳过", len(tr))
            test_start = te_end + pd.Timedelta(days=1)
            continue

        # 样本内打标签 + 特征
        tr = tr.dropna(subset=["ar", "title_tok"]).copy()
        if label_horizon == "t15":
            tr["fwd_excess"] = _compute_fwd_excess(tr, daily_cache, k_max=5)
            tr = tr.dropna(subset=["fwd_excess"]).copy()
            tr["label"] = make_labels(tr["fwd_excess"]) + 1
        else:
            tr["label"] = make_labels(tr["ar"]) + 1  # -1/0/1 -> 0/1/2（XGBoost 类别须 0 起）
        vec = SUEVectorizer()
        vec.fit(tr["title_tok"], tr["summary_tok"])
        X_tr = vec.transform(tr["title_tok"], tr["summary_tok"])
        y_tr = tr["label"].values
        # 事件级分组键：同一 (code, event_date) 的多篇研报须同属一折，
        # 否则文本高度相似的研究会泄漏到验证集造成 AUC 虚高。
        groups_tr = tr[["code", "event_date"]].astype(str).agg("|".join, axis=1).values

        # 训练（grouped CV 防泄漏）
        if model_name == "logit":
            model, auc = train_logit(X_tr, y_tr, groups=groups_tr)
        else:
            model, auc = train_xgb(X_tr, y_tr, groups=groups_tr)
        # 对比基线：行级泄漏的旧 CV（诊断用，不参与选模）
        if model_name == "logit":
            _, auc_leak = train_logit(X_tr, y_tr, groups=None)
        else:
            _, auc_leak = train_xgb(X_tr, y_tr, groups=None)
        log.info("  最佳模型 CV AUC(grouped)=%.4f | AUC(leak-baseline)=%.4f | Δ=%.4f",
                 auc, auc_leak, auc_leak - auc)
        joblib.dump(model, OUT_DIR / f"sue_txt_model_{model_name}_{pool}_r{round_no}.joblib")
        # 注意：SUEVectorizer 内含 lambda tokenizer 不可 pickle，不落盘；
        # 因子构建（build_factor_from_pred）只依赖预测值 sue0，无需 vec。

        # 测试集预测（样本外）
        te = te.dropna(subset=["ar", "title_tok"]).copy()
        X_te = vec.transform(te["title_tok"], te["summary_tok"])
        te["sue0"] = _sue0_from_model(model, X_te)
        all_pred.append(te[["code", "event_date", "report_date", "sue0"]])

        test_start = te_end + pd.Timedelta(days=1)
        if test_start > pd.Timestamp(str(end)):
            break

    if not all_pred:
        log.error("无测试样本，检查数据区间")
        return pd.DataFrame()
    pred = pd.concat(all_pred, ignore_index=True)

    # 因子截面（月末回溯 3 个月 + 衰减）——用逐轮预测已含滚动模型，
    # 这里直接用 pred 构建因子；衰减基准 T 取当月最后交易日（研报口径）
    calendar = _load_trading_calendar()
    factor = build_factor_from_pred(pred, model_name, pool, calendar=calendar)
    log.info("因子面板: %d 行, 覆盖 %d 只", len(factor), factor.index.get_level_values("code").nunique())
    return factor


def _load_trading_calendar() -> pd.DatetimeIndex:
    """从 daily_{pool}.parquet 缓存提取全市场交易日序列（每月最后交易日基准）。"""
    try:
        from config import Config
        p = Path(str(Config.cache()["root"]).replace("//", "/")) / "daily_hs300.parquet"
        if p.exists():
            dates = pd.read_parquet(p, columns=[])  # 仅读索引
            return pd.DatetimeIndex(dates.index.get_level_values("date"))
    except Exception:  # noqa: BLE001
        pass
    return pd.DatetimeIndex([])


def _month_last_trading_day(dates: pd.DatetimeIndex) -> dict:
    """每月最后一个交易日：{Period('M') -> Timestamp}。用于衰减基准 T。"""
    d = pd.DatetimeIndex(dates).normalize().unique()
    s = pd.Series(d, index=d.to_period("M")).groupby(level=0).max()
    return {p: t for p, t in s.items()}


def _sue0_from_model(model, X: np.ndarray) -> np.ndarray:
    """SUE0 = log-odds(上涨) - log-odds(下跌)。

    - Logistic（OvR，对齐 AI 51）：每类二分类器的 decision_function 即
      log(p_c/(1-p_c))，直接相减（比从归一化 predict_proba 更贴近研报公式）。
    - XGBoost（multi:softprob）：用 predict_proba 的类别概率算 log-odds。
    """
    if isinstance(model, LogisticRegression):
        d = model.decision_function(X)  # (n, n_classes)，第 c 列 = 类 c vs 其他 的 log-odds
        cls = list(model.classes_)
        return d[:, cls.index(2)] - d[:, cls.index(0)]  # label 0/1/2: 2=上涨, 0=下跌
    probs = model.predict_proba(X)
    cls = list(model.classes_)
    i_up, i_dn = cls.index(2), cls.index(0)
    return (np.log(probs[:, i_up] / (1 - probs[:, i_up] + 1e-12)) -
            np.log(probs[:, i_dn] / (1 - probs[:, i_dn] + 1e-12)))


def build_factor_from_pred(pred: pd.DataFrame, model_name: str, pool: str = "hs300",
                           calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """用逐轮预测的 sue0 构建月末截面因子（回溯 3 个月 + 0.95 指数衰减）。

    对齐 AI 51（修正版）：
    - 衰减基准 T = **当月最后一个交易日**（研报："每个自然月最后一个交易日
      作为截面期"），days = (T - 事件发布日).days，因子 = SUE0 * 0.95^days；
    - 聚合粒度 = **事件级**：同一 (code, event_date) 的多篇研报先取 SUE.txt
      均值（研报："同一支股票同一业绩预告对应多篇研报，取多篇研报的
      SUE.txt 均值"）；同股回溯期内出现多次预告（少见）时取**最新事件**
      （衰减最小、信息最新），并记录 n_event 供诊断。
    """
    pred = pred.copy()
    pred["event_date"] = pd.to_datetime(pred["event_date"]).dt.normalize()
    months = sorted(set(pd.Timestamp(d.year, d.month, 1) for d in pred["event_date"]))
    if calendar is not None and len(calendar):
        mlt = _month_last_trading_day(calendar)  # {Period('M'): 最后交易日}
    else:
        mlt = {}
    rows = []
    for m in months:
        lb = m - pd.DateOffset(months=LOOKBACK_MONTHS)
        sub = pred[(pred["event_date"] >= lb) & (pred["event_date"] <= m)]
        if sub.empty:
            continue
        # 衰减基准 T：当月最后交易日（无交易日历时退化为月末）
        T = mlt.get(m.to_period("M"), m + pd.offsets.MonthEnd(0))
        days = (pd.Timestamp(T) - sub["event_date"]).dt.days.clip(lower=0)
        sub = sub.assign(suew=sub["sue0"] * DECAY ** days)
        # 事件级聚合：同 (code, event_date) 多研报取均值
        ev = sub.groupby(["code", "event_date"], as_index=False).agg(
            suew=("suew", "mean"), n_report=("sue0", "size"))
        # 同股多事件：取最新事件（衰减最小）
        ev = ev.sort_values("event_date")
        latest = ev.groupby("code", as_index=False).last()
        n_event = ev.groupby("code")["event_date"].nunique()
        latest["n_event"] = latest["code"].map(n_event)
        latest["date"] = m + pd.offsets.MonthEnd(0)
        rows.append(latest.rename(columns={"suew": "factor"})[
            ["date", "code", "factor", "n_report", "n_event"]])
    f = pd.concat(rows, ignore_index=True).set_index(["date", "code"]).sort_index()
    f.to_parquet(OUT_DIR / f"sue_txt_factor_{model_name}_{pool}.parquet",
                 compression="snappy")
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--begin", type=int, default=20190101)
    ap.add_argument("--end", type=int, default=20261231)
    ap.add_argument("--model", default="xgb", choices=["xgb", "logit"])
    ap.add_argument("--pool", default="hs300", choices=["hs300", "zz1000"])
    ap.add_argument("--force-tokenize", action="store_true")
    ap.add_argument("--label-horizon", default=LABEL_HORIZON_DEFAULT,
                    choices=["t01", "t15"],
                    help="t01=T-1~T+1 两日AR（AI51原版）；t15=T+1~T+5 累计超额")
    args = ap.parse_args()

    log_suffix = f"_{args.label_horizon}" if args.label_horizon != "t01" else ""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(
                                      OUT_DIR / f"sue_txt_train_{args.model}_{args.pool}{log_suffix}.log",
                                      encoding="utf-8")])
    run(args.begin, args.end, args.model, args.force_tokenize,
        pool=args.pool, label_horizon=args.label_horizon)
