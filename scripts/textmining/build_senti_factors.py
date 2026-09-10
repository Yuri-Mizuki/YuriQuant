"""
AI 41 研报情感四因子构建（senti / senti_adj / report_num / report_score）
========================================================================

对齐华泰 AI 41《基于BERT的分析师研报情感因子》(20210118)：

1. **senti**：报告情感得分 = P(pos) - P(neg)（研报为二分类 P(pos)-0.5；
   我们的三分类取正负边际差，同为以 0 为中点）。同股同日多篇研报取均值
   （研报步骤3）；月末交易日 T 回溯 **90 个自然日**，线性衰减加权
   （w = (90-lag)/90，越近权重越大）。研报为"加权和"，我们对得分型因子
   用加权均值（保持 [-1,1] 量纲，截面排序几乎不变），数量型用加权和。
2. **senti_adj**：负面得分（<0）×3 后同口径聚合——研报发现分析师正面
   评价约占 3/4，加权负面后才有相对评级/数量的增量信息。
3. **report_num**：研报数量因子。同股同日研报数，90 日线性衰减加权和。
4. **report_score**：研报评级因子。评级映射 买入=7/增持=5/中性=3/减持=2/
   卖出=1（朝阳永续 SCORE_ID 口径，AI 41 图表11），同日均值后 90 日衰减加权。
5. **残差因子** senti_res / senti_adj_res：月度截面去极值(中位数±5×MAD)、
   标准化后对 [report_score, report_num] 回归取残差（AI 41：senti 大部分
   信息可被评级/数量解释，senti_adj 残差才是增量）。

用法：
    python -m scripts.textmining.build_senti_factors --pool zz1000

产出（reports/textmining/senti/features/）：
    senti_factor_{name}_{pool}.parquet   name ∈ senti/senti_adj/report_num/
                                         report_score/senti_res/senti_adj_res
    senti_factors_{pool}.parquet         六因子宽表
    senti_factor_corr_{pool}.txt         因子相关矩阵（AI 41 图表17 口径）
    senti_sentence_check_{pool}.txt      报告级 vs 逐句打分一致性抽样验证
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cli_common import setup_logging  # noqa: E402
from scripts.textmining._paths import Out  # noqa: E402
from scripts.textmining.build_sue_txt_samples import _load_daily  # noqa: E402

OUT_DIR = Out("senti")
log = setup_logging("senti_factors")

WINDOW_DAYS = 90  # AI 41：90 个自然日滚动窗口线性衰减

RATING_MAP = {"买入": 7, "增持": 5, "中性": 3, "持有": 3, "减持": 2, "卖出": 1}


def pool_daily(pool: str) -> pd.DataFrame:
    return _load_daily(None, [], 20190101, 20261231, pool=pool)


def month_end_days(daily: pd.DataFrame, begin: str) -> list[pd.Timestamp]:
    """池内交易日序列上的每月最后一个交易日。"""
    dates = pd.DatetimeIndex(sorted(daily.reset_index()["date"].unique()))
    dates = dates[dates >= pd.Timestamp(begin)]
    return list(dates.to_series().groupby(dates.to_period("M")).last())


def aggregate_window(daily_agg: pd.DataFrame, ends: list[pd.Timestamp],
                     value_col: str, how: str = "mean") -> pd.DataFrame:
    """月末截面回溯 90 自然日线性衰减聚合。

    daily_agg: code, date, <value_col>（同股同日已聚合）
    how=mean: 加权均值（得分型）；how=sum: 加权和（数量型）。
    """
    rows = []
    dates = daily_agg["date"].values
    for t in ends:
        lo = t - pd.Timedelta(days=WINDOW_DAYS - 1)
        m = (dates >= np.datetime64(lo)) & (dates <= np.datetime64(t))
        sub = daily_agg[m]
        if sub.empty:
            continue
        lag = (t - sub["date"]).dt.days.values
        w = (WINDOW_DAYS - lag) / WINDOW_DAYS
        v = sub[value_col].values
        ok = ~np.isnan(v)
        if not ok.any():
            continue
        w, v = w[ok], v[ok]
        g = pd.DataFrame({"code": sub["code"].values[ok], "w": w, "v": v})
        if how == "sum":
            fac = g.groupby("code").apply(lambda x: (x["w"] * x["v"]).sum(),
                                          include_groups=False)
        else:
            fac = g.groupby("code").apply(
                lambda x: (x["w"] * x["v"]).sum() / x["w"].sum(),
                include_groups=False)
        rows.append(pd.DataFrame({"date": t, "code": fac.index,
                                  "factor": fac.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def winsor_zscore(s: pd.Series) -> pd.Series:
    """AI 41 预处理：中位数去极值（±5×|离中位|的中位数）+ 标准化。"""
    med = s.median()
    mad = (s - med).abs().median()
    if mad and mad > 0:
        s = s.clip(lower=med - 5 * mad, upper=med + 5 * mad)
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0


def residual_factor(target: pd.DataFrame, controls: list[pd.DataFrame]) -> pd.DataFrame:
    """月度截面：target 对 controls 回取残差（各自已 winsor+标准化）。"""
    merged = target.rename(columns={"factor": "y"})
    for i, c in enumerate(controls):
        merged = merged.merge(c.rename(columns={"factor": f"x{i}"}),
                              on=["date", "code"], how="inner")
    rows = []
    for t, g in merged.groupby("date"):
        X = np.column_stack([np.ones(len(g))] +
                            [g[f"x{i}"].values for i in range(len(controls))])
        y = g["y"].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        # 保留 y 量纲（y 已按月标准化），不再对残差标准化：
        # R²→1 的月份残差数值上是 0，再标准化会把浮点噪声放大成假因子
        rows.append(pd.DataFrame({
            "date": t, "code": g["code"].values, "factor": res}))
    return pd.concat(rows, ignore_index=True)


def build_daily_agg(pool: str, begin: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """两类日度聚合表：情感得分表（BERT）+ 评级/数量表（结构化字段）。"""
    from config import Config
    cache_root = Path(str(Config.cache()["root"]).replace("//", "/"))
    if pool == "zz1000":
        daily_path = cache_root / "archive_zz1000" / "daily_zz1000_only.parquet"
    else:
        daily_path = cache_root / f"daily_{pool}.parquet"
    codes = set(pd.read_parquet(daily_path, columns=[])
                .index.get_level_values("code").unique())
    rep = pd.read_parquet(cache_root / "text_ths_report.parquet")
    rep = rep[rep["code"].isin(codes)].copy()
    rep["date"] = pd.to_datetime(rep["date"]).dt.normalize()
    rep = rep[rep["date"] >= pd.Timestamp(begin)]

    # 评级/数量（无 BERT 依赖，全历史）
    rep["rating_num"] = rep["rating"].map(RATING_MAP)
    rated = rep.dropna(subset=["rating_num"])
    rating_agg = rated.groupby(["code", "date"])["rating_num"].mean().reset_index()
    num_agg = rep.groupby(["code", "date"]).size().rename("n").reset_index()

    # 情感得分（BERT 打分产物）
    sp = OUT_DIR / f"senti_scores_{pool}.parquet"
    if not sp.exists():
        raise FileNotFoundError(f"{sp} 不存在，先跑 build_senti_scores")
    sc = pd.read_parquet(sp)
    sc["s"] = sc["p_pos"] - sc["p_neg"]
    sc["s_adj"] = sc["s"].where(sc["s"] >= 0, sc["s"] * 3)  # 负面×3（AI 41）
    senti_agg = (sc.groupby(["code", "date"])["s"].mean().reset_index()
                 .rename(columns={"s": "v"}).assign(kind="senti"))
    senti_adj_agg = (sc.groupby(["code", "date"])["s_adj"].mean().reset_index()
                     .rename(columns={"s_adj": "v"}).assign(kind="senti_adj"))
    return pd.concat([senti_agg, senti_adj_agg],
                     ignore_index=True), pd.concat(
        [num_agg.rename(columns={"n": "v"}).assign(kind="report_num"),
         rating_agg.rename(columns={"rating_num": "v"}).assign(kind="report_score")],
        ignore_index=True)


def validate_sentence_agreement(pool: str, n_reports: int = 300) -> str:
    """抽样验证：报告级得分 vs AI 41 逐句均值得分的一致性（Spearman）。"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    try:
        from config import Config
        d = Config.get().get("textmining", {}).get("bert_model_dir")
    except Exception:
        d = None
    model_dir = str(d).replace("//", "/") if d else r"E:/data/models/finbert_tone_chinese"
    torch.set_num_threads(22)

    sp = OUT_DIR / f"senti_scores_{pool}.parquet"
    sc = pd.read_parquet(sp)
    from config import Config as C
    from scripts.textmining.build_senti_scores import clean_text
    cache_root = Path(str(C.cache()["root"]).replace("//", "/"))
    rep = pd.read_parquet(cache_root / "text_ths_report.parquet")
    rep["title_head"] = rep["title"].fillna("").str.slice(0, 60)
    rep = rep.merge(sc[["code", "date", "title_head", "p_pos", "p_neg"]],
                    on=["code", "date", "title_head"], how="inner")
    rep = rep.drop_duplicates(["code", "date", "title_head"])
    rep = rep.sample(min(n_reports, len(rep)), random_state=42)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    def _prob(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 3), dtype=np.float32)
        for i in range(0, len(texts), 48):
            enc = tok(texts[i:i + 48], padding=True, truncation=True,
                      max_length=96, return_tensors="pt")
            with torch.no_grad():
                lg = model(**enc).logits
            out[i:i + 48] = torch.softmax(lg, dim=1).numpy()
        return out

    rep_level, sent_level = [], []
    for _, r in rep.iterrows():
        text = clean_text(r["title"], r["summary"])
        sents = [s for s in re.split(r"[。！？；]", text) if 8 < len(s) < 300][:12]
        if not sents:
            continue
        p = _prob(sents)
        sent_score = float((p[:, 1] - p[:, 2]).mean())
        rep_level.append(r["p_pos"] - r["p_neg"])
        sent_level.append(sent_score)
    rho = pd.Series(rep_level).corr(pd.Series(sent_level), method="spearman")
    line = (f"抽样 {len(rep_level)} 篇：报告级(max_len=256截断) vs 逐句均值"
            f"(前12句, max_len=96) 情感得分 Spearman = {rho:.3f}")
    out = OUT_DIR / f"senti_sentence_check_{pool}.txt"
    out.write_text(line + "\n", encoding="utf-8")
    log.info(line)
    return line


def run(pool: str = "zz1000", begin: str = "20190101",
        factor_begin: str = "20200601", validate: bool = True):
    senti_agg, struct_agg = build_daily_agg(pool, begin)
    daily = pool_daily(pool)
    ends = month_end_days(daily, factor_begin)
    log.info("月末交易日 %d 个（%s ~ %s）", len(ends), ends[0].date(), ends[-1].date())

    panels: dict[str, pd.DataFrame] = {}
    for kind, agg in [("senti", senti_agg[senti_agg["kind"] == "senti"]),
                      ("senti_adj", senti_agg[senti_agg["kind"] == "senti_adj"]),
                      ("report_num", struct_agg[struct_agg["kind"] == "report_num"]),
                      ("report_score", struct_agg[struct_agg["kind"] == "report_score"])]:
        agg = agg.rename(columns={"s": "v", "s_adj": "v", "n": "v",
                                  "rating_num": "v"}, errors="ignore")
        how = "sum" if kind == "report_num" else "mean"
        panels[kind] = aggregate_window(agg[["code", "date", "v"]], ends, "v", how)
        log.info("%s: %d 行 / %d 只", kind, len(panels[kind]),
                 panels[kind]["code"].nunique())

    # 残差因子（截面 winsor+标准化后对评级/数量回归）
    for name, src in [("senti_res", "senti"), ("senti_adj_res", "senti_adj")]:
        t = panels[src][["date", "code", "factor"]].copy()
        t["factor"] = t.groupby("date")["factor"].transform(winsor_zscore)
        ctrl = []
        for c in ["report_score", "report_num"]:
            cc = panels[c][["date", "code", "factor"]].copy()
            cc["factor"] = cc.groupby("date")["factor"].transform(winsor_zscore)
            ctrl.append(cc)
        panels[name] = residual_factor(t, ctrl)
        log.info("%s: %d 行", name, len(panels[name]))

    # 落盘：单因子面板（兼容 evaluate 框架）+ 宽表
    wide = None
    for name, df in panels.items():
        out = OUT_DIR / f"senti_factor_{name}_{pool}.parquet"
        df.set_index(["date", "code"]).sort_index().to_parquet(
            out, compression="snappy")
        w = df.rename(columns={"factor": name})[["date", "code", name]]
        wide = w if wide is None else wide.merge(w, on=["date", "code"], how="outer")
    wide = wide.sort_values(["date", "code"]).reset_index(drop=True)
    OUT_DIR / f"senti_factors_{pool}.parquet"
    wide.to_parquet(OUT_DIR / f"senti_factors_{pool}.parquet", compression="snappy")
    log.info("宽表 %d 行 → senti_factors_%s.parquet", len(wide), pool)

    # 因子相关矩阵（AI 41 图表17 口径：pairwise Spearman）
    corr = wide.drop(columns=["date", "code"]).corr(method="spearman").round(3)
    (OUT_DIR / f"senti_factor_corr_{pool}.txt").write_text(
        corr.to_string(), encoding="utf-8")
    log.info("因子相关矩阵:\n%s", corr.to_string())

    if validate:
        validate_sentence_agreement(pool)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="zz1000", choices=["hs300", "zz1000"])
    ap.add_argument("--begin", default="20190101")
    ap.add_argument("--factor-begin", default="20200601")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()
    run(args.pool, args.begin, args.factor_begin, validate=not args.no_validate)
