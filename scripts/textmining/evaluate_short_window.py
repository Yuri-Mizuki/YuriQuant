"""
短窗口日频超额评估（T+1~T+5）
============================

动机：当前月度评估（月末→下月）可能 horizon 错配 PEAD/FADT 文本效应
（2-3 周窗口）。本脚本评估事件后 T+1~T+5 逐日累计超额收益，看因子排序
后逐日超额是否单调、方向是否符合。

口径：
- 事件日 T：SUE.txt = 业绩预告日；FADT = 研报发布日（已 PIT，同 build_samples）
- T+k 累计超额 = (P_{T+k}/P_T - 1) - (Idx_{T+k}/Idx_T - 1)，k=1..5
  注意 P_T 用 T 当日收盘（公告盘后发布，T 当日收盘不含公告信息，
  T+1 才是第一个信息交易日）。
- 因子值：直接用逐轮预测的 sue0（样本外，无前视），按事件日截面分 5 层，
  各层 T+1~T+5 平均累计超额曲线 + 日频 RankIC 衰减曲线。

用法：
    python -m scripts.textmining.evaluate_short_window --task sue --model xgb --pool hs300
    python -m scripts.textmining.evaluate_short_window --task fadt --model logit --pool zz1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\YuriQuant")
from scripts.textmining.build_sue_txt_samples import _load_daily, _to_naive  # noqa: E402

OUT_DIR = Path(r"E:\YuriQuant\reports\textmining")
BENCH = "000905.SH"


def _load_pred_with_factor(task: str, model: str, pool: str) -> pd.DataFrame:
    """加载逐轮预测（sue0）—— 这是事件级文本得分，短窗口评估的事件级信号。

    不用月末聚合因子（那个已经衰减/平均过），直接用原始 sue0 看事件级
    短窗口效应是否成立。
    """
    if model == "bert_xgb" or model == "bert_logit":
        return _recompute_sue0_bert(task, model.split("_")[1], pool)
    if task == "sue":
        # SUE.txt 训练保存的是 sue_txt_samples_*_tokenized 里的 sue0 列？不，
        # 逐轮 pred 没有单独落盘。需要从因子面板反推或重算。最干净的做法：
        # 从 tokenized 样本 + joblib 模型重算 sue0。
        # 但更简单：因子面板本身是月末聚合，事件级 sue0 没落盘。
        # 这里改为直接从样本 + 模型重算事件级 sue0。
        return _recompute_sue0_sue(model, pool)
    else:
        return _recompute_sue0_fadt(model, pool)


def _recompute_sue0_sue(model: str, pool: str) -> pd.DataFrame:
    """重算 SUE.txt 逐轮事件级 sue0（复用训练时的滚动逻辑）。"""
    from scripts.textmining.train_sue_txt import (
        SUEVectorizer, TITLE_TOP, SUMMARY_TOP, TRAIN_MONTHS, TEST_MONTHS,
        _sue0_from_model, make_labels, tokenize_title, tokenize_summary,
    )
    import joblib

    tok_path = OUT_DIR / f"sue_txt_samples_tokenized_{pool}.parquet"
    if not tok_path.exists():
        raise FileNotFoundError(f"{tok_path} 不存在，先跑 train_sue_txt")
    tok = pd.read_parquet(tok_path)
    tok["event_date"] = pd.to_datetime(tok["event_date"]).dt.normalize()

    all_pred = []
    test_start = pd.Timestamp("20210101")
    round_no = 0
    while True:
        tr_end = test_start - pd.Timedelta(days=1)
        tr_start = tr_end - pd.DateOffset(months=TRAIN_MONTHS) + pd.Timedelta(days=1)
        te_end = test_start + pd.DateOffset(months=TEST_MONTHS) - pd.Timedelta(days=1)
        tr = tok[(tok["event_date"] >= tr_start) & (tok["event_date"] <= tr_end)].copy()
        te = tok[(tok["event_date"] >= test_start) & (tok["event_date"] <= te_end)].copy()
        if len(te) == 0:
            break
        round_no += 1
        mpath = OUT_DIR / f"sue_txt_model_{model}_{pool}_r{round_no}.joblib"
        if not mpath.exists():
            test_start = te_end + pd.Timedelta(days=1)
            continue
        if len(tr) < 50 or len(te) == 0:
            test_start = te_end + pd.Timedelta(days=1)
            continue
        tr = tr.dropna(subset=["ar", "title_tok"]).copy()
        tr["label"] = make_labels(tr["ar"]) + 1
        vec = SUEVectorizer()
        vec.fit(tr["title_tok"], tr["summary_tok"])
        m = joblib.load(mpath)
        te = te.dropna(subset=["ar", "title_tok"]).copy()
        X_te = vec.transform(te["title_tok"], te["summary_tok"])
        te["sue0"] = _sue0_from_model(m, X_te)
        all_pred.append(te[["code", "event_date", "sue0"]])
        test_start = te_end + pd.Timedelta(days=1)
    if not all_pred:
        raise RuntimeError("无预测样本")
    return pd.concat(all_pred, ignore_index=True)


def _recompute_sue0_fadt(model: str, pool: str) -> pd.DataFrame:
    """重算 FADT 逐轮事件级 sue0。"""
    from scripts.textmining.train_fadt import TITLE_TOP, SUMMARY_TOP, TRAIN_MONTHS, TEST_MONTHS
    from scripts.textmining.train_sue_txt import (
        SUEVectorizer, _sue0_from_model, make_labels,
    )
    import joblib

    tok_path = OUT_DIR / f"fadt_samples_tokenized_{pool}.parquet"
    if not tok_path.exists():
        raise FileNotFoundError(f"{tok_path} 不存在，先跑 train_fadt")
    tok = pd.read_parquet(tok_path)
    tok["event_date"] = pd.to_datetime(tok["event_date"]).dt.normalize()

    all_pred = []
    test_start = pd.Timestamp("20210101")
    round_no = 0
    while True:
        tr_end = test_start - pd.Timedelta(days=1)
        tr_start = tr_end - pd.DateOffset(months=TRAIN_MONTHS) + pd.Timedelta(days=1)
        te_end = test_start + pd.DateOffset(months=TEST_MONTHS) - pd.Timedelta(days=1)
        tr = tok[(tok["event_date"] >= tr_start) & (tok["event_date"] <= tr_end)].copy()
        te = tok[(tok["event_date"] >= test_start) & (tok["event_date"] <= te_end)].copy()
        if len(te) == 0:
            break
        round_no += 1
        mpath = OUT_DIR / f"fadt_model_{model}_{pool}_r{round_no}.joblib"
        if not mpath.exists():
            test_start = te_end + pd.Timedelta(days=1)
            continue
        if len(tr) < 100 or len(te) == 0:
            test_start = te_end + pd.Timedelta(days=1)
            continue
        tr = tr.dropna(subset=["ar", "title_tok"]).copy()
        tr["label"] = make_labels(tr["ar"]) + 1
        vec = SUEVectorizer(title_top=TITLE_TOP, summary_top=SUMMARY_TOP)
        vec.fit(tr["title_tok"], tr["summary_tok"])
        m = joblib.load(mpath)
        te = te.dropna(subset=["ar", "title_tok"]).copy()
        X_te = vec.transform(te["title_tok"], te["summary_tok"])
        te["sue0"] = _sue0_from_model(m, X_te)
        all_pred.append(te[["code", "event_date", "sue0"]])
        test_start = te_end + pd.Timedelta(days=1)
    if not all_pred:
        raise RuntimeError("无预测样本")
    return pd.concat(all_pred, ignore_index=True)


def _recompute_sue0_bert(task: str, base_model: str, pool: str) -> pd.DataFrame:
    """重算 BERT 版逐轮事件级 sue0（复用 train_fadt_bert 的滚动逻辑 + CLS 特征）。

    model 参数传 "bert_xgb"/"bert_logit"，base_model 取 xgb/logit。
    """
    from scripts.textmining.train_fadt_bert import (
        _load_cls, BERT_XGB_GRID, train_bert_xgb,
    )
    from scripts.textmining.train_fadt import TRAIN_MONTHS, TEST_MONTHS
    from scripts.textmining.train_sue_txt import (
        _sue0_from_model, make_labels, train_logit,
    )
    import joblib
    import numpy as np

    sample_path = OUT_DIR / (f"{task}_samples_{pool}.parquet" if task == "fadt"
                             else f"sue_txt_samples_{pool}.parquet")
    samples = pd.read_parquet(sample_path)
    samples["event_date"] = pd.to_datetime(samples["event_date"]).dt.normalize()
    samples = samples.reset_index(names="row_idx")
    cls = _load_cls(task, pool)
    samples = samples.merge(cls, on=["row_idx", "code", "event_date"], how="left")
    cls_cols = [c for c in samples.columns if c.startswith("cls_")]

    all_pred = []
    test_start = pd.Timestamp("20210101")
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
        mpath = OUT_DIR / f"{task}_bert_model_{base_model}_{pool}_r{round_no}.joblib"
        if not mpath.exists():
            test_start = te_end + pd.Timedelta(days=1)
            continue
        if len(tr) < 100 or len(te) == 0:
            test_start = te_end + pd.Timedelta(days=1)
            continue
        tr = tr.dropna(subset=["ar", cls_cols[0]]).copy()
        tr["label"] = make_labels(tr["ar"]) + 1
        m = joblib.load(mpath)
        te = te.dropna(subset=["ar", cls_cols[0]]).copy()
        X_te = te[cls_cols].values.astype(np.float32)
        te["sue0"] = _sue0_from_model(m, X_te)
        all_pred.append(te[["code", "event_date", "sue0"]])
        test_start = te_end + pd.Timedelta(days=1)
    if not all_pred:
        raise RuntimeError("无预测样本（先跑 train_fadt_bert）")
    return pd.concat(all_pred, ignore_index=True)


def _compute_cum_excess(pred: pd.DataFrame, daily: pd.DataFrame,
                        k_max: int = 5, k_start: int = 1) -> pd.DataFrame:
    """对每个事件计算 T+k_start~T+k_max 累计超额收益（相对中证500）。

    P_T = 事件日 T 当日收盘（公告盘后发布，T 收盘不含公告信息）。
    累计超额_k = (P_{T+k}/P_T - 1) - (Idx_{T+k}/Idx_T - 1)
    """
    # 每只股票的交易日序列
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

    rows = []
    for _, r in pred.iterrows():
        code = r["code"]
        d = pd.Timestamp(r["event_date"])
        p0 = _close_at(code, d, 0)
        b0 = _close_at(BENCH, d, 0)
        if p0 is None or b0 is None:
            continue
        row = {"code": code, "event_date": d, "sue0": r["sue0"]}
        for k in range(k_start, k_max + 1):
            pk = _close_at(code, d, k)
            bk = _close_at(BENCH, d, k)
            if pk is not None and bk is not None:
                row[f"cum_excess_t{k}"] = (pk / p0 - 1) - (bk / b0 - 1)
            else:
                row[f"cum_excess_t{k}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _layer_excess_curve(df: pd.DataFrame, k_max: int = 5,
                        n_layers: int = 5) -> pd.DataFrame:
    """按 sue0 分 5 层，算 T+1~T+k 各层平均累计超额。

    分层是全样本截面（不分日期），因为事件分散在各日，短窗口评估不分
    日期截面（每个事件的 T+k 相对该事件本身）。
    """
    df = df.dropna(subset=["sue0"]).copy()
    df["layer"] = df["sue0"].rank(pct=True).apply(
        lambda p: min(int(p * n_layers) + 1, n_layers))
    cols = [f"cum_excess_t{k}" for k in range(1, k_max + 1)]
    g = df.groupby("layer")[cols].mean()
    # 多空（L1 - L5，假设 sue0 正→上涨）
    g.loc["L1-L5"] = g.loc[1] - g.loc[n_layers]
    g.loc["L5-L1"] = g.loc[n_layers] - g.loc[1]  # 反向，便于看方向
    return g


def _daily_rank_ic_decay(df: pd.DataFrame, k_max: int = 5) -> pd.DataFrame:
    """日频 RankIC 衰减：sue0 vs cum_excess_t{k} 的 spearman 相关。

    全样本一个 IC（不分日期，因为事件 T+k 收益本身就是相对该事件的短窗口）。
    """
    from scipy.stats import spearmanr
    rows = []
    for k in range(1, k_max + 1):
        col = f"cum_excess_t{k}"
        sub = df.dropna(subset=["sue0", col])
        if len(sub) < 30:
            rows.append({"k": k, "ic": np.nan, "n": len(sub)})
            continue
        ic, _ = spearmanr(sub["sue0"], sub[col])
        rows.append({"k": k, "ic": ic, "n": len(sub)})
    return pd.DataFrame(rows).set_index("k")


def main(task: str = "sue", model: str = "xgb", pool: str = "hs300",
         k_max: int = 5):
    print(f"== {task.upper()} ({model}) 短窗口日频超额评估 ==")
    print(f"事件后 T+1~T+{k_max} 累计超额（相对中证500）\n")

    pred = _load_pred_with_factor(task, model, pool)
    print(f"事件级预测: {len(pred)} 条, 覆盖 {pred['code'].nunique()} 只")

    daily = _load_daily(None, [], 20190101, 20261231)
    df = _compute_cum_excess(pred, daily, k_max=k_max)
    valid = df.dropna(subset=[f"cum_excess_t{k_max}"])
    print(f"有效（T+{k_max} 有收益）: {len(valid)} 条\n")

    print("[分层累计超额 T+1~T+5]（按 sue0 分 5 层，全样本截面）")
    layers = _layer_excess_curve(df, k_max=k_max)
    # 百分比格式
    print((layers * 100).round(3).to_string())
    print()

    print("[日频 RankIC 衰减]")
    ic = _daily_rank_ic_decay(df, k_max=k_max)
    print(ic.round(4).to_string())
    print()

    # 结论判断
    l1l5_t1 = layers.loc["L1-L5", "cum_excess_t1"] if "L1-L5" in layers.index else np.nan
    l1l5_t5 = layers.loc["L1-L5", "cum_excess_t5"] if "L1-L5" in layers.index else np.nan
    ic1 = ic.loc[1, "ic"] if 1 in ic.index else np.nan
    ic5 = ic.loc[k_max, "ic"] if k_max in ic.index else np.nan
    print(f"[判断] L1-L5 T+1={l1l5_t1*100:.3f}% T+{k_max}={l1l5_t5*100:.3f}% | "
          f"IC T+1={ic1:.4f} T+{k_max}={ic5:.4f}")
    if not np.isnan(ic1) and not np.isnan(ic5):
        if abs(ic1) > 0.02 and abs(ic1) > abs(ic5):
            print("→ 短窗口有信号且衰减，符合 PEAD/FADT 2-3 周效应假设（月度 horizon 错配确认）")
        elif abs(ic1) < 0.01:
            print("→ 短窗口也无信号，文本分类本身泛化能力不足（与 grouped CV 结论一致）")
        else:
            print("→ 信号模式不典型，需进一步检查")

    out = OUT_DIR / f"{task}_short_window_{model}_{pool}.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"== {task.upper()} ({model}) 短窗口日频超额评估 ==\n")
        fh.write(f"事件级预测: {len(pred)} 条\n有效: {len(valid)} 条\n\n")
        fh.write(f"[分层累计超额 T+1~T+{k_max} (%)]\n{(layers*100).round(3).to_string()}\n\n")
        fh.write(f"[日频 RankIC 衰减]\n{ic.round(4).to_string()}\n\n")
        fh.write(f"L1-L5 T+1={l1l5_t1*100:.3f}% T+{k_max}={l1l5_t5*100:.3f}% | "
                 f"IC T+1={ic1:.4f} T+{k_max}={ic5:.4f}\n")
    print(f"评估已存: {out}")
    return layers, ic


def _load_samples_ar(task: str, pool: str) -> pd.DataFrame:
    """加载样本里的 ar（两日异常收益），用于 ablation 对比。"""
    if task == "sue":
        sp = OUT_DIR / f"sue_txt_samples_{pool}.parquet"
    else:
        sp = OUT_DIR / f"fadt_samples_{pool}.parquet"
    if not sp.exists():
        return pd.DataFrame()
    s = pd.read_parquet(sp)
    s["event_date"] = pd.to_datetime(s["event_date"]).dt.normalize()
    # 事件级 ar（同事件多研报取均值）
    return s.groupby(["code", "event_date"], as_index=False)["ar"].mean()


def _event_price_features(pred: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """事件日 T 收盘时已知的价格特征（无前视，不含 T+1 信息）：

    - pre_mom5 = (P_T/P_{T-5} − 1) − (Idx_T/Idx_{T-5} − 1)  事件前 5 日动量
    - day_ret  = (P_T/P_{T-1} − 1) − (Idx_T/Idx_{T-1} − 1)  T 当日反应
    全部在 T 日收盘即可得，可公平地与 sue0（文本，T 日可得）比较。
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

    rows = []
    for _, r in pred.iterrows():
        code = r["code"]
        d = pd.Timestamp(r["event_date"])
        p0 = _close_at(code, d, 0)
        b0 = _close_at(BENCH, d, 0)
        if p0 is None or b0 is None:
            continue

        def _exc(sh: int) -> float | None:
            p = _close_at(code, d, sh)
            b = _close_at(BENCH, d, sh)
            if p is None or b is None:
                return np.nan
            return (p / p0 - 1) - (b / b0 - 1)

        rows.append({"code": code, "event_date": d,
                     "pre_mom5": _exc(-5), "day_ret": _exc(-1)})
    return pd.DataFrame(rows)


def _orthogonal_ic(df: pd.DataFrame, factor: str, target_col: str,
                   bench_cols: list[str]) -> float | None:
    """增量 IC：factor 对价格基准做 rank 正交化后的残差 vs 目标的 RankIC。

    spearman(factor, target) 去除 bench_cols 贡献后的净相关性。
    实现：rank 变换后 OLS 回归 factor~bench_cols，取残差与 target 的 spearman。
    """
    from sklearn.linear_model import LinearRegression
    sub = df.dropna(subset=[factor, target_col] + bench_cols)
    if len(sub) < 50:
        return None
    X = sub[bench_cols].rank().values
    y = sub[factor].rank().values
    resid = y - LinearRegression().fit(X, y).predict(X)
    from scipy.stats import spearmanr
    ic, _ = spearmanr(resid, sub[target_col])
    return float(ic)


def ablation_sue0_vs_price(task: str, model: str, pool: str, k_max: int = 5):
    """无泄漏 ablation：sue0（文本） vs 价格基准（T 日已知）+ ar（T+2 可知）。

    修正旧版 ablation_sue0_vs_ar 的泄漏：
      ar = (P_{T+1}/P_{T-1} − 1) 含 T+1 价格，直接与 T+1 目标比是机械相关
      （旧版 IC 0.65 是假数）。新版：
      - 主表（基准 T 收盘已知）：pre_mom5 / day_ret vs sue0，目标 T+1~T+5
      - 漂移表（T+2~T+5）：加入 ar（T+1 收盘后才可知，T+2 开盘可交易）
      - 增量 IC：sue0 对价格基准 rank 正交化后的残差 RankIC
    """
    from scipy.stats import spearmanr
    print(f"\n== {task.upper()} ({model}) 无泄漏 ablation: sue0 vs 价格基准 ==")
    pred = _load_pred_with_factor(task, model, pool)
    ar_df = _load_samples_ar(task, pool)
    if ar_df.empty:
        print("ar 样本缺失，跳过")
        return
    # pred 是研报级（同事件多研报多行）→ 聚合为事件级（sue0 取均值），
    # 与 ar_df/feat 同口径，避免 merge 笛卡尔积 + 事件重复计数
    pred = pred.groupby(["code", "event_date"], as_index=False)["sue0"].mean()
    pred = pred.merge(ar_df, on=["code", "event_date"], how="inner")

    daily = _load_daily(None, [], 20190101, 20261231)
    feat = _event_price_features(pred[["code", "event_date"]], daily)
    df = pred.merge(feat, on=["code", "event_date"], how="inner")
    targets = _compute_cum_excess(df[["code", "event_date", "sue0"]], daily, k_max)
    df = df.merge(targets.drop(columns=["sue0"]), on=["code", "event_date"], how="inner")
    print(f"样本: {len(df)} 事件, 覆盖 {df['code'].nunique()} 只\n")

    bench_cols = ["pre_mom5", "day_ret"]
    print("[" + "=" * 78 + "]")
    print("[主表] 基准因子 T 日收盘已知，目标 = T+k 累计超额（相对中证500）")
    print(f"{'k':>3} {'IC(sue0)':>10} {'IC(pre_mom5)':>14} {'IC(day_ret)':>13} "
          f"{'inc_IC(sue0|price)':>20} {'n':>7}")
    print("-" * 78)
    for k in range(1, k_max + 1):
        col = f"cum_excess_t{k}"
        sub = df.dropna(subset=["sue0"] + bench_cols + [col])
        if len(sub) < 30:
            print(f"{k:>3} {'nan':>10} {'nan':>14} {'nan':>13} {'nan':>20} {len(sub):>7}")
            continue
        ic_s, _ = spearmanr(sub["sue0"], sub[col])
        ic_p, _ = spearmanr(sub["pre_mom5"], sub[col])
        ic_d, _ = spearmanr(sub["day_ret"], sub[col])
        inc = _orthogonal_ic(df, "sue0", col, bench_cols)
        inc_s = "nan" if inc is None else f"{inc:.4f}"
        print(f"{k:>3} {ic_s:>10.4f} {ic_p:>14.4f} {ic_d:>13.4f} {inc_s:>20} {len(sub):>7}")

    print("\n[漂移表] 目标 = T+2~T+5（T+1 反应之后的持续漂移）")
    print("         ar = (P_{T+1}/P_{T-1}−1) 含 T+1，T+2 开盘才可知 → 只进此表")
    print(f"{'k':>3} {'IC(sue0)':>10} {'IC(ar)':>10} {'IC(pre_mom5)':>14} "
          f"{'inc_IC(sue0|price+ar)':>24} {'n':>7}")
    print("-" * 78)
    for k in range(2, k_max + 1):
        col = f"cum_excess_t{k}"
        sub = df.dropna(subset=["sue0", "ar"] + bench_cols + [col])
        if len(sub) < 30:
            print(f"{k:>3} {'nan':>10} {'nan':>10} {'nan':>14} {'nan':>24} {len(sub):>7}")
            continue
        ic_s, _ = spearmanr(sub["sue0"], sub[col])
        ic_a, _ = spearmanr(sub["ar"], sub[col])
        ic_p, _ = spearmanr(sub["pre_mom5"], sub[col])
        inc = _orthogonal_ic(df, "sue0", col, bench_cols + ["ar"])
        inc_s = "nan" if inc is None else f"{inc:.4f}"
        print(f"{k:>3} {ic_s:>10.4f} {ic_a:>10.4f} {ic_p:>14.4f} {inc_s:>24} {len(sub):>7}")

    # 分层多空（T+1 与 T+5 目标）
    print("\n[分层多空 L5-L1（%）]（各因子独立分 5 层）")
    print(f"{'target':>8} {'sue0':>10} {'pre_mom5':>12} {'day_ret':>11}")
    print("-" * 45)
    for k in [1, k_max]:
        col = f"cum_excess_t{k}"
        rows = {}
        for f in ["sue0", "pre_mom5", "day_ret"]:
            sub = df.dropna(subset=[f, col]).copy()
            sub["layer"] = sub[f].rank(pct=True).apply(
                lambda p: min(int(p * 5) + 1, 5))
            g = sub.groupby("layer")[col].mean()
            rows[f] = g.loc[5] - g.loc[1]
        print(f"T+{k:<5} {rows['sue0']*100:>10.3f} {rows['pre_mom5']*100:>12.3f} "
              f"{rows['day_ret']*100:>11.3f}")

    # sue0 与价格基准截面相关（诊断共线性）
    corr = df[["sue0"] + bench_cols].corr().loc["sue0", bench_cols]
    print(f"\n[诊断] sue0 与价格基准截面相关: "
          f"{', '.join(f'{c}={corr[c]:.4f}' for c in bench_cols)}")
    if corr.abs().max() > 0.3:
        print("→ 中度以上相关，sue0 与价格信息重叠较多")
    else:
        print("→ 低相关，sue0 与价格基准近似独立")

    out = OUT_DIR / f"{task}_ablation_noleak_{model}_{pool}.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"== {task.upper()} ({model}) 无泄漏 ablation: sue0 vs 价格基准 ==\n")
        fh.write(f"样本: {len(df)} 事件\n\n")
        fh.write("[主表] 基准 T 收盘已知，目标 T+k 累计超额\n")
        fh.write(f"{'k':>3} {'IC(sue0)':>10} {'IC(pre_mom5)':>14} {'IC(day_ret)':>13} "
                 f"{'inc_IC(sue0|price)':>20} {'n':>7}\n")
        for k in range(1, k_max + 1):
            col = f"cum_excess_t{k}"
            sub = df.dropna(subset=["sue0"] + bench_cols + [col])
            if len(sub) < 30:
                continue
            ic_s, _ = spearmanr(sub["sue0"], sub[col])
            ic_p, _ = spearmanr(sub["pre_mom5"], sub[col])
            ic_d, _ = spearmanr(sub["day_ret"], sub[col])
            inc = _orthogonal_ic(df, "sue0", col, bench_cols)
            inc_s = "nan" if inc is None else f"{inc:.4f}"
            fh.write(f"{k:>3} {ic_s:>10.4f} {ic_p:>14.4f} {ic_d:>13.4f} {inc_s:>20} "
                     f"{len(sub):>7}\n")
        fh.write("\n[漂移表] T+2~T+5，ar 含 T+1（T+2 开盘可知）\n")
        for k in range(2, k_max + 1):
            col = f"cum_excess_t{k}"
            sub = df.dropna(subset=["sue0", "ar"] + bench_cols + [col])
            if len(sub) < 30:
                continue
            ic_s, _ = spearmanr(sub["sue0"], sub[col])
            ic_a, _ = spearmanr(sub["ar"], sub[col])
            ic_p, _ = spearmanr(sub["pre_mom5"], sub[col])
            inc = _orthogonal_ic(df, "sue0", col, bench_cols + ["ar"])
            inc_s = "nan" if inc is None else f"{inc:.4f}"
            fh.write(f"{k:>3} {ic_s:>10.4f} {ic_a:>10.4f} {ic_p:>14.4f} {inc_s:>24} "
                     f"{len(sub):>7}\n")
        fh.write(f"\n[诊断] sue0 与价格基准相关: "
                 f"{', '.join(f'{c}={corr[c]:.4f}' for c in bench_cols)}\n")
    print(f"\n已存: {out}")


def daily_rebalance_backtest(task: str, model: str, pool: str,
                             hold_days: int = 5, n_layers: int = 5):
    """日频调仓回测：每个事件日 T 构建截面，分 5 层，持 T+1~T+hold_days 等权。

    口径：
    - 每个事件日 T（当日有事件覆盖的股票集合 = 当日截面）
    - 按 sue0 分 n_layers 层
    - 持有 T+1~T+hold_days 的日频收益，算各层年化 + 多空
    - 不做跨日持仓（每个事件独立持有窗口），与事件级评估口径一致
    """
    print(f"\n== {task.upper()} ({model}) 日频调仓回测（持 T+1~T+{hold_days}）==")
    pred = _load_pred_with_factor(task, model, pool)
    daily = _load_daily(None, [], 20190101, 20261231)

    # 每只股票的交易日序列 + 收益
    by_code = {}
    for code, sub in daily.reset_index().groupby("code"):
        sub = sub.copy()
        sub["date"] = _to_naive(sub["date"])
        sub = sub.sort_values("date").reset_index(drop=True)
        sub["ret"] = sub["close"].pct_change()
        by_code[code] = sub[["date", "close", "ret"]]

    def _daily_ret(code: str, d: pd.Timestamp, shift: int) -> float | None:
        tbl = by_code.get(code)
        if tbl is None or tbl.empty:
            return None
        dates = tbl["date"].values
        pos = int(pd.Index(dates).searchsorted(
            pd.Timestamp(d).to_datetime64(), side="right")) - 1
        if pos < 0:
            return None
        j = pos + shift
        if 0 <= j < len(dates):
            return float(tbl["ret"].iloc[j])
        return None

    # 每个事件日的日频收益序列 T+1..T+hold_days
    rows = []
    for _, r in pred.iterrows():
        code = r["code"]
        d = pd.Timestamp(r["event_date"])
        rets = {}
        for k in range(1, hold_days + 1):
            rets[f"ret_t{k}"] = _daily_ret(code, d, k)
        rows.append({"code": code, "event_date": d, "sue0": r["sue0"], **rets})
    df = pd.DataFrame(rows).dropna(subset=["sue0"])

    # 分层（全样本截面，事件级）
    df["layer"] = df["sue0"].rank(pct=True).apply(
        lambda p: min(int(p * n_layers) + 1, n_layers))

    # 各层 T+1~T+hold 平均日频收益 → 年化（假设 250 交易日）
    cols = [f"ret_t{k}" for k in range(1, hold_days + 1)]
    g = df.groupby("layer")[cols].mean()

    # 年化口径：各层 hold_days 平均日收益 × 250
    print(f"\n[各层平均日频收益 T+1~T+{hold_days}]")
    print((g * 100).round(4).to_string())
    print(f"\n[年化（日均收益 × 250, %）]")
    g_annual = g.mean(axis=1) * 250 * 100
    print(g_annual.round(2).to_string())

    # 多空年化
    long_short_daily = g.loc[n_layers] - g.loc[1]
    ls_annual = long_short_daily.mean() * 250 * 100
    print(f"\n[多空年化] L{n_layers}-L1 = {ls_annual:.2f}%")

    # 各 T+k 的多空收益（看哪天最强）
    print(f"\n[逐日多空收益 T+1~T+{hold_days}（L{n_layers}-L1, %）]")
    for k in range(1, hold_days + 1):
        col = f"ret_t{k}"
        ls = (g.loc[n_layers, col] - g.loc[1, col]) * 100
        print(f"  T+{k}: {ls:+.4f}%")

    return g, ls_annual


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sue", choices=["sue", "fadt"])
    ap.add_argument("--model", default="xgb",
                    choices=["xgb", "logit", "bert_xgb", "bert_logit"])
    ap.add_argument("--pool", default="hs300", choices=["hs300", "zz1000"])
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--ablation", action="store_true",
                    help="跑无泄漏 sue0 vs 价格基准 ablation")
    ap.add_argument("--backtest", action="store_true",
                    help="跑日频调仓回测")
    ap.add_argument("--hold-days", type=int, default=5,
                    help="日频调仓回测持有天数")
    args = ap.parse_args()

    if args.ablation:
        ablation_sue0_vs_price(args.task, args.model, args.pool, args.k_max)
    elif args.backtest:
        daily_rebalance_backtest(args.task, args.model, args.pool, args.hold_days)
    else:
        main(args.task, args.model, args.pool, args.k_max)
