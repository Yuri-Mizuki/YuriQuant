"""
日频因子面板构建
================

动机：ablation 对比发现文本因子 sue0 信号弱，ar（两日异常收益）在日频口径
（滞后投影）下相对更强，且集中在 T+1。月末截面恰好在信号衰减后。日频面板让
每个交易日 T 都有因子值，日频调仓可抓 T+1 信号。
注意（2026-08-18 修正）：事件级 ablation 中 ar 含 T+1 价格、与 T+1 目标机械
相关（旧版 IC 0.65 是泄漏假数）；本脚本日频面板的 ar 是"过去事件收益向后续
日期投影"，不含未来信息，作为动量基准是干净的。

口径：
- 每个交易日 T，对当日有事件覆盖的股票（过去 lookback 日内有事件），
  取最近事件的 sue0 / ar 作为因子值。
- 事件 horizon：ar 是 T-1~T+1 两日（事件日已知），sue0 是文本预测（事件日已知）。
- 因子值 = 最近事件的 sue0 或 ar（可加衰减，但 ar 本身无衰减含义，默认不衰减）。
- 面板：date × code × factor，用于日频调仓回测 + 入库。

用法：
    python -m scripts.textmining.build_daily_factor --task sue --model xgb --pool hs300
    python -m scripts.textmining.build_daily_factor --task fadt --model xgb --pool zz1000 --factor ar
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\YuriQuant")
from scripts.textmining.build_sue_txt_samples import _load_daily, _to_naive  # noqa: E402
from scripts.textmining.evaluate_short_window import (  # noqa: E402
    _load_pred_with_factor, _load_samples_ar,
)

OUT_DIR = Path(r"E:\YuriQuant\reports\textmining")


def build_daily_panel(task: str, model: str, pool: str,
                      factor_col: str = "sue0",
                      lookback_days: int = 5) -> pd.DataFrame:
    """构建日频因子面板。

    factor_col: "sue0"（文本预测）或 "ar"（两日异常收益，真信号源）。
    lookback_days: 过去 N 日内有事件视为有效覆盖（PEAD 效应在 T+1~T+5 内）。
    """
    # 加载事件级因子值
    if factor_col == "sue0":
        pred = _load_pred_with_factor(task, model, pool)
        pred["event_date"] = pd.to_datetime(pred["event_date"]).dt.normalize()
        events = pred[["code", "event_date", "sue0"]].rename(
            columns={"sue0": "factor"})
    elif factor_col == "ar":
        ar_df = _load_samples_ar(task, pool)
        ar_df["event_date"] = pd.to_datetime(ar_df["event_date"]).dt.normalize()
        events = ar_df.rename(columns={"ar": "factor"})
    else:
        raise ValueError(f"factor_col must be sue0 or ar, got {factor_col}")

    print(f"事件级 {factor_col}: {len(events)} 条, 覆盖 {events['code'].nunique()} 只")

    # 加载交易日历
    daily = _load_daily(None, [], 20190101, 20261231)
    all_dates = sorted(_to_naive(daily.reset_index()["date"]).unique())
    # 只保留 2021 之后（样本外）
    all_dates = [d for d in all_dates if d >= pd.Timestamp("20210101")]
    print(f"交易日: {len(all_dates)} 个, {all_dates[0].date()} ~ {all_dates[-1].date()}")

    # 对每个交易日 T，找过去 lookback_days 内的事件
    rows = []
    events_sorted = events.sort_values(["code", "event_date"]).reset_index(drop=True)
    # 按 code 分组，用 searchsorted 定位
    for code, grp in events_sorted.groupby("code"):
        grp = grp.sort_values("event_date").reset_index(drop=True)
        ev_dates = grp["event_date"].values
        ev_factors = grp["factor"].values
        for T in all_dates:
            # 过去 lookback_days 内的事件
            lb = T - pd.Timedelta(days=lookback_days)
            # 找 <= T 且 >= lb 的事件
            pos_right = int(pd.Index(ev_dates).searchsorted(
                pd.Timestamp(T).to_datetime64(), side="right"))
            pos_left = int(pd.Index(ev_dates).searchsorted(
                pd.Timestamp(lb).to_datetime64(), side="left"))
            if pos_right > pos_left:
                # 取最近事件（最后一个）
                rows.append({
                    "date": T, "code": code,
                    "factor": float(ev_factors[pos_right - 1]),
                    "event_date": pd.Timestamp(ev_dates[pos_right - 1]),
                    "n_events": pos_right - pos_left,
                })
    f = pd.DataFrame(rows)
    if f.empty:
        print("无有效日频面板")
        return f, pd.Series(dtype=float)
    f = f.set_index(["date", "code"]).sort_index()
    # 覆盖度统计
    cov = f.groupby(level="date").size()
    print(f"日频面板: {len(f)} 行, 覆盖 {f.index.get_level_values('code').nunique()} 只")
    print(f"日均覆盖: {cov.mean():.0f} 只 (中位 {cov.median():.0f})")
    return f, cov


def main(task: str = "sue", model: str = "xgb", pool: str = "hs300",
         factor_col: str = "sue0", lookback_days: int = 5):
    print(f"== {task.upper()} ({model}) 日频因子面板（{factor_col}）==")
    f, cov = build_daily_panel(task, model, pool, factor_col, lookback_days)
    if f.empty:
        return

    # 日频 RankIC（因子 vs 次日收益）
    daily = _load_daily(None, [], 20190101, 20261231)
    close = daily["close"].unstack("code")
    ret = close.pct_change().shift(-1)  # 次日收益
    # 合并
    ret_stacked = ret.stack().rename("ret").reset_index()
    ret_stacked.columns = ["date", "code", "ret"]
    ret_stacked["date"] = _to_naive(ret_stacked["date"])
    m = f.reset_index().merge(ret_stacked, on=["date", "code"], how="inner")
    m = m.dropna(subset=["factor", "ret"])
    if len(m) < 100:
        print(f"合并后样本过少({len(m)})，无法算 IC")
        return

    from scipy.stats import spearmanr
    ics = m.groupby("date").apply(
        lambda g: spearmanr(g["factor"], g["ret"])[0] if len(g) >= 5 else np.nan,
        include_groups=False)
    ics = ics.dropna()
    if len(ics) == 0:
        print("无有效 IC")
        return
    ic_mean = ics.mean()
    ic_std = ics.std()
    icir = ic_mean / ic_std if ic_std > 0 else np.nan
    pos_ratio = (ics > 0).mean()
    print(f"\n[日频 RankIC] {len(ics)} 个交易日")
    print(f"  IC 均值 {ic_mean:.4f} | ICIR {icir:.2f} | 正占比 {pos_ratio:.0%}")

    # 分层（5 层，日频截面）
    m["layer"] = m.groupby("date")["factor"].rank(pct=True).apply(
        lambda p: min(int(p * 5) + 1, 5))
    layer_ret = m.groupby(["date", "layer"])["ret"].mean().unstack("layer")
    # 多空日频收益
    ls = layer_ret[5] - layer_ret[1]
    ls_annual = ls.mean() * 250 * 100
    print(f"\n[分层日频回测]")
    print(f"  L5 年化 {(layer_ret[5].mean()*250*100):.2f}%")
    print(f"  L1 年化 {(layer_ret[1].mean()*250*100):.2f}%")
    print(f"  多空年化 {ls_annual:.2f}%")
    print(f"  多空 Sharpe {(ls.mean()/ls.std()*(250**0.5)):.2f}")

    # 存盘
    out_path = OUT_DIR / f"{task}_daily_factor_{factor_col}_{model}_{pool}.parquet"
    f.to_parquet(out_path, compression="snappy")
    print(f"\n日频面板已存: {out_path}")

    # 评估存盘
    out_txt = OUT_DIR / f"{task}_daily_eval_{factor_col}_{model}_{pool}.txt"
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(f"== {task.upper()} ({model}) 日频因子面板（{factor_col}）==\n")
        fh.write(f"面板: {len(f)} 行, 覆盖 {f.index.get_level_values('code').nunique()} 只\n")
        fh.write(f"日均覆盖: {cov.mean():.0f} 只 (中位 {cov.median():.0f})\n\n")
        fh.write(f"[日频 RankIC] {len(ics)} 个交易日\n")
        fh.write(f"  IC 均值 {ic_mean:.4f} | ICIR {icir:.2f} | 正占比 {pos_ratio:.0%}\n\n")
        fh.write(f"[分层日频回测]\n")
        fh.write(f"  L5 年化 {(layer_ret[5].mean()*250*100):.2f}%\n")
        fh.write(f"  L1 年化 {(layer_ret[1].mean()*250*100):.2f}%\n")
        fh.write(f"  多空年化 {ls_annual:.2f}%\n")
        fh.write(f"  多空 Sharpe {(ls.mean()/ls.std()*(250**0.5)):.2f}\n")
    print(f"评估已存: {out_txt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sue", choices=["sue", "fadt"])
    ap.add_argument("--model", default="xgb", choices=["xgb", "logit"])
    ap.add_argument("--pool", default="hs300", choices=["hs300", "zz1000"])
    ap.add_argument("--factor", default="sue0", choices=["sue0", "ar"])
    ap.add_argument("--lookback-days", type=int, default=5)
    args = ap.parse_args()
    main(args.task, args.model, args.pool, args.factor, args.lookback_days)
