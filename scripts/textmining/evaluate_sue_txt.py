"""
SUE.txt 因子评估（对齐华泰 AI 51）
=================================

评估内容：
1. 因子面板统计：覆盖度（每月有因子的股票数）。
2. 单因子分层回测：分 5 层（研报 AI 51 用分 5 层），基准中证500，
   月度调仓等权，考察分层单调性与多头第 1 层年化。
3. IC 检验：月度 RankIC 均值/ICIR（Newey-West t 值可复用 robust_stats）。
4. 与 2 日 AR 因子对比：研报结论——SUE.txt 显著强于 2 日 AR 因子。

用法：
    python -m scripts.textmining.evaluate_sue_txt --model xgb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\YuriQuant")
OUT_DIR = Path(r"E:\YuriQuant\reports\textmining")


def load_factor(model: str, pool: str = "hs300") -> pd.DataFrame:
    p = OUT_DIR / f"sue_txt_factor_{model}_{pool}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} 不存在，先跑 train_sue_txt")
    f = pd.read_parquet(p)
    f["date"] = pd.to_datetime(f.index.get_level_values("date"))
    f["code"] = f.index.get_level_values("code")
    return f.reset_index(drop=True)


def load_next_ret(begin: int = 20190101, end: int = 20261231) -> pd.DataFrame:
    """个股下月收益（月度截面，用于分层回测收益计算）。

    直接读 daily_{pool}.parquet 缓存（不调 DataCache.get_daily_kline，
    避免 calendar 增量写入触发 PermissionError——沙箱对 e:/data 写锁）。
    """
    from config import Config
    import json
    daily = pd.read_parquet(
        Path(str(Config.cache()["root"]).replace("//", "/")) / "daily_hs300.parquet")
    daily = daily[daily.index.get_level_values("date") >= pd.Timestamp(str(begin))]
    close = daily["close"].unstack("code")  # date × code
    # 月末截面：月内最后一个交易日的 close → 下月收益
    monthly_last = close.groupby(close.index.to_period("M")).last()
    ret = monthly_last.pct_change().shift(-1)  # 下月收益
    ret = ret.stack().rename("ret").reset_index()
    ret.columns = ["month", "code", "ret"]
    ret["date"] = ret["month"].dt.to_timestamp() + pd.offsets.MonthEnd(0)
    return ret


def stratified_backtest(factor: pd.DataFrame, n_layers: int = 5,
                        ret: pd.DataFrame | None = None,
                        min_stocks: int = 10) -> pd.DataFrame:
    """分 n 层回测（月度调仓等权，基准中证500）。

    min_stocks: 每月因子覆盖股票数低于该值的月份剔除（覆盖度过低时分层噪声大）。
    """
    if ret is None:
        ret = load_next_ret()
    m = factor.merge(ret[["date", "code", "ret"]], on=["date", "code"], how="inner")
    # 覆盖度过滤
    cov_n = m.groupby("date")["code"].nunique()
    valid_dates = cov_n[cov_n >= min_stocks].index
    m = m[m["date"].isin(valid_dates)]

    # 分层
    m["layer"] = m.groupby("date")["factor"].rank(pct=True).apply(
        lambda p: min(int(p * n_layers) + 1, n_layers))

    # 中证500月度收益（基准）
    bench = load_bench_ret()

    # 每层月度收益均值
    layer_ret = m.groupby(["date", "layer"])["ret"].mean().unstack("layer")
    layer_ret = layer_ret.join(bench, how="left")

    # 年化（月频复利）
    def _annual(col):
        s = col.dropna()
        if len(s) < 12:
            return np.nan
        nav = (1 + s).prod()
        return nav ** (12 / len(s)) - 1

    rows = []
    for col in layer_ret.columns:
        if col == "bench":
            continue
        rows.append({"layer": f"L{col}", "annual_ret": _annual(layer_ret[col]),
                     "excess": _annual(layer_ret[col]) - _annual(layer_ret["bench"]),
                     "n_months": int(layer_ret[col].notna().sum())})
    rows.append({"layer": "L1-L5",
                 "annual_ret": _annual(layer_ret[1]) - _annual(layer_ret[n_layers]),
                 "excess": np.nan,
                 "n_months": int(layer_ret[1].notna().sum())})
    out = pd.DataFrame(rows)
    return out


def load_bench_ret() -> pd.Series:
    """中证500月度收益（date → bench_ret）。"""
    from config import Config
    daily = pd.read_parquet(
        Path(str(Config.cache()["root"]).replace("//", "/")) / "daily_hs300.parquet")
    bench = daily[daily.index.get_level_values("code") == "000905.SH"]["close"]
    bench.index = bench.index.get_level_values("date")
    mlast = bench.groupby(bench.index.to_period("M")).last()
    bret = mlast.pct_change(fill_method=None).shift(-1)  # 与个股 ret 对齐：T 月末→T+1 月末
    bret.index = bret.index.to_timestamp() + pd.offsets.MonthEnd(0)
    return bret.rename("bench")


def rank_ic(factor: pd.DataFrame, ret: pd.DataFrame | None = None) -> pd.DataFrame:
    """月度 RankIC。"""
    if ret is None:
        ret = load_next_ret()
    m = factor.merge(ret[["date", "code", "ret"]], on=["date", "code"], how="inner")
    ics = m.groupby("date").apply(
        lambda g: g["factor"].corr(g["ret"], method="spearman"), include_groups=False)
    ics = ics.dropna()
    return pd.DataFrame({
        "ic_mean": ics.mean(), "ic_std": ics.std(),
        "icir": ics.mean() / ics.std() if ics.std() > 0 else np.nan,
        "n_months": len(ics), "positive_ratio": (ics > 0).mean(),
    }, index=[0])


def coverage(factor: pd.DataFrame) -> pd.DataFrame:
    g = factor.groupby(factor["date"]).size()
    return pd.DataFrame({
        "n_stocks_mean": g.mean(), "n_stocks_median": g.median(),
        "n_months": len(g), "min": g.min(), "max": g.max(),
    }, index=[0])


def main(model: str = "xgb", pool: str = "hs300"):
    f = load_factor(model, pool)
    print(f"== SUE.txt ({model}) 因子评估 ==")
    print(f"因子行数: {len(f)}, 覆盖 {f['code'].nunique()} 只, "
          f"{f['date'].min().date()} ~ {f['date'].max().date()}")

    cov = coverage(f)
    print(f"\n[覆盖度] 月均 {cov['n_stocks_mean'].iloc[0]:.0f} 只, "
          f"中位 {cov['n_stocks_median'].iloc[0]:.0f}, "
          f"区间 [{cov['min'].iloc[0]:.0f}, {cov['max'].iloc[0]:.0f}], "
          f"{cov['n_months'].iloc[0]} 个月")

    print("\n[分层回测 5 层]（月度调仓等权，下月收益）")
    sb = stratified_backtest(f, n_layers=5)
    print(sb.to_string(index=False))

    print("\n[RankIC]")
    ic = rank_ic(f)
    print(f"IC 均值 {ic['ic_mean'].iloc[0]:.4f} | ICIR {ic['icir'].iloc[0]:.2f} | "
          f"正占比 {ic['positive_ratio'].iloc[0]:.0%} | {ic['n_months'].iloc[0]} 个月")

    # 与 2 日 AR 因子对比（研报结论：SUE.txt 强于 AR）
    samples = pd.read_parquet(OUT_DIR / "sue_txt_samples.parquet")
    samples["event_date"] = pd.to_datetime(samples["event_date"])
    samples["month"] = samples["event_date"].dt.to_period("M")
    ar_f = samples.groupby(["month", "code"])["ar"].mean().reset_index()
    ar_f["date"] = ar_f["month"].dt.to_timestamp() + pd.offsets.MonthEnd(0)
    m2 = ar_f.merge(f[["date", "code", "factor"]], on=["date", "code"], how="inner")
    if len(m2) > 50:
        corr = m2["factor"].corr(m2["ar"])
        print(f"\n[与 2 日 AR 因子相关] {corr:.4f}（研报：SUE.txt 显著强于 AR）")

    out = OUT_DIR / f"sue_txt_eval_{model}_{pool}.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"== SUE.txt ({model}) 因子评估 ==\n")
        fh.write(f"覆盖度: {cov.to_string()}\n")
        fh.write(f"分层: \n{sb.to_string()}\n")
        fh.write(f"RankIC: {ic.to_string()}\n")
        if len(m2) > 50:
            fh.write(f"与AR相关: {corr:.4f}\n")
    print(f"\n评估已存: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgb")
    ap.add_argument("--pool", default="hs300", choices=["hs300", "zz1000"])
    args = ap.parse_args()
    main(args.model, args.pool)
