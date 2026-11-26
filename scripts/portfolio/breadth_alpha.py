"""广度档的真超额：换掉「上证指数」这个虚高基准（2026-11-25）。

`reports/yq_core/README.md` 对外报的超额是相对**上证指数**（`Config.benchmarks()["all_a"]`
= 000001.SH，2019-2026 年化 6.54%）。这个基准同时错在两个方向：

- 上证是**价格指数**（不含股息），而三档净值是复权口径 ⇒ 基准被低估约 2%/年；
- 上证是**市值加权 + 沪市单市场**，而策略是**全市场等权选股** ⇒ 风格完全不匹配。

本脚本把基准换成五类，看超额还剩多少：

===================  ==========================================  ============
基准                  构造                                          用途
===================  ==========================================  ============
``sse``               上证指数 000001.SH（既有口径）                  与对外数字对齐
``eqw_daily``         全A 日频等权（``_base/bench_eqw``，零成本）      上界，不可实现
``eqw_monthly``       **全A 月频等权 buy-and-hold**（本脚本自建）      **公平基准（毛）**
``eqw_monthly_net``   同上，扣价格漂移换手成本                         **公平基准（净）**
``csi300/500/1000``   000300 / 000905 / 000852（价格指数）            常见参照
===================  ==========================================  ============

``eqw_monthly`` 是本节的主角：月初等权买入、月内持有、月末换仓。等权 buy-and-hold 的
月收益**精确等于**个股月收益的截面均值（金额加权后每只都是 1/N），所以不需要模拟。
再按「价格漂移导致的实际换手」扣成本，得到一个**同样月频、同样换手约束、零 alpha**
的可实现基准。

用法::

    python -m scripts.portfolio.breadth_alpha

产物：``reports/breadth_alpha/{bench_compare.csv,excess_stats.csv,yearly_excess.csv}``。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.cache_helpers import load_index_returns  # noqa: E402
from scripts.evaluation import caliber as cal  # noqa: E402

OUT = ROOT / "reports" / "breadth_alpha"
NAV_CSV = ROOT / "reports" / "yq_core" / "nav.csv"
TIER = "yqcore-breadth"
LO, HI = "2019-01-02", "2026-09-07"
COST_SIDE = 0.00135  # 与生产档一致的 13.5bp 单边


def month_last(idx: pd.DatetimeIndex) -> pd.Series:
    """每个自然月的**实际最后交易日**（``resample("ME")`` 给的是日历月末，会错位）。"""
    per = idx.to_period("M")
    return pd.Series(idx, index=per).groupby(level=0).last()


def to_monthly(daily: pd.Series) -> pd.Series:
    """日收益 → 月收益，索引换成实际最后交易日。"""
    per = daily.index.to_period("M")
    m = (1 + daily.fillna(0.0)).groupby(per).prod() - 1
    m.index = month_last(daily.index).reindex(m.index).to_numpy()
    m.index = pd.DatetimeIndex(m.index)
    return m


def eqw_monthly(grid: pd.DatetimeIndex) -> tuple[pd.Series, float]:
    """全A 月频等权 buy-and-hold 的毛收益（对齐 ``grid``）+ 月均单边换手。

    月收益 = 个股**月末对月末**收益的截面均值（等权 buy-and-hold 的精确解）。
    🚨 不能用「当月首日收盘 / 当月末日收盘」：那会漏掉「上月末 → 本月首个交易日」
    这一段，8 年累计 92 个交易日，实测低估约多少见 ``_diag``。

    换手 = 月初等权 1/N、月内价格漂移后的权重与 1/N 的单边偏离。
    """
    px = pd.read_parquet(cal.BASE_DIR / "close_adj.parquet").loc[LO:HI].ffill()
    last = px.groupby(px.index.to_period("M")).last()
    last.index = month_last(px.index).reindex(last.index).to_numpy()
    last.index = pd.DatetimeIndex(last.index)
    r = last.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    bench = r.mean(axis=1)

    # 月初等权 → 月末漂移权重 → 再平衡的单边换手
    gross = 1.0 + r
    w_end = gross.div(gross.sum(axis=1), axis=0)
    n = gross.notna().sum(axis=1)
    turn = float((w_end.sub(1.0 / n, axis=0).abs().sum(axis=1) / 2.0).mean())
    del px, last, gross, w_end
    return bench.reindex(grid), turn


def capm(strat: pd.Series, bench: pd.Series) -> dict:
    """单因子回归 ``r_s = α + β·r_b + ε``（月度，无无风险利率）。

    简单超额（``strat − bench``）混了 ``(β−1)·r_b``：本档 β≈0.82，牛市里 β 拖累会
    把真实 α 压掉一半以上。**判「有没有 alpha」看这里的 α 与 t，不看简单超额。**
    """
    d = pd.concat([strat.rename("s"), bench.rename("b")], axis=1).dropna()
    X = np.column_stack([np.ones(len(d)), d["b"].to_numpy()])
    y = d["s"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    cov = (resid @ resid / (n - k)) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    ex = d["s"] - d["b"]
    return {
        "alpha_m": float(coef[0]),
        "alpha_ann": float((1 + coef[0]) ** 12 - 1),
        "t_alpha": float(coef[0] / se[0]),
        "beta": float(coef[1]),
        "t_beta": float(coef[1] / se[1]),
        "r2": float(1 - resid.var() / y.var()),
        "n": int(n),
        "corr_excess_bench": float(np.corrcoef(ex, d["b"])[0, 1]),
        "excess_when_down": float(ex[d["b"] < 0].mean()),
        "excess_when_up": float(ex[d["b"] > 0].mean()),
        "n_down": int((d["b"] < 0).sum()),
        "n_up": int((d["b"] > 0).sum()),
    }


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    dd = (1 + r).cumprod()
    return {
        "ann": float(ann),
        "vol": float(vol),
        "sharpe": float(ann / vol) if vol > 0 else float("nan"),
        "max_dd": float((dd / dd.cummax() - 1).min()),
        "n_month": int(len(r)),
    }


def main() -> int:
    nav = pd.read_csv(NAV_CSV, index_col=0, parse_dates=True)[TIER].loc[LO:HI]
    strat = to_monthly(nav.pct_change(fill_method=None).fillna(0.0)).iloc[1:]
    grid = strat.index

    bench: dict[str, pd.Series] = {}
    for code, name in [("000001.SH", "sse"), ("000300.SH", "csi300"),
                       ("000905.SH", "csi500"), ("000852.SH", "csi1000")]:
        s = load_index_returns(code, begin=int(LO.replace("-", "")),
                               end=int(HI.replace("-", "")))
        if s is None:
            print(f"跳过 {name}（缓存缺失）")
            continue
        bench[name] = to_monthly(s).reindex(grid)

    eqw_d = pd.read_parquet(cal.BASE_DIR / "bench_eqw.parquet")["ret"]
    bench["eqw_daily"] = to_monthly(eqw_d.loc[LO:HI]).reindex(grid)

    m_eqw, turn = eqw_monthly(grid)
    cost_m = turn * 2 * COST_SIDE
    bench["eqw_monthly"] = m_eqw
    bench["eqw_monthly_net"] = m_eqw - cost_m
    print(f"全A月频等权：换手 {turn:.2%}/月 → 成本 {cost_m * 100:.3f}%/月 "
          f"= {cost_m * 12 * 100:.2f}%/年")

    rows = [{"bench": "strategy", **stats(strat)}]
    for k, v in bench.items():
        rows.append({"bench": k, **stats(v)})
    df = pd.DataFrame(rows)

    ex = []
    for k, v in bench.items():
        d = (strat - v).dropna()
        sd = d.std()
        ex.append({
            "vs": k,
            "ann_diff": stats(strat)["ann"] - stats(v)["ann"],
            "excess_ann": float((1 + d).prod() ** (12 / len(d)) - 1),
            "te": float(sd * np.sqrt(12)),
            "ir": float(d.mean() / sd * np.sqrt(12)) if sd > 0 else np.nan,
            "t": float(d.mean() / sd * np.sqrt(len(d))) if sd > 0 else np.nan,
            "win": float((d > 0).mean()),
            "n": int(len(d)),
        })
    exdf = pd.DataFrame(ex)

    # 单因子回归：简单超额混了 (β−1)·r_b，判 alpha 看这里
    cm = capm(strat, m_eqw)
    half = len(strat) // 2
    cm["alpha_1st_half"] = capm(strat.iloc[:half], m_eqw.iloc[:half])["alpha_m"]
    cm["t_alpha_1st_half"] = capm(strat.iloc[:half], m_eqw.iloc[:half])["t_alpha"]
    cm["alpha_2nd_half"] = capm(strat.iloc[half:], m_eqw.iloc[half:])["alpha_m"]
    cm["t_alpha_2nd_half"] = capm(strat.iloc[half:], m_eqw.iloc[half:])["t_alpha"]

    yr = pd.DataFrame({"strategy": (1 + strat).groupby(strat.index.year).prod() - 1})
    for k, v in bench.items():
        yr[k] = (1 + v).groupby(v.index.year).prod() - 1
        yr[f"ex_{k}"] = yr["strategy"] - yr[k]

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "bench_compare.csv", index=False)
    exdf.to_csv(OUT / "excess_stats.csv", index=False)
    yr.to_csv(OUT / "yearly_excess.csv")
    pd.DataFrame([cm]).to_csv(OUT / "capm.csv", index=False)

    print("\n=== 基准对照（月频）")
    print(df.round(4).to_string(index=False))
    print("\n=== 超额")
    print(exdf.round(4).to_string(index=False))
    print("\n=== CAPM（vs 全A 月频等权，毛）")
    for k, v in cm.items():
        print(f"  {k:22s} {v:,.4f}")
    print("\n=== 分年")
    print(yr[["strategy", "eqw_monthly_net", "ex_eqw_monthly_net",
              "eqw_daily", "ex_eqw_daily"]].round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
