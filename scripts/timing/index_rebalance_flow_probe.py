"""指数调样 + ETF 申赎流方向预测探针（纯择时线第九篇，reports/index_rebalance_flow_probe/）。

问题：指数调样与 ETF 申赎是两条「机制性强制流」。它们能不能预测**大盘方向**？

两条线分开检验，因为它们的作用对象不同：

A. **ETF 申赎流**（主判据）：宽基 ETF 的份额变化 = 一级市场净申购/赎回，
   是真实资金流。信号 = 20 日份额变化率的因果 z（252 日），
   `z>+1` 记为申购潮、`z<−1` 记为赎回潮。预注册方向 = **逆向**
   （赎回潮后现货涨、申购潮后跌，即 ETF 资金是「聪明钱反向」）。
   标的：510300→hs300、510500→zz500、512100→zz1000（主判据），
   510050/159915/588000/159901 为控制品种。

B. **指数调样**（信息行，不判指数方向门）：成分表只有生效日（in_date/out_date），
   没有公告日。用 `生效日 − 14 自然日` 作公告代理（保守：真实公告通常更早），
   做「公告代理 → 生效 → 生效+20」的个股事件研究。调样流是**截面**的
   （调入/调出个股），指数点位本身不因换样本而跳变，所以它原则上不是
   指数方向信号；这里用数字确认这一点，并量化调样流的真实幅度。

预注册门（A 线，跑前定死）：

  G1 主门 h=20：赎回潮(z<−1) − 申购潮(z>+1) 的前瞻 20 日收益差 >0，
     HAC-t ≥2.0，hs300/zz500/zz1000 ≥2/3。
  G2 短窗同向：h=1 与 h=5 的差 >0，≥2/3。
  G3 两半窗：2016–2020 / 2021+ 的 h=20 差同号 >0，≥2/3。
  G4 循环平移安慰剂：z 随机平移 200 次，实际 |HAC-t| 的 p≤0.05，≥2/3。
  G5 可交易性：`pos=1{z<0}`（净赎回时做多）长平，30bp 超额 vs 半仓 >0 且
     NW-t ≥2.0，≥2/3。
  G6 非本地伪影：h=1 日内 open→close 分量差 >0（≥2/3）；控制 20 日动量后
     z 的 HAC-t ≤−1.5（≥2/3）。

判读：G1–G4 全过 ⇒ ETF 流含方向信息（候选）；G1 不过 ⇒ 判负/信息不足。
B 线只报「调样流是否真实、是否影响指数方向」，不参与 A 线判门。

口径：份额按 ``ann_date`` 做 PIT（t 日只知 ann_date ≤ t 的最新一期），
前瞻收益从 T+1 open 起算；h=1 即 T+1 日内 open→close。份额数据止 2026-10-24，
样本末端以份额表为准（不把 ffill 出来的假 0 变化当信号）。成本 10bp 主 / 30bp 门槛。

用法::

    python -m scripts.timing.index_rebalance_flow_probe --self-check
    python -m scripts.timing.index_rebalance_flow_probe
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest.metrics import annual_return, max_drawdown, sharpe_ratio
from config import Config
from data.futures import load_spot
from scripts.timing.futures_basis_probe import hac_ols, placebo_p, slope_t
from scripts.timing.hf_vol_timing_probe import causal_z, run_continuous
from scripts.timing.shell import ROOT
from stats.robust_stats import nw_tstat

OUT = ROOT / "reports" / "index_rebalance_flow_probe"
CACHE = Path(str(Config.cache()["root"]).replace("//", "/"))
FUND_SHARE = CACHE / "fund_share.parquet"
DAILY_ALL_A = CACHE / "daily_all_a.parquet"

#: 宽基 ETF → 标的指数。前三只为主判据（份额池里对应最大宽基）。
ETF_MAP = {
    "510300.SH": ("000300.SH", "hs300"),
    "510500.SH": ("000905.SH", "zz500"),
    "512100.SH": ("000852.SH", "zz1000"),
    "510050.SH": ("000016.SH", "sse50"),
    "159915.SZ": ("399006.SZ", "chinext"),
    "588000.SH": ("000688.SH", "star50"),
    "159901.SZ": ("399330.SZ", "sz100"),
}
MAIN = ("510300.SH", "510500.SH", "512100.SH")
BEGIN = "2016-01-01"
HALF = "2021-01-01"
SHARE_END = pd.Timestamp("2026-10-24")   # 份额表末端（不延伸到 11 月）
ZWIN = 252
DEEP, RICH = -1.0, 1.0
HORIZONS = (1, 5, 20)
COSTS = (10.0, 30.0)
PLACEBO_N = 200
SEED = 20261123

REBAL_INDEX = {"000300.SH": "hs300", "000905.SH": "zz500", "000852.SH": "zz1000"}


# ---------------------------------------------------------------- 统计工具
def fwd_ret(open_px: pd.Series, close_px: pd.Series, h: int) -> pd.Series:
    """T+1 open → T+h close（signal 在 t 收盘可知）。"""
    return close_px.shift(-h) / open_px.shift(-1) - 1.0


def state_diff(y: pd.Series, z: pd.Series, maxlags: int) -> tuple[float, float, int]:
    """赎回潮(z<−1) − 申购潮(z>+1) 的前瞻收益差 + HAC-t。"""
    out = (z < DEEP).fillna(False)
    inn = (z > RICH).fillna(False)
    sel = out | inn
    d = pd.DataFrame({"y": y, "out": out.astype(float)}).loc[sel].dropna()
    if len(d) < 20 or d["out"].nunique() < 2:
        return float("nan"), float("nan"), int(len(d))
    res = sm.OLS(d["y"], sm.add_constant(d[["out"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.params["out"]), float(res.tvalues["out"]), int(len(d))


def state_table(y: pd.Series, z: pd.Series, h: int) -> pd.DataFrame:
    out, inn = z < DEEP, z > RICH
    neutral = z.notna() & ~out & ~inn
    rows = []
    for name, mask in (("all", z.notna()), ("redeem(z<-1)", out),
                       ("neutral", neutral), ("subscribe(z>1)", inn)):
        v = y[mask].dropna()
        rows.append({"state": name, "n": len(v), "mean": float(v.mean()) if len(v) else np.nan,
                     "hit": float((v > 0).mean()) if len(v) else np.nan,
                     "ann": float(v.mean() * 252 / h) if len(v) else np.nan})
    return pd.DataFrame(rows)


def half_ret(open_px: pd.Series) -> pd.Series:
    return run_continuous(pd.Series(0.5, index=open_px.index), open_px, 0.0)["ret"]


def arm_row(pos: pd.Series, open_px: pd.Series, cost: float, bench: pd.Series,
            begin: str = BEGIN) -> dict:
    m = run_continuous(pos, open_px, cost)
    r = m["ret"].loc[begin:]
    b = bench.loc[begin:]
    d = (r - b).dropna()
    yrs = len(r) / 252.0
    return {"annual": annual_return(r), "sharpe": sharpe_ratio(r), "mdd": max_drawdown(r),
            "avg_pos": float(pos.loc[begin:].mean()),
            "turn_yr": m["turnover"] / max(yrs, 1e-9),
            "excess": annual_return(r) - annual_return(b),
            "nw_t": float(nw_tstat(d)[0]), "n": len(r),
            "h1": annual_return(r.loc[:HALF]) - annual_return(b.loc[:HALF]),
            "h2": annual_return(r.loc[HALF:]) - annual_return(b.loc[HALF:])}


# ---------------------------------------------------------------- A 线数据
def load_share_pit(etf: str, cal: pd.DatetimeIndex) -> pd.Series:
    """份额的 PIT 日频序列：t 日只知 ann_date ≤ t 的最新一期。"""
    fs = pd.read_parquet(FUND_SHARE)
    sub = fs.xs(etf, level="code").reset_index()
    sub["ann"] = pd.to_datetime(sub["ann_date"], errors="coerce")
    sub["value"] = sub["total_share"].fillna(sub["fund_share"])
    sub = sub.dropna(subset=["ann", "value"]).sort_values("ann")
    sub = sub.drop_duplicates("ann", keep="last")
    s = pd.Series(sub["value"].to_numpy(), index=pd.DatetimeIndex(sub["ann"]))
    return s.reindex(cal, method="ffill")


def build_flow_panel(etf: str, index_code: str, begin: str = BEGIN,
                     ds=None) -> pd.DataFrame:
    idx = load_spot(index_code, begin=20150101, cache_dir=OUT / "data", ds=ds)
    idx = idx.loc[:SHARE_END]
    cal = idx.index
    share = load_share_pit(etf, cal)
    dsh5 = share / share.shift(5) - 1.0
    dsh20 = share / share.shift(20) - 1.0
    z = causal_z(dsh20, ZWIN)
    panel = pd.DataFrame({"share": share, "dsh5": dsh5, "dsh20": dsh20, "z": z,
                          "index_open": idx["open"], "index_close": idx["close"]})
    return panel.loc[begin:]


# ---------------------------------------------------------------- B 线工具
def _car(stock: pd.Series, index: pd.Series, start, end) -> float:
    s = stock.loc[start:end].dropna()
    if len(s) < 5:
        return float("nan")
    i = index.loc[s.index].ffill()
    return float(s.iloc[-1] / s.iloc[0] - 1.0 - (i.iloc[-1] / i.iloc[0] - 1.0))


def rebalance_study(begin: str = "2016-01-01",
                    ds=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """调样事件研究：公告代理→生效→+20 的个股 CAR，以及指数方向检验。"""
    close_long = pd.read_parquet(DAILY_ALL_A, columns=["close"])["close"]
    events, cross = [], []
    for index_code, name in REBAL_INDEX.items():
        idx = load_spot(index_code, begin=20150101, cache_dir=OUT / "data", ds=ds)
        idx = idx.loc[:SHARE_END]
        cal = idx.index
        cons = pd.read_parquet(CACHE / f"index_constituent_{index_code.replace('.', '_')}.parquet")
        cons["in_date"] = pd.to_datetime(cons["in_date"], errors="coerce")
        cons["out_date"] = pd.to_datetime(cons["out_date"], errors="coerce")
        codes = sorted(set(cons["con_code"]))
        px = close_long[close_long.index.get_level_values("code").isin(codes)]
        px = px.unstack("code").reindex(cal)
        px = px.ffill(limit=5)  # 短暂停牌按前收；长期缺失（未上市/退市）保持 NaN

        add = cons[cons["in_date"].between(pd.Timestamp(begin), SHARE_END)]
        rem = cons[cons["out_date"].between(pd.Timestamp(begin), SHARE_END)]
        # 调出的最后一日 = out_date，生效日 = 其后第一个交易日
        eff_dates = sorted(set(add["in_date"].dropna().tolist()) |
                           {cal[cal.searchsorted(d) + 1] for d in rem["out_date"].dropna()
                            if cal.searchsorted(d) + 1 < len(cal)})
        for eff in eff_dates:
            prev = cal[cal < eff]
            if len(prev) == 0:
                continue
            n_add = add[add["in_date"] == eff]
            out_prev = prev[-1]
            n_del = rem[rem["out_date"] == out_prev]
            if len(n_add) == 0 and len(n_del) == 0:
                continue
            # 只保留常规半年度调样（6/12 月、调入≥5），剔除单票特殊调整
            if eff.month not in (6, 12) or len(n_add) < 5:
                continue
            n_total = int(((cons["in_date"] <= eff) &
                           (cons["out_date"].isna() | (cons["out_date"] >= eff))).sum())
            start_proxy = cal[cal <= eff - pd.Timedelta(days=14)]
            start_proxy = start_proxy[-1] if len(start_proxy) else prev[0]
            end_pre, start_post = prev[-1], eff
            pos_eff = int(cal.searchsorted(eff))
            end_post = cal[min(pos_eff + 20, len(cal) - 1)]

            def _mean_car(sub: pd.DataFrame) -> float:
                vals = [_car(px[c], idx["close"], start_proxy, end_pre)
                        if start_proxy <= end_pre else np.nan for c in sub["con_code"]
                        if c in px.columns]
                vals = [v for v in vals if np.isfinite(v)]
                return float(np.mean(vals)) if vals else float("nan")

            def _mean_post(sub: pd.DataFrame) -> float:
                vals = [_car(px[c], idx["close"], start_post, end_post)
                        for c in sub["con_code"] if c in px.columns]
                vals = [v for v in vals if np.isfinite(v)]
                return float(np.mean(vals)) if vals else float("nan")

            a_pre, d_pre = _mean_car(n_add), _mean_car(n_del)
            a_post, d_post = _mean_post(n_add), _mean_post(n_del)
            fwd20 = float(idx["close"].loc[end_post] / idx["open"].loc[eff] - 1.0) \
                if eff in idx.index else float("nan")
            events.append({"index": name, "eff_date": eff.date(), "n_add": len(n_add),
                           "n_del": len(n_del), "n_total": n_total,
                           "net_ratio": (len(n_add) - len(n_del)) / max(n_total, 1),
                           "gross_ratio": (len(n_add) + len(n_del)) / max(n_total, 1),
                           "add_pre": a_pre, "del_pre": d_pre,
                           "add_post": a_post, "del_post": d_post,
                           "pre_diff": a_pre - d_pre, "post_diff": a_post - d_post,
                           "index_fwd20": fwd20})
    ev = pd.DataFrame(events)
    if len(ev) == 0:
        return ev, pd.DataFrame()
    for name, g in ev.groupby("index"):
        cross.append({"index": name, "n_events": len(g),
                      "add_pre": g["add_pre"].mean(), "del_pre": g["del_pre"].mean(),
                      "pre_diff": g["pre_diff"].mean(),
                      "add_post": g["add_post"].mean(), "del_post": g["del_post"].mean(),
                      "post_diff": g["post_diff"].mean(),
                      "net_ratio_mean": g["net_ratio"].mean(),
                      "gross_ratio_mean": g["gross_ratio"].mean(),
                      "rho_gross_fwd20": g[["gross_ratio", "index_fwd20"]].corr(method="spearman").iloc[0, 1],
                      "fwd20_mean": g["index_fwd20"].mean(),
                      "fwd20_hit": float((g["index_fwd20"] > 0).mean())})
    return ev, pd.DataFrame(cross)


# ---------------------------------------------------------------- 自检
def self_check(panels: dict[str, pd.DataFrame], events: pd.DataFrame) -> list[str]:
    bad = []
    fs = pd.read_parquet(FUND_SHARE)
    ratio = (pd.to_datetime(fs["ann_date"]) >= pd.to_datetime(fs.index.get_level_values(0)))
    if float(ratio.mean()) < 0.99:
        bad.append(f"份额 ann_date < change_date 占比 {1-ratio.mean():.1%}，PIT 假设破裂")
    for etf, panel in panels.items():
        if panel["z"].notna().sum() < 300:
            bad.append(f"{etf}: z 有效样本 <300")
        if (panel["share"] <= 0).any():
            bad.append(f"{etf}: 份额出现 ≤0")
        if panel.index.max() > SHARE_END + pd.Timedelta(days=1):
            bad.append(f"{etf}: 面板越过份额表末端")
        # 前瞻收益对齐：h=1 = T+1 close/open−1
        if len(panel) > 400:
            i = 300
            manual = panel["index_close"].iloc[i + 1] / panel["index_open"].iloc[i + 1] - 1.0
            got = fwd_ret(panel["index_open"], panel["index_close"], 1).iloc[i]
            if not np.isclose(manual, got, atol=1e-12):
                bad.append(f"{etf}: h=1 前瞻收益错位")
    if len(events) and events["eff_date"].isna().any():
        bad.append("调样事件出现空生效日")
    return bad


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    from data.datasource import create_datasource
    ds = create_datasource()
    panels = {etf: build_flow_panel(etf, meta[0], ds=ds) for etf, meta in ETF_MAP.items()}
    events, cross = rebalance_study(ds=ds)

    if args.self_check:
        bad = self_check(panels, events)
        for b in bad:
            print("SELF-CHECK FAIL:", b)
        if bad:
            return 1
        for etf, meta in ETF_MAP.items():
            p = panels[etf]
            print(f"{etf}→{meta[1]} n={len(p)} z有效 {p['z'].notna().sum()} "
                  f"dsh20 中位 {p['dsh20'].median():+.2%}")
        print(f"调样事件 {len(events)} 条（常规半年度），覆盖 "
              f"{events['index'].nunique() if len(events) else 0} 个指数")
        print("SELF-CHECK PASS（份额 PIT / 份额>0 / 末端闸门 / 前瞻收益对齐 / 事件日）")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    rows, yearly, decomp, orth = [], [], [], []
    for etf, (index_code, name) in ETF_MAP.items():
        p = panels[etf]
        z = p["z"]
        for h in HORIZONS:
            y = fwd_ret(p["index_open"], p["index_close"], h)
            st = state_table(y, z, h)
            st.insert(0, "index", name)
            st.insert(1, "h", h)
            st.to_csv(OUT / f"states_{name}_h{h}.csv", index=False, encoding="utf-8-sig")
            s, st_t = slope_t(y, z, maxlags=max(h, 1))
            diff, diff_t, n = state_diff(y, z, maxlags=max(h, 1))
            obs_t, pval = placebo_p(y, z, maxlags=max(h, 1))
            rows.append({"index": name, "etf": etf, "h": h, "slope": s, "slope_t": st_t,
                         "redeem_minus_sub": diff, "diff_t": diff_t, "n_diff": n,
                         "placebo_abs_t": obs_t, "placebo_p": pval})
            if h == 20:
                yy = pd.DataFrame({"year": p.index.year, "y": y, "z": z}).dropna()
                for yr, g in yy.groupby("year"):
                    out, inn = g["z"] < DEEP, g["z"] > RICH
                    yearly.append({"index": name, "year": int(yr),
                                   "redeem_n": int(out.sum()), "sub_n": int(inn.sum()),
                                   "redeem_mean": float(g.loc[out, "y"].mean()) if out.any() else np.nan,
                                   "sub_mean": float(g.loc[inn, "y"].mean()) if inn.any() else np.nan,
                                   "diff": (float(g.loc[out, "y"].mean())
                                            - float(g.loc[inn, "y"].mean()))
                                   if (out.any() and inn.any()) else np.nan})
        # 分段分解：隔夜 vs 日内（h=1）
        ov = p["index_open"].shift(-1) / p["index_close"] - 1.0
        intra = p["index_close"].shift(-1) / p["index_open"].shift(-1) - 1.0
        for comp, y in (("overnight", ov), ("intraday", intra),
                        ("total", fwd_ret(p["index_open"], p["index_close"], 1))):
            diff, diff_t, n = state_diff(y, z, maxlags=1)
            decomp.append({"index": name, "component": comp,
                           "redeem_minus_sub": diff, "diff_t": diff_t, "n": n})
        # 正交：控制 20 日动量
        mom = p["index_close"] / p["index_close"].shift(20) - 1.0
        res = hac_ols(fwd_ret(p["index_open"], p["index_close"], 20),
                      pd.DataFrame({"z": z, "mom20": mom}), maxlags=20)
        orth.append({"index": name, "z_coef": float(res.params["z"]),
                     "z_t": float(res.tvalues["z"]),
                     "mom_coef": float(res.params["mom20"]),
                     "mom_t": float(res.tvalues["mom20"])})

    pred = pd.DataFrame(rows)
    pred.to_csv(OUT / "predictive_tests.csv", index=False, encoding="utf-8-sig")
    print("\n[表1] ETF 申赎流 z 的预测检验（T+1 open；方向假设=逆向）")
    print(pred.round(4).to_string(index=False))

    pd.DataFrame(yearly).to_csv(OUT / "yearly_redeem_sub.csv", index=False, encoding="utf-8-sig")
    dec = pd.DataFrame(decomp)
    dec.to_csv(OUT / "decomposition_h1.csv", index=False, encoding="utf-8-sig")
    print("\n[表2] h=1 分段分解（赎回−申购）")
    print(dec.round(4).to_string(index=False))
    ort = pd.DataFrame(orth)
    ort.to_csv(OUT / "orthogonality.csv", index=False, encoding="utf-8-sig")
    print("\n[表3] 控制 20 日动量后 z 的系数（预期 <0）")
    print(ort.round(4).to_string(index=False))

    # 可交易性：净赎回做多（逆向）
    real = []
    for etf, (index_code, name) in ETF_MAP.items():
        p = panels[etf]
        pos = (p["z"] < 0).fillna(False).astype(float)
        half = half_ret(p["index_open"])
        hold = run_continuous(pd.Series(1.0, index=p.index), p["index_open"], 0.0)["ret"]
        avg = float(pos.mean())
        exmatch = run_continuous(pd.Series(avg, index=p.index), p["index_open"], 0.0)["ret"]
        # 顺势反向臂（净申购做多），只作信息行
        pos_trend = (p["z"] > 0).fillna(False).astype(float)
        for cost in COSTS:
            r = arm_row(pos, p["index_open"], cost, half)
            rt = arm_row(pos_trend, p["index_open"], cost, half)
            real.append({"index": name, "cost_bp": cost, **r,
                         "ex_vs_hold": r["annual"] - annual_return(hold.loc[BEGIN:]),
                         "ex_vs_exmatch": r["annual"] - annual_return(exmatch.loc[BEGIN:]),
                         "trend_excess": rt["excess"], "trend_nw_t": rt["nw_t"]})
    rt = pd.DataFrame(real)
    rt.to_csv(OUT / "realization.csv", index=False, encoding="utf-8-sig")
    print("\n[表4] 可交易性：redeem 臂 pos=1{z<0}（vs 半仓；trend 列=反向信息行）")
    print(rt.round(4).to_string(index=False))

    # 调样事件研究（信息行，已在入口算好）
    events.to_csv(OUT / "rebalance_events.csv", index=False, encoding="utf-8-sig")
    cross.to_csv(OUT / "rebalance_cross_section.csv", index=False, encoding="utf-8-sig")
    print("\n[表5·信息行] 调样事件研究（公告代理→生效→+20；CAR vs 指数）")
    print(cross.round(4).to_string(index=False))

    # ================= 预注册门（A 线） =================
    gates: list[dict] = []
    main = [ETF_MAP[e][1] for e in MAIN]

    def _by(idx: str, h: int) -> pd.Series:
        return pred[(pred["index"] == idx) & (pred["h"] == h)].iloc[0]

    g1_detail, n1 = [], 0
    for i in main:
        r = _by(i, 20)
        ok = bool(r["redeem_minus_sub"] > 0 and r["diff_t"] >= 2.0)
        n1 += ok
        g1_detail.append(f"{i} 差 {r['redeem_minus_sub']:+.2%}/t {r['diff_t']:.2f}"
                         f" ⇒ {'PASS' if ok else 'FAIL'}")
    g1 = n1 >= 2
    gates.append({"gate": "G1 主门 h=20（赎回−申购>0 且 HAC-t≥2，≥2/3）",
                  "pass": g1, "detail": "；".join(g1_detail)})

    g2_detail, n2 = [], 0
    for i in main:
        ok = all(_by(i, h)["redeem_minus_sub"] > 0 for h in (1, 5))
        n2 += ok
        g2_detail.append(f"{i} h1 {_by(i,1)['redeem_minus_sub']:+.2%}/"
                         f"h5 {_by(i,5)['redeem_minus_sub']:+.2%}"
                         f" ⇒ {'PASS' if ok else 'FAIL'}")
    g2 = n2 >= 2
    gates.append({"gate": "G2 短窗同向（h=1 与 h=5 差>0，≥2/3）",
                  "pass": g2, "detail": "；".join(g2_detail)})

    by_name = {ETF_MAP[e][1]: e for e in MAIN}
    g3_detail, n3, n_eval = [], 0, 0
    for i in main:
        p = panels[by_name[i]]
        y = fwd_ret(p["index_open"], p["index_close"], 20)
        h1 = state_diff(y.loc[:HALF], p["z"].loc[:HALF], 20)[0]
        h2 = state_diff(y.loc[HALF:], p["z"].loc[HALF:], 20)[0]
        if not np.isfinite(h1):
            g3_detail.append(f"{i} 无前半窗，豁免")
            continue
        n_eval += 1
        ok = bool(np.isfinite(h2) and h1 > 0 and h2 > 0)
        n3 += ok
        g3_detail.append(f"{i} 前半 {h1:+.2%}/后半 {h2:+.2%} ⇒ {'PASS' if ok else 'FAIL'}")
    g3 = bool(n_eval >= 2 and n3 == n_eval)
    gates.append({"gate": "G3 两半窗同号（h=20 差>0，≥2/3）",
                  "pass": g3, "detail": "；".join(g3_detail)})

    g4_detail, n4 = [], 0
    for i in main:
        r = _by(i, 20)
        ok = bool(r["placebo_p"] <= 0.05)
        n4 += ok
        g4_detail.append(f"{i} |t| {r['placebo_abs_t']:.2f}/p {r['placebo_p']:.3f}"
                         f" ⇒ {'PASS' if ok else 'FAIL'}")
    g4 = n4 >= 2
    gates.append({"gate": "G4 循环平移安慰剂（p≤0.05，≥2/3）",
                  "pass": g4, "detail": "；".join(g4_detail)})

    g5_detail, n5 = [], 0
    for i in main:
        r = rt[(rt["index"] == i) & (rt["cost_bp"] == 30.0)].iloc[0]
        ok = bool(r["excess"] > 0 and r["nw_t"] >= 2.0)
        n5 += ok
        g5_detail.append(f"{i} 30bp 超额 {r['excess']:+.2%}/NW-t {r['nw_t']:.2f}"
                         f"（均仓 {r['avg_pos']:.0%}）⇒ {'PASS' if ok else 'FAIL'}")
    g5 = n5 >= 2
    gates.append({"gate": "G5 可交易性（z<0 长平，30bp 超额>0 且 NW-t≥2，≥2/3）",
                  "pass": g5, "detail": "；".join(g5_detail)})

    intra_ok = sum(dec[(dec["index"] == i) & (dec["component"] == "intraday")]
                   ["redeem_minus_sub"].iloc[0] > 0 for i in main)
    orth_ok = sum(ort[ort["index"] == i]["z_t"].iloc[0] <= -1.5 for i in main)
    g6 = bool(intra_ok >= 2 and orth_ok >= 2)
    gates.append({"gate": "G6 非本地伪影（日内分量>0 且控动量后 t≤−1.5，各 ≥2/3）",
                  "pass": g6,
                  "detail": f"日内同向 {intra_ok}/3、控动量后显著 {orth_ok}/3"})

    print("\n===== 判定 =====")
    for g in gates:
        print(f"{g['gate']}: {'PASS' if g['pass'] else 'FAIL'}  {g['detail']}")
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False, encoding="utf-8-sig")

    core = bool(g1 and g2 and g3 and g4)
    print("\n判读（预注册）：", "ETF 流含方向信息 → 候选" if core
          else "核心门未全过 → 判负/信息不足")
    print("可交易性：", "30bp 后仍显著" if g5 else "统计信息未转化为可交易超额")
    print("伪影检查：", "非隔夜/反转伪影" if g6 else "存在隔夜或反转伪影")
    print("\n调样线（信息行）：见 rebalance_cross_section.csv；调样是截面流，"
          "指数点位不因换样本跳变，不作为指数方向门。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
