"""股指期货基差方向预测探针（纯择时线第八篇，reports/futures_basis_probe/）。

问题：股指期货基差（期货相对现货的贴水/升水）是否含**大盘方向**的领先信息？
前置结论是「同期交易类指标只描述状态、不预测方向」，基差是唯一还没测过的
机制性变量：它由套保盘/受限融券的强制供需决定，且到期收敛是机械的——但收敛
只影响期货，不影响现货。所以预测现货方向是对「基差是否含情绪/持仓压力」的
干净检验，不是把收敛当 alpha。

信号（全部 ≤t 收盘可知）：``z = causal_z(ann_basis_main, 252)``
  - ``ann_basis_main`` = 主连日化基差 = ``(main_close/spot_close-1)*365/main_days``；
    主连在近月到期前 5 个自然日切次近月，避开到期日年化分母趋 0 的尖峰。
  - 方向假设（预注册）：**逆向**——深贴水（z<−1）后现货涨、深升水（z>+1）后跌。

预注册门（跑前定死）：

  G1 主门（h=20）：deep(z<−1) − rich(z>+1) 的前瞻 20 日收益差 >0，HAC-t ≥2.0，
    在 IF/IC/IM 至少 2/3 品种成立。
  G2 短窗同向：h=5 与 h=1 的胜率差/收益差同向 >0，至少 2/3。
  G3 两半窗：2016–2020 / 2021+ 的 h=20 差值同号 >0，至少 2/3。
  G4 循环平移安慰剂：z 随机平移 200 次，实际 |HAC-t| 的 p≤0.05，至少 2/3。
  G5 可交易性：``pos = 1{z<0}`` 长平，30bp 超额（vs 半仓）>0 且 NW-t ≥2.0，至少 2/3。
  G6 非本地伪影：h=1 的「日内 open→close」分量 deep−rich >0，至少 2/3；
    且控制 20 日动量后 z 的 HAC-t ≤ −1.5（不是反转的代理），至少 2/3。

判读：G1–G4 全过 ⇒ 基差含方向信息（候选）；G1 不过 ⇒ 判负/信息不足。G5 不过
⇒ 只算统计信息、不具可交易性。G6 不过 ⇒ 是隔夜/反转伪影，不采纳。

口径：spot 收盘与期货收盘同为 15:00；前瞻收益从 **T+1 open** 起算（可交易时点），
h=1 即 T+1 日内 open→close，天然剔除隔夜缺口；h=5/20 为 T+1 open→T+h close。
成本 10bp 主 / 30bp 门槛。样本 2016-01-01 起（IC 2015-04、IM 2022-07 上市）。

用法::

    python -m scripts.timing.futures_basis_probe --self-check
    python -m scripts.timing.futures_basis_probe
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest.metrics import annual_return, max_drawdown, sharpe_ratio
from data.futures import (CFFEX_INDEX, DEFAULT_CACHE, build_basis, coverage,
                          fetch_contracts, load_futures_daily, load_spot)
from scripts.timing.hf_vol_timing_probe import causal_z, run_continuous
from scripts.timing.shell import ROOT
from stats.robust_stats import nw_tstat

OUT = ROOT / "reports" / "futures_basis_probe"
PRODUCTS = {
    "IF": {"spot": CFFEX_INDEX["IF"], "name": "hs300"},
    "IC": {"spot": CFFEX_INDEX["IC"], "name": "zz500"},
    "IM": {"spot": CFFEX_INDEX["IM"], "name": "zz1000"},
    "IH": {"spot": CFFEX_INDEX["IH"], "name": "sse50"},   # 控制品种，不进主判据
}
MAIN = ("IF", "IC", "IM")
BEGIN = "2016-01-01"
HALF = "2021-01-01"
HORIZONS = (1, 5, 20)
ZWIN = 252
DEEP, RICH = -1.0, 1.0
ROLL_DAYS = 5
COSTS = (10.0, 30.0)
PLACEBO_N = 200
SEED = 20261123


# ---------------------------------------------------------------- 数据
def build_data(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """拉/读期货与现货，返回每品种基差面板（含 spot_open/spot_close）。

    全流程只登录一次 SDK（``ds`` 透传），避免同进程多连接触发账号连接上限。
    """
    from data.datasource import create_datasource

    today = int(dt.date.today().strftime("%Y%m%d"))
    ds = create_datasource()
    codes = fetch_contracts(20140101, today, products=tuple(PRODUCTS), ds=ds)
    fut = load_futures_daily(codes, 20140101, today, cache_path=DEFAULT_CACHE,
                             refresh=refresh, ds=ds)
    panels: dict[str, pd.DataFrame] = {}
    for p, meta in PRODUCTS.items():
        spot = load_spot(meta["spot"], begin=20140101, cache_dir=OUT / "data",
                         refresh=refresh, ds=ds)
        panel = build_basis(fut, p, spot["close"], roll_days=ROLL_DAYS)
        panel["spot_open"] = spot["open"].reindex(panel.index)
        panel["spot_close"] = spot["close"].reindex(panel.index)
        panels[p] = panel
    return panels


# ---------------------------------------------------------------- 统计工具
def fwd_ret(panel: pd.DataFrame, h: int) -> pd.Series:
    """T+1 open → T+h close 的前瞻收益（signal 在 t 收盘可知）。"""
    entry = panel["spot_open"].shift(-1)
    exit_ = panel["spot_close"].shift(-h)
    return exit_ / entry - 1.0


def hac_ols(y: pd.Series, x: pd.DataFrame, maxlags: int):
    d = pd.concat([y.rename("y"), x], axis=1).dropna()
    res = sm.OLS(d["y"], sm.add_constant(d[x.columns])).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return res


def slope_t(y: pd.Series, x: pd.Series, maxlags: int) -> tuple[float, float]:
    res = hac_ols(y, x.to_frame(x.name or "x"), maxlags)
    name = x.name or "x"
    return float(res.params[name]), float(res.tvalues[name])


def state_diff(y: pd.Series, z: pd.Series, maxlags: int) -> tuple[float, float, int]:
    """deep(z<−1) 与 rich(z>+1) 的前瞻收益均值差 + HAC-t（只用两端状态）。"""
    deep = (z < DEEP).fillna(False)
    rich = (z > RICH).fillna(False)
    sel = deep | rich
    d = pd.DataFrame({"y": y, "deep": deep.astype(float)}).loc[sel].dropna()
    if len(d) < 20 or d["deep"].nunique() < 2:
        return float("nan"), float("nan"), int(len(d))
    res = sm.OLS(d["y"], sm.add_constant(d[["deep"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.params["deep"]), float(res.tvalues["deep"]), int(len(d))


def placebo_p(y: pd.Series, z: pd.Series, maxlags: int,
              n: int = PLACEBO_N, seed: int = SEED) -> tuple[float, float]:
    """循环平移 z（保留自相关），返回 (实际 |t|, 单侧 p)。"""
    d = pd.concat([y.rename("y"), z.rename("z")], axis=1).dropna()
    if len(d) < 300:
        return float("nan"), float("nan")
    obs = abs(slope_t(d["y"], d["z"], maxlags)[1])
    rng = np.random.default_rng(seed)
    ts = []
    vals = d["z"].to_numpy()
    for _ in range(n):
        k = int(rng.integers(60, len(vals) - 60))
        zs = pd.Series(np.roll(vals, k), index=d.index, name="z")
        ts.append(abs(slope_t(d["y"], zs, maxlags)[1]))
    return obs, float(np.mean(np.asarray(ts) >= obs))


def state_table(y: pd.Series, z: pd.Series, h: int) -> pd.DataFrame:
    deep, rich = z < DEEP, z > RICH
    neutral = z.notna() & ~deep & ~rich
    rows = []
    for name, mask in (("all", z.notna()), ("deep(z<-1)", deep),
                       ("neutral", neutral), ("rich(z>1)", rich)):
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


# ---------------------------------------------------------------- 自检
def self_check(panels: dict[str, pd.DataFrame]) -> list[str]:
    bad = []
    for p, panel in panels.items():
        if panel.index.has_duplicates:
            bad.append(f"{p}: 日期重复")
        if panel["spot_close"].isna().any():
            bad.append(f"{p}: spot_close 有缺失")
        cov = coverage(panel)
        if cov["n"] < 400:
            bad.append(f"{p}: 样本过短 n={cov['n']}")
        b = panel["basis_front"].dropna()
        if len(b) and (b.abs() > 0.20).any():
            bad.append(f"{p}: 近月基差越界 {b.abs().max():.1%}")
        if (panel["days_to_exp"].dropna() < 0).any():
            bad.append(f"{p}: 出现已到期合约仍被选为近月")
        # 前瞻收益对齐：h=1 必须等于 T+1 close/open−1
        if len(panel) <= 1000:
            continue
        t = panel.index[1000]
        i = panel.index.get_loc(t)
        if i + 1 < len(panel):
            manual = panel["spot_close"].iloc[i + 1] / panel["spot_open"].iloc[i + 1] - 1.0
            got = fwd_ret(panel, 1).iloc[i]
            if not np.isclose(manual, got, atol=1e-12):
                bad.append(f"{p}: h=1 前瞻收益错位")
    # 现货与期货末端不能相差超过 7 天（防共享缓存陈旧截断）
    for p, panel in panels.items():
        if len(panel) and panel.index.max() < pd.Timestamp(dt.date.today()) - pd.Timedelta(days=7):
            bad.append(f"{p}: 面板末端 {panel.index.max().date()} 过旧")
    return bad


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    panels = build_data(refresh=args.refresh_data)

    if args.self_check:
        bad = self_check(panels)
        for b in bad:
            print("SELF-CHECK FAIL:", b)
        if bad:
            return 1
        for p in PRODUCTS:
            c = coverage(panels[p])
            print(f"{p} 覆盖 n={c['n']} {c['begin']}→{c['end']} "
                  f"基差缺失 {c['na_basis']:.1%} 区间 [{c['min_basis']:.3f}, {c['max_basis']:.3f}]")
        print("SELF-CHECK PASS（日期/缺失/基差界/到期合约/前瞻收益对齐/末端新鲜度）")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    rows, yearly, decomp, orth = [], [], [], []
    sigs: dict[str, pd.Series] = {}

    for p, meta in PRODUCTS.items():
        panel = panels[p].loc[BEGIN:]
        z = causal_z(panel["ann_basis_main"], ZWIN)
        sigs[p] = z
        fwd = {h: fwd_ret(panel, h) for h in HORIZONS}
        for h in HORIZONS:
            st = state_table(fwd[h], z, h)
            st.insert(0, "index", meta["name"])
            st.insert(1, "h", h)
            st.to_csv(OUT / f"states_{meta['name']}_h{h}.csv", index=False, encoding="utf-8-sig")
            s, st_t = slope_t(fwd[h], z, maxlags=max(h, 1))
            diff, diff_t, n = state_diff(fwd[h], z, maxlags=max(h, 1))
            obs_t, pval = placebo_p(fwd[h], z, maxlags=max(h, 1))
            rows.append({"index": meta["name"], "product": p, "h": h,
                         "slope": s, "slope_t": st_t,
                         "deep_minus_rich": diff, "diff_t": diff_t, "n_diff": n,
                         "placebo_abs_t": obs_t, "placebo_p": pval})
            if h == 20:
                yy = pd.DataFrame({"year": panel.index.year, "y": fwd[h], "z": z}).dropna()
                for yr, g in yy.groupby("year"):
                    deep, rich = g["z"] < DEEP, g["z"] > RICH
                    yearly.append({"index": meta["name"], "year": int(yr),
                                   "deep_n": int(deep.sum()), "rich_n": int(rich.sum()),
                                   "deep_mean": float(g.loc[deep, "y"].mean()) if deep.any() else np.nan,
                                   "rich_mean": float(g.loc[rich, "y"].mean()) if rich.any() else np.nan,
                                   "diff": (float(g.loc[deep, "y"].mean())
                                            - float(g.loc[rich, "y"].mean()))
                                   if (deep.any() and rich.any()) else np.nan})
        # 分段分解：h=1 的隔夜 vs 日内
        ov = panel["spot_open"].shift(-1) / panel["spot_close"] - 1.0
        intra = panel["spot_close"].shift(-1) / panel["spot_open"].shift(-1) - 1.0
        for comp, y in (("overnight", ov), ("intraday", intra), ("total", fwd[1])):
            diff, diff_t, n = state_diff(y, z, maxlags=1)
            decomp.append({"index": meta["name"], "component": comp,
                           "deep_minus_rich": diff, "diff_t": diff_t, "n": n})
        # 正交：控制 20 日动量
        mom = panel["spot_close"] / panel["spot_close"].shift(20) - 1.0
        res = hac_ols(fwd[20], pd.DataFrame({"z": z, "mom20": mom}), maxlags=20)
        orth.append({"index": meta["name"], "z_coef": float(res.params["z"]),
                     "z_t": float(res.tvalues["z"]),
                     "mom_coef": float(res.params["mom20"]),
                     "mom_t": float(res.tvalues["mom20"])})

    pred = pd.DataFrame(rows)
    pred.to_csv(OUT / "predictive_tests.csv", index=False, encoding="utf-8-sig")
    print("\n[表1] 基差 z 的预测检验（T+1 open 起算；HAC-t 按 h 滞后）")
    print(pred.round(4).to_string(index=False))

    pd.DataFrame(yearly).to_csv(OUT / "yearly_deep_rich.csv", index=False, encoding="utf-8-sig")
    dec = pd.DataFrame(decomp)
    dec.to_csv(OUT / "decomposition_h1.csv", index=False, encoding="utf-8-sig")
    print("\n[表2] h=1 收益分段分解（deep−rich）")
    print(dec.round(4).to_string(index=False))
    ort = pd.DataFrame(orth)
    ort.to_csv(OUT / "orthogonality.csv", index=False, encoding="utf-8-sig")
    print("\n[表3] 控制 20 日动量后 z 的系数（预期 <0）")
    print(ort.round(4).to_string(index=False))

    # ================= 可交易性（state arm：z<0 长、否则平） =================
    real = []
    for p, meta in PRODUCTS.items():
        panel = panels[p].loc[BEGIN:]
        pos = (sigs[p] < 0).fillna(False).astype(float)
        half = half_ret(panel["spot_open"])
        hold = run_continuous(pd.Series(1.0, index=panel.index), panel["spot_open"], 0.0)["ret"]
        avg = float(pos.mean())
        exmatch = run_continuous(pd.Series(avg, index=panel.index), panel["spot_open"], 0.0)["ret"]
        for cost in COSTS:
            r = arm_row(pos, panel["spot_open"], cost, half)
            real.append({"index": meta["name"], "cost_bp": cost, **r,
                         "ex_vs_hold": r["annual"] - annual_return(hold.loc[BEGIN:]),
                         "ex_vs_exmatch": r["annual"] - annual_return(exmatch.loc[BEGIN:])})
    rt = pd.DataFrame(real)
    rt.to_csv(OUT / "realization.csv", index=False, encoding="utf-8-sig")
    print("\n[表4] 可交易性：pos=1{z<0}（vs 半仓；另附 vs 持有 / 等暴露）")
    print(rt.round(4).to_string(index=False))

    # ================= 预注册门 =================
    gates: list[dict] = []
    main = [PRODUCTS[p]["name"] for p in MAIN]

    def _by(idx: str, h: int) -> pd.Series:
        return pred[(pred["index"] == idx) & (pred["h"] == h)].iloc[0]

    g1_detail, n1 = [], 0
    for i in main:
        r = _by(i, 20)
        ok = bool(r["deep_minus_rich"] > 0 and r["diff_t"] >= 2.0)
        n1 += ok
        g1_detail.append(f"{i} 差 {r['deep_minus_rich']:+.2%}/t {r['diff_t']:.2f}"
                         f" ⇒ {'PASS' if ok else 'FAIL'}")
    g1 = n1 >= 2
    gates.append({"gate": "G1 主门 h=20（deep−rich>0 且 HAC-t≥2，≥2/3）",
                  "pass": g1, "detail": "；".join(g1_detail)})

    g2_detail, n2 = [], 0
    for i in main:
        ok = all(_by(i, h)["deep_minus_rich"] > 0 for h in (1, 5))
        n2 += ok
        g2_detail.append(f"{i} h1 {_by(i,1)['deep_minus_rich']:+.2%}/"
                         f"h5 {_by(i,5)['deep_minus_rich']:+.2%}"
                         f" ⇒ {'PASS' if ok else 'FAIL'}")
    g2 = n2 >= 2
    gates.append({"gate": "G2 短窗同向（h=1 与 h=5 差>0，≥2/3）",
                  "pass": g2, "detail": "；".join(g2_detail)})

    by_name = {PRODUCTS[p]["name"]: p for p in MAIN}
    g3_detail, n3, n_eval = [], 0, 0
    for i in main:
        panel = panels[by_name[i]].loc[BEGIN:]
        z = sigs[by_name[i]]
        y = fwd_ret(panel, 20)
        h1 = state_diff(y.loc[:HALF], z.loc[:HALF], 20)[0]
        h2 = state_diff(y.loc[HALF:], z.loc[HALF:], 20)[0]
        if not np.isfinite(h1):  # 上市晚（IM 2022-07），前半窗无样本 ⇒ 豁免
            g3_detail.append(f"{i} 无前半窗（上市晚），豁免")
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
                   ["deep_minus_rich"].iloc[0] > 0 for i in main)
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
    print("\n判读（预注册）：", "基差含方向信息 → 候选" if core
          else "核心门未全过 → 判负/信息不足")
    print("可交易性：", "30bp 后仍显著" if g5 else "统计信息未转化为可交易超额")
    print("伪影检查：", "非隔夜/反转伪影" if g6 else "存在隔夜或反转伪影")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
