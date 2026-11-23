"""三腿兑现与组合探针（纯择时线第五篇，reports/legs_combo_probe/）。

动机：四源 IC 已真而兑现死于换手/独立注少（第四篇诊断：每日期望边际 ≈IC×σ≈9bp，
换仓成本 20bp）⇒ 不再找新信息，测「低频独立腿组合能否抬 IR」（Grinold：IR=IC×√独立注数，
三腿低相关 ⇒ 组合层把注数乘起来）。预注册见 TODO.md「第五篇预注册」（2026-11-23）。

三腿（全部 000300、10bp、2016-01→now 同窗，半窗 2016–2020 / 2021–now）：
  L1 rev_shell   周频均值回归壳（shell.run_rebalance_shell 原被门臂，close-close 口径）
  L2 us_ret1-z5  标普隔夜 5 日平滑 z 长平两态（hf_vol 修正版 build_us_ret1，open-open）
  L3 vr-z5       量能比 5 日平滑 z 长平两态（open-open）
门：G0 独立性（两两 |ρ|<0.3）/ G1 单腿弱门（超额>0 双半窗同号、30bp 仍>0）/
G2 组合主门（等权 vs 半仓 NW-t≥2 且 Sharpe≥最强单腿+0.2）/ G3 剔 vr 双腿仍正 /
G4 30bp 组合正 + 等权 vs 因果逆波动率排序半窗不翻转。000001 同流程作信息行。

用法::

    python -m scripts.timing.legs_combo_probe --self-check
    python -m scripts.timing.legs_combo_probe
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest.metrics import annual_return, calmar_ratio, max_drawdown, sharpe_ratio
from scripts.timing.hf_vol_timing_probe import build_us_ret1, causal_z, run_continuous
from scripts.timing.shell import ROOT, load_index_daily, run_rebalance_shell
from stats.robust_stats import nw_tstat

OUT = ROOT / "reports" / "legs_combo_probe"
INDEX_CODE = "000300.SH"
INFO_CODE = "000001.SH"
BEGIN = "2016-01-01"
HALF = "2021-01-01"
WIN = 20
COST = 10.0


# ---------------------------------------------------------------- 腿构建
def build_legs(daily: pd.DataFrame, cost_bps: float = COST,
               refresh: bool = False) -> dict[str, pd.Series]:
    """三条腿的**日净收益**（各腿已含自身换仓成本）。L1 close-close 壳口径、
    L2/L3 open-open 段口径——沿用各自已验证形态（口径异质是预注册边界，README §三）。"""
    close, open_px, amt = daily["close"], daily["open"], daily["amount"]
    # L1 rev_shell：sig≡0 = 均值回归模式（被门臂原参数）。⚠️ 必须从 BEGIN 直接起跑——
    # 壳的仓位是路径依赖的（无回归中枢），2013 起暖机再裁窗会得到另一条轨迹
    # （实测 000300 超额 +0.09pp vs 原形态 +0.90pp），预注册要求「原被门臂」
    cl = close.loc[BEGIN:]
    shell = run_rebalance_shell(cl, pd.Series(0.0, index=cl.index),
                                cost_bps=cost_bps, step_pct=1.0, step=0.1,
                                init_pos=0.5, weekday=2)
    # L2/L3：信号用全历史（2013 起，causal_z 252 日暖机）计算，收益统一裁到窗口
    z_us = causal_z(build_us_ret1(daily, refresh=refresh)).rolling(5).mean()
    z_vr = causal_z(amt / amt.rolling(WIN).mean() - 1.0).rolling(5).mean()
    r_us = run_continuous((z_us > 0).astype(float), open_px, cost_bps)["ret"]
    r_vr = run_continuous((z_vr > 0).astype(float), open_px, cost_bps)["ret"]
    legs = {"L1_rev_shell": shell["ret"], "L2_us_ret1_z5": r_us, "L3_vr_z5": r_vr}
    bench_half = run_continuous(pd.Series(0.5, index=close.index), open_px, 0.0)["ret"]
    bench_hold = run_continuous(pd.Series(1.0, index=close.index), open_px, 0.0)["ret"]
    out = {**legs, "B_half": bench_half, "B_hold": bench_hold}
    return {k: v.loc[BEGIN:] for k, v in out.items()}


def combo_series(legs: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """等权 / 因果逆波动率（252d，滞后一日）/ 剔 vr 双腿。"""
    df = pd.concat([legs["L1_rev_shell"], legs["L2_us_ret1_z5"], legs["L3_vr_z5"]],
                   axis=1, join="inner")
    df.columns = ["L1", "L2", "L3"]
    equal = df.mean(axis=1)
    inv = 1.0 / df.rolling(252, min_periods=126).std().shift(1)
    # ⚠️ DataFrame/Series 相除默认按列对齐（dates≠列名 ⇒ 全 NaN），必须 div(axis=0)
    w = inv.div(inv.sum(axis=1, min_count=1), axis=0)
    invvol = (df * w).sum(axis=1, min_count=1)
    ex_vr = df[["L1", "L2"]].mean(axis=1)
    return {"C_equal": equal, "C_invvol": invvol, "C_ex_vr": ex_vr}


# ---------------------------------------------------------------- 指标
def perf_row(ret: pd.Series, bench: pd.Series | None = None) -> dict:
    row = {"annual": annual_return(ret), "sharpe": sharpe_ratio(ret),
           "mdd": max_drawdown(ret), "calmar": calmar_ratio(ret), "n": len(ret)}
    if bench is not None:
        d = (ret - bench).dropna()
        row["excess"] = annual_return(ret) - annual_return(bench)
        row["nw_t"] = nw_tstat(d)[0]
    return row


def window_table(series: dict[str, pd.Series], half_bench: pd.Series) -> pd.DataFrame:
    rows = []
    for name, r in series.items():
        for wtag, seg in (("full", r), ("h1", r.loc[:HALF]), ("h2", r.loc[HALF:])):
            row = {"name": name, "win": wtag, **perf_row(seg, half_bench.loc[seg.index[0]:seg.index[-1]]
                                                        if wtag != "full" else half_bench)}
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 自检
def self_check() -> list[str]:
    bad = []
    daily = load_index_daily(INDEX_CODE, begin=20130101, cache_dir=OUT / "data")
    legs = build_legs(daily)
    for k in ("L1_rev_shell", "L2_us_ret1_z5", "L3_vr_z5"):
        r = legs[k].dropna()
        if len(r) < 2000:
            bad.append(f"{k} 样本过少（{len(r)}）")
        if r.abs().max() > 0.20:
            # 上界 20%：2024-10-08 节后跳空（09-30 open 3846.7→10-08 open 4450.4，
            # +15.7% open-to-open，信息日 09-27）与 2020-02-03 疫情缺口（−10.7%）为真实行情
            bad.append(f"{k} 出现 |日收益|>20%")
    # us_ret1 无内部 NaN（第四篇修正版不变量）
    u = build_us_ret1(daily).dropna()
    first = build_us_ret1(daily).first_valid_index()
    if build_us_ret1(daily).loc[first:].isna().any():
        bad.append("us_ret1 内部 NaN（修正版失效？）")
    # 执行段对齐抽查（hf_vol 同款）
    op = daily["open"].dropna()
    for loc in (300, 1200, 2500):
        if loc + 2 >= len(op):
            continue
        expect = op.iloc[loc + 2] / op.iloc[loc + 1] - 1
        got = (daily["open"].shift(-2) / daily["open"].shift(-1) - 1.0).iloc[loc]
        if abs(float(expect) - float(got)) > 1e-12:
            bad.append(f"目标段错位 @{op.index[loc].date()}")
    # 组合分散不变量：等权日收益波动 ≤ 最大单腿波动（相关<1 时必成立）
    df = pd.concat([legs["L1_rev_shell"], legs["L2_us_ret1_z5"], legs["L3_vr_z5"]],
                   axis=1, join="inner").dropna()
    if df.std().max() < df.mean(axis=1).std() - 1e-12:
        bad.append("等权波动 ≥ 最大单腿波动（组合构造错？）")
    return bad


# ---------------------------------------------------------------- 主流程
def run_asset(code: str, refresh: bool = False) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = load_index_daily(code, begin=20130101, cache_dir=OUT / "data", refresh=refresh)
    legs10 = build_legs(daily, COST, refresh)
    legs30 = build_legs(daily, 30.0, refresh)
    combos = combo_series(legs10)
    series = {**legs10, **combos}
    half = legs10["B_half"]
    tbl = window_table(series, half)
    tbl30 = window_table({**{k: legs30[k] for k in ("L1_rev_shell", "L2_us_ret1_z5", "L3_vr_z5")},
                          "C_equal@30bp": combo_series(legs30)["C_equal"]}, half)

    # 相关性
    ldf = pd.concat([legs10["L1_rev_shell"], legs10["L2_us_ret1_z5"], legs10["L3_vr_z5"]],
                    axis=1, join="inner").dropna()
    corr = ldf.corr()

    # 分年超额（等权 vs 半仓）
    yr = []
    for y, seg in series["C_equal"].groupby(series["C_equal"].index.year):
        yr.append({"year": y, "excess": annual_return(seg) - annual_return(half.loc[seg.index]),
                   "n": len(seg)})
    yearly = pd.DataFrame(yr)
    return series, corr, tbl, tbl30, yearly


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        bad = self_check()
        for b in bad:
            print("SELF-CHECK FAIL:", b)
        if bad:
            return 1
        print("SELF-CHECK PASS（腿样本量/收益量级/us_ret1 无缺口/执行段对齐/分散不变量全过）")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []

    for code, tag in ((INDEX_CODE, "主资产 000300"), (INFO_CODE, "信息行 000001")):
        series, corr, tbl, tbl30, yearly = run_asset(code, args.refresh_data)
        if code != INDEX_CODE:
            print(f"\n[{tag}] 等权 vs 半仓（信息行，不判门）")
            row = perf_row(series["C_equal"], series["B_half"])
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
            tbl.assign(asset=code).to_csv(OUT / f"perf_{code.split('.')[0]}.csv",
                                          index=False, encoding="utf-8-sig")
            continue

        print(f"\n[{tag}] 三腿 + 组合（10bp；full/h1=2016-2020/h2=2021+；excess/nw_t 对半仓）")
        print(tbl.round(4).to_string(index=False))
        tbl.to_csv(OUT / "perf_000300.csv", index=False, encoding="utf-8-sig")
        tbl30.to_csv(OUT / "perf_000300_30bp.csv", index=False, encoding="utf-8-sig")
        yearly.to_csv(OUT / "yearly_excess.csv", index=False, encoding="utf-8-sig")
        print("分年超额（等权−半仓）：")
        print(yearly.round(4).to_string(index=False))

        print("\n三腿日收益相关矩阵：")
        print(corr.round(3).to_string())
        corr.to_csv(OUT / "legs_corr.csv", encoding="utf-8-sig")

        # ---------- 门 ----------
        pair = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack()
        g0 = bool(pair.max() < 0.3)
        gates.append({"gate": "G0 独立性（两两|ρ|<0.3）", "pass": g0,
                      "detail": f"max|ρ|={pair.max():.3f}，mean|ρ|={pair.mean():.3f}"})

        g1_detail, g1_pass = [], True
        for k in ("L1_rev_shell", "L2_us_ret1_z5", "L3_vr_z5"):
            h = tbl[(tbl["name"] == k)]
            full_ex = float(h[h["win"] == "full"]["excess"].iloc[0])
            h1_ex = float(h[h["win"] == "h1"]["excess"].iloc[0])
            h2_ex = float(h[h["win"] == "h2"]["excess"].iloc[0])
            ex30 = float(tbl30[(tbl30["name"] == k) & (tbl30["win"] == "full")]["excess"].iloc[0])
            ok = full_ex > 0 and h1_ex > 0 and h2_ex > 0 and ex30 > 0
            g1_pass &= ok
            g1_detail.append(f"{k} {full_ex:+.2%}(h1 {h1_ex:+.1%}/h2 {h2_ex:+.1%}/30bp {ex30:+.1%})")
        gates.append({"gate": "G1 单腿弱门", "pass": bool(g1_pass), "detail": "；".join(g1_detail)})

        eq_full = tbl[(tbl["name"] == "C_equal") & (tbl["win"] == "full")].iloc[0]
        best_leg = tbl[(tbl["name"].str.startswith("L")) & (tbl["win"] == "full")]
        best = best_leg.loc[best_leg["sharpe"].idxmax()]
        g2 = bool(eq_full["nw_t"] >= 2.0 and eq_full["sharpe"] >= best["sharpe"] + 0.2)
        gates.append({"gate": "G2 组合主门（NW-t≥2 且 Sharpe≥最强腿+0.2）", "pass": g2,
                      "detail": f"等权 NW-t={eq_full['nw_t']:.2f}、Sharpe={eq_full['sharpe']:.2f} "
                                f"vs 最强腿 {best['name']} {best['sharpe']:.2f}"})

        exvr = tbl[(tbl["name"] == "C_ex_vr") & (tbl["win"] == "full")].iloc[0]
        g3 = bool(exvr["excess"] > 0)
        gates.append({"gate": "G3 剔 vr 双腿仍正", "pass": g3,
                      "detail": f"C_ex_vr 超额 {exvr['excess']:+.2%}、NW-t {exvr['nw_t']:.2f}"})

        eq30 = tbl30[(tbl30["name"] == "C_equal@30bp") & (tbl30["win"] == "full")].iloc[0]
        ranks = {}
        for wtag, sl in (("h1", slice(None, HALF)), ("h2", slice(HALF, None))):
            a = sharpe_ratio(series["C_equal"].loc[sl].dropna())
            b = sharpe_ratio(series["C_invvol"].loc[sl].dropna())
            ranks[wtag] = "invalid" if np.isnan(b) else ("eq" if a >= b else "inv")
        g4 = bool(eq30["excess"] > 0 and ranks["h1"] == ranks["h2"] and "invalid" not in ranks.values())
        gates.append({"gate": "G4 成本/权重稳健", "pass": g4,
                      "detail": f"30bp 等权超额 {eq30['excess']:+.2%}；排序 h1={ranks['h1']} "
                                f"h2={ranks['h2']}（需一致）"})

    print("\n===== 判定（000300 主资产） =====")
    for g in gates:
        print(f"{g['gate']}: {'PASS' if g['pass'] else 'FAIL'}  {g['detail']}")
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
