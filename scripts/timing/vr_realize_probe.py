"""vr 量能腿预注册兑现探针（纯择时线第七篇，reports/vr_realize_probe/）。

动机：第四篇把 vr=amt/MA20−1 连续量能特征升级为线内最强候选腿（T+1 口径，全A
+6.48pp/NW-t 2.05@10bp），但它是 2026-11-23 全样本内发现的，无真 OOS。本篇按
TODO.md「线级订正 ③」的判据草案把它当**独立腿**做兑现检验（不再找新信号）。

门（2026-11-23 跑前定死，对应草案 a–f）：

  G1 主门：全A(399317) 与 zz1000 上，10bp 超额（vs 半仓）>0 且 NW-t ≥2.0，
     30bp 超额 >0，两半窗（2016–2020 / 2021+）超额同号 >0。
  G2 退化对照：全A 上 vr_z5 相对「成交额水平 z5」（近亲）的配对日收益差 >0 且
     NW-t ≥2.0，zz1000/hs300 方向一致 >0；价格趋势四件套保留为对照行（第四篇
     已测全负，不判门）。
  G3 参数邻域：MA 窗 10/20/40 × 平滑 3/5/10 × z 窗 126/252/504（27 组，只报
     不选峰）；全A 与 zz1000 上 10bp 超额正比例 ≥22/27 且中位 >0，30bp 正比例
     ≥22/27。
  G4 稳健性下界：全A 与 zz1000 分年超额 ≥9/11 年正。⚠️ 这是稳健性下界、不是
     OOS；前向检验自 2026-11-23 起纸面跟踪（本脚本每次运行刷新种子台账）。
  G5 组合同台门：壳@000300 + vr_z5@000300 等权，30bp 超额 >0 且 NW-t ≥2.0
     （第五篇教训：任何 vr 组合的主判据必须写 30bp）。
  G6 机制归因（不判门）：拆「高量做多」与「低量空仓」两半贡献 + 分年超额，
     核对 2017 低波慢牛伤口。

判读：G1–G4 全过 ⇒ vr 达采纳候选（独立腿）；G1 不过 ⇒ 判负/差一档（记录缺口）。
G5 过 ⇒ 可与壳同台组合；不过 ⇒ 仅独立腿、不与壳组合。

口径：仓位层 2016-01-01→数据末端（指数日K），信号 ≤t 收盘可知，T+1 open 执行
（复用 hf_vol 的 ``run_continuous``，已带仓位探针不变量），成本 10bp 主 / 30bp
门槛，对照 pos≡0.5（半仓）与 pos≡1（持有）。

用法::

    python -m scripts.timing.vr_realize_probe --self-check
    python -m scripts.timing.vr_realize_probe
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd

from backtest.metrics import annual_return, max_drawdown, sharpe_ratio
from scripts.timing.hf_vol_timing_probe import causal_z, run_continuous
from scripts.timing.shell import ROOT, load_index_daily, run_rebalance_shell
from stats.robust_stats import nw_tstat

OUT = ROOT / "reports" / "vr_realize_probe"
MAIN = {"399317.SZ": "alla", "000852.SH": "zz1000"}   # 主判据资产（草案 a）
CONTROL_CODES = {"000300.SH": "hs300", "000001.SH": "sse"}  # G2 方向一致 + G5 同台
ALL_CODES = list(MAIN) + list(CONTROL_CODES)
BEGIN = "2016-01-01"
HALF = "2021-01-01"
WIN = 20
SMOOTH = 5
ZWIN = 252
COSTS = (10.0, 30.0)
YEARS = tuple(range(2016, 2027))
# 发现日 2026-11-23（数据止 2026-11-20）之后的纸面跟踪起点
TRACK_START = pd.Timestamp("2026-11-21")
TREND_CONTROLS = ("close>MA20", "close>MA60", "ret20>0", "z5(5日收益)>0")
CACHE_SRC = (
    ROOT / "reports" / "hf_vol_timing_probe" / "data" / "data_cache",
    ROOT / "reports" / "legs_combo_probe" / "data" / "data_cache",
)


# ---------------------------------------------------------------- 数据
def seed_cache() -> None:
    """复用上游探针同日拉取的指数日K缓存，避免重复联网（缺则回落到数据源）。"""
    dst = OUT / "data" / "data_cache"
    dst.mkdir(parents=True, exist_ok=True)
    for code in ALL_CODES:
        tgt = dst / f"{code}.parquet"
        if tgt.exists():
            continue
        for src in CACHE_SRC:
            p = src / f"{code}.parquet"
            if p.exists():
                shutil.copy2(p, tgt)
                break


# ---------------------------------------------------------------- 信号
def build_signal(daily: pd.DataFrame, win: int = WIN, smooth: int = SMOOTH,
                 zwin: int = ZWIN) -> pd.Series:
    """vr_z5（全部 ≤t 收盘可知）：vr = amount/MA_win(amount) − 1，
    因果 z（zwin 日，min_periods=zwin//2），再取 smooth 日均值。"""
    amt = daily["amount"]
    return causal_z(amt / amt.rolling(win).mean() - 1.0, zwin).rolling(smooth).mean()


def pos_of(sig: pd.Series) -> pd.Series:
    """长平两态：信号 >0 满仓、否则空仓（NaN 段按 0 = 不建仓）。"""
    return (sig > 0).fillna(False).astype(float)


# ---------------------------------------------------------------- 指标
def arm_row(pos: pd.Series, open_px: pd.Series, cost: float, bench: pd.Series) -> dict:
    """10bp/30bp 单臂指标（窗口 BEGIN 起；excess/NW-t 对 bench）。"""
    m = run_continuous(pos, open_px, cost)
    r = m["ret"].loc[BEGIN:]
    b = bench.loc[BEGIN:]
    d = (r - b).dropna()
    years = len(r) / 252.0
    return {"annual": annual_return(r), "sharpe": sharpe_ratio(r), "mdd": max_drawdown(r),
            "avg_pos": float(pos.loc[BEGIN:].mean()),
            "turn_yr": m["turnover"] / max(years, 1e-9),
            "excess": annual_return(r) - annual_return(b),
            "nw_t": float(nw_tstat(d)[0]), "n": len(r),
            "h1": annual_return(r.loc[:HALF]) - annual_return(b.loc[:HALF]),
            "h2": annual_return(r.loc[HALF:]) - annual_return(b.loc[HALF:])}


def yearly_excess(pos: pd.Series, open_px: pd.Series, cost: float,
                  bench: pd.Series) -> pd.DataFrame:
    r = run_continuous(pos, open_px, cost)["ret"]
    rows = []
    for y in YEARS:
        seg = r.loc[f"{y}-01-01":f"{y}-12-31"]
        if len(seg) == 0:
            rows.append({"year": y, "excess": np.nan, "n": 0})
            continue
        b = bench.loc[seg.index]
        rows.append({"year": y, "excess": annual_return(seg) - annual_return(b), "n": len(seg)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 自检
def self_check(daily: dict[str, pd.DataFrame]) -> list[str]:
    bad = []
    d = daily["399317.SZ"]
    sig = build_signal(d)
    pos = pos_of(sig)
    if not pos.dropna().between(0, 1).all():
        bad.append("pos 越界")
    # 因果性：截断到 t 重算，末值必须与全样本一致（不许用未来信息）
    for t in ("2017-06-15", "2020-03-16", "2024-10-08"):
        tt = pd.Timestamp(t)
        trunc = build_signal(d.loc[:tt])
        if abs(float(trunc.iloc[-1]) - float(sig.loc[tt])) > 1e-10:
            bad.append(f"vr 信号非因果 @{t}")
    # 执行滞后不变量：单点仓位必须恰在 s 行吃 open_{s+1}→open_{s+2}
    op = d["open"]
    probe = pd.Series(0.0, index=op.index)
    probe.iloc[1000] = 1.0
    nz = run_continuous(probe, op, 0.0)["ret"]
    nz = nz[nz != 0]
    if len(nz) != 1 or abs(float(nz.iloc[0]) - float(op.iloc[1002] / op.iloc[1001] - 1)) > 1e-12:
        bad.append("执行滞后错位：pos_s 应恰在 s 行吃 open_{s+1}→open_{s+2}（T+1 open）")
    # 壳：仓位在周三调仓、次日起生效 ⇒ 生效变仓日的前一交易日必须是周三
    cl = d["close"].loc[BEGIN:]
    sh = run_rebalance_shell(cl, pd.Series(0.0, index=cl.index), cost_bps=10.0,
                             step_pct=1.0, step=0.1, init_pos=0.5, weekday=2)
    chg = sh["pos"][sh["pos"].diff().fillna(0) != 0].index
    idx = sh["pos"].index
    if any(idx.get_loc(t) == 0 or idx[idx.get_loc(t) - 1].weekday() != 2 for t in chg):
        bad.append("壳的生效变仓日不在周三调仓的次日")
    return bad


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    seed_cache()
    daily = {c: load_index_daily(c, begin=20130101, cache_dir=OUT / "data",
                                 refresh=args.refresh_data) for c in ALL_CODES}

    if args.self_check:
        bad = self_check(daily)
        for b in bad:
            print("SELF-CHECK FAIL:", b)
        if bad:
            return 1
        print("SELF-CHECK PASS（信号因果性/仓位界/T+1 执行滞后/壳周三变仓全过）")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []
    half = {c: run_continuous(pd.Series(0.5, index=df.index), df["open"], 0.0)["ret"]
            for c, df in daily.items()}
    hold = {c: run_continuous(pd.Series(1.0, index=df.index), df["open"], 0.0)["ret"]
            for c, df in daily.items()}
    sigs = {c: build_signal(df) for c, df in daily.items()}
    poss = {c: pos_of(sigs[c]) for c in ALL_CODES}

    # ================= 主表：10bp 指标 + 30bp 超额 + vs 持有 =================
    print("\n[表] vr_z5 长平（T+1 open；excess/NW-t/h1/h2 对半仓；窗口 2016→now）")
    rows = []
    for c in ALL_CODES:
        d = daily[c]
        r10 = arm_row(poss[c], d["open"], 10.0, half[c])
        r30 = arm_row(poss[c], d["open"], 30.0, half[c])
        rows.append({"index": MAIN.get(c) or CONTROL_CODES[c], **r10,
                     "ex30": r30["excess"], "ex_hold": r10["annual"] - annual_return(hold[c].loc[BEGIN:])})
    tbl = pd.DataFrame(rows)
    print(tbl.round(4).to_string(index=False))
    tbl.to_csv(OUT / "main_table.csv", index=False, encoding="utf-8-sig")

    # ================= G1 主门 =================
    g1_detail, g1_ok = [], True
    for c, name in MAIN.items():
        row = tbl[tbl["index"] == name].iloc[0]
        ok = bool(row["excess"] > 0 and row["nw_t"] >= 2.0 and row["ex30"] > 0
                  and row["h1"] > 0 and row["h2"] > 0)
        g1_ok &= ok
        g1_detail.append(f"{name} 10bp {row['excess']:+.2%}/NW-t {row['nw_t']:.2f}、"
                         f"30bp {row['ex30']:+.2%}、h1 {row['h1']:+.1%}/h2 {row['h2']:+.1%}"
                         f" ⇒ {'PASS' if ok else 'FAIL'}")
    gates.append({"gate": "G1 主门（10bp NW-t≥2 + 30bp>0 + 两半窗同号）",
                  "pass": bool(g1_ok), "detail": "；".join(g1_detail)})

    # ================= G2 退化对照 =================
    print("\n[G2] 退化对照（长平，10bp，对照=半仓；diff = vr − 对照，diffNW = 配对 NW-t）")
    rows2 = []
    for c in list(MAIN) + ["000300.SH"]:
        d = daily[c]
        cl, amt = d["close"], d["amount"]
        ctrls: dict[str, pd.Series] = {
            "vr_z5": poss[c],
            "成交额水平 z5": (causal_z(amt).rolling(5).mean() > 0).fillna(False).astype(float),
            "close>MA20": (cl > cl.rolling(WIN).mean()).fillna(False).astype(float),
            "close>MA60": (cl > cl.rolling(60).mean()).fillna(False).astype(float),
            "ret20>0": (cl / cl.shift(WIN) - 1.0 > 0).fillna(False).astype(float),
            "z5(5日收益)>0": (causal_z(cl / cl.shift(5) - 1.0).rolling(5).mean() > 0)
                             .fillna(False).astype(float),
        }
        rets = {k: run_continuous(v, d["open"], 10.0)["ret"] for k, v in ctrls.items()}
        b = half[c].loc[BEGIN:]
        base = rets["vr_z5"].loc[BEGIN:]
        for k, r in rets.items():
            rr = r.loc[BEGIN:]
            diff = (base - rr).dropna()
            rows2.append({"index": MAIN.get(c) or CONTROL_CODES[c], "arm": k,
                          "excess": annual_return(rr) - annual_return(b),
                          "diff_vs_vr": annual_return(base) - annual_return(rr),
                          "diff_nw_t": float(nw_tstat(diff)[0])})
    tbl2 = pd.DataFrame(rows2)
    print(tbl2.round(4).to_string(index=False))
    tbl2.to_csv(OUT / "degenerate_controls.csv", index=False, encoding="utf-8-sig")
    near = tbl2[tbl2["arm"] == "成交额水平 z5"].set_index("index")
    g2_ok = bool(near.loc["alla", "diff_vs_vr"] > 0 and near.loc["alla", "diff_nw_t"] >= 2.0
                 and near.loc["zz1000", "diff_vs_vr"] > 0 and near.loc["hs300", "diff_vs_vr"] > 0)
    gates.append({"gate": "G2 退化对照（vr 须显著优于成交额水平 z5）",
                  "pass": g2_ok,
                  "detail": "；".join(f"{i} diff {near.loc[i, 'diff_vs_vr']:+.2%}"
                                     f"/NW-t {near.loc[i, 'diff_nw_t']:.2f}"
                                     for i in ("alla", "zz1000", "hs300"))})

    # ================= G3 参数邻域（只报不选峰） =================
    print("\n[G3] 参数邻域 27 组（MA×平滑×z 窗；只报不选峰）")
    grid = []
    for c, name in MAIN.items():
        d = daily[c]
        for win in (10, 20, 40):
            for sm in (3, 5, 10):
                for zw in (126, 252, 504):
                    p = pos_of(build_signal(d, win, sm, zw))
                    r10 = arm_row(p, d["open"], 10.0, half[c])
                    r30 = arm_row(p, d["open"], 30.0, half[c])
                    grid.append({"index": name, "ma_win": win, "smooth": sm, "z_win": zw,
                                 "excess10": r10["excess"], "nw_t10": r10["nw_t"],
                                 "excess30": r30["excess"]})
    gtbl = pd.DataFrame(grid)
    gtbl.to_csv(OUT / "param_grid.csv", index=False, encoding="utf-8-sig")
    g3_detail, g3_ok = [], True
    for name in MAIN.values():
        sub = gtbl[gtbl["index"] == name]
        n10 = int((sub["excess10"] > 0).sum())
        n30 = int((sub["excess30"] > 0).sum())
        med = float(sub["excess10"].median())
        ok = bool(n10 >= 22 and med > 0 and n30 >= 22)
        g3_ok &= ok
        g3_detail.append(f"{name} 10bp 正 {n10}/27、中位 {med:+.2%}、30bp 正 {n30}/27"
                         f" ⇒ {'PASS' if ok else 'FAIL'}")
        print(f"  {name}: 10bp {sub['excess10'].min():+.1%}~{sub['excess10'].max():+.1%}、"
              f"30bp {sub['excess30'].min():+.1%}~{sub['excess30'].max():+.1%}")
    gates.append({"gate": "G3 参数邻域（10bp/30bp 正比例 ≥22/27 且中位>0）",
                  "pass": bool(g3_ok), "detail": "；".join(g3_detail)})

    # ================= G4 稳健性下界 + 分年（明示不是 OOS） =================
    print("\n[G4] 分年超额（vs 半仓，10bp；稳健性下界，不是 OOS）")
    g4_detail, g4_ok = [], True
    for c, name in MAIN.items():
        yt = yearly_excess(poss[c], daily[c]["open"], 10.0, half[c])
        yt.insert(0, "index", name)
        yt.to_csv(OUT / f"yearly_{name}.csv", index=False, encoding="utf-8-sig")
        n_pos = int((yt["excess"] > 0).sum())
        ok = bool(n_pos >= 9)
        g4_ok &= ok
        worst = yt.loc[yt["excess"].idxmin()]
        g4_detail.append(f"{name} {n_pos}/11 年正（最差 {int(worst['year'])} "
                         f"{worst['excess']:+.1%}）")
        print(f"  {name}: " + " ".join(f"{int(r.year)} {r.excess:+.1%}" for r in yt.itertuples()))
    gates.append({"gate": "G4 稳健性下界（分年 ≥9/11 正；不是 OOS）",
                  "pass": bool(g4_ok), "detail": "；".join(g4_detail)})

    # 纸面跟踪种子（前向检验：发现日之后逐日累计；数据未延伸时 n=0）
    track = []
    for c, name in MAIN.items():
        rr = run_continuous(poss[c], daily[c]["open"], 10.0)["ret"]
        after = rr.loc[TRACK_START:]
        b = half[c].loc[TRACK_START:]
        track.append({"index": name, "track_start": TRACK_START.date(), "cost_bp": 10.0,
                      "signal_date": poss[c].index[-1].date(),
                      "last_data_date": rr.index.max().date(), "n_days": len(after),
                      "pos_at_last": float(poss[c].iloc[-1]),
                      "vr_z5_at_last": float(sigs[c].iloc[-1]),
                      "cum_excess_since_start": (annual_return(after) - annual_return(b))
                      if len(after) else 0.0})
    pd.DataFrame(track).to_csv(OUT / "paper_tracking.csv", index=False, encoding="utf-8-sig")

    # ================= G5 与壳同台（000300，30bp 主判据） =================
    print("\n[G5] 壳@000300 + vr_z5@000300 等权（壳为 close-close、vr 为 open-open，口径异质）")
    d3 = daily["000300.SH"]
    cl = d3["close"].loc[BEGIN:]
    sh = {cost: run_rebalance_shell(cl, pd.Series(0.0, index=cl.index), cost_bps=cost,
                                    step_pct=1.0, step=0.1, init_pos=0.5, weekday=2)["ret"]
          for cost in COSTS}
    rows5 = []
    for cost in COSTS:
        r_vr = run_continuous(poss["000300.SH"], d3["open"], cost)["ret"].loc[BEGIN:]
        dfc = pd.concat([sh[cost], r_vr], axis=1, join="inner").dropna()
        combo = dfc.mean(axis=1)
        b = half["000300.SH"].loc[BEGIN:]
        rows5.append({"cost_bp": cost, "annual": annual_return(combo),
                      "sharpe": sharpe_ratio(combo), "mdd": max_drawdown(combo),
                      "excess": annual_return(combo) - annual_return(b),
                      "nw_t": float(nw_tstat((combo - b).dropna())[0]),
                      "h1": annual_return(combo.loc[:HALF]) - annual_return(b.loc[:HALF]),
                      "h2": annual_return(combo.loc[HALF:]) - annual_return(b.loc[HALF:]),
                      "rho_shell_vr": float(dfc.iloc[:, 0].corr(dfc.iloc[:, 1]))})
    t5 = pd.DataFrame(rows5)
    print(t5.round(4).to_string(index=False))
    t5.to_csv(OUT / "shell_vr_combo.csv", index=False, encoding="utf-8-sig")
    r30 = t5[t5["cost_bp"] == 30.0].iloc[0]
    g5_ok = bool(r30["excess"] > 0 and r30["nw_t"] >= 2.0)
    gates.append({"gate": "G5 组合同台门（壳+vr 等权，30bp 超额>0 且 NW-t≥2）",
                  "pass": g5_ok,
                  "detail": f"30bp 超额 {r30['excess']:+.2%}/NW-t {r30['nw_t']:.2f}；"
                            f"ρ(壳,vr) {r30['rho_shell_vr']:+.3f}"})

    # 跨资产信息行（不判门）
    print("\n[信息行·不判门] 壳@000001 + vr_z5@000300 + vr_z5@000852 等权 vs 三指数等权×半仓")
    rowsx = []
    for cost in COSTS:
        sse = daily["000001.SH"]
        cl1 = sse["close"].loc[BEGIN:]
        legs = {"shell": run_rebalance_shell(cl1, pd.Series(0.0, index=cl1.index), cost_bps=cost,
                                             step_pct=1.0, step=0.1, init_pos=0.5,
                                             weekday=2)["ret"]}
        segs = [(sse["open"].shift(-2) / sse["open"].shift(-1) - 1.0).loc[BEGIN:]]
        for c in ("000300.SH", "000852.SH"):
            legs[f"vr@{c[:6]}"] = run_continuous(poss[c], daily[c]["open"], cost)["ret"].loc[BEGIN:]
            segs.append((daily[c]["open"].shift(-2) / daily[c]["open"].shift(-1) - 1.0).loc[BEGIN:])
        combo = pd.concat(legs, axis=1, join="inner").dropna().mean(axis=1)
        bench = 0.5 * pd.concat(segs, axis=1, join="inner").mean(axis=1)
        rowsx.append({"cost_bp": cost, "excess": annual_return(combo) - annual_return(bench),
                      "nw_t": float(nw_tstat((combo - bench).dropna())[0]),
                      "h1": annual_return(combo.loc[:HALF]) - annual_return(bench.loc[:HALF]),
                      "h2": annual_return(combo.loc[HALF:]) - annual_return(bench.loc[HALF:])})
    print(pd.DataFrame(rowsx).round(4).to_string(index=False))
    pd.DataFrame(rowsx).to_csv(OUT / "cross_asset.csv", index=False, encoding="utf-8-sig")

    # ================= G6 机制归因（不判门） =================
    print("\n[G6] 机制归因：高量做多 / 低量空仓 两半贡献（10bp，vs 半仓）")
    rows6 = []
    for c, name in MAIN.items():
        d = daily[c]
        p = poss[c]
        arms = {"full": p,
                "high_only（低量回半仓）": p.where(p > 0, 0.5),
                "low_avoid（高量回半仓）": p * 0.5}
        b = half[c].loc[BEGIN:]
        vals = {k: annual_return(run_continuous(v, d["open"], 10.0)["ret"].loc[BEGIN:])
                - annual_return(b) for k, v in arms.items()}
        rows6.append({"index": name, **vals,
                      "sum_check": vals["high_only（低量回半仓）"] + vals["low_avoid（高量回半仓）"]})
    t6 = pd.DataFrame(rows6)
    print(t6.round(4).to_string(index=False))
    t6.to_csv(OUT / "attribution.csv", index=False, encoding="utf-8-sig")

    print("\n===== 判定 =====")
    for g in gates:
        print(f"{g['gate']}: {'PASS' if g['pass'] else 'FAIL'}  {g['detail']}")
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False, encoding="utf-8-sig")
    leg_ok = bool(g1_ok and g2_ok and g3_ok and g4_ok)
    failed = [g["gate"].split("（")[0] for g in gates[:4] if not g["pass"]]
    print("\n判读（预注册）：独立腿",
          "达标 → 采纳候选" if leg_ok else f"未全过（未过：{'、'.join(failed)}）→ 判负/差一档")
    print("组合层：", "可与壳同台" if g5_ok else "维持独立腿、不与壳组合")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
