"""华福量价择时探针（纯择时线第四篇，reports/hf_vol_timing_probe/）。

研报：华福证券 2025-05-16《基于放量与缩量的量价择时策略研究》——指数级量能事件信号：
横盘缩量（防回撤辅助）/ 放量未涨（布局）。预注册四臂见 TODO.md「第四篇预注册」
（2026-11-23，参数主口径照抄研报）：

  ① 事件复现：20 日窗、std/mean(close)<2% 横盘、amt<0.8×MA20 缩量（MA 含当日）、
     amt>1.5×MA20 且 close<前 20 日最高收盘为放量；事件=条件链首日。全A 用 399317
     替代。G1 = 分年次数 Spearman ≥0.7 或 |年差|≤2 年份过半，且三锚点日
     （2024-08-12/12-24/2025-03-25）≥2/3 在 ±5 交易日内。
  ② 横截面单调性：事件后 20 日均值收益按 zz1000≥zz500≥全A≥hs300≥sz50（G2=Spearman≥0.8）。
  ③ us_ret1 叠加（000300）：③a 裸兑现（causal_z>0 长平两态）/ ③b 缩量状态清零 /
     ③c 放量状态满仓。G3-风控 = MDD 改善≥2pp 且年化恶化≤0.5pp；G3-进攻 = 年化
     +≥0.5pp 且 MDD 恶化≤2pp。对照 pos≡0/≡0.5/≡1。T+1 open 执行，10bp（敏感性 30bp）。
  ④ 连续特征：vr=amt/MA20−1、flat=std/mean(close)。门 = |日IC|≥2% 或 |周IC|≥3%
     且半窗同号（frontier 线口径）。
  OOS = 事件日 > 2025-05-12（研报数据截止）。敏感性网格只报不选峰。
  事件层口径注记：fwd 收益从 T 收盘起算（研报同口径，信息可知性检验）；
  策略臂才用 T+1 open 可执行口径。

用法::

    python -m scripts.timing.hf_vol_timing_probe --self-check
    python -m scripts.timing.hf_vol_timing_probe
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest.metrics import annual_return, calmar_ratio, max_drawdown, sharpe_ratio
from scripts.timing.shell import ROOT, load_index_daily

OUT = ROOT / "reports" / "hf_vol_timing_probe"

# ②的单调性声称序（流动性敏感度：1=最强）；sse 只作信息行不进 G2
INDICES: dict[str, tuple[str, int]] = {
    "000852.SH": ("zz1000", 1),
    "000905.SH": ("zz500", 2),
    "399317.SZ": ("alla", 3),
    "000300.SH": ("hs300", 4),
    "000016.SH": ("sz50", 5),
    "000001.SH": ("sse", 0),
}
OVERLAY_CODE = "000300.SH"
REPORT_CUTOFF = pd.Timestamp("2025-05-12")
ANCHORS = ("2024-08-12", "2024-12-24", "2025-03-25")
HORIZONS = (1, 5, 10, 20, 40)

# 研报 p8/p11 万得全A 分年触发次数（399317 替代口径的校准靶；0 表示研报该年无此行）
REPORT_COUNTS = {
    "shrink": {2010: 5, 2011: 7, 2012: 6, 2013: 5, 2014: 4, 2015: 1, 2016: 4, 2017: 5,
               2018: 2, 2019: 6, 2020: 5, 2021: 2, 2022: 4, 2023: 5, 2024: 2, 2025: 1},
    "surge": {2010: 2, 2011: 2, 2012: 4, 2013: 2, 2014: 2, 2015: 3, 2016: 2, 2017: 0,
              2018: 1, 2019: 1, 2020: 1, 2021: 0, 2022: 0, 2023: 0, 2024: 3, 2025: 0},
}

WIN = 20          # 滚动窗（研报 20 日）
FLAT_TH = 0.02    # 横盘阈值（std/mean）
SHRINK_TH = 0.8   # 缩量阈值（×MA20）
SURGE_TH = 1.5    # 放量阈值（×MA20）
HIGH_WIN = 20     # 「未破近期高点」窗口（研报未给，预注册取 20 日）
COST_BPS = 10.0


# ---------------------------------------------------------------- 数据源
def load_us_daily(refresh: bool = False) -> pd.Series:
    """标普500 日收盘（akshare sina；frontier 探针同源，新版接口名 index_us_stock_sina）。"""
    cache = OUT / "data" / "spx.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)["close"]
    import akshare as ak

    df = ak.index_us_stock_sina(symbol=".INX")
    s = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float).sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame("close").to_parquet(cache)
    return s


def build_us_ret1(daily: pd.DataFrame, refresh: bool = False) -> pd.Series:
    """us_ret1（frontier 探针采纳口径 + 周一补全）：标普 t-1（最后一个严格早于 A 股 t 的
    美股交易日，北京时间 t 凌晨收盘）→ A 股 t 日特征。frontier 原式在 t-1 为非美股
    交易日且非 A 股交易日（周日等）处留 NaN——IC 层无害（dropna），仓位层会把周一
    误当空仓 ⇒ 把 t-1 日历日并进 ffill 索引集补全（因果性不变：所取美股收盘恒早于
    A 股 t 日开盘）。"""
    us = load_us_daily(refresh)
    us_ret = us.pct_change()
    cal = daily.index - pd.Timedelta(days=1)
    s = us_ret.reindex(us_ret.index.union(cal)).ffill().reindex(cal)
    return pd.Series(s.values, index=daily.index)


def causal_z(x: pd.Series, win: int = 252) -> pd.Series:
    mu = x.rolling(win, min_periods=win // 2).mean()
    sd = x.rolling(win, min_periods=win // 2).std()
    return (x - mu) / sd


# ---------------------------------------------------------------- ① 信号
def build_conditions(daily: pd.DataFrame, flat_th: float = FLAT_TH,
                     shrink_th: float = SHRINK_TH, surge_th: float = SURGE_TH,
                     win: int = WIN, high_win: int = HIGH_WIN,
                     ma_shift: int = 0) -> dict[str, pd.Series]:
    """量能状态与事件（全部 ≤t 收盘可知）。ma_shift=1 为敏感性臂（MA 不含当日）。"""
    close, amt = daily["close"], daily["amount"]
    flat = close.rolling(win, min_periods=win).std() / close.rolling(win, min_periods=win).mean() < flat_th
    ma = amt.rolling(win, min_periods=win).mean().shift(ma_shift)
    shrink = amt < shrink_th * ma
    no_new_high = close < close.rolling(high_win, min_periods=high_win).max().shift(1)
    surge = (amt > surge_th * ma) & no_new_high
    cond = {"shrink": flat & shrink, "surge": surge}
    out = {f"{k}_state": v for k, v in cond.items()}
    for k, v in cond.items():  # 事件 = 条件链首日
        out[f"{k}_event"] = v & ~v.shift(1, fill_value=False)
    return out


def event_table(close: pd.Series, event: pd.Series,
                horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """事件日前瞻收益（T 收盘起算，研报口径）。"""
    rows = []
    fwd = {h: close.shift(-h) / close - 1.0 for h in horizons}
    for t in close.index[event.fillna(False)]:
        row = {"date": t, "is_oos": t > REPORT_CUTOFF}
        row.update({f"fwd{h}": float(fwd[h].loc[t]) for h in horizons})
        rows.append(row)
    return pd.DataFrame(rows)


def yearly_counts(event: pd.Series) -> pd.Series:
    ev = event.fillna(False)
    years = pd.Series(ev.index.year, index=ev.index)
    return years[ev].value_counts().sort_index()


# ---------------------------------------------------------------- ③ 回测壳
def run_continuous(pos: pd.Series, open_px: pd.Series, cost_bps: float) -> dict:
    """连续仓位（multisrc_timing_probe 口径复制）：pos_t 于 t 收盘可知，
    t+1 open→t+2 open 段生效；换仓成本在 t+1 open。"""
    seg = open_px.shift(-2) / open_px.shift(-1) - 1.0
    earn = pos.shift(1)
    cost = earn.diff().abs().fillna(0.0) * cost_bps * 1e-4
    ret = (earn * seg - cost).dropna()
    bench = seg.dropna()
    return {
        "annual": annual_return(ret),
        "sharpe": sharpe_ratio(ret),
        "mdd": max_drawdown(ret),
        "calmar": calmar_ratio(ret),
        "bench_annual": annual_return(bench),
        "excess": annual_return(ret) - annual_return(bench),
        "avg_pos": float(earn.dropna().mean()),
        "turnover": float(earn.diff().abs().fillna(0).sum()),
        "ret": ret,
    }


def ts_corr_of(x: pd.Series, y: pd.Series) -> float:
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) < 200:
        return float("nan")
    return float(s.iloc[:, 0].corr(s.iloc[:, 1]))


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


# ---------------------------------------------------------------- 自检
def self_check(daily: dict[str, pd.DataFrame]) -> list[str]:
    bad = []
    alla = daily["399317.SZ"]
    if not (alla["amount"] > 0).all():
        bad.append("amount 出现 ≤0")
    # us_ret1 抽查：2020-03-16 标普 −11.98%（美东收盘=北京 03-17 凌晨）→ 03-17 特征
    us_ret1 = build_us_ret1(daily[OVERLAY_CODE])
    v = us_ret1.loc["2020-03-17"] if "2020-03-17" in us_ret1.index else np.nan
    if not (v < -0.10):
        bad.append(f"us_ret1@2020-03-17 应≈−0.12，实际 {v:.4f}（映射错位？）")
    if us_ret1.dropna().abs().max() > 0.15:
        bad.append("us_ret1 出现 >±15%（映射错位？）")
    first = us_ret1.first_valid_index()
    if first is not None and us_ret1.loc[first:].isna().any():
        bad.append("us_ret1 首个有效值后仍有 NaN（周一映射缺口？）")
    # 条件重算抽查（3 个随机时点）
    cond = build_conditions(alla)
    rng = np.random.default_rng(7)
    idx = alla.index[25:-5]
    for t in rng.choice(idx, 3, replace=False):
        w = alla.loc[:t].tail(WIN)
        flat = (w["close"].std() / w["close"].mean() < FLAT_TH)
        shr = (alla.at[t, "amount"] < SHRINK_TH * w["amount"].mean())
        if bool(cond["shrink_state"].loc[t]) != bool(flat and shr):
            bad.append(f"缩量条件重算不一致 @{pd.Timestamp(t).date()}")
    # 事件=链首：不允许相邻事件日
    for k in ("shrink", "surge"):
        ev = cond[f"{k}_event"]
        if (ev & ev.shift(1, fill_value=False)).any():
            bad.append(f"{k}_event 出现相邻事件日")
    # 执行段对齐抽查（multisrc 同款）
    op = daily[OVERLAY_CODE]["open"]
    d = op.dropna()
    for loc in (300, 1200, 2500):
        if loc + 2 >= len(d):
            continue
        expect = d.iloc[loc + 2] / d.iloc[loc + 1] - 1
        got = (op.shift(-2) / op.shift(-1) - 1.0).iloc[loc]
        if abs(float(expect) - float(got)) > 1e-12:
            bad.append(f"目标段错位 @{d.index[loc].date()}")
    return bad


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    daily = {c: load_index_daily(c, begin=20130101, cache_dir=OUT / "data",
                                 refresh=args.refresh_data) for c in INDICES}
    OUT.mkdir(parents=True, exist_ok=True)

    if args.self_check:
        bad = self_check(daily)
        for b in bad:
            print("SELF-CHECK FAIL:", b)
        if bad:
            return 1
        print("SELF-CHECK PASS（us_ret1 锚点/条件重算/事件链首/执行段对齐全过）")
        return 0

    gates: list[dict] = []

    # ================= ① 事件复现（含 G1 校准） =================
    cond = {c: build_conditions(df) for c, df in daily.items()}
    events = {c: {k: event_table(daily[c]["close"], cond[c][f"{k}_event"]) for k in ("shrink", "surge")}
              for c in daily}
    for k in ("shrink", "surge"):
        pd.concat([events[c][k].assign(index=INDICES[c][0]) for c in INDICES],
                  ignore_index=True).to_csv(OUT / f"events_{k}.csv", index=False, encoding="utf-8-sig")

    print("\n[①] 事件复现——各指数事件数（IS≤2025-05-12 / OOS 之后）与 20 日前瞻（分 IS/OOS）")
    rows1 = []
    for c, (name, _) in INDICES.items():
        for k in ("shrink", "surge"):
            ev = events[c][k]
            n_is, n_oos = int((~ev["is_oos"]).sum()), int(ev["is_oos"].sum())
            rows1.append({
                "index": name, "event": k, "n_total": len(ev), "n_is": n_is, "n_oos": n_oos,
                "fwd20_mean": ev["fwd20"].mean(), "fwd20_median": ev["fwd20"].median(),
                "fwd20_win": (ev["fwd20"] > 0).mean(),
                "fwd20_mean_is": ev.loc[~ev["is_oos"], "fwd20"].mean(),
                "fwd20_mean_oos": ev.loc[ev["is_oos"], "fwd20"].mean() if n_oos else np.nan,
                "fwd20_win_oos": (ev.loc[ev["is_oos"], "fwd20"] > 0).mean() if n_oos else np.nan,
            })
    t1 = pd.DataFrame(rows1)
    print(t1.round(4).to_string(index=False))
    t1.to_csv(OUT / "events_summary.csv", index=False, encoding="utf-8-sig")

    print("\n[①] 前瞻 horizon（全A=399317 与 zz1000；均值/胜率，全样本）")
    rows1b = []
    for c in ("399317.SZ", "000852.SH"):
        for k in ("shrink", "surge"):
            ev = events[c][k]
            row = {"index": INDICES[c][0], "event": k}
            for h in HORIZONS:
                row[f"h{h}_mean"] = ev[f"fwd{h}"].mean()
                row[f"h{h}_win"] = (ev[f"fwd{h}"] > 0).mean()
            rows1b.append(row)
    print(pd.DataFrame(rows1b).round(4).to_string(index=False))

    # G1：399317 分年次数 vs 研报表 + 三锚点日
    alla_ev = events["399317.SZ"]
    g1_parts = []
    for k in ("shrink", "surge"):
        ours = yearly_counts(cond["399317.SZ"][f"{k}_event"])
        common = [y for y in range(2014, 2026)]
        o = pd.Series({y: int(ours.get(y, 0)) for y in common})
        r = pd.Series({y: REPORT_COUNTS[k].get(y, 0) for y in common})
        rho = spearman(o, r)
        close_years = float((abs(o - r) <= 2).mean())
        g1_parts.append((k, rho, close_years, o, r))
        print(f"  G1[{k}] 分年次数 Spearman={rho:.3f} | 年差≤2 占比={close_years:.2f}")
        print(f"      ours: {dict(o)}")
        print(f"      研报: {dict(r)}")
    anchor_pos = {a: daily["399317.SZ"].index.get_indexer([pd.Timestamp(a)], method="nearest")[0]
                  for a in ANCHORS}
    ev_pos = [daily["399317.SZ"].index.get_loc(t) for t in alla_ev["shrink"]["date"]]
    hits = []
    for a, ap_ in anchor_pos.items():
        dist = min((abs(p - ap_) for p in ev_pos), default=999)
        hits.append(dist <= 5)
        print(f"  G1 锚点 {a}: 最近缩量事件距离 {dist} 交易日 {'HIT' if dist <= 5 else 'MISS'}")
    g1_rho_ok = all(rho >= 0.7 or cy >= 0.5 for _, rho, cy, _, _ in g1_parts)
    g1 = g1_rho_ok and sum(hits) >= 2
    gates.append({"gate": "G1 口径复现（次数+锚点）", "pass": bool(g1),
                  "detail": f"Spearman {[f'{r:.2f}' for _, r, _, _, _ in g1_parts]}；"
                            f"锚点 {int(sum(hits))}/3"})

    # ================= ② 横截面单调性 =================
    print("\n[②] 横截面：缩量/放量后 20 日均值（声称序 zz1000≥zz500≥全A≥hs300≥sz50）")
    rows2 = []
    for c, (name, claimed) in INDICES.items():
        if not claimed:
            continue
        row = {"index": name, "claimed_rank": claimed}
        for k in ("shrink", "surge"):
            row[f"{k}_fwd20_mean"] = events[c][k]["fwd20"].mean()
        rows2.append(row)
    t2 = pd.DataFrame(rows2).sort_values("claimed_rank")
    print(t2.round(4).to_string(index=False))
    t2.to_csv(OUT / "cross_section.csv", index=False, encoding="utf-8-sig")
    sp = {}
    for k, asc in (("shrink", True), ("surge", False)):  # shrink: 越小盘越负；surge: 越小盘越正
        actual = t2[f"{k}_fwd20_mean"].rank(ascending=asc)
        sp[k] = spearman(actual, t2["claimed_rank"])
        print(f"  G2[{k}] Spearman(实际序, 声称序) = {sp[k]:.3f}")
    g2 = bool(sp["shrink"] >= 0.8 and sp["surge"] >= 0.8)
    gates.append({"gate": "G2 横截面单调性", "pass": g2,
                  "detail": f"shrink {sp['shrink']:.2f} / surge {sp['surge']:.2f}（门 ≥0.8）"})

    # ================= ③ us_ret1 叠加 =================
    print("\n[③] us_ret1 叠加（000300，T+1 open，成本 10bp；对照含退化形态）")
    d3 = daily[OVERLAY_CODE]
    us_ret1 = build_us_ret1(d3, refresh=args.refresh_data)
    pos_base = (causal_z(us_ret1) > 0).astype(float)
    c3 = cond[OVERLAY_CODE]
    arms = {
        "pos0（退化）": pd.Series(0.0, index=d3.index),
        "pos_half（退化）": pd.Series(0.5, index=d3.index),
        "pos1（退化=持有）": pd.Series(1.0, index=d3.index),
        "③a us_ret1 裸兑现": pos_base,
        "③b +缩量清零": pos_base * (~c3["shrink_state"].fillna(False)).astype(float),
        "③c +放量满仓": pd.concat([pos_base, c3["surge_state"].astype(float)], axis=1).max(axis=1),
    }
    rows3 = []
    for name, pos in arms.items():
        m = run_continuous(pos, d3["open"], COST_BPS)
        rows3.append({"arm": name, "annual": m["annual"], "sharpe": m["sharpe"],
                      "mdd": m["mdd"], "calmar": m["calmar"], "avg_pos": m["avg_pos"],
                      "turnover": m["turnover"]})
    t3 = pd.DataFrame(rows3)
    print(t3.round(4).to_string(index=False))
    t3.to_csv(OUT / "overlay_arms.csv", index=False, encoding="utf-8-sig")

    base = run_continuous(arms["③a us_ret1 裸兑现"], d3["open"], COST_BPS)
    filt = run_continuous(arms["③b +缩量清零"], d3["open"], COST_BPS)
    atk = run_continuous(arms["③c +放量满仓"], d3["open"], COST_BPS)
    g3_risk = bool(base["mdd"] - filt["mdd"] >= 0.02 and filt["annual"] - base["annual"] >= -0.005)
    g3_atk = bool(atk["annual"] - base["annual"] >= 0.005 and atk["mdd"] - base["mdd"] <= 0.02)
    for label, pos in (("③b", arms["③b +缩量清零"]), ("③c", arms["③c +放量满仓"])):
        mm = run_continuous(pos, d3["open"], 30.0)
        print(f"  敏感性 30bp {label}: 年化 {mm['annual']:.2%} / MDD {mm['mdd']:.2%}")

    # 非预注册诊断（归因）：③a 日频翻转换手 1654×，10bp 成本独杀（multisrc v1 同病）。
    # 5 日平滑 z 的低换手基线上复测叠加增量——不判门，只回答「base 太噪还是量能无叠加价值」
    z5 = causal_z(us_ret1).rolling(5).mean()
    pos_base5 = (z5 > 0).astype(float)
    arms5 = {
        "③a' z5 裸兑现": pos_base5,
        "③b' z5+缩量清零": pos_base5 * (~c3["shrink_state"].fillna(False)).astype(float),
        "③c' z5+放量满仓": pd.concat([pos_base5, c3["surge_state"].astype(float)], axis=1).max(axis=1),
    }
    rows3b = []
    for name, pos in arms5.items():
        m = run_continuous(pos, d3["open"], COST_BPS)
        rows3b.append({"arm": name, "annual": m["annual"], "sharpe": m["sharpe"],
                       "mdd": m["mdd"], "calmar": m["calmar"], "avg_pos": m["avg_pos"],
                       "turnover": m["turnover"]})
    print("  诊断（非预注册，5 日平滑 z 基线，10bp）：")
    print(pd.DataFrame(rows3b).round(4).to_string(index=False))
    pd.DataFrame(rows3b).to_csv(OUT / "overlay_arms_diag.csv", index=False, encoding="utf-8-sig")
    gates.append({"gate": "G3-风控（③b 缩量清零）",
                  "pass": g3_risk,
                  "detail": f"MDD {base['mdd']:.2%}→{filt['mdd']:.2%}，年化 "
                            f"{base['annual']:.2%}→{filt['annual']:.2%}"})
    gates.append({"gate": "G3-进攻（③c 放量满仓）",
                  "pass": g3_atk,
                  "detail": f"年化 {base['annual']:.2%}→{atk['annual']:.2%}，MDD "
                            f"{base['mdd']:.2%}→{atk['mdd']:.2%}"})

    # ================= ④ 连续特征 =================
    print("\n[④] 连续特征 IC（日=open_{{t+1}}→open_{{t+2}}，周=→open_{{t+6}}；门 |日|≥2%/|周|≥3% 且半窗同号）")
    rows4 = []
    for c in (OVERLAY_CODE, "399317.SZ", "000852.SH"):
        d4 = daily[c]
        f = pd.DataFrame({
            "vr": d4["amount"] / d4["amount"].rolling(WIN).mean() - 1.0,
            "flat": d4["close"].rolling(WIN).std() / d4["close"].rolling(WIN).mean(),
        })
        tgt_d = d4["open"].shift(-2) / d4["open"].shift(-1) - 1.0
        tgt_w = d4["open"].shift(-6) / d4["open"].shift(-1) - 1.0
        half = len(f) // 2
        for feat in f.columns:
            ic_d, ic_w = ts_corr_of(f[feat], tgt_d), ts_corr_of(f[feat], tgt_w)
            h1d, h2d = ts_corr_of(f[feat].iloc[:half], tgt_d.iloc[:half]), \
                ts_corr_of(f[feat].iloc[half:], tgt_d.iloc[half:])
            h1w, h2w = ts_corr_of(f[feat].iloc[:half], tgt_w.iloc[:half]), \
                ts_corr_of(f[feat].iloc[half:], tgt_w.iloc[half:])
            stable = (np.sign(h1d) == np.sign(h2d)) or (np.sign(h1w) == np.sign(h2w))
            passed = (abs(ic_d) >= 0.02 or abs(ic_w) >= 0.03) and bool(stable)
            rows4.append({"index": INDICES[c][0], "feature": feat, "ic_daily": ic_d,
                          "ic_weekly": ic_w, "h1_d": h1d, "h2_d": h2d,
                          "h1_w": h1w, "h2_w": h2w, "pass": passed})
    t4 = pd.DataFrame(rows4)
    print(t4.round(4).to_string(index=False))
    t4.to_csv(OUT / "cont_features_ic.csv", index=False, encoding="utf-8-sig")
    gates.append({"gate": "G4 连续特征（vr/flat 过门）",
                  "pass": bool(t4["pass"].any()),
                  "detail": f"过门 {int(t4['pass'].sum())}/{len(t4)}（000300/全A/zz1000 × 2 特征）"})

    # ④b 过门 ⇒ 最小长平兑现（预注册：过门才做；口径同 ③a，另附 5 日平滑降换手版）
    print("\n[④b] vr 最小兑现（长平两态，T+1 open，10bp；对照=持有）")
    rows4b = []
    for c in (OVERLAY_CODE, "399317.SZ", "000852.SH"):
        d4 = daily[c]
        z_vr = causal_z(d4["amount"] / d4["amount"].rolling(WIN).mean() - 1.0)
        for tag, pos in (("vr_z", (z_vr > 0).astype(float)),
                         ("vr_z5", (z_vr.rolling(5).mean() > 0).astype(float))):
            m = run_continuous(pos, d4["open"], COST_BPS)
            rows4b.append({"index": INDICES[c][0], "arm": tag, "annual": m["annual"],
                           "sharpe": m["sharpe"], "mdd": m["mdd"],
                           "excess_vs_hold": m["excess"], "turnover": m["turnover"]})
    print(pd.DataFrame(rows4b).round(4).to_string(index=False))
    pd.DataFrame(rows4b).to_csv(OUT / "cont_features_realize.csv", index=False, encoding="utf-8-sig")

    # ================= 敏感性网格（只报不选峰） =================
    grid = []
    alla_c, alla_close = daily["399317.SZ"], daily["399317.SZ"]["close"]
    for flat_th in (0.015, 0.02, 0.025):
        for shrink_th in (0.7, 0.8, 0.9):
            for surge_th in (1.3, 1.5, 1.7):
                for ma_shift in (0, 1):
                    for high_win in (20, 60):
                        cc = build_conditions(alla_c, flat_th, shrink_th, surge_th,
                                              high_win=high_win, ma_shift=ma_shift)
                        row = {"flat_th": flat_th, "shrink_th": shrink_th,
                               "surge_th": surge_th, "ma_shift": ma_shift, "high_win": high_win}
                        for k in ("shrink", "surge"):
                            ev = event_table(alla_close, cc[f"{k}_event"])
                            row[f"{k}_n"] = len(ev)
                            row[f"{k}_fwd20_mean"] = ev["fwd20"].mean()
                        grid.append(row)
    g = pd.DataFrame(grid)
    g.to_csv(OUT / "sensitivity.csv", index=False, encoding="utf-8-sig")
    print(f"\n敏感性网格 {len(grid)} 组合已落盘（不用于选峰）")
    for k in ("shrink", "surge"):
        col = g[f"{k}_fwd20_mean"]
        print(f"  {k}: fwd20 均值 负/正 = {(col < 0).sum()}/{(col >= 0).sum()}，"
              f"四分位 {col.quantile(.25):.4f} ~ {col.quantile(.75):.4f}（全A）")

    print("\n===== 判定 =====")
    for g in gates:
        print(f"{g['gate']}: {'PASS' if g['pass'] else 'FAIL'}  {g['detail']}")
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
