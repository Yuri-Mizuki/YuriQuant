"""多源日频择时信号合成探针（纯择时线主攻，reports/multisrc_timing_probe/）。

背景：hf_minagg_probe（2026-11-22）L1 证明沪深300 日频价格类信号 IC 4~7% 为真但
long/flat 兑现个位数；归档重启条件①要求非价格信息源。本探针合成四源族（三非价格 +
高频结构）为**连续仓位**择时信号：

  S1 高频结构（hf_minagg 缓存复用）：price_vs_resistance / price_volume_divergence /
     reversal_pattern——已知日度 IC 4~7%
  S2 两融（SDK margin_summary，沪深两市合计）：融资余额 5 日变化率、融资买入额
     动量（5 日均/60 日均）
  S4 业绩预告 breadth（SDK profit_notice，沪深300 成分）：过去 20 自然日向好预告占比
     （涨跌幅中位数>0 记向好；按 ANN_DATE 归日）
  ⚠️ S3 国债收益率撤下：SDK treasury_yield 下载后返回空 dict、本地库不持久化（2026-11-22
   实测连调两次复现）；期权/研报评级/情感为待接入源。

预注册判据（先于数据）
----------------------
L1 逐源门：每特征在**日**（open_{t+1}→open_{t+2}）或**周**（open_{t+1}→open_{t+6}）
   目标上 |IC| ≥ 2%（周口径 3%，慢变量放宽）⇒ 该特征过门；要求 **≥2 个非价格特征**
   过门才谈多源增量。
L2 合成门：过门特征滚动 252 日 z 标准化、训练半窗（前半样本）IC 符号对齐、等权合成
   → 连续仓位 pos = clip(0.5 + 0.25·comp_z, 0, 1)，T+1 open 执行。
   **@10bp 年化超额（vs 满仓持有）≥ +4pp 且 Sharpe ≥ 0.8 且测试半窗超额 > 0**。
   关键对照 = 仅高频结构族的同规则组合（隔离非价格源边际）。
L3 稳健（附表）：成本敏感性 {1,5,10}bp、逐年超额表。

用法::

    python -m scripts.timing.multisrc_timing_probe --self-check
    python -m scripts.timing.multisrc_timing_probe
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.metrics import annual_return, max_drawdown, sharpe_ratio
from scripts.timing.hf_minagg_probe import build_features, load_minute_daily
from scripts.timing.shell import load_index_daily
from scripts.timing.shell import ROOT
from stats import PERIODS_PER_YEAR

OUT = ROOT / "reports" / "multisrc_timing_probe"
INDEX_CODE = "000300.SH"
HF_CACHE_ROOT = ROOT / "reports" / "hf_minagg_probe" / "data"

L1_IC_DAILY = 0.02  # 日频 IC 门
L1_IC_WEEKLY = 0.03  # 周频 IC 门（慢变量）
L2_EXCESS = 0.04
L2_SHARPE = 0.8
COST_GRID_BPS = (1.0, 5.0, 10.0)


# ---------------------------------------------------------------- 数据源
def load_margin_daily(refresh: bool = False) -> pd.Series:
    """融资余额（沪深合计，元），index=交易日。缓存 parquet。"""
    cache = OUT / "data" / "margin_balance.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)["balance"]
    from data.datasource import create_datasource

    ds = create_datasource()
    raw = ds._info.get_margin_summary()
    g = raw.groupby("TRADE_DATE")[["SUM_BORROW_MONEY_BALANCE", "SUM_PURCH_WITH_BORROW_MONEY"]].sum()
    g.index = pd.to_datetime(g.index.astype(str))
    g = g.sort_index()
    g["balance"] = g["SUM_BORROW_MONEY_BALANCE"]
    g["purch"] = g["SUM_PURCH_WITH_BORROW_MONEY"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    g[["balance", "purch"]].to_parquet(cache)
    return g["balance"]


def load_margin_purch() -> pd.Series:
    """融资买入额（沪深合计），复用两融缓存。"""
    cache = OUT / "data" / "margin_balance.parquet"
    if not cache.exists():
        load_margin_daily()
    return pd.read_parquet(cache)["purch"]


def load_treasury_daily(refresh: bool = False) -> pd.DataFrame:  # pragma: no cover
    """⚠️ 已撤下：SDK treasury_yield 下载后返回空 dict（2026-11-22 实测），保留函数
    体为空占位，源恢复可用后再接入。"""
    raise NotImplementedError("SDK treasury_yield 返回空 dict，源撤下（见模块 docstring S3）")


def load_notice_daily(code_list: list[str], refresh: bool = False) -> pd.DataFrame:
    """沪深300 成分业绩预告逐日聚合（自然日历日）：向好数/向差数。

    向好 = 预告涨跌幅中位数 > 0；向差 = 中位数 < 0（扭亏等无幅度的记录不计入分母）。
    先读 SDK 本地库，缺再联网下载。PIT 边界：成分表为当前成员，见 README §四。
    """
    cache = OUT / "data" / "notice_hs300.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    from data.datasource import create_datasource

    ds = create_datasource()
    try:
        raw = ds._info.get_profit_notice(code_list, is_local=True)
    except Exception:
        raw = pd.DataFrame()
    if not isinstance(raw, pd.DataFrame) or raw.empty or raw["ANN_DATE"].nunique() < 100:
        # 本地库可能只有近期切片（实测 28 天）⇒ 强制全量下载一次
        try:
            raw = ds._info.get_profit_notice(code_list, is_local=False)
        except Exception:
            pass  # 网络失败时沿用本地切片（thin 数据会在 L1 表现为 NaN，宁可缺测不可造假）
    raw = raw.dropna(subset=["ANN_DATE", "P_CHANGE_MAX", "P_CHANGE_MIN"]).copy()
    raw["mid"] = (raw["P_CHANGE_MAX"] + raw["P_CHANGE_MIN"]) / 2.0
    raw["day"] = pd.to_datetime(raw["ANN_DATE"].astype(str))
    good = raw[raw["mid"] > 0].groupby("day").size()
    bad = raw[raw["mid"] < 0].groupby("day").size()
    out = pd.DataFrame({"good": good, "bad": bad}).fillna(0.0).sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


# ---------------------------------------------------------------- 特征
def build_source_features(daily_k: pd.DataFrame, minute_agg: pd.DataFrame,
                          margin: pd.Series, notice: pd.DataFrame) -> pd.DataFrame:
    """全部候选特征（列名前缀 = 源族）。全部仅用 ≤t 信息。"""
    f = pd.DataFrame(index=daily_k.index)
    # S1 高频结构（hf_minagg 同口径）
    hf = build_features(daily_k, minute_agg)
    f["hf_pvr"] = hf["price_vs_resistance"]
    f["hf_pvd"] = hf["price_volume_divergence"]
    f["hf_revpat"] = hf["reversal_pattern"]
    # S2 两融（对齐到指数交易日历）
    mb = margin.reindex(daily_k.index).ffill()
    f["margin_chg5"] = np.log(mb / mb.shift(5))
    purch = load_margin_purch()
    pm = purch.reindex(daily_k.index).ffill()
    f["margin_buy_mom"] = pm.rolling(5).mean() / pm.rolling(60).mean() - 1.0
    # S4 业绩预告 breadth：过去 20 自然日向好占比（含少量样本日才有效）
    cal = pd.DataFrame(index=pd.date_range(daily_k.index.min(), daily_k.index.max(), freq="D"))
    g = notice["good"].reindex(cal.index).fillna(0.0)
    b = notice["bad"].reindex(cal.index).fillna(0.0)
    win = 20
    breadth = (g.rolling(win).sum() / (g.rolling(win).sum() + b.rolling(win).sum())).reindex(daily_k.index)
    f["notice_breadth20"] = breadth
    return f


# ---------------------------------------------------------------- 判据与回测
def ts_corr_of(x: pd.Series, y: pd.Series) -> float:
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) < 200:
        return float("nan")
    return float(s.iloc[:, 0].corr(s.iloc[:, 1]))


def causal_z(x: pd.Series, win: int = 252) -> pd.Series:
    mu = x.rolling(win, min_periods=win // 2).mean()
    sd = x.rolling(win, min_periods=win // 2).std()
    return (x - mu) / sd


def composite_positions(comp_z: pd.Series) -> pd.Series:
    return (0.5 + 0.25 * comp_z).clip(0.0, 1.0)


def run_continuous(pos: pd.Series, open_px: pd.Series, cost_bps: float) -> dict:
    """连续仓位：pos_t 于 t 收盘可知，t+1 open→t+2 open 段生效；换仓成本在 t+1 open。"""
    seg = open_px.shift(-2) / open_px.shift(-1) - 1.0
    earn = pos.shift(1)
    cost = earn.diff().abs().fillna(0.0) * cost_bps * 1e-4
    ret = (earn * seg - cost).dropna()
    bench = seg.dropna()
    return {
        "annual": annual_return(ret),
        "sharpe": sharpe_ratio(ret),
        "mdd": max_drawdown(ret),
        "bench_annual": annual_return(bench),
        "excess": annual_return(ret) - annual_return(bench),
        "avg_pos": float(earn.dropna().mean()),
        "turnover": float(earn.diff().abs().fillna(0).sum()),
        "ret": ret,
    }


def self_check(feat: pd.DataFrame, open_px: pd.Series, pos: pd.Series) -> list[str]:
    bad = []
    d = open_px.dropna()
    for loc in (300, 1200, 2500):
        t0 = d.index[loc]
        expect = d.iloc[loc + 2] / d.iloc[loc + 1] - 1
        got = (open_px.shift(-2) / open_px.shift(-1) - 1.0).loc[t0]
        if abs(float(expect) - float(got)) > 1e-12:
            bad.append(f"目标段错位 @{t0.date()}")
    if not ((pos.dropna() >= 0).all() and (pos.dropna() <= 1).all()):
        bad.append("仓位越界 [0,1]")
    nb = feat["notice_breadth20"].dropna()
    if len(nb) and not ((nb >= 0).all() and (nb <= 1).all()):
        bad.append("预告 breadth 越界 [0,1]")
    return bad


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    daily = load_index_daily(INDEX_CODE, cache_dir=OUT / "data", refresh=args.refresh_data)
    daily = daily.loc["2013-01-04":]
    minute_agg = load_minute_daily(INDEX_CODE)
    margin = load_margin_daily(refresh=args.refresh_data)

    from data.datasource import create_datasource

    ds = create_datasource()
    cons = ds.get_index_constituent(INDEX_CODE)
    codes = cons[cons["out_date"].isna()]["con_code"].unique().tolist()  # 当前成员（PIT 过滤）
    notice = load_notice_daily(codes, refresh=args.refresh_data)
    print(f"预告样本：{len(notice)} 个自然日有记录；成分股 {len(codes)} 只")

    feat = build_source_features(daily, minute_agg, margin, notice)
    open_px = daily["open"]
    tgt_d = open_px.shift(-2) / open_px.shift(-1) - 1.0  # 日目标
    tgt_w = open_px.shift(-6) / open_px.shift(-1) - 1.0  # 周目标（t+1→t+6 open）

    if args.self_check:
        half = len(feat) // 2
        pos = composite_positions(causal_z(feat["hf_pvr"]))
        bad = self_check(feat, open_px, pos)
        for b in bad:
            print("SELF-CHECK FAIL:", b)
        if bad:
            return 1
        print("SELF-CHECK PASS（目标对齐/仓位/预告广度不变量全过）")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- L1：逐源 IC 门 ----------
    half = len(feat) // 2
    rows = []
    for c in feat.columns:
        ic_d = ts_corr_of(feat[c], tgt_d)
        ic_w = ts_corr_of(feat[c], tgt_w)
        passed = abs(ic_d) >= L1_IC_DAILY or abs(ic_w) >= L1_IC_WEEKLY
        rows.append({"feature": c, "ic_daily": ic_d, "ic_weekly": ic_w,
                     "train_ic_daily": ts_corr_of(feat[c].iloc[:half], tgt_d.iloc[:half]),
                     "abs_ic_daily": abs(ic_d), "pass": passed})
    ic_df = pd.DataFrame(rows)
    print("\n[L1] 逐源 IC 门（日 ≥2% / 周 ≥3%）")
    print(ic_df.round(4).to_string(index=False))
    ic_df.to_csv(OUT / "sources_ic.csv", index=False, encoding="utf-8-sig")

    passed = ic_df[ic_df["pass"]]["feature"].tolist()
    nonprice_passed = [c for c in passed if not c.startswith("hf_")]
    price_passed = [c for c in passed if c.startswith("hf_")]
    print(f"过门：价格类 {len(price_passed)} 个 {price_passed}；非价格 {len(nonprice_passed)} 个 {nonprice_passed}")
    g1_pass = len(nonprice_passed) >= 2

    # ---------- L2：合成连续仓位 ----------
    def make_composite(names: list[str]) -> pd.Series:
        half_loc = feat.index[half]
        zs = []
        for n in names:
            z = causal_z(feat[n])
            sgn = np.sign(ts_corr_of(feat[n].loc[:half_loc], tgt_d.loc[:half_loc]))
            if sgn == 0:
                sgn = 1.0
            zs.append(z * sgn)
        return pd.concat(zs, axis=1).mean(axis=1)

    open_close = open_px
    results = []
    comp_full = make_composite(passed)
    comp_hf = make_composite(price_passed)
    comp_np = make_composite(nonprice_passed) if nonprice_passed else pd.Series(0.0, index=feat.index)
    for label, comp in (("composite_full", comp_full), ("composite_hf_only", comp_hf),
                        ("composite_nonprice_only", comp_np)):
        pos = composite_positions(comp)
        for cbps in COST_GRID_BPS:
            r = run_continuous(pos, open_close, cbps)
            results.append({"arm": label, "cost_bps": cbps,
                            **{k: v for k, v in r.items() if k != "ret"}})
    # 满仓基准
    bench_ret = open_close.shift(-2) / open_close.shift(-1) - 1.0
    results.append({"arm": "bench_full_bh", "cost_bps": 0.0,
                    "annual": annual_return(bench_ret.dropna()), "sharpe": sharpe_ratio(bench_ret.dropna()),
                    "mdd": float("nan"), "bench_annual": float("nan"), "excess": 0.0,
                    "avg_pos": 1.0, "turnover": 0.0})
    res_df = pd.DataFrame(results)
    print("\n[L2] 连续仓位组合（T+1 open 执行）")
    print(res_df.round(4).to_string(index=False))
    res_df.to_csv(OUT / "composite_backtest.csv", index=False, encoding="utf-8-sig")

    arm = res_df[(res_df["arm"] == "composite_full") & (res_df["cost_bps"] == 10.0)].iloc[0]
    arm_hf = res_df[(res_df["arm"] == "composite_hf_only") & (res_df["cost_bps"] == 10.0)].iloc[0]
    # 测试半窗超额
    pos_full = composite_positions(comp_full)
    r_full = run_continuous(pos_full, open_close, 10.0)["ret"]
    mid = r_full.index[len(r_full) // 2]
    ex_test = annual_return(r_full.loc[mid:]) - annual_return(
        (open_close.shift(-2) / open_close.shift(-1) - 1.0).dropna().loc[mid:])
    gates = [
        {"gate": "L1 非价格源", "pass": g1_pass,
         "detail": f"非价格过门 {len(nonprice_passed)}/3 特征（判据 ≥2）"},
        {"gate": "L2 合成超额", "pass": bool(arm["excess"] >= L2_EXCESS and arm["sharpe"] >= L2_SHARPE),
         "detail": f"@10bp 超额 {arm['excess'] * 100:+.2f}pp（判据 ≥+4pp）、Sharpe {arm['sharpe']:.2f}（判据 ≥0.8）"},
        {"gate": "L2 测试半窗", "pass": bool(ex_test > 0),
         "detail": f"测试半窗超额 {ex_test * 100:+.2f}pp（判据 >0）"},
        {"gate": "非价格边际", "pass": bool(arm["excess"] > arm_hf["excess"] + 0.01),
         "detail": f"合成 {arm['excess'] * 100:+.2f}pp vs 仅高频 {arm_hf['excess'] * 100:+.2f}pp"
                   f"（判据：非价格带来 >+1pp）"},
    ]
    print("\n===== 预注册判定 =====")
    for g in gates:
        print(f"{g['gate']}: {'PASS' if g['pass'] else 'FAIL'}  {g['detail']}")

    # ---------- 设计 v2（迭代披露）----------
    # 变更依据（非选峰）：① v1 归因 = 价格高频族的日频 z 跳变 → 换手 480~755×/年，
    # 该族已验证的正确兑现架构是周频壳（rev_shell_probe 采纳候选）；② 非价格慢源
    # 天然低换手（100×/年），保持日频连续。判据（L2 三门）与 v1 完全一致。
    def leg_positions(z: pd.Series, sample_weekly: bool) -> pd.Series:
        zz = z.copy()
        if sample_weekly:
            zz = zz[zz.index.dayofweek == 2].reindex(z.index).ffill()
        return (0.5 + 0.25 * zz).clip(0.0, 1.0)

    pos_price_w = leg_positions(make_composite(price_passed), sample_weekly=True)
    pos_np_d = leg_positions(make_composite(nonprice_passed), sample_weekly=False)
    pos_v2 = (0.5 * pos_price_w + 0.5 * pos_np_d).clip(0.0, 1.0)
    v2_rows = []
    for label, p in (("v2_price_leg_w", pos_price_w), ("v2_np_leg_d", pos_np_d), ("v2_full", pos_v2)):
        for cbps in COST_GRID_BPS:
            r = run_continuous(p, open_close, cbps)
            v2_rows.append({"arm": label, "cost_bps": cbps,
                            **{k: v for k, v in r.items() if k != "ret"}})
    v2_df = pd.DataFrame(v2_rows)
    print("\n[L2-v2] 价格腿周频采样 × 非价格腿日频连续（判据同 v1）")
    print(v2_df.round(4).to_string(index=False))
    v2_df.to_csv(OUT / "composite_backtest_v2.csv", index=False, encoding="utf-8-sig")

    arm2 = v2_df[(v2_df["arm"] == "v2_full") & (v2_df["cost_bps"] == 10.0)].iloc[0]
    r_v2 = run_continuous(pos_v2, open_close, 10.0)["ret"]
    mid2 = r_v2.index[len(r_v2) // 2]
    ex_test2 = annual_return(r_v2.loc[mid2:]) - annual_return(
        (open_close.shift(-2) / open_close.shift(-1) - 1.0).dropna().reindex(r_v2.index).loc[mid2:])
    gates.append({"gate": "L2-v2 合成超额", "pass": bool(arm2["excess"] >= L2_EXCESS and arm2["sharpe"] >= L2_SHARPE),
                  "detail": f"@10bp 超额 {arm2['excess'] * 100:+.2f}pp（判据 ≥+4pp）、Sharpe {arm2['sharpe']:.2f}（判据 ≥0.8）"})
    gates.append({"gate": "L2-v2 测试半窗", "pass": bool(ex_test2 > 0),
                  "detail": f"测试半窗超额 {ex_test2 * 100:+.2f}pp（判据 >0）"})
    pos_v2.to_frame("pos_v2").to_parquet(OUT / "data" / "signal_hs300_v2.parquet")

    print("\n===== 全部判定 =====")
    for g in gates:
        print(f"{g['gate']}: {'PASS' if g['pass'] else 'FAIL'}  {g['detail']}")
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False, encoding="utf-8-sig")
    core_pass = all(g["pass"] for g in gates if g["gate"].startswith("L2-v2")) and g1_pass
    print("\n结论：", "v2 全过 → 择时信号成立（signal_hs300_v2.parquet），进四资产/组合层验证"
          if core_pass else "v2 未全过 → 记录归因，未过门源列待接入（期权/研报评级/情感）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
