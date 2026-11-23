"""降频兑现探针（纯择时线第六篇，reports/downfreq_realize_probe/）。

动机：第五篇成本分层（z5 类周频翻转 ≈5%/年@10bp、30bp 死；壳 53 次换手 30bp 无感）
指明 us_ret1 兑现的唯一开放方向 = 降频。预注册见 TODO.md「第六篇预注册」（2026-11-23）：

  A0 = z5>0 日频基线（第五篇 L2 锚）   A1 = z5 月频采样（月末交易日取值）
  A2 = z5 周三采样（对齐壳惯例）        A3/A4 = 滞回带（z5>+θ 进 / z5<−θ 出，θ=0.5/0.25）
  vr-z 同构造 A1–A4 作信息臂（线索级，不判门）

门：G1 成本存活（预注册臂 30bp 净超额>0 且两半窗同号、avg_pos≥0.3）/
G2 显著性（该臂 10bp NW-t≥2.0）/ G3 不劣（10bp 超额 ≥ A0 的 50%）。
判读：G1+G2 过 ⇒ 兑现成立；G1 过 G2 不过 ⇒ 成本稳健、信息量天花板；G1 不过 ⇒ 收线。
机制归因：A1 存活=边缘在 regime 级、仅 A3/A4 存活=大动量门控、全灭=纯日频不可分。

用法::

    python -m scripts.timing.downfreq_realize_probe --self-check
    python -m scripts.timing.downfreq_realize_probe
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest.metrics import annual_return, calmar_ratio, max_drawdown, sharpe_ratio
from scripts.timing.hf_vol_timing_probe import build_us_ret1, causal_z, run_continuous
from scripts.timing.shell import ROOT, load_index_daily
from stats.robust_stats import nw_tstat

OUT = ROOT / "reports" / "downfreq_realize_probe"
INDEX_CODE = "000300.SH"
BEGIN = "2016-01-01"
HALF = "2021-01-01"
WIN = 20
COSTS = (10.0, 30.0)
ARMS = ("A0_z5_daily", "A1_monthly", "A2_weekly", "A3_hyst05", "A4_hyst025")


# ---------------------------------------------------------------- 构造
def hysteresis(z5: pd.Series, th: float) -> pd.Series:
    """滞回带：z5>+θ 进（1）、z5<−θ 出（0），带内持有；初始 0。仅用 ≤t 信息。"""
    state, out = 0.0, []
    for v in z5.values:
        if not np.isnan(v):
            if v > th:
                state = 1.0
            elif v < -th:
                state = 0.0
        out.append(state)
    return pd.Series(out, index=z5.index, dtype=float)


def build_positions(z5: pd.Series) -> dict[str, pd.Series]:
    """A0–A4（z5 于 t 收盘可知；A1/A2 采样日在采样时点取值、段内持有）。"""
    sig = (z5 > 0).astype(float)
    idx = z5.index
    me_dates = idx.to_series().groupby(idx.to_period("M")).tail(1).index  # type: ignore[union-attr]
    month_end = pd.Series(idx.isin(me_dates), index=idx)
    weekly = pd.Series(idx.weekday == 2, index=idx)
    sampled = sig.where(month_end).ffill()  # 首个采样日前 NaN（不建仓）
    a1 = sampled
    a2 = sig.where(weekly).ffill()
    return {
        "A0_z5_daily": sig,
        "A1_monthly": a1,
        "A2_weekly": a2,
        "A3_hyst05": hysteresis(z5, 0.5),
        "A4_hyst025": hysteresis(z5, 0.25),
    }


def arm_metrics(pos: pd.Series, open_px: pd.Series, cost: float, half_bench: pd.Series) -> tuple[dict, pd.Series]:
    m = run_continuous(pos, open_px, cost)
    r = m["ret"].loc[BEGIN:]
    years = len(r) / 252.0
    d = (r - half_bench).dropna()
    row = {"annual": annual_return(r), "sharpe": m["sharpe"], "mdd": m["mdd"],
           "avg_pos": m["avg_pos"], "turn_yr": m["turnover"] / max(years, 1e-9),
           "excess": annual_return(r) - annual_return(half_bench),
           "nw_t": float(nw_tstat(d)[0]), "n": len(r)}
    return row, r


# ---------------------------------------------------------------- 自检
def self_check() -> list[str]:
    bad = []
    daily = load_index_daily(INDEX_CODE, begin=20130101, cache_dir=OUT / "data")
    z5 = causal_z(build_us_ret1(daily)).rolling(5).mean()
    pos = build_positions(z5)
    idx = daily.index
    # A1 只在月末当日记录采样决策（执行 T+1 open 由 run_continuous 保证）：
    # 变仓日 ∈ 月末集合 ∪ {首个有效日}
    a1 = pos["A1_monthly"].dropna()
    first_day = a1.index[0]
    chg = a1[a1.diff().fillna(0) != 0].index
    me_dates = set(pd.Index(idx.to_series().groupby(idx.to_period("M")).tail(1)))  # type: ignore[union-attr]
    for t in chg:
        if t != first_day and t not in me_dates:
            bad.append(f"A1 在非月末变仓 @{t.date()}")
            break
    # 滞回抽查：变仓日 z5 必在带外
    for name, th in (("A3_hyst05", 0.5), ("A4_hyst025", 0.25)):
        p = pos[name]
        for t in p.index[p.diff().fillna(0) != 0]:
            v = z5.loc[t]
            if not ((p.loc[t] == 1.0 and v > th) or (p.loc[t] == 0.0 and v < -th)):
                bad.append(f"{name} 变仓时 z5={v:.2f} 不在带外 @{t.date()}")
                break
    # us_ret1 无内部 NaN + 仓位界内
    u = build_us_ret1(daily)
    first = u.first_valid_index()
    if u.loc[first:].isna().any():
        bad.append("us_ret1 内部 NaN")
    for k, p in pos.items():
        if not p.dropna().between(0, 1).all():
            bad.append(f"{k} 仓位越界")
    # 执行段对齐抽查
    op = daily["open"].dropna()
    for loc in (300, 1200, 2500):
        if loc + 2 >= len(op):
            continue
        expect = op.iloc[loc + 2] / op.iloc[loc + 1] - 1
        got = (daily["open"].shift(-2) / daily["open"].shift(-1) - 1.0).iloc[loc]
        if abs(float(expect) - float(got)) > 1e-12:
            bad.append(f"目标段错位 @{op.index[loc].date()}")
    return bad


# ---------------------------------------------------------------- 主流程
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
        print("SELF-CHECK PASS（A1 变仓时点/滞回带/仓位界/us_ret1 无缺口/执行段对齐全过）")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    daily = load_index_daily(INDEX_CODE, begin=20130101, cache_dir=OUT / "data",
                             refresh=args.refresh_data)
    open_px = daily["open"]
    z_us = causal_z(build_us_ret1(daily, refresh=args.refresh_data)).rolling(5).mean()
    z_vr = causal_z(daily["amount"] / daily["amount"].rolling(WIN).mean() - 1.0).rolling(5).mean()
    half_bench = run_continuous(pd.Series(0.5, index=daily.index), open_px, 0.0)["ret"].loc[BEGIN:]

    rows, gates = [], []
    for tag, z5, gated in (("us_ret1", z_us, True), ("vr(信息臂)", z_vr, False)):
        pos = build_positions(z5)
        for arm in ARMS:
            for cost in COSTS:
                m, seg = arm_metrics(pos[arm], open_px, cost, half_bench)
                h1 = annual_return(seg.loc[:HALF]) - annual_return(half_bench.loc[:HALF])
                h2 = annual_return(seg.loc[HALF:]) - annual_return(half_bench.loc[HALF:])
                rows.append({"signal": tag, "arm": arm, "cost_bp": cost, **m,
                             "h1_excess": h1, "h2_excess": h2})
    tbl = pd.DataFrame(rows)
    print("\n[表] 降频臂 × 成本（000300，T+1 open，vs 半仓；窗口 2016→now）")
    print(tbl.round(4).to_string(index=False))
    tbl.to_csv(OUT / "arms_perf.csv", index=False, encoding="utf-8-sig")

    u = tbl[tbl["signal"] == "us_ret1"]
    a0_ex = float(u[(u["arm"] == "A0_z5_daily") & (u["cost_bp"] == 10.0)]["excess"].iloc[0])
    cand = []
    for arm in ARMS[1:]:
        r30 = u[(u["arm"] == arm) & (u["cost_bp"] == 30.0)].iloc[0]
        if r30["excess"] > 0 and r30["h1_excess"] > 0 and r30["h2_excess"] > 0 and r30["avg_pos"] >= 0.3:
            cand.append(arm)
    g1 = bool(cand)
    g2_arm = [a for a in cand
              if float(u[(u["arm"] == a) & (u["cost_bp"] == 10.0)]["nw_t"].iloc[0]) >= 2.0]
    g2 = bool(g2_arm)
    g3_arm = [a for a in cand
              if float(u[(u["arm"] == a) & (u["cost_bp"] == 10.0)]["excess"].iloc[0]) >= 0.5 * a0_ex]
    g3 = bool(set(g2_arm) & set(g3_arm)) if g2_arm else False
    gates = [
        {"gate": "G1 成本存活（30bp 超额>0 双半窗同号 avg_pos≥0.3）", "pass": g1,
         "detail": f"存活臂：{cand or '无'}（A0 基线 30bp "
                   f"{float(u[(u['arm'] == 'A0_z5_daily') & (u['cost_bp'] == 30.0)]['excess'].iloc[0]):+.2%}）"},
        {"gate": "G2 显著性（存活臂 10bp NW-t≥2.0）", "pass": g2,
         "detail": f"过门臂：{g2_arm or '无'}；各臂 NW-t@10bp "
                   f"{ {a: round(float(u[(u['arm'] == a) & (u['cost_bp'] == 10.0)]['nw_t'].iloc[0]), 2) for a in ARMS} }"},
        {"gate": "G3 不劣（10bp 超额 ≥ A0 的 50%）", "pass": g3,
         "detail": f"同时过 G2/G3 的臂：{sorted(set(g2_arm) & set(g3_arm)) or '无'}"
                   f"（门槛 {0.5 * a0_ex:.2%}）"},
    ]

    print("\n===== 判定 =====")
    for g in gates:
        print(f"{g['gate']}: {'PASS' if g['pass'] else 'FAIL'}  {g['detail']}")
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False, encoding="utf-8-sig")
    print("\n判读（预注册）：", "兑现成立 → 进采纳候选（降频形态）" if g1 and g2 else
          ("成本稳健、显著性未达标 → 兑现降档开放" if g1 else
           "降频也救不了 → us_ret1 兑现收线（IC 层结论保留）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
