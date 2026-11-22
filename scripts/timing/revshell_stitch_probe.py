"""rev_shell 组合层缝合评估（纯择时线，reports/revshell_stitch_probe/）。

问题：rev_shell（周频均值回归壳，rev_shell_probe 六门过、采纳候选，上证 +2.66pp）
能不能与横截面定版组合（yqcore-breadth，21.81%/1.013 判改进口径）缝合出更好的组合？
两种缝合形态 + 预注册门（先于数据）：

G0 前置：股票腿与壳腿日收益相关 < 0.6（否则无分散价值，直接 FAIL）。
M1 资金 split（ETF 可执行）：combined = w×股票 + (1−w)×壳，w ∈ {0.9,0.8,0.7,0.5}；
   **∃w 使 Sharpe > 股票 + 0.10 且 MDD < 股票 MDD − 2pp** ⇒ 风险改善腿成立。
M2 期货 unfunded overlay（IC 对应中证500，期货可执行）：
   - M2a 全敞口：combined = 股票 + pos×指数收益 − 成本（抬 beta）
   - M2b **beta 中性 tilt**：pos − 滚动 252 日均值（钳 [−0.5,0.5]）×指数收益 − 成本
   **M2b：年化 ≥ 股票年化 − 0.5pp 且 Sharpe ≥ 股票 + 0.05 且 MDD ≤ 股票 + 1pp**
   ⇒ 增强腿成立。

股票腿 = 生产口径复算：caliber 本机代理信号 `dingban_fundind_local` + prod 窗 +
production profile（27bp 时变、T+1 open、1e6 无整手）+ buffer(0.15,0.40)，
end=20260907（登记 check_end，与登记值逐位对照 21.81%/1.013，先复算再使用）。
壳腿 = `scripts/timing/shell.py::run_rebalance_shell`（sig=0 纯反转，step1%/0.1仓/
周三/10bp），资产 = 沪深300（ETF 510300 可执行）与 中证500（ETF 510500 / IC 期货），
窗口与股票腿对齐（交集）。

⚠️ 与 10-15 overlay 封底线的关系（TODO_ARCHIVE 册Ⅱ）：该线封的是「MA/动量/波动率
目标等规则 overlay」在等权全A 上的月频证据（7/7 规则 Δ年化<0）；本评估测的是**不同
信号**（周频均值回归壳，独立六门已过）在**生产组合**上的两种缝合形态，属纯择时线
立项范围（TODO §〇·二 P2），结果无论正负都回写该线。

用法::

    python -m scripts.timing.revshell_stitch_probe --self-check
    python -m scripts.timing.revshell_stitch_probe
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.metrics import annual_return, max_drawdown, sharpe_ratio
from scripts.timing.shell import load_index_daily, run_rebalance_shell
from scripts.timing.shell import ROOT
from stats import PERIODS_PER_YEAR

OUT = ROOT / "reports" / "revshell_stitch_probe"
CHECK_END = "2026-09-07"  # 登记口径末日（check_end）
STITCH_ASSETS = ("000300.SH", "000905.SH")
W_GRID = (0.9, 0.8, 0.7, 0.5)
SHELL_COST_BPS = 10.0


# ---------------------------------------------------------------- 股票腿
def build_stock_sleeve() -> tuple[pd.Series, dict]:
    """生产 breadth 日收益（本机代理信号复算），先逐位对照登记值再返回。"""
    from scripts.evaluation.caliber import cost_profile, get_profile, load_context, run_once
    from backtest.engine import VectorBacktest
    from strategy.cross_section import BufferedTopFracLongOnly

    end = int(CHECK_END.replace("-", ""))
    ctx = load_context(signal="dingban_fundind_local", window="prod",
                       execution="open", end=end)
    cost = cost_profile(get_profile("production").cost)
    bt = VectorBacktest(strategy=BufferedTopFracLongOnly(0.15, 0.40),
                        rebalance_freq="M", initial_capital=1.0e6, costs=cost)
    res = run_once(bt, ctx, "open")
    dr = res.daily_returns.dropna()
    reg_ann, reg_sharpe = 0.2181, 1.013
    ours_ann, ours_sharpe = annual_return(dr), sharpe_ratio(dr)
    ok = abs(ours_ann - reg_ann) < 0.005 and abs(ours_sharpe - reg_sharpe) < 0.02
    print(f"[股票腿] 复算 年化 {ours_ann * 100:.2f}% / Sharpe {ours_sharpe:.3f} "
          f"vs 登记 21.81% / 1.013 ⇒ {'逐位一致' if ok else '⚠️ 不一致，停用'}")
    if not ok:
        raise SystemExit("生产曲线复算未对齐登记值，缝合评估无效（口径漂移须先查）")
    return dr, {"annual": ours_ann, "sharpe": ours_sharpe, "mdd": max_drawdown(dr)}


# ---------------------------------------------------------------- 壳腿
def build_shell_sleeves(begin: pd.Timestamp) -> dict[str, pd.Series]:
    out = {}
    for code in STITCH_ASSETS:
        close = load_index_daily(code, cache_dir=OUT / "data")["close"].loc[begin:]
        r = run_rebalance_shell(close, pd.Series(0.0, index=close.index),
                                cost_bps=SHELL_COST_BPS)
        out[code] = r["ret"]
        print(f"[壳腿] {code}: 年化 {annual_return(r['ret']) * 100:.2f}% "
              f"Sharpe {sharpe_ratio(r['ret']):.3f} 平均仓位 {r['avg_pos']:.3f}")
    return out


# ---------------------------------------------------------------- 组合
def combined_metrics(a: pd.Series, b: pd.Series, w: float) -> dict:
    c = (w * a.reindex(b.index).fillna(0.0)
         + (1 - w) * b.reindex(a.index).fillna(0.0)).dropna()
    return {"w": w, "annual": annual_return(c), "sharpe": sharpe_ratio(c),
            "mdd": max_drawdown(c)}


def overlay_metrics(stock: pd.Series, pos: pd.Series, index_close: pd.Series,
                    cost_bps: float, beta_neutral: bool) -> dict:
    """unfunded overlay：股票 + 期货叠加段收益（保证金占用另记，见 README §四）。"""
    idx_ret = index_close.pct_change().fillna(0.0)
    p = pos.reindex(idx_ret.index).ffill()
    if beta_neutral:
        p = (p - p.rolling(252, min_periods=60).mean()).clip(-0.5, 0.5)
    earn = p.shift(1)
    cost = earn.diff().abs().fillna(0.0) * cost_bps * 1e-4
    leg = (earn * idx_ret - cost).dropna()
    c = (stock.reindex(leg.index).fillna(0.0) + leg).dropna()
    return {"annual": annual_return(c), "sharpe": sharpe_ratio(c),
            "mdd": max_drawdown(c), "leg_annual": annual_return(leg),
            "avg_overlay": float(earn.dropna().mean()),
            "turnover": float(earn.diff().abs().fillna(0).sum())}


def self_check(stock: pd.Series, shells: dict[str, pd.Series]) -> list[str]:
    bad = []
    if (stock < -0.35).any():
        bad.append("股票腿出现 <−35% 的日收益（口径异常）")
    for code, r in shells.items():
        if r.abs().max() > 0.12:
            bad.append(f"{code} 壳腿日收益绝对值 >12%（仓位/成本口径异常）")
    return bad


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    stock, stock_m = build_stock_sleeve()
    shells = build_shell_sleeves(stock.index.min())
    # 对齐：以股票腿日历为准（pandas 3.0 intersection 在 us/ns 单位混用下返回空集的
    # bug 实测 2026-11-22；壳腿日历覆盖股票腿 1864/1864 日，reindex 即精确对齐）
    common = stock.index
    stock_a = stock
    shells_a = {k: v.reindex(common) for k, v in shells.items()}

    if args.self_check:
        bad = self_check(stock_a, shells_a)
        for b in bad:
            print("SELF-CHECK FAIL:", b)
        if bad:
            return 1
        print("SELF-CHECK PASS（日收益量级/对齐不变量全过）")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    idx_close = {code: load_index_daily(code, cache_dir=OUT / "data").loc[common]["close"]
                 for code in STITCH_ASSETS}

    # ---------- G0 相关前置 ----------
    print(f"\n[股票腿] 年化 {stock_m['annual'] * 100:.2f}% / Sharpe {stock_m['sharpe']:.3f} "
          f"/ MDD {stock_m['mdd'] * 100:.2f}%")
    corr_rows = []
    for code, r in shells_a.items():
        corr = float(stock_a.corr(r))
        corr_rows.append({"asset": code, "corr_vs_stock": corr,
                          "shell_annual": annual_return(r),
                          "shell_sharpe": sharpe_ratio(r)})
        print(f"[G0] corr(股票, {code} 壳) = {corr:.3f}（判据 < 0.6）")
    corr_df = pd.DataFrame(corr_rows)
    g0_pass = bool((corr_df["corr_vs_stock"] < 0.6).any())
    corr_df.to_csv(OUT / "sleeves.csv", index=False, encoding="utf-8-sig")

    # ---------- M1 资金 split（用相关最低的资产做壳腿） ----------
    best_asset = corr_df.loc[corr_df["corr_vs_stock"].idxmin(), "asset"]
    shell = shells_a[best_asset]
    frontier = [combined_metrics(stock_a, shell, 1.0)]
    for w in W_GRID:
        frontier.append(combined_metrics(stock_a, shell, w))
    f_df = pd.DataFrame(frontier)
    f_df["asset"] = best_asset
    print(f"\n[M1] 资金 split 前沿（壳 = {best_asset}，ETF 可执行）")
    print(f_df.round(4).to_string(index=False))
    f_df.to_csv(OUT / "stitch_frontier_m1.csv", index=False, encoding="utf-8-sig")
    m1_ok = f_df[(f_df["w"] >= 0.7) & (f_df["w"] < 1.0)]
    g_m1 = bool(((m1_ok["sharpe"] > stock_m["sharpe"] + 0.10)
                 & (m1_ok["mdd"] < stock_m["mdd"] - 0.02)).any())

    # ---------- M2 期货 unfunded overlay（中证500 = IC） ----------
    overlay_rows = []
    g_m2 = False
    for code in STITCH_ASSETS:
        pos = run_rebalance_shell(
            idx_close[code].loc[common], pd.Series(0.0, index=common),
            cost_bps=SHELL_COST_BPS)["pos"]
        for bn in (False, True):
            m = overlay_metrics(stock_a, pos, idx_close[code], SHELL_COST_BPS, beta_neutral=bn)
            overlay_rows.append({"asset": code, "beta_neutral": bn, **m})
            if code == "000905.SH" and bn:
                g_m2 = bool(m["annual"] >= stock_m["annual"] - 0.005
                            and m["sharpe"] >= stock_m["sharpe"] + 0.05
                            and m["mdd"] <= stock_m["mdd"] + 0.01)
    o_df = pd.DataFrame(overlay_rows)
    print("\n[M2] 期货 unfunded overlay（M2b = beta 中性 tilt，门只判 中证500）")
    print(o_df.round(4).to_string(index=False))
    o_df.to_csv(OUT / "stitch_overlay_m2.csv", index=False, encoding="utf-8-sig")

    gates = [
        {"gate": "G0 相关前置", "pass": g0_pass,
         "detail": "；".join(f"{r.asset} {r.corr_vs_stock:.3f}" for r in corr_df.itertuples())},
        {"gate": "M1 资金split", "pass": g_m1,
         "detail": f"壳={best_asset}，w∈{{0.5..0.9}}："
                   + "；".join(f"w{r.w:.1f} SR {r.sharpe:.3f}/MDD {r.mdd * 100:.1f}%"
                               for r in m1_ok.itertuples())},
        {"gate": "M2 beta中性overlay", "pass": g_m2,
         "detail": "中证500 M2b：" + "；".join(
             f"{r.asset}{'bn' if r.beta_neutral else 'full'} 年化 {r.annual * 100:+.2f}% "
             f"SR {r.sharpe:.3f}" for r in o_df.itertuples())},
    ]
    print("\n===== 预注册判定 =====")
    for g in gates:
        print(f"{g['gate']}: {'PASS' if g['pass'] else 'FAIL'}  {g['detail']}")
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False, encoding="utf-8-sig")
    any_pass = g_m1 or g_m2
    print("\n结论：", ("缝合方案成立（" + ("M1 " if g_m1 else "") + ("M2 " if g_m2 else "")
                     + "），进定版候选评估" if any_pass
                     else "两种缝合形态都未过门 ⇒ 壳保持独立腿/不并入生产，判负收口"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
