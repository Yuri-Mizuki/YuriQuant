"""磐石（``champ``）· 相位交叉矩阵的**稳健性诊断**（配套 ``etf_champ_phase_matrix``）。

``etf_champ_phase_matrix`` 给的是相位平均主表（ann / SR / MDD / Calmar）。本脚本回答
主表回答不了的三个问题，三者都是 §5.1 判否的依据：

1. **``pair`` 收益差显著吗** —— 非重叠月度配对 t（82 个月），对定版 ``inverse_vol``。
   主表上 ``hrp`` 相位平均年化 11.10% > 定版 10.62%，但这是否只是噪声？
2. **``mdd`` 单一 MDD 可信吗** —— 12 个月滚动窗口最差回撤的**分布**（~60 个起点）。
   🚨 单一 MDD 是**一条路径的一个统计量**：``hrp`` 在 5 个相位上 MDD 都比定版低 2.4–4.5pp，
   看似稳赢；滚动分布上缩到 ~0.9pp 且有一个相位反号 ⇒ 大部分是"哪一段恰好最深"。
3. **``conc`` 集中度是什么形状** —— 单只权重的分位数 + 首位持仓归属。
   🚨 磐石"单只中位 17.7%"是**双峰分布的中位数**：国债 ETF 入选时 ``inverse_vol`` 按 1/vol
   把它顶到 50%+，不在榜时回落到 5 只各 15%。均值 29%、p75 48%、p90 57%。

附带 ``year`` 模式：逐年最优相位 + 相位间持仓 Jaccard —— 判断"月内相位是真日历效应
还是抽样噪声"。五档相位 Jaccard 0.72–0.81，有效独立试验只有 1–2 个。

用法::

    python -m scripts.portfolio.etf_champ_phase_robust            # 全部四个模式
    python -m scripts.portfolio.etf_champ_phase_robust --mode conc
    python -m scripts.portfolio.etf_champ_phase_robust --rules inverse_vol,hrp

口径与 ``etf_champ_phase_matrix`` 完全同源（同一 ``build_env`` / ``run_champ`` /
月内相位定义），**DEF 同时补进 close 与 open 两个面板**，否则防御腿权重被吞。
环境 = 系统 Python 3.12（``.venv`` 缺 ``AmazingData``）。
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from scripts.portfolio.etf_champ_phase_matrix import (
    build_env,
    month_phase_rebalance_days,
    run_champ,
)
from scripts.portfolio.etf_champ_signals import PROFILES

#: 池内国债/货币 ETF —— 用来识别「首位持仓是不是防御性资产」
TREASURY = {"511010.SH", "511260.SH", "511880.SH", "511990.SH"}
PHASES = (1, 2, 3, 4, 5)


def _monthly(eq: pd.Series) -> pd.Series:
    """月末净值 → 非重叠月度收益。"""
    return eq.resample("ME").last().pct_change().dropna()


def _roll_mdd(eq: pd.Series, win: int = 252, step: int = 21) -> pd.Series:
    """每个交易日起未来 12 个月的最差回撤（正数深度）。起点间隔 ``step`` 交易日。"""
    v, idx = eq.to_numpy(), eq.index
    out = {}
    for i in range(0, len(v) - win, step):
        seg = v[i:i + win + 1]
        out[idx[i]] = float(np.min(seg / np.maximum.accumulate(seg) - 1.0)) * -1
    return pd.Series(out)


def _hold_snapshot(r1: dict) -> tuple[pd.DataFrame, list[set]]:
    """逐日多头腿快照：最大单只权重 + 首位持仓代码；另附逐日持仓集合（算 Jaccard）。"""
    w = r1["weights"]
    held = [c for c in w.columns if c not in ("CASH", "DEF")]
    live = w[w.abs().sum(axis=1) > 1e-9][held]
    df = pd.DataFrame({"maxw": live.max(axis=1), "top1": live.idxmax(axis=1)})
    sets = [set(r[r > 1e-9].index) for _, r in live.iterrows()]
    return df, sets


def mode_pair(eq: dict, rules: list[str], base: str) -> None:
    print("\n== ① 非重叠月度配对 t（vs 定版 %s，同相位）==" % base)
    print(f"  {'规则':<13}{'k':>3}{'Δ月均值':>9}{'t':>7}{'胜率':>8}{'n':>5}")
    for wt in rules:
        if wt == base:
            continue
        for k in PHASES:
            a, b = _monthly(eq[(wt, k)]), _monthly(eq[(base, k)])
            j = a.index.intersection(b.index)
            d = (a[j] - b[j]).dropna()
            t = d.mean() / d.std(ddof=1) * np.sqrt(len(d))
            print(f"  {wt:<13}{k:>3}{d.mean() * 100:>+8.2f}%{t:>7.2f}"
                  f"{(d > 0).mean():>8.1%}{len(d):>5}")


def mode_mdd(eq: dict, rules: list[str], base: str) -> None:
    print("\n== ② 12 个月滚动窗口最差回撤分布（每 21 交易日一个起点）==")
    print(f"  {'规则':<13}{'k':>3}{'中位':>8}{'p90':>8}{'最差':>8}{'Δ中位vs定版':>12}")
    for k in PHASES:
        ref = _roll_mdd(eq[(base, k)])
        for wt in rules:
            r = _roll_mdd(eq[(wt, k)])
            j = r.index.intersection(ref.index)
            dm = r[j].median() - ref[j].median()
            print(f"  {wt:<13}{k:>3}{r.median():>8.2%}{r.quantile(0.9):>8.2%}"
                  f"{r.max():>8.2%}{dm * 100:>+11.2f}pp")
        print()


def mode_conc(hs: dict, rules: list[str]) -> None:
    print("\n== ③ 单只权重的分布形状（k=1，看中位数是否掩盖双峰）==")
    for wt in rules:
        h = hs[(wt, 1)]
        mw, ist = h["maxw"], h["top1"].isin(TREASURY)
        q = "  ".join(f"p{int(x * 100)}={mw.quantile(x):.0%}"
                      for x in (0.1, 0.25, 0.5, 0.75, 0.9))
        print(f"\n  -- {wt}（n={len(mw)} 日）--")
        print(f"     分位 {q}  均值 {mw.mean():.1%}")
        print(f"     >30% 占 {float((mw > 0.30).mean()):.1%}；"
              f">40% 占 {float((mw > 0.40).mean()):.1%}；"
              f">60% 占 {float((mw > 0.60).mean()):.1%}")
        print(f"     首位是国债 ETF 的日数占 {float(ist.mean()):.1%}"
              f"（该子集 maxw 中位 {mw[ist].median():.1%}）"
              f"；其余日子中位 {mw[~ist].median():.1%}")


def mode_year(eq: dict, sets: dict, base: str) -> None:
    print(f"\n== ④ 逐年最优相位 + 相位间持仓重叠（{base}）==")
    last = {k: eq[(base, k)].resample("YE").last() for k in PHASES}
    # ⚠️ 必须先 ``pct_change`` 再补首行：直接对水平值建表会把净值当成年收益打印。
    tab = pd.DataFrame(last).pct_change()
    tab.iloc[0] = [last[k].iloc[0] / float(eq[(base, k)].iloc[0]) - 1.0 for k in PHASES]
    print("  逐年收益：")
    print((tab * 100).round(1).to_string())
    win = tab.idxmax(axis=1)
    print(f"\n  逐年最优相位 {dict(zip(tab.index.year, win))}"
          f"；k=1 最优 {int((win == 1).sum())}/{len(win)} 年"
          f"；各相位最优次数 {win.value_counts().to_dict()}")
    print("  ⚠️ 2026 为不完整年（截 10 月），只作参考；判「是否稳定」看完整年。")
    print("  相位间逐日 Jaccard（对 k=1，越高说明五档本质是同一组合的微调）：")
    a = sets[(base, 1)]
    for k in PHASES[1:]:
        b = sets[(base, k)]
        n = min(len(a), len(b))
        j = [len(a[i] & b[i]) / max(len(a[i] | b[i]), 1) for i in range(n)]
        print(f"    k=1 vs k={k}: 均值 {np.mean(j):.3f}  最低 {np.min(j):.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rules", default="inverse_vol,hrp,min_var,erc")
    ap.add_argument("--phase", type=int, default=1,
                    help="③ 集中度诊断用哪个相位（默认 k=1，与主表锚点一致）")
    ap.add_argument("--mode", default="pair,mdd,conc",
                    help="逗号分隔：pair / mdd / conc / year")
    args = ap.parse_args(argv)

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    modes = {m.strip() for m in args.mode.split(",") if m.strip()}
    base = "inverse_vol"

    print("加载环境（全量 ETF 面板 + DEF 合成腿）…", flush=True)
    t0 = time.time()
    env = build_env()
    rb = month_phase_rebalance_days(env["close"].index)
    eq, hs, sets = {}, {}, {}
    for wt in rules:
        spec = dict(PROFILES["champ"])
        spec["weighting"] = wt
        for k in PHASES:
            _r0, r1 = run_champ(env, spec, rb[k])
            eq[(wt, k)] = r1["equity"].dropna()
            hs[(wt, k)], sets[(wt, k)] = _hold_snapshot(r1)
            print(f"  {wt:<13} k={k} 年化 {r1['annual_return']:>7.2%}  "
                  f"MDD {r1['max_drawdown']:>7.2%}", flush=True)
    print(f"回测完成 {time.time() - t0:.0f}s", flush=True)

    if "pair" in modes:
        mode_pair(eq, rules, base)
    if "mdd" in modes:
        mode_mdd(eq, rules, base)
    if "conc" in modes:
        mode_conc(hs, rules)
    if "year" in modes:
        mode_year(eq, sets, base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
