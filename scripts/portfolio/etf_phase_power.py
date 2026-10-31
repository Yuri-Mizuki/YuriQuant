"""ETF 轮动 · 三锋「相位检验力」分析：多相位重测到底值多少样本（2026-10-31）。

## 为什么需要这个脚本

`reports/etf_contest5/README.md` §七 的 40 条判否里，绝大多数是**在 k=1 相位**
（周一调仓）测的。长期实盘只能用相位平均口径（§4.7）。于是有个真问题：

> 把判否的实验拿到 5 个相位上重测，能提高多少检验力？

磐石侧已答过（§5.3.3）：相位日收益 ρ=0.943 ⇒ `n_eff = 5/(1+4ρ) = 1.05`
（名义 5×，真实 1.05×）。**但三锋结构不同**：单只中位权重 80.7%，相位极差
20.88pp 是磐石（4.2pp）的 5 倍 ⇒ 不能把磐石的 ρ 搬过来用。

本脚本测三锋自己的 ρ、n_eff，以及决定性的一步：
**相位平均之后，`hrp` 相对其它档位的优势还显著吗？**

## 三个量

1. **ρ 与 n_eff**：5 个相位日收益的两两相关。`n_eff = k/(1+(k−1)ρ)`。
2. **配对 t 的相位结构**：对每个相位 k 算 `Δ_k = A−B` 的 2 月窗口配对 t；
   再看**逐窗口的 5 相位平均** `Δ̄_t` 的 t —— 后者才是「相位平均口径」的检验。
3. **判否翻案检查**：k=1 的排序 vs 相位平均排序（Spearman）。

⚠️ 口径：T+1 open、PIT 修复后（2026-10-29）、同池 45 只、同成本 —— 与
`etf_weight_probe.run` 完全同源，只改 `weighting` 与 `rebalance_days`。
相位分组用全索引（含 2018-2019 暖机，与 `weight_probe_phase.csv` 同法）。

## 用法

    python -m scripts.portfolio.etf_phase_power
    python -m scripts.portfolio.etf_phase_power --rules hrp,erc --from-csv

产物：`reports/etf_contest5/phase_power.csv`
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.portfolio.etf_contest5_eval import two_month_windows
from scripts.portfolio.etf_contest_spec import FINAL
from scripts.portfolio.etf_weight_probe import load_env, run

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "reports/etf_contest5/phase_power.csv"

PHASES = (1, 2, 3, 4, 5)
#: 对照档：`hrp` 是定版，其余是 §七 #27 里被否掉的、且差距最大的三个。
DEFAULT_RULES = ("hrp", "erc", "min_var", "equal")


def phase_rebalance_days(index: pd.DatetimeIndex) -> dict[int, set]:
    """周内第 k 个交易日调仓（k=1..5）。与 `etf_weight_probe.mode_phase` 同法。"""
    per = pd.Series(list(index), index=index).groupby(index.to_period("W"))
    out: dict[int, set] = {}
    for k in PHASES:
        rb = set()
        for _, g in per:
            v = list(g)
            if len(v) > k - 1:
                rb.add(v[k - 1])
        out[k] = rb
    return out


def collect(env, rules, phases) -> tuple[dict, dict]:
    """跑 (rule, phase) 全格，返回 equity 与 2 月窗口序列。"""
    rb_map = phase_rebalance_days(env["close"].index)
    eq: dict[tuple[str, int], pd.Series] = {}
    win: dict[tuple[str, int], pd.Series] = {}
    n = len(rules) * len(phases)
    for i, wt in enumerate(rules, 1):
        spec = dict(FINAL)
        spec["weighting"] = wt
        for k in phases:
            t0 = time.time()
            _r0, r1 = run(env, spec, rebalance_days=rb_map[k])
            eq[(wt, k)] = r1["equity"].dropna()
            win[(wt, k)] = two_month_windows(r1["equity"], env["mfs"])
            print(f"  [{i}/{len(rules)}] {wt:<13} k={k}  "
                  f"年化 {r1['annual_return']:>7.2%}  SR {r1['sharpe']:>5.2f}  "
                  f"MDD {r1['max_drawdown']:>7.2%}  ({time.time() - t0:.0f}s)",
                  flush=True)
    print(f"回测完成，共 {n} 格", flush=True)
    return eq, win


def corr_block(eq, rules, phases) -> tuple[pd.DataFrame, float, float]:
    """相位间相关（日收益 / 净值水平），返回 (矩阵, 平均 ρ, n_eff)。"""
    base = rules[0]
    R = pd.DataFrame({k: eq[(base, k)].pct_change(fill_method=None)
                      for k in phases}).dropna()
    L = pd.DataFrame({k: eq[(base, k)] for k in phases}).dropna()
    c = R.corr()
    cl = L.corr()
    upper = np.triu(np.ones(c.shape), 1).astype(bool)
    tri = c.where(upper).stack()
    rho_level = float(cl.where(upper).stack().mean())
    rho = float(tri.mean())
    k = len(phases)
    n_eff = k / (1.0 + (k - 1) * rho)
    print(f"\n== ① 相位间相关（{base}，日收益，n={len(R)} 日）==")
    print("      " + "".join(f"{'k=' + str(x):>8}" for x in c.columns))
    for r in c.index:
        print(f"  k={r}" + "".join(f"{c.loc[r, x]:>8.3f}" for x in c.columns))
    print(f"  平均 ρ = {rho:.3f}（min {tri.min():.3f} / max {tri.max():.3f}）"
          f"；净值水平相关均值 {rho_level:.3f}")
    print(f"  ⇒ n_eff = {k}/(1+{k - 1}×{rho:.3f}) = **{n_eff:.2f}**"
          f"（名义 {k}× ，真实 {n_eff:.2f}×）")
    return c, rho, n_eff


def paired_t(a: pd.Series, b: pd.Series) -> tuple[float, float, float, int]:
    """同窗口配对差 a−b 的 (均值, t, 胜率, n)。重叠窗口 ⇒ t 偏乐观，只作同口径对照。"""
    idx = a.index.intersection(b.index)
    d = (a[idx] - b[idx]).dropna()
    if len(d) < 3:
        return float("nan"), float("nan"), float("nan"), len(d)
    t = float(d.mean() / d.std(ddof=1) * np.sqrt(len(d)))
    return float(d.mean()), t, float((d > 0).mean()), len(d)


def t_block(win, rules, phases) -> pd.DataFrame:
    """配对 t：每个相位单独 vs 逐窗口 5 相位平均（= 相位平均口径）。"""
    base = rules[0]
    rows = []
    print(f"\n== ② 配对 t：Δ = {base} − 对照，2 月窗口（重叠）==")
    print(f"  {'对照':<11}{'口径':<10}{'Δ均值':>9}{'t':>7}{'胜率':>8}{'n':>5}"
          f"{'年化差pp':>10}")
    for wt in rules[1:]:
        # 相位平均口径：逐窗口把 5 个相位的 Δ 取平均
        dfs = []
        for k in phases:
            i = win[(base, k)].index.intersection(win[(wt, k)].index)
            dfs.append((win[(base, k)][i] - win[(wt, k)][i]).rename(k))
        avg = pd.concat(dfs, axis=1).mean(axis=1).dropna()
        t_avg = float(avg.mean() / avg.std(ddof=1) * np.sqrt(len(avg)))
        print(f"  {wt:<11}{'相位平均':<10}{avg.mean():>+9.3%}{t_avg:>7.2f}"
              f"{(avg > 0).mean():>8.1%}{len(avg):>5}"
              f"{avg.mean() * 6 * 100:>+10.2f}")
        m, t, w, n = paired_t(win[(base, 1)], win[(wt, 1)])
        print(f"  {wt:<11}{'k=1':<10}{m:>+9.3%}{t:>7.2f}{w:>8.1%}{n:>5}"
              f"{m * 6 * 100:>+10.2f}")
        # 各相位单独，看 t 的相位离散
        ts = []
        for k in phases:
            _m, tk, _w, _n = paired_t(win[(base, k)], win[(wt, k)])
            ts.append(tk)
        print(f"  {wt:<11}{'k=1..5 各':<10}{'':>9}"
              f"{np.mean(ts):>7.2f}{'':>8}{'':>5}{'':>10}   "
              f"[{', '.join(f'{x:+.2f}' for x in ts)}] 范围 {max(ts) - min(ts):.2f}")
        rows.append({"rule": wt, "t_phase_avg": t_avg, "t_k1": t,
                     "d_phase_avg": float(avg.mean()), "n_win": len(avg),
                     "t_by_phase": ";".join(f"{x:.2f}" for x in ts)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="三锋相位检验力分析")
    ap.add_argument("--rules", default=",".join(DEFAULT_RULES))
    ap.add_argument("--phases", default=",".join(str(p) for p in PHASES))
    ap.add_argument("--csv", default=str(OUT_CSV))
    args = ap.parse_args(argv)

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    phases = tuple(int(p) for p in args.phases.split(",") if p.strip())

    print(f"== 三锋相位检验力分析（{len(rules)} 档 × {len(phases)} 相位）==")
    t_all = time.time()
    env = load_env()
    print(f"环境加载 {time.time() - t_all:.0f}s", flush=True)
    eq, win = collect(env, rules, phases)
    _c, rho, n_eff = corr_block(eq, rules, phases)
    d = t_block(win, rules, phases)

    d["rho"] = rho
    d["n_eff"] = n_eff
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已落盘 {out}")
    print(f"\n总耗时 {time.time() - t_all:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
