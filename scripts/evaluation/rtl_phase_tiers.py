"""三档 21 相位（RTL）补齐 —— focus / breadth 的相位风险有多大。

2026-11-13 的追问
-----------------
`reports/rtl_rebalance_timing/` 只扫了 **solo** 的 21 个相位，量出极差 **17.85pp**、
中位 22.98%、叠加 25.00%，并据此判定「solo 的登记值 35.57/36.58% 是 21 次抽签的
最大值，可期望水平约 23~25%」。

但 `yqcore` 是**三档**产品线，focus（15 只）与 breadth（605 只）的相位风险
**从未测过**。当时只能按 `RTL ≈ 0.95·σ/√N`（`VARIANTS.md` 10 臂实测，CV 1.1%）
外推：focus ≈ 11pp、breadth < 1pp。**外推不是证据** —— 本脚本把两档补齐。

为什么值得单独跑
----------------
若 focus / breadth 的可期望值基本等于登记值，那么三档的**真实排序**会从
solo 36.58 / focus 28.03 / breadth 21.81（差 16.58pp）缩到
solo ≈ 25 / focus ≈ 26 / breadth ≈ 21（差 5pp 以内）——
**这会直接改变「按容量选档」的产品叙事**，也决定对外报数该报哪个数字。

口径
----
- 档位构造走 `strategy.registry`（`get_tier(name).build()`），**不在这里复制参数**
  —— 参数真源是 `config/baselines.yaml`，复制一份就会静默漂移。
- 资金/整手走 `yq_core_report.NATIVE_CAPITAL` + 整手 100（产品线「达得到多少」口径）。
  solo 在 1e5+整手下应逐位复现 `rtl_rebalance_timing` 的 36.58%（本脚本自检）。
- 成本 `production`（27bp 往返时变）、成交 T+1 `open`、窗口 `prod`、末日截 2026-09-07。
  🚨 `rtl_rebalance_timing` 是 **36bp 旧口径**，绝对数不可与本脚本并列；
  可比的只有「相位之间的极差/标准差」这类相对量。
- `RTL` 三口径并列（见 `rtl_probe.py` 头部）：跟踪误差（文献）/ 年化横截面 std / 极差。

用法::

    python -m scripts.evaluation.rtl_phase_tiers                    # 三档 21 相位（约 8 分钟）
    python -m scripts.evaluation.rtl_phase_tiers --tiers yqcore-focus-k15
    python -m scripts.evaluation.rtl_phase_tiers --phases 0-2       # 冒烟
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.evaluation.caliber as cal
from scripts.evaluation import yq_core_report as ycr
from strategy.registry import get_tier

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "rtl_phase_tiers"
N_PHASE = 21
TIERS = ("yqcore-focus-k8", "yqcore-focus-k15", "yqcore-breadth")


def _parse_phases(txt: str) -> list[int]:
    if "-" in txt:
        a, b = txt.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in txt.split(",")]


def run_tier(name: str, phases: list[int], end: int) -> tuple[pd.DataFrame, dict]:
    """一个档位的全部相位。返回 (phase 明细, 日收益 dict[phase] -> Series)。"""
    spec = get_tier(name)
    ctx = cal.load_context(signal=spec.runnable_signal, window="prod",
                           end=end, execution="open")
    overrides = {}
    if spec.panel is not None:
        overrides[spec.panel] = ycr.panel_for(spec.panel, ctx)
    capital = ycr.NATIVE_CAPITAL[name]
    px = ycr.price_panel(ctx)
    cost = cal.cost_profile(ycr.PROFILE)

    rows, rets = [], {}
    for p in phases:
        t0 = time.time()
        strat = spec.build(**overrides)
        bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                initial_capital=float(capital), costs=cost)
        res = cal.run_once(bt, ctx, "open", freq="M", phase=p,
                           price_panel=px, lot_size=ycr.LOT)
        m = cal.metrics_row(res, capital)
        rets[p] = res.daily_returns
        row = {"tier": name, "phase": p,
               "n_reb": len(cal.rebalance_dates(ctx.pred.index, freq="M",
                                               phase=p)),
               "ann": m["ann"], "sr_arith": m["sharpe_arith"],
               "max_dd": m["max_dd"], "turnover": m["turnover"],
               "cost_bp": m["cost_bp"], "n_hold": m["n_active"]}
        # 逐年：判「phase 0 的优势是结构性还是某一年的运气」——只看 21 相位
        # 均值会被单年主导，必须拆年看它是否 7 年都在前 1/3。
        dr = res.daily_returns
        for y, sub in dr.groupby(dr.index.year):
            row[f"y{y}"] = float((1 + sub).prod() - 1)
        rows.append(row)
        print(f"  [{name}] phase={p:<3} ann={m['ann'] * 100:7.2f}% "
              f"SR={m['sharpe_arith']:6.3f} 持仓={m['n_active']:5.1f} "
              f"换手={m['turnover']:5.3f} ({time.time() - t0:.0f}s)", flush=True)

    meta = {"tier": name, "capital": capital, "signal": spec.runnable_signal,
            "fingerprint": ctx.fingerprint, "n_days": ctx.n_days,
            "window": f"{ctx.pred.index.min().date()}..{ctx.pred.index.max().date()}"}
    return pd.DataFrame(rows), {"meta": meta, "rets": rets, "ctx": ctx}


def _stats(s: pd.Series) -> dict:
    return {"mean": float(s.mean()), "median": float(s.median()),
            "std": float(s.std(ddof=1)), "range": float(s.max() - s.min()),
            "min": float(s.min()), "max": float(s.max()),
            "argmax": int(s.idxmax()) if s.index.dtype.kind in "iu" else -1}


def summarize(df: pd.DataFrame, rets: dict[int, pd.Series]) -> dict:
    """一个档位的 RTL 三口径 + 叠加（tranching）+ 选择偏差校正。"""
    s = df.set_index("phase")["ann"]
    st = _stats(s)

    R = pd.DataFrame({p: series for p, series in rets.items()}).dropna(how="all")
    blend = R.mean(axis=1)
    n_y = len(blend) / 252.0
    blend_ann = float((1 + blend).cumprod().iloc[-1] ** (1 / n_y) - 1)
    blend_sr = float(blend.mean() / blend.std(ddof=1) * np.sqrt(252))
    eq = (1 + blend).cumprod()
    blend_dd = float((eq / eq.cummax() - 1).min())

    # 文献口径 RTL：各相位相对叠加组合的**月度跟踪误差**年化
    tes = []
    for p in R.columns:
        mon = (1 + (R[p] - blend)).resample("ME").prod() - 1
        tes.append(float(mon.std(ddof=1) * np.sqrt(12)))
    te = pd.Series(tes)

    # 选择偏差：21 个相位取最大值的期望（正态近似）
    rng = np.random.default_rng(20261113)
    e_max = float(rng.standard_normal((200_000, len(s))).max(axis=1).mean())

    return {**st,
            "phase0": float(s.loc[0]) if 0 in s.index else np.nan,
            "blend_ann": blend_ann, "blend_sr": blend_sr, "blend_dd": blend_dd,
            "rtl_te_mean": float(te.mean()), "rtl_te_median": float(te.median()),
            "e_max_z": e_max, "e_max_ann": st["mean"] + st["std"] * e_max,
            "n_hold": float(df["n_hold"].mean()),
            "turnover": float(df["turnover"].mean())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--phases", default=f"0-{N_PHASE - 1}")
    ap.add_argument("--end", type=int, default=ycr.END_ALIGN)
    a = ap.parse_args(argv)
    tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]
    phases = _parse_phases(a.phases)

    OUT.mkdir(parents=True, exist_ok=True)
    allrows, summary = [], []
    for name in tiers:
        print(f"\n=== {name} ===", flush=True)
        df, pack = run_tier(name, phases, a.end)
        df.to_csv(OUT / f"phase21_{name.replace('yqcore-', '')}.csv",
                  index=False, encoding="utf-8-sig")
        sm = summarize(df, pack["rets"])
        sm.update(pack["meta"])
        summary.append(sm)
        allrows.append(df)

    df_all = pd.concat(allrows, ignore_index=True)
    df_all.to_csv(OUT / "phase_all.csv", index=False, encoding="utf-8-sig")
    sm_all = pd.DataFrame(summary).set_index("tier")
    sm_all.to_csv(OUT / "summary_tiers.csv", encoding="utf-8-sig")

    print("\n=== 汇总（production 27bp / T+1 open / native 资金 + 整手）===")
    cols = ["n_hold", "phase0", "mean", "median", "std", "range",
            "blend_ann", "e_max_ann", "rtl_te_mean"]
    view = sm_all[cols].copy()
    for c in cols:
        if c != "n_hold":
            view[c] = (view[c] * 100).round(2)
    print(view.round(1).to_string())

    # √N 外推检验：RTL(std) / (1/√N) 应为常数
    if len(sm_all) >= 2 and N_PHASE > 4:
        ratio = sm_all["std"] * np.sqrt(sm_all["n_hold"])
        print("\n=== √N 检验：std·√N（同口径下应为常数）===")
        print(ratio.round(5).to_string())
        print(f"  CV = {ratio.std(ddof=1) / ratio.mean() * 100:.1f}%")

    print(f"\n产物 → {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
