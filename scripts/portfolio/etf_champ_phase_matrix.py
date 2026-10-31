"""ETF 磐石（``champ``）· 「月内调仓相位 × 权重规则」交叉矩阵（相位平均口径）。

## 为什么需要这个脚本（P1）

长期实盘选型的最后一个空白。``reports/etf_contest5/README.md`` §4.7 已经证明：

- 三锋（周频 + 80% 单只）的 k=1 单点 **30.63% / SR 1.16 / MDD 24.97%** 是 5 选 1 的
  实现值，相位平均只有 **17.17% / 0.64 / MDD 36.29% / Calmar 0.47**。
- 磐石（月频 + 防御腿 + 单只 17.6%）的 **12.97% / 1.02 / MDD 14.03%** 是**单点读数**，
  从未做相位平均。月内相位极差 3.76pp 只是 HRP 那一档的旁证，不是交叉矩阵。

⇒ 「磐石在相位平均口径下的年化 / Sharpe / MDD / Calmar」与「换权重规则是否更稳」
从未实测。本脚本补这张矩阵，也是 §4.7 里唯一标注为「推断不是实测」的那一格。

## 口径（与 ``etf_contest5_eval`` 严格同源）

- 窗口 ``BEGIN = 20200101 → 末日``，单次连续回测，**不**逐窗独立回测。
- T+1 open：``build_execution_split`` 由**信号日**构造；``execution_shift=True``。
- 🚨 **DEF 合成腿必须同时补进 close 与 open 两个面板**。只补 close 时
  ``build_execution_split`` 拿到的 fill 面板缺 DEF 列 ⇒ 防御腿拿不到执行价、权重被吞，
  年化会从 12.97% 掉到 ~9.5%（2026-10-29 查明的复现失败根因，见
  ``etf_contest5_eval.DEF_VARIANTS`` 注释）。本脚本抄的是那条正确路径。
- 相位 = **月内第 k 个交易日**（k=1..5）作为**信号日**，执行日 = 其下一个交易日。
  k=1 应逐位复现对外口径 12.97%/1.02/14.03% —— 这是本脚本的自检项。

## 判别量（长期实盘）

- ``ann_mean`` / ``sr_mean`` / ``mdd_mean``：相位平均的收益 / 风险调整 / 回撤
- ``ann_range``：相位极差 —— 集中度读数（三锋 20.9pp，磐石若 <5pp 即配置可外推）
- ``phase_luck`` = ``ann_max − ann_mean``：单点读数虚高多少
- ``calmar``：**长期实盘唯一该看的比值**，因为降杠杆不改变它、换池才改变它（§4.7.5）

## 用法

    python -m scripts.portfolio.etf_champ_phase_matrix
    python -m scripts.portfolio.etf_champ_phase_matrix --rules inverse_vol,equal,erc
    python -m scripts.portfolio.etf_champ_phase_matrix --from-csv   # 只重算汇总

产物：``reports/etf_champ/phase_weight_matrix.csv`` + 终端汇总表。
环境：**系统 Python 3.12**（``.venv`` 缺 ``AmazingData``，跑不了数据源）。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import build_execution_split
from scripts.portfolio.etf_champ_signals import PROFILES
from scripts.portfolio.etf_rotation_panels import build_defense, level_from_returns
from scripts.portfolio.etf_rotation_probe import run_variant
from scripts.portfolio.etf_weight_probe import load_env

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "reports/etf_champ/phase_weight_matrix.csv"

BEGIN = 20200101  # 与 etf_contest5_eval 一致

#: 磐石定版 ``inverse_vol`` 在前（对照基准），其余是分散化方向上的替代档。
#: 不跑 rank_decay / risk_budget / hrp_avgcorr / hrp_ew / hrp_anticorr —— 三锋上
#: 已判负 ≥15pp，且磐石 top_k=5 时它们连机制都不同（§七 #27）。
DEFAULT_RULES = ("inverse_vol", "equal", "erc", "min_var", "hrp")

#: 三锋相位平均读数（``reports/etf_contest5/phase_rule_matrix.csv``，hrp 档）
CONTEST3_PA = {"ann": 0.1717, "sharpe": 0.64, "mdd": 0.3629, "calmar": 0.47,
               "ann_range": 20.9}


def month_phase_rebalance_days(close_index: pd.DatetimeIndex) -> dict[int, set]:
    """月内第 k 个交易日（k=1..5）。与 ``phase_rebalance_days`` 的周内版同构。"""
    per = pd.Series(close_index, index=close_index).groupby(close_index.to_period("M"))
    groups = [list(g) for _, g in per]
    out: dict[int, set] = {}
    for k in range(5):
        rb: set = set()
        for v in groups:
            if len(v) > k:
                rb.add(v[k])
        out[k + 1] = rb
    return out


def build_env():
    """在 ``load_env`` 之上补 DEF 合成腿（同时补 close 与 open 两个面板）。"""
    env = load_env()
    close = env["close"]
    def_level = level_from_returns(build_defense(close)["mix"])
    extra_def = pd.DataFrame({"DEF": def_level})
    env["extra_def"] = extra_def
    env["close_def"] = close.join(extra_def)
    env["open_def"] = env["open_adj"].join(extra_def)
    return env


def run_champ(env, spec: dict, rb_days: set | None = None):
    """磐石单次回测（close 取调仓日 → T+1 open），DEF 全程在面板内。"""
    ex = env["extra_def"]
    r0 = run_variant(env["bundle"], env["panels"], spec, BEGIN, None,
                     extra_close=ex, rebalance_days=rb_days)
    spl = build_execution_split(env["open_def"], env["close_def"],
                                set(pd.DatetimeIndex(r0["rebalance_dates"])))
    r1 = run_variant(env["bundle"], env["panels"], spec, BEGIN, None,
                     extra_close=ex, rebalance_days=rb_days,
                     execution_split=spl, execution_shift=True)
    return r0, r1


def _hold_stats(w: pd.DataFrame) -> dict:
    """多头腿集中度快照：最大单只（中位）与有效只数。"""
    held = [c for c in w.columns if c not in ("CASH", "DEF")]
    live = w[(w.abs().sum(axis=1) > 1e-9)][held]
    if not len(live):
        return {"maxw": float("nan"), "neff": float("nan")}
    return {"maxw": float(live.max(axis=1).median()),
            "neff": float((1.0 / (live ** 2).sum(axis=1).replace(0, np.nan)).median())}


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    """相位平均汇总。``max_drawdown`` 是**正的深度** ⇒ 最差相位取 max。"""
    agg = d.groupby("rule").agg(
        ann_mean=("ann", "mean"), ann_min=("ann", "min"), ann_max=("ann", "max"),
        sr_mean=("sharpe", "mean"), sr_min=("sharpe", "min"),
        mdd_mean=("mdd", "mean"), mdd_worst=("mdd", "max"),
        maxw=("maxw", "median"), neff=("neff", "median"),
        turnover=("turnover", "mean"), n_rb=("n_rebalance", "median"),
    ).reset_index()
    agg["ann_range"] = (agg["ann_max"] - agg["ann_min"]) * 100
    agg["phase_luck"] = (agg["ann_max"] - agg["ann_mean"]) * 100
    agg["calmar"] = agg["ann_mean"] / agg["mdd_mean"].abs()
    agg = agg.sort_values(["calmar", "sr_mean"], ascending=False).reset_index(drop=True)
    return agg


def print_summary(agg: pd.DataFrame, base_rule: str = "inverse_vol") -> None:
    print("\n== 磐石 · 月内相位平均口径（长期实盘）==")
    print(f"  {'规则':<13}{'年化均':>8}{'年化区间':>17}{'极差':>7}{'虚高':>7}"
          f"{'SR均':>7}{'SR最差':>8}{'MDD均':>8}{'MDD最差':>9}{'Calmar':>8}"
          f"{'单只':>7}{'有效只':>7}")
    for _, x in agg.iterrows():
        print(f"  {x['rule']:<13}{x['ann_mean']:>8.2%}"
              f"{x['ann_min']:>8.2%}~{x['ann_max']:>7.2%}"
              f"{x['ann_range']:>6.1f}pp{x['phase_luck']:>+6.1f}pp"
              f"{x['sr_mean']:>7.2f}{x['sr_min']:>8.2f}"
              f"{x['mdd_mean']:>8.2%}{x['mdd_worst']:>9.2%}{x['calmar']:>8.2f}"
              f"{x['maxw']:>7.1%}{x['neff']:>7.1f}")

    b = agg[agg.rule == base_rule]
    if not len(b):
        return
    b = b.iloc[0]
    print(f"\n  定版 {base_rule}（相位平均）：年化 {b['ann_mean']:.2%} / "
          f"SR {b['sr_mean']:.2f} / MDD {b['mdd_mean']:.2%} / Calmar {b['calmar']:.2f}")
    better_c = agg[(agg.calmar > b.calmar) & (agg.rule != base_rule)]
    better_s = agg[(agg.sr_mean > b.sr_mean) & (agg.rule != base_rule)]
    print(f"  Calmar 优于定版的档位：{', '.join(better_c.rule) if len(better_c) else '无 ✅'}")
    print(f"  Sharpe 优于定版的档位：{', '.join(better_s.rule) if len(better_s) else '无 ✅'}")

    print("\n-- 与三锋（周频）相位平均对照 --")
    c = CONTEST3_PA
    print(f"  {'方案':<16}{'年化':>8}{'SR':>7}{'MDD':>9}{'Calmar':>8}{'极差':>8}")
    print(f"  {'三锋 hrp':<16}{c['ann']:>8.2%}{c['sharpe']:>7.2f}{c['mdd']:>9.2%}"
          f"{c['calmar']:>8.2f}{c['ann_range']:>7.1f}pp")
    print(f"  {'磐石 ' + base_rule:<16}{b['ann_mean']:>8.2%}{b['sr_mean']:>7.2f}"
          f"{b['mdd_mean']:>9.2%}{b['calmar']:>8.2f}{b['ann_range']:>7.1f}pp")
    print(f"  ⇒ Calmar 比 {b['calmar'] / c['calmar']:.2f}×，"
          f"回撤为三锋的 {b['mdd_mean'] / c['mdd']:.2f}×，"
          f"年化差 {(b['ann_mean'] - c['ann']) * 100:+.2f}pp")


def print_phase_detail(d: pd.DataFrame, base_rule: str = "inverse_vol") -> None:
    sub = d[d.rule == base_rule].sort_values("phase")
    if not len(sub):
        return
    print(f"\n-- {base_rule} 逐相位明细 --")
    print(f"  {'k':<4}{'年化':>9}{'SR':>7}{'MDD':>9}{'换手':>8}{'调仓次数':>10}")
    for _, x in sub.iterrows():
        print(f"  {int(x['phase']):<4}{x['ann']:>9.2%}{x['sharpe']:>7.2f}"
              f"{x['mdd']:>9.2%}{x['turnover']:>8.1%}{int(x['n_rebalance']):>10}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rules", default=",".join(DEFAULT_RULES),
                    help="逗号分隔的 weighting 档位")
    ap.add_argument("--phases", default="1,2,3,4,5")
    ap.add_argument("--csv", default=str(OUT_CSV))
    ap.add_argument("--from-csv", action="store_true",
                    help="跳过回测，直接读 --csv 重算汇总")
    args = ap.parse_args(argv)

    if args.from_csv:
        d = pd.read_csv(Path(args.csv), encoding="utf-8-sig")
        agg = summarize(d)
        print_summary(agg)
        print_phase_detail(d)
        return 0

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    phases = [int(p) for p in args.phases.split(",") if p.strip()]

    print("加载环境（全量 ETF 面板 + DEF 合成腿，约 1-2 分钟）…", flush=True)
    t0 = time.time()
    env = build_env()
    rb_map = month_phase_rebalance_days(env["close"].index)
    print(f"环境就绪 {time.time() - t0:.0f}s；{len(rules)} 档 × {len(phases)} 相位 "
          f"= {len(rules) * len(phases)} 次回测（每次含 close/open 两遍）", flush=True)

    base = dict(PROFILES["champ"])
    rows: list[dict] = []
    for wt in rules:
        spec = dict(base)
        spec["weighting"] = wt
        for k in phases:
            t1 = time.time()
            _r0, r1 = run_champ(env, spec, rb_map[k])
            hs = _hold_stats(r1["weights"])
            rows.append({
                "rule": wt, "phase": k,
                "ann": r1["annual_return"], "sharpe": r1["sharpe"],
                "mdd": r1["max_drawdown"], "turnover": r1["avg_turnover"],
                "n_rebalance": r1.get("n_rebalance", 0),
                "maxw": hs["maxw"], "neff": hs["neff"],
                "secs": round(time.time() - t1, 1),
            })
            print(f"  {wt:<13} k={k}  年化 {r1['annual_return']:>7.2%}  "
                  f"SR {r1['sharpe']:>5.2f}  MDD {r1['max_drawdown']:>7.2%}  "
                  f"单只 {hs['maxw']:>5.1%}  ({time.time() - t1:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False, encoding="utf-8-sig")

    agg = summarize(d)
    print_summary(agg)
    print_phase_detail(d)
    print(f"\n明细已落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
