"""ETF 轮动 · 「权重规则 × 调仓相位」交叉矩阵（**相位平均口径**）。

## 为什么需要这个脚本

现有产物两侧从未交叉，是本次实盘选型的直接缺口：

- ``reports/etf_contest5/weight_probe_rule.csv`` —— 13 档权重规则，但**只在 k=1 相位**
  （周一调仓）下测过；分年度、样本外配对 t 也全是 k=1。
- ``reports/etf_contest5/weight_probe_phase.csv`` —— 5 个相位，但**只在 HRP** 下测过，
  且只存 ann/sharpe/w_mean，**没有 MDD**。

⇒ 「相位平均」口径下各权重规则的排序、以及**相位平均 MDD**（长期实盘真正在意的
回撤）从未被测过。本脚本补这张矩阵。

## 相位平均口径为什么是长期实盘的正确答案

``30.63%`` 与 ``MDD 24.97%`` 都是 k=1 这一个相位的**实现值**（§4.4.7：5 相位极差
20.88pp、k=1 为最高档）。实盘无法事先选相位（§7.9 三条路径全否：WF 16.78%、
摊平 17.47%/0.70），所以长期实盘的可达期望 = **相位平均**，不是 k=1。

判别量：
- ``ann_mean``  相位平均年化 —— 唯一可外推的收益读数
- ``mdd_mean``  相位平均 MDD  —— 长期实盘真正承担的回撤（k=1 的 24.97% 是最好那档）
- ``mdd_worst`` 最差相位 MDD
- ``sr_mean``   相位平均 Sharpe
- ``ann_range`` 相位极差 —— 集中度的读数（§4.5.4：HRP 20.88 vs 等权 5.72）

## 用法

    python -m scripts.portfolio.etf_phase_rule_matrix
    python -m scripts.portfolio.etf_phase_rule_matrix --rules hrp,erc,equal
    python -m scripts.portfolio.etf_phase_rule_matrix --phases 1,2,3,4,5 --csv out.csv

产物：``reports/etf_contest5/phase_rule_matrix.csv``（明细）+ 终端汇总表。
结论回写 ``reports/etf_contest5/README.md`` §4.7（长期实盘口径）。

口径：T+1 open、PIT 修复后、同池 45 只、同成本 —— 与 ``etf_weight_probe.run`` 同源，
只改 ``weighting`` 与 ``rebalance_days`` 两个自由度。
⚠️ 相位分组用**全索引**（含 2018-2019 暖机，与 ``weight_probe_phase.csv`` 同法），
与 README §4.5.7 的「回测起点之后」分组差 ≤1.4pp，两版不可混引。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from scripts.portfolio.etf_contest_spec import FINAL
from scripts.portfolio.etf_weight_probe import ALL_RULES, load_env, run

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "reports/etf_contest5/phase_rule_matrix.csv"

#: 默认只跑有理论依据、且是有竞争力的档位（跳 rank_decay / risk_budget /
#: hrp_avgcorr / hrp_ew / hrp_anticorr —— 前四档已判负 ≥15pp，末档是安慰剂）。
DEFAULT_RULES = ("hrp", "hrp_shrunk", "min_var", "max_div",
                 "inv_var", "inverse_vol", "erc", "equal")


def phase_rebalance_days(close_index: pd.DatetimeIndex) -> dict[int, set]:
    """周内第 k 个交易日（k=1..5）。与 ``etf_weight_probe.mode_phase`` 同法。"""
    per = pd.Series(close_index, index=close_index).groupby(close_index.to_period("W"))
    groups = [list(g) for _, g in per]
    out: dict[int, set] = {}
    for k in range(5):
        rb: set = set()
        for v in groups:
            if len(v) > k:
                rb.add(v[k])
        out[k + 1] = rb
    return out


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    """相位平均汇总 + 「同收益降杠杆」对照。

    最后一列 ``delever_gap`` 是本脚本的核心判据：把该档的相位平均年化按 HRP 的
    收益/MDD 比例折算出「若只是降杠杆会得到多深回撤」，再与它实测的相位平均 MDD
    相减。**正数 = 该档比降杠杆更差**（降杠杆零成本、零新增失效模式）。
    """
    # ⚠️ `max_drawdown` 是**正的深度**（0.2497 = −24.97%）⇒ 最差相位取 max 不是 min。
    agg = d.groupby("rule").agg(
        ann_mean=("ann", "mean"), ann_min=("ann", "min"), ann_max=("ann", "max"),
        sr_mean=("sharpe", "mean"), sr_min=("sharpe", "min"),
        mdd_mean=("mdd", "mean"), mdd_worst=("mdd", "max"),
    ).reset_index()
    agg["ann_range"] = (agg["ann_max"] - agg["ann_min"]) * 100
    agg["phase_luck"] = (agg["ann_max"] - agg["ann_mean"]) * 100
    agg["calmar"] = agg["ann_mean"] / agg["mdd_mean"].abs()
    ref = agg[agg.rule == "hrp"]
    if len(ref):
        a0, m0 = float(ref.ann_mean.iloc[0]), float(ref.mdd_mean.iloc[0])
        agg["mdd_if_delever"] = agg["ann_mean"] / a0 * m0
        agg["delever_gap"] = (agg["mdd_mean"] - agg["mdd_if_delever"]) * 100
    agg = agg.sort_values(["sr_mean", "calmar"], ascending=False).reset_index(drop=True)
    return agg


def print_summary(agg: pd.DataFrame) -> None:
    print("\n== 相位平均口径（长期实盘）==")
    print(f"  {'规则':<13}{'年化均':>8}{'年化区间':>17}{'极差':>7}"
          f"{'SR均':>7}{'SR最差':>8}{'MDD均':>8}{'MDD最差':>9}{'Calmar':>8}")
    for _, x in agg.iterrows():
        print(f"  {x['rule']:<13}{x['ann_mean']:>8.2%}"
              f"{x['ann_min']:>8.2%}~{x['ann_max']:>7.2%}"
              f"{x['ann_range']:>6.1f}pp{x['sr_mean']:>7.2f}{x['sr_min']:>8.2f}"
              f"{x['mdd_mean']:>8.2%}{x['mdd_worst']:>9.2%}{x['calmar']:>8.2f}")

    if "delever_gap" in agg.columns:
        print("\n-- 「同收益降杠杆」对照（正数 = 该档比降杠杆更差）--")
        print(f"  {'规则':<13}{'年化均':>8}{'实测MDD':>9}{'降杠杆MDD':>10}{'差距':>8}")
        for _, x in agg.iterrows():
            print(f"  {x['rule']:<13}{x['ann_mean']:>8.2%}{x['mdd_mean']:>9.2%}"
                  f"{x['mdd_if_delever']:>10.2%}{x['delever_gap']:>+7.1f}pp")

    print("\n  最好相位虚高（ann_max − ann_mean）：")
    for _, x in agg.iterrows():
        print(f"    {x['rule']:<13}{x['phase_luck']:>+6.2f}pp")

    hrp = agg[agg.rule == "hrp"]
    if len(hrp):
        h = hrp.iloc[0]
        print(f"\n  对照 HRP（相位平均）：年化 {h['ann_mean']:.2%} / "
              f"SR {h['sr_mean']:.2f} / MDD {h['mdd_mean']:.2%} / "
              f"Calmar {h['calmar']:.2f}")
        better = agg[(agg.sr_mean > h.sr_mean) & (agg.rule != "hrp")]
        if len(better):
            print(f"  🚨 相位平均 Sharpe 优于 HRP 的档位：{', '.join(better.rule)}")
        else:
            print("  ✅ 相位平均 Sharpe 没有档位优于 HRP")
        better_c = agg[(agg.calmar > h.calmar) & (agg.rule != "hrp")]
        if len(better_c):
            print(f"  🚨 相位平均 Calmar 优于 HRP 的档位：{', '.join(better_c.rule)}")
        else:
            print("  ✅ 相位平均 Calmar 没有档位优于 HRP")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rules", default=",".join(DEFAULT_RULES),
                    help=f"逗号分隔；可选 = {','.join(ALL_RULES)}")
    ap.add_argument("--phases", default="1,2,3,4,5")
    ap.add_argument("--csv", default=str(OUT_CSV))
    ap.add_argument("--from-csv", action="store_true",
                    help="跳过回测，直接读 --csv 重算汇总（改汇总逻辑后用）")
    args = ap.parse_args(argv)

    if args.from_csv:
        d = pd.read_csv(args.csv, encoding="utf-8-sig")
        print_summary(summarize(d))
        return 0

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    phases = [int(p) for p in args.phases.split(",") if p.strip()]

    print("加载环境（全量 ETF 面板，约 1-2 分钟）…", flush=True)
    t0 = time.time()
    env = load_env()
    base_spec = dict(FINAL)
    rb_map = phase_rebalance_days(env["close"].index)
    print(f"环境就绪 {time.time() - t0:.0f}s；规则 {len(rules)} 档 × 相位 "
          f"{len(phases)} 个 = {len(rules) * len(phases)} 次回测", flush=True)

    rows: list[dict] = []
    for wt in rules:
        spec = dict(base_spec)
        spec["weighting"] = wt
        for k in phases:
            t1 = time.time()
            _r0, r1 = run(env, spec, rebalance_days=rb_map[k])
            rows.append({
                "rule": wt, "phase": k,
                "ann": r1["annual_return"], "sharpe": r1["sharpe"],
                "mdd": r1["max_drawdown"],
                "n_rebalance": r1.get("n_rebalance", 0),
                "secs": round(time.time() - t1, 1),
            })
            print(f"  {wt:<13} k={k}  年化 {r1['annual_return']:>7.2%}  "
                  f"SR {r1['sharpe']:>5.2f}  MDD {r1['max_drawdown']:>7.2%}  "
                  f"({time.time() - t1:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False, encoding="utf-8-sig")

    # ---- 相位汇总：长期实盘真正要看的表
    print_summary(summarize(d))

    print(f"\n明细已落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
