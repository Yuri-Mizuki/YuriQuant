"""定版信号慢腿权重 λ 的正式裁决：0.5 要不要动。

背景
----
定版信号 ``(1−λ)·rank(h1020) + λ·rank(slow)`` 的 λ=0.5 是**先验取的整点**
（写死在 ``fundamental_blend.py`` 的 ``h1020_s0.5`` 臂里），历史只在 ``ens_h1h5``
底座上扫过 0.3/0.5/0.7。2026-11-06 在正确底座（h1020）上补了九档
（``blend_weight_920.py``），得到：

    年化  16.93 / 17.67 / 16.90 / 16.82 / 16.31 / 15.82 / 14.77 / 10.96
           λ=0   0.2     0.3    0.4    0.5    0.6    0.7    1.0

两个问题没答，所以不能直接改也不能直接否：

1. **λ=0.2 是不是单点尖峰**。两侧邻点（0.0 / 0.3）都比它低 0.75pp，而网格在
   0~0.3 之间只有三个点，无法分辨「真峰」与「一个点的噪声」。
2. **动它的收益是不是只是「多冒了换手」**。λ 从 0→0.5 年化降 0.62pp、Sharpe 只涨
   0.006，但换手从 33.0% 降到 20.6%——慢腿看起来**不是 alpha，是降换手**。
   若如此，把 λ 调低等于换回换手，而固定 10bp 成本模型看不见冲击成本。

做法
----
把 0~0.6 补到 0.05 步长（补齐 0.1/0.15/0.25/0.35/0.45），对每个候选点跑**预注册的
七道门**（写在 :data:`GATES` 里，跑之前定好），并做成本抬升敏感性（费率 ×2 / ×3）
——固定 10bp 之外的冲击成本只能这样近似。

⚠️ 本裁决**不是盲测**：九档稀疏网格此前已看过，密集网格只用来分辨 0.2 是不是尖峰。
所以门槛刻意定在「收益要比定版多 1pp」这种量级，而不是「比定版高一点点就换」——
全样本选峰 + 9 年单路径，PBO≈0.7，1pp 以下的边际一律视为不可分辨。

用法：
    python -m scripts.evaluation.blend_verdict_920
    python -m scripts.evaluation.blend_verdict_920 --no-cost-sens
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402
from stats.significance import mean_t  # noqa: E402

log = setup_logging("blend_verdict_920")

#: 定版 λ 与它必须复现的读数（buffer_arm_920 --variant defv，2026-10-21 复算）
ANCHOR_LAM = 0.5
ANCHOR_ANNUAL, ANCHOR_SHARPE = 0.1631, 0.738

#: 密集网格：0.05 步长覆盖平台区，两端留 0 与 0.6 看方向
LAMS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]

#: 成本抬升敏感性：只跑有竞争力的几档
COST_LAMS = [0.2, 0.3, 0.4, 0.5]
COST_MULTS = [1.0, 2.0, 3.0]

#: 预注册门槛（跑之前写死）。全过才换。
GATES = {
    "G1 收益": "+年化 ≥ 定版 +1.00pp（全窗与 2019 起两个窗口都要）",
    "G2 风险": "Sharpe ≥ 定版（不降就是底线，动权重必须更好）",
    "G3 回撤": "max_dd ≤ 定版 +1.00pp",
    "G4 逐年": "逐年胜率 ≥ 5/9（全窗）且 ≥ 4/8（2019 起）",
    "G5 换手": "turnover ≤ 定版 ×1.25（冲击成本不进回测，只能设上限）",
    "G6 平台": "两侧最近邻（步长 0.05）年化都不低于候选 −0.30pp（排除单点尖峰）",
    "G7 可分辨": "与定版日收益配对 t ≥ 2.0（否则是同一策略在 buffer 边界摆动）",
}
G1_PP, G3_PP, G6_PP = 1.00, 1.00, 0.30
G5_MULT, G7_T = 1.25, 2.0


def _sub(dr: pd.Series, start: str | None) -> dict:
    """子区间指标（与 ``AC.res_metrics`` 同一套 ``calc_all_metrics``）。"""
    from backtest.metrics import calc_all_metrics

    s = dr.dropna()
    if start:
        s = s.loc[s.index >= pd.Timestamp(start)]
    m = calc_all_metrics(s, pd.Series(0.0, index=s.index), None)
    return {"annual": float(m.get("annual_return", 0.0)),
            "sharpe": float(m.get("sharpe", 0.0)),
            "max_dd": float(m.get("max_drawdown", 0.0))}


def _paired(cand: pd.Series, anc: pd.Series) -> dict:
    """日收益配对检验：相关系数、均值差、差收益 t。"""
    a, b = cand.align(anc, join="inner")
    d = (a - b).dropna()
    sd = float(d.std(ddof=1))
    # 单序列均值 t 走统一实现（stats.significance.mean_t），不在这里内联
    t = mean_t(d) if sd > 0 and len(d) > 2 else np.nan
    return {"n": len(d), "corr": float(a.corr(b)),
            "diff_ann": float(d.mean() * 252), "t": t,
            "tracking_error": float(sd * np.sqrt(252))}


def _gate_row(table: pd.DataFrame, daily: dict, lam: float) -> dict:
    """对一个候选 λ 逐门判定。"""
    row = table.set_index("lam")
    c, a = row.loc[lam], row.loc[ANCHOR_LAM]
    years = sorted(int(k.split("_")[1]) for k in table.columns if k.startswith("ret_"))
    ret_cols = [f"ret_{y}" for y in years]

    c_full = _sub(daily[lam], None)
    a_full = _sub(daily[ANCHOR_LAM], None)
    c_19 = _sub(daily[lam], "2019-01-01")
    a_19 = _sub(daily[ANCHOR_LAM], "2019-01-01")

    d_ann_full = (c_full["annual"] - a_full["annual"]) * 100
    d_ann_19 = (c_19["annual"] - a_19["annual"]) * 100
    wins_full = int((c[ret_cols].astype(float) > a[ret_cols].astype(float) + 1e-9).sum())
    ret_19 = [f"ret_{y}" for y in years if y >= 2019]
    wins_19 = int((c[ret_19].astype(float) > a[ret_19].astype(float) + 1e-9).sum())

    # G6 平台：网格里紧邻的两个点
    lams = sorted(float(x) for x in table["lam"])
    below = max((x for x in lams if x < lam), default=None)
    above = min((x for x in lams if x > lam), default=None)
    nb = {}
    for tag, x in (("below", below), ("above", above)):
        nb[tag] = float("nan") if x is None else (float(row.loc[x, "annual"]) - c["annual"]) * 100
    nb_vals = [v for v in nb.values() if not np.isnan(v)]
    worst_nb = min(nb_vals) if nb_vals else float("nan")

    pair = _paired(daily[lam], daily[ANCHOR_LAM])

    checks = {
        "G1 收益": (d_ann_full >= G1_PP) and (d_ann_19 >= G1_PP),
        "G2 风险": (c_full["sharpe"] - a_full["sharpe"]) >= 0,
        "G3 回撤": (a_full["max_dd"] - c_full["max_dd"]) * 100 >= -G3_PP,
        "G4 逐年": (wins_full >= 5) and (wins_19 >= 4),
        "G5 换手": c["turnover"] <= a["turnover"] * G5_MULT,
        "G6 平台": bool(nb_vals) and worst_nb >= -G6_PP,
        "G7 可分辨": pair["t"] >= G7_T,
    }
    return {
        "lam": lam,
        "annual_full": c_full["annual"], "annual_2019": c_19["annual"],
        "d_annual_full_pp": d_ann_full, "d_annual_2019_pp": d_ann_19,
        "sharpe_full": c_full["sharpe"], "d_sharpe": c_full["sharpe"] - a_full["sharpe"],
        "max_dd_full": c_full["max_dd"], "d_max_dd_pp": (c_full["max_dd"] - a_full["max_dd"]) * 100,
        "turnover": float(c["turnover"]), "turnover_ratio": float(c["turnover"] / a["turnover"]),
        "wins_full": wins_full, "wins_2019": wins_19,
        "nb_below_pp": nb["below"], "nb_above_pp": nb["above"],
        "pair_corr": pair["corr"], "pair_t": pair["t"],
        "pair_diff_ann_pp": pair["diff_ann"] * 100,
        **{k: ("✅" if v else "❌") for k, v in checks.items()},
        "passed": all(checks.values()),
        "n_pass": sum(checks.values()),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="定版慢腿权重 λ 裁决")
    ap.add_argument("--pred-dir", default=str(ROOT / "reports" / "alla_rolling_ortho920fundind"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "blend_verdict_920"))
    ap.add_argument("--no-cost-sens", action="store_true")
    ap.add_argument("--entry", type=float, default=0.15)
    ap.add_argument("--exit-frac", type=float, default=0.40)
    args = ap.parse_args(argv)

    from scripts.evaluation.blend_weight_920 import build_ctx, run_arms

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    log.info("门槛（跑前预注册）：")
    for k, v in GATES.items():
        log.info("  %-8s %s", k, v)

    ctx = build_ctx(Path(args.pred_dir))
    table, daily = run_arms(ctx, [str(x) for x in LAMS],
                            entry=args.entry, exit_frac=args.exit_frac)
    table.to_csv(dest / "lam_dense.csv", index=False, encoding="utf-8-sig")

    # 锚自检：λ=0.5 必须复现定版读数，否则整张网格不可用
    a = table[np.isclose(table["lam"], ANCHOR_LAM)].iloc[0]
    if abs(float(a["annual"]) - ANCHOR_ANNUAL) > 0.0015 or \
            abs(float(a["sharpe"]) - ANCHOR_SHARPE) > 0.006:
        raise SystemExit(f"锚不符：λ=0.5 = {a['annual']:.4f}/{a['sharpe']:.3f}，"
                         f"登记 {ANCHOR_ANNUAL:.4f}/{ANCHOR_SHARPE:.3f}")

    rows = [_gate_row(table, daily, lam)
            for lam in table["lam"] if not np.isclose(lam, ANCHOR_LAM)]
    gates = pd.DataFrame(rows)
    gates.to_csv(dest / "gate_check.csv", index=False, encoding="utf-8-sig")

    cols = ["lam", "annual_full", "annual_2019", "d_annual_full_pp", "d_annual_2019_pp",
            "sharpe_full", "d_sharpe", "turnover_ratio", "wins_full", "wins_2019",
            "pair_corr", "pair_t", "n_pass", "passed"]
    print("\n=== 密集网格逐门 ===")
    print(gates[cols].round(3).to_string(index=False))
    print()
    gate_cols = [c for c in gates.columns if c.startswith("G")]
    print(gates[["lam"] + gate_cols].to_string(index=False))

    if not args.no_cost_sens:
        sens = []
        for mult in COST_MULTS:
            t, _d = run_arms(ctx, [str(x) for x in COST_LAMS], entry=args.entry,
                             exit_frac=args.exit_frac, cost_mult=mult)
            t["cost_mult"] = mult
            sens.append(t[["cost_mult", "lam", "annual", "sharpe", "turnover"]])
            log.info("成本 ×%.1f 完成", mult)
        sens = pd.concat(sens, ignore_index=True)
        sens.to_csv(dest / "cost_sens.csv", index=False, encoding="utf-8-sig")
        print("\n=== 成本抬升敏感性（年化 %）===")
        piv = sens.pivot(index="lam", columns="cost_mult", values="annual") * 100
        print(piv.round(2).to_string())
        piv2 = sens.pivot(index="lam", columns="cost_mult", values="sharpe")
        print("\n=== Sharpe ===")
        print(piv2.round(3).to_string())

    winners = gates[gates["passed"]]
    print("\n=== 裁决 ===")
    if winners.empty:
        best = gates.sort_values("n_pass", ascending=False).iloc[0]
        print(f"无候选过全部门 → 维持 λ={ANCHOR_LAM}")
        print(f"最接近的是 λ={best['lam']:.2f}（过 {int(best['n_pass'])}/7 门，"
              f"差 {[c for c in gate_cols if best[c] == '❌']}）")
    else:
        print("候选过全部门：" + ", ".join(f"λ={x:.2f}" for x in winners["lam"]))
    print(f"\n产物 → {dest}")


if __name__ == "__main__":
    main()
