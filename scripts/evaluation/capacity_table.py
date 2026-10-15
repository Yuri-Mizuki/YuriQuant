"""组合层容量表：集中度 × AUM（统一口径，2026-10-13）。

在**唯一真源口径**（:mod:`scripts.evaluation.caliber`）上量化生产执行层的容量边界：

- **横轴 = 组合形态（集中度）**：
  ``TopFracLongOnly(0.10)``（旧研究口径，对照）→ 依次加宽的 buffer 带。
- **纵轴 = AUM**：按逐股冲击 ``η·σ·√(Q/ADV)`` 的规模敏感项扫描。
- **指标**：净年化 / Sharpe / 换手 / 总成本 bp / **冲击成本 bp 单列** / 活跃只数。

回答两个问题：
1. **该策略能承载多大资金**（净年化跌破基准时的 AUM = 容量拐点）；
2. **集中度如何影响容量** —— 集中（少持仓）在低 AUM 下 alpha 更纯，但冲击随持仓
   权重上升而更快吃掉收益；分散（多持仓）单股权重小、冲击低，但 alpha 被摊薄。

⚠️ 本脚本刻意**不再自造基线**：比例全部取 ``caliber.SIGNALS`` / ``caliber.WINDOWS``。
历史教训见 ``reports/docs/design/定版主模型与统一口径.md`` §二。

用法::

    D:/python/Python312/python.exe scripts/evaluation/capacity_table.py \\
        --window prod --capitals 1e7,3e7,1e8,3e8,1e9 --out-tag prod
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import caliber  # noqa: E402
from scripts.evaluation.caliber import (  # noqa: E402
    cost_profile,
    evaluate_strategy,
    load_context,
)
from strategy.examples import BufferedTopFracLongOnly, TopFracLongOnly  # noqa: E402

#: 组合形态（横轴）。键 = 展示名，值 = 策略实例。生产定版 = buf0.15_0.40。
DEFAULT_ARMS: dict[str, dict] = {
    "top10pct_nobuf": {"kind": "frac", "frac": 0.10,
                       "desc": "Top10% 等权（旧研究口径，无缓冲带）"},
    "buf0.05_0.20": {"kind": "buf", "entry": 0.05, "exit": 0.20,
                     "desc": "集中：约 Top5%~20% 缓冲"},
    "buf0.10_0.30": {"kind": "buf", "entry": 0.10, "exit": 0.30,
                     "desc": "中：约 Top10%~30% 缓冲"},
    "buf0.15_0.40": {"kind": "buf", "entry": 0.15, "exit": 0.40,
                     "desc": "**生产定版** buffer(15/40)"},
    "buf0.30_0.60": {"kind": "buf", "entry": 0.30, "exit": 0.60,
                     "desc": "分散：约 Top30%~60% 缓冲"},
}


def _make(spec: dict):
    if spec["kind"] == "frac":
        return TopFracLongOnly(frac=spec["frac"])
    return BufferedTopFracLongOnly(spec["entry"], spec["exit"])


def build_table(ctx, capitals, arms, cost, *, include_flat_ref: bool = True) -> pd.DataFrame:
    rows = []
    flat = cost_profile("flat")
    for cap in capitals:
        for name, spec in arms.items():
            m = evaluate_strategy(_make(spec), ctx, cost, cap)
            row = {"arm": name, "capital": float(cap), "desc": spec["desc"], **m}
            if include_flat_ref:
                # 固定 10bp 参照（与 AUM 无关）—— 用于读「冲击吃掉了多少」
                mf = evaluate_strategy(_make(spec), ctx, flat, cap)
                row["ann_flat10bp"] = mf["ann"]
                row["sharpe_flat10bp"] = mf["sharpe"]
                row["cost_bp_flat10bp"] = mf["cost_bp"]
            rows.append(row)
            print("  %-16s cap=%-9.0f 年化 %6.2f%% (flat %6.2f%%)  S %.3f  成本 %5.0fbp"
                  " (冲击 %5.0fbp)  换手 %.3f  活跃 %.0f只"
                  % (name, cap, m["ann"] * 100, row.get("ann_flat10bp", float("nan")) * 100,
                     m["sharpe"], m["cost_bp"], m["impact_bp"], m["turnover"], m["n_active"]))
    return pd.DataFrame(rows)


def _capacity_threshold(df: pd.DataFrame, col: str, thr: float) -> pd.DataFrame:
    """每个 arm 的「容量拐点」= 净年化首次跌破 ``thr`` 的 AUM（不跌则记 inf）。"""
    out = []
    for arm, g in df.groupby("arm", sort=False):
        g = g.sort_values("capital")
        hit = g[g[col] < thr]
        out.append({"arm": arm, "n_active_1e8": float(
            g.loc[g["capital"].idxmin(), "n_active"]),
            "capacity_at_thr": float(hit["capital"].iloc[0]) if len(hit) else float("inf"),
            "threshold": thr})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signal", default="ens_h1h5", choices=sorted(caliber.SIGNALS))
    ap.add_argument("--window", default="prod", choices=sorted(caliber.WINDOWS))
    ap.add_argument("--capitals", default="1e7,3e7,1e8,3e8,1e9")
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--buy-mult", type=float, default=None)
    ap.add_argument("--no-flat-ref", action="store_true")
    ap.add_argument("--thr", type=float, default=0.12,
                    help="容量拐点判据：净年化跌破该值（默认 0.12 = 12%）")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    capitals = [float(x) for x in args.capitals.split(",") if x.strip()]
    ctx = load_context(signal=args.signal, window=args.window)
    cost = cost_profile("impact", eta=args.eta, buy_mult=args.buy_mult)
    print(caliber.describe(ctx, "impact"))

    df = build_table(ctx, capitals, DEFAULT_ARMS, cost,
                     include_flat_ref=not args.no_flat_ref)
    out = ROOT / "reports" / "capacity_table"
    out.mkdir(parents=True, exist_ok=True)
    suf = f"_{args.out_tag}" if args.out_tag else ""
    df.to_csv(out / f"capacity{suf}.csv", index=False)

    cap = _capacity_threshold(df, "ann", args.thr)
    cap.to_csv(out / f"capacity_thr{suf}.csv", index=False)
    print(f"\n=== 容量拐点（净年化 < {args.thr:.0%} 的首个 AUM）===")
    print(cap.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # markdown 透视：行 = AUM，列 = 形态，值 = 净年化
    piv = df.pivot_table(index="capital", columns="arm", values="ann")
    print("\n=== 净年化透视（行=AUM，列=形态）===")
    print((piv * 100).round(2).to_string())
    print(f"\n产物: {out / f'capacity{suf}.csv'} / {out / f'capacity_thr{suf}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
