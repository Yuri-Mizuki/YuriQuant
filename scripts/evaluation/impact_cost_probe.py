"""成本摩擦探针：固定滑点 vs 逐股冲击 vs 整手约束（P0-1 / P1，2026-10-13）。

在**真实回测口径**（`reports/alla_rolling/_base` + 定版 pred）上，量化三件事：

1. **固定滑点 vs 逐股冲击**：把 `config` 里的 10bp 假设，换成逐股
   ``η·σ·√(Q/ADV)``，看年化差多少、以及**随资金规模**如何变化（冲击是 AUM 的
   函数，固定滑点不是）。
2. **整手约束**：回测权重层按 100 股取整，量化集中组合的收益虚高幅度。
3. 三者都跑同一 `TopFracLongOnly(0.10)` 月频等权形态，可直接对照。

用法::

    D:/python/Python312/python.exe scripts/evaluation/impact_cost_probe.py \\
        --capital 1e8 --begin 20180101 --end 20260901

口径与边界见
``reports/docs/design/链路优化调查_预测组合执行_2026-10-13.md`` §9.5：
η/买卖非对称是研报经验值；本项目无真实成交数据 ⇒ 产出的是**估计成本**。

⚠️ **2026-10-13 起，全项目统一口径见 :mod:`scripts.evaluation.caliber` +
``config/baselines.yaml``（设计文档 ``reports/docs/design/定版主模型与统一口径.md``）。**
本脚本仍自持 ``BASE_DIR`` / ``_load_daily_amount`` / 自算 Sharpe ⇒ **其数字在按
``caliber`` 迁移并校验之前，不得与定版登记表并列**（历史上正是这类「自家口径」
造成 5 处不同源 ⇒ 结论三次反转）。同类分析请直接用 ``caliber`` 或
``scripts/evaluation/capacity_table.py``。
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

from backtest.costs import ImpactCostModel, TransactionCosts  # noqa: E402
from backtest.engine import VectorBacktest  # noqa: E402
from strategy.examples import TopFracLongOnly  # noqa: E402

BASE_DIR = ROOT / "reports" / "alla_rolling" / "_base"
PRED_DEFAULT = ROOT / "reports" / "alla_rolling_ortho" / "pred" / "ens_h1h5__h1.parquet"
# 覆盖最全的 daily_all_a（E: 2015 起；D: 仅 2022 起）→ 保证 2022 前 ADV 不缺
# 而被静默低估为 0（2026-10-13 核对：两份 amount 逐位一致 ratio=1.0）。
DAILY_CANDIDATES = (Path("e:/data/parquet/daily_all_a.parquet"),
                    Path("d:/data/parquet/daily_all_a.parquet"))


def _load_daily_amount() -> pd.DataFrame:
    best, best_n = None, -1
    for p in DAILY_CANDIDATES:
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        lv = d.index.levels[0] if isinstance(d.index, pd.MultiIndex) else d.index
        if len(lv) > best_n:
            best, best_n = d, len(lv)
    if best is None:
        raise FileNotFoundError(f"无可用 daily_all_a：{DAILY_CANDIDATES}")
    out = best["amount"].unstack("code").astype(float)
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out


def ann(dr: pd.Series) -> float:
    dr = dr.dropna()
    if dr.empty:
        return float("nan")
    return float((1.0 + dr).prod() ** (252.0 / len(dr)) - 1.0)


def load_inputs(begin: int, end: int):
    close = pd.read_parquet(BASE_DIR / "close_adj.parquet")
    close_raw = pd.read_parquet(BASE_DIR / "close_raw.parquet")
    mask = pd.read_parquet(BASE_DIR / "tradable_mask.parquet")
    pred = pd.read_parquet(PRED_DEFAULT)

    amt = _load_daily_amount()

    idx = pred.index
    cols = pred.columns
    lo, hi = pd.Timestamp(str(begin)), pd.Timestamp(str(end))
    keep = idx[(idx >= lo) & (idx <= hi)]
    close = close.reindex(index=keep, columns=cols)
    close_raw = close_raw.reindex(index=keep, columns=cols)
    mask = mask.reindex(index=keep, columns=cols).fillna(True)
    amt = amt.reindex(index=keep, columns=cols).astype(float)
    return close, close_raw, mask, pred.reindex(index=keep, columns=cols), amt


def run_arm(sig, fwd, mask, *, costs, capital, frac, price=None, lot=None,
            adv=None, vol=None) -> dict:
    bt = VectorBacktest(strategy=TopFracLongOnly(frac=frac),
                        rebalance_freq="M", initial_capital=float(capital),
                        costs=costs)
    res = bt.run(sig, fwd, executable_mask=mask,
                 price_panel=price, lot_size=lot,
                 adv_panel=adv, vol_panel=vol)
    dr = res.daily_returns
    return {
        "ann": ann(dr),
        "sharpe": float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else float("nan"),
        "turnover": float(res.turnover_series.dropna().mean()),
        "cost_bp": float((res.cost_series.sum() / float(capital)) * 1e4),
        "impact_bp": float((res.impact_cost_series.sum() / float(capital)) * 1e4),
        "exposure": float(res.weights_history.abs().sum(axis=1).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--begin", type=int, default=20180101)
    ap.add_argument("--end", type=int, default=20260901)
    ap.add_argument("--capital", type=float, default=1e8,
                    help="组合资金规模（元）——冲击是它的函数")
    ap.add_argument("--capitals", default=None,
                    help="多档资金规模（逗号分隔），用于画 AUM-冲击曲线")
    ap.add_argument("--eta", type=float, default=0.4)
    ap.add_argument("--buy-mult", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=0.10,
                    help="Top-Frac 多头比例（生产形态 0.10≈全池 Top10% 约 400 只；"
                         "0.01≈45 只的集中形态，用于检验整手敏感性）")
    ap.add_argument("--out-tag", default="",
                    help="产物后缀，避免不同 frac 场景互相覆盖（如 prod / conc）")
    args = ap.parse_args()

    close, close_raw, mask, pred, amt = load_inputs(args.begin, args.end)
    fwd = close.pct_change(fill_method=None)          # h=1 未 shift（引擎口径）
    vol = fwd.rolling(60, min_periods=20).std()       # 日波动率（无前视）
    sig = pred                                        # 掩码交给引擎（主实验调用形态）

    fixed = TransactionCosts(commission_rate=0.0003, stamp_duty=0.001, slippage_bp=10)
    impact_costs = TransactionCosts(commission_rate=0.0003, stamp_duty=0.001,
                                    slippage_bp=0.0,
                                    impact=ImpactCostModel(eta=args.eta,
                                                           buy_multiplier=args.buy_mult))

    rows = []
    print(f"\n=== 基线（固定 10bp 滑点），资金 {args.capital:,.0f}，frac={args.frac} ===")
    base = run_arm(sig, fwd, mask, costs=fixed, capital=args.capital, frac=args.frac)
    rows.append(("固定10bp", args.capital, base))
    print(f"  年化 {base['ann']:.2%}  Sharpe {base['sharpe']:.3f}  "
          f"换手 {base['turnover']:.3f}  成本 {base['cost_bp']:.1f}bp  敞口 {base['exposure']:.3f}")

    capitals = ([float(x) for x in args.capitals.split(",")]
                if args.capitals else [args.capital])
    print("\n=== 逐股冲击（η=%.2f, 买入×%.1f）===" % (args.eta, args.buy_mult))
    for cap in capitals:
        r = run_arm(sig, fwd, mask, costs=impact_costs, capital=cap, frac=args.frac,
                    adv=amt, vol=vol)
        rows.append(("冲击", cap, r))
        print(f"  资金 {cap:>14,.0f}  年化 {r['ann']:.2%}  成本 {r['cost_bp']:.1f}bp"
              f"（其中冲击 {r['impact_bp']:.1f}bp）  换手 {r['turnover']:.3f}")

    print(f"\n=== 整手约束（lot=100, 资金 {args.capital:,.0f}）===")
    lot_r = run_arm(sig, fwd, mask, costs=impact_costs, capital=args.capital,
                    frac=args.frac, price=close_raw, lot=100, adv=amt, vol=vol)
    rows.append(("整手", args.capital, lot_r))
    print(f"  年化 {lot_r['ann']:.2%}  敞口 {lot_r['exposure']:.3f}"
          f"（欠配 {1 - lot_r['exposure']:.3f}）  换手 {lot_r['turnover']:.3f}")

    out = ROOT / "reports" / "impact_cost_probe"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"arm": a, "capital": c, "frac": args.frac, **m}
                       for a, c, m in rows])
    suf = f"_{args.out_tag}" if args.out_tag else ""
    df.to_csv(out / f"summary{suf}.csv", index=False)
    print(f"\n产物: {out / f'summary{suf}.csv'}")


if __name__ == "__main__":
    main()
