"""
GP 评估预算 × 多 seed 调参实验（三段式，test 段受保护）
=========================================================

在 PIT 并集池 + 三段式框架下回答两个问题：
1. 固定评估预算（≈pop×(1+gen)）时，pop/gen 怎么分配挖出的因子最好？
2. GP 随机性强不强（多 seed 下 hof 稳定性）？

划分（test 段严格受保护）：
    train 2022-2023（PIT 并集池）→ GP 挖掘（内部再留 20% 防过拟合）
    valid 2024        → 对 hof 公式重算 IC 并筛选（**调参只看这里**）
    test  2025        → 仅对 valid 最优配置做最终验证一次，绝不用于选参

用法:
    python scripts/gp_tune_budget.py --grid "100x20,200x10,400x5,1000x3" --seeds 0,1,2
    python scripts/gp_tune_budget.py --grid "200x10,1000x3" --seeds 0,1 --jobs 4

输出: reports/gp_tune/overview.csv（全部配置×seed）+ chosen_test.csv（test 最终验证）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cli_common import setup_logging  # noqa: E402


log = setup_logging("gp_tune_budget")

OUT_DIR = Path("reports") / "gp_tune"

def split_dates(index, train=(20220101, 20231231), valid=(20240101, 20241231),
                test=(20250101, 20251231)):
    def rng(b, e):
        return index[(index >= pd.Timestamp(str(b))) & (index <= pd.Timestamp(str(e)))]
    return rng(*train), rng(*valid), rng(*test)

def load_pit_core_panels(begin: int = 20220101, end: int = 20251231):
    """轻量加载：PIT 并集池 + 核心 OHLCV（调参专用，跳过技术/日内/基本面）。

    与 walk_forward.load_full_panels 同口径（PIT 并集池 + membership mask +
    后复权），但只构建 GP 需要的核心特征，避免分钟线特征工程的分钟级耗时。
    """
    from data.cache import DataCache
    from data.offline import OfflineDataSource
    from data.universe import Universe
    from data.cache_helpers import _pit_universe_codes, _apply_membership_mask
    cache = DataCache(OfflineDataSource())
    uni = Universe(cache)
    codes = _pit_universe_codes(uni, "000300.SH", begin, end)
    df = pd.read_parquet(Path(str(cache.root)) / "daily_hs300.parquet").reset_index()
    df["date"] = df["date"].dt.normalize()
    df = df[(df["date"] >= pd.Timestamp(str(begin))) & (df["date"] <= pd.Timestamp(str(end)))]
    df = df[df["code"].isin(codes)]
    df = _apply_membership_mask(df.set_index(["date", "code"]), uni, "000300.SH").reset_index()

    def piv(col):
        return df.pivot(index="date", columns="code", values=col).sort_index()

    panels = {k: piv(k) for k in ("close", "open", "high", "low", "volume", "amount")}
    bf = pd.read_parquet(Path(str(cache.root)) / "backward_factor.parquet")
    bf = bf.reindex(index=panels["close"].index).reindex(columns=panels["close"].columns).ffill()
    for k in ("close", "open", "high", "low"):
        panels[k] = panels[k] * bf
    panels["returns"] = panels["close"].pct_change()
    panels["vwap"] = panels["amount"] / panels["volume"]
    returns_panel = panels["close"].pct_change().shift(-1)
    return panels, returns_panel

def main():
    ap = argparse.ArgumentParser(description="GP 评估预算 × 多 seed 调参")
    ap.add_argument("--grid", default="100x20,200x10,400x5,1000x3",
                    help="pop×gen 组合，逗号分隔")
    ap.add_argument("--seeds", default="0,1,2", help="随机种子列表")
    ap.add_argument("--jobs", type=int, default=4, help="GP 并行进程数")
    ap.add_argument("--top", type=int, default=15, help="valid 段筛选 top N")
    ap.add_argument("--cand", type=int, default=30, help="hof 前 N 个进 valid 复检")
    ap.add_argument("--train-frac", type=float, default=0.8, help="GP 内部样本外切分")
    ap.add_argument("--windows", default="5,10,20,60")
    ap.add_argument("--k", type=int, default=50, help="test 回测 TopK")
    ap.add_argument("--freq", default="M", choices=["D", "W", "M"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = [tuple(int(x) for x in g.split("x")) for g in args.grid.split(",") if "x" in g]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    log.info("预算网格: %s × seeds %s", grid, seeds)

    # ---- 数据（PIT 并集池 + 核心特征，轻量加载）----
    from scripts.gp_tune_budget import load_pit_core_panels
    panels, returns_panel = load_pit_core_panels(20220101, 20251231)
    features = ["close", "open", "high", "low", "volume", "amount", "returns", "vwap"]
    windows = tuple(int(x) for x in args.windows.split(",") if x.strip())
    log.info("全区间 %d 日 × %d 股（PIT 并集池）", len(returns_panel), len(returns_panel.columns))

    train_dates, valid_dates, test_dates = split_dates(returns_panel.index)
    log.info("三段: train %d 日 / valid %d 日 / test %d 日",
             len(train_dates), len(valid_dates), len(test_dates))
    train_returns = returns_panel.loc[train_dates]
    valid_returns = returns_panel.loc[valid_dates]
    test_returns = returns_panel.loc[test_dates]
    train_panel = {k: p.loc[train_dates] for k, p in panels.items()}

    from factor.formula import formula_builder
    from factor.genetic_mining import run_gp_mining
    from research.factor_analysis import calc_ic_series, calc_ir

    rows = []
    best_cfg = None          # (pop, gen, seed)
    best_valid_ir = -np.inf
    for pop, gen in grid:
        for seed in seeds:
            log.info("==> GP pop=%d gen=%d seed=%d（预算 %d 次评估）", pop, gen, seed,
                     pop * (1 + gen))
            gp_df, _ = run_gp_mining(
                train_panel, train_returns, features=features, windows=windows,
                population=pop, generations=gen, min_depth=2, max_depth=5,
                tournament=5, train_frac=args.train_frac, monthly_weight=0.5,
                fitness_mode="tstat", n_jobs=args.jobs, seed=seed, verbose=False,
            )
            if gp_df is None or gp_df.empty:
                log.warning("pop=%d gen=%d seed=%d 无有效 hof", pop, gen, seed)
                continue
            # valid 段复检：hof 前 N 个公式重算 IC
            sel = []
            for _, r in gp_df.head(args.cand).iterrows():
                f = r["formula"]
                try:
                    fp = formula_builder(f, features=features)(panels)
                except Exception:
                    continue
                ic = calc_ic_series(fp.loc[valid_dates], valid_returns).dropna()
                if len(ic) < 20:
                    continue
                m, s = float(ic.mean()), float(ic.std())
                t = m / (s / np.sqrt(len(ic))) if s > 0 else 0.0
                sel.append({"formula": f, "valid_ic": m, "valid_ir": m / s if s > 0 else 0.0,
                            "valid_t": t, "train_t": r.get("t_stat", np.nan)})
            if not sel:
                log.warning("pop=%d gen=%d seed=%d valid 段无有效因子", pop, gen, seed)
                continue
            sdf = pd.DataFrame(sel).sort_values("valid_t", key=abs, ascending=False)
            top = sdf.head(args.top)
            valid_ic = float(top["valid_ic"].abs().mean())
            valid_ir = float(top["valid_ir"].mean())
            valid_t = float(top["valid_t"].abs().mean())
            rows.append({
                "pop": pop, "gen": gen, "budget": pop * (1 + gen), "seed": seed,
                "n_hof": len(gp_df), "n_valid_ok": len(sdf),
                "valid_ic_mean": valid_ic, "valid_ir_mean": valid_ir,
                "valid_t_mean": valid_t, "best_formula": sdf.iloc[0]["formula"],
            })
            log.info("    valid: %d 因子通过, 平均|IC|=%.4f 平均IR=%.2f",
                     len(sdf), valid_ic, valid_ir)
            if valid_ir > best_valid_ir:
                best_valid_ir = valid_ir
                best_cfg = (pop, gen, seed)
                best_sdf = sdf

    overview = pd.DataFrame(rows)
    overview.to_csv(OUT_DIR / "overview.csv", index=False)
    print("\n===== 评估预算 × 多 seed 调参（valid 段指标）=====")
    with pd.option_context("display.width", 220, "display.float_format", lambda v: f"{v:.4f}"):
        print(overview.sort_values("valid_ir_mean", ascending=False).to_string(index=False))
    print("\n说明: 调参只看 valid 段；test 段仅对最优配置最终验证一次")

    # ---- test 段最终验证（仅一次，受保护）----
    if best_cfg is None:
        log.error("无有效配置")
        sys.exit(1)
    pop, gen, seed = best_cfg
    log.info("valid 最优配置: pop=%d gen=%d seed=%d（valid IR=%.2f）—— test 段最终验证",
             pop, gen, seed, best_valid_ir)
    chosen = best_sdf.copy()
    chosen["pop"], chosen["gen"], chosen["seed"] = pop, gen, seed

    from factor.synthesis import CompositeInput, synthesize_ic_weighted
    comps = []
    for _, r in chosen.iterrows():
        fp = formula_builder(r["formula"], features=features)(panels)
        comps.append(CompositeInput(name=r["formula"], panel=fp,
                                    ic=r["valid_ic"], ir=r["valid_ir"]))
    comp_all = synthesize_ic_weighted(comps, weight_by="ic_abs")

    test_rows = []
    for label, dts, rts in [("train", train_dates, train_returns),
                            ("valid", valid_dates, valid_returns),
                            ("test", test_dates, test_returns)]:
        ic = calc_ic_series(comp_all.loc[dts], rts).dropna()
        test_rows.append({
            "阶段": label, "IC": float(ic.mean()),
            "IR": calc_ir(ic),
            "t": float(ic.mean() / (ic.std() / np.sqrt(len(ic)))) if ic.std() > 0 else 0.0,
            "n_days": len(ic),
        })
    try:
        from backtest import VectorBacktest
        from strategy.examples import TopKLongOnly
        bt = VectorBacktest(TopKLongOnly(k=args.k), rebalance_freq=args.freq)
        res = bt.run(comp_all.loc[test_dates], test_returns)
        m = res.metrics()
        test_rows.append({"阶段": f"test回测Top{args.k}{args.freq}", "IC": np.nan,
                          "IR": np.nan, "t": np.nan, "n_days": args.k})
        res.equity_curve.to_csv(OUT_DIR / "equity_chosen_test.csv")
        bt_note = (f"sharpe={m.get('sharpe', 0):.2f} 年化={m.get('annual_return', 0) * 100:+.1f}% "
                   f"回撤={m.get('max_drawdown', 0) * 100:.1f}%")
    except Exception as e:
        bt_note = f"回测跳过: {e}"
    summary = pd.DataFrame(test_rows)
    summary.to_csv(OUT_DIR / "chosen_test.csv", index=False)
    chosen.to_csv(OUT_DIR / "chosen_factors.csv", index=False)
    print(f"\n===== 最优配置（pop={pop} gen={gen} seed={seed}）三段验证 =====")
    with pd.option_context("display.width", 120, "display.float_format", lambda v: f"{v:.4f}"):
        print(summary.to_string(index=False))
    print(bt_note)
    log.info("结果: %s", OUT_DIR)

if __name__ == "__main__":
    main()