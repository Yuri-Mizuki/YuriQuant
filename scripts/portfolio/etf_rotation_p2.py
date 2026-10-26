"""
ETF 轮动优化验证（P2 前置，2026-10-26）
======================================

在 P1 基线 ``dual_ivvt15_m5`` 之上验证三项「低成本、可借自业界」的增量：

  1. **防御腿**：风险关闭时不再留 0 收益现金，而是退到国债 / 黄金 ETF；
  2. **趋势一致性**：以 60 日对数价格回归 R² 缩放 12-1 动量（借自聚宽「斜率×R²」）；
  3. **拥挤度**：份额历史分位（PIT，按 ``ANN_DATE`` 生效）过滤高拥挤标的
     （借自银河金工动量择势 / 国盛景气-趋势-拥挤）。

判据：相对 P1 基线 ΔSharpe > 0.1、MDD 不恶化、成本后为正（方案 §四）。

用法:
    python -m scripts.portfolio.etf_rotation_p2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from data.cache import DataCache
from data.datasource import create_datasource
from scripts.portfolio.etf_rotation_probe import (
    BENCH,
    MIN_AMOUNT,
    SEGMENTS,
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "etf_rotation_p2"
CACHE_ROOT = Path(str(Config.cache()["root"]))

BOND, GOLD = "511010.SH", "518880.SH"
P1_BASE = {"signal": "mom12_1", "freq": "M", "top_k": 5, "absolute": True,
           "weighting": "inverse_vol", "target_vol": 0.15}

VARIANTS: list[dict] = [
    {**P1_BASE, "name": "p1_base"},
    {**P1_BASE, "name": "def_bond", "defense": "bond"},
    {**P1_BASE, "name": "def_gold", "defense": "gold"},
    {**P1_BASE, "name": "def_mix", "defense": "mix"},
    {**P1_BASE, "name": "mom_r2", "signal": "mom_r2"},
    {**P1_BASE, "name": "crowd_f20", "signal": "mom_crowd_f20"},
    {**P1_BASE, "name": "crowd_f10", "signal": "mom_crowd_f10"},
    {**P1_BASE, "name": "def_mix_crowd_f20", "signal": "mom_crowd_f20", "defense": "mix"},
]

# 调仓频率对照：月频（基准）/ 每 10 个交易日 / 周频。
FREQ: list[dict] = [
    {**P1_BASE, "name": "champ_m", "signal": "mom_crowd_f20", "defense": "mix"},
    {**P1_BASE, "name": "champ_2w", "signal": "mom_crowd_f20", "defense": "mix", "every_n": 10},
    {**P1_BASE, "name": "champ_w", "signal": "mom_crowd_f20", "defense": "mix", "freq": "W"},
]

# 成本对照：现行假设（佣金 5bp）vs 用户实际费率（万分之一 = 1bp）× 滑点档位。
COST: list[dict] = [
    {**P1_BASE, "name": "cost_c5_s5", "signal": "mom_crowd_f20", "defense": "mix"},
    {**P1_BASE, "name": "cost_c1_s5", "signal": "mom_crowd_f20", "defense": "mix",
     "commission_bp": 1.0},
    {**P1_BASE, "name": "cost_c1_s2", "signal": "mom_crowd_f20", "defense": "mix",
     "commission_bp": 1.0, "slippage_bp": 2.0},
    {**P1_BASE, "name": "cost_c1_s10", "signal": "mom_crowd_f20", "defense": "mix",
     "commission_bp": 1.0, "slippage_bp": 10.0},
]

# 持仓只数分布（单变体诊断）。
HOLD: list[dict] = [
    {**P1_BASE, "name": "champ_hold", "signal": "mom_crowd_f20", "defense": "mix"},
]

# 最优组合项（def_mix_crowd_f20）的邻域网格：防御腿类型 / 成本 / 目标波动 / 集中度。
ROBUST: list[dict] = [
    {**P1_BASE, "name": "dm_mix_c20", "signal": "mom_crowd_f20", "defense": "mix"},
    {**P1_BASE, "name": "dm_bond_c20", "signal": "mom_crowd_f20", "defense": "bond"},
    {**P1_BASE, "name": "dm_mix_c20_cost2x", "signal": "mom_crowd_f20", "defense": "mix",
     "cost_x": 2.0},
    {**P1_BASE, "name": "dm_mix_c20_vt12", "signal": "mom_crowd_f20", "defense": "mix",
     "target_vol": 0.12},
    {**P1_BASE, "name": "dm_mix_c20_vt18", "signal": "mom_crowd_f20", "defense": "mix",
     "target_vol": 0.18},
    {**P1_BASE, "name": "dm_mix_c20_m3", "signal": "mom_crowd_f20", "defense": "mix",
     "top_k": 3},
    {**P1_BASE, "name": "dm_mix_c20_m8", "signal": "mom_crowd_f20", "defense": "mix",
     "top_k": 8},
]


def level_from_returns(ret: pd.Series) -> pd.Series:
    """把日收益串成净值水平线（末尾乘以 100，量纲无关）。"""
    return (1.0 + ret.fillna(0.0)).cumprod() * 100.0


def build_defense(close: pd.DataFrame) -> dict[str, pd.Series]:
    """防御腿净值水平线：债 / 金 / 债金等权。"""
    ret = close.pct_change(fill_method=None)
    bond, gold = ret[BOND], ret[GOLD]
    return {"bond": bond, "gold": gold, "mix": 0.5 * bond + 0.5 * gold}


def build_share_pit(close: pd.DataFrame) -> pd.DataFrame:
    """份额的 PIT 日频面板：t 日只知 ``ANN_DATE <= t`` 的最新一期。

    份额表字段以「万份」计，量纲不影响分位排序。缺失公告日或份额的行丢弃；
    同一公告日多条取最后一条。
    """
    fs = pd.read_parquet(CACHE_ROOT / "fund_share.parquet").reset_index()
    fs["ann_date"] = pd.to_datetime(fs["ann_date"], errors="coerce")
    fs["value"] = fs["total_share"].fillna(fs["fund_share"])
    fs = fs.dropna(subset=["ann_date", "value"])

    cols = {}
    for code, g in fs.groupby("code"):
        if code not in close.columns:
            continue
        g = g.sort_values("ann_date").drop_duplicates("ann_date", keep="last")
        s = pd.Series(g["value"].to_numpy(), index=pd.DatetimeIndex(g["ann_date"]))
        cols[code] = s.reindex(close.index, method="ffill")
    pit = pd.DataFrame(cols).reindex(columns=close.columns)
    return pit


def build_crowding(close: pd.DataFrame) -> pd.DataFrame:
    """份额的 250 日滚动分位（0=历史最低，1=历史最高）。"""
    pit = build_share_pit(close)
    return pit.rolling(250, min_periods=120).rank(pct=True)


def build_r2(close: pd.DataFrame, win: int = 60) -> pd.DataFrame:
    """对数价格对时间的滚动回归 R²（趋势一致性）。"""
    logp = np.log(close.where(close > 0))
    t = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    return logp.rolling(win, min_periods=45).corr(t).pow(2)


def _rebalance_days(close: pd.DataFrame, spec: dict,
                    w0: int | None, w1: int | None) -> set | None:
    """``every_n`` 型调仓日：窗口内每 n 个交易日调一次；其余变体返回 None（用引擎频率）。"""
    n = spec.get("every_n")
    if not n:
        return None
    lo = pd.Timestamp(str(w0)) if w0 is not None else close.index.min()
    hi = pd.Timestamp(str(w1)) if w1 is not None else close.index.max()
    idx = close.index[(close.index >= lo) & (close.index <= hi)]
    return set(idx[:: int(n)])


def build_extra_panels(close: pd.DataFrame, panels: dict) -> dict:
    """在 P1 面板上追加 P2 三个信号面板。"""
    r2 = build_r2(close)
    crowd = build_crowding(close)
    mom = panels["mom12_1"]
    panels = dict(panels)
    panels["mom_r2"] = mom * r2
    panels["mom_crowd_f20"] = mom.where(crowd <= 0.80)
    panels["mom_crowd_f10"] = mom.where(crowd <= 0.90)
    return panels


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="ETF 轮动 P2 前置优化验证")
    ap.add_argument("--begin", type=int, default=20200102)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--variants", choices=("main", "robust", "freq", "cost", "hold"),
                    default="main")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = DataCache(create_datasource())
    bundle = load_bundle(cache, WARMUP_BEGIN, args.end)
    close = bundle[1]
    panels = build_extra_panels(close, build_panels(close, bundle[2]))
    def_levels = {k: level_from_returns(v) for k, v in build_defense(close).items()}
    print(f"份额 PIT 面板覆盖 {panels['mom_crowd_f20'].notna().sum().sum():,} 个有效格", flush=True)

    rows, seg_rows, eq = [], [], {}
    variants = {"main": VARIANTS, "robust": ROBUST, "freq": FREQ,
                "cost": COST, "hold": HOLD}[args.variants]
    tag = {"main": "", "robust": "_robust", "freq": "_freq",
           "cost": "_cost", "hold": "_hold"}[args.variants]
    for spec in variants:
        extra = None
        if spec.get("defense"):
            extra = pd.DataFrame({"DEF": def_levels[spec["defense"]]})
        full = run_variant(bundle, panels, spec, args.begin, args.end, extra_close=extra,
                           rebalance_days=_rebalance_days(close, spec, args.begin, args.end))
        segs = []
        for y0, y1 in SEGMENTS:
            lo, hi = max(args.begin, y0 * 10000 + 101), y1 * 10000 + 1231
            r = run_variant(bundle, panels, spec, lo, hi, extra_close=extra,
                            rebalance_days=_rebalance_days(close, spec, lo, hi))
            segs.append(r["annual_return"])
        n_pos = int(sum(s > 0 for s in segs))
        rows.append({**spec, **{k: v for k, v in full.items()
                                 if k not in ("equity", "hold_counts")},
                     "seg_pos": n_pos,
                     **{f"seg_{i + 1}": s for i, s in enumerate(segs)}})
        seg_rows.append({"name": spec["name"], **{f"seg_{i + 1}": s for i, s in enumerate(segs)}})
        eq[spec["name"]] = full["equity"]
        print(f"{spec['name']:<22} ann={full['annual_return']:+.2%} "
              f"sh={full['sharpe']:.2f} mdd={full['max_drawdown']:.2%} "
              f"to={full['avg_turnover']:.2%} hold={full['avg_holdings']:.1f} "
              f"cash={full['avg_cash']:.1%} cost={full['cost_drag']:.2%}/yr "
              f"seg={n_pos}/3", flush=True)
        if args.variants == "hold":
            cnt = full["hold_counts"]
            dist = cnt.value_counts(normalize=True).sort_index()
            print("  持仓只数分布: " + " / ".join(
                f"{int(k)}只 {v:.1%}" for k, v in dist.items()) +
                f"；平均 {cnt.mean():.2f} 只、最少 {int(cnt.min())} 只", flush=True)

    summary = pd.DataFrame(rows)
    if (summary["name"] == "p1_base").any():
        base = summary.loc[summary["name"] == "p1_base"].iloc[0]
        summary["d_sharpe"] = summary["sharpe"] - base["sharpe"]
        summary["d_mdd"] = summary["max_drawdown"] - base["max_drawdown"]
    summary = summary.sort_values("sharpe", ascending=False)
    summary.to_csv(out / f"summary{tag}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(seg_rows).to_csv(out / f"segments{tag}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eq).to_csv(out / f"equity{tag}.csv", encoding="utf-8-sig")
    (out / f"run_config{tag}.json").write_text(json.dumps({
        "begin": args.begin, "end": args.end, "bench": BENCH,
        "pool_size": len(close.columns), "min_amount": MIN_AMOUNT,
        "segments": SEGMENTS,
        "judge": "dSharpe>0.1 vs p1_base, MDD not worse, cost-positive",
        "tag": tag or "main", "variants": variants,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
