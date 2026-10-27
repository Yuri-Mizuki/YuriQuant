"""
ETF 轮动升级探针（P1）
======================

对照 `reports/docs/design/ETF轮动_研究与实施计划_2026-10-26.md` §三 P1：
在扩池（50 只）与 PIT 流动性掩码之上，检验四个常见改进相对等权 TopK 的增量：

  1. 12-1 动量（跳过最近一个月）          —— 信号层
  2. Dual Momentum（相对排序 + 绝对动量） —— 信号层
  3. inverse-vol 加权                    —— 组合构造
  4. 波动率目标（15% 年化）              —— 组合构造
  5. 缓冲区换手（进入前 15 / 跌出前 40）  —— 组合构造

判据（先验固定，跑完不改）：
  - 全样本 Sharpe ≥ 0.5
  - 全样本 MDD ≤ 35%
  - 三段（2020-21 / 2022-23 / 2024-26）年化收益 ≥ 2/3 为正

用法:
    python -m scripts.portfolio.etf_rotation_probe
    python -m scripts.portfolio.etf_rotation_probe --begin 20200102 --end 20261023
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import EtfCosts
from backtest.engine import VectorBacktest
from config import Config
from data.cache import DataCache
from data.datasource import create_datasource
from data.etf_universe import EtfUniverse
from strategy.etf_rotation import EtfRotationAdvanced
from strategy.multi_signal import compose_signals

BENCH = Config.benchmarks()["etf"]          # 沪深300
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "etf_rotation_probe"

MIN_AMOUNT = 3e7        # 近 60 日均成交额下限（元）
SEGMENTS = [(2020, 2021), (2022, 2023), (2024, 2026)]
WARMUP_BEGIN = 20180101  # 12-1 动量需要 ~273 个交易日暖机

# 变体清单：signal / freq / top_k / 构造开关
VARIANTS: list[dict] = [
    {"name": "baseline_mom_week_nomask", "signal": "composite", "freq": "W",
     "top_k": 5, "mask": False},
    {"name": "baseline_mom_week", "signal": "composite", "freq": "W",
     "top_k": 5, "mask": True},
    {"name": "mom12_1_m5", "signal": "mom12_1", "freq": "M", "top_k": 5},
    {"name": "dual_m5", "signal": "mom12_1", "freq": "M", "top_k": 5, "absolute": True},
    {"name": "dual_m8", "signal": "mom12_1", "freq": "M", "top_k": 8, "absolute": True},
    {"name": "dual_iv_m5", "signal": "mom12_1", "freq": "M", "top_k": 5,
     "absolute": True, "weighting": "inverse_vol"},
    {"name": "dual_vt15_m5", "signal": "mom12_1", "freq": "M", "top_k": 5,
     "absolute": True, "target_vol": 0.15},
    {"name": "dual_buf_m5", "signal": "mom12_1", "freq": "M", "top_k": 5,
     "absolute": True, "enter_rank": 15, "exit_rank": 40},
    # 组合项：同时叠加两个构造（先验上是「降波动 + 降换手」的自然搭配）
    {"name": "baseline_mom_week_vt15", "signal": "composite", "freq": "W",
     "top_k": 5, "target_vol": 0.15},
    {"name": "dual_vt15_m8", "signal": "mom12_1", "freq": "M", "top_k": 8,
     "absolute": True, "target_vol": 0.15},
    {"name": "dual_vt20_m5", "signal": "mom12_1", "freq": "M", "top_k": 5,
     "absolute": True, "target_vol": 0.20},
    {"name": "dual_ivvt15_m5", "signal": "mom12_1", "freq": "M", "top_k": 5,
     "absolute": True, "weighting": "inverse_vol", "target_vol": 0.15},
    {"name": "dual_bufvt15_m5", "signal": "mom12_1", "freq": "M", "top_k": 5,
     "absolute": True, "enter_rank": 15, "exit_rank": 40, "target_vol": 0.15},
]

# 主表 PASS 项的稳健性网格（**在看到主表结果后追加**，故单独成表、不加判据，
# 只用于判断「唯一过线项是不是邻域尖峰」）。目标波动 / top_k / 缓冲区 / 双倍成本。
ROBUST_VARIANTS: list[dict] = [
    {"name": "ivvt15_m5", "signal": "mom12_1", "freq": "M", "top_k": 5,
     "absolute": True, "weighting": "inverse_vol", "target_vol": 0.15},
    {"name": "ivvt12_m5", "signal": "mom12_1", "freq": "M", "top_k": 5,
     "absolute": True, "weighting": "inverse_vol", "target_vol": 0.12},
    {"name": "ivvt18_m5", "signal": "mom12_1", "freq": "M", "top_k": 5,
     "absolute": True, "weighting": "inverse_vol", "target_vol": 0.18},
    {"name": "ivvt15_m3", "signal": "mom12_1", "freq": "M", "top_k": 3,
     "absolute": True, "weighting": "inverse_vol", "target_vol": 0.15},
    {"name": "ivvt15_m8", "signal": "mom12_1", "freq": "M", "top_k": 8,
     "absolute": True, "weighting": "inverse_vol", "target_vol": 0.15},
    {"name": "ivvt15_buf_m5", "signal": "mom12_1", "freq": "M", "top_k": 5,
     "absolute": True, "weighting": "inverse_vol", "target_vol": 0.15,
     "enter_rank": 15, "exit_rank": 40},
    {"name": "ivvt15_m5_cost2x", "signal": "mom12_1", "freq": "M", "top_k": 5,
     "absolute": True, "weighting": "inverse_vol", "target_vol": 0.15, "cost_x": 2.0},
]


def load_bundle(cache, begin: int = WARMUP_BEGIN, end: int | None = None):
    """构建全样本面板束（含暖机），后续按窗口切片复用。"""
    uni = EtfUniverse(cache)
    close = uni.load_close(adjust=True)
    hi = pd.Timestamp(str(end)) if end else close.index.max()
    close = close.loc[(close.index >= pd.Timestamp(str(begin))) & (close.index <= hi)]

    raw = cache.read_daily("etf")
    amount = raw["amount"].unstack("code").reindex(index=close.index, columns=close.columns)

    idx = cache.get_index_daily(BENCH, begin, int(hi.strftime("%Y%m%d")))
    bench_close = idx["close"].unstack("code")
    bcol = [c for c in bench_close.columns if c == BENCH or c == BENCH.replace(".SH", "")]
    bench = (bench_close[bcol[0]].reindex(close.index).ffill()
             .pct_change(fill_method=None) if bcol else pd.Series(dtype=float))
    return uni, close, amount, bench


def build_panels(close: pd.DataFrame, amount: pd.DataFrame) -> dict:
    """由收盘价/成交额构建因子、波动率、流动性掩码面板。"""
    ret = close.pct_change(fill_method=None)
    composite, _, _ = compose_signals(close)
    return {
        "composite": composite,
        "mom12_1": close.shift(21) / close.shift(252) - 1.0,
        "vol": ret.rolling(20).std() * np.sqrt(252.0),
        "mask": amount.rolling(60, min_periods=40).mean() >= MIN_AMOUNT,
    }


def slice_bundle(bundle, panels: dict, w0: int | None, w1: int | None):
    """按 [w0, w1] 切片所有面板（因子已在暖机后计算，切片不损失暖机）。"""
    uni, close, amount, bench = bundle
    if w0 is None:
        return bundle, panels
    hi = pd.Timestamp(str(w1)) if w1 is not None else close.index.max()
    lo = pd.Timestamp(str(w0))
    m = (close.index >= lo) & (close.index <= hi)
    sub = (uni, close.loc[m],
           amount.loc[m], bench.loc[bench.index.isin(close.index[m])] if len(bench) else bench)
    sub_panels = {k: v.loc[m] for k, v in panels.items()}
    return sub, sub_panels


def run_variant(bundle, panels: dict, spec: dict, w0: int | None = None,
                w1: int | None = None, extra_close: pd.DataFrame | None = None,
                rebalance_days: set | None = None) -> dict:
    """在（可选）窗口上跑一个变体，返回指标与净值。

    ``extra_close`` 用于注入合成资产列（如防御腿 DEF：债+金的净值水平线）。
    合成资产的因子/波动率为 NaN（不会被选中），掩码恒为 True。
    """
    (uni, close, _amount, bench), p = slice_bundle(bundle, panels, w0, w1)
    extra_cols: list[str] = []
    if extra_close is not None:
        extra_close = extra_close.reindex(close.index)
        close = close.join(extra_close)
        extra_cols = list(extra_close.columns)
    # 现金资产：常数 1（0 收益）。引擎在施加掩码后会把多头腿归一化到 1，
    # 波动率目标的「留现金」只能显式落在一个 0 收益列上。
    returns = close.assign(CASH=1.0).pct_change(fill_method=None)
    factor = p[spec["signal"]].copy()
    factor["CASH"] = np.nan
    vol = p["vol"].copy()
    vol["CASH"] = np.nan
    mask = p["mask"].copy()
    mask["CASH"] = True
    for c in extra_cols:
        factor[c] = np.nan
        vol[c] = np.nan
        mask[c] = True
    cash = spec.get("cash_code")
    if cash is None and spec.get("target_vol"):
        cash = "DEF" if spec.get("defense") else "CASH"
    # 选标只在可交易集合内进行：不可执行标的的因子置 NaN。否则策略会先选中
    # 它们，引擎再把权重清零并重新归一化多头腿 ⇒ 持仓只数被动缩水、
    # 收益被"第 6 名"顶替，与「不许持有不达标标的」的真实约束不符。
    if spec.get("mask", True):
        factor = factor.where(mask)
    strategy = EtfRotationAdvanced(
        top_k=spec.get("top_k", 5),
        absolute_filter=spec.get("absolute", False),
        abs_threshold=spec.get("abs_threshold", 0.0),
        weighting=spec.get("weighting", "equal"),
        vol_panel=vol,
        target_vol=spec.get("target_vol"),
        enter_rank=spec.get("enter_rank"),
        exit_rank=spec.get("exit_rank"),
        cash_code=cash,
        ret_panel=returns,
        cov_window=spec.get("cov_window", 60),
        max_weight=spec.get("max_weight"),
        dedup_corr=spec.get("dedup_corr"),
        dedup_window=spec.get("dedup_window", 120),
        dedup_fill=spec.get("dedup_fill", False),
    )
    cx = float(spec.get("cost_x", 1.0))
    commission_bp = float(spec.get("commission_bp", 5.0))
    slippage_bp = float(spec.get("slippage_bp", 5.0))
    bt = VectorBacktest(strategy=strategy, rebalance_freq=spec.get("freq", "M"),
                        initial_capital=1_000_000.0,
                        costs=EtfCosts(commission_rate=commission_bp * 1e-4 * cx,
                                       commission_min=0.1 * cx,
                                       slippage_bp=slippage_bp * cx))
    res = bt.run(factor_panel=factor, returns_panel=returns,
                 executable_mask=mask if spec.get("mask", True) else None,
                 horizon=1, rebalance_days=rebalance_days)
    m = res.metrics(benchmark_returns=bench if len(bench) else None)
    w = res.weights_history
    live = w[w.abs().sum(axis=1) > 1e-9]
    held = [c for c in w.columns if c not in ("CASH", "DEF")]
    cash_w = live[cash] if cash in live.columns else pd.Series(0.0, index=live.index)
    years = max(len(res.daily_returns) / 252.0, 1e-9)
    cost_drag = float(res.cost_series.sum()) / 1_000_000.0 / years
    rb_dates = res.turnover_series.dropna().index
    wd = w.loc[rb_dates] if len(rb_dates) else w
    hold_counts = (wd[held].abs() > 1e-9).sum(axis=1)
    return {
        "annual_return": m["annual_return"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "excess": m["annual_return"] - (m.get("benchmark_annual_return") or 0.0),
        "avg_turnover": m.get("avg_turnover", 0.0),
        "n_rebalance": int(res.turnover_series.notna().sum()),
        "avg_holdings": float((live[held].abs() > 1e-9).sum(axis=1).mean()) if len(live) else 0.0,
        "avg_cash": float(cash_w.mean()) if len(cash_w) else 0.0,
        "cost_drag": cost_drag,
        "hold_counts": hold_counts,
        "rebalance_dates": rb_dates,
        "weights": w,
        "equity": res.equity_curve,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ETF 轮动升级探针（P1）")
    ap.add_argument("--begin", type=int, default=20200102, help="回测起点（暖机自动前移）")
    ap.add_argument("--end", type=int, default=None, help="回测终点，默认数据末日")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--variants", choices=("main", "robust"), default="main",
                    help="main=先验主表；robust=主表 PASS 项的稳健性网格")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = DataCache(create_datasource())
    bundle = load_bundle(cache, WARMUP_BEGIN, args.end)
    panels = build_panels(bundle[1], bundle[2])

    variants = VARIANTS if args.variants == "main" else ROBUST_VARIANTS
    tag = "" if args.variants == "main" else "_robust"
    rows, seg_rows, eq = [], [], {}
    for spec in variants:
        full = run_variant(bundle, panels, spec, args.begin, args.end)
        segs = []
        for y0, y1 in SEGMENTS:
            r = run_variant(bundle, panels, spec,
                            max(args.begin, y0 * 10000 + 101), y1 * 10000 + 1231)
            segs.append(r["annual_return"])
        n_pos = int(sum(s > 0 for s in segs))
        passed = (full["sharpe"] >= 0.5
                  and abs(full["max_drawdown"]) <= 0.35
                  and n_pos >= 2)
        rows.append({**{k: v for k, v in spec.items() if k != "mask"},
                     "name": spec["name"],
                     "mask": spec.get("mask", True),
                     **{k: v for k, v in full.items() if k != "equity"},
                     "seg_pos": n_pos,
                     **{f"seg_{i + 1}": s for i, s in enumerate(segs)},
                     "pass": passed})
        seg_rows.append({"name": spec["name"], **{f"seg_{i + 1}": s for i, s in enumerate(segs)}})
        eq[spec["name"]] = full["equity"]
        print(f"{spec['name']:<26} ann={full['annual_return']:+.2%} sh={full['sharpe']:.2f} "
              f"mdd={full['max_drawdown']:.2%} ex={full['excess']:+.2%} "
              f"seg={n_pos}/3 {'PASS' if passed else ''}", flush=True)

    summary = pd.DataFrame(rows).sort_values(["pass", "sharpe"], ascending=[False, False])
    summary.to_csv(out / f"summary{tag}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(seg_rows).to_csv(out / f"segments{tag}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eq).to_csv(out / f"equity{tag}.csv", encoding="utf-8-sig")
    (out / f"run_config{tag}.json").write_text(json.dumps({
        "begin": args.begin, "end": args.end, "bench": BENCH,
        "pool_size": len(bundle[1].columns), "min_amount": MIN_AMOUNT,
        "segments": SEGMENTS, "judge": "Sharpe>=0.5, |MDD|<=35%, seg>=2/3 positive",
        "tag": tag or "main", "variants": variants,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
