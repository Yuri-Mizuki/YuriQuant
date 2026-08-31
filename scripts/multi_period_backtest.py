"""
多期组合执行回测脚本（P3，2026-08-20）
====================================

把单期精确优化（QP：滚动 Σ / 换手 / 行业风格约束）+ 向量化回测（成本/借券/指标）
拼成逐调仓日回放的**完整多期执行**。口径与单因子回测一致。

用法:
    python scripts/multi_period_backtest.py                     # mock 月度（默认）
    python scripts/multi_period_backtest.py --freq M --method mvo
    python scripts/multi_period_backtest.py --freq W --method min_var
    python scripts/multi_period_backtest.py --freq M --allow-short --short-limit 0.3

    # 真实数据（需先 python -m scripts.update_data 拉日线/行业/股本；PIT 并集池）
    python scripts/multi_period_backtest.py --real --begin 20230101 --end 20241231
    python -m scripts.multi_period_backtest --real --begin 20250101 --end 20251231 --method tev

输出（默认 reports/multi_period/）:
    mp_backtest_{mode}_{freq}_{method}_{begin}_{end}.csv   绩效与换手汇总行
    mp_target_{mode}_{freq}_{method}_{begin}_{end}.csv     目标权重（调仓日 × code）
"""

from __future__ import annotations

import sys
import argparse
import warnings
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cli_common import add_real_mock_args, setup_logging  # noqa: E402


from factor.preprocessing import winsorize_mad  # noqa: E402
from optimize.multi_period import RebalanceConfig, run_multi_period_backtest  # noqa: E402

log = setup_logging("multi_period_backtest")

warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

OUT_DIR = Path("reports/multi_period")

def build_mock(n_days: int = 500, n_codes: int = 60, seed: int = 0) -> dict[str, pd.DataFrame]:
    """生成含可预测 AR(1) 信号的面板（因子=滞后收益，与收益真实相关）。"""
    from scripts.compare_portfolio_methods import gen_mock_panel
    return gen_mock_panel(n_days=n_days, n_codes=n_codes, seed=seed)

def build_real(begin: int, end: int, index: str = "000300.SH"):
    """真实 PIT 并集池：收益面板 + 行业/市值风格面板；因子=滞后收益动量代理。"""
    from scripts.compare_portfolio_methods import load_real_data
    data = load_real_data(begin, end, index)
    returns = data["returns"]
    data["factor"] = winsorize_mad(returns.shift(1))
    data["factor"] = data["factor"].reindex(index=returns.index, columns=returns.columns)
    data["industry_panel"] = data.get("industry_panel")
    data["style_panels"] = data.get("style_panels")
    return data

def to_cfg(args, data) -> RebalanceConfig:
    kw = dict(
        rebalance_freq=args.freq,
        method=args.method,
        max_weight=args.max_weight,
        max_turnover=args.max_turnover,
        turnover_penalty=args.turnover_penalty,
        allow_short=args.allow_short,
    )
    if args.short_limit is not None:
        kw["short_limit"] = args.short_limit
    if data.get("industry_panel") is not None:
        kw["industry_panel"] = data["industry_panel"]
    if data.get("style_panels"):
        kw["style_exposures"] = data["style_panels"]
    if args.method == "tev":
        factor = data["factor"]
        kw["benchmark_weights"] = pd.Series(1.0 / len(factor.columns), index=factor.columns)
    return RebalanceConfig(**kw)

def main() -> None:
    ap = argparse.ArgumentParser(description="多期组合执行回测（P3）")
    ap.add_argument("--freq", default="M", choices=["D", "W", "M"], help="调仓频率")
    ap.add_argument("--method", default="mvo",
                    choices=["min_var", "tev", "mvo", "risk_parity", "bl"], help="单期优化方法")
    ap.add_argument("--max-weight", type=float, default=0.1)
    ap.add_argument("--max-turnover", type=float, default=0.5)
    ap.add_argument("--turnover-penalty", type=float, default=0.0)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--short-limit", type=float, default=None)
    ap.add_argument("--n-days", type=int, default=500)
    ap.add_argument("--n-codes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    add_real_mock_args(ap, real_help="真实本地数据（需先 update_data）")
    ap.add_argument("--begin", type=int, default=None, help="真实数据起始日 YYYYMMDD")
    ap.add_argument("--end", type=int, default=None, help="真实数据结束日 YYYYMMDD")
    ap.add_argument("--index", default="000300.SH")
    ap.add_argument("--initial-capital", type=float, default=None)
    args = ap.parse_args()

    if args.real:
        from config import Config
        cfg_c = Config.get()
        begin = args.begin or cfg_c["fetch"]["begin_date"]
        end = args.end
        if end is None:
            from data import get_cache
            cal = get_cache().get_calendar(begin)
            end = cal[-1] if cal else begin
        data = build_real(int(begin), int(end), args.index)
        mode = "real"
        log.info("真实数据: %d 天 × %d 只（%s~%s），因子=滞后收益动量代理（可换因子库合成因子）",
                 len(data["returns"]), len(data["returns"].columns), begin, end)
    else:
        data = build_mock(n_days=args.n_days, n_codes=args.n_codes, seed=args.seed)
        mode = "mock"
        log.info("mock 面板: %d 天 × %d 股（seed=%d），因子=滞后收益",
                 args.n_days, args.n_codes, args.seed)

    factor, returns = data["factor"], data["returns"]
    cfg = to_cfg(args, data)
    result, target = run_multi_period_backtest(
        factor, returns, cfg=cfg,
        initial_capital=args.initial_capital,
    )

    # ---- 汇总输出 ----
    m = result.metrics()
    period = f"{begin}_{end}" if args.real else f"{args.n_days}d_s{args.seed}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "mode": mode, "freq": args.freq, "method": args.method,
        "window": f"{cfg.window}/{cfg.min_periods}",
        "rebalances": len(target),
        "n_days": m.get("n_days", 0),
        **{k: m.get(k) for k in ("annual_return", "annual_volatility", "sharpe", "sortino",
                                "max_drawdown", "calmar", "win_rate", "avg_turnover")},
    }
    meta_path = out_dir / f"mp_backtest_{mode}_{args.freq}_{args.method}_{period}.csv"
    pd.DataFrame([row]).to_csv(meta_path, index=False, encoding="utf-8-sig")
    target.to_csv(out_dir / f"mp_target_{mode}_{args.freq}_{args.method}_{period}.csv",
                  encoding="utf-8-sig")

    print(f"\n多期组合执行回测（{mode} · {args.freq} · {args.method}，调仓数={len(target)}）")
    for k, fmt in (
        ("annual_return", "{:.2%}"), ("annual_volatility", "{:.2%}"),
        ("sharpe", "{:.3f}"), ("sortino", "{:.3f}"),
        ("max_drawdown", "{:.2%}"), ("calmar", "{:.3f}"),
        ("avg_turnover", "{:.2%}"),
    ):
        if k in m:
            print(f"  {k:<20}" + fmt.format(m[k]))
    print(f"  {'n_days':<20} {m.get('n_days', 0)}")
    short_k = [k for k in ("avg_long_exposure", "avg_short_exposure",
                           "borrow_fee_drag_annual") if k in m]
    for k in short_k:
        if k == "borrow_fee_drag_annual":
            print(f"  {k:<20} {m[k]:.4%}")
        else:
            print(f"  {k:<20} {m[k]:.4f}")
    log.info("已输出汇总: %s", meta_path)

if __name__ == "__main__":
    main()