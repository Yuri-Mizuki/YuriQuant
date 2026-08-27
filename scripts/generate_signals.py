"""
每日交易信号生成脚本（P3，2026-08-20）
====================================

把多期组合执行的**目标权重**转成带约束校验的**每日可执行交易信号**并导出。

用法:
    # mock（无涨跌停/停牌，全可交易）
    python -m scripts.generate_signals --freq M --method mvo --capital 100000000

    # 真实（需系统 Python D:\\python\\Python312\\python.exe，且先 update_data）
    python -m scripts.generate_signals --real --begin 20250101 --end 20251231 --freq M --method mvo

输出（默认 reports/signals/）:
    signals_{mode}_{freq}_{method}_{period}.csv  长表: 每调仓日×股票的目标股数/方向/受阻
    signals_{mode}_{freq}_{method}_{period}.txt  单日可读清单（最新调仓日）
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from optimize.multi_period import RebalanceConfig, run_multi_period_backtest
from optimize.signals import signals_from_rebalances

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_signals")

OUT_DIR = Path("reports/signals")


def build_rebalance_config(args, data) -> RebalanceConfig:
    kw = dict(rebalance_freq=args.freq, method=args.method, max_weight=args.max_weight)
    if data.get("industry_panel") is not None:
        kw["industry_panel"] = data["industry_panel"]
    if data.get("style_panels"):
        kw["style_exposures"] = data["style_panels"]
    return RebalanceConfig(**kw)


def main() -> None:
    ap = argparse.ArgumentParser(description="每日交易信号生成（P3）")
    ap.add_argument("--freq", default="M", choices=["D", "W", "M"])
    ap.add_argument("--method", default="mvo",
                    choices=["min_var", "tev", "mvo", "risk_parity", "bl"])
    ap.add_argument("--max-weight", type=float, default=0.1)
    ap.add_argument("--capital", type=float, default=100_000_000.0, help="组合总资金")
    ap.add_argument("--lot", type=int, default=100, help="最小交易单位（股）")
    ap.add_argument("--n-days", type=int, default=500)
    ap.add_argument("--n-codes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--begin", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--index", default="000300.SH")
    args = ap.parse_args()

    if args.real:
        from config import Config
        from scripts.compare_portfolio_methods import load_real_data
        cfg_c = Config.get()
        begin = args.begin or cfg_c["fetch"]["begin_date"]
        end = args.end
        if end is None:
            from data import get_cache
            cal = get_cache().get_calendar(begin)
            end = cal[-1] if cal else begin
        data = load_real_data(int(begin), int(end), args.index)
        price = data["close_raw"]
        from factor.preprocessing import winsorize_mad
        data["factor"] = winsorize_mad(data["returns"].shift(1)).reindex(
            index=data["returns"].index, columns=data["returns"].columns
        )
        mode = "real"
        period = f"{begin}_{end}"
        log.info("真实数据 %d 天 × %d 只", len(price), price.shape[1])

        buyable = sellable = None
        try:  # 状态数据缺失/不可用时降级为不做方向校验（不阻断信号产出）
            pass
        except Exception as e:
            log.warning("跳过涨跌停/停牌校验（SDK 不可用）: %s", e)
            buyable = sellable = None
        else:
            buyable, sellable = _load_directional_masks(
                data["codes"].tolist(), int(begin), int(end), price
            )
    else:
        from scripts.compare_portfolio_methods import gen_mock_panel
        data = gen_mock_panel(n_days=args.n_days, n_codes=args.n_codes, seed=args.seed)
        price = data["close"]
        mode = "mock"
        period = f"{args.n_days}d_s{args.seed}"
        buyable = sellable = None
        log.info("mock 面板 %d 天 × %d 股", len(price), price.shape[1])

    factor, returns = data["factor"], data["returns"]
    result, target = run_multi_period_backtest(
        factor, returns, cfg=build_rebalance_config(args, data),
    )
    signals = signals_from_rebalances(
        target, price, capital=args.capital,
        buyable=buyable, sellable=sellable, lot=args.lot,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"signals_{mode}_{args.freq}_{args.method}_{period}.csv"
    signals.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 最新调仓日的可读清单
    latest = target.index.max() if len(target) else None
    print(f"\n每日交易信号（{mode} · {args.freq} · {args.method}，调仓日数={len(target)}）")
    print(f"资金 ${args.capital/1e6:.0f}M · 最小单位 {args.lot} 股 · 换股数用未复权价")
    if latest is not None:
        latest_frame = signals[signals["signal_date"] == latest] if len(signals) else pd.DataFrame()
        txt_path = out_dir / f"signals_{mode}_{args.freq}_{args.method}_{period}.txt"
        _write_latest(txt_path, signals, latest, args.capital)
        if not latest_frame.empty:
            print(f"\n最新调仓日 {pd.Timestamp(latest).date()} 信号（前 15 行）：")
            cols = ["code", "qty_target", "qty_order", "direction", "status", "note"]
            print(latest_frame[cols].head(15).to_string(index=False))
            gc = latest_frame["status"].value_counts()
            print("\n当日状态分布：", dict(gc.to_dict()))
            print("\n受阻明细：")
            blocked = latest_frame[latest_frame["status"].str.startswith("BLOCKED")]
            print(blocked[["code", "direction", "status", "note"]].to_string(index=False)
                  if not blocked.empty else "  (无受阻交易)")
    log.info("已输出: %s", csv_path)


def _load_directional_masks(codes, begin, end, price):
    from data.cache import DataCache
    from data.datasource import create_datasource
    from data.tradability import build_directional_masks
    cache = DataCache(create_datasource())
    status = cache.get_history_stock_status(list(codes), int(begin), int(end))
    buyable, sellable = build_directional_masks(
        status, price.index, price.columns, close_panel=price
    )
    n_blocked_buy = (~buyable).sum().sum()
    n_blocked_sell = (~sellable).sum().sum()
    log.info("方向校验就绪：不可买入 %d 例 / 不可卖出 %d 例", n_blocked_buy, n_blocked_sell)
    return buyable, sellable


def _write_latest(txt_path, signals, latest, capital):
    rows = signals[signals["signal_date"] == latest]
    lines = [f"最新调仓日 {pd.Timestamp(latest).date()} 交易信号（资金 ${capital/1e6:.0f}M）",
             f"{'code':<12}{'目标股数':>10}{'委托股数':>10}{'方向':<7}{'状态':<11}说明"]
    for _, r in rows.sort_values("qty_target", key=abs, ascending=False).iterrows():
        lines.append(f"{r['code']:<12}{r['qty_target']:>10.0f}{r['qty_order']:>10}{r['direction']:<7}"
                     f"{r['status']:<11}{r['note']}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("最新清单: %s", txt_path)


if __name__ == "__main__":
    main()