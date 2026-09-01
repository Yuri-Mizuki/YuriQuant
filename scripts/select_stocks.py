"""
因子库 → 选股回测（合成/选股演示入口）
========================================

从 FactorLibrary 加载因子面板（raw 或 composite），用三种策略做截面选股：
TopK 多空 / TopK 纯多 / 分位多空，日/周/月调仓，输出：

1. 每日选股明细（多头/空头持仓代码 + 权重）CSV
2. 回测绩效对比表（年化/夏普/最大回撤/换手）
3. 最优组合的净值曲线图 + 分层净值图

用法
----
    # 用最佳复合因子选股（默认）
    python -m scripts.select_stocks --dataset hs300_2025 --factor composite2_orthogonal

    # 用单因子选股
    python -m scripts.select_stocks --dataset hs300_2025 --factor close30_ret

    # 全 raw 因子逐个跑对比（耗时较长）
    python -m scripts.select_stocks --dataset hs300_2025 --all-raw --top 10

    # 指定策略与频率
    python -m scripts.select_stocks --dataset hs300_2025 --factor composite2_orthogonal \
        --strategy topk_lo --k 50 --freq W

    # 只输出选股清单，不画图
    python -m scripts.select_stocks --dataset hs300_2025 --factor composite2_orthogonal --list-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cli_common import setup_logging  # noqa: E402


from backtest import VectorBacktest  # noqa: E402
from backtest.costs import ShortCostModel  # noqa: E402
from config import Config  # noqa: E402
from data.cache import DataCache  # noqa: E402
from data.cache_helpers import returns_from_cache  # noqa: E402
from data.offline import OfflineDataSource  # noqa: E402
from research.factor_library import FactorLibrary  # noqa: E402
from strategy.examples import build_strategy  # noqa: E402

log = setup_logging("select_stocks")

def extract_holdings(factor_panel: pd.DataFrame, strategy_name: str, k: int,
                     freq: str, returns_panel: pd.DataFrame) -> pd.DataFrame:
    """按调仓频率提取每个调仓日的选股清单（多头/空头代码 + 权重）。"""
    strat = build_strategy(strategy_name, k)
    VectorBacktest(strategy=strat, rebalance_freq=freq)
    # 复用引擎内部 rebalance 日期（因子面板索引 = 交易日）
    freq_map = {"D": 1, "W": 5, "M": 21}
    step = freq_map.get(freq, 21)
    dates = list(factor_panel.index)
    rows = []
    for i, d in enumerate(dates):
        if i % step != 0:
            continue
        vals = factor_panel.loc[d].dropna()
        if vals.empty:
            continue
        w = strat.get_weights(vals)
        if w.empty:
            continue
        for code, weight in w.items():
            rows.append({"date": d, "code": code, "weight": round(float(weight), 4),
                         "side": "long" if weight > 0 else "short"})
    return pd.DataFrame(rows)

def run_one(factor_name: str, panel: pd.DataFrame, returns_panel: pd.DataFrame,
            strategy_name: str, k: int, freq: str,
            short_costs=None, deleverage: bool = False) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    strat = build_strategy(strategy_name, k)
    bt = VectorBacktest(strategy=strat, rebalance_freq=freq,
                        short_costs=short_costs, deleverage=deleverage)
    result = bt.run(panel, returns_panel)
    holdings = extract_holdings(panel, strategy_name, k, freq, returns_panel)
    m = result.metrics()
    return holdings, result, m

def main():
    parser = argparse.ArgumentParser(description="因子库选股回测演示")
    parser.add_argument("--dataset", default="hs300_2025")
    parser.add_argument("--factor", default="composite2_orthogonal",
                        help="因子库因子名（raw 或 composite）")
    parser.add_argument("--all-raw", action="store_true", help="全部 raw 因子逐个回测对比")
    parser.add_argument("--top", type=int, default=10, help="--all-raw 时只取 |IC| 最大的 N 个")
    parser.add_argument("--strategy", default="topk_ls", choices=["topk_ls", "topk_lo", "quantile"])
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--freq", default="M", choices=["D", "W", "M"])
    parser.add_argument("--list-only", action="store_true", help="只输出选股清单，不画图")
    parser.add_argument("--no-short-cost", action="store_true",
                        help="关闭空头腿成本（借券费=0，旧口径；默认启用 8%% 年化）")
    parser.add_argument("--borrow-rate", type=float, default=None, help="年化借券费率（默认读配置 0.08）")
    parser.add_argument("--margin-ratio", type=float, default=None, help="融券保证金比例（默认读配置 1.0）")
    parser.add_argument("--deleverage", action="store_true",
                        help="1 倍资金约束：总保证金需求>1 时按比例降杠杆")
    args = parser.parse_args()

    # 空头腿成本模型：默认从配置读并启用
    _cfg_bt = dict(Config.get().get("backtest", {}))
    short_costs = ShortCostModel(
        borrow_rate=0.0 if args.no_short_cost else (args.borrow_rate if args.borrow_rate is not None
                                                    else _cfg_bt.get("short_borrow_rate", 0.08)),
        margin_ratio=args.margin_ratio if args.margin_ratio is not None
                      else _cfg_bt.get("short_margin_ratio", 1.0),
    )

    cache = DataCache(OfflineDataSource())
    lib = FactorLibrary(dataset=args.dataset)
    reg = lib.list_all()
    if reg.empty:
        log.error("数据集 %s 为空", args.dataset)
        sys.exit(1)

    returns_panel = returns_from_cache(cache, 20250101, 20251231)

    if args.all_raw:
        raws = reg[reg["kind"] == "raw"].copy()
        raws = raws.sort_values("ic_mean", key=lambda s: s.abs(), ascending=False).head(args.top)
        targets = list(raws["name"])
    else:
        targets = [args.factor]

    out_dir = Path("reports") / f"select_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    best_result = None
    best_name = None
    for name in targets:
        panel = lib.get_panel(name)
        if panel is None or panel.empty:
            log.warning("跳过 %s（无面板）", name)
            continue
        rp = returns_panel.reindex(index=panel.index, columns=panel.columns)
        try:
            holdings, result, m = run_one(name, panel, rp, args.strategy, args.k, args.freq,
                                          short_costs=short_costs, deleverage=args.deleverage)
        except Exception as e:
            log.warning("%s 回测失败: %s", name, e)
            continue

        # 存选股清单
        slug = name.replace("(", "_").replace(")", "").replace(",", "_").replace(" ", "")
        hpath = out_dir / f"holdings_{slug}.csv"
        holdings.to_csv(hpath, index=False)

        summary_rows.append({
            "factor": name,
            "strategy": args.strategy,
            "k": args.k,
            "freq": args.freq,
            "annual_return": m.get("annual_return"),
            "sharpe": m.get("sharpe"),
            "sortino": m.get("sortino"),
            "max_drawdown": m.get("max_drawdown"),
            "calmar": m.get("calmar"),
            "win_rate": m.get("win_rate"),
            "avg_turnover": m.get("avg_turnover"),
            "holdings_file": str(hpath),
        })
        if best_result is None or (m.get("sharpe") or -99) > (summary_rows[-1].get("sharpe") or -99):
            best_result = result
            best_name = name
        log.info("  %-28s sharpe=%.3f 年化=%.1f%% 回撤=%.1f%% 换手=%.2f",
                 name, m.get("sharpe") or 0, (m.get("annual_return") or 0) * 100,
                 (m.get("max_drawdown") or 0) * 100, m.get("avg_turnover") or 0)

    if not summary_rows:
        log.error("无成功回测")
        sys.exit(1)

    summary = pd.DataFrame(summary_rows)
    sum_path = out_dir / "summary.csv"
    summary.to_csv(sum_path, index=False)
    print("\n===== 选股回测汇总（%s / %s / k=%d / %s）=====" %
          (args.dataset, args.strategy, args.k, args.freq))
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.4f}"):
        print(summary[["factor", "annual_return", "sharpe", "sortino",
                       "max_drawdown", "calmar", "win_rate", "avg_turnover"]].to_string(index=False))
    print(f"\n选股清单目录: {out_dir}")

    # 净值曲线（仅单因子模式）
    if not args.list_only and best_result is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            eq = best_result.equity_curve
            ax.plot(eq.index, eq.values, linewidth=1.2)
            ax.set_title(f"净值曲线: {best_name} ({args.strategy}, k={args.k}, {args.freq})")
            ax.set_xlabel("日期"); ax.set_ylabel("净值")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_path = out_dir / f"equity_{best_name.replace('(','_').replace(')','').replace(',','_').replace(' ','')}.png"
            fig.savefig(fig_path, dpi=120)
            plt.close(fig)
            log.info("净值图: %s", fig_path)
        except Exception as e:
            log.warning("画图失败: %s", e)

if __name__ == "__main__":
    main()