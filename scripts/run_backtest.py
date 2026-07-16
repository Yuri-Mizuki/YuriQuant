"""
回测入口脚本
============

一键跑完: 数据 → 因子 → 策略 → 回测 → 报告

支持多因子批量回测，统一对比。

用法（安装为可编辑包后，python scripts/run_backtest.py 与
python -m scripts.run_backtest 都可以在任意 cwd 下运行；参见 README 的安装说明）:
    # 单因子（Mock）
    python scripts/run_backtest.py --factor momentum_20

    # 多因子对比（Mock）
    python scripts/run_backtest.py --factors momentum_20,volatility_20,turnover_20

    # 全部因子
    python scripts/run_backtest.py --factors all

    # 用真实数据
    python scripts/run_backtest.py --real --factors all

    # 指定策略和调仓
    python scripts/run_backtest.py --factors all --strategy topk_ls --k 30 --freq M
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import VectorBacktest
from factor import ALL_FACTORS
from research import generate_comparison_report, generate_excel_report, generate_single_report
from research.factor_analysis import factor_summary
from strategy import QuantileLongShort, TopKLongOnly, TopKLongShort

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("backtest")


# ---------------------------------------------------------------------------
# Mock 数据
# ---------------------------------------------------------------------------
def gen_mock_data(begin: int = 20200101, end: int = 20231231, n_codes: int = 100) -> dict[str, pd.DataFrame]:
    """生成 mock 日频面板数据。"""
    dates = pd.date_range(str(begin), str(end), freq="B")
    codes = [f"{600000+i:06d}.SH" for i in range(n_codes)]
    rng = np.random.default_rng(42)

    base = 10.0 + rng.uniform(0, 50, n_codes)
    rets = rng.normal(0, 0.02, (len(dates), n_codes))
    close = pd.DataFrame(0.0, index=dates, columns=codes)
    for i in range(n_codes):
        close.iloc[:, i] = base[i] * np.exp(np.cumsum(rets[:, i]))
    high = close * (1 + rng.uniform(0, 0.03, (len(dates), n_codes)))
    low = close * (1 - rng.uniform(0, 0.03, (len(dates), n_codes)))
    volume = pd.DataFrame(rng.integers(1e6, 1e8, (len(dates), n_codes)), index=dates, columns=codes)
    amount = volume * close

    return {"close": close, "high": high, "low": low, "volume": volume, "amount": amount}


# ---------------------------------------------------------------------------
# 策略工厂
# ---------------------------------------------------------------------------
def build_strategy(name: str, k: int):
    if name == "topk_ls":
        return TopKLongShort(k=k)
    elif name == "topk_lo":
        return TopKLongOnly(k=k)
    else:
        return QuantileLongShort(n_quantiles=5)


# ---------------------------------------------------------------------------
# 单因子回测
# ---------------------------------------------------------------------------
def run_single_factor(
    factor_name: str,
    panel: dict[str, pd.DataFrame],
    strategy_name: str,
    k: int,
    freq: str,
    returns_panel: pd.DataFrame,
    executable_mask: pd.DataFrame | None = None,
) -> tuple[str, object, dict]:
    """跑单个因子回测，返回 (因子名, BacktestResult, factor_summary)。"""
    factor_cls = ALL_FACTORS[factor_name]()
    factor_values = factor_cls.calc(panel)
    factor_panel = factor_values if isinstance(factor_values, pd.DataFrame) else factor_values.unstack("code")

    strat = build_strategy(strategy_name, k)
    bt = VectorBacktest(strategy=strat, rebalance_freq=freq)
    result = bt.run(factor_panel, returns_panel, executable_mask=executable_mask)

    # 因子分析
    common_dates = factor_panel.index.intersection(returns_panel.index)
    common_codes = factor_panel.columns.intersection(returns_panel.columns)
    fp = factor_panel.loc[common_dates, common_codes]
    rp = returns_panel.loc[common_dates, common_codes]
    fs = factor_summary(fp, rp)

    return factor_name, result, fs


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="YuriQuant 回测")
    parser.add_argument("--real", action="store_true", help="使用真实本地数据")
    parser.add_argument("--factor", default=None, help="单因子名称")
    parser.add_argument("--factors", default=None, help="多因子名称(逗号分隔)或 all")
    parser.add_argument("--strategy", default="topk_ls", choices=["topk_ls", "topk_lo", "quantile"], help="策略")
    parser.add_argument("--k", type=int, default=30, help="持仓数")
    parser.add_argument("--freq", default="M", choices=["D", "W", "M"], help="调仓频率")
    args = parser.parse_args()

    # 确定因子列表
    if args.factors:
        if args.factors == "all":
            factor_list = list(ALL_FACTORS.keys())
        else:
            factor_list = [f.strip() for f in args.factors.split(",")]
    elif args.factor:
        factor_list = [args.factor]
    else:
        factor_list = ["momentum_20"]

    log.info("=== YuriQuant 回测 ===")
    log.info("因子: %s", factor_list)

    # 1. 数据
    executable_mask = None
    if args.real:
        log.info("加载真实本地数据 ...")
        from data import get_cache
        cache = get_cache()
        from config import Config
        cfg = Config.get()
        begin = cfg["fetch"]["begin_date"]
        end = cfg["fetch"].get("end_date") or 20241231
        from data.tradability import build_executable_mask
        from data.universe import Universe
        uni = Universe(cache)
        codes = uni.get_hs300(end)
        kline = cache.get_daily_kline(codes, begin, end)
        close = kline["close"].unstack("code")
        high = kline["high"].unstack("code")
        low = kline["low"].unstack("code")
        volume = kline["volume"].unstack("code")
        amount = kline["amount"].unstack("code")

        # 复权：按 config.universe.adjust 决定是否把原始价转成后复权价
        adjust_mode = cfg.get("universe", {}).get("adjust", "backward")
        if adjust_mode == "backward":
            log.info("拉取后复权因子并应用到价格 ...")
            backward = cache.get_backward_factor(codes)
            backward = backward.reindex(index=close.index, columns=close.columns).ffill()
            close = close * backward
            high = high * backward
            low = low * backward
            log.info("后复权因子行数: %d, 列数: %d", len(backward), backward.shape[1])
        elif adjust_mode != "none":
            log.warning("未知的 universe.adjust=%s，跳过复权", adjust_mode)

        panel = {"close": close, "high": high, "low": low, "volume": volume, "amount": amount}

        # 可执行性掩码：涨跌停/停牌 AND 动态(point-in-time)成分股归属
        log.info("拉取历史涨跌停/停牌状态 ...")
        status = cache.get_history_stock_status(codes, begin, end)
        tradability_mask = build_executable_mask(
            status, close.index, close.columns, close_panel=close
        )
        membership_mask = uni.get_membership_mask(cfg["universe"]["index_code"], close.index)
        membership_mask = membership_mask.reindex(
            index=close.index, columns=close.columns
        ).fillna(False)
        executable_mask = tradability_mask & membership_mask
        log.info(
            "可执行性掩码: 平均每日可交易 %.1f / %d 只",
            executable_mask.sum(axis=1).mean(), len(close.columns),
        )
    else:
        log.info("使用 Mock 数据 ...")
        panel = gen_mock_data()

    # 2. 收益率面板
    returns_panel = panel["close"].pct_change().shift(-1)

    # 3. 批量回测
    results = {}
    factor_summaries = {}
    for fn in factor_list:
        log.info("回测因子: %s (策略: %s, 调仓: %s)", fn, args.strategy, args.freq)
        name, result, fs = run_single_factor(
            fn, panel, args.strategy, args.k, args.freq, returns_panel, executable_mask
        )
        results[name] = result
        factor_summaries[name] = fs
        log.info("  年化收益: %.2f%%, 夏普: %.4f, IC: %.4f",
                 result.metrics()["annual_return"] * 100,
                 result.metrics()["sharpe"],
                 fs.get("ic_mean", 0))

    # 4. 报告
    report_dir = Path("reports")
    if len(results) == 1:
        # 单因子 → 完整报告
        name, result = list(results.items())[0]
        fs = factor_summaries.get(name, {})
        generate_single_report(result, name, factor_summary=fs, output_dir=report_dir)
        log.info("单因子报告: %s/report_%s.png", report_dir.resolve(), name)
    else:
        # 多因子 → 每个单独报告 + 对比报告
        for name, result in results.items():
            fs = factor_summaries.get(name, {})
            generate_single_report(result, name, factor_summary=fs, output_dir=report_dir)
        generate_comparison_report(results, factor_summaries, output_dir=report_dir)
        log.info("多因子对比报告: %s/comparison.png", report_dir.resolve())

    # Excel 报告（所有情况都生成）
    xlsx_path = report_dir / "yuriquant_report.xlsx"
    generate_excel_report(results, factor_summaries, output_path=xlsx_path)
    log.info("Excel 报告: %s", xlsx_path.resolve())

    log.info("=== 完成 ===")


if __name__ == "__main__":
    main()
