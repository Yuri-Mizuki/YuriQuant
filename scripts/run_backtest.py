"""
回测入口脚本
============

一键跑完: 数据 → 因子 → 策略 → 回测 → 报告

用法:
    python scripts/run_backtest.py                      # 默认 Mock 模式
    python scripts/run_backtest.py --real               # 用真实本地数据
    python scripts/run_backtest.py --factor momentum_20 --strategy topk_ls --k 30
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from backtest import VectorBacktest
from factor import ALL_FACTORS, FactorEngine
from research import generate_report
from strategy import QuantileLongShort, TopKLongOnly, TopKLongShort

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("backtest")


# ---------------------------------------------------------------------------
# Mock 数据（与 test_data_layer.py 一致）
# ---------------------------------------------------------------------------
def gen_mock_data(begin: int = 20200101, end: int = 20231231, n_codes: int = 100) -> dict[str, pd.DataFrame]:
    """生成 mock 日频面板数据。"""
    dates = pd.date_range(str(begin), str(end), freq="B")
    codes = [f"{600000+i:06d}.SH" for i in range(n_codes)]
    rng = np.random.default_rng(42)

    # 随机价格（几何布朗运动）
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
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="YuriQuant 回测")
    parser.add_argument("--real", action="store_true", help="使用真实本地数据")
    parser.add_argument("--factor", default="momentum_20", help="因子名称")
    parser.add_argument("--strategy", default="topk_ls", choices=["topk_ls", "topk_lo", "quantile"], help="策略")
    parser.add_argument("--k", type=int, default=30, help="持仓数")
    parser.add_argument("--freq", default="M", choices=["D", "W", "M"], help="调仓频率")
    args = parser.parse_args()

    log.info("=== YuriQuant 回测 ===")

    # 1. 数据
    if args.real:
        log.info("加载真实本地数据 ...")
        from data import get_cache
        cache = get_cache()
        from config import Config
        cfg = Config.get()
        begin = cfg["fetch"]["begin_date"]
        end = cfg["fetch"].get("end_date") or 20241231
        from data.universe import Universe
        uni = Universe(cache)
        codes = uni.get_hs300(end)
        kline = cache.get_daily_kline(codes, begin, end)
        # 转成面板
        close = kline["close"].unstack("code")
        high = kline["high"].unstack("code")
        low = kline["low"].unstack("code")
        volume = kline["volume"].unstack("code")
        amount = kline["amount"].unstack("code")
        panel = {"close": close, "high": high, "low": low, "volume": volume, "amount": amount}
    else:
        log.info("使用 Mock 数据 ...")
        panel = gen_mock_data()

    # 2. 因子
    log.info("计算因子: %s", args.factor)
    factor_cls = ALL_FACTORS[args.factor]()
    factor_values = factor_cls.calc(panel)
    # 因子面板: date × code
    factor_panel = factor_values if isinstance(factor_values, pd.DataFrame) else factor_values.unstack("code")

    # 3. 收益率面板（次日收益）
    returns_panel = panel["close"].pct_change().shift(-1)  # 信号日次日收益

    # 4. 策略
    if args.strategy == "topk_ls":
        strat = TopKLongShort(k=args.k)
    elif args.strategy == "topk_lo":
        strat = TopKLongOnly(k=args.k)
    else:
        strat = QuantileLongShort(n_quantiles=5)
    log.info("策略: %s, 调仓: %s", strat.name, args.freq)

    # 5. 回测
    log.info("执行回测 ...")
    bt = VectorBacktest(strategy=strat, rebalance_freq=args.freq)
    result = bt.run(factor_panel, returns_panel)

    # 6. 绩效
    log.info("\n%s", result.summary())

    # 7. 报告
    report_dir = Path("reports")
    generate_report(result, output_dir=report_dir)
    log.info("报告已生成: %s/", report_dir.resolve())

    # 8. 因子分析
    log.info("因子分析 ...")
    from research import factor_summary
    # 对齐
    common_dates = factor_panel.index.intersection(returns_panel.index)
    common_codes = factor_panel.columns.intersection(returns_panel.columns)
    fp = factor_panel.loc[common_dates, common_codes]
    rp = returns_panel.loc[common_dates, common_codes]
    summary = factor_summary(fp, rp)
    log.info("IC mean: %.4f, IR: %.4f", summary["ic_mean"], summary["ir"])

    log.info("=== 完成 ===")


if __name__ == "__main__":
    main()
