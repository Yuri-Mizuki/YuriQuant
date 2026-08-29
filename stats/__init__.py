"""
stats —— 全库共享的纯统计公共层
==================================

2026-08-29 由 research/robust_stats、research/factor_analysis（IC 族）与
optimize/monitor 下沉合并而成。定位：**只依赖 numpy/pandas/scipy 的底层
统计工具**，供 config 之上的所有层（factor/model/backtest/research/optimize/
monitoring/scripts）直接消费——核心包不得再反向 import research 取统计函数
（此前 factor→research 与 research↔optimize 两个包级循环由此而生）。

模块清单：
- ``stats.robust_stats`` : Newey-West HAC 稳健推断（nw_tstat / ols_newey_west / auto_lag）
- ``stats.ic``           : 因子 IC 统计（calc_ic_series / calc_ir / calc_ic_decay /
                           quantile_backtest / factor_autocorr）
- ``stats.monitor``      : IC 漂移监控统计（rolling_ic / monitor_ic_series / monitor_report）

兼容转出口（保持旧 import 路径可用，真源在本包）：
- ``research.robust_stats`` / ``research.factor_analysis``（IC 族 re-export）
- ``optimize.monitor``
"""
from __future__ import annotations

# 年化交易日数（单一真源）：全库所有年化口径（收益/波动/Sharpe/IR/换手）
# 统一引用本常量，禁止各处再写 244/252 字面量（2026-08-29 收敛：
# e2e_backtest.perf_stats 曾用 244，与引擎 252 分裂导致两族报告差 ~3%）。
# backtest.metrics 对本常量做 re-export（历史 import 路径继续可用）。
PERIODS_PER_YEAR = 252

__all__ = ["PERIODS_PER_YEAR"]
