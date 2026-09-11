"""
optimize —— 03 优化层「组合优化 / 风险 / 监控」。

**与 strategy/ 的边界（2026-09-11 第 3 批口径统一后定案）**

- **不需要风险模型**的权重生产 —— 信号构建 + 行业中性 / 权重上下限 / 换手投影
  —— 属"组合构建"，真源在 :mod:`strategy.constraints`。本包的
  ``portfolio.optimize_weights`` 已退化为**面板级薄门面**，不再持有独立实现。
- **需要协方差 Σ** 的组合优化（QP / HRP / BL）才是优化层本体，见 :mod:`optimize.solver`。
- 两层统一于 :class:`strategy.base.Strategy` 契约：优化产物经
  :class:`optimize.multi_period.PrecomputedWeightsStrategy` 适配后即可喂
  ``backtest.VectorBacktest``——这是优化结果接入回测引擎的唯一路径。

能力清单：

- 组合优化：``solver.optimize_weights_qp``（滚动 Ledoit-Wolf Σ + cvxpy QP：
  min_var / tev / mvo / risk_parity / bl）、``solver.optimize_weights_hrp``
  （层次聚类递归二分，免逆矩阵）。
- 多期执行：``multi_period.run_multi_period_backtest``（逐调仓日重解 QP，
  prev_weights 进换手/成本约束，由回测引擎按日记账评估）。
- 风险归因：``risk.risk_attribution``（α/β + 基准对照 + Brinson）、
  ``risk.risk_decomposition``（Euler 分解 + 风格/行业方差贡献 + VaR/CVaR）。
- 执行信号：``signals.build_signal_frame``（目标权重 → 整手可下单信号）。
- 持续监控：``monitor.monitor_report``（转发 :mod:`stats.monitor`）。

进阶能力（均已就绪）：风格中性化（``style_exposures``）、行业偏离
（``industry_deviation``）、多空（``allow_short`` + ``short_limit``/``gross_limit``）、
Almgren-Chriss 成本惩罚（``turnover_penalty`` 线性 + ``quadratic_cost`` 二次冲击）、
Black-Litterman（``bl_posterior`` / ``bl_views_from_factor``）。
对比脚本 ``scripts/compare_portfolio_methods.py``（--mock / --real PIT 并集池四窗口）。
待建（P3）：风险预算非等权、真实四窗口结论分析。
"""
from optimize.monitor import monitor_report, rolling_ic
from optimize.multi_period import PrecomputedWeightsStrategy
from optimize.portfolio import optimize_weights
from optimize.risk import risk_attribution, risk_decomposition
from optimize.solver import (
    bl_posterior,
    bl_views_from_factor,
    hrp_weights,
    optimize_weights_hrp,
    optimize_weights_qp,
    rolling_covariance,
    solve_portfolio,
)

__all__ = [
    "optimize_weights",
    "optimize_weights_qp",
    "solve_portfolio",
    "rolling_covariance",
    "bl_posterior",
    "bl_views_from_factor",
    "hrp_weights",
    "optimize_weights_hrp",
    "risk_attribution",
    "risk_decomposition",
    "rolling_ic",
    "monitor_report",
    "PrecomputedWeightsStrategy",
]
