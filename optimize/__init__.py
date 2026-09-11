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
对比脚本 ``scripts/portfolio/compare_portfolio_methods.py``（--mock / --real PIT 并集池四窗口）。
待建（P3）：风险预算非等权、真实四窗口结论分析。

**惰性导入（2026-09-11 P2）**：本包的子模块各有重依赖（solver→cvxpy 7.3s 冷启、
monitor→stats.ic→scipy.stats 4.0s、risk→research.xlsx_report→openpyxl 2.4s），
此前 ``__init__`` 静态 import 全部子模块，导致 ``import optimize`` 实测 10.2s
（**与 cvxpy 有关但不全是它**）。现改为 PEP 562 模块级 ``__getattr__`` 按需加载：
``import optimize`` 本身不再拉起 scipy/cvxpy/openpyxl，访问 ``optimize.<name>``
（或 ``from optimize import <name>``）时才加载对应子模块并缓存。
公开名字与对象身份不变（见 ``tests/test_optimize_lazy_import.py``）。
"""
from __future__ import annotations

import importlib

#: 公开 API 名 → 所在子模块（首次访问时加载并缓存到 globals）
_LAZY_API: dict[str, str] = {
    "monitor_report": "optimize.monitor",
    "rolling_ic": "optimize.monitor",
    "PrecomputedWeightsStrategy": "optimize.multi_period",
    "optimize_weights": "optimize.portfolio",
    "risk_attribution": "optimize.risk",
    "risk_decomposition": "optimize.risk",
    "bl_posterior": "optimize.solver",
    "bl_views_from_factor": "optimize.solver",
    "hrp_weights": "optimize.solver",
    "optimize_weights_hrp": "optimize.solver",
    "optimize_weights_qp": "optimize.solver",
    "rolling_covariance": "optimize.solver",
    "solve_portfolio": "optimize.solver",
}

#: 子模块本身也惰性加载（`optimize.solver` 这类属性访问不报 AttributeError）
_LAZY_SUBMODULES = ("monitor", "multi_period", "portfolio", "risk", "solver")

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


def __getattr__(name: str):
    """PEP 562：按需加载公开名 / 子模块（避免 import optimize 付重依赖代价）。"""
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f"optimize.{name}")
        globals()[name] = mod
        return mod
    if name in _LAZY_API:
        mod = importlib.import_module(_LAZY_API[name])
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_API) | set(_LAZY_SUBMODULES))
