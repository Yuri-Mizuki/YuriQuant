"""
optimize —— 03 优化层（对齐聚宽 AI 投研流程）。

组合优化（portfolio.optimize_weights 启发式投影 + solver.optimize_weights_qp
求解器版：滚动 Ledoit-Wolf 协方差 + cvxpy QP，TEV/最小方差/MVO）
→ 风险归因（risk.risk_attribution：α/β + 基准对照 + Brinson；
  risk.risk_decomposition：Euler 分解 + 风格/行业方差贡献 + VaR/CVaR + 预算校验）
→ 持续监控（monitor.monitor_report：IC 漂移 / 衰减 / 自相关）。

现状（2026-08-14）：
- 启发式投影（行业中性 / 权重上下限 / 换手约束）已就绪，无 cvxpy 环境可用（mock venv）。
- 求解器版 P0 已就绪（optimize/solver.py，依赖 cvxpy，系统 python 3.12）：
  滚动 Ledoit-Wolf 收缩 Σ（防前视）+ cvxpy QP，约束线性精确满足（预算/上限/行业中性/换手）。
- P1 已就绪：风格中性化（style_exposures）、行业偏离限制（industry_deviation）、
  风险平价（method="risk_parity"）、HRP（hrp_weights / optimize_weights_hrp，免逆矩阵）。
- P2 已就绪：Black-Litterman（bl_posterior / bl_views_from_factor / method="bl"，
  因子得分→观点）、多空（allow_short + short_limit/gross_limit，回测多空口径）、
  Almgren-Chriss 成本惩罚（turnover_penalty 线性 + quadratic_cost 二次冲击）。
  对比脚本 scripts/compare_portfolio_methods.py（--mock / --real PIT 并集池四窗口）。
  待建（P3）：完整多期最优执行、风险预算非等权、真实四窗口结论分析。
"""
from optimize.monitor import monitor_report, rolling_ic
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
]
