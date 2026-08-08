"""
optimize —— 03 优化层（对齐聚宽 AI 投研流程）。

组合优化（portfolio.optimize_weights，约束优化 TODO）
→ 风险归因（risk.risk_attribution：α/β + 基准对照 + Brinson）
→ 持续监控（monitor.monitor_report：IC 漂移 / 衰减 / 自相关）。

现状：简单加权 + 约束增强（行业中性 / 权重上下限 / 换手约束，启发式投影）
已就绪；均值方差 / 风险平价等求解器优化与组合级风险拆解、预警自动化
为待建缺口（见各模块 TODO）。
"""
from optimize.monitor import monitor_report, rolling_ic
from optimize.portfolio import optimize_weights
from optimize.risk import risk_attribution

__all__ = ["optimize_weights", "risk_attribution", "rolling_ic", "monitor_report"]
