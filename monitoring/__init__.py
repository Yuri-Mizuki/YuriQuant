"""生产化性能监控 —— 因子与模型预测的统一监控层。

监控对象：因子库全部因子（含 ``model:*`` 模型预测因子）。指标口径与
``reports/ml_synthesis_h1`` 实验评估一致（rank IC / NW-t / 分位多空），
并补充生产视角的近期窗口指标：

- ic 漂移：全期 vs 近期 window 交易日（复用库内预存 evals ic 序列，秒级）
- 覆盖率 / 数据新鲜度：面板非 NaN 比例、落后数据源交易日数
- 分位单调性：近期窗口 Q1~Q5 日均收益与组序号的 Spearman 相关
  （模型因子结构恶化先于 IC 归零 —— 实验报告 8.3 的退出预警信号）
- 模型基线对比：model:* 因子以注册时（上线）的 ic_mean 为期望基线

子模块：
- metrics  指标计算（纯函数，可单测）
- alerts   阈值告警规则引擎（阈值来自 config monitoring 段）
- ledger   快照 / 告警账本（幂等 append，同 run_date 重跑覆盖）
- runner   编排（扫描→计算→告警→落盘→HTML 报告）与调度时间函数
"""

from monitoring.alerts import evaluate_alerts, rollup_status
from monitoring.ledger import MonitoringLedger
from monitoring.metrics import (
    MonitorMetrics,
    compute_factor_metrics,
    load_close_panel,
    load_returns_panel,
)
from monitoring.runner import generate_html_report, next_run_time, run_monitoring

__all__ = [
    "MonitorMetrics",
    "MonitoringLedger",
    "compute_factor_metrics",
    "evaluate_alerts",
    "rollup_status",
    "generate_html_report",
    "load_close_panel",
    "load_returns_panel",
    "next_run_time",
    "run_monitoring",
]
