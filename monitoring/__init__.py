"""因子与模型预测的健康度监控 —— 面向定版因子库（默认 ``all_a_2018_2026``）。

监控对象：该库 registry 的全部因子，含 ``model:*`` 模型预测因子（模型预测
回写因子库后自动纳管）。它为「已在跑的东西有没有退化」服务，与「哪个因子
更强」的因子看板是两件事：看板是横截面比较、离线全量；本包是时间序列、
按日推进、带阈值告警。

三类指标：

- **IC 漂移**：全期 vs 近期双窗口（短窗 60d 灵敏预警、长窗 252d 稳健确认），
  附 NW-t 与显著性。IC 序列优先从因子看板缓存读（已算好、同源），
  缺失时回落库 ``evals/``——巡检只读不算，避免重跑 41.6 GiB 面板。
- **覆盖率 / 数据新鲜度**：近期面板非 NaN 比例、最后有效日落后数据源多少个
  交易日。面板按需懒加载（默认只对已告警的因子补）。
- **分位单调性**：近期 Q1~Q5 日均收益与组序号的 Spearman。分位结构恶化
  通常先于 IC 归零，是退出预警。
- **中性化 IC**：剥离风格后的纯 alpha IC。raw IC 衰减而中性化 IC 稳定 =
  风格轮动（该重新中性化），两者同时衰减 = 真 alpha 衰减（该退出）。
- **模型基线对比**：``model:*`` 因子以注册时（上线）的 OOS 基线判断保留率。

子模块：

- ``metrics``  指标计算（纯函数，可单测）
- ``alerts``   阈值告警规则引擎（阈值来自 ``config.settings.monitoring``）
- ``ledger``   快照 / 告警账本（幂等 append，同 run_date 重跑覆盖）
- ``state``    告警去抖（连续 N 期才算确认）
- ``runner``   编排（扫描 → 计算 → 告警 → 落盘 → HTML 报告）
- ``crowding`` 库级拥挤度（因子 IC 相关矩阵 → 分散度是否真实）
- ``signal_monitor`` 组合/信号层快照（新鲜度、HHI、换手、受阻比例）
- ``production_ic``  生产口径逐日 IC 台账（出榜信号本身的表现，另一条线）
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
