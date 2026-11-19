"""全A 数据集构建与回补管线（2026-09-11 迁入，纳入版本控制）。

⚠️ 为什么必须被跟踪：本包的模块是**可重跑的生产/实验管线**（一次性探针一律不留仓库）——
- `alla_daily_rank`（生产每日推理，注册为 Windows 计划任务）直接 import 本包；
- 它们产出受跟踪数据集 `all_a_2018_2026` 的 panels 与其上游 parquet。
放在仓库外会导致**干净 clone 上生产入口 ImportError**。

布局：`build_alla_*` = 因子面板构建器；`backfill_*`/`fetch_*`/`refetch_*` = 数据回补；
`run_evt_margin_build` = 回补+构建编排器。
"""
