"""持续监控 —— 兼容转出口。

真源已下沉至 ``stats/monitor.py``（2026-08-29，解 research↔optimize 包级循环：
research/factor_library 此前懒导入本模块）。本文件仅为历史 import 路径保留
re-export，新代码请直接 ``from stats.monitor import ...``。
"""
from stats.monitor import monitor_ic_series, monitor_report, rolling_ic  # noqa: F401

__all__ = ["rolling_ic", "monitor_ic_series", "monitor_report"]
