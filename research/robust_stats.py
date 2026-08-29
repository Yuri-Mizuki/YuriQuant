"""稳健统计推断 —— 兼容转出口。

真源已下沉至 ``stats/robust_stats.py``（2026-08-29，解 factor/model/optimize
对 research 的反向依赖）。本文件仅为历史 import 路径保留 re-export，
新代码请直接 ``from stats.robust_stats import ...``。
"""
from stats.robust_stats import auto_lag, nw_tstat, ols_newey_west  # noqa: F401

__all__ = ["auto_lag", "nw_tstat", "ols_newey_west"]
