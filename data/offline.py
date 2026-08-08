"""
离线只读数据源桩
================

用于"数据全在本地 Parquet 缓存"的只读场景：select_stocks / synthesize /
build_intraday_factors / build_technical_factors / build_fundamental_factors /
intraday_analysis / walk_forward / backtest_two_periods 等脚本。

设计依据：DataCache 的"覆盖短路"机制——缓存已覆盖请求区间时不会回调数据源
（见 data/cache._refresh_long_table 的 covered 判断）。因此离线桩无需真正实现
数据逻辑：任何数据源方法被意外调用（缓存缺失）时立即抛 RuntimeError 暴露，
避免静默返回空数据污染结果。

覆盖 DataSource 的全部 17 个抽象方法（2026-08-05 统一）。此前 8 个脚本各自
内联了覆盖不一致的桩（11~17 个方法不等），漏掉的方法被调用时抛 AttributeError
而非友好报错，且 DataSource 新增接口时无法同步。
"""
from __future__ import annotations

from data.datasource import DataSource  # noqa: F401  （仅用于文档对齐/类型标注）


class OfflineDataSource:
    """离线模式数据源桩：任何数据源方法被调用都抛 RuntimeError。

    用法:
        from data.offline import OfflineDataSource
        cache = DataCache(OfflineDataSource(), cache_root=...)
    """

    _DEFAULT_MSG = "offline 模式不连接数据源：请先运行 scripts.update_data 拉取缓存"

    def __init__(self, message: str | None = None):
        self._msg = message or self._DEFAULT_MSG

    def _raise(self, *a, **k):
        raise RuntimeError(self._msg)

    # ---- 覆盖 DataSource 全部 17 个抽象方法 ----
    get_calendar = get_code_list = get_index_constituent = _raise
    get_daily_kline = get_minute_kline = get_adj_factor = get_backward_factor = _raise
    get_code_info = get_history_stock_status = get_industry_classification = _raise
    get_equity_structure = get_dividend = get_share_holder = get_holder_num = _raise
    get_balance_sheet = get_cash_flow = get_income = _raise
