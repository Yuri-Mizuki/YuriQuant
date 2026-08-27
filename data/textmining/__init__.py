"""
文本挖掘数据层（Phase 0）
========================

统一入口：
    from data.textmining import fetch_docs
    docs = fetch_docs(["600519.SH"], begin_date=20190101, end_date=20261231)

主源：同花顺研报（标题+摘要全文+评级+机构+研究员，2011 年至今全历史，免费）
辅源：巨潮公告（业绩预告/定期报告/调研纪要，可选开关）

详见 reports/文本选股研报解读与数据源调研.md §八（数据源可行性实测）。
"""
from data.textmining.fetch import (
    TextMiningCache,
    fetch_docs,
    to_code_std,
)

__all__ = ["TextMiningCache", "fetch_docs", "to_code_std"]
