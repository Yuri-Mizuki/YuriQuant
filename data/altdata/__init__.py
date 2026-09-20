"""另类数据域（altdata）—— 事件/资金/注意力类非结构化与半结构化数据。

与既有 ``data/textmining/``（文本域）并列：textmining 管研报/公告正文，
altdata 管**结构化事件表**（增减持 / 内部人交易 / 实控人变动 / 宏观日历）。

分层（与 textmining 同构）：
    source_akshare.py   L1 传输：akshare 现成封装（免自建四接口）
    fetch.py            L2 结构化：``AltDataCache`` parquet 落盘 + 增量 + 统一入口

设计原则（2026-09-20 计划定稿）：
    - **能一行接口拿到的绝不自己抓** —— P0 四接口零抓取器、零 enckey、零分页；
    - **不为 akshare 再包一层万能 wrapper** —— 只在「列名标准化 + 落盘」处薄薄一层，
      额外代码仅用于 PIT 标注与增量水位；
    - **无事件记 NaN，不填 0**（alt 惯例）；
    - 日期统一 ``datetime64``；``ann_date`` = 数据可得时点（PIT 唯一合法对齐轴）。

⚠️ 关键口径：四个接口的 PIT 质量**不同**，详见 ``source_akshare.PIT_QUALITY``。
"""
from data.altdata.fetch import AltDataCache, fetch_altdata

__all__ = ["AltDataCache", "fetch_altdata"]
