"""
报告共享的格式化规则（xlsx / html 报告共用）
============================================

2026-08-05 抽取：两个报告模块（research/xlsx_report.py 与
research/html_report.py）各自维护了重复的"月度复利计算"与"正负着色规则"。
其中月度计算曾因双份实现出现双重加 1 bug（一个月收益 ≈ 2^n，Excel 与 HTML
同时中招）——收敛到此处后，只有一份实现，不会再"一处改另一处漏"。

注意：**两报告的可视化习惯不同，不强行合并展示格式化**：
- xlsx 指标单元格用数值小数（0.0800），html 指标卡用百分比（8.00%）
- 因此仅共享：月度收益序列计算 + 正负着色规则（红涨绿跌）
"""
from __future__ import annotations

import pandas as pd


# 收益/绩效类指标：正涨红、负跌绿（A 股习惯）
POS_NEG_KEYS = frozenset({
    "annual_return", "total_return", "sharpe", "sortino", "calmar",
    "excess_return", "information_ratio", "ir", "ic_mean",
    "avg_daily_return", "profit_loss_ratio",
})


def is_pos_neg(key: str) -> bool:
    """该指标是否按正负着色。"""
    return key in POS_NEG_KEYS


def monthly_returns(daily_returns: pd.Series) -> pd.Series:
    """日收益 → 月度复利收益序列（唯一实现）。

    历史教训：早期两处报告各自实现，写成
    ``(1 + daily_returns).resample("ME").apply(lambda x: (1 + x).prod() - 1)``
    双重加 1，一个月收益 ≈ (2+日收益)^21 ≈ 2^n 天文数字（+838860700%），
    Excel 月度热力图与 HTML 月度表同时中招（2026-08-05 修复并收敛到此）。
    """
    if daily_returns is None or len(daily_returns) == 0:
        return pd.Series(dtype=float)
    return daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
