"""研究层入口。"""
from research.factor_analysis import (
    calc_ic_decay,
    calc_ic_series,
    calc_ir,
    factor_summary,
    quantile_backtest,
)
from research.xlsx_report import generate_excel_report

__all__ = [
    "calc_ic_series", "calc_ir", "calc_ic_decay",
    "quantile_backtest", "factor_summary",
    "generate_excel_report",
]
