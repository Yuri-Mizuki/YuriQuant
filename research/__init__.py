"""研究层入口。"""
from research.factor_analysis import (
    calc_ic_decay,
    calc_ic_series,
    calc_ir,
    factor_summary,
    quantile_backtest,
)
from research.report import (
    generate_comparison_report,
    generate_report,
    generate_single_report,
    plot_drawdown,
    plot_equity_curve,
    plot_ic_series,
    plot_layer_nav,
)
from research.xlsx_report import generate_excel_report

__all__ = [
    "calc_ic_series", "calc_ir", "calc_ic_decay",
    "quantile_backtest", "factor_summary",
    "generate_single_report", "generate_comparison_report", "generate_report",
    "generate_excel_report",
    "plot_equity_curve", "plot_drawdown",
    "plot_ic_series", "plot_layer_nav",
]
