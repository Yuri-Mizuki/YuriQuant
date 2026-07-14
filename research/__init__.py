"""研究层入口。"""
from research.factor_analysis import (
    calc_ic_decay,
    calc_ic_series,
    calc_ir,
    factor_summary,
    quantile_backtest,
)
from research.report import generate_report, plot_drawdown, plot_equity_curve, plot_ic_series, plot_layer_nav

__all__ = [
    "calc_ic_series", "calc_ir", "calc_ic_decay",
    "quantile_backtest", "factor_summary",
    "generate_report", "plot_equity_curve", "plot_drawdown",
    "plot_ic_series", "plot_layer_nav",
]
