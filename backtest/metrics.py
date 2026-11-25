"""
绩效评估指标
============

计算回测结果的常用绩效指标（含中英文标签）:
- 年化收益率 Annual Return
- 累计收益 Total Return
- 年化波动率 Annual Volatility
- 夏普比率 Sharpe Ratio
- 索提诺比率 Sortino Ratio
- 最大回撤 Max Drawdown
- 卡玛比率 Calmar Ratio
- 胜率 Win Rate
- 盈亏比 Profit/Loss Ratio
- 平均日收益 Avg Daily Return
- 平均换手率 Avg Turnover
- 信息比率 Information Ratio（需基准）
- 超额年化 Excess Return（需基准）
- 夏普 t 统计量 Sharpe t-stat（Newey-West，H0: 日均值=0）
- 算术夏普 Sharpe (arithmetic) = mean/std·√N —— **换算显著性专用**
- 证明所需年数 Years to Prove = (1.96/|SR|)²（几何 SR，偏保守）
- 证明所需年数(算术) Years to Prove (arith) = (1.96/|SR_arith|)²（与 t 同口径）
- 超额 t 统计量 Excess t-stat（需基准，H0: 超额均值=0）

所有函数接收日频收益率 Series(index=date)。
"""
from __future__ import annotations

import pandas as pd

# 年化交易日数（单一真源在 stats/ 包；此处 re-export 保持
# ``from backtest.metrics import PERIODS_PER_YEAR`` 的历史路径可用）。
# 全库所有年化口径统一引用该常量，禁止各处再写 244/252 字面量
# （2026-08-29 收敛：e2e_backtest.perf_stats 曾用 244，与引擎 252 分裂
# 导致两族报告差 ~3%）。
from stats import PERIODS_PER_YEAR  # noqa: F401
from stats.perf import (  # noqa: F401  (纯计算真源在 stats.perf，此处 re-export)
    annual_return,
    annual_return_nan,
    annual_volatility,
    calmar_ratio,
    drawdown_series,
    information_ratio,
    max_drawdown,
    profit_loss_ratio,
    sharpe_arithmetic,
    sharpe_arithmetic_nan,
    sharpe_ratio,
    sharpe_ratio_nan,
    sortino_ratio,
    turnover_rate,
    win_rate,
    years_to_prove_from_sharpe,
)
from stats.robust_stats import nw_tstat


def calc_short_metrics(
    weights_history: pd.DataFrame,
    borrow_fee_series: pd.Series,
    initial_capital: float = 1.0,
    margin_ratio: float = 1.0,
    n_days: int = 0,
    periods_per_year: int = PERIODS_PER_YEAR,
    exposure: pd.DataFrame | None = None,
) -> dict:
    """空头成本与资金占用指标（配合 ShortCostModel 使用）。

    Args:
        weights_history: 权重历史（date × code），多头 > 0、空头 < 0。
        borrow_fee_series: 每日借券费（元）。
        initial_capital: 初始资金（元），用于把费用折算为年化拖累。
        margin_ratio: 融券保证金比例（与回测引擎一致）。
        n_days: 交易日数（用于年化拖累；缺省取 borrow_fee_series 长度）。
        periods_per_year: 年化周期数。
        exposure: 每日实际敞口（DataFrame，列 long/short/margin，index=date）。
                  由回测引擎在 run() 内按日记录，**比 weights_history（仅调仓日有
                  记录）更准确地反映全年真实资金占用**；缺省时回退 weights_history。
    """
    if exposure is not None and len(exposure) > 0:
        long_exp = exposure["long"].astype(float)
        short_exp = exposure["short"].astype(float)
        margin = exposure["margin"].astype(float)
    else:
        if weights_history is None or len(weights_history) == 0:
            return {}
        w = weights_history.fillna(0.0)
        long_exp = w.clip(lower=0).sum(axis=1)
        short_exp = (-w.clip(upper=0)).sum(axis=1)
        margin = long_exp + short_exp * margin_ratio
    fee = borrow_fee_series.reindex(long_exp.index).fillna(0.0)
    n = n_days or len(long_exp)
    n_years = n / periods_per_year
    return {
        "avg_long_exposure": float(long_exp.mean()),
        "avg_short_exposure": float(short_exp.mean()),
        "avg_margin_usage": float(margin.mean()),
        "max_margin_usage": float(margin.max()),
        "borrow_fee_total": float(fee.sum()),
        "borrow_fee_drag_annual": (
            float(fee.sum() / initial_capital / n_years)
            if n_years > 0 and initial_capital > 0 else 0.0
        ),
    }


# ===========================================================================
# 指标中英文标签
# ===========================================================================
METRIC_LABELS = {
    "annual_return":       "年化收益率 Annual Return",
    "total_return":        "累计收益 Total Return",
    "annual_volatility":   "年化波动率 Annual Volatility",
    "sharpe":              "夏普比率 Sharpe Ratio",
    "sharpe_arith":        "算术夏普 Sharpe (arithmetic, mean/std·√N)",
    "sharpe_t_stat":       "夏普 t 统计量 Sharpe t-stat",
    "years_to_prove":      "证明所需年数(1.96/|SR|)² Years to Prove",
    "years_to_prove_arith": "证明所需年数(算术SR) Years to Prove (arith SR)",
    "excess_t_stat":       "超额 t 统计量 Excess t-stat",
    "sortino":             "索提诺比率 Sortino Ratio",
    "max_drawdown":        "最大回撤 Max Drawdown",
    "calmar":              "卡玛比率 Calmar Ratio",
    "win_rate":            "胜率 Win Rate",
    "profit_loss_ratio":   "盈亏比 Profit/Loss Ratio",
    "avg_daily_return":    "平均日收益 Avg Daily Return",
    "avg_turnover":        "平均换手率 Avg Turnover",
    "n_days":              "交易日数 Trading Days",
    "benchmark_annual_return": "基准年化 Benchmark Annual Return",
    "excess_return":       "超额年化 Excess Return",
    "information_ratio":   "信息比率 Information Ratio",
    "ic_mean":             "IC均值 IC Mean",
    "ic_std":              "IC标准差 IC Std",
    "ic_win_rate":         "IC胜率 IC Win Rate",
    "ir":                  "信息比率 IR (IC-based)",
    "avg_long_exposure":   "平均多头敞口 Avg Long Exposure",
    "avg_short_exposure":  "平均空头敞口 Avg Short Exposure",
    "avg_margin_usage":    "平均保证金占用 Avg Margin Usage",
    "max_margin_usage":    "最大保证金占用 Max Margin Usage",
    "borrow_fee_total":    "借券费总额 Borrow Fee Total (CNY)",
    "borrow_fee_drag_annual": "借券费年化拖累 Borrow Fee Drag (Ann.)",
}


def calc_all_metrics(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    weights_history: pd.DataFrame | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
    rf: float = 0.0,
) -> dict:
    """一次性计算所有指标。

    Args:
        daily_returns: 日收益序列。
        benchmark_returns: 基准日收益（可选，计算 IR / 超额收益）。
        weights_history: 权重历史（可选，计算平均换手率）。
        periods_per_year: 年化周期数（默认 PERIODS_PER_YEAR 交易日）。
        rf: 年化无风险利率（默认 0），用于 Sharpe / Sortino 的超额收益。
    """
    sharpe = sharpe_ratio(daily_returns, rf=rf, periods_per_year=periods_per_year)
    # 算术 Sharpe 只用于换算显著性（见 sharpe_arithmetic 的 docstring）：
    # 2026-10-28 起与几何版并列登记，避免「拿几何 SR 套 t=SR·√T」的口径错配。
    sharpe_arith = sharpe_arithmetic(daily_returns, rf=rf,
                                     periods_per_year=periods_per_year)
    m = {
        "annual_return": annual_return(daily_returns, periods_per_year),
        "total_return": (1 + daily_returns).prod() - 1,
        "annual_volatility": annual_volatility(daily_returns, periods_per_year),
        "sharpe": sharpe,
        "sharpe_arith": sharpe_arith,
        "sortino": sortino_ratio(daily_returns, rf=rf, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(daily_returns),
        "calmar": calmar_ratio(daily_returns, periods_per_year),
        "win_rate": win_rate(daily_returns),
        "profit_loss_ratio": profit_loss_ratio(daily_returns),
        "avg_daily_return": daily_returns.mean(),
        # 日收益均值的 Newey-West t 检验（H0: 均值=0）。i.i.d. 下与
        # t = SR*sqrt(年数) 恒等；自相关极弱的日收益上两者几乎一致，
        # 低频重采（周/月调仓）序列 NW 校正才有实质差异。
        "sharpe_t_stat": nw_tstat(daily_returns.dropna())[0],
        # 证明策略所需年数 (1.96/|SR|)^2（paperswithbacktest 复现口径）：
        # t = SR*sqrt(T) 达到 1.96 所需的样本年限。
        # ⚠️ 这里传的是**几何** SR ⇒ 与 t 统计量口径错配（非退化区间年数偏保守、
        # 近零区间虚短，详见 sharpe_arithmetic）；严格同口径的是下面那行（算术 SR），
        # 两者并列登记、不要混用。
        "years_to_prove": years_to_prove_from_sharpe(sharpe),
        "years_to_prove_arith": years_to_prove_from_sharpe(sharpe_arith),
        "n_days": len(daily_returns),
    }
    if benchmark_returns is not None:
        aligned = daily_returns.align(benchmark_returns, join="inner")[0]
        bench_aligned = daily_returns.align(benchmark_returns, join="inner")[1]
        m["benchmark_annual_return"] = annual_return(bench_aligned, periods_per_year)
        m["information_ratio"] = information_ratio(aligned, bench_aligned, periods_per_year)
        m["excess_return"] = m["annual_return"] - m["benchmark_annual_return"]
        # 超额收益是否显著非零的直接检验（主动管理是否"真实存在"）。
        m["excess_t_stat"] = nw_tstat((aligned - bench_aligned).dropna())[0]
    if weights_history is not None:
        m["avg_turnover"] = turnover_rate(weights_history)
    return m


def format_metrics(metrics: dict) -> str:
    """格式化为可读字符串（中英文标签）。"""
    lines = []
    for k, v in metrics.items():
        label = METRIC_LABELS.get(k, k)
        if isinstance(v, float):
            if k in ("win_rate", "ic_win_rate"):
                lines.append(f"  {label:<35s} {v:>10.2%}")
            elif abs(v) < 100:
                lines.append(f"  {label:<35s} {v:>10.4f}")
            else:
                lines.append(f"  {label:<35s} {v:>10.0f}")
        else:
            lines.append(f"  {label:<35s} {v}")
    return "\n".join(lines)
