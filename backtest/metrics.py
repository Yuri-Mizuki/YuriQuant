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

import numpy as np
import pandas as pd

# 年化交易日数（单一真源在 stats/ 包；此处 re-export 保持
# ``from backtest.metrics import PERIODS_PER_YEAR`` 的历史路径可用）。
# 全库所有年化口径统一引用该常量，禁止各处再写 244/252 字面量
# （2026-08-29 收敛：e2e_backtest.perf_stats 曾用 244，与引擎 252 分裂
# 导致两族报告差 ~3%）。
from stats import PERIODS_PER_YEAR  # noqa: F401
from stats.robust_stats import nw_tstat


def annual_return(daily_returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """年化收益率 Annual Return。"""
    total = (1 + daily_returns).prod() - 1
    n_years = len(daily_returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return (1 + total) ** (1 / n_years) - 1


def annual_volatility(daily_returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """年化波动率 Annual Volatility。"""
    return daily_returns.std() * np.sqrt(periods_per_year)


def _rf_daily(rf: float, periods_per_year: int) -> float:
    """年化 rf → 日频 rf（复利折算），用于超额收益序列。"""
    if rf <= 0:
        return 0.0
    return (1 + rf) ** (1 / periods_per_year) - 1


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """夏普比率 Sharpe Ratio。

    rf 按日频复利折算为超额收益序列（(1+r)/(1+rf_daily)-1），再用几何年化口径：
    Sharpe = 年化超额收益 / 年化波动率。rf=0 时与经典实现完全一致。
    """
    rfd = _rf_daily(rf, periods_per_year)
    excess = (1 + daily_returns) / (1 + rfd) - 1
    vol = annual_volatility(daily_returns, periods_per_year)
    if vol == 0:
        return 0.0
    return annual_return(excess, periods_per_year) / vol


def sharpe_arithmetic(daily_returns: pd.Series, rf: float = 0.0,
                      periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """**算术**夏普比率 ``mean(excess)/std(excess)·√N``。

    与 :func:`sharpe_ratio`（几何年化收益 / 年化波动）并列存在，**用途不同**：

    - ``sharpe_ratio``（几何）→ **报数**（生产定版口径，`caliber` 的 ``sharpe``）；
    - 本函数（算术）→ **换算显著性**。文献的 ``t = SR·√T`` 恒等式只认算术口径
      （因为 ``t = mean/std·√n`` ≡ ``SR_arith·√(n/252)``）；用几何 SR 套公式会
      **错配**。2026-10-28 实测 ``v1w-ortho882``：几何 0.589 / 算术 0.675，
      ``years_to_prove`` 虚高 31%。

    ⚠️ **几何年化收益 ≤ 算术年化收益 恒成立**（波动拖累 ``−b`` 压过 ``exp`` 凸性），
    其中 ``b = 年化方差/2``、``a = 年化算术收益``。⇒ 偏差方向由**收益符号**决定：

    - ``a > 0``（策略赚钱，本项目主流）：几何 SR 偏小 ⇒ ``|几何| < |算术|``
      ⇒ 几何口径年数**偏保守（高估）**。实测 ``v1w-ortho882``（vol 24.8% /
      算术 SR 0.675，``b/a ≈ 0.18``）：几何 0.589 vs 算术 0.675，年数虚高 31%。
    - ``a < 0``（策略亏钱）：几何更负 ⇒ ``|几何| > |算术|`` ⇒ 几何口径年数**虚短**
      （实测日均 −0.066% / 日波 1.56%：几何 −0.717 vs 算术 −0.665）。
    - ``a ≈ 0``（退化）：``b`` 单独主导 ⇒ 几何年化 ≈ ``−b`` 明显为负而算术≈0
      ⇒ ``|几何| ≫ |算术|``、年数虚短（天文数 → 几百年），此区间无解释力。

    ⇒ **判据：正收益用几何版是保守的，负收益/近零时必须用算术版**；
    跨策略比较一律用算术版，否则正负两组的「证明年数」不可比。

    **引用任何外部显著性公式前，先确认它要的是哪种 SR。**
    """
    rfd = _rf_daily(rf, periods_per_year)
    excess = (1 + daily_returns.dropna()) / (1 + rfd) - 1
    sd = float(excess.std())
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def years_to_prove_from_sharpe(sr: float, z: float = 1.96) -> float:
    """``(z/|SR|)²`` —— 让 ``t = SR·√T`` 达到 ``z`` 所需的样本年数。

    ``SR=0`` → ``inf``。**必须在算术 SR 上传入才与 t 统计量同口径**；
    传几何 SR 得到的是**错配**值（正收益时高估年数、负收益/近零时低估，
    见 :func:`sharpe_arithmetic`）。
    """
    return round((z / abs(sr)) ** 2, 1) if sr else float("inf")


def sortino_ratio(daily_returns: pd.Series, rf: float = 0.0, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """索提诺比率 Sortino Ratio: 只用下行波动。

    下行偏差 = 半方差 downside deviation = sqrt(mean(min(excess, 0)²))，
    衡量亏损深度（相对目标收益 0）；早期实现误用"负收益日样本的 std"，
    度量的是亏损日之间的离散度，系统性高估 |Sortino|（实证 ~30%）。
    rf 按日频折算为超额序列后再计算（与 Sharpe 同口径）。
    """
    rfd = _rf_daily(rf, periods_per_year)
    excess = (1 + daily_returns) / (1 + rfd) - 1
    ar = annual_return(excess, periods_per_year)
    dd = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    if dd == 0:
        return 0.0
    downside_vol = dd * np.sqrt(periods_per_year)
    return ar / downside_vol


def max_drawdown(daily_returns: pd.Series) -> float:
    """最大回撤 Max Drawdown（正值）。"""
    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return abs(dd.min())


def calmar_ratio(daily_returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """卡玛比率 Calmar Ratio: 年化收益 / 最大回撤。"""
    mdd = max_drawdown(daily_returns)
    if mdd == 0:
        return 0.0
    return annual_return(daily_returns, periods_per_year) / mdd


def information_ratio(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """信息比率 Information Ratio。"""
    excess = daily_returns - benchmark_returns
    te = excess.std() * np.sqrt(periods_per_year)
    if te == 0:
        return 0.0
    return excess.mean() * periods_per_year / te


def win_rate(daily_returns: pd.Series) -> float:
    """胜率 Win Rate。"""
    if len(daily_returns) == 0:
        return 0.0
    return (daily_returns > 0).sum() / len(daily_returns)


def profit_loss_ratio(daily_returns: pd.Series) -> float:
    """盈亏比 Profit/Loss Ratio: 平均盈利 / 平均亏损。"""
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    if len(losses) == 0 or losses.mean() == 0:
        return 0.0
    return wins.mean() / abs(losses.mean())


def turnover_rate(weights_history: pd.DataFrame) -> float:
    """平均换手率 Avg Turnover。"""
    if len(weights_history) < 2:
        return 0.0
    diffs = weights_history.diff().abs().sum(axis=1) / 2
    return diffs.mean()


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
