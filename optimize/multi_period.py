"""
多期组合执行适配器 —— 03 优化层「完整多期执行」（P3，2026-08-20）。

把已有的**单期精确优化**（solver.optimize_weights_qp：滚动 Ledoit-Wolf Σ、
跨期 prev_weights 换手、行业/风格中性、BL/多空、A-C 成本惩罚）与**向量化回测**
（backtest.VectorBacktest：成本/借券费/净值/换手指标）拼成逐调仓日回放的
多期执行闭环，口径与单因子回测完全一致：

  1. 决策：仅在调仓日（D/W/M）重解 QP。不可交易标的（涨跌停/停牌掩码）在
     该日 alpha 置 NaN → QP 强制 w=0，不浪费权重；prev_weights 取上一调仓日
     的实际持仓，换手/成本显式进约束或目标。
  2. 持有：调仓日之间目标权重不变（无信号沿用持仓），由回测引擎按日持仓记账。
  3. 评估：复用 VectorBacktest → BacktestResult（含日收益/净值/换手/成本/借券费），
     run() 后可用 .summary() 出与单因子一致的绩效指标。

设计要点（业界"滚动再优化 + 换手/成本抑制"baseline 的落地形态）：
- 不引入多期随机优化（MPC）——单期重解 + turnover penalty 已逼近其效果，
  且实现/调试/解释成本低得多（见 solver.solve_portfolio 的 A-C 惩罚）。
- 决策-执行解耦预留：本模块只产出目标权重，执行/对账层（接券商）后续叠加在
  BacktestResult.weights_history 之上即可，无需改造内部。

用法：
    from optimize.multi_period import RebalanceConfig, run_multi_period_backtest
    cfg = RebalanceConfig(rebalance_freq="M", method="mvo")
    result, target = run_multi_period_backtest(factor, returns, cfg=cfg)
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from backtest.engine import VectorBacktest
from optimize.solver import optimize_weights_qp
from strategy.base import Strategy

__all__ = ["RebalanceConfig", "optimize_rebalance_weights", "run_multi_period_backtest"]


@dataclass
class RebalanceConfig:
    """多期执行的调仓频率与单期 QP 优化参数（透传给 optimize_weights_qp）。"""

    rebalance_freq: str = "M"          # D / W / M
    method: str = "mvo"                # min_var | tev | mvo | risk_parity | bl
    risk_aversion: float = 1.0
    window: int = 120                  # 滚动协方差窗口（交易日）
    min_periods: int = 60              # 窗口最小样本，不足该期空仓
    cov_method: str = "ledoit_wolf"
    shrinkage: float = 0.5
    max_weight: float | None = 0.1
    min_weight: float | None = None
    max_turnover: float | None = 0.5   # 单边换手硬约束
    turnover_penalty: float = 0.0      # 线性冲击成本系数（进目标）
    quadratic_cost: float = 0.0        # 二次冲击成本系数（A-C）
    budget: float = 1.0
    allow_short: bool = False
    short_limit: float | None = None
    gross_limit: float | None = None
    benchmark_weights: pd.Series | None = None
    industry_map: Any | None = None
    industry_target: Any | None = None
    industry_deviation: float | None = None
    industry_panel: pd.DataFrame | None = None
    style_exposures: dict[str, pd.DataFrame] | None = None
    style_tolerance: float = 1e-5
    views: dict | None = None
    market_weights: pd.Series | None = None
    tau: float = 0.05
    delta: float = 2.5

    _QP_FIELDS = {
        "method", "risk_aversion", "window", "min_periods", "cov_method", "shrinkage",
        "max_weight", "min_weight", "max_turnover", "turnover_penalty", "quadratic_cost",
        "budget", "allow_short", "short_limit", "gross_limit", "benchmark_weights",
        "industry_map", "industry_target", "industry_deviation", "industry_panel",
        "style_exposures", "style_tolerance", "views", "market_weights", "tau", "delta",
    }

    def qp_kwargs(self) -> dict[str, Any]:
        """透传给 optimize_weights_qp 的关键字参数（去掉值为 None 的可选项）。"""
        d = asdict(self)
        return {k: v for k, v in d.items() if k in self._QP_FIELDS and v is not None}


def _rebalance_days(dates: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    """调仓日集合（与 VectorBacktest._get_rebalance_days 同口径，保证两处对齐）。"""
    if freq == "D":
        return list(dates)
    period = "W" if freq == "W" else "M"
    s = pd.Series(dates, index=dates)
    return list(s.groupby(s.index.to_period(period)).first())


def optimize_rebalance_weights(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    cfg: RebalanceConfig,
    executable: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """仅在调仓日求解目标权重（date×code），供回放引擎逐期回放。

    Args:
        factor: date×code 因子值（方向已对齐，越大越看好）。
        returns: date×code 日收益（估计协方差；Σ 只用 < 调仓日的收益，防前视）。
        cfg: RebalanceConfig（调仓频率 + QP 参数）。
        executable: date×code 布尔可执行掩码（涨跌停/停牌）。提供时把不可交易
            股票的该日 alpha 置 NaN → QP 强约束 w=0，避免把权重浪费给不可执行标的。
    Returns:
        DataFrame(index=调仓日, columns=code) 目标权重；协方差窗口不足的调仓日全 0。
    """
    common = factor.index.intersection(returns.index)
    f = factor.loc[common]
    reb_days = _rebalance_days(common, cfg.rebalance_freq)
    f_reb = f.loc[reb_days]

    if executable is not None:
        ex = executable.reindex(index=f_reb.index).astype(bool)
        f_reb = f_reb.where(ex)

    return optimize_weights_qp(f_reb, returns, **cfg.qp_kwargs())


class PrecomputedWeightsStrategy(Strategy):
    """按调仓日回放预计算的目标权重；无该日记录则沿用最近一期（持仓）。"""

    name = "multi_period_executor"

    def __init__(self, target: pd.DataFrame):
        self.target = target

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        idx = factor_values.index
        if not len(self.target):
            return pd.Series(0.0, index=idx, dtype=float)
        if factor_values.name in self.target.index:
            row = self.target.loc[factor_values.name]
        else:
            row = self.target.iloc[-1]
        return row.reindex(idx).fillna(0.0).astype(float)


def run_multi_period_backtest(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    cfg: RebalanceConfig,
    executable: pd.DataFrame | None = None,
    initial_capital: float | None = None,
    costs=None,
    short_costs=None,
    deleverage: bool = False,
    backtest_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, pd.DataFrame]:
    """多期组合执行的整体入口：优化目标权重 → 喂回测引擎记账评效。

    Returns:
        (backtest.VectorBacktest 的结果对象, 目标权重 DataFrame(date×code))。
        结果对象含 daily_returns / weights_history / equity_curve /
        turnover_series / cost_series / borrow_fee_series，可 .summary() 出绩效。
    """
    target = optimize_rebalance_weights(factor, returns, cfg, executable)
    strat = PrecomputedWeightsStrategy(target)

    if initial_capital is None:
        from config import Config
        btc = Config.get().get("backtest", {})
        initial_capital = float(btc.get("initial_capital", 1_000_000))

    bt_kw = dict(backtest_kwargs or {})
    bt = VectorBacktest(
        strat,
        rebalance_freq=cfg.rebalance_freq,
        initial_capital=float(initial_capital),
        costs=costs,
        short_costs=short_costs,
        deleverage=deleverage,
        **bt_kw,
    )
    result = bt.run(factor, returns, executable_mask=executable)
    result.target_weights = target  # type: ignore[attr-defined]
    return result, target