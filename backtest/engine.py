"""
向量化回测引擎
==============

日频向量化回测，信号→权重→收益直接矩阵运算。

流程:
1. 输入: 因子值面板(date, code) + 收益率面板(date, code)
2. 按调仓频率(rebalance_freq)取出截面
3. 每个调仓日: 用策略算权重 → 持有至下个调仓日
4. 计算组合日收益 = Σ(weight_i * return_i)
5. 减去交易成本（佣金/印花税/滑点）
6. 减去空头腿持有成本（借券费按日计提，ShortCostModel）

输出:
- daily_returns: Series(index=date), 组合日收益（净收益，含交易成本与借券费）
- weights_history: DataFrame(index=date, columns=code), 持仓权重
- equity_curve: Series(index=date), 净值曲线
- borrow_fee_series: Series(index=date), 每日借券费（元）
- margin_usage_series: Series(index=date), 每日保证金占用倍数
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.costs import ShortCostModel, TransactionCosts
from backtest.metrics import calc_all_metrics, calc_short_metrics, format_metrics
from config import Config
from strategy.base import Strategy


def _apply_executable_mask(
    weights: np.ndarray,
    executable: np.ndarray,
) -> np.ndarray:
    """将不可执行标的权重置零，并按多空组独立重新归一化。

    多头（weights > 0）和空头（weights < 0）各自独立归一化，
    保持原有资金在多空两端的分配比例。
    """
    weights = weights.copy()
    weights[~executable] = 0.0

    long_mask = weights > 0
    if long_mask.any():
        weights[long_mask] /= weights[long_mask].sum()

    short_mask = weights < 0
    if short_mask.any():
        weights[short_mask] /= abs(weights[short_mask].sum())

    return weights


@dataclass
class BacktestResult:
    """回测结果容器。"""
    daily_returns: pd.Series
    weights_history: pd.DataFrame
    equity_curve: pd.Series
    turnover_series: pd.Series
    cost_series: pd.Series
    config: dict = field(default_factory=dict)
    borrow_fee_series: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float, name="borrow_fee")
    )
    margin_usage_series: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float, name="margin_usage")
    )
    long_exposure_series: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float, name="long_exposure")
    )
    short_exposure_series: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float, name="short_exposure")
    )

    def metrics(self, benchmark_returns: pd.Series | None = None) -> dict:
        m = calc_all_metrics(
            self.daily_returns,
            benchmark_returns,
            self.weights_history,
        )
        # avg_turnover 用引擎逐次调仓记录（含换手为 0 的调仓事件），修正
        # weights_history diff 口径被非调仓日稀释的问题
        rb_turnover = self.turnover_series.dropna()
        m["avg_turnover"] = float(rb_turnover.mean()) if len(rb_turnover) else 0.0
        if len(self.borrow_fee_series) > 0:
            exposure = None
            if len(self.long_exposure_series) > 0 and len(self.margin_usage_series) > 0:
                exposure = pd.DataFrame({
                    "long": self.long_exposure_series,
                    "short": self.short_exposure_series,
                    "margin": self.margin_usage_series,
                })
            short_m = calc_short_metrics(
                self.weights_history,
                self.borrow_fee_series,
                initial_capital=float(self.config.get("initial_capital", 1.0)),
                margin_ratio=float(self.config.get("short_margin_ratio", 1.0)),
                n_days=len(self.daily_returns),
                exposure=exposure,
            )
            m.update(short_m)
        return m

    def summary(self, benchmark_returns: pd.Series | None = None) -> str:
        m = self.metrics(benchmark_returns)
        return format_metrics(m)


class VectorBacktest:
    """向量化回测引擎。"""

    def __init__(
        self,
        strategy: Strategy,
        rebalance_freq: str = "M",  # D / W / M
        initial_capital: float = 1_000_000.0,
        costs: TransactionCosts | None = None,
        short_costs: ShortCostModel | None = None,
        deleverage: bool = False,
    ):
        self.strategy = strategy
        self.rebalance_freq = rebalance_freq
        self.initial_capital = float(initial_capital)
        if costs is None:
            cfg = Config.get().get("backtest", {})
            costs = TransactionCosts(
                commission_rate=cfg.get("commission_rate", 0.0001),
                commission_min=cfg.get("commission_min", 5.0),
                stamp_duty=cfg.get("stamp_duty", 0.001),
                slippage_bp=cfg.get("slippage_bp", 5.0),
            )
        self.costs = costs
        # 空头腿成本：默认从配置读取并【启用】（修正空头腿乐观偏差）。
        # 显式传 ShortCostModel 可自定义；borrow_rate=0 等价关闭借券费。
        if short_costs is None:
            cfg = Config.get().get("backtest", {})
            short_costs = ShortCostModel(
                borrow_rate=cfg.get("short_borrow_rate", 0.08),
                margin_ratio=cfg.get("short_margin_ratio", 1.0),
            )
        self.short_costs = short_costs
        # 1 倍资金约束：总保证金需求（多头+空头×保证金比例）> 1 时按比例降杠杆。
        self.deleverage = deleverage

    def run(
        self,
        factor_panel: pd.DataFrame,
        returns_panel: pd.DataFrame,
        executable_mask: pd.DataFrame | None = None,
        horizon: int = 1,
        rebalance_days: set | None = None,
        check_convention: bool = True,
    ) -> BacktestResult:
        """执行回测。

        Args:
            factor_panel: DataFrame(index=date, columns=code), 因子值。
            returns_panel: DataFrame(index=date, columns=code), 收益率。
                          两种合法口径（结算顺序自动适配）：
                          - **未 shift**（推荐）：第 i 行是 i-1→i 的单日收益
                            （``close.pct_change()``）——先结算后调仓，
                            daily_returns 与指数 pct_change 标签严格对齐，
                            beta/alpha 不失真；
                          - **shift(-1) 前视收益贴标签**（因子库 canonical 口径）：
                            第 i 行是 i→i+1 收益——调仓日 t 权重赚 rp[t]=t→t+1，
                            同样无前视，但 daily_returns 标签比指数偏移一天，
                            与指数基准对齐算 IR/beta 时会有一天错位。用此口径须传
                            ``check_convention=False`` 显式确认，且不要与基准对齐
                            算相对指标。
            executable_mask: DataFrame(index=date, columns=code), dtype=bool，
                            True 表示当日该股票可交易。为 None 时不做过滤。
            horizon: 信号前瞻天数。>1 时启用「horizon 对齐结算」——收益面板必须传
                     forward 段累计（``close.pct_change(h).shift(-h)``，第 i 行 =
                     i→i+h），每次调仓赚取 factor[t] 对应的段收益，区间内按几何
                     平均逐日摊派复利。
            rebalance_days: 可选自定义调仓日集合(真实交易日)。传入时覆盖
                     rebalance_freq 生成的固定日历调仓日（用于自适应/事件
                     驱动调仓，如择时触发）。为 None 时维持 rebalance_freq。
            check_convention: h=1 时校验面板是否为 shift(-h) 指纹（末行全 NaN、
                     首行有值），是则报错防止一天错位。因子库等内部明确使用
                     shift 口径且不对齐基准算指标的场景可显式关掉。
        Returns:
            BacktestResult

        Raises:
            ValueError: h=1、check_convention=True 且收益面板疑似被 shift(-h)
                        前视（与引擎对齐口径错位一天）。
        """
        # ---- 口径守卫（BUG-3 修复）：h=1 默认禁止 shift 前视面板 ----
        if horizon == 1 and check_convention and not returns_panel.empty:
            first_row_nonnull = returns_panel.iloc[0].notna()
            last_row_nonnull = returns_panel.iloc[-1].notna()
            # shift(-h) 指纹：最后一行几乎全 NaN + 首行几乎全有值；
            # 未 shift 面板相反（首行 pct_change 为 NaN）。两者皆非则放行
            # （如已裁剪过边界的面板，无法判定，不做武断拦截）。
            if last_row_nonnull.mean() < 0.05 and first_row_nonnull.mean() > 0.95:
                raise ValueError(
                    "returns_panel 疑似为 shift(-h) 后的收益面板（末行全 NaN、"
                    "首行有值），与本引擎的对齐口径（第 i 行=i-1→i 收益）错位一天"
                    "——beta/alpha/信息比对基准时会失真。请改传 close.pct_change() "
                    "原始面板；若确要使用 shift 贴标签口径（因子库 canonical 风格），"
                    "传 check_convention=False 显式声明，且勿与指数基准对齐算指标。"
                )
        # 对齐日期和代码
        common_dates = factor_panel.index.intersection(returns_panel.index)
        common_codes = factor_panel.columns.intersection(returns_panel.columns)
        fp = factor_panel.loc[common_dates, common_codes]
        rp = returns_panel.loc[common_dates, common_codes]

        dates = fp.index
        codes = fp.columns
        n_days = len(dates)
        n_codes = len(codes)

        # 对齐 executable_mask
        if executable_mask is not None:
            executable_mask = executable_mask.reindex(
                index=common_dates, columns=common_codes
            ).fillna(True)

        # 调仓日：可用自定义调仓日覆盖固定频率（自适应/事件驱动调仓）
        rebalance_days = rebalance_days if rebalance_days is not None \
            else self._get_rebalance_days(dates)

        # 用 numpy 数组存储结果，避免 pandas 链式赋值问题
        rp_values = rp.values  # (n_days, n_codes)

        daily_ret_arr = np.zeros(n_days, dtype=np.float64)
        # 换手率仅调仓日有值（NaN 其余天）：avg_turnover 按"每次调仓"平均，
        # 不会被非调仓日的零稀释，也不会漏掉换手为 0 的调仓事件
        turnover_arr = np.full(n_days, np.nan, dtype=np.float64)
        cost_arr = np.zeros(n_days, dtype=np.float64)
        borrow_fee_arr = np.zeros(n_days, dtype=np.float64)
        margin_arr = np.zeros(n_days, dtype=np.float64)
        long_arr = np.zeros(n_days, dtype=np.float64)
        short_arr = np.zeros(n_days, dtype=np.float64)
        equity_arr = np.full(n_days, self.initial_capital, dtype=np.float64)
        weights_arr = np.zeros((n_days, n_codes), dtype=np.float64)

        current_weights = np.zeros(n_codes, dtype=np.float64)
        capital = self.initial_capital

        # horizon 对齐结算（h>1）：
        # 每个调仓区间 [rb, next_rb) 在区间首日按【当日因子决定的新权重】计算
        # 段收益 seg = Σ w·fwd_h[rb]，再按几何平均摊到区间内**每个**交易日复利，
        # 使 prod_{d∈区间}(1+日均摊) == 1+seg —— 净值逐日平滑且口径与信号前瞻一致。
        # 守卫：区间跨度必须 ≥ horizon（重叠窗口重复结算是虚高假收益，见回归测试）。
        fwd_starts: dict[int, int] = {}  # 调仓日行号 -> 下一个调仓日行号（不含时=末尾）
        if horizon > 1:
            rb_list = sorted(i for i, d in enumerate(dates) if d in rebalance_days)
            for k, t in enumerate(rb_list):
                nxt = rb_list[k + 1] if k + 1 < len(rb_list) else n_days
                fwd_starts[t] = nxt
            bad = [t for t in rb_list if (fwd_starts[t] - t) < horizon]
            if bad:
                raise ValueError(
                    f"回测口径错误: horizon={horizon} 但存在调仓区间跨度 < horizon "
                    f"(调仓频率过高于预测视野)，会把 {horizon} 日累计收益摊到更短区间"
                    f"造成重复结算的虚高收益。请提高调仓频率(如 rebalance_freq=D 时 "
                    f"用 horizon=1)，或将 rebalance_freq 调整为与 horizon 匹配的跨度。"
                )

        # 结算-调仓顺序：
        # h=1（收益面板按惯例传未 shift 的 close.pct_change()，第 i 行是 i-1→i 收益）：
        #   先结算后调仓——当日收益由上一调仓日设定的权重赚取，daily_returns[t]
        #   与指数 pct_change[t] 标签严格对齐，beta/alpha 不失真；调仓日收盘换仓，
        #   新权重次日生效。
        # h>1（收益面板传 forward_returns(close,h)，第 i 行是 i→i+h 段累计）：
        #   区间首日先用当日因子换仓定权（成本计入当日），随后整个区间按段收益的
        #   几何日均摊复利。
        active_interval_end = -1   # 当前 h>1 区间的排他终点行号
        interval_daily_equiv = 0.0
        for i in range(n_days):
            cap_before = capital

            # 1a) h=1 结算：当前权重赚当日收益。
            if horizon == 1:
                gross_ret = float(np.nansum(current_weights * rp_values[i]))
                capital *= (1 + gross_ret)

            # 1b) 调仓（h=1 在结算之后；h>1 在区间结算之前，保证段收益用新权重）
            if dates[i] in rebalance_days and i < n_days - 1:
                factor_vals = fp.iloc[i].dropna()
                if len(factor_vals) > 0:
                    new_weights = self.strategy.get_weights_at(dates[i], factor_vals)
                    new_arr = new_weights.reindex(codes).fillna(0.0).values.astype(np.float64)
                    if executable_mask is not None:
                        new_arr = _apply_executable_mask(
                            new_arr, executable_mask.iloc[i].values.astype(bool)
                        )
                    if self.deleverage:
                        long_exp = max(0.0, new_arr[new_arr > 0].sum())
                        short_exp = np.abs(new_arr[new_arr < 0]).sum()
                        mu = self.short_costs.margin_usage(long_exp, short_exp)
                        if mu > 1.0:
                            new_arr = new_arr / mu
                else:
                    new_arr = np.zeros(n_codes, dtype=np.float64)

                turnover = np.abs(new_arr - current_weights).sum() / 2
                turnover_arr[i] = turnover

                cost = self.costs.calc(current_weights, new_arr, turnover, capital)
                cost_arr[i] = cost
                capital -= cost

                current_weights = new_arr

            # 1c) h>1 区间结算：首日以新权重预计算整段均摊收益，区间内逐日复利
            if horizon > 1:
                if i in fwd_starts:
                    span = fwd_starts[i] - i
                    seg_gross = float(np.nansum(current_weights * rp_values[i]))
                    interval_daily_equiv = (
                        (1.0 + seg_gross) ** (1.0 / span) - 1.0 if span > 0 else seg_gross
                    )
                    active_interval_end = fwd_starts[i]
                    capital *= (1 + interval_daily_equiv)
                elif i < active_interval_end:
                    capital *= (1 + interval_daily_equiv)

            # 2) 记录持仓权重（每日都记，非仅调仓日——下游 metrics/监控依赖真实持仓；
            #    此前只记调仓日、其余天为 0 行，导致 avg_turnover 等指标失真）
            weights_arr[i] = current_weights

            # 3) 空头腿成本：按日计提借券费（基于当日持仓）。
            long_exp = max(0.0, current_weights[current_weights > 0].sum())
            short_exp = np.abs(current_weights[current_weights < 0].sum())
            long_arr[i] = long_exp
            short_arr[i] = short_exp
            margin_arr[i] = self.short_costs.margin_usage(long_exp, short_exp)
            fee = self.short_costs.daily_borrow_fee(short_exp, capital)
            capital -= fee
            borrow_fee_arr[i] = fee

            # 4) 净日收益，与 equity_curve 严格一致。
            daily_ret_arr[i] = (capital / cap_before - 1.0) if cap_before > 0 else 0.0
            equity_arr[i] = capital

        # 转 pandas
        daily_rets = pd.Series(daily_ret_arr, index=dates, name="portfolio_return")
        equity_curve = pd.Series(equity_arr / self.initial_capital, index=dates, name="equity")
        turnover_s = pd.Series(turnover_arr, index=dates, name="turnover")
        cost_s = pd.Series(cost_arr, index=dates, name="cost")
        borrow_fee_s = pd.Series(borrow_fee_arr, index=dates, name="borrow_fee")
        margin_s = pd.Series(margin_arr, index=dates, name="margin_usage")
        long_s = pd.Series(long_arr, index=dates, name="long_exposure")
        short_s = pd.Series(short_arr, index=dates, name="short_exposure")
        weights_df = pd.DataFrame(weights_arr, index=dates, columns=codes)

        return BacktestResult(
            daily_returns=daily_rets,
            weights_history=weights_df,
            equity_curve=equity_curve,
            turnover_series=turnover_s,
            cost_series=cost_s,
            borrow_fee_series=borrow_fee_s,
            margin_usage_series=margin_s,
            long_exposure_series=long_s,
            short_exposure_series=short_s,
            config={
                "strategy": self.strategy.name,
                "rebalance_freq": self.rebalance_freq,
                "initial_capital": self.initial_capital,
                "short_borrow_rate": self.short_costs.borrow_rate,
                "short_margin_ratio": self.short_costs.margin_ratio,
                "deleverage": self.deleverage,
            },
        )

    def _get_rebalance_days(self, dates: pd.DatetimeIndex) -> set:
        """根据频率返回调仓日集合。"""
        if self.rebalance_freq == "D":
            return set(dates)
        elif self.rebalance_freq == "W":
            s = pd.Series(dates, index=dates)
            return set(s.groupby(s.index.to_period("W")).first())
        elif self.rebalance_freq == "M":
            s = pd.Series(dates, index=dates)
            return set(s.groupby(s.index.to_period("M")).first())
        else:
            return set(dates)
