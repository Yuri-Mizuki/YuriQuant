"""
交易成本模型
============

支持:
- 佣金: 费率制 + 最低5元
- 印花税: 卖出单边，**支持按日期时变**（2023-08-28 减半，见 ``STAMP_DUTY_CHANGES``）
- 滑点: 按基点扣除，**或**逐股冲击成本模型（ImpactCostModel，P0-1）
- 空头成本（ShortCostModel）: 借券费率按日计提 + 融券保证金占用

用法:
    costs = TransactionCosts(commission_rate=0.0001, commission_min=5.0,
                             stamp_duty=0.001, slippage_bp=5)
    fee = costs.calc(old_weights, new_weights, turnover, capital)

    # 逐股冲击（需逐股 ADV/波动率）：滑点项由 ImpactCostModel 替代固定 slippage_bp
    cost = costs.calc(old_w, new_w, turnover, capital, adv=adv_i, vol=vol_i)

    # 时变印花税（2026-11-11 P0 修正）：传 date 才按当时税率，
    # 不传则用构造时给的 stamp_duty（向后兼容，但跨政策窗口的窗口需显式传）
    cost = costs.calc(old_w, new_w, turnover, capital, date=dates[i])
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backtest.metrics import PERIODS_PER_YEAR


# 证券交易印花税历史税率（卖出单边）—— 财政部/税务总局公告为准。
# 2023-08-28 起减半：千1 → 万5（财政部 税务总局公告 2023 年第 39 号）。
# 只列**有变更**的时点，取「该日期起生效」；查找时取 <= date 的最后一条。
STAMP_DUTY_CHANGES: tuple[tuple[str, float], ...] = (
    ("2008-09-19", 0.001),   # 由双边改单边征收，税率千1
    ("2023-08-28", 0.0005),  # 减半至万5
)


def stamp_duty_on(date) -> float:
    """返回 ``date`` 当日适用的证券交易印花税税率（卖出单边）。

    Args:
        date: 交易日（8 位 int YYYYMMDD / str / datetime / Timestamp / None）。

    Returns:
        税率（小数）。``date`` 为 None 时返回现行税率（万5）。
    """
    if date is None:
        return STAMP_DUTY_CHANGES[-1][1]
    # 日历是 int YYYYMMDD（PIT 口径）：int→Timestamp 必须先转 str，否则落 1970
    if isinstance(date, (int, np.integer)):
        key = str(int(date))
        stamp = f"{key[:4]}-{key[4:6]}-{key[6:8]}"
    elif hasattr(date, "strftime"):
        stamp = date.strftime("%Y-%m-%d")
    else:
        stamp = str(date)[:10]
    rate = STAMP_DUTY_CHANGES[0][1]
    for eff, r in STAMP_DUTY_CHANGES:
        if stamp >= eff:
            rate = r
    return rate


@dataclass
class ImpactCostModel:
    """逐股冲击成本模型（平方根律 + A 股买卖非对称），P0-1（2026-10-13）。

    机制（Gatheral 2010 平方根律；A 股参数取自陈奥林等《不容忽视的交易成本》）::

        participation_i = Q_i / ADV_i            # Q=成交金额(元), ADV=日均成交额(元)
        impact_frac_i   = eta · sigma_i · sqrt(participation_i)
        cost_i          = Q_i · impact_frac_i · (buy_mult if Δw_i>0 else 1.0)

    - ``eta``：冲击系数，文献 η≈0.3~0.5；
    - ``buy_multiplier``：A 股买入冲击约为卖出的 1.5 倍（散户挂单偏好）；
    - ``max_impact``：单名单边冲击上限（默认 10%），防极端参与率爆表；
    - ``min_adv``：ADV 下限（元），防除零/极端参与率。

    与固定 ``slippage_bp`` 的关系：``TransactionCosts`` 中二者**互斥**——传了逐股
    ``adv``/``vol`` 且挂了本模型时，滑点项 = 本模型算出的冲击成本（元）；否则退回
    固定 ``slippage_bp``。佣金/印花税始终照常计。

    ⚠️ 参数是文献/研报经验值，**须用本项目数据重新校准**；且项目无真实成交数据，
    本模型产出的是「估计成本」而非「实测成本」（见
    ``reports/docs/design/链路优化调查_预测组合执行_2026-10-13.md`` §9.5 复现边界）。
    """

    eta: float = 0.4
    buy_multiplier: float = 1.5
    max_impact: float = 0.10
    min_adv: float = 1e6

    def calc_per_name(
        self,
        old_weights,
        new_weights,
        capital: float,
        adv,
        vol,
    ) -> np.ndarray:
        """逐股冲击成本（元），返回与权重同长的数组。

        Args:
            old_weights / new_weights: 调仓前后权重（pd.Series 或 np.ndarray）。
            capital: 当前总资金（元）。
            adv: 逐股日均成交额（元）；NaN/非正 → 该股不计冲击（避免制造假成本）。
            vol: 逐股日波动率（小数）；NaN/非正 → 该股不计冲击。
        """
        old = np.asarray(old_weights, dtype=float).reshape(-1)
        new = np.asarray(new_weights, dtype=float).reshape(-1)
        if old.shape != new.shape:
            raise ValueError(f"权重维度不一致: {old.shape} vs {new.shape}")
        adv_arr = np.asarray(adv, dtype=float).reshape(-1)
        vol_arr = np.asarray(vol, dtype=float).reshape(-1)
        if adv_arr.shape != old.shape or vol_arr.shape != old.shape:
            raise ValueError(
                f"adv/vol 维度须与权重一致: {adv_arr.shape}/{vol_arr.shape} vs {old.shape}")

        delta = new - old
        notional = np.abs(delta) * float(capital)
        valid = (
            np.isfinite(adv_arr) & (adv_arr > 0)
            & np.isfinite(vol_arr) & (vol_arr > 0)
            & np.isfinite(notional) & (notional > 0)
        )
        out = np.zeros_like(notional)
        if not valid.any():
            return out
        safe_adv = np.maximum(adv_arr[valid], self.min_adv)
        participation = notional[valid] / safe_adv
        impact = self.eta * vol_arr[valid] * np.sqrt(np.clip(participation, 0.0, None))
        impact = np.clip(impact, 0.0, self.max_impact)
        side = np.where(delta[valid] > 0, self.buy_multiplier, 1.0)
        out[valid] = notional[valid] * impact * side
        return out

    def calc(self, old_weights, new_weights, capital: float, adv, vol) -> float:
        """逐股冲击成本合计（元）。"""
        return float(np.sum(self.calc_per_name(old_weights, new_weights, capital, adv, vol)))


class TransactionCosts:
    """交易成本计算。"""

    def __init__(
        self,
        commission_rate: float = 0.0001,
        commission_min: float = 5.0,
        stamp_duty: float = 0.0005,
        slippage_bp: float = 5.0,
        impact: ImpactCostModel | None = None,
        stamp_duty_schedule: bool = False,
    ):
        self.commission_rate = commission_rate
        self.commission_min = commission_min
        self.stamp_duty = stamp_duty
        self.slippage_bp = slippage_bp
        # 逐股冲击模型（可选）：挂上且调用方传了逐股 adv/vol 时，滑点项改走它。
        self.impact = impact
        # 印花税是否按调仓日时变。**默认关闭**：`stamp_duty` 是显式给定的固定值，
        # 不能被日期悄悄覆盖 —— 否则 `default_costs(factor_cost=False)` 这类
        # 「显式置零」的无成本对照会被重新收上税（2026-11-11 踩过：两个引擎
        # 单测因净收益被扣税而失败）。只有 `default_costs()` 生产口径才显式开启。
        self.stamp_duty_schedule = stamp_duty_schedule

    def calc(
        self,
        old_weights,
        new_weights,
        turnover: float,
        capital: float,
        adv=None,
        vol=None,
        date=None,
    ) -> float:
        """计算单期交易成本（金额）。

        Args:
            old_weights: 调仓前权重（pd.Series 或 np.ndarray）
            new_weights: 调仓后权重（pd.Series 或 np.ndarray）
            turnover: 单边换手率 (|new - old| 之和 / 2)，即买入额 = 卖出额
            capital: 当前总资金
            adv: 可选，逐股日均成交额（元）。与 ``vol`` 同给、且挂了 ``impact``
                模型时，滑点项改为逐股冲击成本。
            vol: 可选，逐股日波动率（小数）。
            date: 可选，调仓日。给了则印花税按当日税率（时变，见
                ``stamp_duty_on``）；不给则用构造时传入的固定 ``stamp_duty``。

        Returns:
            成本总额（元）

        说明：
            ``turnover`` 为单边换手率，对应买入金额 = 卖出金额 = turnover*capital。
            佣金 / 滑点对买卖双边计费，印花税仅对卖出单边计费。早期实现三者均少算
            一倍（佣金/滑点按单边、印花税按四分之一边），且零换手时仍按最低佣金扣费。
        """
        # 单边交易金额 = 买入额 = 卖出额
        single_sided_value = float(turnover) * capital
        if single_sided_value <= 0:
            return 0.0
        two_sided_value = 2.0 * single_sided_value  # 买 + 卖

        # 佣金: 买卖双边，不低于最低佣金（仅在实际有交易时收取）
        # 简化: 按总交易额算佣金，实际每笔单独算最低5元
        commission = max(two_sided_value * self.commission_rate, self.commission_min)

        # 印花税: 仅卖出方（单边）。仅当对象显式开启时变（`stamp_duty_schedule=True`）
        # 且调用方给了 date 时才按当时税率，否则一律用构造时的固定 `stamp_duty`。
        duty = (stamp_duty_on(date)
                if (date is not None and self.stamp_duty_schedule)
                else self.stamp_duty)
        stamp = single_sided_value * duty

        # 滑点: 逐股冲击（挂在 impact 且给了 adv/vol）优先，否则固定基点
        if self.impact is not None and adv is not None and vol is not None:
            slip = self.impact.calc(old_weights, new_weights, capital, adv, vol)
        else:
            slip = two_sided_value * self.slippage_bp / 10000

        return commission + stamp + slip


class EtfCosts(TransactionCosts):
    """ETF 场内交易成本：免印花税、佣金按场内基金口径（更低且最低档 0.1 元）、滑点同股票。

    与股票档（含印花税千 1）不同：ETF 买卖免印花税，且佣金费率与最低门槛更低。
    用于 ETF 轮动回测（VectorBacktest 要求成本对象实现 ``calc`` 接口）。
    """

    def __init__(
        self,
        commission_rate: float = 0.0005,
        commission_min: float = 0.1,
        slippage_bp: float = 5.0,
    ):
        super().__init__(
            commission_rate=commission_rate,
            commission_min=commission_min,
            stamp_duty=0.0,  # ETF 免印花税
            slippage_bp=slippage_bp,
        )


@dataclass
class ShortCostModel:
    """空头腿成本模型：借券费率（按日计提）+ 融券保证金占用（资金效率报告）。

    修正空头腿乐观偏差：旧引擎只计交易成本（佣金/印花税/滑点），空头持仓的
    借券费完全忽略、保证金占用不报告 —— 多空策略（TopKLongShort 等）的收益
    系统性虚高，长持空头时偏差尤其明显。

    - borrow_rate: 年化借券费率。A 股融券常见 8%~10%，个股差异大，可配置；
      设为 0 等价于关闭借券费（旧口径）。
    - margin_ratio: 融券保证金比例，监管最低 100%（=1.0）。保证金占用 =
      多头名义 + 空头名义 × margin_ratio，相对 1 倍资金计。
    - days_per_year: 年化天数（交易日口径）。
    """

    borrow_rate: float = 0.08
    margin_ratio: float = 1.0
    days_per_year: int = PERIODS_PER_YEAR

    def daily_borrow_fee(self, short_exposure: float, capital: float) -> float:
        """按日计提借券费（金额）：空头名义金额 × 年化费率 / 交易日数。

        Args:
            short_exposure: 空头名义敞口，相对资金的权重和（如 1.0 = 满仓做空 1 倍）。
            capital: 当前总资金（元）。
        """
        if short_exposure <= 0.0 or capital <= 0.0 or self.borrow_rate <= 0.0:
            return 0.0
        return short_exposure * capital * self.borrow_rate / self.days_per_year

    def margin_usage(self, long_exposure: float, short_exposure: float) -> float:
        """保证金占用倍数：多头名义 + 空头名义×保证金比例（相对 1 倍资金）。

        > 1 表示组合隐含超过 1 倍资金的杠杆需求（如等权多空各 1 倍 = 2.0，
        需要 2 倍资金才能同时满仓两端）。
        """
        return long_exposure + short_exposure * self.margin_ratio


def impact_from_config(cfg: dict) -> ImpactCostModel | None:
    """从 ``Config.costs()`` 字典构建冲击模型；``impact_model`` 非 "sqrt" 时返回 None。

    P0-1（2026-10-13）：默认 **关闭**（``impact_model: none``），保持既有回测的
    固定滑点口径零回归；显式把配置改成 ``sqrt`` 才启用逐股冲击。
    """
    model = str(cfg.get("impact_model", "none")).lower()
    if model != "sqrt":
        return None
    return ImpactCostModel(
        eta=float(cfg.get("impact_eta", 0.4)),
        buy_multiplier=float(cfg.get("impact_buy_multiplier", 1.5)),
        max_impact=float(cfg.get("impact_max", 0.10)),
    )


def default_costs(factor_cost: bool = True) -> TransactionCosts:
    """交易成本单一真源：从 config 顶层 ``costs`` 段构建（2026-09-11 下沉）。

    原先定义在 ``scripts/pipelines/run_model_portfolio.py``——一个实验入口脚本，却被主实验
    （``rolling_grid_alla``）、生产（``alla_daily_rank``）和多个实验脚本**反向
    import**，属归属倒挂。成本是回测层概念，归 ``backtest/``。

    2026-09-10：费率真源由 ``model_portfolio.cost_*`` 上移到顶层 ``costs`` 段，
    与 :mod:`backtest.engine` 的缺省费率共用同一份配置（原先两套差 2~3 倍，
    网格实验选出的最优参数无法用引擎默认复跑）。

    Args:
        factor_cost: ``False`` 时置零（无成本对照）。

    消费方一律走本函数，禁止再硬编码费率字面量（防 config 改动后漂移）。
    """
    if not factor_cost:
        # 无成本对照必须真的为零：commission_min（最低 5 元）也要置零，
        # 否则每次调仓仍扣 5 元（2026-10-19 审计 P2）。
        return TransactionCosts(commission_rate=0.0, stamp_duty=0.0,
                                slippage_bp=0.0, commission_min=0.0)
    from config import Config  # 延迟导入：让本模块顶层保持"纯成本模型"无配置副作用

    cfg = Config.costs()
    return TransactionCosts(commission_rate=cfg["commission_rate"],
                            commission_min=cfg["commission_min"],
                            stamp_duty=cfg["stamp_duty"],
                            slippage_bp=cfg["slippage_bp"],
                            # 生产口径开启时变印花税：窗口跨 2023-08-28 政策变更，
                            # 单一常数必然导致一半窗口错（见 stamp_duty_on）。
                            stamp_duty_schedule=True,
                            impact=impact_from_config(cfg))
