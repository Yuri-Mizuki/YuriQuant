"""
rank IC 奖励（Phase 1：市值中性化 + 调仓周期）
==============================================

研报对齐（系列之二十二 §2.3 基础上，叠加系列之二十四 §1.3 的改进奖励）：

- 基础项 = **市值中性化后的 |IC|**：直接 abs(IC) 会让因子过分暴露在小市值风格上
  （研报明确观察到），因此先把因子对（对数）市值做截面回归取残差，再算 IC。
- **多头 IR 奖励**（系列之二十四）：纯 IC 目标挖出的因子多头超额往往很差，
  在 IC 上乘 ``(1 + LONG_IR_LAMBDA × clip(long_ir, 0, LONG_IR_CAP))``——
  long_ir = 多头桶（因子前 ``top_q`` 分位等权组合）相对全市场等权基准的
  日度超额收益的 mean/std（未年化）。
- **Barra 风格时序相关惩罚**（系列之二十四）：因子多空价差与预计算的风格代理
  价差（动量 20d / 波动率 20d / 对数成交额）的最大 |时序相关| 越高越接近风险因子，
  乘 ``(1 − BARRA_TS_PENALTY_MU × clip(corr, 0, 1))``。风格代理在
  :func:`build_barra_styles` 一处生成，如需接入真实 Barra 因子只改该函数。
- 因子评估**调仓周期为 10 日**：``factor[t]`` 对应未来 10 日收益
  （``close.pct_change(h).shift(-h)``）。

综合（研报原文口径）::

    reward = abs(train_ic)
             × (1 + LONG_IR_LAMBDA × clip(train_long_ir, 0, LONG_IR_CAP))
             × (1 − BARRA_TS_PENALTY_MU × clip(barra_ts_corr, 0, 1))

IC 符号为负的因子按"取反向"处理：long_ir 以 sign(mean IC) 定向后再进奖励
（等价于把因子翻转后评估多头），barra 相关用 |corr| 天然符号无关。

口径与项目一致：因子面板 ``factor[t]`` 与收益面板逐截面 spearman 相关；
常数/全 NaN 因子给最小奖励 ``eps``。

``temp``：奖励温度。``None`` = 线性（默认）；数值时锐化为 ``exp(R/temp)``。

**奖励缓存**：canonical 字符串 -> 奖励（研报 §2.2 用 ExprNode 简化降缓存重复；
TB 训练中同一公式会被反复采到，缓存是 CPU 训练可行性的关键）。

**IC 口径分层（2026-09-11 声明）**：本模块的 ``returns_rank`` 是**训练期**性能
捷径——预先算好收益的截面 rank，省去每天重排（约 30% IC 耗时）。它与 canonical
（``stats.ic.calc_ic_series`` 逐日取共同有效点再算 Spearman）**不总等价**：因子
整行缺失无影响，**行内散点缺失才分叉**（30% 散点缺失时日均 IC 差 ~1.1e-2，与
IC 量级同阶）。分层约定：

- **训练**（``scripts/factors/run_gflownet_phase1.py`` -> ``RewardPool``）：走捷径。奖励
  只是 batch 内的相对排序信号（决定哪些公式进 hof），1e-2 偏差不改变相对次序，
  而训练时间敏感。
- **入库 / 评估**（``FactorLibrary.register`` -> ``calc_ic_series``，不传该参数）：
  走 canonical。写进因子库、对外呈现的 IC 必须与全库唯一口径一致。

**因此两处 IC 数字不可直接对比**；已完成的 Phase 0 / Phase 1 结论基于捷径口径，
若改用 canonical 需重跑才有可比性（代价约 30% 训练时间）。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from factor.gflownet.expr import ExprBuilder, canonical_formula
from factor.formula import formula_builder

__all__ = ["rank_ic_series", "RewardCache", "make_reward_fn",
           "build_horizon_returns", "neutralize_market_cap",
           "factor_long_stats", "long_excess_ir", "build_barra_styles",
           "max_style_corr", "rank_stability", "composed_factor_reward",
           "LONG_IR_LAMBDA", "LONG_IR_CAP", "BARRA_TS_PENALTY_MU",
           "DEFAULT_TOP_Q"]

# 系列之二十四 §1.3 的奖励塑形系数（研报正文只给出函数形式，未披露数值，
# 默认值按"奖励主导项仍是 |IC|、IR 与惩罚为修正量级"的原则选取）
LONG_IR_LAMBDA = 0.5          # 多头 IR 奖励强度 λ
LONG_IR_CAP = 1.0             # long_ir 进入乘子的上限（防极端 IR 主导）
BARRA_TS_PENALTY_MU = 0.5     # Barra 时序相关惩罚强度 μ
DEFAULT_TOP_Q = 0.2           # 多头桶分位（前 20%）


def neutralize_market_cap(factor_panel: pd.DataFrame,
                          market_cap: pd.DataFrame) -> pd.DataFrame:
    """对数市值中性化（研报 §2.3 奖励用）。

    薄封装 :func:`factor.preprocessing.neutralize_single`（2026-09-10 收口：
    此前本文件自实现一份向量化版本、preprocessing 另有逐日 lstsq 版，
    同一件事两处维护）。保留本入口是因为 GFlowNet 训练热路径按语义命名调用。
    """
    from factor.preprocessing import neutralize_single
    return neutralize_single(factor_panel, market_cap, log_covariate=True)


def rank_ic_series(factor_panel: pd.DataFrame, returns_panel: pd.DataFrame,
                   returns_rank: pd.DataFrame | None = None) -> pd.Series:
    """逐日截面 spearman IC（因子 t vs 收益面板同日起始，如次日/horizon 收益）。

    薄封装 :func:`stats.ic.calc_ic_series`（全库唯一 IC 口径实现；2026-09-10
    收口，此前本文件自实现一份逐行 rank + 归一化相关）。本入口只额外负责
    **把索引对齐到 ``factor_panel.index``**——训练面板是主索引，而
    ``calc_ic_series`` 按两面板交集返回。

    ⚠️ ``returns_rank``（预计算的收益 rank 面板，训练中省约 30% IC 耗时）**只在
    因子不比收益多出任何 NaN 时**才与默认路径完全等价；因子整行缺失（窗口预热）
    不受影响，行内散点缺失才有差异，详见 ``stats.ic.calc_ic_series`` 的 docstring。
    """
    from stats.ic import calc_ic_series
    ic = calc_ic_series(factor_panel, returns_panel, method="spearman",
                        returns_rank=returns_rank)
    ic = ic.reindex(factor_panel.index)
    # 非有限值保护（收敛前本文件实现用行归一化数学，常数截面行天然给 NaN）：
    # canonical 走 pandas corrwith，常数截面行可能返回 ±inf，而奖励端做
    # abs(ic).mean()，inf 会直接毒化整个奖励。此处保留 NaN 语义。
    return ic.where(np.isfinite(ic))


def build_horizon_returns(close: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """未来 ``horizon`` 日收益：``close[t+h]/close[t] - 1``（t 对齐因子日）。"""
    if horizon <= 1:
        return close.pct_change().shift(-1) if horizon == 1 else close.pct_change(horizon).shift(-horizon)
    return close.pct_change(horizon).shift(-horizon)


def _bucket_top_bot(panel_df: pd.DataFrame, top_q: float):
    """按行构造「因子前 top_q 分位 / 后 top_q 分位」的布尔掩码。

    每行桶大小 k = max(round(top_q × 有效数), 2)；全 NaN 行 rank 为 NaN，
    与阈值比较结果为 False，天然被排除。
    """
    fr = panel_df.rank(axis=1)
    n = panel_df.notna().sum(axis=1)
    k = np.maximum((top_q * n).round(), 2.0)
    top = fr.ge(n - k + 1.0, axis=0)
    bot = fr.le(k, axis=0)
    return top, bot


def factor_long_stats(fp: pd.DataFrame, rets: pd.DataFrame,
                      top_q: float = DEFAULT_TOP_Q) -> tuple[pd.Series, pd.Series]:
    """因子的多头超额序列与多空价差序列（long_ir / barra_ts_corr 的输入）。

    - ``long_excess``：多头桶等权收益 − 全市场等权基准收益（逐日）；
    - ``spread``：多头桶收益 − 空头桶收益（逐日）。

    全部向量化；因子/收益任一侧无效的样本由掩码与 skipna 自动排除。
    """
    top, bot = _bucket_top_bot(fp, top_q)
    r = rets.reindex_like(fp)
    long_ret = r.where(top).mean(axis=1)
    bot_ret = r.where(bot).mean(axis=1)
    mkt = r.mean(axis=1)
    return long_ret - mkt, long_ret - bot_ret


def long_excess_ir(long_excess: pd.Series, min_obs: int = 30) -> float:
    """多头超额的 mean/std（未年化日度 IR）；样本不足或零方差返回 0。"""
    s = long_excess.dropna()
    if len(s) < min_obs:
        return 0.0
    sd = float(s.std())
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return float(s.mean()) / sd


def _style_spread(style_panel: pd.DataFrame, rets_h: pd.DataFrame,
                  top_q: float) -> pd.Series:
    """风格面板自身的多空价差序列（风格组合按其排名分桶）。"""
    _, spread = factor_long_stats(style_panel, rets_h, top_q=top_q)
    return spread


def build_barra_styles(panel: dict[str, pd.DataFrame], rets_h: pd.DataFrame,
                       top_q: float = DEFAULT_TOP_Q) -> dict[str, pd.Series]:
    """从原始字段预计算风格代理的多空价差序列（Barra 惩罚的参照系）。

    代理集（研报用完整 Barra 风格，此处为可得字段的最小近似；
    接入真实 Barra 因子时只需替换本函数）：
    - momentum20   = 20 日动量（close.pct_change(20)）
    - volatility20 = 20 日收益标准差
    - liquidity    = 对数成交额（panel 无 amount 时跳过）

    返回 {风格名: 价差 Series}；有效观测 < 60 日的风格自动剔除。
    """
    styles: dict[str, pd.Series] = {}
    close = panel.get("close")
    if close is not None and not close.empty:
        cands = {
            "momentum20": close.pct_change(20),
            "volatility20": close.pct_change().rolling(20).std(),
        }
        amount = panel.get("amount")
        if amount is not None and not amount.empty:
            with np.errstate(divide="ignore", invalid="ignore"):
                cands["liquidity"] = np.log(amount.where(amount > 0))
        for name, sp in cands.items():
            if sp is None or not np.isfinite(sp.to_numpy(dtype=float)).any():
                continue
            s = _style_spread(sp, rets_h, top_q)
            if s.dropna().shape[0] >= 60:
                styles[name] = s
    return styles


def max_style_corr(spread: pd.Series, styles: dict[str, pd.Series],
                   min_obs: int = 60) -> float:
    """价差序列与各风格价差的 |时序相关| 最大值（无可用重叠返回 0）。"""
    x = spread.dropna()
    best = 0.0
    for ref in styles.values():
        j = pd.concat([x.rename("f"), ref.rename("s")], axis=1,
                      join="inner").dropna()
        if len(j) < min_obs:
            continue
        c = j["f"].corr(j["s"])
        if np.isfinite(c):
            best = max(best, abs(float(c)))
            if best >= 1.0:
                break
    return best


def rank_stability(factor_panel: pd.DataFrame, lag: int = 1) -> float:
    """截面排名自相关（换手率/RRE 筛选的核心量，Alphalens 风格）。

    越接近 1 = 排序越稳定（换手低、可落地）；≈0 = 每日大换血。
    stats 公共层建立后（2026-08-29）直接委托 ``stats.ic.factor_autocorr``
    统一口径（lag=1 与历史实现完全一致），不再独立维护同款实现。
    """
    from stats.ic import factor_autocorr
    return factor_autocorr(factor_panel, max_lag=lag)


def composed_factor_reward(fp: Optional[pd.DataFrame], rets: pd.DataFrame,
                           returns_rank: Optional[pd.DataFrame] = None,
                           market_cap: Optional[pd.DataFrame] = None, *,
                           eps: float = 1e-4, temp: Optional[float] = None,
                           long_ir_lambda: float = LONG_IR_LAMBDA,
                           long_ir_cap: float = LONG_IR_CAP,
                           barra_mu: float = BARRA_TS_PENALTY_MU,
                           styles: Optional[dict[str, pd.Series]] = None,
                           top_q: float = DEFAULT_TOP_Q) -> float:
    """单因子完整奖励核心（make_reward_fn 与并行 RewardPool 共享）::

        R = |mean(IC)| × (1 + λ·clip(long_ir, 0, cap)) × (1 − μ·clip(barra_corr, 0, 1))

    ``fp`` 为空或 base≤0 → 0（调用方线性口径下再落到 eps 下限）。
    IC 为负的因子按"取反向"评估多头：long_ir 以 sign(mean IC) 定向；
    barra 相关用 |corr|，天然符号无关。"""
    if fp is None or fp.empty:
        return 0.0
    if market_cap is not None:
        fp = neutralize_market_cap(fp, market_cap)
    ic = rank_ic_series(fp, rets, returns_rank=returns_rank)
    base = float(ic.abs().mean()) if len(ic) else 0.0
    if not np.isfinite(base) or base <= 0:
        return 0.0

    mult = 1.0
    want_ir = long_ir_lambda is not None and long_ir_lambda > 0
    want_barra = (barra_mu is not None and barra_mu > 0
                  and styles is not None and len(styles) > 0)
    if want_ir or want_barra:
        long_excess, spread = factor_long_stats(fp, rets, top_q=top_q)
    if want_ir:
        orient = -1.0 if float(ic.mean()) < 0 else 1.0
        ir = orient * long_excess_ir(long_excess)
        mult *= 1.0 + float(long_ir_lambda) * float(np.clip(ir, 0.0, float(long_ir_cap)))
    if want_barra:
        sc = max_style_corr(spread, styles)
        mult *= 1.0 - float(barra_mu) * float(np.clip(sc, 0.0, 1.0))

    v = base * mult
    if temp is not None:
        vv = v if v > 0 else eps
        return float(np.exp(vv / temp))
    return v


def _linear(v: float, eps: float = 1e-4) -> float:
    """线性口径下限：无效或非正的奖励落到 eps（避免零奖励淹没 TB 梯度）。"""
    return v if np.isfinite(v) and v > 0 else eps


def _exp(v: float, temp: float, eps: float = 1e-4) -> float:
    """exp 锐化口径：R = exp(max(v, eps)/temp)。"""
    vv = v if np.isfinite(v) and v > 0 else eps
    return float(np.exp(vv / temp))


class RewardCache:
    """canonical -> 奖励 缓存（带容量上限，防内存膨胀）。"""

    def __init__(self, max_size: int = 200_000):
        self._cache: dict[str, float] = {}
        self.max_size = max_size

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, key: str) -> Optional[float]:
        return self._cache.get(key)

    def put(self, key: str, value: float) -> None:
        if len(self._cache) >= self.max_size:
            self._cache.clear()          # 简单策略：满了清空重建
        self._cache[key] = value

    def stats(self) -> dict:
        return {"cache_size": len(self._cache)}


def make_reward_fn(panel: dict[str, pd.DataFrame], returns: pd.DataFrame,
                   features: list[str], eps: float = 1e-4,
                   cache: Optional[RewardCache] = None,
                   evaluator: Optional[Callable] = None,
                   temp: Optional[float] = None,
                   market_cap: Optional[pd.DataFrame] = None,
                   horizon: int = 1,
                   node_cache: Optional[dict] = None,
                   long_ir_lambda: float = LONG_IR_LAMBDA,
                   long_ir_cap: float = LONG_IR_CAP,
                   barra_mu: float = BARRA_TS_PENALTY_MU,
                   styles: Optional[dict[str, pd.Series]] = None,
                   top_q: float = DEFAULT_TOP_Q):
    """构造奖励函数 ``reward(builder) -> float``（含缓存；evaluator 可注入 mock）。

    Args:
        panel: 特征面板 dict（date×code）。
        returns: 收益面板（horizon=1 时用；horizon>1 时从 panel['close'] 构造）。
        market_cap: 市值面板（date×code）；非 None 时先做市值中性化再算 IC。
        horizon: 调仓周期（研报 = 10）。
        temp: 奖励温度，None = 线性（研报口径），数值 = exp 锐化。
        node_cache: 子树级求值缓存（见 formula._eval_node_cached），训练中
            大量共享 ``ts_min_10(amount)`` 级子表达式，可显著提速深树求值。
        long_ir_lambda / long_ir_cap: 多头 IR 奖励 λ 与 clip 上限
            （研报系列之二十四；0 = 关闭，退回纯 |IC| 奖励）。
        barra_mu: Barra 风格时序相关惩罚强度 μ（0 = 关闭）。
        styles: 风格价差 dict；None 且 barra_mu>0 时从 panel 自动构建
            （momentum20/volatility20/liquidity 代理）。
        top_q: 多头/空头桶分位。
    """
    cache = cache if cache is not None else RewardCache()
    if horizon > 1 and "close" in panel:
        rets = build_horizon_returns(panel["close"], horizon)
    else:
        rets = returns
    # 预计算收益 rank（训练中收益固定，省去每因子重复 rank）
    rets_rank = rets.rank(axis=1)

    # 风格价差只依赖收益与原始字段，训练全程不变 -> 预计算一次
    want_barra = barra_mu is not None and barra_mu > 0
    if want_barra and styles is None:
        styles = build_barra_styles(panel, rets, top_q=top_q)
    use_shaping = (long_ir_lambda is not None and long_ir_lambda > 0) or want_barra

    def reward(builder: ExprBuilder | None) -> float:
        # evaluator 注入模式下允许 builder=None（直接按固定面板求值）
        formula = canonical_formula(builder) if builder is not None else ""
        hit = cache.get(formula)
        if hit is not None:
            return hit
        if evaluator is not None:
            fp = evaluator(formula)
        else:
            fp = formula_builder(formula, features=features,
                                 node_cache=node_cache)(panel)
        if use_shaping and fp is not None and not fp.empty:
            # 完整口径走共享核心（含多头 IR + Barra 惩罚）
            v = composed_factor_reward(
                fp, rets, returns_rank=rets_rank, market_cap=market_cap,
                eps=eps, temp=None,  # 温度在下方统一施加，避免双重变换
                long_ir_lambda=long_ir_lambda, long_ir_cap=long_ir_cap,
                barra_mu=barra_mu, styles=styles, top_q=top_q)
        else:
            # 纯 |IC| 口径（shaping 全关时的兼容路径）
            if fp is None or fp.empty:
                v = 0.0
            else:
                if market_cap is not None:
                    fp = neutralize_market_cap(fp, market_cap)
                ic = rank_ic_series(fp, rets, returns_rank=rets_rank)
                v = float(ic.abs().mean()) if len(ic) else 0.0
        val = _exp(v, temp=temp, eps=eps) if temp is not None else _linear(v, eps=eps)
        cache.put(formula, val)
        return val

    return reward
