"""
rank IC 奖励（Phase 1：市值中性化 + 调仓周期）
==============================================

研报对齐（系列之二十二 §2.3）：

- 奖励 = **市值中性化后的 |IC|**：直接 abs(IC) 会让因子过分暴露在小市值风格上
  （研报明确观察到），因此先把因子对（对数）市值做截面回归取残差，再算 IC。
- 因子评估**调仓周期为 10 日**：``factor[t]`` 对应未来 10 日收益
  （``close.pct_change(h).shift(-h)``）。

口径与项目一致：因子面板 ``factor[t]`` 与收益面板逐截面 spearman 相关；
奖励 = 全期 ``mean(|ic_t|)``（跳过 NaN 日），常数/全 NaN 因子给最小奖励 ``eps``。

``temp``：奖励温度。``None`` = 线性 ``R = max(mean|IC|, eps)``（研报口径，默认）；
数值时锐化为 ``R = exp(mean|IC| / temp)``（单调变换，Phase 1 后可选实验）。

**奖励缓存**：canonical 字符串 -> 奖励（研报 §2.2 用 ExprNode 简化降缓存重复；
TB 训练中同一公式会被反复采到，缓存是 CPU 训练可行性的关键）。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from factor.gflownet.expr import ExprBuilder, canonical_formula
from factor.formula import formula_builder

__all__ = ["rank_ic_series", "RewardCache", "make_reward_fn",
           "build_horizon_returns", "neutralize_market_cap"]


def neutralize_market_cap(factor_panel: pd.DataFrame,
                          market_cap: pd.DataFrame) -> pd.DataFrame:
    """**逐行向量化**的对数市值中性化（研报 §2.3 奖励用）。

    等价于「因子 ~ 截距 + log(市值)」的逐日截面回归残差：行中心化后单变量
    回归无截距项，beta = Σ(xa·fa)/Σ(xa²)，残差 = fa − β·xa。单次调用毫秒级
    （vs 逐日 lstsq 的 ~1s/因子，是 Phase 1 训练可行性的关键）。
    """
    x = np.log(market_cap.reindex_like(factor_panel))
    m = factor_panel.notna() & x.notna()
    fp = factor_panel.where(m)
    x = x.where(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        fa = fp.sub(fp.mean(axis=1), axis=0)      # 逐行中心化（axis=0 行向广播）
        xa = x.sub(x.mean(axis=1), axis=0)
        beta = (fa * xa).sum(axis=1) / (xa * xa).sum(axis=1)
        resid = fa.sub(xa.mul(beta, axis=0), axis=0)
    return resid


def rank_ic_series(factor_panel: pd.DataFrame, returns_panel: pd.DataFrame,
                   returns_rank: pd.DataFrame | None = None) -> pd.Series:
    """逐日截面 spearman IC（因子 t vs 收益面板同日起始，如次日/horizon 收益）。

    向量化：先对齐 NaN 位置，再对两面板逐行 rank（axis=1），逐行 Pearson 相关
    即等价于 spearman（rank 后 Pearson）。单次调用无 Python 级逐日循环。

    ``returns_rank``：可传入**预计算的收益 rank 面板**（训练中收益固定，
    省去每次重复 rank，可省约 40% IC 耗时）。
    """
    r = returns_panel.reindex_like(factor_panel)
    m = factor_panel.notna() & r.notna()
    cnt = m.sum(axis=1)
    f = factor_panel.where(m).rank(axis=1)
    y = returns_rank.reindex_like(factor_panel).where(m) if returns_rank is not None \
        else r.where(m).rank(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fa = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1, ddof=0), axis=0)
        ya = y.sub(y.mean(axis=1), axis=0).div(y.std(axis=1, ddof=0), axis=0)
        ic = (fa * ya).mean(axis=1)
    ic = ic.where((cnt >= 5) & np.isfinite(ic))
    return ic


def build_horizon_returns(close: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """未来 ``horizon`` 日收益：``close[t+h]/close[t] - 1``（t 对齐因子日）。"""
    if horizon <= 1:
        return close.pct_change().shift(-1) if horizon == 1 else close.pct_change(horizon).shift(-horizon)
    return close.pct_change(horizon).shift(-horizon)


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
                   node_cache: Optional[dict] = None):
    """构造奖励函数 ``reward(builder) -> float``（含缓存；evaluator 可注入 mock）。

    Args:
        panel: 特征面板 dict（date×code）。
        returns: 收益面板（horizon=1 时用；horizon>1 时从 panel['close'] 构造）。
        market_cap: 市值面板（date×code）；非 None 时奖励 = **市值中性化后**的
            |IC|（研报 §2.3：对对数市值截面回归取残差，避免小市值风格暴露）。
        horizon: 调仓周期（研报 = 10）。
        temp: 奖励温度，None = 线性（研报口径），数值 = exp 锐化。
        node_cache: 子树级求值缓存（见 formula._eval_node_cached），训练中
            大量共享 ``ts_min_10(amount)`` 级子表达式，可显著提速深树求值。
    """
    cache = cache if cache is not None else RewardCache()
    if horizon > 1 and "close" in panel:
        rets = build_horizon_returns(panel["close"], horizon)
    else:
        rets = returns
    # 预计算收益 rank（训练中收益固定，省去每因子重复 rank）
    rets_rank = rets.rank(axis=1)

    def _linear(v: float) -> float:
        return v if np.isfinite(v) and v > 0 else eps

    def _exp(v: float) -> float:
        vv = v if np.isfinite(v) and v > 0 else eps
        return float(np.exp(vv / temp))

    def reward(builder: ExprBuilder) -> float:
        formula = canonical_formula(builder)
        hit = cache.get(formula)
        if hit is not None:
            return hit
        if evaluator is not None:
            fp = evaluator(formula)
        else:
            fp = formula_builder(formula, features=features,
                                 node_cache=node_cache)(panel)
        if fp is None or fp.empty:
            v = 0.0
        else:
            if market_cap is not None:
                fp = neutralize_market_cap(fp, market_cap)
            ic = rank_ic_series(fp, rets, returns_rank=rets_rank)
            v = float(ic.abs().mean())
        val = _exp(v) if temp is not None else _linear(v)
        cache.put(formula, val)
        return val

    return reward
