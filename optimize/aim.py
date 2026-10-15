"""α 衰减感知的 aim portfolio（Gârleanu-Pedersen 2013 简版，P0-3，2026-10-13）。

动机
----
外部分析与项目自身实测同向：**「最优近期收益预测器 ≠ 最优远期收益预测器」**
（Jensen et al.《Portfolio-ML》）。信号有持续性——同样的当期打分，持续性高的
（慢信号）未来累计预测力更大，值得给更高目标权重；而衰减快的信号要尽快兑现。
本项目既有「慢信号 s0.5 blend」与多周期 h1/h5/h10/h20，正是这一机制的天然素材。

本模块实现 GP 的两个可落地机制（**简版，非其完整闭式解**）
----------------------------------------------------------
1. **持续性缩放（aim 水平）**：AR(1) 信号持续性 ρ（本模块记 ``phi``），未来折现
   累计预测力 = f_t/(1−ρ)。故 ``alpha_eff = alpha/(1−phi)``——phi 越接近 1（慢信号）
   目标权重越大。phi 可以是标量（全体同质）或逐股 Series（用
   :func:`persistence_from_horizons` 从多周期预测估）。
2. **偏调整（trading rate）**：不一步换到 aim，而是
   ``w_t = w_{t−1} + speed·(aim − w_{t−1})``，其中
   ``speed = (1−phi)/(1+cost_kappa)``——衰减越快（phi 小）交易越急、成本越高越慢。
   这直接压制无谓换手。

⚠️ **复现边界（务必读）**
- 这是 GP(2013) 的**机制级代理**，不是其含信号自协方差结构的完整最优解；
  两个参数（``phi`` / ``cost_kappa``）**须用本项目数据校准**，本模块只给结构。
- phi 缺省 0.5、cost_kappa 缺省 0 → 退化为「半步偏向目标」的中性基线，
  **不构成已证明的增益**。见
  ``reports/docs/design/链路优化调查_预测组合执行_2026-10-13.md`` §9.2 P0-3。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

__all__ = [
    "persistence_scaled_alpha",
    "aim_target",
    "aim_weights",
    "optimize_weights_aim",
    "persistence_from_horizons",
]

_PHI_MAX = 0.95  # 持续性上限：防 1/(1−phi) 爆表（phi=0.95 → 乘子 20）


def _as_phi(phi: float | pd.Series, index) -> pd.Series:
    """把标量/Series 的 phi 规整为对齐 index、截断到 [0, _PHI_MAX] 的 Series。"""
    if isinstance(phi, pd.Series):
        p = phi.reindex(index)
    else:
        p = pd.Series(float(phi), index=index)
    p = p.astype(float).clip(lower=0.0, upper=_PHI_MAX)
    return p.where(np.isfinite(p), 0.0).fillna(0.0)


def _speed_from_phi(phi: float | pd.Series, cost_kappa: float) -> float:
    """偏调整速度 speed = (1−φ̄)/(1+κ)，截断到 [0,1]。"""
    if isinstance(phi, pd.Series):
        p = float(np.nanmean(np.clip(phi.to_numpy(dtype=float), 0.0, _PHI_MAX)))
    else:
        p = float(np.clip(phi, 0.0, _PHI_MAX))
    if not np.isfinite(p):
        p = 0.0
    kappa = max(float(cost_kappa), 0.0)
    return float(np.clip((1.0 - p) / (1.0 + kappa), 0.0, 1.0))


def persistence_scaled_alpha(
    alpha: pd.Series,
    phi: float | pd.Series,
) -> pd.Series:
    """持续性缩放：``alpha_eff = alpha / (1 − phi)``（phi→1 时乘子更大）。

    Args:
        alpha: 截面因子值（方向已对齐，越大越看好）。
        phi: 信号持续性 ∈ [0, 0.95]，标量或逐股 Series。
    Returns:
        Series，与 alpha 同索引（NaN 保持 NaN）。
    """
    p = _as_phi(phi, alpha.index)
    out = alpha.astype(float) / (1.0 - p)
    return out.where(alpha.notna())


def aim_target(
    alpha_eff: pd.Series,
    Sigma: np.ndarray,
    *,
    gamma: float = 1.0,
    max_weight: float | None = None,
    budget: float = 1.0,
) -> pd.Series:
    """aim 目标仓（一次性最优）：``w ∝ (1/γ)·Σ⁻¹·alpha_eff``，long-only 截断。

    用「解线性方程组 + 负值截零 + 归一」实现 long-only MVO（Jagannathan-Ma：
    long-only 约束本身是隐式收缩，比直接反转 Σ 稳健）。Σ 须半正定（见
    ``optimize.solver.estimate_covariance``）。

    Args:
        alpha_eff: 持续性缩放后的因子值（见 :func:`persistence_scaled_alpha`）。
        Sigma: (N,N) 半正定协方差，列序对齐 alpha_eff.index。
        gamma: 风险厌恶（越大越保守，目标越接近最小方差）。
        max_weight: 个股上限（可选）。
        budget: 目标权重和。
    Returns:
        Series(index=alpha_eff.index) 非负、和 ≈ budget；无有效信号时全 0。
    """
    if len(alpha_eff) == 0:
        return pd.Series(dtype=float)
    n = len(alpha_eff)
    S = np.asarray(Sigma, dtype=float)
    if S.shape != (n, n):
        raise ValueError(f"Sigma 形状 {S.shape} 与 alpha 长度 {n} 不匹配")
    a = alpha_eff.fillna(0.0).to_numpy(dtype=float)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a)
    if np.allclose(a, 0.0):
        return pd.Series(0.0, index=alpha_eff.index)
    try:
        x = np.linalg.solve(S, a)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(S, a, rcond=None)[0]
    w = x / max(float(gamma), 1e-12)
    w = np.clip(w, 0.0, None)  # long-only 截断
    if max_weight is not None and max_weight < np.inf:
        w = np.clip(w, 0.0, float(max_weight))
    s = w.sum()
    if s <= 0:
        return pd.Series(0.0, index=alpha_eff.index)
    w = w / s * float(budget)
    return pd.Series(w, index=alpha_eff.index)


def aim_weights(
    alpha: pd.Series,
    Sigma: np.ndarray,
    *,
    phi: float | pd.Series = 0.5,
    gamma: float = 1.0,
    cost_kappa: float = 0.0,
    prev_weights: pd.Series | None = None,
    max_weight: float | None = None,
    budget: float = 1.0,
) -> pd.Series:
    """单截面 aim 权重（持续性缩放 + 偏调整）。

    Args:
        alpha: 截面因子值（方向已对齐）。
        Sigma: (N,N) 协方差，列序对齐 alpha.index。
        phi: 信号持续性（标量或逐股）；缺省 0.5。
        gamma: 风险厌恶。
        cost_kappa: 成本强度（越大偏调整越慢、换手越低）。
        prev_weights: 上期权重；None 或空仓 → 直接跳到 aim。
        max_weight / budget: 见 :func:`aim_target`。
    Returns:
        Series(index=alpha.index) 非负、和 ≈ budget。
    """
    alpha_eff = persistence_scaled_alpha(alpha, phi)
    target = aim_target(alpha_eff, Sigma, gamma=gamma,
                        max_weight=max_weight, budget=budget)
    if prev_weights is None:
        return target
    prev = prev_weights.reindex(alpha.index).fillna(0.0).astype(float)
    if float(prev.abs().sum()) <= 0.0:
        return target  # 空仓起步：直接建仓
    speed = _speed_from_phi(phi, cost_kappa)
    w = prev + speed * (target - prev)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s > 0:
        w = w / s * float(budget)
    return w


def optimize_weights_aim(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    *,
    window: int = 120,
    min_periods: int = 60,
    cov_method: str = "ledoit_wolf",
    shrinkage: float = 0.5,
    phi: float | pd.Series = 0.5,
    gamma: float = 1.0,
    cost_kappa: float = 0.0,
    max_weight: float | None = None,
    budget: float = 1.0,
) -> pd.DataFrame:
    """面板级 aim 权重（逐调仓日：滚动 Σ + 上期权重偏调整）。

    防前视：每个截面 t 的 Σ 只用 ``returns_panel`` 中 < t 的最近 window 行
    （与 ``optimize.solver.rolling_covariance`` 同纪律）。窗口不足的截面输出全 0。
    ``prev_weights`` 由本函数在时序中自动维护（上期输出）。
    """
    from optimize.solver import rolling_covariance

    dates = factor_panel.index
    codes = factor_panel.columns
    rows: list[pd.Series] = []
    prev: pd.Series | None = None
    for t in dates:
        Sigma = rolling_covariance(
            returns_panel, t, window=window, min_periods=min_periods,
            method=cov_method, shrinkage=shrinkage,
        )
        if Sigma is None:
            rows.append(pd.Series(0.0, index=codes))
            prev = None  # 空仓 → 下期直接建仓
            continue
        w = aim_weights(
            factor_panel.loc[t], Sigma, phi=phi, gamma=gamma,
            cost_kappa=cost_kappa, prev_weights=prev,
            max_weight=max_weight, budget=budget,
        )
        w = w.reindex(codes).fillna(0.0)
        rows.append(w)
        prev = w
    return pd.DataFrame(rows, index=dates, columns=codes).fillna(0.0)


def persistence_from_horizons(
    preds: Mapping[int, pd.Series],
) -> pd.Series:
    """从多周期预测的面板估计**逐股持续性** phi ∈ [0, 0.95]（实验性）。

    直觉：把各 horizon 的截面预测先 zscore，再对 log(horizon) 做逐股线性拟合，
    斜率 > 0 表示该股预测随 horizon 走强（慢信号），斜率 < 0 表示随 horizon 衰减
    （快信号）。斜率经 sigmoid 映射到 (0,1)。

    Args:
        preds: {horizon: Series(index=code)}，horizon 为正整数（如 {1,5,10,20}）。
    Returns:
        Series(index=code) ∈ [0, 0.95]。样本不足（<2 个 horizon）时返回中值 0.5。
    """
    hs = sorted(int(h) for h in preds)
    if len(hs) < 2:
        codes = pd.Index([])
        for s in preds.values():
            codes = s.index if len(codes) == 0 else codes.union(s.index)
        return pd.Series(0.5, index=codes)
    Z = []
    for h in hs:
        s = preds[h].astype(float)
        if s.std(ddof=0) > 0:
            s = (s - s.mean()) / s.std(ddof=0)
        else:
            s = s * 0.0
        Z.append(s)
    Zdf = pd.concat(Z, axis=1)
    Zdf.columns = [np.log(float(h)) for h in hs]
    x = Zdf.columns.to_numpy(dtype=float)
    # 逐股最小二乘斜率：slope = cov(x, z)/var(x)
    zc = Zdf.sub(Zdf.mean(axis=1), axis=0)
    xc = x - x.mean()
    var_x = float((xc ** 2).sum())
    slopes = zc.to_numpy(dtype=float) @ xc / var_x if var_x > 0 else np.zeros(len(Zdf))
    phi = 1.0 / (1.0 + np.exp(-slopes))  # sigmoid → (0,1)
    phi = np.clip(phi, 0.0, _PHI_MAX)
    return pd.Series(phi, index=Zdf.index)
