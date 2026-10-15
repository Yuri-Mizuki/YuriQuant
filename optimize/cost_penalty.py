"""成本感知组合优化：惩罚系数的校准与扫描（P0-2，2026-10-13）。

把「交易成本」从回测里的事后假设，变成 QP 目标函数里的一项。
``optimize.solver.solve_portfolio`` 早已支持 ``turnover_penalty``（线性冲击）与
``quadratic_cost``（二次冲击，A-C 型），**默认都是 0、从未校准**——本模块负责
「该填多少」，并给出可复现的推导与扫描入口。

理论起点（A 股冲击研报 + Almgren-Chriss 二次近似）
--------------------------------------------------
逐股冲击（元，见 ``backtest.costs.ImpactCostModel``）::

    Q_i·impact_frac_i = Q_i·η·σ_i·√(Q_i/ADV_i)

折成**收益单位**（除以 capital）：记单边权重变动 x_i = |Δw_i|，capital 为总资金::

    Q_i = x_i·capital
    cost_return_i(x) = x_i·impact_frac_i = c_i·x_i^{1.5},
        c_i = η·σ_i·√(capital/ADV_i)

``x^{1.5}`` 不是二次型，不能直接进 QP。用「在典型单边换手 x* 处**值与斜率双匹配**」
的线性+二次近似 ``a·x + b·x²``::

    a·x* + b·x*² = c·x*^{1.5}          (值匹配)
    a + 2b·x*    = 1.5·c·√x*           (斜率匹配)

两式相减 / 代入可得闭式解::

    a = 0.5·c·√x*,   b = 0.5·c/√x*   ⇒   **b = a / x***

即二次项系数 = 线性项系数 ÷ 典型单边换手。这给出成本惩罚的**形状**（唯一自由
参数是整体尺度 a）；a 的绝对尺度还取决于 ``solve_portfolio`` 目标里 α 项（rank pct）
与 w'Σw 的相对量级，必须**用本项目数据扫描定标**（见 :func:`scan_cost_aware`）。

⚠️ 复现边界：c_i 依赖 η/σ/ADV，其中 η 是研报经验值；本项目无真实成交数据，
产出的是「估计惩罚」不是「实测最优」。见
``reports/docs/design/链路优化调查_预测组合执行_2026-10-13.md`` §9.5。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from optimize.solver import optimize_weights_qp

__all__ = [
    "typical_impact_c",
    "calibrate_cost_penalty",
    "cost_aware_weights",
    "scan_cost_aware",
]


def typical_impact_c(
    adv,
    vol,
    capital: float,
    eta: float = 0.4,
) -> np.ndarray:
    """逐股冲击的比例常数 c_i = η·σ_i·√(capital/ADV_i)。

    使逐股收益口径的成本可写成 ``cost_return_i = c_i·|Δw_i|^{1.5}``。

    Args:
        adv: 逐股日均成交额（元）。
        vol: 逐股日波动率（小数）。
        capital: 组合总资金（元）。
        eta: 冲击系数。
    Returns:
        np.ndarray，与 adv/vol 同长；adv/vol 无效处为 nan（调用方 dropna）。
    """
    adv_arr = np.asarray(adv, dtype=float).reshape(-1)
    vol_arr = np.asarray(vol, dtype=float).reshape(-1)
    if adv_arr.shape != vol_arr.shape:
        raise ValueError(f"adv/vol 维度不一致: {adv_arr.shape} vs {vol_arr.shape}")
    out = np.full(adv_arr.shape, np.nan)
    valid = np.isfinite(adv_arr) & (adv_arr > 0) & np.isfinite(vol_arr) & (vol_arr > 0)
    out[valid] = eta * vol_arr[valid] * np.sqrt(float(capital) / adv_arr[valid])
    return out


def calibrate_cost_penalty(
    adv,
    vol,
    capital: float,
    x_star: float | None = None,
    eta: float = 0.4,
    n_names: int | None = None,
) -> dict:
    """由冲击参数推导 ``turnover_penalty``(a) / ``quadratic_cost``(b) 的理论起点。

    见模块 docstring 的推导：``a = 0.5·c̄·√x*``、``b = a/x*``，其中 c̄ 是逐股 c_i
    的均值（等权近似）。x* 缺省取等权单边换手 ``1/n``（n 为有效股票数）。

    Args:
        adv / vol: 逐股日均成交额（元）/ 日波动率。
        capital: 组合总资金（元）。
        x_star: 典型单边权重变动；None → 用 ``1/n``（等权换手）。
        eta: 冲击系数。
        n_names: x* 缺省时的股票数；None → 用 adv 中的有效样本数。

    Returns:
        dict(k1=a, k2=b, x_star, c_mean, n_valid)；adv/vol 全无效时 k1=k2=0。
    """
    c = typical_impact_c(adv, vol, capital, eta=eta)
    c_valid = c[np.isfinite(c)]
    if c_valid.size == 0:
        return {"k1": 0.0, "k2": 0.0, "x_star": float(x_star or 0.0),
                "c_mean": 0.0, "n_valid": 0}
    n = int(n_names) if n_names else int(c_valid.size)
    xs = float(x_star) if x_star is not None else (1.0 / max(n, 1))
    if xs <= 0:
        raise ValueError(f"x_star 须为正: {xs}")
    c_mean = float(c_valid.mean())
    a = 0.5 * c_mean * np.sqrt(xs)
    b = a / xs
    return {"k1": float(a), "k2": float(b), "x_star": xs,
            "c_mean": c_mean, "n_valid": int(c_valid.size)}


def _mean_valid(panel: pd.DataFrame | None, codes) -> np.ndarray | None:
    """把 date×code 面板在时间维上取均值，reindex 到 codes；None 透传。"""
    if panel is None:
        return None
    sub = panel.reindex(columns=codes)
    return sub.mean(axis=0).to_numpy(dtype=float)


def cost_aware_weights(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    *,
    adv_panel: pd.DataFrame,
    vol_panel: pd.DataFrame | None = None,
    capital: float,
    k1_scale: float = 1.0,
    x_star: float | None = None,
    eta: float = 0.4,
    window: int = 120,
    min_periods: int = 60,
    **qp_kwargs: Any,
) -> pd.DataFrame:
    """一步到位：由冲击参数自动定标成本惩罚，跑成本感知 QP。

    先用 :func:`calibrate_cost_penalty` 得到 (k1, k2)，再乘整体尺度 ``k1_scale``
    （k2 随 k1 同比例，保持 b = a/x* 的形状）后喂给
    :func:`optimize.solver.optimize_weights_qp`（``prev_weights`` 由后者在滚动中
    自动维护，故换手惩罚真正生效）。

    Args:
        factor_panel: date×code 因子（方向已对齐，越大越看好）。
        returns_panel: date×code 日收益（估协方差用）。
        adv_panel: date×code 日均成交额（元）；时间维均值用于定标。
        vol_panel: date×code 日波动率；None → 用 returns_panel 的 60 日滚动标准差。
        capital: 组合总资金（元），用于把冲击折成收益口径。
        k1_scale: 整体尺度倍率（1.0 = 理论起点；扫描时改它）。
        x_star / eta: 见 :func:`calibrate_cost_penalty`。
        window / min_periods: 协方差窗口。
        **qp_kwargs: 透传 ``optimize_weights_qp``（如 method、max_weight、budget）。
    Returns:
        DataFrame(date×code) 权重。
    """
    codes = factor_panel.columns
    if vol_panel is None:
        vol_panel = returns_panel.rolling(60, min_periods=20).std()
    adv_mean = _mean_valid(adv_panel, codes)
    vol_mean = _mean_valid(vol_panel, codes)
    if adv_mean is None or vol_mean is None:
        raise ValueError("adv_panel/vol_panel 不能为空")
    cal = calibrate_cost_penalty(adv_mean, vol_mean, capital, x_star=x_star, eta=eta,
                                 n_names=len(codes))
    k1 = cal["k1"] * float(k1_scale)
    k2 = cal["k2"] * float(k1_scale)
    return optimize_weights_qp(
        factor_panel,
        returns_panel,
        window=window,
        min_periods=min_periods,
        turnover_penalty=k1,
        quadratic_cost=k2,
        **qp_kwargs,
    )


def scan_cost_aware(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    *,
    adv_panel: pd.DataFrame,
    vol_panel: pd.DataFrame | None = None,
    capital: float,
    scales: list[float] | tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0),
    **kwargs: Any,
) -> dict[float, pd.DataFrame]:
    """在一组 ``k1_scale`` 上跑成本感知 QP，返回 ``{scale: 权重面板}``。

    ``scale=0`` 即「无成本惩罚」基线；调用方对各面板跑同一回测，比较
    净收益 / 换手，即可经验定标（理论起点只是形状，绝对尺度须扫描）。
    """
    scale_list = list(scales)
    out: dict[float, pd.DataFrame] = {}
    for s in scale_list:
        out[float(s)] = cost_aware_weights(
            factor_panel, returns_panel,
            adv_panel=adv_panel, vol_panel=vol_panel, capital=capital,
            k1_scale=float(s), **kwargs,
        )
    return out
