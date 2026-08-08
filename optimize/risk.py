"""
风险归因 —— 03 优化层「风险归因」。

收编现有能力（research/attribution.py + research/benchmarks.py）：

- α/β 分解（CAPM / 多因子回归，NW 校正）：alpha_beta
- 基准对照指标（超额年化 / IR / 跟踪误差 / 相关 / beta）：compare_to_benchmark
- Brinson 归因（配置 / 选择 / 交互，Carino 链接）：brinson_attribution

TODO（待建）：组合级风险拆解（行业/风格暴露、VaR / CVaR 贡献、风险预算分解）。

口径约定（与 research/attribution.py 一致）：
- returns_panel 为 date×code **当期日收益**（未前移）。
- 组合收益序列可由权重×收益面板逐日合成得到。
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.attribution import alpha_beta, brinson_attribution
from research.benchmarks import compare_to_benchmark

__all__ = ["risk_attribution"]


def risk_attribution(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    factor_returns: pd.DataFrame | None = None,
    returns_panel: pd.DataFrame | None = None,
    port_weights: pd.DataFrame | None = None,
    bench_weights: pd.DataFrame | None = None,
    category: Mapping[str, str] | pd.Series | None = None,
    freq: str = "M",
) -> dict[str, Any]:
    """组合风险归因标准入口（03 优化层「风险归因」）。

    Args:
        portfolio_returns: 组合日收益 Series(index=date)。
        benchmark_returns: 基准（市场）日收益 Series。
        factor_returns: 可选额外因子收益 DataFrame(date, factor)，进入 α/β 回归。
        returns_panel: date×code 当期日收益（做 Brinson 时需要）。
        port_weights / bench_weights: date×code 组合/基准权重（Brinson 需要）。
        category: code → 类别名（Brinson 需要，如申万行业）。
        freq: Brinson 归因周期 'D' | 'W' | 'M'。

    Returns:
        dict: alpha_beta(dict) / benchmark(dict) / brinson(可选 (df, summary))。
    """
    out: dict[str, Any] = {
        "alpha_beta": alpha_beta(portfolio_returns, benchmark_returns,
                                 factor_returns=factor_returns),
        "benchmark": compare_to_benchmark(portfolio_returns, benchmark_returns),
    }
    if (
        returns_panel is not None
        and port_weights is not None
        and bench_weights is not None
        and category is not None
    ):
        out["brinson"] = brinson_attribution(
            returns_panel, port_weights, bench_weights, category, freq=freq,
        )
    else:
        out["brinson"] = None
        out["brinson_note"] = "传入 returns_panel + port/bench_weights + category 可启用 Brinson 归因"
    return out
