"""
因子分析（研究层）
==================

单因子检验的研究级 API：

- ``standard_factor_summary`` : Alphalens 式单因子标准摘要（IC/NW-t/衰减/自相关一体）
- ``factor_summary``          : 因子检验汇总报告（含分层回测净值）
- ``calc_neutral_ic_series``  : 风格中性化后的纯 Alpha IC（依赖 factor.preprocessing）

核心统计量（calc_ic_series / calc_ir / calc_ic_decay / quantile_backtest /
factor_autocorr）已于 2026-08-29 下沉至 ``stats/ic.py`` 并在此 re-export——
factor/model/optimize/monitoring 等核心包请直接从 stats 导入，勿再经本模块
（此前 factor→research 反向依赖造成包级循环）。

所有函数接收:
- factor_panel: DataFrame(date, code), 因子值
- returns_panel: DataFrame(date, code), 日收益率
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from factor.preprocessing import neutralize, build_style_covariates
from stats.ic import (  # noqa: F401  re-export：保持历史 import 路径可用
    calc_ic_decay,
    calc_ic_series,
    calc_ir,
    factor_autocorr,
    quantile_backtest,
)

__all__ = [
    "calc_ic_series", "calc_ir", "calc_ic_decay",
    "quantile_backtest", "factor_autocorr",
    "standard_factor_summary", "factor_summary", "calc_neutral_ic_series",
]


def standard_factor_summary(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    decay_lags: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Alphalens 式单因子标准摘要（业界通用的"因子体检表"）。

    一张表给出所有核心指标，便于跨因子统一比较：
    - ic_mean / ic_std / ir / t_stat / p_value / ic_win_rate：预测力与显著性
    - t_stat_nw / p_value_nw / nw_lag：**Newey-West 自相关稳健**显著性
      （IC 序列强自相关时 OLS t 会虚高，NW 版本才是可信的统计推断）
    - ic_decay：不同持有期 IC（信号持久度）
    - autocorr：截面排名自相关（换手率代理）
    """
    ic = calc_ic_series(factor_panel, returns_panel)
    ic_valid = ic.dropna()
    n = len(ic_valid)
    ic_mean = float(ic_valid.mean()) if n else float("nan")
    ic_std = float(ic_valid.std()) if n else float("nan")
    ir = calc_ir(ic)
    t = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 and n > 0 else 0.0
    p = 2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 1)) if n > 1 else float("nan")
    # Newey-West 自相关稳健推断（业界标准：Andrews 1991 带宽）
    from stats.robust_stats import nw_tstat
    t_nw, _se_nw, nw_lag = nw_tstat(ic_valid) if n > 1 else (0.0, 0.0, 0)
    p_nw = 2.0 * (1.0 - stats.t.cdf(abs(t_nw), df=max(n - 1, 1))) if n > 1 else float("nan")
    decay = calc_ic_decay(factor_panel, returns_panel, max_lag=max(decay_lags))
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ir": ir,
        "t_stat": t,
        "p_value": p,
        "t_stat_nw": t_nw,
        "p_value_nw": p_nw,
        "nw_lag": nw_lag,
        "ic_win_rate": float((ic_valid > 0).mean()) if n else float("nan"),
        "ic_decay": {int(k): float(v) for k, v in decay.items()},
        "autocorr": factor_autocorr(factor_panel),
        "n": n,
    }


def factor_summary(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
) -> dict:
    """因子检验汇总报告。"""
    ic = calc_ic_series(factor_panel, returns_panel)
    ir = calc_ir(ic)
    decay = calc_ic_decay(factor_panel, returns_panel, max_lag=5)
    layers = quantile_backtest(factor_panel, returns_panel, n_quantiles=5)

    return {
        "ic_mean": ic.mean(),
        "ic_std": ic.std(),
        "ic_win_rate": (ic > 0).mean(),
        "ir": ir,
        "ic_decay": decay.to_dict(),
        "layer_returns": layers.iloc[-1].to_dict(),
        "ic_series": ic,
        "layer_nav": layers,
    }


# ===========================================================================
# 中性化 IC（风格剥离后的纯 Alpha IC）
# ===========================================================================
def calc_neutral_ic_series(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    style_covariates: dict[str, pd.DataFrame] | None = None,
    panel: dict[str, pd.DataFrame] | None = None,
    market_cap_panel: pd.DataFrame | None = None,
    industry_panel: pd.DataFrame | None = None,
    method: str = "spearman",
) -> pd.Series:
    """计算中性化 IC：先对因子做风格中性化取残差，再算 Rank IC。

    中性化流程（逐日截面）::

        因子_raw = β₀ + β₁·Size + β₂·Industry + β₃·Mom + β₄·Vol + β₅·Turn + ε
        IC_neutral = Spearman(ε, 未来收益)

    风格因子来源（优先级递减）：
    1. ``style_covariates``：调用方直接传入的协变量 dict（最灵活）
    2. ``panel`` + ``market_cap_panel`` + ``industry_panel``：从原始面板自动构建
       （调用 build_style_covariates，华泰五因子口径）
    3. 两者都没有：退化为 raw IC（不中性化，等于 calc_ic_series）

    Args:
        factor_panel: DataFrame(date, code), 因子值。
        returns_panel: DataFrame(date, code), 未来一期收益。
        style_covariates: 风格协变量面板 dict，key 为 ``size``/``industry``/``mom``/
            ``vol``/``turn``（或任意子集）。优先使用。
        panel: 原始 OHLCV 面板 dict（含 close/volume 等），用于自动构建协变量。
        market_cap_panel: 市值面板（date×code）。
        industry_panel: 行业面板（date×code，值=行业名）。
        method: IC 计算方法，默认 'spearman'（Rank IC）。

    Returns:
        Series(index=date), 每日中性化 IC。name='ic_neutral'。
    """
    # 确定协变量来源
    covariates = style_covariates
    if covariates is None:
        if panel is not None:
            covariates = build_style_covariates(
                panel,
                market_cap_panel=market_cap_panel,
                industry_panel=industry_panel,
            )
        elif market_cap_panel is not None or industry_panel is not None:
            # 没有完整 panel 但有市值/行业面板——只做 2 因子中性化
            covariates = {}
            if market_cap_panel is not None:
                covariates["size"] = market_cap_panel
            if industry_panel is not None:
                covariates["industry"] = industry_panel

    if not covariates:
        # 无协变量——退化为 raw IC
        ic = calc_ic_series(factor_panel, returns_panel, method=method)
        ic.name = "ic_neutral"
        return ic

    # 从 covariates 中拆分出 industry（特殊处理：行业面板是分类变量，走 industry_panel 参数）
    industry_cov = covariates.get("industry")
    size_cov = covariates.get("size")

    # 其余连续协变量（mom/vol/turn 等）
    extra_cov: dict[str, pd.DataFrame] = {}
    for k, v in covariates.items():
        if k not in ("industry", "size"):
            extra_cov[k] = v

    # 逐日截面回归取残差
    residual_panel = neutralize(
        factor_panel,
        market_cap_panel=size_cov,
        industry_panel=industry_cov,
        extra_covariates=extra_cov if extra_cov else None,
    )

    # 对残差算 IC
    ic = calc_ic_series(residual_panel, returns_panel, method=method)
    ic.name = "ic_neutral"
    return ic
