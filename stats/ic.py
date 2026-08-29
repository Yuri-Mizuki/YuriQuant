"""
因子 IC 统计
============

面板级因子检验的核心统计量（date×code 面板 → 标量/序列指标）：

- IC (Information Coefficient): 因子值与未来收益的截面相关系数
- IR (Information Ratio): IC 均值 / IC 标准差 × √PERIODS_PER_YEAR
- IC 衰减: 不同持有期的 IC 变化
- 分层回测: 按因子值分 N 组，看各组收益单调性
- 截面排名自相关: 换手率代理（factor_autocorr）

所有函数接收:
- factor_panel: DataFrame(date, code), 因子值
- returns_panel: DataFrame(date, code), 日收益率（未来一期口径）

真源历史：2026-08-29 自 research/factor_analysis.py 下沉（该模块保留
standard_factor_summary / factor_summary / calc_neutral_ic_series 等研究级
API 并对本模块 re-export）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats  # noqa: F401  （保留原模块 import 面，无直接使用）

from stats import PERIODS_PER_YEAR

__all__ = [
    "calc_ic_series", "calc_ir", "calc_ic_decay",
    "quantile_backtest", "factor_autocorr",
]


def calc_ic_series(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """计算每日 IC（因子值与未来收益的截面相关系数）。

    Args:
        factor_panel: DataFrame(date, code), 因子值。
        returns_panel: DataFrame(date, code), 未来一期收益。
        method: 'pearson' 或 'spearman'（默认，Rank IC）。
    Returns:
        Series(index=date), 每日 IC。

    **向量化实现（2026-08-03 优化）**：spearman IC 等价于对因子与收益做
    截面排名后再算 pearson 相关，因此用 ``rank(axis=1) + corrwith(axis=1)``
    一次向量化算出全部日期的 IC，替代原逐日 ``stats.spearmanr`` 循环
    （evaluate_candidates 全量评估 350 候选耗时 117s → 优化后大幅下降）。
    """
    common_dates = factor_panel.index.intersection(returns_panel.index)
    common_codes = factor_panel.columns.intersection(returns_panel.columns)
    fp = factor_panel.loc[common_dates, common_codes]
    rp = returns_panel.loc[common_dates, common_codes]

    if method == "spearman":
        # 只在因子与收益均有效的股票子集上做截面排名（与旧实现"先剔缺失再排名"
        # 严格一致，避免全截面排名带来的 rank 基准漂移），再 pearson = spearman。
        valid = fp.notna() & rp.notna()
        fr = fp.where(valid).rank(axis=1)
        rr = rp.where(valid).rank(axis=1)
        ic = fr.corrwith(rr, axis=1, method="pearson")
    else:
        ic = fp.corrwith(rp, axis=1, method="pearson")
    ic = ic.astype(float)

    # 与旧实现一致：有效观测 <5 的日期视为缺失
    valid_cnt = (fp.notna() & rp.notna()).sum(axis=1)
    ic[valid_cnt < 5] = np.nan
    ic.name = "ic"
    return ic


def calc_ir(ic_series: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """信息比率 = IC均值 / IC标准差 × √PERIODS_PER_YEAR。"""
    ic = ic_series.dropna()
    if len(ic) < 2 or ic.std() == 0:
        return 0.0
    return ic.mean() / ic.std() * np.sqrt(periods_per_year)


def calc_ic_decay(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    max_lag: int = 10,
) -> pd.Series:
    """IC 衰减: 持有 lag 期收益的 IC（lag=1 即次日，与主 IC 口径完全一致）。

    ``returns_panel`` 约定与 ``calc_ic_series`` 相同：**未来一期收益面板**
    （即 returns[d] = close[d+1]/close[d]-1，已前移一期）。因此:

        decay[lag] = corr( factor[t], 未来第 lag 期收益 ) = corr(factor[t], aligned[t+lag-1])

    早期实现内部再 shift(-lag)，把"未来一期"又前移一期，导致 decay[1] 实际是
    "未来第 2 天"（off-by-one）—— 已修复为 shift(-(lag-1))。
    """
    decay = {}
    for lag in range(1, max_lag + 1):
        shifted_returns = returns_panel.shift(-(lag - 1))
        ic = calc_ic_series(factor_panel, shifted_returns)
        decay[lag] = ic.mean()
    return pd.Series(decay, name="ic_decay")


def quantile_backtest(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """分层回测: 按因子值分 N 组，计算各组累计收益。

    **口径（2026-08-03 修复）**：与主回测引擎 / IC 完全一致 —— 当日因子
    ``factor[t]`` 赚当日（未来一期）收益 ``returns_panel[t]``。早期实现用
    ``factor[t-1]`` 赚 ``returns_panel[t]``，因子整体晚一天生效，分层单调性
    检验系统性丢失一天信号（与 IC/回测结论错位）。

    Returns:
        DataFrame(index=date, columns=quantile_1~N), 各组累计净值。
    """
    common_dates = factor_panel.index.intersection(returns_panel.index)
    common_codes = factor_panel.columns.intersection(returns_panel.columns)
    fp = factor_panel.loc[common_dates, common_codes]
    rp = returns_panel.loc[common_dates, common_codes]

    group_returns = pd.DataFrame(0.0, index=common_dates, columns=[f"Q{i+1}" for i in range(n_quantiles)])

    for i, date in enumerate(common_dates):
        f = fp.iloc[i].dropna()  # 当日因子
        r = rp.iloc[i]           # 当日（未来一期）收益，与 IC 同口径

        common = f.index.intersection(r.index)
        if len(common) < n_quantiles:
            continue
        f_aligned = f.loc[common]
        r_aligned = r.loc[common]

        try:
            groups = pd.qcut(f_aligned, n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        n_actual = groups.nunique()
        for g in range(n_actual):
            mask = groups == g
            if mask.sum() > 0:
                group_returns.iloc[i, g] = r_aligned[mask].mean()

    # 累计净值
    cum = (1 + group_returns).cumprod()
    return cum


def factor_autocorr(factor_panel: pd.DataFrame, max_lag: int = 1) -> float:
    """因子截面排名自相关（换手率代理，Alphalens 风格）。

    对相邻期计算因子排名的 spearman 相关并取均值。越接近 1 表示因子排序越稳定
    （换手越低、交易成本越小）；越接近 0 表示每日大换血（换手高、成本高）。
    无需跑回测即可估算因子本身的"粘性"，是 IC/IR 之外判断因子能否落地的关键维度。

    实现：先对每期做截面排名，再用向量化 corrwith(method='spearman') 算相邻期
    排名相关，避免逐日调 scipy 的性能问题。
    """
    ranked = factor_panel.rank(axis=1)
    vals = []
    for lag in range(1, max_lag + 1):
        prev = ranked.shift(lag)
        c = prev.corrwith(ranked, axis=1, method="spearman").dropna()
        if len(c):
            vals.append(float(c.mean()))
    return float(np.mean(vals)) if vals else 0.0
