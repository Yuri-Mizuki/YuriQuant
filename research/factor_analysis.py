"""
因子分析
========

因子检验指标:
- IC (Information Coefficient): 因子值与未来收益的截面相关系数
- IR (Information Ratio): IC 均值 / IC 标准差 × √252
- IC 衰减: 不同持有期的 IC 变化
- 分层回测: 按因子值分 5 组，看各组收益单调性

所有函数接收:
- factor_panel: DataFrame(date, code), 因子值
- returns_panel: DataFrame(date, code), 日收益率
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from factor.preprocessing import neutralize, build_style_covariates


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


def calc_ir(ic_series: pd.Series, periods_per_year: int = 252) -> float:
    """信息比率 = IC均值 / IC标准差 × √252。"""
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
    from research.robust_stats import nw_tstat
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
