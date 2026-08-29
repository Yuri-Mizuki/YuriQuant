"""
收益归因（Performance Attribution）
==================================

三大业内公认的归因框架：

1. **Brinson 归因**（Brinson & Fachler 1985，机构最常用的持仓归因）
   把组合相对基准的超额收益分解为三类决策效应（单期算术恒等式）:
       Allocation_j   = (w_p,j - w_b,j) × (r_b,j - R_b)      # 行业/类别配置
       Selection_j    = w_b,j × (r_p,j - r_b,j)              # 类别内选股
       Interaction_j  = (w_p,j - w_b,j) × (r_p,j - r_b,j)    # 配置×选股交互
       R_p - R_b      = Σ Allocation + Σ Selection + Σ Interaction
   多期用 **Carino (1999) 平滑**把各期效应链接到累计超额（单期效应跨期
   不能直接相加，业界标配是 Carino / Menchero / GRAP 链接）。

2. **Fama-MacBeth (1973) 两步回归**（因子溢价显著性检验的标准方法）
   第一步：每个时点 t 做横截面回归（收益 ~ 因子暴露），得到系数序列 β_t；
   第二步：对 β_t 序列做时间序列检验（均值 = 因子溢价，t 检验 + **Newey-West
   自相关校正**——业界一致认为 FM 第二步必须处理 β_t 的自相关，否则 t 值虚高）。

3. **α/β 分解**（CAPM / 多因子回归）
   r_p - r_f = α + β(r_m - r_f) + Σ β_k F_k + ε
   α 为未被系统性风险解释的超额（经理人能力），各因子贡献 = β_k × mean(F_k)。
   系数显著性同样用 Newey-West HAC 标准误（业内对日频回归的标配）。

口径约定（与项目其他模块一致）：
- 面板一律 date×code（DataFrame，index=date, columns=code）。
- ``fama_macbeth`` 的 returns_panel 用**未来一期收益**（factor[t] 预测 ret[t+1]，
  与 IC 口径一致）；``brinson_attribution`` 与 ``alpha_beta`` 用**当期收益**。
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from backtest.metrics import PERIODS_PER_YEAR
from stats.robust_stats import nw_tstat, ols_newey_west

__all__ = ["fama_macbeth", "brinson_attribution", "alpha_beta"]


# ===========================================================================
# Fama-MacBeth 两步回归
# ===========================================================================
def fama_macbeth(
    factor_panels: Mapping[str, pd.DataFrame] | pd.DataFrame,
    returns_panel: pd.DataFrame,
    controls: Mapping[str, pd.DataFrame] | None = None,
    add_intercept: bool = True,
    lag: int | None = None,
    min_obs: int = 10,
) -> pd.DataFrame:
    """Fama-MacBeth 两步回归：估计各因子（含控制变量）的**溢价**及其显著性。

    第一步：每个交易日做横截面 OLS —— returns[t] ~ 1 + factor[t] + controls[t]，
            得到每个变量的系数序列 β_t（"溢价时序"）。
    第二步：对 β_t 序列做时间序列检验：均值（premium）、普通 t 与
            **Newey-West 自相关稳健 t**（β_t 强自相关时普通 t 虚高）。

    Args:
        factor_panels: {因子名: date×code 面板}；也可传单个 DataFrame（视为一个因子）。
        returns_panel: date×code **未来一期收益**（与 IC 口径一致）。
        controls: 可选控制变量 {名: date×code}（如市值、行业哑变量）。
        add_intercept: 是否含截距项（α_t；FM 通常保留以吸收公共漂移）。
        lag: Newey-West 滞后截断，None → Andrews 规则自动选择。
        min_obs: 横截面有效观测下限。
    Returns:
        DataFrame(index=变量名), 列:
        premium / se_ols / t_ols / se_nw / t_nw / p_nw / mean_r2 / n_periods
    """
    if isinstance(factor_panels, pd.DataFrame):
        factor_panels = {"factor": factor_panels}
    all_panels: dict[str, pd.DataFrame] = dict(factor_panels)
    if controls:
        all_panels.update(controls)

    # 对齐网格
    idx = returns_panel.index
    cols = returns_panel.columns
    for p in all_panels.values():
        idx = idx.intersection(p.index)
        cols = cols.intersection(p.columns)

    names = list(all_panels.keys())
    if add_intercept:
        names = ["intercept"] + names
    coef_series: dict[str, list[float]] = {nm: [] for nm in names}
    dates_used: list = []
    r2_list: list[float] = []

    for d in idx:
        y = returns_panel.loc[d, cols].astype(float)
        Xcols: list[np.ndarray] = []
        valid = y.notna()
        for nm in names:
            if nm == "intercept":
                Xcols.append(np.ones(len(cols)))
            else:
                v = all_panels[nm].loc[d, cols].astype(float)
                Xcols.append(v.values)
                valid &= v.notna()
        if valid.sum() < max(min_obs, len(names) + 2):
            continue
        Xm = np.column_stack(Xcols)[valid.values]
        ym = y.values[valid.values]
        beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        for nm, b in zip(names, beta):
            coef_series[nm].append(float(b))
        # R²（含截距时）
        ybar = ym.mean()
        ss_tot = float(np.sum((ym - ybar) ** 2))
        ss_res = float(np.sum((ym - Xm @ beta) ** 2))
        r2_list.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
        dates_used.append(d)

    if not dates_used:
        return pd.DataFrame(columns=[
            "premium", "se_ols", "t_ols", "se_nw", "t_nw", "p_nw", "mean_r2", "n_periods",
        ])

    rows = {}
    for nm in names:
        s = pd.Series(coef_series[nm], index=dates_used)
        n = len(s)
        mean = float(s.mean())
        sd = float(s.std())
        t_ols = mean / (sd / np.sqrt(n)) if sd > 0 else 0.0
        se_ols = sd / np.sqrt(n) if n > 1 else 0.0
        t_nw, se_nw, _lag = nw_tstat(s, lag=lag)
        p_nw = 2.0 * (1.0 - stats.t.cdf(abs(t_nw), df=max(n - 1, 1)))
        rows[nm] = {
            "premium": mean, "se_ols": se_ols, "t_ols": t_ols,
            "se_nw": se_nw, "t_nw": t_nw, "p_nw": p_nw,
            "mean_r2": float(np.nanmean(r2_list)) if r2_list else np.nan,
            "n_periods": n,
        }
    return pd.DataFrame(rows).T


# ===========================================================================
# Brinson 归因（Brinson-Fachler + Carino 链接）
# ===========================================================================
def _period_returns(
    returns_panel: pd.DataFrame,
    weights: pd.Series,
    cat_map: Mapping[str, str],
) -> tuple[float, pd.Series]:
    """给定期初权重（Series(code)），计算该持有期组合总收益与各类别收益。

    假设期内权重固定不变（买入持有口径，Brinson 单期前提）。
    **算术口径（2026-08-03 修复）**：期收益 = 期内日收益简单加总（Σ r_t），
    而非复利连乘 —— Brinson 单期恒等式 R_p = Σ_j w_p,j·r_p,j 是**算术关系**，
    复利交叉项会使恒等式不成立（月度多日 recon_error ≠ 0）。Carino 链接
    负责把各期算术效应链接到累计几何超额，单期本身保持算术。
    Returns:
        (组合总收益, 类别收益 Series(index=category))，均为期内算术加总。
    """
    w = weights.reindex(returns_panel.columns).fillna(0.0)
    # 组合日收益 = Σ_i w_i r_t,i，期内简单加总（算术收益）
    daily_port = returns_panel.mul(w, axis=1).sum(axis=1, min_count=1)
    total = float(daily_port.fillna(0.0).sum())

    # 类别收益：类别内权重归一后，日收益加权加总（算术）
    codes = list(returns_panel.columns)
    cat_ret: dict[str, float] = {}
    for c in sorted({cat_map[code] for code in codes}):
        in_cat = [code for code in codes if cat_map[code] == c]
        wc = float(w[in_cat].sum())
        if wc <= 0:
            cat_ret[c] = 0.0
            continue
        wc_norm = w[in_cat] / wc
        daily_cat = returns_panel[in_cat].mul(wc_norm, axis=1).sum(axis=1, min_count=1)
        cat_ret[c] = float(daily_cat.fillna(0.0).sum())
    return total, pd.Series(cat_ret)


def _carino_k(r_p: float, r_b: float) -> float:
    """Carino (1999) 单期平滑系数；|R_p - R_b| 极小时退化为 1/(1+R_p)。"""
    if abs(r_p - r_b) < 1e-12:
        return 1.0 / (1.0 + r_p) if (1 + r_p) > 0 else 1.0
    return (np.log1p(r_p) - np.log1p(r_b)) / (r_p - r_b)


def brinson_attribution(
    returns_panel: pd.DataFrame,
    port_weights: pd.DataFrame,
    bench_weights: pd.DataFrame,
    category: Mapping[str, str] | pd.Series,
    freq: str = "M",
    min_obs: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Brinson-Fachler 收益归因（多期 + Carino 链接）。

    把组合相对基准的超额收益分解为：配置效应 / 选择效应 / 交互效应。

    **算术口径（2026-08-03 修复）**：Brinson 是算术归因框架 —— 单期收益
    （组合 / 类别）为期内日收益**简单加总**，单期恒等式
    R_p = Σ_j w_p,j·r_p,j 严格成立（recon_error ≈ 0）；Carino 链接再把各期
    算术效应映射到累计几何超额。早期实现用复利期收益，月度多日下恒等式
    不成立（recon_error ≠ 0，效应数值被复利交叉项污染）。summary 中的
    portfolio_return / benchmark_return 均为算术口径（与回测净值的复利口径
    不同，属 Brinson 框架固有特征）。

    Args:
        returns_panel: date×code **当期日收益**（未前移）。
        port_weights:  date×code 组合期初权重（可只在调仓日有值，内部前向填充；
                       每个归因周期取该周期首个交易日的权重作为期初仓位）。
        bench_weights: date×code 基准期初权重（同口径，须与组合同类别划分）。
        category:      code → 类别名（互斥且完备；如申万行业）。
        freq:          归因周期 'D' | 'W' | 'M'（默认 'M'，月频归因）。
        min_obs:       周期内最少交易日，不足则跳过该期。
    Returns:
        (attribution_df, summary)：
        - attribution_df: DataFrame(index=类别名+TOTAL, columns=[allocation,
          selection, interaction, total])，均为 Carino 链接后的累计效应（小数）。
        - summary: dict，含 portfolio_return / benchmark_return / active_return /
          allocation / selection / interaction / n_periods / recon_error
          （收益均为算术口径，recon_error ≈ 0）。
    """
    if isinstance(category, pd.Series):
        category = category.to_dict()

    # 对齐网格
    idx = returns_panel.index.intersection(port_weights.index).intersection(bench_weights.index)
    cols = returns_panel.columns.intersection(port_weights.columns).intersection(bench_weights.columns)
    cols = [c for c in cols if c in category]
    rp = returns_panel.loc[idx, cols]
    wp = port_weights.loc[idx, cols].ffill()
    wb = bench_weights.loc[idx, cols].ffill()

    cats = sorted(set(category[c] for c in cols))
    # 类别映射 Series（code → category），挂到 weights 上供 _period_returns 使用
    cat_series = pd.Series({c: category[c] for c in cols}, name="cat")

    periods = pd.Series(idx).groupby(idx.to_period(freq)).groups  # period → positions
    # 每期效应向量（已乘 Carino 单期系数 K_t，最后统一除以总系数 K_T 完成链接）
    per_alloc: dict[str, list[float]] = {c: [] for c in cats}
    per_sel: dict[str, list[float]] = {c: [] for c in cats}
    per_int: dict[str, list[float]] = {c: [] for c in cats}
    rp_periods: list[float] = []
    rb_periods: list[float] = []

    for per, pos in periods.items():
        pos = sorted(pos)
        if len(pos) < min_obs:
            continue
        d0 = idx[pos[0]]
        w_p0 = wp.loc[d0].fillna(0.0)
        w_b0 = wb.loc[d0].fillna(0.0)
        if w_p0.notna().sum() < 1 or w_b0.notna().sum() < 1:
            continue

        sub = rp.iloc[pos]
        R_p, cat_ret_p = _period_returns(sub, w_p0, cat_series.to_dict())
        R_b, cat_ret_b = _period_returns(sub, w_b0, cat_series.to_dict())
        w_p_cat = w_p0.groupby(cat_series).sum().reindex(cats).fillna(0.0)
        w_b_cat = w_b0.groupby(cat_series).sum().reindex(cats).fillna(0.0)
        r_p_cat = cat_ret_p.reindex(cats).fillna(0.0)
        r_b_cat = cat_ret_b.reindex(cats).fillna(0.0)
        # 组合未持有类别（w_p,j <= 0）时，r_p,j 退回基准类别收益 r_b,j：
        # 无持仓 = 无选股决策 → selection/interaction 为 0，仅保留配置效应。
        # （早期实现把 r_p,j 当 0，制造虚假 selection = -w_b,j·r_b,j）
        no_hold = w_p_cat <= 0
        r_p_cat = r_p_cat.mask(no_hold, r_b_cat)
        k = _carino_k(R_p, R_b)

        for c in cats:
            per_alloc[c].append((w_p_cat[c] - w_b_cat[c]) * (r_b_cat[c] - R_b) * k)
            per_sel[c].append(w_b_cat[c] * (r_p_cat[c] - r_b_cat[c]) * k)
            per_int[c].append((w_p_cat[c] - w_b_cat[c]) * (r_p_cat[c] - r_b_cat[c]) * k)
        rp_periods.append(R_p)
        rb_periods.append(R_b)

    if not rp_periods:
        empty = pd.DataFrame(0.0, index=cats + ["TOTAL"],
                             columns=["allocation", "selection", "interaction", "total"])
        return empty, {"n_periods": 0}

    # Carino 链接：累计效应 = Σ_t (K_t / K_T) × effect_t
    R_pT = float(np.prod([1 + x for x in rp_periods]) - 1.0)
    R_bT = float(np.prod([1 + x for x in rb_periods]) - 1.0)
    K_T = _carino_k(R_pT, R_bT)

    rows = {}
    tot_a = tot_s = tot_i = 0.0
    for c in cats:
        a = float(np.sum(per_alloc[c]) / K_T)
        s = float(np.sum(per_sel[c]) / K_T)
        i = float(np.sum(per_int[c]) / K_T)
        rows[c] = {"allocation": a, "selection": s, "interaction": i, "total": a + s + i}
        tot_a += a
        tot_s += s
        tot_i += i
    rows["TOTAL"] = {"allocation": tot_a, "selection": tot_s, "interaction": tot_i,
                     "total": tot_a + tot_s + tot_i}
    df = pd.DataFrame(rows).T

    active = R_pT - R_bT
    summary = {
        "portfolio_return": R_pT,
        "benchmark_return": R_bT,
        "active_return": active,
        "allocation": tot_a,
        "selection": tot_s,
        "interaction": tot_i,
        "n_periods": len(rp_periods),
        "recon_error": active - (tot_a + tot_s + tot_i),
    }
    return df, summary


# ===========================================================================
# α/β 分解（CAPM / 多因子回归，Newey-West 推断）
# ===========================================================================
def alpha_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: float = 0.0,
    factor_returns: pd.DataFrame | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
    lag: int | None = None,
) -> dict:
    """α/β 分解：r_p - r_f = α + β(r_m - r_f) + Σ β_k F_k + ε。

    Args:
        portfolio_returns: 组合日收益 Series(index=date)。
        benchmark_returns: 基准（市场）日收益 Series。
        rf: 年化无风险利率（按日频折算为超额）。
        factor_returns: 可选额外因子收益 DataFrame(date, factor)（已对齐日期）。
        periods_per_year: 年化周期数（α 年化用）。
        lag: Newey-West 滞后截断，None → 自动。
    Returns:
        dict: alpha / alpha_annual / alpha_t_nw / alpha_p_nw / beta / beta_t_nw /
              factors(DataFrame: coef, t_nw, p_nw, contribution) / r2 / n / lag_used。
    """
    rfd = (1 + rf) ** (1 / periods_per_year) - 1 if rf > 0 else 0.0
    p_ex = (1 + portfolio_returns) / (1 + rfd) - 1
    m_ex = (1 + benchmark_returns) / (1 + rfd) - 1

    dates = p_ex.index.intersection(m_ex.index)
    X = np.column_stack([np.ones(len(dates)), m_ex.loc[dates].values])
    fnames: list[str] = []
    if factor_returns is not None:
        fdf = factor_returns.loc[dates]
        fnames = list(fdf.columns)
        X = np.column_stack([X, fdf.values])
    y = p_ex.loc[dates].values

    res = ols_newey_west(X, y, lag=lag)
    beta = res["beta"]
    n = res["n"]

    alpha = float(beta[0]) if n else np.nan
    beta_m = float(beta[1]) if n else np.nan
    alpha_ann = (1 + alpha) ** periods_per_year - 1 if n and not np.isnan(alpha) else np.nan

    factors = pd.DataFrame(columns=["coef", "t_nw", "p_nw", "contribution"])
    if fnames:
        fmeans = fdf.mean().values
        factors = pd.DataFrame({
            "coef": beta[2:],
            "t_nw": res["t_nw"][2:],
            "p_nw": res["p_nw"][2:],
            "contribution": beta[2:] * fmeans,
        }, index=fnames)

    # R²
    if n:
        resid = y - X @ beta
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    else:
        r2 = np.nan

    return {
        "alpha": alpha,
        "alpha_annual": alpha_ann,
        "alpha_t_nw": float(res["t_nw"][0]) if n else np.nan,
        "alpha_p_nw": float(res["p_nw"][0]) if n else np.nan,
        "beta": beta_m,
        "beta_t_nw": float(res["t_nw"][1]) if n else np.nan,
        "factors": factors,
        "r2": r2,
        "n": n,
        "lag_used": res["lag_used"],
    }
