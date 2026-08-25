"""
风险归因 —— 03 优化层「风险归因」。

收编现有能力（research/attribution.py + research/benchmarks.py）：

- α/β 分解（CAPM / 多因子回归，NW 校正）：alpha_beta
- 基准对照指标（超额年化 / IR / 跟踪误差 / 相关 / beta）：compare_to_benchmark
- Brinson 归因（配置 / 选择 / 交互，Carino 链接）：brinson_attribution

组合级风险拆解（risk_decomposition）：

- 边际风险贡献 MRC + 成分风险贡献 CR + 占比（Euler 分解恒等式）
- 风格 / 行业因子方差贡献（B'ΣB 因子分解）
- VaR / CVaR 成分分解（历史模拟法）
- 风险预算校验（实际占比 vs 目标预算）

口径约定（与 research/attribution.py 一致）：
- returns_panel 为 date×code **当期日收益**（未前移）。
- 组合收益序列可由权重×收益面板逐日合成得到。
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from optimize.solver import _to_psd, estimate_covariance
from research.attribution import alpha_beta, brinson_attribution
from research.benchmarks import compare_to_benchmark

__all__ = ["risk_attribution", "risk_decomposition"]


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


# ===========================================================================
# 组合级风险分解
# ===========================================================================
def _ffill_weights(weights_history: pd.DataFrame) -> pd.DataFrame:
    """前向填充非调仓日权重。

    BacktestResult.weights_history 仅在调仓日有非零值，非调仓日为全 0
    （与 brinson_attribution:231 同口径处理）。
    """
    w = weights_history.copy()
    # 用 0 替换 NaN 后 ffill（调仓日之间的空行沿用上一次持仓）
    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    return w


def _component_var_cvar(
    weights: np.ndarray,
    returns: np.ndarray,
    var_quantile: float = 0.05,
) -> dict[str, Any]:
    """历史模拟法 VaR/CVaR 及成分分解。

    Args:
        weights: (N,) 组合权重向量。
        returns: (T, N) 个股收益历史矩阵（已对齐 weights 列）。
        var_quantile: 左尾分位（0.05 = 5% VaR）。

    Returns:
        dict: portfolio_VaR, portfolio_CVaR, component_VaR(Series),
              component_CVaR(Series), marginal_VaR(Series).
    """
    port_returns = returns @ weights  # (T,)
    T = len(port_returns)
    if T < 20:
        return {"portfolio_VaR": np.nan, "portfolio_CVaR": np.nan,
                "component_VaR": None, "component_CVaR": None,
                "marginal_VaR": None, "note": "样本不足（<20）"}

    var_p = np.percentile(port_returns, var_quantile * 100)
    tail_mask = port_returns <= var_p
    cvar_p = port_returns[tail_mask].mean() if tail_mask.any() else var_p

    # 成分 VaR（Epperlein 式）: VaR_k = w_k * E[r_k | r_p <= VaR]
    # 边际 VaR: ∂VaR/∂w_k = E[r_k | r_p <= VaR]
    marginal_var = np.zeros(len(weights))
    if tail_mask.any():
        marginal_var = returns[tail_mask].mean(axis=0)  # (N,)
    component_var = weights * marginal_var  # (N,)

    marginal_cvar = marginal_var  # 同口径（历史模拟下 ∂CVaR/∂w_k = E[r_k | r_p <= VaR]）
    component_cvar = weights * marginal_cvar

    return {
        "portfolio_VaR": float(var_p),
        "portfolio_CVaR": float(cvar_p),
        "marginal_VaR": marginal_var,
        "component_VaR": component_var,
        "marginal_CVaR": marginal_cvar,
        "component_CVaR": component_cvar,
    }


def risk_decomposition(
    weights_history: pd.DataFrame,
    returns_panel: pd.DataFrame,
    style_exposures: dict[str, pd.DataFrame] | None = None,
    industry_panel: pd.DataFrame | None = None,
    covariance_window: int = 120,
    min_periods: int = 60,
    cov_method: str = "ledoit_wolf",
    shrinkage: float = 0.5,
    var_quantile: float = 0.05,
    risk_budgets: dict[str, float] | None = None,
    freq: str = "M",
) -> dict[str, Any]:
    """组合级风险分解：暴露分解 + 方差贡献 + VaR/CVaR + 预算校验。

    基于 Euler 分解恒等式：σ²_p = w'Σw = Σ_k w_k (Σw)_k，
    边际风险贡献 MRC_k = (Σw)_k / σ_p，成分贡献 CR_k = w_k · MRC_k。

    Args:
        weights_history: date×code 持仓权重（BacktestResult.weights_history）。
            非调仓日为 0，内部会 ffill。
        returns_panel: date×code **当期日收益**（与回测口径一致）。
        style_exposures: {风格名: date×code 暴露面板}，如
            build_style_covariates 产出 {size/mom/vol/turn}。
            会在截面内做 zscore 统一量纲。
        industry_panel: date×code 行业分类面板（如申万一级）。
            用于按行业分组计算方差贡献。
        covariance_window: 协方差估计窗口（交易日）。
        min_periods: 协方差最小样本数。
        cov_method: 传给 estimate_covariance（"ledoit_wolf"/"shrinkage"）。
        shrinkage: 手动收缩强度（cov_method="shrinkage" 时生效）。
        var_quantile: VaR 左尾分位（0.05 = 5%）。
        risk_budgets: 目标风险预算 {名称: 占比}，如 {"行业A": 0.2, "风格": 0.3}。
            提供时校验实际贡献占比 vs 目标。
        freq: 汇总周期 'D'/'W'/'M'，用于逐期结果聚合。

    Returns:
        dict:
        - "exposure": DataFrame(date×[style/industry]), 组合逐期暴露
        - "variance_decomp": dict(总方差, 风格因子贡献, 行业贡献, 特质风险)
        - "risk_contributions": DataFrame(code×[MRC, CR, CR_pct])
        - "var_cvar": dict(portfolio_VaR, portfolio_CVaR, component_VaR, component_CVaR)
        - "budget_check": dict(实际占比 vs 目标, 偏离) — risk_budgets 提供时
        - "summary": dict, 汇总指标
    """
    # ── 对齐权重与收益 ──
    w = _ffill_weights(weights_history)
    common_codes = w.columns.intersection(returns_panel.columns)
    w = w[common_codes]
    r = returns_panel[common_codes]

    # ── 按 freq 聚合调仓截面 ──
    rebalance_dates = w.index[w.abs().sum(axis=1) > 1e-10]
    if freq != "D":
        # 取每个 freq 周期的最后一个调仓日
        s = pd.Series(rebalance_dates, index=rebalance_dates)
        grouped = s.groupby(pd.Grouper(freq="ME" if freq == "M" else freq))
        rebalance_dates = grouped.last().dropna().values
        rebalance_dates = pd.DatetimeIndex(rebalance_dates)

    results: list[dict] = []
    for dt in rebalance_dates:
        w_t = w.loc[dt]
        active = w_t.abs() > 1e-12
        if active.sum() < 2:
            continue

        w_vec = w_t[active].values
        codes = w_t[active].index
        r_sub = r[active.index[active]].loc[:dt]
        if len(r_sub) < 2:
            continue

        # ── 估计协方差（防前视：只用 < dt 的收益） ──
        hist = r_sub.iloc[:-1].tail(covariance_window).dropna(how="any")
        if len(hist) < min_periods:
            continue

        try:
            Sigma = estimate_covariance(hist, method=cov_method, shrinkage=shrinkage)
        except (ValueError, np.linalg.LinAlgError):
            continue
        Sigma = _to_psd(Sigma, eps=1e-6)

        # ── Euler 分解 ──
        port_var = float(w_vec @ Sigma @ w_vec)
        port_vol = np.sqrt(port_var) if port_var > 0 else 0.0
        mrc = (Sigma @ w_vec) / port_vol if port_vol > 0 else np.zeros(len(w_vec))
        cr = w_vec * mrc  # 成分风险贡献
        cr_sum = cr.sum()
        cr_pct = cr / cr_sum if abs(cr_sum) > 0 else np.zeros(len(w_vec))

        # ── 风格因子方差贡献 (B'ΣB 分解) ──
        # 口径：将风格暴露 b 归一化为单位向量 b̂ = b/||b||，
        # 组合在该因子上的暴露 = b̂'w，
        # 因子解释的方差 = (b̂'w)² × b̂'Σb̂（单因子回归 R² 的方差口径）
        style_contrib: dict[str, float] = {}
        if style_exposures:
            for name, panel in style_exposures.items():
                b_col = panel.reindex(columns=codes).loc[:dt]
                if b_col.empty:
                    continue
                b_vec = b_col.iloc[-1].values.astype(float)
                # 截面 zscore 统一量纲
                valid = ~np.isnan(b_vec)
                if valid.sum() < 3:
                    continue
                b_zscore = np.zeros_like(b_vec)
                b_zscore[valid] = (b_vec[valid] - b_vec[valid].mean()) / (
                    b_vec[valid].std() + 1e-12
                )
                # 归一化为单位向量
                norm = np.linalg.norm(b_zscore)
                if norm < 1e-12:
                    continue
                b_hat = b_zscore / norm
                # 因子暴露 = b̂'w
                factor_exposure = float(b_hat @ w_vec)
                # 因子方差 = b̂'Σb̂
                factor_var = float(b_hat @ Sigma @ b_hat)
                # 该风格因子解释的组合方差 = 暴露² × 因子方差
                style_contrib[name] = factor_exposure ** 2 * factor_var

        # ── 行业方差贡献（按行业分组求 CR 之和） ──
        industry_contrib: dict[str, float] = {}
        if industry_panel is not None:
            ind_col = industry_panel.reindex(columns=codes).loc[:dt]
            if not ind_col.empty:
                ind_t = ind_col.iloc[-1]
                for ind_name, group_codes in ind_t.groupby(ind_t).groups.items():
                    if pd.isna(ind_name):
                        continue
                    idx = [i for i, c in enumerate(codes) if c in set(group_codes)]
                    if idx:
                        industry_contrib[str(ind_name)] = float(cr[idx].sum())

        # ── VaR/CVaR 成分分解 ──
        ret_matrix = hist.values  # (T, N_active)
        vc = _component_var_cvar(w_vec, ret_matrix, var_quantile)

        results.append({
            "date": dt,
            "n_holdings": int(active.sum()),
            "portfolio_vol": port_vol,
            "portfolio_var": port_var,
            "codes": codes,
            "weights": w_vec,
            "mrc": mrc,
            "cr": cr,
            "cr_pct": cr_pct,
            "style_contrib": style_contrib,
            "industry_contrib": industry_contrib,
            "var_cvar": vc,
        })

    if not results:
        return {"error": "无有效截面（协方差样本不足或持仓过少）"}

    # ── 聚合 ──
    # 1) 逐期暴露
    exposure_rows = []
    for res in results:
        row = {"date": res["date"], "portfolio_vol": res["portfolio_vol"]}
        for name, val in res["style_contrib"].items():
            row[f"style_{name}"] = val
        for name, val in res["industry_contrib"].items():
            row[f"industry_{name}"] = val
        exposure_rows.append(row)
    exposure_df = pd.DataFrame(exposure_rows).set_index("date")

    # 2) 方差分解汇总
    avg_var = np.mean([r["portfolio_var"] for r in results])
    style_vars = {}
    for name in set(k for r in results for k in r["style_contrib"]):
        vals = [r["style_contrib"].get(name, 0.0) for r in results]
        style_vars[name] = float(np.mean(vals))
    industry_vars = {
        name: float(np.mean([r["industry_contrib"].get(name, 0.0) for r in results]))
        for name in set(k for r in results for k in r["industry_contrib"])
    }
    explained_var = sum(style_vars.values())
    idiosyncratic_var = max(avg_var - explained_var, 0.0)

    # 3) 最新截面的风险贡献详情
    latest = results[-1]
    cr_df = pd.DataFrame({
        "code": latest["codes"],
        "weight": latest["weights"],
        "MRC": latest["mrc"],
        "CR": latest["cr"],
        "CR_pct": latest["cr_pct"],
    }).set_index("code").sort_values("CR_pct", ascending=False)

    # 4) VaR/CVaR 汇总（最新截面）
    vc_latest = latest["var_cvar"]
    var_cvar_summary = {
        "portfolio_VaR": vc_latest.get("portfolio_VaR", np.nan),
        "portfolio_CVaR": vc_latest.get("portfolio_CVaR", np.nan),
    }
    if vc_latest.get("component_VaR") is not None:
        var_cvar_summary["component_VaR"] = pd.Series(
            vc_latest["component_VaR"], index=latest["codes"]
        ).sort_values()
        var_cvar_summary["component_CVaR"] = pd.Series(
            vc_latest["component_CVaR"], index=latest["codes"]
        ).sort_values()

    # 5) 风险预算校验
    budget_check = None
    if risk_budgets:
        total_industry_cr = sum(industry_vars.values())
        actual_pct = {}
        for name, budget_target in risk_budgets.items():
            actual = industry_vars.get(name, style_vars.get(name, 0.0))
            pct = actual / total_industry_cr if total_industry_cr > 0 else 0.0
            actual_pct[name] = {
                "target": budget_target,
                "actual": pct,
                "deviation": pct - budget_target,
            }
        budget_check = actual_pct

    # 6) 汇总指标
    summary = {
        "n_periods": len(results),
        "avg_portfolio_vol": float(np.mean([r["portfolio_vol"] for r in results])),
        "avg_portfolio_var": float(avg_var),
        "style_variance_contrib": style_vars,
        "industry_variance_contrib": industry_vars,
        "explained_variance_ratio": explained_var / avg_var if avg_var > 0 else 0.0,
        "idiosyncratic_variance": float(idiosyncratic_var),
        "idiosyncratic_ratio": idiosyncratic_var / avg_var if avg_var > 0 else 0.0,
        "var_quantile": var_quantile,
        "portfolio_VaR": var_cvar_summary["portfolio_VaR"],
        "portfolio_CVaR": var_cvar_summary["portfolio_CVaR"],
        "top5_risk_contributors": cr_df.head(5)["CR_pct"].to_dict(),
    }

    return {
        "exposure": exposure_df,
        "variance_decomp": {
            "total_variance": float(avg_var),
            "style_factor_contrib": style_vars,
            "industry_contrib": industry_vars,
            "explained_variance": float(explained_var),
            "idiosyncratic_risk": float(idiosyncratic_var),
        },
        "risk_contributions": cr_df,
        "var_cvar": var_cvar_summary,
        "budget_check": budget_check,
        "summary": summary,
    }
