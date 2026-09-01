"""
组合优化 —— 03 优化层「组合优化」求解器版（P0，2026-08-14）。

在 optimize_weights（启发式投影）基础上新增**带求解器的精确优化**：

1. 协方差估计：
   - rolling_covariance：滚动窗口估计，**防前视**（只用调仓日及以前数据）。
   - estimate_covariance：Ledoit-Wolf 收缩（sklearn，自动收缩强度）；
     高维（N 接近/超过 T）时 fallback 向对角手动收缩，保证半正定。
2. 求解器优化：
   - solve_portfolio：cvxpy QP 单截面求解。
   - optimize_weights_qp：面板级入口（与 optimize_weights 同签名风格、同输出约定）。
   - 目标：最小方差 min w'Σw / TEV min (w-wb)'Σ(w-wb) − λ·α'w / MVO min w'Σw − λ·α'w。
   - 约束（线性、**精确满足**，优于投影法的近似）：
     预算 sum(w)=budget、个股上限 w≤max、行业中性（等式）、
     换手（|w−w_prev| 线性化，单边换手口径与回测一致 0.5·Σ|Δw|）。
   - 可选换手惩罚（L1 进目标，对应线性冲击成本）。

与启发式投影的定位：
- 本模块是「约束精确 + 风险显式权衡」的正式路径（业界 TEV+风险模型的最小闭环）。
- optimize_weights（启发式投影）保留为无 cvxpy 环境的基线（mock venv 3.13 未装 cvxpy）。

TODO（P2）：
- min_weight 为非凸约束，当前做近似后处理
- 行业偏离限制（相对基准 ±x%）已支持（industry_deviation）；风格中性化（Barra 风格
  因子暴露）已支持（style_exposures）；风险平价（method="risk_parity"）与
  HRP（hrp_weights / optimize_weights_hrp）已落地
- 多空、成本冲击模型（Almgren-Chriss）、Black-Litterman 观点融合

依赖：cvxpy（系统 python 3.12 已装 1.9.2；延迟导入，缺失时给出清晰报错）。
HRP 仅需 scipy（项目已有）。
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

try:  # 延迟导入：mock venv 3.13 无 cvxpy，真实模式（系统 python 3.12）已装
    import cvxpy as cp

    _HAS_CVXPY = True
except Exception:  # pragma: no cover - 环境探测
    cp = None
    _HAS_CVXPY = False

__all__ = [
    "estimate_covariance",
    "rolling_covariance",
    "solve_portfolio",
    "optimize_weights_qp",
    "bl_posterior",
    "bl_views_from_factor",
    "hrp_weights",
    "optimize_weights_hrp",
]


# ===========================================================================
# 协方差估计
# ===========================================================================
def _to_psd(S: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """对称化 + 对角正则，保证 cvxpy quad_form 可用的半正定矩阵。

    eps 用 1e-6（绝对）：日频 Σ 元素 ~1e-4，若正则过小（如 1e-10），
    高维真实数据（382 只）特征值 ~1e-10 → 条件数爆炸，OSQP 数值崩溃
    产出 NaN（2026-08-14 真实四窗口踩坑）。
    """
    S = (S + S.T) / 2.0
    return S + eps * np.eye(S.shape[0])


def estimate_covariance(
    returns_sub: pd.DataFrame,
    method: str = "ledoit_wolf",
    shrinkage: float = 0.5,
) -> np.ndarray:
    """由窗口内收益（date×code）估计 N×N 协方差（半正定）。

    Args:
        returns_sub: 窗口内日收益面板（index=date, columns=code），需已去缺失。
        method: "ledoit_wolf"（sklearn 自动收缩强度，样本数需大于特征数）或
                "shrinkage"（向对角矩阵手动收缩，高维兜底）。
        shrinkage: method="shrinkage" 时的收缩强度 δ（0=不收缩，1=完全向对角）。
    Returns:
        np.ndarray (N, N) 半正定协方差矩阵。
    """
    X = returns_sub.dropna(how="any").values  # (T, N)
    if X.shape[0] < 2:
        raise ValueError(f"窗口有效样本不足（{X.shape[0]} < 2）")
    if method == "ledoit_wolf" and X.shape[0] >= X.shape[1] + 2:
        try:
            from sklearn.covariance import LedoitWolf

            return _to_psd(LedoitWolf().fit(X).covariance_)
        except Exception:
            pass  # 高维/数值问题 → 走手动收缩兜底
    S = np.nan_to_num(np.cov(X, rowvar=False), nan=0.0)
    target = np.diag(np.diag(S))  # 对角目标（保留各自波动，抹掉联动噪声）
    return _to_psd(shrinkage * target + (1.0 - shrinkage) * S)


def rolling_covariance(
    returns_panel: pd.DataFrame,
    date,
    window: int = 120,
    min_periods: int = 60,
    method: str = "ledoit_wolf",
    shrinkage: float = 0.5,
) -> np.ndarray | None:
    """调仓日 date 的协方差（防前视）。

    returns_panel 按回测口径理解：`returns_panel[t]` 是 t→t+1 的收益，
    **在 t 日收盘时未知**。因此 Σ_t 只用 `returns_panel.loc[:date]` 中
    **不含 date 当日**（即 < date）的最近 window 行。

    窗口内共同交易日不足 min_periods 时返回 None（外层应跳过该截面，
    避免用不充分样本估计风险）。
    """
    hist = returns_panel.loc[:date]
    if len(hist) <= 1:  # 首日无历史收益可用
        return None
    sub = hist.iloc[:-1].tail(window).dropna(how="any")
    if len(sub) < min_periods:
        return None
    return estimate_covariance(sub, method=method, shrinkage=shrinkage)


# ===========================================================================
# 求解器优化
# ===========================================================================
def _alpha_score(alpha: pd.Series) -> pd.Series:
    """把因子方向对齐后的原始值转为 0..1 的 rank 分数（与 factor_weighted 同语义，
    对异常稳健，NaN 股票保持 NaN）。"""
    return alpha.rank(pct=True)


def _filter_min_weight(w: pd.Series, min_weight: float) -> pd.Series:
    """min_weight 近似后处理：微仓清零后按多空组独立重新归一（非凸约束无法精确进 QP）。

    多空口径与回测引擎 _apply_executable_mask 一致：多头组（w>0）与空头组（w<0）
    各自归一，保持净敞口结构。注：行业/预算等式约束可能被轻微破坏（近似）。
    """
    small = ((w > 0) & (w < min_weight)) | ((w < 0) & (w > -min_weight))
    if not small.any():
        return w
    out = w.copy()
    out[small] = 0.0
    pos, neg = out > 0, out < 0
    if pos.any():
        s = out[pos].sum()
        if s > 0:
            out[pos] /= s
    if neg.any():
        s = out[neg].sum()
        if s < 0:
            out[neg] /= abs(s)
    return out


def solve_portfolio(
    alpha: pd.Series,
    Sigma: np.ndarray,
    method: str = "min_var",
    risk_aversion: float = 1.0,
    benchmark: pd.Series | None = None,
    max_weight: float | None = None,
    min_weight: float | None = None,
    industry_map: Mapping[str, str] | pd.Series | None = None,
    industry_target: Mapping[str, float] | pd.Series | None = None,
    industry_deviation: float | None = None,
    style_exposures: pd.DataFrame | None = None,
    style_tolerance: float = 1e-5,
    prev_weights: pd.Series | None = None,
    max_turnover: float | None = None,
    turnover_penalty: float = 0.0,
    quadratic_cost: float = 0.0,
    budget: float = 1.0,
    allow_short: bool = False,
    short_limit: float | None = None,
    gross_limit: float | None = None,
    views: dict | None = None,
    market_weights: pd.Series | None = None,
    tau: float = 0.05,
    delta: float = 2.5,
) -> pd.Series:
    """cvxpy QP 单截面求解（P2：BL 观点融合、多空、A-C 成本惩罚）。

    Args:
        alpha: Series(index=code) 因子值（方向已对齐，越大越看好）；NaN = 不可持仓。
        Sigma: (N, N) 半正定协方差（estimate_covariance / rolling_covariance 产出）。
        method: "min_var" | "tev"（需 benchmark）| "mvo" | "risk_parity"
                | "bl"（Black-Litterman：均衡先验 + 观点 → 后验 μ_BL 进目标）。
        risk_aversion: λ，越大越保守。
        benchmark: tev 的基准权重 w_b。
        max_weight / min_weight: 个股上限 / 微仓过滤（非凸 → 近似后处理）。
        industry_map / industry_target / industry_deviation: 行业约束（等式或相对偏离区间）。
        style_exposures: code×style 暴露矩阵（列已 zscore），|B'w| ≤ tol 风格中性化。
        style_tolerance: 风格中性化容忍度。
        prev_weights: 上一期权重。
        max_turnover: 单边换手率硬约束（0.5·Σ|Δw| ≤ max_turnover，回测口径）。
        turnover_penalty: 线性冲击成本系数 κ₁·Σ|Δw|（进目标）。
        quadratic_cost: 二次冲击成本系数 κ₂·ΣΔw²（进目标；与线性项组成
            Almgren-Chriss 风格成本惩罚——完整多期最优执行留 P3）。
        budget: 净多头预算（默认 1.0；allow_short 时 w 可负，净敞口=budget）。
        allow_short: 是否允许做空（P2 完整支持）。
        short_limit: 空头总权重上限（|Σw<0| ≤ short_limit）。
        gross_limit: 总杠杆上限（Σ|w| ≤ gross_limit）。
        views: BL 观点 dict（见 bl_posterior）；method="bl" 时生效，None 则纯均衡。
        market_weights: BL 市场权重（反向优化均衡收益用，默认等权）。
        tau / delta: BL 先验标度 / 风险厌恶。
    Returns:
        Series(index=code) 最优权重；不可持仓（alpha NaN）股票恒为 0。
    """
    if not _HAS_CVXPY:
        raise RuntimeError(
            "需要 cvxpy：真实模式请用系统 python 3.12（已装 1.9.2），"
            "mock venv 3.13 未安装 cvxpy"
        )
    if method not in ("min_var", "tev", "mvo", "risk_parity", "bl"):
        raise ValueError(f"未知求解方法 {method!r}，可选: min_var / tev / mvo / risk_parity / bl")
    if method == "tev" and benchmark is None:
        raise ValueError("method='tev' 需要 benchmark 基准权重")
    if method == "risk_parity" and allow_short:
        raise ValueError("risk_parity 要求纯多头（w > 0）")

    codes = list(alpha.index)
    n = len(codes)
    if n == 0:
        return pd.Series(dtype=float)
    # 不可持仓股票（alpha NaN）权重强制为 0
    nan_mask = alpha.isna().values
    if nan_mask.all():
        return pd.Series(0.0, index=codes)

    w = cp.Variable(n)
    constraints: list[Any] = []
    obj_parts: list[Any] = []

    # ---- 约束：净多头预算、多空、上限 ----
    if budget is not None:
        constraints.append(cp.sum(w) == budget)
    if not allow_short:
        constraints.append(w >= 0)
    if allow_short:
        if short_limit is not None and short_limit > 0:
            # 空头总量 = Σmax(-w,0) = 0.5·(Σ|w| − Σw)
            constraints.append(0.5 * (cp.norm1(w) - cp.sum(w)) <= short_limit)
        if gross_limit is not None and gross_limit > 0:
            constraints.append(cp.norm1(w) <= gross_limit)
    if max_weight is not None and max_weight < 1.0:
        constraints.append(w <= max_weight)
    if nan_mask.any():
        nan_pos = np.flatnonzero(nan_mask)
        constraints.append(w[nan_pos] == 0)

    # ---- 约束：行业（精确等式 / 相对目标偏离区间，P1）----
    if industry_map is not None:
        ind = pd.Series(industry_map).reindex(codes)
        known = ind.notna()
        if known.any():
            inds = ind[known].unique()
            tgt = pd.Series(industry_target or {}, dtype=float).reindex(inds).fillna(0.0)
            if tgt.sum() > 0:
                tgt = tgt / tgt.sum() * budget  # 归一到预算
            else:
                tgt = pd.Series(budget / len(inds), index=inds)
            for name in inds:
                pos = [i for i in range(n) if known.iloc[i] and ind.iloc[i] == name]
                if not pos:
                    continue
                if industry_deviation is not None:
                    # 指增风格：行业权重允许在目标 ±dev 内浮动（比精确等式更贴近实盘）
                    constraints.append(
                        cp.abs(cp.sum(w[pos]) - float(tgt[name])) <= industry_deviation
                    )
                else:
                    constraints.append(cp.sum(w[pos]) == float(tgt[name]))

    # ---- 约束：风格中性化（P1）：|B'w| ≤ tol，把风格 beta 从组合中剔除 ----
    if style_exposures is not None:
        B = style_exposures.reindex(codes).fillna(0.0).values  # (n, n_style)，列需已 zscore
        if B.shape[1] > 0:
            constraints.append(cp.abs(B.T @ w) <= style_tolerance)

    # ---- 目标函数 ----
    score = _alpha_score(alpha).fillna(0.0).values
    if method == "min_var":
        obj_parts.append(cp.quad_form(w, Sigma))
    elif method == "tev":
        wb = benchmark.reindex(codes).fillna(0.0).values
        obj_parts.append(cp.quad_form(w - wb, Sigma))
        obj_parts.append(-risk_aversion * score @ w)
    elif method == "mvo":
        obj_parts.append(cp.quad_form(w, Sigma))
        obj_parts.append(-risk_aversion * score @ w)
    elif method == "bl":
        # Black-Litterman：均衡先验 + 观点 → 后验 μ_BL / Σ_BL，直接作线性项
        mw = (
            market_weights.reindex(codes).fillna(0.0).values
            if market_weights is not None
            else np.full(n, 1.0 / n)
        )
        mu_bl, Sigma_bl = bl_posterior(Sigma, mw, views=views, tau=tau, delta=delta)
        obj_parts.append(cp.quad_form(w, Sigma_bl))
        obj_parts.append(-risk_aversion * mu_bl @ w)
    else:  # risk_parity：等风险预算，w>0 由 ln 项保证（Spinu 2013 凸配方）
        constraints.append(w >= 1e-8)
        rb = np.full(n, 1.0 / n)
        # 数值关键：日频 Σ 元素 ~1e-4，而 -Σln(w) 量级 ~n·ln(n)。
        # 若 ln 项不缩放，风险项梯度（~Σw）被 ln 梯度（~1/w）淹没，求解器退化为等权。
        # rp_tau 取「等权处两项梯度同量级」：rp_tau = max|Σ·1|/n。最优解 wᵢ(Σw)ᵢ = rp_tau·bᵢ 仍等贡献。
        # 注意：对数障碍配方的数值解风险贡献比通常 ~1.5（等权 ~4），
        # 属近似风险平价；追求更高精度可用固定点迭代 wᵢ←bᵢ/(Σw)ᵢ（P3）。
        rp_tau = float(np.abs(Sigma @ np.ones(n)).max() / max(n, 1))
        obj_parts.append(0.5 * cp.quad_form(w, Sigma))
        obj_parts.append(-rp_tau * cp.sum(cp.multiply(rb, cp.log(w))))

    # ---- 换手/成本：硬约束（线性化）+ A-C 风格成本惩罚（线性 + 二次冲击，P2）----
    if prev_weights is not None:
        w0 = prev_weights.reindex(codes).fillna(0.0).values
        if max_turnover is not None and max_turnover > 0:
            # 单边换手 = 0.5·Σ|Δw|，与回测 turnover 口径一致
            constraints.append(cp.sum(cp.abs(w - w0)) <= 2.0 * max_turnover)
        if turnover_penalty and turnover_penalty > 0:
            obj_parts.append(turnover_penalty * cp.sum(cp.abs(w - w0)))
        if quadratic_cost and quadratic_cost > 0:
            # 二次冲击项：大单边际成本递增（Almgren-Chriss 风格）
            obj_parts.append(quadratic_cost * cp.sum_squares(w - w0))

    objective = cp.Minimize(sum(obj_parts))
    prob = cp.Problem(objective, constraints)
    # risk_parity 含 cp.log（指数锥）→ OSQP（仅 QP）不可用，需 SCS/ECOS；其余 QP 用 OSQP
    if method == "risk_parity":
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=100_000)
    else:
        prob.solve(solver=cp.OSQP, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP 求解失败，status={prob.status}")

    out = pd.Series(np.asarray(w.value).reshape(-1), index=codes)
    if nan_mask.any():
        out[nan_mask] = 0.0  # 不可持仓股票强制精确 0（OSQP 数值上仅近似满足）
    if min_weight is not None and min_weight > 0:
        out = _filter_min_weight(out, min_weight)
    return out


def optimize_weights_qp(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame | None = None,
    method: str = "min_var",
    window: int = 120,
    min_periods: int = 60,
    cov_method: str = "ledoit_wolf",
    shrinkage: float = 0.5,
    risk_aversion: float = 1.0,
    benchmark_weights: pd.Series | None = None,
    max_weight: float | None = None,
    min_weight: float | None = None,
    industry_map: Mapping[str, str] | pd.Series | None = None,
    industry_target: Mapping[str, float] | pd.Series | None = None,
    industry_deviation: float | None = None,
    industry_panel: pd.DataFrame | None = None,
    style_exposures: dict[str, pd.DataFrame] | None = None,
    style_tolerance: float = 1e-5,
    prev_weights: pd.Series | pd.DataFrame | None = None,
    max_turnover: float | None = None,
    turnover_penalty: float = 0.0,
    quadratic_cost: float = 0.0,
    budget: float = 1.0,
    allow_short: bool = False,
    short_limit: float | None = None,
    gross_limit: float | None = None,
    views: dict | None = None,
    market_weights: pd.Series | None = None,
    tau: float = 0.05,
    delta: float = 2.5,
    **kwargs: Any,
) -> pd.DataFrame:
    """面板级求解器组合优化（与 optimize_weights 同签名风格、同输出约定）。

    Args:
        factor_panel: date×code 因子值（方向已对齐）。
        returns_panel: date×code 日收益（估计 Σ 用；为 None 时无法估计风险 → 报错，
            因为本模块必须有风险模型才有意义）。
        method / window / min_periods / cov_method / shrinkage: 见 solve_portfolio。
        industry_deviation: 行业权重相对目标的允许偏离（None=精确等式）。
        style_exposures: dict{风格名 → date×code 暴露面板（列已 zscore）}，
            逐日拼成 code×style 传给 solve_portfolio 做风格中性化。
        industry_panel: date×code 的行业分类面板（值=行业名，PIT 口径），
            逐日取该行生成行业映射（优先级高于静态 industry_map，真实数据用）。
        allow_short / short_limit / gross_limit: 多空（净多头=budget，空头总量≤short_limit，
            总杠杆≤gross_limit）。
        views / market_weights / tau / delta: Black-Litterman（method="bl" 时生效）。
        quadratic_cost: A-C 二次冲击成本系数（与 turnover_penalty 线性项组合）。
        prev_weights: 面板级 DataFrame 时取**上一期输出权重**（滚动持仓）；
            Series 时每期广播。仅 max_turnover / turnover_penalty / quadratic_cost
            启用时生效。
    Returns:
        DataFrame(date×code) 权重；窗口样本不足的截面输出全 0（空仓，等价无法开仓）。
    """
    if returns_panel is None:
        raise ValueError("optimize_weights_qp 需要 returns_panel 估计协方差（风险模型是 QP 优化的前提）")

    dates = factor_panel.index
    codes = factor_panel.columns
    rows: list[pd.Series] = []
    prev_series: pd.Series | None = None
    turnover_on = (
        (max_turnover is not None and max_turnover > 0)
        or (turnover_penalty and turnover_penalty > 0)
        or (quadratic_cost and quadratic_cost > 0)
    )

    for t in dates:
        alpha = factor_panel.loc[t]
        Sigma = rolling_covariance(
            returns_panel, t, window=window, min_periods=min_periods,
            method=cov_method, shrinkage=shrinkage,
        )
        if Sigma is None:
            rows.append(pd.Series(0.0, index=codes))
            prev_series = None  # 空仓 → 换手基线上期为空仓
            continue
        pv = prev_series if turnover_on else None
        if pv is None and isinstance(prev_weights, pd.Series) and turnover_on:
            pv = prev_weights
        elif isinstance(prev_weights, pd.DataFrame) and t in prev_weights.index and turnover_on:
            pv = prev_weights.loc[t]
        # 风格暴露逐日拼接：{style → date×code} → code×style
        style_b = None
        if style_exposures:
            style_b = pd.concat(
                {name: df.loc[t] for name, df in style_exposures.items() if t in df.index},
                axis=1,
            )
        # 行业映射：PIT 面板优先（逐日），否则静态 industry_map
        ind_map_t = industry_map
        if industry_panel is not None and t in industry_panel.index:
            row = industry_panel.loc[t].dropna()
            if not row.empty:
                ind_map_t = row.to_dict()
        try:
            w = solve_portfolio(
                alpha, Sigma, method=method, risk_aversion=risk_aversion,
                benchmark=benchmark_weights, max_weight=max_weight, min_weight=min_weight,
                industry_map=ind_map_t, industry_target=industry_target,
                industry_deviation=industry_deviation,
                style_exposures=style_b, style_tolerance=style_tolerance,
                prev_weights=pv, max_turnover=max_turnover,
                turnover_penalty=turnover_penalty, quadratic_cost=quadratic_cost,
                budget=budget, allow_short=allow_short,
                short_limit=short_limit, gross_limit=gross_limit,
                views=views, market_weights=market_weights, tau=tau, delta=delta,
            )
        except RuntimeError:  # 单截面求解失败 → 该期空仓，不中断面板
            w = pd.Series(0.0, index=codes)
            prev_series = None
        rows.append(w.reindex(codes).fillna(0.0))
        prev_series = w

    out = pd.DataFrame(rows, index=dates, columns=codes)
    return out.fillna(0.0)


# ===========================================================================
# Black-Litterman（P2）：均衡收益先验 + 观点 → 后验 μ / Σ
# ===========================================================================
def bl_posterior(
    Sigma: np.ndarray,
    market_weights: np.ndarray | None = None,
    views: dict | None = None,
    tau: float = 0.05,
    delta: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Black-Litterman 后验（1992，高盛）。

    先验 = 市场均衡收益（反向优化）π = δ·Σ·w_mkt；叠加观点后贝叶斯合成：

        μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹·[(τΣ)⁻¹π + P'Ω⁻¹q]
        Σ_BL = Σ + [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹

    Args:
        Sigma: (N,N) 协方差。
        market_weights: 市场权重（默认等权 1/N），反向优化得均衡收益。
        views: {"P": (K,N) 视图矩阵, "q": (K,) 观点收益, "Omega": (K,K) 观点方差
               （可省略 → 用 τ·P·Σ·P' 自动定标）}；None 时退化为纯均衡。
        tau: 先验标度（均衡 vs 观点的相对权重，默认 0.05）。
        delta: 风险厌恶系数（均衡收益尺度）。
    Returns:
        (mu_bl, Sigma_bl)。
    """
    n = Sigma.shape[0]
    mw = market_weights if market_weights is not None else np.full(n, 1.0 / n)
    pi = delta * Sigma @ mw  # 反向优化：市场隐含均衡收益
    if views is None or views.get("P") is None:
        return pi, Sigma
    P = np.atleast_2d(np.asarray(views["P"], dtype=float))
    q = np.asarray(views["q"], dtype=float).reshape(-1)
    Om = views.get("Omega")
    if Om is None:
        Om = P @ (tau * Sigma) @ P.T  # 观点方差按均衡尺度自动定标
    else:
        Om = np.atleast_2d(np.asarray(Om, dtype=float))
        if Om.shape == (1, 1) and P.shape[0] > 1:
            Om = np.eye(P.shape[0]) * float(Om[0, 0])
    inv_tS = np.linalg.inv(tau * Sigma)
    M = inv_tS + P.T @ np.linalg.inv(Om) @ P
    mu_bl = np.linalg.solve(M, inv_tS @ pi + P.T @ np.linalg.inv(Om) @ q)
    Sigma_bl = Sigma + np.linalg.inv(M)
    return mu_bl, Sigma_bl


def bl_views_from_factor(
    factor: pd.Series,
    n_top: int = 10,
    view_scale: float = 0.001,
    uncertainty: float | None = None,
) -> dict:
    """把因子截面构造成 BL「相对观点」：P 行 = (top-n 等权组合 − bottom-n 等权组合)。

    观点含义：因子打分最高的 n 只相对最低的 n 只，未来相对收益方向 = 因子方向，
    强度由 view_scale 定（量级应接近日频收益的预期 spread）。

    Args:
        factor: Series(index=code) 因子值（越大越看好；NaN 剔除）。
        n_top: 观点两端持仓数。
        view_scale: 观点收益强度的绝对值（正负号取因子方向）。
        uncertainty: 观点方差；None → 用 τ·P·Σ·P' 自动定标（见 bl_posterior）。
    Returns:
        {"P": (1,N), "q": (1,), "Omega": (1,1) 或 None}，N=len(factor)。
    """
    vals = factor.dropna()
    if len(vals) < 2 * n_top:
        raise ValueError(f"因子有效样本不足（{len(vals)} < 2·{n_top}），无法构造观点")
    order = vals.sort_values(ascending=False)
    top, bot = order.index[:n_top], order.index[-n_top:]
    pos = {c: i for i, c in enumerate(factor.index)}
    P = np.zeros((1, len(factor)))
    for c in top:
        P[0, pos[c]] += 1.0 / n_top
    for c in bot:
        P[0, pos[c]] -= 1.0 / n_top
    direction = 1.0 if float(vals[top].mean() - vals[bot].mean()) >= 0 else -1.0
    q = np.array([direction * view_scale])
    views = {"P": P, "q": q}
    if uncertainty is not None:
        views["Omega"] = np.array([[uncertainty]])
    return views


# ===========================================================================
# HRP（Lopez de Prado 2016）：层次聚类 + 递归二分 + 逆方差，免逆矩阵
# ===========================================================================
def _quasi_diag(link: np.ndarray) -> list[int]:
    """把层次聚类 linkage 的叶节点重排为「准对角」顺序（聚类块连续）。"""
    link = link.astype(int)
    num_items = int(link[-1, 3])
    sort_ix = [int(link[-1, 0]), int(link[-1, 1])]
    while max(sort_ix) >= num_items:
        new_ix: list[int] = []
        for ix in sort_ix:
            if ix >= num_items:
                j = ix - num_items
                new_ix += [int(link[j, 0]), int(link[j, 1])]
            else:
                new_ix.append(ix)
        sort_ix = new_ix
    return sort_ix


def _cluster_variance(sub_cov: np.ndarray, pos: list[int]) -> float:
    """簇内方差：用逆方差分配（IVP）加权。"""
    inv_diag = 1.0 / np.maximum(np.diag(sub_cov), 1e-12)
    ivp = inv_diag[pos] / inv_diag.sum()
    return float(ivp @ sub_cov[np.ix_(pos, pos)] @ ivp)


def _hrp_recursive(cov: np.ndarray, idx: list[int]) -> dict[int, float]:
    """递归二分：每层按左右簇方差比例分配权重，叶子返回。"""
    if len(idx) <= 1:
        return {idx[0]: 1.0}
    mid = len(idx) // 2
    left, right = idx[:mid], idx[mid:]
    sub = cov[np.ix_(idx, idx)]
    lv = _cluster_variance(sub, list(range(mid)))
    rv = _cluster_variance(sub, list(range(mid, len(idx))))
    alpha = 1.0 - lv / (lv + rv)
    out: dict[int, float] = {}
    for k, v in _hrp_recursive(cov, left).items():
        out[k] = alpha * v
    for k, v in _hrp_recursive(cov, right).items():
        out[k] = (1.0 - alpha) * v
    return out


def hrp_weights(
    cov: np.ndarray,
    codes=None,
    linkage_method: str = "ward",
) -> pd.Series:
    """HRP 权重（Lopez de Prado 2016）：层次聚类 + 递归二分 + 逆方差。

    优势：全程不需求 Σ⁻¹（病态/奇异协方差免疫，数值稳定），天然非负、
    权重和 ≈ 1。缺点：不用收益信号（纯风险结构）。
    """
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform

    n = cov.shape[0]
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))  # 相关距离
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    link = sch.linkage(squareform(dist), method=linkage_method)
    order = _quasi_diag(link)
    w = _hrp_recursive(cov, order)
    out = np.zeros(n)
    for pos, wgt in w.items():
        out[pos] = wgt
    s = out.sum()
    if s > 0:
        out /= s
    return pd.Series(out, index=codes if codes is not None else range(n))


def optimize_weights_hrp(
    returns_panel: pd.DataFrame,
    window: int = 120,
    min_periods: int = 60,
    cov_method: str = "ledoit_wolf",
    shrinkage: float = 0.5,
) -> pd.DataFrame:
    """面板级 HRP 权重（纯风险结构，不需因子信号与求解器）。

    Returns:
        DataFrame(date×code) 权重；窗口不足截面全 0（空仓）。
    """
    dates = returns_panel.index
    codes = returns_panel.columns
    rows: list[pd.Series] = []
    for t in dates:
        Sigma = rolling_covariance(
            returns_panel, t, window=window, min_periods=min_periods,
            method=cov_method, shrinkage=shrinkage,
        )
        if Sigma is None:
            rows.append(pd.Series(0.0, index=codes))
            continue
        rows.append(hrp_weights(Sigma, codes=codes))
    return pd.DataFrame(rows, index=dates, columns=codes).fillna(0.0)
