"""Barra CNE6（简化版）截面回归链路：暴露预处理 → WLS 因子收益 → 协方差。

分层（``test_layering``）：research 层，允许 import factor/data（统计内核在
``factor.preprocessing``、行情面板来自 ``reports/alla_rolling/_base``）。

链路三步（每日独立、可并行）：
1. :func:`standardize_exposures` —— CNE6 暴露预处理：MAD 去极值 →
   **市值加权均值 / 等权标准差** 标准化（市值加权标准化，lncap/nl_size 例外
   只做去极值等权标准化——它们本身就是市值函数，市值加权会自反）。
2. :func:`cross_sectional_regression` —— 逐日 WLS 截面回归：
   ``r = X·f + u``，X = [国家截距 | 行业哑变量 | 风格暴露]，
   权重 = sqrt(市值)（Barra 标准做法；行业用哑变量、不加约束）。
   返回因子收益率 f、R²、调整 R²、个股特异性收益 u。
3. :func:`factor_covariance` / :func:`specific_risk` —— 因子收益时序协方差
   （Neway-West 期限调整）与特异性风险（EWMA）→ :func:`assemble_sigma`
   合成个股协方差 Σ = X·F·X' + D。

用法::

    from research.barra import load_barra_inputs, run_factor_returns
    inputs = load_barra_inputs()          # 暴露面板 + 行情 + 行业 + 市值
    fr = run_factor_returns(inputs)        # DataFrame(index=date, columns=因子)
    fr.to_csv("reports/barra_cne6/factor_returns.csv")
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = "all_a_2018_2026"

#: 一级风格因子 → 描述变量（panels_barra 文件名）
STYLE_LEVEL1: dict[str, tuple[str, ...]] = {
    "size": ("lncap",),
    "beta": ("beta",),
    "resid_vol": ("resid_vol",),
    "momentum": ("momentum",),
    "nl_size": ("nl_size",),
    "liquidity": ("liquidity",),
    "earnings_yld": ("ep_ttm",),
    "value": ("btop",),
    "growth": ("growth",),
    "leverage": ("mlev", "dtoa", "blev"),
    "dividend_yld": ("dtop",),
}

#: 自身即市值函数的描述变量（标准化时不做市值加权，防自反）
_CAP_NATIVE = {"lncap", "nl_size"}


# ---------------------------------------------------------------------------
# 1. 暴露预处理
# ---------------------------------------------------------------------------
def winsorize_cs(s: pd.Series, n_mad: float = 5.0) -> pd.Series:
    """截面 MAD 去极值（Barra 用更宽的 5×MAD，alpha 因子才用 3×）。"""
    med = s.median()
    mad = (s - med).abs().median()
    if not mad or np.isnan(mad):
        return s
    lo, hi = med - n_mad * mad, med + n_mad * mad
    return s.clip(lo, hi)


def standardize_exposures(raw_row: pd.Series, cap_row: pd.Series,
                          var: str) -> pd.Series:
    """单个描述变量的截面标准化（CNE6：MAD → 市值加权均值/等权 sd）。

    - MAD 去极值 5×MAD；
    - 均值 = 市值加权；标准差 = 等权（DolphinDB 教程口径）；
    - ``var`` 在 :data:`_CAP_NATIVE`（lncap/nl_size）时均值也用等权——
      该变量本身是市值的确定性函数，市值加权会自反。
    """
    s = raw_row.dropna()
    if len(s) < 30:
        return pd.Series(np.nan, index=raw_row.index)
    s = winsorize_cs(s)
    w = cap_row.reindex(s.index).clip(lower=0)
    if var in _CAP_NATIVE or w.notna().sum() < 30:
        mu = s.mean()
    else:
        ww = w.fillna(0.0)
        mu = float((s * ww).sum() / ww.sum()) if ww.sum() > 0 else s.mean()
    sd = s.std()
    if not sd or np.isnan(sd):
        return pd.Series(np.nan, index=raw_row.index)
    return (s - mu) / sd


# ---------------------------------------------------------------------------
# 2. 逐日 WLS 截面回归
# ---------------------------------------------------------------------------
def cross_sectional_regression(ret_row: pd.Series,
                               exposures: dict[str, pd.Series],
                               industry_row: pd.Series,
                               cap_row: pd.Series) -> dict[str, Any]:
    """单日 WLS 截面回归。

    Args:
        ret_row: 当日个股收益（T→T+1 或 T→T，链路外决定）。
        exposures: {描述变量: 当日标准化暴露}。
        industry_row: 当日申万一级行业（code→行业名）。
        cap_row: 当日市值（WLS 权重 = sqrt）。
    Returns:
        {"factor_returns": Series(行业+风格+国家), "r2", "adj_r2",
         "resid": Series, "n", "t_stats": Series}
    """
    y = ret_row.dropna()
    common = y.index
    for v in exposures.values():
        common = common.intersection(v.dropna().index)
    common = common.intersection(cap_row.dropna().index)
    if len(common) < 100:  # 样本下限（全A 5800 只，正常日 >4000）
        return {}

    y = y.loc[common].astype(float)
    cap = cap_row.loc[common].astype(float)
    cap_ok = np.isfinite(cap.values) & (cap.values > 0)
    w = np.sqrt(cap.clip(lower=0)).values          # sqrt 市值权重
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

    cols: dict[str, np.ndarray] = {}
    ind_names = sorted(set(industry_row.loc[common].dropna()))
    for ind in ind_names:                          # 行业哑变量（k 个，含截距共线性不处理）
        cols[f"ind:{ind}"] = (industry_row.loc[common] == ind).astype(float).values
    for var, s in exposures.items():
        cols[f"sty:{var}"] = s.reindex(common).astype(float).values

    X = pd.DataFrame(cols, index=common)
    keep = X.notna().all(axis=1).values & np.isfinite(y.values) & cap_ok
    X = X.loc[keep]
    yv = y.values[keep]
    wv = w[keep]
    if len(yv) < 100 or X.shape[1] >= len(yv) - 1:
        return {}

    # WLS: β = (X'WX)^-1 X'Wy
    sw = np.sqrt(wv)
    Xw = X.values * sw[:, None]
    yw = yv * sw
    try:
        beta, _, rank, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return {}
    fitted = X.values @ beta
    resid = yv - fitted

    # 加权 R²/t：与回归目标权重 wv = sqrt(cap) 同尺度（Barra 标准），
    # σ̂² = Σw·e²/(n-k) 与 Cov(β) = σ̂²(X'WX)⁻¹ 配套，勿改成 w²（会打碎 t）。
    ss_res = float((wv * resid ** 2).sum())
    ss_tot = float((wv * (yv - np.average(yv, weights=wv)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n, k = len(yv), X.shape[1]
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else np.nan

    # t 统计（WLS 标准误）：Cov(β) = σ² (X'WX)^-1，σ² = SSR/(n-k)
    dof = n - k
    sigma2 = ss_res / dof if dof > 0 else np.nan
    xtwx_inv = np.linalg.pinv(Xw.T @ Xw)
    se = np.sqrt(np.maximum(np.diag(xtwx_inv) * sigma2, 0.0))
    t_stats = pd.Series(beta / np.where(se > 0, se, np.nan), index=X.columns)

    fr = pd.Series(beta, index=X.columns)
    resid_s = pd.Series(resid, index=X.index)
    return {"factor_returns": fr, "r2": r2, "adj_r2": adj_r2,
            "resid": resid_s, "n": n, "t_stats": t_stats}


# ---------------------------------------------------------------------------
# 3. 时序层：因子协方差 + 特异性风险 + Σ
# ---------------------------------------------------------------------------
def factor_covariance(fr: pd.DataFrame, halflife: int = 250,
                      max_lag: int = 3) -> pd.DataFrame:
    """因子收益协方差 + Newey-West 期限调整（lag ≤ max_lag）。

    Γ = γ₀ + Σ_l (γ_l + γ_l')·(1 - l/(max_lag+1))，γ_l = Cov(f_t, f_{t-l})
    （EWMA 加权；γ_l 与其转置各计一次，标量情形即标准 NW 长期方差）。
    """
    x = fr.dropna(how="all")
    ewm_cov = x.ewm(halflife=halflife, min_periods=int(halflife * 0.6)).cov(bias=False)
    f_full = x.columns
    out = (ewm_cov.xs(x.index[-1], level=0)
           .reindex(index=f_full, columns=f_full).fillna(0.0))
    for lag in range(1, max_lag + 1):
        # pandas 2.x 的 DataFrame.cov() 不接受 other 参数 → concat 成宽表取交叉块
        wide = pd.concat([x.add_suffix("__l0"),
                          x.shift(lag).add_suffix("__l1")], axis=1)
        cov_all = wide.cov(min_periods=int(halflife * 0.3))
        l0 = [f"{c}__l0" for c in f_full]
        l1 = [f"{c}__l1" for c in f_full]
        g = cov_all.loc[l0, l1]
        g.index, g.columns = f_full, f_full
        out = out + (g + g.T).fillna(0.0) * (1.0 - lag / (max_lag + 1.0))
    # 对称化 + 半正定化（特征值裁剪）
    out = (out + out.T) / 2.0
    vals, vecs = np.linalg.eigh(out.values)
    vals = np.clip(vals, 1e-12, None)
    return pd.DataFrame(vecs @ np.diag(vals) @ vecs.T, index=f_full, columns=f_full)


def specific_risk(resid: pd.DataFrame, halflife: int = 90) -> pd.DataFrame:
    """个股特异性方差（EWMA，逐股独立）。返回 Σu² 面板（date×code）。"""
    return (resid ** 2).ewm(halflife=halflife, min_periods=30).mean()


def assemble_sigma(exposure_row: pd.Series, industry_row: pd.Series,
                   f_cov: pd.DataFrame,
                   spec_var: pd.Series) -> pd.DataFrame:
    """Σ = X·F·X' + D（单日：N×N 个股协方差）。

    X = [行业哑变量 | 风格暴露（标准化后）]；行业因子收益协方差并入 F。
    特异性对角块 D = diag(σ²_u)。
    """
    codes = exposure_row.index
    ind_names = sorted(f_cov.index[f_cov.index.str.startswith("ind:")])
    cols = {}
    for ind in ind_names:
        ind_name = ind[4:]
        cols[ind] = (industry_row.reindex(codes) == ind_name).astype(float)
    for var in f_cov.index:
        if var.startswith("sty:"):
            cols[var] = exposure_row.reindex(codes)
    X = pd.DataFrame(cols, index=codes).reindex(columns=f_cov.index).fillna(0.0)
    # 全程 ndarray 矩阵乘：pandas 3.0 中 ndarray @ DataFrame 返回带 0..2 整数
    # 索引的 DataFrame，再被外层构造器按 codes 重索引 → 全 NaN（静默炸）
    common = pd.DataFrame(X.values @ f_cov.values @ X.values.T,
                          index=codes, columns=codes)
    D = pd.DataFrame(np.diag(spec_var.reindex(codes).fillna(0.0).values),
                     index=codes, columns=codes)
    return common + D


# ---------------------------------------------------------------------------
# 输入装载 + 主跑
# ---------------------------------------------------------------------------
def load_barra_inputs() -> dict[str, Any]:
    """装载暴露面板（panels_barra）、行情、行业、市值。"""
    from config import Config
    ds_dir = Path(str(Config.get()["factor_library"]["root"])) / DATASET
    barra_dir = ds_dir / "panels_barra"
    base = ROOT / "reports" / "alla_rolling" / "_base"

    from scripts.builders.build_alla_barra_styles import STYLE_VARS
    expos: dict[str, pd.DataFrame] = {}
    for v in STYLE_VARS:
        p = barra_dir / f"{v}.parquet"
        if p.exists():
            expos[v] = pd.read_parquet(p)
    close = pd.read_parquet(base / "close_adj.parquet")
    cap = pd.read_parquet(base / "market_cap.parquet")
    ind = pd.read_parquet(base / "cov_industry.parquet")
    return {"exposures": expos, "close": close, "cap": cap, "industry": ind}


def run_factor_returns(inputs: dict[str, Any],
                       begin: str | None = None,
                       end: str | None = None,
                       min_names: int = 300) -> pd.DataFrame:
    """逐日 WLS 截面回归主循环，返回因子收益宽表（date × [ind:*, sty:*]）。

    收益口径 = T 收盘信号 → T→T+1 收益（与项目 ``pct_change().shift(-1)`` 同源）。
    """
    expos: dict[str, pd.DataFrame] = inputs["exposures"]
    close = inputs["close"]
    cap = inputs["cap"]
    ind = inputs["industry"]

    ret = close.pct_change(fill_method=None).shift(-1)
    dates = close.index
    if begin:
        dates = dates[dates >= pd.Timestamp(begin)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]

    rows: dict[pd.Timestamp, pd.Series] = {}
    meta: list[dict] = []
    for d in dates:
        ret_row = ret.loc[d]
        if ret_row.notna().sum() < min_names:
            continue
        raw = {v: p.loc[d] for v, p in expos.items() if d in p.index}
        if len(raw) < len(expos) * 0.8:
            continue
        cap_row = cap.loc[d] if d in cap.index else pd.Series(dtype=float)
        ind_row = ind.loc[d] if d in ind.index else pd.Series(dtype=float)
        stded = {v: standardize_exposures(raw[v], cap_row, v) for v in raw}
        res = cross_sectional_regression(ret_row, stded, ind_row, cap_row)
        if not res:
            continue
        rows[d] = res["factor_returns"]
        meta.append({"date": d, "r2": res["r2"], "adj_r2": res["adj_r2"],
                     "n": res["n"]})
    fr = pd.DataFrame(rows).T.sort_index()
    fr.attrs["meta"] = pd.DataFrame(meta).set_index("date") if meta else None
    return fr


def run_checkup(inputs: dict[str, Any], out_dir: Path,
                begin: str | None = None, end: str | None = None) -> pd.DataFrame:
    """端到端跑因子收益 + 元信息，落盘 factor_returns / meta / 摘要。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    fr = run_factor_returns(inputs, begin=begin, end=end)
    if fr.empty:
        raise SystemExit("因子收益为空（检查暴露面板覆盖与日期区间）")
    fr.to_csv(out_dir / "factor_returns.csv", encoding="utf-8-sig")
    meta = fr.attrs.get("meta")
    if meta is not None and len(meta):
        meta.to_csv(out_dir / "regression_meta.csv", encoding="utf-8-sig")
        summary = {
            "n_days": int(len(meta)),
            "r2_mean": float(meta["r2"].mean()),
            "r2_median": float(meta["r2"].median()),
            "adj_r2_mean": float(meta["adj_r2"].mean()),
            "n_mean": float(meta["n"].mean()),
        }
        pd.Series(summary).to_csv(out_dir / "summary.csv", header=["value"],
                                  encoding="utf-8-sig")
        print(summary)
    # 因子收益显著性（NW 稳健 t，H0: 均值=0）
    from stats.robust_stats import nw_tstat
    t = fr.apply(lambda c: nw_tstat(c.dropna())[0])
    t.rename("t_nw").to_csv(out_dir / "factor_t.csv", encoding="utf-8-sig")
    return fr
