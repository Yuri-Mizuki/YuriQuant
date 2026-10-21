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
    x = fr.dropna(how="all").copy()
    # 🚨 pandas 会把 attrs 传播到 add_suffix 的副本，concat 时逐项比较 attrs；
    # fr.attrs["meta"] 是 DataFrame → `all(obj.attrs == ...)` 触发 __bool__ 报错。
    x.attrs = {}
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


def assemble_sigma(exposure_row: pd.Series | dict[str, pd.Series],
                   industry_row: pd.Series,
                   f_cov: pd.DataFrame,
                   spec_var: pd.Series) -> pd.DataFrame:
    """Σ = X·F·X' + D（单日：N×N 个股协方差）。

    X = [行业哑变量 | 风格暴露（标准化后）]；行业因子收益协方差并入 F。
    特异性对角块 D = diag(σ²_u)。

    Args:
        exposure_row: ``{描述变量名: Series}``（多风格，推荐）或单个 Series
            （向后兼容：只有一个 ``sty:*`` 因子时才无歧义——早期版本把所有
            ``sty:*`` 列都赋成同一个 Series，多风格时静默错位，10-21 修）。
    """
    codes = exposure_row.index if isinstance(exposure_row, pd.Series) \
        else next(iter(exposure_row.values())).index
    if isinstance(exposure_row, dict):
        sty_map = exposure_row
    else:
        sty_map = {k[4:]: exposure_row for k in f_cov.index if k.startswith("sty:")}

    ind_names = sorted(f_cov.index[f_cov.index.str.startswith("ind:")])
    cols = {}
    for ind in ind_names:
        ind_name = ind[4:]
        cols[ind] = (industry_row.reindex(codes) == ind_name).astype(float)
    for var in f_cov.index:
        if var.startswith("sty:"):
            name = var[4:]
            if name not in sty_map:
                raise KeyError(f"暴露缺少风格变量 {name!r}（f_cov 里有 {var}）")
            cols[var] = sty_map[name].reindex(codes)
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
                       min_names: int = 300,
                       collect_resid: bool = False) -> pd.DataFrame:
    """逐日 WLS 截面回归主循环，返回因子收益宽表（date × [ind:*, sty:*]）。

    收益口径 = T 收盘信号 → T→T+1 收益（与项目 ``pct_change().shift(-1)`` 同源）。

    ``collect_resid=True`` 时把逐日特异性收益面板塞进 ``fr.attrs["resid"]``
    （date×code，未参与回归的股票为 NaN），供 :func:`specific_risk` 用。
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
    resid_rows: dict[pd.Timestamp, pd.Series] = {}
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
        if collect_resid:
            resid_rows[d] = res["resid"]
        meta.append({"date": d, "r2": res["r2"], "adj_r2": res["adj_r2"],
                     "n": res["n"]})
    fr = pd.DataFrame(rows).T.sort_index()
    fr.attrs["meta"] = pd.DataFrame(meta).set_index("date") if meta else None
    if collect_resid:
        fr.attrs["resid"] = (pd.DataFrame(resid_rows).T
                             .reindex(index=fr.index, columns=close.columns))
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


# ---------------------------------------------------------------------------
# 4. 第二期：贝叶斯压缩 + 波动率偏误调整 + Σ 落盘与 Bias 验收
# ---------------------------------------------------------------------------
def bayes_shrink_fcov(fr: pd.DataFrame,
                      base_cov: pd.DataFrame | None = None,
                      delta: float | None = None) -> tuple[pd.DataFrame, float]:
    """贝叶斯压缩：向「等相关」先验收缩因子相关矩阵（Barra 简化实现）。

    - 先验 ``F0``：对角保留各因子样本方差 ``F_ii``，非对角 =
      ``ρ̄·sqrt(F_ii·F_jj)``，``ρ̄`` = 样本平均相关（「所有因子共享同一相关」
      是最弱的结构假设）；
    - 收缩强度 ``δ``：缺省用 Ledoit-Wolf 在等权因子收益上的解析解
      （Frobenius 损失下的最优 δ ≈ 相关噪声占总变异比例），钳制 [0.05, 0.95]。

    与 Barra 原版差异：原版用 B 统计量（依赖因子数与样本量的解析式）并对
    相关矩阵分块压缩；这里只做全局单一 δ（44 个因子、2000+ 样本下两者差异很小）。

    Returns:
        (压缩后 F, δ)
    """
    if base_cov is None:
        base_cov = factor_covariance(fr)
    f = base_cov.values.astype(float)
    d = np.sqrt(np.clip(np.diag(f), 0.0, None))
    n = len(f)
    corr = np.eye(n)
    dd = np.outer(d, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(dd > 0, f / dd, 0.0)
    corr = np.where(np.isfinite(c), c, 0.0)
    np.fill_diagonal(corr, 1.0)
    off = ~np.eye(n, dtype=bool)
    rho_bar = float(corr[off].mean()) if off.any() else 0.0
    prior = np.full((n, n), rho_bar)
    np.fill_diagonal(prior, 1.0)

    if delta is None:
        delta = 0.5
        x = fr.dropna(how="any").values
        if x.shape[1] > 1 and x.shape[0] > x.shape[1] + 2:
            try:
                from sklearn.covariance import LedoitWolf
                delta = float(LedoitWolf().fit(x).shrinkage_)
            except Exception:  # noqa: BLE001  数值/维度问题走固定 0.5
                pass
    delta = float(np.clip(delta, 0.05, 0.95))
    shrunk = (1.0 - delta) * corr + delta * prior
    out = shrunk * dd
    vals, vecs = np.linalg.eigh(out)
    out = vecs @ np.diag(np.clip(vals, 1e-14, None)) @ vecs.T
    return (pd.DataFrame(out, index=base_cov.index, columns=base_cov.columns),
            delta)


def vol_regime_adjust(f_cov: pd.DataFrame,
                      fr_hist: pd.DataFrame | None = None,
                      resid_hist: pd.DataFrame | None = None,
                      window: int = 60,
                      bounds: tuple[float, float] = (0.5, 2.0),
                      ) -> tuple[pd.DataFrame, float, float]:
    """波动率偏误调整（VRA）：短期实现波动 / 长期预测 → 缩放因子。

    Barra VRA 用近 ``window`` 日实现波动与长期 EWMA 预测之比（跨因子/跨股
    取中位数——个股噪声大，中位数稳健），钳制在 ``bounds``。市场波动上行期
    Σ 会系统性低估风险，VRA 是修正它的标准手段。

    Args:
        resid_hist: 残差**面板**（date×code），不是方差面板。
    Returns:
        (调整后 F, 因子 scale, 特异 scale)
    """
    f_scale = s_scale = 1.0
    out = f_cov
    if fr_hist is not None and len(fr_hist) >= window + 60:
        long_sd = fr_hist.ewm(halflife=250, min_periods=60).std().iloc[-1]
        short_sd = fr_hist.tail(window).std()
        r = (short_sd / long_sd).replace([np.inf, -np.inf], np.nan).dropna()
        if len(r):
            f_scale = float(np.clip(r.median(), *bounds))
            out = f_cov * (f_scale ** 2)
    if resid_hist is not None and len(resid_hist) >= window + 60:
        sq = resid_hist ** 2
        # 长期预测用 halflife=250（更平滑，才不会把近 60 日的波动跳变自己
        # 吸收掉——halflife=90 时 60 日跳变已占长期均值 37%，VRA 会失效）
        long_var = sq.ewm(halflife=250, min_periods=60).mean().iloc[-1]
        short_var = sq.tail(window).mean()
        r = np.sqrt(short_var / long_var.replace(0.0, np.nan))
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        if len(r):
            s_scale = float(np.clip(r.median(), *bounds))
    return out, f_scale, s_scale


def risk_bias_stat(sigma: pd.DataFrame, fut_ret: pd.Series,
                   horizon: int = 20) -> dict:
    """Barra Bias 统计量：实际 horizon 收益 / 预测 horizon 波动的 RMS。

    ``z_i = r_i / (σ_i·sqrt(horizon))``，``Bias = sqrt(mean(z²))``
    （理想 = 1.0；>1 = 低估风险，<1 = 高估）。
    """
    pred = pd.Series(np.sqrt(np.clip(np.diag(sigma.values), 0, None)),
                     index=sigma.index)
    x = pd.concat([pred.rename("pred"), fut_ret.rename("real")], axis=1).dropna()
    x = x[x["pred"] > 0]
    if len(x) < 50:
        return {}
    z = x["real"] / (x["pred"] * np.sqrt(horizon))
    return {"n": int(len(x)), "bias": float(np.sqrt((z ** 2).mean())),
            "mean_z": float(z.mean()), "std_z": float(z.std())}


def build_sigma_artifacts(inputs: dict[str, Any], out_dir: Path,
                          begin: str | None = None, end: str | None = None,
                          n_sample: int = 40, horizon: int = 20,
                          window: int = 60) -> pd.DataFrame:
    """第二期主入口：残差面板 + 特异风险 + 抽样日 Σ + Bias 验收。

    抽样日均匀分布在暖机后 ~ 末日前 horizon 日；每期只用当日**之前**的
    因子收益/残差估 F 与 D（防前视），Σ = X·F·X' + D 后与未来 horizon 日
    实现收益比算 Bias。

    落盘（``out_dir``）：
        resid_panel.parquet   逐日特异性收益（date×code）
        spec_vol.parquet      特异波动（EWMA，date×code）
        f_cov_last.parquet    最后抽样日的因子协方差（压缩+VRA 后）
        sigma_summary.csv     逐抽样日 n / f_scale / s_scale / bias / mean_z
        sigma_diag_last.csv   最后抽样日的预测波动（code, pred_sigma）
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fr = run_factor_returns(inputs, begin=begin, end=end, collect_resid=True)
    if fr.empty:
        raise SystemExit("因子收益为空（检查暴露面板覆盖）")
    resid = fr.attrs.pop("resid")
    fr.to_csv(out_dir / "factor_returns.csv", encoding="utf-8-sig")
    resid.astype("float32").to_parquet(out_dir / "resid_panel.parquet")
    spec_var = specific_risk(resid)
    spec_var.astype("float32").to_parquet(out_dir / "spec_vol.parquet")

    close = inputs["close"]
    cap = inputs["cap"]
    ind = inputs["industry"]
    expos = inputs["exposures"]

    cand = fr.index[300:-horizon]
    if len(cand) == 0:
        raise SystemExit("样本不足（<320 日）")
    pos = (np.linspace(0, len(cand) - 1, n_sample).astype(int)
           if len(cand) > n_sample else np.arange(len(cand)))
    sample = cand[pos]

    rows, last = [], None
    for d in sample:
        hist = fr.loc[:d].iloc[:-1].tail(500)      # 防前视：不含当日
        if len(hist) < 200:
            continue
        f_cov = factor_covariance(hist)
        f_cov, fs, ss = vol_regime_adjust(f_cov, hist,
                                          resid.loc[:d].iloc[:-1])
        f_cov, _ = bayes_shrink_fcov(hist, f_cov)

        raw = {v: p.loc[d] for v, p in expos.items() if d in p.index}
        if len(raw) < len(expos) * 0.8:
            continue
        cap_row = cap.loc[d] if d in cap.index else pd.Series(dtype=float)
        ind_row = ind.loc[d] if d in ind.index else pd.Series(dtype=float)
        stded = {v: standardize_exposures(raw[v], cap_row, v) for v in raw}

        spec_row = spec_var.loc[d] * (ss ** 2) if d in spec_var.index else None
        if spec_row is None:
            continue
        codes = spec_row.dropna().index
        for v in stded.values():
            codes = codes.intersection(v.dropna().index)
        codes = codes.intersection(ind_row.dropna().index)
        if len(codes) < 200:
            continue
        sub_sty = {k: v.reindex(codes) for k, v in stded.items()}
        sigma = assemble_sigma(sub_sty, ind_row.reindex(codes), f_cov,
                               spec_row.reindex(codes))
        fut = close.shift(-horizon).loc[d] / close.loc[d] - 1.0
        st = risk_bias_stat(sigma, fut.reindex(codes), horizon)
        if not st:
            continue
        rows.append({"date": d, "f_scale": fs, "s_scale": ss, **st})
        last = (d, f_cov, sigma, fs, ss)
        if len(rows) % 10 == 0:
            print(f"  [{len(rows)}/{len(sample)}] {d.date()} bias={st['bias']:.3f}",
                  flush=True)

    summ = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    if not summ.empty:
        summ.to_csv(out_dir / "sigma_summary.csv", encoding="utf-8-sig")
        print(f"Bias 均值 = {summ['bias'].mean():.4f} "
              f"(理想 1.0) | 中位数 = {summ['bias'].median():.4f} "
              f"| 样本 {len(summ)} 期", flush=True)
    if last is not None:
        d, f_cov, sigma, fs, ss = last
        f_cov.to_parquet(out_dir / "f_cov_last.parquet")
        pd.DataFrame({
            "code": sigma.index,
            "pred_sigma": np.sqrt(np.clip(np.diag(sigma.values), 0, None)),
            "spec_vol": np.sqrt(spec_var.loc[d].reindex(sigma.index).values),
        }).to_csv(out_dir / "sigma_diag_last.csv", index=False,
                  encoding="utf-8-sig")
        print(f"最后抽样日 {d.date()}：Σ {sigma.shape}，"
              f"因子协方差 {f_cov.shape}（f_scale={fs:.3f}, s_scale={ss:.3f}）",
              flush=True)
    return summ
