"""
因子增强 Top25 组合（对齐华泰 AI 63/92 组合构建）
==================================================

AI 63 案例：以文本因子分 10 层第 1 层为基础股票池，用 12 个基本面/技术面/
市值因子等权合成（合成前行业市值中性化 + 方向调整），选 Top25 月度调仓。
AI 92 图表10 的 12 因子清单。本脚本对照映射（如实声明替代）：

    AI 63/92 因子                本实现                          方向
    --------------------------------------------------------------
    ROE_q 单季ROE              roe_ttm（TTM 口径）              +
    EP                          ep_ttm                          +
    OCFP                        fcf_yield（自由现金流/市值近似） +
    Profit_G_q 净利YTD同比      np_growth_yoy                    +
    Sales_G_q 营收YTD同比       rev_growth_yoy                   +
    Holder_avgpctchange 户均   holder_num_zs（股东户数zscore，   -
      户均持股增长              户数降=户均升）
    exp_wgt_return_1m          exp_wgt_return_1m（换手衰减加权）-
    exp_wgt_return_12m         exp_wgt_return_12m               -
    bias_turn_1m               bias_turn（20日均换手/年均换手-1）-
    trans_at_last_ratio 尾盘   （跳过：分钟数据 2022+ 才有，
      成交占比                   历史不对称）
    Amihud_illiq               amihud（|ret|/成交额，成交额=
                                 20日均换手×市值 近似）          +
    ln_capital                 ln_mktcap                       -

风格因子合成前按月做行业+市值中性化（AI 63 口径）；文本因子不中性化
（基础池由原始文本因子排名决定）。换手衰减反转权重 exp(-x_i/N/4)。

基准：全A等权（_base/bench_eqw）。交易成本双边 3‰。

用法：
    python -m scripts.textmining.build_enhanced_top25 \
        --text-factor fadt_factor_bert_xgb_zz1000.parquet

产出：
    reports/textmining/fadt/features/enhanced_top25_{tag}.parquet （月收益）
    reports/textmining/fadt/reports/enhanced_top25_{tag}.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cli_common import setup_logging  # noqa: E402
from scripts.textmining._paths import Out  # noqa: E402

OUT_DIR = Out("fadt")
BASE = ROOT / "reports" / "alla_rolling" / "_base"
STYLE_CACHE = OUT_DIR / "style_panels_monthly.parquet"
log = setup_logging("enhanced_top25")

COST_RT = 0.003  # 双边 3‰


# ---------------------------------------------------------------------------
# 风格因子面板（月度）
# ---------------------------------------------------------------------------
def _technical_panels() -> dict[str, pd.DataFrame]:
    """技术面因子（月度截面）：来自 _base 复权价/换手/市值面板。"""
    close = pd.read_parquet(BASE / "close_adj.parquet")
    turn = pd.read_parquet(BASE / "cov_turn.parquet")
    cap = pd.read_parquet(BASE / "market_cap.parquet")

    ret = close.pct_change(fill_method=None)

    def _exp_wgt_return(n_days: int, half_life_scale: float) -> pd.DataFrame:
        """换手率衰减加权动量（AI 63：w ∝ exp(-x_i/N/4)，最近权重最大）。

        F(t) = Σ_j r(j)·w(t-j) 恰为卷积定义 conv[t] = Σ_j x[j]·k[t-j]
        （kernel 用原序 w），故 F = conv[:T]；前 n-1 行窗口不完整置 NaN。
        """
        from scipy.signal import oaconvolve
        w = np.exp(-np.arange(n_days) / n_days / half_life_scale)
        conv = oaconvolve(np.nan_to_num(ret.values, nan=0.0),
                          w[:, None], axes=0, mode="full")
        F = conv[:len(ret)]
        full_window = (ret.notna().astype(float)
                       .rolling(n_days, min_periods=n_days).sum() == n_days).values
        F = np.where(full_window, F, np.nan)
        return pd.DataFrame(F / w.sum(), index=ret.index, columns=ret.columns)

    out = {}
    out["exp_wgt_return_1m"] = _exp_wgt_return(20, 4)
    out["exp_wgt_return_12m"] = _exp_wgt_return(240, 4)
    turn_y = turn.rolling(240, min_periods=120).mean()
    out["bias_turn"] = turn.rolling(20, min_periods=10).mean() / turn_y - 1
    amount = turn * cap  # 成交额近似 = 换手率 × 市值
    out["amihud"] = (ret.abs() / amount.replace(0, np.nan)).rolling(
        20, min_periods=10).mean()
    return out


def _fundamental_panels() -> dict[str, pd.DataFrame]:
    """基本面因子（月度）：复用 build_fundamental_factors 的 PIT 面板构建。

    注意 load_daily 需显式 pool="all_a"（config 默认池是 hs300，
    会导致基本面只覆盖 ~340 只）。
    """
    from data.cache import DataCache
    from data.cache_helpers import load_daily, load_financial_tables
    from data.offline import OfflineQuietDataSource
    from data.universe import Universe
    from scripts.build_fundamental_factors import build_factor_panels

    cache = DataCache(OfflineQuietDataSource())
    uni = Universe(cache)
    codes, cal, daily = load_daily(cache, uni, "000300.SH", 20190101, None,
                                   pool="all_a")
    fin = load_financial_tables(cache, codes)
    log.info("基本面数据(全A): daily %d 行, income %d 行, %d 只",
             len(daily), len(fin["income"]), len(codes))
    panels = build_factor_panels(daily, cal, fin["income"],
                                 fin["balance_sheet"], fin["cash_flow"],
                                 fin["equity_structure"], fin["dividend"],
                                 fin["share_holder"], fin["holder_num"])
    keep = ["roe_ttm", "ep_ttm", "fcf_yield", "np_growth_yoy",
            "rev_growth_yoy", "holder_num_zs", "ln_mktcap"]
    return {k: panels[k] for k in keep if k in panels}


def ensure_style_panels(force: bool = False) -> pd.DataFrame:
    """月度风格因子长表缓存（date=月末交易日, code, factor, value）。"""
    if STYLE_CACHE.exists() and not force:
        return pd.read_parquet(STYLE_CACHE)
    panels: dict[str, pd.DataFrame] = {}
    panels.update(_technical_panels())
    panels.update(_fundamental_panels())
    log.info("风格因子: %s", sorted(panels))
    close = pd.read_parquet(BASE / "close_adj.parquet")
    month_end = close.index.to_series().groupby(close.index.to_period("M")).last()
    rows = []
    for name, pnl in panels.items():
        pnl = pnl.copy()
        pnl.index = pd.to_datetime(pnl.index).normalize()
        pnl = pnl.reindex(index=month_end.values)
        stack = pnl.stack().rename("value").reset_index()
        stack.columns = ["date", "code", "value"]
        stack["factor"] = name
        rows.append(stack)
    long_df = pd.concat(rows, ignore_index=True)
    long_df.to_parquet(STYLE_CACHE, compression="snappy")
    log.info("风格面板缓存: %d 行 → %s", len(long_df), STYLE_CACHE)
    return long_df


# ---------------------------------------------------------------------------
# 月度截面处理
# ---------------------------------------------------------------------------
def _winsor_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if mad and mad > 0:
        s = s.clip(med - 5 * mad, med + 5 * mad)
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * np.nan


def _neutralize(g: pd.DataFrame, factor: str) -> pd.Series:
    """行业+市值中性化：对 [行业哑变量, ln_mktcap] 回归取残差。"""
    y = g[factor]
    X = [pd.Series(1.0, index=g.index)]
    if "ln_mktcap" in g:
        X.append(g["ln_mktcap"].fillna(g["ln_mktcap"].median()))
    if "industry" in g:
        X.extend(pd.get_dummies(g["industry"], drop_first=True).astype(float)
                 .reindex(g.index).fillna(0).pipe(lambda d: [d[c] for c in d.columns]))
    X = pd.concat(X, axis=1)
    ok = y.notna() & X.notna().all(axis=1)
    if ok.sum() < 30:
        return y
    beta, *_ = np.linalg.lstsq(X[ok].values, y[ok].values, rcond=None)
    res = pd.Series(np.nan, index=g.index)
    res[ok] = y[ok].values - X[ok].values @ beta
    return res


# 因子方向（AI 63 图表10）：+ 为取值越大越好
DIRECTION = {
    "roe_ttm": +1, "ep_ttm": +1, "fcf_yield": +1,
    "np_growth_yoy": +1, "rev_growth_yoy": +1,
    "holder_num_zs": -1,
    "exp_wgt_return_1m": -1, "exp_wgt_return_12m": -1,
    "bias_turn": -1, "amihud": +1, "ln_mktcap": -1,
}


def run(text_factor_file: str, tag: str | None = None,
        top_n: int = 25, force_style: bool = False):
    tag = tag or Path(text_factor_file).stem.replace("fadt_factor_", "")

    # ── 数据 ──
    close = pd.read_parquet(BASE / "close_adj.parquet")
    monthly_last = close.groupby(close.index.to_period("M")).last()
    ret_wide = monthly_last.pct_change().shift(-1)  # Period index
    industry = pd.read_parquet(BASE / "industry.parquet")
    cap = pd.read_parquet(BASE / "market_cap.parquet")

    style = ensure_style_panels(force=force_style)
    style["month"] = pd.to_datetime(style["date"]).dt.to_period("M")
    style_pv = style.pivot_table(index=["month", "code"], columns="factor",
                                 values="value")

    tf = pd.read_parquet(OUT_DIR / text_factor_file).reset_index()
    tf.columns = [c.lower() for c in tf.columns]
    tf["month"] = pd.to_datetime(tf["date"]).dt.to_period("M")
    tf = tf.rename(columns={"factor": "text"})[["month", "code", "text"]]

    # 行业/市值取月末交易日行后转 Period（日频直转会产生重复月标签）
    month_ends = set(close.index.to_series()
                     .groupby(close.index.to_period("M")).last())

    def _monthly(pnl: pd.DataFrame) -> pd.DataFrame:
        p = pnl[pnl.index.isin(month_ends)].copy()
        p.index = pd.to_datetime(p.index).to_period("M")
        return p[~p.index.duplicated(keep="last")]

    ind_m = _monthly(industry)
    cap_m = _monthly(cap)

    months = sorted(set(tf["month"]) & set(ret_wide.index))
    log.info("文本因子 %d 月（%s ~ %s）", len(months), months[0], months[-1])

    # ── 逐月构建组合 ──
    port_rows: dict[str, list] = {k: [] for k in
                                  ("enhanced", "text_only", "style_only")}
    hold_rows: dict[str, list] = {k: [] for k in port_rows}
    prev: dict[str, set] = {}
    for m in months:
        g = tf[tf["month"] == m].set_index("code").copy()
        if len(g) < 50:
            continue
        sv = style_pv.xs(m, level="month") if m in style_pv.index.get_level_values(0) else None
        g = g.join(sv) if sv is not None else g
        g["ln_mktcap"] = np.log(cap_m.loc[m].reindex(g.index)) if m in cap_m.index else np.nan
        g["industry"] = ind_m.loc[m].reindex(g.index) if m in ind_m.index else np.nan

        # 风格 z（中性化后）
        zcols = []
        for f, d in DIRECTION.items():
            if f not in g:
                continue
            z = _winsor_z(g[f].astype(float))
            z = _neutralize(g.assign(**{f: z}), f) if f != "ln_mktcap" else z
            g[f"z_{f}"] = z * d
            zcols.append(f"z_{f}")
        zc = g[zcols]
        g["style_z"] = zc.mean(axis=1).where(zc.notna().sum(axis=1) >= 5)

        g["text_z"] = _winsor_z(g["text"].astype(float))
        base = g.dropna(subset=["text_z"])
        n_top = max(int(len(base) * 0.1), 30)  # 分 10 层首层（≥30 只）
        base = base.nlargest(n_top, "text_z")

        cand = {
            "enhanced": base.dropna(subset=["style_z"]).nlargest(top_n, "style_z").index,
            "text_only": base.nlargest(top_n, "text_z").index,
            "style_only": g.dropna(subset=["style_z"]).nlargest(top_n, "style_z").index,
        }
        for k, codes in cand.items():
            hold_rows[k].append({"month": m, "codes": "|".join(map(str, codes)),
                                 "n": len(codes)})
            prev_set = prev.get(k, set())
            codes_set = set(codes)
            traded = len(codes_set - prev_set) / max(len(codes_set), 1)
            port_rows[k].append({"month": m, "ret_gross": ret_wide.loc[m, list(codes)].mean(),
                                 "turnover": traded})
            prev[k] = codes_set

    # ── 组装净值与绩效 ──
    def perf(df: pd.DataFrame) -> dict:
        r = df["ret_gross"].values
        to = df["turnover"].values
        r_net = r - COST_RT * to  # 单边换手 × 双边费率
        nav = (1 + pd.Series(r_net)).cumprod()
        ann = (1 + r_net).prod() ** (12 / len(r_net)) - 1
        vol = np.std(r_net, ddof=1) * np.sqrt(12)
        dd = (nav / nav.cummax() - 1).min()
        return {"ann_ret": ann, "ann_vol": vol, "sharpe": ann / vol if vol else np.nan,
                "max_dd": dd, "n_months": len(r_net),
                "avg_turnover_1s": to.mean()}

    summary = {"tag": tag}
    ports = {}
    for k in port_rows:
        df = pd.DataFrame(port_rows[k]).set_index("month").sort_index()
        ports[k] = df
        p = perf(df)
        # 全A等权基准（同一批月份）
        bench_m = ret_wide.loc[df.index].mean(axis=1)
        ex = df["ret_gross"].values - COST_RT * df["turnover"].values - bench_m.values
        ir = ex.mean() / ex.std() * np.sqrt(12) if ex.std() else np.nan
        p["excess_ann_eqw"] = (1 + ex).prod() ** (12 / len(ex)) - 1
        p["ir_eqw"] = ir
        p["win_rate_eqw"] = (ex > 0).mean()
        summary[k] = {k2: round(v, 4) for k2, v in p.items()}

    sm = pd.DataFrame(summary).T
    print("\n== 因子增强 Top25 组合（基准 全A等权，双边3‰）==")
    print(sm.to_string())

    out_md = OUT_DIR / f"enhanced_top25_{tag}.md"
    lines = [f"# 因子增强 Top25（text={text_factor_file}，top{top_n}，双边3‰）",
             "", sm.to_string(), "",
             "口径：基础池=文本因子十分层首层（≥30 只）；enhanced=池内风格合成 Top25；"
             "text_only=池内文本 Top25；style_only=全截面风格 Top25。"
             "风格合成=11 因子行业市值中性化 z 均值（≥5 个有效）。",
             "AI 63/92 因子映射与替代见脚本 docstring（trans_at_last_ratio 跳过）。"]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    for k, df in ports.items():
        out = df.copy()
        out.index = out.index.to_timestamp()
        out.assign(ret_net=out["ret_gross"] - COST_RT * out["turnover"]).to_parquet(
            OUT_DIR / f"enhanced_top25_{tag}_{k}.parquet", compression="snappy")
    log.info("已存: %s", out_md)
    return sm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-factor", default="fadt_factor_bert_xgb_zz1000.parquet")
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force-style", action="store_true")
    args = ap.parse_args()
    run(args.text_factor, args.tag, args.top_n, args.force_style)
