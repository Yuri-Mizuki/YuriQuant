"""yqcore 三档 Barra 风格归因（TODO P1 ⑬，2026-11-25 建）。

问题：三档对外报的「超额」只控了市场一个因子。已测暴露是 breadth resid_vol
z −22 / btop z +24 / 股息 z +9 ⇒ 大概率是「低波 + 价值」的打包价，不是选股 α。
本脚本把月收益对「市场 + 风格」做时序回归，看控住风格后残差 α 还剩多少。

口径
----
- **风格收益**：``research.barra.run_factor_returns`` 的逐日 WLS 截面因子收益
  （申万一级哑变量并入截面回归 ⇒ 风格收益已行业中性），日频复利到月频。
- **市场**：全A 月频等权 buy-and-hold（``breadth_alpha.eqw_monthly``，与公平基准
  同口径；不用上证，理由见 ``reports/breadth_alpha``）。
- **因变量**：三档月收益（``reports/yq_core/nav_attrib.csv``，uniform 1e6 无整手）。
  **不扣无风险利率**（rf=0）⇒ α 是「相对现金」的月度超额，与既有 CAPM 读数同口径。
- **显著性**：Newey-West HAC，月度 lag=3（``stats.robust_stats.ols_newey_west``）。

模型
----
====  ==========================================================
M1    市场单因子（单因子 CAPM；对齐旧归档「+7.67%/年」那一族读数）
M2    市场 + 5 核心风格：规模 / 低波 / 价值 / 动量 / 流动性（TODO 口径）
M3    市场 + 全部 13 个风格描述变量（稳健性；含 nl_size / 杠杆 3 项等）
====  ==========================================================

判据（TODO ⑬）：M2 的 α 仍 >0 且 t > 2 → 保留 alpha 称谓；否则三档降级为
「风格产品」。

**信号口径（2026-11-25 由莉酱定）**：三档统一用 **f50 数据态**。breadth 的登记信号
``dingban_f50`` 本机现成；集中档登记信号 ``gbdt_mid13`` 指向的 882 ortho pred 本机
已不存在，且 feature budget 已切 f50 ⇒ 用 ``reports/alla_rolling_ab_f50/pred`` 的
四个 gbdt 面板现场拼 ``gbdt_mid13``（2:1:1:2 秩平均），落盘为
``ab_f50/pred/gbdt_mid13__h1.parquet``。这是**迁移后的集中档**，不是 882 旧口径。

用法::

    python -m scripts.portfolio.style_attribution --build-nav  # 三档净值（f50 数据态）
    python -m scripts.portfolio.style_attribution --build-nav --arm reg   # 登记 882 口径
    python -m scripts.portfolio.style_attribution --compare     # 集中档 882 vs f50 对照
    python -m scripts.portfolio.style_attribution              # 归因（缓存风格月收益）
    python -m scripts.portfolio.style_attribution --rebuild    # 强制重算风格月收益
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

from research.barra import (  # noqa: E402
    STYLE_VARS, load_barra_inputs, run_factor_returns)
from scripts.portfolio.breadth_alpha import eqw_monthly, to_monthly  # noqa: E402
from stats.robust_stats import ols_newey_west  # noqa: E402

OUT = ROOT / "reports" / "style_attribution"
NAV_CSV = ROOT / "reports" / "yq_core" / "nav_attrib.csv"
NAV_CSV_REG = ROOT / "reports" / "yq_core" / "nav_attrib_882.csv"
TIERS = ("yqcore-breadth", "yqcore-focus-k15", "yqcore-focus-k8")
CORE5 = ("lncap", "resid_vol", "btop", "momentum", "liquidity")
NW_LAG = 3
FR_CACHE = OUT / "factor_returns_monthly.csv"

#: f50 数据态：信号面板目录 + 集中档复合信号（2:1:1:2 秩平均）
F50_PRED = ROOT / "reports" / "alla_rolling_ab_f50" / "pred"
F50_COMPOSITE = F50_PRED / "gbdt_mid13__h1.parquet"
END_ALIGN = "2026-09-07"      # 与登记表 check_end 对齐（三档同末日）
CAPITAL = 1.0e6               # uniform 口径：三档同资金、无整手


# ---------------------------------------------------------------------------
# 净值（f50 数据态）
# ---------------------------------------------------------------------------
def build_f50_composite(force: bool = False) -> Path:
    """用 ab_f50 的 gbdt h1/h5/h10/h20 拼 ``gbdt_mid13``（与 caliber 复合口径同源）。"""
    if F50_COMPOSITE.exists() and not force:
        return F50_COMPOSITE
    from scripts.evaluation.caliber import weighted_rank_average
    parts = [pd.read_parquet(F50_PRED / f"gbdt__h{h}.parquet")
             for h in (1, 5, 10, 20)]
    comp = weighted_rank_average(parts, [2.0, 1.0, 1.0, 2.0])
    comp.to_parquet(F50_COMPOSITE)          # 保留 date 索引（load_context 用它当交易日轴）
    print(f"复合信号落盘 {F50_COMPOSITE.name}  {comp.shape}")
    return F50_COMPOSITE


def build_nav(force: bool = False, arm: str = "f50") -> pd.DataFrame:
    """三档日净值（uniform 1e6 无整手，末日至 END_ALIGN）。

    ``arm="f50"``：集中档用 ``ab_f50`` 拼的 ``gbdt_mid13``（迁移后口径）；
    ``arm="reg"``：集中档走登记口径（``reports/alla_rolling_ortho/pred`` 现场复合）。
    """
    import gc

    from scripts.evaluation import caliber as cal
    from strategy.registry import get_tier

    comp = build_f50_composite(force) if arm == "f50" else None
    cost = cal.cost_profile("production")
    curves, recs = {}, []
    for tier in TIERS:
        spec = get_tier(tier)
        pred_path = None if (tier == "yqcore-breadth" or arm == "reg") else comp
        ctx = cal.load_context(signal=spec.runnable_signal, window="prod",
                               end=int(END_ALIGN.replace("-", "")),
                               execution="open", pred_path=pred_path)
        overrides = {}
        if spec.panel is not None:
            ind = pd.read_parquet(cal.BASE_DIR / f"{spec.panel}.parquet")
            overrides[spec.panel] = ind.reindex(index=ctx.pred.index,
                                                columns=ctx.pred.columns)
        strat = spec.build(**overrides)
        bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                initial_capital=CAPITAL, costs=cost)
        res = cal.run_once(bt, ctx, "open")
        m = cal.metrics_row(res, CAPITAL)
        dr = res.daily_returns
        curves[tier] = (1 + dr).cumprod()
        recs.append({"tier": tier, "signal": ctx.signal,
                     "fingerprint": ctx.fingerprint,
                     "window": f"{ctx.pred.index.min().date()}..{ctx.pred.index.max().date()}",
                     "ann": m["ann"], "sharpe": m["sharpe"],
                     "max_dd": m["max_dd"], "turnover": m["turnover"]})
        del ctx, res, strat, bt
        gc.collect()
    nav = pd.DataFrame(curves).sort_index()
    nav.index.name = "date"
    out_csv = NAV_CSV if arm == "f50" else NAV_CSV_REG
    nav.to_csv(out_csv)
    pd.DataFrame(recs).to_csv(OUT / f"nav_metrics_{arm}.csv", index=False)
    print(pd.DataFrame(recs).round(4).to_string(index=False))
    print(f"净值写入 {out_csv}")
    return nav


# ---------------------------------------------------------------------------
# 数据层
# ---------------------------------------------------------------------------
def style_monthly(rebuild: bool = False) -> pd.DataFrame:
    """13 个风格的月度组合收益（Barra WLS 因子收益 → 月频复利）。"""
    if FR_CACHE.exists() and not rebuild:
        return pd.read_csv(FR_CACHE, index_col=0, parse_dates=True)

    inputs = load_barra_inputs()
    fr = run_factor_returns(inputs, begin="2018-12-01")
    sty_cols = [c for c in fr.columns if c.startswith("sty:")]
    if len(sty_cols) != len(STYLE_VARS):
        raise RuntimeError(f"风格因子列缺失：{set(STYLE_VARS) - {c[4:] for c in sty_cols}}")
    out = pd.DataFrame({c[4:]: to_monthly(fr[c]) for c in sty_cols})
    meta = fr.attrs.get("meta")
    if meta is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        meta.to_csv(OUT / "factor_regression_meta.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(FR_CACHE)
    return out


def tier_monthly() -> pd.DataFrame:
    """三档月度收益（首月为不完整月，剔除；末日 2026-09 为半月，保留并与既有
    ``breadth_alpha`` 的 ``iloc[1:]`` 口径一致）。"""
    nav = pd.read_csv(NAV_CSV, index_col=0, parse_dates=True)
    missing = [t for t in TIERS if t not in nav.columns]
    if missing:
        raise KeyError(f"{NAV_CSV.name} 缺档位 {missing}")
    daily = nav[list(TIERS)].pct_change(fill_method=None).fillna(0.0)
    m = pd.DataFrame({t: to_monthly(daily[t]) for t in TIERS})
    return m.iloc[1:]


# ---------------------------------------------------------------------------
# 回归
# ---------------------------------------------------------------------------
def fit(y: pd.Series, mkt: pd.Series, styles: pd.DataFrame,
        names: tuple[str, ...]) -> dict:
    """``y = α + β_m·mkt + Σ β_s·style_s + ε``，HAC(lag=3)。"""
    X = np.column_stack([np.ones(len(y)), mkt.to_numpy()]
                        + [styles[n].to_numpy() for n in names])
    r = ols_newey_west(X, y.to_numpy(), lag=NW_LAG)
    beta, t, p = r["beta"], r["t_nw"], r["p_nw"]
    resid = r["resid"]
    out = {
        "n": int(r["n"]),
        "alpha_m": float(beta[0]),
        "alpha_ann": float((1 + beta[0]) ** 12 - 1),
        "t_alpha": float(t[0]),
        "p_alpha": float(p[0]),
        "beta_mkt": float(beta[1]),
        "t_mkt": float(t[1]),
        "r2": float(1 - resid.var(ddof=1) / y.var(ddof=1)),
        "lag_used": int(r["lag_used"]),
        "betas": {n: (float(b), float(tt), float(pp))
                  for n, b, tt, pp in zip(names, beta[2:], t[2:], p[2:])},
    }
    return out


def _stats(r: pd.Series) -> dict:
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    dd = (1 + r).cumprod()
    return {"ann": float(ann),
            "sharpe": float(ann / vol) if vol > 0 else float("nan"),
            "mdd": float((dd / dd.cummax() - 1).min())}


def compare() -> int:
    """集中档 f50 vs 882 同口径对照（净值 → 月收益；配对差 + HAC t）。"""
    from stats.robust_stats import nw_tstat

    a = pd.read_csv(NAV_CSV, index_col=0, parse_dates=True)      # f50
    b = pd.read_csv(NAV_CSV_REG, index_col=0, parse_dates=True)  # 882 登记口径
    rows = []
    for tier in TIERS[1:]:
        j = pd.concat([a[tier].pct_change().rename("f50"),
                       b[tier].pct_change().rename("reg")],
                      axis=1).dropna()
        ma = to_monthly(j["f50"]).iloc[1:]
        mb = to_monthly(j["reg"]).iloc[1:]
        t, _, _ = nw_tstat((ma - mb).to_numpy(), lag=NW_LAG)
        sa, sb = _stats(ma), _stats(mb)
        rows.append({"tier": tier, "n_month": len(ma),
                     "f50_ann": sa["ann"], "reg_ann": sb["ann"],
                     "diff_ann": sa["ann"] - sb["ann"],
                     "f50_sharpe": sa["sharpe"], "reg_sharpe": sb["sharpe"],
                     "f50_mdd": sa["mdd"], "reg_mdd": sb["mdd"],
                     "t_diff": t})
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    df.to_csv(OUT / "compare_882_vs_f50.csv", index=False)
    return 0


def timing() -> int:
    """信号 TopK 篮子的**事前风格暴露**是否领先下一月风格收益（beta 择时检验）。

    ``e_t`` = 信号 TopK 篮子在月末 t 的 13 风格**截面 z 均值**（只用到 t 时点信息）；
    ``f_{t+1}`` = 该风格 t+1 月收益（Barra WLS 因子收益）。
    ``corr(e_t, f_{t+1})>0`` 显著 ⇒ 信号在**提前摆 beta**（择时）；
    ≈0 ⇒ 收益全部来自静态 carry（暴露长期不变、吃因子均值）。
    同时给 ``corr(e_t, f_t)``（同期）：显著为正说明只是追当期强势风格，不是预测。
    """
    from scripts.common.alla_common import ds_root
    from scripts.evaluation import caliber as cal
    from stats.robust_stats import nw_tstat

    sty = style_monthly()
    sig = cal._composite_panel("gbdt_mid13_f50")
    dates = sty.index.intersection(sig.index)
    k_map = {"yqcore-focus-k15": 15, "yqcore-focus-k8": 8}
    rows = []
    exps: dict[tuple[str, str], pd.Series] = {}
    for var in STYLE_VARS:
        raw = pd.read_parquet(ds_root() / "panels_barra" / f"{var}.parquet")
        r = raw.reindex(index=dates, columns=sig.columns).astype("float64")
        z = r.sub(r.mean(axis=1), axis=0).div(
            r.std(axis=1).replace(0.0, np.nan), axis=0)
        sv = sig.reindex(index=dates, columns=sig.columns).to_numpy(dtype="float64")
        zv = z.to_numpy()
        for tier, k in k_map.items():
            e = np.full(len(dates), np.nan)
            for i in range(len(dates)):
                s, zz = sv[i], zv[i]
                ok = np.isfinite(s) & np.isfinite(zz)
                if ok.sum() < max(3 * k, 30):
                    continue
                pick = np.argsort(-s[ok])[:k]
                e[i] = zz[ok][pick].mean()
            es = pd.Series(e, index=dates)
            exps[(tier, var)] = es
            f = sty[var].reindex(dates)
            d = pd.concat([es.rename("e"), f.rename("f")], axis=1).dropna()
            if len(d) < 24:
                continue
            ec, fc = d["e"] - d["e"].mean(), d["f"] - d["f"].mean()
            fn = fc.shift(-1)
            t_next, _, _ = nw_tstat((ec * fn).dropna().to_numpy(), lag=NW_LAG)
            rows.append({"tier": tier, "style": var, "n": len(d),
                         "exp_mean": float(d["e"].mean()),
                         "exp_sd": float(d["e"].std()),
                         "corr_next": float(ec.corr(fn)),
                         "t_next": t_next,
                         "corr_same": float(ec.corr(fc))})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "timing.csv", index=False)
    for tier in k_map:
        d = df[df.tier == tier].reindex(
            df[df.tier == tier]["corr_next"].abs().sort_values(ascending=False).index)
        print(f"\n=== {tier}：事前暴露 vs 下一月风格收益（|corr_next| 降序）")
        print(d[["style", "exp_mean", "exp_sd", "corr_next", "t_next",
                 "corr_same"]].round(3).to_string(index=False))
        print(f"  |corr_next|>0.2 且 t>2 的风格数："
              f"{int(((d['corr_next'].abs() > 0.2) & (d['t_next'] > 2)).sum())}/13")

    # ---- 多周期：暴露 vs 未来 3/6/12 个月风格收益（季度重训 ⇒ 兑现可能更慢）----
    print("\n=== 多周期检验（e_t vs 未来 H 月累计风格收益，t 用 lag=H）")
    for tier in k_map:
        for H in (3, 6, 12):
            rec = []
            for var in STYLE_VARS:
                e = exps[(tier, var)]
                f = sty[var].reindex(dates)
                fwd = ((1 + f).rolling(H).apply(np.prod, raw=True) - 1).shift(-H)
                d = pd.concat([e.rename("e"), fwd.rename("f")], axis=1).dropna()
                if len(d) < 20:
                    continue
                ec, fc = d["e"] - d["e"].mean(), d["f"] - d["f"].mean()
                t, _, _ = nw_tstat((ec * fc).to_numpy(), lag=H)
                rec.append({"style": var, "corr": float(ec.corr(fc)), "t": t})
            rr = pd.DataFrame(rec).sort_values(
                "corr", key=lambda s: s.abs(), ascending=False)
            print(f"\n  {tier} H={H}月")
            print(rr.round(3).head(5).to_string(index=False))
            print(f"  |corr|>0.2 且 t>2："
                  f"{int(((rr['corr'].abs() > 0.2) & (rr['t'] > 2)).sum())}/13")

    # ---- 反向：暴露是否在追「已经发生」的风格（重训拟合近期 regime）----
    print("\n=== 暴露 vs 过去 12 月风格收益（追涨已发生的风格？）")
    for tier in k_map:
        rec = []
        for var in STYLE_VARS:
            e = exps[(tier, var)]
            f = sty[var].reindex(dates)
            past = ((1 + f).rolling(12).apply(np.prod, raw=True) - 1)
            d = pd.concat([e.rename("e"), past.rename("p")], axis=1).dropna()
            if len(d) < 24:
                continue
            rec.append({"style": var, "corr": float(d["e"].corr(d["p"]))})
        rr = pd.DataFrame(rec).sort_values(
            "corr", key=lambda s: s.abs(), ascending=False)
        print(f"\n  {tier}")
        print(rr.round(3).head(5).to_string(index=False))

    # ---- H=12 非重叠复核：12 个起点子样本 + 循环块 bootstrap ----
    print("\n=== H=12 非重叠复核（12 个起点各取每 12 个月；块 bootstrap B=500）")
    rng = np.random.default_rng(42)
    for tier in k_map:
        rec = []
        for var in STYLE_VARS:
            e = exps[(tier, var)]
            f = sty[var].reindex(dates)
            fwd = ((1 + f).rolling(12).apply(np.prod, raw=True) - 1).shift(-12)
            d = pd.concat([e.rename("e"), fwd.rename("f")], axis=1).dropna()
            if len(d) < 60:
                continue
            ev, fv = d["e"].to_numpy(), d["f"].to_numpy()
            cs = []
            for off in range(12):                      # 12 个相位起点，各 ~7 个独立点
                a, b = ev[off::12], fv[off::12]
                if len(a) >= 5 and a.std() > 0 and b.std() > 0:
                    cs.append(float(np.corrcoef(a, b)[0, 1]))
            n, nb, boot = len(d), int(np.ceil(len(d) / 12)), []
            for _ in range(500):                       # 循环块 bootstrap（块长 12）
                starts = rng.integers(0, n, nb)
                idx = np.concatenate(
                    [s + np.arange(12) for s in starts]) % n
                idx = idx[:n]
                if ev[idx].std() > 0 and fv[idx].std() > 0:
                    boot.append(float(np.corrcoef(ev[idx], fv[idx])[0, 1]))
            rec.append({"style": var,
                        "full": float(np.corrcoef(ev, fv)[0, 1]),
                        "off_med": float(np.median(cs)) if cs else np.nan,
                        "off_pos": int(sum(c > 0 for c in cs)), "off_n": len(cs),
                        "boot_p5": float(np.percentile(boot, 5)),
                        "boot_p50": float(np.percentile(boot, 50)),
                        "boot_p95": float(np.percentile(boot, 95))})
        rr = pd.DataFrame(rec).sort_values(
            "full", key=lambda s: s.abs(), ascending=False)
        print(f"\n  {tier}")
        print(rr.round(3).head(6).to_string(index=False))

    pd.DataFrame({f"{t}|{v}": s for (t, v), s in exps.items()}).to_csv(
        OUT / "basket_exposures.csv")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rebuild", action="store_true", help="重算风格月收益（默认读缓存）")
    ap.add_argument("--build-nav", action="store_true", help="只出三档净值")
    ap.add_argument("--arm", choices=["f50", "reg"], default="f50",
                    help="净值口径：f50=集中档用 ab_f50 拼 gbdt_mid13；reg=登记 882 口径")
    ap.add_argument("--compare", action="store_true",
                    help="集中档 f50 vs 882 同口径对照（需先出两套净值）")
    ap.add_argument("--timing", action="store_true",
                    help="事前风格暴露 vs 下一月风格收益（beta 择时检验）")
    ap.add_argument("--force-nav", action="store_true", help="一并重建复合信号与净值")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    if a.build_nav:
        build_nav(a.force_nav, a.arm)
        return 0
    if a.compare:
        return compare()
    if a.timing:
        return timing()

    if not NAV_CSV.exists():
        raise SystemExit(f"缺 {NAV_CSV}，先跑：python -m scripts.portfolio.style_attribution --build-nav")

    tm = tier_monthly()
    styles = style_monthly(a.rebuild).reindex(tm.index)
    mkt, turn = eqw_monthly(tm.index)
    if mkt.isna().any():
        raise RuntimeError("市场因子与三档月度网格未对齐")
    # 末日 2026-09 只有 09-07 前的 5 个交易日（净值被 END_ALIGN 截断），风格月收益
    # 按整月算 ⇒ 该月风格为 NaN。为保证 M1/M2/M3 同样本，一律剔除这些不完整月。
    keep = styles.notna().all(axis=1)
    if (~keep).any():
        print(f"剔除不完整月 {list(tm.index[~keep].date)}")
    tm, styles, mkt = tm[keep], styles[keep], mkt[keep]
    print(f"月度样本 {tm.index[0].date()} ~ {tm.index[-1].date()}（{len(tm)} 月）；"
          f"市场 = 全A月频等权（月换手 {turn:.2%}）")

    all_styles = tuple(v for v in STYLE_VARS)
    models = {
        "M1_市场": (),
        "M2_核心5风格": CORE5,
        "M3_全13风格": all_styles,
    }
    rows, beta_rows = [], []
    for tier in TIERS:
        y = tm[tier]
        m1 = None
        print(f"\n=== {tier}")
        for mname, names in models.items():
            r = fit(y, mkt, styles, names)
            if m1 is None:
                m1 = r
            rows.append({
                "tier": tier, "model": mname, "n": r["n"],
                "alpha_m": r["alpha_m"], "alpha_ann": r["alpha_ann"],
                "t_alpha": r["t_alpha"], "p_alpha": r["p_alpha"],
                "beta_mkt": r["beta_mkt"], "t_mkt": r["t_mkt"], "r2": r["r2"],
                "alpha_ann_drop_vs_M1": r["alpha_ann"] - m1["alpha_ann"],
            })
            for k, (b, t, p) in r["betas"].items():
                f_mean = float(styles[k].mean() * 12)
                beta_rows.append({"tier": tier, "model": mname, "factor": k,
                                  "beta": b, "t": t, "p": p,
                                  "f_mean_ann": f_mean,
                                  "carry_ann": b * f_mean})
            print(f"  {mname:14s} α {r['alpha_ann']*100:+7.2f}%/年  "
                  f"t {r['t_alpha']:+6.2f}  β_mkt {r['beta_mkt']:.2f}  "
                  f"R² {r['r2']:.2f}  较 M1 缩水 {rows[-1]['alpha_ann_drop_vs_M1']*100:+.2f}pp")

    OUT.mkdir(parents=True, exist_ok=True)
    tm.to_csv(OUT / "tier_monthly.csv")
    att = pd.DataFrame(rows)
    att.to_csv(OUT / "attribution.csv", index=False)
    pd.DataFrame(beta_rows).to_csv(OUT / "betas.csv", index=False)

    print("\n=== 判据（M2 α>0 且 t>2 ⇒ 保留 alpha；否则降级为风格产品）")
    for tier in TIERS:
        r = att[(att.tier == tier) & (att.model == "M2_核心5风格")].iloc[0]
        verdict = "PASS" if (r.alpha_ann > 0 and r.t_alpha > 2) else "FAIL"
        print(f"  {tier:20s} {verdict}  α {r.alpha_ann*100:+.2f}%/年  t {r.t_alpha:+.2f}")
    print(f"\n产物：{OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
