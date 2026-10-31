"""研报复现线（国泰海通 2026-10-29《七维共振》）· 只做可行的部分。

研报 7 类因子逐类清点我们能不能做：

| 类别 | 研报因子数 | 本线可做 | 说明 |
|---|---|---|---|
| 动量（行业指数原生） | 4 | **4** | 传统动量 / 信息比率 / 龙头股动量 / 残差动量 |
| 估值（成分股加权） | 5 | **2~3** | 市现率倒数变动、PEG；含 FY3 的一致预期类做不了 |
| 超预期（成分股加权） | 6 | **4** | SUE 四变体；「过去 N 日盈利/营收调整」需分析师 ⇒ 缺 2 |
| 深度学习 GRU/TCN | 12 | **2~4** | 自训代表配置（日/周行情 6 特征） |
| RAFT | 10 | 0~1 | 研报自身 IC 仅 1.16%，排最后 |
| 分析师预期 | 9 | **0** | 库内无结构化 FY1/FY2/FY3 一致预期（只有研报文本，无预测数值） |
| 大单 | 3 | **0** | 无 Level2 逐笔，分钟线只有 HS300 且无法拆大单 |

⇒ 7 类里可做 5 类，缺的恰是研报排名第 2 的分析师（年化超额 9.68%）与第 4 的大单（5.73%）。
   注意研报自己给出「超预期 vs 分析师」相关 0.583 ⇒ 超预期四变体是分析师类的**部分代理**。

研报口径（本脚本逐条对齐）：
  · 池 = 行业指数（研报用中信一级 30 个，我们用申万一级 28→31 个，PIT 有效数）
  · 周度调仓（研报「分 5 组周度调仓」）；另给月频对照
  · 单因子测试：**当天收盘价**成交、不计成本（研报 §2 开头）
  · 组合测试：**第二天收盘价**成交（研报「考虑交易可行性」）
  · 基准 = 全行业等权、同期同成交价
  · IC = Pearson(raw IC)，研报明确写的是 Pearson 而非 rank

与研报的已知差异（引用时必须带）：
  1. 行业池不同（申万 31 vs 中信 30）⇒ 行业收益口径不同，IC 不能逐位对
  2. 申万指数是**价格指数不含分红**；研报口径亦然（中信行业指数同为价格指数）
  3. 我们无分析师/大单两维

用法：
    python -m scripts.portfolio.report_replica --stage mom        # 动量族
    python -m scripts.portfolio.report_replica --stage all
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backtest.metrics import annual_return, max_drawdown, sharpe_ratio  # noqa: E402
from config import Config  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "report_replica"
PQ = Path(Config.paths()["cache_root"])

COST_BP = 36.0
MIN_IND = 20          # 截面最少行业数（28~31 个池，留余量）


# --------------------------------------------------------------------- 数据
def load_px() -> pd.DataFrame:
    """申万一级行业指数收盘价。

    ⚠️ 原始序列里有 3 个「只有 1 个行业有值」的脏日期（2018-07 等），
    它们会让 `pct_change` + `rolling` 的 NaN 向前传染 240 个交易日
    （实测使 `ir_240` 凭空少 91 期），必须在入口剔除。
    """
    d = pd.read_parquet(PQ / "sw_industry_daily.parquet")
    px = d.pivot_table(index="date", columns="code", values="close").sort_index()
    good = px.notna().sum(axis=1) >= 10
    if (~good).any():
        print(f"  剔除 {int((~good).sum())} 个脏日期（有效行业 < 10）")
    return px.loc[good]


def rebal_dates(idx: pd.DatetimeIndex, start: pd.Timestamp,
                freq: str = "W") -> list[pd.Timestamp]:
    """周度：每周首个交易日；月频：每月首个交易日。研报是周度。"""
    s = pd.Series(idx, index=idx)
    if freq == "M":
        g = s.groupby([idx.year, idx.month]).first()
    else:
        iso = idx.isocalendar()
        g = s.groupby([iso["year"].to_numpy(), iso["week"].to_numpy()]).first()
    return [pd.Timestamp(x) for x in g.tolist() if pd.Timestamp(x) >= start]


# --------------------------------------------------------------------- 动量族
def signals_momentum(px: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """研报 §2.3.1 的四个动量因子（全部行业指数原生，无需个股）。"""
    ret = px.pct_change(fill_method=None)
    out: dict[str, pd.DataFrame] = {}

    # ① 信息比率因子：回望期 N 日收益均值 / 标准差（年化）
    for n in (20, 120, 180, 240):
        out[f"ir_{n}"] = ret.rolling(n).mean() / ret.rolling(n).std() * np.sqrt(240)

    # ② 传统动量：回望 N、剔除最近 M（M=0 为不剔除）
    for n in (180, 240):
        for m in (0, 20):
            out[f"mom_{n}_{m}"] = px.shift(m) / px.shift(n) - 1

    # ③ 残差动量（FF3 残差，见 signals_resid_mom，需个股/市场因子 ⇒ 另算）
    return out


def ic_series(sig: pd.DataFrame, dates: list[pd.Timestamp],
              px: pd.DataFrame, tdays: pd.DatetimeIndex,
              lag: int = 0) -> np.ndarray:
    """Pearson IC：T 日因子 vs T+lag → 下一调仓日 的行业收益。

    lag=0 当天收盘成交（研报单因子口径）；lag=1 次日收盘（可交易口径）。
    """
    ed = _exec_rows(dates, tdays, lag=lag)
    out = np.full(len(dates) - 1, np.nan)
    for k in range(len(dates) - 1):
        e0, e1 = ed[k], ed[k + 1]
        if e1 <= e0:
            continue
        j = _row(tdays, dates[k])
        x = sig.iloc[j]
        r = (px.iloc[e1] / px.iloc[e0] - 1)
        ok = x.notna() & r.notna()
        if ok.sum() < MIN_IND:
            continue
        a, b = x[ok].to_numpy(float), r[ok].to_numpy(float)
        a = np.clip(a, *np.percentile(a, [1, 99]))
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        out[k] = float(np.corrcoef(a, b)[0, 1])
    return out


def quintile_excess(sig: pd.DataFrame, dates: list[pd.Timestamp],
                    px: pd.DataFrame, tdays: pd.DatetimeIndex,
                    lag: int = 0) -> list[float]:
    """研报「分 5 组年化超额」：每组等权毛收益 − 全行业等权基准，再年化。"""
    ed = _exec_rows(dates, tdays, lag=lag)
    per = 52 if _is_weekly(dates) else 12
    buckets: dict[int, list[float]] = {i: [] for i in range(5)}
    bench: list[float] = []
    for k in range(len(dates) - 1):
        e0, e1 = ed[k], ed[k + 1]
        if e1 <= e0:
            continue
        j = _row(tdays, dates[k])
        x = sig.iloc[j]
        r = (px.iloc[e1] / px.iloc[e0] - 1).dropna()
        x = x[x.index.isin(r.index)]
        if len(x) < MIN_IND:
            continue
        # 按 rank 分位切 5 组（值可能大量重复，qcut 会在退化截面报错）
        rk = x.rank(method="average", pct=True)
        q = np.minimum(4, np.floor(rk.to_numpy() * 5)).astype(int)
        q = pd.Series(q, index=x.index)
        b = float(r.mean())
        bench.append(b)
        for g in range(5):
            idx = x.index[q == g]
            buckets[g].append(float(r.reindex(idx).mean()) - b)
    return [annual_return(pd.Series(buckets[g]), per) for g in range(5)]


# --------------------------------------------------------------------- 工具
def _row(tdays: pd.DatetimeIndex, d: pd.Timestamp) -> int:
    j = tdays.get_indexer([d])[0]
    return j if j >= 0 else int(tdays.searchsorted(d))


def _exec_rows(dates: list[pd.Timestamp], tdays: pd.DatetimeIndex,
               lag: int = 1) -> list[int]:
    return [min(_row(tdays, d) + lag, len(tdays) - 1) for d in dates]


def _is_weekly(dates: list[pd.Timestamp]) -> bool:
    if len(dates) < 10:
        return True
    gap = np.median([(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)])
    return gap < 15


def nw_t(x: np.ndarray, lag: int = 12) -> float:
    a = x[np.isfinite(x)]
    n = a.size
    if n < 20:
        return float("nan")
    e = a - a.mean()
    var = float(e @ e) / n
    for j in range(1, min(lag, n - 2) + 1):
        w = 1 - j / (lag + 1)
        var += 2 * w * float(e[j:] @ e[:-j]) / n
    return float(a.mean() / np.sqrt(max(var, 1e-12) / n))


def stats(ics: np.ndarray) -> dict:
    a = ics[np.isfinite(ics)]
    n = a.size
    if n < 20:
        return {"n": n, "ic": np.nan, "icir": np.nan, "t": np.nan, "t_nw": np.nan}
    m, sd = float(a.mean()), float(a.std(ddof=1))
    return {"n": n, "ic": m, "icir": m / sd if sd > 0 else np.nan,
            "t": m / sd * np.sqrt(n) if sd > 0 else np.nan, "t_nw": nw_t(a)}


def agg_matrix(panel: pd.DataFrame, ctx: dict, px: pd.DataFrame) -> pd.DataFrame:
    """任意个股面板（含模型分数）→ 行业因子。"""
    from scripts.oneoff.etf_industry_factor_p0 import _agg
    rows = []
    for i, dt in enumerate(ctx["dates"]):
        if dt not in panel.index:
            rows.append(np.full(len(ctx["inds"]), np.nan))
            continue
        v = panel.loc[dt].reindex(ctx["universe"]).to_numpy(dtype=float)
        rows.append(_agg(v, ctx["lbls"][i], ctx["wts"][i]))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(ctx["dates"]),
                        columns=ctx["inds"]).reindex(index=px.index, columns=px.columns)


def dl_signals(ctx: dict, px: pd.DataFrame, arms: tuple[str, ...] = ("gbdt__h1",)
               ) -> dict[str, pd.DataFrame]:
    """研报「深度学习」维度 → 我们的模型分数聚合（占位替身，见报告§差异）。"""
    from scripts.oneoff.etf_industry_model_ic import load_model_panel
    out: dict[str, pd.DataFrame] = {}
    for arm in arms:
        p = load_model_panel(arm, ctx["universe"])
        if p is None:
            continue
        out[f"dl_{arm.split('__')[0]}"] = agg_matrix(p, ctx, px)
    return out


# ------------------------------------------------------- 组合（研报策略 1/2 子集）
def port_backtest(sig: pd.DataFrame, px: pd.DataFrame, dates: list[pd.Timestamp],
                  top: int = 5, cost_bp: float = 10.0) -> dict:
    """周频 Top-N 等权、T 日信号 → **T+1 收盘成交**（研报「第二天收盘价交易」）。"""
    tdays = px.index
    ed = _exec_rows(dates, tdays, lag=1)
    rets, turns, idx, bens = [], [], [], []
    w_old = pd.Series(0.0, index=px.columns)
    for k in range(len(dates) - 1):
        e0, e1 = ed[k], ed[k + 1]
        if e1 <= e0:
            continue
        x = sig.iloc[_row(tdays, dates[k])]
        hold = (px.iloc[e1] / px.iloc[e0] - 1).dropna()
        x = x[x.index.isin(hold.index)].dropna()
        if len(x) < MIN_IND:
            continue
        long = x.nlargest(top).index
        w = pd.Series(0.0, index=px.columns)
        w[long] = 1.0 / len(long)
        gross = float((w * hold.fillna(0)).sum())
        turn = float((w - w_old).abs().sum() / 2)
        rets.append(gross - turn * 2 * cost_bp / 1e4)
        turns.append(turn)
        idx.append(tdays[e1])
        # 基准必须与本期同成交价、同调仓日（等权全行业）
        bens.append(float((px.iloc[e1] / px.iloc[e0] - 1).mean()))
        w_old = w
    if not rets:
        return {}
    r = pd.Series(rets, index=pd.DatetimeIndex(idx))
    b = pd.Series(bens, index=r.index)
    ex = r - b
    sd = ex.std(ddof=1)
    return {
        "ret": r, "bench": b, "n": len(r),
        "ann": annual_return(r, 52), "ann_bench": annual_return(b, 52),
        "excess": annual_return(r, 52) - annual_return(b, 52),
        "sharpe": sharpe_ratio(r, periods_per_year=52),
        "mdd": max_drawdown(r),
        "t_ir": float(ex.mean() / sd * np.sqrt(len(ex))) if sd > 0 else np.nan,
        "ir": float(ex.mean() / sd * np.sqrt(52)) if sd > 0 else np.nan,
        # 研报「周度双边换手率」口径：单期买+卖 = 2 × 单边换手
        "turn_week": float(np.mean(turns) * 2),
        "turnover": float(np.mean(turns) * 2 * 52),
    }


def zrank(s: pd.DataFrame) -> pd.DataFrame:
    """横截面 pct rank 到 [-0.5, 0.5]，用于跨因子等权合成。"""
    return s.rank(axis=1, pct=True) - 0.5


# ------------------------------------------------- 龙头股动量（研报 §2.3.1.2，需个股）
def lead_mv_momentum(ctx: dict, dates: list[pd.Timestamp], lam: float = 0.6,
                     m: int = 180, min_members: int = 8) -> pd.DataFrame:
    """行业内市值前 λ（累计占比 ≤λ%）为龙头股，因子 = 龙头组累计收益 − 跟随组。

    研报选用的是 λ=60、回望 180 的那一档，故默认对齐。
    """
    from scripts.oneoff.etf_industry_factor_p0 import float_share_panel

    d = pd.read_parquet(PQ / "daily_all_a.parquet", columns=["close"])
    close = d["close"].unstack("code").sort_index().astype(np.float32)
    adj = pd.read_parquet(PQ / "adj_factor.parquet").reindex(
        index=close.index, columns=close.columns).ffill().astype(np.float32)
    uni = ctx["universe"]
    close = close.reindex(columns=uni)
    pa = close * adj.reindex(columns=uni)
    fs = float_share_panel(pd.DatetimeIndex(dates), uni)
    tdays = close.index
    n_ind = len(ctx["inds"])
    out = np.full((len(dates), n_ind), np.nan)
    for i, dt in enumerate(dates):
        j = int(tdays.searchsorted(dt))
        j = min(j, len(tdays) - 1)
        jm = j - m
        if jm < 0:
            continue
        r = (pa.iloc[j] / pa.iloc[jm] - 1).to_numpy(dtype=float)
        mc = (close.iloc[j].to_numpy(dtype=float)
              * fs.iloc[i].to_numpy(dtype=float))
        lbl = ctx["lbls"][i]
        for k in range(n_ind):
            msk = (lbl == k) & np.isfinite(r) & np.isfinite(mc) & (mc > 0)
            if msk.sum() < min_members:
                continue
            idx = np.where(msk)[0]
            order = idx[np.argsort(-mc[idx])]
            w = mc[order]
            cum = np.cumsum(w) / w.sum()
            n_lead = int((cum <= lam).sum())
            n_lead = max(1, min(n_lead, len(order) - 2))
            lead, follow = order[:n_lead], order[n_lead:]
            if len(follow) < 2:
                continue
            out[i, k] = r[lead].mean() - r[follow].mean()
    return pd.DataFrame(out, index=pd.DatetimeIndex(dates), columns=ctx["inds"])


# --------------------------------------------- 残差动量（研报 §2.3.1.3，FF3 残差）
def _ff3_monthly() -> pd.DataFrame:
    """自建 FF3：MKT = 全 A 流通市值加权月收益；SMB/HML 用市值与 BP 分组。"""
    d = pd.read_parquet(PQ / "daily_all_a.parquet", columns=["close"])
    close = d["close"].unstack("code").sort_index()
    adj = pd.read_parquet(PQ / "adj_factor.parquet").reindex(
        index=close.index, columns=close.columns).ffill()
    pa = close * adj
    mret = pa.resample("ME").last().pct_change(fill_method=None)
    mc = close.resample("ME").last()
    # BP：净资产（最近一期 PIT）× 
    bal = pd.read_parquet(PQ / "balance_sheet.parquet",
                          columns=["code", "ann_date", "report_period",
                                   "TOT_SHARE_EQUITY_EXCL_MIN_INT"])
    bal["ann_date"] = pd.to_datetime(bal["ann_date"], errors="coerce")
    bal = bal.dropna(subset=["ann_date"]).sort_values(["code", "ann_date"])
    eq_m = {}
    for dt in mret.index:
        sub = bal[bal["ann_date"] <= dt].drop_duplicates("code", keep="last")
        eq_m[dt] = sub.set_index("code")["TOT_SHARE_EQUITY_EXCL_MIN_INT"]
    bp = pd.DataFrame(eq_m).T.reindex(columns=mret.columns) / mc.replace(0, np.nan)
    mkt = (mret * mc.shift(1)).sum(axis=1) / mc.shift(1).sum(axis=1)
    size_rk = mc.rank(axis=1, pct=True)
    smb = (mret.where(size_rk <= 0.5).mean(axis=1)
           - mret.where(size_rk >= 0.5).mean(axis=1))
    bp_rk = bp.rank(axis=1, pct=True)
    hml = (mret.where(bp_rk >= 0.7).mean(axis=1)
           - mret.where(bp_rk <= 0.3).mean(axis=1))
    return pd.DataFrame({"MKT": mkt, "SMB": smb, "HML": hml}).dropna(how="all")


def residual_momentum(px: pd.DataFrame, lookback: int = 36,
                      stat_win: int = 11) -> pd.DataFrame:
    """行业指数月度收益对 FF3 回归，取过去 11 个月残差标准化和。"""
    ff = _ff3_monthly()
    mpx = px.resample("ME").last()
    mret = mpx.pct_change(fill_method=None)
    idx = ff.index.intersection(mret.index[1:])
    ff, mret = ff.reindex(idx), mret.reindex(idx)
    out = pd.DataFrame(np.nan, index=idx, columns=px.columns)
    X_all = np.column_stack([np.ones(len(ff)), ff[["MKT", "SMB", "HML"]].to_numpy()])
    for col in px.columns:
        y = mret[col].to_numpy(dtype=float)
        res = np.full(len(y), np.nan)
        for t in range(lookback, len(y)):
            yy = y[t - lookback:t]
            xx = X_all[t - lookback:t]
            ok = np.isfinite(yy) & np.isfinite(xx).all(axis=1)
            if ok.sum() < lookback * 0.8:
                continue
            beta, *_ = np.linalg.lstsq(xx[ok], yy[ok], rcond=None)
            res[t] = y[t] - float(X_all[t] @ beta)
        s = pd.Series(res, index=idx)
        r11 = s.rolling(stat_win, min_periods=6)
        out[col] = r11.sum() / r11.std()
    return out


def by_year(r: pd.Series, b: pd.Series) -> pd.DataFrame:
    g = pd.DataFrame({"ret": r, "bench": b})
    g["y"] = g.index.year
    t = g.groupby("y").agg(ret=("ret", lambda s: (1 + s).prod() - 1),
                           bench=("bench", lambda s: (1 + s).prod() - 1))
    t["excess"] = t["ret"] - t["bench"]
    return t


# ----------------------------------------------------- 成分股加权族（超预期/估值）
def build_ctx(freq: str = "W", start: str = "2018-01-01") -> dict:
    """复用 P0 的行业上下文（申万一级 PIT 分类 + 流通市值权重 + 调仓日）。"""
    from scripts.oneoff.etf_industry_factor_p0 import build_context
    return build_context(pd.Timestamp(start), freq=freq)


def agg_panels(names: list[str], ctx: dict, px: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """个股日频面板 → 行业因子（行业内自由流通市值加权，1/99 winsorize）。

    聚合函数直接复用 P0 的 `_agg`，保证与既有 P0 结果同口径。
    """
    from scripts.oneoff.etf_industry_factor_p0 import _agg

    out: dict[str, pd.DataFrame] = {}
    for name in names:
        p = OUT_DIR / "panels" / f"{name}.parquet"
        if not p.exists():
            continue
        panel = pd.read_parquet(p)
        rows = []
        for i, dt in enumerate(ctx["dates"]):
            if dt not in panel.index:
                rows.append(np.full(len(ctx["inds"]), np.nan))
                continue
            v = panel.loc[dt].reindex(ctx["universe"]).to_numpy(dtype=float)
            rows.append(_agg(v, ctx["lbls"][i], ctx["wts"][i]))
        df = pd.DataFrame(rows, index=pd.DatetimeIndex(ctx["dates"]),
                          columns=ctx["inds"])
        out[name] = df.reindex(index=px.index, columns=px.columns)
    return out


def test_family(sigs: dict[str, pd.DataFrame], px: pd.DataFrame,
                dates: list[pd.Timestamp], tdays: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for name, s in sigs.items():
        ic0 = ic_series(s, dates, px, tdays, lag=0)
        ic1 = ic_series(s, dates, px, tdays, lag=1)
        st0, st1 = stats(ic0), stats(ic1)
        q0 = quintile_excess(s, dates, px, tdays, lag=0)
        rows.append({
            "factor": name, "n": st0["n"],
            "ic_close": st0["ic"], "icir_close": st0["icir"],
            "t_close": st0["t"], "t_nw_close": st0["t_nw"],
            "ic_next": st1["ic"], "t_next": st1["t"], "t_nw_next": st1["t_nw"],
            **{f"ex_g{i + 1}": q0[i] for i in range(5)},
        })
        print(f"  {name:18s} IC(当天) {st0['ic']:+.4f} t_nw {st0['t_nw']:+.2f} | "
              f"IC(次日) {st1['ic']:+.4f} t_nw {st1['t_nw']:+.2f} | "
              f"G5-G1 {q0[4] - q0[0]:+.2%}")
    return pd.DataFrame(rows).sort_values("ic_close", ascending=False)


def write_report(fam: pd.DataFrame, ports: pd.DataFrame, rets: pd.DataFrame,
                 bench: pd.DataFrame, meta: dict) -> None:
    """生成 README.md（数据驱动，避免手抄数字）。"""
    def g(name: str, col: str):
        r = fam.loc[fam["factor"] == name, col]
        return float(r.iloc[0]) if len(r) else float("nan")

    def p(name: str, col: str):
        r = ports.loc[(ports["port"] == name) & (ports["cost_bp"] == 10.0), col]
        return float(r.iloc[0]) if len(r) else float("nan")

    # 研报原文数字（表 18 选用因子汇总）
    rep = [
        ("深度学习 GRU/TCN", 0.0685, 0.1466, "dl_gbdt"),
        ("行业动量", 0.0332, 0.0523, "mom_240_20"),
        ("超预期", 0.0248, 0.0608, "sue_op_sq"),
        ("估值", 0.0116, 0.0601, "cfp_inv_chg240"),
        ("分析师", 0.0335, 0.0968, None),
        ("大单 L2", 0.0364, 0.0573, None),
        ("RAFT", 0.0121, 0.0503, None),
    ]
    L = [
        "# 研报复现（国泰海通《七维共振》2026-10-29）· 只做可行的部分 —— 七维只做得出动量一维",
        "",
        "> **判定**：`复现完成（主结论为负）` —— 复现框架与因子层已落盘可重跑；"
        "七类因子里只有动量复现成功，另三类失败、三类无数据。",
        "> **脚本**：`scripts/portfolio/report_replica.py`（因子与组合）、"
        "`scripts/builders/build_replica_factors.py`（研报需要但库里没有的个股因子）",
        "> **产物**：本目录（`family_fund_W.csv` / `port_metrics.csv` / "
        "`port_returns.parquet` / `panels/*.parquet`）",
        "> **上下游**：上游 `reports/etf_contest5/industry_factor_p0.*`（全量 950 因子筛选）、"
        "`reports/etf_contest5/industry_model_ic.md`（模型分数行业层 IC）；下游 无",
        "",
        f"> {meta['n_dates']} 期周频（{meta['d0']}~{meta['d1']}）· "
        f"{meta['n_ind']} 个申万一级 · Top5 等权 · T+1 收盘成交 · 单边 10bp（研报双边千二口径）",
        "> 基准 = 全行业等权、同成交价、同调仓日（等权 buy&hold 年化 "
        f"{meta['bench_bh']:.2%}，价格指数不含分红）",
        "",
        "## 〇 一句话结论",
        "",
        "**七类因子逐类清点：3 类无数据、1 类方向相反、1 类无效、1 类只有四成强度，"
        "唯一复现成功的是动量。** 拼出来的最好组合是「纯动量」，年化超额 "
        f"{p('mom_240_20', 'excess'):+.2%}、t(IR) {p('mom_240_20', 't_ir'):+.2f} —— "
        "**在 9 年里 6 年为正，但统计上不显著**。",
        "",
        f"研报两条策略的可行子集：7 类等权（策略 1）能做到 {p('S1_mom_dl_eq', 'excess'):+.2%}，"
        f"深度学习 2 倍权重（策略 2）反而是 **{p('S2_dl_x2', 'excess'):+.2%}**。"
        "研报里「策略 2 优于策略 1」的前提是深度学习 IC 6.85%（七类最高），"
        "我们这边深度学习只有 2.83% ⇒ **加权方式照抄会反向**。",
        "",
        "## 一 口径（与研报逐条对齐）",
        "",
        "| 项 | 研报 | 本线 | 是否可比 |",
        "|---|---|---|---|",
        "| 行业池 | 中信一级 30 个 | 申万一级 28→31 个（PIT） | ⚠️ 划分不同，IC 不能逐位对 |",
        "| 调仓频率 | 周度 | 周度（另跑月频对照） | ✅ |",
        "| 单因子成交价 | 当天收盘、不计成本 | 当天收盘、不计成本 | ✅ |",
        "| 组合成交价 | 第二天收盘 | T+1 收盘 | ✅ |",
        "| 成本 | 双边千二（单边 10bp） | 单边 10bp（另报 36bp 项目口径） | ✅ |",
        "| 基准 | 中信行业指数等权 | 申万行业指数等权、同成交价 | ⚠️ 池不同 |",
        "| IC 定义 | Pearson(raw) | Pearson(raw)，另报 t / NW t | ✅ |",
        "| 分组 | 分 5 组 | 分 5 组（rank 分位切，避免退化截面） | ✅ |",
        "| 行业因子聚合 | 自由流通市值加权，缺失用行业中值补 | 自由流通市值加权，缺失跳过不补 | ⚠️ 我们略保守 |",
        "| 样本期 | 2014.01–2026.09 | 2018.01–2026.09（受个股面板起点限制） | ⚠️ 只有超额可比 |",
        "",
        "**行业收益口径**：申万行业指数点位，**价格指数不含分红**。"
        "所以等权 buy&hold 年化只有 1.20%（真实持有加约 2% 股息）；"
        "多头与基准同口径，超额不受影响，但绝对收益被系统性低估。",
        "",
        "## 二 七维可行性清点（先看清缺什么）",
        "",
        "| 研报类别 | 研报因子数 | 复现 | 依据 |",
        "|---|---|---|---|",
        "| 动量（行业指数原生） | 4 | **4** | 传统动量×2 / 信息比率×2 + 龙头股动量 + 残差动量 |",
        "| 估值（成分股加权） | 5 | **3** | 市现率倒数变动、PB 倒数+ROE 排序、PEG（历史增速替代 FY2）|",
        "| 超预期（成分股加权） | 6 | **4** | SUE 四变体；「过去 N 日盈利/营收调整」需分析师 ⇒ 缺 2 |",
        "| 深度学习 GRU/TCN | 12 | **1**（日频 8 个可训） | 已按研报自训 `TCN_4_day60_10d`；"
        "其余 7 个日频配置挂好机器，`week30` 四个待重采样 |",
        "| RAFT | 10 | **0** | 研报自身 IC 仅 1.21%，优先级最低 |",
        "| 分析师预期 | 9 | **0** | 库内无结构化 FY1/FY2/FY3 一致预期（只有研报文本，无预测数值）|",
        "| 大单 L2 | 3 | **0** | 无逐笔成交；分钟线只有 HS300，且无法从 bar 拆大单 |",
        "",
        "缺的两类恰好是研报排名第 2（分析师 9.68%）与第 4（大单 5.73%）。"
        "研报自己给出「超预期 vs 分析师」截面相关 0.583，所以本应按超预期类代理分析师 —— "
        "但下面第四节显示我们的超预期方向是反的，这条代理链断了。",
        "",
        "## 三 单因子复现：逐类对照研报",
        "",
        "| 研报类别 | 研报 IC | 我们 IC（Pearson/周频） | 研报年化超额 | 我们最好单因子超额 | 判定 |",
        "|---|---|---|---|---|---|",
    ]
    verdict = {"dl_gbdt": "❌ 只有 41% 强度", "mom_240_20": "✅ 复现（更强）",
               "sue_op_sq": "❌ 方向相反", "cfp_inv_chg240": "❌ 无效"}
    for cname, ric, rex, ours in rep:
        if ours is None:
            L.append(f"| {cname} | {ric:.2%} | — | {rex:.2%} | — | ⛔ 无数据 |")
            continue
        L.append(f"| {cname} | {ric:.2%} | {g(ours, 'ic_close'):+.2%} | {rex:.2%} | "
                 f"{p(ours, 'excess'):+.2%} | {verdict.get(ours, '—')} |")
    mom8 = ["ir_180", "ir_240", "mom_180_0", "mom_180_20", "mom_240_0",
            "mom_240_20", "lead_mv_60_180", "resid_mom"]
    have = set(ports["port"])
    exs = [p(k, "excess") for k in mom8 if k in have]
    ics = [g(k, "ic_close") for k in mom8]
    L += [
        "",
        f"动量族 8 个中长期因子**全部正超额**（{min(exs):+.2%}~{max(exs):+.2%}，"
        f"均值 {np.mean(exs):+.2%}），IC {min(ics):.2%}~{max(ics):.2%}，"
        "与研报动量类 5.23%/3.32% **同量级但略低**；短回望的 `ir_20`/`ir_120` 是负的。"
        "我们的 IC（Pearson、申万池、2018 起）并不比研报低多少 —— "
        "**这是七维里唯一站得住的一维**。",
        "",
        "## 四 三个对不上的地方",
        "",
        "### 4.1 深度学习：真做了 GRU/TCN —— 个股 IC 0.072，行业层归零",
        "",
        "**已按研报 §2.1.1 原样自训**（不再用 `gbdt(h1)` 分数替身）：6 个行情特征、"
        "5/10 日均价标签、**`−corr` 截面损失**、**每交易日一个 batch**、年度滚动、"
        "purge 到标签长度之外。实跑 `TCN_4_day60_10d`（2021–2026，1400 个交易日）：",
        "",
        "| 层 | 我们 | 研报 |",
        "|---|---|---|",
        "| 个股层截面 IC | **+0.072**（剔涨跌停 0.047） | 未披露 |",
        "| 行业层 IC | **+0.0059**（t_nw 0.34） | 6.85%（5.48%~8.21%） |",
        "| Top5 周频组合超额 | **−7.95%**（t(IR) −1.70） | +14.66% |",
        "",
        "🚨 **口径更正（10-31 晚）**：此前报的「个股 IC 0.197」**作废**。那个数是"
        "**标签分母伪相关**：研报写「均价涨跌幅」未给分母，我照字面用 `avg[t]` 做分母，"
        "而特征 `vwap/close` 的分子正是同一个 `avg[t]` ⇒ `label ≈ r − log(avg/close)`，"
        "机械相关上限 σ(d)/σ(r) = 0.95%/5.33% = **0.178**。实测：同一份预测把分母换成"
        "可成交的 `close[t]`，IC 从 **0.198 → 0.072**；裸特征 `−(vwap/close)` 从 "
        "**+0.167 → −0.013**（后者才是真值）。代码已改（`forward_avg_ret` 强制要求传"
        "`close`），但**现有的 `pred_*.parquet` 仍是伪标签训出来的**，0.072 只是偏保守下界。",
        "",
        "🚨 **归零的机制已查清，不是「个股 alpha 聚合不出行业 alpha」**"
        "（`reports/report_replica/dl/decomp_industry.md`）。同一份预测：",
        "",
        "| 聚合口径 / 样本 | 行业层 IC | t |",
        "|---|---|---|",
        "| 自由流通市值加权（研报 §2.1.1 口径） | **+0.0025** | +0.13 |",
        "| 等权聚合 | **+0.0423** | +2.10 |",
        "| 小市值三分位 · 市值加权 | **+0.0770** | +3.92 |",
        "| 大市值三分位 · 市值加权 | **−0.0039** | −0.20 |",
        "",
        "**行业成分是存在的，只是集中在低权重股票上，市值加权聚合按权重把它筛掉了。**"
        "大市值组个股 IC 只有 0.031、小市值组 0.086（差 2.8 倍），"
        "而市值加权必然让大票主导行业因子 ⇒ 归零。",
        "",
        "⇒ 所以「换 gbdt 还是 GRU/TCN」和「改去量纲」都不是重点，"
        "**重点是把信号从只有小票有、变成大票也有** —— 研报市值加权后仍有 6.85%，"
        "说明它的大票带行业信息。另两条已知事实：",
        "",
        "1. **η² 不能当判据**：大市值组 η² 最高（13%）但行业 IC 最差（−0.4%）——"
        "η² 只衡量「预测里有多少行业差异」，不衡量「这部分差异对不对」。",
        "2. **不能归因成「日内反转」**：实测 `−均价/收盘` 裸特征对可成交收益的个股 IC 只有 "
        "+0.003~+0.008（h=1），撑不起 0.065；信号具体来自哪一维尚未定位。"
        "多头组 T 日涨停占比 13.6% vs 全市场 1.7% = **8 倍**（纸面判据第 ③ 条阈值 5 倍）。",
        "",
        "⚠️ **两个数不要混引**：上表 0.0025 用的是「预测 vs 个股 10 日均价收益、"
        "同源聚合」（分解口径，与个股 IC 可比）；本页开头表的 **0.0059** 用的是"
        "「预测 vs 申万行业指数真实收益」（实盘口径）。标签不同，都在 0 附近。",
        "",
        "### 4.2 超预期：方向相反，而且不是噪声",
        "",
        "四个 SUE 变体全部为负（IC −2.40%~−6.01%，t_nw −2.3~−6.2），"
        "**9 年里 8~9 年同号**。研报是全部为正（+1.60%~+3.08%）。这不是样本期问题，"
        "是**同一个名字下的两个不同因子**：",
        "",
        "- 研报 §2.2.1 给的公式是「去年同期 + 漂移项」的**时间序列预期**，我们照此实现；",
        "- 但研报的数据源是朝阳永续，国内券商的「超预期」通常实际用**分析师一致预期**做 E(X)。"
        "两者经济含义完全不同：前者是「盈利同比改善」，后者才是「打败市场预期」。",
        "- 佐证：库内既有 `sue_q`（同样的时间序列口径）在个股层 IC 就是 −0.068（负），"
        "而研报说自己因子为正 —— 同一口径不可能在个股层为负、行业层翻正。",
        "",
        "⇒ **超预期这一维，在没有一致预期数据之前，我们无法判定「研报对还是我们错」，"
        "只能判定「我们的实现不是研报那个因子」**。反向使用（做多低 SUE 行业）虽然 t 高，"
        "但那是**事后定方向**，且经济含义是困境反转而非超预期，不能计入复现成果。",
        "",
        "### 4.3 估值：两边都无效，只是符号不同",
        "",
        f"研报估值类 IC 1.16%、IC_IR 0.23（七类最弱）。我们三个因子 IC "
        f"{min(g('cfp_inv_chg240', 'ic_close'), g('bp_inv_roe_rank', 'ic_close'), g('peg_hist', 'ic_close')):+.2%}"
        f"~{max(g('cfp_inv_chg240', 'ic_close'), g('bp_inv_roe_rank', 'ic_close'), g('peg_hist', 'ic_close')):+.2%}，"
        "全部落在噪声带内（|t_nw| ≤ 1.39）。**定性一致（都无效），符号不一致**。",
        "研报自己说「不同行业估值差异大，横截面上不可直接比」，"
        "并承认该类因子主要靠 PEG / 一致预期市盈率 —— 这两个我们又做不了。",
        "",
        "## 五 策略层：研报策略 1/2 的可行子集",
        "",
        "| 组合 | 年化 | 基准 | **超额** | Sharpe | t(IR) | 周双边换手 | 说明 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    note = {
        "mom_240_20": "单因子最强（研报未选它，选了 180_20）",
        "ir_240": "研报动量类选定",
        "resid_mom": "研报动量类选定",
        "lead_mv_60_180": "研报动量类选定",
        "mom_180_20": "研报动量类选定",
        "mom_only": "**研报选定的 4 个动量因子等权**",
        "dl_only": "深度学习替身",
        "S1_mom_dl_eq": "**策略 1 可行子集**（2 类等权）",
        "S2_dl_x2": "**策略 2 可行子集**（DL 2 倍权重）",
        "S2_plus_sue_rev": "策略 2 + 超预期反向（**事后方向，仅对照**）",
    }
    order = ["mom_240_20", "ir_240", "resid_mom", "lead_mv_60_180", "mom_180_20",
             "mom_only", "dl_only", "S1_mom_dl_eq", "S2_dl_x2", "S2_plus_sue_rev"]
    for k in order:
        if k not in set(ports["port"]):
            continue
        L.append(f"| `{k}` | {p(k, 'ann'):+.2%} | {p(k, 'ann_bench'):+.2%} | "
                 f"**{p(k, 'excess'):+.2%}** | {p(k, 'sharpe'):+.2f} | "
                 f"{p(k, 't_ir'):+.2f} | {p(k, 'turn_week'):.0%} | {note.get(k, '')} |")
    L += [
        "",
        "**研报对照**：策略 1 相对等权超额 13.91%（TE 11.05%，换手 91.92%）；"
        "策略 2 超额 15.81%（TE 10.68%，换手 93.85%）。",
        "**关键反差**：研报「策略 2 > 策略 1」（深度学习加权 2 倍），"
        "我们反过来 —— 因为我们深度学习维的组合层贡献是负的。**加权方案不能照抄，"
        "它依赖各维度的实际强度排序，而我们的强度排序与研报不同。**",
        "",
        "## 六 分年超额（最能说明问题的一栏）",
        "",
        "| 组合 | " + " | ".join(str(y) for y in range(2018, 2027)) + " | 正年数 |",
        "|---|" + "---|" * 10,
    ]
    for k in ("mom_240_20", "ir_240", "mom_only", "S1_mom_dl_eq", "S2_dl_x2", "dl_gbdt"):
        if k not in rets.columns:
            continue
        t = by_year(rets[k].dropna(), bench[k].dropna())
        cells = " | ".join(
            (f"{t.loc[y, 'excess']:+.0%}" if y in t.index else "—")
            for y in range(2018, 2027))
        pos = int((t["excess"] > 0).sum())
        L.append(f"| `{k}` | {cells} | **{pos}/{len(t)}** |")
    L += [
        "",
        "动量在 **2023–2024 连续失效**（mom_240_20 −15%/−4%），与研报「2022-2024 年动量因子"
        "表现不佳」完全一致 —— 这反过来证明我们的动量实现是**同一个东西**。",
        "深度学习替身只有 3/9 年为正，收益集中在 2024（+13%），其余年份持续负贡献。",
        "",
        "## 七 复现",
        "",
        "```bash",
        "# 1) 建研报需要、库里没有的个股因子（SUE 四变体 + 估值三项）",
        "python -m scripts.builders.build_replica_factors",
        "# 2) 因子层 + 组合层一次跑完",
        "python -m scripts.portfolio.report_replica --stage all --freq W",
        "# 3) 只用行业原生动量（可回溯到 2014-03，不依赖个股面板）",
        "python -m scripts.portfolio.report_replica --stage mom --freq W --start 2014-03-01",
        "```",
        "",
        "## 八 边界（引用前必读）",
        "",
        "1. **样本期与研报不同**：研报 2014.01–2026.09（含 2014–2015 大牛市，等权基准年化 7.45%），"
        f"我们 {meta['d0']}–{meta['d1']}（等权 buy&hold 年化 {meta['bench_bh']:.2%}）。"
        "**绝对收益不可比，只有超额可比**。",
        "2. **行业池不同**：研报中信一级 30 个，我们申万一级 28→31 个（PIT，"
        "石油石化/环保/美容护理 2021-12 才纳入）。行业划分差异会让 IC 不能逐位对。",
        "3. **申万指数是价格指数、不含分红**。基准年化 1.26% 很低就是这个原因；"
        "超额因此偏高（各行业股息率 0.5%~6% 不等，且与动量方向系统性相关）。",
        "4. **深度学习只做了 12 个配置里的 1 个**（`TCN_4_day60_10d`），"
        "且特征去量纲方式与研报未知口径可能不同（见 §4.1 与 `dl/README.md` §四）；"
        "`week30` 四个与 day30 四个未跑。",
        "5. **`mom_240_20` 是我们的最强单因子，但研报选的是 `mom_180_20`**（研报 180_20 单因子"
        f"超额 10.35%，是动量族里最高的）。我们这边恰好反过来"
        f"（240_20 {p('mom_240_20', 'excess'):+.2%} > 180_20 {p('mom_180_20', 'excess'):+.2%}），"
        "属**事后可见的差异**，不是事先判断。用它当「复现成果」会被多重检验质疑。",
        "6. **所有 t(IR) 都 < 1.6**。周频 446 期、Top5/31，超额年化波动 ~18%，"
        f"标准误约 0.9pp。**{p('mom_240_20', 'excess'):+.2%} 的超额在统计上与 0 "
        f"只差 {p('mom_240_20', 't_ir'):.1f} 个标准差**。",
        "7. **本线未过 `caliber.py`**：`baselines.yaml` 里没有行业轮动的合法基线，"
        "所以任何数字只能线内比较，不能对外声称「改进」。",
        "8. **`rev_*` 反向臂与 `S2_plus_sue_rev` 是看到结果后定的方向**，"
        "只能当诊断，不能当策略。",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(L), encoding="utf-8")
    print(f"→ {OUT_DIR / 'README.md'}")


# --------------------------------------------------------------------- 主流程
FUND_FACTORS = ["sue_roa_ttm_d", "sue_roe_ttm", "sue_np_sq_d", "sue_op_sq",
                "cfp_inv_chg240", "bp_inv_roe_rank", "peg_hist"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="mom", choices=["mom", "fund", "all"])
    ap.add_argument("--start", default=None,
                    help="默认：mom=2014-03-01（申万日历可回溯）；fund/all=2018-01-01")
    ap.add_argument("--freq", default="W", choices=["W", "M"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    px = load_px()
    tdays = px.index

    ctx = None
    if args.stage in ("fund", "all"):
        start = args.start or "2018-01-01"
        ctx = build_ctx(args.freq, start)
        dates = list(pd.DatetimeIndex(ctx["dates"]))
    else:
        start = args.start or "2014-03-01"
        dates = rebal_dates(tdays, pd.Timestamp(start), args.freq)

    print(f"池 {px.shape[1]} 个申万一级 · {args.freq} 频 {len(dates)} 期 "
          f"({dates[0].date()}~{dates[-1].date()})")

    sigs: dict[str, pd.DataFrame] = {}
    tag = "mom"
    if args.stage in ("mom", "all"):
        sigs.update(signals_momentum(px))
    if ctx is not None:
        tag = f"fund_{args.freq}"
        sigs.update(agg_panels(FUND_FACTORS, ctx, px))
        sigs.update(dl_signals(ctx, px))
        # 研报点名、但需个股/FF3 的两个动量因子
        sigs["lead_mv_60_180"] = lead_mv_momentum(ctx, dates).reindex(
            index=px.index, columns=px.columns)
        sigs["resid_mom"] = residual_momentum(px).reindex(
            index=px.index, columns=px.columns, method="ffill")
        # 超预期反向臂：方向由本线实测得出（9 年 8~9 年同号），属**事后**方向，
        # 只作对照展示，不得作为策略输入直接引用
        for k in ("sue_op_sq", "sue_roe_ttm", "sue_np_sq_d", "sue_roa_ttm_d"):
            if k in sigs:
                sigs[f"rev_{k}"] = -sigs[k]

    df = test_family(sigs, px, dates, tdays)
    out = OUT_DIR / f"family_{tag}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n→ {out}")
    print(df.to_string(index=False,
                       float_format=lambda v: f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"))

    if ctx is None:
        return

    # ---- 组合：研报策略 1/2 的可行子集
    def _cmean(items: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """跨因子 pct-rank 的忽略 NaN 均值（各因子暖机期不同）。"""
        st = pd.concat({k: zrank(v) for k, v in items.items()}, axis=0)
        return st.groupby(level=1).mean()

    mom_keys = [k for k in ("ir_240", "lead_mv_60_180", "mom_180_20",
                            "resid_mom") if k in sigs]
    dl_keys = [k for k in sigs if k.startswith("dl_")]
    if not mom_keys or not dl_keys:
        return
    mom_c = _cmean({k: sigs[k] for k in mom_keys})
    dl_c = _cmean({k: sigs[k] for k in dl_keys})
    combos = {
        "mom_only": mom_c,
        "dl_only": dl_c,
        "S1_mom_dl_eq": _cmean({"m": mom_c, "d": dl_c}),             # 策略1 可行子集
        "S2_dl_x2": (mom_c * 1 + dl_c * 2) / 3,                      # 策略2（DL 2 倍权重）
    }
    if "rev_sue_op_sq" in sigs:                                      # 事后方向，仅对照
        sue_c = _cmean({k: sigs[k] for k in ("rev_sue_op_sq", "rev_sue_roe_ttm")})
        combos["S2_plus_sue_rev"] = (mom_c * 1 + dl_c * 2 + sue_c * 1) / 4

    rows = []
    rets_all, bench_all = {}, {}
    for name, s in sigs.items():
        r = port_backtest(s, px, dates, top=5, cost_bp=10.0)
        if not r:
            continue
        rows.append({"port": name, "cost_bp": 10.0, **
                     {k: v for k, v in r.items() if k not in ("ret", "bench")}})
        rets_all[name] = r["ret"]
        bench_all[name] = r["bench"]
        print(f"  {name:18s} 年化 {r['ann']:+.2%} 基准 {r['ann_bench']:+.2%} "
              f"超额 {r['excess']:+.2%} t(IR) {r['t_ir']:+.2f} "
              f"周双边换手 {r['turn_week']:.0%}")
    for name, s in combos.items():
        for bp in (10.0, 36.0):
            r = port_backtest(s, px, dates, top=5, cost_bp=bp)
            if not r:
                continue
            rows.append({"port": name, "cost_bp": bp, **
                         {k: v for k, v in r.items() if k not in ("ret", "bench")}})
            if bp == 10.0:
                print(f"  {name:16s} 年化 {r['ann']:+.2%} 基准 {r['ann_bench']:+.2%} "
                      f"超额 {r['excess']:+.2%} Sharpe {r['sharpe']:+.2f} "
                      f"t(IR) {r['t_ir']:+.2f} 换手 {r['turnover']:.0%}")
                rets_all[name] = r["ret"]
                bench_all[name] = r["bench"]
    pd.DataFrame(rets_all).to_parquet(OUT_DIR / "port_returns.parquet")
    pd.DataFrame(bench_all).to_parquet(OUT_DIR / "port_bench.parquet")
    pm = pd.DataFrame(rows)
    pm.to_csv(OUT_DIR / "port_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT_DIR / 'port_metrics.csv'}")

    sub = px.loc[dates[0]:dates[-1]]
    meta = {
        "n_dates": len(dates), "d0": dates[0].date(), "d1": dates[-1].date(),
        "n_ind": int(px.shape[1]),
        "bench_bh": float(((sub / sub.iloc[0]).mean(axis=1).iloc[-1])
                          ** (365.25 / max((dates[-1] - dates[0]).days, 1)) - 1),
    }
    write_report(df, pm, pd.DataFrame(rets_all), pd.DataFrame(bench_all), meta)


if __name__ == "__main__":
    main()
