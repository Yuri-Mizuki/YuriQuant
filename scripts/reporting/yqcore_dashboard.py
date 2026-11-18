"""`yqcore` 定版策略前端页面（自包含 HTML，零后端）——**历史回测**与**模拟交易**分两页。

为什么分两页（2026-11-17）
--------------------------
原来一张 `dashboard.html` 把「七年回测」和「这一期拿着什么」堆在同一页，两套时间轴
（回测末日 2026-09-07 vs 持仓估值日 今天）、两套口径（引擎净值 vs 股数×收盘重算）
在同一条页面上滚动，读者分不清哪个数是回测的、哪个是实盘的。现在拆成：

- ``backtest.html``：历史回测详情（布局仿聚宽「回测详情」）——收益概述指标墙 +
  三面板收益曲线（策略/基准/超额 + 每日收益）+ 分年 + 月度热力 + 回撤 + 交易统计。
- ``live.html``：模拟交易（布局仿聚宽「模拟交易」）——当前持仓、持仓期表现、个股 K 线。

两页共享顶栏并互相跳转；各自只出现自己那套口径。

回测页的基准与派生指标
----------------------
基准取 ``reports/alla_rolling/_base/bench_eqw.parquet``（全A等权，与策略同池、含退市），
可切 ``bench_index``（沪深300）。α/β 为日频 CAPM + Newey-West HAC；超额、胜率、盈亏比、
超额回撤均由净值与基准现场算，**不落盘缓存**。基准序列晚于回测末日的部分自然被裁掉。

口径与诚实性约束
----------------
- 回测末日 = 信号面板末日 ``2026-09-07``（``END_ALIGN``），**不是今天**；实盘持仓另行
  估值到最新交易日。两页分开写，不混成一个数。
- 持仓期曲线按**股数 × 收盘**重算，末值须与 ``meta_<信号日>.json::portfolio_ret``
  一致；偏差 >1e-4 时页面显式报「自检未通过」，不静默吞掉。
- 持仓期基准由 ``daily_all_a.parquet`` **现场算**（同窗口、同开盘起步），不复用
  ``_base/bench_eqw.parquet``（月频调仓基准，与持仓期起点不同）。
- Sharpe 同时给几何（``report_native.json``）与算术（自算），引用必带口径。
- 股票名在本地库为空，表格按「有则显、无则省略该列」处理，**不编造**。

用法::

    python -m scripts.reporting.yqcore_dashboard                 # → backtest.html + live.html
    python -m scripts.reporting.yqcore_dashboard --open          # 生成后打开回测页
    python -m scripts.reporting.yqcore_dashboard --period 20261102
    python -m scripts.reporting.yqcore_dashboard --tier yqcore-focus-k15
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROD_DIR = ROOT / "reports" / "yq_core"
LIVE_DIR = PROD_DIR / "live"
BASE_DIR = ROOT / "reports" / "alla_rolling" / "_base"
ECHARTS_JS = ROOT / "reports" / "docs" / "design" / "_shared" / "js" / "echarts.min.js"

TIERS = ("yqcore-focus-k8", "yqcore-focus-k15", "yqcore-breadth")
TIER_COLORS = ("#1f4e79", "#d98324", "#5a7d9a")
#: K 线回溯的自然日跨度（前端再按区间切片，后端只读一次）
KLINE_LOOKBACK_DAYS = 420

#: 成本口径注记（与 ``caliber.cost_profile("production")`` 对应，改口径必须同步改这里）
COST_NOTE = ("往返 27bp 时变（佣金万1双边2 + 印花万5单边·2023-08-28 前千1 "
             "+ 滑点 10bp 双边20）")


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------
def _cache_root() -> Path:
    from config import Config
    return Path(str(Config.cache()["root"]))


def load_backtest(prod_dir: Path, tier: str) -> dict:
    """读 ``yq_core_report`` 的四件产物（指标 / 净值 / 回撤 / 分年）。"""
    rep = json.loads((prod_dir / "report_native.json").read_text(encoding="utf-8"))
    if tier not in rep["tiers"]:
        raise SystemExit(f"{prod_dir}/report_native.json 里没有档位 {tier}；"
                         f"可选 {list(rep['tiers'])}")
    nav = pd.read_csv(prod_dir / "nav_native.csv", index_col=0, parse_dates=True)
    dd = pd.read_csv(prod_dir / "drawdown_native.csv", index_col=0, parse_dates=True)
    yearly = pd.read_csv(prod_dir / "yearly_native.csv", index_col=0)
    tiers = [t for t in TIERS if t in nav.columns]

    def _col(df: pd.DataFrame, t: str) -> list:
        if t not in df.columns:
            return []
        return [None if pd.isna(v) else round(float(v), 6) for v in df[t].to_numpy()]

    years = [str(i) for i in yearly.index if str(i).isdigit()]
    return {
        "meta": rep,
        "kpi": rep["tiers"][tier],
        "nav_dates": [d.strftime("%Y-%m-%d") for d in nav.index],
        "nav": {t: _col(nav, t) for t in tiers},
        "dd_dates": [d.strftime("%Y-%m-%d") for d in dd.index],
        "dd": _col(dd, tier),
        "years": years,
        "yearly": {t: [round(float(yearly.loc[y, t]), 2) if t in yearly.columns else None
                       for y in years] for t in tiers},
    }


def list_periods(live_dir: Path) -> list[str]:
    out = []
    for p in live_dir.glob("holdings_*.csv"):
        stem = p.stem.rsplit("_", 1)[-1]
        if len(stem) == 8 and stem.isdigit():
            out.append(stem)
    return sorted(out)


def load_period(live_dir: Path, period: str | None = None) -> tuple[str, pd.DataFrame, dict]:
    """定位一期持仓（缺省 = 最新），返回（信号日, 持仓表, meta）。"""
    periods = list_periods(live_dir)
    if not periods:
        raise SystemExit(f"{live_dir} 下没有任何持仓期次，先跑 "
                         f"`python -m scripts.live.yqcore holdings --commit`")
    ds = period or periods[-1]
    h = pd.read_csv(live_dir / f"holdings_{ds}.csv", dtype={"code": str}, index_col="code")
    meta_p = live_dir / f"meta_{ds}.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return ds, h, meta


# ---------------------------------------------------------------------------
# 回测页：基准与派生指标
# ---------------------------------------------------------------------------
BENCH_LABEL = {
    "bench_eqw": "全A等权（同池基准）",
    "bench_index": "沪深300（大盘）",
}
BENCH_SHORT = {"bench_eqw": "全A等权", "bench_index": "沪深300"}


def load_bench(base_dir: Path) -> dict[str, pd.Series]:
    """读基准日收益（``_base/bench_*.parquet`` 的 ``ret`` 列）。

    这两份文件由 `alla_rolling` 落盘，含 inf 脏值（停牌股除权日），必须清洗；
    末日可能晚于回测末日（2026-11-04 / 09-01），对齐时自然裁掉。
    """
    out: dict[str, pd.Series] = {}
    for k in BENCH_LABEL:
        p = base_dir / f"{k}.parquet"
        if not p.exists():
            continue
        s = pd.read_parquet(p)["ret"]
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        out[k] = s
    return out


def _dd_curve(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def _mdd_info(nav: pd.Series) -> dict:
    """最大回撤及其区间（峰 → 谷）。峰值取谷底之前的最高点，与 dd 曲线定义一致。"""
    dd = _dd_curve(nav)
    if not len(dd):
        return {"mdd": None, "peak": None, "trough": None, "days": None}
    trough = dd.idxmin()
    upto = nav.loc[:trough]
    peak = upto.idxmax()
    return {
        "mdd": float(dd.min()),
        "peak": peak.strftime("%Y-%m-%d"),
        "trough": trough.strftime("%Y-%m-%d"),
        "days": int(len(upto.loc[peak:]) - 1),
    }


def _capm(r: pd.Series, m: pd.Series) -> dict:
    """日频 CAPM（NW HAC 自动滞后）：α 年化 = a×252，β 与 t 值同出。"""
    from stats.robust_stats import ols_newey_west
    d = pd.concat([r.rename("y"), m.rename("m")], axis=1).dropna()
    if len(d) < 60:
        return {"alpha": None, "t_alpha": None, "beta": None, "t_beta": None,
                "r2": None, "rho": None, "resid": None}
    X = np.column_stack([np.ones(len(d)), d["m"].to_numpy()])
    res = ols_newey_west(X, d["y"].to_numpy())
    resid = pd.Series(res["resid"], index=d.index)
    return {
        "alpha": float(res["beta"][0]), "t_alpha": float(res["t_nw"][0]),
        "beta": float(res["beta"][1]), "t_beta": float(res["t_nw"][1]),
        "r2": float(1 - resid.var() / d["y"].var()) if d["y"].var() else None,
        "rho": float(d["y"].corr(d["m"])),
        "resid": resid,
    }


def _perf(nav: pd.Series, bench_ret: pd.Series, ann: int = 252) -> dict:
    """一组净值 + 基准日收益 → 页面需要的全部派生指标（自算，不读缓存）。"""
    nav = nav.dropna()
    if len(nav) < 2:
        return {}
    r = nav.pct_change().dropna()
    br = bench_ret.reindex(nav.index).ffill().fillna(0.0)
    bn = (1.0 + br).cumprod()
    bn = bn / bn.iloc[0]                        # 基准净值与策略同起点
    n = len(nav)

    cum_p = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    cum_b = float(bn.iloc[-1] / bn.iloc[0] - 1.0)
    cum_x = float(nav.iloc[-1] / bn.iloc[-1] - 1.0)
    ann_p = float((1 + cum_p) ** (ann / n) - 1)
    ann_b = float((1 + cum_b) ** (ann / n) - 1)
    # 超额曲线：策略净值 / 基准净值（比值，不是两条收益曲线相减）
    ratio = (nav / nav.iloc[0]) / (bn / bn.iloc[0])

    # 超额口径（2026-11-18 统一，两类不能混）：
    #   · 累计 / 年化 / 月度 / 年度 → **比值** (1+p)/(1+b)−1，与 cum_x、年度表同源；
    #   · 日均超额 / TE / IR      → **算术差** r_p − r_b（标准定义）。
    # 旧版年化超额用 ann_p−ann_b（算术），与累计超额（比值）不同口径：
    # 本期 6.62% vs 5.74%，差 0.87pp；日均超额与 IR 用几何差则把均值低估约 13%。
    ann_excess = float((1 + ann_p) / (1 + ann_b) - 1)
    ex_ari = (r - br.reindex(r.index)).dropna()

    rr = r.reindex(br.index).fillna(0.0)
    win = float((rr > 0).mean())
    up, dn = float(rr[rr > 0].sum()), float(abs(rr[rr < 0].sum()))
    pl_ratio = (up / dn) if dn else None

    mp = (1 + rr).resample("ME").prod() - 1
    mb = (1 + br).resample("ME").prod() - 1
    monthly = pd.DataFrame({"p": mp, "b": mb})
    monthly["x"] = (1 + monthly["p"]) / (1 + monthly["b"]) - 1

    capm = _capm(r, br)
    te = float(ex_ari.std() * np.sqrt(ann)) if ex_ari.std() else None
    mdd = _mdd_info(nav)
    mdd_x = _mdd_info(ratio)

    return {
        "n_days": n,
        "start": nav.index[0].strftime("%Y-%m-%d"),
        "end": nav.index[-1].strftime("%Y-%m-%d"),
        "cum": cum_p, "cum_bench": cum_b, "cum_excess": cum_x,
        "ann": ann_p, "ann_bench": ann_b,
        "ann_excess": ann_excess,
        "vol": float(rr.std() * np.sqrt(ann)),
        "vol_bench": float(br.std() * np.sqrt(ann)),
        # 几何 Sharpe = 年化几何收益 / 年化波动（与 report_native.json 的 sharpe 同式）
        "sharpe_geo": float(ann_p / (rr.std() * np.sqrt(ann))) if rr.std() else None,
        "sharpe_arith": float(rr.mean() / rr.std() * np.sqrt(ann)) if rr.std() else None,
        "calmar": float(ann_p / abs(mdd["mdd"])) if mdd["mdd"] else None,
        "win_day": win,
        "win_month": float((monthly["p"] > 0).mean()),
        "pl_ratio": pl_ratio,
        "avg_excess_day": float(ex_ari.mean()),
        "te": te,
        "ir": float((ex_ari.mean() * ann) / te) if te else None,
        "mdd": mdd, "mdd_excess": mdd_x,
        "alpha": capm["alpha"] * ann if capm["alpha"] is not None else None,
        "t_alpha": capm["t_alpha"],
        "beta": capm["beta"], "t_beta": capm["t_beta"],
        "r2": capm["r2"], "rho": capm["rho"],
        "resid_sr": (float(capm["resid"].mean() / capm["resid"].std() * np.sqrt(ann))
                     if capm.get("resid") is not None and capm["resid"].std() else None),
        "monthly": monthly,
        "dd": _dd_curve(nav),
        "nav": nav, "bench_nav": bn,
        "daily_ret": rr, "bench_ret": br,
    }


def build_backtest_payload(prod_dir: Path = PROD_DIR, base_dir: Path = BASE_DIR,
                           tier: str = "yqcore-focus-k8",
                           primary: str = "bench_eqw") -> dict:
    """回测页数据：三档净值 + 每档对每个基准的派生指标 + 逐日序列（用于三面板图）。"""
    bt = load_backtest(prod_dir, tier)
    bench = load_bench(base_dir)
    if primary not in bench:
        primary = next(iter(bench), None)
    if primary is None:
        raise SystemExit(f"{base_dir} 下没有 bench_*.parquet，回测页算不出基准类指标")

    nav = bt["nav"]
    dates = bt["nav_dates"]
    idx = pd.to_datetime(dates)
    series_nav = {t: pd.Series([np.nan if v is None else v for v in nav[t]], index=idx)
                  for t in nav}
    perf = {t: {k: _perf(series_nav[t], bench[k]) for k in bench} for t in nav}

    cur = perf[tier][primary]
    monthly = cur["monthly"]
    rows = []
    for d, row in monthly.iterrows():
        rows.append({"ym": d.strftime("%Y-%m"), "y": d.year, "m": d.month,
                     "p": round(float(row["p"]), 6) if pd.notna(row["p"]) else None,
                     "b": round(float(row["b"]), 6) if pd.notna(row["b"]) else None,
                     "x": round(float(row["x"]), 6) if pd.notna(row["x"]) else None})

    def _ser(s: pd.Series, nd: int = 6) -> list:
        return [None if pd.isna(v) else round(float(v), nd) for v in s.to_numpy()]

    # 三面板图用的逐日序列：策略累计收益 / 基准累计收益 / 超额 / 每日收益
    cum_p = (cur["nav"] / cur["nav"].iloc[0] - 1.0)
    cum_b = (cur["bench_nav"] / cur["bench_nav"].iloc[0] - 1.0)
    cum_x = (cur["nav"] * cur["bench_nav"].iloc[0]) / (cur["bench_nav"] * cur["nav"].iloc[0]) - 1.0

    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "kind": "backtest",
        "tier": tier,
        "tier_label": bt["kpi"].get("label", tier),
        "tiers": list(nav),
        "signal": bt["kpi"].get("signal_used", ""),
        "fingerprint": bt["kpi"].get("signal_fingerprint", ""),
        "window": bt["kpi"].get("window", ""),
        "n_days": bt["kpi"].get("n_days"),
        "capital": bt["kpi"].get("capital"),
        "cost_note": COST_NOTE,
        "trade_note": ("T+1 开盘成交 · 整手 100 股进引擎 · "
                       "price_panel=close_raw.ffill().shift(1)"),
        "profile": bt["meta"].get("profile"),
        "kpi": bt["kpi"],
        "meta_tiers": {t: {"ann": v.get("ann"), "sharpe": v.get("sharpe"),
                           "sharpe_arith": v.get("sharpe_arith"),
                           "max_dd": v.get("max_dd"), "vol": v.get("vol"),
                           "turnover": v.get("turnover"), "nhold": v.get("nhold_mean"),
                           "invest": v.get("invest_mean")}
                       for t, v in bt["meta"]["tiers"].items()},
        "cur": {k: v for k, v in cur.items()
                if k not in ("monthly", "dd", "nav", "bench_nav", "daily_ret", "bench_ret")},
        "perf": {t: {k: {kk: vv for kk, vv in v.items()
                         if kk not in ("monthly", "dd", "nav", "bench_nav",
                                       "daily_ret", "bench_ret")}
                     for k, v in d.items()} for t, d in perf.items()},
        "bench": {"primary": primary, "labels": BENCH_LABEL, "short": BENCH_SHORT,
                  "available": list(bench)},
        "curve": {
            "dates": [d.strftime("%Y-%m-%d") for d in cur["nav"].index],
            "strategy": _ser(cum_p), "bench": _ser(cum_b), "excess": _ser(cum_x),
            # 三面板图统一存**小数比率**（0.0736 = +7.36%），显示时由前端 ×100；
            # 2026-11-18 前 daily 存百分数、strategy 存小数，同一 tooltip 里混用两种单位。
            "daily": _ser(cur["daily_ret"], 5),
            "daily_bench": _ser(cur["bench_ret"], 5),
        },
        "dd": {"dates": [d.strftime("%Y-%m-%d") for d in cur["dd"].index],
               "values": _ser(cur["dd"])},
        "monthly": rows,
        # 基准年收益：日收益按自然年复合（与策略 yearly 同法，可直接对比）
        "yearly": {"years": bt["years"], "series": bt["yearly"],
                   "bench": {str(y): round(float((1 + g).prod() - 1), 6)
                             for y, g in cur["bench_ret"].groupby(cur["bench_ret"].index.year)}},
        "nav": {"dates": dates, "series": nav},
        "sources": {
            "backtest": str((prod_dir / "report_native.json").relative_to(ROOT)),
            "nav": str((prod_dir / "nav_native.csv").relative_to(ROOT)),
            "bench": str((base_dir / f"{primary}.parquet").relative_to(ROOT)),
        },
    }


def _read_daily(cols: list[str], start: pd.Timestamp | None = None,
                codes: list[str] | None = None) -> pd.DataFrame:
    """读日线面板（只取需要的列 / 日期 / 代码，走 parquet 谓词下推）。

    返回 MultiIndex ``(date, code)``，code 强制为 str（parquet 里偶有混合类型）。
    """
    filters = []
    if start is not None:
        filters.append(("date", ">=", start))
    if codes is not None:
        filters.append(("code", "in", list(codes)))
    df = pd.read_parquet(_cache_root() / "daily_all_a.parquet", columns=cols,
                         filters=filters or None)
    idx = df.index
    df.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(idx.get_level_values(0)), idx.get_level_values(1).astype(str)],
        names=["date", "code"])
    return df.sort_index()


def price_window(codes: list[str], start: pd.Timestamp,
                 end: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    """取一段窗口的开盘/收盘宽表（行=日期，列=代码）。"""
    px = _read_daily(["open", "close"], start=start)
    out = {}
    for c in ("open", "close"):
        w = px[c].unstack("code").sort_index()
        if end is not None:
            w = w.loc[:end]
        out[c] = w
    return out


def period_frame(codes: list[str], holdings: pd.DataFrame,
                 open_: pd.DataFrame, close: pd.DataFrame) -> dict:
    """持仓期内逐日重估：组合（按股数）vs 全A等权（同起点、同开盘买入口径）。"""
    dates = close.index
    cols = [c for c in codes if c in close.columns]
    missing = [c for c in codes if c not in close.columns]

    sub = close[cols].ffill()
    shares = holdings.loc[cols, "shares"].astype(float)
    cost = float(holdings.loc[cols, "cost"].sum())
    mv = sub.mul(shares, axis=1).sum(axis=1, min_count=1)
    port = mv / cost - 1.0

    # 全A等权：首日 = 开盘买入到收盘；其后 = 日收益截面等权均值（跳过缺前收的票）
    r0 = (close.iloc[0] / open_.iloc[0] - 1.0).replace([np.inf, -np.inf], np.nan)
    daily = close.pct_change(fill_method=None).iloc[1:].mean(axis=1)
    bench_r = pd.concat([pd.Series([r0.mean()], index=dates[:1]), daily])
    bench = (1.0 + bench_r.fillna(0.0)).cumprod() - 1.0

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "port": [None if pd.isna(v) else round(float(v), 6) for v in port],
        "bench": [None if pd.isna(v) else round(float(v), 6) for v in bench],
        "missing": missing,
    }


def kline_payload(holdings: pd.DataFrame, signal_date: pd.Timestamp,
                  exec_date: pd.Timestamp, end: pd.Timestamp | None) -> dict:
    """每只持仓股的日 K（前复权锚定最新日 = 真实报价，历史按比例回溯）。"""
    codes = [c for c in holdings.index]
    start = signal_date - pd.Timedelta(days=KLINE_LOOKBACK_DAYS)
    raw = _read_daily(["open", "high", "low", "close", "volume"], start=start, codes=codes)
    adj = pd.read_parquet(BASE_DIR / "close_adj.parquet", columns=codes)
    rawc = pd.read_parquet(BASE_DIR / "close_raw.parquet", columns=codes)
    ratio = (adj / rawc).replace([np.inf, -np.inf], np.nan)

    out: dict[str, dict] = {}
    for c in codes:
        if c not in raw.index.get_level_values("code"):
            continue
        d = raw.xs(c, level="code").sort_index()
        if end is not None:
            d = d.loc[:end]
        if d.empty:
            continue
        k = ratio[c].reindex(d.index).ffill().bfill().fillna(1.0)
        k_last = float(k.iloc[-1]) or 1.0
        scale = k / k_last                       # 末日比例 = 1 ⇒ 末日价 = 真实报价

        px = pd.DataFrame({
            "open": d["open"] * scale, "high": d["high"] * scale,
            "low": d["low"] * scale, "close": d["close"] * scale,
        })
        ohlc = [[round(float(r[0]), 2), round(float(r[1]), 2),
                 round(float(r[2]), 2), round(float(r[3]), 2)]
                for r in px[["open", "close", "low", "high"]].to_numpy()]
        cser = px["close"]

        # 成本价换算到同一条前复权轴上：cost_real × k(成交日) / k(末日)
        k_exec = float(k.asof(exec_date)) if exec_date >= k.index[0] else k_last
        cost_raw = float(holdings.loc[c, "open_exec"])
        cost_px = round(cost_raw * (k_exec or k_last) / k_last, 3)

        out[c] = {
            "dates": [t.strftime("%Y-%m-%d") for t in px.index],
            "ohlc": ohlc,
            "vol": [int(v) if not pd.isna(v) else 0 for v in d["volume"].to_numpy()],
            "ma5": [None if pd.isna(v) else round(float(v), 2)
                    for v in cser.rolling(5).mean()],
            "ma20": [None if pd.isna(v) else round(float(v), 2)
                     for v in cser.rolling(20).mean()],
            "cost_px": cost_px,
            "cost_raw": round(cost_raw, 3),
        }
    return {"codes": [c for c in codes if c in out], "data": out,
            "adj_last": adj.index.max().strftime("%Y-%m-%d") if len(adj.index) else None,
            "px_last": raw.index.get_level_values("date").max().strftime("%Y-%m-%d")
            if len(raw.index) else None}


def build_live_payload(prod_dir: Path = PROD_DIR, live_dir: Path = LIVE_DIR,
                       tier: str = "yqcore-focus-k8", period: str | None = None) -> dict:
    """模拟交易页数据：当前一期持仓 + 持仓期表现 + K 线（**不含**历史回测序列）。"""
    bt = load_backtest(prod_dir, tier)
    ds, holdings, meta = load_period(live_dir, period)

    signal_date = pd.Timestamp(ds)
    exec_date = pd.Timestamp(meta.get("exec_date") or ds)
    # meta.mark_date 是「信号生成当日」的快照日，不是估值终点：持仓期一过就是过期值。
    # 估值终点取行情面板最新交易日（≤ 今天），快照日只用于与落盘 meta 做自检。
    snap_date = pd.Timestamp(meta["mark_date"]) if meta.get("mark_date") else None
    codes = list(holdings.index)

    today = pd.Timestamp(datetime.now().date())
    px = {k: v.loc[:today] for k, v in price_window(codes, exec_date).items()}
    per = period_frame(codes, holdings, px["open"], px["close"])

    close = px["close"]
    latest = close.ffill().iloc[-1].reindex(codes)
    latest_date = close.index[-1].strftime("%Y-%m-%d")
    capital = float(meta.get("capital") or 0.0)

    rows = []
    for code, r in holdings.iterrows():
        last_px = float(latest.get(code, np.nan))
        cost_px = float(r["open_exec"])
        shares = float(r["shares"])
        cost = float(r["cost"])
        ret = last_px / cost_px - 1.0 if cost_px and not np.isnan(last_px) else None
        nm = r.get("name")
        rows.append({
            "code": code,
            "name": nm if isinstance(nm, str) else "",
            "rank": int(r["rank"]) if not pd.isna(r.get("rank")) else None,
            "industry": r.get("industry") if isinstance(r.get("industry"), str) else "",
            "shares": int(shares),
            "cost_px": round(cost_px, 3),
            "last_px": None if np.isnan(last_px) else round(last_px, 3),
            "ret": None if ret is None else round(float(ret), 6),
            "mv": None if np.isnan(last_px) else round(shares * last_px, 0),
            "pnl": None if ret is None else round(cost * ret, 0),
            "cost": round(cost, 0),
            "weight": round(cost / capital, 4) if capital else None,
        })

    mv_sum = float(sum(x["mv"] for x in rows if x["mv"] is not None))
    cost_sum = float(sum(x["cost"] for x in rows))
    pnl_sum = float(sum(x["pnl"] for x in rows if x["pnl"] is not None))

    # 自检：用「快照日」收盘重算组合收益 vs 落盘 meta（容忍 1e-4）。
    # 估值日已推进到最新交易日，拿最新价跟旧快照比必然不等 —— 那是口径错，不是数据错。
    snap_ts = None
    if snap_date is not None:
        avail = close.index[close.index <= snap_date]
        if len(avail):
            snap_ts = avail[-1]
    snap_mv = mv_sum
    if snap_ts is not None:
        row = close.ffill().loc[snap_ts]
        vals = [float(holdings.loc[c, "shares"]) * float(row.get(c, np.nan))
                for c in codes]
        if vals and not any(np.isnan(vals)):
            snap_mv = float(sum(vals))
    computed = (snap_mv / cost_sum - 1.0) if cost_sum else None
    claimed = meta.get("portfolio_ret")
    check = {
        "date": None if snap_ts is None else snap_ts.strftime("%Y-%m-%d"),
        "computed": None if computed is None else round(computed, 6),
        "claimed": None if claimed is None else round(float(claimed), 6),
        "ok": (computed is not None and claimed is not None
               and abs(computed - float(claimed)) <= 1e-4),
    }

    contrib = sorted(
        [{"code": x["code"], "pnl": x["pnl"] or 0.0,
          "ret": x["ret"] if x["ret"] is not None else 0.0} for x in rows],
        key=lambda d: d["pnl"], reverse=True)

    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "kind": "live",
        "tier": tier,
        "tier_label": bt["kpi"].get("label", tier),
        "signal": bt["kpi"].get("signal_used", ""),
        "fingerprint": bt["kpi"].get("signal_fingerprint", ""),
        "window": bt["kpi"].get("window", ""),
        "n_days": bt["kpi"].get("n_days"),
        "cost_note": COST_NOTE,
        "trade_note": ("T+1 开盘成交 · 整手 100 股进引擎 · "
                       "price_panel=close_raw.ffill().shift(1)"),
        "periods": list_periods(live_dir),
        "hold": {
            "signal_date": ds,
            "exec_date": exec_date.strftime("%Y-%m-%d"),
            "snap_date": None if snap_ts is None else snap_ts.strftime("%Y-%m-%d"),
            "latest_date": latest_date,
            "capital": capital,
            "k": meta.get("k"),
            "max_per_sector": meta.get("max_per_sector"),
            "lot": meta.get("lot"),
            "rows": rows,
            "totals": {
                "mv": round(mv_sum, 0), "cost": round(cost_sum, 0),
                "pnl": round(pnl_sum, 0),
                "ret": round(pnl_sum / cost_sum, 6) if cost_sum else None,
                "cash": round(capital - mv_sum, 0) if capital else None,
                "invest": round(mv_sum / capital, 4) if capital else None,
            },
            "n_flipped": (meta.get("tradable") or {}).get("n_flipped"),
        },
        "period": per,
        "contrib": contrib,
        "kline": kline_payload(holdings, signal_date, exec_date, today),
        "check": check,
        "sources": {
            "backtest": str((prod_dir / "report_native.json").relative_to(ROOT)),
            "nav": str((prod_dir / "nav_native.csv").relative_to(ROOT)),
            "live": str((live_dir / f"holdings_{ds}.csv").relative_to(ROOT)),
        },
    }


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def _fmt_pct(v, nd: int = 2, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v * 100:+.{nd}f}%" if sign else f"{v * 100:.{nd}f}%"


def _cls(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return "pos" if v > 0 else ("neg" if v < 0 else "")


def _num(v, nd: int = 0, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+,.{nd}f}" if sign else f"{v:,.{nd}f}"


def _hold_table(rows: list[dict], total_mv: float | None = None,
                has_name: bool = True) -> str:
    """持仓明细行（聚宽「持仓详情」列序：标的 / 数量 / 现价 / 市值 / 浮动盈亏 / 占比）。"""
    out = []
    for r in rows:
        w = (r["mv"] / total_mv) if (total_mv and r.get("mv") is not None) else None
        name_cell = f"<td>{r['name']}</td>" if has_name else ""
        out.append(
            "<tr>"
            f"<td class='num'>{r['rank'] if r['rank'] is not None else '—'}</td>"
            f"<td class='mono'>{r['code']}</td>"
            f"{name_cell}"
            f"<td>{r['industry'] or '—'}</td>"
            f"<td class='num'>{r['shares']:,}</td>"
            f"<td class='num'>{r['cost_px']:.2f}</td>"
            f"<td class='num'>{_num(r['last_px'], 2)}</td>"
            f"<td class='num {_cls(r['ret'])}'>{_fmt_pct(r['ret'], 2, True)}</td>"
            f"<td class='num'>{_num(r['mv'], 0)}</td>"
            f"<td class='num {_cls(r['pnl'])}'>{_num(r['pnl'], 0, True)}</td>"
            f"<td class='num'>{_fmt_pct(w, 2)}</td>"
            "</tr>")
    return "".join(out)


def _topbar(active: str, generated: str) -> str:
    """四页共享顶栏；`active` ∈ {board, backtest, live, picks}。

    因子看板是另一个模块（`reports/factor_board/`），必须在这里也给入口，
    否则从看板点进回测/模拟后回不去（2026-11-18 修）。
    `picks` 页在 `picks_board.py` 里生成，同目录 `picks.html`。
    """
    def tab(key: str, href: str, label: str) -> str:
        cls = " class='on'" if key == active else ""
        return f'<a href="{href}"{cls}>{label}</a>'
    return (
        '<div class="topbar">'
        '<div class="brand"><i></i>YuriQuant</div>'
        '<div class="sep"></div>'
        '<div class="nav">'
        + tab("board", "../factor_board/index.html", "因子看板")
        + tab("backtest", "backtest.html", "历史回测")
        + tab("live", "live.html", "模拟交易")
        + tab("picks", "picks.html", "本期出榜")
        + '</div>'
        '<div class="right"><span>只读 · 零后端 · 可离线打开</span>'
        f'<span>{generated}</span></div></div>'
    )


def render_live_body(p: dict) -> str:
    t = p["hold"]["totals"]
    per = p["period"]
    per_port = next((v for v in reversed(per["port"]) if v is not None), 0.0) or 0.0
    per_bench = next((v for v in reversed(per["bench"]) if v is not None), 0.0) or 0.0
    per_ex = (1 + per_port) / (1 + per_bench) - 1 if (1 + per_bench) else per_port - per_bench
    per_port_s = _fmt_pct(per_port, 2, True)
    per_bench_s = _fmt_pct(per_bench, 2, True)
    per_ex_s = _fmt_pct(per_ex, 2, True)
    hold_ret_s = _fmt_pct(t["ret"], 2, True)
    k_txt = f"{p['hold']['k']} 只 · 每申万一级 ≤{p['hold']['max_per_sector']}"
    n_hold = len(p["hold"]["rows"])
    n_days = len(per["dates"])

    # 名称列在本地库为空（code_info.parquet 无名称字段）→ 整列为空时直接不渲染该列
    has_name = any(r.get("name") for r in p["hold"]["rows"])
    name_th = "<th>名称</th>" if has_name else ""

    chk = p["check"]
    chk_day = chk.get("date") or "快照日"
    chk_txt = ("✅ 自检通过" if chk["ok"] else "⚠️ <b>自检未通过</b>") + \
        f"：按股数×{chk_day} 收盘重算 {_fmt_pct(chk['computed'], 3, True)} vs " \
        f"落盘 meta {_fmt_pct(chk['claimed'], 3, True)}"
    latest_s = p["hold"]["latest_date"]
    snap_s = p["hold"].get("snap_date") or "—"
    stale_txt = "（与 CSV 快照同日）"
    if snap_s != latest_s:
        stale_txt = (f"（CSV 里的 <code>last_close</code> 是 <b>{snap_s}</b> "
                     "快照，已过期不用）")

    kl = p["kline"]
    adj_lag_txt = ""
    if kl.get("adj_last") and kl.get("px_last") and kl["adj_last"] < kl["px_last"]:
        adj_lag_txt = (f"<br>⚠️ 复权面板（<code>_base</code>）末日 <b>{kl['adj_last']}</b> "
                       f"早于行情末日 <b>{kl['px_last']}</b>：其后价格沿用最后可用复权因子，"
                       f"该段若发生除权会失真（需重建 <code>reports/alla_rolling/_base</code>）。")
    flip = p["hold"]["n_flipped"]
    flip_txt = (f"⚠️ 信号日实时状态核验：榜单内 <b>{flip}</b> 条 tradable 记录已过期，"
                f"已按实时状态重判（名单以此份为准，不等于持仓调整数）。") if flip else ""

    chk_ok = bool(chk.get("ok"))
    kpis = [
        ("期间收益", per_port_s, _cls(per_port), ""),
        ("超额（vs 全A等权）", per_ex_s, _cls(per_ex), ""),
        ("持仓市值", _num(t["mv"]), "", "元"),
        ("浮动盈亏", _num(t["pnl"], 0, True), _cls(t["pnl"]), "元"),
        ("可用资金", _num(t.get("cash")), "", "元"),
        ("仓位占比", _fmt_pct(t.get("invest"), 1), "", ""),
        ("实持股数", str(n_hold), "", f"只 / 目标 {p['hold']['k']}"),
    ]
    kpi_html = "".join(
        '<div class="kpi"><span>' + n + '</span><b class="' + c + '">' + v
        + (("<small>" + u + "</small>") if u else "") + "</b></div>"
        for n, v, c, u in kpis)

    return f"""
{_topbar('live', p['generated'])}

<div class="subbar">
  <span class="title">{p['tier']} · 模拟交易</span>
  <span class="badge">production 口径</span>
  <span class="badge ok">整手 100</span>
  <span class="badge {'ok' if chk_ok else 'warn'}">{'自检通过' if chk_ok else '自检未通过'}</span>
  <span class="badge gray">T+1 开盘</span>
  <span class="badge gray">信号日 {p['hold']['signal_date']} · 持仓 {n_days} 日</span>
  <div class="acts">
    <button class="btn" onclick="location.href='backtest.html'">历史回测 →</button>
    <button class="btn"
      onclick="document.getElementById('caliber').scrollIntoView()">口径说明</button>
    <button class="btn pri" onclick="window.print()">导出 PDF</button>
  </div>
</div>

<div class="kpibar">{kpi_html}</div>

<div class="wrap">
  <div class="card">
    <div class="card-hd">
      <h2>持仓详情（{p['hold']['signal_date']}）</h2>
      <span class="sub2">成交 {p['hold']['exec_date']} · 估值 {p['hold']['latest_date']}</span>
      <div class="right">
        <button class="btn" onclick="exportHoldings()">导出 CSV</button>
      </div>
    </div>
    <div class="minikpi">
      <span>目标<b>{k_txt}</b></span>
      <span>实持<b>{n_hold} 只</b></span>
      <span>资金<b>{_num(p['hold']['capital'])}</b></span>
      <span>持仓市值<b>{_num(t['mv'])}</b></span>
      <span>成本<b>{_num(t['cost'])}</b></span>
      <span>浮盈亏<b class="{_cls(t['pnl'])}">{_num(t['pnl'], 0, True)}</b></span>
      <span>期间收益<b class="{_cls(t['ret'])}">{hold_ret_s}</b></span>
      <span>仓位<b>{_fmt_pct(t['invest'], 1)}</b></span>
      <span>现金<b>{_num(t.get('cash'))}</b></span>
    </div>
    <div class="tbl-wrap">
      <table class="tbl" id="holdTable"><thead><tr>
      <th class="num">排名</th><th>代码</th>{name_th}<th>行业</th><th class="num">数量</th>
      <th class="num">成本价</th><th class="num">现价</th><th class="num">期间收益</th>
      <th class="num">市值</th><th class="num">浮动盈亏</th><th class="num">仓位占比</th>
      </tr></thead><tbody>{_hold_table(p['hold']['rows'], t['mv'], has_name)}</tbody></table>
    </div>
    <div class="note" style="padding:10px 14px 12px">{chk_txt} · 现价按
      <b>{latest_s}</b> 收盘重算{stale_txt}。<br>
      名称列在本地库为空（<code>code_info.parquet</code> 无名称字段、榜单 name 列同样空），
      整列为空时直接省略，不编造。{flip_txt}</div>
  </div>

  <div class="sec">持仓期内表现 · {p['hold']['exec_date']} → {p['hold']['latest_date']}</div>
  <div class="card">
    <div class="card-hd"><h2>持仓期内表现</h2>
      <div class="right">
        <span class="sub2">组合 <b class="{_cls(per_port)}">{per_port_s}</b> · 全A等权
        <b class="{_cls(per_bench)}">{per_bench_s}</b> · 超额
        <b class="{_cls(per_ex)}">{per_ex_s}</b></span>
      </div>
    </div>
    <div class="card-bd">
      <div id="periodChart" class="chart-s"></div>
      <div class="note">曲线按<b>持仓股数 × 当日收盘</b>重估，起点 = 成交日开盘（与基准同起点）；
        基准 = 全A等权（首日开盘买至收盘，其后日收益截面等权），由
        <code>daily_all_a.parquet</code> 现场计算，不复用 <code>_base/bench_eqw.parquet</code>
        （月频调仓基准，与持仓期起点不同）。共 {n_days} 个交易日。</div>
    </div>
    <div class="card-hd" style="border-top:1px solid #eef1f5"><h2>逐只盈亏贡献</h2></div>
    <div class="card-bd"><div id="contribChart" class="chart-s"></div></div>
  </div>

  <div class="sec">个股 K 线</div>
  <div class="card">
    <div class="card-hd"><h2>个股行情</h2>
      <div class="right"><div class="chips" id="winChips" style="margin:0"></div></div>
    </div>
    <div class="card-bd">
      <div class="chips" id="stockChips"></div>
      <div id="klChart" class="chart-xl"></div>
      <div class="note" id="klNote"></div>
      <div class="note"><b>复权</b>：前复权锚定最新可用复权日（该日价 = 真实报价，历史按
        <code>close_adj/close_raw</code> 比例回溯），成本线换算到同一条轴上；成交量单位＝股。
        {adj_lag_txt}</div>
    </div>
  </div>

  <div class="card" id="caliber">
    <div class="card-hd"><h2>口径说明</h2></div>
    <div class="card-bd">
      <div class="minikpi" style="border:0;background:transparent;padding:0">
        <span>信号<b>{p['signal']}</b></span>
        <span>指纹<b>{p['fingerprint']}</b></span>
        <span>回测窗口<b>{p['window']}</b></span>
        <span>成本<b>{p['cost_note']}</b></span>
        <span>成交<b>{p['trade_note']}</b></span>
      </div>
      <div class="note" style="margin-top:10px">
        数据源：<code>{p['sources']['live']}</code> · <code>daily_all_a.parquet</code>（OHLCV）·
        <code>_base/close_adj.parquet</code>（复权）<br>
        本页只回答「现在拿着什么、这一期到现在怎么样」；历史七年的收益/风险见
        <a href="backtest.html">backtest.html</a>，两页口径不同，<b>不可相减</b>。<br>
        生成命令：<code>python -m scripts.reporting.yqcore_dashboard</code> ·
        本页只读、零后端、可离线打开。
      </div>
    </div>
  </div>
</div>
"""


def _mcell(label: str, value: str, cls: str = "", sub: str = "") -> str:
    inner = f"<i>{sub}</i>" if sub else ""
    return (f'<div class="mcell"><span>{label}</span>'
            f'<b class="{cls}">{value}</b>{inner}</div>')


def render_backtest_body(p: dict) -> str:
    c = p["cur"]
    k = p["kpi"]
    bshort = p["bench"]["short"].get(p["bench"]["primary"], "基准")
    nav_last = next((v for v in reversed(p["nav"]["series"].get(p["tier"]) or [])
                     if v is not None), None)

    def pct(v, nd=2, sign=True):
        return _fmt_pct(v, nd, sign)

    mdd = c["mdd"]
    mdd_s = "—" if mdd.get("mdd") is None else "−" + _fmt_pct(abs(mdd["mdd"]), 2)
    mx = c["mdd_excess"]
    mx_s = "—" if mx.get("mdd") is None else "−" + _fmt_pct(abs(mx["mdd"]), 2)
    mdd_sub = ("—" if mdd.get("peak") is None
               else f"{mdd['peak']} → {mdd['trough']} · {mdd['days']} 日")
    mx_sub = ("—" if mx.get("peak") is None
              else f"{mx['peak']} → {mx['trough']} · {mx['days']} 日")
    alpha_s = "—" if c["alpha"] is None else f"{c['alpha'] * 100:.2f}%"
    beta_s = "—" if c["beta"] is None else f"{c['beta']:.3f}"
    sharpe_s = "—" if c["sharpe_geo"] is None else f"{c['sharpe_geo']:.3f}"
    ir_s = "—" if c["ir"] is None else f"{c['ir']:.2f}"

    cells = [
        _mcell("策略收益", pct(c["cum"]), _cls(c["cum"]),
               f"{c['start']} ~ {c['end']}"),
        _mcell("策略年化收益", pct(c["ann"]), _cls(c["ann"]), "几何年化 · 252 日"),
        _mcell("超额收益", pct(c["cum_excess"]), _cls(c["cum_excess"]),
               f"年化超额 {pct(c['ann_excess'])} · 比值口径"),
        _mcell("基准收益", pct(c["cum_bench"]), _cls(c["cum_bench"]), bshort),
        _mcell("阿尔法（年化）", alpha_s, _cls(c["alpha"]),
               f"t = {c['t_alpha']:.2f} · CAPM 日频" if c["t_alpha"] is not None else "—"),
        _mcell("贝塔", beta_s, "", f"t = {c['t_beta']:.1f} · R² = {c['r2']:.2f}"
               if c["t_beta"] is not None else "—"),
        _mcell("夏普比率", sharpe_s, "",
               f"算术 {k.get('sharpe_arith', 0):.3f} · t = {k.get('sharpe_t_stat', 0):.2f}"),
        _mcell("信息比率", ir_s, "",
               f"跟踪误差 {pct(c['te'], 2, False)} · 算术日超额" if c["te"] else "—"),
        _mcell("最大回撤", mdd_s, "neg", mdd_sub),
        _mcell("超额收益最大回撤", mx_s, "neg", mx_sub),
        _mcell("日均超额收益", f"{c['avg_excess_day'] * 1e4:.2f} bp", "",
               f"算术均值 · 年化≈{pct(c['avg_excess_day'] * 252)}"),
        _mcell("卡玛比率", f"{c['calmar']:.2f}" if c["calmar"] else "—", "", "年化 / 最大回撤"),
        _mcell("年化波动", pct(c["vol"], 2, False), "", "日收益 × √252"),
        _mcell("基准波动率", pct(c["vol_bench"], 2, False), "", bshort),
        _mcell("日胜率", f"{c['win_day'] * 100:.1f}%", "",
               f"上涨 {int(round(c['win_day'] * c['n_days']))} / {c['n_days']} 日"),
        _mcell("月胜率", f"{c['win_month'] * 100:.1f}%", "", "自然月收益 > 0"),
        _mcell("盈亏比", f"{c['pl_ratio']:.2f}" if c["pl_ratio"] else "—", "",
               "总盈利 / 总亏损（日）"),
        _mcell("年换手", pct(k.get("turnover"), 1, False), "", "单边 · 年化"),
        _mcell("平均持仓", f"{k.get('nhold_mean', 0):.1f} 只", "",
               f"最少 {k.get('nhold_min', '—')} 只"),
        _mcell("实投率均值", pct(k.get("invest_mean"), 1, False), "",
               f"最低 {pct(k.get('invest_min'), 1, False)}"),
    ]
    grid = "".join(cells)

    # 年收益表：三档 + 基准 + 超额
    ye = p["yearly"]
    ybench = ye.get("bench", {})
    year_cols = p["tiers"]
    ybody = ""
    for i, y in enumerate(ye["years"]):
        tds = "".join(
            f'<td class="num">{ye["series"][t][i]:.2f}</td>'
            if ye["series"].get(t) and ye["series"][t][i] is not None
            else '<td class="num">—</td>' for t in year_cols)
        bv = ybench.get(str(y))
        sv = ye["series"].get(p["tier"], [None] * len(ye["years"]))[i]
        xv = None
        if bv is not None and sv is not None:
            xv = (1 + sv / 100) / (1 + bv) - 1
        brow = f"{bv * 100:.2f}" if bv is not None else "—"
        ybody += f'<tr><td>{y}</td>{tds}<td class="num">{brow}</td>'
        ybody += f'<td class="num {_cls(xv)}">{_fmt_pct(xv, 2, True)}</td></tr>'

    # 交易统计表（三档对照）
    stat_rows = [
        ("策略收益", lambda d: _fmt_pct(d["cum"], 2, True)),
        ("年化收益", lambda d: _fmt_pct(d["ann"], 2, True)),
        ("年化波动", lambda d: _fmt_pct(d["vol"], 2, False)),
        ("夏普（几何）", lambda d: f"{d['sharpe_geo']:.3f}" if d["sharpe_geo"] else "—"),
        ("夏普（算术）", lambda d: f"{d['sharpe_arith']:.3f}" if d["sharpe_arith"] else "—"),
        ("最大回撤", lambda d: "−" + _fmt_pct(abs(d["mdd"]["mdd"]), 2) if d["mdd"]["mdd"] else "—"),
        ("卡玛比率", lambda d: f"{d['calmar']:.2f}" if d["calmar"] else "—"),
        ("信息比率", lambda d: f"{d['ir']:.2f}" if d["ir"] else "—"),
        ("阿尔法（年化）", lambda d: _fmt_pct(d["alpha"], 2, True) if d["alpha"] else "—"),
        ("贝塔", lambda d: f"{d['beta']:.3f}" if d["beta"] else "—"),
        ("日胜率", lambda d: f"{d['win_day'] * 100:.1f}%"),
        ("盈亏比", lambda d: f"{d['pl_ratio']:.2f}" if d["pl_ratio"] else "—"),
    ]
    stat_head = "".join(
        f'<th class="num">{t.replace("yqcore-", "")}</th>' for t in p["tiers"])
    stat_body = ""
    for label, fn in stat_rows:
        tds = "".join(
            f'<td class="num">{fn(p["perf"][t][p["bench"]["primary"]])}</td>'
            if t in p["perf"] else '<td class="num">—</td>' for t in p["tiers"])
        stat_body += f"<tr><td>{label}</td>{tds}</tr>"

    return f"""
{_topbar('backtest', p['generated'])}

<div class="subbar">
  <span class="title">{p['tier']} · 历史回测</span>
  <span class="badge ok">回测完成</span>
  <span class="badge">production 口径 · 27bp 时变</span>
  <span class="badge gray">T+1 开盘 · 整手 100</span>
  <span class="badge gray">{p['window']} · {p['n_days']} 日 · 资金 ¥{_num(p['capital'])}</span>
  <div class="acts">
    <button class="btn" onclick="location.href='live.html'">模拟交易 →</button>
    <button class="btn pri" onclick="window.print()">导出 PDF</button>
  </div>
</div>

<div class="bt">
  <aside class="bt-side">
    <a href="#ov" class="on">收益概述</a>
    <a href="#curve">收益曲线</a>
    <a href="#year">分年收益</a>
    <a href="#month">月度收益</a>
    <a href="#dd">回撤分析</a>
    <a href="#trade">三档对照</a>
    <a href="#caliber">口径说明</a>
  </aside>
  <div class="bt-main">

    <div class="card" id="ov">
      <div class="card-hd">
        <h2>收益概述</h2>
        <span class="sub2">基准 = {bshort}｜α/β 为日频 CAPM + Newey-West</span>
        <div class="right"><span class="sub2">累计净值 <b>{nav_last:.2f}×</b></span></div>
      </div>
      <div class="mgrid">{grid}</div>
      <div class="note" style="padding:0 14px 12px">回测末日 <b>{c['end']}</b>
        （信号面板末日，<b>不是今天</b>）。收益与回撤均为<b>净</b>值（扣 {p['cost_note']}）。
        Sharpe 同时给几何与算术：几何用于对外报数，算术用于 <code>t = SR·√T</code>。
      </div>
    </div>

    <div class="card" id="curve">
      <div class="card-hd">
        <h2>收益曲线</h2>
        <span class="sub2">上：累计收益（策略/基准/超额）· 中：每日收益 · 下：每日超额</span>
        <div class="right"><div class="seg" id="btRange"></div></div>
      </div>
      <div class="card-bd">
        <div class="legend">
          <span><i style="background:#1f4e79"></i>策略</span>
          <span><i style="background:#8a8f98"></i>基准 {bshort}</span>
          <span><i style="background:#d98324"></i>超额</span>
          <span style="color:#a0a7b6">轴：累计收益率 %</span>
        </div>
        <div id="btChart" class="chart-bt"></div>
        <div class="note">超额 = 策略净值 ÷ 基准净值 − 1 =
          <code>(P − B) / (1 + B)</code>，P、B 为策略/基准的累计收益率。
          <b>分母 (1+B) 不能省</b>：基准七年 +184.65% 时，直接相减
          <code>903.05 − 184.65 = 718pp</code> 要再除以 2.8465 才是 +252.37%；
          只有基准≈0（现金）时「相减」才与超额近似相等。
          下两栏柱条按日收益着色：<b class="pos">正红</b> / <b class="neg">负绿</b>。
          本页<b>累计 / 年化 / 月度 / 年度超额</b>一律用此比值口径；
          <b>日均超额、跟踪误差、信息比率</b>用算术差 <code>r<sub>p</sub>−r<sub>b</sub></code>
          （IR 的标准定义），两者数值不同是口径差异，不可互相校验。</div>
      </div>
    </div>

    <div class="card" id="year">
      <div class="card-hd"><h2>分年收益</h2><span class="sub2">按自然年 · 单位 %</span></div>
      <div class="card-bd"><div id="yearChart" class="chart-m"></div></div>
      <div class="tbl-wrap" style="max-height:none">
        <table class="tbl narrow"
          style="width:1180px;max-width:100%;table-layout:fixed"><thead><tr><th>年份</th>
        {''.join(f'<th class="num">{t}</th>' for t in year_cols)}
        <th class="num">{bshort}</th><th class="num">超额（{p['tier'].replace('yqcore-', '')}）</th>
        </tr></thead><tbody>{ybody}</tbody></table>
      </div>
      <div class="note" style="padding:8px 14px 12px">超额 = (1+策略)/(1+基准)−1，
        按年复合后计算；基准为日收益按年复合。</div>
    </div>

    <div class="card" id="month">
      <div class="card-hd"><h2>月度收益</h2>
        <span class="sub2">年 × 月 · 单位 % · 悬停看当月数值</span>
        <div class="right"><div class="seg" id="mMode"></div></div>
      </div>
      <div class="card-bd"><div id="monthChart" class="chart-month"></div></div>
    </div>

    <div class="card" id="dd">
      <div class="card-hd"><h2>回撤分析</h2>
        <span class="sub2">从历史高点回落 · 最大回撤区间已标注</span>
        <div class="right"><span class="sub2">最大回撤 <b class="neg">{mdd_s}</b></span></div>
      </div>
      <div class="card-bd">
        <div id="ddChart" class="chart-s"></div>
        <div class="note">最大回撤区间 {mdd_sub}（峰 = 谷底之前的净值最高点）。</div>
      </div>
    </div>

    <div class="card" id="trade">
      <div class="card-hd"><h2>三档对照</h2>
        <span class="sub2">同窗口、同信号、同成本，仅资金与 K 不同</span></div>
      <div class="tbl-wrap" style="max-height:none">
        <table class="tbl narrow"
          style="width:1180px;max-width:100%;table-layout:fixed">
        <thead><tr><th>指标</th>{stat_head}</tr></thead>
        <tbody>{stat_body}</tbody></table>
      </div>
      <div class="note" style="padding:8px 14px 12px">
        三档<b>资金不同</b>（10 万 / 50 万 / 200 万），曲线与指标<b>不可跨档相减</b>；
        对照只用于看执行层（整手取整、可买股数）的影响。</div>
    </div>

    <div class="card" id="caliber">
      <div class="card-hd"><h2>口径说明</h2></div>
      <div class="card-bd">
        <div class="minikpi" style="border:0;background:transparent;padding:0">
          <span>信号<b>{p['signal']}</b></span>
          <span>指纹<b>{p['fingerprint']}</b></span>
          <span>窗口<b>{p['window']}</b></span>
          <span>成本<b>{p['cost_note']}</b></span>
          <span>成交<b>{p['trade_note']}</b></span>
        </div>
        <div class="note" style="margin-top:10px">
          数据源：<code>{p['sources']['backtest']}</code> · <code>{p['sources']['nav']}</code> ·
          <code>{p['sources']['bench']}</code><br>
          α/β 用日频 CAPM（Newey-West HAC，自动滞后），α 按 ×252 线性年化；单因子，
          <b>未</b>控制市值/波动暴露，因此 α 偏乐观。超额、胜率、盈亏比、超额回撤均由
          净值与基准现场计算，不落盘缓存。<br>
          本页只回答「这套历史挣了多少」；当前持仓与本期表现见
          <a href="live.html">live.html</a>。两页时间轴不同（回测末日 {c['end']} vs 持仓估值日），
          <b>不可相减</b>。
        </div>
      </div>
    </div>

  </div>
</div>
"""


def _echarts_tag() -> str:
    if ECHARTS_JS.exists():
        return "<script>" + ECHARTS_JS.read_text(encoding="utf-8") + "</script>"
    return ('<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js">'
            "</script>")


_EXTRA_CSS = """
/* ===== 聚宽模拟交易页风格（2026-11-17 改版）=====
   布局：深蓝顶栏 → 策略副栏 → 7 格 KPI 条 → 卡片流（主图/回撤/分年/持仓/持仓期/K线）
   要点：全宽、方角小圆角、12~13px、表头浅灰、涨红跌绿 */

:root{--navy:#16304f;--blue:#2f6fdb;--bg:#f0f2f5;--line:#e4e7ec;
      --ink:#22262e;--muted:#7a8296;--pos:#d9363e;--neg:#12a05c}
html{background:var(--bg)}
body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;padding:0;
     background:var(--bg);color:var(--ink);font-size:13px;line-height:1.6}
h1,h2,h3{font-weight:600;margin:0}

/* 顶栏 */
.topbar{background:var(--navy);color:#fff;height:50px;display:flex;align-items:center;
        padding:0 20px;gap:12px;position:sticky;top:0;z-index:30}
.brand{display:flex;align-items:center;gap:8px;font-size:15px;font-weight:600}
.brand i{width:17px;height:17px;border-radius:3px;background:#3f7ad6;display:inline-block;
         box-shadow:inset 0 0 0 4px rgba(255,255,255,.85)}
.topbar .sep{width:1px;height:17px;background:rgba(255,255,255,.28)}
.topbar .nav{display:flex;gap:20px;font-size:13px;color:rgba(255,255,255,.8)}
.topbar .nav b{color:#fff;font-weight:500}
.topbar .right{margin-left:auto;display:flex;gap:16px;font-size:12px;
               color:rgba(255,255,255,.7)}

/* 策略副栏 */
.subbar{background:#fff;border-bottom:1px solid var(--line);min-height:54px;display:flex;
        align-items:center;padding:0 20px;gap:10px;flex-wrap:wrap}
.subbar .title{font-size:15px;font-weight:600;color:#1b2029}
.subbar .acts{margin-left:auto;display:flex;gap:8px}
.badge{font-size:11px;padding:2px 7px;border-radius:2px;border:1px solid #cfe0fb;
       background:#eaf2ff;color:var(--blue);white-space:nowrap}
.badge.ok{background:#eafaf1;border-color:#c6efdb;color:#0f8f58}
.badge.warn{background:#fff4e8;border-color:#ffdcb5;color:#b26a00}
.badge.gray{background:#f3f5f8;border-color:#e2e6ee;color:#6c7486}
.btn{font-family:inherit;font-size:12px;padding:5px 12px;border:1px solid var(--line);
     background:#fff;border-radius:3px;color:#3d4557;cursor:pointer}
.btn:hover{border-color:#c3cbda;color:var(--blue)}
.btn.pri{background:var(--blue);border-color:var(--blue);color:#fff}
.btn.pri:hover{filter:brightness(1.06);color:#fff}

/* KPI 条 */
.kpibar{background:#fff;border-bottom:1px solid var(--line);display:flex;
        box-shadow:0 1px 2px rgba(20,30,50,.04)}
.kpibar .kpi{flex:1;text-align:center;padding:12px 4px 13px;border-right:1px solid #f0f2f5}
.kpibar .kpi:last-child{border-right:0}
.kpibar .kpi span{display:block;font-size:12px;color:var(--muted);margin-bottom:3px}
.kpibar .kpi b{font-size:21px;font-weight:600;font-variant-numeric:tabular-nums;
               color:#1b2029;letter-spacing:-.2px}
.kpibar .kpi small{font-size:12px;color:var(--muted);font-weight:400;margin-left:1px}
.pos{color:var(--pos)}.neg{color:var(--neg)}

/* 主区与卡片 */
.wrap{padding:14px 18px 48px;max-width:1680px;margin:0 auto}
.card{background:#fff;border:1px solid var(--line);border-radius:4px;margin-bottom:12px}
.card-hd{display:flex;align-items:center;gap:10px;padding:0 14px;min-height:42px;
         border-bottom:1px solid #eef1f5;flex-wrap:wrap}
.card-hd h2{font-size:14px;color:#1b2029}
.card-hd .sub2{font-size:12px;color:var(--muted)}
.card-hd .right{margin-left:auto;display:flex;align-items:center;gap:10px}
.card-bd{padding:12px 14px 14px}
.sec{font-size:13px;font-weight:600;color:#3d4557;margin:20px 0 8px;padding-left:8px;
     border-left:3px solid var(--blue)}

/* 分段控件 / 图例 */
.seg{display:inline-flex;border:1px solid #dfe3e8;border-radius:3px;overflow:hidden}
.seg button{border:0;border-right:1px solid #e8ebf0;background:#fff;padding:4px 10px;
            font-size:12px;color:#5d6474;cursor:pointer;font-family:inherit}
.seg button:last-child{border-right:0}
.seg button.active{background:#eaf2ff;color:var(--blue);font-weight:600}
.legend{display:flex;gap:16px;font-size:12px;color:#5d6474;align-items:center;
        margin-bottom:6px;flex-wrap:wrap}
.legend i{display:inline-block;width:16px;height:2px;margin-right:5px;vertical-align:middle}

/* 表格 */
.tbl-wrap{overflow:auto;max-height:560px;border-top:1px solid #f2f4f7}
table.tbl{width:100%;border-collapse:collapse;font-size:12.5px}
table.tbl th{background:#f7f8fa;color:#5d6474;font-weight:400;text-align:left;
             padding:0 10px;height:34px;border-bottom:1px solid #e8ebf0;white-space:nowrap}
table.tbl td{padding:0 10px;height:32px;border-bottom:1px solid #f2f4f7;
             font-variant-numeric:tabular-nums;white-space:nowrap}
table.tbl tbody tr:hover td{background:#f5f9ff}
table.tbl td.num,table.tbl th.num{text-align:right}
table.tbl.compact{width:auto;min-width:460px}

/* 杂项 */
.note{color:#8a92a3;font-size:12px;line-height:1.75;margin-top:8px}
.minikpi{display:flex;gap:22px;flex-wrap:wrap;padding:9px 14px;border-bottom:1px solid #f2f4f7;
         font-size:12px;color:var(--muted);background:#fcfdfe}
.minikpi b{color:#1b2029;font-weight:600;font-variant-numeric:tabular-nums;margin-left:4px}
.row2{display:grid;grid-template-columns:1.45fr 1fr;gap:12px}
@media(max-width:1100px){.row2{grid-template-columns:1fr}}
.mono{font-family:Consolas,monospace}
code{font-family:Consolas,monospace;font-size:11.5px;color:#5b6478;background:#f5f6f8;
     padding:1px 4px;border-radius:2px}
.chart-h{height:400px}.chart-m{height:280px}.chart-s{height:230px}.chart-xl{height:470px}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 10px}
.chip{padding:4px 10px;border-radius:3px;background:#f2f4f7;cursor:pointer;font-size:12px;
      border:1px solid transparent;user-select:none;color:#4a5162}
.chip.active{background:#eaf2ff;color:var(--blue);border-color:#cfe0fb;font-weight:600}
.chip:hover{border-color:#d5dae4}
@media(max-width:960px){
  .kpibar{flex-wrap:wrap}.kpibar .kpi{flex:1 0 33%;border-bottom:1px solid #f0f2f5}
  .topbar .nav{display:none}.wrap{padding:10px 10px 32px}
}

/* ===== 两页共享：顶栏 tab ===== */
.topbar .nav a{color:rgba(255,255,255,.72);text-decoration:none;padding:3px 0}
.topbar .nav a:hover{color:#fff}
.topbar .nav a.on{color:#fff;font-weight:600;box-shadow:inset 0 -2px 0 #5b9bf5}

/* ===== 回测页：左导航 + 指标墙（聚宽「回测详情」布局） ===== */
.bt{display:flex;gap:14px;padding:14px 18px 48px;max-width:1680px;margin:0 auto;
    align-items:flex-start}
.bt-side{width:132px;flex:0 0 132px;position:sticky;top:64px;background:#fff;
         border:1px solid var(--line);border-radius:4px;padding:6px 0}
.bt-side a{display:block;padding:8px 14px;font-size:12.5px;color:#4a5162;
           text-decoration:none;border-left:2px solid transparent}
.bt-side a:hover{background:#f5f9ff;color:var(--blue)}
.bt-side a.on{color:var(--blue);border-left-color:var(--blue);background:#f2f7ff;
              font-weight:600}
.bt-main{flex:1;min-width:0}
.mgrid{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid #f0f2f5}
.mcell{padding:11px 14px 12px;border-right:1px solid #f2f4f7;border-bottom:1px solid #f2f4f7}
.mcell:nth-child(4n){border-right:0}
.mcell span{display:block;font-size:11.5px;color:#8a92a3;margin-bottom:2px}
.mcell b{font-size:19px;font-weight:600;font-variant-numeric:tabular-nums;
         letter-spacing:-.2px;color:#1b2029;line-height:1.35}
.mcell i{display:block;font-style:normal;font-size:11px;color:#a8aeba;margin-top:2px;
         white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chart-bt{height:470px}
.chart-month{height:400px}
table.tbl.narrow{width:1180px;max-width:100%;table-layout:fixed}
table.tbl.narrow th{white-space:normal;line-height:1.35}
table.tbl.narrow td:first-child,table.tbl.narrow th:first-child{width:132px;
  color:#3d4557}
.bt a{color:var(--blue)}
@media(max-width:1200px){.mgrid{grid-template-columns:repeat(3,1fr)}
  .mcell:nth-child(4n){border-right:1px solid #f2f4f7}
  .mcell:nth-child(3n){border-right:0}}
@media(max-width:980px){.bt-side{display:none}}
@media(max-width:760px){.mgrid{grid-template-columns:repeat(2,1fr)}
  .mcell:nth-child(3n){border-right:1px solid #f2f4f7}
  .mcell:nth-child(2n){border-right:0}}
@media print{
  .topbar,.subbar .acts,.bt-side{display:none}
  body{background:#fff}.card{break-inside:avoid}
}
"""


_LIVE_JS = r"""
const D = window.__YQ__, H = D.hold, T = H.totals;
const pos = '#c62828', neg = '#2e7d32', blue = '#1f4e79', gray = '#8a8f98';
const pctS = v => (v==null || isNaN(v)) ? '—' : (v>0?'+':'')+(v*100).toFixed(2)+'%';
const charts = [];
function mk(id, opt){ const el = document.getElementById(id); if (!el) return null;
  const c = echarts.init(el); if (opt) c.setOption(opt); charts.push(c); return c; }
window.addEventListener('resize', () => charts.forEach(c => c.resize()));
window.addEventListener('load', () => charts.forEach(c => c.resize()));

/* ---------- 1. 持仓期 ---------- */
mk('periodChart', {
  animation: false, grid: {left: 62, right: 26, top: 36, bottom: 40},
  legend: {top: 2, textStyle: {fontSize: 12}},
  tooltip: {trigger: 'axis', valueFormatter: v => v==null ? '—' : (v*100).toFixed(2)+'%'},
  xAxis: {type: 'category', data: D.period.dates, axisLabel: {fontSize: 11}},
  yAxis: {type: 'value', axisLabel: {formatter: v => (v*100).toFixed(1)+'%', fontSize: 11},
          splitLine: {lineStyle: {color: '#f0f1f3'}}},
  series: [
    {name: D.tier, type: 'line', data: D.period.port, symbol: 'circle', symbolSize: 5,
     lineStyle: {width: 2, color: blue}, itemStyle: {color: blue}},
    {name: '全A等权', type: 'line', data: D.period.bench, symbol: 'circle', symbolSize: 4,
     lineStyle: {width: 1.5, color: gray, type: 'dashed'}, itemStyle: {color: gray}},
  ],
});
const contribRev = D.contrib.slice().reverse();
mk('contribChart', {
  animation: false, grid: {left: 100, right: 64, top: 10, bottom: 30},
  tooltip: {trigger: 'axis', axisPointer: {type: 'shadow'},
    formatter: ps => { const c = contribRev[ps[0].dataIndex];
      return `${c.code}<br>盈亏 ${c.pnl >= 0 ? '+' : ''}${c.pnl.toFixed(0)} 元<br>` +
             `个股收益 ${(c.ret*100).toFixed(2)}%`; }},
  xAxis: {type: 'value', axisLabel: {fontSize: 11}, splitLine: {lineStyle: {color: '#f0f1f3'}}},
  yAxis: {type: 'category', data: contribRev.map(c => c.code), axisLabel: {fontSize: 11}},
  series: [{type: 'bar', barMaxWidth: 18,
    data: contribRev.map(c => ({
      value: c.pnl, itemStyle: {color: c.pnl >= 0 ? pos : neg},
      label: {position: c.pnl >= 0 ? 'right' : 'insideLeft',
              color: c.pnl >= 0 ? '#374151' : '#fff'}})),
    label: {show: true, fontSize: 11,
            formatter: o => (o.value >= 0 ? '+' : '') + o.value.toFixed(0)}}],
});

/* ---------- 2. K 线 ---------- */
const KL = D.kline;
if (!KL.codes.length) {
  document.getElementById('klChart').innerHTML =
    '<div class="note">没有可用的行情数据（daily_all_a.parquet 缺这些代码）</div>';
} else {
  let curCode = KL.codes[0], curWin = 2;   // 缺省「近3月」：持仓期只有 3~6 根 K，读不出形态
  const WINS = [['持仓期', null], ['近1月', 30], ['近3月', 90], ['近半年', 180],
                ['近1年', 365], ['全部', 0]];
  const klChart = mk('klChart');

  function sliceK(code, winDays){
    const d = KL.data[code]; const n = d.dates.length;
    if (winDays === null){
      const i = d.dates.findIndex(x => x >= H.exec_date);
      return (i >= 0) ? [i, n] : [Math.max(0, n - 20), n];
    }
    if (winDays === 0) return [0, n];
    const last = new Date(d.dates[n-1] + 'T00:00:00');
    const start = new Date(last.getTime() - winDays*86400000)
      .toISOString().slice(0, 10);
    const i = d.dates.findIndex(x => x >= start);
    return [i < 0 ? 0 : i, n];
  }

  function drawK(code, winDays){
    const d = KL.data[code];
    if (!d) return;
    const [from, to] = sliceK(code, winDays);
    const dates = d.dates.slice(from, to);
    const ohlc = d.ohlc.slice(from, to);
    const vol = d.vol.slice(from, to).map((v, i) => ({value: v,
      itemStyle: {color: ohlc[i][1] >= ohlc[i][0] ? pos : neg}}));
    const ma5 = d.ma5.slice(from, to), ma20 = d.ma20.slice(from, to);
    const cost = d.cost_px;

    const marks = [];
    if (cost != null){
      marks.push({yAxis: cost, lineStyle: {color: gray, type: 'dashed', width: 1},
        label: {formatter: '成本 ' + cost.toFixed(2), fontSize: 11, color: '#555',
                position: 'insideEndTop'}});
    }
    klChart.setOption({
      animation: false,
      axisPointer: {link: [{xAxisIndex: 'all'}], label: {backgroundColor: '#555'}},
      grid: [{left: 76, right: 30, top: 18, height: 320},
             {left: 76, right: 30, top: 376, height: 78}],
      tooltip: {trigger: 'axis', axisPointer: {type: 'cross'},
        formatter: ps => {
          const ax = ps.find(p => p.seriesType === 'candlestick');
          if (!ax) return '';
          const i = ax.dataIndex, o = ohlc[i];
          const chg = (o[1] / o[0] - 1) * 100;
          return `${dates[i]}<br>开 ${o[0]}　收 ${o[1]}<br>低 ${o[2]}　高 ${o[3]}<br>` +
                 `涨跌 <b style="color:${chg >= 0 ? pos : neg}">` +
                 `${chg >= 0 ? '+' : ''}${chg.toFixed(2)}%</b><br>` +
                 `量 ${(vol[i].value / 1e4).toFixed(1)} 万`;
        }},
      xAxis: [{type: 'category', data: dates, gridIndex: 0, boundaryGap: true,
               axisLabel: {fontSize: 11}},
              {type: 'category', data: dates, gridIndex: 1, axisLabel: {show: false}}],
      yAxis: [{scale: true, gridIndex: 0, axisLabel: {fontSize: 11},
               splitLine: {lineStyle: {color: '#f0f1f3'}}},
              {gridIndex: 1, axisLabel: {show: false}, splitLine: {show: false},
               axisLine: {show: false}, axisTick: {show: false}}],
      dataZoom: [{type: 'inside', xAxisIndex: [0, 1]},
                 {type: 'slider', xAxisIndex: [0, 1], height: 14, bottom: 6}],
      series: [
        {name: 'K', type: 'candlestick', data: ohlc,
         itemStyle: {color: pos, color0: neg, borderColor: pos, borderColor0: neg},
         markLine: {silent: true, symbol: 'none', data: marks},
         markPoint: {symbol: 'pin', symbolSize: 30, itemStyle: {color: blue},
           label: {fontSize: 10, color: '#fff'},
           data: (cost != null) ? [{name: '成交', coord: [H.exec_date, cost], value: '买'}]
                                : []}},
        {name: 'MA5', type: 'line', data: ma5, symbol: 'none',
         lineStyle: {width: 1, color: '#d98324'}},
        {name: 'MA20', type: 'line', data: ma20, symbol: 'none',
         lineStyle: {width: 1, color: '#5a7d9a'}},
        {name: '量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: vol, barMaxWidth: 6},
      ],
    }, true);

    const r = H.rows.find(x => x.code === code) || {};
    document.getElementById('klNote').innerHTML =
      `<b>${code}</b> · ${r.industry || '—'} · 榜内排名 ${r.rank == null ? '—' : r.rank} · ` +
      `股数 ${(r.shares || 0).toLocaleString()} · 成本 ${r.cost_px} → 现价 ` +
      `${r.last_px == null ? '—' : r.last_px} · 期间收益 ` +
      `<b style="color:${(r.ret || 0) >= 0 ? pos : neg}">${pctS(r.ret)}</b> · ` +
      `市值 ${(r.mv || 0).toLocaleString()} · 盈亏 ${(r.pnl || 0).toFixed(0)} 元 · ` +
      `窗口 ${dates.length} 个交易日`;
  }

  const sc = document.getElementById('stockChips');
  KL.codes.forEach((c, i) => {
    const b = document.createElement('div');
    b.className = 'chip' + (i === 0 ? ' active' : ''); b.textContent = c;
    b.onclick = () => { curCode = c;
      sc.querySelectorAll('.chip').forEach(x => x.classList.remove('active'));
      b.classList.add('active'); drawK(curCode, WINS[curWin][1]); };
    sc.appendChild(b);
  });
  const wc = document.getElementById('winChips');
  WINS.forEach(([lbl, v], i) => {
    const b = document.createElement('div');
    b.className = 'chip' + (i === curWin ? ' active' : ''); b.textContent = lbl;
    b.onclick = () => { curWin = i;
      wc.querySelectorAll('.chip').forEach(x => x.classList.remove('active'));
      b.classList.add('active'); drawK(curCode, v); };
    wc.appendChild(b);
  });
  drawK(curCode, WINS[curWin][1]);
}

/* ---------- 3. 导出持仓 CSV（聚宽「导出全部」同款）---------- */
function exportHoldings(){
  const tb = document.querySelector('#holdTable tbody');
  if (!tb) return;
  const rows = Array.from(document.querySelectorAll('#holdTable tr')).map(tr =>
    Array.from(tr.children).map(td =>
      '"' + td.textContent.trim().replace(/"/g, '""') + '"').join(','));
  const blob = new Blob(['\ufeff' + rows.join('\n')], {type: 'text/csv;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'holdings_' + (H.signal_date || '') + '.csv';
  document.body.appendChild(a); a.click(); a.remove();
}
"""


_BT_JS = r"""
const D = window.__YQ__, C = D.curve, FY = D.yearly;
const pos = '#c62828', neg = '#2e7d32', blue = '#1f4e79', orange = '#d98324', gray = '#8a8f98';
const AR = ['#1f4e79', '#d98324', '#5a7d9a'];
const charts = [];
function mk(id, opt){ const el = document.getElementById(id); if (!el) return null;
  const c = echarts.init(el); if (opt) c.setOption(opt); charts.push(c); return c; }
window.addEventListener('resize', () => charts.forEach(c => c.resize()));
window.addEventListener('load', () => charts.forEach(c => c.resize()));

/* ---------- 1. 三面板收益曲线（策略 / 基准 / 超额 + 每日柱） ---------- */
const bt = mk('btChart');
const N = C.dates.length;
const XAX = (i, show) => ({type: 'category', data: C.dates, gridIndex: i,
  boundaryGap: i > 0, axisLabel: {show: show, fontSize: 11, color: '#7a8296',
    hideOverlap: true}, axisLine: {lineStyle: {color: '#e4e7ec'}},
  axisTick: {show: false}});
const YAX = (i, name) => ({type: 'value', gridIndex: i, name: name || '',
  nameTextStyle: {fontSize: 10, color: '#a0a7b6', align: 'right'},
  axisLabel: {fontSize: 10.5, color: '#7a8296',
    formatter: v => (v * 100).toFixed(0) + '%'},
  splitLine: {lineStyle: {color: '#f4f5f8'}}, axisLine: {show: false},
  axisTick: {show: false}, scale: i === 0, splitNumber: i === 0 ? 6 : 2});

bt.setOption({
  animation: false,
  axisPointer: {link: [{xAxisIndex: 'all'}], label: {backgroundColor: '#555'}},
  grid: [{left: 56, right: 34, top: 22, height: 254},
         {left: 56, right: 34, top: 300, height: 62},
         {left: 56, right: 34, top: 392, height: 62}],
  tooltip: {trigger: 'axis', axisPointer: {type: 'line'},
    formatter: ps => {
      const i = ps[0].dataIndex;
      const f = v => v == null ? '—' : (v > 0 ? '+' : '') + (v * 100).toFixed(2) + '%';
      return C.dates[i] + '<br>' +
        '策略 <b>' + f(C.strategy[i]) + '</b>　基准 ' + f(C.bench[i]) + '<br>' +
        '超额 <b style="color:' + orange + '">' + f(C.excess[i]) + '</b><br>' +
        '当日 ' + f(C.daily[i]) + '　基准 ' + f(C.daily_bench[i]);
    }},
  xAxis: [XAX(0, false), XAX(1, false), XAX(2, true)],
  yAxis: [YAX(0), YAX(1), YAX(2)],
  dataZoom: [{type: 'inside', xAxisIndex: [0, 1, 2]},
             {type: 'slider', xAxisIndex: [0, 1, 2], height: 16, bottom: 8,
              borderColor: '#e4e7ec', fillerColor: 'rgba(47,111,219,.10)',
              handleStyle: {color: '#2f6fdb'}, textStyle: {fontSize: 10, color: '#a0a7b6'}}],
  series: [
    {name: '策略', type: 'line', data: C.strategy, showSymbol: false, sampling: 'lttb',
     lineStyle: {width: 1.8, color: blue}, itemStyle: {color: blue},
     areaStyle: {color: 'rgba(31,78,121,.10)'},
     markLine: {silent: true, symbol: 'none', label: {show: false},
                lineStyle: {color: '#c3c9d4', type: 'solid', width: 1},
                data: [{yAxis: 0}]}},
    {name: '基准', type: 'line', data: C.bench, showSymbol: false, sampling: 'lttb',
     lineStyle: {width: 1.2, color: gray}, itemStyle: {color: gray}},
    {name: '超额', type: 'line', data: C.excess, showSymbol: false, sampling: 'lttb',
     lineStyle: {width: 1.4, color: orange}, itemStyle: {color: orange}},
    {name: '每日收益', type: 'bar', xAxisIndex: 1, yAxisIndex: 1,
     data: C.daily.map(v => ({value: v,
       itemStyle: {color: v >= 0 ? pos : neg}})), barMaxWidth: 4},
    {name: '每日超额', type: 'bar', xAxisIndex: 2, yAxisIndex: 2,
     data: C.daily.map((v, i) => ({value: (v - C.daily_bench[i]),
       itemStyle: {color: (v - C.daily_bench[i]) >= 0 ? pos : neg}})), barMaxWidth: 4},
  ],
});

const RANGES = [['1月', 1], ['3月', 3], ['6月', 6], ['1年', 12], ['全部', 0]];
function setRange(m){
  let i = 0;
  if (m > 0){
    const last = new Date(C.dates[N - 1] + 'T00:00:00');
    const s = new Date(last.getTime()); s.setMonth(s.getMonth() - m);
    const key = s.toISOString().slice(0, 10);
    const k = C.dates.findIndex(x => x >= key);
    i = k < 0 ? 0 : k;
  }
  bt.dispatchAction({type: 'dataZoom', startValue: i, endValue: N - 1});
}
const rb = document.getElementById('btRange');
RANGES.forEach(([lbl, m]) => {
  const b = document.createElement('button');
  b.textContent = lbl; if (m === 0) b.className = 'active';
  b.onclick = () => { rb.querySelectorAll('button').forEach(x => x.classList.remove('active'));
    b.classList.add('active'); setRange(m); };
  rb.appendChild(b);
});

/* ---------- 2. 分年柱 ---------- */
const YEARS = FY.years, yb = FY.bench || {};
mk('yearChart', {
  animation: false, grid: {left: 56, right: 26, top: 36, bottom: 30},
  legend: {top: 0, left: 4, itemWidth: 12, itemHeight: 9,
           textStyle: {fontSize: 12, color: '#5d6474'}},
  tooltip: {trigger: 'axis', valueFormatter: v => v == null ? '—' : v.toFixed(2) + '%'},
  xAxis: {type: 'category', data: YEARS, axisLabel: {fontSize: 11}},
  yAxis: {type: 'value', name: '%', nameTextStyle: {fontSize: 11},
          axisLabel: {fontSize: 11}, splitLine: {lineStyle: {color: '#f2f4f7'}}},
  series: [
    {name: D.tier.replace('yqcore-', ''), type: 'bar', barMaxWidth: 26,
     data: FY.series[D.tier], itemStyle: {color: AR[0]}},
    {name: D.bench.short[D.bench.primary], type: 'bar', barMaxWidth: 26,
     data: YEARS.map(y => yb[y] == null ? null : yb[y] * 100),
     itemStyle: {color: '#b9c0cc'}},
  ],
});

/* ---------- 3. 月度热力图（策略 / 超额） ---------- */
const MO = D.monthly;
const yset = [...new Set(MO.map(r => r.y))];
const MNAME = Array.from({length: 12}, (_, i) => (i + 1) + '月');
function heatData(key, abs){
  return MO.filter(r => r[key] != null).map(r => {
    const v = r[key] * 100;
    return {value: [r.m - 1, yset.indexOf(r.y), v],
            label: {color: Math.abs(v) > abs * 0.6 ? '#fff' : '#40485a'}};
  });
}
const ymOf = v => yset[v[1]] + '-' + String(v[0] + 1).padStart(2, '0');
const monthChart = mk('monthChart');
function drawMonth(key){
  const raw = heatData(key, 0);
  const abs = raw.reduce((a, d) => Math.max(a, Math.abs(d.value[2])), 1);
  const data = heatData(key, abs);
  monthChart.setOption({
    animation: false,
    grid: {left: 62, right: 26, top: 14, bottom: 46},
    tooltip: {formatter: o => {
      const ym = ymOf(o.value), r = MO.find(x => x.ym === ym) || {};
      const f = v => v == null ? '—' : (v > 0 ? '+' : '') + (v * 100).toFixed(2) + '%';
      return `<b>${ym}</b><br>策略 ${f(r.p)}<br>基准 ${f(r.b)}<br>超额 ${f(r.x)}`;
    }},
    xAxis: {type: 'category', data: MNAME, splitArea: {show: true},
            axisLabel: {fontSize: 11, color: '#7a8296'}, axisLine: {show: false}},
    yAxis: {type: 'category', data: yset.map(String), splitArea: {show: true},
            axisLabel: {fontSize: 11, color: '#7a8296'}, axisLine: {show: false}},
    visualMap: {min: -abs, max: abs, calculable: false, orient: 'horizontal',
      left: 'center', bottom: 0, itemWidth: 12, itemHeight: 90, precision: 0,
      text: ['+%', '−%'], textStyle: {fontSize: 10, color: '#8a92a3'},
      inRange: {color: ['#12a05c', '#8ecfae', '#ffffff', '#f2b0b2', '#d9363e']}},
    series: [{type: 'heatmap', data: data,
      label: {show: true, fontSize: 10,
              formatter: o => (o.value[2] > 0 ? '+' : '') + o.value[2].toFixed(1)},
      emphasis: {itemStyle: {borderColor: '#16304f', borderWidth: 1}}}],
  }, true);
}
drawMonth('p');
const mm = document.getElementById('mMode');
[['策略收益', 'p'], ['超额收益', 'x']].forEach(([lbl, key], i) => {
  const b = document.createElement('button');
  b.textContent = lbl; if (!i) b.className = 'active';
  b.onclick = () => { mm.querySelectorAll('button').forEach(x => x.classList.remove('active'));
    b.classList.add('active'); drawMonth(key); };
  mm.appendChild(b);
});

/* ---------- 4. 回撤 ---------- */
const mdd = D.cur.mdd || {};
mk('ddChart', {
  animation: false, grid: {left: 56, right: 30, top: 16, bottom: 38},
  tooltip: {trigger: 'axis', valueFormatter: v => v == null ? '—' : (v * 100).toFixed(2) + '%'},
  xAxis: {type: 'category', data: D.dd.dates, axisLabel: {fontSize: 11, color: '#7a8296',
    hideOverlap: true}, axisLine: {lineStyle: {color: '#e4e7ec'}}},
  yAxis: {type: 'value', axisLabel: {formatter: v => (v * 100).toFixed(0) + '%', fontSize: 11,
    color: '#7a8296'}, splitLine: {lineStyle: {color: '#f4f5f8'}}},
  series: [{type: 'line', showSymbol: false, data: D.dd.values,
    lineStyle: {width: 1, color: neg},
    areaStyle: {color: 'rgba(46,125,50,.14)', origin: 'start'},
    markArea: (mdd.peak && mdd.trough) ? {
      silent: true, itemStyle: {color: 'rgba(217,54,62,.07)'},
      label: {fontSize: 10, color: '#b04a50', position: 'insideBottom'},
      data: [[{xAxis: mdd.peak, name: '最大回撤区间'}, {xAxis: mdd.trough}]]} : undefined}],
});

/* ---------- 5. 左侧导航滚动高亮 ---------- */
const links = Array.from(document.querySelectorAll('.bt-side a'));
const secs = links.map(a => document.querySelector(a.getAttribute('href')));
function syncNav(){
  const y = window.scrollY + 96;
  let idx = 0;
  secs.forEach((el, i) => { if (el && el.offsetTop <= y) idx = i; });
  links.forEach((a, i) => a.classList.toggle('on', i === idx));
}
window.addEventListener('scroll', syncNav, {passive: true});
syncNav();
"""




def render_html(payload: dict, body: str, js: str, title: str) -> str:
    from research.html_report import page
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    scripts = f"<script>window.__YQ__ = {data};</script>\n<script>{js}</script>"
    return page(
        title=title,
        header="",
        body=body,
        extra_css=_EXTRA_CSS,
        head_extra=_echarts_tag(),
        scripts=scripts,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="yqcore 定版策略前端页面（回测 / 模拟交易，自包含 HTML）")
    ap.add_argument("--tier", default="yqcore-focus-k8", choices=list(TIERS))
    ap.add_argument("--period", default=None, help="信号日 YYYYMMDD（缺省 = 最新一期）")
    ap.add_argument("--bench", default="bench_eqw", choices=list(BENCH_LABEL),
                    help="回测页基准（bench_eqw=全A等权，bench_index=沪深300）")
    ap.add_argument("--prod-dir", default=None)
    ap.add_argument("--live-dir", default=None)
    ap.add_argument("--base-dir", default=None,
                    help="基准 parquet 目录（缺省 reports/alla_rolling/_base）")
    ap.add_argument("--out-dir", default=None, help="输出目录（缺省 reports/yq_core）")
    ap.add_argument("--open", action="store_true", help="生成后用默认浏览器打开")
    ap.add_argument("--only", default="both", choices=["both", "backtest", "live"])
    args = ap.parse_args()

    prod = Path(args.prod_dir) if args.prod_dir else PROD_DIR
    live = Path(args.live_dir) if args.live_dir else prod / "live"
    base = Path(args.base_dir) if args.base_dir else BASE_DIR
    out_dir = Path(args.out_dir) if args.out_dir else prod
    out_dir.mkdir(parents=True, exist_ok=True)

    wrote: list[tuple[Path, float]] = []

    if args.only in ("both", "backtest"):
        bt = build_backtest_payload(prod, base, args.tier, args.bench)
        out = out_dir / "backtest.html"
        out.write_text(render_html(bt, render_backtest_body(bt), _BT_JS,
                                   f"{args.tier} 历史回测"), encoding="utf-8")
        wrote.append((out, out.stat().st_size / 1e6))
        c = bt["cur"]
        print(f"[回测] {out}  ({out.stat().st_size / 1e6:.2f} MB)")
        print(f"        {c['start']} ~ {c['end']} · {c['n_days']} 日 · "
              f"累计 {c['cum'] * 100:+.2f}% · 年化 {c['ann'] * 100:+.2f}% · "
              f"α {c['alpha'] * 100:+.2f}% · β {c['beta']:.3f} · "
              f"SR {c['sharpe_geo']:.3f} · MDD {c['mdd']['mdd'] * 100:.2f}%")

    if args.only in ("both", "live"):
        lp = build_live_payload(prod, live, args.tier, args.period)
        out = out_dir / "live.html"
        out.write_text(render_html(lp, render_live_body(lp), _LIVE_JS,
                                   f"{args.tier} 模拟交易"), encoding="utf-8")
        wrote.append((out, out.stat().st_size / 1e6))
        tot = lp["hold"]["totals"]
        print(f"[实盘] {out}  ({out.stat().st_size / 1e6:.2f} MB)")
        print(f"        信号日 {lp['hold']['signal_date']} → 成交 {lp['hold']['exec_date']}"
              f" · 估值 {lp['hold']['latest_date']}"
              f" · 实持 {len(lp['hold']['rows'])} 只"
              f" · 期间 {tot['ret'] * 100:+.2f}%")
        print(f"        自检 {'通过' if lp['check']['ok'] else '未通过'}"
              f"（重算 {lp['check']['computed']} vs meta {lp['check']['claimed']}）")

    if args.open and wrote:
        name = "backtest.html" if (out_dir / "backtest.html").exists() else "live.html"
        target = out_dir / name
        webbrowser.open(target.resolve().as_uri())


if __name__ == "__main__":
    main()
