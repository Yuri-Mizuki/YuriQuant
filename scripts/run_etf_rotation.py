"""
ETF 轮动最小闭环回测脚本
========================

用法:
    python -m scripts.run_etf_rotation                    # 默认参数跑通最小闭环
    python -m scripts.run_etf_rotation --n 60 --top_k 3 --freq M
    python -m scripts.run_etf_rotation --begin 20220101 --end 20260821
    python -m scripts.run_etf_rotation --grid             # 跑参数稳健性交叉
    python -m scripts.run_etf_rotation --out reports/etf_rotation/report.html

流程:
    1. 加载 ETF 候选池（后复权收盘价 / 收益率 / 动量分）与沪深300基准。
    2. 用 VectorBacktest + EtfCosts 跑动量轮动回测（周频调仓）。
    3. 输出绩效指标（vs 沪深300）并生成自包含 HTML 收益报告。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import EtfCosts
from backtest.engine import VectorBacktest
from backtest.metrics import format_metrics
from config import Config
from data.cache import DataCache
from data.datasource import create_datasource
from data.etf_universe import EtfUniverse
from research.html_report import page
from strategy.etf_rotation import EtfRotation
from strategy.multi_signal import compose_signals, DEFAULT_WEIGHTS

BENCH = "000300.SH"

# 流动性下限：平均日成交额（元）低于该值剔除（默认 0 = 不过滤）
MIN_AVG_AMOUNT = 0.0

# ETF 轮动报告主题样式（page 外壳 + 自定义 CSS）
_CSS = """
  :root { --bg:#fafbfc; --bg2:#eef2f7; --ink:#17233b; --muted:#5a6b85;
           --rule:#dfe6f0; --accent:#1559b6; --accent2:#0f8a80; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
          background:var(--bg); color:var(--ink); line-height:1.7; padding:2.5rem 1rem 4rem; }
  .page { max-width:960px; margin:0 auto; }
  header { border-bottom:1px solid var(--rule); padding-bottom:1.4rem; margin-bottom:2rem; }
  h1 { font-size:1.7rem; letter-spacing:-.01em; }
  .meta { font-size:.82rem; color:var(--muted); margin-top:.6rem; }
  .meta span { margin-right:1.2rem; }
  h2 { font-size:1.15rem; margin:2.2rem 0 .8rem; padding-bottom:.4rem; border-bottom:2px solid var(--accent); }
  .kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:.9rem; margin:1.2rem 0; }
  .kpi { background:var(--bg2); border-radius:10px; padding:.9rem 1rem; border-top:3px solid var(--accent); }
  .kpi b { font-family:inherit; font-size:1.35rem; display:block; color:var(--accent); }
  .kpi span { font-size:.78rem; color:var(--muted); }
  .panel { background:#ffffff; border:1px solid var(--rule); border-radius:10px; padding:1.2rem; margin:1.2rem 0; }
  .legend { font-size:.8rem; color:var(--muted); margin-top:.5rem; }
  .legend i { display:inline-block; width:14px; height:3px; margin-right:5px; vertical-align:middle; }
  .table-wrap { overflow-x:auto; margin:1rem 0; }
  table { width:100%; border-collapse:collapse; font-size:.88rem; }
  th,td { padding:.55rem .8rem; text-align:left; border-bottom:1px solid var(--rule); }
  thead th { background:var(--bg2); }
  .note { margin:1rem 0; padding:.8rem 1rem; border-left:4px solid #b45309; background:var(--bg2); font-size:.85rem; }
  @media (max-width:600px) { .kpi-row { grid-template-columns:repeat(2,1fr); } }
"""


def _build_panels(cache, begin: int, end: int, adjust: bool = True):
    """构建资产池动量分 / 收益率 / 成交额面板（严格按 [begin, end] 切片）。"""
    uni = EtfUniverse(cache)
    close = uni.load_close(adjust=adjust)
    # 切片到回测窗口：动量暖机依赖窗口起点之前的收盘价，故先取全量再截断
    lo, hi = pd.Timestamp(str(begin)), pd.Timestamp(str(end))
    close = close.loc[(close.index >= lo) & (close.index <= hi)]
    returns = close.pct_change(fill_method=None)

    # 流动性过滤（可选）
    raw = cache.read_daily("etf")
    if raw is not None and MIN_AVG_AMOUNT > 0:
        amt = raw["amount"].unstack("code")
        avg_amt = amt.mean()
        keep = [c for c in returns.columns if avg_amt.get(c, 0) >= MIN_AVG_AMOUNT]
        returns = returns[keep]
        close = close[keep]

    return uni, close, returns


def _load_benchmark(cache, begin: int, end: int) -> pd.Series:
    idx = cache.get_index_daily(BENCH, begin, end)
    if idx is None or idx.empty:
        return pd.Series(dtype=float)
    close = idx["close"].unstack("code")
    col = [c for c in close.columns if c in (BENCH, BENCH.replace(".SH", "") )]
    if not col:
        return pd.Series(dtype=float)
    s = close[col[0]].dropna()
    return s.pct_change(fill_method=None)


def _load_benchmark_close(cache, begin: int, end: int) -> pd.Series:
    """返回基准指数收盘价序列（供趋势择时门控）。"""
    idx = cache.get_index_daily(BENCH, begin, end)
    if idx is None or idx.empty:
        return pd.Series(dtype=float)
    close = idx["close"].unstack("code")
    col = [c for c in close.columns if c in (BENCH, BENCH.replace(".SH", "") )]
    if not col:
        return pd.Series(dtype=float)
    return close[col[0]].dropna()


def _regime_mask(bench_close: pd.Series, timing: str, lookback: int) -> pd.Series:
    """返回布尔 Series（risk_on=True）。timing: 'ma' | 'momentum'。"""
    if timing == "ma":
        ma = bench_close.rolling(lookback).mean()
        return bench_close > ma
    # momentum
    mom = bench_close.pct_change(lookback, fill_method=None)
    return mom > 0.0


def _build_bundle(cache, begin: int, end: int, adjust: bool = True):
    """一次性构建面板束（ETF面板 + 基准收益 + 基准收盘价），供多窗口回测复用。"""
    uni, close, returns = _build_panels(cache, begin, end, adjust=adjust)
    bench = _load_benchmark(cache, begin, end)
    bench_close = _load_benchmark_close(cache, begin, end)
    return uni, close, returns, bench, bench_close


def _run(bundle, top_k: int, freq: str, weights: dict[str, float] | None = None,
         signal: str | None = None, w0: int | None = None, w1: int | None = None,
         timing: str = "none", timing_lookback: int = 60) -> dict:
    """在面板束的（可选）窗口 [w0, w1] 上跑一次回测。

    timing: 'none'（不加择时）| 'ma'（大盘趋势门控）| 'momentum'（大盘动量门控）。
    门控在 risk-off 时把合成因子置 NaN → 策略返回空权重 → 整体转现金。
    """
    uni, close, returns, bench, bench_close = bundle
    if w0 is not None:
        lo, hi = pd.Timestamp(str(w0)), pd.Timestamp(str(w1))
        m = (close.index >= lo) & (close.index <= hi)
        close, returns = close.loc[m], returns.loc[m]
        if not bench.empty:
            bench = bench.loc[bench.index.isin(close.index)]
        if not bench_close.empty:
            bench_close = bench_close.loc[bench_close.index.isin(close.index)]

    if signal:
        factor, _norm, used_weights = compose_signals(close, weights={signal: 1.0})
        used_signal = signal
    else:
        factor, _norm, used_weights = compose_signals(close, weights=weights)
        used_signal = "+".join(used_weights)

    # 资产级择时门控：risk-off 日子 -> 因子置 NaN -> 空仓（与 cash_filter 叠加）
    if timing != "none" and not bench_close.empty:
        risk_on = _regime_mask(bench_close, timing, timing_lookback)
        off_days = factor.index[~risk_on.reindex(factor.index).fillna(False)]
        if len(off_days):
            factor.loc[off_days, :] = np.nan

    strategy = EtfRotation(top_k=top_k, cash_filter=True)
    bt = VectorBacktest(
        strategy=strategy,
        rebalance_freq=freq,
        initial_capital=1_000_000.0,
        costs=EtfCosts(),
    )
    result = bt.run(factor_panel=factor, returns_panel=returns, horizon=1)
    metrics = result.metrics(benchmark_returns=bench if not bench.empty else None)

    return {
        "strategy": strategy,
        "metrics": metrics,
        "bench_returns": bench,
        "equity_curve": result.equity_curve,
        "weights_history": result.weights_history,
        "cost_series": result.cost_series,
        "turnover_series": result.turnover_series,
        "uni": uni,
        "signal": used_signal,
        "weights": used_weights,
    }


def run_once(cache, begin: int, end: int, top_k: int, freq: str,
             weights: dict[str, float] | None = None, signal: str | None = None,
             adjust: bool = True) -> dict:
    """兼容入口：构建全样本面板束后跑一次。"""
    bundle = _build_bundle(cache, begin, end, adjust=adjust)
    return _run(bundle, top_k, freq, weights=weights, signal=signal)


def _row(m) -> dict:
    bench = m.get("benchmark_annual_return")
    return {
        "total_return": m["total_return"],
        "annual_return": m["annual_return"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "excess": m["annual_return"] - (bench or 0),
        "avg_turnover": m.get("avg_turnover", 0),
    }


def _grid_signals(bundle, top_k: int, freq: str, weights: dict[str, float]) -> list[dict]:
    """单信号 vs 合成 对比（同一 top_k / 频率口径，复用面板束）。"""
    rows = []
    for name in list(DEFAULT_WEIGHTS) + ["__composite__"]:
        sig = None if name == "__composite__" else name
        r = _run(bundle, top_k, freq, weights=weights, signal=sig)
        m = _row(r["metrics"])
        rows.append({"name": r["signal"], **m})
    return rows


def _grid_robust(bundle, weights: dict[str, float]) -> list[dict]:
    """合成信号的 top_k × 调仓频率 稳健性交叉。"""
    rows = []
    for top_k in (3, 5):
        for freq in ("W", "M"):
            r = _run(bundle, top_k, freq, weights=weights)
            m = _row(r["metrics"])
            rows.append({"top_k": top_k, "freq": freq, **m})
    return rows


# 动量族权重离散网格（相对权重，z 标准化后只比比例）
WEIGHT_GRID: list[tuple[str, dict[str, float]]] = [
    ("等权(1:1:1)", {"mom20": 1.0, "mom60": 1.0, "voladj": 1.0}),
    ("当前(m20×2+m60×1.5+v×1)", {"mom20": 2.0, "mom60": 1.5, "voladj": 1.0}),
    ("短动主导(3:1:1)", {"mom20": 3.0, "mom60": 1.0, "voladj": 1.0}),
    ("去掉中动(2:0:1)", {"mom20": 2.0, "voladj": 1.0}),
    ("中动主导(1:3:1)", {"mom20": 1.0, "mom60": 3.0, "voladj": 1.0}),
    ("波动主导(1:1:3)", {"mom20": 1.0, "mom60": 1.0, "voladj": 3.0}),
    ("短动+去波动(3:1:0)", {"mom20": 3.0, "mom60": 1.0}),
    ("短中均衡(1.5:1.5:1)", {"mom20": 1.5, "mom60": 1.5, "voladj": 1.0}),
    ("动能偏强(2:2:1)", {"mom20": 2.0, "mom60": 2.0, "voladj": 1.0}),
    ("风险保守(1:1:2)", {"mom20": 1.0, "mom60": 1.0, "voladj": 2.0}),
]

# 分段边界：label -> (segment_begin, segment_end)，覆盖不同市场状态
SEGMENTS: list[tuple[str, int, int]] = [
    ("2020-21", 20200101, 20220101),
    ("2022-23", 20220101, 20240101),
    ("2024-26", 20240101, 20260821),
]


def _grid_weights(bundle, top_k: int, freq: str, end: int) -> dict:
    """动量族权重离散网格 × 分段稳健性（复用面板束）。

    Returns:
        {rows: 逐组合指标, segs: 分段列名}
    """
    segs = [(l, b, e) for l, b, e in SEGMENTS if b < end]
    rows = []
    for name, w in WEIGHT_GRID:
        r = _run(bundle, top_k, freq, weights=w, signal=None)
        m = _row(r["metrics"])
        seg_ann: dict[str, float] = {}
        for label, sb, se in segs:
            rs = _run(bundle, top_k, freq, weights=w, signal=None, w0=sb, w1=se)
            seg_ann[label] = rs["metrics"]["annual_return"]
        n_pos = sum(1 for v in seg_ann.values() if v > 0)
        rows.append({
            "name": name,
            "full_ann": m["annual_return"],
            "full_sharpe": m["sharpe"],
            "full_mdd": m["max_drawdown"],
            **{f"seg_{l}": seg_ann[l] for l, *_ in segs},
            "n_pos_seg": n_pos,
        })
    return {"rows": rows, "seg_labels": [l for l, *_ in segs]}


TIMING_METHODS = ["none", "ma", "momentum"]


def _grid_timing(bundle, top_k: int, freq: str, weights: dict[str, float],
                 end: int, lookback: int = 60) -> dict:
    """择时方法 × 分段稳健性：检验门控能否修复 2022-23 阴跌段。"""
    segs = [(l, b, e) for l, b, e in SEGMENTS if b < end]
    rows = []
    for tm in TIMING_METHODS:
        r = _run(bundle, top_k, freq, weights=weights, timing=tm, timing_lookback=lookback)
        m = _row(r["metrics"])
        seg_ann: dict[str, float] = {}
        for label, sb, se in segs:
            rs = _run(bundle, top_k, freq, weights=weights, w0=sb, w1=se,
                      timing=tm, timing_lookback=lookback)
            seg_ann[label] = rs["metrics"]["annual_return"]
        n_pos = sum(1 for v in seg_ann.values() if v > 0)
        rows.append({
            "name": tm,
            "full_ann": m["annual_return"],
            "full_sharpe": m["sharpe"],
            "full_mdd": m["max_drawdown"],
            **{f"seg_{l}": seg_ann[l] for l, *_ in segs},
            "n_pos_seg": n_pos,
        })
    return {"rows": rows, "seg_labels": [l for l, *_ in segs], "lookback": lookback}


def _top_holdings(weights_history: pd.DataFrame, uni: EtfUniverse, top_n: int = 8) -> list[dict]:
    """统计各 ETF 的平均持有权重（非零日），输出最常持有的前 top_n。"""
    w = weights_history  # date × code
    freq_held = (w > 0).mean()  # 持有天数占比
    avg_w = (w[w > 0]).mean()  # 平均权重（仅持有日）
    out = []
    for code in w.columns:
        if w[code].sum() == 0:
            continue
        out.append({
            "code": code,
            "name": uni.label(code),
            "category": uni.category(code),
            "held_pct": float(freq_held.get(code, 0)),
            "avg_weight": float(avg_w.get(code, 0)),
        })
    out.sort(key=lambda x: -x["held_pct"])
    return out[:top_n]


# ---------------------------------------------------------------------------
# 收益报告（自包含 HTML，内联 SVG 净值曲线）
# ---------------------------------------------------------------------------
def _svg_equity(dates: list[pd.Timestamp], port: np.ndarray, bench: np.ndarray) -> str:
    W, H = 880, 300
    pad_l, pad_r, pad_t, pad_b = 16, 8, 10, 22
    vals = list(port) + list(bench)
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        hi = lo + 1
    span_pad = (hi - lo) * 0.06
    lo -= span_pad
    hi += span_pad

    def xy(arr):
        pts = []
        for i, v in enumerate(arr):
            if np.isnan(v):
                continue
            x = pad_l + (W - pad_l - pad_r) * i / max(1, len(arr) - 1)
            y = pad_t + (H - pad_t - pad_b) * (hi - v) / (hi - lo)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    def gridline(y_frac, label):
        x0, x1 = pad_l, W - pad_r
        y = pad_t + (H - pad_t - pad_b) * y_frac
        return (f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                f'stroke="var(--rule)" stroke-width="1"/>'
                f'<text x="{W - pad_r}" y="{y - 4:.1f}" text-anchor="end" '
                f'font-size="11" fill="var(--muted)">{label}</text>')

    def yearline(i, label):
        x = pad_l + (W - pad_l - pad_r) * i / max(1, len(port) - 1)
        y_bot = H - pad_b
        return (f'<text x="{x:.1f}" y="{y_bot + 14:.1f}" text-anchor="middle" '
                f'font-size="11" fill="var(--muted)">{label}</text>')

    # 三年划分标签：0 / 1/3 / 2/3 / 1
    ticks = [0, len(port) // 3, 2 * len(port) // 3, len(port) - 1]
    year_texts = "".join(yearline(i, dates[i].year) if i < len(dates) else "" for i in ticks)

    return (
        f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'{gridline(0.25, f"{hi:.2f}")}{gridline(0.5, f"{(hi+lo)/2:.2f}")}{gridline(0.75, f"{lo:.2f}")}'
        f'<polyline fill="none" stroke="var(--muted)" stroke-width="1.5" '
        f'stroke-dasharray="5 4" points="{xy(bench)}"/>'
        f'<polyline fill="none" stroke="var(--accent)" stroke-width="2.2" '
        f'points="{xy(port)}"/>'
        f'{year_texts}'
        f'</svg>'
    )


def _render_report(out_path: Path, r: dict, sig_rows: list[dict], robust_rows: list[dict],
                   params: dict, weight_grid: dict | None = None,
                   timing_grid: dict | None = None) -> str:
    m = r["metrics"]
    eq = r["equity_curve"]
    bench = r["bench_returns"]
    n_etf = len(r["weights_history"].columns)
    common = eq.index.intersection(bench.index) if not bench.empty else eq.index
    port_cum = eq.reindex(common).values
    bench_cum = (1 + bench.reindex(common).fillna(0)).cumprod().values if not bench.empty else np.zeros(len(eq))

    meta = params
    kpi = [
        ("累计收益", f"{m['total_return']:.2%}"),
        ("年化收益", f"{m['annual_return']:.2%}"),
        ("夏普比率", f"{m['sharpe']:.2f}"),
        ("最大回撤", f"{m['max_drawdown']:.2%}"),
        ("卡玛比率", f"{m.get('calmar', 0):.2f}"),
        ("胜率", f"{m['win_rate']:.2%}"),
        ("平均换手", f"{m.get('avg_turnover', 0):.2%}"),
        ("超额年化", f"{m.get('excess_return', 0):.2%}"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><b>{v}</b><span>{k}</span></div>' for k, v in kpi
    )

    top = _top_holdings(r["weights_history"], r["uni"])
    holds_html = "".join(
        f'<tr><td>{t["code"]}</td><td>{t["name"]}</td><td>{t["category"]}</td>'
        f'<td>{t["held_pct"]:.1%}</td><td>{t["avg_weight"]:.1%}</td></tr>'
        for t in top
    )

    wt_html = " + ".join(f"{k}×{w:g}" for k, w in r["weights"].items())
    sig_html = ("<table><thead><tr><th>信号</th><th>累计收益</th><th>年化</th>"
                "<th>夏普</th><th>最大回撤</th><th>超额年化</th><th>平均换手</th></tr></thead><tbody>"
                + "".join(
                    f'<tr><td>{g["name"]}</td><td>{g["total_return"]:.2%}</td>'
                    f'<td>{g["annual_return"]:.2%}</td><td>{g["sharpe"]:.2f}</td>'
                    f'<td>{g["max_drawdown"]:.2%}</td><td>{g["excess"]:+.2%}</td>'
                    f'<td>{g["avg_turnover"]:.2%}</td></tr>'
                    for g in sig_rows
                ) + "</tbody></table>")

    if robust_rows:
        robust_html = ("<table><thead><tr><th>TopK</th><th>频率</th>"
                       "<th>累计收益</th><th>年化</th><th>夏普</th><th>最大回撤</th>"
                       "<th>超额年化</th><th>平均换手</th></tr></thead><tbody>"
                       + "".join(
                           f'<tr><td>{g["top_k"]}</td><td>{g["freq"]}</td>'
                           f'<td>{g["total_return"]:.2%}</td><td>{g["annual_return"]:.2%}</td>'
                           f'<td>{g["sharpe"]:.2f}</td><td>{g["max_drawdown"]:.2%}</td>'
                           f'<td>{g["excess"]:+.2%}</td><td>{g["avg_turnover"]:.2%}</td></tr>'
                           for g in robust_rows
                       ) + "</tbody></table>")
    else:
        robust_html = ""

    if weight_grid and weight_grid["rows"]:
        seg_heads = "".join(f"<th>{l} 年化</th>" for l in weight_grid["seg_labels"])
        wg_html = ("<table><thead><tr><th>权重组合</th><th>全样本年化</th>"
                   "<th>全样本夏普</th><th>全样本MDD</th>" + seg_heads +
                   "<th>正分段数</th></tr></thead><tbody>"
                   + "".join(
                       f'<tr><td>{g["name"]}</td><td>{g["full_ann"]:.2%}</td>'
                       f'<td>{g["full_sharpe"]:.2f}</td><td>{g["full_mdd"]:.2%}</td>'
                       + "".join(f'<td>{g[f"seg_{l}"]:.2%}</td>' for l in weight_grid["seg_labels"])
                       + f'<td>{g["n_pos_seg"]}/{len(weight_grid["seg_labels"])}</td></tr>'
                       for g in weight_grid["rows"]
                   ) + "</tbody></table>")
        wg_section = (f'<h2>动量族权重网格 × 分段稳健性（TopK={params["top_k"]}, {params["freq"]}）</h2>'
                      f'<div class="table-wrap">{wg_html}</div>'
                      f'<div class="note">权 z 标准化后仅比相对比例；分段 2020-21(波动) / 2022-23(熊市) / 2024-26(近期)。'
                      f'「正分段数」= 各分段年化&gt;0 的个数，衡量组合是否跨市场状态一致为正。</div>')
    else:
        wg_section = ""

    if timing_grid and timing_grid["rows"]:
        tg_heads = "".join(f"<th>{l} 年化</th>" for l in timing_grid["seg_labels"])
        tg_html = ("<table><thead><tr><th>择时方法</th><th>全样本年化</th>"
                   "<th>全样本夏普</th><th>全样本MDD</th>" + tg_heads +
                   "<th>正分段数</th></tr></thead><tbody>"
                   + "".join(
                       f'<tr><td>{g["name"]}</td><td>{g["full_ann"]:.2%}</td>'
                       f'<td>{g["full_sharpe"]:.2f}</td><td>{g["full_mdd"]:.2%}</td>'
                       + "".join(f'<td>{g[f"seg_{l}"]:.2%}</td>' for l in timing_grid["seg_labels"])
                       + f'<td>{g["n_pos_seg"]}/{len(timing_grid["seg_labels"])}</td></tr>'
                       for g in timing_grid["rows"]
                   ) + "</tbody></table>")
        tg_section = (f'<h2>自适应择时方法 × 分段稳健性（大盘门控, 回看={timing_grid["lookback"]}日）</h2>'
                      f'<div class="table-wrap">{tg_html}</div>'
                      f'<div class="note">none=不加择时；ma=大盘收盘价&gt;N日均线才持仓；momentum=大盘N日动量为正才持仓。'
                      f'门控在 risk-off 时整体转现金。</div>')
    else:
        tg_section = ""

    cost_total = float(r["cost_series"].sum())
    ts = r["turnover_series"]
    rebal = ts[ts > 0]
    avg_turnover_rebal = float(rebal.mean()) if len(rebal) else 0.0
    n_rebal = int(len(rebal))

    body = f"""<div class="page">
<header>
  <h1>ETF 轮动策略收益报告</h1>
  <div class="meta">
    <span>策略: {r['strategy'].name}</span>
    <span>信号: {r['signal']}</span>
    <span>标的池: {n_etf} 只宽基+行业</span>
    <span>调仓: {params['freq']}</span>
    <span>成本: ETF 场内(免印花税)</span>
    <span>基准: 沪深300</span>
  </div>
</header>

<h2>绩效指标</h2>
<div class="kpi-row">{kpi_html}</div>
<div class="note">样本区间 {meta['begin']} – {meta['end']}；合成信号权重 {wt_html}，持仓 TopK={meta['top_k']}，{meta['freq']} 调仓，等权持有，方向过滤（TopK 综合分均值≤0 转现金）。权重为透明合成设定，非样本内调优。本报告为 MVP 最小闭环演示。</div>

<h2>净值曲线</h2>
<div class="panel">
  {_svg_equity(common, port_cum, bench_cum)}
  <div class="legend"><i style="background:var(--accent)"></i>ETF 轮动净值(归一化)
    <i style="background:var(--muted)"></i>沪深300 净值(归一化)</div>
</div>

<h2>成本与换手</h2>
<div class="kpi-row">
  <div class="kpi"><b>{cost_total:,.0f}</b><span>累计交易成本(元, 100万本金)</span></div>
  <div class="kpi"><b>{avg_turnover_rebal:.2%}</b><span>平均单次调仓单边换手</span></div>
  <div class="kpi"><b>{n_rebal}</b><span>调仓次数</span></div>
  <div class="kpi"><b>{len(top)}</b><span>实际持有 ETF 数</span></div>
</div>

<h2>持有分布（按持有时间占比）</h2>
<div class="table-wrap">
  <table>
    <thead><tr><th>代码</th><th>名称</th><th>类别</th><th>持有时间占比</th><th>平均权重</th></tr></thead>
    <tbody>{holds_html}</tbody>
  </table>
</div>

<h2>单信号 vs 合成（TopK={params['top_k']}, {params['freq']}）</h2><div class="table-wrap">{sig_html}</div>
{f'<h2>合成信号稳健性（TopK × 调仓频率）</h2><div class="table-wrap">{robust_html}</div>' if robust_html else ''}
{wg_section}
{tg_section}

</div>"""
    html = page("ETF 轮动策略收益报告", header="", body=body, css=_CSS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


def _parse_weights(s: str | None) -> dict[str, float] | None:
    """把 'mom20:2,trend:1,voladj:1' 解析为权重 dict；空则用默认。"""
    if not s:
        return None
    out = {}
    for item in s.split(","):
        k, _, v = item.partition(":")
        out[k.strip()] = float(v)
    return out


def main():
    parser = argparse.ArgumentParser(description="ETF 轮动回测（多信号合成）")
    parser.add_argument("--begin", type=int, default=20200101)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--freq", default="W", choices=["D", "W", "M"])
    parser.add_argument("--weights", default=None,
                        help="合成权重，逗号分隔 如 'mom20:2,trend:1,voladj:1'；默认 "
                             + ",".join(f"{k}:{v:g}" for k, v in DEFAULT_WEIGHTS.items()))
    parser.add_argument("--grid", action="store_true", help="跑单信号对比与合成稳健性交叉")
    parser.add_argument("--weight-grid", action="store_true", help="跑动量族权重离散网格 × 分段稳健性")
    parser.add_argument("--timing-grid", action="store_true", help="跑自适应择时方法 × 分段稳健性")
    parser.add_argument("--timing", default="none", choices=["none", "ma", "momentum"],
                        help="自适应择时：none/ma(大盘N日均线)/momentum(大盘N日动量)")
    parser.add_argument("--timing_lookback", type=int, default=60, help="择时回看（交易日）")
    parser.add_argument("--out", default=None, help="报告输出路径")
    parser.add_argument("--adjust", default=True, action="store_true")
    args = parser.parse_args()

    weights = _parse_weights(args.weights)
    cache = DataCache(create_datasource())
    end = args.end
    if end is None:
        cal = cache.get_calendar(args.begin, None)
        end = cal[-1] if cal else 20260821

    bundle = _build_bundle(cache, args.begin, end, adjust=args.adjust)
    r = _run(bundle, args.top_k, args.freq, weights=weights,
             timing=args.timing, timing_lookback=args.timing_lookback)
    print("=== ETF 轮动主结果（多信号合成 + 自适应择时）===")
    wt_s = " + ".join(f"{k}×{w:g}" for k, w in r["weights"].items())
    print(f"策略: {r['strategy'].name}  信号: {r['signal']} ({wt_s})  择时: {args.timing}(回看{args.timing_lookback})"
          f"  区间 {args.begin}-{end} TopK={args.top_k} 调仓={args.freq}")
    print(format_metrics(r["metrics"]))

    sig_rows = _grid_signals(bundle, args.top_k, args.freq, r["weights"]) if args.grid else []
    robust_rows = _grid_robust(bundle, r["weights"]) if args.grid else []
    weight_grid = _grid_weights(bundle, args.top_k, args.freq, end) if args.weight_grid else None
    timing_grid = _grid_timing(bundle, args.top_k, args.freq, r["weights"], end,
                               args.timing_lookback) if args.timing_grid else None

    if weight_grid:
        print("\n=== 动量族权重网格 × 分段稳健性（TopK={}，{}) ===".format(args.top_k, args.freq))
        head = "\t".join(["权重组合", "全样本年化", "全样本夏普"] + weight_grid["seg_labels"] + ["正分段数"])
        print(head)
        for g in weight_grid["rows"]:
            cells = "\t".join([
                g["name"],
                f"{g['full_ann']:.2%}",
                f"{g['full_sharpe']:.2f}",
                *[f"{g[f'seg_{l}']:.2%}" for l in weight_grid["seg_labels"]],
                f"{g['n_pos_seg']}/{len(weight_grid['seg_labels'])}",
            ])
            print(cells)

    if timing_grid:
        print("\n=== 自适应择时方法 × 分段稳健性（回看{}日）===".format(timing_grid["lookback"]))
        head = "\t".join(["择时", "全样本年化", "全样本夏普"] + timing_grid["seg_labels"] + ["正分段数"])
        print(head)
        for g in timing_grid["rows"]:
            cells = "\t".join([
                g["name"],
                f"{g['full_ann']:.2%}",
                f"{g['full_sharpe']:.2f}",
                *[f"{g[f'seg_{l}']:.2%}" for l in timing_grid["seg_labels"]],
                f"{g['n_pos_seg']}/{len(timing_grid['seg_labels'])}",
            ])
            print(cells)

    cfg = Config.get()
    out = args.out or str(Path(cfg.get("backtest", {}).get("ledger_root", "reports/etf_rotation")) / "report.html")
    path = _render_report(Path(out), r, sig_rows, robust_rows, {
        "begin": args.begin, "end": end, "top_k": args.top_k, "freq": args.freq,
    }, weight_grid=weight_grid, timing_grid=timing_grid)
    print(f"\n报告已生成: {path}")


if __name__ == "__main__":
    main()