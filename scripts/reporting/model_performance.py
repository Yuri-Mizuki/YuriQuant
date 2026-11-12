"""主模型回测表现页（净值 / 回撤 / 分年 / 月度热图，单文件 HTML）。

数据链路与 `scripts.evaluation.caliber` **同源**，不另算一套口径：

    caliber.load_context()          → 同一份面板、同一掩码、同一窗口
    caliber._arm_strategies()       → 同一批臂（TopFrac / BufferedTopFrac）
    caliber.run_once + _metrics     → 同一引擎、同一成本、同一指标实现
    research.benchmarks.segmented_equal_weight_returns()
                                    → 池化等权基准（项目唯一实现，investable= 传池掩码）

🚨 只画**本机跑得出来**的臂。`config/baselines.yaml` 里登记了但信号面板不在本机的
条目（如 920 定版 `dingban_fundind`：`reports/frontier_audit_920_reval/` 与
`alla_rolling_ortho920fundind/pred/` 本机均为空）**不画曲线**，只在页尾「登记基线
对照」表里列出登记值并标注 `未在本机重跑`——不拿别人的数字冒充自己的回测。

用法:
    python -m scripts.reporting.model_performance
    python -m scripts.reporting.model_performance --signal ens_h1h5 --window prod
    python -m scripts.reporting.model_performance --signal dingban_fundind   # 缺面板会报错
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import VectorBacktest  # noqa: E402
from research.benchmarks import segmented_equal_weight_returns  # noqa: E402
from scripts.evaluation import caliber as cal  # noqa: E402

OUT_DIR = ROOT / "reports" / "model_performance"
CHART_CDN = ('<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/'
             'dist/chart.umd.min.js"></script>')

ARM_LABEL = {
    "eqw_top10pct": "Top10% 等权",
    "eqw_buf0.15_0.4": "Buffer 15/40（交付臂）",
    "eqw_buf0.15_0.4_0.2_0.5": "Buffer 15/40 + 20/50",
}


# ---------------------------------------------------------------- 回测
def run_arm(strategy, ctx, cost, capital: float, execution: str):
    """跑一个臂，返回引擎结果（与 caliber.evaluate_strategy 同一路径）。"""
    bt = VectorBacktest(strategy=strategy, rebalance_freq="M",
                        initial_capital=float(capital), costs=cost)
    return cal.run_once(bt, ctx, execution)


def build_benchmark(ctx) -> pd.Series:
    """池化等权基准：与策略同池、月频重置、段内随价漂移（不是日频再平衡）。

    🚨 必须传「窗口首日**前一交易日**起」的价格切片 + 显式 ``rebalance_dates``：
    直接传 close 全历史会让 2016 年那些月份成为调仓日（当时列里大半股票未上市），
    触发函数内「面板被裁剪」守卫（有效列数 < 50%）。
    """
    from research.benchmarks import monthly_first_trading_days

    close_full = pd.read_parquet(cal.BASE_DIR / "close_adj.parquet")
    mask_full = pd.read_parquet(cal.BASE_DIR / "tradable_mask.parquet")
    cols = ctx.pred.columns
    first, last = ctx.pred.index[0], ctx.pred.index[-1]
    prev = close_full.index[close_full.index < first]
    start = prev[-1] if len(prev) else first
    px = close_full.loc[start:last, cols]
    reb = monthly_first_trading_days(ctx.pred.index)
    bench = segmented_equal_weight_returns(
        px, rebalance_dates=reb, index=ctx.pred.index,
        investable=mask_full.reindex(columns=cols))
    return bench.reindex(ctx.pred.index).fillna(0.0)


def eq_curve(dr: pd.Series) -> pd.Series:
    s = dr.reindex(dr.index).fillna(0.0)
    return (1.0 + s).cumprod()


def drawdown(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


def yearly(dr: pd.Series) -> dict[str, float]:
    out = {}
    for y, sub in dr.groupby(dr.index.year):
        out[str(y)] = float((1 + sub.fillna(0.0)).prod() - 1)
    return out


def monthly(dr: pd.Series) -> dict[str, float]:
    m = (1 + dr.fillna(0.0)).resample("ME").prod() - 1
    return {p.strftime("%Y-%m"): float(v) for p, v in m.items()}


def week_sample(s: pd.Series) -> tuple[list[str], list[float]]:
    """净值按周采样（图表点数减半，形状不失真）。"""
    w = s.resample("W-FRI").last().dropna()
    w = pd.concat([s.iloc[[0]], w]).drop_duplicates()
    return [d.strftime("%Y-%m-%d") for d in w.index], [round(float(v), 4) for v in w]


# ---------------------------------------------------------------- 登记值对照
def registered_arms(signal: str, window: str) -> list[dict]:
    """从 config/baselines.yaml 取同 signal+window 的登记指标，供页面并列展示。"""
    p = ROOT / "config" / "baselines.yaml"
    if not p.exists():
        return []
    import yaml
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rows = []
    for b in doc.get("baselines") or []:
        prof = ((b.get("metrics") or {}).get("profiles") or {}).get("production") or {}
        rows.append({
            "id": b.get("id"), "status": b.get("status"), "label": b.get("label"),
            "signal_key": b.get("signal_key"), "window": b.get("window"),
            "production_all": prof,
            "production_arm": prof.get("eqw_buf0.15_0.4"),
            "verified_local": b.get("verified_local"),
            "same_as_run": b.get("signal_key") == signal and b.get("window") == window,
        })
    return rows


# ---------------------------------------------------------------- 渲染
_CSS = """
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
background:#f6f7f9;color:#1f2328;font-size:13px}
.wrap{max-width:1180px;margin:0 auto;padding:18px 20px 40px}
h1{font-size:19px;margin:0 0 4px}
.sub{color:#6a737d;font-size:12px;margin-bottom:14px}
.banner{background:#fff;border:1px solid #e3e6ea;border-left:3px solid #2f6feb;
border-radius:6px;padding:10px 12px;margin-bottom:14px;line-height:1.7}
.banner b{color:#0b3d91}
.warn{border-left-color:#d1242f;background:#fff8f8}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
.kpi{background:#fff;border:1px solid #e3e6ea;border-radius:6px;padding:10px 12px}
.kpi .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.kpi .k{color:#6a737d;font-size:11px;margin-top:2px}
.pos{color:#c62828}.neg{color:#1b7f3b}
.card{background:#fff;border:1px solid #e3e6ea;border-radius:6px;padding:12px 14px;
margin-bottom:14px}
.card h2{font-size:13px;margin:0 0 8px;color:#24292f}
.chart{height:300px;position:relative}
.chart.sm{height:220px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid #eef0f3;text-align:right;white-space:nowrap}
th{background:#fafbfc;color:#57606a;font-weight:600;font-size:11px;text-align:right}
th:first-child,td:first-child{text-align:left}
tr.hl td{background:#f4f8ff;font-weight:600}
.hm td{padding:3px 5px;font-size:11px;border:1px solid #fff;text-align:center;color:#1f2328}
.hm th{text-align:center;font-size:11px}
.foot{color:#6a737d;font-size:11px;line-height:1.8;margin-top:16px}
code{background:#f2f4f7;padding:1px 4px;border-radius:3px;font-size:11px}
"""


def _pct(v, nd=2):
    return "—" if v is None or not np.isfinite(v) else f"{v*100:.{nd}f}%"


def _f(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def render_html(payload: dict) -> str:
    meta = payload["meta"]
    arms = payload["arms"]
    bench = payload["bench"]
    main_tag = payload["main_arm"]
    main = arms[main_tag]

    kpis = [
        ("年化收益", _pct(main["ann"]), main["ann"]),
        ("Sharpe（几何）", _f(main["sharpe"]), main["sharpe"]),
        ("Sharpe（算术）", _f(main["sharpe_arith"]), main["sharpe_arith"]),
        ("NW-t（显著性）", _f(main["sharpe_t_stat"], 2), main["sharpe_t_stat"]),
        ("最大回撤", _pct(main["max_dd"]), -main["max_dd"]),
        ("年化换手（单边）", _f(main["turnover"], 3), None),
        ("累计成本", f"{main['cost_bp']:.0f} bp", None),
        ("平均持股数", f"{main['n_active']:.0f}", None),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for k, v, _ in kpis)

    # 指标对比表（各臂 + 基准）
    head = ("<tr><th>口径 / 臂</th><th>年化</th><th>Sharpe几何</th><th>Sharpe算术</th>"
            "<th>NW-t</th><th>波动</th><th>最大回撤</th><th>换手</th><th>成本 bp</th>"
            "<th>持股</th></tr>")
    rows = []
    for tag, m in arms.items():
        cls = ' class="hl"' if tag == main_tag else ""
        reg = meta.get("registered", {}).get(tag)
        mark = ""
        if reg:
            d = (m["ann"] - reg["ann"]) * 100
            mark = (f' <span style="color:#6a737d">（登记 {reg["ann"]*100:.2f}%，'
                    f'Δ{d:+.2f}pp）</span>')
        rows.append(
            f"<tr{cls}><td>{ARM_LABEL.get(tag, tag)}{mark}</td>"
            f'<td class="pos">{_pct(m["ann"])}</td><td>{_f(m["sharpe"])}</td>'
            f'<td>{_f(m["sharpe_arith"])}</td><td>{_f(m["sharpe_t_stat"], 2)}</td>'
            f'<td>{_pct(m["vol"])}</td><td class="neg">{_pct(m["max_dd"])}</td>'
            f'<td>{_f(m["turnover"], 3)}</td><td>{m["cost_bp"]:.0f}</td>'
            f'<td>{m["n_active"]:.0f}</td></tr>')
    rows.append(
        f'<tr><td>等权基准（同池月度重置）</td><td class="pos">{_pct(bench["ann"])}</td>'
        f'<td>{_f(bench["sharpe"])}</td><td>{_f(bench["sharpe_arith"])}</td>'
        f'<td>{_f(bench["sharpe_t_stat"], 2)}</td><td>{_pct(bench["vol"])}</td>'
        f'<td class="neg">{_pct(bench["max_dd"])}</td><td>—</td><td>0</td><td>—</td></tr>')

    # 分年表
    years = payload["years"]
    yhead = "<tr><th>年份</th><th>策略</th><th>基准</th><th>超额</th></tr>"
    yrows = "".join(
        f"<tr><td>{y}</td><td class='{ 'pos' if v['s'] >= 0 else 'neg' }'>{_pct(v['s'])}</td>"
        f"<td class='{ 'pos' if v['b'] >= 0 else 'neg' }'>{_pct(v['b'])}</td>"
        f"<td class='{ 'pos' if v['e'] >= 0 else 'neg' }'>{_pct(v['e'])}</td></tr>"
        for y, v in years.items())

    # 月度热图
    mon = payload["monthly"]
    yrs = sorted({k[:4] for k in mon})
    hm = ['<tr><th>年</th>' + "".join(f"<th>{m}月</th>" for m in range(1, 13))
          + "<th>全年</th></tr>"]
    for y in yrs:
        cells = []
        for m in range(1, 13):
            k = f"{y}-{m:02d}"
            if k in mon:
                v = mon[k]
                bg = "#fdecea" if v > 0 else ("#e8f5ec" if v < 0 else "#f5f5f5")
                cells.append(f'<td style="background:{bg}">{v*100:+.1f}</td>')
            else:
                cells.append("<td>·</td>")
        yv = years.get(y, {}).get("s")
        cells.append(f"<td><b>{_pct(yv, 1)}</b></td>")
        hm.append(f"<tr><th>{y}</th>{''.join(cells)}</tr>")

    # 登记基线对照（含本机跑不了的条目）
    reg_head = ("<tr><th>id</th><th>status</th><th>窗口</th><th>登记年化</th>"
                "<th>登记 Sharpe</th><th>NW-t</th><th>本机可重跑</th></tr>")
    reg_rows = []
    for r in payload["registry"]:
        pa = r.get("production_arm") or {}
        ok = "✅ 本次跑的正是这条" if r["same_as_run"] else (
            "❌ 信号面板不在本机" if r.get("status") == "production" else "—")
        reg_rows.append(
            f"<tr><td>{r['id']}</td><td>{r['status']}</td><td>{r['window']}</td>"
            f"<td>{_pct(pa.get('ann'))}</td><td>{_f(pa.get('sharpe'))}</td>"
            f"<td>{_f(pa.get('sharpe_t_stat'), 2)}</td><td>{ok}</td></tr>")

    warn = ""
    if payload.get("missing_production"):
        warn = (f'<div class="banner warn"><b>⚠️ 本页不是生产定版</b>：'
                f'config/baselines.yaml 里 status=production 的是 '
                f'<b>{payload["missing_production"]}</b>（920 因子 fundind 面板），'
                '但它的信号面板 <code>reports/frontier_audit_920_reval/defv_sig__h1.parquet</code> '
                '与 <code>alla_rolling_ortho920fundind/pred/</code> 在本机不存在 ⇒ 跑不出来。'
                '本页曲线是 <b>working 代理线（882 ortho·h1+h5）</b>，'
                f'定版指标见页尾登记表（{payload["missing_production"]} 那行，'
                '标注「未在本机重跑」）。'
                '重建定版信号的成本另计：须先跑 920 pred 的 4 个周期，再 '
                '<code>buffer_arm_920 --variant defv --save-sig</code>，'
                '见 baselines.yaml 的 regen_cmd。</div>')

    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>主模型回测表现 — {meta['signal']} / {meta['window']}</title>
{CHART_CDN}<style>{_CSS}</style></head><body><div class="wrap">
<h1>主模型回测表现 — {meta['signal']}</h1>
<div class="sub">{meta['signal_desc']} ｜ 窗口 {meta['window_label']}（{meta['n_days']} 日）
｜ 指纹 <code>{meta['fingerprint']}</code> ｜ 口径 <b>{meta['profile']}</b>
（成本 {meta['cost']} / 成交价 {meta['execution']} / 资金 {meta['capital']:.0e}）
｜ 生成 {meta['generated']}</div>
{warn}
<div class="banner"><b>怎么读</b>：曲线全部由 <code>scripts.evaluation.caliber</code>
同一套上下文、引擎与成本重跑得到，不是抄登记值；下表每臂都带「登记值 + Δ」，
Δ≠0 说明数据态或信号面板变了。Sharpe 引用**几何**（生产口径），显著性用 NW-t
（t = SR·√T 里的 SR 是**算术**，两列都给了）。基准 = 同池月度重置等权，
段内随价漂移、非日频再平衡。</div>
<div class="kpis">{kpi_html}</div>

<div class="card"><h2>累计净值（起点 = 1）</h2>
<div class="chart"><canvas id="cEq"></canvas></div></div>

<div class="card"><h2>回撤（{ARM_LABEL.get(main_tag, main_tag)}）</h2>
<div class="chart sm"><canvas id="cDd"></canvas></div></div>

<div class="card"><h2>分年收益</h2>
<div class="chart sm"><canvas id="cYr"></canvas></div>
<table>{yhead}{yrows}</table></div>

<div class="card"><h2>指标对比（{meta['window_label']}）</h2>
<table>{head}{''.join(rows)}</table></div>

<div class="card"><h2>月度收益热图（{ARM_LABEL.get(main_tag, main_tag)}，%）</h2>
<table class="hm">{''.join(hm)}</table></div>

<div class="card"><h2>登记基线对照（config/baselines.yaml，含本机跑不了的条目）</h2>
<table>{reg_head}{''.join(reg_rows)}</table></div>

<div class="foot">口径与边界：① 窗口与信号由 caliber 登记决定，换窗口必须
<code>--window</code> 重跑（截断日收益 ≠ 换起跑日）；② 成本 = 项目
<code>Config.costs()</code> 固定 10bp（每单位单边换手的往返），无冲击项；
③ 掩码 = 方案② mask-lag 1（信号 T、掩码 T+1、成交 T+1 open）；
④ 全 A 含退市回补，但**因子面板**里约 4.8% 的股票整列无值（见
reports/factor_dashboard/README.md 边界），策略池因此不含它们；
⑤ 本页只做展示，不是新实验，判改进一律回 <code>caliber --check</code>。</div>
</div>
<script>const D = {json.dumps(payload['chart'], ensure_ascii=False)};
const fmtPct = v => (v*100).toFixed(1) + '%';
const baseOpt = {{responsive:true,maintainAspectRatio:false,
  interaction:{{mode:'index',intersect:false}},
  plugins:{{legend:{{labels:{{boxWidth:12,font:{{size:11}}}}}},
    tooltip:{{callbacks:{{label:c=>c.dataset.label+': '+
      (c.dataset.yAxisID==='y1'?fmtPct(c.parsed.y):Number(c.parsed.y).toFixed(3))}}}}}},
  scales:{{x:{{ticks:{{maxTicksLimit:10,font:{{size:10}}}}}},
    y:{{ticks:{{callback:v=>Number(v).toFixed(2),font:{{size:10}}}}}}}}}};
const eqSeries = D.equity.datasets.map(d=>Object.assign({{}}, d, {{
  borderWidth:1.6, pointRadius:0, tension:.15}}));
new Chart(document.getElementById('cEq'), {{type:'line',
  data:{{labels:D.equity.labels, datasets:eqSeries}}, options:baseOpt}});
new Chart(document.getElementById('cDd'), {{type:'line',
  data:{{labels:D.dd.labels, datasets:[{{label:'回撤', data:D.dd.data,
    borderColor:'#d1242f', backgroundColor:'rgba(209,36,47,.12)',
    fill:true, borderWidth:1.4, pointRadius:0}}]}},
  options:Object.assign({{}}, baseOpt, {{plugins:{{legend:{{display:false}},
    tooltip:{{callbacks:{{label:c=>fmtPct(c.parsed.y)}}}}}},
    scales:{{x:{{ticks:{{maxTicksLimit:10,font:{{size:10}}}}}},
      y:{{ticks:{{callback:v=>fmtPct(v),font:{{size:10}}}}}}}}}})}});
new Chart(document.getElementById('cYr'), {{type:'bar',
  data:{{labels:D.yearly.labels, datasets:[{{label:'策略', data:D.yearly.s,
    backgroundColor:'#c62828'}}, {{label:'基准', data:D.yearly.b,
    backgroundColor:'#9aa4b2'}}, {{label:'超额', data:D.yearly.e,
    backgroundColor:'#2f6feb'}}]}},
  options:Object.assign({{}}, baseOpt, {{plugins:{{legend:{{labels:{{boxWidth:12,
    font:{{size:11}}}}}}, tooltip:{{callbacks:{{label:c=>c.dataset.label+': '+
    fmtPct(c.parsed.y)}}}}}},
    scales:{{x:{{ticks:{{font:{{size:10}}}}}},
      y:{{ticks:{{callback:v=>fmtPct(v),font:{{size:10}}}}}}}}}})}});
</script></body></html>"""


# ---------------------------------------------------------------- 主流程
def build(signal: str, window: str, profile: str, buffer: str,
          execution: str | None, capital: float | None) -> dict:
    prof = cal.get_profile(profile)
    ex = execution or prof.execution
    cap = float(capital if capital is not None else prof.capital)
    ctx = cal.load_context(signal=signal, window=window, execution=ex)
    cost = cal.cost_profile(prof.cost)

    arms_metrics, curves = {}, {}
    arms = cal._arm_strategies(buffer, "")
    for tag, strat in arms:
        res = run_arm(strat, ctx, cost, cap, ex)
        m = cal._metrics(res, cap)
        arms_metrics[tag] = m
        curves[tag] = res.daily_returns.reindex(ctx.pred.index).fillna(0.0)

    bench_dr = build_benchmark(ctx)
    bench_res = type("R", (), {})()          # 基准只需要 _metrics 的两处字段
    bench_res.daily_returns = bench_dr
    bench_res.impact_cost_series = None
    bench_res.turnover_series = pd.Series([np.nan])
    bench_res.cost_series = pd.Series([0.0])
    bench_res.weights_history = pd.DataFrame()
    try:
        bench_m = cal._metrics(bench_res, cap)
    except Exception:                          # noqa: BLE001 —— 基准无持仓统计
        bench_m = {k: float("nan") for k in arms_metrics[next(iter(arms_metrics))]}
        sp = cal.sharpe_pair(bench_dr)
        bench_m.update(sp)
        bench_m["ann"] = float(cal.annual_return(bench_dr))
        bench_m["vol"] = float(cal.annual_volatility(bench_dr))
        bench_m["max_dd"] = float(cal.max_drawdown(bench_dr))
        bench_m["sharpe_t_stat"] = float(cal.nw_tstat(bench_dr.dropna())[0])
    bench_m["n_active"] = float("nan")
    bench_m["turnover"] = float("nan")
    bench_m["cost_bp"] = 0.0

    main_tag = next((t for t in arms_metrics if t.startswith("eqw_buf")),
                    next(iter(arms_metrics)))

    # 净值 / 回撤 / 分年 / 月度
    labels, eq_sets = None, []
    for tag, dr in curves.items():
        eq = eq_curve(dr)
        lb, vals = week_sample(eq)
        if labels is None:
            labels = lb
        eq_sets.append({"label": ARM_LABEL.get(tag, tag), "data": vals,
                        "borderColor": "#2f6feb" if tag == main_tag else "#8b95a5",
                        "borderDash": [] if tag == main_tag else [4, 3]})
    b_eq = eq_curve(bench_dr)
    _, b_vals = week_sample(b_eq)
    eq_sets.append({"label": "等权基准", "data": b_vals, "borderColor": "#c62828",
                    "borderDash": [2, 3]})
    dd = drawdown(eq_curve(curves[main_tag]))
    ddw = dd.resample("W-FRI").last().reindex(pd.to_datetime(labels)).ffill()
    y_strat = yearly(curves[main_tag])
    y_bench = yearly(bench_dr)
    yrs = sorted(set(y_strat) | set(y_bench))

    payload = {
        "meta": {
            "signal": signal, "window": window,
            "signal_desc": cal.SIGNALS.get(signal, {}).get("desc", signal),
            "window_label": f"{ctx.pred.index[0]:%Y-%m-%d}..{ctx.pred.index[-1]:%Y-%m-%d}"
                            f"（{ctx.n_days} 日 / {ctx.n_days/252:.2f} 年）",
            "n_days": ctx.n_days, "fingerprint": ctx.fingerprint,
            "profile": profile, "cost": prof.cost, "execution": ex,
            "capital": cap, "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "registered": {},
        },
        "arms": arms_metrics, "bench": bench_m, "main_arm": main_tag,
        "years": {y: {"s": y_strat.get(y, float("nan")), "b": y_bench.get(y, float("nan")),
                      "e": (y_strat.get(y, np.nan) - y_bench.get(y, np.nan))}
                  for y in yrs},
        "monthly": monthly(curves[main_tag]),
        "registry": registered_arms(signal, window),
        "chart": {
            "equity": {"labels": labels, "datasets": eq_sets},
            "dd": {"labels": [d.strftime("%Y-%m-%d") for d in ddw.index],
                   "data": [round(float(v), 4) for v in ddw]},
            "yearly": {"labels": yrs, "s": [round(y_strat.get(y, 0) * 100, 2) / 100 for y in yrs],
                       "b": [round(y_bench.get(y, 0) * 100, 2) / 100 for y in yrs],
                       "e": [round((y_strat.get(y, 0) - y_bench.get(y, 0)) * 100, 2) / 100
                             for y in yrs]},
        },
    }
    # 登记值（同 signal+window 那条的 production 各臂）用于表内 Δ
    for r in payload["registry"]:
        if r["same_as_run"]:
            for tag in arms_metrics:
                reg = (r.get("production_all") or {}).get(tag)
                if reg and "ann" in reg:
                    payload["meta"]["registered"][tag] = reg
    # 生产定版不是本次跑的那条 ⇒ 页面顶部必须挂警告（别让读者把代理线当定版）
    prod = [r["id"] for r in payload["registry"]
            if r.get("status") == "production" and not r["same_as_run"]]
    payload["missing_production"] = prod[0] if prod else None
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signal", default="ens_h1h5", choices=sorted(cal.SIGNALS))
    ap.add_argument("--window", default="prod", choices=sorted(cal.WINDOWS))
    ap.add_argument("--profile", default="production", choices=sorted(cal.PROFILES))
    ap.add_argument("--buffer", default="0.15:0.40")
    ap.add_argument("--execution", default=None, choices=list(cal.EXECUTIONS))
    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    payload = build(args.signal, args.window, args.profile, args.buffer,
                    args.execution, args.capital)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.out or f"perf_{args.signal}_{args.window}"
    hp, jp = OUT_DIR / f"{stem}.html", OUT_DIR / f"{stem}.json"
    hp.write_text(render_html(payload), encoding="utf-8")
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str),
                  encoding="utf-8")
    m = payload["arms"][payload["main_arm"]]
    print(f"主臂 {payload['main_arm']}：年化 {m['ann']*100:.2f}%  "
          f"Sharpe {m['sharpe']:.3f} / 算术 {m['sharpe_arith']:.3f}  "
          f"NW-t {m['sharpe_t_stat']:.2f}  回撤 {m['max_dd']*100:.2f}%")
    print(f"产物: {hp}\n      {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
