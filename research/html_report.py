"""
交互式 HTML 报告生成（Chart.js CDN + 数据内嵌）
================================================

更现代、更直观的因子报告形态：单个自包含 .html 文件，浏览器直接打开
（无需 Excel），图表可 hover / 缩放，对比表可点击表头排序。

设计目标（对照 research/xlsx_report.py 的 Excel 报告）：
1. 单因子视图：指标卡（大数字）+ 净值 / 回撤 / IC / 分层净值交互图
   + 月度收益热力表格 + 指标明细表
2. 多因子视图：顶部可排序对比表 + 叠加净值图；标签页切换每个因子详情
3. 因子库全览：render_sortable_table 输出任意记录表的可排序 HTML

技术方案（A 路线）：
- 图表：Chart.js 4（CDN: cdn.jsdelivr.net），数据以 JSON 内嵌 <script>，
  单文件自包含（图表库需联网加载一次，数据离线可用）
- 颜色：遵循 A 股习惯 —— 涨红 (pos) / 跌绿 (neg)
- 安全：JSON 注入前转义 "</"，防 script 闭合

用法：
    from research.html_report import generate_html_report, render_sortable_table
    generate_html_report(results, factor_summaries, "reports/yuriquant_report.html")
    render_sortable_table(reg, "reports/factor_library_report.html", title="因子库全览")
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult
from backtest.metrics import METRIC_LABELS
from research.metrics_format import is_pos_neg, monthly_returns

# ===========================================================================
# 指标格式化
# ===========================================================================
_PCT_KEYS = {
    "annual_return", "total_return", "annual_volatility", "max_drawdown",
    "win_rate", "ic_win_rate", "avg_turnover", "excess_return",
    "borrow_fee_drag_annual",
}
# 正负着色规则共享自 research.metrics_format（红涨绿跌）
_POS_NEG_KEYS = frozenset()
# 指标卡（详情视图顶部大数字），按展示顺序
_CARD_KEYS = [
    "annual_return", "annual_volatility", "sharpe", "sortino",
    "max_drawdown", "calmar", "win_rate", "avg_turnover",
    "ic_mean", "ir", "avg_margin_usage", "borrow_fee_drag_annual",
]
_CMP_KEYS = [
    "name", "annual_return", "sharpe", "sortino", "max_drawdown",
    "calmar", "win_rate", "avg_turnover", "ic_mean", "ir",
    "avg_margin_usage", "borrow_fee_drag_annual",
]
_METRIC_ALIAS = {
    "ic_mean": "IC 均值", "ir": "IR", "ic_win_rate": "IC 胜率",
    "annual_return": "年化收益", "total_return": "累计收益",
    "annual_volatility": "年化波动", "sharpe": "夏普", "sortino": "索提诺",
    "max_drawdown": "最大回撤", "calmar": "卡玛", "win_rate": "胜率",
    "profit_loss_ratio": "盈亏比", "avg_daily_return": "日均收益",
    "avg_turnover": "平均换手", "avg_long_exposure": "平均多头敞口",
    "avg_short_exposure": "平均空头敞口", "avg_margin_usage": "保证金占用",
    "max_margin_usage": "最大保证金占用", "borrow_fee_total": "借券费总额",
    "borrow_fee_drag_annual": "借券费年化拖累", "n_days": "交易日数",
}


def _fmt(v, key: str = "") -> str:
    """格式化指标值（百分比/数值），NaN 显示 -。"""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "-"
    if isinstance(v, (np.integer, int)):
        return f"{int(v):,}"
    if isinstance(v, float):
        if key in _PCT_KEYS:
            return f"{v * 100:.2f}%"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.4f}"
    return str(v)


def _pos_neg(v, key: str = "") -> str:
    """涨红跌绿：收益/绩效类指标按正负着色（规则共享 metrics_format）。"""
    if is_pos_neg(key) and isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
        return "pos" if v >= 0 else "neg"
    return ""


# ===========================================================================
# 序列数据构造
# ===========================================================================
def _series_pairs(s: pd.Series) -> list[list]:
    s = s.dropna()
    return [[str(idx.date()), float(v)] for idx, v in s.items()]


def _monthly_table(daily_returns: pd.Series) -> dict:
    """年 → {月: 收益} 热力表格数据（月度复利共享 metrics_format.monthly_returns）。

    历史教训：双重加 1 导致一个月收益 ≈ 2^n 天文数字（2026-08-05 修复）。
    """
    if daily_returns is None or len(daily_returns) == 0:
        return {}
    monthly = monthly_returns(daily_returns)
    out: dict = {}
    for dt, v in monthly.items():
        out.setdefault(int(dt.year), {})[int(dt.month)] = round(float(v), 6)
    return out


def _month_cell_style(v: float) -> str:
    """月度收益单元格背景（红涨绿跌，色深随幅度）。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "background:#fafafa;color:#bbb"
    amp = min(abs(v), 0.12) / 0.12
    if v >= 0:
        return f"background:rgba(198,40,40,{0.10 + 0.55 * amp:.2f});color:#fff"
    return f"background:rgba(46,125,50,{0.10 + 0.55 * amp:.2f});color:#fff"


# ===========================================================================
# HTML 模板（CSS / JS）
# ===========================================================================
# 统一基础样式：多套报告（html_report / report_pipeline / 各脚本）共用，
# 单一实现。特定报告的个性化主题在 page(css=...) 覆盖。
BASE_CSS = """
body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;padding:26px 30px;background:#f6f7f9;color:#1f2430}
h1{font-size:21px;margin:0 0 4px;font-weight:600}
.sub{color:#6b7280;font-size:13px;margin-bottom:18px}
.card{background:#fff;border-radius:12px;padding:16px 20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
h2{font-size:15px;margin:0 0 12px;font-weight:600;color:#111827}
h2.section{font-size:17px;margin:28px 0 10px;font-weight:600;color:#111827;border-bottom:2px solid #e5e7eb;padding-bottom:6px}
h3{font-size:13px;margin:14px 0 8px;font-weight:600;color:#374151}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:2px}
.stat{background:#fff;border:1px solid #eef0f3;border-radius:10px;padding:10px 14px;min-width:96px}
.stat b{font-size:18px;display:block;font-weight:600;margin-top:2px}
.stat span{color:#8a8f98;font-size:12px}
.pos{color:#c62828}.neg{color:#2e7d32}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;color:#6b7280;font-weight:500;padding:7px 10px;border-bottom:1px solid #e5e7eb;cursor:pointer;user-select:none;white-space:nowrap}
th:hover{color:#111827}
td{padding:6px 10px;border-bottom:1px solid #f3f4f6;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:hover td{background:#fafbfc}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.tab{padding:6px 14px;border-radius:999px;background:#eef0f3;cursor:pointer;font-size:13px;border:1px solid transparent;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tab.active{background:#1f4e79;color:#fff}
.detail{display:none}.detail.active{display:block}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:6px 20px}
.month-grid td{text-align:center;font-size:12px;padding:4px 6px;border-radius:4px}
.month-grid th{font-size:11.5px;text-align:center}
canvas{max-height:300px}
.mono{font-family:Consolas,monospace;font-size:12px;color:#555}
.note{color:#8a8f98;font-size:12px;margin-top:6px}
code{font-family:Consolas,monospace;font-size:12px;color:#555}
"""

# 兼容别名（旧版内部以 _CSS 引用）
_CSS = BASE_CSS

# 模板中的 __DATA__ 由 Python 注入；__EQ__/__DD__/__IC__/__LAYERS__ 为
# 每因子动态 datasets 表达式（f = DATA.factors[idx]，由 renderFactor 传入）。
_JS = r"""
const DATA = __DATA__;
const fmtPct = v => (v*100).toFixed(2) + '%';
const fmtNum = v => Math.abs(v) >= 1000 ? v.toLocaleString() : (+v).toFixed(4);
const scaleOpts = {
  responsive: true, maintainAspectRatio: true,
  interaction: { mode: 'index', intersect: false },
  plugins: { legend: { labels: { boxWidth: 12, font: { size: 11 } } } },
};
const dateScale = {
  x: { type: 'time', time: { unit: 'month', tooltipFormat: 'yyyy-MM-dd' }, grid: { display: false } },
  y: { ticks: { callback: v => (+v).toFixed(2) } },
};
const charts = {};
function makeChart(id, cfg) {
  const el = document.getElementById(id);
  if (!el) return;
  charts[id] = new Chart(el, cfg);
}
function renderFactor(f, idx) {
  makeChart('eq_'+idx, { type:'line', data:{datasets: __EQ__}, options: { ...scaleOpts, scales: { ...dateScale } } });
  makeChart('dd_'+idx, { type:'line', data:{datasets: __DD__}, options: { ...scaleOpts, scales: { ...dateScale, y: { ticks: { callback: v => (v*100).toFixed(1)+'%' } } } } });
  if (f.ic && f.ic.length) {
    makeChart('ic_'+idx, { data:{datasets: __IC__}, options: {
      ...scaleOpts,
      scales: { x: { type:'time', time:{unit:'month'}, grid:{display:false} },
        y: { position:'left', ticks:{callback:v=>v.toFixed(3)} },
        y1: { position:'right', grid:{display:false}, ticks:{callback:v=>v.toFixed(3)} } } } });
  }
  if (f.layers && Object.keys(f.layers).length) {
    makeChart('ly_'+idx, { type:'line', data:{datasets: __LAYERS__}, options: { ...scaleOpts, scales: { ...dateScale } } });
  }
}
function renderCmp() {
  const tbody = document.getElementById('cmp-body');
  if (!tbody || !DATA.compare.length) return;
  const keys = DATA.compareKeys;
  tbody.innerHTML = DATA.compare.map(r =>
    '<tr>' + keys.map(k => {
      const v = r[k];
      if (v === null || v === undefined) return '<td>-</td>';
      const cls = (typeof v === 'number' && DATA.colorKeys.includes(k)) ? (v >= 0 ? 'pos' : 'neg') : '';
      let txt = v;
      if (typeof v === 'number') txt = DATA.pctKeys.includes(k) ? fmtPct(v) : fmtNum(v);
      return '<td class="'+cls+'">'+txt+'</td>';
    }).join('') + '</tr>'
  ).join('');
  makeChart('cmp-equity', { type:'line', data:{datasets: DATA.cmpEquity}, options: { ...scaleOpts, scales: { ...dateScale } } });
}
document.addEventListener('DOMContentLoaded', () => {
  DATA.factors.forEach((f, i) => renderFactor(f, i));
  renderCmp();
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach(t => t.addEventListener('click', () => {
    tabs.forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.detail').forEach(d => d.classList.remove('active'));
    t.classList.add('active');
    document.getElementById(t.dataset.target).classList.add('active');
    Object.values(charts).forEach(c => c.resize());
  }));
  document.querySelectorAll('#cmp-head th').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.k;
      const asc = th.dataset.asc !== '1';
      th.dataset.asc = asc ? '1' : '0';
      const tbody = document.getElementById('cmp-body');
      const rows = Array.from(tbody.rows);
      const i = Array.from(th.parentNode.children).indexOf(th);
      rows.sort((a, b) => {
        const va = a.cells[i].textContent.replace(/[%,]/g,'');
        const vb = b.cells[i].textContent.replace(/[%,]/g,'');
        let x = parseFloat(va), y = parseFloat(vb);
        if (isNaN(x)) { x = va; } if (isNaN(y)) { y = vb; }
        if (typeof x === 'number' && typeof y === 'number') return asc ? x - y : y - x;
        return asc ? String(x).localeCompare(String(y)) : String(y).localeCompare(String(x));
      });
      rows.forEach(r => tbody.appendChild(r));
    });
  });
});
"""


def _js_datasets_expressions() -> dict[str, str]:
    """每因子图表 datasets 的动态构造表达式（在 JS 中引用 f = DATA.factors[idx]）。"""
    return {
        "__EQ__": (
            "[{label:f.name,data:f.equity,borderColor:'#1f4e79',"
            "backgroundColor:'rgba(31,78,121,.08)',fill:true,borderWidth:1.5,pointRadius:0,tension:.1}]"
        ),
        "__DD__": (
            "[{label:'回撤',data:f.drawdown,borderColor:'#c62828',"
            "backgroundColor:'rgba(198,40,40,.25)',fill:true,borderWidth:1.2,pointRadius:0}]"
        ),
        "__IC__": (
            "[{label:'日IC',data:f.ic,type:'bar',"
            "backgroundColor:f.ic.map(p=>p[1]>=0?'#c62828':'#2e7d32'),yAxisID:'y',barPercentage:.9},"
            "{label:'累计IC',data:f.ic_cum,type:'line',borderColor:'#185fa5',borderWidth:1.5,pointRadius:0,yAxisID:'y1'}]"
        ),
        "__LAYERS__": (
            "Object.entries(f.layers||{}).map(([k,s],i)=>("
            "{label:k,data:s,borderColor:['#1f4e79','#d85a30','#1d9e75','#854f0b','#993556','#534ab7'][i%6],"
            "borderWidth:1.3,pointRadius:0,fill:false,tension:.1}))"
        ),
    }


# ===========================================================================
# 公共基座：页面外壳 / 表格渲染 / Chart.js / SVG sparkline / 图片内嵌
# （多套报告共享的底层原语，统一以本模块为唯一实现）
# ===========================================================================
def page(
    title: str,
    meta: str = "",
    body: str = "",
    css: str | None = None,
    header: str | None = None,
    head_extra: str = "",
    scripts: str = "",
    footer: str = "",
) -> str:
    """自包含 HTML 页面外壳（统一 doctype/head/标题/样式/脚本块）。

    - css: 页面样式（默认 BASE_CSS；个性化主题可传入整段 CSS 覆盖）
    - header: 页首 HTML（默认 ``<h1>标题 + <div class=sub>meta</div>``；可传自定义块或空串）
    - head_extra: <head> 内追加标签（如 Chart.js CDN <script src>）
    - scripts: </body> 前的原始 <script>...</script> 块（可多个）
    """
    css = css or BASE_CSS
    header = header if header is not None else f"<h1>{title}</h1>\n<div class=\"sub\">{meta}</div>"
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{head_extra}
<style>{css}</style>
</head>
<body>
{header}
{body}
{scripts}
{footer}
</body></html>"""


# 统一的表头排序处理（thead/tbody 分离，JS 只排 tbody 行）
_SORT_HANDLER = """
  document.querySelectorAll('th').forEach(th => {
    th.addEventListener('click', () => {
      const tbody = th.closest('table').querySelector('tbody');
      if (!tbody) return;
      const rows = Array.from(tbody.rows);
      const i = Array.from(th.parentNode.children).indexOf(th);
      const asc = th.dataset.asc !== '1';
      th.dataset.asc = asc ? '1' : '0';
      rows.sort((a, b) => {
        const va = a.cells[i].textContent.replace(/[%,]/g,'');
        const vb = b.cells[i].textContent.replace(/[%,]/g,'');
        let x = parseFloat(va), y = parseFloat(vb);
        if (isNaN(x)) { x = va; } if (isNaN(y)) { y = vb; }
        if (typeof x === 'number' && typeof y === 'number') return asc ? x - y : y - x;
        return asc ? String(x).localeCompare(String(y)) : String(y).localeCompare(String(x));
      });
      rows.forEach(r => tbody.appendChild(r));
    });
  });
"""

SORT_JS = f"document.addEventListener('DOMContentLoaded', () => {{{_SORT_HANDLER}\n}});"


def base_js(charts_js: str = "") -> str:
    """统一页面 JS：定义 makeChart + 表头排序，并执行传入的图表初始化代码。"""
    return (f"function makeChart(id, cfg) {{\n"
            f"  const el = document.getElementById(id);\n"
            f"  if (!el) return;\n"
            f"  new Chart(el, cfg);\n"
            f"}}\n"
            f"document.addEventListener('DOMContentLoaded', () => {{{_SORT_HANDLER}\n"
            f"  {charts_js}\n"
            f"}});")


def chart_js_line(canvas_id: str, datasets: list, y_fmt: str = "value", title: str = "") -> str:
    """Chart.js 折线图初始化 JS 代码块（统一折线图配置，时间轴 x）。

    datasets: [{label, data: [[x, y], ...], borderColor, ...}]
    """
    cfg = {
        "type": "line",
        "data": {"datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"labels": {"boxWidth": 12, "font": {"size": 11}}}},
            "scales": {
                "x": {"type": "time", "time": {"unit": "month"}, "grid": {"display": False}},
                "y": {"ticks": {"callback": f"v=>v.toFixed(2)"}},
            },
        },
    }
    cfg_json = json.dumps(cfg, ensure_ascii=False, separators=(",", ":"))
    return f"makeChart('{canvas_id}', {cfg_json});\n"


def render_table(
    df: pd.DataFrame,
    pct_cols: list[str] | None = None,
    color_cols: list[str] | None = None,
    max_rows: int | None = None,
) -> str:
    """DataFrame → 可排序 HTML 表（thead/tbody 分离 + 表头 data-k 供排序 JS）。

    统一 render_sortable_table 与 report_pipeline._df_to_html_table 的渲染逻辑。
    """
    pct_cols = set(pct_cols or [])
    color_cols = set(color_cols or [])
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    head = "".join(f"<th data-k='{c}'>{c}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            cls, txt = "", "-"
            if isinstance(v, (int, float, np.integer, np.floating)) and not (
                isinstance(v, float) and (np.isnan(v) or np.isinf(v))
            ):
                if c in pct_cols:
                    txt = f"{float(v) * 100:.2f}%"
                elif isinstance(v, (np.integer, int)):
                    txt = f"{int(v):,}"
                else:
                    txt = f"{float(v):.4f}"
                if c in color_cols:
                    cls = "pos" if float(v) >= 0 else "neg"
            elif isinstance(v, str):
                txt = v
            cells.append(f"<td class='{cls}'>{txt}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def svg_sparkline(series, width: int = 150, height: int = 28,
                  up_color: str = "#0E6E5C", down_color: str = "#B3402A") -> str:
    """inline SVG sparkline（自包含、无外部依赖；末值≥首值用升色，否则降色）。"""
    if series is None or len(series) < 2:
        return ""
    vals = np.asarray(series.values if hasattr(series, "values") else series, dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max())
    rng = (vmax - vmin) or 1.0
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = i * (width - 2) / (n - 1) + 1
        y = height - 2 - (v - vmin) / rng * (height - 4)
        pts.append(f"{x:.1f},{y:.1f}")
    zero_y = height - 2 - (0 - vmin) / rng * (height - 4)
    zero_y = max(2, min(height - 2, zero_y))
    up = vals[-1] >= vals[0]
    color = up_color if up else down_color
    return (
        f'<svg class="spark" width="{width}" height="{height}">'
        f'<line x1="0" y1="{zero_y:.1f}" x2="{width}" y2="{zero_y:.1f}" '
        f'stroke="#D5D4CA" stroke-width="1" stroke-dasharray="2,2"/>'
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
        f'stroke-width="1.4"/></svg>'
    )


def svg_sparkline_monthly(series, width: int = 150, height: int = 28) -> str:
    """月频 IC 趋势 sparkline：先 resample 月均 IC 再画线（噪音更小、全库不膨胀）。"""
    if series is None or len(series) < 2:
        return ""
    monthly = series.resample("ME").mean().dropna()
    if len(monthly) < 2:
        return svg_sparkline(series, width, height)
    return svg_sparkline(monthly, width, height)


def embed_image_b64(fig) -> str:
    """matplotlib 图 → base64 PNG data URI（报告内嵌图统一入口）。

    实现复用 research.factor_report._fig_to_b64（单一实现，避免复制）。
    """
    from research.factor_report import _fig_to_b64

    return _fig_to_b64(fig)


# ===========================================================================
# 数据准备
# ===========================================================================
def _build_factor_dict(name: str, result: BacktestResult,
                       summary: dict | None = None) -> dict:
    metrics = dict(result.metrics())
    equity = _series_pairs(result.equity_curve)
    cum = (1 + result.daily_returns).cumprod()
    drawdown = _series_pairs((cum - cum.cummax()) / cum.cummax())
    ic = _series_pairs(summary["ic_series"]) if summary and "ic_series" in summary else []
    ic_cum = []
    if ic:
        acc = 0.0
        for d, v in ic:
            acc += v
            ic_cum.append([d, round(acc, 6)])
    layers: dict = {}
    if summary and summary.get("layer_nav") is not None:
        try:
            ln = summary["layer_nav"]
            for col in ln.columns:
                layers[str(col)] = _series_pairs(ln[col])
        except Exception:
            layers = {}
    return {
        "name": name,
        "metrics": metrics,
        "equity": equity,
        "drawdown": drawdown,
        "ic": ic,
        "ic_cum": ic_cum,
        "layers": layers,
        "monthly_html": _monthly_html(_monthly_table(result.daily_returns)),
        "metrics_html": _metrics_detail_html(metrics),
    }


def _metric_card_html(key: str, v) -> str:
    label = _METRIC_ALIAS.get(key, METRIC_LABELS.get(key, key))
    return f'<div class="stat"><span>{label}</span><b class="{_pos_neg(v, key)}">{_fmt(v, key)}</b></div>'


def _metrics_detail_html(metrics: dict) -> str:
    rows = []
    for k, v in metrics.items():
        label = _METRIC_ALIAS.get(k, METRIC_LABELS.get(k, k))
        rows.append(f"<tr><td style='color:#6b7280'>{label}</td><td class='{_pos_neg(v, k)}'>{_fmt(v, k)}</td></tr>")
    return f"<table>{''.join(rows)}</table>"


def _monthly_html(monthly: dict) -> str:
    if not monthly:
        return "<div class='note'>无月度数据</div>"
    years = sorted(monthly)
    head = "<tr><th>年</th>" + "".join(f"<th>{m}月</th>" for m in range(1, 13)) + "<th>全年</th></tr>"
    body = []
    for y in years:
        cells, yr_vals = [], []
        for m in range(1, 13):
            v = monthly[y].get(m)
            if v is not None:
                yr_vals.append(v)
                cells.append(f"<td style='{_month_cell_style(v)}'>{v * 100:+.1f}%</td>")
            else:
                cells.append("<td style='background:#fafafa'></td>")
        yr_sum = (1 + pd.Series(yr_vals)).prod() - 1 if yr_vals else 0.0
        cells.append(f"<td style='font-weight:600'>{yr_sum * 100:+.1f}%</td>")
        body.append(f"<tr><td style='color:#6b7280'>{y}</td>{''.join(cells)}</tr>")
    return f"<table class='month-grid'>{head}{''.join(body)}</table>"


def _comparison_html(n_factors: int) -> str:
    if n_factors < 2:
        return ""
    labels = ["因子"] + [METRIC_LABELS.get(k, k).split(" ")[0] for k in _CMP_KEYS[1:]]
    head = "".join(f"<th data-k='{k}'>{l}</th>" for k, l in zip(_CMP_KEYS, labels))
    return f"""
<div class="card">
  <h2>因子对比（点击表头排序）</h2>
  <table><thead id="cmp-head"><tr>{head}</tr></thead><tbody id="cmp-body"></tbody></table>
  <h3>净值曲线叠加</h3>
  <canvas id="cmp-equity"></canvas>
</div>
"""


# ===========================================================================
# 公共入口
# ===========================================================================
def generate_html_report(
    results: dict[str, BacktestResult],
    factor_summaries: dict[str, dict] | None = None,
    output_path: str | Path = "reports/yuriquant_report.html",
    title: str = "YuriQuant 因子回测报告",
    meta: str | None = None,
) -> Path:
    """生成交互式 HTML 因子回测报告。

    Args:
        results: {factor_name: BacktestResult}
        factor_summaries: {factor_name: factor_summary dict}（可选，提供 IC/分层）
        output_path: 输出 .html 路径
        title: 报告标题
        meta: 副标题/元信息（如参数说明）
    Returns:
        Path to the html file.
    """
    summaries = factor_summaries or {}
    factors = [_build_factor_dict(name, res, summaries.get(name))
               for name, res in results.items()]
    if not factors:
        raise ValueError("results 为空，无法生成 HTML 报告")

    tabs_html = ""
    if len(factors) > 1:
        tabs_html = "<div class='tabs'>" + "".join(
            f"<div class='tab{' active' if i == 0 else ''}' data-target='f{i}'>{f['name']}</div>"
            for i, f in enumerate(factors)
        ) + "</div>"

    details = []
    for i, f in enumerate(factors):
        cards = "".join(_metric_card_html(k, f["metrics"].get(k))
                        for k in _CARD_KEYS if f["metrics"].get(k) is not None)
        blocks = (f"<div><h3>净值曲线</h3><canvas id='eq_{i}'></canvas></div>"
                  f"<div><h3>回撤</h3><canvas id='dd_{i}'></canvas></div>")
        if f["ic"]:
            blocks += f"<div><h3>IC 序列</h3><canvas id='ic_{i}'></canvas></div>"
        if f["layers"]:
            blocks += f"<div><h3>分层净值</h3><canvas id='ly_{i}'></canvas></div>"
        details.append(
            f'<div class="card detail{" active" if i == 0 else ""}" id="f{i}">'
            f'<h2>{f["name"]}</h2><div class="stats">{cards}</div>'
            f'<div class="grid2">{blocks}</div>'
            f'<h3>月度收益（红涨绿跌）</h3>{f["monthly_html"]}'
            f'<h3>指标明细</h3>{f["metrics_html"]}'
            f'</div>'
        )

    cmp_equity = [
        {"label": f["name"], "data": f["equity"], "borderWidth": 1.3,
         "pointRadius": 0, "fill": False, "tension": .1}
        for f in factors
    ]
    data = {
        "factors": factors,
        "compare": [
            {"name": f["name"], **{k: f["metrics"].get(k) for k in _CMP_KEYS[1:]}}
            for f in factors
        ],
        "compareKeys": _CMP_KEYS,
        "pctKeys": [k for k in _CMP_KEYS if k in _PCT_KEYS],
        "colorKeys": [k for k in _CMP_KEYS if k in _POS_NEG_KEYS or k == "name"],
        "cmpEquity": cmp_equity,
    }
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    js = _JS.replace("__DATA__", data_json)
    for token, expr in _js_datasets_expressions().items():
        js = js.replace(token, expr)

    body = f"""{_comparison_html(len(factors))}
{tabs_html}
{''.join(details)}"""
    scripts = (
        f"<script>const DATA = {data_json};</script>\n"
        f'<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>\n'
        f"<script>\n{js}\n</script>"
    )
    html = page(
        title,
        meta=meta or '生成时间 ' + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        body=body,
        scripts=scripts,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def render_sortable_table(
    records: pd.DataFrame,
    output_path: str | Path,
    title: str = "数据表",
    meta: str | None = None,
    pct_cols: list[str] | None = None,
    color_cols: list[str] | None = None,
) -> Path:
    """把任意 DataFrame 渲染为可排序 HTML 表（单文件，无图表）。

    用于因子库全览 / walk-forward 汇总等"记录表"型报告。
    """
    body = render_table(records, pct_cols=pct_cols, color_cols=color_cols)
    html = page(
        title,
        meta=meta or f"共 {len(records)} 行 · 点击表头排序",
        body=f'<div class="card">{body}</div>',
        scripts=f"<script>{SORT_JS}</script>",
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
