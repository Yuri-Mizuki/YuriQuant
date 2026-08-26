"""交互式因子检测报告：时间段选择 + 来源筛选 + 指标排序 + 因子详情。

数据预计算（脚本端）：
- 对每个因子从 evals 读逐日 ic / equity_ls_M / equity_lo_M
- 生成**月度指标**（monthly ic 均值、LS/LO 月收益、LS/LO 净值终点）+ **日度序列**
  以 JSON 内嵌到 HTML —— 前端按用户选的时间段切片重算汇总指标，无需后端。

前端交互（Chart.js + 原生 JS）：
- 时间段选择（起始/结束日期，下拉预设：全部/近1年/近2年/2024起/自定义）
- 来源筛选（alpha101/158/191/360/gp/model/全部）
- 可排序表格：IC / ICIR / NW-t / LS月收益 / LO月收益 / LS Sharpe / 换手
- 点击行 -> 详情面板：净值曲线（LS/LO）+ 月度 IC 柱状图 + 指标卡 + 公式

用法:
    python scripts/factor_explorer_report.py [--dataset hs300_2022_2025] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("factor_explorer")

from research.factor_library import FactorLibrary  # noqa: E402


def family_of(source) -> str:
    s = str(source) if pd.notna(source) else "unknown"
    for pre in ("alpha101", "alpha158", "alpha191", "alpha360", "gp", "model"):
        if s.startswith(pre):
            return pre
    return s.split(":")[0]


def monthly_series(ic: pd.Series) -> dict:
    """IC 按月的均值序列（dict month_str -> float）。"""
    s = ic.dropna()
    if len(s) == 0:
        return {}
    m = s.groupby(s.index.to_period("M")).mean()
    return {str(p): round(float(v), 4) for p, v in m.items()}


def monthly_nav(equity: pd.Series) -> dict:
    """分层净值按月的月末值（dict month_str -> float），供前端画净值曲线。"""
    s = equity.dropna()
    if len(s) == 0:
        return {}
    last = s.groupby(s.index.to_period("M")).last()
    return {str(p): round(float(v), 4) for p, v in last.items()}


def build_factor_data(lib: FactorLibrary, reg: pd.DataFrame) -> list[dict]:
    """每个因子 -> {name, family, formula, ic_series, ls_nav, lo_nav, full_metrics}。"""
    out = []
    for _, r in reg.iterrows():
        name = r["name"]
        ev = lib._load_eval(name)
        if ev is None or len(ev) == 0:
            continue
        ic = ev["ic"].dropna() if "ic" in ev.columns else pd.Series(dtype=float)
        ls_eq = ev["equity_ls_M"].dropna() if "equity_ls_M" in ev.columns else pd.Series(dtype=float)
        lo_eq = ev["equity_lo_M"].dropna() if "equity_lo_M" in ev.columns else pd.Series(dtype=float)

        # 全期静态指标（注册表已有，直接带上，前端选全期时用）
        n = len(ic)
        ic_mean = float(ic.mean()) if n else np.nan
        ic_std = float(ic.std()) if n > 1 else np.nan
        icir = ic_mean / ic_std * np.sqrt(12) if ic_std and ic_std > 0 else np.nan  # 月频年化
        ls_ret = float(ls_eq.iloc[-1] - 1) if len(ls_eq) else np.nan
        lo_ret = float(lo_eq.iloc[-1] - 1) if len(lo_eq) else np.nan
        ls_sharpe = None
        if "dret_ls_M" in ev.columns:
            d = ev["dret_ls_M"].dropna()
            if len(d) > 1 and d.std() > 0:
                ls_sharpe = float(d.mean() / d.std() * np.sqrt(12))

        out.append({
            "name": name,
            "family": family_of(r.get("source")),
            "formula": str(r.get("formula", ""))[:300],
            "ic_series": monthly_series(ic),          # 月度 IC
            "ls_nav": monthly_nav(ls_eq),             # 月度多空净值
            "lo_nav": monthly_nav(lo_eq),             # 月度多头净值
            "m_full": {
                "ic": round(ic_mean, 4) if not np.isnan(ic_mean) else None,
                "icir": round(icir, 3) if not np.isnan(icir) else None,
                "t_nw": round(float(r.get("t_stat_nw", np.nan)), 3) if pd.notna(r.get("t_stat_nw")) else None,
                "ls_ret": round(ls_ret, 4) if not np.isnan(ls_ret) else None,
                "lo_ret": round(lo_ret, 4) if not np.isnan(lo_ret) else None,
                "ls_sharpe": round(ls_sharpe, 3) if ls_sharpe is not None else None,
                "win": round(float(r.get("ic_win_rate", np.nan)), 3) if pd.notna(r.get("ic_win_rate")) else None,
                "turn": round(float(r.get("avg_turnover_ls_M", np.nan)), 4) if pd.notna(r.get("avg_turnover_ls_M")) else None,
                "sig": bool(r.get("significant", False)),
            },
        })
    return out


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>YuriQuant 交互式因子检测报告 — {dataset}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,'PingFang SC',sans-serif; background:#f5f7fa; color:#1a1a2e; line-height:1.55; }}
.container {{ max-width:1280px; margin:0 auto; padding:20px 16px; }}
.header {{ background:linear-gradient(135deg,#1a1a2e,#16213e); color:#fff; padding:24px; border-radius:12px; margin-bottom:16px; }}
.header h1 {{ font-size:19px; margin-bottom:4px; }}
.header .meta {{ font-size:12px; opacity:.85; }}
.controls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; background:#fff; padding:14px 16px; border-radius:10px; margin-bottom:14px; box-shadow:0 1px 4px rgba(0,0,0,.06); }}
.controls label {{ font-size:13px; color:#555; }}
.controls select, .controls input {{ padding:6px 10px; border:1px solid #ddd; border-radius:6px; font-size:13px; background:#fff; }}
.btn {{ padding:7px 14px; border:1px solid #16213e; background:#16213e; color:#fff; border-radius:6px; font-size:13px; cursor:pointer; }}
.btn:hover {{ opacity:.9; }}
.card {{ background:#fff; border-radius:10px; padding:16px; margin-bottom:14px; box-shadow:0 1px 4px rgba(0,0,0,.06); }}
.card h2 {{ font-size:15px; color:#16213e; margin-bottom:10px; border-bottom:2px solid #e8e8e8; padding-bottom:6px; }}
table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
th {{ padding:7px 9px; background:#f0f2f5; text-align:right; border-bottom:2px solid #ddd; position:sticky; top:0; cursor:pointer; user-select:none; white-space:nowrap; }}
th:first-child, td:first-child {{ text-align:left; }}
td {{ padding:6px 9px; border-bottom:1px solid #f0f0f0; text-align:right; white-space:nowrap; }}
tr:hover {{ background:#f8f9ff; cursor:pointer; }}
.scroll {{ max-height:520px; overflow-y:auto; border:1px solid #eee; border-radius:8px; }}
.sig {{ color:#A32D2D; font-weight:600; }}
.fam-badge {{ display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; color:#fff; }}
.chart-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.chart-box {{ position:relative; height:260px; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px; margin:12px 0; }}
.metric {{ background:#f8f9fa; border-radius:8px; padding:10px; }}
.metric .l {{ font-size:11px; color:#888; }} .metric .v {{ font-size:17px; font-weight:600; }}
.formula {{ font-family:monospace; font-size:11.5px; color:#555; background:#f8f9fa; padding:8px 12px; border-radius:6px; margin-top:8px; overflow-x:auto; }}
.hidden {{ display:none; }}
#detail {{ position:sticky; top:8px; }}
.empty {{ padding:30px; text-align:center; color:#999; font-size:14px; }}
</style></head><body><div class="container">
<div class="header">
<h1>YuriQuant 交互式因子检测报告 — {dataset}</h1>
<div class="meta">{n_factors} 个因子 | 面板 2022-01 ~ 2026-08 | 选择时间段后自动重算 IC/IR/收益，点击因子查看详情</div>
</div>

<div class="controls">
<label>时间段：
<select id="preset">
<option value="all">全部 (2022-01~2026-08)</option>
<option value="y1">近 1 年</option>
<option value="y2">近 2 年</option>
<option value="2024">2024 起</option>
<option value="custom">自定义</option>
</select></label>
<label>从 <input type="month" id="dstart"></label>
<label>至 <input type="month" id="dend"></label>
<label>来源：
<select id="ffamily">
<option value="all">全部来源</option>
<option>alpha101</option><option>alpha158</option><option>alpha191</option><option>alpha360</option><option>gp</option><option>model</option>
</select></label>
<label>搜索 <input type="text" id="fsearch" placeholder="因子名..." style="width:160px"></label>
<button class="btn" onclick="applyFilter()">筛选</button>
<span id="rcount" style="font-size:12px;color:#888;"></span>
</div>

<div class="card"><h2>因子检验表（点击表头排序，点击行看详情）</h2>
<div class="scroll"><table id="ftable">
<thead><tr>
<th data-k="name">因子名</th><th data-k="family">来源</th><th data-k="ic">IC</th><th data-k="icir">ICIR</th>
<th data-k="t_nw">NW-t</th><th data-k="ls_ret">多空收益</th><th data-k="ls_sharpe">多空Sharpe</th>
<th data-k="lo_ret">多头收益</th><th data-k="win">IC胜率</th><th data-k="turn">换手</th><th data-k="sig">显著</th>
</tr></thead>
<tbody></tbody></table></div></div>

<div class="card hidden" id="detail"><h2 id="dtitle">因子详情</h2>
<div class="metrics" id="dmetrics"></div>
<div class="chart-grid">
<div class="chart-box"><canvas id="cNav" role="img" aria-label="多空多头净值曲线">净值曲线</canvas></div>
<div class="chart-box"><canvas id="cIc" role="img" aria-label="月度IC柱状图">月度IC</canvas></div>
</div>
<div class="formula" id="dformula"></div>
</div>
</div>

<script>
const DATA = {factor_data};
const MONTHS = {months};
const FAM_COLORS = {fam_colors};

// ---------- 状态 ----------
let current = {{ sortKey: 'ic', asc: false }};
const navChart = null, icChart = null;
let chartNav = null, chartIc = null;

// ---------- 工具 ----------
const fmt = (v, d=3) => v === null || v === undefined ? '—' : (typeof v === 'number' ? v.toFixed(d) : v);
const fmtPct = (v, d=1) => v === null || v === undefined ? '—' : (v*100).toFixed(d) + '%';

// 选时间段内重算指标：对 month 序列切片
function rangeMetrics(f) {{
  const s = current.start, e = current.end;
  const icMs = Object.entries(f.ic_series).filter(([m]) => m >= s && m <= e).map(([,v]) => v);
  const lsEnd = Object.entries(f.ls_nav).filter(([m]) => m >= s && m <= e);
  const loEnd = Object.entries(f.lo_nav).filter(([m]) => m >= s && m <= e);
  const lsStart = Object.entries(f.ls_nav).filter(([m]) => m < s);
  const loStart = Object.entries(f.lo_nav).filter(([m]) => m < s);
  const n = icMs.length;
  const icMean = n ? icMs.reduce((a,b)=>a+b,0)/n : null;
  const icStd = n > 1 ? Math.sqrt(icMs.reduce((a,m)=>a+(m-icMean)**2,0)/(n-1)) : 0;
  const icir = (icStd > 0 && icMean !== null) ? icMean/icStd*Math.sqrt(12) : null;
  // 净值：区间起点归一，终点相对起点
  const navFrom = (list) => list.length ? list[0][1] : 1;
  const navTo = (list) => list.length ? list[list.length-1][1] : null;
  const ls0 = navFrom(lsStart.length ? lsStart : lsEnd);
  const ls1 = navTo(lsEnd);
  const lo0 = navFrom(loStart.length ? loStart : loEnd);
  const lo1 = navTo(loEnd);
  const lsRet = ls1 ? ls1/ls0 - 1 : null;
  const loRet = lo1 ? lo1/lo0 - 1 : null;
  // NW-t 近似：月度 IC 自相关稳健（lag=3 简单 NW）
  let tNw = null;
  if (n > 4 && icStd > 0) {{
    const mean = icMean;
    let s2 = 0, sAc = 0;
    for (let i=0;i<n;i++) s2 += (icMs[i]-mean)**2;
    for (let lag=1;lag<=3;lag++) {{
      let ac = 0, c=0;
      for (let i=lag;i<n;i++) {{ ac += (icMs[i]-mean)*(icMs[i-lag]-mean); c++; }}
      if (c) sAc += (1 - lag/(4+1)) * (ac/c);
    }}
    const varNw = (s2 + 2*sAc) / n;
    tNw = varNw > 0 ? mean/Math.sqrt(varNw) : null;
  }}
  return {{ ic: icMean, icir, t_nw: tNw, ls_ret: lsRet, lo_ret: loRet, n }};
}}

// ---------- 表格 ----------
function render() {{
  const tbody = document.querySelector('#ftable tbody');
  const rows = DATA.filter(f => {{
    if (current.family !== 'all' && f.family !== current.family) return false;
    if (current.q && !f.name.toLowerCase().includes(current.q)) return false;
    return true;
  }});
  document.getElementById('rcount').textContent = rows.length + ' 个因子';
  rows.forEach(f => {{
    const m = rangeMetrics(f);
    const sig = f.m_full.sig;
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td><b>${{f.name}}</b></td><td><span class="fam-badge" style="background:${{FAM_COLORS[f.family]}}">${{f.family}}</span></td>` +
      `<td>${{fmt(m.ic)}}</td><td>${{fmt(m.icir)}}</td><td>${{fmt(m.t_nw)}}</td>` +
      `<td>${{fmtPct(m.ls_ret)}}</td><td>${{fmt(f.m_full.ls_sharpe, 2)}}</td>` +
      `<td>${{fmtPct(m.lo_ret)}}</td><td>${{f.m_full.win!==null? (f.m_full.win*100).toFixed(0)+'%':'—'}}</td>` +
      `<td>${{f.m_full.turn!==null? (f.m_full.turn*100).toFixed(2)+'%':'—'}}</td>` +
      `<td class="${{sig?'sig':''}}">${{sig?'显著':'—'}}</td>`;
    tr.onclick = () => showDetail(f);
    tbody.appendChild(tr);
  }});
}}

// ---------- 排序 ----------
document.querySelectorAll('#ftable th').forEach(th => {{
  th.onclick = () => {{
    const k = th.dataset.k;
    if (current.sortKey === k) current.asc = !current.asc; else {{ current.sortKey = k; current.asc = false; }}
    const tbody = document.querySelector('#ftable tbody');
    const rows = [...tbody.rows];
    rows.sort((a,b) => {{
      if (k === 'name' || k === 'family') return current.asc ? a.cells[0].innerText.localeCompare(b.cells[0].innerText) : b.cells[0].innerText.localeCompare(a.cells[0].innerText);
      const av = parseFloat(a.cells[Array.from(th.parentNode.children).indexOf(th)].innerText.replace('%','').replace('—','NaN'));
      const bv = parseFloat(b.cells[Array.from(th.parentNode.children).indexOf(th)].innerText.replace('%','').replace('—','NaN'));
      if (isNaN(av)) return 1; if (isNaN(bv)) return -1;
      return current.asc ? av-bv : bv-av;
    }});
    rows.forEach(r => tbody.appendChild(r));
  }};
}});

// ---------- 详情 ----------
function showDetail(f) {{
  document.getElementById('detail').classList.remove('hidden');
  document.getElementById('dtitle').textContent = f.name + '  —  ' + (current.start) + ' ~ ' + (current.end);
  document.getElementById('dformula').textContent = '公式: ' + f.formula;
  const m = rangeMetrics(f);
  document.getElementById('dmetrics').innerHTML = [
    ['IC', fmt(m.ic,4)], ['ICIR', fmt(m.icir)], ['NW-t', fmt(m.t_nw)],
    ['多空收益', fmtPct(m.ls_ret)], ['多头收益', fmtPct(m.lo_ret)],
    ['样本月数', m.n], ['IC胜率', f.m_full.win!==null?(f.m_full.win*100).toFixed(0)+'%':'—'],
    ['显著', f.m_full.sig?'是':'否']
  ].map(([l,v]) => `<div class="metric"><div class="l">${{l}}</div><div class="v">${{v}}</div></div>`).join('');

  // 净值曲线
  const months = Object.keys(f.ls_nav).filter(mo => mo >= current.start && mo <= current.end);
  const ls = months.map(mo => f.ls_nav[mo] || null);
  const lo = months.map(mo => f.lo_nav[mo] || null);
  const ls0 = Object.entries(f.ls_nav).filter(([mo]) => mo < current.start);
  const lo0 = Object.entries(f.lo_nav).filter(([mo]) => mo < current.start);
  const baseL = ls0.length ? ls0[ls0.length-1][1] : (ls.length ? ls[0] : 1);
  const baseO = lo0.length ? lo0[lo0.length-1][1] : (lo.length ? lo[0] : 1);
  const lsNorm = ls.map(v => v ? v/baseL : null);
  const loNorm = lo.map(v => v ? v/baseO : null);
  if (chartNav) chartNav.destroy();
  chartNav = new Chart(document.getElementById('cNav'), {{
    type: 'line',
    data: {{ labels: months, datasets: [
      {{ label: '多空(LS)', data: lsNorm, borderColor: '#A32D2D', borderWidth: 1.6, pointRadius: 0, tension: .15 }},
      {{ label: '多头(LO)', data: loNorm, borderColor: '#185FA5', borderWidth: 1.6, pointRadius: 0, tension: .15 }}
    ]}},
    options: {{ responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 12, font: {{ size: 11 }} }} }}, title: {{ display: true, text: '分层净值（区间起点=1）', font: {{ size: 13 }} }} }},
      scales: {{ y: {{ title: {{ display: true, text: '累计净值' }} }}, x: {{ ticks: {{ maxTicksLimit: 12, font: {{ size: 10 }} }} }} }} }}
  }});
  // 月度 IC
  const icMs = Object.entries(f.ic_series).filter(([mo]) => mo >= current.start && mo <= current.end);
  if (chartIc) chartIc.destroy();
  chartIc = new Chart(document.getElementById('cIc'), {{
    type: 'bar',
    data: {{ labels: icMs.map(([m]) => m), datasets: [{{ label: '月度IC', data: icMs.map(([,v]) => v),
      backgroundColor: icMs.map(([,v]) => v >= 0 ? 'rgba(163,45,45,.8)' : 'rgba(59,109,17,.8)') }}]}},
    options: {{ responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: '月度 IC', font: {{ size: 13 }} }} }},
      scales: {{ y: {{ title: {{ display: true, text: 'IC' }} }}, x: {{ ticks: {{ maxTicksLimit: 12, font: {{ size: 10 }} }} }} }} }}
  }});
}}

// ---------- 时间段 ----------
function applyFilter() {{
  const p = document.getElementById('preset').value;
  let s = MONTHS[0], e = MONTHS[MONTHS.length-1];
  if (p === 'y1') {{ s = MONTHS[MONTHS.length-13] || s; }}
  else if (p === 'y2') {{ s = MONTHS[MONTHS.length-25] || s; }}
  else if (p === '2024') {{ s = MONTHS.find(m => m >= '2024-01') || s; }}
  else if (p === 'custom') {{
    const a = document.getElementById('dstart').value, b = document.getElementById('dend').value;
    if (a) s = a + '-01'; if (b) e = b + '-01';
  }}
  current.start = s; current.end = e;
  document.getElementById('dstart').value = s.slice(0,7);
  document.getElementById('dend').value = e.slice(0,7);
  current.family = document.getElementById('ffamily').value;
  current.q = document.getElementById('fsearch').value.trim().toLowerCase();
  document.querySelector('#ftable tbody').innerHTML = '';
  render();
  document.getElementById('detail').classList.add('hidden');
}}

document.getElementById('preset').onchange = applyFilter;
document.getElementById('ffamily').onchange = applyFilter;
document.getElementById('fsearch').oninput = applyFilter;

// 初始化
current.start = MONTHS[0]; current.end = MONTHS[MONTHS.length-1];
render();
</script>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="交互式因子检测报告")
    ap.add_argument("--dataset", default="hs300_2022_2025")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lib = FactorLibrary(dataset=args.dataset)
    reg = lib.list_all()
    log.info("因子库 %s: %d 因子", args.dataset, len(reg))

    data = build_factor_data(lib, reg)
    log.info("预计算完成: %d 因子", len(data))

    # 全月份轴
    all_months = sorted({m for f in data for m in f["ic_series"]})
    fam_colors = {
        "alpha101": "#378ADD", "alpha158": "#1D9E75", "alpha191": "#BA7517",
        "alpha360": "#534AB7", "gp": "#D85A30", "model": "#E24B4A", "unknown": "#888780",
    }
    html = HTML_TEMPLATE.format(
        dataset=args.dataset,
        n_factors=len(data),
        factor_data=json.dumps(data, ensure_ascii=False),
        months=json.dumps(all_months),
        fam_colors=json.dumps(fam_colors),
    )
    out = Path(args.out) if args.out else Path("reports") / f"factor_explorer_{args.dataset}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    log.info("报告已生成: %s (%d KB, %d 因子)", out, len(html) // 1024, len(data))


if __name__ == "__main__":
    main()
