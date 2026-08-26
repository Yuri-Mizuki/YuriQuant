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


def qcut_rebal(factor: pd.DataFrame, returns: pd.DataFrame, n: int, freq: str,
               monthly_points: bool = True) -> pd.DataFrame | None:
    """调仓日分层回测：每调仓日按因子分 n 层，组内等权，持有至下个调仓日。

    - freq='M': 每月首个交易日调仓；'W': 每周首个交易日调仓。
    - 口径与主回测一致：调仓日 t 收盘后建仓，t+1 起赚收益（未来一期）。
    - monthly_points=True 时返回【月度末点】净值（每层每月 1 个值），
      供前端画图（数据量小 20 倍）；False 返回逐日净值。
    """
    common = factor.dropna(how="all").index.intersection(returns.dropna(how="all").index)
    if len(common) < 60:
        return None
    s = pd.Series(common, index=common)
    rebal = list(s.groupby(s.index.to_period("M" if freq == "M" else "W")).first())
    daily_grp = pd.DataFrame(0.0, index=common, columns=[f"Q{i+1}" for i in range(n)])
    ret_sub = returns.reindex(common)
    for i, t in enumerate(rebal):
        nxt = rebal[i + 1] if i + 1 < len(rebal) else common[-1]
        hold = common[(common > t) & (common <= nxt)]
        if len(hold) == 0:
            continue
        f = factor.loc[t].dropna()
        cc = f.index.intersection(ret_sub.columns)
        if len(cc) < n:
            continue
        f = f[cc]
        try:
            groups = pd.qcut(f, n, labels=False, duplicates="drop")
        except ValueError:
            continue
        for g in range(n):
            mask = groups == g
            if mask.sum() >= 3:
                seg = ret_sub.loc[hold, cc[mask]].mean(axis=1).fillna(0)
                daily_grp.loc[hold, f"Q{g+1}"] = seg.values
    nav = (1 + daily_grp).cumprod()
    if monthly_points:
        last = nav.groupby(nav.index.to_period("M")).last()
        return last
    return nav


def layer_stats_from_nav(nav: pd.DataFrame) -> dict:
    """从分层净值（月度末点）算每层绩效：年化收益 / 年化波动 / Sharpe / 回撤 / 换手代理。"""
    out = []
    for col in nav.columns:
        s = nav[col].dropna()
        if len(s) < 3:
            out.append(None)
            continue
        rets = s.pct_change().dropna()
        total = float(s.iloc[-1] - 1)
        n_years = len(s) / 12
        annual = float((1 + total) ** (1 / max(n_years, 1e-9)) - 1) if n_years > 0.3 else total
        vol = float(rets.std() * np.sqrt(12)) if len(rets) > 1 else 0.0
        sharpe = float(annual / vol) if vol > 0 else None
        dd = float((s / s.cummax() - 1).min())
        out.append({
            "annual": round(annual, 4), "vol": round(vol, 4),
            "sharpe": round(sharpe, 3) if sharpe is not None else None,
            "dd": round(dd, 4), "total": round(total, 4),
        })
    return out


def weekly_ic_series(ic: pd.Series) -> dict:
    """日频 IC -> 周频序列（每周最后一个有效值），供画 IC 时间序列 + 4 周 MA。"""
    s = ic.dropna()
    if len(s) == 0:
        return {}
    last = s.groupby(s.index.to_period("W")).last()
    return {str(p.end_time.date()): round(float(v), 4) for p, v in last.items()}


def ic_decay_series(ic: pd.Series, max_lag: int = 10) -> list:
    """IC 衰减：IC(t) 与 IC(t+lag) 的相关（信号持久度），lag=1..max_lag。"""
    s = ic.dropna()
    if len(s) < max_lag + 5:
        return []
    out = []
    for lag in range(1, max_lag + 1):
        a, b = s.iloc[:-lag].values, s.iloc[lag:].values
        if len(a) < 10 or a.std() == 0 or b.std() == 0:
            out.append(None)
        else:
            out.append(round(float(np.corrcoef(a, b)[0, 1]), 4))
    return out


def ic_heatmap(monthly_ic: dict) -> dict:
    """月度 IC -> {year: [12 个月的 IC]}（无数据填 None），供热力图。"""
    years = {}
    for mo, v in monthly_ic.items():
        y, m = mo.split("-")
        years.setdefault(y, [None] * 12)[int(m) - 1] = v
    return {y: vals for y, vals in sorted(years.items())}


def layer_avg_ret(nav: pd.DataFrame) -> list:
    """每层平均每期收益（月频净值的简单收益均值），供分层收益柱状图。"""
    out = []
    for col in nav.columns:
        s = nav[col].dropna()
        if len(s) < 3:
            out.append(None)
            continue
        rets = s.pct_change().dropna()
        out.append(round(float(rets.mean()), 4))
    return out


def build_factor_data(lib: FactorLibrary, reg: pd.DataFrame,
                      returns: pd.DataFrame | None = None) -> list[dict]:
    """每个因子 -> {name, family, formula, 分层/IC/热力图等详情数据}。"""
    out = []
    for _, r in reg.iterrows():
        name = r["name"]
        ev = lib._load_eval(name)
        if ev is None or len(ev) == 0:
            continue
        ic = ev["ic"].dropna() if "ic" in ev.columns else pd.Series(dtype=float)
        ls_eq = ev["equity_ls_M"].dropna() if "equity_ls_M" in ev.columns else pd.Series(dtype=float)
        lo_eq = ev["equity_lo_M"].dropna() if "equity_lo_M" in ev.columns else pd.Series(dtype=float)

        n = len(ic)
        ic_mean = float(ic.mean()) if n else np.nan
        ic_std = float(ic.std()) if n > 1 else np.nan
        icir = ic_mean / ic_std * np.sqrt(12) if ic_std and ic_std > 0 else np.nan
        ls_ret = float(ls_eq.iloc[-1] - 1) if len(ls_eq) else np.nan
        lo_ret = float(lo_eq.iloc[-1] - 1) if len(lo_eq) else np.nan
        ls_sharpe = None
        if "dret_ls_M" in ev.columns:
            d = ev["dret_ls_M"].dropna()
            if len(d) > 1 and d.std() > 0:
                ls_sharpe = float(d.mean() / d.std() * np.sqrt(12))

        item = {
            "name": name,
            "family": family_of(r.get("source")),
            "formula": str(r.get("formula", ""))[:300],
            "ic_series": monthly_series(ic),          # 月度 IC（柱状图/热力图/重算）
            "ls_nav": monthly_nav(ls_eq),             # 月度多空净值
            "lo_nav": monthly_nav(lo_eq),             # 月度多头净值
            "ic_w": weekly_ic_series(ic),             # 周频 IC（时间序列 + MA）
            "ic_decay": ic_decay_series(ic),          # IC 衰减 lag1~10
            "heat": ic_heatmap(monthly_series(ic)),   # 月度 IC 热力图
            "q5_M": None, "q10_M": None, "q5_W": None, "q10_W": None,
            "layer_stats_M": None, "layer_avg_M": None,
            "layer_stats_W": None, "layer_avg_W": None,
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
        }

        # 分层回测（5层/10层 × 月/周），逐日 -> 月度末点（每种只算一次，复用）
        if returns is not None:
            panel = pd.read_parquet(Path(str(r["panel_path"])))
            caches = {}
            for n_q, freq in [(5, "M"), (10, "M"), (5, "W"), (10, "W")]:
                caches[(n_q, freq)] = qcut_rebal(panel, returns, n_q, freq)
            for (n_q, freq), key in [((5, "M"), "q5_M"), ((10, "M"), "q10_M"),
                                     ((5, "W"), "q5_W"), ((10, "W"), "q10_W")]:
                nav = caches[(n_q, freq)]
                if nav is not None:
                    item[key] = {str(p): [round(float(x), 4) if pd.notna(x) else None
                                          for x in row.values]
                                 for p, row in nav.iterrows()}
            nav5M = caches[(5, "M")]
            if nav5M is not None:
                item["layer_stats_M"] = layer_stats_from_nav(nav5M)
                item["layer_avg_M"] = layer_avg_ret(nav5M)
            nav5W = caches[(5, "W")]
            if nav5W is not None:
                item["layer_stats_W"] = layer_stats_from_nav(nav5W)
                item["layer_avg_W"] = layer_avg_ret(nav5W)

        out.append(item)
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
.chart-grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:14px; }}
.chart-box {{ position:relative; height:260px; }}
.dtabs {{ margin:10px 0 4px; display:flex; gap:6px; }}
.dtab {{ padding:5px 14px; border:1px solid #ddd; background:#fff; border-radius:6px; font-size:12.5px; cursor:pointer; color:#555; }}
.dtab.active {{ background:#16213e; color:#fff; border-color:#16213e; }}
.layer-tbl {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:8px; }}
.layer-tbl th {{ padding:5px 8px; background:#f0f2f5; text-align:right; border-bottom:1px solid #ddd; }}
.layer-tbl td {{ padding:5px 8px; text-align:right; border-bottom:1px solid #f0f0f0; }}
.layer-tbl th:first-child, .layer-tbl td:first-child {{ text-align:left; }}
.heat-grid {{ display:grid; grid-template-columns:repeat(13, 1fr); gap:2px; font-size:10px; margin-top:10px; }}
.heat-cell {{ padding:4px 2px; text-align:center; border-radius:3px; color:#fff; }}
.heat-label {{ color:#888; padding:4px 2px; text-align:center; }}
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
<div class="dtabs">
<button class="dtab active" onclick="setFreq('M')">月频调仓</button>
<button class="dtab" onclick="setFreq('W')">周频调仓</button>
<button class="dtab" onclick="setLayers(5)" id="tab5" >5层</button>
<button class="dtab" onclick="setLayers(10)" id="tab10">10层</button>
</div>
<div class="chart-grid2">
<div class="chart-box"><canvas id="cLayers" role="img" aria-label="分层净值曲线">分层净值</canvas></div>
<div class="chart-box"><canvas id="cBar" role="img" aria-label="分层收益柱状图">分层收益</canvas></div>
<div class="chart-box"><canvas id="cIcTs" role="img" aria-label="IC时间序列">IC序列</canvas></div>
<div class="chart-box"><canvas id="cDecay" role="img" aria-label="IC衰减">IC衰减</canvas></div>
</div>
<div class="chart-grid2">
<div class="chart-box"><canvas id="cHeat" role="img" aria-label="月度IC热力图">IC热力图</canvas></div>
<div class="chart-box" id="cLayerTable"></div>
</div>
<div class="formula" id="dformula"></div>
</div>
</div>

<script>
const DATA = {factor_data};
const MONTHS = {months};
const FAM_COLORS = {fam_colors};

// ---------- 状态 ----------
let current = {{ sortKey: 'ic', asc: false, family: 'all', q: '', start: null, end: null }};
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
    tr.onclick = () => plotDetail(f);
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
let dFreq = 'M', dLayers = 5, curFactor = null;
let chartLayers = null, chartBar = null, chartIcTs = null, chartDecay = null;

function setFreq(f) {{ dFreq = f; document.querySelectorAll('.dtab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.dtab')[f==='M'?0:1].classList.add('active'); if (curFactor) plotDetail(curFactor); }}
function setLayers(n) {{ dLayers = n; document.querySelectorAll('.dtab').forEach(b => b.classList.remove('active'));
  document.getElementById('tab'+(n===5?5:10)).classList.add('active'); if (curFactor) plotDetail(curFactor); }}

// 分层净值数据：按频率/层数取月度序列
function layerSeries(f) {{
  const key = (dFreq==='M'?'q':'q') + dLayers + '_' + dFreq;
  const nav = f[key] || {{}};
  const months = Object.keys(nav).filter(mo => mo >= current.start && mo <= current.end);
  return {{ months, nav }};
}}

function plotDetail(f) {{
  curFactor = f;
  document.getElementById('detail').classList.remove('hidden');
  document.getElementById('dtitle').textContent = f.name + '  —  ' + current.start + ' ~ ' + current.end;
  document.getElementById('dformula').textContent = '公式: ' + f.formula;
  const m = rangeMetrics(f);
  document.getElementById('dmetrics').innerHTML = [
    ['IC', fmt(m.ic,4)], ['ICIR', fmt(m.icir)], ['NW-t', fmt(m.t_nw)],
    ['多空收益', fmtPct(m.ls_ret)], ['多头收益', fmtPct(m.lo_ret)],
    ['样本月数', m.n], ['IC胜率', f.m_full.win!==null?(f.m_full.win*100).toFixed(0)+'%':'—'],
    ['显著', f.m_full.sig?'是':'否']
  ].map(([l,v]) => `<div class="metric"><div class="l">${{l}}</div><div class="v">${{v}}</div></div>`).join('');

  const {{ months, nav }} = layerSeries(f);
  const nQ = dLayers;

  // 1) 分层净值曲线（每层一条 + 多空线 Qn-Q1）
  if (chartLayers) chartLayers.destroy();
  const layerColors = ['#A32D2D','#BA7517','#639922','#1D9E75','#378ADD','#534AB7','#993556','#0F6E56','#854F0B','#185FA5'];
  const ds = [];
  const base = {{}};
  // 起点：区间前最后一个月（归一），无则取区间首月
  const first = months.length ? months[0] : null;
  for (let q=1; q<=nQ; q++) {{
    const prev = Object.entries(nav).filter(([mo]) => mo < current.start && nav[mo][q-1] != null);
    base[q] = prev.length ? prev[prev.length-1][1][q-1] : (first && nav[first][q-1] != null ? nav[first][q-1] : 1);
  }}
  for (let q=1; q<=nQ; q++) {{
    const vals = months.map(mo => nav[mo] && nav[mo][q-1] != null ? nav[mo][q-1]/base[q] : null);
    ds.push({{ label: 'Q'+q, data: vals, borderColor: layerColors[(q-1)%10], borderWidth: 1.2, pointRadius: 0, tension: .15 }});
  }}
  // 多空线 = Qn - Q1 归一后差（独立累计：用 Qn/Q1 相对净值）
  const lsVals = months.map(mo => {{ if (!nav[mo] || nav[mo][0]==null || nav[mo][nQ-1]==null) return null;
    return (nav[mo][nQ-1]/base[nQ]) / (nav[mo][0]/base[1]); }});
  ds.push({{ label: 'Q'+nQ+'-Q1(多空)', data: lsVals, borderColor: '#000', borderWidth: 1.8, borderDash: [4,3], pointRadius: 0, tension: .15 }});
  chartLayers = new Chart(document.getElementById('cLayers'), {{
    type: 'line', data: {{ labels: months, datasets: ds }},
    options: {{ responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 10, font: {{ size: 10 }} }} }},
        title: {{ display: true, text: '分层净值（'+nQ+'层·'+(dFreq==='M'?'月频':'周频')+'，区间起点=1）', font: {{ size: 13 }} }} }},
      scales: {{ y: {{ title: {{ display: true, text: '累计净值' }} }}, x: {{ ticks: {{ maxTicksLimit: 10, font: {{ size: 9 }} }} }} }} }}
  }});

  // 2) 分层平均每期收益柱状图
  if (chartBar) chartBar.destroy();
  const avgKey = 'layer_avg_' + dFreq;
  const avg = f[avgKey] || [];
  chartBar = new Chart(document.getElementById('cBar'), {{
    type: 'bar',
    data: {{ labels: avg.map((_,i) => 'Q'+(i+1)), datasets: [{{ label: '平均每期收益',
      data: avg, backgroundColor: avg.map(v => v>=0 ? 'rgba(163,45,45,.75)' : 'rgba(59,109,17,.75)') }}]}},
    options: {{ responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: '分层平均每期收益（'+ (dFreq==='M'?'月':'周') +'）', font: {{ size: 13 }} }} }},
      scales: {{ y: {{ title: {{ display: true, text: '每期收益' }} }} }} }}
  }});

  // 3) IC 时间序列 + 4周移动平均（周频数据）
  if (chartIcTs) chartIcTs.destroy();
  const icW = Object.entries(f.ic_w).filter(([w]) => w.slice(0,7) >= current.start && w.slice(0,7) <= current.end);
  const icWv = icW.map(([,v]) => v);
  const icMa = icWv.map((_,i) => i>=3 ? icWv.slice(i-3,i+1).reduce((a,b)=>a+b,0)/4 : null);
  chartIcTs = new Chart(document.getElementById('cIcTs'), {{
    type: 'line',
    data: {{ labels: icW.map(([w]) => w), datasets: [
      {{ label: 'IC(周)', data: icWv, borderColor: '#378ADD', borderWidth: 1.2, pointRadius: 0, tension: .1 }},
      {{ label: 'IC 4周MA', data: icMa, borderColor: '#A32D2D', borderWidth: 1.8, pointRadius: 0, tension: .2 }}
    ]}},
    options: {{ responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 10, font: {{ size: 10 }} }} }},
        title: {{ display: true, text: 'IC 时间序列（周频 + 4周移动平均）', font: {{ size: 13 }} }} }},
      scales: {{ y: {{ title: {{ display: true, text: 'IC' }} }}, x: {{ ticks: {{ maxTicksLimit: 10, font: {{ size: 9 }} }} }} }} }}
  }});

  // 4) IC 衰减
  if (chartDecay) chartDecay.destroy();
  chartDecay = new Chart(document.getElementById('cDecay'), {{
    type: 'line',
    data: {{ labels: (f.ic_decay||[]).map((_,i) => 'lag'+(i+1)), datasets: [{{ label: 'IC',
      data: f.ic_decay||[], borderColor: '#534AB7', backgroundColor: 'rgba(83,74,183,.15)',
      borderWidth: 1.6, pointRadius: 2.5, tension: .15, fill: true }}]}},
    options: {{ responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: 'IC 衰减（信号持久度）', font: {{ size: 13 }} }} }},
      scales: {{ y: {{ title: {{ display: true, text: '自相关' }} }} }} }}
  }});

  // 5) 月度 IC 热力图（年份×月份）
  const heatEl = document.getElementById('cHeat');
  heatEl.style.display = 'block';
  const parent = heatEl.parentNode;
  let heatHtml = '<div style="font-size:12px;color:#16213e;margin-bottom:4px;font-weight:500;">月度 IC 热力图</div>';
  heatHtml += '<div class="heat-grid"><div class="heat-label"></div><div class="heat-label">1月</div><div class="heat-label">2月</div><div class="heat-label">3月</div><div class="heat-label">4月</div><div class="heat-label">5月</div><div class="heat-label">6月</div><div class="heat-label">7月</div><div class="heat-label">8月</div><div class="heat-label">9月</div><div class="heat-label">10月</div><div class="heat-label">11月</div><div class="heat-label">12月</div>';
  const h = f.heat || {{}};
  Object.entries(h).forEach(([yr, vals]) => {{
    heatHtml += '<div class="heat-label">'+yr+'</div>';
    vals.forEach(v => {{
      if (v === null || v === undefined) {{ heatHtml += '<div class="heat-cell" style="background:#f0f0f0;color:#999;">—</div>'; return; }}
      const t = Math.max(0, Math.min(1, (v + 0.1) / 0.2)); // IC -0.1~0.1 映射 0~1
      const r = Math.round(200 - t*170), g = Math.round(60 + t*100), b = Math.round(40 + t*60);
      heatHtml += '<div class="heat-cell" style="background:rgb('+r+','+g+','+b+');" title="'+yr+' 月IC='+v.toFixed(3)+'">'+v.toFixed(2)+'</div>';
    }});
  }});
  heatHtml += '</div>';
  parent.innerHTML = '<div style="position:relative;height:260px;overflow:auto;">' + heatHtml + '</div>';

  // 6) 各层绩效表
  const statsKey = 'layer_stats_' + dFreq;
  const stats = f[statsKey] || [];
  if (stats.length) {{
    let th = '<table class="layer-tbl"><tr><th>层级</th><th>年化收益</th><th>年化波动</th><th>Sharpe</th><th>最大回撤</th></tr>';
    stats.forEach((s, i) => {{
      th += `<tr><td>Q${{i+1}}</td><td>${{s&&s.annual!=null?(s.annual*100).toFixed(1)+'%':'—'}}</td><td>${{s&&s.vol!=null?(s.vol*100).toFixed(1)+'%':'—'}}</td><td>${{s&&s.sharpe!=null?s.sharpe.toFixed(2):'—'}}</td><td>${{s&&s.dd!=null?(s.dd*100).toFixed(1)+'%':'—'}}</td></tr>`;
    }});
    th += '</table>';
    document.getElementById('cLayerTable').innerHTML = '<div style="font-size:12px;color:#16213e;margin-bottom:4px;font-weight:500;">各层绩效（'+ (dFreq==='M'?'月频':'周频') +'）</div>' + th;
  }} else {{
    document.getElementById('cLayerTable').innerHTML = '<div style="color:#999;font-size:12px;">无分层绩效数据</div>';
  }}
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

// 初始化：默认展示全部时间段 + 全部来源
current.start = MONTHS[0]; current.end = MONTHS[MONTHS.length-1];
document.getElementById('dstart').value = current.start.slice(0,7);
document.getElementById('dend').value = current.end.slice(0,7);
document.getElementById('ffamily').value = 'all';
document.getElementById('preset').value = 'all';
render();
</script>
</body></html>"""


def _build_one(args) -> list[dict] | None:
    """多进程 worker：构建单因子详情数据。args=(registry_row_dict, returns)。"""
    import pandas as pd
    from research.factor_library import FactorLibrary
    try:
        row, returns = args
        lib = FactorLibrary(dataset=row.get("dataset", "hs300_2022_2025"))
        return build_factor_data(lib, pd.DataFrame([row]), returns)
    except Exception as e:
        log.warning("因子 %s 失败: %s", args[0].get("name") if args else "?", e)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="交互式因子检测报告")
    ap.add_argument("--dataset", default="hs300_2022_2025")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=4, help="分层回测并行进程数")
    args = ap.parse_args()

    lib = FactorLibrary(dataset=args.dataset)
    reg = lib.list_all()
    log.info("因子库 %s: %d 因子", args.dataset, len(reg))

    # 收益面板（与 IC/回测同口径：未来一期收益）
    from scripts.e2e_common import load_daily_data
    px, _ = load_daily_data(begin=20220101)
    returns = px["close"].pct_change(fill_method=None).shift(-1)
    log.info("收益面板: %d 日 × %d 股", returns.shape[0], returns.shape[1])

    # 多进程分层回测（每因子独立，可并行）
    import multiprocessing as mp

    n_cpu = max(1, args.jobs)
    tasks = [(r.to_dict() | {"dataset": args.dataset}, returns) for _, r in reg.iterrows()]
    log.info("并行分层回测（%d 进程）...", n_cpu)
    with mp.Pool(n_cpu) as pool:
        results = pool.map(_build_one, tasks)
    data = [d[0] for d in results if d]
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
