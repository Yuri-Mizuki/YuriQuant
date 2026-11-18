"""本期出榜解释页（为什么买这些）—— 接在模拟交易页之后。

页面结构
--------
``reports/yq_core/picks.html`` 单页，信号日下拉切换，全部期内嵌：
  指标条（信号日 / 链路 / 候选池 / Top10% / 实持 / 行业数）
  本期选股 Top 20（排名 · 代码 · 行业 · 分数 · SHAP 前三驱动）
  行业热度（申万一级 31 类，按均分取前 15）
  模型今天看什么（特征重要性 Top 20 + 累积贡献）
  候选池前 60（可搜索）

数据源
------
``reports/alla_daily_defv/``（定版链）与 ``reports/alla_daily_solo/``（实盘链）的
``picks_ / explain_top_ / industry_rank_ / feature_importance_`` 四件套；同一信号日两链
都有时取**定版**。实持数取自 ``reports/yq_core/live/holdings_<信号日>.csv``（仅最新期有）。

口径
----
frac=0.1（Top10% ≈ 459 只等权）、buffer=0.15/0.40、slow_blend=0.5；
三因子 = 预测日 SHAP 贡献绝对值前三（``predict_contrib``），``z`` 为该股在此因子上的
截面标准化值，**正负表示该股在此因子上的高低，不是看多看空的符号**；
实持与 Top10% 不是同一口径（k=8、每行业≤1、整手 100 股），**不可直接对数**。

用法::

    python -m scripts.reporting.picks_board             # 全部期，写 picks.html
    python -m scripts.reporting.picks_board --period 20261102   # 只留一期
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.reporting.yqcore_dashboard import render_html, topbar

ROOT = Path(__file__).resolve().parents[2]
DEFV = ROOT / "reports" / "alla_daily_defv"
SOLO = ROOT / "reports" / "alla_daily_solo"
LIVE = ROOT / "reports" / "yq_core" / "live"
OUT = ROOT / "reports" / "yq_core" / "picks.html"

POOL_TOP = 60          # 候选池表最多展示
IND_TOP = 15           # 行业图最多展示
SRC_LABEL = {"defv": "定版", "solo": "实盘"}


def _num(v, nd: int = 4):
    """NaN / inf / 非数值 → None（JSON 里不出现 null 之外的脏值）。"""
    try:
        if v is None or pd.isna(v):
            return None
        f = float(v)
        return round(f, nd) if pd.notna(f) else None
    except (TypeError, ValueError):
        return None


def _csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def _discover() -> list[tuple[str, str, Path]]:
    """[(信号日, 链路, 目录)]，按日期倒序；同日期定版优先。"""
    got: dict[str, tuple[str, Path]] = {}
    for src, d in (("defv", DEFV), ("solo", SOLO)):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("picks_*.csv")):
            date = f.stem.split("_")[-1]
            if date.isdigit() and len(date) == 8:
                got.setdefault(date, (src, d))
    return [(k, v[0], v[1]) for k, v in sorted(got.items(), reverse=True)]


def _cfg(d: Path, date: str) -> dict:
    h = _csv(d / "history.csv")
    if h is None or "predict_date" not in h:
        return {}
    row = h[h["predict_date"].astype(str) == date]
    if not len(row):
        return {}
    r = row.iloc[0]
    hz = r.get("horizons")
    return {
        "frac": _num(r.get("frac"), 3),
        "buffer": None if pd.isna(r.get("buffer")) else str(r.get("buffer")),
        "blend": _num(r.get("slow_blend"), 2),
        "model": None if pd.isna(r.get("model")) else str(r.get("model")),
        "hz": None if pd.isna(hz) else str(hz),
    }


def _holdings(date: str) -> dict:
    p = LIVE / f"holdings_{date}.csv"
    d = _csv(p)
    if d is None or not len(d) or "code" not in d:
        return {}
    return {str(r["code"]): _num(r.get("weight_actual"), 4) for _, r in d.iterrows()}


def _explain(d: Path, date: str) -> list[dict]:
    x = _csv(d / f"explain_top_{date}.csv")
    if x is None or not len(x):
        return []
    out = []
    for _, r in x.iterrows():
        drv = []
        for i in (1, 2, 3):
            nm = r.get(f"drv{i}_name")
            if pd.isna(nm):
                continue
            drv.append([str(nm), _num(r.get(f"drv{i}_z"), 2), _num(r.get(f"drv{i}_contrib"), 4)])
        out.append({
            "c": str(r["code"]), "r": int(r["rank"]), "s": _num(r.get("score"), 4),
            "i2": None if pd.isna(r.get("industry_l2")) else str(r.get("industry_l2")),
            "i1": None if pd.isna(r.get("industry_l1")) else str(r.get("industry_l1")),
            "d": drv,
        })
    return out


def _industry(d: Path, date: str) -> list[dict]:
    x = _csv(d / f"industry_rank_{date}.csv")
    if x is None or not len(x):
        return []
    if "level" in x:
        x = x[x["level"] == "L1"]
    out = []
    for _, r in x.iterrows():
        out.append({
            "n": str(r["industry"]), "k": int(r["rank"]),
            "cnt": _num(r.get("n_stocks"), 0), "sc": _num(r.get("mean_score"), 4),
            "sh": _num(r.get("top_frac_share"), 3),
        })
    return sorted(out, key=lambda z: (z["k"] if z["k"] is not None else 999))


def _features(d: Path, date: str) -> list[dict]:
    x = _csv(d / f"feature_importance_{date}.csv")
    if x is None or not len(x):
        return []
    out = []
    for _, r in x.head(25).iterrows():
        out.append({
            "r": int(r["rank"]), "f": str(r["feature"]),
            "cn": None if pd.isna(r.get("name_cn")) else str(r.get("name_cn")),
            "fam": None if pd.isna(r.get("family")) else str(r.get("family")),
            "sh": _num(r.get("share"), 4), "cum": _num(r.get("cum_share"), 4),
        })
    return out


def _pool(d: Path, date: str) -> list[dict]:
    x = _csv(d / f"picks_{date}.csv")
    if x is None or not len(x):
        return []
    out = []
    for _, r in x.head(POOL_TOP).iterrows():
        out.append({
            "c": str(r["code"]), "r": int(r["rank"]), "s": _num(r.get("score"), 4),
            "p": _num(r.get("pct_rank"), 4),
            "i2": None if pd.isna(r.get("industry_l2")) else str(r.get("industry_l2")),
            "i1": None if pd.isna(r.get("industry_l1")) else str(r.get("industry_l1")),
            "t": 1 if bool(r.get("top_frac")) else 0,
        })
    return out


def _period(date: str, src: str, d: Path) -> dict | None:
    picks = _csv(d / f"picks_{date}.csv")
    if picks is None or not len(picks):
        return None
    ex = _explain(d, date)
    if not ex:                      # 没有解释文件就不成页（早期期数可能缺）
        return None
    hold = _holdings(date)
    return {
        "d": date, "src": src, "cfg": _cfg(d, date),
        "np": int(len(picks)),
        "nt": int(picks["top_frac"].sum()) if "top_frac" in picks else None,
        "nh": len(hold) or None,
        "hd": hold,
        "ex": ex,
        "ind": _industry(d, date),
        "ft": _features(d, date),
        "pl": _pool(d, date),
    }


def build_payload(period: str | None = None) -> dict:
    periods = []
    for date, src, d in _discover():
        if period and period != date:
            continue
        p = _period(date, src, d)
        if p:
            periods.append(p)
    if not periods:
        raise SystemExit("没有可用的出榜产物（picks_*.csv + explain_top_*.csv）")
    return {"gen": datetime.now().strftime("%Y-%m-%d %H:%M"), "periods": periods}


_BODY = """
<style>
.sel{padding:4px 8px;border:1px solid #dfe3e8;border-radius:3px;background:#fff;
     font-size:12.5px;color:#22262e}
.s2{font-size:12px;color:#7a8296;font-weight:400;margin-left:6px}
.lbl{font-size:12px;color:#7a8296;margin-right:6px}
table.tbl tr.hold td{background:#fffaf0}
.tag-hold{display:inline-block;padding:0 5px;border-radius:2px;background:#fdecc8;
          color:#8a5a00;font-size:11px;margin-left:5px;vertical-align:1px}
td.drv{font-size:12px;color:#3f4657;white-space:nowrap}
td.drv .z{color:#8a92a3;margin-left:5px;font-variant-numeric:tabular-nums}
#q{padding:4px 8px;border:1px solid #dfe3e8;border-radius:3px;font-size:12.5px;width:180px}
.dot{display:inline-block;width:6px;height:6px;border-radius:3px;background:#d98324;
     margin-right:5px;vertical-align:1px}
</style>

<div class="wrap">
  <div class="subbar">
    <div class="title">本期出榜<span class="s2">为什么是这些票 · 三因子 SHAP 驱动</span></div>
    <div class="acts"><span class="lbl">信号日</span><select id="pd" class="sel"></select></div>
  </div>

  <div class="kpibar">
    <div class="kpi"><span>信号日</span><b id="k_date">—</b></div>
    <div class="kpi"><span>链路</span><b id="k_src">—</b></div>
    <div class="kpi"><span>打分名单</span><b id="k_pool">—</b><small>只</small></div>
    <div class="kpi"><span>入选（top_frac）</span><b id="k_top">—</b><small id="k_top2"></small></div>
    <div class="kpi"><span>实持</span><b id="k_hold">—</b><small>只</small></div>
    <div class="kpi"><span>申万一级</span><b id="k_ind">—</b><small>类</small></div>
  </div>

  <div class="row2">
    <div class="card">
      <div class="card-hd"><h2>本期选股 Top 20</h2>
        <span class="sub2">按模型打分降序 · 灰色为特征 z 值</span></div>
      <div class="card-bd" style="padding:0">
        <div class="tbl-wrap" style="max-height:640px">
          <table class="tbl" id="pickTable">
            <thead><tr>
              <th class="num" style="width:48px">排名</th><th style="width:104px">代码</th>
              <th style="width:150px">行业</th><th class="num" style="width:78px">分数</th>
              <th>驱动 1</th><th>驱动 2</th><th>驱动 3</th>
            </tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-hd"><h2>行业热度</h2>
        <span class="sub2">申万一级 · 按均分取前 <span id="indN">15</span></span></div>
      <div class="card-bd"><div id="indChart" class="chart-m"></div></div>
    </div>
  </div>

  <div class="card">
    <div class="card-hd"><h2>模型今天看什么</h2>
      <span class="sub2">特征重要性 Top 15（gain 归一化）· <span id="featCum"></span></span></div>
    <div class="card-bd">
      <div id="featFam" class="legend" style="margin:0 0 4px"></div>
      <div id="featChart" class="chart-h"></div>
    </div>
  </div>

  <div class="card">
    <div class="card-hd"><h2>候选池前 <span id="poolN">60</span></h2>
      <span class="sub2">全池按打分排序 · <span class="dot"></span>Top10% 入选</span>
      <div class="right"><input id="q" placeholder="搜代码 / 行业"></div></div>
    <div class="card-bd" style="padding:0">
      <div class="tbl-wrap" style="max-height:430px">
        <table class="tbl" id="poolTable">
          <thead><tr>
            <th class="num" style="width:56px">排名</th><th style="width:110px">代码</th>
            <th style="width:150px">二级行业</th><th>一级行业</th>
            <th class="num" style="width:86px">分数</th><th class="num" style="width:86px">分位</th>
          </tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="note">
    <b>口径</b>：frac=0.1（Top10% ≈ 459 只等权）· buffer=0.15/0.40 · slow_blend=0.5。
    <b>三因子</b>为预测日 SHAP 贡献绝对值前三的因子；<code>z</code> 是该股在此因子上的
    截面标准化值，<b>正负表示该股在此因子上的高低，不是看多看空的方向</b>。
    <b>实持</b>与 Top10% 不是同一口径（k=8、每行业≤1、整手 100 股约束），
    <b>不可直接对数</b>；两者差多少只属正常。
    数据源：<code>reports/alla_daily_defv</code>（定版链）与 <code>reports/alla_daily_solo</code>
    （实盘链），同一信号日两链都有时取定版。两链的 <code>picks</code> 落盘范围不同：
    定版落**全池**（760 只，其中 top_frac 459 只），实盘只落**入选名单**（518 只，全部
    top_frac）—— 所以「打分名单」两期不可直接比大小。
  </div>
</div>
"""

_JS = r"""
const P = window.__YQ__;
const IND_N = 15;   /* 与 py 端 IND_TOP 保持一致（行业图最多展示条数） */
const S = document.getElementById('pd');
S.innerHTML = P.periods.map((p, i) =>
  `<option value="${i}">${p.d.slice(0,4)}-${p.d.slice(4,6)}-${p.d.slice(6)} · ${p.src === 'defv' ? '定版' : '实盘'}</option>`
).join('');
let cur = 0, indChart = null, featChart = null;

const fmtD = d => d.slice(0,4) + '-' + d.slice(4,6) + '-' + d.slice(6);

function renderPick(p) {
  const rows = p.ex.map(e => {
    const isHold = p.hd && p.hd[e.c] !== undefined;
    const drv = (e.d || []).map(x =>
      `<td class="drv">${x[0]}<span class="z">${x[1] > 0 ? '+' : ''}${(+x[1]).toFixed(2)}σ</span></td>`
    ).join('');
    const ind = (e.i2 || '') + (e.i1 ? `<span class="z" style="margin-left:6px">${e.i1}</span>` : '');
    return `<tr class="${isHold ? 'hold' : ''}"><td class="num">${e.r}</td>`
      + `<td class="mono">${e.c}${isHold ? '<span class="tag-hold">实持</span>' : ''}</td>`
      + `<td class="drv">${ind}</td><td class="num">${(+e.s).toFixed(4)}</td>${drv}</tr>`;
  }).join('');
  document.querySelector('#pickTable tbody').innerHTML = rows;
}

function renderInd(p) {
  const n = Math.min(IND_N, p.ind.length);
  document.getElementById('indN').textContent = n;
  const rows = p.ind.slice(0, n).reverse();
  if (!indChart) indChart = echarts.init(document.getElementById('indChart'));
  const mx = Math.max(0.1, ...rows.map(x => x.sc || 0));
  indChart.setOption({
    animation: false,
    grid: {left: 76, right: 18, top: 6, bottom: 24},
    tooltip: {trigger: 'item', formatter: o => {
      const d = o.data;
      const sh = d.sh == null ? '—' : (d.sh * 100).toFixed(0) + '%';
      return `${d.name}<br>均分 ${(+d.value).toFixed(4)}<br>成分 ${d.cnt} 只 · Top10% 占 ${sh}`;
    }},
    xAxis: {type: 'value', min: 0, max: Math.ceil(mx * 10) / 10, splitNumber: 3,
            axisLabel: {fontSize: 10, formatter: v => v.toFixed(2)},
            splitLine: {lineStyle: {color: '#f4f5f8'}}},
    yAxis: {type: 'category', data: rows.map(x => x.n),
            axisLabel: {fontSize: 11, color: '#4b5563'}, axisTick: {show: false},
            axisLine: {show: false}},
    series: [{type: 'bar', barMaxWidth: 14,
      data: rows.map(x => ({value: x.sc, name: x.n, cnt: x.cnt, sh: x.sh})),
      itemStyle: {color: '#2f6fdb', borderRadius: [0, 2, 2, 0]},
      label: {show: true, position: 'insideRight', fontSize: 9.5, color: '#fff',
              formatter: o => (+o.value).toFixed(3)}}]
  }, true);
}

const FAMC = {'量价因子': '#2f6fdb', '估值因子': '#d98324', '筹码因子': '#12a05c',
              '规模因子': '#7a5af8', '财务质量': '#8a8f98'};

function renderFeat(p) {
  const ft = p.ft.slice(0, 15).slice().reverse();     /* 横向条形：最重要的在最上 */
  const cum5 = p.ft.length >= 5 ? p.ft[4].cum : null;
  const cum10 = p.ft.length >= 10 ? p.ft[9].cum : null;
  document.getElementById('featCum').textContent =
    cum5 == null ? '' : `Top5 累积 ${(cum5 * 100).toFixed(1)}% · Top10 ${(cum10 * 100).toFixed(1)}%`;
  document.getElementById('featFam').innerHTML = Object.entries(FAMC).map(
    ([k, v]) => `<span style="display:inline-block;margin-right:10px">`
      + `<i style="display:inline-block;width:8px;height:8px;border-radius:2px;`
      + `background:${v};margin-right:4px"></i>${k}</span>`).join('');
  if (!featChart) featChart = echarts.init(document.getElementById('featChart'));
  featChart.setOption({
    animation: false,
    grid: {left: 152, right: 52, top: 8, bottom: 26},
    tooltip: {trigger: 'item', formatter: o => {
      const x = o.data;
      return `${x.cn || x.f}<br><span style="color:#8a92a3">${x.f}</span><br>`
        + `族：${x.fam || '—'}<br>单项 ${(x.sh * 100).toFixed(2)}% · 累积 ${(x.cum * 100).toFixed(1)}%`;
    }},
    xAxis: {type: 'value', axisLabel: {fontSize: 10, formatter: v => v + '%'},
            splitLine: {lineStyle: {color: '#f4f5f8'}}},
    yAxis: {type: 'category', data: ft.map(x => x.cn || x.f),
            axisLabel: {fontSize: 10, color: '#4b5563', width: 140, overflow: 'truncate'},
            axisTick: {show: false},
            axisLine: {show: false}},
    series: [{type: 'bar', barMaxWidth: 15,
      data: ft.map(x => ({value: +(x.sh * 100).toFixed(2), cn: x.cn, f: x.f,
                          fam: x.fam, sh: x.sh, cum: x.cum})),
      itemStyle: {borderRadius: [0, 2, 2, 0],
        color: o => FAMC[o.data.fam] || '#5a7d9a'},
      label: {show: true, position: 'right', fontSize: 9.5, color: '#6b7280',
              formatter: o => o.value.toFixed(1) + '%'}}]
  }, true);
}

function renderPool(p) {
  const q = (document.getElementById('q').value || '').trim().toUpperCase();
  const rows = !q ? p.pl : p.pl.filter(x =>
    (x.c || '').includes(q) || (x.i2 || '').includes(q) || (x.i1 || '').includes(q));
  document.getElementById('poolN').textContent = p.pl.length;
  document.querySelector('#poolTable tbody').innerHTML = rows.map(x =>
    `<tr><td class="num">${x.r}</td><td class="mono">`
    + `${x.t ? '<span class="dot"></span>' : ''}${x.c}</td>`
    + `<td>${x.i2 || ''}</td><td>${x.i1 || ''}</td>`
    + `<td class="num">${(+x.s).toFixed(4)}</td>`
    + `<td class="num">${(x.p * 100).toFixed(2)}%</td></tr>`
  ).join('') || `<tr><td colspan="6" class="mut" style="padding:14px">无匹配</td></tr>`;
}

function render(i) {
  cur = i;
  const p = P.periods[i], cfg = p.cfg || {};
  document.getElementById('k_date').textContent = fmtD(p.d);
  document.getElementById('k_src').textContent = p.src === 'defv' ? '定版' : '实盘';
  document.getElementById('k_pool').textContent = p.np;
  document.getElementById('k_top').textContent = p.nt == null ? '—' : p.nt;
  document.getElementById('k_top2').textContent =
    p.nt == null || !p.np ? '' : (p.nt >= p.np ? '（全部）' : '（占' + (p.nt / p.np * 100).toFixed(0) + '%）');
  document.getElementById('k_hold').textContent = p.nh == null ? '—' : p.nh;
  document.getElementById('k_ind').textContent = p.ind.length;
  renderPick(p);
  renderInd(p);
  renderFeat(p);
  renderPool(p);
  S.title = `frac=${cfg.frac} · buffer=${cfg.buffer} · blend=${cfg.blend}`;
}

S.onchange = () => render(+S.value);
document.getElementById('q').oninput = () => renderPool(P.periods[cur]);
render(0);
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="本期出榜解释页（自包含 HTML）")
    ap.add_argument("--period", default=None, help="只保留某一信号日 YYYYMMDD")
    ap.add_argument("--out", default=None, help="输出路径（缺省 reports/yq_core/picks.html）")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    payload = build_payload(args.period)
    body = topbar("picks", payload["gen"]) + _BODY
    html = render_html(payload, body, _JS, "本期出榜 · 选股解释")
    out = Path(args.out) if args.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    n = len(payload["periods"])
    first = payload["periods"][0]
    print(f"[出榜] {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"        {n} 期 · 最新 {first['d']}（{SRC_LABEL.get(first['src'], first['src'])}）"
          f" · 候选池 {first['np']} 只 · Top10% {first['nt']} 只 · 实持 {first['nh'] or '—'} 只")
    if args.open:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
