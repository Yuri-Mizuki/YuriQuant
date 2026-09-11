"""最佳策略收益曲线页（聚宽模拟交易风格）——策略 vs 中证全指口径基准。

消费 rolling_grid_alla 实验产物，复刻聚宽模拟交易页版式：
- 顶部统计条：累计收益 / 年化收益 / 超额收益 / 夏普 / 最大回撤 / 日胜率 /
  年化波动（+「其他指标」展开：α/β/IR/Calmar/换手/基准收益）；
- 主图「历史收益」：策略收益（蓝）vs 基准收益（红）vs 超额收益（橙），
  y 轴百分比、十字线 tooltip、周期切换（全部/近1年/近6月/近1月）；
- 下方「回撤」面积图（与主图联动同一时间窗）；
- 分年绩效表（对应聚宽持仓/收益明细区的信息密度）。

基准说明：用户要求中证全指（000985），但 AmazingData 该代码行情只到 2016-06；
采用覆盖等价的**国证A指 399317.SZ**（全市场A股指数，深交所编制），
页面上如实标注。

用法:
    python scripts/reporting/jq_style_report.py                          # 自动选全期超额第一
    python scripts/reporting/jq_style_report.py --run-id gbdt__h1__M__raw__f0.10
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

from config import Config  # noqa: E402
from scripts.common.cli_common import setup_logging  # noqa: E402
from research.html_report import page  # noqa: E402

log = setup_logging("jq_style_report")

OUT = Path("reports") / "alla_rolling"
_BM = Config.benchmarks()
BENCH_INDEX = _BM["report_a_share"]   # 国证A指（中证全指 000985 行情缺失的替代）
BENCH_LABEL = _BM["report_a_share_label"]

# 聚宽风格配色
C_STRAT = "#3f8cd6"     # 策略：蓝
C_BENCH = "#e05d5d"     # 基准：红
C_EXCESS = "#f5a623"    # 超额：橙
C_DD = "#9db3c9"        # 回撤：灰蓝

# 报告主题（经 research.html_report.page 外壳注入；本文件不再自拼 HTML 外壳）
_CHART_CDN = ('<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/'
              'dist/chart.umd.min.js"></script>')
_CSS = """
 * { box-sizing: border-box; }
 body { font-family: "Microsoft YaHei","PingFang SC",sans-serif; margin: 0;
        background: #eef1f5; color: #333; font-size: 13px; }
 .navbar { background: #2d3a4a; color: #fff; padding: 10px 20px;
           display: flex; align-items: baseline; gap: 14px; }
 .navbar .title { font-size: 15px; font-weight: 600; }
 .navbar .sub { font-size: 12px; color: #aab7c8; }
 .wrap { max-width: 1120px; margin: 14px auto; padding: 0 12px; }
 .card { background: #fff; border: 1px solid #e3e8ee; border-radius: 4px;
         margin-bottom: 12px; }
 .strip { display: flex; flex-wrap: wrap; padding: 14px 10px 6px; }
 .stat { min-width: 118px; padding: 2px 14px 10px; }
 .stat .v { font-size: 20px; font-weight: 600; }
 .stat .l { font-size: 12px; color: #8f9bb3; margin-top: 2px; }
 .pos { color: #e04b4b; } .neg { color: #1fa06a; }
 .more-toggle { cursor: pointer; }
 .tabs { display: flex; border-bottom: 1px solid #e3e8ee; padding: 0 12px; }
 .tab { padding: 10px 16px; color: #666; border-bottom: 2px solid transparent; }
 .tab.active { color: #2d6fb2; border-bottom-color: #2d6fb2; font-weight: 600; }
 .ranges { margin-left: auto; display: flex; gap: 4px; align-items: center;
          padding: 6px 0; }
 .ranges button { border: 1px solid #d8dee8; background: #fff; color: #555;
                 padding: 3px 10px; border-radius: 3px; cursor: pointer; font-size: 12px; }
 .ranges button.on { background: #2d6fb2; color: #fff; border-color: #2d6fb2; }
 .chartbox { padding: 10px 14px 14px; }
 .main { height: 360px; } .sub { height: 140px; }
 .legendline { padding: 4px 16px 0; color: #666; font-size: 12px; }
 table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
 th, td { border-bottom: 1px solid #eef1f5; padding: 6px 12px; text-align: right; }
 th { color: #8f9bb3; font-weight: 500; background: #fafbfd; }
 td:first-child, th:first-child { text-align: left; }
 .note { font-size: 12px; color: #8f9bb3; padding: 10px 16px 14px; line-height: 1.8; }
"""


def load_run(run_id: str | None) -> tuple[str, pd.DataFrame]:
    mo = pd.read_csv(OUT / "metrics_overall.csv")
    if run_id is None:
        row = mo.sort_values("excess_idx", ascending=False).iloc[0]
        run_id = str(row["run_id"])
        log.info("自动选择全期超额第一: %s", run_id)
    path = OUT / "equity" / f"eq__{run_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}")
    return run_id, pd.read_csv(path, index_col=0, parse_dates=True)


def load_benchmark(days: pd.DatetimeIndex) -> pd.Series:
    from config import Config
    p = Path(str(Config.cache()["root"])) / f"index_daily_{BENCH_INDEX.replace('.', '_')}.parquet"
    df = pd.read_parquet(p)
    # 过滤供应商当日零值占位行
    close = df.xs(BENCH_INDEX, level="code")["close"]
    close = close[(close > 0)].sort_index()
    ret = close.pct_change().fillna(0.0)
    return ret.reindex(days).fillna(0.0)


def _sharpe(dr: pd.Series) -> float:
    sd = dr.std()
    return float(dr.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def stats_block(dr: pd.Series, bench: pd.Series, turnover: pd.Series | None) -> dict:
    n = len(dr)
    cum = float((1 + dr).prod() - 1)
    ann = float((1 + dr).prod() ** (252 / n) - 1)
    b_cum = float((1 + bench).prod() - 1)
    b_ann = float((1 + bench).prod() ** (252 / n) - 1)
    eq = (1 + dr).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    vol = float(dr.std() * np.sqrt(252))
    # β/α（日频回归年化）
    cov = np.cov(dr, bench)[0, 1]
    beta = float(cov / np.var(bench)) if np.var(bench) > 0 else 0.0
    alpha = ann - beta * b_ann
    ex = dr - bench
    ex_sd = ex.std()
    ir = float(ex.mean() / ex_sd * np.sqrt(252)) if ex_sd > 0 else 0.0
    to = turnover.dropna() if turnover is not None else pd.Series(dtype=float)
    calmar = ann / abs(mdd) if mdd < 0 else 0.0
    return {
        "cum": cum, "ann": ann, "excess_cum": cum - b_cum,
        "sharpe": _sharpe(dr), "mdd": mdd, "win": float((dr > 0).mean()),
        "vol": vol, "bench_cum": b_cum, "bench_ann": b_ann,
        "alpha": alpha, "beta": beta, "ir": ir, "calmar": calmar,
        "turnover": float(to.mean()) if len(to) else 0.0,
        "n_days": n,
    }


def yearly_rows(dr: pd.Series, bench: pd.Series) -> list[dict]:
    rows = []
    for y, sub in dr.groupby(dr.index.year):
        b = bench.reindex(sub.index).fillna(0.0)
        if len(sub) < 20:
            continue
        ann = float((1 + sub).prod() ** (252 / len(sub)) - 1)
        b_ann = float((1 + b).prod() ** (252 / len(b)) - 1)
        eq = (1 + sub).cumprod()
        rows.append({
            "year": int(y), "ann": ann, "bench": b_ann, "excess": ann - b_ann,
            "mdd": float((eq / eq.cummax() - 1).min()),
            "sharpe": _sharpe(sub),
        })
    return rows


def _pts(s: pd.Series) -> list[list]:
    s = s.dropna()
    return [[int(t.timestamp() * 1000), round(float(v), 6)]
            for t, v in s.items()]


def _pct(v, digits=2, sign=False):
    if pd.isna(v):
        return "—"
    s = f"{v * 100:+.{digits}f}%" if sign else f"{v * 100:.{digits}f}%"
    return s


def _cls(v):
    return "pos" if v > 0 else ("neg" if v < 0 else "")


def main():
    ap = argparse.ArgumentParser(description="最佳策略聚宽风格收益曲线页")
    ap.add_argument("--run-id", default=None, help="组合 run_id（默认全期超额第一）")
    args = ap.parse_args()

    run_id, eq = load_run(args.run_id)
    dr = eq["daily_ret"].dropna()
    bench = load_benchmark(dr.index)
    st = stats_block(dr, bench, eq.get("turnover"))
    yearly = yearly_rows(dr, bench)

    strat_cum = (1 + dr).cumprod() - 1
    bench_cum = (1 + bench).cumprod() - 1
    excess_cum = strat_cum - bench_cum
    dd = (1 + dr).cumprod()
    dd = dd / dd.cummax() - 1

    data = {
        "strat": _pts(strat_cum), "bench": _pts(bench_cum),
        "excess": _pts(excess_cum), "dd": _pts(dd),
        "st": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in st.items()},
        "yearly": [{**r, "ann": round(r["ann"], 6), "bench": round(r["bench"], 6),
                    "excess": round(r["excess"], 6), "mdd": round(r["mdd"], 6),
                    "sharpe": round(r["sharpe"], 4)} for r in yearly],
    }

    # ---- 顶部统计条 ----
    def card(v: str, label: str, cls: str = "") -> str:
        return (f'<div class="stat"><div class="v {cls}">{v}</div>'
                f'<div class="l">{label}</div></div>')

    strip_cards = [
        card(_pct(st["cum"], 2, True), "累计收益", _cls(st["cum"])),
        card(_pct(st["ann"], 2, True), "年化收益", _cls(st["ann"])),
        card(_pct(st["excess_cum"], 2, True),
             f"超额收益(vs {BENCH_LABEL})", _cls(st["excess_cum"])),
        card(f"{st['sharpe']:.2f}", "夏普比率"),
        card(_pct(st["mdd"]), "最大回撤", "neg"),
        card(_pct(st["win"], 1), "日胜率"),
        card(_pct(st["vol"], 1), "年化波动率"),
        '<div class="stat more-toggle" id="moreToggle">'
        '<div class="v" style="color:#8f9bb3">···</div>'
        '<div class="l">其他指标</div></div>',
    ]
    strip = '<div class="strip">' + "".join(strip_cards) + "</div>"

    extra_cards = [
        card(_pct(st["alpha"], 2, True), "阿尔法 α(年化)", _cls(st["alpha"])),
        card(f"{st['beta']:.2f}", "贝塔 β"),
        card(f"{st['ir']:.2f}", "信息比率 IR"),
        card(f"{st['calmar']:.2f}", "卡玛比率"),
        card(_pct(st["turnover"], 1), "次均换手(每次调仓)"),
        card(_pct(st["bench_cum"], 2, True), "基准累计收益", _cls(st["bench_cum"])),
        card(_pct(st["bench_ann"], 2, True), "基准年化收益", _cls(st["bench_ann"])),
    ]
    strip += ('<div class="strip extra" id="moreStrip" style="display:none">'
              + "".join(extra_cards) + "</div>")

    year_rows = "".join(
        f"<tr><td>{r['year']}</td>"
        f"<td class='{_cls(r['ann'])}'>{_pct(r['ann'], 2, True)}</td>"
        f"<td class='{_cls(r['bench'])}'>{_pct(r['bench'], 2, True)}</td>"
        f"<td class='{_cls(r['excess'])}'>{_pct(r['excess'], 2, True)}</td>"
        f"<td class='neg'>{_pct(r['mdd'])}</td>"
        f"<td>{r['sharpe']:.2f}</td></tr>"
        for r in yearly)

    body = f"""<div class="navbar">
  <span class="title">YuriQuant · 全A滚动训练实验</span>
  <span class="sub">策略 <code>{run_id}</code>
   · 样本外 {str(dr.index[0].date())} ~ {str(dr.index[-1].date())}
   · 初始资金 ¥1,000,000 · 含成本（佣金万3+印花税千1+滑点10bp）</span>
</div>
<div class="wrap">
  <div class="card">{strip}</div>
  <div class="card">
    <div class="tabs">
      <div class="tab active">历史收益</div>
      <div class="ranges">
        <button data-r="all" class="on">全部</button><button data-r="1y">近1年</button>
        <button data-r="6m">近6月</button><button data-r="1m">近1月</button>
      </div>
    </div>
    <div class="chartbox"><div class="main"><canvas id="mainChart"></canvas></div></div>
    <div class="chartbox" style="border-top:1px dashed #e3e8ee">
      <div class="legendline">回撤（策略净值相对历史高点）</div>
      <div class="sub"><canvas id="ddChart"></canvas></div>
    </div>
  </div>
  <div class="card">
    <div class="tabs"><div class="tab active">分年绩效</div></div>
    <table><thead><tr><th>年份</th><th>策略收益</th><th>基准收益</th><th>超额收益</th>
      <th>年内最大回撤</th><th>夏普</th></tr></thead>
      <tbody>{year_rows}</tbody></table>
  </div>
  <div class="card"><div class="note">
    口径：策略为全A 股票池（含已退市股，SDK 历史清单回补、无幸存者偏差）的量价+基本面
    公因子 walk-forward 滚动训练样本外组合（每年特征选择与训练只用当年之前数据，季度再训练；
    TopFrac 等权多头，涨跌停封板/停牌/ST 不可交易过滤，含交易成本）。
    基准 {BENCH_LABEL}：用户指定中证全指（000985），但数据源该代码行情仅到 2016-06，
    改用成分覆盖等价的国证A指（399317.SZ，全市场A股）。
    组合为 ~500-1000 只等权分散持仓的研究型组合，容量与冲击成本未建模。
  </div></div>
</div>
<script>
const DATA = {json.dumps(data, ensure_ascii=False)};
const ALL = {json.dumps({
    "strat": data["strat"], "bench": data["bench"],
    "excess": data["excess"], "dd": data["dd"]}, ensure_ascii=False)};
const ts = t => new Date(t).toLocaleDateString('zh-CN');
const pctTick = v => (v*100).toFixed(0)+'%';

function sliceRange(arr, months) {{
  if (!months) return arr;
  if (!arr.length) return arr;
  const end = arr[arr.length-1][0];
  const start = end - months*30.5*86400*1000;
  return arr.filter(p => p[0] >= start);
}}
function applyRange(months) {{
  [ 'strat','bench','excess','dd' ].forEach(k => {{
    const d = chartMain.data.datasets.find(x => x.k === k);
    d.data = sliceRange(ALL[k], months);
    const d2 = chartDD.data.datasets[0];
    if (k === 'dd') d2.data = sliceRange(ALL[k], months);
    else d2.data = sliceRange(ALL.dd, months);
  }});
  chartMain.update('none'); chartDD.update('none');
}}

const scaleOpts = {{ ticks: {{ callback: pctTick }} , grid: {{ color: '#f0f3f7' }} }};
const chartMain = new Chart(document.getElementById('mainChart'), {{
  type: 'line',
  data: {{ datasets: [
    {{ k:'strat', label:'策略收益', data: DATA.strat, borderColor: '{C_STRAT}',
       borderWidth: 1.8, pointRadius: 0, tension: 0.08, fill: false }},
    {{ k:'bench', label:'基准收益（{BENCH_LABEL}）', data: DATA.bench, borderColor: '{C_BENCH}',
       borderWidth: 1.5, pointRadius: 0, tension: 0.08, fill: false }},
    {{ k:'excess', label:'超额收益', data: DATA.excess, borderColor: '{C_EXCESS}',
       borderWidth: 1.2, pointRadius: 0, tension: 0.08, fill: false,
       borderDash: [4, 3] }},
  ]}},
  options: {{
    animation: false, responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ labels: {{ usePointStyle: false, boxWidth: 24, font: {{size: 12}} }} }},
      tooltip: {{ callbacks: {{
        label: c => c.dataset.label + ': '
                 + (c.parsed.y * 100).toFixed(2) + '%',
        title: it => ts(it[0].parsed.x) }} }}
    }},
    scales: {{
      x: {{ type: 'linear', grid: {{ display: false }},
           ticks: {{ callback: v => new Date(v).toISOString().slice(0, 7),
                    maxRotation: 0, autoSkipPadding: 30 }} }},
      y: {{ ...scaleOpts }}
    }}
  }}
}});
const chartDD = new Chart(document.getElementById('ddChart'), {{
  type: 'line',
  data: {{ datasets: [{{ label: '回撤', data: DATA.dd, borderColor: '{C_DD}',
     backgroundColor: 'rgba(157,179,201,0.35)', borderWidth: 1.2, pointRadius: 0,
     fill: true, tension: 0.08 }}]}},
  options: {{
    animation: false, responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{ legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: c => '回撤: ' + (c.parsed.y*100).toFixed(2) + '%',
                              title: it => ts(it[0].parsed.x) }} }} }},
    scales: {{
      x: {{ type: 'linear', grid: {{ display: false }},
           ticks: {{ callback: v => new Date(v).toISOString().slice(0, 7),
                    maxRotation: 0, autoSkipPadding: 30 }} }},
      y: {{ ticks: {{ callback: pctTick }}, grid: {{ color: '#f0f3f7' }} }}
    }}
  }}
}});
document.querySelectorAll('.ranges button').forEach(b => {{
  b.onclick = () => {{
    document.querySelectorAll('.ranges button').forEach(x => x.classList.remove('on'));
    b.classList.add('on');
    applyRange(b.dataset.r === 'all' ? null : parseInt(b.dataset.r));
  }};
}});
document.getElementById('moreToggle').onclick = () => {{
  const s = document.getElementById('moreStrip');
  s.style.display = s.style.display === 'none' ? 'flex' : 'none';
}};
</script>
"""
    html = page(f"{run_id} · 收益曲线（vs {BENCH_LABEL}）", header="", body=body,
                css=_CSS, head_extra=_CHART_CDN)

    out = OUT / "best_strategy_jq.html"
    out.write_text(html, encoding="utf-8")
    log.info("生成: %s（%.2f MB）| %s 累计 %s vs 基准 %s | 超额 %s",
             out.resolve(), out.stat().st_size / 1e6, run_id,
             _pct(st["cum"]), _pct(st["bench_cum"]), _pct(st["excess_cum"], 2, True))


if __name__ == "__main__":
    main()
