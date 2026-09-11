"""全A滚动训练实验报告生成器：收益曲线 + 分年绩效 + 维度对比（自包含 HTML）。

消费 scripts/rolling_grid_alla.py 的产物：
- reports/alla_rolling/metrics_overall.csv / metrics_yearly.csv / ic_stats.csv
- reports/alla_rolling/equity/eq__*.csv
- reports/alla_rolling/_base/bench_*.parquet

产出 reports/alla_rolling/report.html（Chart.js CDN + 数据内嵌，A股红涨绿跌）。

用法:
    python scripts/rolling_grid_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cli_common import setup_logging  # noqa: E402
from research.html_report import page  # noqa: E402

log = setup_logging("rolling_grid_report")

OUT = Path("reports") / "alla_rolling"

# 报告主题（经 research.html_report.page 外壳注入；本文件不再自拼 HTML 外壳）
_TITLE = "全A滚动训练实验报告（2018~now）"
_CHART_CDN = ('<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/'
              'dist/chart.umd.min.js"></script>')
_CSS = """
 body { font-family: "Microsoft YaHei", sans-serif; margin: 24px auto; max-width: 1280px;
        color: #222; background: #fafafa; }
 h1 { font-size: 22px; } h2 { font-size: 18px; border-left: 4px solid #c0392b;
      padding-left: 8px; margin-top: 36px; } h3 { font-size: 15px; color: #444; }
 table { border-collapse: collapse; font-size: 12px; width: 100%; margin: 8px 0 20px; }
 th, td { border: 1px solid #ddd; padding: 4px 7px; text-align: right; white-space: nowrap; }
 th { background: #eee; cursor: pointer; position: sticky; top: 0; }
 td.rid { text-align: left; font-family: Consolas, monospace; }
 .pos { color: #c0392b; } .neg { color: #27ae60; }
 .chartwrap { height: 380px; background: #fff; border: 1px solid #ddd; padding: 8px; }
 .note { background: #fff8e1; border-left: 4px solid #f1c40f; padding: 10px 14px;
         font-size: 13px; line-height: 1.7; }
 .meta { color: #666; font-size: 12px; }
"""

# 代表性对比组（run_id 片段匹配）
GROUPS = {
    "horizon 对比（gbdt · 中性化 · Top20% · 各 horizon 合法频率）": [
        "gbdt__h1__M__neut__f0.20", "gbdt__h5__M__neut__f0.20",
        "gbdt__h10__M__neut__f0.20", "gbdt__h20__2M__neut__f0.20",
    ],
    "调仓频率对比（gbdt · h=1 · 中性化 · Top20%）": [
        "gbdt__h1__D__neut__f0.20", "gbdt__h1__W__neut__f0.20",
        "gbdt__h1__M__neut__f0.20",
    ],
    "模型与超参对比（h=1 · 月频 · 中性化 · Top20%）": [
        "ridge__h1__M__neut__f0.20", "gbdt__h1__M__neut__f0.20",
        "gbdt_fast__h1__M__neut__f0.20", "gbdt_deep__h1__M__neut__f0.20",
        "ranker__h1__M__neut__f0.20", "gbdt_w750__h1__M__neut__f0.20",
        "ridge_w750__h1__M__neut__f0.20",
    ],
    "风格中性化对比（gbdt · h=1 · 月频 · Top20%）": [
        "gbdt__h1__M__neut__f0.20", "gbdt__h1__M__raw__f0.20",
    ],
    "持仓集中度对比（gbdt · h=1 · 月频 · 中性化）": [
        "gbdt__h1__M__neut__f0.20", "gbdt__h1__M__neut__f0.10",
    ],
}
HEATMAP_FILTER = dict(neut=True, frac=0.20)   # 热力图用规范子集
LINE_COLORS = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#16a085",
               "#2980b9", "#8e44ad", "#7f8c8d", "#d35400", "#2c3e50"]


def load_run_ids() -> list[str]:
    return sorted(p.stem[3:] for p in (OUT / "equity").glob("eq__*.csv"))


def curve_json(run_id: str, col: str = "equity") -> list[list]:
    p = OUT / "equity" / f"eq__{run_id}.csv"
    eq = pd.read_csv(p, index_col=0, parse_dates=True)
    s = eq[col].dropna()
    if col == "equity":
        s = s / s.iloc[0]      # 归一化
    w = s.resample("W").last().dropna()
    ts = (w.index.astype("int64") // 10**6).tolist()
    return list(zip(ts, [round(float(x), 5) for x in w.values]))


def bench_curve() -> tuple[list, list]:
    oos_start = pd.Timestamp("2018-01-01")
    idx = pd.read_parquet(OUT / "_base" / "bench_index.parquet")["ret"]
    eqw = pd.read_parquet(OUT / "_base" / "bench_eqw.parquet")["ret"]
    out = []
    for name, s in (("上证指数", idx), ("全A等权", eqw)):
        s = s[s.index >= oos_start]
        w = ((1 + s.fillna(0)).cumprod()).resample("W").last().dropna()
        w = w / w.iloc[0]
        out.append(list(zip((w.index.astype("int64") // 10**6).tolist(),
                            [round(float(x), 5) for x in w.values])))
    return out[0], out[1]


def pct(v, digits=1) -> str:
    return "—" if pd.isna(v) else f"{v * 100:.{digits}f}%"


def f2(v) -> str:
    return "—" if pd.isna(v) else f"{v:.2f}"


def overall_table_html(mo: pd.DataFrame) -> str:
    mo = mo.sort_values(["excess_idx"], ascending=False)
    head = ["配置", "年化", "超额(指数)", "超额(等权)", "Sharpe", "t(SR)",
            "t(超额)", "证明年数", "IR", "MaxDD",
            "换手/次", "β"]
    rows = []
    for _, r in mo.iterrows():
        t_sr = r.get("sharpe_t_stat", np.nan)
        t_exc = r.get("excess_t_stat", np.nan)
        y2p = r.get("years_to_prove", np.nan)
        rows.append(
            f"<tr><td class='rid'>{r['run_id']}</td>"
            f"<td class='pos'>{pct(r['annual'])}</td>"
            f"<td class='{'pos' if r['excess_idx'] >= 0 else 'neg'}'>{pct(r['excess_idx'])}</td>"
            f"<td class='{'pos' if r['excess_eqw'] >= 0 else 'neg'}'>{pct(r['excess_eqw'])}</td>"
            f"<td>{f2(r['sharpe'])}</td>"
            f"<td class='{'pos' if t_exc >= 1.96 else ''}'>{f2(t_sr)}</td>"
            f"<td class='{'pos' if t_exc >= 1.96 else ''}'>{f2(t_exc)}</td>"
            f"<td>{'—' if pd.isna(y2p) or np.isinf(y2p) else f'{y2p:.1f}'}</td>"
            f"<td>{f2(r['ir'])}</td>"
            f"<td class='neg'>−{pct(r['max_dd'])}</td>"
            f"<td>{pct(r['turnover'])}</td><td>{f2(r['beta'])}</td></tr>")
    return ("<table class='sortable'><thead><tr>" +
            "".join(f"<th>{h}</th>" for h in head) +
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def heatmap_html(my: pd.DataFrame) -> str:
    sub = my[(my["neut"] == HEATMAP_FILTER["neut"]) &
             (my["frac"] == HEATMAP_FILTER["frac"])]
    pv_ann = sub.pivot_table(index="run_id", columns="year", values="annual")
    pv_exc = sub.pivot_table(index="run_id", columns="year", values="excess")
    years = sorted(pv_exc.columns)
    # 全期超额与正超额年数
    agg = (sub.groupby("run_id")
           .agg(mean_excess=("excess", "mean"), pos_years=("excess", lambda s: int((s > 0).sum())),
                n_years=("excess", "size"))
           .sort_values("mean_excess", ascending=False))
    head = (["配置"] + [str(y) for y in years] +
            ["均值超额", "正超额年数"])
    rows = []
    vmax = float(np.nanmax(np.abs(pv_exc.values))) or 1.0
    for rid, r in agg.iterrows():
        cells = []
        for y in years:
            e = pv_exc.loc[rid, y] if rid in pv_exc.index and y in pv_exc.columns else np.nan
            a = pv_ann.loc[rid, y] if rid in pv_ann.index else np.nan
            if pd.isna(e):
                cells.append("<td>—</td>")
            else:
                alpha = min(1.0, abs(e) / vmax) * 0.75 + 0.12
                color = (f"rgba(192,57,43,{alpha:.2f})" if e >= 0
                         else f"rgba(39,174,96,{alpha:.2f})")
                txt = "#fff" if alpha > 0.45 else "#222"
                cells.append(
                    f"<td style='background:{color};color:{txt}' "
                    f"title='年化 {pct(a)} / 超额 {pct(e)}'>{e * 100:+.1f}</td>")
        mean_e = r["mean_excess"]
        rows.append(
            f"<tr><td class='rid'>{rid}</td>" + "".join(cells) +
            f"<td class='{'pos' if mean_e >= 0 else 'neg'}'>{mean_e * 100:+.1f}%</td>"
            f"<td>{int(r['pos_years'])}/{int(r['n_years'])}</td></tr>")
    return ("<table class='heat'><thead><tr>" +
            "".join(f"<th>{h}</th>" for h in head) +
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def ic_table_html(ic: pd.DataFrame) -> str:
    if ic is None or ic.empty:
        return "<p>无 IC 统计。</p>"
    years = [c for c in ic.columns if c.startswith("ic_20")]
    head = ["模型", "horizon", "IC均值", "ICIR(年化)", "IC>0占比"] + \
        [c[3:] for c in years]
    rows = []
    for _, r in ic.sort_values(["horizon", "model"]).iterrows():
        rows.append(
            f"<tr><td class='rid'>{r['model']}</td><td>{int(r['horizon'])}</td>"
            f"<td>{r['ic_mean']:+.4f}</td><td>{f2(r['ic_ir'])}</td>"
            f"<td>{pct(r['ic_win'])}</td>" +
            "".join(f"<td>{r[c]:+.4f}</td>" if pd.notna(r[c]) else "<td>—</td>"
                    for c in years) + "</tr>")
    return ("<table class='sortable'><thead><tr>" +
            "".join(f"<th>{h}</th>" for h in head) +
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def group_chart_html(title: str, run_ids: list[str], bench_idx_pts,
                     bench_eqw_pts) -> str:
    datasets = []
    for i, rid in enumerate(run_ids):
        p = OUT / "equity" / f"eq__{rid}.csv"
        if not p.exists():
            continue
        datasets.append({
            "label": rid, "data": curve_json(rid),
            "borderColor": LINE_COLORS[i % len(LINE_COLORS)],
            "borderWidth": 1.8, "pointRadius": 0, "tension": 0.1, "fill": False})
    datasets.append({"label": "上证指数", "data": bench_idx_pts,
                     "borderColor": "#555", "borderDash": [6, 3],
                     "borderWidth": 1.2, "pointRadius": 0, "fill": False})
    datasets.append({"label": "全A等权", "data": bench_eqw_pts,
                     "borderColor": "#999", "borderDash": [2, 2],
                     "borderWidth": 1.2, "pointRadius": 0, "fill": False})
    cid = "chart_" + str(abs(hash(title)) % 10**8)
    ds_json = json.dumps(datasets, ensure_ascii=False, default=str)
    return f"""
<h3>{title}</h3><div class="chartwrap"><canvas id="{cid}"></canvas></div>
<script>
new Chart(document.getElementById('{cid}'), {{
  type: 'line',
  data: {{ datasets: {ds_json} }},
  options: {{
    animation: false, responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: 'nearest', intersect: false }},
    scales: {{
      x: {{ type: 'linear', ticks: {{ callback: v => new Date(v).toISOString().slice(0,7) }} }},
      y: {{ ticks: {{ callback: v => v.toFixed(2) }} }}
    }}
  }}
}});
</script>"""


def yearly_bar_html(my: pd.DataFrame, run_ids: list[str]) -> str:
    """代表配置的分年超额柱状图。"""
    datasets = []
    for i, rid in enumerate(run_ids):
        sub = my[my["run_id"] == rid].set_index("year")["excess"]
        datasets.append({
            "label": rid,
            "data": {str(int(y)): round(float(v) * 100, 2)
                     for y, v in sub.items()},
            "backgroundColor": LINE_COLORS[i % len(LINE_COLORS)],
        })
    cid = "chart_yearly_bar"
    ds_json = json.dumps(datasets, ensure_ascii=False, default=str)
    return f"""
<h3>分年超额收益（vs 上证指数，%）</h3><div class="chartwrap"><canvas id="{cid}"></canvas></div>
<script>
new Chart(document.getElementById('{cid}'), {{
  type: 'bar',
  data: {{ datasets: {ds_json} }},
  options: {{
    animation: false, responsive: true, maintainAspectRatio: false,
    scales: {{ x: {{ stacked: false }}, y: {{ ticks: {{ callback: v => v + '%' }} }} }}
  }}
}});
</script>"""


def robust_list_html(robust: pd.DataFrame) -> str:
    if not len(robust):
        return "<li>（无）</li>"
    items = []
    for rid, r in robust.head(12).iterrows():
        items.append(f"<li><code>{rid}</code>: 平均超额 "
                     f"{r['mean_excess'] * 100:+.1f}%/年, "
                     f"正超额 {int(r['pos'])}/{int(r['n'])} 年</li>")
    return "".join(items)


def main():
    mo = pd.read_csv(OUT / "metrics_overall.csv")
    my = pd.read_csv(OUT / "metrics_yearly.csv")
    icp = OUT / "ic_stats.csv"
    ic = pd.read_csv(icp) if icp.exists() else pd.DataFrame()
    bench_idx_pts, bench_eqw_pts = bench_curve()
    run_ids = load_run_ids()
    log.info("产物: %d 组合, %d 分年行, %d IC 行", len(mo), len(my), len(ic))

    charts = []
    for title, rids in GROUPS.items():
        charts.append(group_chart_html(title, rids, bench_idx_pts, bench_eqw_pts))
    rep_runs = [r for r in ["gbdt__h1__M__neut__f0.20", "gbdt__h5__M__neut__f0.20",
                            "ranker__h1__M__neut__f0.20"] if r in set(mo["run_id"])]
    charts.append(yearly_bar_html(my, rep_runs))

    n_years = my["year"].nunique()
    best = (my.groupby("run_id")["excess"].agg(["mean", lambda s: int((s > 0).sum()), "size"])
            .rename(columns={"mean": "mean_excess", "<lambda_0>": "pos", "size": "n"})
            .sort_values("mean_excess", ascending=False))
    robust = best[(best["pos"] >= max(3, best["n"].max() - 1)) & (best["mean_excess"] > 0)]

    body = f"""<h1>全A多年度滚动训练实验报告（2018 ~ now）</h1>
<p class="meta">生成于 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
股票池：全A（数据缓存内全部 A 股，{len(run_ids)} 个组合配置） ·
预测评估年数：{n_years} 年 · 基准：上证指数 000001.SH 与 全A等权</p>

<div class="note">
<b>实验协议（模拟真实预测场景）</b>：对每个目标年 Y，特征选择（IC 质量窗 500 交易日 +
覆盖率 + 相关去冗余 Top50）与模型训练只用 Y 年之前的数据；年内按季度再训练
（walk-forward 前推折，embargo=horizon），拼接 2018~今连续 OOS 信号。
回测含成本（佣金万3 + 印花税千1卖出 + 滑点 10bp）、涨跌停封板/停牌/ST 不可交易过滤，
月频/周频/日频调仓按引擎「收盘信号、次日生效」口径。<br>
<b>诚实披露</b>：① 股票池含退市股——按 SDK 历史清单（沪深A 2015 至今含退市 ∪ 当前全A）
回补，238 只退市类代码中 237 只有完整历史 K 线（1 只 000562.SZ 于 2015-01 换股退市、
实验窗口内本无行情），基本面/股本/复权因子同步回补，幸存者偏差已消除；
② 状态表已按完整历史清单全量重拉（2015 至今，分年覆盖 2814→5572 单调递增），
涨跌停封板/停牌/ST 掩码在全部年份生效（不可交易占比 3%~5.5%）；
③ 本网格为验证型（全量报告所有配置，非挑优后报告）；④ 模型超参沿用项目固化默认，
未在本数据上调参；⑤ h&gt;1 组合由引擎"区间几何均摊"逐日结算，显式压低了日收益
波动——其 Sharpe 与 h=1 不可直接横向比较（跨 horizon 请用年化收益/超额对比）；
⑥ 特征筛选用 DPP（行列式点过程，质量 500 日 IC + 覆盖率 Top50 去冗余）逐 horizon
独立选取，各 horizon 每年均选出 50 个互补特征，解决了旧版 pairwise 贪心去重在长
horizon 只剩 3~16 特征的连锁误杀问题；⑦ 基本面/股东因子面板曾混入 32 只 ETF 列
（财务表含基金数据），已剔除并重算对应 IC；样本末端 2026-09-02 为盘中半拉数据，
已从全链路裁剪，水位回退至 2026-09-01 待整日重拉。
</div>

<h2>一、全部配置总览（按超额排序，点击表头可排序）</h2>
{overall_table_html(mo)}

<h2>二、模型 OOS 预测力（IC，分 horizon × 模型，分年）</h2>
{ic_table_html(ic)}

<h2>三、分年超额热力图（中性化 · Top20% 子集；数值=年内超额 vs 上证指数 %）</h2>
{heatmap_html(my)}

<h2>四、维度对比收益曲线（净值归一，含双基准）</h2>
{''.join(charts)}

<h2>五、稳健性小结</h2>
<div class="note">
全期平均超额为正且正超额年数 ≥ {max(3, int(best['n'].max()) - 1)} 的配置（共 {len(robust)} 个）：
<ul>
{robust_list_html(robust)}
</ul>
</div>

<p class="meta">数据与产物目录：reports/alla_rolling/（metrics_overall.csv ·
metrics_yearly.csv · ic_stats.csv · equity/*.csv · pred/*.parquet）</p>
"""
    html = page(_TITLE, header="", body=body, css=_CSS, head_extra=_CHART_CDN)
    out = OUT / "report.html"
    out.write_text(html, encoding="utf-8")
    log.info("报告已生成: %s（%.1f MB）", out.resolve(), out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
