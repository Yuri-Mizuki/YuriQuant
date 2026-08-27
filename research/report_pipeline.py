"""
端到端研究报告管线（插件化收集器 + 统一组装器）
==================================================

把散落在 reports/ 下的各层产物（CSV/JSON/parquet）收集起来，组装成单个自包含
HTML 研究报告 —— 浏览器直接打开，Chart.js 交互图表，带时间戳与 git 版本。

核心设计：**插件化收集器**。每个产物来源对应一个 ``collect_xxx()`` 函数，注册到
``COLLECTORS`` 字典；组装器遍历字典，自动发现并组装。加新实验报告只需写一个
collect 函数 + 注册一行，不碰任何已有代码。

收集函数统一契约::

    def collect_xxx(ctx: ReportContext) -> Section | None:
        '''读已有产物文件 → 返回 Section，产物缺失返回 None（自动跳过）'''

    ctx:  ReportContext（dataset, report_dir, git_hash, timestamp）
    返回: Section(title, html, order) 或 None

章节顺序由 COLLECTORS 的 key 顺序决定（有序字典）。

用法::

    from research.report_pipeline import generate_research_report
    generate_research_report("hs300_2025", out="reports/research_report.html")

    # 或 CLI
    python scripts/generate_report.py --dataset hs300_2025
"""
from __future__ import annotations

import json
import logging
import subprocess
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("report_pipeline")

# ===========================================================================
# 数据结构
# ===========================================================================
@dataclass
class ReportContext:
    """报告上下文，传给每个 collect 函数。"""
    dataset: str
    report_dir: Path
    git_hash: str = ""
    timestamp: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class Section:
    """一个报告章节。组装器按 order 排序后拼接。"""
    title: str
    html: str                       # 已渲染的 HTML 片段（不含 <html>/<body>）
    order: int = 0                  # 章节排序（小→大）
    charts_js: str = ""             # 可选：Chart.js 图表初始化 JS


# ===========================================================================
# 工具函数
# ===========================================================================
def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    """容错读 CSV，文件不存在/格式错返回 None。"""
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception as e:
        log.warning("读取 CSV 失败 %s: %s", path, e)
    return None


def _safe_read_json(path: Path) -> dict | None:
    try:
        if path.exists() and path.stat().st_size > 0:
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("读取 JSON 失败 %s: %s", path, e)
    return None


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "-"
    return f"{float(v) * 100:.2f}%"


def _fmt_num(v, dp: int = 4) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "-"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return f"{float(v):.{dp}f}"


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 30,
                      pct_cols: set | None = None) -> str:
    """DataFrame → 可排序 HTML 表（精简版，复用 html_report CSS）。"""
    pct_cols = pct_cols or set()
    if len(df) > max_rows:
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
                elif isinstance(v, (int, np.integer)):
                    txt = f"{int(v):,}"
                else:
                    txt = f"{float(v):.4f}"
            elif isinstance(v, str):
                txt = v
            cells.append(f"<td>{txt}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    n_more = f"<div class='note'>仅展示前 {max_rows} 行，共 {len(df)} 行</div>" if len(df) > max_rows else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>{n_more}"


def _chart_js_line(canvas_id: str, labels: list, datasets: list,
                   y_fmt: str = "value", title: str = "") -> str:
    """生成一个 Chart.js 折线图初始化 JS 代码块。

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
                "x": {"type": "time", "time": {"unit": "month"},
                      "grid": {"display": False}},
                "y": {"ticks": {"callback": f"v=>v.toFixed(2)"}},
            },
        },
    }
    cfg_json = json.dumps(cfg, ensure_ascii=False, separators=(",", ":"))
    return f"makeChart('{canvas_id}', {cfg_json});\n"


def _git_hash() -> str:
    """获取当前 git commit hash（无 git 环境返回空字符串）。"""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


# ===========================================================================
# 各层收集器（每个返回 Section | None）
# ===========================================================================

def collect_overview(ctx: ReportContext) -> Section | None:
    """概览章节：数据集、因子库规模、实验计数、git 版本。"""
    parts = []
    parts.append(f"<div class='stats'>")
    parts.append(f"<div class='stat'><span>数据集</span><b>{ctx.dataset}</b></div>")
    # 因子库规模
    lib_reg = ctx.report_dir / "factor_library_report.html"
    lib_size = lib_reg.stat().st_size // 1024 if lib_reg.exists() else 0
    parts.append(f"<div class='stat'><span>因子库报告</span><b>{lib_size} KB</b></div>")
    # 实验日志计数
    exp = _safe_read_csv(ctx.report_dir / "experiments.csv")
    n_exp = len(exp) if exp is not None else 0
    parts.append(f"<div class='stat'><span>实验记录</span><b>{n_exp}</b></div>")
    parts.append(f"<div class='stat'><span>生成时间</span><b>{ctx.timestamp}</b></div>")
    if ctx.git_hash:
        parts.append(f"<div class='stat'><span>Git 版本</span><b>{ctx.git_hash}</b></div>")
    parts.append("</div>")

    # 产物清单
    found = []
    checks = [
        ("因子库报告", "factor_library_report.html"),
        ("回测报告", "yuriquant_report.html"),
        ("DPP 筛选对比", "dpp_vs_pairwise_hs300_2025_cross.csv"),
        ("合成对比", "gflownet_vs_gp_synthesis.csv"),
        ("组合方法对比", "portfolio_methods_compare.csv"),
        ("滚动窗口汇总", "walk_forward/summary.csv"),
        ("OOS 方法对比", "oos_method_comparison.html"),
        ("实验日志", "experiments.csv"),
        ("两期对比", "two_periods/summary.csv"),
    ]
    for label, rel in checks:
        p = ctx.report_dir / rel
        ok = p.exists()
        found.append(
            f"<tr><td>{label}</td><td><code>{rel}</code></td>"
            f"<td>{'存在' if ok else '<span style=color:#999>缺失</span>'}</td></tr>"
        )
    parts.append(f"<h3>报告产物清单</h3><table><thead><tr><th>产物</th><th>路径</th><th>状态</th></tr></thead>"
                  f"<tbody>{''.join(found)}</tbody></table>")
    return Section("概览", "".join(parts), order=-100)


def collect_factor_library(ctx: ReportContext) -> Section | None:
    """因子库全览：从 registry 读规模/来源/IC 分布。"""
    try:
        from research.factor_library import FactorLibrary
        lib = FactorLibrary(dataset=ctx.dataset)
        reg = lib.list_all()
    except Exception:
        return None
    if reg.empty:
        return None
    parts = [f"<div class='stats'>"]
    parts.append(f"<div class='stat'><span>因子总数</span><b>{len(reg)}</b></div>")
    if "kind" in reg.columns:
        for k, g in reg.groupby("kind"):
            parts.append(f"<div class='stat'><span>{k}</span><b>{len(g)}</b></div>")
    if "ic_mean" in reg.columns:
        ic = reg["ic_mean"].dropna()
        if len(ic):
            parts.append(f"<div class='stat'><span>IC 均值中位</span><b>{ic.median():.4f}</b></div>")
    if "significant" in reg.columns:
        n_sig = int(reg["significant"].sum()) if reg["significant"].dtype == bool else 0
        parts.append(f"<div class='stat'><span>显著因子</span><b>{n_sig}</b></div>")
    parts.append("</div>")

    # top 20 因子
    show = reg.copy()
    if "ic_ir" in show.columns:
        show = show.sort_values("ic_ir", ascending=False)
    cols = [c for c in ("name", "family", "ic_mean", "ic_ir", "t_stat_nw",
                        "best_sharpe", "maturity") if c in show.columns]
    show = show[cols].head(20)
    parts.append(f"<h3>Top 20 因子（按 IR）</h3>")
    parts.append(_df_to_html_table(show, max_rows=20))
    return Section("因子库全览", "".join(parts), order=0)


def collect_dpp(ctx: ReportContext) -> Section | None:
    """DPP 集合级筛选对比。"""
    # 尝试 cross 优先，fallback 到无后缀
    for suffix in ("_cross", ""):
        df = _safe_read_csv(ctx.report_dir / f"dpp_vs_pairwise_hs300_2025{suffix}.csv")
        if df is not None:
            break
    if df is None:
        return None
    parts = ["<p>DPP 集合级多样性筛选 vs 两两贪心去重对比（真实因子池实测）：</p>"]
    parts.append(_df_to_html_table(df, max_rows=10))
    parts.append("<div class='note'>log-det 越接近 0 → 信息空间越大、冗余越低</div>")
    return Section("DPP 多样性筛选", "".join(parts), order=10)


def collect_synthesis(ctx: ReportContext) -> Section | None:
    """多因子合成方法对比。"""
    # 优先 gflownet_vs_gp_synthesis，fallback 到 synthesis_*.csv
    df = _safe_read_csv(ctx.report_dir / "gflownet_vs_gp_synthesis.csv")
    if df is None:
        # glob 匹配
        for p in sorted(ctx.report_dir.glob("synthesis_*.csv"), reverse=True):
            df = _safe_read_csv(p)
            if df is not None:
                break
    if df is None:
        return None
    parts = ["<p>各合成方法（IC 加权 / PCA / 正交化 / GBDT）在不同因子池上的对比：</p>"]
    parts.append(_df_to_html_table(df, max_rows=20))
    return Section("多因子合成对比", "".join(parts), order=20)


def collect_portfolio(ctx: ReportContext) -> Section | None:
    """组合优化方法对比。"""
    df = _safe_read_csv(ctx.report_dir / "portfolio_methods_compare.csv")
    if df is None:
        # glob 匹配 real
        for p in sorted(ctx.report_dir.glob("portfolio_methods_real_*.csv")):
            df = _safe_read_csv(p)
            if df is not None:
                break
    if df is None:
        return None
    parts = ["<p>组合优化方法（min_var / tev / mvo / 风险平价 / BL / HRP）对比：</p>"]
    parts.append(_df_to_html_table(df, max_rows=20))
    return Section("组合优化对比", "".join(parts), order=30)


def collect_walk_forward(ctx: ReportContext) -> Section | None:
    """滚动窗口训练评估。"""
    df = _safe_read_csv(ctx.report_dir / "walk_forward" / "summary.csv")
    if df is None:
        df = _safe_read_csv(ctx.report_dir / "walk_forward_A" / "summary.csv")
    if df is None:
        return None
    parts = ["<p>滚动窗口训练评估（训练→测试逐窗口递进）：</p>"]
    parts.append(_df_to_html_table(df, max_rows=20))
    return Section("滚动窗口评估", "".join(parts), order=40)


def collect_oos(ctx: ReportContext) -> Section | None:
    """OOS 合成方法对比 + 选股曲线。"""
    df = _safe_read_csv(ctx.report_dir / "oos_selection_summary.csv")
    if df is None:
        return None
    parts = ["<p>样本外合成方法对比（TabICL 滚动 vs 静态 vs 基线）：</p>"]
    parts.append(_df_to_html_table(df, max_rows=20))
    # 净值曲线 CSV → Chart.js（兼容长格式与宽格式）
    curves = _safe_read_csv(ctx.report_dir / "oos_selection_curves.csv")
    charts_js = ""
    if curves is not None and len(curves) > 2:
        if "nav" in curves.columns and "date" in curves.columns:
            # 长格式：window,method,variant,date,nav → 按 variant/method 透视
            label_col = None
            for cand in ("variant", "method"):
                if cand in curves.columns:
                    label_col = cand
                    break
            datasets = []
            if label_col:
                for label, g in curves.groupby(label_col):
                    g = g[["date", "nav"]].dropna()
                    if len(g) > 5:
                        datasets.append({
                            "label": str(label),
                            "data": [[str(r["date"]), float(r["nav"])] for _, r in g.iterrows()],
                            "borderWidth": 1.3, "pointRadius": 0, "fill": False, "tension": 0.1,
                        })
        elif "date" in curves.columns:
            # 宽格式：date,method1,method2,...
            datasets = []
            for col in [c for c in curves.columns if c != "date"]:
                s = curves[["date", col]].dropna()
                if len(s) > 5:
                    try:
                        datasets.append({
                            "label": col,
                            "data": [[str(r["date"]), float(r[col])] for _, r in s.iterrows()],
                            "borderWidth": 1.3, "pointRadius": 0, "fill": False, "tension": 0.1,
                        })
                    except (ValueError, TypeError):
                        pass
        if datasets:
            parts.append(f"<h3>样本外净值曲线</h3><div style='position:relative;height:300px'>"
                         f"<canvas id='oos_curves' role='img' aria-label='OOS 净值曲线'></canvas></div>")
            charts_js = _chart_js_line("oos_curves", [], datasets)
    return Section("OOS 合成对比", "".join(parts), order=50, charts_js=charts_js)


def collect_two_periods(ctx: ReportContext) -> Section | None:
    """两期对比（训练期 vs 测试期）。"""
    df = _safe_read_csv(ctx.report_dir / "two_periods" / "summary.csv")
    if df is None:
        return None
    parts = ["<p>两期对比（训练段 fit → 测试段评估）：</p>"]
    parts.append(_df_to_html_table(df, max_rows=20))
    return Section("两期对比", "".join(parts), order=45)


def collect_experiments(ctx: ReportContext) -> Section | None:
    """实验日志（元数据汇总）。"""
    df = _safe_read_csv(ctx.report_dir / "experiments.csv")
    if df is None:
        return None
    parts = [f"<p>共 {len(df)} 条实验记录（最近 20 条）：</p>"]
    show = df.tail(20) if len(df) > 20 else df
    parts.append(_df_to_html_table(show, max_rows=20))
    return Section("实验日志", "".join(parts), order=900)


# ===========================================================================
# 收集器注册表（有序，key 顺序 = 章节顺序）
# ===========================================================================
COLLECTORS: OrderedDict[str, callable] = OrderedDict([
    ("overview", collect_overview),
    ("factor_library", collect_factor_library),
    ("dpp", collect_dpp),
    ("synthesis", collect_synthesis),
    ("portfolio", collect_portfolio),
    ("walk_forward", collect_walk_forward),
    ("oos", collect_oos),
    ("two_periods", collect_two_periods),
    ("experiments", collect_experiments),
])


# ===========================================================================
# 组装器
# ===========================================================================
_CSS = """
body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;padding:26px 30px;background:#f6f7f9;color:#1f2430}
h1{font-size:21px;margin:0 0 4px;font-weight:600}
.sub{color:#6b7280;font-size:13px;margin-bottom:18px}
h2.section{font-size:17px;margin:28px 0 10px;font-weight:600;color:#111827;border-bottom:2px solid #e5e7eb;padding-bottom:6px}
.card{background:#fff;border-radius:12px;padding:16px 20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
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
.note{color:#8a8f98;font-size:12px;margin-top:6px}
canvas{max-height:300px}
code{font-family:Consolas,monospace;font-size:12px;color:#555}
"""

_JS_TEMPLATE = r"""
function makeChart(id, cfg) {
  const el = document.getElementById(id);
  if (!el) return;
  new Chart(el, cfg);
}
document.addEventListener('DOMContentLoaded', () => {
  __CHARTS_JS__
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
});
"""


def generate_research_report(
    dataset: str = "hs300_2025",
    out: str | Path | None = None,
    report_dir: str | Path = "reports",
    title: str | None = None,
    extra_collectors: dict | None = None,
) -> Path:
    """生成端到端研究报告（单 HTML 自包含）。

    Args:
        dataset: 数据集名（如 hs300_2025）。
        out: 输出路径（None=自动带时间戳）。
        report_dir: reports 目录。
        title: 报告标题（None=自动生成）。
        extra_collectors: 额外收集器 {name: func}，合并进 COLLECTORS（增量加入）。

    Returns:
        Path: 生成的 HTML 文件路径。
    """
    rdir = Path(report_dir)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    ctx = ReportContext(
        dataset=dataset,
        report_dir=rdir,
        git_hash=_git_hash(),
        timestamp=ts,
    )
    # 合并收集器（允许外部增量注入）
    collectors = OrderedDict(COLLECTORS)
    if extra_collectors:
        for k, fn in extra_collectors.items():
            collectors[k] = fn

    # 收集各章节
    sections: list[Section] = []
    for name, fn in collectors.items():
        try:
            sec = fn(ctx)
            if sec is not None:
                sections.append(sec)
                log.info("收集 [%s] %s: %d chars", name, sec.title, len(sec.html))
        except Exception as e:
            log.warning("收集器 %s 异常（跳过）: %s", name, e)

    if not sections:
        raise RuntimeError("无可用报告产物——请先跑实验脚本生成 reports/ 下的产物")

    # 按 order 排序
    sections.sort(key=lambda s: s.order)

    # 组装 HTML
    title = title or f"YuriQuant 研究报告 · {dataset}"
    meta = f"数据集: {dataset} · 生成时间: {ts}"
    if ctx.git_hash:
        meta += f" · Git: {ctx.git_hash}"

    section_html = []
    all_charts_js = []
    for sec in sections:
        section_html.append(
            f'<h2 class="section">{sec.title}</h2>'
            f'<div class="card">{sec.html}</div>'
        )
        if sec.charts_js:
            all_charts_js.append(sec.charts_js)

    charts_js_str = "\n  ".join(all_charts_js)
    js = _JS_TEMPLATE.replace("__CHARTS_JS__", charts_js_str)

    # 输出路径
    if out is None:
        ts_file = datetime.now().strftime("%Y%m%d_%H%M")
        out = rdir / f"research_report_{ts_file}.html"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{title}</h1>
<div class="sub">{meta}</div>
{''.join(section_html)}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<script>
{js}
</script>
</body></html>"""
    out.write_text(html, encoding="utf-8")
    log.info("研究报告已生成: %s (%d KB, %d 章节)", out, out.stat().st_size // 1024, len(sections))
    return out
