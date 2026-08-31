"""监控编排 —— 扫描因子库 → 计算指标 → 触发告警 → 落盘账本 → HTML 报告。

调度入口（scripts/monitor_performance.py）：
- 单次：cron / Windows 计划任务每日调 ``monitor_performance.py --dataset ...``
- 常驻：``monitor_performance.py --daemon 17:30``（stdlib 循环，无额外依赖）
- ``next_run_time`` 为纯函数，跨零点正确，可单测。
"""

from __future__ import annotations

import html as _html
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.metrics import PERIODS_PER_YEAR
from config import Config
from monitoring.alerts import attach_alerts
from monitoring.ledger import MonitoringLedger
from monitoring.metrics import (
    MonitorMetrics,
    compute_factor_metrics,
    load_close_panel,
    load_returns_panel,
)
from monitoring.state import confirm_rows
from research.html_report import (
    page,
    svg_sparkline as _sparkline,
    svg_sparkline_monthly as _sparkline_monthly,
)

log = logging.getLogger("monitoring")


def next_run_time(now: datetime, hhmm: str) -> datetime:
    """下一次调度时刻（纯函数）：now 早于今日 hhmm → 今日，否则明日。"""
    hh, mm = (int(x) for x in hhmm.split(":")[:2])
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return target if target > now else target + timedelta(days=1)


def run_monitoring(
    dataset: str = "hs300_2022_2025",
    window: int | None = None,
    window_long: int | None = None,
    as_of: str | None = None,
    factor_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    ledger_root: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
    record: bool = True,
    signal_path: str | Path | None = None,
    confirm_n: int | None = None,
    run_stamp: str | None = None,
) -> dict:
    """跑一轮完整监控，返回汇总 dict（factors/models/告警计数 + 产物路径）。

    Args:
        dataset: 因子库数据集（监控对象全集 = 该库 registry 所有因子）。
        window: 近期短窗（默认 config monitoring.window，如 60d，提前预警）。
        window_long: 近期长窗（默认 config monitoring.window_long，如 252d，稳健确认）。
        as_of: 监控基准日 YYYYMMDD（默认 = 数据源最新交易日）。
        factor_root / cache_root / ledger_root: 路径覆盖（测试隔离用）。
        record: 是否写 experiments 账本。
    """
    from research.factor_library import FactorLibrary

    conf = cfg or Config.monitoring()
    window = int(window or conf["window"])
    window_long = int(window_long or conf.get("window_long", 252))
    factor_root = Path(factor_root) if factor_root else None
    cache_root = Path(cache_root) if cache_root else Path(Config.cache()["root"])

    lib = FactorLibrary(root=factor_root, dataset=dataset)
    registry = lib.list_all()
    if registry.empty:
        raise RuntimeError(f"因子库为空: dataset={dataset} root={factor_root}")

    close = load_close_panel(cache_root)
    as_of_ts = pd.Timestamp(as_of) if as_of else close.index[-1]
    returns = load_returns_panel(close, as_of=as_of_ts)

    snapshots: list[MonitorMetrics] = []
    ic_history: dict[str, pd.Series] = {}
    for _, row in registry.iterrows():
        name = row["name"]
        panel = lib.get_panel(name)
        eval_path = Path(str(row.get("eval_path", "")))
        ic = pd.read_parquet(eval_path)["ic"] if eval_path.exists() else pd.Series(dtype=float)
        if panel is None or ic.empty:
            log.warning("跳过（面板或 eval 缺失）: %s", name)
            continue
        m = compute_factor_metrics(
            name, row, panel, ic, returns, as_of_ts,
            window=window, window_long=window_long,
        )
        m = attach_alerts(m, conf)
        snapshots.append(m)
        ic_history[name] = ic[ic.index <= as_of_ts].tail(PERIODS_PER_YEAR)

    # 库级拥挤度监控：因子 IC 相关矩阵 → 同质化/分散度虚假
    if len(ic_history) >= 2:
        from monitoring.crowding import compute_crowding
        crowd = compute_crowding(ic_history)
        if crowd is not None:
            crowd = attach_alerts(crowd, conf)
            snapshots.append(crowd)
            if crowd.alerts:
                log.warning(
                    "因子拥挤度: IC 平均相关 %.2f / 首主成分解释 %.0f%% 超限 (%d 因子)",
                    crowd.corr_mean_full, crowd.pc1_share_full * 100, crowd.corr_n,
                )

    # 组合/信号层监控：读取每日交易信号产物，折叠成一条 signal 快照
    if signal_path is not None:
        from monitoring.signal_monitor import build_signal_metrics, load_signals
        sig = load_signals(signal_path)
        if not sig.empty:
            sm = build_signal_metrics(sig, returns.index, as_of_ts, conf)
            snapshots.append(sm)
            log.info(
                "信号层监控: 最新信号日=%s 持仓=%d HHI=%.3f 换手=%s",
                sm.signal_date, sm.n_signal_stocks,
                sm.hhi_recent if sm.hhi_recent == sm.hhi_recent else float("nan"),
                "-" if sm.net_turnover != sm.net_turnover else f"{sm.net_turnover:.2f}",
            )
        else:
            log.warning("信号文件为空或不可读: %s", signal_path)

    ledger_root = ledger_root or conf["ledger_root"]
    ledger = MonitoringLedger(Path(ledger_root))
    run_date = run_stamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snap_df = pd.DataFrame([m.as_row() for m in snapshots])
    snap_df.insert(0, "run_date", run_date)
    snap_df.insert(1, "as_of", as_of_ts.strftime("%Y-%m-%d"))
    ledger.append_snapshots(snap_df, run_date)

    alert_rows = []
    for m in snapshots:
        for a in m.alerts:
            alert_rows.append(
                {
                    "run_date": run_date,
                    "as_of": as_of_ts.strftime("%Y-%m-%d"),
                    "name": m.name,
                    "category": m.category,
                    "source": m.source,
                    "rule": a["rule"],
                    "level": a["level"],
                    "message": a["message"],
                }
            )
    confirm_n = int(confirm_n if confirm_n is not None else conf.get("confirm_n", 3))
    # 去抖：只把达到 confirm_n 连续期的告警算作"确认"用于报告与计数；
    # 但所有触发（含观察中）都落台账并带 confirmed 标记，否则历史里看不到
    # 连续次数，永远无法形成确认。
    final_alert_rows, n_pending = confirm_rows(alert_rows, ledger.load_alerts(), confirm_n)
    confirmed = {(r["name"], r["rule"]) for r in final_alert_rows}
    for r in alert_rows:
        r["confirmed"] = (r["name"], r["rule"]) in confirmed
    ledger.append_alerts(pd.DataFrame(alert_rows), run_date)

    report_path = Path(ledger_root) / "monitor_report.html"
    generate_html_report(
        snapshots,
        final_alert_rows,
        ic_history,
        report_path,
        dataset=dataset,
        window=window,
        window_long=window_long,
        as_of=as_of_ts,
        pending_count=n_pending,
    )

    n_model = sum(1 for m in snapshots if m.category == "model")
    summary = {
        "run_date": run_date,
        "as_of": str(as_of_ts.date()),
        "dataset": dataset,
        "n_factors": len(snapshots),
        "n_models": n_model,
        "n_critical": sum(1 for m in snapshots if m.status == "critical"),
        "n_warning": sum(1 for m in snapshots if m.status == "warning"),
        "n_normal": sum(1 for m in snapshots if m.status == "normal"),
        "n_pending": n_pending,
        "snapshots_path": str(ledger.snapshots_path),
        "alerts_path": str(ledger.alerts_path),
        "report_path": str(report_path),
    }

    if record:
        from research.experiments import record_experiment

        record_experiment(
            kind="performance_monitor",
            command=f"monitor_performance --dataset {dataset}",
            params={
                "dataset": dataset,
                "window": window,
                "window_long": window_long,
                "as_of": str(as_of_ts.date()),
            },
            result_path=str(report_path),
            metrics={
                "n_factors": summary["n_factors"],
                "n_models": n_model,
                "n_critical": summary["n_critical"],
                "n_warning": summary["n_warning"],
            },
            note="生产化监控：因子与模型预测性能快照 + 告警 + HTML 报告",
        )
    log.info("监控完成: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# HTML 报告（自包含：无外部 JS/CSS 依赖，inline SVG sparkline）
# ---------------------------------------------------------------------------

_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #FAFAF8;
       color: #1F241C; font-size: 14px; line-height: 1.6; padding: 2rem 1rem 4rem; }
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin-bottom: 0.3rem; }
.meta { color: #6B7065; font-size: 0.82rem; font-family: Consolas, monospace;
        margin-bottom: 1.6rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
         gap: 0.8rem; margin-bottom: 1.6rem; }
.card { background: #F0EFE9; border-radius: 8px; padding: 0.9rem 1.1rem;
        border: 1px solid #E0DFD5; }
.card .v { font-size: 1.6rem; font-weight: 700; font-family: Consolas, monospace; }
.card .l { font-size: 0.75rem; color: #6B7065; }
.card.crit .v { color: #B3402A; }
.card.warn .v { color: #B07A1E; }
.card.ok .v { color: #0E6E5C; }
h2 { font-size: 1.05rem; margin: 1.8rem 0 0.7rem; padding-bottom: 0.35rem;
     border-bottom: 2px solid #0E6E5C; }
table { width: 100%; border-collapse: collapse; font-size: 0.8rem;
        margin: 0.6rem 0; background: #fff; }
th { text-align: left; padding: 0.5rem 0.6rem; background: #F0EFE9;
     border-bottom: 2px solid #E0DFD5; font-size: 0.72rem; color: #555A50;
     white-space: nowrap; }
td { padding: 0.42rem 0.6rem; border-bottom: 1px solid #ECEBE2; vertical-align: top; }
td.num { text-align: right; font-family: Consolas, monospace; white-space: nowrap; }
.tag { display: inline-block; padding: 0.05rem 0.5rem; border-radius: 9px;
       font-size: 0.7rem; font-weight: 600; color: #fff; }
.tag.critical { background: #B3402A; }
.tag.warning { background: #C08A2D; }
.tag.normal { background: #3C8375; }
.spark { vertical-align: middle; }
.rule { font-family: Consolas, monospace; font-size: 0.75rem; color: #555A50; }
footer { margin-top: 2.5rem; color: #8A8F84; font-size: 0.75rem; }
details { margin: 0.6rem 0; }
summary { cursor: pointer; font-size: 0.85rem; font-weight: 600;
           padding: 0.3rem 0; color: #0E6E5C; user-select: none; }
details[open] summary { margin-bottom: 0.3rem; }
th.sortable { cursor: pointer; user-select: none; }
th.sortable:hover { background: #E0DFD5; }
th.sortable::after { content: ' ↕'; font-size: 0.65rem; color: #999; }
.scroll-table { max-height: 70vh; overflow: auto; border: 1px solid #E0DFD5;
                border-radius: 4px; position: relative; }
.scroll-table table { margin: 0; }
.scroll-table thead th { position: sticky; top: 0; z-index: 1;
                         background: #F0EFE9; box-shadow: 0 1px 0 #E0DFD5; }
th.sortable.asc::after { content: ' ↑'; color: #0E6E5C; }
th.sortable.desc::after { content: ' ↓'; color: #0E6E5C; }
.filter-bar { margin: 0.6rem 0; display: flex; flex-wrap: wrap; gap: 0.4rem; }
.filter-btn { padding: 0.2rem 0.7rem; border: 1px solid #D5D4CA; border-radius: 6px;
              background: #F0EFE9; font-size: 0.78rem; cursor: pointer; color: #555A50; }
.filter-btn:hover { background: #E0DFD5; }
.filter-btn.active { background: #0E6E5C; color: #fff; border-color: #0E6E5C; }
"""

# 页面交互 JS：表头点击排序（thead/tbody 分离，只排 tbody 行）+ 来源筛选
_JS = """// 表头点击排序：所有表均用 thead/tbody 分离，JS 只排 tbody 行。
document.addEventListener('click', function(e) {
    var th = e.target.closest('th.sortable');
    if (!th) return;
    var table = th.closest('table');
    var tbody = table.querySelector('tbody');
    if (!tbody) return;
    var rows = Array.from(tbody.querySelectorAll('tr'));
    if (!rows.length) return;
    var idx = Array.from(th.parentNode.children).indexOf(th);
    var asc = th.classList.contains('asc');
    th.parentNode.querySelectorAll('th.sortable').forEach(function(t) {
        t.classList.remove('asc', 'desc');
    });
    th.classList.add(asc ? 'desc' : 'asc');
    var dir = asc ? -1 : 1;
    var useAbs = th.classList.contains('abs-sort');
    rows.sort(function(a, b) {
        var va = a.children[idx] ? a.children[idx].textContent.trim() : '';
        var vb = b.children[idx] ? b.children[idx].textContent.trim() : '';
        var na = parseFloat(va.replace(/[%+—−\\s]/g, ''));
        var nb = parseFloat(vb.replace(/[%+—−\\s]/g, ''));
        if (useAbs) { na = Math.abs(na); nb = Math.abs(nb); }
        if (isNaN(na) && isNaN(nb)) return va.localeCompare(vb, 'zh') * dir;
        if (isNaN(na)) return 1;
        if (isNaN(nb)) return -1;
        return (na - nb) * dir;
    });
    rows.forEach(function(r) { tbody.appendChild(r); });
});

// 来源筛选
document.addEventListener('click', function(e) {
    var btn = e.target.closest('.filter-btn');
    if (!btn) return;
    var bar = btn.parentNode;
    bar.querySelectorAll('.filter-btn').forEach(function(b) {
        b.classList.remove('active');
    });
    btn.classList.add('active');
    var src = btn.getAttribute('data-src');
    var table = document.getElementById(btn.getAttribute('data-table'));
    if (!table) return;
    var rows = table.querySelectorAll('tbody tr');
    rows.forEach(function(r) {
        if (!src || src === 'ALL') {
            r.style.display = '';
        } else {
            r.style.display = r.getAttribute('data-src') === src ? '' : 'none';
        }
    });
});
"""


def _fmt(v, fmt: str = "{:+.4f}") -> str:
    return fmt.format(v) if v == v else "—"


def _med(vals) -> float:
    """中位数（空列表 → NaN）。"""
    return float(np.median(vals)) if len(vals) else float("nan")


def _med_abs(ms: list) -> float:
    """一组监控对象 |近252日IC| 的中位数（缺值跳过；全缺 → NaN）。"""
    vals = [abs(m.ic_mean_recent_252) for m in ms
            if m.ic_mean_recent_252 == m.ic_mean_recent_252]
    return _med(vals)


# 报告展示用中文映射（CSV 账本仍存原始英文值，供程序消费）
_STATUS_ZH = {"normal": "正常", "warning": "预警", "critical": "严重"}
_CATEGORY_ZH = {"factor": "因子", "model": "模型", "signal": "信号", "crowd": "拥挤度"}
_RULE_ZH = {
    "stale_data": "数据滞后",
    "coverage_drop": "覆盖率下降",
    "ic_decay": "IC衰减",
    "significance_loss": "显著性丢失",
    "monotonicity_break": "单调性恶化",
    "factor_crowding": "因子拥挤",
    "signal_stale": "信号滞后",
    "signal_coverage": "信号覆盖不足",
    "signal_concentration": "持仓集中",
    "signal_turnover": "换手过高",
    "signal_blocked": "交易受阻",
}


def _status_zh(s: str) -> str:
    return _STATUS_ZH.get(s, s)


def _category_zh(c: str) -> str:
    return _CATEGORY_ZH.get(c, c)


def _rule_zh(r: str) -> str:
    """规则中文展示名（保留英文 ID 便于与 alerts.csv 对照）。"""
    zh = _RULE_ZH.get(r)
    return f"{zh}({r})" if zh else r


def generate_html_report(
    snapshots: list[MonitorMetrics],
    alert_rows: list[dict],
    ic_history: dict[str, pd.Series],
    out_path: Path,
    dataset: str = "",
    window: int = 60,
    window_long: int = 252,
    as_of: pd.Timestamp | None = None,
    pending_count: int = 0,
) -> None:
    """生成自包含 HTML 监控报告（inline SVG，无外部依赖，可直接归档/邮件转发）。

    章节顺序：概要 → 模型预测 → 全部快照 → 拥挤度 → 告警明细。
    所有表格使用 thead/tbody 分离，排序 JS 只排 tbody 行，不会把表头挤走。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_model = sum(1 for m in snapshots if m.category == "model")
    n_factor = sum(1 for m in snapshots if m.category == "factor")
    n_crit = sum(1 for m in snapshots if m.status == "critical")
    n_warn = sum(1 for m in snapshots if m.status == "warning")
    n_ok = sum(1 for m in snapshots if m.status == "normal")

    parts = [
        "<div class='wrap'>",
        "<h1>因子与模型预测 · 性能监控报告</h1>",
        f"<div class='meta'>数据集={_html.escape(dataset)} · 基准日="
        f"{as_of.date() if as_of is not None else ''} · 双窗口={window}/{window_long}日"
        f"（近{window}日提前预警 / 近{window_long}日稳健确认）· "
        f"生成时间={datetime.now().strftime('%Y-%m-%d %H:%M')}</div>",
        "<div class='cards'>",
        f"<div class='card'><div class='v'>{len(snapshots)}</div>"
        f"<div class='l'>监控对象（因子 {n_factor} + 模型 {n_model}）</div></div>",
        f"<div class='card crit'><div class='v'>{n_crit}</div><div class='l'>严重</div></div>",
        f"<div class='card warn'><div class='v'>{n_warn}</div><div class='l'>预警</div></div>",
        f"<div class='card ok'><div class='v'>{n_ok}</div><div class='l'>正常</div></div>",
        "</div>",
    ]

    # ---- 1. 模型预测因子（最优先展示）----
    parts.append("<h2>模型预测因子</h2>")
    models = [m for m in snapshots if m.category == "model"]
    if models:
        parts.append(
            "<table><thead><tr>"
            "<th class='sortable'>名称</th><th>月频IC趋势</th>"
            "<th class='sortable num abs-sort'>近60日IC</th>"
            "<th class='sortable num abs-sort'>近252日IC</th>"
            "<th class='sortable num'>保留率</th>"
            "<th class='sortable num abs-sort'>NW-t</th>"
            "<th class='sortable num abs-sort'>单调性</th>"
            "<th class='sortable num'>多空日均</th>"
            "<th class='sortable'>状态</th></tr></thead><tbody>"
        )
        for m in sorted(models, key=lambda x: x.status != "critical"):
            parts.append(
                f"<tr><td>{_html.escape(m.name)}<br><span class='rule'>{m.model_id}</span></td>"
                f"<td>{_sparkline_monthly(ic_history.get(m.name))}</td>"
                f"<td class='num'>{_fmt(m.ic_mean_recent)}</td>"
                f"<td class='num'>{_fmt(m.ic_mean_recent_252)}</td>"
                f"<td class='num'>{_fmt(m.ic_retention, '{:.0%}')}</td>"
                f"<td class='num'>{_fmt(m.ic_t_nw_recent, '{:.2f}')}</td>"
                f"<td class='num'>{_fmt(m.monotonicity_recent, '{:+.2f}')}</td>"
                f"<td class='num'>{_fmt(m.ls_daily_recent, '{:+.3%}')}</td>"
                f"<td><span class='tag {m.status}'>{_status_zh(m.status)}</span></td></tr>"
            )
        parts.append("</tbody></table>")
    else:
        parts.append(
            "<p>库内无 model:* 因子（运行 monitor_performance.py "
            "--register-model-factors 回写）。</p>"
        )

    # ---- 2. 全部快照（主表：统一表 + 来源筛选 + 可排序）----
    parts.append("<h2>全部快照</h2>")
    all_factors = [m for m in snapshots if m.category in ("factor", "model")]
    groups_all: dict[str, list[MonitorMetrics]] = {}
    for m in all_factors:
        groups_all.setdefault(m.source_group, []).append(m)

    # 来源分组概要（精简为一行小表，不再单独占大段）
    if len(groups_all) >= 1:
        parts.append("<table><thead><tr>")
        parts.append(
            "<th>来源组</th><th class='num'>对象数</th>"
            "<th class='num'>中位|12个月IC|</th>"
            "<th class='num'>NW-t显著占比</th>"
            "<th class='num'>严重</th><th class='num'>预警</th>"
            "<th class='num'>正常</th></tr></thead><tbody>"
        )
        for g, ms in sorted(groups_all.items(), key=lambda kv: -_med_abs(kv[1])):
            ic252 = [m.ic_mean_recent_252 for m in ms
                     if m.ic_mean_recent_252 == m.ic_mean_recent_252]
            t252 = [m.ic_t_nw_recent_252 for m in ms
                    if m.ic_t_nw_recent_252 == m.ic_t_nw_recent_252]
            sig = (sum(1 for t in t252 if abs(t) > 2.0) / len(t252)) if t252 else float("nan")
            parts.append(
                f"<tr><td>{_html.escape(g)}</td><td class='num'>{len(ms)}</td>"
                f"<td class='num'>{_fmt(_med_abs(ms))}</td>"
                f"<td class='num'>{_fmt(sig, '{:.0%}')}</td>"
                f"<td class='num'>{sum(1 for m in ms if m.status == 'critical')}</td>"
                f"<td class='num'>{sum(1 for m in ms if m.status == 'warning')}</td>"
                f"<td class='num'>{sum(1 for m in ms if m.status == 'normal')}</td></tr>"
            )
        parts.append("</tbody></table>")

    # 筛选按钮栏
    parts.append("<div class='filter-bar'>")
    parts.append(
        f"<button class='filter-btn active' data-src='ALL' "
        f"data-table='snap-table'>全部（{len(all_factors)}）</button>"
    )
    for g, ms in sorted(groups_all.items(), key=lambda kv: -_med_abs(kv[1])):
        n_c = sum(1 for m in ms if m.status == "critical")
        parts.append(
            f"<button class='filter-btn' data-src='{_html.escape(g)}' "
            f"data-table='snap-table'>{_html.escape(g)}（{len(ms)} · 严重 {n_c}）</button>"
        )
    parts.append("</div>")

    # 统一表（thead/tbody 分离，排序 JS 只操作 tbody）
    parts.append("<div class='scroll-table'>")
    parts.append(
        "<table id='snap-table'><thead><tr>"
        "<th class='sortable'>名称</th><th class='sortable'>来源</th>"
        "<th>月频IC趋势</th>"
        "<th class='sortable num abs-sort'>全期IC</th>"
        "<th class='sortable num abs-sort'>近60日IC</th>"
        "<th class='sortable num abs-sort'>近252日IC</th>"
        "<th class='sortable num'>覆盖率</th>"
        "<th class='sortable num abs-sort'>单调性</th>"
        "<th class='sortable num'>滞后日</th>"
        "<th class='sortable'>状态</th></tr></thead><tbody>"
    )
    for m in sorted(
        all_factors,
        key=lambda x: -(
            abs(x.ic_mean_recent_252)
            if x.ic_mean_recent_252 == x.ic_mean_recent_252
            else -1
        ),
    ):
        parts.append(
            f"<tr data-src='{_html.escape(m.source_group)}'>"
            f"<td>{_html.escape(m.name)}</td>"
            f"<td>{_html.escape(m.source_group)}</td>"
            f"<td>{_sparkline_monthly(ic_history.get(m.name))}</td>"
            f"<td class='num'>{_fmt(m.ic_mean_full)}</td>"
            f"<td class='num'>{_fmt(m.ic_mean_recent)}</td>"
            f"<td class='num'>{_fmt(m.ic_mean_recent_252)}</td>"
            f"<td class='num'>{_fmt(m.coverage_recent, '{:.0%}')}</td>"
            f"<td class='num'>{_fmt(m.monotonicity_recent, '{:+.2f}')}</td>"
            f"<td class='num'>{m.stale_days}</td>"
            f"<td><span class='tag {m.status}'>{_status_zh(m.status)}</span></td></tr>"
        )
    parts.append("</tbody></table></div>")

    # ---- 3. 组合 & 信号层（有数据时才展示）----
    signals_ = [m for m in snapshots if m.category == "signal"]
    if signals_:
        parts.append("<h2>组合 & 信号层</h2>")
        parts.append(
            "<table><thead><tr><th>对象</th><th class='num'>最新信号日</th>"
            "<th class='num'>可交易股数</th><th class='num'>集中度 HHI</th>"
            "<th class='num'>净换手</th><th class='num'>受阻占比</th>"
            "<th class='num'>信号滞后(交易日)</th><th>状态</th>"
            "</tr></thead><tbody>"
        )
        for m in sorted(signals_, key=lambda x: x.status != "critical"):
            parts.append(
                f"<tr><td>{_html.escape(m.name)}</td>"
                f"<td class='num'>{m.signal_date or '—'}</td>"
                f"<td class='num'>{m.n_signal_stocks if m.n_signal_stocks else '—'}</td>"
                f"<td class='num'>{_fmt(m.hhi_recent, '{:.3f}')}</td>"
                f"<td class='num'>{_fmt(m.net_turnover, '{:.0%}')}</td>"
                f"<td class='num'>{_fmt(m.blocked_ratio, '{:.0%}')}</td>"
                f"<td class='num'>{m.signal_freshness_days}</td>"
                f"<td><span class='tag {m.status}'>{_status_zh(m.status)}</span></td></tr>"
            )
        parts.append("</tbody></table>")

    # ---- 4. 因子拥挤度（库级）----
    crowds = [m for m in snapshots if m.category == "crowd"]
    if crowds:
        parts.append("<h2>因子拥挤度</h2>")
        parts.append(
            "<table><thead><tr><th>对象</th><th class='num'>因子数</th>"
            "<th class='num'>IC平均相关</th><th class='num'>主成分解释</th>"
            "<th>状态</th></tr></thead><tbody>"
        )
        for m in sorted(crowds, key=lambda x: x.status != "critical"):
            parts.append(
                f"<tr><td>{_html.escape(m.name)}</td><td class='num'>{m.corr_n}</td>"
                f"<td class='num'>{_fmt(m.corr_mean_full, '{:+.2f}')}</td>"
                f"<td class='num'>{_fmt(m.pc1_share_full, '{:.0%}')}</td>"
                f"<td><span class='tag {m.status}'>{_status_zh(m.status)}</span></td></tr>"
            )
        parts.append("</tbody></table>")

    # ---- 5. 告警明细（统一表 + 来源筛选 + 滚动）----
    parts.append("<h2>告警明细</h2>")
    if alert_rows:
        # 按来源分组统计，生成筛选按钮
        alert_groups: dict[str, list[dict]] = {}
        for a in alert_rows:
            sg = a.get("source", "").split(":")[0] if a.get("source") else "(未标注)"
            alert_groups.setdefault(sg, []).append(a)

        parts.append("<div class='filter-bar'>")
        parts.append(
            f"<button class='filter-btn active' data-src='ALL' "
            f"data-table='alert-table'>全部（{len(alert_rows)}）</button>"
        )
        for g, alerts_in_g in sorted(alert_groups.items()):
            n_c = sum(1 for a in alerts_in_g if a["level"] == "critical")
            parts.append(
                f"<button class='filter-btn' data-src='{_html.escape(g)}' "
                f"data-table='alert-table'>{_html.escape(g)}（{len(alerts_in_g)} · 严重 {n_c}）</button>"
            )
        parts.append("</div>")

        parts.append("<div class='scroll-table'>")
        parts.append(
            "<table id='alert-table'><thead><tr>"
            "<th class='sortable'>级别</th>"
            "<th class='sortable'>来源</th>"
            "<th class='sortable'>对象</th><th>规则</th><th>说明</th>"
            "</tr></thead><tbody>"
        )
        order = {"critical": 0, "warning": 1}
        for a in sorted(alert_rows, key=lambda x: (order.get(x["level"], 9), x["name"])):
            sg = a.get("source", "").split(":")[0] if a.get("source") else "(未标注)"
            parts.append(
                f"<tr data-src='{_html.escape(sg)}'>"
                f"<td><span class='tag {a['level']}'>{_status_zh(a['level'])}</span></td>"
                f"<td>{_html.escape(sg)}</td>"
                f"<td>{_html.escape(a['name'])}</td>"
                f"<td class='rule'>{_rule_zh(a['rule'])}</td>"
                f"<td>{_html.escape(a['message'])}</td></tr>"
            )
        parts.append("</tbody></table></div>")
        if pending_count:
            parts.append(
                f"<p class='rule'>另有 {pending_count} 条告警处于观察中"
                f"（须连续触发约定期数才确认）。</p>"
            )
    else:
        parts.append("<p>无已确认告警。</p>")
        if pending_count:
            parts.append(
                f"<p class='rule'>{pending_count} 条告警处于观察中"
                f"（须连续触发约定期数才确认）。</p>"
            )

    parts.append(
        "<footer>YuriQuant monitoring · 快照/告警账本见 reports/monitoring/*.csv · "
        "同一 run_date 重跑幂等覆盖</footer></div>"
    )
    html = page(
        "性能监控报告",
        header="",
        body="\n".join(parts),
        css=_CSS,
        scripts=f"<script>\n{_JS}\n</script>",
    )
    out_path.write_text(html, encoding="utf-8")
