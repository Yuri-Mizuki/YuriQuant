"""
回测可视化报告
==============

支持两种报告模式:
1. 单因子完整报告  generate_single_report()
   - 绩效指标表（中英文）
   - 净值曲线
   - 回撤曲线
   - IC 时序 + 累计 IC
   - 分层回测净值
   - 月度收益热力图

2. 多因子对比报告  generate_comparison_report()
   - 净值曲线对比图
   - 指标对比表（所有因子并排）
   - 雷达图对比
   - 排名表
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字体设置：Windows 用微软雅黑 / SimHei，Linux 用文泉驿
for font_name in ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Arial Unicode MS"]:
    try:
        fm.findfont(font_name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue
else:
    plt.rcParams["axes.unicode_minus"] = False

from backtest.engine import BacktestResult
from backtest.metrics import METRIC_LABELS, format_metrics


# ===========================================================================
# 图表组件
# ===========================================================================
def _plot_equity(ax, equity_curve: pd.Series, label: str = "Portfolio", color: str = None):
    ax.plot(equity_curve.index, equity_curve.values, label=label, linewidth=1.5, color=color)
    ax.set_ylabel("NAV")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_drawdown(ax, daily_returns: pd.Series):
    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)


def _plot_ic(ax1, ax2, ic_series: pd.Series):
    ax1.bar(ic_series.index, ic_series.values, width=1, alpha=0.6, color="steelblue")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_ylabel("Daily IC")
    ax1.grid(True, alpha=0.3)
    cum_ic = ic_series.cumsum()
    ax2.plot(cum_ic.index, cum_ic.values, linewidth=1, color="darkblue")
    ax2.fill_between(cum_ic.index, 0, cum_ic.values, alpha=0.2, color="darkblue")
    ax2.set_ylabel("Cumulative IC")
    ax2.grid(True, alpha=0.3)


def _plot_layers(ax, layer_nav: pd.DataFrame):
    for col in layer_nav.columns:
        ax.plot(layer_nav.index, layer_nav[col].values, label=col, linewidth=1)
    ax.set_ylabel("NAV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_monthly_heatmap(ax, daily_returns: pd.Series):
    """月度收益热力图。"""
    monthly = (1 + daily_returns).resample("ME").apply(lambda x: (1 + x).prod() - 1)
    if monthly.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    pivot = monthly.to_frame("ret")
    pivot["year"] = pivot.index.year
    pivot["month"] = pivot.index.month
    table = pivot.pivot_table(index="year", columns="month", values="ret")
    table = table.reindex(columns=range(1, 13))
    month_labels = [f"{m}月" for m in range(1, 13)]
    im = ax.imshow(table.values, cmap="RdYlGn", aspect="auto", vmin=-0.08, vmax=0.08)
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels, fontsize=8)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=8)
    # 标注数值
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)


def _metrics_to_table(metrics: dict, factor_name: str = "") -> pd.DataFrame:
    """指标 dict → DataFrame（中英文标签）。"""
    rows = []
    for k, v in metrics.items():
        label = METRIC_LABELS.get(k, k)
        if isinstance(v, float):
            if k in ("win_rate", "ic_win_rate"):
                val_str = f"{v:.2%}"
            elif abs(v) < 100:
                val_str = f"{v:.4f}"
            else:
                val_str = f"{v:.0f}"
        else:
            val_str = str(v)
        rows.append({"指标 Metric": label, factor_name or "值": val_str})
    return pd.DataFrame(rows)


# ===========================================================================
# 单因子完整报告
# ===========================================================================
def generate_single_report(
    result: BacktestResult,
    factor_name: str = "",
    benchmark: pd.Series | None = None,
    factor_summary: dict | None = None,
    output_dir: str | Path = "reports",
) -> Path:
    """生成单因子完整报告（一张大图 + 指标表 + 文本摘要）。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = result.metrics(benchmark)

    # ---- 组合大图 ----
    has_ic = factor_summary and "ic_series" in factor_summary
    has_layers = factor_summary and "layer_nav" in factor_summary

    n_rows = 4 if (has_ic or has_layers) else 2
    fig = plt.figure(figsize=(16, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.35, wspace=0.2)

    # 1. 净值曲线
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_equity(ax1, result.equity_curve, factor_name or "Portfolio")
    if benchmark is not None:
        bench_nav = (1 + benchmark).cumprod()
        ax1.plot(bench_nav.index, bench_nav.values, label="Benchmark", linewidth=1, alpha=0.7, color="orange")
        ax1.legend(fontsize=9)
    ax1.set_title(f"净值曲线 Equity Curve — {factor_name}", fontsize=11)

    # 2. 回撤
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_drawdown(ax2, result.daily_returns)
    ax2.set_title("回撤 Drawdown", fontsize=11)

    # 3. 月度热力图
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_monthly_heatmap(ax3, result.daily_returns)
    ax3.set_title("月度收益热力图 Monthly Returns", fontsize=11)

    # 4. 指标表（文本渲染）
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    metrics_text = format_metrics(metrics)
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes,
             fontsize=8, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax4.set_title("绩效指标 Performance Metrics", fontsize=11)

    row = 2
    # 5. IC
    if has_ic:
        ax5 = fig.add_subplot(gs[row, 0])
        ax5b = fig.add_subplot(gs[row, 1])
        _plot_ic(ax5, ax5b, factor_summary["ic_series"])
        ax5.set_title("IC 时序 Daily IC", fontsize=11)
        ax5b.set_title("累计 IC Cumulative IC", fontsize=11)
        row += 1

    # 6. 分层
    if has_layers:
        ax6 = fig.add_subplot(gs[row, :])
        _plot_layers(ax6, factor_summary["layer_nav"])
        ax6.set_title("分层回测净值 Quantile Portfolio NAV", fontsize=11)

    fig.suptitle(f"YuriQuant 回测报告 — {factor_name}", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(out / f"report_{factor_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 文本摘要 ----
    text = f"YuriQuant Backtest Report — {factor_name}\n{'='*60}\n\n"
    text += format_metrics(metrics)
    if factor_summary:
        text += f"\n\n因子分析 Factor Analysis\n{'-'*40}\n"
        text += f"  IC均值 IC Mean:      {factor_summary['ic_mean']:.4f}\n"
        text += f"  IC标准差 IC Std:     {factor_summary['ic_std']:.4f}\n"
        text += f"  IC胜率 IC Win Rate:  {factor_summary['ic_win_rate']:.2%}\n"
        text += f"  信息比率 IR:          {factor_summary['ir']:.4f}\n"
        if "ic_decay" in factor_summary:
            text += "\n  IC衰减 IC Decay:\n"
            for lag, val in factor_summary["ic_decay"].items():
                text += f"    lag={lag}: {val:.4f}\n"
        if "layer_returns" in factor_summary:
            text += "\n  分层累计收益 Layer Returns:\n"
            for q, val in factor_summary["layer_returns"].items():
                text += f"    {q}: {val:.4f}\n"

    txt_path = out / f"report_{factor_name}.txt"
    txt_path.write_text(text, encoding="utf-8")

    return txt_path


# ===========================================================================
# 多因子对比报告
# ===========================================================================
def generate_comparison_report(
    results: dict[str, BacktestResult],
    factor_summaries: dict[str, dict] | None = None,
    benchmark: pd.Series | None = None,
    output_dir: str | Path = "reports",
) -> Path:
    """生成多因子对比报告。

    Args:
        results: {factor_name: BacktestResult}
        factor_summaries: {factor_name: factor_summary dict}
        benchmark: 基准日收益
        output_dir: 输出目录
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 收集所有指标
    all_metrics = {}
    for name, res in results.items():
        m = res.metrics(benchmark)
        if factor_summaries and name in factor_summaries:
            fs = factor_summaries[name]
            m["ic_mean"] = fs.get("ic_mean", np.nan)
            m["ic_std"] = fs.get("ic_std", np.nan)
            m["ic_win_rate"] = fs.get("ic_win_rate", np.nan)
            m["ir"] = fs.get("ir", np.nan)
        all_metrics[name] = m

    # ---- 大图 ----
    n_factors = len(results)
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # 1. 净值对比
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.tab10(np.linspace(0, 1, n_factors))
    for (name, res), color in zip(results.items(), colors):
        ax1.plot(res.equity_curve.index, res.equity_curve.values,
                 label=name, linewidth=1.2, color=color)
    if benchmark is not None:
        bench_nav = (1 + benchmark).cumprod()
        ax1.plot(bench_nav.index, bench_nav.values, label="Benchmark",
                 linewidth=1, alpha=0.7, color="black", linestyle="--")
    ax1.set_title("净值曲线对比 Equity Curve Comparison", fontsize=12)
    ax1.set_ylabel("NAV")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 回撤对比
    ax2 = fig.add_subplot(gs[1, 0])
    for (name, res), color in zip(results.items(), colors):
        cum = (1 + res.daily_returns).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        ax2.plot(dd.index, dd.values, label=name, linewidth=0.8, color=color)
    ax2.set_title("回撤对比 Drawdown Comparison", fontsize=12)
    ax2.set_ylabel("Drawdown")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. 指标对比表
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    # 构建对比表
    compare_df = _build_comparison_table(all_metrics)
    ax3.table(cellText=compare_df.values,
              rowLabels=compare_df.index,
              colLabels=compare_df.columns,
              cellLoc="center",
              loc="center",
              fontsize=8)
    ax3.set_title("指标对比 Metrics Comparison", fontsize=12)

    fig.suptitle("YuriQuant 多因子对比报告 Multi-Factor Comparison", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(out / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 指标对比 CSV + 文本 ----
    compare_df.to_csv(out / "comparison_metrics.csv", encoding="utf-8-sig")

    # 排名（按夏普降序）
    ranking = _build_ranking(all_metrics)
    text = "YuriQuant 多因子对比报告 Multi-Factor Comparison\n"
    text += f"{'='*70}\n\n"
    text += "指标对比表 Metrics Comparison:\n\n"
    text += compare_df.to_string() + "\n\n"
    text += "排名 Ranking (by Sharpe):\n\n"
    text += ranking.to_string() + "\n"
    (out / "comparison.txt").write_text(text, encoding="utf-8")

    return out / "comparison.txt"


def _build_comparison_table(all_metrics: dict[str, dict]) -> pd.DataFrame:
    """构建因子×指标对比表。"""
    # 选取关键指标列
    key_metrics = [
        "annual_return", "annual_volatility", "sharpe", "sortino",
        "max_drawdown", "calmar", "win_rate", "profit_loss_ratio",
        "avg_turnover", "ic_mean", "ir",
    ]
    rows = {}
    for name, m in all_metrics.items():
        row = []
        for k in key_metrics:
            v = m.get(k, np.nan)
            if k in ("win_rate", "ic_win_rate"):
                row.append(f"{v:.2%}" if pd.notna(v) else "-")
            elif pd.notna(v):
                row.append(f"{v:.4f}")
            else:
                row.append("-")
        rows[name] = row
    labels = [METRIC_LABELS.get(k, k) for k in key_metrics]
    df = pd.DataFrame(rows, index=labels).T
    return df


def _build_ranking(all_metrics: dict[str, dict]) -> pd.DataFrame:
    """按夏普降序排名。"""
    data = []
    for name, m in all_metrics.items():
        data.append({
            "因子 Factor": name,
            "年化收益 Annual Return": m.get("annual_return", np.nan),
            "夏普 Sharpe": m.get("sharpe", np.nan),
            "最大回撤 Max DD": m.get("max_drawdown", np.nan),
            "IC均值 IC Mean": m.get("ic_mean", np.nan),
            "IR": m.get("ir", np.nan),
        })
    df = pd.DataFrame(data)
    df = df.sort_values("夏普 Sharpe", ascending=False).reset_index(drop=True)
    df.insert(0, "排名 Rank", range(1, len(df) + 1))
    return df


# ===========================================================================
# 保留旧接口兼容
# ===========================================================================
def plot_equity_curve(result, benchmark=None, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    _plot_equity(ax, result.equity_curve, "Portfolio")
    if benchmark is not None:
        bench_nav = (1 + benchmark).cumprod()
        ax.plot(bench_nav.index, bench_nav.values, label="Benchmark", linewidth=1, alpha=0.7)
        ax.legend()
    ax.set_title("Equity Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_drawdown(result, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    _plot_drawdown(ax, result.daily_returns)
    ax.set_title("Drawdown")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_ic_series(ic_series, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    _plot_ic(ax1, ax2, ic_series)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_layer_nav(layer_nav, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    _plot_layers(ax, layer_nav)
    ax.set_title("Quantile Portfolio NAV")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def generate_report(result, benchmark=None, factor_summary=None, output_dir="reports"):
    """旧接口兼容 → 调用单因子报告。"""
    return generate_single_report(result, "", benchmark, factor_summary, output_dir)
