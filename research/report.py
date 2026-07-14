"""
回测可视化报告
==============

生成回测绩效图表:
- 净值曲线（组合 vs 基准）
- 回撤曲线
- IC 时序 + 衰减
- 分层回测净值
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面环境也能保存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.engine import BacktestResult


def plot_equity_curve(
    result: BacktestResult,
    benchmark: pd.Series | None = None,
    save_path: str | Path | None = None,
) -> None:
    """净值曲线。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(result.equity_curve.index, result.equity_curve.values, label="Portfolio", linewidth=1.5)
    if benchmark is not None:
        bench_nav = (1 + benchmark).cumprod()
        ax.plot(bench_nav.index, bench_nav.values, label="Benchmark", linewidth=1, alpha=0.7)
    ax.set_title("Equity Curve")
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_drawdown(result: BacktestResult, save_path: str | Path | None = None) -> None:
    """回撤曲线。"""
    cum = (1 + result.daily_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_ic_series(ic_series: pd.Series, save_path: str | Path | None = None) -> None:
    """IC 时序图。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1])
    ax1.bar(ic_series.index, ic_series.values, width=1, alpha=0.6)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_title("Daily IC")
    ax1.grid(True, alpha=0.3)
    # 累计 IC
    cum_ic = ic_series.cumsum()
    ax2.plot(cum_ic.index, cum_ic.values, linewidth=1, color="blue")
    ax2.set_title("Cumulative IC")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_layer_nav(layer_nav: pd.DataFrame, save_path: str | Path | None = None) -> None:
    """分层回测净值图。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in layer_nav.columns:
        ax.plot(layer_nav.index, layer_nav[col].values, label=col, linewidth=1)
    ax.set_title("Quantile Portfolio NAV")
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def generate_report(
    result: BacktestResult,
    benchmark: pd.Series | None = None,
    factor_summary: dict | None = None,
    output_dir: str | Path = "reports",
) -> Path:
    """生成完整报告（图片 + 文本摘要）。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 图表
    plot_equity_curve(result, benchmark, out / "equity_curve.png")
    plot_drawdown(result, out / "drawdown.png")

    if factor_summary and "ic_series" in factor_summary:
        plot_ic_series(factor_summary["ic_series"], out / "ic_series.png")
    if factor_summary and "layer_nav" in factor_summary:
        plot_layer_nav(factor_summary["layer_nav"], out / "layer_nav.png")

    # 文本摘要
    summary = result.summary(benchmark)
    text = f"YuriQuant Backtest Report\n{'='*50}\n\n{summary}\n"
    if factor_summary:
        text += f"\nFactor Analysis\n{'-'*30}\n"
        text += f"  IC mean:     {factor_summary['ic_mean']:.4f}\n"
        text += f"  IC std:      {factor_summary['ic_std']:.4f}\n"
        text += f"  IC win rate: {factor_summary['ic_win_rate']:.2%}\n"
        text += f"  IR:          {factor_summary['ir']:.4f}\n"

    (out / "summary.txt").write_text(text, encoding="utf-8")
    return out
