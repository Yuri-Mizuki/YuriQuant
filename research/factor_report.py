"""因子报告的数据准备与绘图辅助（factor_explorer_report / investment_report 共用）。

从 scripts 下沉的纯计算/绘图函数：分层回测（``qcut_rebal``）、分层绩效
（``layer_stats_from_nav`` / ``layer_avg_ret``）、月度/周频序列与 IC 衰减/热力图、
模型预测作为因子的检验（``factor_test``）与 matplotlib 绘图。HTML 模板与 CLI
入口留在脚本层。
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import pandas as pd

from backtest.metrics import (
    annual_return,
    annual_volatility,
    drawdown_series,
    sharpe_ratio,
)

log = logging.getLogger("factor_report")


# ---------------------------------------------------------------------------
# 分层回测与绩效（因子检测报告用）
# ---------------------------------------------------------------------------
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
        annual = float(annual_return(rets, 12)) if n_years > 0.3 else total
        vol = float(annual_volatility(rets, 12)) if len(rets) > 1 else 0.0
        sharpe = float(sharpe_ratio(rets, periods_per_year=12)) if vol > 0 else None
        dd = float(drawdown_series(s).min())
        out.append({
            "annual": round(annual, 4), "vol": round(vol, 4),
            "sharpe": round(sharpe, 3) if sharpe is not None else None,
            "dd": round(dd, 4), "total": round(total, 4),
        })
    return out


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


# ---------------------------------------------------------------------------
# 模型预测作为因子的检验（investment_report 用）
# ---------------------------------------------------------------------------
def factor_test(pred_panel: pd.DataFrame, fwd: pd.DataFrame, close: pd.DataFrame,
                out_dir: Path, label: str) -> dict:
    """把模型预测面板作为因子做标准检验。

    两种口径：
    - 稀疏：只在调仓日有预测值（纯信号检验，样本=调仓次数）
    - 持仓：预测值 ffill 到下一次调仓前（与组合持仓一致，样本=交易日）

    口径纪律（修复 2026-08-25）：
    - IC 检验用 fwd（未来 horizon 累计收益）——模型预测什么就评估什么；
    - **分层回测必须用日频未来一期收益**（close.pct_change().shift(-1)），
      若把 horizon 累计收益按日累乘 (1+r).cumprod()，5 日滚动窗口重叠会把
      同一段收益重复算约 5 次（净值虚高 5 倍，Q1 曾显示 +556% 假象）；
    - 网格裁剪到因子有效起始（2024-01），消除回测前全 NaN 的 1.0 平线段。

    Returns:
        {summary_sparse, summary_hold, layer_nav, ic_sparse, ic_hold, monthly_ic}
    """
    from research.factor_analysis import calc_ic_series, quantile_backtest, standard_factor_summary

    common = fwd.dropna(how="all").index
    sparse = pred_panel.reindex(index=common)
    hold = sparse.ffill()

    # 裁剪到因子有效起始（2024-01）：消除 2022~2024 全 NaN 平线
    valid = hold.dropna(how="all").index
    if len(valid):
        common = common[common >= valid[0]]
        sparse = sparse.reindex(index=common)
        hold = hold.reindex(index=common)
    fwd_aligned = fwd.reindex(index=common)
    log.info("因子检验网格: %s ~ %s (%d 日)", common[0].date(), common[-1].date(), len(common))

    # IC 口径（模型预测 horizon 累计收益）
    sum_sparse = standard_factor_summary(sparse, fwd_aligned)
    sum_hold = standard_factor_summary(hold, fwd_aligned)

    # 分层净值（持仓口径）：日频未来一期收益，避免 horizon 重叠放大
    ret_d = close.pct_change(fill_method=None).shift(-1).reindex(index=common)
    layer_nav = quantile_backtest(hold, ret_d, n_quantiles=5)
    layer_nav.to_csv(out_dir / f"layer_nav_{label}.csv", encoding="utf-8-sig")

    # 月度 IC（持仓口径，horizon 累计收益）
    ic_hold = calc_ic_series(hold, fwd_aligned)
    ic_hold.name = "ic"
    monthly_ic = ic_hold.groupby(ic_hold.index.to_period("M")).mean()
    monthly_ic.to_csv(out_dir / f"monthly_ic_{label}.csv", encoding="utf-8-sig")

    pd.DataFrame([sum_sparse, sum_hold], index=["稀疏(调仓日)", "持仓(日频)"]) \
        .to_csv(out_dir / f"factor_summary_{label}.csv", encoding="utf-8-sig")

    log.info("因子检验[%s]: 稀疏 IC=%.4f IR=%.2f t_nw=%.2f | 持仓 IC=%.4f IR=%.2f t_nw=%.2f",
             label, sum_sparse["ic_mean"], sum_sparse["ir"], sum_sparse["t_stat_nw"],
             sum_hold["ic_mean"], sum_hold["ir"], sum_hold["t_stat_nw"])
    return {"sum_sparse": sum_sparse, "sum_hold": sum_hold, "layer_nav": layer_nav,
            "ic_hold": ic_hold, "monthly_ic": monthly_ic}


# ---------------------------------------------------------------------------
# 绘图（matplotlib 懒加载，避免纯数据使用者背 matplotlib 依赖）
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    _plt().close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_monthly_ic(monthly_ic: pd.Series) -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    colors = ["#A32D2D" if v > 0 else "#3B6D11" for v in monthly_ic]
    ax.bar(monthly_ic.index.astype(str), monthly_ic.values, color=colors)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_title("模型预测因子月度 IC（持仓口径）")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.grid(alpha=0.3, axis="y")
    return fig_to_b64(fig)
