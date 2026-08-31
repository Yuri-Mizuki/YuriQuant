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

import numpy as np
import pandas as pd

log = logging.getLogger("factor_report")


# ---------------------------------------------------------------------------
# 序列整理（因子检测报告用）
# ---------------------------------------------------------------------------
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
    ic_sparse = calc_ic_series(sparse, fwd_aligned)

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


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    _plt().close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_layers(layer_nav: pd.DataFrame) -> str:
    """分层累计收益图：Q1~Q5 与多空全部用累计收益（净值-1，0 起点），
    同一量纲、Y 轴按百分比自动展开——避免净值 1 起点导致差异被压扁。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4))
    ret = layer_nav - 1.0  # 累计收益，起点 0
    for q in layer_nav.columns:
        ax.plot(ret.index, ret[q], label=q, linewidth=1.2)
    ls = ret["Q5"] - ret["Q1"]  # 多空 = 收益差，同样 0 起点
    ax.plot(ls.index, ls, label="Q5-Q1(多空)", linewidth=1.6,
            color="#A32D2D", linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_title("模型预测因子分层累计收益（Q1=预测最低组, Q5=最高组）")
    ax.set_ylabel("累计收益")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def plot_monthly_ic(monthly_ic: pd.Series) -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    colors = ["#A32D2D" if v > 0 else "#3B6D11" for v in monthly_ic]
    ax.bar(monthly_ic.index.astype(str), monthly_ic.values, color=colors)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_title("模型预测因子月度 IC（持仓口径）")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _fig_to_b64(fig)
