"""标签构建 —— 模型层 ② LabelBuilder。

把未来收益变换为监督学习的训练标签，统一管理三个维度：

- ``horizon``：预测视野（交易日数）。标签[t] 只允许使用 t+1..t+horizon 的收益
  （对 t 日特征可见，严格无未来函数）；同时 horizon 决定训练/测试边界的
  **隔离带长度（embargo）**——训练段最后 horizon 日的标签会"望进"测试段，
  必须剔除（纪律继承：stacking 的 purge 思想推广到任意 horizon）。
- ``mode``：目标变换。
    - ``rank``   当日截面百分比秩 - 0.5（推荐：与 rank IC 评价口径对齐，
                 对收益厚尾稳健；Spearman 下与 raw 收益完全等价）
    - ``zscore`` 当日截面标准化（保留收益强弱幅度信息）
    - ``raw``    原始 horizon 日收益（回归直接拟合收益值）
- ``method``（华泰 AI29 另类标签，2026-09-23 接入）：把 horizon 日区间
  映射为标量的方式，作用于**超额**口径（需 ``bench_close_panel``）：
    - ``return`` 区间收益本身（默认，历史行为逐位一致）
    - ``ir``     区间超额收益 ÷ 区间内日度超额收益标准差
                 （AI29 式 (P₁/P₀ − B₁/B₀)/σ₂ 的连续版：σ 取区间内逐日
                 超额收益的样本标准差，含无风险=0 简化）
    - ``calmar`` 区间超额收益 ÷ 区间内超额收益最大回撤

注意：zscore/rank 均为**当日截面内**的单调变换，因此对任一 mode，
Spearman(pred, label) == Spearman(pred, 原始 horizon 收益)——
用 label 面板直接算 IC 与用原始收益算 IC 口径一致。
**例外**：ir/calmar 标签含区间路径信息（σ/MaxDD），与区间收益**不再单调
等价**——这正是 AI29 引入它们的动机（开辟标签蓝海）；评价时对另类标签
应报"pred vs 标签自身"的 IC，且**须同口径披露超额最大回撤**（AI29 所有
测试中另类标签的超额 MaxDD 一致更差，见研读笔记 AI43_AI29）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor.preprocessing import standardize_zscore

__all__ = ["build_labels", "build_label_pair", "forward_returns",
           "forward_excess_stats"]


def forward_returns(
    close_panel: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """未来 horizon 日收益率面板：fwd[t] = close[t+horizon]/close[t] - 1。

    尾部 horizon 日无完整前瞻窗口 → NaN（自然截断，无未来函数）。
    fill_method=None：价格缺口（停牌等）不前向填充，缺口的收益如实为 NaN。
    """
    if horizon < 1:
        raise ValueError(f"horizon 必须 >= 1，收到 {horizon}")
    return close_panel.pct_change(horizon, fill_method=None).shift(-horizon)


def forward_excess_stats(
    close_panel: pd.DataFrame,
    bench_close: pd.DataFrame | pd.Series,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """未来 horizon 日**超额**口径三件套：(区间超额收益, 日度超额 σ, 超额 MaxDD)。

    超额 = 个股 − 基准（AI29 简化：无风险利率 = 0）。三者都只使用
    t+1..t+horizon 的数据（对 t 日特征可见，严格无未来函数），尾部
    horizon 日无完整前瞻窗口 → NaN。

    Returns:
        (excess_total, excess_vol, excess_mdd) 三个 date×code 面板：
        - excess_total[t] = (1+r_stock)·...·(1+r_stock[h]) − (1+r_bench)·...·
          (1+r_bench[h])：**复合**区间超额（AI29 的 P/P₀ − B/B₀ 同构，
          复合口径下个股与基准各自累积后相减）
        - excess_vol[t] = 区间内逐日超额收益的样本标准差（ddof=1；窗口
          <2 个有效日 → NaN）
        - excess_mdd[t] = 区间内**几何超额净值曲线**（个股复合净值 ÷ 基准复合
          净值）的最大回撤（恒 ≤ 0，AI29 的 Calmar 分母取其绝对值）
    """
    if horizon < 1:
        raise ValueError(f"horizon 必须 >= 1，收到 {horizon}")
    stock_ret = close_panel.pct_change(fill_method=None)
    b = bench_close
    if isinstance(b, pd.Series):
        b = b.to_frame("bench")
    if isinstance(b, pd.DataFrame) and b.shape[1] == 1:
        b = pd.DataFrame(np.repeat(b.to_numpy(), len(close_panel.columns), axis=1),
                         index=b.index, columns=close_panel.columns)
    bench_ret = b.pct_change(fill_method=None)

    # 复合区间超额：个股累积因子 − 基准累积因子（各 h 日窗口内 cumprod）
    n = len(stock_ret)
    idx = stock_ret.index
    et = pd.DataFrame(np.nan, index=idx, columns=stock_ret.columns)
    ev = pd.DataFrame(np.nan, index=idx, columns=stock_ret.columns)
    em = pd.DataFrame(np.nan, index=idx, columns=stock_ret.columns)
    sr = stock_ret.to_numpy(dtype=float)
    br = bench_ret.reindex(columns=stock_ret.columns).to_numpy(dtype=float)
    for t in range(n - horizon):
        ws = sr[t + 1: t + 1 + horizon]   # t+1..t+h 的日收益（不含 t 当日）
        wb = br[t + 1: t + 1 + horizon]
        # 有效掩码：个股或基准任一缺口 → 该日超额无效
        valid = np.isfinite(ws) & np.isfinite(wb)
        if not valid.any():
            continue
        cum_s = np.where(valid, ws, 0.0)
        cum_b = np.where(valid, wb, 0.0)
        # 复合：仅统计有效日链（缺口日不重置、不计入——与 pct_change(h) 的
        # "链上遇 NaN 即 NaN" 保守口径相比，这里按有效日计数）
        n_valid = valid.sum(axis=0)
        total_s = np.prod(1.0 + cum_s, axis=0) - 1.0
        total_b = np.prod(1.0 + cum_b, axis=0) - 1.0
        # 全窗口有效才出数（部分缺口的"跳链复合"会低估/高估波动，保守置 NaN）
        full = (n_valid == horizon)
        et.iloc[t] = np.where(full, total_s - total_b, np.nan)
        # 日度超额 σ（样本 std，全窗口有效日）
        with np.errstate(invalid="ignore"):
            mean_e = np.where(n_valid > 0,
                              np.nansum(np.where(valid, ws - wb, 0.0), axis=0)
                              / np.maximum(n_valid, 1), np.nan)
            dev2 = np.where(valid, (ws - wb - mean_e[None, :]) ** 2, 0.0)
            var = np.where(n_valid >= 2,
                           dev2.sum(axis=0) / np.maximum(n_valid - 1, 1),
                           np.nan)
            ev.iloc[t] = np.where(full, np.sqrt(var), np.nan)
        # 超额净值曲线 MaxDD：几何超额 = 个股复合净值 ÷ 基准复合净值
        # （相对强弱曲线——业内"超额净值"标准口径），起点 1
        cs = np.cumprod(1.0 + np.where(valid, ws, 0.0), axis=0)
        cb = np.cumprod(1.0 + np.where(valid, wb, 0.0), axis=0)
        nav = cs / cb
        run_max = np.maximum.accumulate(nav, axis=0)
        mdd = np.min(nav / run_max - 1.0, axis=0)
        em.iloc[t] = np.where(full, mdd, np.nan)
    return et, ev, em


def build_labels(
    close_panel: pd.DataFrame | None = None,
    horizon: int = 5,
    mode: str = "rank",
    fwd_returns_panel: pd.DataFrame | None = None,
    tradable_mask: pd.DataFrame | None = None,
    method: str = "return",
    bench_close_panel: pd.DataFrame | pd.Series | None = None,
) -> tuple[pd.DataFrame, int]:
    """构建模型训练标签 + 对应 embargo 长度。

    Args:
        close_panel: date×code 收盘价面板（与 horizon 一起给出 horizon 日前瞻收益）。
        horizon: 预测视野（交易日）。embargo = horizon。
        mode: rank / zscore / raw（见模块 docstring）。
        fwd_returns_panel: 已算好的 horizon 日前瞻收益面板（给了则忽略
            close_panel/horizon 的收益推导，仅做 mode 变换；embargo 仍取
            horizon 参数——调用方须保证两者一致）。**method≠"return" 时忽略
            此参数**（另类标签须从价格面板重算区间路径，不接受预制收益）。
        tradable_mask: 可选的可交易性掩码（date×code bool，**T+1 成交口径**，
            即 ``data.tradability::build_tradable_mask`` 的产物）。给定时，
            ``False`` 处的标签置 NaN —— 等价于把这些样本从训练集剔除。
            默认 ``None`` = 不加掩码（与历史行为逐位一致）。
        method: ``return`` / ``ir`` / ``calmar``（AI29 另类标签，见模块
            docstring）。``ir``/``calmar`` 须同时给 ``bench_close_panel``。
        bench_close_panel: 基准收盘价面板（date×1）或 Series（date×，广播到
            全部个股列）。仅 ``method``≠``return`` 时需要。

    Returns:
        (labels: date×code 标签面板, embargo_days: int = horizon)

    为什么需要 ``tradable_mask``（2026-09-17 接入）：
        无掩码时标签把"T 日封涨停 → T+1 继续封板"的**买不进的**收益当成监督
        信号，模型便学出"高封板位置 → 高收益"的纸面关系（实测 `limit_pos`
        在可交易口径下 IC 为 **负**，而它拿到 Top1 权重 20.6%，
        reports/limit_pos/tradable_p0/report.md）。掩掉不可交易样本后，这部分
        纸面收益不再进入损失函数，纸面因子会**自然失去预测力**——治本位置在
        标签层，不在因子层。

    作用时机：**先 mode 变换、后置 NaN**（不重排 rank）。即 ``rank`` 仍是
        "全样本截面百分位"，掩码只把该样本从训练集剔除；这样标签的截面定义
        与既有 IC / 分层评价口径保持可比，避免"修标签"同时改变 rank 语义而
        无法归因。若日后要改成"可交易池内重排 rank"（更贴近实盘任务），须作为
        独立实验对照，不要在本函数里静默切换。
    """
    if method not in ("return", "ir", "calmar"):
        raise ValueError(f"未知 method {method!r}，可选: return / ir / calmar")
    if method == "return":
        if fwd_returns_panel is not None:
            fwd = fwd_returns_panel
        else:
            if close_panel is None:
                raise ValueError("close_panel 与 fwd_returns_panel 至少给一个")
            fwd = forward_returns(close_panel, horizon)
    else:
        if close_panel is None:
            raise ValueError(
                f"method={method!r} 须给 close_panel（区间路径标签不接受预制收益面板）")
        if bench_close_panel is None:
            raise ValueError(f"method={method!r} 须给 bench_close_panel（AI29 超额口径）")
        exc_total, exc_vol, exc_mdd = forward_excess_stats(
            close_panel, bench_close_panel, horizon)
        if method == "ir":
            # AI29: 区间超额收益 / 区间内日度超额 σ（σ→0 处保守置 NaN，
            # 防 +1e-4 型常数改写经济含义——同类前科见 MEMORY §已收口）
            fwd = exc_total / exc_vol.where(exc_vol > 0)
        else:  # calmar: 区间超额收益 / |超额 MaxDD|（mdd=0 即无回撤 → NaN）
            fwd = exc_total / exc_mdd.abs().where(exc_mdd.abs() > 0)

    if mode == "rank":
        labels = fwd.rank(axis=1, pct=True) - 0.5
    elif mode == "zscore":
        labels = standardize_zscore(fwd)
    elif mode == "raw":
        labels = fwd.copy()
    else:
        raise ValueError(f"未知 mode {mode!r}，可选: rank / zscore / raw")

    labels = labels.astype(float)

    if tradable_mask is not None:
        # 对齐到标签面板；掩码未覆盖的 (date, code) 保守视为可交易（与回测
        # VectorBacktest 的 mask.reindex(...).fillna(True) 同口径），避免因
        # 掩码面板缺行/缺列而静默丢掉正常样本。
        m = tradable_mask.reindex(index=labels.index, columns=labels.columns,
                                  fill_value=True)
        labels = labels.where(m.fillna(True).astype(bool))

    # 全 NaN 列保留（对齐网格交给 Predictor），但至少要有一些有效标签
    if not labels.notna().any().any():
        if tradable_mask is not None:
            raise ValueError(
                "标签面板全为 NaN：加 tradable_mask 后无剩余有效标签——"
                "检查掩码方向（True = 可交易）与覆盖范围")
        raise ValueError("标签面板全为 NaN：检查 horizon 与面板长度")
    return labels, int(horizon)


def build_label_pair(
    close_panel: pd.DataFrame,
    horizon: int = 5,
    tradable_mask: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rank 标签, 原始 horizon 收益) 二元组 —— 预测/回测共口径（e2e 下沉）。

    即 ``build_labels(mode="rank")`` 与 ``forward_returns`` 的组合：标签用于
    训练，原始收益用于 IC/分层评价，两者出自同一前瞻窗口（h=horizon）。

    ``tradable_mask`` 只作用于**标签**（置 NaN，见 ``build_labels``），返回的原始
    收益面板保持完整 —— 评价口径（IC / 分层）仍应看到全部样本，掩码是训练侧的事。
    """
    fwd = forward_returns(close_panel, horizon)
    labels, _ = build_labels(close_panel=close_panel, horizon=horizon, mode="rank",
                             tradable_mask=tradable_mask)
    return labels, fwd
