"""标签构建 —— 模型层 ② LabelBuilder。

把未来收益变换为监督学习的训练标签，统一管理两个维度：

- ``horizon``：预测视野（交易日数）。标签[t] 只允许使用 t+1..t+horizon 的收益
  （对 t 日特征可见，严格无未来函数）；同时 horizon 决定训练/测试边界的
  **隔离带长度（embargo）**——训练段最后 horizon 日的标签会"望进"测试段，
  必须剔除（纪律继承：stacking 的 purge 思想推广到任意 horizon）。
- ``mode``：目标变换。
    - ``rank``   当日截面百分比秩 - 0.5（推荐：与 rank IC 评价口径对齐，
                 对收益厚尾稳健；Spearman 下与 raw 收益完全等价）
    - ``zscore`` 当日截面标准化（保留收益强弱幅度信息）
    - ``raw``    原始 horizon 日收益（回归直接拟合收益值）

注意：zscore/rank 均为**当日截面内**的单调变换，因此对任一 mode，
Spearman(pred, label) == Spearman(pred, 原始 horizon 收益)——
用 label 面板直接算 IC 与用原始收益算 IC 口径一致。
"""
from __future__ import annotations

import pandas as pd

from factor.preprocessing import standardize_zscore

__all__ = ["build_labels", "build_label_pair", "forward_returns"]


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


def build_labels(
    close_panel: pd.DataFrame | None = None,
    horizon: int = 5,
    mode: str = "rank",
    fwd_returns_panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, int]:
    """构建模型训练标签 + 对应 embargo 长度。

    Args:
        close_panel: date×code 收盘价面板（与 horizon 一起给出 horizon 日前瞻收益）。
        horizon: 预测视野（交易日）。embargo = horizon。
        mode: rank / zscore / raw（见模块 docstring）。
        fwd_returns_panel: 已算好的 horizon 日前瞻收益面板（给了则忽略
            close_panel/horizon 的收益推导，仅做 mode 变换；embargo 仍取
            horizon 参数——调用方须保证两者一致）。

    Returns:
        (labels: date×code 标签面板, embargo_days: int = horizon)
    """
    if fwd_returns_panel is not None:
        fwd = fwd_returns_panel
    else:
        if close_panel is None:
            raise ValueError("close_panel 与 fwd_returns_panel 至少给一个")
        fwd = forward_returns(close_panel, horizon)

    if mode == "rank":
        labels = fwd.rank(axis=1, pct=True) - 0.5
    elif mode == "zscore":
        labels = standardize_zscore(fwd)
    elif mode == "raw":
        labels = fwd.copy()
    else:
        raise ValueError(f"未知 mode {mode!r}，可选: rank / zscore / raw")

    labels = labels.astype(float)
    # 全 NaN 列保留（对齐网格交给 Predictor），但至少要有一些有效标签
    if not labels.notna().any().any():
        raise ValueError("标签面板全为 NaN：检查 horizon 与面板长度")
    return labels, int(horizon)


def build_label_pair(
    close_panel: pd.DataFrame,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rank 标签, 原始 horizon 收益) 二元组 —— 预测/回测共口径（e2e 下沉）。

    即 ``build_labels(mode="rank")`` 与 ``forward_returns`` 的组合：标签用于
    训练，原始收益用于 IC/分层评价，两者出自同一前瞻窗口（h=horizon）。
    """
    fwd = forward_returns(close_panel, horizon)
    labels, _ = build_labels(close_panel=close_panel, horizon=horizon, mode="rank")
    return labels, fwd
