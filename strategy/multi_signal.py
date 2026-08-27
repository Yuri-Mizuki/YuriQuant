"""
ETF 多信号合成（Multi-Signal Composition）
==========================================

在单信号动量之上，用「多信号加权合成」得到 ETF 级综合分，再喂给轮动策略。

原则
----
- 所有信号均为 date × code 面板，与项目既有因子/回测约定一致。
- 每个信号在**每个交易日截面 z-score**（跨 code 标准化），消除量纲后再加权。
- 不引入 GBDT 等黑盒：ETF 池仅 15 只，训练样本太少，加权合成更透明可解释。

信号集（均可用后复权收盘价构建）
--------------------------------
- mom20   : 20 日动量（短期收益持续性）
- mom60   : 60 日动量（中期趋势）
- trend   : 20/60 日均线多头状态（ma20/ma60 - 1，反映趋势阶段）
- voladj  : 波动率调整动量（mom20 / 20 日收益 std，风险平价修正）
"""
from __future__ import annotations

import pandas as pd

# 默认信号权重：收敛到「动量家族」（短/中动 + 波动率修正），
# 避免混入弱信号（trend）拉低合成；权重可经 CLI 覆盖
DEFAULT_WEIGHTS: dict[str, float] = {"mom20": 2.0, "mom60": 1.5, "voladj": 1.0}


def _zscore_rowwise(panel: pd.DataFrame) -> pd.DataFrame:
    """每个交易日对 code 做截面 z 标准化（跳过 NaN）。"""
    mean = panel.mean(axis=1)
    std = panel.std(axis=1)
    return panel.sub(mean, axis=0).div(std.replace(0, float("nan")), axis=0)


def build_signals(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """由后复权收盘价面板构建各信号（date × code）。"""
    ret = close.pct_change(fill_method=None)
    return {
        "mom20": close.pct_change(20),
        "mom60": close.pct_change(60),
        "trend": close.rolling(20).mean() / close.rolling(60).mean() - 1.0,
        "voladj": close.pct_change(20, fill_method=None)
                  / ret.rolling(20).std().replace(0, float("nan")),
    }


def compose_signals(
    close: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, float]]:
    """构建多信号并加权合成综合分。

    Returns:
        (composite, normalized_signals, final_weights)
        - composite: date × code 综合信号（各分量截面 z 化后加权和）。
        - normalized_signals: 每个信号截面 z 化后的面板。
        - final_weights: 实际使用的权重（含未出现在 close 的清洗）。
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    sig_all = build_signals(close)

    norm: dict[str, pd.DataFrame] = {}
    for name in weights:
        if name in sig_all:
            norm[name] = _zscore_rowwise(sig_all[name])

    # 剔除在 close 中无法构建的信号，避免权重悬空
    final_weights = {n: w for n, w in weights.items() if n in norm}
    if not final_weights:
        raise ValueError(f"权重信号 {list(weights)} 均无法构建")

    composite = sum(final_weights[n] * norm[n] for n in final_weights)
    return composite, norm, final_weights