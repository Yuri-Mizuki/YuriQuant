"""东吴 LLM-MCTS 的 Seed 层：29 个 Alpha158 rolling 因子（窗口统一 20）。

东吴研报 §3「日频 Seed 定向改造」用 **29 个 Alpha158 Seed、窗口统一 %d=20**
（代表案例 vstd20 / std20 / cntn20 / rank20 全部对应 rolling 类）。Qlib
Alpha158 的 rolling 部分恰好是 **29 个类别**（ROC…VSUMD）× 5 窗口，本项目
``factor/alpha158.py`` 已全量实现——本模块取 29 类 × w=20 作为 Seed 集。

两类表达，各司其职：

- **求值**走 :data:`factor.alpha158.ALPHA158` 的原生 callable（零转译风险，
  与主实验因子库逐位同源）；``display`` 只是给 LLM 读的伪代码。
- **display**（研报语法风格）含 7 个 Seed 专属算子（Slope/Rsquare/Resi/
  Quantile/Rank/IdxMax/IdxMin——不在 ``REPORT_OPERATORS`` 变体空间里）。
  LLM 变体**只允许标准算子集**（与 AI97 llm_pool 同一空间，见
  ``check_report_formula``），Seed 专属算子仅出现在 prompt 的 Seed 展示中，
  ``prompt`` 里显式声明"变体不得使用 Seed 专属算子"。

复现边界：东吴未公布 29 Seed 的确切清单，本清单按"rolling 全类 × w=20"
构成（与其案例因子名完全吻合），属自拟口径。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["SEEDS", "SEED_WINDOW", "SeedSpec", "compute_seed_panels"]

#: 东吴口径：Seed 窗口统一 20（研报原文 "%d=20"）。
SEED_WINDOW: int = 20


@dataclass(frozen=True)
class SeedSpec:
    """单个 Seed：名字 / alpha158 注册键 / 展示伪代码 / 经济含义。"""

    name: str          # 如 "VSTD20"（东吴案例口径的短名）
    key: str           # ALPHA158 注册键，如 "alpha158_VSTD20"
    display: str       # 研报语法风格伪代码（给 LLM 读，不求值）
    note: str          # 一句话经济含义（进 prompt）


def _s(name: str, display: str, note: str) -> SeedSpec:
    return SeedSpec(name=name, key=f"alpha158_{name}{SEED_WINDOW}",
                    display=display, note=note)


_W = SEED_WINDOW

#: 29 个 Seed = Alpha158 rolling 全类 × w=20（顺序同 alpha158.py 注册序）。
SEEDS: tuple[SeedSpec, ...] = (
    _s("ROC", f"Div(Ref($CLOSE, {_W}), $CLOSE)", "20 日前收盘价/现价，反转锚"),
    _s("MA", f"Div(Mean($CLOSE, {_W}), $CLOSE)", "20 日均价相对现价的偏离"),
    _s("STD", f"Div(Std($CLOSE, {_W}), $CLOSE)", "20 日价格波动率（变异系数）"),
    _s("BETA", f"Div(Slope($CLOSE, {_W}), $CLOSE)",
       "20 日趋势斜率（Seed 专属算子 Slope）"),
    _s("RSQR", f"Rsquare($CLOSE, {_W})", "趋势拟合优度 R²（Seed 专属算子）"),
    _s("RESI", f"Div(Resi($CLOSE, {_W}), $CLOSE)", "趋势回归残差（Seed 专属算子）"),
    _s("MAX", f"Div(Max($HIGH, {_W}), $CLOSE)", "20 日最高价相对现价"),
    _s("MIN", f"Div(Min($LOW, {_W}), $CLOSE)", "20 日最低价相对现价"),
    _s("QTLU", f"Div(Quantile($CLOSE, {_W}, 0.8), $CLOSE)",
       "20 日 80% 分位相对现价（Seed 专属算子）"),
    _s("QTLD", f"Div(Quantile($CLOSE, {_W}, 0.2), $CLOSE)",
       "20 日 20% 分位相对现价（Seed 专属算子）"),
    _s("RANK", f"Rank($CLOSE, {_W})", "收盘价 20 日时序百分位（Seed 专属算子）"),
    _s("RSV", f"Div(Sub($CLOSE, Min($LOW, {_W})), "
              f"Sub(Max($HIGH, {_W}), Min($LOW, {_W})))",
       "价格在 20 日区间中的位置"),
    _s("IMAX", f"IdxMax($HIGH, {_W})", "最高价距当期天数（Seed 专属算子）"),
    _s("IMIN", f"IdxMin($LOW, {_W})", "最低价距当期天数（Seed 专属算子）"),
    _s("IMXD", f"Sub(IdxMax($HIGH, {_W}), IdxMin($LOW, {_W}))",
       "高低点时间差（Seed 专属算子）"),
    _s("CORR", f"Corr($CLOSE, Log($VOLUME), {_W})", "量价相关性"),
    _s("CORD", f"Corr(Div($CLOSE, Ref($CLOSE, 1)), Div($VOLUME, "
               f"Ref($VOLUME, 1)), {_W})", "量价变化相关性"),
    _s("CNTP", f"Mean(Greater(Sub($CLOSE, Ref($CLOSE, 1)), 0), {_W})",
       "20 日上涨天数占比"),
    _s("CNTN", f"Mean(Less(Sub($CLOSE, Ref($CLOSE, 1)), 0), {_W})",
       "20 日下跌天数占比"),
    _s("CNTD", f"Sub(Mean(Greater(Sub($CLOSE, Ref($CLOSE, 1)), 0), {_W}), "
               f"Mean(Less(Sub($CLOSE, Ref($CLOSE, 1)), 0), {_W}))",
       "20 日涨跌天数差"),
    _s("SUMP", f"Div(Sum(Greater(Sub($CLOSE, Ref($CLOSE, 1)), 0), {_W}), "
               f"Sum(Abs(Sub($CLOSE, Ref($CLOSE, 1))), {_W}))",
       "20 日总涨幅占绝对波动比"),
    _s("SUMN", f"Div(Sum(Greater(Sub(Ref($CLOSE, 1), $CLOSE), 0), {_W}), "
               f"Sum(Abs(Sub($CLOSE, Ref($CLOSE, 1))), {_W}))",
       "20 日总跌幅占绝对波动比"),
    _s("SUMD", f"Div(Sub(Sum(Greater(Sub($CLOSE, Ref($CLOSE, 1)), 0), {_W}), "
               f"Sum(Greater(Sub(Ref($CLOSE, 1), $CLOSE), 0), {_W})), "
               f"Sum(Abs(Sub($CLOSE, Ref($CLOSE, 1))), {_W}))",
       "20 日涨跌净值占比"),
    _s("VMA", f"Div(Mean($VOLUME, {_W}), $VOLUME)", "20 日均量/现量"),
    _s("VSTD", f"Div(Std($VOLUME, {_W}), $VOLUME)", "成交量变异系数"),
    _s("WVMA", f"Div(Std(Mul(Abs(Sub(Div($CLOSE, Ref($CLOSE, 1)), 1)), $VOLUME), {_W}), "
               f"Mean(Mul(Abs(Sub(Div($CLOSE, Ref($CLOSE, 1)), 1)), $VOLUME), {_W}))",
       "成交量加权的价格波动率"),
    _s("VSUMP", f"Div(Sum(Greater(Sub($VOLUME, Ref($VOLUME, 1)), 0), {_W}), "
                f"Sum(Abs(Sub($VOLUME, Ref($VOLUME, 1))), {_W}))",
       "20 日放量占比"),
    _s("VSUMN", f"Div(Sum(Greater(Sub(Ref($VOLUME, 1), $VOLUME), 0), {_W}), "
                f"Sum(Abs(Sub($VOLUME, Ref($VOLUME, 1))), {_W}))",
       "20 日缩量占比"),
    _s("VSUMD", f"Div(Sub(Sum(Greater(Sub($VOLUME, Ref($VOLUME, 1)), 0), {_W}), "
                f"Sum(Greater(Sub(Ref($VOLUME, 1), $VOLUME), 0), {_W})), "
                f"Sum(Abs(Sub($VOLUME, Ref($VOLUME, 1))), {_W}))",
       "20 日量能净变化占比"),
)

assert len(SEEDS) == 29, f"东吴口径 29 Seed，当前 {len(SEEDS)}"


def compute_seed_panels(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """在原始价量面板上求值全部 29 个 Seed（原生 ALPHA158 callable）。

    Args:
        panel: 需含 open/high/low/close/volume/amount（vwap 可选）的宽面板。
    Returns:
        ``{SeedSpec.name: date×code 面板}``（±inf→NaN，与 compute_alpha158 同清理）。
    """
    from factor.alpha158 import ALPHA158
    from factor.alpha_base import AlphaData

    d = AlphaData(panel)
    out: dict[str, pd.DataFrame] = {}
    for spec in SEEDS:
        fp = ALPHA158[spec.key](d)
        out[spec.name] = fp.replace([np.inf, -np.inf], np.nan)
    return out
