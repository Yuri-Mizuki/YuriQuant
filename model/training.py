"""
模型训练 —— 02 模型层「模型训练」。

当前系统内 ML 能力集中在因子合成（ML stacking），位于 ``factor/synthesis.py``
的 ``synthesize_stacking`` 系列。这里做薄封装 + 自动注册，把「训练一次模型」
变成可追踪、可迭代的流程对象：

- ``train_stacking_model``：调 factor/synthesis 的 stacking 方法，返回预测面板与描述
- ``train_and_register``：训练 + 用评价指标 + 注册进 ModelRegistry（血缘/指纹/区间齐全）

后续新增独立收益预测模型（如 GBDT / NN 直接预测收益）时，在同一接口下扩展，
训练产物统一走 ModelRegistry 落盘，保证「模型迭代」链条可追溯。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from factor.synthesis import (
    CompositeInput,
    build_components,
    synthesize_stacking,
    synthesize_stacking_gbdt,
    synthesize_stacking_gbdt_tuned,
    synthesize_stacking_lambdarank,
)
from model.registry import ModelRegistry, default_model_root
from research.factor_analysis import calc_ic_series, calc_ir

__all__ = ["train_stacking_model", "train_and_register"]

_METHODS = {
    "ridge": synthesize_stacking,
    "gbdt": synthesize_stacking_gbdt,
    "gbdt_tuned": synthesize_stacking_gbdt_tuned,
    "lambdarank": synthesize_stacking_lambdarank,
}


def _to_components(components: Sequence[CompositeInput | pd.DataFrame],
                   returns_panel: pd.DataFrame) -> list[CompositeInput]:
    """把入参统一成 list[CompositeInput]（允许直接传 date×code 面板）。"""
    out: list[CompositeInput] = []
    for i, c in enumerate(components):
        if isinstance(c, CompositeInput):
            out.append(c)
        else:
            out.append(CompositeInput(panel=c, name=f"component{i}",
                                      ic=calc_ic_series(c, returns_panel).mean()))
    return out


def train_stacking_model(
    components: Sequence[CompositeInput | pd.DataFrame],
    returns_panel: pd.DataFrame,
    method: str = "ridge",
    target_mode: str = "raw",
    **kwargs: Any,
) -> dict:
    """训练一个 ML stacking 合成模型（薄封装 factor/synthesis）。

    Args:
        components: 参与合成的因子（CompositeInput 或 date×code 面板）。
        returns_panel: 未来一期收益面板（与 IC 口径一致）。
        method: ridge / gbdt / gbdt_tuned / lambdarank。
        target_mode: 目标变换，透传给底层 stacking。
    Returns:
        dict: {"panel": 模型预测面板(date×code), "spec": 训练规格, "ic": 训练期 IC}
    """
    if method not in _METHODS:
        raise ValueError(f"未知模型方法 {method!r}，可选: {sorted(_METHODS)}")
    comps = _to_components(components, returns_panel)
    synth = _METHODS[method]
    try:
        panel = synth(comps, returns_panel, target_mode=target_mode, **kwargs)
    except TypeError:
        # 个别实现签名不一致时退化为默认参数重试
        panel = synth(comps, returns_panel, **kwargs)
    spec = {"method": method, "target_mode": target_mode,
            "components": [c.name for c in comps], **kwargs}
    ic_series = calc_ic_series(panel, returns_panel)
    return {"panel": panel, "spec": spec,
            "ic_mean": float(ic_series.mean()), "ic_ir": calc_ir(ic_series)}


def train_and_register(
    name: str,
    components: Sequence[CompositeInput | pd.DataFrame],
    returns_panel: pd.DataFrame,
    method: str = "ridge",
    fingerprint: str | None = None,
    train_begin: int | None = None,
    train_end: int | None = None,
    parents: Sequence[str] | None = None,
    registry: ModelRegistry | None = None,
    **kwargs: Any,
) -> tuple[str, dict]:
    """训练 + 评价 + 注册一次完整调用（模型层的标准入口）。

    Returns:
        (model_id, result_dict)：result 含 panel/spec/ic_mean/ic_ir。
    """
    reg = registry or ModelRegistry(default_model_root())
    result = train_stacking_model(components, returns_panel, method=method, **kwargs)
    spec = dict(result["spec"])
    metrics = {"ic_mean": result["ic_mean"], "ic_ir": result["ic_ir"]}
    mid = reg.register(
        name=name, kind="ml_stacking", spec=spec,
        fingerprint=fingerprint, train_begin=train_begin, train_end=train_end,
        metrics=metrics, parents=parents,
    )
    return mid, result
