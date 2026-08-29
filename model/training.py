"""
模型训练 —— 02 模型层「模型训练」。

当前系统内 ML 能力集中在因子合成（ML stacking），位于 ``factor/synthesis.py``
的 ``synthesize_stacking`` 系列。这里做薄封装 + 自动注册，把「训练一次模型」
变成可追踪、可迭代的流程对象：

- ``train_stacking_model``：调 factor/synthesis 的 stacking 方法，返回预测面板与描述
- ``train_predictor_model``：独立截面预测模型（模型层 ③ Predictor），CV 产出 OOS 面板
- ``train_and_register``：训练 + 评价 + 注册进 ModelRegistry（血缘/指纹/区间齐全）；
  ``kind="ml_stacking"``（默认）走因子合成，``kind="predictor"`` 走独立预测模型
  ——训练产物统一走 ModelRegistry 落盘，保证「模型迭代」链条可追溯。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from factor.synthesis import (
    CompositeInput,
    synthesize_stacking,
    synthesize_stacking_gbdt,
    synthesize_stacking_gbdt_tuned,
    synthesize_stacking_lambdarank,
)
from model.registry import ModelRegistry, default_model_root
from stats.ic import calc_ic_series, calc_ir
from stats.robust_stats import nw_tstat

__all__ = ["train_stacking_model", "train_predictor_model", "train_and_register"]

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


def train_predictor_model(
    feature_panels: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame],
    labels: pd.DataFrame,
    method: str = "gbdt",
    n_splits: int = 5,
    embargo_days: int = 5,
    horizon: int | None = None,
    target_mode: str | None = None,
    **kwargs: Any,
) -> dict:
    """训练一个独立截面预测模型（模型层 ③），CV 产出 OOS 预测面板。

    与 ``train_stacking_model`` 的区别：特征可以是任意因子面板（不必是精选
    合成组件），目标由 LabelBuilder 预先构建（多 horizon 支持），模型经
    ``model.predictor.PREDICTORS`` 注册表分发——预测器而非合成器。

    Args:
        feature_panels: {name: date×code 面板}（已对齐/标准化，如
            ``model.features.build_feature_set`` 的输出）。
        labels: LabelBuilder 产出的标签面板（mode 为截面单调变换时，
            IC 口径与原始 horizon 收益一致）。
        method: PREDICTORS 键（ridge / gbdt）。
        n_splits / embargo_days: 时序 CV 折数与隔离带（embargo 应 >= horizon）。
        horizon / target_mode: 仅记录进 spec（标签由调用方构建，这里存档溯源）。
        **kwargs: 透传预测器超参。

    Returns:
        dict: {"panel": OOS 预测面板, "spec": 训练规格,
               "ic_mean" / "ic_ir" / "ic_t_nw": 预测面板 vs 标签的 IC 统计}
    """
    from model.predictor import PREDICTORS, fit_predict_oos

    if method not in PREDICTORS:
        raise ValueError(f"未知预测器 {method!r}，可选: {sorted(PREDICTORS)}")
    if not isinstance(feature_panels, Mapping):
        feature_panels = {f"feature{i}": p for i, p in enumerate(feature_panels)}
    if not feature_panels:
        raise ValueError("feature_panels 为空")

    panel = fit_predict_oos(
        PREDICTORS[method], feature_panels, labels,
        n_splits=n_splits, embargo_days=embargo_days, **kwargs,
    )
    spec = {
        "method": method,
        "features": sorted(feature_panels.keys()),
        "n_splits": n_splits,
        "embargo_days": embargo_days,
        "horizon": horizon,
        "target_mode": target_mode,
        **kwargs,
    }
    ic_series = calc_ic_series(panel, labels)
    ic_valid = ic_series.dropna()
    t_nw = nw_tstat(ic_valid.values)[0] if len(ic_valid) > 1 else 0.0
    return {
        "panel": panel, "spec": spec,
        "ic_mean": float(ic_series.mean()),
        "ic_ir": calc_ir(ic_series),
        "ic_t_nw": float(t_nw),
    }


def train_and_register(
    name: str,
    components: Sequence[CompositeInput | pd.DataFrame] | Mapping[str, pd.DataFrame],
    returns_panel: pd.DataFrame,
    method: str = "ridge",
    kind: str = "ml_stacking",
    fingerprint: str | None = None,
    train_begin: int | None = None,
    train_end: int | None = None,
    parents: Sequence[str] | None = None,
    registry: ModelRegistry | None = None,
    **kwargs: Any,
) -> tuple[str, dict]:
    """训练 + 评价 + 注册一次完整调用（模型层的标准入口，训练即注册）。

    Args:
        kind: "ml_stacking"（components=合成因子，returns_panel=未来一期收益）
              | "predictor"（components=特征面板 dict，returns_panel=LabelBuilder
                标签；method 默认建议 "gbdt"）。
        其余参数：kind="predictor" 时透传 ``train_predictor_model``
        （n_splits / embargo_days / horizon / target_mode / 预测器超参），
        否则透传 ``train_stacking_model``。

    Returns:
        (model_id, result_dict)：result 含 panel/spec 与 IC 统计。
    """
    reg = registry or ModelRegistry(default_model_root())

    if kind == "predictor":
        if not isinstance(components, Mapping):
            raise TypeError('kind="predictor" 需要 components 为 {name: 面板} 特征字典')
        result = train_predictor_model(components, returns_panel, method=method, **kwargs)
        spec = dict(result["spec"])
        metrics = {"ic_mean": result["ic_mean"], "ic_ir": result["ic_ir"],
                   "ic_t_nw": result["ic_t_nw"]}
        if parents is None:
            parents = sorted(components.keys())
        mid = reg.register(
            name=name, kind="predictor", spec=spec,
            fingerprint=fingerprint, train_begin=train_begin, train_end=train_end,
            metrics=metrics, parents=parents,
        )
        return mid, result

    result = train_stacking_model(components, returns_panel, method=method, **kwargs)
    spec = dict(result["spec"])
    metrics = {"ic_mean": result["ic_mean"], "ic_ir": result["ic_ir"]}
    mid = reg.register(
        name=name, kind="ml_stacking", spec=spec,
        fingerprint=fingerprint, train_begin=train_begin, train_end=train_end,
        metrics=metrics, parents=parents,
    )
    return mid, result
