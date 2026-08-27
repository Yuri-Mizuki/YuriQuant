"""模型消费出口 —— 模型层 ⑤。

预测面板的分发：策略 / 优化器直接吃面板（零适配，date×code 即接口），
因子库经 ``register_model_as_factor`` 回写（联动②：模型预测注册为
meta 因子，即刻享受因子库的 monitor 衰减监控 / regime 分析 / compare
排行 / canonical 回测全套基础设施）。

命名约定：``model:<名称>_h<horizon>``（如 ``model:lgbm_h5``），同名再注册
即版本迭代（因子库 register 语义）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

__all__ = ["register_model_as_factor"]


def register_model_as_factor(
    name: str,
    pred_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    parents: Sequence[str] | None = None,
    dataset: str | None = None,
    model_id: str | None = None,
    horizon: int | None = None,
    oos: bool = True,
    note: str = "",
    root: str | Path | None = None,
) -> dict:
    """把模型预测面板注册进因子库（血缘回写）。

    Args:
        name: 因子名（建议 ``model:<名称>_h<horizon>``）。
        pred_panel: date×code 预测面板（OOS 或 CV 拼接面板）。
        returns_panel: 与预测口径一致的前瞻收益面板（horizon 日）——
            因子库 IC / 回测预计算以此为目标。
        parents: 父特征（因子）名列表 → 血缘。
        dataset: 因子库数据集名（mock → "mock"；真实按 mine_factors 约定推导）。
        model_id: ModelRegistry 里的模型 ID（写进 note，双向可追溯）。
        horizon: 预测视野（写进 note）。
        oos: 预测是否样本外（True → maturity="oos_verified"，False →
            "experimental"——例如全样本拟合的面板必须标 experimental）。
        note: 附加备注。
        root: 因子库根目录（测试隔离用；默认走 config factor_library.root）。

    Returns:
        因子库 register() 的返回（该因子的 registry 行 dict）。
    """
    from research.factor_library import FactorLibrary

    lib = FactorLibrary(root=root, dataset=dataset)
    meta = []
    if model_id:
        meta.append(f"model_id={model_id}")
    if horizon is not None:
        meta.append(f"horizon={horizon}d")
    if oos:
        meta.append("OOS")
    full_note = "; ".join(meta + ([note] if note else []))

    return lib.register(
        name, pred_panel, returns_panel,
        kind="composite",
        formula=name,
        parents=list(parents) if parents else [],
        source=f"model:{name}",
        family="模型",
        frequency="日频",
        maturity="oos_verified" if oos else "experimental",
        note=full_note,
    )
