"""模型超参默认值（单一真源）

2026-09-11 由 ``scripts/pipelines/run_model_portfolio.py`` **下沉**到此处。

背景：该常量原本定义在实验入口脚本 ``run_model_portfolio.py`` 里，却被
主实验 ``scripts/pipelines/rolling_grid_alla.py``、生产 ``scripts/pipelines/alla_daily_rank.py``
和 ``scripts/evaluation/multiyear_oos.py`` **反向 import**——实验脚本成了生产链路的
依赖库，属归属倒挂。超参是模型层的公共契约，归属 ``model/``。

消费方一律从这里取，禁止在脚本里复制字面量（防网格实验与生产漂移）。
"""
from __future__ import annotations

__all__ = ["DEFAULT_MODEL_PARAMS"]

#: 固化模型超参（与 ``config/settings.yaml`` 的 ``model_portfolio`` 段配套）。
#: - ``gbdt``   : 主实验/生产口径的 LightGBM 回归参数
#: - ``ridge``  : 无超参（占位，便于按模型名统一索引）
#: - ``ranker`` : LambdaRank 变体（``objective="rank_xendcg"``）
DEFAULT_MODEL_PARAMS: dict[str, dict] = {
    "gbdt":   dict(n_estimators=150, learning_rate=0.03, num_leaves=15,
                   min_child_samples=50, seed=0),
    "ridge":  {},
    "ranker": dict(n_estimators=200, learning_rate=0.05, num_leaves=15,
                   min_child_samples=50, seed=0, labels_bins=2,
                   objective="rank_xendcg"),
}
