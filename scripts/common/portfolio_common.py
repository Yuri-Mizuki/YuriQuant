"""跨实验脚本共享的组合编排组件（无 CLI 入口）

2026-09-11 由 ``scripts/pipelines/run_model_portfolio.py`` **下沉**：该模块是主实验
入口（全A正交化管线），却被 ``buffer_tune`` / ``freq_tune`` / ``multiyear_oos`` /
``rolling_grid_alla`` 等脚本 **反向 import 其内部函数**——入口脚本成了依赖库。

2026-11-18 收敛：只保留 :func:`neutralize_panel`（``rolling_grid_alla`` 在用）。
同模块原有的 ``build_model_panel`` / ``load_index_benchmark`` /
``build_style_covariates_panel`` / ``LEGACY_DATASET`` 全部是 **HS300 口径**
（读死 ``hs300_2022_2025`` 库 + ``daily_hs300.parquet``），唯一消费者是
``buffer_tune`` / ``freq_tune``——两者随 HS300 时代下线一并删除，故这些函数
与其 ``e2e_common.select_features`` 依赖同时移除。
"""
from __future__ import annotations

__all__ = ["neutralize_panel"]


def neutralize_panel(signal, cov):
    """信号层风格中性化（保留给对照实验；主口径 neutralize=false 不走此路径）。

    ``cov`` 是 ``{"size": 市值面板, "industry": 行业面板, **额外连续协变量}``
    形态的协变量集合，这里解包后交给 :func:`factor.preprocessing.neutralize`
    这个唯一真源。
    """
    from factor.preprocessing import neutralize
    size = cov.get("size")
    ind = cov.get("industry")
    extra = {k: v for k, v in cov.items() if k not in ("size", "industry")}
    return neutralize(signal, market_cap_panel=size, industry_panel=ind,
                      extra_covariates=extra)
