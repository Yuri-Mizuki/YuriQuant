"""因子层入口。"""
from factor.base import Factor, FactorEngine
from factor.library import (
    ALL_FACTORS,
    Amplitude,
    Momentum,
    PriceMA,
    Reversal,
    Turnover,
    Volatility,
    VolumeRatio,
)
from factor.operators import (
    CS_OPS, DEFAULT_FEATURES, DEFAULT_WINDOWS, ELEMENT_OPS, TS_OPS,
    OpSpec, all_operators, op_registry,
)
from factor.preprocessing import (
    neutralize,
    preprocess_factor,
    standardize_rank,
    standardize_zscore,
    winsorize_mad,
    winsorize_quantile,
)

__all__ = [
    "Factor", "FactorEngine", "ALL_FACTORS",
    "Momentum", "Reversal", "Volatility", "Amplitude",
    "Turnover", "VolumeRatio", "PriceMA",
    "winsorize_mad", "winsorize_quantile",
    "standardize_zscore", "standardize_rank",
    "neutralize", "preprocess_factor",
    # 算子空间
    "ELEMENT_OPS", "TS_OPS", "CS_OPS", "OpSpec",
    "all_operators", "op_registry", "DEFAULT_WINDOWS", "DEFAULT_FEATURES",
]
# 注意：factor.mining / factor.genetic_mining / factor.synthesis 不在此导出，
# 避免引入 scipy/deap 等重依赖到所有调用方；按需 `from factor.synthesis import ...`。
