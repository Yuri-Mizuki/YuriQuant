"""兼容转出口 —— 判据真源已下沉到 :mod:`research.dup_criteria`（2026-11-19）。

本文件**不再持任何逻辑**。历史上它是真源，但 ``research.factor_criteria``
需要复用它，而 research 反向 import scripts 属分层倒挂（``factor_library``
此前已在函数级这么干了两处）。因此判据本体下沉到 research 层，这里只做转发。

🚨 **禁止在本文件加任何新逻辑或新阈值** —— 那会立刻变成第二份实现。
守卫断言见 ``tests/test_factor_criteria.py::test_dup_criteria_shim_is_same_object``：
``scripts.factors.dup_criteria.classify is research.dup_criteria.classify``。
"""
from __future__ import annotations

from research.dup_criteria import (
    CORR_TRIGGER,
    MIN_SPAN,
    REL_STD_DEGENERATE,
    REL_STD_TINY,
    RESID_IC_NEGLIGIBLE,
    TIER_T1,
    TIER_T2,
    TIER_T3,
    V_BELOW_TRIGGER,
    V_DEGENERATE,
    V_INCREMENT,
    V_INDEPENDENT,
    V_TINY,
    V_UNTESTED,
    Verdict,
    classify,
    classify_independent,
    explain,
    should_check,
    tier_of,
)

__all__ = [
    "CORR_TRIGGER", "MIN_SPAN", "REL_STD_DEGENERATE", "REL_STD_TINY",
    "RESID_IC_NEGLIGIBLE", "TIER_T1", "TIER_T2", "TIER_T3",
    "V_BELOW_TRIGGER", "V_DEGENERATE", "V_INCREMENT", "V_INDEPENDENT",
    "V_TINY", "V_UNTESTED", "Verdict", "classify", "classify_independent",
    "explain", "should_check", "tier_of",
]
