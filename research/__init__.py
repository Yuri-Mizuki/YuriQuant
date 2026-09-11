"""研究层入口。

**惰性导入（2026-09-11 P2，与 ``optimize/__init__`` 同配方）**：本包多个子模块
拖重第三方依赖——``xlsx_report`` → matplotlib + openpyxl、``factor_analysis``
→ scipy（合计 ~7s 级冷启成本）。此前 ``__init__`` 静态 import 三个子模块，
任何 ``import research``（包括经 monitoring / optimize 的函数级引用首次触发）
都要付全款。现改为 PEP 562 模块级 ``__getattr__`` 按需加载：
``from research import FactorLibrary`` / ``from research import factor_analysis``
等用法不变，首次访问时才加载并缓存。
"""
from __future__ import annotations

import importlib

#: 公开 API 名 → 所在子模块（首次访问时加载并缓存到 globals）
_LAZY_API: dict[str, str] = {
    "calc_ic_decay": "research.factor_analysis",
    "calc_ic_series": "research.factor_analysis",
    "calc_ir": "research.factor_analysis",
    "factor_summary": "research.factor_analysis",
    "quantile_backtest": "research.factor_analysis",
    "FactorLibrary": "research.factor_library",
    "generate_excel_report": "research.xlsx_report",
}

#: 子模块本身也惰性加载（`research.xlsx_report` 这类属性访问不报 AttributeError）
_LAZY_SUBMODULES = ("factor_analysis", "factor_library", "xlsx_report")

__all__ = [
    "calc_ic_series", "calc_ir", "calc_ic_decay",
    "quantile_backtest", "factor_summary",
    "FactorLibrary",
    "generate_excel_report",
]


def __getattr__(name: str):
    """PEP 562：按需加载公开名 / 子模块（避免 import research 付重依赖代价）。"""
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f"research.{name}")
        globals()[name] = mod
        return mod
    if name in _LAZY_API:
        mod = importlib.import_module(_LAZY_API[name])
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_API) | set(_LAZY_SUBMODULES))
