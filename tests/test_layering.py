"""
分层守卫测试（2026-08-29 stats 公共层下沉后固化）
==================================================

静态扫描 + 运行时同一性校验，防止包级循环依赖回潮：

1. factor/ 不得 import research（统计工具应走 stats/，历史违规已清除）
2. research/ 不得 import optimize（monitor 统计已下沉 stats/monitor.py）
3. stats/ 是纯统计底层：不得 import 任何业务包（只允许 numpy/pandas/scipy/stdlib）
4. 兼容转出口必须指向 stats 真源（同一对象，防 shim 漂移成第二实现）
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# import 语句的正则（覆盖 from X import / import X 两种形式）
_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)|import\s+([\w.]+))", re.M)


def _imports_of(path: Path) -> set[str]:
    src = path.read_text(encoding="utf-8")
    roots = set()
    for m in _IMPORT_RE.finditer(src):
        mod = m.group(1) or m.group(2)
        if mod:
            roots.add(mod.split(".")[0])
    return roots


BUSINESS_PKGS = {
    "config", "data", "factor", "model", "optimize",
    "backtest", "research", "monitoring", "strategy", "scripts",
}


def test_factor_does_not_import_research():
    """factor 层不得反向依赖 research（IC/NW 统计走 stats）。"""
    offenders = []
    for f in (ROOT / "factor").rglob("*.py"):
        bad = _imports_of(f) & {"research"}
        if bad:
            offenders.append(f"{f.relative_to(ROOT)} -> {sorted(bad)}")
    assert not offenders, "factor -> research 违规:\n" + "\n".join(offenders)


def test_research_does_not_import_optimize():
    """research 层不得依赖 optimize（monitor 统计已下沉 stats/monitor.py）。"""
    offenders = []
    for f in (ROOT / "research").rglob("*.py"):
        bad = _imports_of(f) & {"optimize"}
        if bad:
            offenders.append(f"{f.relative_to(ROOT)} -> {sorted(bad)}")
    assert not offenders, "research -> optimize 违规:\n" + "\n".join(offenders)


def test_stats_is_pure():
    """stats/ 只依赖 numpy/pandas/scipy/stdlib，不得 import 任何业务包。"""
    offenders = []
    for f in (ROOT / "stats").rglob("*.py"):
        bad = _imports_of(f) & BUSINESS_PKGS
        if bad:
            offenders.append(f"{f.relative_to(ROOT)} -> {sorted(bad)}")
    assert not offenders, "stats 引入了业务依赖:\n" + "\n".join(offenders)


def test_compat_shims_point_to_stats():
    """兼容转出口与 stats 真源必须是同一对象（防 shim 漂移成第二实现）。"""
    import stats.ic
    import stats.monitor
    import stats.robust_stats

    from research import factor_analysis as fa
    from research import robust_stats as rs
    from optimize import monitor as om

    assert fa.calc_ic_series is stats.ic.calc_ic_series
    assert fa.calc_ir is stats.ic.calc_ir
    assert fa.calc_ic_decay is stats.ic.calc_ic_decay
    assert fa.quantile_backtest is stats.ic.quantile_backtest
    assert fa.factor_autocorr is stats.ic.factor_autocorr
    assert rs.nw_tstat is stats.robust_stats.nw_tstat
    assert rs.ols_newey_west is stats.robust_stats.ols_newey_west
    assert om.monitor_report is stats.monitor.monitor_report
    assert om.rolling_ic is stats.monitor.rolling_ic


def test_periods_per_year_single_source():
    """年化常数单一真源：stats 定义，backtest.metrics re-export 同一对象。"""
    import stats
    from backtest import metrics

    assert stats.PERIODS_PER_YEAR == metrics.PERIODS_PER_YEAR == 252


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
