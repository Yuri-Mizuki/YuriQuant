"""
分层守卫测试（2026-08-29 stats 公共层下沉后固化；2026-09-11 补 factor/model 与
optimize/strategy 边界）
================================================================================

静态扫描 + 运行时同一性校验，防止包级循环依赖与归属倒挂回潮：

1. factor/ 不得 import research（统计工具应走 stats/，历史违规已清除）
2. factor/ 不得 import model（模型层消费因子层，不能反向依赖）
3. ML stacking 实现必须落在 model/，factor/ 不得转发（2026-09-11 归属修复）
4. research/ 不得 import optimize（monitor 统计已下沉 stats/monitor.py）
5. stats/ 是纯统计底层：不得 import 任何业务包（只允许 numpy/pandas/scipy/stdlib）
6. 兼容转出口必须指向 stats 真源（同一对象，防 shim 漂移成第二实现）
7. strategy/ 是回测引擎的基础件：不得 import 上层包，也不得引入 cvxpy 等重依赖
8. 组合构建约束（无需风险模型）真源在 strategy.constraints，optimize 只做编排
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


def test_factor_does_not_import_model():
    """factor 层不得反向依赖 model（模型层消费因子层，不能倒挂）。"""
    offenders = []
    for f in (ROOT / "factor").rglob("*.py"):
        bad = _imports_of(f) & {"model"}
        if bad:
            offenders.append(f"{f.relative_to(ROOT)} -> {sorted(bad)}")
    assert not offenders, "factor -> model 违规:\n" + "\n".join(offenders)


def test_ml_stacking_lives_in_model_layer():
    """ML stacking 属模型层能力：实现必须在 model/，factor/ 不得转发。

    2026-09-11 归属修复（原 ``factor/synthesis.py`` 第 4 节）：stacking 拟合的是
    有监督模型，与因子层「确定性组合」职责不同，故迁入 :mod:`model.stacking`；
    factor 层不得留兼容转出口，否则倒挂会静默回潮。
    """
    import model.stacking as ms
    from factor import synthesis as fs

    for name in ("synthesize_stacking", "synthesize_stacking_gbdt",
                 "synthesize_stacking_gbdt_tuned", "synthesize_stacking_lambdarank"):
        assert hasattr(ms, name), f"model.stacking 缺少 {name}"
        assert not hasattr(fs, name), f"{name} 仍留在 factor 层（归属倒挂回潮）"

    # 因子层保留的确定性组合接口不得被搬走
    for name in ("synthesize_ic_weighted", "synthesize_pca",
                 "synthesize_orthogonal", "build_components"):
        assert hasattr(fs, name), f"factor.synthesis 缺少因子层接口 {name}"

    # 跨层共享的工具必须是同一对象（防复制成第二实现）
    assert fs.CompositeInput is ms.CompositeInput
    assert fs.long_matrix is ms.long_matrix


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


# strategy/ 必须保持"基础件"地位：回测引擎直接依赖它
_HEAVY_DEPS = {"cvxpy", "deap", "lightgbm", "torch", "shap"}


def test_strategy_layer_has_no_upward_or_heavy_deps():
    """strategy/ 是回测引擎的基础件：不得依赖上层包，也不得引入重依赖。"""
    offenders = []
    for f in (ROOT / "strategy").rglob("*.py"):
        bad = _imports_of(f) & ({"optimize", "backtest", "research", "model"} | _HEAVY_DEPS)
        if bad:
            offenders.append(f"{f.relative_to(ROOT)} -> {sorted(bad)}")
    assert not offenders, "strategy 层依赖违规:\n" + "\n".join(offenders)


def test_combination_constraints_live_in_strategy():
    """组合构建约束（无需风险模型）真源在 strategy.constraints，optimize 只做编排。

    2026-09-11 第三批口径统一：``optimize/portfolio.py`` 的信号构建与工程约束
    （行业中性 / 上下限 / 换手投影）**不需要协方差**，属"组合构建"而非"组合优化"，
    已下沉为 :mod:`strategy.constraints`（面板级、零三方依赖）。只有需要 Σ 的
    QP / HRP / BL 才留在 optimize 层。
    """
    import numpy as np
    import pandas as pd

    import strategy.constraints as sc
    from optimize import portfolio as op

    names = ("build_signal_weights", "neutralize_industry", "apply_bounds",
             "apply_turnover", "apply_constraints")
    for n in names:
        assert hasattr(sc, n), f"strategy.constraints 缺少 {n}"

    # optimize/portfolio.py 不得重新定义这些算子（只能转发，防第二实现）
    src = (ROOT / "optimize" / "portfolio.py").read_text(encoding="utf-8")
    for n in names:
        assert f"def {n}" not in src, f"optimize/portfolio.py 重新定义了 {n}（真源漂移）"

    # 门面行为 == 真源组合（逐位）
    rng = np.random.default_rng(0)
    p = pd.DataFrame(rng.normal(0, 1, (20, 10)),
                     index=pd.date_range("2024-01-01", periods=20, freq="B"),
                     columns=[f"C{i}" for i in range(10)])
    a = op.optimize_weights(p, method="factor_weighted", max_weight=0.2)
    b = sc.apply_constraints(sc.build_signal_weights(p, "factor_weighted"), max_weight=0.2)
    assert np.allclose(a.values, b.values, atol=1e-15)


def test_periods_per_year_single_source():
    """年化常数单一真源：stats 定义，backtest.metrics re-export 同一对象。"""
    import stats
    from backtest import metrics

    assert stats.PERIODS_PER_YEAR == metrics.PERIODS_PER_YEAR == 252


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
