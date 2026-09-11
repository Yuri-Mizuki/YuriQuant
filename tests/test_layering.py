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
9. 模型超参真源在 model.params，scripts 层不得持第二份定义（2026-09-11 下沉）
10. 实验入口脚本不得被其他 scripts 反向 import（2026-09-11 下沉）
11. 跨模块引用的私有函数必须公开化：scripts 不得 import 带下划线前缀的名字
    （2026-09-11 P1#3 倒挂收口）
12. 入库代码不得 import gitignored 目录（scripts/oneoff 等）——否则干净 clone /
    生产环境必 ImportError（2026-09-11 P0+P1：全A 构建管线迁入 scripts/builders/）
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


def test_model_params_live_in_model_layer():
    """模型超参真源在 model.params；scripts 层不得持第二份定义。

    2026-09-11 下沉：原 ``scripts/run_model_portfolio.DEFAULT_MODEL_PARAMS``
    被主实验 ``rolling_grid_alla`` 与生产 ``alla_daily_rank`` **反向 import**
    ——实验入口脚本成了生产链路的依赖库。超参是模型层公共契约，归 ``model/``。
    """
    from model.params import DEFAULT_MODEL_PARAMS

    assert set(DEFAULT_MODEL_PARAMS) == {"gbdt", "ridge", "ranker"}

    offenders = []
    for f in (ROOT / "scripts").rglob("*.py"):
        if {"oneoff", "archive"} & set(f.parts):
            continue
        src = f.read_text(encoding="utf-8")
        if re.search(r"^DEFAULT_MODEL_PARAMS\s*[:=]", src, re.M):
            offenders.append(str(f.relative_to(ROOT)))
    assert not offenders, (
        "scripts 层重新定义了 DEFAULT_MODEL_PARAMS（应 import model.params）:\n"
        + "\n".join(offenders))


def test_experiment_entry_scripts_are_not_imported_by_other_scripts():
    """实验入口脚本不得被其他 scripts 反向 import（防"入口变依赖库"回潮）。

    2026-09-11 收口：``scripts/pipelines/run_model_portfolio.py`` 是主实验入口（全A正交化
    管线），却把 ``DEFAULT_MODEL_PARAMS`` / ``default_costs`` 以及 4 个 legacy
    组件定义在自身，被 ``rolling_grid_alla``（主实验）、``alla_daily_rank``（生产）
    与 ``buffer_tune`` / ``freq_tune`` / ``multiyear_oos`` 反向 import。现四类
    共享件均已下沉：超参 → :mod:`model.params`，成本 → :mod:`backtest.costs`，
    legacy 组件 → :mod:`scripts.common.portfolio_common`。本守卫钉住"入口脚本只进不出"。
    """
    entry_mods = {"scripts.pipelines.run_model_portfolio"}
    entry_files = {m.rsplit(".", 1)[-1] + ".py" for m in entry_mods}
    offenders = []
    for f in (ROOT / "scripts").rglob("*.py"):
        if {"oneoff", "archive"} & set(f.parts) or f.name in entry_files:
            continue
        src = f.read_text(encoding="utf-8")
        for mod in entry_mods:
            pkg, leaf = mod.rsplit(".", 1)
            pats = (
                rf"^\s*from\s+{re.escape(mod)}\s+import\b",       # from scripts.pkg.mod import x
                rf"^\s*import\s+{re.escape(mod)}\b",              # import scripts.pkg.mod
                rf"^\s*from\s+{re.escape(pkg)}\s+import\s+[^\n]*\b{re.escape(leaf)}\b",  # from scripts.pkg import mod
            )
            if any(re.search(p, src, re.M) for p in pats):
                offenders.append(f"{f.relative_to(ROOT)} -> {mod}")
    assert not offenders, (
        "实验入口脚本被其他 scripts 反向 import（应改用共享模块）:\n"
        + "\n".join(offenders))


def test_periods_per_year_single_source():
    """年化常数单一真源：stats 定义，backtest.metrics re-export 同一对象。"""
    import stats
    from backtest import metrics

    assert stats.PERIODS_PER_YEAR == metrics.PERIODS_PER_YEAR == 252


# ---------------------------------------------------------------------------
# 私有名跨模块引用守卫（2026-09-11 P1#3 / 第四批 / P0+P1 收口）
# ---------------------------------------------------------------------------
#: 业务包根（这些包内的私有名不得被跨模块 import）
_PRIV_GUARD_PKGS = {"config", "data", "factor", "model", "research", "stats",
                    "strategy", "optimize", "backtest", "monitoring", "scripts"}

#: tests 白盒例外：允许测试引用的 (模块, 私有名)。这些名**仍为私有**、未对外承诺，
#: 只是测试需要直接驱动内部实现（构造边界输入、逐位对拍）。新增例外请写进这里并
#: 注明理由——**生产代码一律不得引用**，这条由本守卫强制。
_TESTS_ONLY_PRIVATE: set[tuple[str, str]] = {
    # 私有名前两批收口时保留的"仅测试白盒"用例
    ("factor.genetic_mining", "_adjust_crowding"),
    ("factor.genetic_mining", "_restore_crowding"),
    ("factor.genetic_mining", "_dedup_hof_by_correlation"),
    ("factor.genetic_mining", "_ensure_creator"),
    ("factor.genetic_mining", "_seg_ic_stats"),
    ("scripts.pipelines.rolling_grid_alla", "_yearly_metrics"),
    ("scripts.pipelines.rolling_grid_alla", "_rebalance_days_validated"),
    ("scripts.pipelines.alla_daily_rank", "_limit_from_daily"),
    ("scripts.pipelines.alla_daily_rank", "_ALPHA_PREFIXES"),
    ("scripts.pipelines.alla_daily_rank", "_CONSTRUCTED_KEYS"),
    ("scripts.pipelines.alla_daily_rank", "_HOLDER_NUM_KEYS"),
    ("scripts.pipelines.alla_daily_rank", "_HOLDER_TOP_KEYS"),
    ("scripts.pipelines.alla_daily_rank", "_PLEDGE_KEYS"),
    ("scripts.factors.mine_factors", "_apply_gtja_preset"),
    ("scripts.pipelines.e2e_backtest", "_enforce_caps"),
    # 2026-09-11 把守卫改为**通用规则**后新暴露的测试白盒用例（生产侧已一并清干净）
    ("backtest.engine", "_apply_executable_mask"),
    ("research.factor_library", "_coerce_date"),
    ("research.html_report", "_fmt"),
    ("research.html_report", "_month_cell_style"),
    ("research.html_report", "_monthly_html"),
    ("research.html_report", "_monthly_table"),
    ("optimize.multi_period", "_rebalance_days"),
    ("optimize.solver", "_risk_parity_ccd"),
    ("model.stacking", "_time_fold_masks"),
    ("data.textmining.source_cninfo", "_fmt"),
    ("data.textmining.source_cninfo", "_CATEGORY_MAP"),
    ("data.textmining.source_cninfo", "_download_pdf_text"),
    ("data.textmining.source_ths", "_parse_report_json"),
}

#: 暂豁免的**被引用方**前缀：
#: - ``scripts.textmining`` —— 另一会话正在改（同目录 8 个文件未提交），待其落地后并入；
#: - ``scripts.oneoff`` —— 整目录 gitignored，属本地脚本、不是入库契约。
_PRIVATE_IMPORT_EXEMPT_PREFIXES = ("scripts.textmining", "scripts.oneoff")


def _private_names_from_imports(path: Path) -> set[tuple[str, str]]:
    """提取 ``from <mod> import _x, _y`` 形式的 (mod, 私有名) 对。"""
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - 语法错误由别处覆盖
        return set()
    out: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                if a.name.startswith("_"):
                    out.add((node.module, a.name))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("_"):
                    out.add((a.name.rsplit(".", 1)[0], a.name.rsplit(".", 1)[-1]))
    return out


def test_no_private_names_cross_module():
    """业务包内的私有名不得被**跨模块** import（下划线是"别用"的约定，不是 API）。

    2026-09-11 三轮收口解决了 3 批同类倒挂：
    - P1#3：`data.cache_helpers` 的 2 个 PIT 函数 + `factor.genetic_mining` 的
      5 个适应度组件（被 10 个 scripts 引用）；
    - 第四批：`scripts` 层生产链 8 处（`rolling_grid_alla.existence_mask` 被生产
      `alla_daily_rank` 用等）；
    - P0+P1：全A 构建管线 7 处（`scripts/builders/*` 内部的
      `add_single_quarter` / `add_ttm_yoy` / `sq_growth_long` / `year_offset` /
      `build_panels` / `pit_holder_num` / `pit_share_holder`）。

    本守卫用**通用 AST 规则**（不再逐个模块列表）自动覆盖新增代码：
    只要不是 tests 白名单、不在豁免前缀内，跨模块 import 私有名即失败。定义模块
    自身内部的私有调用不受影响。
    """
    offenders: list[str] = []
    for f in sorted(ROOT.rglob("*.py")):
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        if set(f.parts) & {".git", "__pycache__", ".venv", "venv", "node_modules"}:
            continue
        if rel.startswith(("reports/", "scripts/oneoff/")) or f.name.startswith("_"):
            continue
        file_mod = rel[:-3].replace("/", ".")
        if file_mod.endswith(".__init__"):
            file_mod = file_mod[: -len(".__init__")]
        is_test = rel.startswith("tests/")
        for mod, name in _private_names_from_imports(f):
            if mod.split(".")[0] not in _PRIV_GUARD_PKGS:
                continue
            if mod == file_mod:                      # 自身模块，非跨模块
                continue
            if mod.startswith(_PRIVATE_IMPORT_EXEMPT_PREFIXES):
                continue
            if is_test and (mod, name) in _TESTS_ONLY_PRIVATE:
                continue
            offenders.append(f"{rel}: from {mod} import {name}")
    assert not offenders, (
        "跨模块 import 了私有名（请公开化后再引用）:\n" + "\n".join(offenders))


def test_no_tracked_code_imports_gitignored_dirs():
    """入库代码不得 import **gitignored 目录**（否则干净 clone / 生产环境必 ImportError）。

    2026-09-11 修复：``scripts/pipelines/alla_daily_rank.py``（生产每日推理，注册为 Windows
    计划任务 ``YuriQuant AllaDailyRank``）原有 6 处 ``from scripts.oneoff.* import``，
    而 ``.gitignore`` 第 41 行忽略整个 ``scripts/oneoff/``——生产入口在干净 clone 上
    必然 ImportError。已把全A 数据集构建/回补管线 22 个模块迁入**受跟踪**的
    ``scripts/builders/``（该目录的 ``__init__`` 写明了搬迁理由）。

    本测试此后充当回归守卫：只要入库代码再出现 ``scripts.oneoff.*`` 的 import 即失败。
    用 AST 取**真实 import**（而非 grep 文本），避免 docstring 里的路径提及误伤。
    """
    offenders: list[str] = []
    for f in sorted(ROOT.rglob("*.py")):
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        if set(f.parts) & {".git", "__pycache__", ".venv", "venv", "node_modules"}:
            continue
        if rel.startswith(("reports/", "scripts/oneoff/")):
            continue
        for mod, _name in _all_imported_modules(f):
            if mod.startswith("scripts.oneoff"):
                offenders.append(f"{rel}: import {mod}")
    assert not offenders, (
        "入库代码依赖了 gitignored 的 scripts/oneoff/（请迁入 scripts/builders/）:\n"
        + "\n".join(offenders))


def _all_imported_modules(path: Path) -> set[tuple[str, str]]:
    """模块的全部 import（含非私有名），返回 (module, name) 集合。"""
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover
        return set()
    out: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                out.add((node.module, a.name))
        elif isinstance(node, ast.Import):
            for a in node.names:
                out.add((a.name, a.name.rsplit(".", 1)[-1]))
    return out


def test_frozen_recipe_stays_consistent_across_layers():
    """冻结配方（训练窗/折数/质量窗/最少训练日）在三处声明必须一致。

    `settings.model_portfolio`（生产控制面）与 `rolling_grid_alla`
    （实验真源）各写了一份数字、并在注释里互相声明"必须一致"；`alla_daily_rank`
    同样镜像了一份。这里把耦合**钉死**，避免改一处忘一处导致生产与实验悄悄分叉。
    """
    import importlib

    from config import Config

    rg = importlib.import_module("scripts.pipelines.rolling_grid_alla")
    adr = importlib.import_module("scripts.pipelines.alla_daily_rank")
    mp = Config.get().get("model_portfolio") or {}

    assert int(mp.get("train_window", 500)) == rg.DEFAULT_WINDOW, (
        "settings.model_portfolio.train_window 与 rolling_grid_alla.DEFAULT_WINDOW 分叉")
    assert int(mp.get("n_folds", 4)) == rg.N_FOLDS, (
        "settings.model_portfolio.n_folds 与 rolling_grid_alla.N_FOLDS 分叉")
    assert int(mp.get("quality_window", 500)) == rg.QUALITY_WINDOW, (
        "settings.model_portfolio.quality_window 与 rolling_grid_alla.QUALITY_WINDOW 分叉")
    assert adr.MIN_TRAIN == rg.MIN_TRAIN, (
        "alla_daily_rank.MIN_TRAIN 与 rolling_grid_alla.MIN_TRAIN 分叉（注释声明'同实验'）")
    assert adr.DEFAULT_WINDOW == rg.DEFAULT_WINDOW, (
        "alla_daily_rank.DEFAULT_WINDOW 与 rolling_grid_alla.DEFAULT_WINDOW 分叉")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
