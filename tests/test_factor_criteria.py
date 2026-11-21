"""判据单一真源守卫（2026-11-19 判据收口后固化）。

三件事必须永远成立：

1. **同一对象**：``scripts`` 侧的兼容转出口与 ``research`` 侧的真源指向同一个
   函数对象。守卫 6（`test_layering`）的同型要求 —— 一旦有人把 shim 改成"复制
   一份再改两行"，这里的 ``is`` 断言立刻炸。
2. **定义唯一**：判据函数在全仓只能有白名单里那几个定义处。新增一处就报错，
   并直接点名文件 —— 这是防"又写一遍 IC 统计"的唯一机械手段。
3. **阈值唯一**：准入阈值只出自 ``factor_criteria``、晋升阈值只出自
   IS/OOS 分段常量只出自 ``research.factor_criteria.SEGMENT_DEFAULTS``
   （原 ``MATURITY_GATES`` 晋升系统已退役）、重复性阈值只出自
   ``research.dup_criteria``。

白名单只减不增；收口一个就从白名单删一个，数量只会单调下降。
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

#: 判据函数名 → 允许的定义处（相对仓库根，posix 路径）。
#: 🚨 **只减不增**。新增条目等于宣布"这个判据允许有两份实现"。
#:
#: ⚠️ 这里只收**语义唯一的名字**。``classify`` / ``verdict`` / ``should_check``
#: 这类通用名全仓有十几处同名定义（事件判定、PBO 判定、ETF 判定…），拿来当
#: 守卫会天天误报，所以准入判定改名 ``admission_verdict`` 后才纳管
#: （2026-11-19 实测：按 ``verdict`` 守卫一次抓到 4 个语义无关的同名函数）。
DEFINERS: dict[str, set[str]] = {
    "ic_calibers": {"research/factor_criteria.py"},
    "resid_rank": {"research/factor_criteria.py"},
    "panel_coverage": {"research/factor_criteria.py"},
    "cyclic_placebo": {"research/factor_criteria.py"},
    "effect_size": {"research/factor_criteria.py"},
    "admission_verdict": {"research/factor_criteria.py"},
    # 指标口径（2026-11-19 收口）：显著性 / IC 摘要全仓各一处实现。
    # ``ic_stats`` 8 处复制、``nw_t`` 12 处复制，语义统一后按名字钉死。
    "ic_stats": {"stats/ic.py"},
    "nw_t": {"stats/robust_stats.py"},
    "nw_mean_t": {"stats/robust_stats.py"},
    # 已消除的逐日 IC 序列别名：再出现即视为重写（空集 = 任何文件都不许定义）。
    "daily_ic": set(),
    "ic_of": set(),
}

#: 阈值常量名 → 允许的定义处。
GATE_DEFINERS: dict[str, set[str]] = {
    "ADMISSION_GATES": {"research/factor_criteria.py"},
    "MATURITY_GATES": set(),  # 晋升系统 2026-11-22 退役——再出现即视为复活
    "CORR_TRIGGER": {"research/dup_criteria.py"},
}

_SKIP_DIRS = {".venv", ".git", "__pycache__", "node_modules", "build", "dist"}


def _py_files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        out.append(p)
    return out


def _defined_names(path: Path) -> set[str]:
    """模块里**定义**的函数名与赋值名（不看 import / 调用）。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _find_definers(watched: dict[str, set[str]]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {k: set() for k in watched}
    for f in _py_files():
        rel = f.relative_to(ROOT).as_posix()
        names = _defined_names(f)
        for k in watched:
            if k in names:
                found[k].add(rel)
    return found


# ---------------------------------------------------------------------------
# 1. 同一对象（转出口不许变成第二实现）
# ---------------------------------------------------------------------------
def test_dup_criteria_shim_stays_dead():
    """``scripts.factors.dup_criteria`` 兼容转出口已于 2026-11-20 删除。

    消费方（library_dedup_apply / library_dedup_plan / 本目录判据锁）已全部改走
    :mod:`research.dup_criteria` 真源；本守卫防止 shim 复活成第二入口。
    """
    assert not (ROOT / "scripts/factors/dup_criteria.py").exists(), (
        "scripts/factors/dup_criteria.py 兼容转出口已删除，勿恢复 —— "
        "请直接 import research.dup_criteria")


# ---------------------------------------------------------------------------
# 2. 定义唯一（防"又写一遍"）
# ---------------------------------------------------------------------------
def test_criteria_have_single_definition_site():
    """判据函数全仓只有白名单里的定义处。"""
    found = _find_definers(DEFINERS)
    bad = {k: v - DEFINERS[k] for k, v in found.items() if v - DEFINERS[k]}
    assert not bad, (
        "判据出现第二份实现 —— 请改为调用 research/factor_criteria.py，"
        f"而不是另写一份：{bad}")


def test_gate_constants_have_single_definition_site():
    """阈值常量全仓只有白名单里的定义处。"""
    found = _find_definers(GATE_DEFINERS)
    bad = {k: v - GATE_DEFINERS[k] for k, v in found.items()
           if v - GATE_DEFINERS[k]}
    assert not bad, f"阈值出现第二份定义：{bad}"


# ---------------------------------------------------------------------------
# 3. 阈值聚合视图与真源一致
# ---------------------------------------------------------------------------
def test_all_thresholds_forward_real_sources():
    """``all_thresholds()`` 的各段必须等于各自的真源（转发不是复制）。

    晋升段 2026-11-22 退役：原 ``MATURITY_GATES`` 转发改指向
    :data:`fc.SEGMENT_DEFAULTS`（IS/OOS 分段常量的唯一住址）。
    """
    import research.dup_criteria as dc
    import research.factor_criteria as fc

    th = fc.all_thresholds()
    assert th["admission"] == fc.ADMISSION_GATES
    assert th["segmentation"] == fc.SEGMENT_DEFAULTS
    assert th["dup"]["corr_trigger"] == dc.CORR_TRIGGER
    assert th["dup"]["rel_std_degenerate"] == dc.REL_STD_DEGENERATE


def test_promotion_system_stays_retired():
    """晋升系统 2026-11-22 退役——阈值字典与评定方法不许复活。"""
    import research.factor_criteria as fc

    assert "min_sharpe" not in fc.ADMISSION_GATES
    assert "min_sharpe" not in fc.SEGMENT_DEFAULTS
    assert not (ROOT / "research/factor_library.py").read_text(
        encoding="utf-8").count("MATURITY_GATES"), "MATURITY_GATES 不许复活"


# ---------------------------------------------------------------------------
# 4. 分段统计转发（不另写）
# ---------------------------------------------------------------------------
def test_seg_stability_forwards_window_stats():
    """``seg_stability`` 与 ``factor_library.window_stats`` 逐位一致。"""
    import research.factor_criteria as fc
    from research.factor_library import window_stats

    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=600, freq="B")
    dret = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    ic = pd.Series(rng.normal(0.01, 0.05, len(idx)), index=idx)

    a = fc.seg_stability(dret, ic)
    b = window_stats(dret, ic, split_frac=0.7, recent_days=250, min_days=60)
    for k in b:
        assert a[k] == pytest.approx(b[k], rel=0, abs=0) or (
            np.isnan(a[k]) and np.isnan(b[k])), k


def test_seg_stability_uses_gate_defaults():
    """默认 split_frac / recent_days 取自晋升门槛，不另写 0.7 / 250。"""
    import research.factor_criteria as fc

    g = fc.promotion_gates()
    assert g["split_frac"] == 0.7 and g["recent_days"] == 250
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    out = fc.seg_stability(pd.Series(np.zeros(40), index=idx),
                           pd.Series(np.zeros(40), index=idx))
    # 样本不足 min_seg_days ⇒ 全 NaN，而不是给一个噪声数字
    assert all(np.isnan(v) for v in out.values())


# ---------------------------------------------------------------------------
# 5. 跨层引用方向（research 不得在模块级 import scripts）
# ---------------------------------------------------------------------------
def test_research_does_not_import_scripts_at_module_level():
    """``research.factor_criteria`` / ``dup_criteria`` 模块级不得依赖 scripts。

    函数级例外已存在（``factor_library`` 里读 ``dup_audit``），但新收口的
    判据模块必须是纯函数层，否则 `import research.factor_criteria` 会连带
    拖起整条 scripts 依赖链。
    """
    for rel in ("research/factor_criteria.py", "research/dup_criteria.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        roots = set()
        for node in tree.body:  # 只看模块级
            if isinstance(node, ast.Import):
                roots.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        assert "scripts" not in roots, f"{rel} 在模块级 import 了 scripts"
