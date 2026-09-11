"""optimize 包惰性导入契约守卫（2026-09-11 P2）。

背景：``optimize/__init__`` 原先静态 import 全部子模块，连带 cvxpy（7.3s 冷启）、
scipy.stats（经 stats.ic，4.0s）、openpyxl（经 research.xlsx_report，2.4s），
``import optimize`` 实测 10.2s。改为 PEP 562 模块级 ``__getattr__`` 后，导入本身
不再拉起这些重依赖。

本测试钉住三件事：
1. **契约不变**：``__all__`` 里每个名字都能取到，且与子模块属性是**同一对象**；
2. **子模块属性可访问**：``optimize.solver`` 等不因惰性化而 AttributeError；
3. **惰性真的生效**：``import optimize`` 不把 cvxpy / openpyxl 拉进 sys.modules。

⚠️ 第 3 条要在**干净的子进程**里验证——同进程里别的测试可能已经 import 过 cvxpy。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_optimize_all_names_resolve_to_submodule_objects():
    """``__all__`` 每个名字可访问，且 == 所属子模块的同名属性（同一对象）。"""
    import importlib

    import optimize

    assert set(optimize.__all__) == set(optimize._LAZY_API), (
        "__all__ 与 _LAZY_API 必须一一对应（否则存在未惰性化的公开名）")

    for name in optimize.__all__:
        val = getattr(optimize, name)
        mod = importlib.import_module(optimize._LAZY_API[name])
        assert val is getattr(mod, name), (
            f"optimize.{name} 与 {optimize._LAZY_API[name]}.{name} 不是同一对象")
        # 缓存后再取应命中 globals，而非重复 __getattr__
        assert name in vars(optimize), f"optimize.{name} 未缓存到模块 globals"


def test_optimize_submodule_attributes_accessible():
    """``optimize.solver`` / ``optimize.risk`` 等属性访问必须可用（惰性不破坏契约）。"""
    import optimize

    for sub in optimize._LAZY_SUBMODULES:
        mod = getattr(optimize, sub)
        assert mod.__name__ == f"optimize.{sub}"


def test_optimize_unknown_attribute_raises():
    import optimize

    with pytest.raises(AttributeError):
        optimize.definitely_not_a_real_name  # noqa: B018


def test_import_optimize_does_not_pull_heavy_deps():
    """干净子进程里 ``import optimize`` 不得拉起 cvxpy / openpyxl / scipy.stats。"""
    code = (
        "import sys, optimize;"
        "heavy=[m for m in ('cvxpy','openpyxl','scipy.stats') if m in sys.modules];"
        "print('HEAVY', heavy)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("HEAVY")]
    assert line, out.stdout[-2000:]
    assert line[0].strip() == "HEAVY []", (
        f"import optimize 拉起了重依赖: {line[0]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
