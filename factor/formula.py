"""
统一因子公式解析器
==================

把「公式字符串 → 面板求值器」从候选生成里解耦出来，支持两种语法（业界前缀
表达式，可任意嵌套）：

1. **exhaustive 风格**（``factor/mining.generate_candidates`` 命名）：
   窗口作为显式参数 —— ``ts_delta(amount,20)`` / ``cs_rank(ts_mean(close,5))`` /
   ``div(close,ts_mean(close,20))`` / ``ts_corr(close,volume,20)``

2. **GP 风格**（``factor/genetic_mining`` 命名）：窗口编入算子名 ——
   ``ts_mean_5(close)`` / ``mul(ts_mean_5(close), cs_rank(ts_delta_20(volume)))``

统一解析的意义：

- **build_components 支持 GP 公式还原**：GP 因子（HallOfFame 个体）不在
  exhaustive 候选空间里，原先按名匹配重建会失败；现在统一走本解析器。
- **并行挖掘的 worker 重建**：``Candidate.build`` 是闭包、不可 pickle，无法直接
  送进进程池；公式字符串可序列化，worker 内用 ``formula_builder`` 重建求值器。
- **跨 run 复用**：不依赖 DEAP 的 pset / 模块级 prim_map（后者可能被覆盖）。

约定：
- 终端 = 特征名（须在 ``features`` 中，否则宽容视为特征名）。
- 整数常量只出现在窗口参数位置（``OpSpec.n_window`` 指定）。
- 窗口解析优先级：显式参数 > 算子名后缀（GP 风格）。
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

import pandas as pd

from factor.operators import OpSpec, op_registry

__all__ = ["formula_builder", "parse_formula"]

# GP 风格算子命名：ts_mean_5 / ts_corr_20（窗口编入算子名）
_GP_WINDOWED_RE = re.compile(r"^(ts_[a-z_]+?)_(\d+)$")


def _split_args(s: str) -> list[str]:
    """按顶层逗号分割参数（忽略括号内的逗号）。"""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def _match_spec(name: str, registry: dict[str, OpSpec]) -> tuple[OpSpec, tuple[int, ...]] | None:
    """把算子 token 解析为 (OpSpec, 窗口元组)；GP 风格窗口从名后缀提取。"""
    spec = registry.get(name)
    if spec is not None:
        return spec, ()
    m = _GP_WINDOWED_RE.match(name)
    if m:
        base = m.group(1)
        spec = registry.get(base)
        if spec is not None and spec.n_window >= 1:
            return spec, (int(m.group(2)),)
    return None


def parse_formula(
    formula: str,
    features: Sequence[str] | None = None,
    registry: dict[str, OpSpec] | None = None,
):
    """解析公式字符串为 AST 节点。

    Returns:
        节点元组：
        - ('feat', name) / ('const', value) / ('call', OpSpec, [children], win_from_name)
    """
    reg = registry if registry is not None else op_registry()
    feat_set = set(features) if features is not None else None

    def _parse(token: str):
        token = token.strip()
        if feat_set is not None and token in feat_set:
            return ("feat", token)
        if re.fullmatch(r"-?\d+", token):
            return ("const", int(token))
        if token.endswith(")") and "(" in token:
            name, _, inner = token[:-1].partition("(")
            name = name.strip()
            m = _match_spec(name, reg)
            if m is None:
                raise ValueError(f"未知算子 '{name}'（公式: {formula}）")
            spec, win_name = m
            children = [_parse(a) for a in _split_args(inner)]
            # 参数数校验：arity 个面板参数 + n_window 个窗口参数（GP 风格窗口在名里可少传）
            n_expect = spec.arity + spec.n_window
            if len(children) not in (spec.arity, n_expect):
                raise ValueError(
                    f"算子 '{name}' 参数数不符：期望 {spec.arity}(面板)+{spec.n_window}(窗口)，"
                    f"实际 {len(children)}（公式: {formula}）"
                )
            if len(children) == spec.arity and not win_name and spec.n_window > 0:
                raise ValueError(
                    f"算子 '{name}' 缺少窗口参数：需显式传入或使用 {name}_<w> 风格"
                    f"（公式: {formula}）"
                )
            return ("call", spec, children, win_name)
        # 宽容回退：未知裸 token 视为特征名（features 未传时的兜底）
        return ("feat", token)

    return _parse(formula)


def _node_key(node) -> str:
    """子树的缓存键（公式字符串形式，含窗口）。"""
    kind = node[0]
    if kind == "feat":
        return f"f:{node[1]}"
    if kind == "const":
        return f"c:{node[1]}"
    spec, children, win_name = node[1], node[2], node[3]
    arity = spec.arity
    child_keys = [_node_key(c) for c in children]
    win = ""
    if win_name:
        win = str(win_name[0])
    return f"{spec.name}[{win}]({','.join(child_keys[:arity])})"


def _node_depth(node) -> int:
    """节点子树高度（叶子=1）。"""
    kind = node[0]
    if kind in ("feat", "const"):
        return 1
    return 1 + max([_node_depth(c) for c in node[2]], default=0)


def _eval_node(node, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    kind = node[0]
    if kind == "feat":
        return panel[node[1]]
    if kind == "const":
        return node[1]
    spec, children, win_name = node[1], node[2], node[3]
    arity = spec.arity
    panel_args = [_eval_node(c, panel) for c in children[:arity]]
    win_args: list = []
    for c in children[arity:]:
        win_args.append(c[1] if c[0] == "const" else _eval_node(c, panel))
    if not win_args and win_name:          # GP 风格：窗口来自算子名后缀
        win_args = list(win_name)
    return spec.func(*panel_args, *win_args)


def _eval_node_cached(node, panel: dict[str, pd.DataFrame], cache: dict) -> pd.DataFrame:
    """带子树缓存求值（``cache``: 子公式字符串 -> 面板）。

    只缓存**浅层**子树（深度 ≤ 2，即「算子 + 全叶子参数」级别的组合）——这类
    子树在训练采样中共享率最高（如 ``ts_min_10(amount)``），深层组合共享少且
    面板内存大（~1800×520×8B ≈ 7.5MB/个），不缓存避免内存爆炸。
    """
    kind = node[0]
    if kind == "feat":
        return panel[node[1]]
    if kind == "const":
        return node[1]
    spec, children, win_name = node[1], node[2], node[3]
    arity = spec.arity
    cacheable = _node_depth(node) <= 2
    key = None
    if cacheable:
        key = _node_key(node)
        hit = cache.get(key)
        if hit is not None:
            return hit
    panel_args = [_eval_node_cached(c, panel, cache) for c in children[:arity]]
    win_args: list = []
    for c in children[arity:]:
        win_args.append(c[1] if c[0] == "const" else _eval_node_cached(c, panel, cache))
    if not win_args and win_name:
        win_args = list(win_name)
    out = spec.func(*panel_args, *win_args)
    if cacheable and key is not None:
        cache[key] = out
    return out


def formula_builder(
    formula: str,
    features: Sequence[str] | None = None,
    registry: dict[str, OpSpec] | None = None,
    node_cache: dict | None = None,
) -> Callable[[dict[str, pd.DataFrame]], pd.DataFrame]:
    """从公式字符串构造 ``build(panel) -> panel`` 闭包。

    Args:
        formula: 前缀表达式（exhaustive 或 GP 风格）。
        features: 特征名集合（用于识别终端；None 时未知 token 宽容视为特征）。
        registry: 算子注册表（默认 ``factor.operators.op_registry()``）。
        node_cache: 可选的子树级缓存 dict（跨调用共享，见 ``_eval_node_cached``）。
    Returns:
        build 函数：输入特征面板 dict，输出 date×code 因子面板。
    """
    node = parse_formula(formula, features=features, registry=registry)

    def build(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if node_cache is not None:
            out = _eval_node_cached(node, panel, node_cache)
        else:
            out = _eval_node(node, panel)
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame(out)

    build.__name__ = f"build<{formula}>"
    return build
