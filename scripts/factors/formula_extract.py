"""从 factor/ 源码反解析因子的**可读公式**，供 registry ``formula`` 列与看板展示。

为什么要这个模块
----------------
registry 的 ``formula`` 列长期只有 12/891 行有值（全是手工填的 ``pap_*`` 研报线），
``parents`` / ``panel_path`` / ``note`` 全库为空。后果是：拿到一行 registry，
**回答不了"这个面板是什么"**——去重判「微小增量」时只能靠数值阈值，无法用公式
层证据；新因子入库也无从判断是否与库内某个因子同构。

``scripts/factors/backfill_metadata.py`` 填的是**定义引用**（``GTJA Alpha191 #158``），
只有出处没有内容。本模块补的是**真公式**：从函数源码 AST 反解析出可读表达式，
写成 ``<定义引用> = <表达式>``，一行同时给出来源与内容。

三个提取通道（按可靠性排序）
----------------------------
1. **注册表直查**：``ALPHA101`` / ``ALPHA158`` / ``ALPHA191`` / ``ALPHA360`` 四个
   dict 保存了 ``name → fn``。用 ``inspect.getsource`` + ``ast`` 反解析。
2. **闭包参数绑定**：rolling 族（``_register_rolling``）与 alpha360 族是工厂生成的
   lambda，源码里只有 ``lambda w: lambda d: ts_mean(d.close, w)``。把闭包里的
   ``w`` / ``i`` 提出来当常量代入，才能得到 ``ts_mean(close, 20)`` 而不是 ``w``。
3. **builder 直方图**：非 alpha 族（alt_/notice_/profit_ 等 157 个）没有统一注册表，
   只能取 builder 里的中文标签（``DESCRIPTIONS`` / ``EVENT_ROUTES``）；这类写
   ``<中文标签>``，不伪造表达式。

口径边界（写进 formula 时要能自证）
-----------------------------------
- 表达式是**反解析**结果，与源码语义等价但不是源码原文；源码改了下一次回填会跟着变。
- ``d.close`` → ``close``，``_sum`` → ``sum`` 等去掉实现前缀，保留算子名。
- 长表达式保留 ``let a = ...`` 别名链（对应源码的中间赋值），不做全量内联——
  实测 ``alpha191_159`` 全内联后 700+ 字符，人读不了。
- 无法提取时**返回 None 并计数**，不写占位符、不写"未维护"之类假值。

用法：
    python -m scripts.factors.formula_extract --check          # 只统计覆盖率
    python -m scripts.factors.formula_extract --show alpha191_159
    python -m scripts.factors.formula_extract --apply          # 写回 registry.formula
"""
from __future__ import annotations

import argparse
import ast
import inspect
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.atomic_io import write_csv_atomic  # noqa: E402

# ---------------------------------------------------------------------------
# AST → 表达式字符串
# ---------------------------------------------------------------------------
#: Python 运算符优先级（数值越大结合越紧），用于决定是否补括号
_PREC: dict[type, int] = {
    ast.IfExp: 1, ast.Or: 2, ast.And: 3, ast.Not: 4, ast.Compare: 5,
    ast.BitOr: 6, ast.BitXor: 7, ast.BitAnd: 8, ast.LShift: 9, ast.RShift: 9,
    ast.Add: 10, ast.Sub: 10, ast.Mult: 11, ast.Div: 11, ast.Mod: 11,
    ast.FloorDiv: 11, ast.Pow: 13,
}
_SYM: dict[type, str] = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**",
    ast.Mod: "%", ast.FloorDiv: "//", ast.BitOr: "|", ast.BitXor: "^",
    ast.BitAnd: "&", ast.LShift: "<<", ast.RShift: ">>",
}
_CMP: dict[type, str] = {
    ast.Lt: "<", ast.Gt: ">", ast.LtE: "<=", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=", ast.Is: "is", ast.IsNot: "is not",
}
#: 别名 RHS 超过这个 AST 节点数就强制保留 ``let``（否则内联后不可读）
_ALIAS_NODES = 12
#: 别名在 return / 后续别名里被引用 >= 这个次数就保留 ``let``（重复内联不如命名）
_ALIAS_USES = 2


def _nodes(node: ast.AST) -> int:
    return sum(1 for _ in ast.walk(node))


def _render(node: ast.AST, keep: set[str], prec_parent: int = 0,
            env: dict[str, str] | None = None) -> str:
    """把表达式 AST 渲染成可读字符串。

    Args:
        node: 表达式节点。
        keep: 需要保留为 ``let`` 别名的变量名集合（其余直接内联）。
        prec_parent: 父节点优先级，用于决定是否需要补括号。
        env: 自由变量 → 常量字面量的映射（工厂 lambda 的窗口 ``w``、滞后 ``i``）。
            渲染时优先取 env，避免公式里出现未绑定的 ``w``。
    """
    env = env or {}
    if isinstance(node, ast.Name):
        # 工厂 lambda 的形参在闭包里已绑定具体值，直接代入常量
        return env[node.id] if node.id in env else node.id
    if isinstance(node, ast.Attribute):
        # d.close → close（去掉 AlphaData 命名空间前缀，公式里 close 就是收盘价）
        if isinstance(node.value, ast.Name) and node.value.id == "d":
            return node.attr
        return f"{_render(node.value, keep, env=env)}.{node.attr}"
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, float):
            return repr(v)
        if isinstance(v, str):
            return f'"{v}"'
        if v is None:
            return "None"
        return str(v)
    if isinstance(node, ast.BinOp):
        p = _PREC[type(node.op)]
        sym = _SYM[type(node.op)]
        # 左结合：左操作数同级不加括号，右操作数同级要加（a-(b-c) 与 a-b-c 不同）
        left = _render(node.left, keep, p, env=env)
        right = _render(node.right, keep, p + 1, env=env)
        s = f"{left} {sym} {right}"
        return f"({s})" if p < prec_parent else s
    if isinstance(node, ast.UnaryOp):
        sym = {ast.USub: "-", ast.UAdd: "+", ast.Invert: "~"}.get(type(node.op), "!")
        s = f"{sym}{_render(node.operand, keep, 12, env=env)}"
        return f"({s})" if 12 < prec_parent else s
    if isinstance(node, ast.BoolOp):
        p = _PREC[type(node.op)]
        sym = "and" if isinstance(node.op, ast.And) else "or"
        s = f" {sym} ".join(_render(v, keep, p + 1, env=env) for v in node.values)
        return f"({s})" if p < prec_parent else s
    if isinstance(node, ast.Call):
        fn = _render(node.func, keep, env=env)
        args = [_render(a, keep, env=env) for a in node.args]
        args += [f"{k.arg}={_render(k.value, keep, env=env)}" for k in node.keywords if k.arg]
        return f"{fn}({', '.join(args)})"
    if isinstance(node, ast.Compare):
        left = _render(node.left, keep, 6, env=env)
        parts = [f"{left} {_CMP.get(type(o), '?')} {_render(c, keep, 6, env=env)}"
                 for o, c in zip(node.ops, node.comparators)]
        return " and ".join(parts)
    if isinstance(node, ast.IfExp):
        s = (f"{_render(node.body, keep, env=env)} if {_render(node.test, keep, env=env)} "
             f"else {_render(node.orelse, keep, env=env)}")
        return f"({s})" if 1 < prec_parent else s
    if isinstance(node, ast.Subscript):
        return f"{_render(node.value, keep, env=env)}[{_render(node.slice, keep)}]"
    if isinstance(node, ast.Tuple):
        return "(" + ", ".join(_render(e, keep, env=env) for e in node.elts) + ")"
    if isinstance(node, ast.Slice):
        lo = _render(node.lower, keep, env=env) if node.lower else ""
        hi = _render(node.upper, keep, env=env) if node.upper else ""
        return f"{lo}:{hi}"
    if isinstance(node, ast.List):
        return "[" + ", ".join(_render(e, keep, env=env) for e in node.elts) + "]"
    return f"<{type(node).__name__}>"


# ---------------------------------------------------------------------------
# 闭包参数绑定：工厂 lambda 里的 w / i 换成实际值
# ---------------------------------------------------------------------------
def _free_var_values(fn) -> tuple[dict[str, object], dict[str, object]]:
    """从函数对象取出「自由变量/默认参数 → 具体值」。

    Returns:
        (free, defaults)
        - ``free``: ``__closure__`` 里的自由变量（rolling 族的 ``w``）
        - ``defaults``: ``__defaults__`` 里的默认参数（alpha360 的 ``_fn`` / ``_i``）

    两个来源要分开：``lambda w: lambda d: ...`` 的 ``w`` 是**外层形参**，值在闭包里；
    ``lambda d, _fn=ref_fn, _i=i: ...`` 的值在默认参数里。混在一起会把形参名搞错。
    """
    free: dict[str, object] = {}
    defaults: dict[str, object] = {}
    closure = getattr(fn, "__closure__", None)
    code = getattr(fn, "__code__", None)
    if closure and code is not None:
        names = list(getattr(code, "co_freevars", ()) or ())
        for i, cell in enumerate(closure):
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            free[names[i] if i < len(names) else f"_{i}"] = v
    dflt = getattr(fn, "__defaults__", None)
    if dflt and code is not None:
        argnames = list(code.co_varnames[:code.co_argcount])
        for nm, v in zip(argnames[-len(dflt):], dflt):
            defaults[nm] = v
    return free, defaults


def _leaf_lambda(fn) -> ast.Lambda | ast.FunctionDef | None:
    """取工厂 lambda / 普通函数的最内层 body 节点。

    源码形如 ``_reg("ROC", lambda w: lambda d: safe_div(d.close.shift(w), d.close))``
    时要拿**内层** lambda（``lambda d: ...``），它的 body 才是真公式。
    """
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None
    lams = [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]
    if lams:
        inner = [c for c in lams if not isinstance(c.body, ast.Lambda)]
        return (inner or lams)[-1]
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return n
    return None


def _callable_formula(callable_obj, depth: int = 0) -> str | None:
    """把一个 callable 的 body 反解析成公式字符串（用于 alpha360 的 ``_close_ref``）。

    alpha360 的注册是 ``lambda d, _fn=ref_fn, _i=i: _fn(d, _i)``：外层只做转发，
    真正的公式在 ``_close_ref`` 里，且 ``i`` 要代进内层。所以要**递归展开**一层，
    并把 ``i`` 的值从调用点的实参继承下去。
    """
    if depth > 2:
        return None
    node = _leaf_lambda(callable_obj)
    if node is None:
        return None
    free, defaults = _free_var_values(callable_obj)
    env: dict[str, str] = {}
    for nm, v in {**free, **defaults}.items():
        if isinstance(v, (int, float)) and not callable(v):
            env[nm] = repr(v)
    if isinstance(node, ast.Lambda):
        body, assigns = node.body, []
    else:
        body = None
        assigns = []
        rets = [st for st in node.body
                if isinstance(st, ast.Return) and st.value is not None]
        if not rets:
            return None
        body = rets[-1].value
        assigns = [(st.targets[0].id, st.value) for st in node.body
                   if isinstance(st, ast.Assign) and len(st.targets) == 1
                   and isinstance(st.targets[0], ast.Name)]
    # 转发层：body 是 _fn(d, _i) 这种「单个 call，被调方还是个 callable」
    if isinstance(body, ast.Call):
        fnode = body.func
        target = None
        if isinstance(fnode, ast.Name):
            target = env.get(fnode.id) or defaults.get(fnode.id)
        if callable(target):
            # 把实参里的常量传给内层同名形参。
            # 外层 ``lambda d, _fn=ref_fn, _i=i: _fn(d, _i)`` 的第 2 个实参是
            # Name('_i') 而不是 Constant，所以还要从 env 里把 ``_i`` 的值解出来。
            inner_env: dict[str, str] = {}
            code = getattr(target, "__code__", None)
            if code is not None:
                names = list(code.co_varnames[:code.co_argcount])
                for nm, a in zip(names[1:], body.args[1:]):
                    lit = _const_of(a, env)
                    if lit is not None:
                        inner_env[nm] = lit
            return _expand_with_env(target, inner_env, depth + 1)
    return _render_lines(body, assigns, env)


def _const_of(node: ast.AST, env: dict[str, str]) -> str | None:
    """把实参节点解析成常量字面量：支持字面量与已绑定的 env 变量。"""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        return env.get(node.id)
    return None


def _expand_with_env(fn, extra_env: dict[str, str], depth: int) -> str | None:
    """在给定 env 下渲染 fn 的 body（alpha360 内层 _close_ref 用）。"""
    node = _leaf_lambda(fn)
    if node is None:
        return None
    free, defaults = _free_var_values(fn)
    env = dict(extra_env)
    for nm, v in {**free, **defaults}.items():
        if isinstance(v, (int, float)) and not callable(v):
            env.setdefault(nm, repr(v))
    if isinstance(node, ast.Lambda):
        return _render(node.body, set(), env=env)
    rets = [st for st in node.body
            if isinstance(st, ast.Return) and st.value is not None]
    if not rets:
        return None
    assigns = [(st.targets[0].id, st.value) for st in node.body
               if isinstance(st, ast.Assign) and len(st.targets) == 1
               and isinstance(st.targets[0], ast.Name)]
    return _render_lines(rets[-1].value, assigns, env)


def _render_lines(ret: ast.AST, assigns: list[tuple[str, ast.AST]],
                  env: dict[str, str]) -> str:
    """渲染「可选 let 别名链 + 主表达式」。"""
    used: Counter[str] = Counter()

    def _count(node: ast.AST) -> None:
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                used[n.id] += 1

    _count(ret)
    for _, rhs in assigns:
        _count(rhs)
    keep: set[str] = {nm for nm, rhs in assigns
                      if used[nm] >= _ALIAS_USES or _nodes(rhs) >= _ALIAS_NODES}
    lines = [f"let {nm} = {_render(rhs, keep, env=env)}" for nm, rhs in assigns]
    lines.append(_render(ret, keep, env=env))
    return " | ".join(lines) if len(lines) > 1 else lines[0]


# ---------------------------------------------------------------------------
# 主提取入口
# ---------------------------------------------------------------------------
def extract_alpha_formula(name: str, fn) -> str | None:
    """从因子函数反解析出可读公式；失败返回 ``None``。

    三条通道依次尝试：普通 ``def`` 函数 → 工厂 lambda（含转发层展开）→ 放弃。
    返回的字符串里 ``let x = ... |`` 的别名链对应源码的中间赋值。
    """
    node = _leaf_lambda(fn)
    if node is None:
        return None
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        free, defaults = _free_var_values(fn)
        env = {nm: repr(v) for nm, v in {**free, **defaults}.items()
               if isinstance(v, (int, float)) and not callable(v)}
        rets = [st for st in node.body
                if isinstance(st, ast.Return) and st.value is not None]
        if not rets:
            return None
        assigns = [(st.targets[0].id, st.value) for st in node.body
                   if isinstance(st, ast.Assign) and len(st.targets) == 1
                   and isinstance(st.targets[0], ast.Name)]
        return _render_lines(rets[-1].value, assigns, env)
    return _callable_formula(fn)


#: 实现名 → 公式记号。仓库里**两种命名都存在**，都要处理：
#: - 前缀式 ``_sum`` / ``_mean``（alpha191 里的加权求和）
#: - 后缀式 ``min_`` / ``max_``（避 Python 内置名的写法）
_OP_RENAME = {
    "_sum": "sum", "_mean": "mean", "_min": "min", "_max": "max",
    "_std": "std", "_rank": "rank", "_abs": "abs", "_log": "log",
    "min_": "min", "max_": "max", "sum_": "sum", "mean_": "mean",
}


def polish(expr: str) -> str:
    """把实现名换成公式记号：``_sum`` / ``min_`` → ``sum`` / ``min``，并折叠常量条件。"""
    for a, b in _OP_RENAME.items():
        expr = re.sub(rf"(?<![\w.]){re.escape(a)}\(", f"{b}(", expr)
    # ``x if 20 > 0 else y``：常量条件能静态判定的直接取一支。
    # alpha360 的 `if i > 0` 在 i 代入后全成常量，整式读起来像没化简的代码。
    expr = re.sub(r"(\S.*?) if (-?\d+) > 0 else (\S.*)", _fold_if, expr)
    return expr


def _fold_if(m: re.Match) -> str:
    then_br, lit, else_br = m.group(1), int(m.group(2)), m.group(3)
    return then_br if lit > 0 else else_br


def _registry_for(name: str):
    """按名字前缀找注册表，返回 (set_name, {name: fn}) 或 (None, {})。"""
    if name.startswith("alpha101"):
        import factor.alpha101 as m
        return "alpha101", m.ALPHA101
    if name.startswith("alpha191"):
        import factor.alpha191 as m
        return "alpha191", m.ALPHA191
    if name.startswith("alpha158"):
        import factor.alpha158 as m
        return "alpha158", m.ALPHA158
    if name.startswith("alpha360"):
        import factor.alpha158 as m
        return "alpha360", m.ALPHA360
    return None, {}


def formula_for(name: str) -> tuple[str, str]:
    """返回 ``(formula, source)``；source ∈ {``ast``, ``label``, ``none``}。

    ``label`` 表示该因子只能拿到中文标签（builder 无注册表），formula 里写标签
    原文而不伪造表达式——「有标签」和「有公式」是两件事，不能混。
    """
    set_name, reg = _registry_for(name)
    if set_name:
        fn = reg.get(name)
        if fn is None:
            return "", "none"
        expr = extract_alpha_formula(name, fn)
        if expr:
            from factor.alpha_base import SET_LABELS
            tail = name[len(set_name) + 1:] if name.startswith(set_name + "_") else name
            # alpha101/191 是编号式；alpha158/360 尾部本身是「算子+窗口」
            head = (f"{SET_LABELS[set_name]} #{tail}" if set_name in ("alpha101", "alpha191")
                    else f"{SET_LABELS[set_name]} · {tail}")
            return f"{head} = {polish(expr)}", "ast"
    lab = label_for(name)
    if lab:
        return lab, "label"
    return "", "none"


# ---------------------------------------------------------------------------
# 非 alpha 族：只取 builder 里的中文标签
# ---------------------------------------------------------------------------
def _builder_labels() -> dict[str, str]:
    """收集 builder 里的「因子名 → 中文定义」，静态解析，不 import builder。

    不 import builder 模块：它们的 ``main`` 会连带 import 数据层，代价大且可能
    依赖数据盘。用 AST 扫，认三种写法（都在仓库里实存）：

    1. ``{"name": "中文标签"}`` —— 值直接是字符串（altf / event / pledge）
    2. ``{"name": ("中文定义", "family")}`` —— 值是元组（style 的 ``FACTOR_DEFS``）
    3. 模块 docstring 里的 ``name : 中文说明`` 行（ideal/holder/fundamental 文件头）

    优先级 3 < 2 < 1：元组里的定义式（``资产负债率=总负债/总资产``）比文件头的
    一句话说明更精确，而 dict 里的短标签最贴近展示口径。
    """
    root = ROOT / "scripts" / "builders"
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for py in sorted(root.glob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        # ③ 模块 docstring 的 "name : 中文说明"（允许行首有 ⑦ 之类序号与表格空格）
        doc = ast.get_docstring(tree) or ""
        for line in doc.splitlines():
            m = re.match(r"^\s*(?:[^\w\s]{1,4}\s*)?([A-Za-z_][\w]{2,40})\s*[:：]\s*(\S.*)$",
                         line)
            if m and re.search(r"[\u4e00-\u9fff]", m.group(2)):
                out.setdefault(m.group(1), m.group(2).strip())
            # 同一行里 "a(xxx) / b(yyy)" 这种多因子并列写法的兜底
            for nm, desc in re.findall(r"([A-Za-z_]\w{2,40})\(([^()]{2,20})\)", line):
                if re.search(r"[\u4e00-\u9fff]", desc):
                    out.setdefault(nm, desc.strip())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for k, v in zip(node.keys, node.values):
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                    continue
                val = _dict_val_text(v)
                if is_label(val):
                    out[k.value] = val
    return out


def _dict_val_text(v: ast.AST) -> str:
    """从 dict 的值节点取可读文本。

    两种元组布局都要处理——**不能只取第一个字符串**：

    - ``("中文定义", "family")``    ← style 的 ``FACTOR_DEFS``，中文在前
    - ``("set_name", "中文标签")`` ← moneyflow 的 ``MONEYFLOW_LABELS``，中文在后

    判据是有几个字符串元素含中文：只有一个就取那个；两个都含就取第一个
    （定义式比标签更详细）。
    """
    if isinstance(v, ast.Constant) and isinstance(v.value, str):
        return v.value.strip()
    if isinstance(v, ast.Tuple):
        strs = [e.value.strip() for e in v.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if not strs:
            return ""
        cjk = [s for s in strs if re.search(r"[\u4e00-\u9fff]", s)]
        if len(cjk) == 1:
            return cjk[0]
        if cjk:
            return cjk[0]
        return strs[0] if is_label(strs[0]) else strs[-1]
    return ""


def is_label(text: str) -> bool:
    """判断一段文本像不像「因子定义」而不是正文。

    含中文 ⇒ 是。纯 ASCII 也接受，但**只在短**（≤12 字符）时——``"PEG"`` /
    ``"ROE"`` / ``"Altman Z"`` 是标签，而 ``"block_amt_20d/block_vol_20d 但缺折价率"``
    是正文。中文判据挡不住纯缩写，长度判据挡不住正文，两条并用才干净。
    """
    if not text:
        return False
    if re.search(r"[\u4e00-\u9fff]", text):
        return True
    return len(text) <= 12 and bool(re.fullmatch(r"[A-Za-z0-9 ./%_+\-]+", text))


_LABELS: dict[str, str] | None = None


def label_for(name: str) -> str:
    """非 alpha 因子的中文标签（无则空串）。"""
    global _LABELS
    if _LABELS is None:
        _LABELS = _builder_labels()
    return _LABELS.get(name, "")


def backfill_batch(ds_dir: Path, names: list[str]) -> int:
    """给**刚入库的一批**因子补 ``formula`` / ``formula_src``，返回写入行数。

    入库链（``scripts/builders/common.py::merge_outputs``）在 ``upsert_rows`` 之后
    调用本函数。只碰传入的 ``names``，不动全库——全库回填走 CLI ``--apply``。

    为什么不在 ``upsert_rows`` 里做：那是库层、要兼容多数据集与旧 schema；而公式
    只依赖 ``factor/`` 的源码，与哪个库无关，放在构建侧更简单也不会拖慢库写入。

    Args:
        ds_dir: 数据集根目录（其下有 ``registry.csv``）。
        names: 本批因子名。

    Returns:
        实际写入的行数（无法提取的不计）。
    """
    import pandas as pd

    reg_path = Path(ds_dir) / "registry.csv"
    if not reg_path.exists() or not names:
        return 0
    reg = pd.read_csv(reg_path)
    if "name" not in reg.columns:
        return 0
    want = {str(n) for n in names}
    # 🪤 新建列默认是 float64（全 NaN），往里写字符串会触发
    # ``FutureWarning: Setting an item of incompatible dtype``，未来版本直接报错。
    # 建列时就显式给 object dtype。
    if "formula" not in reg.columns:
        reg["formula"] = pd.Series([""] * len(reg), dtype=object)
    # 🚨 旧库没有 ``formula_src`` 列时，存量有值**无法区分**是手写还是上一版
    # 脚本填的。此时**保守**：当作手写不动（宁可公式旧，不可覆盖人写的）。
    # 该列一旦建立（CLI --apply 会建），后续就能按来源精确分流。
    has_src_col = "formula_src" in reg.columns
    if not has_src_col:
        reg["formula_src"] = pd.Series([""] * len(reg), dtype=object)
    # 已存在的列若因整列空串被 pandas 读成 float64，也得转回 object，
    # 否则 ``reg.at[i, ...] = "ast"`` 触发 FutureWarning（将来是硬报错）。
    for _c in ("formula", "formula_src"):
        if reg[_c].dtype != object:
            reg[_c] = reg[_c].astype(object)

    n = 0
    for i in reg.index:
        nm = str(reg.at[i, "name"])
        if nm not in want:
            continue
        # 🪤 ``str(nan or "")`` 是 ``"nan"``——nan 是**真值**。直接判 ``cur_fml``
        # 非空会把「空 formula」误当成手写，整批静默跳过（实测踩到：写入 0 行
        # 但日志一切正常）。必须先 isna 再 strip。
        raw_src = reg.at[i, "formula_src"]
        raw_fml = reg.at[i, "formula"]
        cur_src = "" if pd.isna(raw_src) else str(raw_src).strip()
        cur_fml = "" if pd.isna(raw_fml) else str(raw_fml).strip()
        if cur_src == "manual":
            continue
        # 无来源列 且 存量有值 ⇒ 保守当手写；有来源列时仅 src 为空才算手动
        if not has_src_col and cur_fml:
            continue
        f, src = formula_for(nm)
        if not f:
            continue
        reg.at[i, "formula"] = f
        reg.at[i, "formula_src"] = src
        n += 1
    if n:
        write_csv_atomic(reg, reg_path)
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="因子的可读公式提取（只读或回填 registry）")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--show", default="", help="打印指定因子（逗号分隔）的公式")
    ap.add_argument("--check", action="store_true", help="只统计覆盖率，不写盘")
    ap.add_argument("--apply", action="store_true", help="写回 registry.formula")
    args = ap.parse_args()

    if args.show:
        for nm in [s.strip() for s in args.show.split(",") if s.strip()]:
            f, src = formula_for(nm)
            print(f"### {nm}  [{src}]")
            print(f"  {f or '<无法提取>'}")
        return

    import pandas as pd

    from research.factor_library import FactorLibrary

    lib = FactorLibrary(dataset=args.dataset)
    base = Path(lib.root)
    if not (base / "registry.csv").exists():
        base = base / args.dataset
    reg_path = base / "registry.csv"
    reg = pd.read_csv(reg_path)

    rows = []
    for nm in reg["name"]:
        f, src = formula_for(str(nm))
        rows.append({"name": nm, "formula_new": f, "src": src})
    got = pd.DataFrame(rows)
    stat = got["src"].value_counts().to_dict()
    print(f"registry {len(reg)} 行 | ast {stat.get('ast', 0)} | "
          f"label {stat.get('label', 0)} | none {stat.get('none', 0)}")
    if "family" in reg.columns:
        print("\n按族（ast 覆盖）:")
        m = reg[["name", "family"]].merge(got, on="name")
        print(m.groupby("family")["src"].value_counts().unstack(fill_value=0).to_string())

    if args.check or not args.apply:
        print("\n（--check / 默认：不写盘；加 --apply 回填）")
        return

    import time
    old = (reg["formula"].fillna("").astype(str).str.strip()
           if "formula" in reg.columns else pd.Series([""] * len(reg)))
    # 🚨 「已有 formula 就跳过」会把**上一轮自动填的**也当手写保护起来，源码改了
    # 公式永远不更新（2026-11-13 实际踩到：第二次回填 0 行）。所以按来源分流：
    #   - ``formula_src == manual``：人写的，任何情况都不覆盖
    #   - ``formula_src ∈ {ast, label}``：脚本生成的，每跑一次都跟着源码刷新
    #   - 旧行（无 formula_src 但有 formula）：视为 manual —— 那是回填前的存量
    prev_src = (reg["formula_src"].fillna("").astype(str)
                if "formula_src" in reg.columns else pd.Series([""] * len(reg)))
    is_manual = (prev_src == "manual") | ((prev_src == "") & (old != ""))
    # 反解析失败但存量有值 ⇒ 存量就是手写（pap_* 那批），标 manual 留档
    new_src = got["src"].where(got["src"] != "none", "")
    new_src = new_src.where(~(is_manual & (new_src == "")), "manual")
    write = (~is_manual) & (got["formula_new"].fillna("") != "")
    reg["formula"] = got["formula_new"].fillna("").where(write, old)
    reg["formula_src"] = new_src.where(write | (new_src != ""), prev_src)
    reg.loc[is_manual & (reg["formula_src"].fillna("") == ""), "formula_src"] = "manual"

    changed = int((reg["formula"].fillna("") != old.fillna("")).sum())
    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = reg_path.with_name(f"registry.csv.bak-formula-{ts}")
    reg.to_csv(bak, index=False, encoding="utf-8")
    write_csv_atomic(reg, reg_path)
    print(f"\n写入 {int(write.sum())} 行（其中内容变化 {changed} 行）；"
          f"保护手写 {int(is_manual.sum())} 行")
    print(f"备份 {bak.name}")
    chk = pd.read_csv(reg_path)
    print(f"复核: formula 非空 {int(chk['formula'].notna().sum())}/{len(chk)} | "
          f"来源分布 {chk['formula_src'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()