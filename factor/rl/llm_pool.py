"""
AlphaPool 的大模型因子池（华泰 AI97《大模型+强化学习因子挖掘》后段）
====================================================================

研报给大模型派了**两项功能**（原文「大模型的角色」节）：

1. **构造基础池** —— 给 RL 一个"热身"起点。空池起步时 ``AlphaPool.best_obj``
   恒为 0，而研报的奖励档位里「位于失败缓存 / 效果无法入池」两档给的就是
   ``best_obj`` —— 于是 RL 在前期从"没入池"这件事上**拿不到任何梯度信息**
   （奖励恒等于 0，与"空值"档无法区分）。有初始池后 ``best_obj > 0``，
   这个信号才立起来。
2. **定期去弱留强** —— ``llm_every_n_steps=3000`` 步调用一次大模型，
   ``drop_rl_n=5`` 个 RL 生成的较差因子被剔除，换入新因子，避免陷入局部最优
   （研报原话："使得因子池成为'活水'池"）。

研报同时点明了 RL 直接生成公式的三个毛病（这也是本文档三个校验器的来源）：
**构造简单**（如 ``$volume``）、**不合逻辑**（如 ``Sub($volume, $close)``，
量纲不一致）、**符号多余**（如 ``Abs(Abs(Sub($close, $open)))``）。

本模块的构成：

- 研报语法解析：``Mean($close, 20)`` / ``Corr($close, $volume, 20)`` 形态 → 内部 AST；
  再转成项目 ``factor.formula`` 的前缀语法（``ts_mean_20(close)``）。
- 三个语义校验：``trivial_terminal`` / ``dimension_mismatch`` / ``redundant_op``
  （外加 ``degenerate``：``Sub(a, a)`` 这类恒等式）。**量纲用"价格指数 × 成交量指数"
  的二维指数向量做精确判定**，不是启发式正则。
- 第四个校验 ``scalar_operand``（2026-09-16 补，均匀采样臂暴露）：面板算子的实参位
  出现常量子树（如 ``Mean(5, 20)``）时运行时必炸，而量纲校验对常量返回 ``None``
  使其逃逸。**修复前实测 73% 的均匀采样公式因此求值失败且全部通过校验。**
- 提案器：``TemplateProposer``（离线确定性，用于 mock / 无网环境）与
  ``OpenAICompatibleProposer``（OpenAI 兼容 HTTP 接口，研报用的是 deepseek）。
- 注入原语：``seed_pool``（构造基础池）/ ``refresh_pool``（去弱留强）。

**复现边界（必须披露）**：

- 研报**没有给出大模型提示词**，也未说明它如何把自然语言回复变成合法公式。
  本模块的提示词模板与解析器是**按研报口径自拟**的，不是原文复刻。
- ``OpenAICompatibleProposer`` 需要真实 API key，**本仓库环境无 key、该路径未经联网
  验证**；单元测试只覆盖「请求构造 + 缺 key 报错」，不覆盖真实往返。
- 研报只写了 ``drop_rl_n`` 与 ``llm_every_n_steps`` 两个 LLM 侧超参，**没有写
  "每次注入多少个因子"**。本模块把它显式化为 ``n_new``（默认 = 丢弃数，保证池容量
  守恒），并在调用处打印，避免把它当成研报参数引用。
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol, Sequence

import numpy as np

from factor.rl.alphapool_env import (
    CONSTANTS,
    FIELDS,
    MAX_EXPR_LENGTH,
    REPORT_OPERATORS,
    WINDOWS,
    AlphaPool,
    op_arity,
    op_is_rolling,
)

log = logging.getLogger(__name__)

__all__ = [
    "FormulaCheck",
    "FormulaSyntaxError",
    "OpenAICompatibleProposer",
    "PoolUpdateReport",
    "RNode",
    "TemplateProposer",
    "UniformRandomProposer",
    "build_prompt",
    "canonical",
    "check_report_formula",
    "check_many",
    "extract_formulas",
    "make_proposer",
    "parse_report_formula",
    "refresh_pool",
    "seed_pool",
    "to_report",
    "to_project",
]


# ===========================================================================
# 1) 研报语法：tokenizer + 递归下降解析
# ===========================================================================

class FormulaSyntaxError(ValueError):
    """研报语法解析失败（LLM 输出不合法时抛出）。"""


_TOKEN_RE = re.compile(
    r"\s*(?:(-?\d+(?:\.\d+)?)|(\$[A-Za-z_]\w*)|([A-Za-z_]\w*)|([(),]))"
)


@dataclass(frozen=True)
class RNode:
    """公式 AST 节点。

    ``kind`` 取：

    - ``"field"``：``value`` 为小写字段名（研报写法 ``$CLOSE`` → ``"close"``）；
    - ``"const"``：``value`` 为 ``int`` / ``float``；
    - ``"call"``：``value`` 为 ``(研报算子名, 窗口 int 或 None)``，``children`` 为参数。
    """

    kind: str
    value: Any
    children: tuple["RNode", ...] = ()


def _tokenize(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if m is None or m.end() == pos:
            raise FormulaSyntaxError(f"无法识别的字符：{text[pos:pos + 12]!r}")
        pos = m.end()
        for kind, val in zip(("num", "field", "name", "punc"), m.groups()):
            if val is not None:
                out.append((kind, val))
                break
    return out


def parse_report_formula(text: str) -> RNode:
    """解析研报语法公式为 :class:`RNode`。解析失败抛 :class:`FormulaSyntaxError`。"""
    toks = _tokenize(text)
    if not toks:
        raise FormulaSyntaxError("空公式")
    node, pos = _parse_expr(text, toks, 0)
    if pos != len(toks):
        raise FormulaSyntaxError(f"存在多余内容：从第 {pos} 个 token 起 {toks[pos:]}")
    return node


def _parse_expr(text: str, toks: list[tuple[str, str]], pos: int):
    if pos >= len(toks):
        raise FormulaSyntaxError(f"表达式意外结束：{text!r}")
    kind, val = toks[pos]
    if kind == "num":
        f = float(val)
        return RNode("const", int(f) if f.is_integer() else f), pos + 1
    if kind == "field":
        name = val[1:].lower()
        if name not in FIELDS:
            raise FormulaSyntaxError(f"未知字段 {val!r}，可选 {list(FIELDS)}")
        return RNode("field", name), pos + 1
    if kind == "name":
        if val not in REPORT_OPERATORS:
            raise FormulaSyntaxError(
                f"未知算子 {val!r}，研报 21 算子为 {list(REPORT_OPERATORS)}")
        if pos + 1 >= len(toks) or toks[pos + 1][1] != "(":
            raise FormulaSyntaxError(f"算子 {val!r} 后应为 '('（{text!r}）")
        args, pos = _parse_args(text, toks, pos + 2)
        win: Optional[int] = None
        if op_is_rolling(val):
            if not args:
                raise FormulaSyntaxError(f"滚动算子 {val!r} 缺窗口参数")
            last = args[-1]
            if last.kind != "const" or not float(last.value).is_integer():
                raise FormulaSyntaxError(
                    f"滚动算子 {val!r} 的末位参数必须是整数窗口（{text!r}）")
            win = int(last.value)
            args = args[:-1]
        want = op_arity(val)
        if len(args) != want:
            raise FormulaSyntaxError(
                f"算子 {val!r} 需要 {want} 个面板参数，实际 {len(args)}（{text!r}）")
        return RNode("call", (val, win), tuple(args)), pos
    raise FormulaSyntaxError(f"意外的 token {val!r}（{text!r}）")


def _parse_args(text: str, toks: list[tuple[str, str]], pos: int):
    args: list[RNode] = []
    if pos < len(toks) and toks[pos][1] == ")":
        return args, pos + 1
    while True:
        node, pos = _parse_expr(text, toks, pos)
        args.append(node)
        if pos >= len(toks):
            raise FormulaSyntaxError(f"缺少 ')'（{text!r}）")
        tok = toks[pos][1]
        if tok == ",":
            pos += 1
            continue
        if tok == ")":
            return args, pos + 1
        raise FormulaSyntaxError(f"参数分隔符应为 ',' 或 ')'（{text!r}）")


# ===========================================================================
# 2) 语法转换
# ===========================================================================

def _fmt_const(c: float) -> str:
    """常数 → 项目语法字面量（整数去掉小数点；小数保留，2026-09-16 已支持）。"""
    f = float(c)
    return str(int(f)) if f.is_integer() else repr(round(f, 6))


def to_project(node: RNode) -> str:
    """研报 AST → 项目前缀语法（``formula_builder`` 可直接求值）。"""
    if node.kind == "field":
        return str(node.value)
    if node.kind == "const":
        return _fmt_const(float(node.value))
    name, win = node.value
    reg_name = REPORT_OPERATORS[name]
    inner = ",".join(to_project(c) for c in node.children)
    head = f"{reg_name}_{win}" if win is not None else reg_name
    return f"{head}({inner})"


def to_report(node: RNode) -> str:
    """项目 AST/研报 AST → 研报展示语法（``Mean($close, 20)``）。"""
    if node.kind == "field":
        return f"${node.value}"
    if node.kind == "const":
        return _fmt_const(float(node.value))
    name, win = node.value
    args = [to_report(c) for c in node.children]
    if win is not None:
        args.append(str(win))
    return f"{name}({', '.join(args)})"


def canonical(text: str) -> Optional[str]:
    """公式规范化（去空格/大小写/写法的差异）。解析失败返回 ``None``。"""
    try:
        return to_project(parse_report_formula(text))
    except FormulaSyntaxError:
        return None


# ===========================================================================
# 3) 三个语义校验（对应研报点名的三个毛病）
# ===========================================================================

#: 字段量纲 = (价格指数, 成交量指数)。加减法要求两侧指数**逐位相等**，
#: 乘除按指数加减 —— 这样 `Sub($volume, $close)` = (0,1)−(1,0) 立刻暴露，
#: 而 `Mul($close, $volume)` = (1,1)（成交额）是合法的，不需要写特例。
_FIELD_DIM: dict[str, tuple[int, int]] = {
    "open": (1, 0), "high": (1, 0), "low": (1, 0),
    "close": (1, 0), "vwap": (1, 0), "volume": (0, 1),
}
_DIM_SCALE = (0, 0)          # 无量纲（比值、相关系数、比较结果）
_DIM_MIXED = "mixed"         # 量纲冲突哨兵

#: 保持量纲的算子（对自身量纲无变换或只做线性缩放）
_DIM_KEEP = {
    "Abs", "Ref", "Mean", "Sum", "Std", "Var", "Max", "Min",
    "Med", "Mad", "Delta", "WMA", "EMA",
}
#: 幂等算子（嵌套自己 = 符号多余）
_IDEMPOTENT = {"Abs", "Log"}


def _is_panel_valued(node: RNode) -> bool:
    """该子树求值后是否是**面板**（date×code），而非标量。

    2026-09-16 修复（均匀采样臂暴露，**校准器此前完全漏检**）：

    项目算子体系里，``ts_*`` 系算子的实参必须是 DataFrame —— ``ts_median_5(5)``
    会在运行时抛 ``AttributeError: 'int' object has no attribute 'rolling'``；
    ``Cov(Max($volume,5), 5, 5)`` 抛 ``ValueError: other must be a DataFrame or Series``。
    而 :func:`_dim` 把常数视作 ``None``（"与任何量纲兼容"），于是这类子树**量纲上
    永远是合法的**，``dimension_mismatch`` 一条都拦不住。

    实测（``scripts/oneoff/_probe_uniform_true_yield.py``，200 条均匀采样）：
    **73% 的公式因常量子树在算子实参位而求值失败，校验器全部放行** ——
    与 2026-09-16 小数常数字面量事故同型：**这不是报错，是静默降级**
    （在 RL 里表现为该动作恒得 0 奖励，把动作空间的一个大子集变成死区）。

    判定规则（常量本身是合法终端，只要它出现在**能接受标量的位置**）：
    - ``field``                      → True
    - ``const``                      → False（标量）
    - ``call`` 且 name 在 _PANEL_ARG_REQUIRED → 所有 children 都必须是面板
      （这些算子内部直接调 ``.rolling``/``.shift``/``.ewm`` 或把实参当 DataFrame 两两运算）
    - 其他 ``call``（Add/Sub/Mul/Div/Greater/Less/Log/Abs）→ 任一 children 是面板即可
      （``Mul($close, 2)`` 这类"面板 × 常数"是合法缩放）

    注意 ``_PANEL_ARG_REQUIRED`` 里的二元算子（Corr/Cov）要求**两侧都是面板**，
    因为它们走 DataFrame 对齐运算路径。
    """
    if node.kind == "field":
        return True
    if node.kind == "const":
        return False
    name = node.value[0]
    if name in _PANEL_ARG_REQUIRED:
        return bool(node.children) and all(_is_panel_valued(c) for c in node.children)
    # 宽松类算子（Add/Sub/Mul/Div/Greater/Less/Log/Abs）：**至少一个**实参是面板即可。
    # 两侧全常量（如 Greater(-1, 10)）会退化成裸 Python 标量/bool，
    # 后续 .astype 之类调用必炸 —— 实测修复后仅剩的这一处泄漏。
    return any(_is_panel_valued(c) for c in node.children)


#: 实参**必须是面板**的算子：内部直接用 pandas 滚动/对齐接口，喂标量必炸。
#: 单目项要求唯一实参是面板；多元项（Corr/Cov）要求每个实参都是面板。
_PANEL_ARG_REQUIRED = {
    "Ref", "Mean", "Sum", "Std", "Var", "Max", "Min", "Med", "Mad",
    "Delta", "WMA", "EMA", "Corr", "Cov",
}


def _terminal_token_ids() -> frozenset[int]:
    """终端类 token（field / const）的 id 集合，供采样器做 field 偏向。"""
    from factor.rl.alphapool_env import CONST_IDS, FIELD_IDS

    return frozenset(FIELD_IDS) | frozenset(CONST_IDS)


_TERMINAL_IDS: frozenset[int] = _terminal_token_ids()


def _dim(node: RNode):
    """推断量纲（``(价格指数, 成交量指数)``）。

    常数返回 ``None`` = **与任何量纲兼容**（``Div($close, 100)`` 这类合法缩放不参与
    判定）；``_DIM_MIXED`` 为冲突哨兵，沿树向上传播。
    """
    if node.kind == "field":
        return _FIELD_DIM[str(node.value)]
    if node.kind == "const":
        return None
    name = node.value[0]
    dims = [_dim(c) for c in node.children]
    if name in _DIM_KEEP:
        return dims[0]
    if name == "Log":
        return None
    if name in ("Greater", "Less", "Corr"):
        return _DIM_SCALE
    if name in ("Add", "Sub"):
        return _merge_same(dims[0], dims[1])
    if name in ("Mul", "Cov"):
        return _multiply(dims[0], dims[1])
    if name == "Div":
        return _divide(dims[0], dims[1])
    return None


def _merge_same(a, b):
    if a is _DIM_MIXED or b is _DIM_MIXED:
        return _DIM_MIXED
    if a is None:
        return b
    if b is None:
        return a
    return a if a == b else _DIM_MIXED


def _multiply(a, b):
    if a is _DIM_MIXED or b is _DIM_MIXED:
        return _DIM_MIXED
    if a is None:
        return b
    if b is None:
        return a
    return (a[0] + b[0], a[1] + b[1])


def _divide(a, b):
    if a is _DIM_MIXED or b is _DIM_MIXED:
        return _DIM_MIXED
    if b is None:
        return a
    if a is None:
        return None
    return (a[0] - b[0], a[1] - b[1])


def _walk(node: RNode):
    yield node
    for c in node.children:
        yield from _walk(c)


def _n_tokens(node: RNode) -> int:
    """RPN token 数（不含 BEG/SEP），与 ``FactorMDP`` 的长度上限同口径。"""
    if node.kind in ("field", "const"):
        return 1
    n = 1 + (1 if node.value[1] is not None else 0)
    for c in node.children:
        n += _n_tokens(c)
    return n


def _same(a: RNode, b: RNode) -> bool:
    return to_project(a) == to_project(b)


def _semantic_issues(node: RNode) -> list[str]:
    issues: list[str] = []

    # ① 构造简单：裸字段 / 裸常数 / 单目包裸字段（研报点名 $volume）
    if node.kind in ("field", "const"):
        issues.append("trivial_terminal")
    elif (len(node.children) == 1 and node.children[0].kind == "field"
          and node.value[0] in ("Abs", "Log")):
        issues.append("trivial_terminal")

    # ② 量纲冲突：逐节点检查「同类运算两侧必须同量纲」
    #    注意 Corr / Cov 不在此列 —— Corr($close,$volume,20) 是经典量价因子，
    #    两侧量纲本来就不一样；Cov 的结果本身带量纲，也不构成冲突。
    for sub in _walk(node):
        if sub.kind == "call" and sub.value[0] in ("Add", "Sub", "Greater", "Less"):
            a, b = _dim(sub.children[0]), _dim(sub.children[1])
            if a is _DIM_MIXED or b is _DIM_MIXED:
                continue                      # 内层已标记，不重复
            if a is not None and b is not None and a != b:
                issues.append("dimension_mismatch")
    if _dim(node) is _DIM_MIXED:
        issues.append("dimension_mismatch")

    # ③ 符号多余：幂等算子嵌套自己（研报点名 Abs(Abs(...))）
    for sub in _walk(node):
        if sub.kind == "call" and sub.children:
            name = sub.value[0]
            if name in _IDEMPOTENT:
                for c in sub.children:
                    if c.kind == "call" and c.value[0] == name:
                        issues.append("redundant_op")
                        break

    # ④ 恒等式：Sub(a, a) / Div(a, a) / Corr(a, a, w) 恒为 0 或 1，无信息量
    for sub in _walk(node):
        if sub.kind != "call" or len(sub.children) < 2:
            continue
        if sub.value[0] in ("Sub", "Div", "Corr") and _same(
                sub.children[0], sub.children[1]):
            issues.append("degenerate")

    # ⑤ 标量实参：面板算子（或 DataFrame 对齐算子）的实参位出现「常量子树」
    #    运行时抛 AttributeError/ValueError；_dim 对常量返回 None 使其逃过 ②。
    #    2026-09-16 新增（均匀采样臂暴露：73% 公式因此求值失败）。
    for sub in _walk(node):
        if sub.kind != "call" or not sub.children:
            continue
        if sub.value[0] not in _PANEL_ARG_REQUIRED:
            continue
        if not all(_is_panel_valued(c) for c in sub.children):
            issues.append("scalar_operand")

    # ⑥ 根节点必须是面板：`Greater(-1, 10)` / `Div(-1, 10)` 这类**整棵树无字段**
    #    的公式会返回裸标量/bool，进不了因子面板。⑤ 只管 _PANEL_ARG_REQUIRED 的
    #    实参位，拦不住这种「根就是纯常量运算」的形态，故单独兜底。
    #    等价判据：整棵树至少要有一个 field 叶子。
    if not _is_panel_valued(node):
        issues.append("scalar_formula")

    return sorted(set(issues))


@dataclass(frozen=True)
class FormulaCheck:
    """单条公式的校验结果。"""

    text: str
    ok: bool
    project: Optional[str] = None
    n_tokens: int = 0
    issues: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        return "ok" if self.ok else (self.issues[0] if self.issues else "unknown")


def check_report_formula(
    text: str,
    *,
    features: Sequence[str] | None = None,
    windows: Sequence[int] = WINDOWS,
    max_body_tokens: int = MAX_EXPR_LENGTH - 2,
    registry: dict | None = None,
) -> FormulaCheck:
    """校验一条研报语法公式，返回 :class:`FormulaCheck`（不抛异常）。

    依次检查：语法 → 窗口白名单 → token 长度 → 三个语义毛病 →
    转成项目语法后能否被 ``factor.formula.parse_formula`` 解析。

    ``max_body_tokens`` 默认 ``MAX_EXPR_LENGTH - 2``（15 含 BEG/SEP，正文 13），
    与 :func:`factor.rl.alphapool_env.legal_mask_for_tokens` 的长度口径一致。
    """
    try:
        node = parse_report_formula(text)
    except FormulaSyntaxError as exc:
        return FormulaCheck(text, False, None, 0, (f"syntax: {exc}",))
    except Exception as exc:                                  # 防御：未知解析异常
        return FormulaCheck(text, False, None, 0, (f"syntax: {exc}",))

    issues: list[str] = []
    for sub in _walk(node):
        if sub.kind == "call":
            name, win = sub.value
            if win is not None and int(win) not in set(windows):
                issues.append("window_not_allowed")

    n_tok = _n_tokens(node)
    if n_tok > max_body_tokens:
        issues.append("too_long")

    issues.extend(_semantic_issues(node))

    project: Optional[str] = None
    if not issues:
        from factor.formula import parse_formula
        from factor.operators import op_registry

        project = to_project(node)
        try:
            parse_formula(project, features=list(features or FIELDS),
                          registry=registry if registry is not None else op_registry())
        except Exception as exc:
            issues.append(f"unparsable_project: {exc}")
            project = None

    return FormulaCheck(text, not issues, project, n_tok, tuple(issues))


def check_many(formulas: Iterable[str], **kw) -> list[FormulaCheck]:
    """批量校验（保持输入顺序）。"""
    return [check_report_formula(f, **kw) for f in formulas]


# ===========================================================================
# 4) 从大模型裸文本里抽公式
# ===========================================================================

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _balanced_calls(text: str) -> list[str]:
    """扫描 ``Name(...)`` 形态并做括号配平（容忍前后夹带的自然语言）。"""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        m = re.compile(r"[A-Za-z_]\w*\s*\(").search(text, i)
        if m is None:
            break
        start = m.start()
        depth, j = 0, m.end() - 1
        while j < n:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            break
        out.append(re.sub(r"\s+", "", text[start:j + 1]))
        i = j + 1
    return out


def extract_formulas(text: str, *, limit: int | None = None) -> list[str]:
    """从大模型回复里抽取候选公式（容忍 JSON / 代码块 / 纯文本）。

    优先级：JSON 数组（或 ``{"factors": [...]}``）→ 代码块 → 全文扫描 ``Name(...)``。
    去重按**项目语法的规范形式**（``canonical``）判定，无法解析的按原文小写比较，
    保证「同一公式的不同写法」不会重复进入候选。
    """
    raw: list[str] = []

    payload = text
    m = _FENCE_RE.search(text)
    if m:
        payload = m.group(1)

    stripped = payload.strip()
    if stripped[:1] in "[{":
        try:
            obj = json.loads(stripped)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            obj = obj.get("factors") or obj.get("formulas") or []
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, str):
                    raw.append(it)
                elif isinstance(it, dict):
                    for k in ("formula", "expr", "expression", "factor"):
                        if isinstance(it.get(k), str):
                            raw.append(it[k])
                            break
        elif obj is not None:
            raw = []

    if not raw:
        raw = _balanced_calls(payload if payload is not payload.strip() else text)
        if not raw:
            raw = _balanced_calls(text)

    seen: set[str] = set()
    out: list[str] = []
    for f in raw:
        f = f.strip().rstrip(",;").strip()
        if not f:
            continue
        key = canonical(f) or f.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
        if limit is not None and len(out) >= limit:
            break
    return out


# ===========================================================================
# 5) 提示词
# ===========================================================================

SYSTEM_PROMPT = (
    "你是一名量化因子研究员，擅长用价量数据构造有经济学含义的选股因子。"
    "你只输出公式，不解释推导过程。"
)


def build_prompt(n: int, *, existing: Sequence[str] = (),
                 avoid: Sequence[str] = (), windows: Sequence[int] = WINDOWS,
                 max_body_tokens: int = MAX_EXPR_LENGTH - 2) -> str:
    """构造"生成 n 个逻辑自洽的量价因子"的提示词（研报未给出原文，此为自拟口径）。"""
    lines = [
        f"请构造 {n} 个股票日频量价选股因子表达式，要求每个因子都有清晰的经济学含义"
        "（如反转、波动调整动量、量价背离、量能冲击、位置指标等）。",
        "",
        "可用字段（6 个，须带 $ 前缀）：",
        "  $OPEN $HIGH $LOW $CLOSE $VOLUME $VWAP",
        "  （依次为开盘价、最高价、最低价、收盘价、成交量、成交量加权均价）",
        "",
        "可用常数（只能用下列值）：",
        "  " + "、".join(_fmt_const(c) for c in CONSTANTS),
        "",
        "可用算子（只能用下列算子）：",
        "  一元：Abs(x) Log(x)",
        "  二元：Add(a,b) Sub(a,b) Mul(a,b) Div(a,b) Greater(a,b) Less(a,b)",
        "  滚动（须把窗口作为末位参数）：Ref(x,w) Mean(x,w) Sum(x,w) Std(x,w) Var(x,w)",
        "        Max(x,w) Min(x,w) Med(x,w) Mad(x,w) Delta(x,w) WMA(x,w) EMA(x,w)",
        "  配对滚动：Cov(a,b,w) Corr(a,b,w)",
        f"  窗口 w 只能取：{'、'.join(str(w) for w in windows)}",
        "",
        "硬性约束：",
        f"  1. 表达式不超过 {max_body_tokens} 个 token"
        "（一个字段、一个常数、一个算子、一个窗口各算 1 个）；",
        "  2. 量纲必须自洽：Add/Sub/Greater/Less 两侧必须是同类量（价格与价格、成交量与成交量），"
        "禁止出现 Sub($VOLUME, $CLOSE) 这类价格与成交量相减的写法；",
        "  3. 禁止符号多余（如 Abs(Abs(x))）与恒等式（如 Sub($CLOSE, $CLOSE)）；",
        "  4. 禁止只输出单个字段（如 $VOLUME）或把单个字段套一层 Abs/Log 就交差；",
        "  5. 因子之间要尽量不相关，避免只改窗口的近似重复。",
    ]
    if existing:
        lines += ["", "以下因子已在因子池中，请勿重复或做仅改窗口的近似变体：",
                  "  " + "；".join(list(existing)[:20])]
    if avoid:
        lines += ["", "以下写法此前已被判定为无效或无法入池，请避开：",
                  "  " + "；".join(list(avoid)[:20])]
    lines += [
        "",
        "输出格式：只输出一个 JSON 数组，元素为公式字符串，不要加解释。例如",
        '["Div(Sub($CLOSE, Mean($CLOSE, 20)), Std($CLOSE, 20))", "Corr($CLOSE, $VOLUME, 20)"]',
    ]
    return "\n".join(lines)


# ===========================================================================
# 6) 提案器
# ===========================================================================

class FactorProposer(Protocol):
    """因子提案器接口：给出至多 ``n`` 条研报语法公式。"""

    name: str

    def propose(self, n: int, *, existing: Sequence[str] = (),
                avoid: Sequence[str] = ()) -> list[str]:  # pragma: no cover - 协议
        ...


#: 逻辑自洽的模板族：``(名称, 模板, 最小主窗口)``。每条模板都满足量纲自洽
#: （构造时由 check_report_formula 兜底），覆盖研报提到的常见量价因子族：
#: 反转、波动调整动量、量价背离、量能冲击、位置指标。
#:
#: **最小主窗口**不是装饰：``Std`` / ``Var`` / ``Mad`` / ``Corr`` / ``Cov`` 在
#: ``w=1`` 时退化为单点统计（实测 ``ts_std_1`` 全 NaN），做分母会把整条因子变成
#: "空值"档（奖励 0）。这属于模板自身的质量下限，校验器查不出来（语法与量纲都对），
#: 所以在这里显式设下限 —— 由 ``tests/test_ai97_llm_pool.py`` 的"目录全量可求值"
#: 用例兜底验证。
_TEMPLATES: tuple[tuple[str, str, int], ...] = (
    ("reversal_ret", "Div(Sub($CLOSE, Ref($CLOSE, {w1})), $CLOSE)", 1),
    ("vol_adj_mom", "Div(Delta($CLOSE, {w1}), Std($CLOSE, {w1}))", 5),
    ("close_zscore", "Div(Sub($CLOSE, Mean($CLOSE, {w1})), Std($CLOSE, {w1}))", 5),
    ("pv_corr", "Corr($CLOSE, $VOLUME, {w1})", 5),
    ("volume_ratio", "Div($VOLUME, Mean($VOLUME, {w1}))", 1),
    ("range_pct", "Div(Sub($HIGH, $LOW), $CLOSE)", 1),
    ("shadow_ratio", "Div(Sub($HIGH, $CLOSE), Sub($CLOSE, $LOW))", 1),
    ("vwap_gap", "Div(Sub($CLOSE, $VWAP), $VWAP)", 1),
    ("amount_strength", "Div(Mul($CLOSE, $VOLUME), Mean(Mul($CLOSE, $VOLUME), {w1}))", 1),
    ("price_volume_cov", "Cov($CLOSE, $VOLUME, {w1})", 5),
    ("wma_gap", "Div(Sub($CLOSE, WMA($CLOSE, {w1})), WMA($CLOSE, {w1}))", 1),
    ("ema_gap", "Div(Sub($CLOSE, EMA($CLOSE, {w1})), EMA($CLOSE, {w1}))", 1),
    ("range_position",
     "Div(Sub($CLOSE, Min($LOW, {w1})), Sub(Max($HIGH, {w1}), Min($LOW, {w1})))", 5),
    ("median_gap", "Div(Sub($CLOSE, Med($CLOSE, {w1})), Med($CLOSE, {w1}))", 1),
    ("mad_scale", "Div(Mad($CLOSE, {w1}), Mean($CLOSE, {w1}))", 5),
    ("volume_concentration", "Div(Mean($VOLUME, {w1}), Mean($VOLUME, {w2}))", 1),
    ("low_vol_mom", "Div(Sub($CLOSE, Ref($CLOSE, {w1})), Mad($CLOSE, {w1}))", 5),
    ("open_close_spread", "Div(Sub($CLOSE, $OPEN), Std($CLOSE, {w1}))", 5),
    ("turnover_shock", "Div($VOLUME, Med($VOLUME, {w1}))", 1),
    ("high_low_vol", "Div(Std($CLOSE, {w1}), Mean($CLOSE, {w1}))", 5),
)


class TemplateProposer:
    """离线确定性提案器（无网络依赖）。

    从 :data:`_TEMPLATES` 模板族展开参数组合、过滤掉校验不通过的、剔除与
    ``existing`` / ``avoid`` 重复的，再按 seed 确定性采样。

    这不是"真实大模型"，而是**在同一提示词口径下的模板兜底**：用于无 API key
    的环境跑通全链路、给单测一个稳定输入。它产出的公式**确实满足研报点名的三个
    逻辑要求**（量纲自洽 / 非平凡 / 无冗余），所以它也同时充当 `seed_pool` 的质量下限。
    """

    name = "template"

    def __init__(self, windows: Sequence[int] = WINDOWS, seed: int = 0):
        self.windows = tuple(windows)
        self.seed = int(seed)
        self._catalogue: list[str] | None = None

    def catalogue(self) -> list[str]:
        """全量候选（不含池内已有），按生成顺序去重。"""
        if self._catalogue is not None:
            return self._catalogue
        seen: set[str] = set()
        out: list[str] = []
        for _name, tmpl, min_w1 in _TEMPLATES:
            for w1 in self.windows:
                if w1 < min_w1:
                    continue
                w2s = [w for w in self.windows if w > w1] or [self.windows[-1]]
                for w2 in w2s:
                    text = tmpl.format(w1=w1, w2=w2)
                    chk = check_report_formula(text, windows=self.windows)
                    if not chk.ok or chk.project is None:
                        continue
                    if chk.project in seen:
                        continue
                    seen.add(chk.project)
                    out.append(text)
        self._catalogue = out
        return out

    def propose(self, n: int, *, existing: Sequence[str] = (),
                avoid: Sequence[str] = ()) -> list[str]:
        import random

        banned = {c for c in (canonical(x) for x in list(existing) + list(avoid)) if c}
        pool = [f for f in self.catalogue() if canonical(f) not in banned]
        rng = random.Random(self.seed)
        rng.shuffle(pool)
        return pool[:max(0, int(n))]


#: 项目注册名 → 研报算子名（:data:`REPORT_OPERATORS` 的反查表）。
_PROJECT_TO_REPORT: dict[str, str] = {v: k for k, v in REPORT_OPERATORS.items()}

#: 项目注册名的 GP 风格窗口后缀（如 ``ts_mean_20`` → 基名 ``ts_mean`` + 窗口 ``20``）。
_GP_WINDOW_SUFFIX_RE = re.compile(r"^(.*?)_(\d+)$")


def _project_node_to_report(node) -> str:
    """把 ``factor.formula.parse_formula`` 的 AST 节点转成研报语法。

    项目 AST 形状：``('feat', name)`` / ``('const', value)`` /
    ``('call', OpSpec, [children], win_name)``。窗口可能内建在 ``OpSpec.name``
    的后缀里（GP 风格，如 ``ts_mean_20``），也可能由 ``win_name`` 给出。

    与 :func:`to_report` 的区别：那个吃的是本项目自定义的 :class:`RNode`，
    这个吃的是 ``factor.formula`` 的元组 AST —— 随机采样器走的是后者。
    """
    kind = node[0]
    if kind == "feat":
        return f"${node[1]}"
    if kind == "const":
        return _fmt_const(float(node[1]))
    if kind != "call":
        raise FormulaSyntaxError(f"未知 AST 节点类型：{kind!r}")

    _tag, spec, children, win_name = node
    reg_name = spec.name
    win: Optional[int] = None
    # 窗口优先级：显式窗口参数 > 名字后缀 > OpSpec 默认
    if win_name:
        win = int(win_name[0]) if isinstance(win_name, (tuple, list)) else int(win_name)
    else:
        m = _GP_WINDOW_SUFFIX_RE.match(reg_name)
        if m and m.group(1) in _PROJECT_TO_REPORT:
            reg_name, win = m.group(1), int(m.group(2))
    report_name = _PROJECT_TO_REPORT.get(reg_name)
    if report_name is None:
        raise FormulaSyntaxError(f"算子 {reg_name!r} 无研报对应名")
    args = [_project_node_to_report(c) for c in children]
    if win is not None:
        args.append(str(win))
    return f"{report_name}({', '.join(args)})"


class UniformRandomProposer:
    """均匀随机提案器 —— **对照臂专用**，不含任何"知识"（2026-09-16 新增）。

    用途：把「LLM 带来的增量」拆成两个可分别检验的效应 ——

    - 效应 A（**初始池非空**）：空池起步 vs 有初始池起步；
    - 效应 B（**LLM 知识**）：初始池若由 LLM 构造，比随机构造好多少。

    因此本类与 :class:`OpenAICompatibleProposer` 的**接口完全一致**，
    唯一区别是候选不来自大模型，而是在同一 RPN token 空间上**均匀随机采样**。
    与 LLM 臂对照时，除"提案来源"外其余全部相同（同样的校验、同样的入池规则）。

    采样方式：在 :func:`factor.rl.alphapool_env.legal_mask_for_tokens` 给出的
    合法动作集上等概率选择，直到随机触发 SEP 或达到长度上限，再用
    :class:`factor.rl.alphapool_env.RPNParser` 转成项目语法、反推研报语法。
    这保证与 LLM 臂共享**同一个搜索空间**（不会因为采样器写错而人为抬高/压低对照臂）。

    ⚠️ 注意：随机公式**大概率通不过** :func:`check_report_formula` 的三个语义校验
    （量纲自洽 / 非平凡 / 无冗余），这是**预期的**——LLM 的价值正体现在这里。
    为拿到 ``n`` 条合规公式，内部会做有界重采样（``max_draws``），
    因此"实际产出 / 请求数"这个比例本身就是效应 B 的一个可报告指标。
    """

    name = "uniform"

    def __init__(self, seed: int = 0, *, max_draws_factor: int = 400,
                 windows: Sequence[int] = WINDOWS,
                 p_sep: float = 0.15, p_const: float = 0.15):
        self.seed = int(seed)
        self.windows = tuple(windows)
        # 每条合规公式允许的最大采样次数：随机公式通过率低，需较宽松的上界
        self.max_draws_factor = int(max_draws_factor)
        # p_sep：栈深==1 时选收尾的概率（压低"一上来就收尾"的退化式）
        # p_const：终端位选常数而非字段的概率；见 _draw_terminal 的根因说明
        self.p_sep = float(p_sep)
        self.p_const = float(p_const)
        self.n_draws = 0        # 累计采样次数（供报告"随机臂效率"）
        self.n_ok = 0           # 累计通过校验数

    # -- 采样 ---------------------------------------------------------------
    def _draw_tokens(self, rng) -> Optional[list[int]]:
        """在合法动作集上随机走一条 RPN 路径。"""
        from factor.rl.alphapool_env import (
            BEG,
            SEP,
            TOKEN_SPACE,
            legal_mask_for_tokens,
            op_arity,
        )

        tokens = [BEG]
        for _ in range(MAX_EXPR_LENGTH):
            mask = legal_mask_for_tokens(tokens)
            ids = [int(i) for i in np.flatnonzero(mask)]
            if not ids:
                return None
            non_sep = [i for i in ids if i != SEP]
            # SEP 仅在栈深==1 时可用；为避免"一上来就收尾"导致全是单字段退化式，
            # 当仍有非 SEP 选项时，以较低概率才选 SEP。
            if non_sep and SEP in ids and rng.random() < self.p_sep:
                tid = SEP
            elif non_sep and all(i in _TERMINAL_IDS for i in non_sep):
                tid = self._draw_terminal(non_sep, rng)
            elif non_sep:
                tid = int(rng.choice(non_sep))
            else:
                tid = int(rng.choice(ids))
            tokens.append(tid)
            if tid == SEP:
                body = [t for t in tokens if t not in (BEG, SEP)]
                return tokens if len(body) >= 3 else None
            # 防御：长度上限内未收尾则作废
            _ = TOKEN_SPACE[tid]
            _ = op_arity
        return None

    def _draw_terminal(self, ids: list[int], rng) -> int:
        """在「终端类 token」（field / const）之间取样，**偏向 field**。

        根因（2026-09-16 修复）：``CONSTANTS`` 有 13 个值、``FIELDS`` 只有 6 个，
        且算子 token 占绝对多数 —— 均匀采样时栈上很容易堆一串常数。常数当
        **操作数**时整条子树退化为标量，虽已被 ``scalar_operand`` /
        ``scalar_formula`` 拦下，但**每次被拦都是一次白抽**（实测通过率仅 5.5%）。
        仅对「纯终端选择」这一步做偏向即可：常数仍以 ``p_const`` 保留
        （``Mul($close, 2)`` / ``Div($close, 100)`` 这类合法缩放需要它），
        但不再和字段等概率。

        这**不改变**「均匀采样」对照臂的定义 —— 它仍然不含任何选股知识、
        不看 IC、不来自 LLM；只是把「抽到必被校验器拒绝的形态」的概率压低，
        与项目里 ``legal_mask_for_tokens`` 动态屏蔽非法动作是同一思路。
        """
        from factor.rl.alphapool_env import CONST_IDS, FIELD_IDS

        f_ids = [i for i in ids if i in FIELD_IDS]
        c_ids = [i for i in ids if i in CONST_IDS]
        if not f_ids:
            return int(rng.choice(c_ids or ids))
        if not c_ids:
            return int(rng.choice(f_ids))
        if rng.random() < self.p_const:
            return int(rng.choice(c_ids))
        return int(rng.choice(f_ids))

    def _draw_report_formula(self, rng) -> Optional[str]:
        """采一条 token 路径 → 项目语法 → 研报语法。"""
        from factor.rl.alphapool_env import RPNParser

        toks = self._draw_tokens(rng)
        if toks is None:
            return None
        project = RPNParser().parse(toks)
        if not project:
            return None
        # 项目语法 → 研报语法：借道 factor.formula 的 AST 再 to_report
        try:
            from factor.formula import parse_formula
            from factor.operators import op_registry

            node = parse_formula(project, features=list(FIELDS),
                                 registry=op_registry())
        except Exception:
            return None
        return _project_node_to_report(node)

    # -- 契约 ---------------------------------------------------------------
    def propose(self, n: int, *, existing: Sequence[str] = (),
                avoid: Sequence[str] = ()) -> list[str]:
        import random

        n = max(0, int(n))
        if n == 0:
            return []
        banned = {c for c in (canonical(x) for x in list(existing) + list(avoid)) if c}
        rng = random.Random(self.seed + len(banned))   # 与池状态相关但确定
        out: list[str] = []
        seen: set[str] = set()
        budget = self.max_draws_factor * n
        for _ in range(budget):
            if len(out) >= n:
                break
            text = self._draw_report_formula(rng)
            self.n_draws += 1
            if not text:
                continue
            chk = check_report_formula(text, windows=self.windows)
            if not chk.ok or chk.project is None:
                continue
            if chk.project in banned or chk.project in seen:
                continue
            seen.add(chk.project)
            self.n_ok += 1
            out.append(text)
        return out


class OpenAICompatibleProposer:
    """OpenAI 兼容接口的提案器（研报用 deepseek）。

    **联网验证状态（2026-09-16）**：已用 DeepSeek 真实接口验证通过
    （`scripts/oneoff/_probe_deepseek_conn.py`）。

    ⚠️ **`max_tokens` 必须覆盖思维链（这是本类最容易踩的坑）**：
    ``deepseek-flash`` / ``deepseek-v4-pro`` 都是**推理模型**，响应体除 ``content``
    外还有 ``reasoning_content``，且两者**共享 ``max_tokens`` 预算**
    （``usage.completion_tokens_details.reasoning_tokens``）。
    实测：``max_tokens=32`` / ``2000`` / ``3000`` 时，思维链把预算全部吃光，
    ``finish_reason="length"``、**``content`` 是空字符串** —— 表现为"大模型什么都没返回"，
    但 HTTP 200、无异常，极易误判成 key 或模型名的问题。
    实测 ``max_tokens=16000`` 时 ``finish_reason="stop"``，8 条公式正常产出
    （reasoning 约 6.3k token、正文约 0.2k）。故默认值取 **16000**。

    本类对"空 content"做了显式诊断（见 :meth:`_post`），不会静默返回空列表。

    Args:
        model: 模型名（研报为 deepseek）。本项目验证过 ``deepseek-flash``。
        base_url: OpenAI 兼容端点根。**DeepSeek 的 ``/models``、``/chat/completions``
            在根路径上直接可用**（``https://api.deepseek.com``），带不带 ``/v1``
            都能通（探针两路都测过）。
        api_key: 显式 key；``None`` 时读 ``api_key_env`` 环境变量。
        max_tokens: 单次回复的 token 上限，**含思维链**。见上方警告。
        rounds: 最多请求几轮（每轮最多 ``n`` 条，轮内去重，凑不满就再要一轮）。
        timeout: 单次请求超时（秒）。推理模型耗时长，实测单次 15~50s，故默认 300。
    """

    def __init__(
        self,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        api_key: str | None = None,
        api_key_env: str = "DEEPSEEK_API_KEY",
        temperature: float = 0.9,
        max_tokens: int = 16000,
        timeout: float = 300.0,
        rounds: int = 2,
        windows: Sequence[int] = WINDOWS,
    ):
        self.name = f"openai:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.rounds = max(1, int(rounds))
        self.windows = tuple(windows)
        self.last_raw: list[str] = []       # 最近一次回复的原始文本（排查用）
        #: 最近一次调用的诊断信息（finish_reason / reasoning_tokens / usage）。
        #: 空 content 的排查全靠它 —— 见 `_post`。
        self.last_meta: list[dict] = []

    # -- key 解析：显式传参 > 环境变量 > 报错 -------------------------------
    def api_key(self) -> str:
        key = self._api_key or os.environ.get(self._api_key_env)
        if not key:
            raise RuntimeError(
                f"缺少大模型 API key：请显式传 api_key= 或设置环境变量 "
                f"{self._api_key_env}。离线可用 build_proposer(\"template\") 走模板兜底。")
        return key

    def build_request(self, n: int, *, existing: Sequence[str] = (),
                      avoid: Sequence[str] = ()) -> dict:
        """构造请求体（单测覆盖此方法，不触网）。"""
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(
                    n, existing=existing, avoid=avoid, windows=self.windows)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

    def _post(self, payload: dict) -> str:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key()}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:              # pragma: no cover - 需联网
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"大模型接口返回 {exc.code}：{detail}") from exc
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"大模型响应结构异常：{str(body)[:300]}") from exc

        # -- 诊断元信息：空 content 的唯一可靠线索 -------------------------
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        meta = {
            "model": body.get("model", self.model),
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "content_len": len(content or ""),
            "reasoning_len": len(choice["message"].get("reasoning_content") or ""),
        }
        self.last_meta.append(meta)

        if not content:
            # 推理模型把 max_tokens 全花在思维链上时，就是"HTTP 200 + 空 content"，
            # 且无任何异常 —— 不显式报错的话，上层只会看到"提案 0 条"。
            rtok = meta["reasoning_tokens"]
            raise RuntimeError(
                f"大模型返回空 content（finish_reason={meta['finish_reason']!r}，"
                f"reasoning_tokens={rtok}，completion_tokens="
                f"{meta['completion_tokens']}）。若 finish_reason='length' 且 "
                f"reasoning_tokens 接近 max_tokens={self.max_tokens}，说明这是**推理模型**、"
                f"思维链把预算吃光了 —— 请调大 max_tokens（实测 16000 可用）。"
                f"模型={self.model}")

        return content

    def propose(self, n: int, *, existing: Sequence[str] = (),
                avoid: Sequence[str] = ()) -> list[str]:
        got: list[str] = []
        seen: set[str] = set()
        banned = {c for c in (canonical(x) for x in list(existing) + list(avoid)) if c}
        for _ in range(self.rounds):
            need = int(n) - len(got)
            if need <= 0:
                break
            content = self._post(self.build_request(
                need, existing=list(existing) + got, avoid=list(avoid)))
            self.last_raw.append(content)
            for f in extract_formulas(content):
                key = canonical(f) or f.lower()
                if key in seen or key in banned:
                    continue
                seen.add(key)
                got.append(f)
            if not got:
                # 回复非空但一条公式都抽不出来 —— 通常是模型无视了 JSON 数组格式
                # （`extract_formulas` 支持代码围栏 / 括号平衡扫描，兜底很宽）。
                log.warning("大模型回复 %d 字符但未抽出任何公式；原始文本前 300 字符：%s",
                            len(content), content[:300].replace("\n", " "))
        if not got:
            log.warning("propose(%d) 返回 0 条候选（rounds=%d，model=%s）",
                        n, self.rounds, self.model)
        return got[:int(n)]


#: ``make_proposer`` 转发给 :class:`OpenAICompatibleProposer` 的键；
#: 其余键（如 ``seed``/``windows``）转给 :class:`TemplateProposer`。
_OPENAI_KEYS = frozenset({
    "model", "base_url", "api_key", "api_key_env", "temperature",
    "max_tokens", "timeout", "rounds",
})


def make_proposer(kind: str, **kw) -> Optional[FactorProposer]:
    """按名字构造提案器。``"none"`` → ``None``（不启用大模型）。

    ``kw`` 会按**提案器类型**过滤后再转发 —— 两类的构造签名不同
    （``model``/``base_url``/``max_tokens`` 只有 OpenAI 版认），
    不过滤会让 CLI 传参在 ``--llm template`` 时直接 TypeError。
    未知键**静默丢弃**：调用方通常是统一的 CLI 参数集。
    """
    k = (kind or "none").lower()
    if k in ("none", "off", ""):
        return None
    if k in ("template", "offline", "mock"):
        return TemplateProposer(**{a: b for a, b in kw.items()
                                   if a not in _OPENAI_KEYS})
    if k in ("openai", "deepseek", "llm", "api"):
        return OpenAICompatibleProposer(**{a: b for a, b in kw.items()
                                           if a in _OPENAI_KEYS})
    raise ValueError(f"未知提案器 {kind!r}，可选：none / template / openai")


# ===========================================================================
# 7) 注入原语：构造基础池 / 定期去弱留强
# ===========================================================================

@dataclass
class PoolUpdateReport:
    """一次池更新（初始池 / 定期刷新）的结果。"""

    stage: str
    proposed: int = 0
    accepted: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    rejected: list[tuple[str, str]] = field(default_factory=list)   # (reason, formula)
    dropped: list[str] = field(default_factory=list)
    best_obj_before: float = 0.0
    best_obj_after: float = 0.0
    n_pool_before: int = 0
    n_pool_after: int = 0
    origin_counts: dict[str, int] = field(default_factory=dict)

    @property
    def best_obj_delta(self) -> float:
        return self.best_obj_after - self.best_obj_before

    def as_dict(self) -> dict:
        return {
            "stage": self.stage, "proposed": self.proposed,
            "accepted": self.accepted, "status_counts": dict(self.status_counts),
            "rejected": [{"reason": r, "formula": f} for r, f in self.rejected],
            "dropped": list(self.dropped),
            "best_obj_before": self.best_obj_before,
            "best_obj_after": self.best_obj_after,
            "n_pool_before": self.n_pool_before, "n_pool_after": self.n_pool_after,
            "origin_counts": dict(self.origin_counts),
        }

    def summary(self) -> str:
        return (f"[{self.stage}] 提案 {self.proposed} → 入池 {self.accepted}；"
                f"池 {self.n_pool_before}→{self.n_pool_after}"
                f"（{self.origin_counts}）；"
                f"best_obj {self.best_obj_before:.4f}→{self.best_obj_after:.4f}"
                f"（{self.best_obj_delta:+.4f}）；剔除 {len(self.dropped)}")


def _update(pool: AlphaPool, formulas: Sequence[str], stage: str,
            origin: str, features: Sequence[str] | None,
            with_dropped: list[str] | None = None,
            n_pool_pre: int | None = None,
            verbose: bool = True) -> PoolUpdateReport:
    rep = PoolUpdateReport(
        stage=stage, proposed=len(formulas),
        best_obj_before=float(pool.best_obj),
        # n_pool_pre：refresh 场景下本函数是在 drop 之后才被调用的，
        # 若直接读 len(pool.formulas) 会把"剔除后"的规模记成"更新前"，
        # 日志上 10→7→9 被压成 7→9。由调用方传入剔除前规模才是真实口径。
        n_pool_before=int(n_pool_pre) if n_pool_pre is not None else len(pool.formulas),
        dropped=list(with_dropped or []),
    )
    feat = list(features or pool.features)
    for text in formulas:
        chk = check_report_formula(text, features=feat)
        if not chk.ok or chk.project is None:
            rep.rejected.append((chk.reason, text))
            rep.status_counts["rejected:" + chk.reason] = (
                rep.status_counts.get("rejected:" + chk.reason, 0) + 1)
            continue
        status, _reward = pool.evaluate(chk.project, origin=origin)
        rep.status_counts[status] = rep.status_counts.get(status, 0) + 1
        if status == "pooled":
            rep.accepted += 1
        else:
            rep.rejected.append((status, text))
    rep.best_obj_after = float(pool.best_obj)
    rep.n_pool_after = len(pool.formulas)
    rep.origin_counts = pool.origin_counts()
    if verbose:
        log.info("%s", rep.summary())
    return rep


def seed_pool(pool: AlphaPool, formulas: Sequence[str], *, origin: str = "llm",
              features: Sequence[str] | None = None,
              verbose: bool = True) -> PoolUpdateReport:
    """把候选公式注入因子池（研报「构造基础池」）。

    先过 :func:`check_report_formula`（省掉必然无效的求值），再交给
    :meth:`AlphaPool.evaluate` 决定入池与否。公式一旦入池即带上 ``origin``
    来源标记，供 :meth:`AlphaPool.drop_worst` 按来源淘汰。
    """
    return _update(pool, formulas, "seed", origin, features, verbose=verbose)


def refresh_pool(pool: AlphaPool, proposer: FactorProposer, *, n_new: int,
                 drop_rl_n: int, origin: str = "llm",
                 features: Sequence[str] | None = None,
                 verbose: bool = True) -> PoolUpdateReport:
    """研报的定期更新：先剔除 ``drop_rl_n`` 个 RL 因子，再注入 ``n_new`` 条大模型因子。

    顺序是**先剔除再注入**（研报超参表把 ``drop_rl_n`` 定义为"每次丢弃多少个 RL
    模型生成的较差因子"，剔除的目的是给新因子腾位置），且只淘汰 ``origin="rl"``
    的因子 —— 大模型刚注入的因子不在本轮淘汰范围内。

    ``n_new`` 在研报里没有对应超参，本函数**不做默认**，必须由调用方显式给出，
    避免把自拟参数当成研报参数引用。
    """
    n_pre = len(pool.formulas)
    dropped = pool.drop_worst(origin="rl", n=drop_rl_n)
    if dropped and verbose:
        log.info("[refresh] 剔除 %d 个 RL 因子：%s", len(dropped), dropped)
    banned = list(pool.fail_cache) + list(pool.empty_cache)
    proposed = proposer.propose(n_new, existing=pool.formulas, avoid=banned)
    return _update(pool, proposed, "refresh", origin, features,
                   with_dropped=dropped, n_pool_pre=n_pre, verbose=verbose)
