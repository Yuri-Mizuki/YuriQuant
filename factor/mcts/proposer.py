"""MCTS 的 LLM 扩展器与离线变异兜底。

东吴三层闭环中 LLM 扮演**扩展器 + 审查器**：候选生成带
``hypothesis / expected_direction / relation_to_seed`` 元数据；反馈吸收读
父节点 RankIC/IR 弱项与历史路径（本文按 RD-Agent 机制③的"Observation /
Evaluation / Decision"三段式结构化传递，而非裸日志）。

吸收的 RD-Agent(Q) 机制（09-24 研读，见笔记 §三）：

- **⑦ JSON 纪律**：prompt 明令"公式必须完整可求值、不得含省略号或占位文本"，
  且 :func:`parse_variant_reply` 对含 ``...`` 的条目直接丢弃（防线双保险）。
- **④ 失败换向**：引擎检测到连续 N 轮无 SOTA 改进时置
  ``PromptContext.reset_complexity``，prompt 追加"降复杂度从简单公式重新起步"。
- （机制①数值去重在 engine 层，机制④的检测端同在 engine。）

离线兜底 :class:`SeedMutationProposer`：不触网、确定性（seed+计数器），
两种生成模式——根节点（Seed 展示含专属算子、不可解析）用"改造模板库 ×
Seed 字段族"；子节点（标准算子、可解析）做单点变异（字段/窗口/算子替换）。
供无 key 环境冒烟与单测，**不是真实大模型**（复现边界同 TemplateProposer）。

复现边界：东吴/论文均未给出 prompt 原文，本模块 prompt 为自拟口径
（与 llm_pool 现有模板同源，语法约束段一致）。
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from factor.rl.alphapool_env import CONSTANTS, FIELDS, MAX_EXPR_LENGTH, WINDOWS
from factor.rl.llm_pool import (
    OpenAICompatibleProposer,
    RNode,
    check_report_formula,
    extract_formulas,
    fmt_const,
    parse_report_formula,
    to_report,
)

__all__ = [
    "MCTSLLMProposer", "PromptContext", "SeedMutationProposer", "Variant",
    "build_mcts_prompt", "parse_variant_reply",
]

MCTS_SYSTEM_PROMPT = (
    "你是一名量化因子研究员，擅长围绕既有 Seed 因子做定向改造。"
    "你只输出一个 JSON 数组，不输出任何解释文字。"
)


@dataclass
class Variant:
    """一条候选变体（LLM 回复单元）。"""

    formula: str                      # 研报语法
    hypothesis: str = ""
    expected_direction: str = ""      # "+" / "-"
    relation_to_seed: str = ""


@dataclass
class PromptContext:
    """一次扩展调用的上下文（prompt 的全部输入件）。"""

    seed_name: str
    seed_display: str
    seed_note: str
    seed_stats: dict                  # {"rankic":…, "rankir":…, "reward":…}
    n_variants: int = 5
    #: 路径反馈（最近 3 个）：``(label, stats_dict)``，根→当前
    path_feedback: list[tuple[str, dict]] = field(default_factory=list)
    avoid: Sequence[str] = ()         # 近期被拒/低分公式（prompt 层避让）
    reset_complexity: bool = False    # RD-Agent 机制④：连续无改进后的换向提示


# ---------------------------------------------------------------------------
# prompt 构造
# ---------------------------------------------------------------------------

def _grammar_lines(windows: Sequence[int] = WINDOWS,
                   max_body_tokens: int = MAX_EXPR_LENGTH - 2) -> list[str]:
    """语法约束段（与 llm_pool.build_prompt 同源，保证变体空间与 AI97 一致）。"""
    return [
        "可用字段（6 个，须带 $ 前缀）：",
        "  $OPEN $HIGH $LOW $CLOSE $VOLUME $VWAP",
        "可用常数（只能用下列值）：",
        "  " + "、".join(fmt_const(c) for c in CONSTANTS),
        "可用算子（只能用下列标准算子，禁止使用 Seed 专属算子如 Slope/Rsquare/"
        "Resi/Quantile/Rank/IdxMax/IdxMin）：",
        "  一元：Abs(x) Log(x)",
        "  二元：Add(a,b) Sub(a,b) Mul(a,b) Div(a,b) Greater(a,b) Less(a,b)",
        "  滚动（窗口为末位参数）：Ref(x,w) Mean(x,w) Sum(x,w) Std(x,w) Var(x,w)",
        "        Max(x,w) Min(x,w) Med(x,w) Mad(x,w) Delta(x,w) WMA(x,w) EMA(x,w)",
        "  配对滚动：Cov(a,b,w) Corr(a,b,w)",
        f"  窗口 w 只能取：{'、'.join(str(w) for w in windows)}",
        "",
        "硬性约束：",
        f"  1. 表达式不超过 {max_body_tokens} 个 token"
        "（一个字段/常数/算子/窗口各算 1 个）；",
        "  2. 量纲必须自洽（价格与价格、成交量与成交量，禁止跨类相减）；",
        "  3. 禁止符号多余（Abs(Abs(x))）与恒等式（Sub(a, a)）；",
        "  4. 公式必须完整可求值，不得含省略号（...）或任何占位文本；",
        "  5. 禁止只输出单个字段或单字段套一层 Abs/Log。",
    ]


def _weakness_diagnosis(stats: dict) -> str:
    """父路径弱项诊断（RD-Agent 机制③的 Decision 段，确定性规则）。"""
    ic, ir, turn = (abs(stats.get("rankic", 0.0) or 0.0),
                    abs(stats.get("rankir", 0.0) or 0.0),
                    stats.get("turnover", float("nan")))
    notes = []
    if ir < 0.60:
        notes.append("RankIR 偏低（稳定性不足），优先考虑降噪：拉长窗口、"
                     "用中位数/加权均值替代简单均值")
    if ic < 0.05:
        notes.append("|RankIC| 偏低（强度不足），优先考虑换主字段或加量能确认项")
    if isinstance(turn, float) and turn == turn and turn > 0.8:
        notes.append("换手偏高，优先考虑平滑信号（滚动均值化、降低调仓敏感度）")
    return "；".join(notes) if notes else "表现均衡，可尝试结构性创新（跨字段组合、新算子嵌套）"


def build_mcts_prompt(ctx: PromptContext) -> str:
    """MCTS 扩展 prompt（东吴口径自拟，结构：Seed→任务→路径反馈→语法→输出）。"""
    st = ctx.seed_stats
    lines = [
        "【Seed 因子（改造锚点）】",
        f"  名称：{ctx.seed_name}",
        f"  公式：{ctx.seed_display}",
        f"  含义：{ctx.seed_note}",
        f"  样本内表现：RankIC={st.get('rankic', float('nan')):.4f}，"
        f"RankIR={st.get('rankir', float('nan')):.4f}，"
        f"reward={st.get('reward', float('nan')):.4f}",
        "",
        f"【任务】提出 {ctx.n_variants} 个针对该 Seed 的改进变体。"
        "每个变体围绕一个明确的经济学假设（如把'股数不稳定'改成'资金流不稳定'、"
        "给反转信号加量能确认、把计数型指标换成资金加权型），不要只改窗口。",
    ]
    if ctx.path_feedback:
        lines += ["", "【父路径反馈（从根到当前节点，最近 3 个已评测公式）】"]
        for label, pst in ctx.path_feedback[-3:]:
            lines.append(
                f"  - {label} → RankIC={pst.get('rankic', float('nan')):.4f}，"
                f"RankIR={pst.get('rankir', float('nan')):.4f}，"
                f"reward={pst.get('reward', float('nan')):.4f}")
        last = ctx.path_feedback[-1][1]
        lines.append(f"  弱项诊断：{_weakness_diagnosis(last)}")
        lines.append("  请针对弱项诊断提出与既有路径不同的改进方向，避免重复。")
    if ctx.avoid:
        lines += ["", "以下写法近期已被判定无效或低分，请避开：",
                  "  " + "；".join(list(ctx.avoid)[:10])]
    if ctx.reset_complexity:
        lines += ["",
                  "【换向提示】最近连续多轮没有出现更优公式：请大幅降低复杂度，"
                  "从简单公式（不超过 2 个算子）重新起步，先保证稳健再叠加结构。"]
    lines += [""] + _grammar_lines() + [
        "",
        "【输出格式】只输出一个 JSON 数组，每个元素含四个键：",
        '  {"formula": "研报语法公式", "hypothesis": "一句话投资假设",',
        '   "expected_direction": "+ 或 -", "relation_to_seed": "相对 Seed 改了什么、'
        '为什么可能更强"}',
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 回复解析（含 RD-Agent 机制⑦ 的省略号防线）
# ---------------------------------------------------------------------------

_ELLIPSIS_RE = re.compile(r"\.\.\.|…")


def parse_variant_reply(text: str) -> list[Variant]:
    """LLM 回复 → 变体列表（JSON 优先，裸公式兜底；含省略号的条目直接丢弃）。"""
    payload = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        payload = m.group(1).strip()
    items: list = []
    if payload[:1] in "[{":
        try:
            obj = json.loads(payload)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            obj = obj.get("variants") or obj.get("factors") or obj.get("formulas") or []
        if isinstance(obj, list):
            items = obj
    out: list[Variant] = []
    if items and all(isinstance(it, (str, dict)) for it in items):
        for it in items:
            if isinstance(it, str):
                out.append(Variant(formula=it))
            else:
                f = it.get("formula") or it.get("expr") or it.get("expression") or ""
                if not isinstance(f, str) or not f.strip():
                    continue
                out.append(Variant(
                    formula=f.strip(),
                    hypothesis=str(it.get("hypothesis", "") or ""),
                    expected_direction=str(it.get("expected_direction", "") or ""),
                    relation_to_seed=str(it.get("relation_to_seed", "") or ""),
                ))
    else:
        out = [Variant(formula=f) for f in extract_formulas(text)]
    # RD-Agent 机制⑦：省略号/占位文本一律丢弃（JSON 纪律的解析端防线）
    return [v for v in out
            if v.formula.strip() and not _ELLIPSIS_RE.search(v.formula)]


# ---------------------------------------------------------------------------
# LLM 扩展器（联网）
# ---------------------------------------------------------------------------

class MCTSLLMProposer(OpenAICompatibleProposer):
    """MCTS 上下文版扩展器：复用 OpenAICompatibleProposer 的 HTTP 通道与
    ``max_tokens`` 陷阱诊断，仅替换消息体（Seed/路径反馈/JSON 元数据）。"""

    def build_mcts_request(self, ctx: PromptContext) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": MCTS_SYSTEM_PROMPT},
                {"role": "user", "content": build_mcts_prompt(ctx)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

    def propose_variants(self, ctx: PromptContext) -> list[Variant]:
        content = self._post(self.build_mcts_request(ctx))
        self.last_raw.append(content)
        return parse_variant_reply(content)

    def judge_ask(self, prompt: str) -> str:
        """logic_review 依赖注入的 ``ask`` 回调（复用同一 HTTP 通道）。"""
        return self._post({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
        })

    @property
    def n_calls(self) -> int:
        return len(self.last_meta)

    @property
    def completion_tokens(self) -> int:
        return int(sum(m.get("completion_tokens") or 0 for m in self.last_meta))


# ---------------------------------------------------------------------------
# 离线变异兜底（确定性，不触网）
# ---------------------------------------------------------------------------

_PRICE_FIELDS = ("close", "vwap", "high", "low", "open")

#: 根节点改造模板（标准算子；``{f}`` 主字段、``{g}`` 交叉字段、``{w}`` 窗口）。
#: 覆盖东吴案例的改造方向：变异系数 / 资金加权 / 量能确认 / 区间位置 / 平滑。
_ROOT_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("cv", "Div(Std({f}, {w}), Mean({f}, {w}))"),
    ("zscore_gap", "Div(Sub({f}, Mean({f}, {w})), Std({f}, {w}))"),
    ("pv_corr", "Corr({f}, $VOLUME, {w})"),
    ("wma_gap", "Div(Sub({f}, WMA({f}, {w})), WMA({f}, {w}))"),
    ("range_pos", "Div(Sub({f}, Min({f}, {w})), Sub(Max({f}, {w}), Min({f}, {w})))"),
    ("fast_slow", "Div(Mean({f}, 5), Mean({f}, {w}))"),
    ("vol_adj_mom", "Div(Delta({f}, 5), Std({f}, {w}))"),
    ("chg_corr", "Corr(Div({f}, Ref({f}, 1)), Div($VOLUME, Ref($VOLUME, 1)), {w})"),
    ("med_gap", "Div(Sub({f}, Med({f}, {w})), Mad({f}, {w}))"),
    ("mom_x_vol", "Mul(Div(Sub({f}, Ref({f}, 5)), {f}), "
                  "Div($VOLUME, Mean($VOLUME, {w})))"),
)

_OP_SWAPS = {"Mean": ("Med", "EMA", "WMA"), "Med": ("Mean", "EMA"),
             "Std": ("Mad", "Var"), "Mad": ("Std", "Var"), "Var": ("Std", "Mad"),
             "Max": ("Min",), "Min": ("Max",), "WMA": ("Mean", "EMA"),
             "EMA": ("Mean", "WMA")}
_FIELD_SWAPS = {"close": ("vwap", "high", "low"), "vwap": ("close", "high"),
                "high": ("low", "close"), "low": ("high", "close"),
                "open": ("close", "vwap"), "volume": ()}


def _mutate_report_formula(text: str, rng: random.Random) -> tuple[str, str] | None:
    """对标准算子公式做单点变异（RNode 不可变 → 路径定位 + 重建子树）。

    返回 ``(新公式, 变异说明)``；不可解析 / 无可变异点返回 ``None``。
    """
    import copy

    try:
        node = parse_report_formula(text)
    except Exception:
        return None

    targets: list[tuple[tuple[int, ...], str]] = []

    def _collect(n, path):
        if n.kind == "call":
            name, win = n.value
            if name in _OP_SWAPS:
                targets.append((path, "op"))
            if win is not None:
                targets.append((path, "window"))
        elif n.kind == "field" and _FIELD_SWAPS.get(n.value):
            targets.append((path, "field"))
        for i, c in enumerate(n.children):
            _collect(c, path + (i,))

    _collect(node, ())
    if not targets:
        return None
    path, mut = rng.choice(targets)

    # 先沿路径取目标节点、选定变异载荷（载荷不可选则放弃该策略）
    target = copy.deepcopy(node)
    for i in path:
        target = target.children[i]
    payload = None
    if mut == "window":
        old_w = target.value[1]
        choices = [w for w in WINDOWS if w != old_w]
        if choices:
            payload = rng.choice(choices)
    elif mut == "op":
        choices = _OP_SWAPS.get(target.value[0], ())
        if choices:
            payload = rng.choice(choices)
    else:
        choices = _FIELD_SWAPS.get(target.value, ())
        if choices:
            payload = rng.choice(choices)
    if payload is None:
        return None
    old_disp = target.value if mut == "field" else target.value[0]

    def _rebuild(n, p):
        if not p:
            if mut == "window":
                return RNode("call", (n.value[0], payload), n.children)
            if mut == "op":
                return RNode("call", (payload, n.value[1]), n.children)
            return RNode("field", payload)
        children = tuple(_rebuild(c, p[1:]) if i == p[0] else c
                         for i, c in enumerate(n.children))
        return RNode(n.kind, n.value, children)

    if mut == "field":
        note = f"字段 ${str(old_disp).upper()}→${payload.upper()}"
    else:
        note = f"{'窗口' if mut == 'window' else '算子'} {old_disp}→{payload}"
    return to_report(_rebuild(node, path)), note


class SeedMutationProposer:
    """离线确定性扩展器（无网络依赖；供冒烟与单测，非真实大模型）。

    根节点：Seed 字段族 × 改造模板库（东吴案例方向）；子节点：父公式单点变异。
    产出全部过 :func:`check_report_formula`，不足 ``n`` 时用窗口扰动补齐。
    """

    name = "seed_mutation"

    def __init__(self, seed: int = 0, windows: Sequence[int] = WINDOWS):
        self._base_seed = int(seed)
        self._counter = 0
        self.windows = tuple(windows)
        self.n_calls = 0
        self.completion_tokens = 0

    def _seed_family_fields(self, seed_display: str) -> tuple[str, ...]:
        vol_heavy = seed_display.count("$VOLUME") >= max(
            1, seed_display.count("$CLOSE"))
        if vol_heavy:
            return ("volume", "vwap", "close")
        return ("close", "vwap", "high", "low")

    def propose_variants(self, ctx: PromptContext) -> list[Variant]:
        self.n_calls += 1
        self._counter += 1
        # 确定性：实例 seed × 调用序号（跨进程/跨次运行可复现）
        rng = random.Random(self._base_seed * 100003 + self._counter)
        out: list[Variant] = []
        seen: set[str] = set()
        # 路径反馈里有可解析的父公式（子节点扩展）→ 单点变异优先
        parent_formula = None
        for label, _ in reversed(ctx.path_feedback):
            if label != ctx.seed_name and "(" in label:
                parent_formula = label
                break
        attempts = 0
        while len(out) < ctx.n_variants and attempts < ctx.n_variants * 8:
            attempts += 1
            if parent_formula is not None:
                mut = _mutate_report_formula(parent_formula, rng)
                if mut is not None:
                    formula, hypothesis = mut
                    hypothesis = f"对父公式做单点变异（{hypothesis}），保留其核心结构"
                else:
                    got = self._from_template(ctx, rng)
                    if got is None:
                        continue
                    formula, hypothesis = got
            else:
                got = self._from_template(ctx, rng)
                if got is None:
                    continue
                formula, hypothesis = got
            chk = check_report_formula(formula, windows=self.windows)
            if not chk.ok or not chk.project or chk.project in seen:
                continue
            seen.add(chk.project)
            out.append(Variant(formula=formula, hypothesis=hypothesis,
                               expected_direction="", relation_to_seed=hypothesis))
        return out

    def _from_template(self, ctx: PromptContext,
                       rng: random.Random) -> tuple[str, str] | None:
        name, tpl = _ROOT_TEMPLATES[rng.randrange(len(_ROOT_TEMPLATES))]
        f = rng.choice(self._seed_family_fields(ctx.seed_display))
        w = rng.choice([w for w in self.windows if w >= 5] or [20])
        formula = tpl.format(f=f"${f.upper()}", w=w)
        return formula, f"{name} 模板作用于 ${f.upper()}（窗口 {w}）——" \
               f"对 Seed「{ctx.seed_name}」的定向结构改造"
