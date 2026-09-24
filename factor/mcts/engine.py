"""MCTS 主循环：select → expand（LLM 扩展 + 三层去重）→ eval → backprop。

去重防线（按序）：

1. **静态校验**：``check_report_formula``（语法/窗口/量纲/冗余/长度，
   AI97 同一套）+ canonical 去重 + 结构去重（``is_structural_duplicate``，
   QuantaAlpha 机制①的 AST 层）。
2. **logic_review**（东吴审查层，opt-in）：``llm_semantic_check`` 审查
   hypothesis ↔ 公式一致性，未通过不进评测（审查通道故障按未通过处理，
   不静默放行）。
3. **数值去重**（RD-Agent 机制①的数值防线）：候选 IS 周度 IC 序列与池内
   任一已收候选的 Spearman 相关 ≥ ``dedup_ic_corr``（默认 0.99）→ 剔除。
   比结构去重多一层「形异值同」防线。

失败换向（RD-Agent 机制④）：连续 ``fail_reset_rounds`` 轮全树无新最优 →
``PromptContext.reset_complexity``（prompt 层"降复杂度从简单公式重来"），
出新最优自动解除。

reward 口径：搜索期含 diversity（压同质候选）；**复评期放开**（东吴口径⑤，
``SearchRecord.reward_replay`` 以 ``max_abs_corr=None`` 重算，diversity 记
满分——避免把好候选压死在搜索期）。

Seed 的 complexity 用展示伪代码的 token 计数近似（专属算子不可解析，无法
走 AST 精确计数；与候选取同一 token 语义，见 seeds 模块说明）。
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from factor.mcts.proposer import (
    MCTSLLMProposer,
    PromptContext,
    SeedMutationProposer,
    Variant,
)
from factor.mcts.reward import CandidateEvaluator, EvalOutcome, WeeklyStats
from factor.mcts.seeds import SeedSpec
from factor.mcts.tree import MCTSNode, MCTSTree
from factor.rl.llm_pool import (
    canonical,
    check_report_formula,
    is_structural_duplicate,
)

log = logging.getLogger(__name__)

__all__ = ["RunConfig", "SearchRecord", "TreeReport", "display_complexity",
           "run_mcts_tree"]


@dataclass
class RunConfig:
    """单树搜索配置（东吴 §3 参数：iterations=10 / variants=5 / max_depth=3）。"""

    iterations: int = 10
    variants: int = 5
    max_depth: int = 3
    exploration: float = 1.0
    virtual_weight: float = 0.08
    #: RD-Agent 机制①：IS 周度 IC Spearman ≥ 此值判数值重复
    dedup_ic_corr: float = 0.99
    #: 结构去重开关（QuantaAlpha 机制①；关闭则只靠 canonical + 数值层）
    struct_dedup: bool = True
    #: logic_review（东吴审查层；llm 臂默认开，离线兜底强制关）
    logic_review: bool = False
    #: RD-Agent 机制④：连续 N 轮无全树新最优 → prompt 降复杂度换向
    fail_reset_rounds: int = 3
    seed: int = 0


@dataclass
class SearchRecord:
    """一条候选的完整台账（含被拒者——候选池诊断口径需要全部记录）。"""

    seed_name: str
    formula_report: str
    formula_project: str
    hypothesis: str
    expected_direction: str
    relation_to_seed: str
    status: str   # accepted / rejected_check / rejected_logic / rejected_struct_dup
                  # / rejected_numeric_dup / rejected_eval / rejected_seen
    reject_reason: str = ""
    depth: int = 0
    iteration: int = 0
    reward_search: float = float("nan")   # 含 diversity
    reward_replay: float = float("nan")   # 复评期放开 diversity（东吴口径⑤）
    max_abs_corr: float = float("nan")
    outcome: Optional[EvalOutcome] = None
    ic_weekly_is: Optional[pd.Series] = None

    def row(self) -> dict:
        o = self.outcome
        row = {
            "seed": self.seed_name, "formula_report": self.formula_report,
            "formula_project": self.formula_project,
            "hypothesis": self.hypothesis,
            "expected_direction": self.expected_direction,
            "relation_to_seed": self.relation_to_seed, "status": self.status,
            "reject_reason": self.reject_reason, "depth": self.depth,
            "iteration": self.iteration, "reward_search": self.reward_search,
            "reward_replay": self.reward_replay,
            "max_abs_corr": self.max_abs_corr,
        }
        for pref, st in (("is_", o.st_is if o else None),
                         ("oos_", o.st_oos if o else None)):
            for k, v in (st.as_dict() if st else WeeklyStats().as_dict()).items():
                row[pref + k] = v
        if o is not None:
            row["complexity"] = o.complexity
            row["digit_count"] = o.digit_count
        return row


@dataclass
class TreeReport:
    """单树搜索结果（给 runner 汇总层）。"""

    seed_name: str
    seed_outcome: EvalOutcome
    seed_reward_replay: float
    records: list[SearchRecord] = field(default_factory=list)
    n_llm_calls: int = 0
    n_completion_tokens: int = 0
    wall_seconds: float = 0.0


_TOKEN_COUNT_RE = re.compile(
    r"[A-Z][A-Za-z]+|\$[A-Z]+|\d+(?:\.\d+)?")


def display_complexity(display: str) -> tuple[int, int]:
    """Seed 展示伪代码的 (token 数, 数字字符数) 近似（与候选取同一 token 语义）。"""
    tokens = _TOKEN_COUNT_RE.findall(display)
    digits = sum(ch.isdigit() for t in tokens for ch in t if not t.startswith("$"))
    return len(tokens), int(digits)


def _ic_series_spearman(a: pd.Series, b: pd.Series) -> float:
    """两条周度 IC 序列的 Spearman 相关（对齐后 ≥8 个共同周才可信）。"""
    df = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(df) < 8:
        return float("nan")
    return float(df["a"].corr(df["b"], method="spearman"))


def run_mcts_tree(
    seed_spec: SeedSpec,
    seed_panel: pd.DataFrame,
    evaluator: CandidateEvaluator,
    proposer: MCTSLLMProposer | SeedMutationProposer,
    cfg: RunConfig,
    llm_judge: Optional[Callable[[str, str], bool]] = None,
) -> TreeReport:
    """跑一棵 Seed 树（东吴 §3 主循环；面板缓存与候选池均在树内生命周期）。"""
    from factor.gflownet.selection import panel_flat_corr
    from stats.ic import calc_ic_series

    t0 = time.perf_counter()
    is_idx = evaluator.panel_is["close"].index

    # -- Seed 评测（complexity 用展示伪代码近似）---------------------------
    seed_outcome = evaluator.evaluate_panel(seed_panel)
    seed_outcome.complexity, seed_outcome.digit_count = display_complexity(
        seed_spec.display)
    seed_reward_replay = seed_outcome.reward("is", max_abs_corr=None)
    seed_stats = {"rankic": seed_outcome.st_is.rankic_mean,
                  "rankir": seed_outcome.st_is.rankir,
                  "turnover": seed_outcome.st_is.turnover,
                  "reward": seed_reward_replay}

    tree = MCTSTree(
        root=MCTSNode(label=seed_spec.name, depth=0, reward=seed_reward_replay),
        max_depth=cfg.max_depth, exploration=cfg.exploration,
        virtual_weight=cfg.virtual_weight)

    # -- 候选池（含 Seed；全部 IS 切片对齐）---------------------------------
    seed_fp_is = seed_panel.reindex(is_idx).replace(
        [np.inf, -np.inf], np.nan)
    pool_fp: dict[str, pd.DataFrame] = {seed_spec.name: seed_fp_is}
    pool_ic: dict[str, pd.Series] = {
        seed_spec.name: calc_ic_series(
            seed_fp_is.reindex(evaluator.grid_is),
            evaluator.ret_is.reindex(evaluator.grid_is),
            method="spearman").dropna()}
    node_stats: dict[str, dict] = {seed_spec.name: seed_stats}
    seen_canonical: set[str] = set()
    seen_struct: list[str] = [seed_spec.display]

    records: list[SearchRecord] = []
    best_reward = seed_reward_replay
    rounds_since_improve = 0
    reset_hint = False
    recent_rejects: list[str] = []

    for it in range(1, cfg.iterations + 1):
        target = tree.select_target()
        path_feedback = [
            (n.label, node_stats.get(n.label, seed_stats))
            for n in target.path()]
        ctx = PromptContext(
            seed_name=seed_spec.name, seed_display=seed_spec.display,
            seed_note=seed_spec.note, seed_stats=seed_stats,
            n_variants=cfg.variants, path_feedback=path_feedback,
            avoid=recent_rejects[-10:], reset_complexity=reset_hint)
        tree.mark_expanded(target)

        try:
            variants: list[Variant] = proposer.propose_variants(ctx)
        except Exception as exc:                      # LLM 通道故障：本轮跳过不中断
            log.warning("iter %d proposer 异常：%s", it, exc)
            continue

        improved_this_round = False
        for var in variants:
            rec = SearchRecord(
                seed_name=seed_spec.name, formula_report=var.formula,
                formula_project="", hypothesis=var.hypothesis,
                expected_direction=var.expected_direction,
                relation_to_seed=var.relation_to_seed, status="accepted",
                depth=target.depth + 1, iteration=it)

            def _reject(status: str, reason: str) -> None:
                rec.status, rec.reject_reason = status, reason
                records.append(rec)
                recent_rejects.append(rec.formula_report)

            # -- 防线 1：静态校验 + canonical/结构去重 -----------------------
            chk = check_report_formula(var.formula)
            if not chk.ok:
                _reject("rejected_check", chk.reason)
                continue
            rec.formula_project = chk.project or ""
            key = canonical(var.formula) or rec.formula_project
            if key in seen_canonical:
                _reject("rejected_seen", "canonical 重复")
                continue
            if cfg.struct_dedup:
                dup, dup_of = is_structural_duplicate(
                    rec.formula_report, seen_struct)
                if dup:
                    _reject("rejected_struct_dup",
                            f"AST 结构同族（Jaccard≥0.6，vs {dup_of[:60]}）")
                    continue

            # -- 防线 2：logic_review（东吴审查层，opt-in）-------------------
            if cfg.logic_review:
                ok = bool(llm_judge(var.hypothesis or var.formula, var.formula)) \
                    if llm_judge is not None else False
                if not ok:
                    _reject("rejected_logic", "假设↔公式一致性未通过")
                    continue

            # -- 评测（IS 定 reward / OOS 只记录）----------------------------
            try:
                outcome = evaluator.evaluate_formula(rec.formula_project)
                fp_is = evaluator.factor_panel(rec.formula_project, "is")
                ic_is = calc_ic_series(
                    fp_is.reindex(evaluator.grid_is),
                    evaluator.ret_is.reindex(evaluator.grid_is),
                    method="spearman").dropna()
            except Exception as exc:
                _reject("rejected_eval", str(exc)[:200])
                continue
            rec.outcome = outcome
            rec.ic_weekly_is = ic_is

            # -- 防线 3：数值去重（RD-Agent 机制①）---------------------------
            dup = ""
            for name, ic0 in pool_ic.items():
                c = _ic_series_spearman(ic_is, ic0)
                if math.isfinite(c) and c >= cfg.dedup_ic_corr:
                    dup = name
                    break
            if dup:
                _reject("rejected_numeric_dup",
                        f"周度 IC 相关≥{cfg.dedup_ic_corr}（vs {dup}）")
                continue

            # -- 收编：diversity → 六项 reward → 挂树回传 ---------------------
            mac = 0.0
            for fp0 in pool_fp.values():
                c = panel_flat_corr(fp_is, fp0, min_overlap=200)
                if math.isfinite(c):
                    mac = max(mac, abs(c))
            rec.max_abs_corr = mac
            rec.reward_search = outcome.reward("is", max_abs_corr=mac)
            rec.reward_replay = outcome.reward("is", max_abs_corr=None)
            if not math.isfinite(rec.reward_search):
                _reject("rejected_eval", "reward NaN")
                continue

            seen_canonical.add(key)
            seen_struct.append(rec.formula_report)
            pool_fp[rec.formula_report] = fp_is
            pool_ic[rec.formula_report] = ic_is
            node_stats[rec.formula_report] = {
                "rankic": outcome.st_is.rankic_mean,
                "rankir": outcome.st_is.rankir,
                "turnover": outcome.st_is.turnover,
                "reward": rec.reward_search}
            tree.add_child(target, rec.formula_report, rec.reward_search)
            records.append(rec)
            if rec.reward_search > best_reward:
                best_reward, improved_this_round = rec.reward_search, True

        # -- 机制④：连续无全树新最优 → 换向提示（出新最优自动解除）-----------
        if improved_this_round:
            rounds_since_improve, reset_hint = 0, False
        else:
            rounds_since_improve += 1
            if rounds_since_improve >= cfg.fail_reset_rounds:
                reset_hint = True

    return TreeReport(
        seed_name=seed_spec.name, seed_outcome=seed_outcome,
        seed_reward_replay=seed_reward_replay, records=records,
        n_llm_calls=int(getattr(proposer, "n_calls", 0)),
        n_completion_tokens=int(getattr(proposer, "completion_tokens", 0)),
        wall_seconds=time.perf_counter() - t0)
