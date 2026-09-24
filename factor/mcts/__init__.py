"""东吴 LLM-MCTS 因子迭代框架（Phase 0 试点）。

研读真源：``reports/docs/research_notes/东吴0623_研读_LLM_MCTS因子迭代框架与Phase0设计.md``
（本包按该设计文档实现，含 RD-Agent 机制抽取 ①④⑦）。

- :mod:`factor.mcts.seeds`：29 个 Alpha158 rolling Seed（窗口统一 20）
- :mod:`factor.mcts.reward`：周度六项 reward（东吴口径）
- :mod:`factor.mcts.tree`：MCTS 节点/UCT/virtual expansion
- :mod:`factor.mcts.proposer`：LLM 扩展器（MCTS 上下文 prompt）+ 离线变异兜底
- :mod:`factor.mcts.engine`：select-expand-eval-backprop 主循环 + 去重防线

四臂对照 runner：``scripts/factors/run_llm_mcts.py``。
"""
from factor.mcts.engine import (
    RunConfig,
    SearchRecord,
    TreeReport,
    display_complexity,
    run_mcts_tree,
)
from factor.mcts.proposer import (
    MCTSLLMProposer,
    PromptContext,
    SeedMutationProposer,
    Variant,
    build_mcts_prompt,
    parse_variant_reply,
)
from factor.mcts.reward import (
    CandidateEvaluator,
    EvalOutcome,
    WeeklyStats,
    reward_components,
    six_part_reward,
    weekly_grid,
)
from factor.mcts.seeds import SEEDS, SEED_WINDOW, SeedSpec, compute_seed_panels
from factor.mcts.tree import MCTSNode, MCTSTree

__all__ = [
    "CandidateEvaluator", "EvalOutcome", "MCTSLLMProposer", "MCTSNode",
    "MCTSTree", "PromptContext", "RunConfig", "SEEDS", "SEED_WINDOW",
    "SearchRecord", "SeedMutationProposer", "SeedSpec", "TreeReport",
    "Variant", "WeeklyStats", "build_mcts_prompt", "compute_seed_panels",
    "display_complexity", "parse_variant_reply", "reward_components",
    "run_mcts_tree", "six_part_reward", "weekly_grid",
]
