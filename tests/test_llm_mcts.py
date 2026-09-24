"""东吴 LLM-MCTS Phase 0（factor/mcts + run_llm_mcts）单元测试。

覆盖：
- 29 Seed 构成（rolling 全类 × w=20）与 alpha158 注册键对齐、面板可求值；
- 周度网格 / 前 10% 换手 / 六项 reward 分量的手算锚点；
- CandidateEvaluator 的防前视切片与 **IS/OOS 子树缓存隔离**（回归锚点：
  共用 node_cache 会把 IS 面板错配给 OOS → 全 NaN）；
- 树：UCT 选择 / virtual expansion 横扩 / 回传取 max / 深度上限；
- proposer：JSON 纪律（省略号丢弃）/ prompt 关键行 / 离线变异确定性 + 可解析；
- 引擎：三层去重各防线单点触发、logic_review 拒绝路径、机制④换向提示。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from factor.alpha158 import ALPHA158
from factor.mcts.engine import RunConfig, display_complexity, run_mcts_tree
from factor.mcts.proposer import (
    PromptContext,
    SeedMutationProposer,
    Variant,
    build_mcts_prompt,
    parse_variant_reply,
)
from factor.mcts.reward import (
    CandidateEvaluator,
    WeeklyStats,
    ast_complexity,
    reward_components,
    six_part_reward,
    top_decile_turnover,
    weekly_grid,
    weekly_stats,
)
from factor.mcts.seeds import SEEDS, SEED_WINDOW, compute_seed_panels
from factor.mcts.tree import MCTSNode, MCTSTree
from factor.rl.llm_pool import check_report_formula


# ---------------------------------------------------------------------------
# 合成面板 fixture（与 run_llm_mcts.build_mock_panel 同构，但独立小样本）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_panel() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2019-01-01", periods=1600)
    cols = [f"C{i:02d}" for i in range(30)]
    close = pd.DataFrame(
        10 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(idx), len(cols))), axis=0)),
        index=idx, columns=cols)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_.values, close.values) * 1.005
    low = np.minimum(open_.values, close.values) * 0.995
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, close.shape),
                          index=idx, columns=cols)
    amount = volume * close
    return {"open": open_,
            "high": pd.DataFrame(high, index=idx, columns=cols),
            "low": pd.DataFrame(low, index=idx, columns=cols),
            "close": close, "volume": volume, "amount": amount,
            "vwap": amount / volume}


@pytest.fixture(scope="module")
def evaluator(mock_panel):
    idx = mock_panel["close"].index
    is_idx = idx[idx < "2024-01-01"]
    oos_idx = idx[idx >= "2024-01-01"]
    return CandidateEvaluator(mock_panel, is_idx, oos_idx, horizon=5)


# ---------------------------------------------------------------------------
# Seed 层
# ---------------------------------------------------------------------------

def test_seeds_are_29_rolling_classes_at_w20():
    assert len(SEEDS) == 29
    assert SEED_WINDOW == 20
    names = {s.name for s in SEEDS}
    # 东吴案例因子名必须在内（研读笔记 §二 代表案例）
    assert {"VSTD", "STD", "CNTN", "RANK"} <= names
    for s in SEEDS:
        assert s.key in ALPHA158, f"{s.key} 未注册于 alpha158"
        assert s.display and s.note


def test_compute_seed_panels_finite(mock_panel):
    panels = compute_seed_panels(mock_panel)
    assert len(panels) == 29
    close = mock_panel["close"]
    tail = slice(-200, None)          # 跳过滚动 warmup
    for name, fp in panels.items():
        frac = fp.iloc[tail].notna().mean().mean()
        assert frac > 0.9, f"Seed {name} 尾部覆盖率 {frac:.2f}"


# ---------------------------------------------------------------------------
# 周度统计与六项 reward
# ---------------------------------------------------------------------------

def test_weekly_grid_last_trading_day_of_week():
    idx = pd.bdate_range("2024-01-01", periods=30)   # 周一到周五
    g = weekly_grid(idx)
    assert len(g) < len(idx)
    # 每个网格日都是其 ISO 周内的最后一个交易日
    for d in g:
        same_week = idx[idx.to_period("W") == d.to_period("W")]
        assert d == same_week[-1]


def test_top_decile_turnover_known_construction():
    idx = pd.bdate_range("2024-01-01", periods=6)
    cols = [f"S{i}" for i in range(10)]
    same = pd.DataFrame(np.arange(60).reshape(6, 10), index=idx, columns=cols)
    assert top_decile_turnover(same) == pytest.approx(0.0)
    # 交替全换：奇偶周 top 集完全不相交 → 换手率 ≈ 1
    alt = pd.DataFrame(
        np.where(np.arange(6)[:, None] % 2 == 0,
                 np.arange(10), -np.arange(10) - 1),
        index=idx, columns=cols)
    assert top_decile_turnover(alt) == pytest.approx(1.0)


def test_reward_components_hand_check():
    st = WeeklyStats(rankic_mean=0.05, rankic_std=0.05, rankir=1.0,
                     coverage=0.9, turnover=0.8)
    comp = reward_components(st, complexity=10, digit_count=4, max_abs_corr=0.4)
    sig = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
    assert comp["effectiveness"] == pytest.approx(sig(0.0))
    assert comp["stability"] == pytest.approx(sig((1.0 - 0.6) / 0.25))
    assert comp["turnover"] == pytest.approx(sig(0.0))
    assert comp["diversity"] == pytest.approx(0.6)
    assert comp["coverage"] == pytest.approx(0.9)
    total = six_part_reward(st, 10, 4, 0.4)
    w = dict(effectiveness=0.40, stability=0.30, coverage=0.02,
             turnover=0.10, diversity=0.10, overfit=0.08)
    assert total == pytest.approx(sum(w[k] * comp[k] for k in w))
    # NaN RankIC → 总分 NaN（无法评测不给假分）
    st_nan = WeeklyStats(rankic_mean=float("nan"))
    assert math.isnan(six_part_reward(st_nan, 10, 4, None))
    # 复评期放开 diversity（东吴口径⑤）
    assert reward_components(st, 10, 4, None)["diversity"] == 1.0


def test_ast_complexity_counts():
    nodes, digits = ast_complexity("ts_corr_20(close, volume)")
    assert nodes == 3 and digits == 2
    # div + sub + close + ts_mean_20 + close + 0.5 = 6 节点；
    # digits = 窗口 "20"(2) + 常数 "0.5" 的数字字符(2) = 4
    nodes, digits = ast_complexity("div(sub(close, ts_mean_20(close)), 0.5)")
    assert nodes == 6 and digits == 4


# ---------------------------------------------------------------------------
# 评测器（防前视 + 缓存隔离回归锚）
# ---------------------------------------------------------------------------

def test_evaluator_formula_equivalent_to_alpha158_rollover(evaluator, mock_panel):
    """项目语法 ROC20 ≈ alpha158_ROC20（safe_div vs div 的 eps 差异内高一致）。"""
    seed_fp = compute_seed_panels(mock_panel)["ROC"]
    formula_fp = evaluator.factor_panel("div(ts_ref_20(close), close)", "is")
    both = seed_fp.notna() & formula_fp.notna()
    assert both.sum().sum() > 1000
    corr = seed_fp[both].corrwith(formula_fp[both])
    assert corr.mean() > 0.999


def test_evaluator_is_oos_cache_isolation(evaluator):
    """回归锚：node_cache 键不含面板——IS/OOS 必须各持缓存，否则 OOS 全 NaN。"""
    o = evaluator.evaluate_formula("ts_corr_20(close, volume)")
    assert o.st_is.n_weeks > 100
    assert o.st_oos.n_weeks > 20, "OOS 周度 IC 为空（子树缓存跨面板污染）"


def test_evaluator_no_lookahead_across_split(evaluator, mock_panel):
    """IS 尾部标签不吃 OOS 价格：ret_is 最后 h 行必须 NaN。"""
    assert evaluator.ret_is.iloc[-1].isna().all()
    assert evaluator.ret_oos.iloc[-1].isna().all()


def test_weekly_stats_via_evaluator(evaluator, mock_panel):
    seed_fp = compute_seed_panels(mock_panel)["ROC"]
    st = evaluator.evaluate_panel(seed_fp).st_is
    assert st.n_weeks > 100
    assert 0 <= st.coverage <= 1
    assert 0 <= st.turnover <= 1


# ---------------------------------------------------------------------------
# 树
# ---------------------------------------------------------------------------

def test_tree_select_expand_backprop():
    root = MCTSNode(label="seed", depth=0, reward=0.5)
    tree = MCTSTree(root, max_depth=3)
    assert tree.select_target() is root            # 空树先扩根
    a = tree.add_child(root, "a", 0.7)
    b = tree.add_child(root, "b", 0.4)
    assert root.visits == 2 and root.best_reward == 0.7
    # a 的 UCT(0.7+√(ln3/1)) 高于 b 与 root 的 virtual(0.7+0.08/2) → 下挖 a
    assert tree.select_target() is a
    # 回传：a 下挂更优子节点后，root.best_reward 跟着抬
    tree.add_child(a, "a1", 0.9)
    assert a.best_reward == 0.9 and root.best_reward == 0.9


def test_tree_depth_cap():
    root = MCTSNode(label="seed", depth=0, reward=0.5)
    tree = MCTSTree(root, max_depth=1)
    c = tree.add_child(root, "c", 0.6)
    # 唯一子节点已到深度上限：select 退回其父（root）而非 c
    assert tree.select_target() is root


def test_tree_virtual_expansion_wins_with_high_weight():
    root = MCTSNode(label="seed", depth=0, reward=0.5)
    tree = MCTSTree(root, max_depth=3, virtual_weight=10.0)   # 极端横扩偏好
    tree.add_child(root, "a", 0.9)
    assert tree.select_target() is root         # virtual 必然压过任何子 UCT


# ---------------------------------------------------------------------------
# proposer
# ---------------------------------------------------------------------------

CTX = PromptContext(seed_name="ROC", seed_display="Div(Ref($CLOSE, 20), $CLOSE)",
                    seed_note="反转锚", seed_stats={"rankic": 0.01, "rankir": 0.1,
                    "turnover": 0.5, "reward": 0.3}, n_variants=3)


def test_parse_variant_reply_json_and_ellipsis_ban():
    reply = ('[{"formula": "Corr($CLOSE, $VOLUME, 20)", "hypothesis": "量价背离",'
             ' "expected_direction": "-", "relation_to_seed": "换相关结构"},'
             ' {"formula": "Mean($CLOSE, ...)", "hypothesis": "占位"}]')
    vs = parse_variant_reply(reply)
    assert len(vs) == 1 and "..." not in vs[0].formula
    assert vs[0].expected_direction == "-"
    # 裸数组 / 纯文本兜底
    assert len(parse_variant_reply('["Std($VOLUME, 20)", "Abs($CLOSE)"]')) == 2


def test_build_mcts_prompt_key_lines():
    p = build_mcts_prompt(CTX)
    assert "Seed 因子" in p and "ROC" in p
    assert "省略号" in p                       # RD-Agent 机制⑦
    assert "禁止使用 Seed 专属算子" in p
    assert "换向提示" not in p                  # 未触发
    ctx2 = PromptContext(**{**CTX.__dict__, "reset_complexity": True})
    assert "换向提示" in build_mcts_prompt(ctx2)   # RD-Agent 机制④
    ctx3 = PromptContext(**{**CTX.__dict__,
                            "path_feedback": [("x", {"rankic": 0.01,
                                                     "rankir": 0.1,
                                                     "turnover": 0.9})]})
    assert "弱项诊断" in build_mcts_prompt(ctx3)   # 机制③三段式反馈


def test_seed_mutation_proposer_deterministic_and_valid():
    p1 = SeedMutationProposer(seed=42)
    p2 = SeedMutationProposer(seed=42)
    out1, out2 = p1.propose_variants(CTX), p2.propose_variants(CTX)
    assert [v.formula for v in out1] == [v.formula for v in out2]
    assert 0 < len(out1) <= 3
    for v in out1:
        chk = check_report_formula(v.formula)
        assert chk.ok, f"{v.formula}: {chk.reason}"
        assert v.hypothesis


def test_seed_mutation_mutates_parent_formula():
    """带可解析父公式的上下文 → 单点变异产出且可解析。"""
    ctx = PromptContext(**{**CTX.__dict__, "path_feedback": [
        ("ROC", CTX.seed_stats),
        ("Corr($CLOSE, $VOLUME, 20)", {"rankic": 0.02, "rankir": 0.3,
                                       "turnover": 0.5})]})
    out = SeedMutationProposer(seed=1).propose_variants(ctx)
    assert out
    for v in out:
        assert check_report_formula(v.formula).ok


# ---------------------------------------------------------------------------
# 引擎（三层去重单点触发 + logic_review + 换向）
# ---------------------------------------------------------------------------

class _FixedProposer:
    """桩扩展器：按脚本顺序吐候选（不触网、可控）。"""

    name = "fixed"

    def __init__(self, script: list[list[str]]):
        self.script = list(script)
        self.n_calls = 0
        self.completion_tokens = 0
        self.ctxs: list[PromptContext] = []

    def propose_variants(self, ctx):
        self.ctxs.append(ctx)
        self.n_calls += 1
        batch = self.script.pop(0) if self.script else []
        return [Variant(formula=f, hypothesis="测试假设") for f in batch]


def _run_engine(evaluator, mock_panel, script, **cfg_over):
    spec = SEEDS[0]      # ROC
    cfg = RunConfig(iterations=max(1, len(script)), variants=5,
                    fail_reset_rounds=3, **cfg_over)
    proposer = _FixedProposer(script)
    rep = run_mcts_tree(spec, compute_seed_panels(mock_panel)["ROC"],
                        evaluator, proposer, cfg)
    return rep, proposer


def test_engine_accepts_valid_candidate(evaluator, mock_panel):
    rep, _ = _run_engine(evaluator, mock_panel,
                         [["Corr($CLOSE, $VOLUME, 20)",
                           "Div(Sub($LOW, Mean($LOW, 60)), Std($LOW, 60))"]])
    assert rep.seed_name == "ROC"
    acc = [r for r in rep.records if r.status == "accepted"]
    assert len(acc) == 2
    for r in acc:
        assert math.isfinite(r.reward_search) and math.isfinite(r.reward_replay)
        assert r.outcome.st_oos.n_weeks > 0


def test_engine_static_check_rejects_garbage(evaluator, mock_panel):
    rep, _ = _run_engine(evaluator, mock_panel,
                         [["Sub($VOLUME, $CLOSE)",        # 量纲不匹配
                           "Mean($CLOSE, 7)",             # 窗口不在白名单
                           "$VOLUME"]])                   # 平凡终端
    assert all(r.status == "rejected_check" for r in rep.records)


def test_engine_seen_and_struct_dup(evaluator, mock_panel):
    rep, _ = _run_engine(evaluator, mock_panel,
                         [["Corr($CLOSE, $VOLUME, 20)"],
                          ["Corr($CLOSE, $VOLUME, 20)",   # canonical 重复
                           "Corr($LOW, $VOLUME, 5)"]])    # 同结构不同窗/字段
    status = [r.status for r in rep.records]
    assert status[0] == "accepted"
    assert "rejected_seen" in status
    assert "rejected_struct_dup" in status


def test_engine_logic_review_gate(evaluator, mock_panel):
    rep, _ = _run_engine(evaluator, mock_panel,
                         [["Corr($CLOSE, $VOLUME, 20)"]],
                         logic_review=True)             # llm_judge=None → 全拒
    assert all(r.status == "rejected_logic" for r in rep.records)


def test_engine_reset_hint_after_stagnation(evaluator, mock_panel):
    # 第 1 轮收编一条，后续 4 轮全部重复同一条（rejected_seen、零改进）
    # → 无论首条是否胜过 Seed，3 轮无改进后必然出现换向提示（机制④）
    x = "Corr($CLOSE, $VOLUME, 20)"
    script = [[x], [x], [x], [x], [x]]
    rep, proposer = _run_engine(evaluator, mock_panel, script)
    assert proposer.n_calls == 5
    assert any(c.reset_complexity for c in proposer.ctxs)


def test_display_complexity():
    n, d = display_complexity("Div(Ref($CLOSE, 20), $CLOSE)")
    assert n == 5 and d == 2
