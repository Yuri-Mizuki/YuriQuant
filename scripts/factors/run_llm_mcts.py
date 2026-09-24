"""东吴 LLM-MCTS 因子迭代 Phase 0 —— 同题四引擎对照 runner
========================================================================

设计真源：``reports/docs/research_notes/东吴0623_研读_LLM_MCTS因子迭代框架与Phase0设计.md``
（reward 六项 / UCT / virtual expansion / 参数表 / 验收口径全在该笔记）。

四臂（同一 29 Seed、同一六项 reward、同一报告层，只换搜索引擎）：

- ``mcts``        ：LLM-MCTS 主臂（``factor/mcts/engine.py``，东吴口径）；
- ``llm_oneshot`` ：LLM 一次性生成（无树无反馈——分离「LLM 语义生成」与
  「MCTS 控制增益」，若本臂 ≈ mcts 臂，结论即 MCTS 控制增益有限，如实报）；
- ``gp``          ：项目 GP 引擎（``factor/genetic_mining.py``）；
- ``gflownet``    ：项目 GFlowNet 引擎（``factor/gflownet/``，国金22 复现线）。

用法::

    P=D:/Python/Python312/python
    # 本机冒烟（合成面板 + 离线变异兜底，零数据/零网络依赖）
    $P -m scripts.factors.run_llm_mcts --panel mock --arm mcts --smoke

    # 正式（好机器 + DEEPSEEK_API_KEY；llm 臂默认开 logic_review）
    $P -u scripts/factors/run_llm_mcts.py --panel hs300_2015_2026 \
        --is 2016-01-01:2023-12-31 --oos 2024-01-01:2026-08-21 --horizon 5 \
        --arm mcts --iterations 10 --variants 5 --max-depth 3 --seeds-n 3 \
        --llm openai --out reports/llm_mcts_phase0
    # 平行臂换 --arm gp / gflownet / llm_oneshot（对照臂不需 key，template 即可）

**口径与复现边界（引用数字必须注明，全清单见设计笔记 §五）**：

- h=5 对齐东吴（非项目生产 h=1）；指标绝对值不可跨报告比，只比同池四臂。
- IS 选型、OOS 只验证；「正式选择 OOS 兑现率」对标东吴 **65.5%**、「候选池
  诊断率」对标 **75.4%**——判据为自拟：**OOS 六项 reward（复评口径，diversity
  放开）> Seed 的 OOS reward**（东吴未明示判据，报告同时给 |RankIC| 版本列）。
- gp / gflownet 臂的搜索内部件保持各自引擎原生（GP 日频 IC 适应度、GFlowNet
  TB 奖励），**报告层统一走本脚本的周度六项评测**——对照的是「搜索引擎」，
  不是「搜索内芯」。
- ``--panel mock`` 为合成随机游走面板（仅链路验证，数字无意义）。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
)
from factor.mcts.reward import CandidateEvaluator
from factor.mcts.seeds import SEEDS, SEED_WINDOW, compute_seed_panels
from factor.rl.alphapool_env import FIELDS

log = logging.getLogger("run_llm_mcts")

#: 面板预设：名字 → (universe, begin, end)。
PANEL_PRESETS: dict[str, tuple[str, int, int]] = {
    "hs300_2015_2026": ("hs300", 20150108, 20260821),
}

DEFAULT_IS = "2016-01-01:2023-12-31"
DEFAULT_OOS = "2024-01-01:2026-08-21"


# ---------------------------------------------------------------------------
# 面板构建
# ---------------------------------------------------------------------------

def build_mock_panel(n_codes: int = 60, years: float = 8.6,
                     seed: int = 7) -> dict[str, pd.DataFrame]:
    """合成随机游走面板（冒烟用；2018 起 ~8.6 年覆盖默认 IS/OOS 窗口）。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=int(244 * years))
    cols = [f"C{i:03d}" for i in range(n_codes)]
    base = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(idx), n_codes)), axis=0))
    close = pd.DataFrame(base, index=idx, columns=cols)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.DataFrame(np.maximum(open_.values, close.values) * (1 + rng.uniform(
        0, 0.01, close.shape)), index=idx, columns=cols)
    low = pd.DataFrame(np.minimum(open_.values, close.values) * (1 - rng.uniform(
        0, 0.01, close.shape)), index=idx, columns=cols)
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, close.shape),
                          index=idx, columns=cols)
    amount = volume * (high + low + close) / 3.0
    vwap = (amount / volume).replace([np.inf, -np.inf], np.nan)
    return {"open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "amount": amount, "vwap": vwap}


def load_panel(name: str, offline: bool) -> tuple[dict[str, pd.DataFrame], int, int]:
    if name == "mock":
        return build_mock_panel(), 20180101, 20240101
    if name not in PANEL_PRESETS:
        raise ValueError(f"未知面板 {name!r}，可选：{sorted(PANEL_PRESETS)} | mock")
    from data.cache_helpers import build_real_panel
    from config import Config

    universe, begin, end = PANEL_PRESETS[name]
    cfg = Config.get()
    cfg["universe"]["default"] = universe
    panel, _ = build_real_panel(cfg, begin, end, offline=offline)

    # ⚠️ vwap 复权口径（2026-09-16 AI97 真实数据暴露的 bug，同款防线）：
    # build_panel 只后复权 OHLC，amount/volume（及缓存里的 vwap）是原始口径，
    # 与后复权 close 不可比（本机缓存实测 vwap/close 中位数 0.25）。丢弃缓存
    # vwap、用后复权因子重建（run_alphapool_ppo.attach_vwap 同款，这里内联
    # 实现避免拖入 rl 依赖）。
    backward = None
    try:
        from data.cache import DataCache
        from data.cache_helpers import load_backward_factor
        from data.datasource import create_datasource

        ds = create_datasource(Config.datasource())
        backward = load_backward_factor(DataCache(ds),
                                        list(panel["close"].columns))
    except Exception as exc:
        log.warning("读取后复权因子失败（%s）→ vwap 保持未复权口径，"
                    "与 close 不可比，量价交叉类公式将失真", exc)
    panel = dict(panel)
    panel.pop("vwap", None)
    vwap = panel["amount"] / panel["volume"].replace(0, np.nan)
    if backward is not None:
        bf = backward.reindex(index=vwap.index, columns=vwap.columns).ffill()
        vwap = vwap * bf
    panel["vwap"] = vwap
    med = float((vwap / panel["close"]).stack().median())
    if abs(med - 1.0) < 0.1:
        log.info("vwap 复权重建通过：vwap/close 中位数 %.4f", med)
    else:
        log.warning("vwap/close 中位数 %.4f 偏离 1 —— 复权因子未对齐，"
                    "vwap 不可信", med)
    return panel, begin, end


def _parse_range(s: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    a, b = s.split(":")
    return pd.Timestamp(a), pd.Timestamp(b)


# ---------------------------------------------------------------------------
# 报告层公共件：非 MCTS 臂的公式 → SearchRecord（同一评测/去重口径）
# ---------------------------------------------------------------------------

class ArmAssimilator:
    """把 GP/GFlowNet/one-shot 产出的公式按 MCTS 同一防线收编为记录。

    数值去重池为**臂级全局**（跨 Seed）；MCTS 臂引擎内是树级池（东吴口径），
    两侧的「唯一公式数」在汇总层用同一口径重数（见 :func:`unique_counts`）。
    """

    def __init__(self, evaluator: CandidateEvaluator, dedup_ic_corr: float,
                 struct_dedup: bool = True):
        from factor.rl.llm_pool import is_structural_duplicate

        self.ev = evaluator
        self.dedup_ic_corr = dedup_ic_corr
        self.struct_dedup = struct_dedup
        self._is_dup = is_structural_duplicate
        from stats.ic import calc_ic_series

        self._calc_ic = calc_ic_series
        self.pool_fp: dict[str, pd.DataFrame] = {}
        self.pool_ic: dict[str, pd.Series] = {}
        self.seen_struct: list[str] = []

    def add_seed(self, name: str, fp_full: pd.DataFrame) -> None:
        fp = fp_full.reindex(self.ev.panel_is["close"].index).replace(
            [np.inf, -np.inf], np.nan)
        self.pool_fp[name] = fp
        self.pool_ic[name] = self._calc_ic(
            fp.reindex(self.ev.grid_is),
            self.ev.ret_is.reindex(self.ev.grid_is), method="spearman").dropna()
        self.seen_struct.append(name)

    def assimilate(self, seed_name: str, formula_project: str,
                   iteration: int = 0, source: str = ""
                   ) -> SearchRecord:
        """评测一条项目语法公式（重复/失败记 rejected 台账，不抛异常）。"""
        from factor.gflownet.selection import panel_flat_corr
        from factor.rl.llm_pool import canonical

        rec = SearchRecord(
            seed_name=seed_name, formula_report=formula_project,
            formula_project=formula_project, hypothesis="",
            expected_direction="", relation_to_seed=source, status="accepted",
            iteration=iteration)

        def _reject(status: str, reason: str) -> SearchRecord:
            rec.status, rec.reject_reason = status, reason
            return rec

        if not formula_project:
            return _reject("rejected_check", "空/不可解析公式")
        try:
            outcome = self.ev.evaluate_formula(formula_project)
            fp_is = self.ev.factor_panel(formula_project, "is")
        except Exception as exc:
            return _reject("rejected_eval", str(exc)[:200])
        rec.outcome = outcome
        ic_is = self._calc_ic(
            fp_is.reindex(self.ev.grid_is),
            self.ev.ret_is.reindex(self.ev.grid_is),
            method="spearman").dropna()
        rec.ic_weekly_is = ic_is

        key = canonical(formula_project) or formula_project
        if key in {canonical(k) or k for k in self.pool_fp}:
            return _reject("rejected_seen", "canonical 重复")
        if self.struct_dedup:
            dup, dup_of = self._is_dup(formula_project, self.seen_struct)
            if dup:
                return _reject("rejected_struct_dup",
                               f"AST 结构同族（vs {dup_of[:60]}）")
        for name, ic0 in self.pool_ic.items():
            df = pd.concat([ic_is, ic0], axis=1).dropna()
            if len(df) >= 8:
                c = df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman")
                if math.isfinite(c) and c >= self.dedup_ic_corr:
                    return _reject("rejected_numeric_dup", f"vs {name}")

        mac = 0.0
        for fp0 in self.pool_fp.values():
            c = panel_flat_corr(fp_is, fp0, min_overlap=200)
            if math.isfinite(c):
                mac = max(mac, abs(c))
        rec.max_abs_corr = mac
        rec.reward_search = outcome.reward("is", max_abs_corr=mac)
        rec.reward_replay = outcome.reward("is", max_abs_corr=None)
        if not math.isfinite(rec.reward_search):
            return _reject("rejected_eval", "reward NaN")
        self.pool_fp[formula_project] = fp_is
        self.pool_ic[formula_project] = ic_is
        self.seen_struct.append(formula_project)
        return rec


# ---------------------------------------------------------------------------
# 四臂
# ---------------------------------------------------------------------------

def _seed_ctx(spec, seed_stats: dict, n: int) -> PromptContext:
    return PromptContext(seed_name=spec.name, seed_display=spec.display,
                         seed_note=spec.note, seed_stats=seed_stats, n_variants=n)


def run_arm_mcts(specs, seed_panels, evaluator, args, out_dir: Path) -> list[TreeReport]:
    proposer = _make_proposer(args)
    judge = None
    if args.logic_review and isinstance(proposer, MCTSLLMProposer):
        from factor.rl.llm_pool import llm_semantic_check

        def judge(h: str, f: str) -> bool:
            verdicts = llm_semantic_check([f], [h], ask=proposer.judge_ask)
            return bool(verdicts[0])          # None（审查故障）按未通过处理
    cfg = RunConfig(iterations=args.iterations, variants=args.variants,
                    max_depth=args.max_depth, exploration=args.exploration,
                    virtual_weight=args.virtual_weight,
                    dedup_ic_corr=args.dedup_ic_corr,
                    struct_dedup=not args.no_struct_dedup,
                    logic_review=args.logic_review and judge is not None,
                    fail_reset_rounds=args.fail_reset_rounds, seed=args.seed)
    reports = []
    for k, spec in enumerate(specs, 1):
        t0 = time.perf_counter()
        rep = run_mcts_tree(spec, seed_panels[spec.name], evaluator, proposer,
                            cfg, llm_judge=judge)
        acc_rewards = [r.reward_search for r in rep.records
                       if r.status == "accepted"]
        log.info("[mcts %d/%d] Seed %s：accepted=%d/%d best_reward=%.4f"
                 "（seed=%.4f） %.1fs llm_calls=%d",
                 k, len(specs), spec.name, len(acc_rewards), len(rep.records),
                 max(acc_rewards + [rep.seed_reward_replay]),
                 rep.seed_reward_replay, time.perf_counter() - t0,
                 rep.n_llm_calls)
        reports.append(rep)
    _dump_prompt_sample(proposer, out_dir)
    return reports


def run_arm_llm_oneshot(specs, seed_panels, evaluator, args,
                        out_dir: Path) -> list[TreeReport]:
    """one-shot 臂：每 Seed 一次性生成预算等量候选（无树、无路径反馈）。

    返回与 TreeReport 同构的轻量对象（汇总层无感）。
    """
    proposer = _make_proposer(args)
    assim = ArmAssimilator(evaluator, args.dedup_ic_corr,
                           not args.no_struct_dedup)
    budget = args.iterations * args.variants
    reports = []
    for k, spec in enumerate(specs, 1):
        assim.add_seed(spec.name, seed_panels[spec.name])
        seed_out = evaluator.evaluate_panel(seed_panels[spec.name])
        seed_out.complexity, seed_out.digit_count = display_complexity(
            spec.display)
        records = []
        got: list[str] = []
        # 分批（每批 ≤10 条，防单次输出超限；existing 传已收清单防重复）
        batch = 10
        while len(got) < budget:
            n = min(batch, budget - len(got))
            ctx = _seed_ctx(spec, {"rankic": seed_out.st_is.rankic_mean,
                                   "rankir": seed_out.st_is.rankir,
                                   "reward": 0.0}, n)
            ctx.avoid = got
            try:
                variants = proposer.propose_variants(ctx)
            except Exception as exc:
                log.warning("[oneshot %s] proposer 异常：%s", spec.name, exc)
                break
            if not variants:
                break
            for v in variants:
                if v.formula in got:
                    continue
                got.append(v.formula)
                rec = assim.assimilate(
                    spec.name, _to_project(v.formula),
                    source=f"oneshot:{v.hypothesis}")
                rec.hypothesis = v.hypothesis
                rec.expected_direction = v.expected_direction
                rec.relation_to_seed = v.relation_to_seed
                rec.formula_report = v.formula
                records.append(rec)
        reports.append(TreeReport(
            seed_name=spec.name, seed_outcome=seed_out,
            seed_reward_replay=seed_out.reward("is", max_abs_corr=None),
            records=records,
            n_llm_calls=getattr(proposer, "n_calls", 0),
            n_completion_tokens=getattr(proposer, "completion_tokens", 0)))
        log.info("[oneshot %d/%d] Seed %s：proposed=%d accepted=%d",
                 k, len(specs), spec.name, len(got),
                 sum(r.status == "accepted" for r in records))
    _dump_prompt_sample(proposer, out_dir)
    return reports


def run_arm_gp(specs, seed_panels, evaluator, args,
               out_dir: Path) -> list[TreeReport]:
    """GP 对照臂：项目 GP 引擎原生搜索 → 报告层统一收编。"""
    from factor.genetic_mining import run_gp_mining

    assim = ArmAssimilator(evaluator, args.dedup_ic_corr,
                           not args.no_struct_dedup)
    panel_is = evaluator.panel_is
    ret_is = evaluator.ret_is
    budget = args.iterations * args.variants * max(1, len(specs))
    pop = max(20, args.variants * 10)
    gen = max(2, budget // pop)
    log.info("[gp] pop=%d gen=%d（预算 ≈ %d 次求值，与 mcts 臂同量级）",
             pop, gen, pop * gen)
    t0 = time.perf_counter()
    results_df, _hof = run_gp_mining(
        panel_is, ret_is, features=list(FIELDS),
        windows=(5, 10, 20, 40, 60), population=pop, generations=gen,
        fitness_mode="rankic_mean", train_frac=1.0, seed=args.seed,
        verbose=False)
    log.info("[gp] 引擎完成 %.1fs，产出 %d 条", time.perf_counter() - t0,
             len(results_df))
    reports = []
    for k, spec in enumerate(specs, 1):
        assim.add_seed(spec.name, seed_panels[spec.name])
        seed_out = evaluator.evaluate_panel(seed_panels[spec.name])
        seed_out.complexity, seed_out.digit_count = display_complexity(
            spec.display)
        # GP 无 Seed 概念：全部产出挂到首个 Seed 名下做同表报告
        anchor = specs[0].name
        if spec.name != anchor:
            reports.append(TreeReport(
                seed_name=spec.name, seed_outcome=seed_out,
                seed_reward_replay=seed_out.reward("is", max_abs_corr=None),
                records=[]))
            continue
        records = [assim.assimilate(anchor, f, source="gp")
                   for f in results_df["formula"].tolist()[: args.iterations
                                                           * args.variants * 5]]
        reports.append(TreeReport(
            seed_name=anchor, seed_outcome=seed_out,
            seed_reward_replay=seed_out.reward("is", max_abs_corr=None),
            records=records))
    return reports


def run_arm_gflownet(specs, seed_panels, evaluator, args,
                     out_dir: Path) -> list[TreeReport]:
    """GFlowNet 对照臂：TB 训练 + 采样 → 报告层统一收编。"""
    from factor.gflownet.env import FactorMDP
    from factor.gflownet.reward import make_reward_fn
    from factor.gflownet.tb import TBPolicy, sample_formulas, train_tb
    from factor.rl.alphapool_env import REPORT_OPERATORS

    assim = ArmAssimilator(evaluator, args.dedup_ic_corr,
                           not args.no_struct_dedup)
    op_names = sorted(set(REPORT_OPERATORS.values()))
    mdp = FactorMDP(op_names, windows=(5, 10, 20, 60), features=list(FIELDS),
                    max_depth=3)
    reward_fn = make_reward_fn(
        evaluator.panel_is, evaluator.ret_is, features=list(FIELDS),
        horizon=args.horizon, long_ir_lambda=0.5, barra_mu=0.0)
    net = TBPolicy(mdp.n_actions)
    budget = args.iterations * args.variants * max(1, len(specs))
    t0 = time.perf_counter()
    train_tb(mdp, reward_fn, net, n_iters=max(30, budget), batch_size=12,
             seed=args.seed, log_every=max(50, budget // 4))
    samples = sample_formulas(net, mdp, reward_fn,
                              n=min(budget, args.iterations * args.variants * 5),
                              seed=args.seed)
    log.info("[gflownet] 训练+采样 %.1fs，产出 %d 条",
             time.perf_counter() - t0, len(samples))
    reports = []
    anchor = specs[0].name
    for k, spec in enumerate(specs, 1):
        assim.add_seed(spec.name, seed_panels[spec.name])
        seed_out = evaluator.evaluate_panel(seed_panels[spec.name])
        seed_out.complexity, seed_out.digit_count = display_complexity(
            spec.display)
        if spec.name != anchor:
            reports.append(TreeReport(
                seed_name=spec.name, seed_outcome=seed_out,
                seed_reward_replay=seed_out.reward("is", max_abs_corr=None),
                records=[]))
            continue
        records = [assim.assimilate(anchor, f, source="gflownet")
                   for f, _r in samples]
        reports.append(TreeReport(
            seed_name=anchor, seed_outcome=seed_out,
            seed_reward_replay=seed_out.reward("is", max_abs_corr=None),
            records=records))
    return reports



def _make_proposer(args) -> MCTSLLMProposer | SeedMutationProposer:
    if args.llm == "openai":
        return MCTSLLMProposer(model=args.llm_model,
                               max_tokens=args.llm_max_tokens,
                               timeout=args.llm_timeout)
    return SeedMutationProposer(seed=args.seed)


def _to_project(report_formula: str) -> str:
    from factor.rl.llm_pool import check_report_formula

    chk = check_report_formula(report_formula)
    return chk.project or ""


def _dump_prompt_sample(proposer, out_dir: Path) -> None:
    try:
        if isinstance(proposer, MCTSLLMProposer) and proposer.last_raw:
            (out_dir / "llm_reply_sample.txt").write_text(
                proposer.last_raw[-1][:8000], encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 汇总层：双口径验收 + 去重计数 + 相关分布 + PBO/DSR
# ---------------------------------------------------------------------------

def _oos_reward(rep: TreeReport) -> float:
    return rep.seed_outcome.reward("oos", max_abs_corr=None)


def summarize(arm: str, reports: list[TreeReport], out_dir: Path,
              args) -> dict:
    rows = [r.row() for rep in reports for r in rep.records]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if len(df):
        df.to_csv(out_dir / "candidates.csv", index=False,
                  encoding="utf-8-sig")
    seed_rows = [{
        "seed": rep.seed_name,
        "is_rankic": rep.seed_outcome.st_is.rankic_mean,
        "is_rankir": rep.seed_outcome.st_is.rankir,
        "oos_rankic": rep.seed_outcome.st_oos.rankic_mean,
        "oos_rankir": rep.seed_outcome.st_oos.rankir,
        "is_reward": rep.seed_reward_replay,
        "oos_reward": _oos_reward(rep),
        "n_records": len(rep.records),
        "n_accepted": sum(r.status == "accepted" for r in rep.records),
        "llm_calls": rep.n_llm_calls,
        "wall_s": round(rep.wall_seconds, 1),
    } for rep in reports]
    pd.DataFrame(seed_rows).to_csv(out_dir / "seeds.csv", index=False,
                                   encoding="utf-8-sig")

    # -- 双口径验收（自拟判据：OOS 六项 reward 复评口径，见模块 docstring）--
    formal_hit, formal_total = 0, 0
    pool_hit, pool_total = 0, 0
    for rep in reports:
        acc = [r for r in rep.records if r.status == "accepted"
               and math.isfinite(r.reward_replay)]
        better_is = [r for r in acc if r.reward_replay > rep.seed_reward_replay]
        if not acc:
            continue
        # 正式选择：IS 复评 reward 最优候选（须优于 Seed，否则该 Seed 无入选）
        if better_is:
            best = max(better_is, key=lambda r: r.reward_replay)
            formal_total += 1
            if (best.outcome.reward("oos", max_abs_corr=None)
                    > _oos_reward(rep)):
                formal_hit += 1
        # 候选池诊断：IS 优于 Seed 的候选中 OOS 仍优于 Seed 的占比
        for r in better_is:
            pool_total += 1
            if r.outcome.reward("oos", max_abs_corr=None) > _oos_reward(rep):
                pool_hit += 1

    summary = {
        "arm": arm, "llm": args.llm, "panel": args.panel,
        "is": args.is_, "oos": args.oos, "horizon": args.horizon,
        "iterations": args.iterations, "variants": args.variants,
        "max_depth": args.max_depth, "seeds_n": len(reports),
        "n_candidates_total": int(len(df)) if len(df) else 0,
        "n_accepted": int((df["status"] == "accepted").sum()) if len(df) else 0,
        "status_counts": (df["status"].value_counts().to_dict()
                          if len(df) else {}),
        "formal_selection_oos_hit": f"{formal_hit}/{formal_total}",
        "formal_selection_oos_ratio": round(formal_hit / formal_total, 4)
        if formal_total else None,
        "formal_ratio_dongwu_benchmark": 0.655,
        "pool_diagnosis_hit": f"{pool_hit}/{pool_total}",
        "pool_diagnosis_ratio": round(pool_hit / pool_total, 4)
        if pool_total else None,
        "pool_ratio_dongwu_benchmark": 0.754,
        "llm_calls": int(sum(r.n_llm_calls for r in reports)),
        "llm_completion_tokens": int(sum(r.n_completion_tokens for r in reports)),
        "wall_seconds": round(sum(r.wall_seconds for r in reports), 1),
    }

    # -- 唯一公式计数（汇总层同口径重数：canonical + 结构 + 数值）----------
    acc = [r for rep in reports for r in rep.records if r.status == "accepted"]
    summary["unique_formulas_canonical"] = len(
        {r.formula_project for r in acc})
    try:
        from factor.rl.llm_pool import is_structural_duplicate

        uniq: list[str] = []
        for r in acc:
            if not is_structural_duplicate(r.formula_report, uniq)[0]:
                uniq.append(r.formula_report)
        summary["unique_formulas_structural"] = len(uniq)
    except Exception:
        pass

    # -- 两两相关性分布（东吴 §4：|corr| 均值/中位/p90，≥0.8 占比）----------
    corrs = _pairwise_corr(acc)
    if corrs:
        arr = np.array(corrs)
        summary["pairwise_abs_corr"] = {
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
            "frac_ge_0.8": round(float((arr >= 0.8).mean()), 4),
            "n_pairs": int(len(arr)),
        }

    # -- PBO / DSR（IC 矩阵口径：T×N 周度 IC，metric=mean/sharpe）----------
    summary.update(_pbo_block(acc))

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(summary, out_dir)
    return summary


def _pairwise_corr(records: list[SearchRecord]) -> list[float]:
    """两两 |Spearman(周度 IC)|（东吴 §4 同质化诊断，IC 序列口径）。"""
    ics = [(r.formula_project, r.ic_weekly_is) for r in records
           if r.ic_weekly_is is not None and len(r.ic_weekly_is) >= 8]
    out: list[float] = []
    for i in range(len(ics)):
        for j in range(i + 1, len(ics)):
            df = pd.concat([ics[i][1], ics[j][1]], axis=1).dropna()
            if len(df) < 8:
                continue
            c = df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman")
            if math.isfinite(c):
                out.append(abs(c))
    return out


def _pbo_block(records: list[SearchRecord]) -> dict:
    out: dict = {}
    ok = [r for r in records if r.ic_weekly_is is not None
          and len(r.ic_weekly_is.dropna()) >= 32]
    if len(ok) < 2:
        return {"pbo": "样本不足（<2 条有效 IC 或 <32 周）"}
    mat = pd.concat({i: r.ic_weekly_is for i, r in enumerate(ok)}, axis=1)
    mat = mat.dropna(how="any")
    if len(mat) < 8 or mat.shape[1] < 2:
        return {"pbo": "对齐后样本不足"}
    try:
        from stats.pbo import cscv_pbo, deflate_best

        n_parts = max(2, min(16, len(mat) // 2))
        pbo = cscv_pbo(mat.to_numpy(), n_partitions=n_parts, metric="mean")
        dsr = deflate_best(mat.to_numpy(), metric="sharpe")
        out["pbo"] = round(float(pbo["pbo"]), 4)
        out["pbo_logit_median"] = round(float(pbo["logit_median"]), 4)
        out["deflate_best_dsr"] = round(float(dsr["dsr"]), 4)
        out["deflate_best_column"] = str(dsr["best_column"])
    except Exception as exc:
        out["pbo"] = f"计算失败：{exc}"
    return out


def _write_md(summary: dict, out_dir: Path) -> None:
    lines = [
        "# 东吴 LLM-MCTS Phase 0 — " + summary["arm"] + " 臂",
        "",
        f"- 面板：{summary['panel']}，IS {summary['is']} / OOS {summary['oos']}"
        f"，h={summary['horizon']}，seeds={summary['seeds_n']}",
        f"- 预算：iterations={summary['iterations']} × "
        f"variants={summary['variants']}（max_depth={summary['max_depth']}）",
        f"- 候选：{summary['n_candidates_total']} 条 / 收编 "
        f"{summary['n_accepted']} 条；状态分布 {summary['status_counts']}",
        "",
        "## 双口径验收（对标东吴 65.5% / 75.4%；判据=OOS 六项 reward 复评口径）",
        f"- 正式选择 OOS 兑现：**{summary['formal_selection_oos_hit']}**"
        f"（{summary['formal_selection_oos_ratio']}；东吴 0.655）",
        f"- 候选池诊断率：**{summary['pool_diagnosis_hit']}**"
        f"（{summary['pool_diagnosis_ratio']}；东吴 0.754）",
        "",
        "## 去重与同质化",
        f"- canonical 唯一：{summary.get('unique_formulas_canonical')}"
        f"｜结构唯一：{summary.get('unique_formulas_structural')}",
    ]
    if "pairwise_abs_corr" in summary:
        pc = summary["pairwise_abs_corr"]
        lines.append(f"- 两两 |corr|：均值 {pc['mean']} / 中位 {pc['median']}"
                     f" / p90 {pc['p90']} / ≥0.8 占比 {pc['frac_ge_0.8']}"
                     f"（{pc['n_pairs']} 对）")
    lines += [
        "",
        "## 过拟合检验",
        f"- PBO：{summary.get('pbo')}｜DSR：{summary.get('deflate_best_dsr')}",
        "",
        "## 成本",
        f"- LLM 调用 {summary['llm_calls']} 次 / "
        f"{summary['llm_completion_tokens']} completion tokens｜"
        f"耗时 {summary['wall_seconds']}s",
        "",
        "> 复现边界：h=5 对齐东吴；判据与去重口径自拟（详见脚本 docstring 与"
        "设计笔记 §五）；指标绝对值不可跨报告比，只比同池四臂。",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", default="hs300_2015_2026",
                    help=f"面板预设 {sorted(PANEL_PRESETS)} 或 mock（合成面板）")
    ap.add_argument("--is", dest="is_", default=DEFAULT_IS,
                    help="IS 窗口 YYYY-MM-DD:YYYY-MM-DD（默认 2016-2023）")
    ap.add_argument("--oos", default=DEFAULT_OOS,
                    help="OOS 窗口（默认 2024-01-01:2026-08-21）")
    ap.add_argument("--horizon", type=int, default=5,
                    help="前瞻收益期限（默认 5，对齐东吴非生产口径）")
    ap.add_argument("--arm", default="mcts",
                    choices=["mcts", "llm_oneshot", "gp", "gflownet"])
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--variants", type=int, default=5)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--seeds-n", type=int, default=3,
                    help="取前 N 个 Seed（东吴全量 29，Phase 0 试点限流）")
    ap.add_argument("--exploration", type=float, default=1.0)
    ap.add_argument("--virtual-weight", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", default="template", choices=["template", "openai"],
                    help="扩展器：template=离线变异兜底（不触网）/ openai=DeepSeek")
    ap.add_argument("--llm-model", default="deepseek-flash")
    ap.add_argument("--llm-max-tokens", type=int, default=16000,
                    help="含思维链（推理模型陷阱，见 llm_pool 文档）")
    ap.add_argument("--llm-timeout", type=float, default=300.0)
    ap.add_argument("--logic-review", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="东吴审查层；--llm openai 时默认开，template 强制关")
    ap.add_argument("--dedup-ic-corr", type=float, default=0.99,
                    help="RD-Agent 机制①：周度 IC 相关去重阈值")
    ap.add_argument("--no-struct-dedup", action="store_true")
    ap.add_argument("--fail-reset-rounds", type=int, default=3,
                    help="RD-Agent 机制④：连续 N 轮无新最优触发换向")
    ap.add_argument("--offline", action=argparse.BooleanOptionalAction,
                    default=True, help="数据面离线缓存模式（默认开）")
    ap.add_argument("--out", default="reports/llm_mcts_phase0")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟档：1 Seed × 3 iters × 3 variants，mock 面板")
    return ap


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.panel = "mock"
        args.seeds_n, args.iterations, args.variants = 1, 3, 3
        if args.llm == "openai":
            log.info("--smoke 保留 --llm openai（真实往返冒烟）")
    if args.arm == "gp":
        args.llm = "template"                     # GP 臂不走 LLM
    if args.logic_review is None:
        args.logic_review = (args.llm == "openai"
                             and args.arm in ("mcts", "llm_oneshot"))
    if args.llm == "template":
        args.logic_review = False                 # 离线兜底无审查通道

    ts = time.strftime("%m%d_%H%M%S")
    run_name = f"{args.arm}_{args.llm}_{ts}" + ("_smoke" if args.smoke else "")
    out_dir = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== 东吴 LLM-MCTS Phase 0：arm=%s llm=%s panel=%s h=%d ===",
             args.arm, args.llm, args.panel, args.horizon)
    panel, _, _ = load_panel(args.panel, offline=args.offline)
    is_a, is_b = _parse_range(args.is_)
    oos_a, oos_b = _parse_range(args.oos)
    close = panel["close"]
    is_idx = close.index[(close.index >= is_a) & (close.index <= is_b)]
    oos_idx = close.index[(close.index >= oos_a) & (close.index <= oos_b)]
    if len(is_idx) < 120 or len(oos_idx) < 60:
        raise SystemExit(f"窗口过短：IS {len(is_idx)} 日 / OOS {len(oos_idx)} 日")

    log.info("计算 29 Seed 面板（%s × w=%d）...", len(SEEDS), SEED_WINDOW)
    seed_panels = compute_seed_panels(panel)
    specs = SEEDS[: args.seeds_n]

    evaluator = CandidateEvaluator(panel, is_idx, oos_idx, horizon=args.horizon)

    runner = {"mcts": run_arm_mcts, "llm_oneshot": run_arm_llm_oneshot,
              "gp": run_arm_gp, "gflownet": run_arm_gflownet}[args.arm]
    reports = runner(specs, seed_panels, evaluator, args, out_dir)

    summary = summarize(args.arm, reports, out_dir, args)
    log.info("汇总落盘 %s；正式选择 %s（对标 0.655），候选池 %s（对标 0.754）",
             out_dir, summary["formal_selection_oos_hit"],
             summary["pool_diagnosis_hit"])


if __name__ == "__main__":
    main()
