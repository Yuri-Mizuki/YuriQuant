"""周度六项 reward（东吴 LLM-MCTS 口径）。

研报公开公式（IS 搜索期）::

    mcts_reward = 0.40·effectiveness + 0.30·stability + 0.02·coverage
                + 0.10·turnover + 0.10·diversity + 0.08·overfit
    effectiveness = sigmoid((absRankIC − 0.05)/0.015)      # 周度截面 RankIC 绝对值
    stability      = sigmoid((absRankIR − 0.60)/0.25)      # RankIR = RankIC 均值/标准差
    coverage       = clip(coverage, 0, 1)                  # 非空覆盖率时间均值
    turnover       = sigmoid(−(turnover − 0.8)/0.4)        # 前10%组合平均换手，越低越好
    diversity      = 1 − max_abs_corr                      # 与已有有效候选的最大绝对相关
    overfit        = sigmoid(−(complexity + 0.15·digit_count − 14)/5)

复现边界（引用数字必须注明）：

- **周度 RankIC 的采样口径自拟**（研报未给细节）：取每周最后一个交易日的
  截面 RankIC（对 h 日前瞻收益），RankIR = 周度 IC 序列的均值/标准差
  （**不年化**——研报数值域 0.60 对应不年化的周度 IR）。
- **换手口径自拟**：前 10% 等权组合在周度调仓网格上的"成分更换率"均值
  （1 − |本期∩上期|/|本期|，单边、不年化——研报 sigmoid 中心 0.8 与该
  口径的数值域匹配）。
- **complexity = AST 节点数、digit_count = 常数（含窗口）数字字符数**，
  按公式 token 语义近似；研报的 overfit 断点 (−14)/5 是为其更深的树设计，
  本项目变体空间 ≤15 token 时该项数值域偏平（~0.45–0.55），如实保留常数。
- diversity 在搜索期压低强相关候选；**复评期（Stage2）放开**（东吴口径，
  避免把好候选压死在搜索期）——:func:`reward_components` 的
  ``max_abs_corr`` 传 ``None`` 即跳过该项（按满分 1.0 计）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from factor.rl.alphapool_env import FIELDS
from stats.ic import calc_ic_series

__all__ = [
    "CandidateEvaluator", "EvalOutcome", "WeeklyStats", "ast_complexity",
    "reward_components", "six_part_reward", "top_decile_turnover",
    "weekly_grid", "weekly_stats",
]

#: 六项权重（东吴/论文一致，全部公开）。
REWARD_WEIGHTS: dict[str, float] = {
    "effectiveness": 0.40, "stability": 0.30, "coverage": 0.02,
    "turnover": 0.10, "diversity": 0.10, "overfit": 0.08,
}

#: 前 10% 组合口径（东吴：前 10% 组合平均换手）。
TOP_FRAC: float = 0.10


def _sigmoid(x: float) -> float:
    if not math.isfinite(x):
        return 0.0 if x < 0 else 1.0
    return 1.0 / (1.0 + math.exp(-x))


def weekly_grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每周最后一个交易日（ISO 周分组，与 research.factor_report 周度口径同族）。"""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("W")).last())


@dataclass
class WeeklyStats:
    """一个因子面板在某窗口上的周度统计（六项 reward 的输入件）。"""

    rankic_mean: float = float("nan")
    rankic_std: float = float("nan")
    rankir: float = float("nan")
    coverage: float = float("nan")
    turnover: float = float("nan")
    n_weeks: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return {"rankic_mean": self.rankic_mean, "rankic_std": self.rankic_std,
                "rankir": self.rankir, "coverage": self.coverage,
                "turnover": self.turnover, "n_weeks": self.n_weeks}


def top_decile_turnover(fp_w: pd.DataFrame, top_frac: float = TOP_FRAC) -> float:
    """前 10% 等权组合在周度网格上的平均成分更换率。

    ``turnover_t = 1 − |S_t ∩ S_{t-1}| / |S_t|``（单边、不年化）。
    首周不计；样本 <2 周返回 NaN。
    """
    if len(fp_w) < 2:
        return float("nan")
    valid = fp_w.notna()
    ranks = fp_w.where(valid).rank(axis=1, ascending=False)
    books: list[frozenset[str]] = []
    for dt in fp_w.index:
        n_valid = int(valid.loc[dt].sum())
        k = max(1, int(math.ceil(n_valid * top_frac)))
        row = ranks.loc[dt]
        cols = row[row <= k].index
        books.append(frozenset(cols))
    turns = [1.0 - len(b & books[i - 1]) / max(len(b), 1)
             for i, b in enumerate(books) if i >= 1 and len(b)]
    return float(np.mean(turns)) if turns else float("nan")


def weekly_stats(fp: pd.DataFrame, fwd_returns: pd.DataFrame,
                 grid: pd.DatetimeIndex) -> WeeklyStats:
    """周度截面统计：RankIC/IR + 覆盖率 + 前10%换手。

    Args:
        fp: 日频因子面板（取周度网格切片后求 IC）。
        fwd_returns: h 日前瞻收益面板（与 fp 同频，尾部 h 行无标签自然为 NaN）。
        grid: 周度网格（由窗口收盘价索引导出）。
    """
    fp_w = fp.reindex(grid)
    ret_w = fwd_returns.reindex(grid)
    valid = fp_w.notna() & ret_w.notna()
    ic = calc_ic_series(fp_w, ret_w, method="spearman").dropna()
    if len(ic) < 2 or ic.std() == 0:
        return WeeklyStats(n_weeks=int(len(ic)))
    cov = float(valid.sum(axis=1).mean() / max(fp_w.shape[1], 1))
    return WeeklyStats(
        rankic_mean=float(ic.mean()), rankic_std=float(ic.std()),
        rankir=float(ic.mean() / ic.std()), coverage=cov,
        turnover=top_decile_turnover(fp_w), n_weeks=int(len(ic)),
    )


def ast_complexity(project_formula: str) -> tuple[int, int]:
    """项目语法 AST 的 (节点数, 常数数字字符数)。

    节点数含算子/字段/常数；digit_count 统计所有数值常量（含窗口后缀）的
    数字字符（"20"→2 位、"0.5"→2 位）。
    """
    from factor.formula import parse_formula

    node = parse_formula(project_formula, features=list(FIELDS))
    n_nodes = 0
    digits = 0

    def _walk(n) -> None:
        nonlocal n_nodes, digits
        kind = n[0]
        if kind == "const":
            digits += sum(ch.isdigit() for ch in str(abs(n[1])))
            n_nodes += 1
            return
        if kind == "feat":
            n_nodes += 1
            return
        n_nodes += 1
        if n[3]:  # GP 风格窗口后缀（win_name 元组）
            digits += sum(sum(ch.isdigit() for ch in str(w)) for w in n[3])
        for c in n[2]:
            _walk(c)

    _walk(node)
    return n_nodes, digits


def reward_components(st: WeeklyStats, complexity: int, digit_count: int,
                      max_abs_corr: float | None) -> dict[str, float]:
    """六项 sigmoid 分量（供报告透明化；NaN 输入对应分量为 NaN）。"""
    ic = abs(st.rankic_mean) if math.isfinite(st.rankic_mean) else float("nan")
    ir = abs(st.rankir) if math.isfinite(st.rankir) else float("nan")
    turn = st.turnover if math.isfinite(st.turnover) else float("nan")
    eff = _sigmoid((ic - 0.05) / 0.015) if math.isfinite(ic) else float("nan")
    sta = _sigmoid((ir - 0.60) / 0.25) if math.isfinite(ir) else float("nan")
    cov = float(np.clip(st.coverage if math.isfinite(st.coverage) else 0.0, 0.0, 1.0))
    tov = _sigmoid(-(turn - 0.8) / 0.4) if math.isfinite(turn) else float("nan")
    # 复评期放开（东吴口径）：max_abs_corr=None → diversity 记满分 1.0
    div = 1.0 if max_abs_corr is None else max(0.0, 1.0 - max_abs_corr)
    over = _sigmoid(-(complexity + 0.15 * digit_count - 14) / 5)
    return {"effectiveness": eff, "stability": sta, "coverage": cov,
            "turnover": tov, "diversity": div, "overfit": over}


def six_part_reward(st: WeeklyStats, complexity: int, digit_count: int,
                    max_abs_corr: float | None) -> float:
    """六项加权总分；任一必需分量为 NaN 时返回 NaN（调用方按"无法评测"处理）。"""
    comp = reward_components(st, complexity, digit_count, max_abs_corr)
    total = 0.0
    for key, w in REWARD_WEIGHTS.items():
        v = comp[key]
        if not math.isfinite(v):
            return float("nan")
        total += w * v
    return total


@dataclass
class EvalOutcome:
    """一次候选评测的完整结果（IS/OOS 双窗口）。"""

    formula_project: str
    st_is: WeeklyStats = field(default_factory=WeeklyStats)
    st_oos: WeeklyStats = field(default_factory=WeeklyStats)
    complexity: int = 0
    digit_count: int = 0
    # 多样性在引擎层算（依赖候选池），这里留槽位由引擎回填
    max_abs_corr: float | None = None

    def reward(self, which: str = "is", max_abs_corr: float | None = None) -> float:
        st = self.st_is if which == "is" else self.st_oos
        corr = max_abs_corr if max_abs_corr is not None else self.max_abs_corr
        return six_part_reward(st, self.complexity, self.digit_count, corr)


class CandidateEvaluator:
    """候选公式 → IS/OOS 周度统计（防前视：先切窗再算前瞻收益）。

    前瞻收益在**切片后**的收盘价上构造（``pct_change(h).shift(-h)``）——
    切片尾部 h 日无标签自然丢弃，IS 尾部标签不会吃到 OOS 价格。
    """

    def __init__(self, panel: dict[str, pd.DataFrame], is_idx: pd.DatetimeIndex,
                 oos_idx: pd.DatetimeIndex, horizon: int = 5,
                 features: tuple[str, ...] = FIELDS):
        self.horizon = int(horizon)
        self.features = list(features)
        # ⚠️ formula 的子树缓存键只含公式文本不含面板——IS/OOS 必须各持一份，
        # 共用会把 IS 面板错配给 OOS 求值（实测全 NaN 的根因）。
        self._nc_is: dict = {}
        self._nc_oos: dict = {}
        self.panel_is = {k: v.reindex(is_idx) for k, v in panel.items()
                         if k in self.features}
        self.panel_oos = {k: v.reindex(oos_idx) for k, v in panel.items()
                          if k in self.features}
        self.ret_is = self.panel_is["close"].pct_change(
            self.horizon, fill_method=None).shift(-self.horizon)
        self.ret_oos = self.panel_oos["close"].pct_change(
            self.horizon, fill_method=None).shift(-self.horizon)
        self.grid_is = weekly_grid(is_idx)
        self.grid_oos = weekly_grid(oos_idx)

    def factor_panel(self, project_formula: str, which: str = "is") -> pd.DataFrame:
        """项目语法公式 → 因子面板（按窗口缓存；±inf→NaN 清理）。"""
        from factor.formula import formula_builder

        panel = self.panel_is if which == "is" else self.panel_oos
        cache = self._nc_is if which == "is" else self._nc_oos
        fp = formula_builder(project_formula, features=self.features,
                             node_cache=cache)(panel)
        return fp.replace([np.inf, -np.inf], np.nan)

    def _fwd_stats(self, fp: pd.DataFrame, which: str) -> WeeklyStats:
        ret = self.ret_is if which == "is" else self.ret_oos
        grid = self.grid_is if which == "is" else self.grid_oos
        return weekly_stats(fp, ret, grid)

    def evaluate_panel(self, fp_full: pd.DataFrame) -> EvalOutcome:
        """对面板已就绪的因子（如 Seed 原生面板）切窗评测。"""
        fp = fp_full.replace([np.inf, -np.inf], np.nan)
        return EvalOutcome(
            formula_project="",
            st_is=self._fwd_stats(fp.reindex(self.panel_is["close"].index), "is"),
            st_oos=self._fwd_stats(fp.reindex(self.panel_oos["close"].index), "oos"),
        )

    def evaluate_formula(self, project_formula: str) -> EvalOutcome:
        """对项目前缀语法公式求值并评测（失败抛异常，由引擎捕获记 rejected）。"""
        complexity, digits = ast_complexity(project_formula)
        # IS/OOS 各自求值（切片头部滚动窗口自然 NaN，与 Seed 面板同口径）
        fp_is = self.factor_panel(project_formula, "is")
        fp_oos = self.factor_panel(project_formula, "oos")
        return EvalOutcome(
            formula_project=project_formula,
            st_is=self._fwd_stats(fp_is, "is"),
            st_oos=self._fwd_stats(fp_oos, "oos"),
            complexity=complexity, digit_count=digits,
        )
