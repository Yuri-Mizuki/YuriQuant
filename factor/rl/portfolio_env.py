"""stage2 组合优化 RL 环境（Phase 0：确定性骨架，银河 0706 口径）
================================================================

设计定稿依据：``reports/docs/research_notes/银河0706_华安226_研读_
RL组合优化全景与stage2设计定稿.md``（主蓝图=银河 0706，T2RL SAC 为可选第四臂）。

**主动权重空间**：状态显式含基准权重/主动偏离/换手/跟踪误差，动作经
``active_share`` 与基准混合——RL 臂与 QP 臂（``optimize.solver.solve_portfolio``）
用同一套 ``(w − wb)`` 口径对照（同时解决 AI39 口径核对发现的绝对/主动空间问题）。

三块确定性核心（均有独立单测，不依赖 gym）：

- :func:`map_actions_to_weights` —— 动作打分 → clip → 温度 softmax → 与基准按
  主动比例混合 → 持仓数上限（银河 3.1 节 2）；
- :func:`reward_components` —— 奖励六项分解：实际超额 / 信号排名暴露 / 风险暴露
  惩罚 / 门槛式 TE 惩罚 / 换手惩罚 / 可选集中度（银河 3.1 节 3，λ 缺省=研报表 7）；
- :func:`banach_projected_value` —— Banach 不动点自融资投影 ``V* = V−/(1+c·TO)``；
  **训练阶段 cost_rate=0**（防训练/回测双重扣费，回测统一扣真实成本，银河 2.5 节）。

:class:`PortfolioEnv` 把三块拼成 gymnasium Env：逐"决策日"推进，状态向量 =
银河表 8 清单（8N + 14 维），动作空间 ``Box(-5, 5, (N,))``。

**数据对齐约定（引用前先读）**：
- ``factor`` / ``aux`` / ``bench`` 面板以**决策日** t 为索引（t 日收盘可见的信号
  与基准权重）；
- ``period_returns`` 以**期间末端日**为索引：``period_returns.loc[t_next]`` 是
  ``(t, t_next]`` 区间内逐票收益——即"在 t 选的权重、吃到 t_next 实现的收益"；
- ``idx_period_returns`` 同口径的指数区间收益（超额与牛熊特征用）。

**复现边界（Phase 0）**：无涨跌停/停牌可成交掩码（银河未在 env 层处理一字板，
项目侧后续经 tradable_mask 注入）；无行业/风格硬约束（银河把这些全放进奖励软惩罚
——硬约束是 QP 臂的职责，对照实验恰好互补）。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# 银河表 7 的缺省参数（主蓝图口径；对照实验须披露池子差异后的重标定）
# ---------------------------------------------------------------------------
DEFAULT_ACTION_KW: dict[str, float] = {
    "active_share": 0.7,       # ρ：主动权重上限（与基准混合比例）
    "softmax_temp": 0.5,       # 动作 softmax 温度
    "clip_range": 5.0,         # 动作裁剪区间 [−5, 5]
    "max_holding_pct": 0.4,    # 单期最大持仓股数占成分股比例
}
DEFAULT_REWARD_KW: dict[str, float] = {
    "lambda_excess": 2.0,      # 实际超额收益主奖励
    "lambda_factor": 1.0,      # 信号排名暴露奖励
    "lambda_aux": 2.0,         # 高风险暴露惩罚（单侧）
    "lambda_tracking": 0.01,   # 门槛式 TE 惩罚
    "target_te": 0.15,         # 目标跟踪误差上限
    "lambda_turnover": 0.01,   # 换手惩罚
    "lambda_conc": 0.0,        # 集中度惩罚（研报默认不约束）
    "conc_top": 5,             # 集中度口径：前 5 大重仓
    "conc_target": 0.5,        # 集中度阈值
}
BANACH_MAX_ITER = 50          # 银河表 7：防死循环
BANACH_TOL = 1e-12


def rank_pct(row: np.ndarray | pd.Series) -> np.ndarray:
    """截面百分位排名（NaN 记 0.5 中性，不参与方向偏置）。"""
    s = pd.Series(np.asarray(row, dtype=float)).fillna(0.5)
    # NaN 先填 0.5 会被 rank 误当作真值——先 rank 再把原 NaN 位置置 0.5
    r = s.rank(pct=True).to_numpy(dtype=float)
    r[np.isnan(np.asarray(row, dtype=float))] = 0.5
    return r


# ---------------------------------------------------------------------------
# 1) 动作 → 可执行目标权重（银河 3.1 节 2）
# ---------------------------------------------------------------------------
def map_actions_to_weights(
    actions: np.ndarray,
    bench: np.ndarray,
    *,
    active_share: float = DEFAULT_ACTION_KW["active_share"],
    softmax_temp: float = DEFAULT_ACTION_KW["softmax_temp"],
    clip_range: float = DEFAULT_ACTION_KW["clip_range"],
    max_holding_pct: float = DEFAULT_ACTION_KW["max_holding_pct"],
) -> np.ndarray:
    """连续动作打分 → 目标权重（非负、和为 1、持仓数受上限约束）。

    流程：clip[-R,R] → 温度 softmax（数值稳定）→ ``w_raw = (1−ρ)·wb + ρ·w_act``
    → 归一化 → 保留权重最大的前 ``max_holding_pct·N`` 只后重归一。

    ``active_share=0`` 退化为基准（被动）；=1 为纯主动 softmax 权重。
    bench 含 NaN/负值按 0 处理后归一化（基准面板缺失票不占预算）。
    """
    a = np.clip(np.asarray(actions, dtype=float), -clip_range, clip_range)
    scaled = a / max(softmax_temp, 1e-9)
    scaled = scaled - scaled.max()                     # 数值稳定
    e = np.exp(scaled)
    w_act = e / e.sum()
    b = np.asarray(bench, dtype=float)
    b = np.where(np.isfinite(b), b, 0.0)
    b = np.clip(b, 0.0, None)
    if b.sum() <= 0:
        b = np.full(len(a), 1.0 / len(a))
    else:
        b = b / b.sum()
    w_raw = (1.0 - active_share) * b + active_share * w_act
    w = w_raw / w_raw.sum()
    # 持仓数上限：保留权重最大的 top_k，其余清零后重归一（银河 3.1 节 2）
    k = max(1, int(round(len(w) * float(max_holding_pct))))
    keep = np.argsort(w)[::-1][:k]
    out = np.zeros_like(w)
    out[keep] = w[keep]
    return out / out.sum()


# ---------------------------------------------------------------------------
# 2) 奖励六项分解（银河 3.1 节 3）
# ---------------------------------------------------------------------------
def reward_components(
    w: np.ndarray,
    w_prev: np.ndarray,
    bench: np.ndarray,
    period_ret: np.ndarray,
    f_row: np.ndarray,
    z_row: np.ndarray,
    *,
    lambda_excess: float = DEFAULT_REWARD_KW["lambda_excess"],
    lambda_factor: float = DEFAULT_REWARD_KW["lambda_factor"],
    lambda_aux: float = DEFAULT_REWARD_KW["lambda_aux"],
    lambda_tracking: float = DEFAULT_REWARD_KW["lambda_tracking"],
    target_te: float = DEFAULT_REWARD_KW["target_te"],
    lambda_turnover: float = DEFAULT_REWARD_KW["lambda_turnover"],
    lambda_conc: float = DEFAULT_REWARD_KW["lambda_conc"],
    conc_top: int = 5,
    conc_target: float = 0.5,
    eps: float = 1e-12,
) -> dict[str, float]:
    """奖励六项分解（z 方向约定：**值越大风险越低**）。

    返回 dict：``er``（实际超额）``fr``（信号排名暴露）``risk_penalty``/
    ``te_penalty``/``turnover_penalty``/``conc_penalty``（三项 ≤0 惩罚）
    ``total``（λ 加权汇总）``te``/``aux_exposure``/``turnover``（诊断量）。
    """
    d = w - bench
    er = float(w @ period_ret) - float(bench @ period_ret)
    fr = float(d @ rank_pct(f_row))
    aux_exp = float(d @ z_row)
    risk_penalty = -lambda_aux * max(0.0, -aux_exp)
    te = float(np.linalg.norm(d))
    te_penalty = -lambda_tracking * max(0.0, te - target_te) ** 2 / (target_te ** 2 + eps)
    turnover = float(np.abs(w - w_prev).sum())
    turnover_penalty = -lambda_turnover * turnover
    conc_penalty = 0.0
    if lambda_conc > 0:
        top_sum = float(np.sort(w)[::-1][:conc_top].sum())
        conc_penalty = -lambda_conc * max(0.0, top_sum - conc_target) ** 2
    total = (lambda_excess * er + lambda_factor * fr
             + risk_penalty + te_penalty + turnover_penalty + conc_penalty)
    return {
        "er": er, "fr": fr,
        "risk_penalty": risk_penalty, "te_penalty": te_penalty,
        "turnover_penalty": turnover_penalty, "conc_penalty": conc_penalty,
        "total": float(total),
        "te": te, "aux_exposure": aux_exp, "turnover": turnover,
    }


# ---------------------------------------------------------------------------
# 3) Banach 自融资投影（银河 2.5 节）
# ---------------------------------------------------------------------------
def banach_projected_value(
    v_prev: float,
    turnover: float,
    cost_rate: float,
    *,
    max_iter: int = BANACH_MAX_ITER,
    tol: float = BANACH_TOL,
) -> float:
    """调仓后组合价值：``V* = V−/(1 + c·TO)``（不动点迭代形式，防分母 0）。

    ``cost_rate=0``（训练阶段）退化为纯自融资投影 ``V* = V−``。
    """
    v = float(v_prev)
    v_minus = float(v_prev)
    to = float(turnover)
    for _ in range(max_iter):
        v_next = v_minus - cost_rate * v * to
        if abs(v_next - v) < tol:
            return v_next
        v = v_next
    return v


# ---------------------------------------------------------------------------
# 4) 状态向量（银河表 8：8N + 14 维）
# ---------------------------------------------------------------------------
def build_observation(
    f_row: np.ndarray,
    z_row: np.ndarray,
    bench: np.ndarray,
    prev_w: np.ndarray,
    hist_factor_mean: np.ndarray,
    hist_aux_mean: np.ndarray,
    hist_ret_mean: np.ndarray,
    hist_ret_vol: np.ndarray,
    portfolio_value: float,
    turnover: float,
    excess_ret: float,
    idx_ret_curr: float,
    idx_ret_mean: float,
    idx_ret_vol: float,
) -> np.ndarray:
    """银河表 8 的状态向量：8 个 N 维块 + 14 个标量。

    N 维块：当期信号 f / 辅助风险 z / 基准 wb / 主动偏离 d=prev_w−wb /
    f 与 z 的回看均值 / 回看收益均值与波动。NaN 一律填 0（缺失信号中性化）。
    标量：组合净值、f/z 暴露（d⊤f、d⊤z）、换手（**上期**口径）、TE、
    主动持仓比例、f 的 max/min/mean、z 的 mean、指数当期/均值/波动收益、
    上期超额收益。
    """
    def _clean(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(np.isfinite(x), x, 0.0)

    d = _clean(prev_w) - _clean(bench)
    f = _clean(f_row)
    z = _clean(z_row)
    blocks = [f, z, _clean(bench), d,
              _clean(hist_factor_mean), _clean(hist_aux_mean),
              _clean(hist_ret_mean), _clean(hist_ret_vol)]
    scalars = [
        portfolio_value, float(d @ f), float(d @ z), turnover,
        float(np.linalg.norm(d)),
        float((np.abs(_clean(prev_w)) > 1e-5).mean()),       # active_cnt
        float(f.max()) if len(f) else 0.0,
        float(f.min()) if len(f) else 0.0,
        float(f.mean()) if len(f) else 0.0,
        float(z.mean()) if len(z) else 0.0,
        idx_ret_curr, idx_ret_mean, idx_ret_vol, excess_ret,
    ]
    obs = np.concatenate([*blocks, np.asarray(scalars, dtype=float)])
    return obs.astype(np.float32)


def observation_dim(n_codes: int) -> int:
    """状态维度 = 8 个 N 维块 + 14 个标量（银河表 8）。"""
    return 8 * n_codes + 14


# ---------------------------------------------------------------------------
# 5) gymnasium 环境
# ---------------------------------------------------------------------------
class PortfolioEnv(gym.Env):
    """银河 0706 口径的指数增强组合优化 env（逐决策日推进）。

    Args:
        factor / aux: date×code 信号面板（z 方向=值越大风险越低）。
        bench: date×code 基准权重面板。
        period_returns: **期间末端日**索引的逐票区间收益（见模块 docstring 对齐约定）。
        idx_period_returns: 同口径指数区间收益（Series）。
        decision_dates: 决策日序列（升序，len ≥ 2；相邻两日为一个持有期）。
        cost_rate: 调仓费率——**训练传 0**（防双重扣费），回测评估传真实费率。
        lookback: 回看窗口（期数，银河默认 6）。
        init_to_bench: 首期持仓是否初始化为基准（默认 True，被动起步）。

    信息流：step(action) → 权重映射 → 奖励（按 t_{i+1} 期间收益）→ Banach 扣费
    → 状态推进。``info`` 含 ``weights`` / ``reward``（六项分解）/ ``portfolio_value``
    / ``date``，供回测层与诊断直接复用。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        factor: pd.DataFrame,
        aux: pd.DataFrame,
        bench: pd.DataFrame,
        period_returns: pd.DataFrame,
        idx_period_returns: pd.Series,
        decision_dates: list[pd.Timestamp],
        *,
        cost_rate: float = 0.0,
        lookback: int = 6,
        init_to_bench: bool = True,
        action_kw: dict[str, float] | None = None,
        reward_kw: dict[str, float] | None = None,
        seed: int = 0,
    ):
        super().__init__()
        if len(decision_dates) < 2:
            raise ValueError("decision_dates 须 ≥ 2（相邻两日构成一个持有期）")
        missing = [d for d in decision_dates if d not in bench.index]
        if missing:
            raise ValueError(f"bench 缺决策日: {missing[:3]}")
        need = decision_dates[1:]
        miss_r = [d for d in need if d not in period_returns.index]
        if miss_r:
            raise ValueError(f"period_returns 缺期间末端日: {miss_r[:3]}")

        self.factor = factor
        self.aux = aux
        self.bench = bench
        self.period_returns = period_returns
        self.idx_period_returns = idx_period_returns
        self.decision_dates = [pd.Timestamp(d) for d in decision_dates]
        self.cost_rate = float(cost_rate)
        self.lookback = int(lookback)
        self.init_to_bench = bool(init_to_bench)
        self.action_kw = {**DEFAULT_ACTION_KW, **(action_kw or {})}
        self.reward_kw = {**DEFAULT_REWARD_KW, **(reward_kw or {})}

        codes = bench.columns
        self.codes = list(codes)
        self.n_codes = len(codes)
        self.action_space = spaces.Box(
            low=-self.action_kw["clip_range"], high=self.action_kw["clip_range"],
            shape=(self.n_codes,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(observation_dim(self.n_codes),), dtype=np.float32)
        self._seed = int(seed)

        # 运行态
        self._i = 0
        self._prev_w = np.zeros(self.n_codes)
        self._prev_turnover = 0.0
        self._portfolio_value = 1.0
        self._excess_ret = 0.0

    # ------------------------------------------------------------------
    def _hist_stats(self, i: int) -> dict[str, np.ndarray]:
        """回看窗（当前决策日之前 lookback 期）内信号均值与收益统计。"""
        start = max(0, i - self.lookback)
        win_dates = self.decision_dates[start:i]
        if not win_dates:
            zero = np.zeros(self.n_codes)
            return {"f_mean": zero, "z_mean": zero,
                    "r_mean": zero, "r_vol": zero}
        win_r = self.period_returns.loc[win_dates].reindex(columns=self.codes)
        win_f = self.factor.loc[win_dates].reindex(columns=self.codes)
        win_z = self.aux.loc[win_dates].reindex(columns=self.codes)
        return {
            "f_mean": win_f.mean(axis=0).to_numpy(dtype=float),
            "z_mean": win_z.mean(axis=0).to_numpy(dtype=float),
            "r_mean": win_r.mean(axis=0).to_numpy(dtype=float),
            "r_vol": win_r.std(axis=0, ddof=0).to_numpy(dtype=float),
        }

    def _idx_hist_stats(self, i: int) -> tuple[float, float]:
        start = max(0, i - self.lookback)
        win = self.idx_period_returns.reindex(self.decision_dates[start:i]).dropna()
        if len(win) == 0:
            return 0.0, 0.0
        return float(win.mean()), float(win.std(ddof=0))

    def _get_obs(self) -> np.ndarray:
        t = self.decision_dates[self._i]
        hist = self._hist_stats(self._i)
        idx_mean, idx_vol = self._idx_hist_stats(self._i)
        return build_observation(
            f_row=self.factor.loc[t].reindex(self.codes).to_numpy(dtype=float),
            z_row=self.aux.loc[t].reindex(self.codes).to_numpy(dtype=float),
            bench=self.bench.loc[t].reindex(self.codes).to_numpy(dtype=float),
            prev_w=self._prev_w,
            hist_factor_mean=hist["f_mean"],
            hist_aux_mean=hist["z_mean"],
            hist_ret_mean=hist["r_mean"],
            hist_ret_vol=hist["r_vol"],
            portfolio_value=self._portfolio_value,
            turnover=self._prev_turnover,   # 观测在动作前取 → 是"上期"换手
            excess_ret=self._excess_ret,
            idx_ret_curr=float(self.idx_period_returns.reindex(
                [self.decision_dates[max(0, self._i - 1)]]).fillna(0).iloc[0]),
            idx_ret_mean=idx_mean,
            idx_ret_vol=idx_vol,
        )

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed if seed is not None else self._seed)
        self._i = 0
        t0 = self.decision_dates[0]
        b0 = self.bench.loc[t0].reindex(self.codes).to_numpy(dtype=float)
        b0 = np.where(np.isfinite(b0), b0, 0.0)
        if self.init_to_bench and b0.sum() > 0:
            self._prev_w = b0 / b0.sum()
        else:
            self._prev_w = np.full(self.n_codes, 1.0 / self.n_codes)
        self._prev_turnover = 0.0
        self._portfolio_value = 1.0
        self._excess_ret = 0.0
        return self._get_obs(), {"date": t0}

    def step(self, action):
        t = self.decision_dates[self._i]
        t_next = self.decision_dates[self._i + 1]
        bench = self.bench.loc[t].reindex(self.codes).to_numpy(dtype=float)
        bench = np.where(np.isfinite(bench), bench, 0.0)
        if bench.sum() <= 0:
            bench = np.full(self.n_codes, 1.0 / self.n_codes)
        else:
            bench = bench / bench.sum()

        w = map_actions_to_weights(np.asarray(action, dtype=float), bench,
                                   **self.action_kw)
        period_ret = self.period_returns.loc[t_next].reindex(self.codes)
        period_ret = np.nan_to_num(period_ret.to_numpy(dtype=float), nan=0.0)
        f_row = self.factor.loc[t].reindex(self.codes).to_numpy(dtype=float)
        z_row = self.aux.loc[t].reindex(self.codes).to_numpy(dtype=float)
        idx_ret = float(self.idx_period_returns.reindex([t_next]).fillna(0).iloc[0])

        rew = reward_components(w, self._prev_w, bench, period_ret, f_row, z_row,
                                **self.reward_kw)
        # Banach 扣费在调仓时点；期间收益按目标权重增长（模块 docstring 约定）
        v_after_cost = banach_projected_value(self._portfolio_value,
                                              rew["turnover"], self.cost_rate)
        self._portfolio_value = v_after_cost * (1.0 + float(w @ period_ret))
        self._excess_ret = rew["er"]
        self._prev_turnover = rew["turnover"]
        self._prev_w = w
        self._i += 1
        terminated = self._i >= len(self.decision_dates) - 1
        obs = self._get_obs() if not terminated else np.zeros(
            observation_dim(self.n_codes), dtype=np.float32)
        info: dict[str, Any] = {
            "date": t, "weights": w, "reward": rew,
            "portfolio_value": self._portfolio_value,
            "idx_ret": idx_ret,
        }
        if terminated:
            info["terminal"] = True
        return obs, float(rew["total"]), bool(terminated), False, info
