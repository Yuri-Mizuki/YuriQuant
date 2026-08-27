"""
MaskablePPO 的 AlphaPool gym 环境（P0）
======================================

把 :mod:`factor.rl.alphapool_env.AlphaPool` 包装成 gymnasium.Env，**复用
:class:`factor.gflownet.env.FactorMDP` + :class:`factor.gflownet.expr.ExprBuilder`
的槽位填充式构造**（对齐研报实际实现 Yu et al. 2023 / AlphaQCM——从根算子逐步
填槽，保证任何完整轨迹都是合法可执行的因子表达式；RPN 只是存储/展示形式）。

- 动作空间：``Discrete(n_actions)``，三段式编码：op 段（研报 21 算子，滚动算子
  的窗口是独立 win 槽位）/ win 段（窗口候选）/ feat 段（6 字段）。
- 观测空间：``Box((MAX_LEN + 3,))`` = 定长动作历史 token 序列 + 3 手工状态特征
  （当前深度/已用算子比例/已用节点比例，对齐 GFlowNet 研报状态编码）。
- 掩码：``FactorMDP.legal_mask``（MaskablePPO 需要），非法动作置 -inf。
- 终止：``ExprBuilder.is_done()``（树填充完毕 = 表达式完整）→ 评估并给奖励。
- 奖励：终止时按研报 4 档（invalid=-1 / empty=0 / fail_cache·no_pool=best_obj /
  pooled=new_obj）；中间步奖励 0。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from factor.rl.alphapool_env import (
    AlphaPool, FIELDS, REPORT_OPERATORS, WINDOWS,
    MAX_EXPR_LENGTH,
)
from factor.gflownet.env import FactorMDP
from factor.gflownet.expr import ExprBuilder, canonical_formula


def build_report_mdp(max_nodes: int = 8, max_depth: int = 3) -> FactorMDP:
    """研报 21 算子 + 6 字段 + 窗口候选 的 MDP（滚动算子窗口为独立槽位）。"""
    op_names = list(REPORT_OPERATORS.values())       # 注册表名
    mdp = FactorMDP(op_names=op_names, windows=tuple(WINDOWS),
                    features=list(FIELDS), max_depth=max_depth,
                    max_nodes=max_nodes, max_len=MAX_EXPR_LENGTH)
    return mdp


class AlphaPoolGymEnv(gym.Env):
    """研报 AlphaPool 的 gym 封装（P0：MSE Pool / IC 指标，槽位填充式构造）。"""

    metadata = {"render_modes": []}

    def __init__(self, panel: dict[str, pd.DataFrame], rets: pd.DataFrame,
                 features: Optional[list[str]] = None, capacity: int = 10,
                 metric: str = "ic", corr_threshold: float = 0.7,
                 max_nodes: int = 8, max_depth: int = 3, seed: int = 0):
        super().__init__()
        self.panel = panel
        self.rets = rets
        self.features = list(features) if features is not None else list(FIELDS)
        self.max_nodes = max_nodes
        self.max_depth = max_depth

        self.mdp = build_report_mdp(max_nodes=max_nodes, max_depth=max_depth)
        self.pool = AlphaPool(panel, rets, self.features, capacity=capacity,
                              metric=metric, corr_threshold=corr_threshold,
                              seed=seed)
        self._rng = np.random.default_rng(seed)

        self.n_actions = self.mdp.n_actions
        self.action_space = spaces.Discrete(self.n_actions)
        self.seq_len = self.mdp.max_len
        # token 历史定长 + 3 手工特征
        self.observation_space = spaces.Box(
            low=0, high=self.n_actions, shape=(self.seq_len + 3,),
            dtype=np.float32)

        self.builder: Optional[ExprBuilder] = None

    # ------------------------------------------------------------------
    # 掩码（MaskablePPO 接口）
    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        return self.mdp.legal_mask(self.builder)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.pool.rng = np.random.default_rng(seed)
        self.builder = self.mdp.reset()
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 防御：非法动作直接终止（不应发生，掩码已保证）
        if not self.mdp.legal_mask(self.builder)[action]:
            return self._obs(), -1.0, True, False, {"status": "invalid_action"}

        self.mdp.step(self.builder, int(action))
        if self.builder.is_done():
            reward, status = self._evaluate()
            return self._obs(), reward, True, False, {
                "status": status,
                "pool_stats": self.pool.stats(),
                "formula": canonical_formula(self.builder),
            }
        return self._obs(), 0.0, False, False, {}

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _evaluate(self) -> tuple[float, str]:
        formula = canonical_formula(self.builder)
        status, reward = self.pool.evaluate(formula)
        return float(reward), status

    def _obs(self) -> np.ndarray:
        ids, hand, _ = self.mdp.encode_state(self.builder, pad_id=self.n_actions)
        return np.concatenate([ids.numpy().ravel().astype(np.float32),
                               hand.numpy().ravel().astype(np.float32)])

    def best_pool(self) -> dict:
        return self.pool.stats()
