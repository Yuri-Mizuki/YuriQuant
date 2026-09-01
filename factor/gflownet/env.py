"""
因子构造 MDP 环境（Phase 0）
===========================

研报对齐（系列之二十二 §2.2/§2.3）：

- **动作空间**：扁平离散空间，三类操作 —— operator（算子子集）/ window（窗口候选）/
  feature（叶子特征），动作 id 分三段编码；总大小 = 三类数量之和。
- **合法动作掩码**：由 ``ExprBuilder.legal_kinds()`` 决定（初始强制 op、等窗口只能选
  window、不超复杂度上限可继续选 op）。
- **状态编码**：动作历史 token 序列（op/win/feat 各自独立 embedding 表）+ 3 个手工
  特征（当前深度 / 已用算子比例 / 已用节点比例），定长 padding。
- **终止**：树填充完毕（无待填槽位）。
"""
from __future__ import annotations


import numpy as np

from factor.gflownet.expr import ExprBuilder

__all__ = ["FactorMDP", "REPORT_ALIAS"]


# 研报（图表6）算子名 -> 项目注册表算子名的别名（neg→reverse、max2→max 等）
REPORT_ALIAS: dict[str, str] = {
    "neg": "reverse", "max2": "max", "min2": "min",
    "ts_argmax": "ts_arg_max", "ts_argmin": "ts_arg_min",
    "ts_mad": "ts_avedev",
}


class FactorMDP:
    """因子构造 MDP。

    Args:
        op_names: 算子名子集（研报名或项目注册表名；``neg``/``max2`` 等研报名
            自动映射到项目别名 reverse/max）。
        windows: 窗口候选（如 (5, 10, 20, 60)）。
        features: 叶子特征名列表。
        max_depth / max_nodes: 复杂度上限（传给 ExprBuilder）。
        max_len: 轨迹 token 定长（padding 用）。
    """

    def __init__(self, op_names: list[str], windows: tuple[int, ...],
                 features: list[str], max_depth: int = 3, max_nodes: int = 9,
                 max_len: int = 20):
        from factor.operators import op_registry
        reg = op_registry()
        self.op_specs = [reg[REPORT_ALIAS.get(n, n)] for n in op_names]
        self.windows = list(windows)
        self.features = list(features)
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_len = max_len

        self.n_op = len(self.op_specs)
        self.n_win = len(self.windows)
        self.n_feat = len(self.features)
        self.n_actions = self.n_op + self.n_win + self.n_feat
        # 动作段偏移
        self._op0, self._win0, self._feat0 = 0, self.n_op, self.n_op + self.n_win

    # ------------------------------------------------------------------
    # 动作 id <-> 语义
    # ------------------------------------------------------------------
    def action_kind(self, a: int) -> str:
        if a < self._win0:
            return "op"
        if a < self._feat0:
            return "win"
        return "feat"

    def action_semantics(self, a: int) -> tuple[str, object, int]:
        """(type, 语义对象, 段内下标)。"""
        if a < self._win0:
            return "op", self.op_specs[a], a
        if a < self._feat0:
            return "win", self.windows[a - self._win0], a - self._win0
        return "feat", self.features[a - self._feat0], a - self._feat0

    def build_actions(self, kinds: list[str]) -> list[int]:
        """把动作类别列表展开为合法动作 id 列表。"""
        ids: list[int] = []
        for k in kinds:
            if k == "op":
                ids += list(range(self._op0, self._win0))
            elif k == "win":
                ids += list(range(self._win0, self._feat0))
            else:
                ids += list(range(self._feat0, self.n_actions))
        return ids

    # ------------------------------------------------------------------
    # 环境交互
    # ------------------------------------------------------------------
    def reset(self) -> ExprBuilder:
        return ExprBuilder(max_depth=self.max_depth, max_nodes=self.max_nodes)

    def legal_mask(self, b: ExprBuilder) -> np.ndarray:
        """返回 bool 掩码（shape=(n_actions,)），按节点上限精确过滤 op/feat。"""
        mask = np.zeros(self.n_actions, dtype=bool)
        for k in b.legal_kinds():
            if k == "op":
                for i, spec in enumerate(self.op_specs):
                    if b.can_add_op(spec):
                        mask[self._op0 + i] = True
            elif k == "win":
                mask[self._win0:self._feat0] = True
            else:
                if b.can_add_feat():
                    mask[self._feat0:] = True
        return mask

    def step(self, b: ExprBuilder, action_id: int) -> bool:
        """执行动作（原地 mutate）。非法动作返回 False。"""
        t, value, idx = self.action_semantics(action_id)
        return b.step(t, value, idx)

    def sample_trajectory(self, policy_logits_fn, rng: np.random.Generator):
        """按策略 logits 采样一条完整轨迹。

        Args:
            policy_logits_fn: callable(state) -> (logits[n_actions], log_probs[n_actions])。
                logits 须为 masked（非法动作 -inf）。
            rng: numpy 随机数生成器。

        Returns:
            (tokens, actions, logp_sum, builder, done_steps)
        """
        b = self.reset()
        actions: list[int] = []
        logp_sum = 0.0
        steps = 0
        while not b.is_done() and steps < self.max_len:
            logits, logp = policy_logits_fn(b)
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            a = rng.choice(self.n_actions, p=probs)
            if not self.step(b, int(a)):
                raise RuntimeError(f"采样到非法动作 {a}（{self.action_kind(a)}）")
            actions.append(int(a))
            logp_sum += float(logp[int(a)])
            steps += 1
        return b, actions, logp_sum, steps

    # ------------------------------------------------------------------
    # 状态编码（torch）
    # ------------------------------------------------------------------
    def encode_state(self, b: ExprBuilder, device="cpu", pad_id: int = -1):
        """token ids + 手工特征。

        token 编码：(type, value_idx) -> 全局 id（op 段偏移 / win 段偏移 / feat 段偏移），
        单表 embedding 查询（词表大小 n_actions，pad_id 由网络在 n_actions 处预留）。
        Returns:
            (token_ids[1, max_len], hand[1, 3], valid_len)
        """
        import torch
        ids = torch.full((1, self.max_len), pad_id, dtype=torch.long)
        for i, (t, v) in enumerate(b.tokens):
            if i >= self.max_len:
                break
            off = self._op0 if t == "op" else (self._win0 if t == "win" else self._feat0)
            ids[0, i] = off + v
        hand = torch.tensor([b.handcrafted_features()], dtype=torch.float32)
        return ids.to(device), hand.to(device), min(len(b.tokens), self.max_len)
