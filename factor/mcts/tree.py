"""MCTS 树：节点 / UCT / virtual expansion（东吴 §一 核心公式）。

::

    UCT(child)  = child.best_reward + exploration·sqrt(ln(parent_visits+1)/visits)
    virtual(node) = node.best_reward + virtual_weight/(1+expansions)

选择循环：从根出发，在每个节点比较「最优子节点 UCT」与「本节点
virtual_score」——子节点胜则下挖（利用），virtual 胜则在本节点横扩
（探索）。virtual expansion 控制树的形状，防止一出生就长成深窄链。

``best_reward`` 语义 = **子树内已评测公式的最高 reward**（reward 回传取
max，与"reward 回答公式好不好、UCT 回答还沿不沿这条路走"的分工一致）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["MCTSNode", "MCTSTree"]


@dataclass
class MCTSNode:
    """一个已评测公式（根节点 = Seed；子节点 = 变体）。"""

    label: str                      # 展示名（Seed 名或公式文本）
    depth: int
    reward: float                   # 本节点自己的 reward（搜索口径，含 diversity）
    visits: int = 0
    #: 本节点被选为扩展目标的次数（virtual_score 的膨胀项）
    expansions: int = 0
    parent: Optional["MCTSNode"] = None
    children: list["MCTSNode"] = field(default_factory=list)
    #: 子树内最高 reward（含本节点）——回传时取 max 维护
    best_reward: float = float("-inf")

    def __post_init__(self) -> None:
        if self.best_reward == float("-inf"):
            self.best_reward = self.reward

    def path(self) -> list["MCTSNode"]:
        """根 → 本节点的路径（prompt 的"历史路径"反馈用）。"""
        out: list[MCTSNode] = []
        node: Optional[MCTSNode] = self
        while node is not None:
            out.append(node)
            node = node.parent
        return list(reversed(out))


class MCTSTree:
    """单 Seed 的搜索树：选择 / 挂子 / 回传。"""

    def __init__(self, root: MCTSNode, max_depth: int = 3,
                 exploration: float = 1.0, virtual_weight: float = 0.08):
        self.root = root
        self.max_depth = int(max_depth)
        self.exploration = float(exploration)
        self.virtual_weight = float(virtual_weight)

    # -- 评分 ---------------------------------------------------------------

    def uct(self, parent: MCTSNode, child: MCTSNode) -> float:
        """子节点 UCT（未访问按 +inf 处理，保证新生子先被探索一次）。"""
        if child.visits == 0:
            return float("inf")
        return (child.best_reward + self.exploration
                * math.sqrt(math.log(parent.visits + 1) / child.visits))

    def virtual_score(self, node: MCTSNode) -> float:
        return node.best_reward + self.virtual_weight / (1 + node.expansions)

    # -- 选择 ---------------------------------------------------------------

    def select_target(self) -> MCTSNode:
        """从根下潜，返回应扩展的节点（terminal 节点不会被选中）。"""
        node = self.root
        while True:
            if node.depth >= self.max_depth:
                # 触底：本节点不能再挂子——退回父节点扩展
                return node.parent if node.parent is not None else node
            best_child, best_uct = None, float("-inf")
            for c in node.children:
                u = self.uct(node, c)
                if u > best_uct:
                    best_child, best_uct = c, u
            if best_child is None or self.virtual_score(node) >= best_uct:
                return node          # 横扩本节点
            node = best_child        # 下挖

    # -- 扩展 / 回传 ---------------------------------------------------------

    def add_child(self, parent: MCTSNode, label: str, reward: float) -> MCTSNode:
        child = MCTSNode(label=label, depth=parent.depth + 1, reward=reward,
                         parent=parent)
        parent.children.append(child)
        self.backprop(child)
        return child

    def backprop(self, node: MCTSNode) -> None:
        """visits+1 沿路径回传、best_reward 取 max 维护。"""
        cur: Optional[MCTSNode] = node
        while cur is not None:
            cur.visits += 1
            sub = max([c.best_reward for c in cur.children], default=float("-inf"))
            cur.best_reward = max(cur.reward, sub)
            cur = cur.parent

    def mark_expanded(self, node: MCTSNode) -> None:
        node.expansions += 1
