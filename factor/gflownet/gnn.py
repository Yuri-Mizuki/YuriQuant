"""
RGCN 状态编码器（AlphaSAGE 探针）
=================================

论文对齐：Chen et al., *AlphaSAGE: Structure-Aware Alpha Mining via GFlowNets
for Robust Exploration*, ICLR 2026, arXiv:2509.25055, §3.3。

动机
----
现有 :class:`factor.gflownet.net.FactorEncoder` 把状态编码成**动作历史 token 序列**
（emb → GRU）。序列里隐含了「先 add 后 close、再 volume」的**顺序信息**，但对
交换律等价式（``add(close,volume)`` vs ``add(volume,close)``）会给出不同表示，
即「长得一样的因子得到不同的状态向量」。论文的论点是：表达式是图，用 AST +
多关系图卷积能拿到对语法变体不变的表示。

**本模块只做一件事：把「状态 → 向量」这一段换掉**，其余（MDP、TB 损失、
奖励、组合层）一律不动，保证探针是单变量对照。

图构造（:func:`build_state_graph`）
-----------------------------------
节点四类（``node_type``）：

===========  ==========================================================
0 op          已填算子节点（带 ``window`` 标量）
1 feat        已填特征叶子
2 arg_slot    待填参数槽位（占位节点，FIFO 队首即 ``is_head``）
3 win_slot    待填窗口槽位（占位节点）
===========  ==========================================================

边四类 × 双向 = 8 种关系（``edge_type``）：

===========  ==========================================================
0 / 1        op → 第 0 / 第 1 个参数（**仅非交换律算子**）
2            op → 任一参数（**交换律算子 add/mul/max/min 共用**）
3            op → 窗口槽位
4 / 5 / 6 / 7  0 / 1 / 2 / 3 的反向边
===========  ==========================================================

**交换律共用一条关系边（关键设计）**：若按参数位区分关系，``add(close,volume)``
会得到「close 在 rel0、volume 在 rel1」的图，而 ``add(volume,close)`` 恰好相反，
编码器就**失去了**论文声称的结构不变性。因此对 ``COMMUTATIVE_OPS`` 只保留一条
``arg_comm`` 关系——这与 :func:`factor.gflownet.expr.canonical_formula` 的
「交换律子节点排序」是同一件事，只是做在图构造层，且对**部分填充**状态同样成立
（无需等整棵树补全）。非交换律算子（sub/div/ts_corr…）仍按参数位区分，保留顺序。

``is_head`` 标记（**必要，不是可选**）
-------------------------------------
原 GRU 编码器通过 token 顺序隐含知道「现在轮到填哪个槽位」；图 + max-pool
会把这个位置信息洗掉，导致同一棵树在填充到不同阶段时状态向量相同、策略无法
区分当前动作空间。因此节点标量里显式加 ``is_head``（当前 FIFO 队首槽位）与
``is_root``。**不加就不是公平对照**——那是在测「信息更少的编码器」而不是
「换成图结构」。

用法::

    from factor.gflownet.gnn import RGCNEncoder, build_state_graph
    enc = RGCNEncoder(n_actions=mdp.n_actions, d=32, n_hand=3, layers=2)
    x = enc.state_repr(builder, mdp)      # (d + n_hand,) —— 与 GRU 输出同宽
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from factor.gflownet.expr import COMMUTATIVE_OPS, ExprBuilder, Node

__all__ = ["StateGraph", "build_state_graph", "RGCNEncoder", "NODE_TYPES", "N_REL"]

#: 节点类型数（op / feat / arg_slot / win_slot）
NODE_TYPES = 4
#: 正向关系数：arg0 / arg1 / arg_comm / win
N_REL_FWD = 4
#: 关系数（4 类边 × 正向/反向）
N_REL = 2 * N_REL_FWD
#: 节点标量维度：is_head / is_root / depth 比例 / window 比例
N_SCALAR = 4
#: 单关系正向偏移（反向 = +N_REL_FWD）
REL_ARG_COMM = 2
REL_WIN = 3


def _rel_of(op_name: str, child_idx: int) -> int:
    """(算子, 参数位) -> 正向关系 id。

    交换律算子所有参数共用 ``REL_ARG_COMM``，保证 ``add(a,b)`` 与 ``add(b,a)`` 同图。
    """
    if op_name in COMMUTATIVE_OPS:
        return REL_ARG_COMM
    return 0 if child_idx == 0 else 1


@dataclass
class StateGraph:
    """一张状态图的张量化表示。

    Attributes:
        node_type: ``(N,)`` long，取值 ``0..3``。
        node_val:  ``(N,)`` long，动作 id（op / win / feat 段偏移后）；占位节点取
            ``pad_val``。
        node_scal: ``(N, 4)`` float。
        edge_index: ``(2, E)`` long，``[src, dst]``。
        edge_type:  ``(E,)`` long，``0..5``。
        n_nodes: 节点数（含占位）。
    """

    node_type: torch.Tensor
    node_val: torch.Tensor
    node_scal: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    n_nodes: int


def build_state_graph(b: ExprBuilder, mdp, pad_val: int | None = None,
                      device: str | torch.device = "cpu") -> StateGraph:
    """把部分表达式转成状态图。

    Args:
        b: 当前 MDP 状态（``ExprBuilder``，可能只填了一部分）。
        mdp: :class:`factor.gflownet.env.FactorMDP`，提供动作分段偏移与算子/特征次序。
        pad_val: 占位节点的 ``node_val``，默认 ``mdp.n_actions``。
        device: 张量设备。

    Returns:
        :class:`StateGraph`。

    Notes:
        - 树为空（``root is None``）时图中只有一个 arg_slot 节点、无任何边；
          RGCN 退化为自环 MLP，等价于「空状态嵌入」。
        - ``b._slots`` 是 FIFO 队列，``_slots[0]`` 记为 head。
    """
    if pad_val is None:
        pad_val = mdp.n_actions

    op_idx = {spec.name: i for i, spec in enumerate(mdp.op_specs)}
    feat_idx = {name: i for i, name in enumerate(mdp.features)}
    win_idx = {w: i for i, w in enumerate(mdp.windows)}

    node_type: list[int] = []
    node_val: list[int] = []
    node_scal: list[list[float]] = []
    edges: list[tuple[int, int, int]] = []

    #: id(node) -> 图下标。Node 是可变 dataclass（不可哈希），只能用 id 做键。
    idx_of: dict[int, int] = {}

    # ---- 1. 已填充的树节点（DFS，记录 op 嵌套深度） ----
    def _add(kind: int, val: int, scal: list[float]) -> int:
        i = len(node_type)
        node_type.append(kind)
        node_val.append(val)
        node_scal.append(scal)
        return i

    def _walk(n: Node, depth: int) -> int:
        if n.kind == "feat":
            i = _add(1, mdp._feat0 + feat_idx.get(n.feat_name, 0),
                     [0.0, 0.0, depth / max(b.max_depth, 1), 0.0])
            idx_of[id(n)] = i
            return i
        win_scal = 0.0
        if n.window is not None and mdp.n_win > 0:
            win_scal = (win_idx.get(n.window, 0) + 1) / mdp.n_win
        i = _add(0, op_idx.get(n.op_spec.name, 0),
                 [0.0, float(b.root is n), depth / max(b.max_depth, 1), win_scal])
        idx_of[id(n)] = i
        for ci, child in enumerate(n.children):
            if child is None:
                continue
            j = _walk(child, depth + 1)
            rel = _rel_of(n.op_spec.name, ci)
            edges.append((i, j, rel))
            edges.append((j, i, rel + N_REL_FWD))
        # 窗口值只走 op 节点标量（win 槽位填完后即出队，图里不再有该节点）
        return i

    if b.root is not None:
        _walk(b.root, 0)

    # ---- 2. 待填槽位（占位节点）；_slots 是 FIFO，i==0 即当前 head ----
    for si, slot in enumerate(b._slots):
        is_head = 1.0 if si == 0 else 0.0
        if slot.kind == "arg":
            i = _add(2, pad_val, [is_head, 0.0,
                                  slot.depth / max(b.max_depth, 1), 0.0])
            parent = slot.parent
            if parent is not None and id(parent) in idx_of:
                p = idx_of[id(parent)]
                ci = slot.child_idx if slot.child_idx >= 0 else 0
                rel = _rel_of(parent.op_spec.name, ci)
                edges.append((p, i, rel))
                edges.append((i, p, rel + N_REL_FWD))
        else:                        # win 槽位
            i = _add(3, pad_val, [is_head, 0.0,
                                  slot.depth / max(b.max_depth, 1), 0.0])
            parent = slot.parent
            if parent is not None and id(parent) in idx_of:
                p = idx_of[id(parent)]
                edges.append((p, i, REL_WIN))
                edges.append((i, p, REL_WIN + N_REL_FWD))

    n = len(node_type)
    st = torch.tensor(node_type, dtype=torch.long, device=device)
    sv = torch.tensor(node_val, dtype=torch.long, device=device)
    ss = torch.tensor(node_scal, dtype=torch.float32, device=device)
    if edges:
        ei = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]],
                          dtype=torch.long, device=device)
        et = torch.tensor([e[2] for e in edges], dtype=torch.long, device=device)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long, device=device)
        et = torch.zeros((0,), dtype=torch.long, device=device)
    return StateGraph(st, sv, ss, ei, et, n)


class RGCNEncoder(nn.Module):
    """关系图卷积编码器（论文 §3.3 的 RGCN + max-pooling）。

    与 :class:`factor.gflownet.net.FactorEncoder` **输出同宽**（``d + n_hand``），
    因此可以直接插进现有 actor 头，构成单变量对照。

    Args:
        n_actions: 动作总数（与前向策略输出维度一致，复用同一词表做节点取值嵌入）。
        d: 隐藏维度（GRU 基线默认 32）。
        n_hand: 手工状态特征数（3，与 GRU 版一致）。
        layers: RGCN 层数（论文 2）。
        n_rel: 关系数，默认 8（4 类边 × 双向）。

    Notes:
        - 消息传递手写（不依赖 torch_geometric）：按关系切片做 ``index_add_``，
          节点数 ≤ ``max_nodes`` + 槽位数（本项目 ≤ 20），开销可忽略。
        - 每个关系独立线性变换 ``W_r``，入度归一化（论文式 7 的 ``1/c_{v,r}``）；
          自环用独立 ``self_w``，等价于式 7 的 ``W_0``。
    """

    kind = "rgcn"

    def __init__(self, n_actions: int, d: int = 32, n_hand: int = 3,
                 layers: int = 2, n_rel: int = N_REL):
        super().__init__()
        self.vocab_size = n_actions + 1
        self.d = d
        self.n_rel = n_rel
        self.n_hand = n_hand
        self.type_emb = nn.Embedding(NODE_TYPES, d)
        self.val_emb = nn.Embedding(n_actions + 1, d)
        self.scal_proj = nn.Linear(N_SCALAR, d, bias=False)
        self.inp = nn.Linear(3 * d, d)
        self.self_w = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.rel_w = nn.ModuleList([
            nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_rel)])
            for _ in range(layers)
        ])

    def forward_graph(self, g: StateGraph) -> torch.Tensor:
        """图 → ``(d,)`` 图嵌入（max-pooling）。"""
        h = self.inp(torch.cat([self.type_emb(g.node_type),
                                self.val_emb(g.node_val),
                                self.scal_proj(g.node_scal)], dim=-1))
        src, dst = g.edge_index[0], g.edge_index[1]
        n = g.n_nodes
        for li in range(len(self.self_w)):
            out = self.self_w[li](h)
            for r in range(self.n_rel):
                m = g.edge_type == r
                if not bool(m.any()):
                    continue
                msg = self.rel_w[li][r](h[src[m]])
                agg = torch.zeros_like(h).index_add_(0, dst[m], msg)
                deg = torch.zeros(n, dtype=h.dtype, device=h.device).index_add_(
                    0, dst[m], torch.ones(int(m.sum()), dtype=h.dtype, device=h.device))
                out = out + agg / deg.clamp(min=1.0).unsqueeze(-1)
            h = torch.relu(out)
        return h.max(dim=0).values

    def state_repr(self, b: ExprBuilder, mdp) -> torch.Tensor:
        """``(d + n_hand,)``，与 :meth:`FactorEncoder.forward_enc` 输出同宽。"""
        dev = next(self.parameters()).device
        pooled = self.forward_graph(build_state_graph(b, mdp, device=dev))
        hand = torch.tensor(b.handcrafted_features(), dtype=torch.float32, device=dev)
        return torch.cat([pooled, hand], dim=-1)
