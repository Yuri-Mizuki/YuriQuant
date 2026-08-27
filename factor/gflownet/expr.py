"""
GFlowNet 因子表达式树（Phase 0）
================================

研报对齐（系列之二十二 §2.2）：表达式以 ExprNode 树结构组织，逐步构建：

- 树由 **op 节点**（引用 :class:`factor.operators.OpSpec`）与 **feature 叶子**组成；
  op 的窗口参数是独立槽位（独立动作，研报「动作空间含算子/窗口/特征三类操作」）。
- 语法约束：初始步强制选 op（不允许纯特征根）；若队首槽位等待窗口则只能选窗口；
  其余情况可选 feature，且在不超复杂度上限时可继续选 op。
- 复杂度上限：``max_depth``（op 嵌套深度）与 ``max_nodes``（节点总数）。
- 简化：交换律算子（add/mul/max/min）子节点按 canonical 排序 + 双重 neg 折叠，
  等价表达式归一为同一 canonical 字符串 —— 用于奖励缓存去重（研报 §2.2）。

canonical 字符串采用 :mod:`factor.formula` 的 GP 风格前缀语法
（``ts_mean_5(close)``），因此可直接被 ``formula_builder`` 解析求值，
与 exhaustive / GP 挖掘共用同一求值链路。

约定：``ExprBuilder`` 只处理**语义化动作**（op=OpSpec 对象 / win=窗口整数 /
feat=特征名），动作 id 与语义的映射由 :mod:`factor.gflownet.env` 负责。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from factor.operators import OpSpec

__all__ = ["Node", "Slot", "ExprBuilder", "COMMUTATIVE_OPS", "canonical_formula"]


# 交换律算子：canonical 化时子节点按字符串排序（ts_corr/ts_cov 含窗口且参数有序，不交换）
COMMUTATIVE_OPS: frozenset[str] = frozenset({"add", "mul", "max", "min"})


@dataclass
class Node:
    """表达式树节点。

    - ``kind="op"``: op 节点，``op_spec`` 非空，``children`` 按 arity 就地填充。
    - ``kind="feat"``: 叶子特征，``feat_name`` 非空。
    """
    kind: str
    op_spec: Optional[OpSpec] = None
    feat_name: Optional[str] = None
    window: Optional[int] = None          # op 的窗口（None = 未定 / 无需窗口）
    children: list["Node"] = field(default_factory=list)


@dataclass
class Slot:
    """待填充槽位。

    - ``kind="arg"``: 可放 op 或 feature（``leaf_only=True`` 时只能放 feature）。
    - ``kind="win"``: 只能放 window（整数）。
    - ``parent``/``child_idx``: 槽位归属（op 节点的参数位置；win 槽位挂其自身节点）。
    - ``depth``: 该槽位若放 op 得到的 op 嵌套深度（初始槽位 depth=0）。
    """
    kind: str                      # "arg" | "win"
    depth: int
    leaf_only: bool = False
    parent: Optional[Node] = None  # 挂载到的 op 节点
    child_idx: int = -1            # 参数位置；win 槽位用 -1


class ExprBuilder:
    """部分因子表达式（MDP 状态）。

    使用 **FIFO 槽位队列**：队首槽位决定当前可执行动作类别，与研报
    「若当前节点等待窗口，则只能选择 window」语义一致。

    Attributes:
        root: 顶层 op 节点（构建完成后即完整因子树）。
        op_count / feat_count / node_count: 复杂度统计。
        max_depth / max_nodes: 复杂度上限。
        tokens: 动作历史 (action_type, value_idx) 序列，供状态编码。
    """

    def __init__(self, max_depth: int = 3, max_nodes: int = 9):
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.root: Optional[Node] = None
        self.op_count = 0
        self.feat_count = 0
        self.tokens: list[tuple[str, int]] = []   # (type, value_idx)
        self._slots: list[Slot] = []
        self._slots.append(Slot(kind="arg", depth=0))  # 初始：必须放 op 的位置

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    @property
    def node_count(self) -> int:
        return self.op_count + self.feat_count

    @property
    def cur_depth(self) -> int:
        """当前最大 op 嵌套深度（空树为 0）。"""
        return max((s.depth for s in self._slots), default=0)

    def is_done(self) -> bool:
        return not self._slots

    def legal_kinds(self) -> list[str]:
        """当前可执行的动作类别（[] = 终止）。

        仅按语法/深度判断（不精确到 max_nodes，那由 ``can_add_op`` /
        ``can_add_feat`` 在掩码层过滤）：初始强制 op；等窗口只能选 window；
        其余可选 feature，且在不超复杂度时可继续选 op。
        """
        if self.is_done():
            return []
        if self.root is None:
            return ["op"]                    # 初始步强制选 op（研报：不允许纯特征根）
        head = self._slots[0]
        if head.kind == "win":
            return ["win"]
        kinds = ["feat"]
        if not head.leaf_only:
            kinds = ["op"] + kinds
        return kinds

    def _pending_args(self) -> int:
        """当前待填 arg 槽位总数（含队首；win 槽位不占节点）。"""
        return sum(1 for s in self._slots if s.kind == "arg")

    def can_add_op(self, spec: OpSpec) -> bool:
        """该槽位是否可放 ``spec``：保证轨迹必可终止（无死锁）。

        保守约束：放该 op（+1 节点）后，剩余所有 arg 槽位（现有 p-1 个 + 新
        引入 arity 个）最坏全部由 feature 叶子填充，总节点数不得超过
        ``max_nodes``：``node_count + pending_args + arity <= max_nodes``。
        """
        if self.is_done():
            return False
        head = self._slots[0]
        if self.root is None:
            return True                       # 初始 op 必可放
        if head.kind != "arg" or head.leaf_only:
            return False
        return self.node_count + self._pending_args() + spec.arity <= self.max_nodes

    def can_add_feat(self) -> bool:
        """是否可放 feature 叶子（保证轨迹必可终止）。"""
        if self.is_done():
            return False
        head = self._slots[0]
        return head.kind == "arg" and \
            self.node_count + self._pending_args() <= self.max_nodes

    # ------------------------------------------------------------------
    # 动作执行（原地 mutate；非法动作返回 False）
    # ------------------------------------------------------------------
    def step(self, action_type: str, value, value_idx: int) -> bool:
        if action_type not in self.legal_kinds():
            return False
        # 先校验（can_add_op/can_add_feat 依赖队首槽位），通过后统一 pop
        if action_type == "op" and not self.can_add_op(value):
            return False
        if action_type == "feat" and not self.can_add_feat():
            return False
        head = self._slots.pop(0)
        self.tokens.append((action_type, value_idx))
        if action_type == "win":
            head.parent.window = value
            return True
        if action_type == "op":
            node = Node(kind="op", op_spec=value,
                        children=[None] * value.arity)
            if head.parent is not None:
                head.parent.children[head.child_idx] = node
            else:
                self.root = node
            self.op_count += 1
            new_depth = head.depth + 1
            if node.op_spec.n_window >= 1:
                self._slots.append(Slot(kind="win", depth=head.depth,
                                        parent=node, child_idx=-1))
            leaf_only = new_depth >= self.max_depth
            for i in range(node.op_spec.arity):
                self._slots.append(Slot(kind="arg", depth=new_depth,
                                        leaf_only=leaf_only,
                                        parent=node, child_idx=i))
            return True
        # feature
        leaf = Node(kind="feat", feat_name=value)
        if head.parent is not None:
            head.parent.children[head.child_idx] = leaf
        self.feat_count += 1
        return True

    # ------------------------------------------------------------------
    def handcrafted_features(self) -> list[float]:
        """3 个手工状态特征（研报 §2.2）：当前深度 / 已用算子比例 / 已用节点比例。"""
        return [
            self.cur_depth / max(self.max_depth, 1),
            self.op_count / max(self.max_nodes, 1),
            self.node_count / max(self.max_nodes, 1),
        ]


# ---------------------------------------------------------------------------
# canonical 字符串（奖励缓存 key；兼容 factor.formula 前缀语法）
# ---------------------------------------------------------------------------
def _neg_fold(node: Node) -> Node:
    """双重 neg 折叠：reverse(reverse(x)) -> x（返回新节点，不改动原树）。"""
    if node.kind == "feat":
        return node
    kids = [_neg_fold(c) for c in node.children]
    if node.op_spec.name == "reverse" and kids and kids[0].kind == "op" \
            and kids[0].op_spec.name == "reverse":
        return kids[0].children[0]
    node.children = kids
    return node


def canonical_formula(builder: ExprBuilder) -> str:
    """构建 canonical 前缀公式字符串（GP 风格窗口后缀，兼容 formula.py）。

    应用：交换律子节点排序 + 双重 neg 折叠。用于奖励缓存 key 与公式入库。
    """
    if builder.root is None:
        raise ValueError("空表达式：初始步必须选 op")
    root = _neg_fold(builder.root)

    def _rec(n: Node) -> str:
        if n.kind == "feat":
            return n.feat_name
        args = [_rec(c) for c in n.children]
        if n.op_spec.name in COMMUTATIVE_OPS and len(args) == 2:
            args.sort()
        head = f"{n.op_spec.name}_{n.window}" if (n.op_spec.n_window >= 1
                                                  and n.window is not None) \
            else n.op_spec.name
        return f"{head}({','.join(args)})"

    return _rec(root)
