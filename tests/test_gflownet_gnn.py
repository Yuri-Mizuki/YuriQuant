"""RGCN 状态编码器（AlphaSAGE 探针）单元测试。

覆盖：图构造（节点/边/head 标记）、结构不变性（交换律等价式表示相同，而 GRU
表示不同）、掩码正确性、梯度可回传、与 GRU 编码器输出同宽、GRU 参数集未变。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from factor.gflownet.env import FactorMDP
from factor.gflownet.gnn import N_REL, RGCNEncoder, build_state_graph
from factor.gflownet.net import TBPolicy, policy_logits

OPS = ["add", "ts_mean", "sign"]
WINS = (5, 10)
FEATS = ["close", "volume"]


@pytest.fixture
def mdp():
    return FactorMDP(OPS, WINS, FEATS, max_depth=3, max_nodes=9)


def _add(b):
    return 0                      # OPS[0] = add


# ---------------------------------------------------------------------------
# 图构造
# ---------------------------------------------------------------------------
def test_empty_state_graph(mdp):
    """初始空状态：只有一个 arg_slot 占位节点、无任何边、且它就是 head。"""
    b = mdp.reset()
    g = build_state_graph(b, mdp)
    assert g.n_nodes == 1
    assert int(g.node_type[0]) == 2                     # arg_slot
    assert float(g.node_scal[0][0]) == 1.0              # is_head
    assert g.edge_index.shape == (2, 0)
    assert g.edge_type.numel() == 0


def test_graph_after_one_op(mdp):
    """选 add 后：op 节点 + 2 个 arg 槽位。add 是交换律算子 ⇒ 参数共用 arg_comm
    关系（rel=2），反向 rel=6，共 4 条边。"""
    b = mdp.reset()
    mdp.step(b, _add(b))
    g = build_state_graph(b, mdp)
    assert g.n_nodes == 3                               # add + 2 slots
    assert int(g.node_type[0]) == 0                     # op
    assert sorted(int(t) for t in g.node_type[1:]) == [2, 2]
    assert g.edge_index.shape == (2, 4)
    assert set(int(t) for t in g.edge_type) == {2, 6}   # arg_comm 正/反向


def test_head_marker_unique(mdp):
    """任意合法轨迹状态下，is_head 恰好标记一个节点。"""
    rng = np.random.default_rng(0)
    for _ in range(40):
        b = mdp.reset()
        while not b.is_done():
            g = build_state_graph(b, mdp)
            n_head = int((g.node_scal[:, 0] > 0.5).sum())
            assert n_head == 1, f"head 标记数={n_head}"
            legal = np.flatnonzero(mdp.legal_mask(b))
            assert mdp.step(b, int(rng.choice(legal)))


def test_window_slot_is_head_when_pending(mdp):
    """ts_mean 选完后窗口槽位是 head，窗口填好后 head 换成参数槽位。"""
    b = mdp.reset()
    mdp.step(b, 1)                                      # OPS[1] = ts_mean
    g = build_state_graph(b, mdp)
    head_idx = int(g.node_scal[:, 0].argmax())
    assert int(g.node_type[head_idx]) == 3               # win_slot
    mdp.step(b, mdp._win0)                              # 选窗口 5
    g2 = build_state_graph(b, mdp)
    head_idx2 = int(g2.node_scal[:, 0].argmax())
    assert int(g2.node_type[head_idx2]) == 2             # arg_slot
    assert not bool((g2.node_type == 3).any())           # win 槽位已出队


# ---------------------------------------------------------------------------
# 结构不变性（探针的核心假设）
# ---------------------------------------------------------------------------
def _build_expr(mdp, order):
    b = mdp.reset()
    mdp.step(b, 0)                                       # add
    for name in order:
        mdp.step(b, mdp._feat0 + FEATS.index(name))
    return b


def test_rgcn_invariant_to_commutative_order(mdp):
    """add(close,volume) 与 add(volume,close) 的图表示逐位相同。"""
    torch.manual_seed(0)
    enc = RGCNEncoder(mdp.n_actions, d=16)
    r1 = enc.state_repr(_build_expr(mdp, ("close", "volume")), mdp)
    r2 = enc.state_repr(_build_expr(mdp, ("volume", "close")), mdp)
    assert r1.shape == r2.shape
    assert torch.allclose(r1, r2, atol=1e-6), "RGCN 表示未对交换律变体保持不变"


def test_gru_not_invariant_to_commutative_order(mdp):
    """同一对等价式，GRU 编码器给出**不同**表示（对照组：说明探针有区分力）。"""
    torch.manual_seed(0)
    net = TBPolicy(mdp.n_actions, d=16, encoder="gru")
    pad = net.vocab_size - 1
    outs = []
    for order in (("close", "volume"), ("volume", "close")):
        b = _build_expr(mdp, order)
        ids, hand, _ = mdp.encode_state(b, pad_id=pad)
        with torch.no_grad():
            outs.append(net.enc.forward_enc(ids, hand))
    assert not torch.allclose(outs[0], outs[1], atol=1e-4), \
        "GRU 表示意外地对交换顺序不变"


# ---------------------------------------------------------------------------
# 与策略头的接口
# ---------------------------------------------------------------------------
def test_encoder_output_width_matches_gru(mdp):
    """两编码器输出同宽 ⇒ 可共用同一 actor 头（单变量对照的前提）。"""
    gru = TBPolicy(mdp.n_actions, d=16, encoder="gru")
    rgcn = TBPolicy(mdp.n_actions, d=16, encoder="rgcn")
    b = _build_expr(mdp, ("close", "volume"))
    ids, hand, _ = mdp.encode_state(b, pad_id=gru.vocab_size - 1)
    with torch.no_grad():
        x_gru = gru.enc.forward_enc(ids, hand)
        x_rgcn = rgcn.enc.state_repr(b, mdp)
    # GRU 带 batch 维、RGCN 返回 1-D，但**最后一维必须相同**（actor 头共享的前提）
    assert x_gru.shape[-1] == x_rgcn.shape[-1] == 16 + 3


def test_rgcn_masked_logits(mdp):
    """非法动作 logits = -inf，合法动作有限值；形状 = n_actions。"""
    net = TBPolicy(mdp.n_actions, d=16, encoder="rgcn")
    b = mdp.reset()
    logits = policy_logits(net, b, mdp)
    assert logits.shape == (mdp.n_actions,)
    mask = mdp.legal_mask(b)
    assert torch.isinf(logits[~torch.from_numpy(mask)]).all()
    assert torch.isfinite(logits[torch.from_numpy(mask)]).all()


def test_rgcn_gradient_flows(mdp):
    """图路径可反传：共享主干全程有梯度，且**图上实际存在**的每条关系都有梯度。"""
    net = TBPolicy(mdp.n_actions, d=16, encoder="rgcn")
    b = mdp.reset()
    mdp.step(b, 0)                                      # add
    mdp.step(b, 1)                                      # ts_mean（窗口待填）
    g = build_state_graph(b, mdp)
    rels = sorted(set(int(t) for t in g.edge_type))
    assert rels, "测试状态不应无关系边"

    logits = policy_logits(net, b, mdp)
    torch.logsumexp(logits, dim=0).backward()

    shared = ["enc.type_emb.weight", "enc.val_emb.weight", "enc.scal_proj.weight",
              "enc.inp.weight", "actor.0.weight"]
    for name in shared:
        p = dict(net.named_parameters())[name]
        assert p.grad is not None, f"共享参数 {name} 无梯度"
    for li in range(len(net.enc.rel_w)):
        for r in rels:
            w = net.enc.rel_w[li][r].weight
            assert w.grad is not None, f"rel_w.{li}.{r}.weight 无梯度（关系在图上存在）"


def test_gru_parameter_set_unchanged():
    """GRU 臂的参数字典键集合与改动前一致（防止「只换编码器」被悄悄破坏）。"""
    net = TBPolicy(10, encoder="gru")
    assert set(net.state_dict().keys()) == {
        "enc.emb.weight",
        "enc.gru.weight_ih_l0", "enc.gru.weight_hh_l0",
        "enc.gru.bias_ih_l0", "enc.gru.bias_hh_l0",
        "actor.0.weight", "actor.0.bias",
        "actor.2.weight", "actor.2.bias",
        "logZ",
    }


def test_invalid_encoder_raises():
    with pytest.raises(ValueError, match="encoder"):
        TBPolicy(10, encoder="transformer")


def test_relation_count():
    """关系数常量与文档一致（4 类边 × 双向）。"""
    assert N_REL == 8
