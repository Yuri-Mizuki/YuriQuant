"""
策略网络（Phase 0）
==================

研报对齐（系列之二十二 §2.2）：状态表示 = 动作历史序列编码 + 3 个手工状态特征。
Phase 0 用 **单层 GRU**（研报用 Transformer，GRU 在短 token 序列上足够验证 TB 闭环，
Phase 1 再升级）：

- ``FactorEncoder``: 共享编码器 —— token embedding（词表 n_actions+1，末位为 pad）+
  单层 GRU + 手工特征 concat。
- ``TBPolicy``: P_F 前向策略头（MLP -> logits）+ 可学习 ``logZ``（TB 总流参数）。
- ``PPONet``: Actor-Critic（actor 头同 TBPolicy，critic 头输出标量 V）。

masked 处理：合法动作掩码由 :class:`factor.gflownet.env.FactorMDP.legal_mask` 提供，
``logits[mask]=-inf`` 后再 softmax/采样。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from factor.gflownet.env import FactorMDP
from factor.gflownet.expr import ExprBuilder

__all__ = ["FactorEncoder", "TBPolicy", "PPONet", "masked_logits", "policy_logits"]


def masked_logits(net, b: ExprBuilder, mdp: FactorMDP,
                  pad_id: int) -> torch.Tensor:
    """返回 masked logits（合法动作有限值，非法 -inf）。"""
    ids, hand, _ = mdp.encode_state(b, device=next(net.parameters()).device,
                                    pad_id=pad_id)
    logits = net.forward_enc(ids, hand)[0]
    mask = torch.from_numpy(mdp.legal_mask(b)).bool().to(logits.device)
    logits = logits.masked_fill(~mask, float("-inf"))
    return logits


def policy_logits(net, b: ExprBuilder, mdp: FactorMDP) -> torch.Tensor:
    """便捷包装（pad_id 取网络词表-1）。"""
    return masked_logits(net, b, mdp, pad_id=net.vocab_size - 1)


class FactorEncoder(nn.Module):
    """动作历史 GRU 编码 + 手工特征 concat。"""

    def __init__(self, n_actions: int, d: int = 32, max_len: int = 20,
                 n_hand: int = 3):
        super().__init__()
        self.vocab_size = n_actions + 1            # 末位 = pad token
        self.emb = nn.Embedding(n_actions + 1, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.n_hand = n_hand

    def forward_enc(self, ids: torch.Tensor, hand: torch.Tensor) -> torch.Tensor:
        """ids: (B, L), hand: (B, n_hand) -> (B, d + n_hand)。"""
        x = self.emb(ids)                          # (B, L, d)
        _, h = self.gru(x)                         # (1, B, d)
        h = h.squeeze(0)                           # (B, d)
        return torch.cat([h, hand], dim=-1)


class TBPolicy(nn.Module):
    """P_F 策略 + 可学习 logZ。"""

    def __init__(self, n_actions: int, d: int = 32, max_len: int = 20,
                 n_hand: int = 3, hidden: int = 64, init_logz: float = 0.0):
        super().__init__()
        self.enc = FactorEncoder(n_actions, d, max_len, n_hand)
        self.vocab_size = self.enc.vocab_size
        self.actor = nn.Sequential(
            nn.Linear(d + n_hand, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        # logZ 经验初始化（≈log(ΣR) 量级）可避免从 0 起步时 Z 跨数量级追赶的
        # 前期损耗；仍可学习（独立更高 lr 优化，见 train_tb）
        self.logZ = nn.Parameter(torch.tensor(float(init_logz)))

    def forward_enc(self, ids, hand):
        return self.actor(self.enc.forward_enc(ids, hand))


class PPONet(nn.Module):
    """Actor-Critic（PPO 对照）。"""

    def __init__(self, n_actions: int, d: int = 32, max_len: int = 20,
                 n_hand: int = 3, hidden: int = 64):
        super().__init__()
        self.enc = FactorEncoder(n_actions, d, max_len, n_hand)
        self.vocab_size = self.enc.vocab_size
        self.actor = nn.Sequential(
            nn.Linear(d + n_hand, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(d + n_hand, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward_enc(self, ids, hand):
        h = self.enc.forward_enc(ids, hand)
        return self.actor(h), self.critic(h).squeeze(-1)
