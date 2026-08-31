"""
简化 PPO 对照（Phase 0）
=======================

研报对齐（系列之二十二 §2.4）：RL 基线用几乎相同的环境/状态空间/网络架构，
Actor-Critic + PPO + 熵奖励。预期现象：RL 训练初期收敛快，但随后**模式崩溃**
（batch 内因子相关性中位数升至接近 1），与 GFlowNet 的低相关性形成对照。

本实现为「PPO-lite」：clipped surrogate + 蒙特卡洛收益（γ=1，终止奖励 R(x) 回溯
到每步）+ 熵奖励，多 epoch 更新。**训练已张量化**（每 round 收集的步级数据堆叠
为一个 batch，一次 forward/backward，避免逐条 backward 的 CPU 瓶颈）。
仅用于 Phase 0 对照，不追求与研报 PPO 完全一致。
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from factor.gflownet.env import FactorMDP
from factor.gflownet.net import PPONet, policy_logits
from factor.gflownet.tb import RewardFn

log = logging.getLogger(__name__)

__all__ = ["train_ppo"]


def _collect(mdp: FactorMDP, reward_fn: RewardFn, net: PPONet,
             n_traj: int, rng: np.random.Generator) -> tuple:
    """收集 n_traj 条轨迹，展开为步级张量（含终止奖励 R 回溯）。"""
    ids_l, hand_l, act_l, lp_l, r_l = [], [], [], [], []
    pad = net.vocab_size - 1
    for _ in range(n_traj):
        b = mdp.reset()
        traj = []
        for _ in range(mdp.max_len):
            if b.is_done():
                break
            ids, hand, _ = mdp.encode_state(b, pad_id=pad)
            logits = policy_logits(net, b, mdp)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            traj.append((ids, hand, int(a), float(dist.log_prob(a).detach())))
            if not mdp.step(b, int(a)):
                raise RuntimeError(f"PPO 采样到非法动作 {int(a)}")
        R = reward_fn(b)
        for ids, hand, a, lp in traj:
            ids_l.append(ids)
            hand_l.append(hand)
            act_l.append(a)
            lp_l.append(lp)
            r_l.append(R)
    return (torch.cat(ids_l), torch.cat(hand_l),
            torch.tensor(act_l), torch.tensor(lp_l), torch.tensor(r_l))


def train_ppo(mdp: FactorMDP, reward_fn: RewardFn, net: PPONet,
              n_rounds: int = 250, traj_per_round: int = 64, epochs: int = 4,
              clip: float = 0.2, ent_coef: float = 0.01, lr: float = 1e-3,
              seed: int = 0, log_every: int = 50) -> list[float]:
    """训练简化 PPO（张量化）。返回逐 round 的 mean loss。"""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    dev = next(net.parameters()).device
    losses: list[float] = []
    for r in range(n_rounds):
        ids, hand, acts, lp_old, R = _collect(mdp, reward_fn, net,
                                              traj_per_round, rng)
        ids, hand, acts = ids.to(dev), hand.to(dev), acts.to(dev)
        lp_old, R = lp_old.to(dev), R.to(dev)
        round_loss = []
        for _ in range(epochs):
            logits, v = net.forward_enc(ids, hand)
            dist = torch.distributions.Categorical(logits=logits)
            lp_new = dist.log_prob(acts)
            ratio = (lp_new - lp_old).exp()
            adv = R - v
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
            loss = -(torch.min(surr1, surr2) - ent_coef * dist.entropy()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            round_loss.append(float(loss.detach()))
        losses.append(float(np.mean(round_loss)))
        if log_every and (r + 1) % log_every == 0:
            log.info("[ppo] round %d/%d  loss=%.4f", r + 1, n_rounds, losses[-1])
    return losses
