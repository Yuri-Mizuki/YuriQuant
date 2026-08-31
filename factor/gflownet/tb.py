"""
Trajectory Balance 训练器（Phase 0）
===================================

研报对齐（系列之二十二 §2.1）：TB 目标

    L = ( log Z + Σ_t log P_F(s_t | s_{t-1}) − log R(x) )²

**Phase 0 简化（诚实标注）**：后向策略 P_B 取恒 1（均匀后退的常数简化），即
forward-only TB。其解仍满足 P(x) ∝ R(x)/Z（Z 吸收归一化常数），用于验证
「多样采样 + IC 非零」闭环足够；完整带 P_B 的 TB / SubTB 升级放到 Phase 1。

训练流程（研报 §2.1）：当前策略采样轨迹 → 计算奖励 R(x) → TB loss → 反向传播
更新 P_F 与 Z → 重复。
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from factor.gflownet.env import FactorMDP
from factor.gflownet.expr import ExprBuilder, canonical_formula
from factor.gflownet.net import TBPolicy, policy_logits

log = logging.getLogger(__name__)

__all__ = ["sample_trajectory_logp", "train_tb", "evaluate_samples", "RewardFn"]


RewardFn = Callable[[ExprBuilder], float]


def sample_trajectory_logp(net: nn.Module, mdp: FactorMDP,
                           reward_fn: RewardFn, rng: np.random.Generator,
                           compute_reward: bool = True):
    """按当前策略采样一条轨迹。

    Args:
        compute_reward: False 时 R 返回 None（配合并行 reward 池，采样后
            batch 统一并行求值）。

    Returns:
        (builder, logp_sum, R) —— logp_sum 为带梯度的 tensor（供 TB/PPO loss 反向传播）。
    """
    b = mdp.reset()
    logp_sum = torch.zeros(())
    for _ in range(mdp.max_len):
        if b.is_done():
            break
        logits = policy_logits(net, b, mdp)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        if not mdp.step(b, int(a)):
            raise RuntimeError(f"策略采样到非法动作 {int(a)}")
        logp_sum = logp_sum + dist.log_prob(a)
    R = reward_fn(b) if compute_reward else None
    return b, logp_sum, R


def train_tb(mdp: FactorMDP, reward_fn: RewardFn, net: TBPolicy,
             n_iters: int = 3000, batch_size: int = 16, lr: float = 1e-3,
             z_lr: float = 1e-2, seed: int = 0, log_every: int = 200,
             reward_pool=None, ckpt_path: str | None = None,
             resume: bool = False) -> list[float]:
    """训练 TB 策略。返回逐 iter 的 loss 序列。

    ``z_lr``：logZ 用**独立且更高**的学习率（GFlowNet 社区经验：Z 需跨数量级
    匹配奖励总量 logZ≈log(ΣR)，若与网络同 lr 会拖慢收敛，出现 loss 只升不降）。

    ``reward_pool``：可选的 ``RewardPool``（进程池并行求值）——batch 采样后
    公式统一并行算 reward，显著提速深树求值。

    ``ckpt_path``：每 ``log_every`` iters 保存 checkpoint（权重 + 迭代号），
    长训练（数小时）被外部中断时可 ``resume=True`` 续训，不从头再来。
    """
    start = 0
    if resume and ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(ckpt["model"])
        net.logZ.data = torch.tensor(float(ckpt["logz"]))
        start = int(ckpt["iter"])
        log.info("[resume] 从 iter %d 续训（ckpt=%s）", start, ckpt_path)

    net_params = [p for p in net.parameters() if p is not net.logZ]
    opt = torch.optim.Adam(net_params, lr=lr)
    opt_z = torch.optim.Adam([net.logZ], lr=z_lr)
    rng = np.random.default_rng(seed + start)
    losses: list[float] = []
    for it in range(start, n_iters):
        if reward_pool is not None:
            # 采样（不计算 reward）→ batch 统一并行求值
            items = []
            for _ in range(batch_size):
                b, logp_sum, _ = sample_trajectory_logp(
                    net, mdp, reward_fn, rng, compute_reward=False)
                items.append((canonical_formula(b), logp_sum))
            r_vals = reward_pool.compute([f for f, _ in items])
            terms = [net.logZ + lp - torch.log(torch.tensor(r))
                     for (_, lp), r in zip(items, r_vals)]
        else:
            terms, r_vals = [], []
            for _ in range(batch_size):
                _, logp_sum, R = sample_trajectory_logp(net, mdp, reward_fn, rng)
                terms.append(net.logZ + logp_sum - torch.log(torch.tensor(R)))
                r_vals.append(R)
        loss = torch.stack([t * t for t in terms]).mean()
        opt.zero_grad()
        opt_z.zero_grad()
        loss.backward()
        opt.step()
        opt_z.step()
        losses.append(float(loss.detach()))
        if log_every and (it + 1) % log_every == 0:
            log.info("[tb] iter %d/%d  loss=%.4f  logZ=%.2f  轨迹R均值=%.4f Rmax=%.4f",
                     it + 1, n_iters, loss.item(), net.logZ.item(),
                     np.mean(r_vals), np.max(r_vals))
            if ckpt_path:
                torch.save({"iter": it + 1, "model": net.state_dict(),
                            "logz": float(net.logZ.item())}, ckpt_path)
    return losses


def sample_formulas(net: nn.Module, mdp: FactorMDP, reward_fn: RewardFn,
                    n: int, seed: int = 0) -> list[tuple[str, float]]:
    """从策略采样 n 个因子，返回 [(formula, R)]（canonical 去重）。"""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    for _ in range(n):
        b, _, R = sample_trajectory_logp(net, mdp, reward_fn, rng)
        out[canonical_formula(b)] = float(R)
    return sorted(out.items(), key=lambda kv: kv[1], reverse=True)


def sample_uniform(mdp: FactorMDP, reward_fn: RewardFn, n: int,
                   seed: int = 0) -> list[tuple[str, float]]:
    """**均匀随机策略基线**：每个合法动作等概率采样。

    用途：TB 训练有效性的关键对照——mock 面板信号强时随机公式也有非零 IC，
    若不对比基线，「采样因子多样 + IC 非零」会被均匀策略轻易满足，无法证明
    P_F 学到了高 R 偏向。训练后 R 分布应显著优于基线。
    """
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    for _ in range(n):
        b = mdp.reset()
        while not b.is_done():
            legal = np.flatnonzero(mdp.legal_mask(b))
            mdp.step(b, int(rng.choice(legal)))
        out[canonical_formula(b)] = reward_fn(b)
    return sorted(out.items(), key=lambda kv: kv[1], reverse=True)


def _pairwise_spearman(panels: list) -> np.ndarray:
    """因子面板两两 spearman（面板 flatten 为向量，成对剔除 NaN）。"""
    arr = np.stack([p.to_numpy().flatten() for p in panels])
    n = len(arr)
    corrs = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = arr[i], arr[j]
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() < 200:
                continue
            ra = np.argsort(np.argsort(a[m]))
            rb = np.argsort(np.argsort(b[m]))
            corrs[i, j] = np.corrcoef(ra, rb)[0, 1]
    return corrs


def evaluate_samples(mdp: FactorMDP, reward_fn: RewardFn, samples: list[tuple[str, float]],
                     features: list[str], panel: dict, n_corr: int = 80,
                     seed: int = 0, nz_threshold: float = 1e-3) -> dict:
    """评估采样因子：IC 分布 / 多样性 / batch 内相关性。

    ``nz_threshold``：判断「有正 IC」的 R 阈值（线性奖励 1e-3；exp 温度奖励
    传 exp(1e-3/temp)）。
    """
    from factor.formula import formula_builder
    rng = np.random.default_rng(seed)
    top = samples[:n_corr]
    panels = []
    for formula, _ in top:
        fp = formula_builder(formula, features=features)(panel)
        panels.append(fp)
    corr = _pairwise_spearman(panels)
    corr_med = float(np.nanmedian(np.abs(corr)))

    ic_vals = [r for _, r in samples]
    n_unique = len(samples)
    return {
        "n_formulas": n_unique,
        "ic_mean_median": float(np.median(ic_vals)),
        "ic_mean_p25": float(np.percentile(ic_vals, 25)),
        "ic_mean_p75": float(np.percentile(ic_vals, 75)),
        "ic_mean_max": float(np.max(ic_vals)),
        "ic_nonzero_ratio": float(np.mean([r > nz_threshold for r in ic_vals])),
        "batch_corr_median_abs": corr_med,
        "top5": samples[:5],
    }
