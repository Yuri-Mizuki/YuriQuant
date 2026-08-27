"""GFlowNet Phase 0 最小闭环测试。

覆盖：canonical 简化（交换律/neg 折叠）、MDP 终止性、canonical 兼容 formula.py、
TB 训练 loss 下降、采样多样性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from factor.gflownet.env import FactorMDP
from factor.gflownet.expr import ExprBuilder, canonical_formula
from factor.gflownet.net import TBPolicy
from factor.gflownet.reward import RewardCache, make_reward_fn
from factor.gflownet.tb import sample_formulas, train_tb

OPS = ["abs", "sign", "log", "sqrt", "add", "sub", "mul", "div",
       "ts_mean", "ts_rank", "ts_delta", "cs_rank", "cs_zscore", "ts_corr"]
WINS = (5, 10, 20)
FEATS = ["close", "volume"]


@pytest.fixture
def mini_panel():
    rng = np.random.default_rng(0)
    n_days, n_codes = 150, 20
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    phi = 0.3
    rets = np.zeros((n_days, n_codes))
    for t in range(1, n_days):
        rets[t] = phi * rets[t - 1] + rng.normal(0, 0.02, n_codes)
    close = pd.DataFrame(np.exp(np.cumsum(rets, axis=0)), idx, codes)
    volume = pd.DataFrame(rng.lognormal(12, 0.5, (n_days, n_codes)), idx, codes)
    return {"close": close, "volume": volume}, close.pct_change().shift(-1)


def test_commutative_canonical_same():
    """add(close, volume) 与 add(volume, close) canonical 相同（交换律排序）。"""
    mdp = FactorMDP(["add", "mul"], WINS, FEATS)
    def mk(order):
        b = mdp.reset()
        mdp.step(b, mdp._op0)                 # add
        mdp.step(b, mdp._feat0 + FEATS.index(order[0]))
        mdp.step(b, mdp._feat0 + FEATS.index(order[1]))
        return canonical_formula(b)
    assert mk(("close", "volume")) == mk(("volume", "close"))


def test_neg_fold_canonical():
    """reverse(reverse(close)) -> close（双重 neg 折叠）。"""
    mdp = FactorMDP(["reverse", "ts_mean"], WINS, FEATS)
    b = mdp.reset()
    mdp.step(b, mdp._op0)                     # reverse
    mdp.step(b, mdp._op0)                     # reverse 内层
    mdp.step(b, mdp._feat0 + FEATS.index("close"))
    assert canonical_formula(b) == "close"


def test_canonical_parseable(mini_panel):
    """随机轨迹的 canonical 都能被 formula.py 解析求值。"""
    from factor.formula import formula_builder
    panel, rets = mini_panel
    mdp = FactorMDP(OPS, WINS, FEATS)
    rng = np.random.default_rng(1)
    cache = RewardCache()
    reward_fn = make_reward_fn(panel, rets, FEATS, cache=cache)
    # 随机均匀策略采样（每个合法动作等概率，走精确掩码）
    for _ in range(30):
        b = mdp.reset()
        while not b.is_done():
            legal = np.flatnonzero(mdp.legal_mask(b))
            a = int(rng.choice(legal))
            mdp.step(b, a)
        f = canonical_formula(b)
        fp = formula_builder(f, features=FEATS)(panel)
        assert fp is not None and fp.shape == panel["close"].shape
        r = reward_fn(b)
        assert r > 0


# ---------------------------------------------------------------------------
# MDP 终止性
# ---------------------------------------------------------------------------
def test_env_terminates_within_max_len():
    """任意合法动作序列必在 max_len 内终止且根非空。"""
    mdp = FactorMDP(OPS, WINS, FEATS, max_depth=3, max_nodes=9)
    rng = np.random.default_rng(2)
    for _ in range(50):
        b = mdp.reset()
        steps = 0
        while not b.is_done() and steps < mdp.max_len:
            legal = np.flatnonzero(mdp.legal_mask(b))
            mdp.step(b, int(rng.choice(legal)))
            steps += 1
        assert b.is_done(), "轨迹未在 max_len 内终止"
        assert b.root is not None
        assert b.node_count <= b.max_nodes


# ---------------------------------------------------------------------------
# TB 训练
# ---------------------------------------------------------------------------
def test_tb_loss_decreases(mini_panel):
    """TB loss 训练后显著下降（短训练）。"""
    panel, rets = mini_panel
    mdp = FactorMDP(OPS, WINS, FEATS)
    reward_fn = make_reward_fn(panel, rets, FEATS, cache=RewardCache())
    torch.manual_seed(0)
    net = TBPolicy(mdp.n_actions)
    losses = train_tb(mdp, reward_fn, net, n_iters=120, batch_size=4,
                      log_every=0)
    assert losses[-1] < losses[0] * 0.7, f"loss 未下降: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_sampling_diverse(mini_panel):
    """短训练后采样因子多样（去重率高、batch 内相关性宽松阈值）。"""
    from factor.gflownet.tb import evaluate_samples
    panel, rets = mini_panel
    mdp = FactorMDP(OPS, WINS, FEATS)
    reward_fn = make_reward_fn(panel, rets, FEATS, cache=RewardCache())
    torch.manual_seed(1)
    net = TBPolicy(mdp.n_actions)
    train_tb(mdp, reward_fn, net, n_iters=100, batch_size=4, log_every=0)
    samples = sample_formulas(net, mdp, reward_fn, 30, seed=3)
    ev = evaluate_samples(mdp, reward_fn, samples, FEATS, panel, n_corr=30)
    assert ev["batch_corr_median_abs"] < 0.5
    assert ev["ic_nonzero_ratio"] > 0.3
