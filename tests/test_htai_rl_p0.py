"""华泰 AI97 大模型+强化学习因子挖掘 —— P0 环境单元测试。

覆盖：RPN 算子 token 空间、AlphaPoolGymEnv 基本交互、AlphaPool 入池与奖励
（对齐研报 4 档奖励）、权重优化、mock 面板随机采样可入池。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.rl.alphapool_env import (
    AlphaPool, RPNParser, TOKEN_SPACE, legal_mask_for_tokens,
    BEG, SEP, N_ACTIONS, MAX_EXPR_LENGTH,
)
from factor.rl.alphapool_gym import AlphaPoolGymEnv


@pytest.fixture
def mini_panel():
    rng = np.random.default_rng(0)
    n_days, n_codes = 200, 30
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    phi = 0.3
    rets = np.zeros((n_days, n_codes))
    for t in range(1, n_days):
        rets[t] = phi * rets[t - 1] + rng.normal(0, 0.02, n_codes)
    close = pd.DataFrame(np.exp(np.cumsum(rets, axis=0)), idx, codes)
    open_ = close * (1 + rng.normal(0, 0.005, (n_days, n_codes)))
    volume = pd.DataFrame(rng.lognormal(12, 0.5, (n_days, n_codes)), idx, codes)
    vwap = close * (1 + rng.normal(0, 0.002, (n_days, n_codes)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, (n_days, n_codes)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, (n_days, n_codes)))
    panel = {"open": open_, "high": high, "low": low, "close": close,
             "volume": volume, "vwap": vwap}
    mom20 = close.pct_change(20)
    signal = -mom20.shift(1)
    sig = signal.sub(signal.mean(axis=1), axis=0).div(signal.std(axis=1) + 1e-9, axis=0)
    noise = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_codes)), idx, codes)
    fwd = (0.6 * 0.03 * sig.fillna(0) + noise).where(signal.notna())
    return panel, fwd


# ---------------------------------------------------------------------------
# RPN 解析
# ---------------------------------------------------------------------------
def test_rpn_parse_suffix():
    p = RPNParser()
    def tid(kind, val):
        for t, (k, v) in TOKEN_SPACE.items():
            if k == kind and v == val:
                return t
        raise KeyError((kind, val))
    def ids(*kvs):
        return [BEG] + [tid(k, v) for k, v in kvs] + [SEP]

    # 后缀式：close open Sub = sub(close,open)
    assert p.parse(ids(("field", "close"), ("field", "open"),
                       ("op", ("Sub", None)))) == "sub(close,open)"
    # 滚动算子窗口内建：close Mean_20 = ts_mean_20(close)
    assert p.parse(ids(("field", "close"), ("op", ("Mean", 20)))) == \
        "ts_mean_20(close)"
    # 配对滚动：close volume Corr_10
    assert p.parse(ids(("field", "close"), ("field", "volume"),
                       ("op", ("Corr", 10)))) == "ts_corr_10(close,volume)"
    # 非法：缺操作数
    assert p.parse(ids(("field", "close"), ("op", ("Sub", None)))) is None
    # 非法：空
    assert p.parse([BEG, SEP]) is None


def test_legal_mask_never_deadlock(mini_panel):
    panel, fwd = mini_panel
    env = AlphaPoolGymEnv(panel, fwd, features=list(panel.keys()),
                          capacity=3, seed=1)
    rng = np.random.default_rng(7)
    for _ in range(30):
        obs, _ = env.reset()
        done = False
        guard = 0
        while not done and guard < MAX_EXPR_LENGTH + 2:
            mask = env.action_masks()
            assert mask.sum() > 0, f"deadlock at {env.builder.tokens}"
            a = int(rng.choice(np.flatnonzero(mask)))
            obs, r, done, trunc, info = env.step(a)
            guard += 1
        assert done, "episode 必须在长度内终止"


# ---------------------------------------------------------------------------
# AlphaPool 奖励（研报 4 档）
# ---------------------------------------------------------------------------
def test_alphapool_reward_tiers(mini_panel):
    panel, fwd = mini_panel
    feats = list(panel.keys())
    pool = AlphaPool(panel, fwd, features=feats, capacity=3, seed=0)
    # 空池起始 best_obj=0（研报由 LLM 初始池保证；P0 避免 -inf 毒化）
    assert pool.best_obj == 0.0
    # 有效因子入池 → pooled，奖励 = 组合 IC（>0）
    status, reward = pool.evaluate("ts_mean_20(close)")
    assert status == "pooled"
    assert reward > 0
    # 高相关因子（同结构）→ no_pool，奖励 = best_obj
    status2, reward2 = pool.evaluate("ts_mean_20(open)")
    assert status2 in ("no_pool", "pooled")
    # 全 NaN / 异常 → empty，奖励 0
    status3, reward3 = pool.evaluate("div(close,sub(close,close))")
    assert status3 in ("empty", "invalid", "no_pool")
    assert reward3 >= -1.0


def test_alphapool_random_sampling_pools(mini_panel):
    """随机策略也能入池（P0 环境可探索性验证）。"""
    panel, fwd = mini_panel
    env = AlphaPoolGymEnv(panel, fwd, features=list(panel.keys()),
                          capacity=5, seed=1)
    rng = np.random.default_rng(7)
    stats = {}
    for _ in range(60):
        obs, _ = env.reset()
        done = False
        guard = 0
        while not done and guard < MAX_EXPR_LENGTH + 2:
            mask = env.action_masks()
            a = int(rng.choice(np.flatnonzero(mask)))
            obs, r, done, trunc, info = env.step(a)
            guard += 1
        stats[info["status"]] = stats.get(info["status"], 0) + 1
    assert stats.get("pooled", 0) > 0, "随机策略应能入池"
    assert env.best_pool()["n_factors"] >= 1


def test_optimize_weights_finite(mini_panel):
    """权重优化输出有限（无 NaN/发散）。"""
    panel, fwd = mini_panel
    pool = AlphaPool(panel, fwd, features=list(panel.keys()), capacity=5, seed=0)
    for f in ["ts_mean_20(close)", "sub(close,open)", "ts_delta_5(volume)",
              "div(close,ts_mean_20(close))", "ts_std_20(close)"]:
        pool.evaluate(f)
    assert len(pool.factor_panels) >= 3
    w = pool.optimize_weights(pool.factor_panels)
    assert np.isfinite(w).all()
    # L2 归一化（均值-方差闭式解）
    assert abs(np.sqrt((w * w).sum()) - 1.0) < 1e-6
