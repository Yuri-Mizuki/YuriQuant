"""stage2 组合优化 env（factor/rl/portfolio_env.py，Phase 0）单元测试。

银河 0706 口径的确定性验证（合成面板，不触真实数据）：
- 权重映射：归一/非负/持仓数上限、active_share=0 退化基准、极端动作集中、确定性；
- 奖励六项：符号方向（超配高信号 FR>0、负风险暴露受罚、TE 超阈才罚、
  换手线性罚、集中度超阈才罚）+ total 逐项精确复算；
- Banach 投影：闭式解一致、成本 0 退化、单调性；
- 观测：维度 8N+14、NaN 中性化、偏离块逐位正确；
- env：reset/step 形状、终止时点、确定性轨迹、成本臂 vs 免费臂、入参校验。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.rl.portfolio_env import (
    PortfolioEnv,
    banach_projected_value,
    build_observation,
    map_actions_to_weights,
    observation_dim,
    reward_components,
)


N = 10
N_STEPS = 8


def _synthetic(seed: int = 3):
    """合成面板：决策日 = 连续交易日；期间收益由 t 日因子驱动（可测性）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=N_STEPS + 1)
    codes = [f"c{i}" for i in range(N)]
    f = pd.DataFrame(rng.normal(0, 1, (len(dates), N)), dates, codes)
    z = pd.DataFrame(rng.normal(0, 1, (len(dates), N)), dates, codes)
    bench = pd.DataFrame(
        rng.dirichlet(np.ones(N) * 3.0, size=len(dates)), dates, codes)
    rets = pd.DataFrame(0.0, dates, codes)
    for i in range(len(dates) - 1):
        rets.iloc[i + 1] = 0.01 * f.iloc[i].to_numpy() \
            + rng.normal(0, 0.005, N)
    idx_ret = rets.mean(axis=1)
    return dict(factor=f, aux=z, bench=bench, period_returns=rets,
                idx_period_returns=idx_ret, decision_dates=list(dates))


def _env(**kw):
    data = _synthetic()
    kw.setdefault("cost_rate", 0.0)
    return PortfolioEnv(data["factor"], data["aux"], data["bench"],
                        data["period_returns"], data["idx_period_returns"],
                        data["decision_dates"], **kw), data


# ===========================================================================
# 权重映射
# ===========================================================================
def test_weights_valid_and_holding_cap():
    rng = np.random.default_rng(0)
    bench = rng.dirichlet(np.ones(N))
    w = map_actions_to_weights(rng.normal(0, 2, N), bench)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert (w >= 0).all()
    assert (w > 1e-12).sum() <= max(1, round(N * 0.4))     # 持仓数上限


def test_weights_active_share_zero_reproduces_bench():
    rng = np.random.default_rng(1)
    bench = rng.dirichlet(np.ones(N))
    w = map_actions_to_weights(rng.normal(0, 3, N), bench,
                               active_share=0.0, max_holding_pct=1.0)
    assert np.allclose(w, bench, atol=1e-9)


def test_weights_extreme_actions_concentrate():
    bench = np.full(N, 1.0 / N)
    actions = np.concatenate([[100.0, -100.0], np.zeros(N - 2)])
    w = map_actions_to_weights(actions, bench, max_holding_pct=1.0)
    # 极端动作下 softmax 项 ≈ 独占，但 active_share=0.7 与均匀基准混合：
    # w[0] ≈ 0.3·(1/N) + 0.7·1 ≈ 0.73 —— 主动上限本身就是集中度天花板
    assert w[0] > 0.7 and w[1] < 0.04
    assert w[0] > w[2:].sum()


def test_weights_higher_temp_flatter():
    rng = np.random.default_rng(2)
    actions = rng.normal(0, 1, N)
    w_low = map_actions_to_weights(actions, np.full(N, 1 / N),
                                   softmax_temp=0.1, max_holding_pct=1.0)
    w_high = map_actions_to_weights(actions, np.full(N, 1 / N),
                                    softmax_temp=10.0, max_holding_pct=1.0)
    assert (w_low.max() - w_low.min()) > (w_high.max() - w_high.min())


def test_weights_deterministic_and_bench_nan_safe():
    rng = np.random.default_rng(4)
    bench = rng.dirichlet(np.ones(N))
    bench[:3] = np.nan                                     # 缺失基准票不占预算
    a = rng.normal(0, 1, N)
    w1 = map_actions_to_weights(a, bench, max_holding_pct=1.0)
    w2 = map_actions_to_weights(a, bench, max_holding_pct=1.0)
    assert np.array_equal(w1, w2)
    assert np.isfinite(w1).all() and w1.sum() == pytest.approx(1.0)


# ===========================================================================
# 奖励六项
# ===========================================================================
def _reward_ctx(seed=5):
    rng = np.random.default_rng(seed)
    bench = rng.dirichlet(np.ones(N))
    w = rng.dirichlet(np.ones(N) * 5.0)
    ret = rng.normal(0, 0.02, N)
    f = rng.normal(0, 1, N)
    z = rng.normal(0, 1, N)
    return w, bench, ret, f, z


def test_reward_fr_sign_follows_signal():
    w, bench, ret, f, z = _reward_ctx()
    # 超配 f 最高的一半 → FR 必为正
    order = np.argsort(f)
    w_signal = np.zeros(N)
    w_signal[order[N // 2:]] = 1.0
    w_signal /= w_signal.sum()
    kw = dict(lambda_excess=0.0, lambda_factor=1.0, lambda_aux=0.0,
              lambda_tracking=0.0, lambda_turnover=0.0)
    assert reward_components(w_signal, w_signal, bench, ret, f, z,
                             **kw)["fr"] > 0


def test_reward_risk_penalty_one_sided():
    w, bench, ret, f, z = _reward_ctx()
    d = w - bench
    z_aligned = z * np.sign(d)                             # 保证 d⊤z>0
    kw = dict(lambda_excess=0.0, lambda_factor=0.0, lambda_aux=2.0,
              lambda_tracking=0.0, lambda_turnover=0.0)
    r_pos = reward_components(w, w, bench, ret, f, z_aligned, **kw)
    assert r_pos["risk_penalty"] == 0.0                    # 低风险暴露不受奖
    r_neg = reward_components(w, w, bench, ret, f, -z_aligned, **kw)
    assert r_neg["risk_penalty"] < 0.0                     # 高风险暴露受罚（单侧）


def test_reward_te_penalized_only_beyond_target():
    w, bench, ret, f, z = _reward_ctx()
    kw = dict(lambda_excess=0.0, lambda_factor=0.0, lambda_aux=0.0,
              lambda_tracking=0.01, target_te=0.9, lambda_turnover=0.0)
    assert reward_components(w, w, bench, ret, f, z, **kw)["te_penalty"] == 0.0
    kw["target_te"] = 0.05
    assert reward_components(w, w, bench, ret, f, z, **kw)["te_penalty"] < 0.0


def test_reward_turnover_linear_and_total_exact():
    w, bench, ret, f, z = _reward_ctx()
    w_prev = np.roll(w, 2)
    kw = dict(lambda_excess=2.0, lambda_factor=1.0, lambda_aux=2.0,
              lambda_tracking=0.01, target_te=0.15,
              lambda_turnover=0.01, lambda_conc=0.5, conc_target=0.4)
    r = reward_components(w, w_prev, bench, ret, f, z, **kw)
    assert r["turnover_penalty"] == pytest.approx(-0.01 * np.abs(w - w_prev).sum())
    expected = (2.0 * r["er"] + 1.0 * r["fr"] + r["risk_penalty"]
                + r["te_penalty"] + r["turnover_penalty"] + r["conc_penalty"])
    assert r["total"] == pytest.approx(expected)


def test_reward_conc_penalty_triggers():
    _, bench, ret, f, z = _reward_ctx()
    w = np.zeros(N)
    w[:3] = [0.5, 0.3, 0.2]                                # 前 3 只 = 100% > 阈值
    r = reward_components(w, w, bench, ret, f, z,
                          lambda_excess=0.0, lambda_factor=0.0,
                          lambda_aux=0.0, lambda_tracking=0.0,
                          lambda_turnover=0.0, lambda_conc=1.0,
                          conc_top=5, conc_target=0.4)
    assert r["conc_penalty"] < 0.0


# ===========================================================================
# Banach 投影
# ===========================================================================
def test_banach_zero_cost_is_identity():
    assert banach_projected_value(1.0, 0.5, 0.0) == pytest.approx(1.0)


def test_banach_matches_closed_form():
    for c in (0.001, 0.01, 0.05):
        v = banach_projected_value(1.234, 0.37, c)
        assert v == pytest.approx(1.234 / (1.0 + c * 0.37), rel=1e-10)


def test_banach_monotone_in_cost():
    vals = [banach_projected_value(1.0, 0.6, c) for c in (0.0, 0.001, 0.01, 0.1)]
    assert all(vals[i + 1] <= vals[i] + 1e-15 for i in range(len(vals) - 1))


# ===========================================================================
# 观测
# ===========================================================================
def test_observation_dim_and_nan_neutral():
    rng = np.random.default_rng(6)
    f = rng.normal(0, 1, N)
    f[0] = np.nan
    obs = build_observation(
        f_row=f, z_row=rng.normal(0, 1, N), bench=rng.dirichlet(np.ones(N)),
        prev_w=rng.dirichlet(np.ones(N) * 5),
        hist_factor_mean=np.zeros(N), hist_aux_mean=np.zeros(N),
        hist_ret_mean=np.zeros(N), hist_ret_vol=np.zeros(N),
        portfolio_value=1.0, turnover=0.1, excess_ret=0.01,
        idx_ret_curr=0.001, idx_ret_mean=0.0005, idx_ret_vol=0.01,
    )
    assert obs.shape == (observation_dim(N),)
    assert np.isfinite(obs).all()
    assert obs[0] == 0.0                                   # NaN 信号中性化


def test_observation_deviation_block_exact():
    rng = np.random.default_rng(7)
    bench = rng.dirichlet(np.ones(N))
    prev = rng.dirichlet(np.ones(N) * 5)
    obs = build_observation(
        f_row=rng.normal(0, 1, N), z_row=rng.normal(0, 1, N), bench=bench,
        prev_w=prev,
        hist_factor_mean=np.zeros(N), hist_aux_mean=np.zeros(N),
        hist_ret_mean=np.zeros(N), hist_ret_vol=np.zeros(N),
        portfolio_value=1.0, turnover=0.0, excess_ret=0.0,
        idx_ret_curr=0.0, idx_ret_mean=0.0, idx_ret_vol=0.0,
    )
    assert np.allclose(obs[3 * N:4 * N], prev - bench, atol=1e-9)


# ===========================================================================
# PortfolioEnv
# ===========================================================================
def test_env_reset_shapes_and_init_to_bench():
    env, data = _env()
    obs, info = env.reset()
    assert obs.shape == (observation_dim(N),)
    assert obs.dtype == np.float32
    assert (env._prev_w > 0).sum() == N                    # 初始=基准
    assert np.allclose(env._prev_w,
                       data["bench"].iloc[0].to_numpy(), atol=1e-9)


def test_env_full_episode_deterministic():
    rng = np.random.default_rng(8)
    actions = [rng.uniform(-5, 5, N) for _ in range(N_STEPS)]

    def run():
        env, _ = _env()
        env.reset()
        out = []
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(actions[env._i - 1 if env._i else 0]
                                                 if False else rng_actions.pop(0))
            out.append((r, info["portfolio_value"]))
            done = term or trunc
        return out

    rng_actions = [a.copy() for a in actions]
    run1 = run()
    rng_actions = [a.copy() for a in actions]
    run2 = run()
    assert run1 == run2
    assert len(run1) == N_STEPS                            # 决策日 9 个 → 8 步
    assert all(np.isfinite(r) for r, _ in run1)


def test_env_terminates_at_last_decision():
    env, _ = _env()
    env.reset()
    for i in range(N_STEPS - 1):
        _obs, _r, term, _tr, _info = env.step(np.zeros(N))
        assert not term
    _obs, _r, term, _tr, info = env.step(np.zeros(N))
    assert term and info.get("terminal") is True


def test_env_cost_arm_value_not_higher_than_free_arm():
    rng = np.random.default_rng(9)
    actions = [rng.uniform(-5, 5, N) for _ in range(N_STEPS)]

    def run(cost):
        env, _ = _env(cost_rate=cost)
        env.reset()
        vals = []
        for a in actions:
            _obs, _r, term, _tr, info = env.step(a)
            vals.append(info["portfolio_value"])
            if term:
                break
        return vals[-1]

    assert run(0.01) <= run(0.0) + 1e-12


def test_env_info_carries_components_and_valid_weights():
    env, _ = _env()
    env.reset()
    _obs, _r, _term, _tr, info = env.step(np.linspace(-2, 2, N))
    assert {"weights", "reward", "portfolio_value", "date"} <= set(info)
    assert info["weights"].sum() == pytest.approx(1.0, abs=1e-9)
    assert set(info["reward"]) == {"er", "fr", "risk_penalty", "te_penalty",
                                   "turnover_penalty", "conc_penalty",
                                   "total", "te", "aux_exposure", "turnover"}


def test_env_init_equal_weights_option():
    env, _ = _env(init_to_bench=False)
    env.reset()
    assert np.allclose(env._prev_w, 1.0 / N)


def test_env_input_validation():
    data = _synthetic()
    with pytest.raises(ValueError, match="≥ 2"):
        PortfolioEnv(data["factor"], data["aux"], data["bench"],
                     data["period_returns"], data["idx_period_returns"],
                     data["decision_dates"][:1])
    with pytest.raises(ValueError, match="缺决策日"):
        PortfolioEnv(data["factor"], data["aux"], data["bench"].iloc[1:],
                     data["period_returns"], data["idx_period_returns"],
                     data["decision_dates"])
    with pytest.raises(ValueError, match="缺期间末端日"):
        PortfolioEnv(data["factor"], data["aux"], data["bench"],
                     data["period_returns"].iloc[:-1],
                     data["idx_period_returns"],
                     data["decision_dates"])
