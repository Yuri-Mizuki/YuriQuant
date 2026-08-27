"""
GFlowNet Phase 0 最小闭环：运行入口
==================================

研报参照：国金证券《Alpha掘金系列之二十二》（2026-04-10）§2。

流程：
1. mock AR(1) 信号注入面板（60 股 × 300 天，含 open/high/low/close/volume/amount）
2. 简化算子子集（22 个，覆盖一元/二元/时序/截面/时序二元）+ 4 窗口 + 6 特征
3. TB 训练（Trajectory Balance，forward-only 简化，奖励缓存）
4. 从 P_F 采样 200 因子：IC 分布 / 多样性 / batch 内 spearman 相关性中位数
5. PPO-lite 对照训练，对比 batch 内相关性（预期模式崩溃）

验收标准（Phase 0）：
- GFlowNet batch 内 |corr| 中位数 < 0.2（研报全配置 < 0.04）
- IC 非零因子占比 > 50%
- PPO 对照 batch 内相关性显著高于 GFlowNet

用法（系统 python 3.12，含 torch）：
    cd E:/YuriQuant && D:/python/Python312/python.exe scripts/run_gflownet_phase0.py
    D:/python/Python312/python.exe scripts/run_gflownet_phase0.py --iters 4000 --ppo-rounds 400
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

# 常数因子行 std=0 产生的无害 RuntimeWarning（结果已 isfinite 过滤）
warnings.filterwarnings("ignore", message="invalid value encountered in reduce")

from factor.gflownet.env import FactorMDP
from factor.gflownet.net import PPONet, TBPolicy
from factor.gflownet.ppo import train_ppo
from factor.gflownet.reward import RewardCache, make_reward_fn
from factor.gflownet.tb import evaluate_samples, sample_formulas, sample_uniform, train_tb

# ---------------------------------------------------------------------------
# mock 面板（对齐 tests/test_genetic_mining.py 的 AR(1) 惯例，加强信号）
# ---------------------------------------------------------------------------
def make_mock_panel(n_days: int = 300, n_codes: int = 60, seed: int = 7,
                    phi: float = 0.3) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """AR(1) 收益注入动量信号（可被 ts_rank/ts_delta 类动量因子捕获）。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    # 收益：个股自相关（动量）+ 截面公共因子 + 噪声
    common = rng.normal(0, 0.005, n_days)
    rets = np.zeros((n_days, n_codes))
    for t in range(1, n_days):
        rets[t] = phi * rets[t - 1] + common[t] + rng.normal(0, 0.015, n_codes)
    close = pd.DataFrame(np.exp(np.cumsum(rets, axis=0)), idx, codes)
    # 由 close 构造 OHLCV
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.003,
                                                                   (n_days, n_codes)))
    high = pd.concat([open_, close], axis=0).groupby(level=0).max() * \
        (1 + rng.uniform(0, 0.004, (n_days, n_codes)))
    low = pd.concat([open_, close], axis=0).groupby(level=0).min() * \
        (1 - rng.uniform(0, 0.004, (n_days, n_codes)))
    volume = pd.DataFrame(rng.lognormal(12, 0.5, (n_days, n_codes)), idx, codes)
    amount = volume * close / 1000
    panel = {"open": open_, "high": high, "low": low,
             "close": close, "volume": volume, "amount": amount}
    returns = close.pct_change().shift(-1)          # 次日收益（factor[t] ↔ ret[t+1]）
    return panel, returns


# 算子子集：覆盖一元/二元/时序一元/截面/时序二元（研报 §2.2 类别的精简版）
OP_NAMES = [
    # element unary
    "abs", "sign", "log", "sqrt", "inv", "reverse",
    # element binary
    "add", "sub", "mul", "div",
    # ts unary (needs window)
    "ts_mean", "ts_std", "ts_rank", "ts_delta", "ts_ema", "ts_max", "ts_min",
    # cs
    "cs_rank", "cs_zscore", "cs_demean",
    # ts binary
    "ts_corr", "ts_cov",
]
WINDOWS = (5, 10, 20, 60)
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2500, help="TB 训练迭代数")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ppo-rounds", type=int, default=250, help="PPO 训练 round 数")
    ap.add_argument("--ppo-traj", type=int, default=64)
    ap.add_argument("--samples", type=int, default=200, help="评估采样因子数")
    ap.add_argument("--temp", type=float, default=None,
                    help="奖励温度：None=线性 R=max(|IC|,eps)（研报口径）；数值=exp 锐化 R=exp(|IC|/temp)")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-nodes", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    panel, returns = make_mock_panel()
    mdp = FactorMDP(OP_NAMES, WINDOWS, FEATURES,
                    max_depth=args.max_depth, max_nodes=args.max_nodes)
    print(f"[phase0] ops={mdp.n_op} win={mdp.n_win} feat={mdp.n_feat} "
          f"n_actions={mdp.n_actions} 面板={panel['close'].shape}")

    cache = RewardCache()
    reward_fn = make_reward_fn(panel, returns, FEATURES, cache=cache,
                               temp=args.temp)
    # 「有正 IC」的 R 阈值（线性=1e-3；exp=exp(1e-3/temp)）
    nz_thr = 1e-3 if args.temp is None else float(np.exp(1e-3 / args.temp))

    # ---- 均匀策略基线（TB 有效性对照：mock 信号强时随机公式也有 IC） ----
    print("\n=== 均匀随机策略基线（训练前对照） ===")
    base = sample_uniform(mdp, reward_fn, args.samples, seed=args.seed)
    base_rs = [r for _, r in base]
    print(f"  基线 R 均值={np.mean(base_rs):.4f} 中位数={np.median(base_rs):.4f} "
          f"top1={base_rs[0]:.4f}")

    # ---- TB ----
    tb_net = TBPolicy(mdp.n_actions)
    print("\n=== TB 训练 ===")
    train_tb(mdp, reward_fn, tb_net, n_iters=args.iters, batch_size=args.batch,
             seed=args.seed)
    print(f"奖励缓存命中规模: {len(cache)}")

    samples = sample_formulas(tb_net, mdp, reward_fn, args.samples, seed=args.seed)
    ev = evaluate_samples(mdp, reward_fn, samples, FEATURES, panel, n_corr=80,
                          nz_threshold=nz_thr)
    tb_rs = [r for _, r in samples]
    print("\n=== GFlowNet 采样因子评估（Phase 0 验收） ===")
    for k in ("n_formulas", "ic_mean_median", "ic_mean_p25", "ic_mean_p75",
              "ic_mean_max", "ic_nonzero_ratio", "batch_corr_median_abs"):
        print(f"  {k}: {ev[k]:.4f}" if isinstance(ev[k], float) else f"  {k}: {ev[k]}")
    print(f"  TB R 均值={np.mean(tb_rs):.4f} 中位数={np.median(tb_rs):.4f} "
          f"top1={tb_rs[0]:.4f}  (基线 均值={np.mean(base_rs):.4f} 中位数={np.median(base_rs):.4f})")
    print("  top5:")
    for f, r in ev["top5"]:
        print(f"    {f}   R={r:.4f}")

    # ---- PPO 对照 ----
    ppo_net = PPONet(mdp.n_actions)
    print("\n=== PPO-lite 对照训练 ===")
    train_ppo(mdp, reward_fn, ppo_net, n_rounds=args.ppo_rounds,
              traj_per_round=args.ppo_traj, seed=args.seed)
    ppo_samples = sample_formulas(ppo_net, mdp, reward_fn, args.samples,
                                  seed=args.seed + 1)
    ppo_ev = evaluate_samples(mdp, reward_fn, ppo_samples, FEATURES, panel,
                              n_corr=80, seed=args.seed + 1,
                              nz_threshold=nz_thr)
    print("\n=== PPO 对照评估 ===")
    print(f"  batch_corr_median_abs: {ppo_ev['batch_corr_median_abs']:.4f}")
    print(f"  去重后公式数: {ppo_ev['n_formulas']}")

    print("\n=== 结论 ===")
    tb_corr = ev["batch_corr_median_abs"]
    ppo_corr = ppo_ev["batch_corr_median_abs"]
    base_med = float(np.median(base_rs))
    tb_med = float(np.median(tb_rs))
    base_mean = float(np.mean(base_rs))
    tb_mean = float(np.mean(tb_rs))
    # 验收：多样 + IC 非零 + TB 优于均匀基线（一阶随机占优用均值；P(x)∝R(x) 的
    # 中位数 ≈ R 中位数是线性奖励下的数学必然，不构成验收） + PPO 崩溃对照
    ok = (tb_corr < 0.2 and ev["ic_nonzero_ratio"] > 0.5 and ppo_corr > tb_corr
          and tb_mean > base_mean)
    print(f"  TB batch 内 |corr| 中位数: {tb_corr:.4f} (验收 < 0.2)")
    print(f"  IC 非零占比: {ev['ic_nonzero_ratio']:.2%} (验收 > 50%)")
    print(f"  TB R 均值 vs 均匀基线: {tb_mean:.4f} vs {base_mean:.4f} (验收 TB 更优, +{100*(tb_mean/base_mean-1):.1f}%)")
    print(f"  TB R 中位数 vs 基线: {tb_med:.4f} vs {base_med:.4f} (参考)")
    print(f"  PPO batch 内 |corr| 中位数: {ppo_corr:.4f} (预期显著高于 TB；去重后 {ppo_ev['n_formulas']} 个公式)")
    print(f"  耗时: {time.time() - t0:.0f}s   结果: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
