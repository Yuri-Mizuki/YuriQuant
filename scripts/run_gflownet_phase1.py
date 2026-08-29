"""
GFlowNet Phase 1：真实 HS300 对齐研报
====================================

研报对齐项（国金《Alpha掘金系列之二十二》）：
- §2.2 算子空间：51 算子（已全覆盖，`factor/operators.py` 现有 69 个）
- §2.2 ExprNode 简化：交换律排序 + neg 折叠（Phase 0 已实现）
- §2.3 奖励：**市值中性化后的 |IC|** + **10 日调仓**
- §2.3 数据切分：训练段 / 测试段（样本外），测试段筛选 **spearman 相关 < 0.4**

数据：真实 HS300 后复权日线（2019-01 ~ 2026-07，缓存）；训练 2019-2024 / 测试 2025-2026。

用法（需安装 torch 的解释器 + AmazingData）：
    cd <仓库根>
    python -u scripts/run_gflownet_phase1.py --iters 600 --batch 12
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="invalid value encountered in reduce")

from factor.gflownet.env import FactorMDP
from factor.gflownet.net import TBPolicy
from factor.gflownet.reward import RewardCache, make_reward_fn
from factor.gflownet.selection import select_low_corr
from factor.gflownet.tb import evaluate_samples, sample_formulas, sample_uniform, train_tb

# ---------------------------------------------------------------------------
# 真实面板（复用 data.cache_helpers.build_panel 的统一数据管道）
# ---------------------------------------------------------------------------
def build_real_panel(begin: int, end: int, cache_root: str | None = None,
                     sdk_cache: str | None = None, offline: bool = False):
    """构建真实 HS300 面板（**历史成分并集池 + membership mask，消除幸存者偏差**）。

    2026-08-17 重构：收敛到 ``data.cache_helpers.build_panel``（与
    ``scripts/mine_factors.build_real_panel`` 共享同一实现），消除两套 PIT 并集池 /
    复权 / 财务 PIT / 市值面板并行逻辑。

    Returns:
        (panel, close, market_cap, mask) —— close/market_cap/panel 均已应用 mask。
    """
    from data.cache_helpers import build_panel as _build_panel

    cfg = {"universe": {"index_code": "000300.SH", "adjust": "backward"}}
    panel, _returns = _build_panel(
        cfg, begin, end,
        cache_root=cache_root, sdk_cache=sdk_cache, offline=offline,
        include_market_cap=True, retry=True,
    )
    mask = panel["mask"]
    daily = int(mask.sum(axis=1).median())
    print(f"membership mask: 每日在册中位数 {daily} 只（{mask.shape}）")
    return panel, panel["close_m"], panel["market_cap"], mask


# 研报图表6 的 51 算子全集（Phase 1 对齐口径）
OP_NAMES = [
    "abs", "neg", "sign", "log", "inv", "sqrt", "signed_power2", "signed_power3",
    "ts_mean", "ts_std", "ts_max", "ts_min", "ts_rank", "ts_skew", "ts_kurt",
    "ts_median", "ts_delay", "ts_delta", "ts_pct_change", "ts_sum", "ts_argmax",
    "ts_argmin", "ts_decay_linear", "ts_var", "ts_mad", "ts_count", "ts_ema",
    "ts_wma", "ts_slope", "ts_rsquare", "ts_residual", "ts_quantile",
    "add", "sub", "mul", "div", "max2", "min2", "greater", "less",
    "ts_corr", "ts_cov", "ts_beta", "ts_orth",
    "cs_rank", "cs_zscore", "cs_demean", "cs_scale", "cs_normalize",
    "cs_winsorize", "cs_truncate",
]
WINDOWS = (5, 10, 20, 30, 60)  # 研报 §2.2 为 5 个可选窗口；正文未列具体值，按华泰同族惯例补 30
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-begin", type=int, default=20190101)
    ap.add_argument("--train-end", type=int, default=20241231)
    ap.add_argument("--test-begin", type=int, default=20250101)
    ap.add_argument("--test-end", type=int, default=20260716)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--horizon", type=int, default=10, help="调仓周期（研报=10 日）")
    ap.add_argument("--select-cor", type=float, default=0.4,
                    help="入选因子与已入选因子 spearman 相关上限（研报=0.4）")
    ap.add_argument("--min-autocorr", type=float, default=0.0,
                    help="RRE 秩稳定性门槛（截面排名自相关下限，0=关闭）")
    ap.add_argument("--no-mc", action="store_true", help="关闭市值中性化奖励")
    ap.add_argument("--long-ir-lambda", type=float, default=0.5,
                    help="多头 IR 奖励强度 λ（研报之二十四；0=关闭）")
    ap.add_argument("--no-barra", action="store_true",
                    help="关闭 Barra 风格时序相关惩罚")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-logz", type=float, default=0.0,
                    help="logZ 经验初始化（≈log(ΣR)，真实数据建议 9~11，加速收敛）")
    ap.add_argument("--jobs", type=int, default=1,
                    help="reward 并行求值进程数（>1 用进程池，需 spawn 可 pickle）")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="TB checkpoint 路径（每 log_every iters 保存；存在时自动续训）")
    ap.add_argument("--offline", action="store_true",
                    help="离线模式：仅用本地 parquet 缓存（TGW 不可用时的降级路径）")
    ap.add_argument("--cache-root", type=str, default=None,
                    help="DataCache 缓存目录（默认 Config；绕开被占用文件时传副本目录）")
    ap.add_argument("--sdk-cache", type=str, default=None,
                    help="SDK 本地缓存目录（默认 settings.yaml 的 sdk_local_path；"
                         "绕开被锁定文件时传新目录）")
    args = ap.parse_args()

    t0 = time.time()
    panel, close, market_cap, _mask = build_real_panel(
        args.train_begin, args.test_end,
        cache_root=args.cache_root, sdk_cache=args.sdk_cache, offline=args.offline)
    mdp = FactorMDP(OP_NAMES, WINDOWS, FEATURES, max_depth=3, max_nodes=9)
    print(f"[phase1] ops={mdp.n_op} win={mdp.n_win} feat={mdp.n_feat} "
          f"n_actions={mdp.n_actions} 面板={panel['close'].shape}")

    # ---- 训练/测试切分 ----
    train_mask = close.index < pd.Timestamp(str(args.test_begin))
    train_close = close.loc[train_mask]
    train_panel = {k: v.loc[train_mask] for k, v in panel.items()}
    train_mc = market_cap.loc[train_mask]
    print(f"训练段: {train_close.index[0].date()} ~ {train_close.index[-1].date()} "
          f"({len(train_close)} 日) | 测试段: {len(close) - len(train_close)} 日")

    # ---- 奖励（市值中性化 + 10 日调仓 + 研报之二十四奖励塑形） ----
    cache = RewardCache()
    node_cache: dict = {}                    # 子树级求值缓存（训练提速）
    mc_arg = None if args.no_mc else train_mc
    reward_fn = make_reward_fn(train_panel, None, FEATURES, cache=cache,
                               market_cap=mc_arg, horizon=args.horizon,
                               node_cache=node_cache,
                               long_ir_lambda=args.long_ir_lambda,
                               barra_mu=0.0 if args.no_barra else 0.5)
    shape_bits = [f"{args.horizon}日调仓"]
    if mc_arg is not None:
        shape_bits.append("市值中性化")
    if args.long_ir_lambda > 0:
        shape_bits.append(f"多头IR(λ={args.long_ir_lambda})")
    if not args.no_barra:
        shape_bits.append("Barra惩罚")
    print("奖励口径: " + " + ".join(shape_bits))

    # ---- 均匀基线（训练段同口径） ----
    base = sample_uniform(mdp, reward_fn, 60, seed=args.seed)
    base_r = [r for _, r in base]
    print(f"均匀基线 R 均值={np.mean(base_r):.4f} 中位数={np.median(base_r):.4f}")

    # ---- 并行 reward 池（可选，--jobs > 1） ----
    reward_pool = None
    if args.jobs > 1:
        from factor.gflownet.parallel import RewardPool
        from factor.gflownet.reward import build_horizon_returns
        rets = build_horizon_returns(train_close, args.horizon)
        reward_pool = RewardPool(train_panel, market_cap=mc_arg, returns=rets,
                                 returns_rank=rets.rank(axis=1),
                                 features=FEATURES, n_jobs=args.jobs,
                                 long_ir_lambda=args.long_ir_lambda,
                                 barra_mu=0.0 if args.no_barra else 0.5)
        print(f"并行 reward 池: {args.jobs} 进程")

    # ---- TB 训练 ----
    tb_net = TBPolicy(mdp.n_actions, init_logz=args.init_logz)
    print(f"\n=== TB 训练（{args.iters} iters x {args.batch}） ===")
    train_tb(mdp, reward_fn, tb_net, n_iters=args.iters, batch_size=args.batch,
             seed=args.seed, reward_pool=reward_pool,
             ckpt_path=args.ckpt, resume=bool(args.ckpt))
    if reward_pool is not None:
        reward_pool.close()
    print(f"奖励缓存: {len(cache)}")

    # ---- 测试段采样 + 低相关筛选（研报 §2.3） ----
    samples = sample_formulas(tb_net, mdp, reward_fn, args.samples, seed=args.seed)
    sr = [r for _, r in samples]
    print(f"\n=== 测试段评估（训练奖励同口径） ===")
    print(f"TB R 均值={np.mean(sr):.4f} 中位数={np.median(sr):.4f} "
          f"top1={sr[0]:.4f} (基线 均值={np.mean(base_r):.4f})")

    print(f"\n=== spearman<{args.select_cor} 低相关筛选"
          f"{' + RRE≥' + str(args.min_autocorr) if args.min_autocorr > 0 else ''} ===")
    selected = select_low_corr(samples, train_panel, FEATURES,
                               threshold=args.select_cor, progress=True,
                               min_autocorr=args.min_autocorr)
    print(f"入选因子: {len(selected)} / {len(samples)}")
    for f, r in selected[:8]:
        print(f"  {f}   R={r:.4f}")

    # 测试段真正样本外 IC（用测试段 close 的 horizon 收益）
    test_panel = {k: v.loc[~train_mask] for k, v in panel.items()}
    test_returns = test_panel["close"].pct_change(args.horizon).shift(-args.horizon)
    from factor.formula import formula_builder
    from factor.gflownet.reward import neutralize_market_cap, rank_ic_series
    oos_rows = []
    for f, r in selected[:40]:
        fp = formula_builder(f, features=FEATURES)(test_panel)
        if mc_arg is not None:
            fp = neutralize_market_cap(fp, market_cap.loc[~train_mask])
        ic = rank_ic_series(fp, test_returns)
        oos_rows.append((f, r, float(ic.abs().mean())))
    if oos_rows:
        oos_ic = [x[2] for x in oos_rows if np.isfinite(x[2])]
        n_nan = len(oos_rows) - len(oos_ic)
        print(f"\n=== 测试段 OOS |IC|（样本外，{len(oos_rows)} 个入选因子"
              f"{'，' + str(n_nan) + ' 个测试段无有效截面' if n_nan else ''}） ===")
        if oos_ic:
            print(f"OOS |IC| 均值={np.mean(oos_ic):.4f} 中位数={np.median(oos_ic):.4f} "
                  f"top={max(oos_ic):.4f}")

    print(f"\n耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
