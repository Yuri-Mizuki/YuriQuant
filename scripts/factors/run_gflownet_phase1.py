"""
GFlowNet Phase 1：真实 HS300 对齐研报
====================================

研报对齐项（国金《Alpha掘金系列之二十二》）：
- §2.2 算子空间：51 算子（已全覆盖，`factor/operators.py` 现有 69 个）
- §2.2 ExprNode 简化：交换律排序 + neg 折叠（Phase 0 已实现）
- §2.3 奖励：**市值中性化后的 |IC|** + **10 日调仓**
- §2.3 数据切分：训练段 / 测试段（样本外），测试段筛选 **spearman 相关 < 0.4**

数据：真实 HS300 后复权日线（2019-01 ~ 2026-07，缓存）；训练 2019-2024 / 测试 2025-2026。

分钟特征接入（2026-09-16，国金22 × 国金24 交叉）
-------------------------------------------------
``--feat-source`` 决定终端特征集：

- ``raw``（默认）：6 个原始量价字段 —— 与改动前完全一致（向后兼容）。
- ``im_independent`` / ``im_all``：只用因子库的 17 / 42 个 ``im_*`` 日内统计特征。
- ``raw+im_independent`` / ``raw+im_all``：两者并用，让搜索自己挑。

**窗口对齐（重要）**：``im_*`` 特征依赖分钟数据，最早只到 2022-01（当前
``min5_hs300.parquet`` 覆盖 2022-01 ~ 2026-06）。启用 im 后本脚本默认把**评估段**
裁到 im 覆盖区间（面板本身不裁行，以免丢掉 rolling 算子的 warm-up 历史），
于是训练段由 2019-2024 变为 2022-2024。

> ⚠️ **公平对照**：裁窗口后，含 im 的臂与默认 ``raw`` 臂的样本期不同，年化/IC
> **不可直接比较**。要拿到可比基线，用 ``--eval-begin`` / ``--eval-end``（**不是**
> ``--train-begin``）跑 raw 臂 —— 后者只动评估窗口，``--train-begin`` 会连面板
> 构建区间、列并集（实测 520→421 列）和 Barra 风格参照系一起改，等于给对照再引入
> 一个差异：
>
>     python -u scripts/factors/run_gflownet_phase1.py --feat-source im_independent
>     python -u scripts/factors/run_gflownet_phase1.py --feat-source raw \
>         --eval-begin 20220104 --eval-end 20260630
>
> 脚本会把两段天数、以及训练段**每日截面宽度**（在册股数 / 其中有 im 值股数）都
> 打印出来 —— 同窗口但截面宽度不同时同样不可比，这一步不量就会漏掉。实测 im 特征
> 只覆盖 HS300 每日 300 只在册股中的 **255 只（85%）**，是必须披露的复现边界。
> 用 ``--keep-window`` 可强制不裁，但会打印告警（不同终端样本期不同 → 不可比）。

用法（需安装 torch 的解释器 + AmazingData）：
    cd <仓库根>
    python -u scripts/factors/run_gflownet_phase1.py --iters 600 --batch 12
    python -u scripts/factors/run_gflownet_phase1.py --feat-source im_independent
    python -u scripts/factors/run_gflownet_phase1.py --feat-source raw+im_all --keep-window
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="invalid value encountered in reduce")

from factor.gflownet.env import FactorMDP  # noqa: E402
from factor.gflownet.net import TBPolicy  # noqa: E402
from factor.gflownet.reward import RewardCache, make_reward_fn  # noqa: E402
from factor.gflownet.selection import select_low_corr  # noqa: E402
from factor.gflownet.tb import sample_formulas, sample_uniform, train_tb  # noqa: E402
from scripts.common.e2e_common import (  # noqa: E402
    FEAT_SOURCES,
    RAW_FEATURES,
    aligned_eval_masks,
    attach_im_features,
)


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
#: ``--feat-source raw`` 时的终端特征（= 改动前的行为）。启用 im 后实际特征集由
#: :func:`scripts.common.e2e_common.attach_im_features` 在 main 内动态给出。
FEATURES = list(RAW_FEATURES)


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
    ap.add_argument("--feat-source", dest="feat_source", default="raw",
                    choices=list(FEAT_SOURCES),
                    help="终端特征集来源：raw=6 原始量价（改动前行为）；"
                         "im_*=因子库分钟特征；raw+im_*=并用。见模块 docstring")
    ap.add_argument("--im-dataset", dest="im_dataset", default="hs300_2022_2025",
                    help="im_* 特征的因子库数据集名")
    ap.add_argument("--keep-window", dest="keep_window", action="store_true",
                    help="启用 im 时不把评估段裁到 im 覆盖区间（会告警：不可比）")
    ap.add_argument("--eval-begin", dest="eval_begin", type=int, default=None,
                    help="评估段起点 YYYYMMDD（默认：启用 im 时 = im 首个有值日）。"
                         "与 --train-begin（面板构建起点）解耦 —— 跑 raw 对照臂时"
                         "用它对齐窗口，可避免面板列集被一并改变")
    ap.add_argument("--eval-end", dest="eval_end", type=int, default=None,
                    help="评估段终点 YYYYMMDD（默认：启用 im 时 = im 最后有值日）")
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

    # ---- 终端特征集：raw 6 字段 ± 因子库 im_* 分钟特征（国金22 × 国金24） ----
    panel, features, im_index = attach_im_features(
        panel, args.feat_source, dataset=args.im_dataset)
    n_raw = sum(1 for f in features if f in RAW_FEATURES)
    n_im = len(features) - n_raw

    # ---- 评估段与 im 覆盖区间对齐（默认开；只裁评估段，不裁面板行以保 warm-up） ----
    idx = close.index
    test_start = pd.Timestamp(str(args.test_begin))
    im_lo = im_hi = None
    if im_index is not None:
        if args.keep_window:
            print("[warn] --keep-window：评估段未裁到 im 覆盖区间。含 im_* 的公式只在"
                  "其有值日参与 IC，纯 raw 公式仍用全段 → 两类样本期不同，结果不可比。")
        else:
            im_lo, im_hi = im_index[0], im_index[-1]
    eval_begin = args.eval_begin if args.eval_begin is not None else (
        int(im_lo.strftime("%Y%m%d")) if im_lo is not None else None)
    eval_end = args.eval_end if args.eval_end is not None else (
        int(im_hi.strftime("%Y%m%d")) if im_hi is not None else None)

    train_mask, test_mask = aligned_eval_masks(
        idx, args.test_begin,
        im_index if im_lo is not None else None,
        keep_window=args.keep_window,
        eval_begin=eval_begin, eval_end=eval_end)
    if not train_mask.any():
        raise SystemExit(
            f"训练段为空：面板 {idx[0].date()}~{idx[-1].date()} 与评估起点"
            f"（{eval_begin}）/ im 覆盖区间无交集，检查 --train-begin/--eval-begin")
    if not test_mask.any():
        raise SystemExit("测试段为空：检查 --test-begin/--test-end 与 im 覆盖区间")

    mdp = FactorMDP(OP_NAMES, WINDOWS, features, max_depth=3, max_nodes=9)
    print(f"[phase1] feat-source={args.feat_source} | 终端特征 {len(features)} 个"
          f"（raw {n_raw} + im {n_im}）")
    print(f"[phase1] ops={mdp.n_op} win={mdp.n_win} feat={mdp.n_feat} "
          f"n_actions={mdp.n_actions} 面板={panel['close'].shape}")
    if n_im:
        print(f"[phase1] im 特征有值日 {im_index[0].date()} ~ {im_index[-1].date()}"
              f"（{len(im_index)} 日）")

    # ---- 训练/测试切分 ----
    train_close = close.loc[train_mask]
    train_panel = {k: v.loc[train_mask] for k, v in panel.items()}
    train_mc = market_cap.loc[train_mask]
    test_close = close.loc[test_mask]
    print(f"训练段: {train_close.index[0].date()} ~ {train_close.index[-1].date()} "
          f"({len(train_close)} 日) | 测试段: {test_close.index[0].date()} ~ "
          f"{test_close.index[-1].date()} ({len(test_close)} 日)")
    if n_im and im_lo is not None:
        raw_train = int((idx < test_start).sum())
        print(f"[窗口对齐] im 覆盖 {im_lo.date()} ~ {im_hi.date()}；训练段由原始 "
              f"{raw_train} 日裁至 {len(train_close)} 日"
              f"（-{raw_train - len(train_close)} 日）")
        print(f"[窗口对齐] 要与 raw 臂公平对照，请用**同一面板 + 同一评估段**跑对照臂：\n"
              f"           --feat-source raw --eval-begin {im_lo:%Y%m%d} "
              f"--eval-end {im_hi:%Y%m%d}\n"
              f"           （务必保持 --train-begin={args.train_begin} 不变 —— 改它会"
              f"改变面板列并集与 Barra 风格参照系，给对照引入第二个差异）")

    # 截面覆盖诊断：即使窗口相同，含 im 的公式与纯 raw 公式的**每日截面宽度**也可能
    # 不同（im 特征只在因子库池上有值）。不把差异量出来，就会"看起来同窗口、其实
    # 不同截面"——rank IC 的样本基数不同，比较依旧不严格。
    members = panel["mask"].loc[train_mask].sum(axis=1)
    m_mem = float(members.median())
    line = f"[截面覆盖] 训练段每日在册中位数 {m_mem:.0f} 只"
    if n_im:
        cov_any = None
        for nm in features[n_raw:]:
            v = panel[nm].loc[train_mask].notna()
            cov_any = v if cov_any is None else (cov_any | v)
        m_im = float(cov_any.sum(axis=1).median())
        line += f"，其中有 im 值 {m_im:.0f} 只（{m_im / max(m_mem, 1):.1%}）"
        if m_im < 0.95 * m_mem:
            line += "\n[截面覆盖] ⚠️ im 未覆盖全部在册股 → 含 im 的公式截面比纯 raw " \
                    "公式窄，两者 IC 样本基数不同，横向比较须把这一点算进去。"
    print(line)

    # ---- 奖励（市值中性化 + 10 日调仓 + 研报之二十四奖励塑形） ----
    cache = RewardCache()
    node_cache: dict = {}                    # 子树级求值缓存（训练提速）
    mc_arg = None if args.no_mc else train_mc
    reward_fn = make_reward_fn(train_panel, None, features, cache=cache,
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
        # returns_rank 是**训练期**性能捷径（省约 30% IC 耗时）；与入库评估走的
        # canonical IC 不总等价（因子整行缺失无影响，行内散点缺失才分叉）。
        # 分层约定见 factor/gflownet/reward.py 模块 docstring 的「IC 口径分层」。
        reward_pool = RewardPool(train_panel, market_cap=mc_arg, returns=rets,
                                 returns_rank=rets.rank(axis=1),
                                 features=features, n_jobs=args.jobs,
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
    selected = select_low_corr(samples, train_panel, features,
                               threshold=args.select_cor, progress=True,
                               min_autocorr=args.min_autocorr)
    print(f"入选因子: {len(selected)} / {len(samples)}")
    for f, r in selected[:8]:
        print(f"  {f}   R={r:.4f}")

    # 测试段真正样本外 IC（用测试段 close 的 horizon 收益）
    test_panel = {k: v.loc[test_mask] for k, v in panel.items()}
    test_returns = test_panel["close"].pct_change(args.horizon).shift(-args.horizon)
    from factor.formula import formula_builder
    from factor.gflownet.reward import neutralize_market_cap, rank_ic_series
    oos_rows = []
    for f, r in selected[:40]:
        fp = formula_builder(f, features=features)(test_panel)
        if mc_arg is not None:
            fp = neutralize_market_cap(fp, market_cap.loc[test_mask])
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
