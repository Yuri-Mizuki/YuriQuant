# -*- coding: utf-8 -*-
"""
GFlowNet v4 因子池入库 + GP 池对比 + 合成评估
=============================================

1. 从 ckpt 恢复训练好的 TB 策略 → 采样 150 → spearman<0.4 低相关筛选（复现 v4 入选集）
2. 注册进 FactorLibrary(hs300_2025)：GFlowNet 池 + GP 池（华泰21 口径 csv）
3. 池间相关性对比（GFlowNet vs GP）
4. 合成评估（训练段 2019-2024 fit / 测试段 2025-2026H1 评估）：
   IC加权 / PCA / Gram-Schmidt / GBDT stacking × {GFlowNet池, GP池, 合并池}
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from backtest.metrics import PERIODS_PER_YEAR
from factor.formula import formula_builder
from factor.gflownet.env import FactorMDP
from factor.gflownet.net import TBPolicy
from factor.gflownet.reward import make_reward_fn, rank_ic_series
from factor.gflownet.selection import select_low_corr
from factor.gflownet.tb import sample_formulas
from factor.synthesis import CompositeInput, synthesize_ic_weighted, synthesize_pca, \
    synthesize_orthogonal, synthesize_stacking_gbdt
from research.factor_library import FactorLibrary

# 与 run_gflownet_phase1.py 保持同一口径
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
WINDOWS = (5, 10, 20, 30, 60)
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def _zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """逐日截面 zscore（华泰预处理口径）。"""
    m = panel.notna()
    out = panel.where(m).sub(panel.mean(axis=1), axis=0).div(panel.std(axis=1, ddof=0), axis=0)
    return out


def build_factors(formulas: list[str], panel: dict[str, pd.DataFrame],
                  node_cache: dict | None = None) -> dict[str, pd.DataFrame]:
    """公式列表 → {formula: date×code 面板}（zscore）。"""
    out = {}
    for f in formulas:
        try:
            fp = formula_builder(f, features=FEATURES, node_cache=node_cache)(panel)
            if fp is None or fp.empty or fp.notna().sum().sum() == 0:
                continue
            out[f] = _zscore(fp)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {f[:50]}... {type(e).__name__}: {str(e)[:40]}", flush=True)
    return out


def eval_ic_ir(factor: pd.DataFrame, returns: pd.DataFrame) -> tuple[float, float]:
    """日频 1 日 IC 均值 / IR（与库口径一致）。"""
    ic = rank_ic_series(factor, returns)
    ic = ic.dropna()
    if ic.empty:
        return 0.0, 0.0
    return float(ic.mean()), float(ic.mean() / ic.std() * np.sqrt(PERIODS_PER_YEAR)) if ic.std() > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="reports/gflownet_tb_v4.pt")
    ap.add_argument("--offline", action="store_true", help="离线模式（本地缓存）")
    ap.add_argument("--gp-csv", default="reports/factor_gp_20260804_174258.csv")
    ap.add_argument("--dataset", default="hs300_2025")
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--train-begin", type=int, default=20190101)
    ap.add_argument("--test-begin", type=int, default=20250101)
    ap.add_argument("--test-end", type=int, default=20260716)
    ap.add_argument("--register", action="store_true", help="注册进 factor_library")
    args = ap.parse_args()

    t0 = time.time()
    from scripts.run_gflownet_phase1 import build_real_panel
    panel, close, market_cap, _mask = build_real_panel(
        args.train_begin, args.test_end, offline=args.offline)
    print(f"面板: {panel['close'].shape}", flush=True)

    train_mask = close.index < pd.Timestamp(str(args.test_begin))
    train_panel = {k: v.loc[train_mask] for k, v in panel.items()}
    train_mc = market_cap.loc[train_mask]
    test_returns = close.pct_change().shift(-1)

    # ---- 恢复 TB 策略并复现 v4 入选集 ----
    mdp = FactorMDP(OP_NAMES, WINDOWS, FEATURES, max_depth=3, max_nodes=9)
    net = TBPolicy(mdp.n_actions, init_logz=9.0)
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    print("ckpt 加载完成（logZ 由训练保存值恢复）", flush=True)
    net.logZ.data = torch.tensor(float(
        torch.load(args.ckpt, map_location="cpu")["logz"]))

    from factor.gflownet.reward import RewardCache
    cache = RewardCache()
    reward_fn = make_reward_fn(train_panel, None, FEATURES, cache=cache,
                               market_cap=train_mc, horizon=10)
    samples = sample_formulas(net, mdp, reward_fn, args.samples, seed=0)
    selected = select_low_corr(samples, train_panel, FEATURES,
                               threshold=args.threshold, progress=True)
    gf_formulas = [f for f, _ in selected]
    print(f"GFlowNet 入选: {len(gf_formulas)} 因子", flush=True)

    # ---- GP 池 ----
    gp_df = pd.read_csv(args.gp_csv)
    gp_formulas = [str(f).strip().strip('"') for f in gp_df["formula"]]
    print(f"GP 池: {len(gp_formulas)} 因子", flush=True)

    # ---- 全期因子面板 ----
    node_cache: dict = {}
    gf_panels = build_factors(gf_formulas, panel, node_cache)
    gp_panels = build_factors(gp_formulas, panel, node_cache)
    print(f"GFlowNet 可用面板: {len(gf_panels)} / {len(gf_formulas)}; "
          f"GP 可用面板: {len(gp_panels)} / {len(gp_formulas)}", flush=True)

    # ---- 注册进因子库 ----
    if args.register:
        fl = FactorLibrary(dataset=args.dataset)
        rets = test_returns.reindex_like(close)
        n_reg = 0
        for f, fp in gf_panels.items():
            try:
                fl.register(
                    name=f"gf_{_slug(f)}", panel=fp, returns_panel=rets,
                    formula=f, source="mining:gflownet_phase1_v4",
                    family="非线性组合", frequency="日频",
                    note="GFlowNet v4: 51算子×5窗口, 10日调仓市值中性化奖励, "
                         "并集池520+mask, 1600iters loss28.95",
                    check_dup=False)
                n_reg += 1
            except Exception as e:  # noqa: BLE001
                print(f"  [reg fail] {f[:50]}... {type(e).__name__}: {str(e)[:50]}", flush=True)
        for f, fp in gp_panels.items():
            try:
                fl.register(
                    name=f"gp_{_slug(f)}", panel=fp, returns_panel=rets,
                    formula=f, source="mining:gp_htai_21",
                    family="非线性组合", frequency="日频",
                    note="华泰21 GP 复现（factor_gp_20260804_174258）", check_dup=False)
                n_reg += 1
            except Exception as e:  # noqa: BLE001
                print(f"  [reg fail] {f[:50]}... {type(e).__name__}: {str(e)[:50]}", flush=True)
        print(f"注册完成: {n_reg} 因子 -> hs300_2025", flush=True)

    # ---- 池间相关性对比 ----
    gf_list = list(gf_panels.keys())
    gp_list = list(gp_panels.keys())
    corrs = []
    for a in gf_list:
        pa = gf_panels[a]
        for b in gp_list:
            pb = gp_panels[b]
            m = pa.notna() & pb.notna()
            c = pd.concat([pa[m].stack(), pb[m].stack()], axis=1).corr().iloc[0, 1]
            corrs.append(c)
    corrs = [c for c in corrs if np.isfinite(c)]
    print(f"\n=== 池间相关性（{len(gf_list)} GFlowNet × {len(gp_list)} GP） ===")
    print(f"平均 |corr|: {np.mean(np.abs(corrs)):.4f}  中位数: "
          f"{np.median(np.abs(corrs)):.4f}  max: {np.max(np.abs(corrs)):.4f}")
    print(f"|corr|>0.7 对数: {sum(1 for c in corrs if abs(c) > 0.7)}")
    print(f"|corr|>0.5 对数: {sum(1 for c in corrs if abs(c) > 0.5)}")

    # ---- 合成评估（训练段 fit / 测试段评估；1 日与 10 日双口径） ----
    # 口径公平性：GFlowNet 因子按 10 日调仓奖励训练，GP 因子按日频 IC 挖掘——
    # 单一口径对一方有利，故两种 horizon 都报告。
    print(f"\n=== 合成评估（训练段 fit / 测试段评估, 双 horizon） ===")
    rets_by_h = {}
    for h in (1, 10):
        rets_by_h[h] = close.pct_change(h).shift(-h)

    def _eval(name: str, comp: pd.DataFrame, te: pd.DataFrame):
        ic_te = rank_ic_series(comp.loc[~train_mask], te).dropna()
        ic_te_mean = float(ic_te.mean()) if len(ic_te) else float("nan")
        ir_te = float(ic_te.mean() / ic_te.std() * np.sqrt(PERIODS_PER_YEAR / h)) if len(ic_te) > 2 and ic_te.std() > 0 else float("nan")
        print(f"  {name:24s} 测试IC={ic_te_mean:+.4f}  测试IR={ir_te:+.2f}")
        return {"method": name, "horizon": h, "test_ic": ic_te_mean, "test_ir": ir_te}

    rows = []
    for h in (1, 10):
        rets_all = rets_by_h[h]
        tr = rets_all.loc[train_mask]
        te = rets_all.loc[~train_mask]
        for pool_name, panels in [("GFlowNet", gf_panels), ("GP", gp_panels),
                                  ("合并", {**gf_panels, **gp_panels})]:
            if not panels:
                continue
            comps = []
            for f, fp in panels.items():
                ic, ir = eval_ic_ir(fp.loc[train_mask], tr)
                comps.append(CompositeInput(name=f[:40], panel=fp, ic=ic, ir=ir))
            icw = synthesize_ic_weighted(comps, returns_panel=rets_all)
            pca = synthesize_pca(comps, returns_panel=rets_all, n_components=1)
            orth = synthesize_orthogonal(comps)
            for m, c in [("IC加权", icw), ("PCA", pca), ("正交", orth)]:
                rows.append(_eval(f"{pool_name}-{m}", c, te))
            try:
                gb = synthesize_stacking_gbdt(comps, returns_panel=rets_all)
                rows.append(_eval(f"{pool_name}-GBDT", gb, te))
            except Exception as e:  # noqa: BLE001
                print(f"  {pool_name}-GBDT 失败: {type(e).__name__}: {str(e)[:60]}")

    out = pd.DataFrame(rows)
    out.to_csv("reports/gflownet_vs_gp_synthesis.csv", index=False)
    print(f"\n结果已存 reports/gflownet_vs_gp_synthesis.csv")
    print(f"总耗时: {time.time() - t0:.0f}s")


def _slug(s: str) -> str:
    import re
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)[:60]
    return s.strip("_") or "f"


if __name__ == "__main__":
    main()
