"""
端到端对比：MI 非线性因子 vs 线性因子 × ML stacking 合成器
==========================================================

验证报告23 的核心论点：「GP 挖出的非线性因子能被机器学习选股模型有效利用」。

设计（2×2 正交 + 噪声对照）：
    因子组   { 线性组(rankic_mean 挖的), 非线性组(mutual_info 挖的) }
    合成器   { ridge(线性模型), LightGBM(非线性模型) }
    对照组   { 随机噪声因子 + GBDT }（证明 GBDT 的提升不是无脑过拟合）

公平性：
    - 两组因子数量相同（top K 对齐）
    - 合成器配置、切分、seed 完全一致
    - 合成目标 = 日频次日收益，expanding-window 时序 CV（严格无未来函数）
    - 评估分段报告：前 40%（冷启动）/ 后 40%（训练充分，真 OOS）/ 全样本

用法:
    python scripts/compare_ml_synthesis.py                 # mock（快）
    python scripts/compare_ml_synthesis.py --real           # 真实 HS300
    python scripts/compare_ml_synthesis.py --real --top 9 --seed 42

输出: reports/_htai_gp/ml_synthesis_<real|mock>.csv + 终端 2×2 对比表
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cli_common import add_real_mock_args, setup_logging  # noqa: E402


log = setup_logging("compare_ml_synthesis")

from research.factor_analysis import calc_ic_series, calc_ir  # noqa: E402

def _htai_features_from_formulas(formulas: list[str], panel: dict, neutral_panels: dict
                                 ) -> list:
    """重建公式面板并做华泰环内预处理（MAD→五因子中性化→zscore），作为合成特征。"""
    from factor.formula import formula_builder
    from factor.genetic_mining import _htai_preprocess

    feats = list(panel.keys())
    out = []
    for f in formulas:
        try:
            fp = _htai_preprocess(formula_builder(f, features=feats)(panel),
                                  neutral_panels=neutral_panels)
        except Exception:
            continue
        if fp is None or fp.notna().sum().sum() < 100:
            continue
        out.append(fp)
    return out

def _noise_panel(panel: dict, seed: int = 0) -> pd.DataFrame:
    """随机噪声因子（截面 zscore），作对照组。"""
    rng = np.random.default_rng(seed)
    close = panel["close"]
    z = pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)
    from factor.preprocessing import standardize_zscore
    return standardize_zscore(z)

def seg_stats(comp: pd.DataFrame, rets: pd.DataFrame, frac_lo: float = 0.0,
              frac_hi: float = 1.0, method: str = "spearman") -> dict:
    """复合因子在 [frac_lo, frac_hi) 时间段的 IC 统计。"""
    ic = calc_ic_series(comp, rets, method=method).dropna()
    seg = ic.iloc[int(len(ic) * frac_lo):int(len(ic) * frac_hi)]
    if len(seg) < 5:
        return {"ic_mean": float("nan"), "ir": float("nan"), "t": float("nan"), "n": len(seg)}
    m, s = float(seg.mean()), float(seg.std())
    return {
        "ic_mean": m,
        "ir": calc_ir(seg),
        "t": m / (s / np.sqrt(len(seg))) if s > 0 else 0.0,
        "n": len(seg),
    }

def top_bottom_excess(comp: pd.DataFrame, rets: pd.DataFrame,
                      top_frac: float = 0.1) -> dict:
    """Top/Bottom 层相对全池等权的平均未来 20 日超额（月频持有口径）。"""
    from factor.genetic_mining import _monthly_forward_returns
    r_forward = _monthly_forward_returns(rets)
    tops, bots = [], []
    for d in comp.index:
        f = comp.loc[d]
        r = r_forward.loc[d]
        m = f.notna() & r.notna()
        if int(m.sum()) < 20:
            continue
        fv, rv = f[m], r[m]
        n_top = max(1, int(len(fv) * top_frac))
        idx_top = fv.sort_values(ascending=False).index[:n_top]
        idx_bot = fv.sort_values(ascending=True).index[:n_top]
        base = float(rv.mean())
        tops.append(float(rv.loc[idx_top].mean()) - base)
        bots.append(float(rv.loc[idx_bot].mean()) - base)
    if not tops:
        return {"top_excess": float("nan"), "bot_excess": float("nan")}
    return {"top_excess": float(np.mean(tops)), "bot_excess": float(np.mean(bots))}

def main():
    ap = argparse.ArgumentParser(description="MI 非线性因子 vs 线性因子 × ML stacking 端到端对比")
    add_real_mock_args(ap)
    ap.add_argument("--begin", type=int, default=20250101, help="真实数据开始（须与挖因子的区间一致）")
    ap.add_argument("--end", type=int, default=20251231, help="真实数据结束（须与挖因子的区间一致）")
    ap.add_argument("--top", type=int, default=9, help="每组使用的因子数（对齐两组）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gbdt-trials", type=int, default=0,
                    help=">0 时用 optuna 调参版 GBDT（较慢）；0 用固定超参 GBDT")
    ap.add_argument("--out", default=None, help="结果 CSV 路径")
    args = ap.parse_args()

    from data.cache_helpers import build_htai_neutral_panels, build_real_panel
    from data.mock import gen_mock_panel_with_signal
    from factor.synthesis import (CompositeInput, synthesize_stacking,
                                  synthesize_stacking_gbdt)

    # ---- 数据 ----
    if args.real:
        from config import Config
        cfg = Config.get()
        panel, rets = build_real_panel(cfg, args.begin, args.end)
    else:
        panel = gen_mock_panel_with_signal()
        rets = panel["close"].pct_change().shift(-1)
    panel["returns"] = panel["close"].pct_change()
    if "amount" in panel and "volume" in panel:
        panel["vwap"] = panel["amount"] / panel["volume"]
    neutral_panels = build_htai_neutral_panels(panel, real=args.real)
    log.info("面板: %d 日 × %d 股, 中性化协变量=%s", len(rets), panel["close"].shape[1],
             list(neutral_panels.keys()))

    # ---- 因子组：复用 compare_htai_fitness 已挖出的公式（避免重跑 GP）----
    src = Path("reports/_htai_gp/compare_hs300.csv" if args.real else "reports/_htai_gp/compare_mock.csv")
    mined = pd.read_csv(src)
    groups = {}
    for mode, tag in [("rankic_mean", "linear"), ("mutual_info", "nonlinear")]:
        sub = mined[mined["mode"] == mode].head(args.top)
        fps = _htai_features_from_formulas(list(sub["formula"]), panel, neutral_panels)
        groups[tag] = [
            CompositeInput(name=f"f{i}", panel=fp, ic=0.0, ir=0.0)
            for i, fp in enumerate(fps)
        ]
        log.info("%s 组: %d 个因子", tag, len(groups[tag]))

    # ---- 合成器矩阵 ----
    def synth_ridge(comps):
        return synthesize_stacking(comps, rets, n_splits=5, alpha=1.0, target_mode="rank")

    def synth_gbdt(comps):
        return synthesize_stacking_gbdt(comps, rets, n_splits=5, embargo_days=5,
                                        n_estimators=300, learning_rate=0.05,
                                        num_leaves=31, max_depth=6,
                                        min_child_samples=20, seed=args.seed,
                                        target_mode="rank")

    rows = []
    for tag, comps in groups.items():
        for algo, fn in [("ridge", synth_ridge), ("gbdt", synth_gbdt)]:
            if not comps:
                log.warning("%s 组无有效因子，跳过 %s", tag, algo)
                continue
            log.info("==> 合成: %s 因子 × %s", tag, algo)
            comp = fn(comps)
            rec = {"factor_group": tag, "algo": algo}
            for seg_name, lo, hi in [("full", 0.0, 1.0), ("head40", 0.0, 0.4),
                                     ("tail40", 0.6, 1.0)]:
                s = seg_stats(comp, rets, lo, hi)
                rec[f"ic_{seg_name}"] = s["ic_mean"]
                rec[f"ir_{seg_name}"] = s["ir"]
                rec[f"t_{seg_name}"] = s["t"]
            s = seg_stats(comp, _monthly_fwd(rets), method="spearman")
            rec["ic_monthly"] = s["ic_mean"]
            rec["ir_monthly"] = s["ir"]
            tb = top_bottom_excess(comp, rets)
            rec.update(tb)
            rows.append(rec)

    # ---- 噪声对照组：随机因子 × GBDT ----
    log.info("==> 合成: noise × gbdt（对照）")
    noise = CompositeInput(name="noise", panel=_noise_panel(panel, seed=args.seed), ic=0.0, ir=0.0)
    comp_n = synth_gbdt([noise])
    rec = {"factor_group": "noise", "algo": "gbdt"}
    for seg_name, lo, hi in [("full", 0.0, 1.0), ("head40", 0.0, 0.4), ("tail40", 0.6, 1.0)]:
        s = seg_stats(comp_n, rets, lo, hi)
        rec[f"ic_{seg_name}"] = s["ic_mean"]
        rec[f"ir_{seg_name}"] = s["ir"]
        rec[f"t_{seg_name}"] = s["t"]
    s = seg_stats(comp_n, _monthly_fwd(rets), method="spearman")
    rec["ic_monthly"] = s["ic_mean"]
    rec["ir_monthly"] = s["ir"]
    rec.update(top_bottom_excess(comp_n, rets))
    rows.append(rec)

    df = pd.DataFrame(rows)
    print("\n===== ML stacking 端到端对比（2×2 + 噪声对照）=====")
    with pd.option_context("display.width", 220, "display.float_format", lambda v: f"{v:.4f}"):
        print(df.to_string(index=False))
    print("\n解读: tail40=后40%训练充分段(真OOS) | ic_monthly=月频20日IC | top_excess=Top10%多头超额")

    out_path = Path(args.out or f"reports/_htai_gp/ml_synthesis_{'real' if args.real else 'mock'}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("结果已保存: %s", out_path)

def _monthly_fwd(rets: pd.DataFrame):
    from factor.genetic_mining import _monthly_forward_returns
    return _monthly_forward_returns(rets)

if __name__ == "__main__":
    main()