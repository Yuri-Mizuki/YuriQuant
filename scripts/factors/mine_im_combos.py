"""
第二层：im_* 统计特征之上的 GP 组合公式挖掘（降维路线主产出）
============================================================

把第一层体检出的独立显著 im_* 特征作为 GP 终端集，搜索时序算子
（ts_mean/ts_rank/...）与截面算子的组合公式——Alpha掘金 22 的
"降维后复用日频挖掘框架"路线（他们 40 特征上挖出 IC 中位数 5.83%）。

设计
----
- **终端集**：``--source im_independent``（默认）= 17 个独立且显著
  （dup_corr1<0.5 且 |t_nw|>2）的特征；``im_all`` = 全部 42 个（3 个完全
  同构的会因与库相关 >0.95 被去相关惩罚压制）。
- **输入面板来源**：因子库已入库的 im_* 面板（含除权掩码 + zscore，
  与第一层体检完全同口径，不重算）。
- **适应度**：|NW 风格 t|（train 段 70%），OOS 30% 只报告不参与进化；
  monthly_weight 融合缓解 horizon 错配；library_penalty 对库内因子去相关。
- **窗口**：(5, 10, 20)——分钟特征多为快信号，长窗口交给库内日频因子。
- **产出**：results_df（含 train/OOS IC）+ HallOfFame 公式逐个评估后
  入库 ``gp_im_*``（hs300_2022_2025，frequency=日内，parents=用到的 im_*），
  与库内因子的去相关体检一并做。

用法
----
    python -m scripts.factors.mine_im_combos                     # 默认预算
    python -m scripts.factors.mine_im_combos --pop 300 --gen 30  # 大预算
    python -m scripts.factors.mine_im_combos --source im_all --no-save
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("mine_im_combos")

PREFIX = "gp_im_"


def load_im_panels(source: str = "im_independent",
                   dataset: str = "hs300_2022_2025") -> tuple[dict, dict]:
    """从因子库取 im_* 面板与公式说明；im_independent 只取独立显著 17 个。

    2026-09-16：实现已上移 ``scripts.common.e2e_common.load_im_panels``，
    与 ``scripts.factors.run_gflownet_phase1``（GFlowNet 接分钟特征）共用同一
    份筛选口径，避免两条链路漂移；此处只保留入口。搬迁等价性由
    ``scripts.oneoff._probe_im_loader_equiv`` 逐位验证（17/42 面板 max|Δ|=0、
    md5 一致）。
    """
    from scripts.common.e2e_common import load_im_panels as _load

    return _load(source, dataset)


def main() -> None:
    ap = argparse.ArgumentParser(description="im_* 特征之上的 GP 组合挖掘（第二层）")
    ap.add_argument("--source", default="im_independent",
                    choices=["im_independent", "im_all"])
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--gen", type=int, default=25)
    ap.add_argument("--windows", default="5,10,20")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--sample-step", type=int, default=2,
                    help="进化期时间子采样步长（hof 汇总仍全样本；1=不采样）")
    ap.add_argument("--top-k", type=int, default=10, help="入库公式数上限")
    ap.add_argument("--no-save", action="store_true", help="只挖不入库")
    args = ap.parse_args()

    from factor.genetic_mining import run_gp_mining
    from scripts.common.e2e_common import load_daily_data

    im_panels, im_docs = load_im_panels(args.source)
    if len(im_panels) < 4:
        raise SystemExit(f"终端集过少: {len(im_panels)}")
    names = sorted(im_panels)
    log.info("终端集 %d 个: %s", len(names), ", ".join(names))

    px, _lib_feats = load_daily_data(begin=20220101)
    END = pd.Timestamp("2026-06-30")
    px = {k: v[v.index <= END] for k, v in px.items()}
    close = px["close"]
    im_panels = {n: p.reindex(index=close.index, columns=close.columns)
                 for n, p in im_panels.items()}
    returns = close.pct_change(fill_method=None).shift(-1)  # 未来一日收益（库/挖掘同口径）
    log.info("日线 %d 日 × %d 码（截断至 %s）",
             close.shape[0], close.shape[1], END.date())

    # 库去相关惩罚面：排除 im_* 自身后的库内 significant 因子（防"重新发明
    # 库内日频因子"——第二层的目标是分钟信息组合，不是日频公式重述）
    from research.factor_library import FactorLibrary
    lib = FactorLibrary(dataset="hs300_2022_2025")
    lib_feats = lib.load_significant_features(exclude_model=True)
    lib_panels = {k: v.reindex(index=close.index, columns=close.columns)
                  for k, v in lib_feats.items() if not k.startswith("im_")}
    log.info("库去相关参照: %d 个库内 significant 因子", len(lib_panels))

    results, hof = run_gp_mining(
        panel=im_panels,
        returns_panel=returns,
        features=names,
        windows=tuple(int(w) for w in args.windows.split(",")),
        population=args.pop,
        generations=args.gen,
        train_frac=0.7,
        monthly_weight=0.5,
        library_penalty=0.3,
        library_panels=lib_panels,
        n_jobs=args.n_jobs,
        seed=args.seed,
        patience=args.patience,
        sample_step=args.sample_step,  # 进化期粗筛（hof 汇总仍全样本）
    )
    out = Path("reports/mine_im_combos.csv")
    results.to_csv(out, index=False, encoding="utf-8-sig")
    log.info("挖掘完成: %d 条结果 → %s", len(results), out)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        want = ("formula", "ic_mean", "ic_train", "ic_oos", "t_stat", "t_oos", "ir", "height")
        cols = [c for c in want if c in results.columns]
        print(results.sort_values("t_stat", ascending=False)[cols].head(15).to_string(index=False))

    if args.no_save:
        log.info("--no-save：不入库")
        return

    # ---- Hof 公式入库：gp_im_<hash8> ----
    # 过拟合三重守卫：train/OOS 同号、OOS 幅度 ≥30% train、t_stat 排序取 top K
    from factor.formula import formula_builder

    ok = results[
        (np.sign(results["ic_oos"]) == np.sign(results["ic_train"]))
        & (results["ic_oos"].abs() >= 0.3 * results["ic_train"].abs())
    ].copy()
    ok["_at"] = ok["t_stat"].abs()
    ok = ok.sort_values("_at", ascending=False)
    log.info("过拟合守卫后 %d/%d 条可入库，取 top %d", len(ok), len(results), args.top_k)

    import hashlib

    registered = 0
    for _, row in ok.head(args.top_k).iterrows():
        formula = str(row["formula"])
        try:
            fp = formula_builder(formula, features=names)(im_panels)
        except Exception as e:  # noqa: BLE001
            log.warning("公式重建失败，跳过: %s (%s)", formula[:60], e)
            continue
        if fp is None or fp.notna().sum().sum() == 0:
            continue
        fp = fp.reindex(index=close.index, columns=close.columns)
        h = hashlib.md5(formula.encode("utf-8")).hexdigest()[:8]
        parents = sorted({t for t in names if t in formula})
        try:
            lib.register(
                name=f"{PREFIX}{h}",
                panel=fp.replace([np.inf, -np.inf], np.nan).pipe(
                    lambda d: (d - d.mean()) / d.std().replace(0, np.nan)
                ).clip(-5, 5),
                returns_panel=returns,
                kind="raw",
                formula=formula,
                parents=parents,
                source=f"mining:im_combos:pop{args.pop}gen{args.gen}",
                family="非线性组合",
                frequency="日内",
                note=f"第二层 GP 组合（终端=im_*），train t={row['t_stat']:.2f} "
                     f"OOS IC={row['ic_oos']:.4f}",
            )
            registered += 1
        except Exception as e:  # noqa: BLE001
            log.warning("入库失败 %s: %s", formula[:60], e)
    log.info("入库完成: %d 个 gp_im_* 因子", registered)


if __name__ == "__main__":
    main()
