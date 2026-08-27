"""
多因子合成 CLI
=============

挖掘闭环的最后一环：把挖掘出的显著因子**合成**为一个复合因子，并对比
不同合成方式（IC 加权 / PCA / 正交化 / ML stacking）的 IC 与回测表现。

用法:
    # mock 数据（自带 AR(1) 信号），跑四种合成方式并对比
    python scripts/synthesize_factors.py
    # 真实数据：先 python -m scripts.update_data
    python scripts/synthesize_factors.py --real --topk 10 --method all
    # 直接基于已有挖掘结果 CSV
    python scripts/synthesize_factors.py --from reports/factor_mining_xxxx.csv --topk 8
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from factor.preprocessing import standardize_zscore
from factor.synthesis import (
    CompositeInput, build_components, composite_stats,
    rebuild_train_weights,
    synthesize_ic_weighted, synthesize_orthogonal, synthesize_pca, synthesize_stacking,
)
from research.factor_library import FactorLibrary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("synthesize_factors")

METHODS = ["ic_weighted", "pca", "orthogonal", "stacking"]


def _run_mining(panel, returns_panel, features, windows, depth, method, fdr_q):
    """返回挖掘结果 DataFrame（含 name/ic_mean/ir/t_stat/significant）。"""
    from factor.mining import dedup_by_formula, evaluate_candidates, generate_candidates
    cands = dedup_by_formula(generate_candidates(features=features, windows=windows, depth=depth))
    log.info("生成候选因子数: %d", len(cands))
    result = evaluate_candidates(cands, panel, returns_panel, method=method, fdr_q=fdr_q, verbose=True)
    log.info("有效评估: %d，显著(FDR q=%.2f): %d",
             len(result), fdr_q, int(result["significant"].sum()) if len(result) else 0)
    return result


def _select_topk(result: pd.DataFrame, topk: int) -> pd.DataFrame:
    """优先选显著因子；显著不足 topk 时按 |t| 补足。"""
    if result.empty:
        return result
    sig = result[result["significant"]]
    if len(sig) >= topk:
        return sig.head(topk)
    extra = result[~result["significant"]].head(topk - len(sig))
    return pd.concat([sig, extra]).head(topk).reset_index(drop=True)


def _backtest_metrics(composite, returns_panel, strategy_name, k, freq):
    """对单个因子面板跑回测，返回绩效指标 dict。"""
    from backtest import VectorBacktest
    from strategy.examples import QuantileLongShort, TopKLongOnly, TopKLongShort

    strat_map = {
        "topk_ls": TopKLongShort(k=k),
        "topk_lo": TopKLongOnly(k=k),
        "quantile_ls": QuantileLongShort(n_quantiles=5),
    }
    strat = strat_map[strategy_name]
    bt = VectorBacktest(strat, rebalance_freq=freq)
    res = bt.run(composite, returns_panel)
    m = res.metrics()
    return {
        "ann_return": m.get("annual_return", float("nan")),
        "sharpe": m.get("sharpe", float("nan")),
        "max_drawdown": m.get("max_drawdown", float("nan")),
        "win_rate": m.get("win_rate", float("nan")),
    }


def _derive_dates_from_dataset(name: str):
    """从数据集名推导 (begin, end)，如 hs300_2025 → (20250101, 20251231)。"""
    if not name:
        return None, None
    toks = name.split("_")
    yrs = [t for t in toks if t.isdigit() and len(t) == 4]
    if len(yrs) >= 2:
        return int(yrs[0] + "0101"), int(yrs[-1] + "1231")
    if len(yrs) == 1:
        return int(yrs[0] + "0101"), int(yrs[0] + "1231")
    return None, None


def _components_from_library(lib) -> list:
    """从因子库加载 raw 因子面板，包装为 CompositeInput（用存储的 ic/ir 加权）。

    直接读回注册时落盘的面板（已截面标准化），比按公式重建更忠实于库内因子。
    """
    reg = lib.list_all(kind="raw")
    out = []
    for _, row in reg.iterrows():
        name = row["name"]
        panel = lib.get_panel(name)
        if panel is None or panel.empty:
            continue
        ic = float(row.get("ic_mean", 0.0))
        ir = float(row.get("ic_ir", 0.0))
        out.append(CompositeInput(
            name=name,
            panel=standardize_zscore(panel),
            ic=ic, ir=ir,
        ))
    # 按 |IC| 降序，重要因子优先（正交化/加权时更重要）
    out.sort(key=lambda c: abs(c.ic), reverse=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="YuriQuant 多因子合成")
    parser.add_argument("--real", action="store_true", help="真实数据（默认 mock）")
    parser.add_argument("--from", dest="from_csv", default=None, help="直接读已有挖掘结果 CSV")
    parser.add_argument("--begin", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--windows", default="5,10,20,60")
    parser.add_argument("--depth", type=int, default=2, choices=[1, 2])
    parser.add_argument("--method", default="all", choices=METHODS + ["all"])
    parser.add_argument("--topk", type=int, default=10, help="参与合成的因子数")
    parser.add_argument("--fdr-q", type=float, default=0.05)
    parser.add_argument("--ic-method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--strategy", default="topk_ls", choices=["topk_ls", "topk_lo", "quantile_ls"])
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--freq", default="M", choices=["D", "W", "M"])
    parser.add_argument("--train-frac", type=float, default=0.7,
                        help="前段比例用作训练段，其 IC 决定 ic_weighted/orthogonal 的"
                             "权重与符号；后段（1-train_frac）为测试段，仅用于外推评估"
                             "（防未来函数，2026-08-17）")
    parser.add_argument("--save-panels", action="store_true", help="额外保存每个合成方式的复合因子面板(parquet)")
    parser.add_argument("--save-library", action="store_true", help="把各合成方式的复合因子入库（参与下一轮迭代）")
    parser.add_argument("--library-dataset", default=None,
                        help="因子库数据集名（按数据集分库根）。不填自动推导：真实→<指数>_<年>，mock→mock")
    parser.add_argument("--from-library", action="store_true",
                        help="直接从因子库加载 raw 因子参与合成（不重新挖掘），须配合 --library-dataset 指定数据集")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    windows = tuple(int(x) for x in args.windows.split(",") if x.strip())

    # 0) 数据集 / 日期推导（--from-library 时从数据集名推导年份）
    dataset = args.library_dataset
    begin = args.begin
    end = args.end
    if args.from_library:
        if dataset is None and (begin is None or end is None):
            from config import Config
            cfg = Config.get()
            idx = cfg["universe"]["index_code"].split(".")[0]
            yr = str(begin or cfg["fetch"]["begin_date"])[:4]
            dataset = f"{idx}_{yr}"
        if begin is None or end is None:
            yb, ye = _derive_dates_from_dataset(dataset) if dataset else (None, None)
            begin = begin or yb
            end = end or ye

    # 1) 面板
    if args.real or args.from_library:
        from config import Config
        from scripts.mine_factors import build_real_panel
        cfg = Config.get()
        begin = begin or cfg["fetch"]["begin_date"]
        end = end or cfg.get("end_date")
        panel, returns_panel = build_real_panel(cfg, begin, end)
    else:
        from scripts.mine_factors import gen_mock_panel_with_signal
        log.info("使用 Mock 数据（注入 AR(1) 动量信号）...")
        panel = gen_mock_panel_with_signal()
        returns_panel = panel["close"].pct_change().shift(-1)

    # 1.5) 训练/测试段切分（2026-08-17 防未来函数）：
    #      训练段只用于【挖掘选因子 + 决定合成权重/符号】，测试段仅用于外推评估，
    #      绝不参与任何因子选择 / 权重估计。切分须在挖掘之前完成。
    all_dates = returns_panel.index
    train_frac = min(max(args.train_frac, 0.1), 0.9)
    n_train = max(1, int(len(all_dates) * train_frac))
    train_dates = all_dates[:n_train]
    test_dates = all_dates[n_train:]
    if len(test_dates) < 20:
        log.error("测试段交易日过少（%d < 20），请调大 --train-frac 或扩大区间", len(test_dates))
        return
    log.info("切分: 训练段 %d 日(%s~%s) / 测试段 %d 日(%s~%s)",
             len(train_dates), train_dates[0], train_dates[-1],
             len(test_dates), test_dates[0], test_dates[-1])
    returns_train = returns_panel.loc[train_dates]
    returns_test = returns_panel.loc[test_dates]

    features = list(panel.keys())
    log.info("特征: %s", features)

    # 2) 组件来源：优先 --from-library（直接加载库内 raw 因子）
    if args.from_library:
        lib = FactorLibrary(dataset=dataset)
        components = _components_from_library(lib)
        if not components:
            log.error("数据集 %s 无可用 raw 因子，终止。", dataset)
            return
        reg = lib.list_all(kind="raw").rename(columns={"ic_ir": "ir"})
        topk = reg[["name", "ic_mean", "ir", "t_stat", "significant"]].copy()
        log.info("从因子库加载 %d 个 raw 因子（数据集=%s，区间 %s~%s）",
                 len(components), dataset, begin, end)
    else:
        # 挖掘结果
        if args.from_csv:
            result = pd.read_csv(args.from_csv)
            log.info("读取挖掘结果: %s (%d 行)", args.from_csv, len(result))
        else:
            # 关键（2026-08-17）：挖掘/选因子只在【训练段】进行，避免"用未来收益
            # 挑选参与合成的因子"这一层 look-ahead。选出因子后用全样本面板重建，
            # 测试段复合值仅由训练段选定的因子产生 → 无未来函数。
            # panel 可能是单 DataFrame（真实）或 dict{因子名: DataFrame}（mock），需兼容。
            panel_tr = {k: v.loc[train_dates] for k, v in panel.items()} \
                if isinstance(panel, dict) else panel.loc[train_dates]
            rts_tr = returns_panel.loc[train_dates]
            log.info("挖掘在训练段进行（%d 日）...", len(train_dates))
            result = _run_mining(panel_tr, rts_tr, features, windows, args.depth,
                                 args.ic_method, args.fdr_q)

        if result.empty:
            log.error("无可用候选因子，终止。")
            return

        topk = _select_topk(result, args.topk)
        log.info("参与合成的因子数: %d（训练段选出）", len(topk))
        components = build_components(topk, panel, features=features, windows=windows, depth=args.depth)
        log.info("成功重建因子面板(全样本): %d / %d", len(components), len(topk))

    if not components:
        log.error("因子面板构建失败，终止。")
        return

    print("\n===== 参与合成的 Top-%d 因子 =====" % len(topk))
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(topk[["name", "ic_mean", "ir", "t_stat", "significant"]].to_string(index=False))

    # 4) 合成 + 评估 + 回测对比
    # 切分已在步骤 1.5 完成（train_dates/test_dates/returns_train/returns_test）。
    method_list = METHODS if args.method == "all" else [args.method]
    rows = []
    panels_out = {}

    # 权重/符号来源：ic_weighted / orthogonal 只用训练段 IC，杜绝全样本 look-ahead
    train_components = rebuild_train_weights(components, returns_panel, train_dates)

    # 基线：最佳单因子（测试段外推）
    best = train_components[0]
    best_stats = composite_stats(best.panel.loc[test_dates], returns_test, args.ic_method)
    best_bt = _backtest_metrics(best.panel.loc[test_dates], returns_test, args.strategy, args.k, args.freq)
    rows.append({"method": "best_single(" + best.name + ")", **best_stats, **best_bt})

    for m in method_list:
        log.info("合成方式: %s", m)
        # 仅 ic_weighted / orthogonal 的权重由训练段 IC 决定（train_components）；
        # pca / stacking 系列内部已是时序防泄漏，传全样本面板即可。
        if m == "ic_weighted":
            comp = synthesize_ic_weighted(train_components, weight_by="ic_abs")
        elif m == "pca":
            comp = synthesize_pca(components, n_components=min(3, len(components)), returns_panel=returns_panel)
        elif m == "orthogonal":
            comp = synthesize_orthogonal(train_components, weight_by="ic_abs")
        elif m == "stacking":
            comp = synthesize_stacking(components, returns_panel, n_splits=5)
        else:
            continue
        comp_test = comp.loc[test_dates].reindex(columns=returns_test.columns)
        st = composite_stats(comp_test, returns_test, args.ic_method)
        bt = _backtest_metrics(comp_test, returns_test, args.strategy, args.k, args.freq)
        rows.append({"method": m, **st, **bt})
        panels_out[m] = comp

    report = pd.DataFrame(rows)
    print("\n===== 合成方式对比（IC / 回测）=====")
    with pd.option_context("display.max_rows", None, "display.width", 220,
                           "display.float_format", lambda v: f"{v:.4f}"):
        cols = ["method", "ic_mean", "ir", "t_stat", "ic_win_rate",
                "ann_return", "sharpe", "max_drawdown", "win_rate"]
        print(report[cols].to_string(index=False))

    # 5) 保存
    out_path = Path(args.out) if args.out else (
        Path("reports") / f"synthesis_{datetime.now():%Y%m%d_%H%M%S}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    log.info("对比报告已保存: %s", out_path)

    # 实验记录
    try:
        import sys
        from data.cache import DataCache
        from data.datasource import create_datasource
        from research.experiments import record_experiment
        fingerprint = DataCache(create_datasource()).get_fingerprint()
        best_row = report.iloc[0] if len(report) else {}
        record_experiment(
            kind="synthesis",
            command=" ".join(sys.argv),
            params={"real": args.real, "from_csv": args.from_csv,
                    "from_library": args.from_library, "library_dataset": args.library_dataset,
                    "windows": list(windows), "depth": args.depth,
                    "method": args.method, "topk": args.topk,
                    "strategy": args.strategy, "k": args.k, "freq": args.freq},
            data_fingerprint=fingerprint,
            result_path=str(out_path),
            metrics={"best_method": str(best_row.get("method", "")),
                     "best_ic": float(best_row.get("ic_mean", 0.0) or 0.0),
                     "best_sharpe": float(best_row.get("sharpe", 0.0) or 0.0)},
            note=f"{len(components)} 个因子参与合成",
        )
    except Exception as e:
        log.warning("实验记录写入失败（不影响结果）: %s", e)

    if args.save_panels:
        for m, comp in panels_out.items():
            pp = out_path.with_suffix("").as_posix() + f"_composite_{m}.parquet"
            comp.to_parquet(pp)
            log.info("复合因子面板已保存: %s", pp)

    # 6) 复合因子入库（参与下一轮迭代）
    if args.save_library:
        lib_dataset = args.library_dataset
        if lib_dataset is None:
            if args.real or args.from_library:
                from config import Config
                cfg = Config.get()
                idx = cfg["universe"]["index_code"].split(".")[0]
                begin = args.begin or cfg["fetch"]["begin_date"]
                end = args.end or cfg.get("end_date") or begin
                yr_b, yr_e = str(begin)[:4], str(end)[:4]
                yr = yr_b if yr_b == yr_e else f"{yr_b}_{yr_e}"
                lib_dataset = f"{idx}_{yr}"
            else:
                lib_dataset = "mock"
        lib = FactorLibrary(dataset=lib_dataset)
        log.info("复合因子入库到数据集: %s", lib_dataset)
        parents = [c.name for c in components]
        for m, comp in panels_out.items():
            lib.register(
                f"composite_{m}",
                comp, returns_train, kind="composite",
                formula=f"{m}({','.join(parents)}; train_frac={train_frac})",
                parents=parents,
                source=f"synthesis:{m}",
            )
        log.info("复合因子入库完成（%d 个，库内 IC 基于训练段），"
                 "可用 `python scripts/mine_factors.py --use-library` 参与下一轮",
                 len(panels_out))


if __name__ == "__main__":
    main()
