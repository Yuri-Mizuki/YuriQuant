"""AlphaEval 式因子筛选流水线
================================

参考国金证券 Alpha 掘金系列之二十四（AlphaEval, KDD 2026）的三步漏斗：

1. **PPS 预测能力硬门槛** — 用 registry 已有 IC/NW-t/IR 做硬过滤
2. **DPP/log-det 多样性选择** — 从通过门槛的因子中选低冗余子集
3. **RRE 秩稳定性过滤** — 按截面排名自相关（autocorr）可选降换手

输出：``reports/screening/selected_factors.csv``（因子清单 + 各维度得分），
供 ``ml_synthesis_experiment.py`` 的特征供给（``build_feature_set``）使用。

用法::

    # 默认：硬门槛 → DPP 选 70% → RRE 过滤
    python -m scripts.factor_screening --dataset hs300_2022_2025

    # 自定义阈值和目标数
    python -m scripts.factor_screening --dataset hs300_2022_2025 \
        --min-ic 0.015 --min-t-nw 2.0 --dpp-k 200 --min-autocorr 0.3

    # 只做 PPS 硬门槛，不做 DPP/RRE
    python -m scripts.factor_screening --dataset hs300_2022_2025 --no-dpp --no-rre

    # 输出筛选报告
    python -m scripts.factor_screening --dataset hs300_2022_2025 --report
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("factor_screening")

OUTPUT_DIR = Path("reports") / "screening"


# ===========================================================================
# Step 1: PPS 预测能力硬门槛
# ===========================================================================
def pps_filter(
    reg: pd.DataFrame,
    min_ic: float = 0.015,
    min_t_nw: float = 2.0,
    min_coverage: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """预测能力硬门槛筛选。

    Args:
        reg: FactorLibrary.list_all() 返回的 registry DataFrame。
        min_ic: |IC 均值| 下限。
        min_t_nw: NW-t 统计量绝对值下限。
        min_coverage: 覆盖率下限（用 n_codes/n_dates 代理，不读面板）。

    Returns:
        (passed, rejected) 两个 DataFrame。
    """
    df = reg.copy()
    df["ic_abs"] = df["ic_mean"].fillna(0).abs()
    df["t_nw_abs"] = df["t_stat_nw"].fillna(0).abs()

    mask = (df["ic_abs"] >= min_ic) & (df["t_nw_abs"] >= min_t_nw)
    # 覆盖率代理：n_codes > 0 且非全 NaN 面板
    if "n_codes" in df.columns:
        mask &= df["n_codes"].fillna(0) > 0

    passed = df[mask].copy()
    rejected = df[~mask].copy()

    log.info("PPS 硬门槛: |IC|>=%.4f, |t_nw|>=%.1f → 通过 %d / 淘汰 %d",
             min_ic, min_t_nw, len(passed), len(rejected))
    return passed, rejected


# ===========================================================================
# Step 2: DPP/log-det 多样性选择
# ===========================================================================
def dpp_filter(
    lib,
    names: list[str],
    k: int | None = None,
    quality_col: str = "ic_mean",
    sigma: float = 0.2,
) -> tuple[list[str], dict]:
    """DPP 集合级多样性筛选。

    Args:
        lib: FactorLibrary 实例。
        names: PPS 通过的因子名列表。
        k: 目标入选数（None=70% 池大小，对齐研报 800/1134）。
        quality_col: 质量列（默认 ic_mean）。
        sigma: 相似度核带宽。

    Returns:
        (selected_names, dpp_result_dict)
    """
    if k is None:
        k = max(1, int(np.ceil(0.7 * len(names))))
    k = min(k, len(names))

    log.info("DPP 筛选: 池 %d → 目标 %d", len(names), k)
    res = lib.select_diverse(
        names=names, k=k, quality_col=quality_col, sigma=sigma,
    )
    selected = res["selected"]
    log.info("DPP 完成: 入选 %d, max|corr| %.3f→%.3f, mean|corr| %.3f→%.3f",
             len(selected),
             res["max_abs_corr_pool"], res["max_abs_corr_selected"],
             res["mean_abs_corr_pool"], res["mean_abs_corr_selected"])
    return selected, res


# ===========================================================================
# Step 3: RRE 秩稳定性过滤
# ===========================================================================
def rre_filter(
    reg: pd.DataFrame,
    names: list[str],
    min_autocorr: float = 0.0,
    max_autocorr: float | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """秩稳定性过滤：按因子截面排名自相关过滤。

    autocorr 已在 registry 中（因子截面排名的日频自相关）。
    - 低 autocorr（→0）= 排名天天变 = 高换手噪声
    - 高 autocorr（→1）= 排名不变 = 信号无更新（也可能有价值）

    Args:
        reg: registry DataFrame。
        names: DPP 入选因子名列表。
        min_autocorr: 自相关下限（低于此剔除，降换手）。
        max_autocorr: 自相关上限（高于此剔除，防信号停滞）；None=不限。

    Returns:
        (selected_names, score_df) — 过滤后因子名 + 各维度得分表。
    """
    df = reg[reg["name"].isin(names)].copy()
    ac = df["autocorr"].fillna(0.0)

    mask = ac >= min_autocorr
    if max_autocorr is not None:
        mask &= ac <= max_autocorr

    kept = df[mask]
    dropped = df[~mask]

    log.info("RRE 秩稳定性: autocorr ∈ [%.2f, %s] → 保留 %d / 淘汰 %d",
             min_autocorr, str(max_autocorr) if max_autocorr else "∞",
             len(kept), len(dropped))

    # 构建综合得分表
    score = kept[["name", "ic_mean", "t_stat_nw", "ic_ir", "autocorr"]].copy()
    score["ic_abs"] = score["ic_mean"].abs()
    score.columns = ["因子名", "IC均值", "NW-t", "IC_IR", "排名自相关", "|IC|"]

    return kept["name"].tolist(), score


# ===========================================================================
# 主流程
# ===========================================================================
def run_screening(
    dataset: str = "",
    min_ic: float = 0.015,
    min_t_nw: float = 2.0,
    dpp_k: int | None = None,
    dpp_ratio: float = 0.7,
    sigma: float = 0.05,
    min_autocorr: float = 0.0,
    max_autocorr: float | None = None,
    no_dpp: bool = False,
    no_rre: bool = False,
    report: bool = False,
) -> dict:
    """运行完整 AlphaEval 式三步筛选流水线。

    Returns:
        dict with keys: pps_passed, dpp_selected, rre_passed, scores, output_path
    """
    from research.factor_library import FactorLibrary

    lib = FactorLibrary(dataset=dataset)
    reg = lib.list_all()
    log.info("因子库 %s: 共 %d 个因子", dataset, len(reg))

    # Step 1: PPS 硬门槛
    pps_passed, pps_rejected = pps_filter(reg, min_ic=min_ic, min_t_nw=min_t_nw)
    if len(pps_passed) == 0:
        log.error("PPS 硬门槛后无因子存活，请调低阈值")
        return {"pps_passed": [], "dpp_selected": [], "rre_passed": []}

    names = pps_passed["name"].tolist()
    dpp_names = list(names)                  # DPP 关闭/退化时 = PPS 通过集

    # Step 2: DPP 多样性选择
    dpp_result = None
    if not no_dpp and len(names) > 1:
        k = dpp_k if dpp_k is not None else int(np.ceil(dpp_ratio * len(names)))
        names, dpp_result = dpp_filter(lib, names, k=k, quality_col="ic_mean",
                                       sigma=sigma)
        dpp_names = list(names)

    # Step 3: RRE 秩稳定性过滤
    scores = None
    if not no_rre and len(names) > 0:
        names, scores = rre_filter(reg, names, min_autocorr=min_autocorr,
                                  max_autocorr=max_autocorr)
    else:
        scores = pps_passed[["name", "ic_mean", "t_stat_nw", "ic_ir", "autocorr"]].copy()
        scores = scores[scores["name"].isin(names)]

    # 输出
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "selected_factors.csv"
    out_df = reg[reg["name"].isin(pps_passed["name"])].copy()
    # 各筛选维度标签：与同名字段一一对应，避免下游按 rre_passed 混用
    out_df["pps_passed"] = True
    out_df["dpp_selected"] = out_df["name"].isin(dpp_names)
    out_df["rre_passed"] = out_df["name"].isin(names)
    out_df["final_selected"] = out_df["rre_passed"]
    out_df = out_df.sort_values("ic_mean", key=lambda s: s.abs(), ascending=False)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("筛选完成: %d 个因子 → %s", len(names), out_path)

    # 筛选报告
    if report:
        _generate_report(dataset, len(reg), len(pps_passed),
                         len(names), dpp_result, pps_rejected, out_path)

    return {
        "pps_passed": pps_passed["name"].tolist(),
        "dpp_selected": dpp_names,
        "rre_passed": names,
        "scores": scores,
        "output_path": str(out_path),
        "n_pool": len(reg),
        "n_pps": len(pps_passed),
        "n_final": len(names),
    }


def _generate_report(dataset, n_pool, n_pps, n_final, dpp_result, pps_rejected, out_path):
    """生成文本筛选报告。"""
    report_path = OUTPUT_DIR / "screening_report.txt"
    lines = [
        f"AlphaEval 式因子筛选报告",
        f"{'='*60}",
        f"数据集: {dataset}",
        f"因子池: {n_pool}",
        f"",
        f"Step 1 — PPS 预测能力硬门槛:",
        f"  通过: {n_pps} / {n_pool}",
        f"  淘汰: {len(pps_rejected)}",
        f"",
    ]
    if dpp_result is not None:
        lines += [
            f"Step 2 — DPP/log-det 多样性选择:",
            f"  入选: {dpp_result['k']} / {dpp_result['n_pool']}",
            f"  max|corr|: {dpp_result['max_abs_corr_pool']:.3f} → {dpp_result['max_abs_corr_selected']:.3f}",
            f"  mean|corr|: {dpp_result['mean_abs_corr_pool']:.3f} → {dpp_result['mean_abs_corr_selected']:.3f}",
            f"  log-det: {dpp_result['logdet_selected']:.1f}",
            f"",
        ]
    lines += [
        f"Step 3 — RRE 秩稳定性过滤:",
        f"  最终入选: {n_final}",
        f"",
        f"输出: {out_path}",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("筛选报告: %s", report_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="AlphaEval 式因子筛选流水线")
    p.add_argument("--dataset", default="hs300_2022_2025", help="因子库数据集名")
    p.add_argument("--min-ic", type=float, default=0.015, help="|IC 均值| 硬门槛")
    p.add_argument("--min-t-nw", type=float, default=2.0, help="NW-t 绝对值硬门槛")
    p.add_argument("--dpp-k", type=int, default=None, help="DPP 目标入选数（默认 70%%）")
    p.add_argument("--dpp-ratio", type=float, default=0.7, help="DPP 入选比例（k 未指定时生效）")
    p.add_argument("--sigma", type=float, default=0.2, help="DPP 相似度核带宽")
    p.add_argument("--min-autocorr", type=float, default=0.0, help="RRE 自相关下限（降换手）")
    p.add_argument("--max-autocorr", type=float, default=None, help="RRE 自相关上限（防信号停滞）")
    p.add_argument("--no-dpp", action="store_true", help="跳过 DPP 多样性筛选")
    p.add_argument("--no-rre", action="store_true", help="跳过 RRE 秩稳定性过滤")
    p.add_argument("--report", action="store_true", help="生成文本筛选报告")
    args = p.parse_args()

    res = run_screening(
        dataset=args.dataset,
        min_ic=args.min_ic,
        min_t_nw=args.min_t_nw,
        dpp_k=args.dpp_k,
        dpp_ratio=args.dpp_ratio,
        min_autocorr=args.min_autocorr,
        max_autocorr=args.max_autocorr,
        no_dpp=args.no_dpp,
        no_rre=args.no_rre,
        report=args.report,
    )
    print(f"\n筛选完成: {res['n_pool']} → PPS {res['n_pps']} → 最终 {res['n_final']}")
    print(f"输出: {res['output_path']}")


if __name__ == "__main__":
    main()
