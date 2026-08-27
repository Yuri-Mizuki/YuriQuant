"""
DPP 集合级筛选 vs 两两贪心去重 —— 真实因子库实测对比
====================================================
研报对齐（国金系列之二十四 §3.1）：多样性筛选用 log-det 最大化（集合级），
对比现有"两两贪心去重"（spearman/相关阈值）。

用法（真实因子库，只读不写）::

    python scripts/dpp_library_compare.py                 # 默认 hs300_2025 全池
    python scripts/dpp_library_compare.py --dataset hs300_2025 --k 80
    python scripts/dpp_library_compare.py --source filter  # 支持后续按 source 过滤

产出：reports/dpp_vs_pairwise_<dataset>.csv + 控制台摘要。
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("dpp_compare")

from research.dpp_selection import corr_matrix, dpp_select, pairwise_dedup  # noqa: E402
from research.factor_library import FactorLibrary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="DPP vs 两两去重实测对比")
    ap.add_argument("--dataset", default="hs300_2025")
    ap.add_argument("--k", type=int, default=None, help="DPP 目标数量（默认 ceil(0.7n)）")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="两两去重相关阈值（默认 0.7，对齐 check_dup）")
    ap.add_argument("--method", default="cross", choices=["cross", "flat"])
    ap.add_argument("--kind", default=None, help="只比较某类因子（raw/composite）")
    args = ap.parse_args()

    lib = FactorLibrary(dataset=args.dataset)
    reg = lib.list_all(kind=args.kind)
    names = list(reg["name"])
    t0 = time.time()
    panels: dict[str, pd.DataFrame] = {}
    for n in names:
        p = lib.get_panel(n)
        if p is not None and not p.empty:
            panels[n] = p
    log.info("载入 %d/%d 因子面板 (%.1fs)", len(panels), len(names), time.time() - t0)

    t0 = time.time()
    corr = corr_matrix(panels, method=args.method)
    log.info("相关矩阵 %d×%d 计算完成 (%.1fs)", *corr.shape, time.time() - t0)

    n = len(corr)
    k = args.k if args.k else int(np.ceil(0.7 * n))

    # ---- DPP（纯多样性 + 质量加权两种） ----
    ic = reg.set_index("name")["ic_mean"].reindex(corr.index)
    res_pure = dpp_select(corr, k=k, quality=None, sigma=0.2)
    res_qual = dpp_select(corr, k=k, quality=ic.abs(), sigma=0.2)

    # ---- 两两贪心去重（基线）：按 |IC| 降序，|corr|>threshold 剔除 ----
    order = list(ic.abs().sort_values(ascending=False).index)
    sel_pair = pairwise_dedup(corr, order=order, threshold=args.threshold)

    # ---- 全池参考 ----
    C = np.abs(corr.to_numpy())
    off = C[~np.eye(n, dtype=bool)]

    def _stats(idx) -> dict:
        if len(idx) <= 1:
            return {"max_abs_corr": float("nan"), "mean_abs_corr": float("nan"),
                    "logdet": float("nan"), "ic_mean_abs": float("nan")}
        sub = C[np.ix_(idx, idx)]
        s = sub[~np.eye(len(idx), dtype=bool)]
        return {
            "max_abs_corr": float(s.max()),
            "mean_abs_corr": float(s.mean()),
            "logdet": float(np.linalg.slogdet(
                np.exp(-(1 - np.abs(corr.to_numpy()[np.ix_(idx, idx)])) / 0.2))[1]),
            "ic_mean_abs": float(ic.iloc[idx].abs().mean()),
        }

    pool_st = _stats(list(range(n)))
    rows = {
        "全池(不筛)": {"n": n, **pool_st},
        "DPP纯多样性": {"n": len(res_pure["selected"]),
                     **_stats([corr.index.get_loc(x) for x in res_pure["selected"]])},
        "DPP质量加权": {"n": len(res_qual["selected"]),
                     **_stats([corr.index.get_loc(x) for x in res_qual["selected"]])},
        f"两两去重|corr|>{args.threshold}": {"n": len(sel_pair),
                     **_stats([corr.index.get_loc(x) for x in sel_pair])},
    }
    out = pd.DataFrame(rows).T
    print("\n===== DPP vs 两两去重（%s，%d 因子） =====" % (args.dataset, n))
    print(out.round(4).to_string())

    # 落盘
    report = Path("reports")
    report.mkdir(exist_ok=True)
    csv = report / f"dpp_vs_pairwise_{args.dataset}_{args.method}.csv"
    out.round(6).to_csv(csv, encoding="utf-8-sig")
    # 入选名单
    for label, res in (("dpp_pure", res_pure), ("dpp_quality", res_qual),
                       ("pairwise", sel_pair)):
        names_out = pd.DataFrame({"name": res["selected"] if isinstance(res, dict)
                                  else res})
        names_out.to_csv(report / f"dpp_{label}_{args.dataset}_{args.method}.csv",
                         index=False, encoding="utf-8-sig")
    log.info("报告已写: %s", csv)


if __name__ == "__main__":
    main()
