"""LLM-MCTS 候选公式入库（all-A 域求值，2026-10-08）。

从 reports/llm_mcts_phase0/<最新>/candidates.csv 读 accepted 公式（raw 6 字段，
formula_project 语法），在全A 面板上求值 → panels/llm_* + registry/ic_h 融合
（merge_outputs，family='llmmcts'）。席位竞争力由 ic_h 排名与 DPP 对照裁决。
用法: python -m scripts.builders.build_alla_llmmcts_factors [--csv path]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = "all_a_2018_2026"
KEEP_FROM = "2016-07-01"
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="candidates.csv（缺省取 llm_mcts_phase0 最新目录）")
    args = ap.parse_args()
    csv = Path(args.csv) if args.csv else sorted(
        Path("reports/llm_mcts_phase0").glob("*/candidates.csv"))[-1]
    cand = pd.read_csv(csv)
    acc = cand[cand["status"] == "accepted"]
    print(f"accepted: {len(acc)}（源 {csv}）", flush=True)

    from scripts.builders.build_alla_alpha_panels import load_alla_panels
    from factor.formula import formula_builder
    from stats.ic import calc_ic_series
    from scripts.builders.common import HORIZONS, IC_CODE_STRIDE, merge_outputs

    panels_px, _ind, close_adj = load_alla_panels(20150101)
    k0 = pd.Timestamp(KEEP_FROM)
    close_kept = close_adj.loc[close_adj.index >= k0]
    fwd = {h: close_kept.pct_change(h, fill_method=None).shift(-h) for h in HORIZONS}
    ic_codes = close_kept.columns[::IC_CODE_STRIDE]

    from config import Config
    ds = Path(str(Config.get()["factor_library"]["root"])) / DATASET
    (ds / "panels").mkdir(parents=True, exist_ok=True)
    stats_path = ds / "factor_stats_llmmcts.jsonl"
    node_cache: dict = {}
    n = 0
    for i, row in enumerate(acc.itertuples(), 1):
        f = str(row.formula_project).strip()
        name = f"llm_{i:02d}"
        try:
            p = formula_builder(f, features=FEATURES, node_cache=node_cache)(panels_px)
        except Exception as e:
            print(f"[跳过] {f}: {e}", flush=True); continue
        p = p.loc[p.index >= k0].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        p.to_parquet(ds / "panels" / f"{name}.parquet")
        cov = float(p.notna().mean().mean())
        icm = {h: float(calc_ic_series(p[ic_codes], fwd[h]).mean()) for h in HORIZONS}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"name": name, "set": "llmmcts", "label": f[:60],
                                 "coverage": cov,
                                 **{f"ic_mean_h{h}": icm[h] for h in HORIZONS}},
                                ensure_ascii=False) + "\n")
        print(f"[{i}/{len(acc)}] {name} cov={cov:.3f} ic_h1={icm[1]:+.4f} | {f[:50]}",
              flush=True)
        n += 1
    if n:
        merge_outputs(ds, "llmmcts")
    print(f"完成 {n} 因子", flush=True)


if __name__ == "__main__":
    main()
