"""E5 变体：基本面族「只对行业中性化」的面板变体（GOOD_MACHINE_TASKS 批次 6）。

对照 build_alla_factor_neutralized（标准三步：去极值 → 行业+市值中性化 → zscore），
本变体对**基本面/股东族**（scripts.pipelines.rolling_grid_alla.FUNDAMENTAL_FAMILY_SETS，
即检验"ortho 二次压制"诊断的目标域）跳过市值中性化、只剥行业——保住 bp/ep/div_yield
的价值/红利风格暴露，其余因子维持标准行业+市值中性化，输出到
``ds_root/panels_neu_fundind/``。

配套跑法（rolling_grid_alla 新增 --preproc ortho_fundind）::

    python scripts/pipelines/rolling_grid_alla.py --preproc ortho_fundind \
        --out-tag ortho920fundind --exclude-features limit_pos \
        --tradable-labels --execution open

用法:
    python -m scripts.builders.build_alla_factor_neutralized_fundind [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = "all_a_2018_2026"
BASE_DIR = Path("reports") / "alla_rolling" / "_base"
MIN_COVERAGE = 0.3
OUT_SUBDIR = "panels_neu_fundind"


def _process_one(name: str) -> tuple[str, float]:
    from factor.preprocessing import preprocess_factor
    from scripts.pipelines.rolling_grid_alla import FUNDAMENTAL_FAMILY_SETS
    ds = Path(str(__import__("config").Config.get()["factor_library"]["root"])) / DATASET
    base = ROOT / BASE_DIR
    p = pd.read_parquet(ds / "panels" / f"{name}.parquet")
    mc = pd.read_parquet(base / "market_cap.parquet")
    ind = pd.read_parquet(base / "cov_industry.parquet")
    common = mc.index.intersection(p.index)
    p = p.reindex(index=common)
    mc = mc.reindex(index=common, columns=p.columns)
    ind = ind.reindex(index=common, columns=p.columns)
    if name in FUNDAMENTAL_FAMILY_SETS:
        # E5 单变量：基本面/股东族只剥行业，保市值（价值/红利）风格信息
        x = preprocess_factor(p, industry_panel=ind)
    else:
        x = preprocess_factor(p, market_cap_panel=mc, industry_panel=ind)
    x = x.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    x = x.clip(-10.0, 10.0)
    cov = float(x.notna().mean().mean())
    x.to_parquet(ds / OUT_SUBDIR / f"{name}.parquet")
    return name, cov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()

    from config import Config
    from scripts.pipelines.rolling_grid_alla import FUNDAMENTAL_FAMILY_SETS

    ds = Path(str(Config.get()["factor_library"]["root"])) / DATASET
    reg = pd.read_csv(ds / "registry.csv")
    names = (args.only if args.only else
             reg.loc[reg["name"].isin(reg["name"]), "name"].tolist())
    if not args.only:
        cov_ok = reg.drop_duplicates(subset="name", keep="last").set_index("name")["coverage"]
        names = [n for n in names if cov_ok.get(n, 0) >= MIN_COVERAGE]
    out_dir = ds / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {f.stem for f in out_dir.glob("*.parquet")}
    todo = [n for n in names if n not in done]
    print(f"E5 变体待中性化因子: {len(todo)}（基本面族 "
          f"{len([n for n in todo if n in FUNDAMENTAL_FAMILY_SETS])}）；"
          f"已存在跳过: {len(done)}", flush=True)

    t0 = time.time()
    stats = []
    if args.workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (name, cov) in enumerate(ex.map(_process_one, todo), 1):
                stats.append({"name": name, "set": "fundind", "coverage": cov})
                print(f"[{i}/{len(todo)}] {name} cov={cov:.3f} ({time.time()-t0:.0f}s)",
                      flush=True)
    else:
        for i, name in enumerate(todo, 1):
            _name, cov = _process_one(name)
            stats.append({"name": _name, "set": "fundind", "coverage": cov})
            print(f"[{i}/{len(todo)}] {_name} cov={cov:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)

    (ds / "factor_stats_neutralized_fundind.jsonl").open("w", encoding="utf-8").write(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in stats) + "\n")
    print(f"E5 变体面板完成 {len(stats)} 个因子 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
