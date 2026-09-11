"""预计算全注册因子的因子层中性化面板（#3 优化）。

对每一注册因子面板做业界标准三步预处理：
    去极值(MAD) → 行业哑变量 + 市值(logERC) 中性化 → 截面 z-score
（factor.preprocessing::preprocess_factor，逐日截面、无未来函数）。

产出写入 ds_root/panels_neu/{name}.parquet，供滚动实验以
`--preproc ortho` 读取做对照 —— 即把风格中性化从"信号层/组合层"
下放到"因子层/模型输入层"，让线性模型(ridge)也能利用基本面慢变量。

中性化所用市值/行业面板来自 reports/alla_rolling/_base（与 close_adj
同 date×code 对齐）。multiprocessing 加速（每因子独立，写不同 parquet）。

用法:
    python -m scripts.builders.build_alla_factor_neutralized [--workers 6]
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
MIN_COVERAGE = 0.3   # 低于此覆盖率的因子不进入模型候选，跳过中性化省算力


def _process_one(name: str) -> tuple[str, float]:
    from factor.preprocessing import preprocess_factor
    ds = Path(str(__import__("config").Config.get()["factor_library"]["root"])) / DATASET
    base = ROOT / BASE_DIR
    p = pd.read_parquet(ds / "panels" / f"{name}.parquet")
    mc = pd.read_parquet(base / "market_cap.parquet")
    ind = pd.read_parquet(base / "cov_industry.parquet")
    common = mc.index.intersection(p.index)
    p = p.reindex(index=common)
    mc = mc.reindex(index=common, columns=p.columns)
    ind = ind.reindex(index=common, columns=p.columns)
    x = preprocess_factor(p, market_cap_panel=mc, industry_panel=ind)
    x = x.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    x = x.clip(-10.0, 10.0)
    cov = float(x.notna().mean().mean())
    x.to_parquet(ds / "panels_neu" / f"{name}.parquet")
    return name, cov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="+", default=None,
                    help="只处理指定因子名（调试用）")
    args = ap.parse_args()

    from config import Config
    ds = Path(str(Config.get()["factor_library"]["root"])) / DATASET
    reg = pd.read_csv(ds / "registry.csv")
    names = (args.only if args.only else
             reg.loc[reg["name"].isin(reg["name"]), "name"].tolist())
    if not args.only:
        cov_ok = reg.set_index("name")["coverage"]
        names = [n for n in names if cov_ok.get(n, 0) >= MIN_COVERAGE]
    out_dir = ds / "panels_neu"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {f.stem for f in out_dir.glob("*.parquet")}
    todo = [n for n in names if n not in done]
    print(f"待中性化因子: {len(todo)}；已存在跳过: {len(done)}", flush=True)

    t0 = time.time()
    stats = []
    if args.workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (name, cov) in enumerate(ex.map(_process_one, todo), 1):
                stats.append({"name": name, "set": "neutralized", "coverage": cov})
                print(f"[{i}/{len(todo)}] {name} cov={cov:.3f} ({time.time()-t0:.0f}s)",
                      flush=True)
    else:
        for i, name in enumerate(todo, 1):
            _name, cov = _process_one(name)
            stats.append({"name": _name, "set": "neutralized", "coverage": cov})
            print(f"[{i}/{len(todo)}] {_name} cov={cov:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)

    (ds / "factor_stats_neutralized.jsonl").open("w", encoding="utf-8").write(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in stats) + "\n")
    print(f"中性化面板完成 {len(stats)} 个因子 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()