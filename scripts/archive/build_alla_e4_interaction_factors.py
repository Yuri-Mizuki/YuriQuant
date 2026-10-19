"""E4 交互特征：基本面状态分桶 × 量价因子组内 zscore（GOOD_MACHINE_TASKS 批次 6）。

检验「基本面当条件变量」假设：量价因子在同基本面状态组内做截面 zscore，
显式把"基本面状态"变成量价信号的评估域，而不是让模型自己学交互。
单变量对照 E5（基本面只剥行业）——两者都回答"基本面信息的正确打开方式"。

设计（3 桶变量 × 5 量价因子 = 15 个候选面板，名 e4_{buk}_x_{px}）：
- 桶变量（读 ds panels 的基本面 raw 面板，tercile 分桶，逐日截面、PIT 安全）：
  bp（价值）、roe_ttm（盈利）、np_growth_yoy（成长）
- 量价因子（从 daily_all_a 现算，与 alpha 建库者同源）：
  mom_20 / mom_60（动量）、rev_5（短反转）、vol_20（波动）、turn_20（换手亲和）
- 组内 zscore：逐日、桶内 mean/std；桶缺失（基本面 NaN）→ 面板 NaN
  （交互特征的定义域就是"有基本面状态"的股票）

产出经 common.merge_outputs 入 registry/ic_h（set=E4X），滚动实验照常消费。

用法:
    python -m scripts.builders.build_alla_e4_interaction_factors [--resume]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = "all_a_2018_2026"
KEEP_FROM = "2016-07-01"

BUCKET_VARS = ("bp", "roe_ttm", "np_growth_yoy")
PRICE_DEFS = {
    "mom_20": ("mom", 20),
    "mom_60": ("mom", 60),
    "rev_5": ("rev", 5),
    "vol_20": ("vol", 20),
    "turn_20": ("turn", 20),
}


def _load_close_vol() -> tuple[pd.DataFrame, pd.DataFrame]:
    from scripts.builders.common import load_close_adj
    import config
    root = Path(str(config.Config.get()["cache"]["root"]))
    close = load_close_adj()
    vol = pd.read_parquet(root / "daily_all_a.parquet")["volume"].unstack("code")
    vol = vol.reindex(index=close.index, columns=close.columns)
    return close, vol


def _price_panel(kind: str, win: int, close: pd.DataFrame,
                 vol: pd.DataFrame) -> pd.DataFrame:
    if kind == "mom":
        p = close / close.shift(win) - 1.0
    elif kind == "rev":
        p = -(close / close.shift(win) - 1.0)
    elif kind == "vol":
        p = close.pct_change(fill_method=None).rolling(win).std()
    elif kind == "turn":
        # 换手亲和：成交量/过去 win 均量（无流通股本口径依赖）
        p = vol / vol.rolling(win).mean()
    else:
        raise ValueError(kind)
    return p.astype(np.float32)


def _tercile_buckets(f: pd.DataFrame) -> pd.DataFrame:
    """逐日截面 tercile：1/2/3；NaN 保持 NaN（无基本面状态 → 交互不定义）。"""
    rank = f.rank(axis=1, pct=True)          # 0~1，NaN 保留
    b = pd.DataFrame(np.nan, index=f.index, columns=f.columns, dtype=float)
    b = b.mask(rank.notna(), 0.0)
    b = b + (rank > 1.0 / 3).astype(float) + (rank > 2.0 / 3).astype(float)
    return b.where(rank.notna())


def _group_zscore(x: pd.DataFrame, bucket: pd.DataFrame) -> pd.DataFrame:
    """逐日、桶内 zscore。"""
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=np.float32)
    for g in (1.0, 2.0, 3.0):
        mask = bucket == g
        xm = x.where(mask)
        mu = xm.mean(axis=1)
        sd = xm.std(axis=1, ddof=0).replace(0.0, np.nan)
        out = out.mask(mask, xm.sub(mu, axis=0).div(sd, axis=0))
    return out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from stats.ic import calc_ic_series
    from scripts.builders.common import HORIZONS, IC_CODE_STRIDE

    import config
    ds_dir = Path(str(config.Config.get()["factor_library"]["root"])) / DATASET
    panels_dir = ds_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    stats_path = ds_dir / "factor_stats_e4x.jsonl"
    done = {json.loads(l)["name"] for l in
            stats_path.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if args.resume and stats_path.exists() else set()

    close, vol = _load_close_vol()
    close_adj = close  # _load_close_adj 已含后复权（动量/波动口径自洽）

    # 标签实现期与建库者同款：ic_h 对齐 close_adj
    fwd = {h: close_adj.pct_change(h, fill_method=None).shift(-h) for h in HORIZONS}
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]

    todo = [(b, p) for b in BUCKET_VARS for p in PRICE_DEFS]
    if args.resume:
        todo = [(b, p) for b, p in todo if f"e4_{b}_x_{p}" not in done]
    print(f"E4 交互因子待构建: {len(todo)}", flush=True)

    t0 = time.time()
    n = 0
    for i, (buk, pname) in enumerate(todo, 1):
        name = f"e4_{buk}_x_{pname}"
        fpath = panels_dir / f"{buk}.parquet"
        if not fpath.exists():
            print(f"[跳过] 桶变量面板缺失: {buk}（先跑 fundamental builder）", flush=True)
            continue
        f = pd.read_parquet(fpath).reindex(index=close_adj.index,
                                           columns=close_adj.columns)
        pdef = PRICE_DEFS[pname]
        px = _price_panel(pdef[0], pdef[1], close_adj, vol)
        bucket = _tercile_buckets(f)
        panel = _group_zscore(px, bucket)
        panel = panel.replace([np.inf, -np.inf], np.nan)
        k0 = pd.Timestamp(KEEP_FROM)
        panel = panel.loc[panel.index >= k0]
        panel.to_parquet(panels_dir / f"{name}.parquet")
        cov = float(panel.notna().mean().mean())
        ic = {h: calc_ic_series(panel[ic_codes], fwd[h].loc[panel.index])
              for h in HORIZONS}
        row = {"name": name, "set": "e4x", "label": f"E4 {buk}×{pname}",
               "coverage": cov,
               **{f"ic_mean_h{h}": float(ic[h].mean()) for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
        print(f"[{i}/{len(todo)}] {name} cov={cov:.3f} "
              f"ic_h1={row['ic_mean_h1']:+.4f} ({time.time()-t0:.0f}s)", flush=True)

    if n:
        from scripts.builders.common import merge_outputs
        merge_outputs(ds_dir, "e4x")
    print(f"E4 完成 {n} 个因子 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
