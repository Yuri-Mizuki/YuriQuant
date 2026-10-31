"""新因子 vs 生产池：最大截面相关扫描（判「是不是老东西换皮」，2026-10-27）。

## 为什么需要这一步

`residual_ic_probe` 的 `resfac` / `both` 口径只对**主信号** rank 正交化。
主信号是 GBDT 集成，它「没 span 掉」某个因子有两种解释：

1. **真有增量信息**（池里没有这种东西）；
2. **池里有，但模型没用好**（拟合不足 / 特征被正交化洗弱 / 树模型对单调线性不敏感）。

只看残差 IC 分不开这两者，而分不开就会把「模型调参问题」误判成「新因子族」。
本脚本补这一步：拿候选因子去和**生产池实际在用的因子**逐个比截面秩相关，
最大相关高 ⇒ 换皮（走 2），低 ⇒ 才有资格谈 1。

## 对照集

- `--pool-from selection`（默认）：`reports/alla_rolling_ortho/selection/y*__h1.json`
  九年 DPP 选中并集 —— **模型真正吃进去的那批**（h1 共 143 个）。这是最有意义
  的对照，因为没被 DPP 选中的因子本来就没进模型。
- `--pool-from all`：整个 `panels_neu`（950 张），作为下界证据（慢，约 20 分钟）。

## 口径

- 两组都是**中性化后**面板（`panels_neu` vs 候选的 `panels_neu`）：先剥掉行业 +
  市值 + 动量/波动/换手，再比内容。否则共同的风格暴露会把相关抬到 0.8+ 的假高。
- 秩相关（Spearman），**按日算再取均值**（与 `residual_ic_probe` 的 IC 口径一致），
  不用「先堆池再算一个总相关」——后者会被时间维的截面规模变化带偏。
- 日期 `--stride` 采样（默认 5，约 420 天）：相关估计的标准误 ~1/√420 ≈ 0.05，
  足够把 0.9 与 0.7 分开，而 IO 只有 1/5。

## 用法

    python -m scripts.evaluation.pool_maxcorr_probe \
        --cand-dir reports/chip_behavior_probe_2026-10-27/all_a_2018_2026/panels_neu

    # 全池下界（慢）
    python -m scripts.evaluation.pool_maxcorr_probe --pool-from all --stride 10
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("pool_maxcorr_probe")

PROD = ROOT / "reports" / "alla_rolling_ortho"


def _ranked(fp: Path, idx: pd.DatetimeIndex,
            cols: pd.Index, mask: pd.DataFrame) -> pd.DataFrame | None:
    """读面板 → 对齐 → 掩码内逐日秩（float32）。"""
    x = pd.read_parquet(fp)
    if x.columns.has_duplicates:
        x = x.loc[:, ~x.columns.duplicated()]
    x = x.select_dtypes(include=[np.number]).reindex(index=idx, columns=cols)
    if x.shape[1] == 0 or x.notna().to_numpy().sum() < 1000:
        return None
    x = x.replace([np.inf, -np.inf], np.nan)
    return x.where(mask).rank(axis=1).astype(np.float32)


def _corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """逐日截面秩相关均值（两面板已在同一掩码内 rank 过）。"""
    return float(a.corrwith(b, axis=1).mean())


def selection_pool() -> list[str]:
    """h1 九年 DPP 选中并集。"""
    names: set[str] = set()
    for f in glob.glob(str(PROD / "selection" / "y*__h1.json")):
        names |= set(json.load(open(f, encoding="utf-8")))
    return sorted(names)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cand-dir", required=True)
    ap.add_argument("--cand", nargs="+", default=None,
                    help="候选因子名（默认目录下全部）")
    ap.add_argument("--pool-dir", default=None,
                    help="池面板目录（默认生产 panels_neu）")
    ap.add_argument("--pool-from", choices=["selection", "all"], default="selection")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--top", type=int, default=5, help="每个候选输出前 K 个最相关")
    ap.add_argument("--out", default=None, help="CSV 输出（默认候选目录同级）")
    args = ap.parse_args(argv)

    from config import Config
    from scripts.common.alla_common import load_base

    ds = Path(str(Config.get()["factor_library"]["root"])) / "all_a_2018_2026"
    pool_dir = Path(args.pool_dir) if args.pool_dir else ds / "panels_neu"
    cand_dir = Path(args.cand_dir)

    base = load_base()
    mask_all = base["mask"].astype(bool)

    cand_files = ([cand_dir / f"{n}.parquet" for n in args.cand]
                  if args.cand else sorted(cand_dir.glob("*.parquet")))
    # 日期/列口径以候选面板为准，采样后与池取交集
    probe = pd.read_parquet(cand_files[0])
    idx = probe.index[:: args.stride]
    cols = probe.columns

    pool_names = (selection_pool() if args.pool_from == "selection"
                  else [p.stem for p in sorted(pool_dir.glob("*.parquet"))])
    pool_names = [n for n in pool_names if (pool_dir / f"{n}.parquet").exists()]
    log.info("候选 %d 个 × 池 %d 个（%s）| 采样 %d 日",
             len(cand_files), len(pool_names), args.pool_from, len(idx))

    mask = mask_all.reindex(index=idx, columns=cols).fillna(False).astype(bool)

    # 候选只读一次（8 个 × 60MB），否则内层循环会把 IO 放大 池因子数 倍
    cands: dict[str, pd.DataFrame] = {}
    for cf in cand_files:
        r = _ranked(cf, idx, cols, mask)
        if r is not None:
            cands[cf.stem] = r

    rows: list[dict] = []
    t0 = time.time()
    for j, pf in enumerate(pool_names, 1):
        b = _ranked(pool_dir / f"{pf}.parquet", idx, cols, mask)
        if b is None:
            log.info("[%3d/%d] %-22s 跳过（空面板）", j, len(pool_names), pf)
            continue
        for name, a in cands.items():
            rows.append({"cand": name, "pool": pf,
                         "corr": round(_corr(a, b), 4)})
        if j % 20 == 0:
            log.info("[%3d/%d] %.0fs", j, len(pool_names), time.time() - t0)

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("没有产出")
        return
    df["abs"] = df["corr"].abs()
    df = df.sort_values(["cand", "abs"], ascending=[True, False])
    # 默认名带 pool 口径与面板目录名：neut vs raw 两组常在同一 parent 下，
    # 共用 pool_maxcorr.csv 会互相覆盖（2026-10-27 踩过）。
    out = (Path(args.out) if args.out
           else cand_dir.parent / f"pool_maxcorr_{args.pool_from}_{cand_dir.name}.csv")
    df.drop(columns="abs").to_csv(out, index=False, encoding="utf-8-sig")

    log.info("=" * 78)
    for name, g in df.groupby("cand", sort=True):
        top = g.head(args.top)
        log.info("%-14s max|r| = %+.3f   ->  %s", name, top["corr"].iloc[0],
                 ", ".join(f"{r.pool}={r.corr:+.3f}" for r in top.itertuples()))
    log.info("完成 %.0fs -> %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
