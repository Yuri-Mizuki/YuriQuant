"""ic_h 融合入口：把指定因子（缺省=库内全部）的 IC 列补进 ic_h{1,5,10,20}。

为什么常驻（2026-11-22）：fd/im 等 factors 通道 builder 不经过 builders.common
的 IC 融合步骤，ic_h 曾停在 11-07、漏掉整个分钟族——「要 IC 先读这里」的
约定会静默漏列。本入口由 run_factor_pipeline 的 evaluate 段按 names 调用，
幂等（已存在列跳过）。

用法::
    python -m scripts.factors.fuse_ic --dataset all_a_2018_2026
    python -m scripts.factors.fuse_ic --dataset all_a_2018_2026 --names a,b,c
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.builders.common import fuse_horizon_ic
from scripts.common.cli_common import setup_logging

log = setup_logging("fuse_ic")


def main() -> None:
    ap = argparse.ArgumentParser(description="ic_h 四面板融合补齐")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--names", default=None, help="逗号分隔；缺省=registry 全部")
    args = ap.parse_args()

    ds = Path(Config_root()) / args.dataset
    if args.names:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
    else:
        import pandas as pd

        reg = pd.read_csv(ds / "registry.csv", usecols=["name"])
        names = reg["name"].astype(str).tolist()
    log.info("待融合 %d 因子 × 4 horizon → %s", len(names), ds)
    for h in (1, 5, 10, 20):
        fuse_horizon_ic(ds, h, family=None, names=names, skip_existing=True)
    log.info("ic_h 融合完成")


def Config_root() -> str:
    from config import Config

    return str(Config.paths()["factor_library_root"])


if __name__ == "__main__":
    main()
