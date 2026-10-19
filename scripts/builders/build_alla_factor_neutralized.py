"""预计算全注册因子的因子层中性化面板（#3 优化）。

对每一注册因子面板做业界标准三步预处理：
    去极值(MAD) → 行业哑变量 + 市值(logERC) 中性化 → 截面 z-score
（factor.preprocessing::preprocess_factor，逐日截面、无未来函数）。

产出写入 ds_root/panels_neu/{name}.parquet，供滚动实验以
`--preproc ortho` 读取做对照 —— 即把风格中性化从"信号层/组合层"
下放到"因子层/模型输入层"，让线性模型(ridge)也能利用基本面慢变量。

中性化所用市值/行业面板来自 reports/alla_rolling/_base（与 close_adj
同 date×code 对齐）。multiprocessing 加速（每因子独立，写不同 parquet）。
共享实现见 ``scripts.builders.common.run_neutralization``（与 fundind 变体
共用，2026-10-19 去重）。

用法:
    python -m scripts.builders.build_alla_factor_neutralized [--workers 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = "all_a_2018_2026"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="+", default=None,
                    help="只处理指定因子名（调试用）")
    ap.add_argument("--force", action="store_true",
                    help="忽略已存在产物、全部重算。**上游 panels/ 重建后必须加**——"
                         "跳过判据只看文件是否存在，不看数据是否已过期（2026-10-16 踩到）")
    args = ap.parse_args()

    from config import Config
    from scripts.builders import common

    ds = Path(str(Config.get()["factor_library"]["root"])) / DATASET
    common.run_neutralization(
        ds, "panels_neu", workers=args.workers, only=args.only,
        force=args.force, set_name="neutralized", label="中性化",
        stats_file="factor_stats_neutralized.jsonl")


if __name__ == "__main__":
    main()
