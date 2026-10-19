"""E5 变体：基本面族「只对行业中性化」的面板变体（GOOD_MACHINE_TASKS 批次 6）。

对照 build_alla_factor_neutralized（标准三步：去极值 → 行业+市值中性化 → zscore），
本变体对**基本面/股东族**（``scripts.common.alla_common.FUNDAMENTAL_FAMILY_SETS``，
即检验"ortho 二次压制"诊断的目标域）跳过市值中性化、只剥行业——保住 bp/ep/div_yield
的价值/红利风格暴露，其余因子维持标准行业+市值中性化，输出到
``ds_root/panels_neu_fundind/``。

共享实现见 ``scripts.builders.common.run_neutralization``（与主变体共用，
2026-10-19 去重；差异仅在 industry_only 集合与输出目录）。

配套跑法（rolling_grid_alla --preproc ortho_fundind）::

    python scripts/pipelines/rolling_grid_alla.py --preproc ortho_fundind \
        --out-tag ortho920fundind --exclude-features limit_pos \
        --tradable-labels --execution open

用法:
    python -m scripts.builders.build_alla_factor_neutralized_fundind [--workers 6]
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
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--force", action="store_true",
                    help="忽略已存在产物、全部重算（上游 panels/ 重建后必须加）")
    args = ap.parse_args()

    from config import Config
    from scripts.builders import common
    from scripts.common.alla_common import FUNDAMENTAL_FAMILY_SETS

    ds = Path(str(Config.get()["factor_library"]["root"])) / DATASET
    common.run_neutralization(
        ds, "panels_neu_fundind", workers=args.workers, only=args.only,
        force=args.force, set_name="fundind", label="E5 变体",
        stats_file="factor_stats_neutralized_fundind.jsonl",
        industry_only=frozenset(FUNDAMENTAL_FAMILY_SETS))


if __name__ == "__main__":
    main()
