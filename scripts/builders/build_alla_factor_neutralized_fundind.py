"""E5 变体：基本面族「只对行业中性化」的面板变体（TODO_ARCHIVE 册Ⅲ 批次 6）。

对照 build_alla_factor_neutralized（标准三步：去极值 → 行业+市值中性化 → zscore），
本变体对**基本面/股东族**（``scripts.common.alla_common.FUNDAMENTAL_FAMILY_SETS``，
即检验"ortho 二次压制"诊断的目标域）跳过市值中性化、只剥行业——保住 bp/ep/div_yield
的价值/红利风格暴露，其余因子维持标准行业+市值中性化，输出到
``ds_root/panels_neu_fundind/``。

**2026-11-18 起该目录是「覆盖层」而不是「全量副本」**：只存与 ``panels_neu``
内容不同的因子（基本面/股东族 + 少数事件族），其余由
``alla_common.fundind_dirs()`` 回落基准层。全量重建会把 863 个逐字节相同的
因子重新写一遍（48.55 GB 纯重复），故默认范围已收窄为
「基本面族 ∪ 现有覆盖层」；要恢复旧的全量行为须显式 ``--only`` 传全名单。

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
    from scripts.common.alla_common import (
        FUNDAMENTAL_FAMILY_SETS,
        FUNDIND_BASE_SUBDIR,
        FUNDIND_SUBDIR,
    )

    ds = Path(str(Config.get()["factor_library"]["root"])) / DATASET
    # 2026-11-18 起覆盖层**只存与基准层内容不同的因子**：默认范围 = 基本面/股东族
    # ∪ 现有覆盖层。其余因子在两层逐字节相同，写进去就是 48.55 GB 纯重复
    # （全量 md5 实测 863/950 个文件相同）。
    if args.only:
        scope = args.only
    else:
        available = {p.stem for p in (ds / FUNDIND_BASE_SUBDIR).glob("*.parquet")}
        current = {p.stem for p in (ds / FUNDIND_SUBDIR).glob("*.parquet")}
        scope = sorted((FUNDAMENTAL_FAMILY_SETS & available) | current)
        print(f"覆盖层范围：基本面/股东族 {len(FUNDAMENTAL_FAMILY_SETS & available)} "
              f"+ 现有覆盖 {len(current)} → 去重 {len(scope)}", flush=True)
    common.run_neutralization(
        ds, FUNDIND_SUBDIR, workers=args.workers, only=scope,
        force=args.force, set_name="fundind", label="E5 变体",
        stats_file="factor_stats_neutralized_fundind.jsonl",
        industry_only=frozenset(FUNDAMENTAL_FAMILY_SETS))


if __name__ == "__main__":
    main()
