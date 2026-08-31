"""因子库数据延长 —— 面板刷新到数据源最新交易日
================================================

背景：``hs300_2022_2025`` 原建到 2025-12-31，而监控按当前日期运行，
全体因子触发 stale_data 告警。本脚本把库内**可重算**的因子在同一口径下
重算并覆盖入库，延长到最新交易日：

- ``alpha101`` / ``alpha191``：warmup 起算（覆盖最大 250 日回看），裁剪回
  [begin, end] 后 zscore 入库（与 ``build_alpha_factors`` 完全同一代码路径）；
- ``gp`` 公式因子：按 ``mine_factors`` 原始口径（``build_panel`` 原始特征
  面板，无预处理——存量面板即原始量纲）重算公式。**注册前先与存量面板做
  重叠区校验**（归一化最大偏差 < 1e-6 才覆盖），历史无法复现的因子跳过
  不动，防止口径漂移悄悄改写历史 IC；
- ``model:*`` 预测因子：模型流水线产物，本脚本不处理（需重跑
  ``walk_forward_model`` 生成新预测段）。

用法：
    python -m scripts.extend_factor_library --offline                  # 刷新到最新
    python -m scripts.extend_factor_library --offline --sets gp        # 只刷 GP
    python -m scripts.extend_factor_library --offline --verify-only    # 只校验不入库
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.factor_extension import (  # noqa: E402
    extend_alpha_factors, extend_gp_factors, warmup_begin,
)
from scripts.cli_common import (  # noqa: E402
    add_build_args, make_data_context, record_experiment_safe,
    setup_logging,
)

log = setup_logging("extend_factor_library")

def main():
    parser = argparse.ArgumentParser(description="因子库面板延长到最新交易日")
    add_build_args(parser)
    parser.add_argument("--sets", default="alpha101,alpha191,gp",
                        help="逗号分隔：alpha101,alpha191,alpha158,alpha360,gp")
    parser.add_argument("--verify-only", action="store_true",
                        help="只校验历史口径一致性，不入库")
    parser.add_argument("--force", action="store_true",
                        help="GP 历史校验不一致时仍强制覆盖（慎用）")
    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = "mock" if args.mock else "hs300_2022_2025"
    if args.begin is None:
        # 必须从原始建库起点重算：make_data_context 的 offline 默认区间是
        # 2025 起（DEFAULT_RANGES），直接沿用会把面板历史截断到 2025 之后
        args.begin = 20230103 if args.mock else 20220101
    cache, uni, begin, end, dataset = make_data_context(args)
    verify_only = args.verify_only or args.no_save

    # 未显式给 end 时取本地缓存最新交易日（make_data_context 的默认 end 是建库区间
    # 终点；离线桩对超范围日历查询会回源报错，故直接读 daily.parquet）
    if args.end is None:
        from pathlib import Path

        d = pd.read_parquet(Path(cache.root) / "daily_hs300.parquet")
        if len(d):
            end = int(d.index.get_level_values(0).max().strftime("%Y%m%d"))
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    warmup = warmup_begin(begin)

    log.info("延长数据集 %s: %d → %d（warmup 起 %d）| 因子集: %s | 模式: %s%s",
             dataset, begin, end, warmup, sets,
             "mock" if args.mock else ("offline" if args.offline else "real"),
             "（仅校验）" if verify_only else "")

    from research.factor_library import FactorLibrary

    lib = FactorLibrary(dataset=dataset)
    reg = lib.list_all()
    if reg.empty:
        raise RuntimeError(f"因子库为空: dataset={dataset}")
    src = reg["source"].fillna("").astype(str).str.split(":").str[0]

    n_alpha = 0
    alpha_sets = [s for s in sets
                  if s in ("alpha101", "alpha191", "alpha158", "alpha360")]
    if alpha_sets:
        n_alpha = extend_alpha_factors(
            lib, cache, uni, args.index, alpha_sets, begin, end, warmup,
            verify_only=verify_only)

    gp_rows = reg[src == "gp"] if "gp" in sets else reg.iloc[0:0]
    ok = skipped = failed = []
    if "gp" in sets and not gp_rows.empty:
        from data.cache_helpers import build_panel
        from config import Config

        cfg = Config.get()
        if args.mock:
            gp_panel, gp_returns = build_panel(cfg, begin, end, cache=cache)
        else:
            gp_panel, gp_returns = build_panel(cfg, begin, end, offline=True)
        ok, skipped, failed = extend_gp_factors(
            lib, gp_rows, gp_panel, gp_returns, begin, end,
            verify_only=verify_only, force=args.force)
        log.info("GP 完成: 校验/入库 %d，口径不一致跳过 %d，重算失败 %d",
                 len(ok), len(skipped), len(failed))
    elif "gp" in sets:
        log.info("库内无 gp 因子，跳过")

    n_model = int((src == "model").sum())
    if n_model:
        log.info("model:* 预测因子 %d 个不在延长范围（需重跑 walk_forward_model）", n_model)

    record_experiment_safe(
        kind="extend_factor_library",
        command=" ".join(sys.argv),
        params={"dataset": dataset, "begin": begin, "end": end,
                "sets": sets, "index": args.index},
        fingerprint=cache.get_fingerprint(),
        result_path=str(lib.root),
        metrics={"n_alpha": n_alpha, "n_gp_ok": len(ok),
                 "n_gp_skipped": len(skipped), "n_gp_failed": len(failed)},
        note="因子库面板延长到最新交易日（alpha 重算覆盖 + GP 公式重算带历史校验）",
    )
    log.info("完成。数据集 %s 现有 %d 个因子", dataset, len(lib.list_all()))

if __name__ == "__main__":
    main()