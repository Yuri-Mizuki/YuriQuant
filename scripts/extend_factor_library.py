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

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from scripts._build_common import (
    add_build_args, make_data_context, record_experiment_safe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extend_factor_library")

# 重叠区归一化偏差容差（同数据同算子的浮点噪声水平）
_OVERLAP_TOL = 1e-6


def _warmup_begin(begin: int) -> int:
    """begin 前推约 2 个自然年（覆盖 alpha 公式最大 250 交易日回看）。"""
    y, md = divmod(begin, 10000)
    return max((y - 2) * 10000 + md, 20190102)


def verify_overlap(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    """新旧面板重叠区一致性校验。

    Returns:
        {n, max_abs, scale, norm}：n=双方均非 NaN 的对比点数；scale=旧值
        典型量级（重叠区 |值| 中位数）；norm=max_abs/scale（归一化偏差，
        < _OVERLAP_TOL 视为同一口径）。
    """
    idx = new.index.intersection(old.index)
    cols = new.columns.intersection(old.columns)
    if len(idx) == 0 or len(cols) == 0:
        return {"n": 0, "max_abs": float("nan"), "scale": float("nan"),
                "norm": float("nan")}
    a = new.loc[idx, cols]
    b = old.loc[idx, cols]
    mask = (a.notna() & b.notna()).to_numpy()
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "max_abs": float("nan"), "scale": float("nan"),
                "norm": float("nan")}
    diff = (a - b).abs().to_numpy()[mask]
    bvals = b.to_numpy()[mask]
    max_abs = float(diff.max())
    scale = float(np.median(np.abs(bvals)))
    norm = max_abs / scale if scale > 0 else max_abs
    return {"n": n, "max_abs": max_abs, "scale": scale, "norm": norm}


def extend_gp_factors(lib, gp_rows, panel, returns, begin, end,
                      verify_only: bool = False, force: bool = False):
    """按原始口径重算 GP 公式因子并覆盖入库。

    Returns:
        (ok, skipped, failed)：成功（或校验通过）名单 / 口径不一致跳过名单 /
        公式重算失败名单。
    """
    from factor.formula import formula_builder

    feats = list(panel.keys())
    b0, e0 = pd.Timestamp(str(begin)), pd.Timestamp(str(end))
    ok: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for _, row in gp_rows.iterrows():
        name = str(row["name"])
        try:
            fp = formula_builder(str(row.get("formula") or name), features=feats)(panel)
        except Exception as e:  # noqa: BLE001
            log.warning("公式重算失败 %s: %s", name, str(e)[:120])
            failed.append(name)
            continue
        fp = fp.loc[b0:e0]
        old = lib.get_panel(name)
        if old is not None and not old.empty:
            v = verify_overlap(fp, old)
            if v["n"] == 0 or (v["norm"] >= _OVERLAP_TOL and not force):
                log.warning("历史口径不一致，跳过 %s（重叠 %d 点，归一化偏差 %.2e）",
                            name, v["n"], v["norm"])
                skipped.append(name)
                continue
            log.info("校验通过 %s（%d 点，归一化偏差 %.2e）", name, v["n"], v["norm"])
        if verify_only:
            ok.append(name)
            continue
        lib.register(
            name, fp, returns,
            kind=str(row.get("kind") or "raw"),
            formula=str(row.get("formula") or name),
            source=str(row.get("source") or ""),
            family=str(row.get("family") or ""),
            frequency=str(row.get("frequency") or ""),
            maturity=str(row.get("maturity") or "experimental"),
            note=str(row.get("note") or ""),
        )
        ok.append(name)
    return ok, skipped, failed


def extend_alpha_factors(lib, cache, uni, index_code: str, sets: list[str],
                         begin: int, end: int, warmup: int,
                         verify_only: bool = False) -> int:
    """重算 alpha101/alpha191/alpha158/alpha360 全集并覆盖入库（与 build_alpha_factors 同路径）。"""
    from factor.alpha101 import compute_alpha101
    from factor.alpha158 import compute_alpha158, compute_alpha360
    from factor.alpha191 import compute_alpha191
    from factor.alpha_base import AlphaData
    from scripts._build_common import register_panels
    from scripts.build_alpha_factors import SET_LABELS, load_panels

    panels_px, industry, close_adj = load_panels(cache, uni, index_code, warmup, end)
    d = AlphaData(panels_px, industry=industry)
    b0, e0 = pd.Timestamp(str(begin)), pd.Timestamp(str(end))
    returns = close_adj.loc[:e0].pct_change().shift(-1).loc[b0:e0]

    _compute_fn = {
        "alpha101": compute_alpha101,
        "alpha191": compute_alpha191,
        "alpha158": compute_alpha158,
        "alpha360": compute_alpha360,
    }

    n_total = 0
    for s in sets:
        fn = _compute_fn[s]
        label = SET_LABELS[s]
        log.info("重算 %s（%s）...", s, label)
        # 与 build_alpha_factors 同口径：裁剪后剔除全 NaN 行列（不在册期股票的
        # 死列会虚增覆盖率分母，触发假 coverage_drop 告警）
        computed = {
            k: p.loc[b0:e0].dropna(axis=1, how="all").dropna(axis=0, how="all")
            for k, p in fn(d).items()
        }
        if verify_only:
            log.info("%s 校验模式：可重算 %d 个（不入库）", s, len(computed))
            n_total += len(computed)
            continue
        defs = {k: f"{label} #{k.split('_')[-1]}" for k in computed}
        rows = register_panels(
            lib, computed, defs, returns,
            source=f"{s}:build_alpha_factors:{begin}-{end}")
        n_total += len(rows)
        log.info("%s 完成: %d 个已覆盖入库", s, len(rows))
    return n_total


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
    warmup = _warmup_begin(begin)

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
