"""公开经典因子集（Alpha101 / GTJA Alpha191）→ 构建入库
====================================================

WorldQuant 101 Formulaic Alphas（Kakushadze 2016）与国泰君安 191 短周期
价量因子，作为公开基准信号源补入因子库，与库内 GP/挖掘因子**同库统一
监控**（12 个月滚动 IC、保留率、拥挤度，按 source 分组对比）。

口径说明
--------
- 价格一律**后复权**（raw × backward_factor），vwap = amount/volume × bf；
  alpha101 的 IndNeutralize 用申万一级行业 PIT 面板（无行业数据时恒等）。
- **warmup**：公式最大回看 250 日（如 alpha191_101 的 corr(...,250)），
  默认从 begin 前约 1.5 年起拉数据，算完面板后裁剪回 [begin, end] 再
  入库，保证入库首日因子即充分预热（非 warmup 前几行的 NaN）。
- 收益面板与库内口径一致：后复权 close 的 pct_change().shift(-1)。
- 与 alpha101 完全重复的 3 个 alpha191 因子（032/040/139）在公式层已去重。
- 面板 zscore 截面标准化后入库（同 build_technical_factors）。

用法
----
    python -m scripts.build_alpha_factors --offline                 # 默认 hs300_2022_2025
    python -m scripts.build_alpha_factors --offline --no-save      # 只算不入库
    python -m scripts.build_alpha_factors --offline --sets alpha101
    python -m scripts.build_alpha_factors --mock                   # mock 验证
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from data.cache_helpers import load_backward_factor, load_daily
from factor.alpha101 import compute_alpha101
from factor.alpha158 import compute_alpha158, compute_alpha360
from factor.alpha191 import compute_alpha191
from factor.alpha_base import AlphaData
from research.factor_library import FactorLibrary
from scripts._build_common import (
    add_build_args, make_data_context, print_no_save, record_experiment_safe,
    register_panels,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_alpha_factors")

# 三态默认：监控主力库 hs300_2022_2025（2022-2025 全区间）；
# warmup 起点取 begin 前约 1.5 年（> 最大回看 250 交易日）
DEFAULTS = {
    "mock": {"begin": 20230103, "end": 20241231, "dataset": "mock",
             "warmup": 20220101},
    "offline": {"begin": 20220101, "end": 20251231, "dataset": "hs300_2022_2025",
                "warmup": 20200701},
    "real": {"begin": 20220101, "end": 20251231, "dataset": "hs300_2022_2025",
             "warmup": 20200701},
}

SET_LABELS = {
    "alpha101": "WorldQuant Alpha101 (Kakushadze 2016)",
    "alpha191": "GTJA Alpha191 短周期价量因子 (2017)",
    "alpha158": "Qlib Alpha158 量价特征集 (Microsoft, 2020)",
    "alpha360": "Qlib Alpha360 原始OHLCV序列 (Microsoft, 2020)",
}


def load_panels(cache, uni, index_code: str, warmup: int, end: int):
    """拉 warmup..end 的 PIT 日线并组装 AlphaData 输入面板（后复权）。"""
    codes, cal, daily = load_daily(cache, uni, index_code, warmup, end)
    bf = load_backward_factor(cache, codes)
    log.info("日线 %d 行 / 复权因子 %d 列", len(daily),
             bf.shape[1] if not bf.empty else 0)

    d = daily.reset_index()
    d["date"] = d["date"].dt.normalize()

    def _panel(col: str) -> pd.DataFrame:
        return d.pivot(index="date", columns="code", values=col).sort_index()

    o, h, l, c = _panel("open"), _panel("high"), _panel("low"), _panel("close")
    v, amt = _panel("volume"), _panel("amount")

    if not bf.empty:
        f = bf.reindex(index=c.index, columns=c.columns).ffill()
        for pnl in (o, h, l, c):
            pnl[:] = pnl.values * f.values
        # vwap = 均价 × 后复权因子（与价格同口径）
        vwap = (amt / v).replace([np.inf, -np.inf], np.nan) * f
    else:
        log.warning("无复权因子，使用原始价（除权日价量关系会有跳变）")
        vwap = (amt / v).replace([np.inf, -np.inf], np.nan)

    # 申万一级行业 PIT 面板（IndNeutralize 用；失败则恒等中性化）
    industry = None
    try:
        from data.industry import IndustryClassification
        industry = IndustryClassification(cache, level=1).get_industry_panel(
            list(c.columns), c.index)
        if industry.isna().all().all():
            industry = None
    except Exception as e:  # noqa: BLE001
        log.warning("行业面板不可用（%s），IndNeutralize 退化为恒等", str(e)[:80])

    panels = {"open": o, "high": h, "low": l, "close": c,
              "volume": v, "amount": amt, "vwap": vwap}
    return panels, industry, c


def main():
    parser = argparse.ArgumentParser(description="Alpha101/Alpha191 公开因子集构建入库")
    add_build_args(parser)
    parser.add_argument("--sets", default="alpha101,alpha191",
                        help="逗号分隔：alpha101,alpha191,alpha158,alpha360")
    parser.add_argument("--warmup", type=int, default=None,
                        help="预热起始日（默认按模式取 begin 前 ~1.5 年）")
    args = parser.parse_args()

    # 先解析模式（make_data_context 依赖 --mock/--offline 标志），再回填默认区间
    mode = "mock" if args.mock else ("offline" if args.offline else "real")
    cfg_d = DEFAULTS[mode]
    args.begin = args.begin or cfg_d["begin"]
    args.end = args.end or cfg_d["end"]
    args.dataset = args.dataset or cfg_d["dataset"]
    warmup = args.warmup or cfg_d["warmup"]

    cache, uni, begin, end, dataset = make_data_context(args)
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    for s in sets:
        if s not in SET_LABELS:
            parser.error(f"未知因子集: {s}（可选 {list(SET_LABELS)}）")

    _compute_fn = {
        "alpha101": compute_alpha101,
        "alpha191": compute_alpha191,
        "alpha158": compute_alpha158,
        "alpha360": compute_alpha360,
    }

    log.info("模式=%s 区间=%d-%d warmup起=%d 数据集=%s 因子集=%s",
             mode, begin, end, warmup, dataset, sets)

    panels_px, industry, close_adj = load_panels(cache, uni, args.index, warmup, end)
    d = AlphaData(panels_px, industry=industry)

    # warmup 区间上算因子，裁剪回 [begin, end] 入库（首日即充分预热）
    b0, e0 = pd.Timestamp(str(begin)), pd.Timestamp(str(end))
    all_panels: dict[str, pd.DataFrame] = {}
    all_defs: dict[str, str] = {}
    for s in sets:
        fn = _compute_fn[s]
        label = SET_LABELS[s]
        log.info("计算 %s（%s）...", s, label)
        for name, p in fn(d).items():
            p = p.loc[b0:e0]
            p = p.dropna(axis=1, how="all").dropna(axis=0, how="all")
            all_panels[name] = p
            all_defs[name] = f"{label} #{name.split('_')[-1]}"
        log.info("  %s 得到 %d 个因子面板", s,
                 sum(1 for n in all_defs if n.startswith(f"{s}_")))

    if args.no_save:
        print_no_save(all_defs, all_panels)
        return

    # 收益面板：与库内口径一致（后复权 close，pct_change().shift(-1)）
    returns_panel = close_adj.loc[:e0].pct_change().shift(-1).loc[b0:e0]

    lib = FactorLibrary(dataset=dataset)
    log.info("入库到数据集: %s", dataset)
    for s in sets:
        names = [n for n in all_defs if n.startswith(f"{s}_")]
        source = f"{s}:build_alpha_factors:{begin}-{end}"
        log.info("注册 %s（%d 个，source=%s）", s, len(names), source)

        def _on_fail(name, e):
            log.warning("注册失败 %s: %s（继续其余因子）", name, str(e)[:120])

        rows = register_panels(
            lib, all_panels, all_defs, returns_panel,
            source=source, names=names, on_fail=_on_fail,
        )
        ok = sum(1 for r in rows if abs(r["ic_mean"]) >= 0.01)
        log.info("%s 完成: 成功 %d / %d，其中 |IC|≥0.01 的 %d 个",
                 s, len(rows), len(names), ok)

    record_experiment_safe(
        kind="alpha_factors",
        command=" ".join(sys.argv),
        params={"index": args.index, "begin": begin, "end": end,
                "dataset": dataset, "sets": sets, "warmup": warmup},
        fingerprint=cache.get_fingerprint(),
        result_path=str(lib.root),
        metrics={"n_factors": len(all_defs)},
        note="公开因子集（Alpha101/GTJA191）入库，统一监控按 source 分组对比",
    )
    log.info("完成。数据集 %s 现有 %d 个因子", dataset, len(lib.list_all()))


if __name__ == "__main__":
    main()
