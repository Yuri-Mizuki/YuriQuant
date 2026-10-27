"""A1 长 horizon 特征清单选择器：h60 IC 质量窗 + DPP 去冗余（2026-10-19）。

## 定位

TODO_ARCHIVE.md 册Ⅰ §六 A1 下一步。首个 h60 模型（`h60_probe_y2023`）用的是 **h20 清单**
（`y2023__h20.json`），IC 0.141 只是「h20 特征在 h60 标签上的下界」。本脚本按
h60 标签的 IC 重选 50 特征，产出 `y{year}__h{h}.json`，供 `retrain_freq_ic
--sel-dir` 消费，回答「h60 最优特征集长什么样、比 h20 清单强多少」。

## 口径（对齐 rolling_grid_alla::select_features_for_year，差异显式列出）

同：
- 质量窗 500 交易日（QUALITY_WINDOW）；候选 top150（DEDUP_TOP_CANDIDATES）；
  registry coverage ≥ 0.5（MIN_COVERAGE）；DPP 去冗余取 50（MAX_FEATURES），
  核参数同 `dpp_select` 缺省；
- 相关矩阵采样：质量窗 stride=4 交易日、列隔 2 取半（对齐 RRE_STRIDE 网格）。

异（简化，meta 会写明）：
- 无基本面/另类**保留席位**（长 horizon 下基本面是否自然上榜，正是 A1 要
  观察的对象，不预设）；
- 无 RRE 秩稳定性筛（首版探针不做）；
- **fwd 防前视更严**：质量窗最后一日满足 t+h 交易日 ≤ cut（ic 的标签完整
  落在选择窗内；rolling_grid 的 ic_cache 允许 fwd 穿透 cut 到 OOS 头部）。
- IC 走可交易掩码 + 截面 spearman（与 horizon_probe / residual_ic_probe 同源）。

## 用法

    python -m scripts.evaluation.select_horizon_features --horizon 60 --year 2023
    python -m scripts.evaluation.select_horizon_features --horizon 60 --year 2023 \
        --quality-window 500 --compare-h20 reports/alla_rolling_ortho/selection/y2023__h20.json
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

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("select_horizon_features")

QUALITY_WINDOW = 500
MAX_FEATURES = 50
DEDUP_TOP_CANDIDATES = 150
MIN_COVERAGE = 0.5
SAMPLE_STRIDE = 4
CODE_STRIDE = 2


def quality_window(all_days: pd.DatetimeIndex, year: int, h: int,
                   qw: int) -> pd.DatetimeIndex:
    """质量窗：截止 ``year`` 年末、末日的 h 日前瞻完整落在年内（防前视）。

    2026-10-19 口径注记：rolling_grid 的 ic_cache 允许 fwd 穿透 cut 到 OOS
    头部；这里更严——窗口最后一日 t 满足 t+h 个交易日 ≤ year 年末，使 IC
    标签不接触 OOS。
    """
    year_end = pd.Timestamp(f"{year}-12-31")
    le = all_days[all_days <= year_end]
    if len(le) <= h:
        raise ValueError(f"{year} 年末前不足 {h} 个交易日")
    cut_eff = le[-1 - h]
    return le[le <= cut_eff][-qw:]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--year", type=int, default=2023,
                    help="清单归属年（质量窗截止 = 该年年末，防前视）")
    ap.add_argument("--quality-window", type=int, default=QUALITY_WINDOW)
    ap.add_argument("--panels-dir", default="panels_neu_fundind",
                    help="特征源（与 retrain_freq_ic 训练特征一致）")
    ap.add_argument("--out", default="reports/frontier_audit_2026-10-19/selection_h60")
    ap.add_argument("--compare-json", default=None,
                    help="对照清单（如 h20 的 y2023__h20.json），报重叠度")
    ap.add_argument("--resume", action="store_true",
                    help="命中已落盘的 ic_cache CSV 时跳过 IC 扫描")
    args = ap.parse_args(argv)

    from scripts.common.alla_common import ds_root, load_base
    from stats.ic import calc_ic_series

    ds = ds_root()
    pdir = ds / args.panels_dir
    # registry 存在同 name 多行（neu/raw 双登记等，1112 行 / 950 唯一名）：
    # set_index 后重复键的 [] 取值返回 Series，布尔链会炸，先去重。
    reg = pd.read_csv(ds / "registry.csv").drop_duplicates("name", keep="last")
    cov_ok = reg.set_index("name")["coverage"]

    base = load_base()
    close, mask_all = base["close"], base["mask"].astype(bool)
    all_days = close.index
    h = args.horizon

    # fwd 防前视：质量窗最后一天的 h 日前瞻完整落在 year 年末之前
    q_days = quality_window(all_days, args.year, h, args.quality_window)
    log.info("h=%d | 质量窗 %s ~ %s（%d 日；fwd 截止 %s）",
             h, q_days[0].date(), q_days[-1].date(), len(q_days),
             f"{args.year}-12-31")

    fwd = (close.pct_change(h, fill_method=None).shift(-h)
           .reindex(index=q_days)
           .replace([np.inf, -np.inf], np.nan).astype(np.float32))
    mask = mask_all.reindex(index=q_days).fillna(False)

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    ic_cache_fp = dest / f"ic_cache_y{args.year}__h{h}.csv"

    files = sorted(pdir.glob("*.parquet"))
    registered = set(reg["name"].astype(str))
    files = [p for p in files if p.stem in registered]
    log.info("候选面板 %d 个（%s）", len(files), pdir)

    qual_cached: dict[str, float] = {}
    if args.resume and ic_cache_fp.exists():
        _c = pd.read_csv(ic_cache_fp)
        qual_cached = dict(zip(_c["name"], _c["ic_mean"]))
        log.info("resume：IC 缓存命中 %d 个", len(qual_cached))

    ic_cache: dict[str, pd.Series] = {}
    skipped = 0
    t0 = time.time()
    todo = [p for p in files if p.stem not in qual_cached]
    for i, fp in enumerate(todo, 1):
        name = fp.stem
        raw = pd.read_parquet(fp)
        if not isinstance(raw.index, pd.DatetimeIndex):
            try:
                raw.index = pd.to_datetime(raw.index)
            except Exception:
                skipped += 1
                continue
        if raw.columns.has_duplicates:
            raw = raw.loc[:, ~raw.columns.duplicated()]
        x = (raw.reindex(index=q_days)
             .replace([np.inf, -np.inf], np.nan)
             .select_dtypes(include=[np.number]).astype(np.float32))
        if x.shape[1] == 0 or float(x.notna().to_numpy(dtype=float).mean()) < 0.05:
            skipped += 1
            continue
        ic_cache[name] = calc_ic_series(x, fwd, tradable=mask)
        if i % 100 == 0:
            log.info("  %d/%d 个面板已评 IC（%.0fs）", i, len(todo),
                     time.time() - t0)
        del raw, x

    qual = {n: float(s.mean()) for n, s in ic_cache.items()}
    qual.update(qual_cached)
    quality = (pd.Series(qual).dropna().abs().sort_values(ascending=False))
    log.info("IC 评估完成：%d 个（跳过 %d，缓存 %d）| 质量窗 |IC| top10：\n%s",
             len(quality), skipped, len(qual_cached),
             quality.head(10).to_string())
    pd.DataFrame({"name": quality.index,
                  "ic_mean": [qual[n] for n in quality.index]}
                 ).to_csv(ic_cache_fp, index=False, encoding="utf-8-sig")

    cands = [n for n in quality.index
             if n in cov_ok.index and cov_ok[n] >= MIN_COVERAGE]
    cands = cands[:DEDUP_TOP_CANDIDATES]
    log.info("进入 DPP 的候选 %d 个（cov≥%.1f + top%d）",
             len(cands), MIN_COVERAGE, DEDUP_TOP_CANDIDATES)

    from research.dpp_selection import corr_matrix, dpp_select

    sample_days = q_days[::SAMPLE_STRIDE]
    sample_codes = None
    panels: dict[str, pd.DataFrame] = {}
    for n in cands:
        x = pd.read_parquet(pdir / f"{n}.parquet")
        if not isinstance(x.index, pd.DatetimeIndex):
            x.index = pd.to_datetime(x.index)
        if sample_codes is None:
            sample_codes = x.columns[::CODE_STRIDE]
        panels[n] = (x.reindex(index=sample_days, columns=sample_codes)
                     .astype(np.float32))
        del x
    corr = corr_matrix(panels, method="cross")
    res = dpp_select(corr, k=min(MAX_FEATURES, len(cands)),
                     quality=quality.reindex(corr.index))
    selected: list[str] = res["selected"]
    log.info("DPP 选中 %d 个 | 池内均值|corr| %.3f → 选中 %.3f",
             len(selected), res["mean_abs_corr_pool"], res["mean_abs_corr_selected"])

    (dest / f"y{args.year}__h{h}.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    qdf = quality.rename("abs_ic_mean").to_frame()
    qdf["ic_mean"] = qdf.index.map(lambda n: qual.get(n, np.nan))
    qdf["rank"] = range(1, len(qdf) + 1)
    qdf["in_top_cands"] = qdf.index.isin(cands)
    qdf["selected"] = qdf.index.isin(selected)
    qdf["coverage"] = qdf.index.map(lambda n: float(cov_ok.get(n, np.nan)))
    set_by_name = dict(zip(reg["name"].astype(str), reg["set"].astype(str)))
    qdf["set"] = qdf.index.map(lambda n: set_by_name.get(n, "?"))
    qdf.to_csv(dest / f"quality_y{args.year}__h{h}.csv",
               encoding="utf-8-sig")

    fam = qdf[qdf.selected]["set"].value_counts().to_dict()
    meta = {"horizon": h, "year": args.year, "panels_dir": args.panels_dir,
            "quality_window": args.quality_window,
            "quality_days": [str(q_days[0].date()), str(q_days[-1].date())],
            "n_panels": len(files), "n_quality": int(len(quality)),
            "n_candidates": len(cands), "selected": selected,
            "selected_family_counts": fam,
            "diff_vs_rolling_grid": ("无保留席位/无 RRE/fwd 不穿透选择窗"
                                     "（更严防前视）；其余常量对齐"),
            "elapsed_s": round(time.time() - t0, 1)}
    if args.compare_json:
        other = json.loads(Path(args.compare_json).read_text(encoding="utf-8"))
        ov = sorted(set(selected) & set(other))
        meta["compare_with"] = args.compare_json
        meta["compare_overlap"] = len(ov)
        meta["compare_overlap_names"] = ov
    (dest / f"meta_y{args.year}__h{h}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("完成 %.0fs -> %s（选中族分布 %s）", time.time() - t0,
             dest / f"y{args.year}__h{h}.json", fam)


if __name__ == "__main__":
    main()
