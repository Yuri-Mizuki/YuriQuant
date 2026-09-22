"""P5 补缺基本面因子族（B5）→ 并入 all_a_2018_2026 数据集。

来源（2026-09-22 数据源盘点，RESEARCH_TODO §一 P5）：无需任何新数据源，
四张已有财务表 + dividend 表即可自建：

  ① SUE 财报版    sue_q(单季扣非季节差/过去8期同差标准差,正)
  ② 资产扩张      asset_growth_yoy(总资产同比,方向由 IC 自判)
  ③ 投资强度      capex_intensity(-投资活动净现金流TTM/总资产,正=扩张投入)
  ④ 送转预期      spsr(每股资本公积,正) / bonus_freq_3y(近3年送转次数,正)

口径：与 B/B+/B++ 族一致——报告期维度先算（长表按 report_period 错位），
PIT 按 ann_date 展开；capex 无"购建固定资产"专项科目（cash_flow 117 列无此列），
以投资活动净现金流 TTM 代理。

用法:
    python -m scripts.builders.build_alla_p5_factors            # 单机
    python -m scripts.builders.build_alla_p5_factors --resume   # 断点续跑
    python -m scripts.builders.build_alla_p5_factors --only sue_q  # 调试
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

log = setup_logging("build_alla_p5")
from scripts.builders import common  # noqa: E402
from scripts.builders.common import HORIZONS, IC_CODE_STRIDE  # noqa: E402

DATASET = "all_a_2018_2026"

from scripts.builders.build_alla_fundamental_factors import (  # noqa: E402
    load_panels, add_ttm_yoy, add_single_quarter,
)
from scripts.builders.build_alla_constructed_factors import (  # noqa: E402
    year_offset, _pit, _safe,
)


def build_panels(close_adj, close_raw, inc, bal, cfo, div):
    codes = close_adj.columns
    cal_idx = close_adj.index
    panels: dict[str, pd.DataFrame] = {}

    # ---- ① SUE 财报版：单季扣非净利的季节差标准化 ----
    inc = add_single_quarter(inc, "NET_PRO_AFTER_DED_NR_GL", "NP_DED_SQ")
    inc = year_offset(inc, "NP_DED_SQ", "delta")          # 当季 - 去年同季
    inc = inc.sort_values(["code", "report_period"])
    inc["SUE_STD"] = (inc.groupby("code")["_off_NP_DED_SQ"]
                      .transform(lambda s: s.rolling(8, min_periods=4).std()))
    inc["SUE_Q"] = inc["_off_NP_DED_SQ"] / inc["SUE_STD"].replace(0.0, np.nan)
    panels["sue_q"] = _pit(inc, cal_idx, codes, "SUE_Q")   # 正：超预期

    # ---- ② 总资产同比 ----
    bal = year_offset(bal, "TOTAL_ASSETS", "ratio")
    panels["asset_growth_yoy"] = _pit(bal, cal_idx, codes, "_off_TOTAL_ASSETS")

    # ---- ③ 投资强度：投资活动净现金流 TTM / 总资产 ----
    cfo = add_ttm_yoy(cfo, "NET_CASH_FLOWS_INV_ACT", "INV_CF_TTM", None)
    inv_ttm = _pit(cfo, cal_idx, codes, "INV_CF_TTM")
    ta = _pit(bal, cal_idx, codes, "TOTAL_ASSETS")
    panels["capex_intensity"] = -_safe(inv_ttm, ta)        # 净流出为正=投入

    # ---- ④ 送转预期：每股资本公积 + 近3年送转次数 ----
    cap_resv = _pit(bal, cal_idx, codes, "CAP_RESV")
    tot_share = _pit(bal, cal_idx, codes, "TOT_SHARE")
    panels["spsr"] = _safe(cap_resv, tot_share)            # 正：高送转潜力
    if div is not None and not div.empty and "bonus_rate" in div.columns:
        dv = div[["code", "ann_date", "bonus_rate"]].copy()
        dv["ann_date"] = pd.to_datetime(dv["ann_date"], errors="coerce")
        dv["bonus_rate"] = pd.to_numeric(dv["bonus_rate"], errors="coerce")
        dv = dv.dropna().query("bonus_rate > 0")
        dv = dv.sort_values(["code", "ann_date"]).drop_duplicates(
            subset=["code", "ann_date"], keep="last")
        dv = dv.set_index("ann_date")
        dv["bonus_freq_3y"] = (dv.groupby("code")["bonus_rate"]
                               .rolling("1095D").count().to_numpy())
        dv = dv.reset_index().drop_duplicates(subset=["code", "ann_date"],
                                              keep="last")
        panels["bonus_freq_3y"] = _pit(dv, cal_idx, codes, "bonus_freq_3y")

    return {n: q.reindex(index=cal_idx, columns=codes) for n, q in panels.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, help="只构建指定因子")
    args = ap.parse_args()

    from config import Config
    from stats.ic import calc_ic_series

    lib_root = Path(str(Config.get()["factor_library"]["root"]))
    ds_dir = lib_root / DATASET
    panels_dir = ds_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    stats_path = ds_dir / "factor_stats_p5.jsonl"
    done = {json.loads(l)["name"] for l in
            stats_path.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if args.resume and stats_path.exists() else set()

    close_adj, close_raw, income, balance, cashflow, equity, dividend = load_panels()
    codes = close_adj.columns
    cal_idx = close_adj.index
    fwd = {h: close_adj.pct_change(h, fill_method=None).shift(-h) for h in HORIZONS}
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]

    names_all = ["sue_q", "asset_growth_yoy", "capex_intensity",
                 "spsr", "bonus_freq_3y"]
    if args.only:
        names_all = [n for n in names_all if n in set(args.only)]
    labels = {
        "sue_q": "单季扣非SUE(标准化季节差)",
        "asset_growth_yoy": "总资产同比",
        "capex_intensity": "投资强度(投资净流出/总资产)",
        "spsr": "每股资本公积(送转潜力)",
        "bonus_freq_3y": "近3年送转次数",
    }

    panels = build_panels(close_adj, close_raw, income, balance, cashflow, dividend)
    t0 = time.time()
    for i, name in enumerate([n for n in names_all if n in panels], 1):
        if name in done:
            continue
        p = panels[name].fillna(np.nan).astype(np.float32)
        p = p.replace([np.inf, -np.inf], np.nan).clip(-1e4, 1e4)
        p.to_parquet(panels_dir / f"{name}.parquet")
        cov = float(p.notna().mean().mean())
        ic = {h: calc_ic_series(p[ic_codes], fwd[h]).astype(np.float32)
              for h in HORIZONS}
        row = {"name": name, "set": "fundamental",
               "label": labels.get(name, name), "coverage": cov,
               **{f"ic_mean_h{h}": float(ic[h].mean()) for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("[%d/%d] %s cov=%.3f ic_h1=%+.4f ic_h20=%+.4f | %.0fs",
                 i, len(names_all), name, cov, row["ic_mean_h1"],
                 row["ic_mean_h20"], time.time() - t0)

    common.merge_outputs(ds_dir, 'p5', skip_existing=True,
                         empty_log='无 P5 补缺因子统计产出')
    log.info("P5 补缺基本面因子完成 %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
