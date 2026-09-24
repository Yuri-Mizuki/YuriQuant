"""P5 补缺基本面因子族（B5）→ 并入 all_a_2018_2026 数据集。

来源（2026-09-22 数据源盘点，RESEARCH_TODO §一 P5）：无需任何新数据源，
四张已有财务表 + dividend 表即可自建：

  ① SUE 财报版    sue_q(单季扣非季节差/过去8期同差标准差,正)
  ② 资产扩张      asset_growth_yoy(总资产同比,方向由 IC 自判)
  ③ 投资强度      capex_intensity(-投资活动净现金流TTM/总资产,正=扩张投入)
  ④ 送转预期      spsr(每股资本公积,正) / bonus_freq_3y(近3年送转次数,正)
  ⑤ 营运效率      ccc(现金转换周期,负向资金占用)
  ⑥ 治理信号      report_delay(披露时滞相对偏移,晚披露=坏消息)
  ⑦ 综合质量      piotroski_f(9 项打分 0-9)
  ⑧ 渠道压力      inv_rev_gap(存货同比-营收同比,正=压货负向)

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
import gc
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
    year_offset, pit, safe,
)


def _pit32(df, cal_idx, codes, field):
    """PIT 展开 + float32（本 builder 面板多，峰值内存敏感）。"""
    return pit(df, cal_idx, codes, field).astype(np.float32)


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
    panels["sue_q"] = pit32(inc, cal_idx, codes, "SUE_Q")   # 正：超预期

    # ---- ② 总资产同比 ----
    bal = year_offset(bal, "TOTAL_ASSETS", "ratio")
    panels["asset_growth_yoy"] = pit32(bal, cal_idx, codes, "_off_TOTAL_ASSETS")

    # ---- ③ 投资强度：投资活动净现金流 TTM / 总资产 ----
    cfo = add_ttm_yoy(cfo, "NET_CASH_FLOWS_INV_ACT", "INV_CF_TTM", None)
    inv_ttm = pit32(cfo, cal_idx, codes, "INV_CF_TTM")
    ta = pit32(bal, cal_idx, codes, "TOTAL_ASSETS")
    panels["capex_intensity"] = -safe(inv_ttm, ta)        # 净流出为正=投入

    # ---- ④ 送转预期：每股资本公积 + 近3年送转次数 ----
    cap_resv = pit32(bal, cal_idx, codes, "CAP_RESV")
    tot_share = pit32(bal, cal_idx, codes, "TOT_SHARE")
    panels["spsr"] = safe(cap_resv, tot_share)            # 正：高送转潜力
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
        panels["bonus_freq_3y"] = pit32(dv, cal_idx, codes, "bonus_freq_3y")

    # ---- ⑤ 现金转换周期 CCC = DSO + DIO - DPO（负向：占用越久越差） ----
    inc = add_ttm_yoy(inc, "OPERA_REV", "OPERA_REV_TTM", None)
    inc = add_ttm_yoy(inc, "LESS_OPERA_COST", "LESS_OPERA_COST_TTM", None)
    inc = add_ttm_yoy(inc, "NET_PRO_INCL_MIN_INT_INC", "NET_PRO_TTM", None)
    rev_ttm = pit32(inc, cal_idx, codes, "OPERA_REV_TTM")
    cost_ttm = pit32(inc, cal_idx, codes, "LESS_OPERA_COST_TTM")
    ar = pit32(bal, cal_idx, codes, "ACC_RECEIVABLE")
    inv = pit32(bal, cal_idx, codes, "INV")
    ap = pit32(bal, cal_idx, codes, "ACCT_PAYABLE")
    dso = safe(ar, rev_ttm) * 244.0
    dio = safe(inv, cost_ttm) * 244.0
    dpo = safe(ap, cost_ttm) * 244.0
    panels["ccc"] = dso + dio.fillna(0.0) - dpo.fillna(0.0)
    del ar, inv, ap, dso, dio, dpo

    # ---- ⑥ 披露时滞：ann_date - 报告期末，相对该 code 同类型过去 8 期中位 ----
    d = inc[["code", "ann_date", "report_period"]].dropna(
        subset=["ann_date", "report_period"]).copy()
    d = d.drop_duplicates(subset=["code", "ann_date", "report_period"], keep="last")
    d["qtype"] = d["report_period"].dt.month.map(
        {3: "Q1", 6: "H1", 9: "Q3", 12: "FY"})
    d["delay"] = (d["ann_date"] - d["report_period"]).dt.days
    d = d[d["delay"].between(1, 400)]
    d = d.sort_values(["code", "qtype", "report_period"])
    d["delay_med"] = (d.groupby(["code", "qtype"])["delay"]
                      .transform(lambda s: s.rolling(8, min_periods=4).median()))
    d["report_delay"] = d["delay"] - d["delay_med"]   # 正=比历史晚披露
    panels["report_delay"] = pit32(d, cal_idx, codes, "report_delay")

    # ---- ⑦ Piotroski F-Score：9 项 0/1 打分 ----
    bal_k = bal[["code", "ann_date", "report_period", "TOTAL_ASSETS",
                 "TOTAL_LIAB", "TOTAL_CUR_ASSETS", "TOTAL_CUR_LIAB",
                 "TOT_SHARE"]].dropna(subset=["report_period"])
    bal_k = bal_k.drop_duplicates(subset=["code", "report_period"], keep="last")
    cfo_k = cfo[["code", "ann_date", "report_period",
                 "NET_CASH_FLOWS_OPERA_ACT"]].dropna(
        subset=["report_period"])
    cfo_k = cfo_k.drop_duplicates(subset=["code", "report_period"], keep="last")
    r = inc[["code", "ann_date", "report_period", "OPERA_REV_TTM",
             "LESS_OPERA_COST_TTM", "NET_PRO_TTM"]].dropna(
        subset=["report_period"]).drop_duplicates(
        subset=["code", "ann_date", "report_period"], keep="last")
    r = r.merge(bal_k.drop(columns=["ann_date"]), on=["code", "report_period"],
                how="left").merge(
        cfo_k.drop(columns=["ann_date"]), on=["code", "report_period"], how="left")
    r["ROA"] = safe(r["NET_PRO_TTM"], r["TOTAL_ASSETS"])
    r["LEV"] = safe(r["TOTAL_LIAB"], r["TOTAL_ASSETS"])
    r["CUR"] = safe(r["TOTAL_CUR_ASSETS"], r["TOTAL_CUR_LIAB"])
    r["TURN"] = safe(r["OPERA_REV_TTM"], r["TOTAL_ASSETS"])
    r["GM_RATIO"] = safe(r["OPERA_REV_TTM"] - r["LESS_OPERA_COST_TTM"],
                          r["OPERA_REV_TTM"])
    roa = pit32(r, cal_idx, codes, "ROA")
    cfo_ttm = pit32(r, cal_idx, codes, "NET_CASH_FLOWS_OPERA_ACT")
    np_ttm = pit32(r, cal_idx, codes, "NET_PRO_TTM")
    score = (roa > 0).astype(np.float32) + (cfo_ttm > 0).astype(np.float32) \
        + (cfo_ttm > np_ttm).astype(np.float32)
    for fld, direction in (("ROA", "up"), ("LEV", "down"), ("CUR", "up"),
                           ("TOT_SHARE", "down"), ("GM_RATIO", "up"),
                           ("TURN", "up")):
        rr = year_offset(r, fld, "delta")
        sig = pit32(rr, cal_idx, codes, f"_off_{fld}")
        good = (sig > 0) if direction == "up" else (sig < 0)
        score = score + good.astype(np.float32)
    panels["piotroski_f"] = score  # 0-9，正=质量高

    # ---- ⑧ 存货异动：存货同比 - 营收同比（正=压货，负向） ----
    bal = year_offset(bal, "INV", "ratio")
    inv_yoy = pit32(bal, cal_idx, codes, "_off_INV")
    inc = add_ttm_yoy(inc, "OPERA_REV", "OPERA_REV_TTM", "REV_YOY")
    rev_yoy = pit32(inc, cal_idx, codes, "REV_YOY")
    panels["inv_rev_gap"] = inv_yoy - rev_yoy

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
    del equity  # 本 builder 不用股本结构表，尽早释放
    # 在 main 里瘦身（而非 build_panels 内）：宽版原始表（income 95 列 /
    # balance 176 列，合计 ~650MB）随覆盖赋值立刻释放，构建期间不再持有
    income = income[["code", "ann_date", "report_period", "OPERA_REV",
                     "LESS_OPERA_COST", "NET_PRO_INCL_MIN_INT_INC",
                     "NET_PRO_AFTER_DED_NR_GL"]].reset_index(drop=True)
    balance = balance[["code", "ann_date", "report_period", "TOTAL_ASSETS",
                       "TOTAL_LIAB", "TOTAL_CUR_ASSETS", "TOTAL_CUR_LIAB",
                       "ACC_RECEIVABLE", "INV", "ACCT_PAYABLE", "CAP_RESV",
                       "TOT_SHARE"]].reset_index(drop=True)
    gc.collect()
    codes = close_adj.columns
    cal_idx = close_adj.index
    fwd = {h: close_adj.pct_change(h, fill_method=None).shift(-h) for h in HORIZONS}
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]

    names_all = ["sue_q", "asset_growth_yoy", "capex_intensity",
                 "spsr", "bonus_freq_3y",
                 "ccc", "report_delay", "piotroski_f", "inv_rev_gap"]
    if args.only:
        names_all = [n for n in names_all if n in set(args.only)]
    labels = {
        "sue_q": "单季扣非SUE(标准化季节差)",
        "asset_growth_yoy": "总资产同比",
        "capex_intensity": "投资强度(投资净流出/总资产)",
        "spsr": "每股资本公积(送转潜力)",
        "bonus_freq_3y": "近3年送转次数",
        "ccc": "现金转换周期(负向资金占用)",
        "report_delay": "披露时滞相对偏移(晚披露=坏消息)",
        "piotroski_f": "Piotroski F-Score(0-9)",
        "inv_rev_gap": "存货同比-营收同比(压货负向)",
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
