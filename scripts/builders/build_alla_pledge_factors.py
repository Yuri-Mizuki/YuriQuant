"""B+ 基本面族因子面板 → 并入 all_a_2018_2026 数据集。

三件套（data 已回补到本地 parquet；原一次性拉取脚本已删，2026-11-20）：
- 商誉     : goodwill_ratio = GOODWILL / 股东权益（balance_sheet 本地已有）
- 股权质押 : pledge_ratio / pledge_holder_ratio / frozen_ratio
             （equity_pledge_freeze，PIT 按公告日）
- 业绩预告 : profit_notice_chg（净利变动区间中值，前瞻信息代理）
- 业绩快报 : profit_express_np_yoy / profit_express_rev_yoy

所有因子严格 PIT（公告日 <= 交易日，前向填充，无未来函数），date×code float32，
与 ic_h{1,5,10,20}.parquet 索引对齐，IC 用次日/未来1/5/10/20日 Spearman。

用法:
    python -m scripts.builders.build_alla_pledge_factors            # 单机
    python -m scripts.builders.build_alla_pledge_factors --resume   # 断点续跑
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("build_alla_pledge")
from scripts.builders import common  # noqa: E402

DATASET = "all_a_2018_2026"


def _safe(x, denom):
    return (x / denom.replace(0.0, np.nan))


def _pit(report_df, cal_idx, codes, field):
    from data.financials import build_pit_panel
    if field not in report_df.columns:
        return pd.DataFrame(np.nan, index=cal_idx, columns=codes)
    return build_pit_panel(report_df, cal_idx, field).reindex(index=cal_idx, columns=codes)


def load_panels():
    """复用 B族 的行情/股本加载：返回 close_adj, close_raw, balance。"""
    from scripts.builders.build_alla_fundamental_factors import load_panels as _load
    close_adj, close_raw, income, balance, cashflow, equity, dividend = _load()
    return close_adj, close_raw, balance


def build_pledge_factors(close_adj, close_raw, balance, pledge=None,
                         notice=None, express=None) -> dict[str, pd.DataFrame]:
    """返回 {因子名: date×code 原始面板}（B+ 三件套）。"""
    codes = close_adj.columns
    cal_idx = close_adj.index
    bal = balance if balance is not None else pd.DataFrame()

    pit = {}
    for f in ("GOODWILL", "TOT_SHARE_EQUITY_EXCL_MIN_INT", "TOT_SHARE"):
        if f in bal.columns:
            pit[f] = _pit(bal, cal_idx, codes, f)

    panels: dict[str, pd.DataFrame] = {}

    # ---- 商誉：商誉 / 净资产（>0 为商誉占比；负净资产剔除）----
    if "GOODWILL" in pit and "TOT_SHARE_EQUITY_EXCL_MIN_INT" in pit:
        equity_pit = pit["TOT_SHARE_EQUITY_EXCL_MIN_INT"]
        gw = _safe(pit["GOODWILL"].clip(lower=0.0), equity_pit.where(equity_pit > 0))
        panels["goodwill_ratio"] = gw.replace([np.inf, -np.inf], np.nan).clip(0.0, 5.0)

    # ---- 股权质押（PIT 按公告日展开最新披露）----
    if pledge is not None and not pledge.empty:
        tot_share = pit.get("TOT_SHARE")
        p_total = _pit(pledge, cal_idx, codes, "total_pledge_shr")
        if tot_share is not None:
            panels["pledge_ratio"] = _safe(p_total, tot_share).replace(
                [np.inf, -np.inf], np.nan).clip(0.0, 1.5)
        for src, name in (("total_holding_shr_ratio", "pledge_holder_ratio"),
                          ("fro_shr_to_total_ratio", "frozen_ratio")):
            if src in pledge.columns:
                panels[name] = (_pit(pledge, cal_idx, codes, src).replace(
                    [np.inf, -np.inf], np.nan).clip(0.0, 1.5))

    # ---- 业绩预告：净利变动区间中值（同比 %）----
    if notice is not None and not notice.empty:
        nt = notice.copy()
        if {"p_change_max", "p_change_min"}.issubset(nt.columns):
            nt["p_change_mid"] = ((nt["p_change_max"] + nt["p_change_min"]) / 2.0)
            panels["profit_notice_chg"] = (
                _pit(nt, cal_idx, codes, "p_change_mid").replace([np.inf, -np.inf], np.nan)
                .clip(-500.0, 1500.0))

    # ---- 业绩快报：净利/营收同比 ---- 
    if express is not None and not express.empty:
        for src, name in (("yoy_gr_net_profit_parent", "profit_express_np_yoy"),
                          ("yoy_gr_gross_rev", "profit_express_rev_yoy")):
            if src in express.columns:
                panels[name] = (_pit(express, cal_idx, codes, src)
                                .replace([np.inf, -np.inf], np.nan).clip(-400.0, 1200.0))

    out = {}
    for n, p in panels.items():
        out[n] = p.reindex(index=cal_idx, columns=codes)
    return out


# ---------------------------------------------------------------------------
# 主流程 & 合并
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from config import Config

    lib_root = Path(str(Config.get()["factor_library"]["root"]))
    ds_dir = lib_root / DATASET
    cache_root = Path(str(Config.cache()["root"]))
    close_adj, close_raw, balance = load_panels()
    pledge = (pd.read_parquet(cache_root / "equity_pledge_freeze.parquet")
              if (cache_root / "equity_pledge_freeze.parquet").exists() else None)
    notice = (pd.read_parquet(cache_root / "profit_notice.parquet")
              if (cache_root / "profit_notice.parquet").exists() else None)
    express = (pd.read_parquet(cache_root / "profit_express.parquet")
               if (cache_root / "profit_express.parquet").exists() else None)

    labels = {
        "goodwill_ratio": "商誉/净资产",
        "pledge_ratio": "股权质押比例",
        "pledge_holder_ratio": "质押占股东持股比",
        "frozen_ratio": "股权冻结比例",
        "profit_notice_chg": "业绩预告净利变动中值",
        "profit_express_np_yoy": "业绩快报净利同比",
        "profit_express_rev_yoy": "业绩快报营收同比",
    }

    panels = build_pledge_factors(close_adj, close_raw, balance, pledge, notice, express)
    common.write_family_panels(
        ds_dir, panels, family="pledge", set_name="fundamental", labels=labels,
        close_adj=close_adj, clip=common.CLIP["WIDE"], resume=args.resume,
        merge_kwargs={"skip_existing": False,
                      "empty_log": "无 B+ 三件套因子统计产出"})
    log.info("B+ 三件套基本面因子构建完成")


if __name__ == "__main__":
    main()