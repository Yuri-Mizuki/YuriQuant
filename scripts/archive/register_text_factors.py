"""
文本因子入库 factor_library（AI 41 情感残差族）
================================================

入库对象（2026-09-10 评估结论）：
- txt_senti_res     情感残差因子（对评级/数量正交），行业市值中性化后
                    NW t=2.61，本次复现中最强的文本信号
- txt_senti_adj_res 负面加权版残差（AI 41 的 senti_adj 口径），中性化 t=2.43

不入库：report_num_rev——原始 IC −0.027 (t=−3.4) 经行业市值中性化后缩到
+0.0055 (t=0.9)，属行业/市值构成效应而非个股信号（详见评估报告）。

口径：月度因子面板 → 截面 winsor+标准化 → 按交易日 ffill 扩展为日频面板
（与库内基本面因子口径一致），returns 面板 = daily_all_a 次日收益。
入库走 register 标准流程（IC 序列 + NW t + canonical 回测 + 冗余预检）。

用法：
    python -m scripts.textmining.register_text_factors
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
from scripts.common.cli_common import returns_from_daily, setup_logging  # noqa: E402
from scripts.textmining._paths import Out  # noqa: E402

OUT_DIR = Out("senti")
log = setup_logging("register_text")

DATASET = "all_a_2018_2026"
FACTORS = {
    "txt_senti_res": ("senti_res", "AI41 情感残差（对评级/数量正交）；"
                      "行业市值中性化后 IC 0.0124 / NW t=2.61，本次文本复现最强信号"),
    "txt_senti_adj_res": ("senti_adj_res", "AI41 senti_adj 残差（负面×3 后正交）；"
                          "中性化后 NW t=2.43"),
}


def _winsor_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if mad and mad > 0:
        s = s.clip(med - 5 * mad, med + 5 * mad)
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * np.nan


def _daily_panel(monthly: pd.DataFrame, trading_days: pd.DatetimeIndex,
                 name: str) -> pd.DataFrame:
    """月度因子长表 → 截面标准化 → 日频 ffill 宽面板（date×code）。"""
    monthly = monthly.copy()
    monthly["z"] = monthly.groupby("date")["factor"].transform(_winsor_z)
    wide = monthly.pivot(index="date", columns="code", values="z")
    wide.index = pd.to_datetime(wide.index).normalize()
    daily = wide.reindex(trading_days).ffill()
    log.info("%s: %d 月 → 日频面板 %d 天 × %d 股",
             name, len(wide), len(daily), daily.shape[1])
    return daily


def run(pool: str = "all_a", panel_begin: str = "20180101"):
    from data.cache import DataCache
    from data.cache_helpers import load_daily
    from data.offline import OfflineQuietDataSource
    from data.universe import Universe
    from research.factor_library import FactorLibrary

    cache = DataCache(OfflineQuietDataSource())
    uni = Universe(cache)
    codes, cal, daily = load_daily(cache, uni, "000300.SH", 20190101, None,
                                   pool="all_a")
    returns_panel = returns_from_daily(daily)
    trading_days = returns_panel.index
    trading_days = trading_days[trading_days >= pd.Timestamp(panel_begin)]

    lib = FactorLibrary(dataset=DATASET)
    for name, (src, note) in FACTORS.items():
        p = OUT_DIR / f"senti_factor_{src}_{pool}.parquet"
        if not p.exists():
            log.warning("%s 不存在，跳过", p)
            continue
        monthly = pd.read_parquet(p).reset_index()
        monthly["date"] = pd.to_datetime(monthly["date"])
        panel = _daily_panel(monthly[["date", "code", "factor"]],
                             trading_days, name)
        row = lib.register(
            name=name,
            panel=panel,
            returns_panel=returns_panel,
            family="情绪",
            frequency="月频",
            source=f"textmining:AI41_senti_factors:{pool}",
            note=note,
            check_dup=True,
        )
        log.info("已入库 %s: ic=%.4f t_nw=%.2f significant=%s resid_t_nw=%s",
                 name, row.get("ic_mean", np.nan), row.get("t_stat_nw", np.nan),
                 row.get("significant"), row.get("resid_t_nw"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="all_a", choices=["hs300", "zz1000", "all_a"])
    ap.add_argument("--panel-begin", default="20180101")
    args = ap.parse_args()
    run(args.pool, args.panel_begin)
