"""
数据质量检查 CLI
================

拉数后的质量体检：K线缺失率 / 财务字段 NaN 率 / 复权因子跳变 / 价格异常 /
成分覆盖度，输出阈值告警（WARN/ERROR）与明细 CSV。

用法:
    python -m scripts.check_data_quality                          # 默认指数成分池
    python -m scripts.check_data_quality --begin 20250101 --end 20251231
    python -m scripts.check_data_quality --strict                 # 有 ERROR 时退出码 1
    python -m scripts.check_data_quality --out reports/data_quality.csv

无 SDK 凭证时自动回退 mock 数据源（离线开发模式）。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cli_common import setup_logging  # noqa: E402

from data.quality import (  # noqa: E402
    ERROR_MISSING, WARN_ADJ_JUMP, WARN_COVERAGE, WARN_MISSING, WARN_NAN_RATE,
    WARN_PRICE_JUMP, check_adjust_factor_jumps, check_coverage,
    check_financial_nan, check_kline_missing, check_price_anomalies, flag,
)

log = setup_logging("check_data_quality")


def main():
    parser = argparse.ArgumentParser(description="YuriQuant 数据质量检查")
    parser.add_argument("--begin", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--index", default=None, help="指数代码，默认 config.universe.index_code")
    parser.add_argument("--strict", action="store_true", help="存在 ERROR 级问题退出码 1")
    parser.add_argument("--out", default=None, help="质量明细 CSV 输出路径")
    args = parser.parse_args()

    from config import Config
    from data.cache import DataCache
    from data.datasource import create_datasource
    from data.universe import Universe

    cfg = Config.get()
    cache = DataCache(create_datasource())
    uni = Universe(cache)
    index_code = args.index or cfg["universe"]["index_code"]
    cal = cache.get_calendar(args.begin or cfg["fetch"]["begin_date"], args.end)
    if not cal:
        log.error("交易日历为空")
        return 1
    target = args.end or cal[-1]
    from data.cache_helpers import _pit_universe_codes
    codes = _pit_universe_codes(uni, index_code, args.begin or cfg["fetch"]["begin_date"], target)
    log.info("检查 %s：%d 只历史在册成分（并集池），%d 个交易日，数据指纹 %s",
             index_code, len(codes), len(cal), cache.get_fingerprint())

    problems: list[tuple[str, str, str]] = []   # (级别, 检查项, 说明)
    details: dict[str, pd.DataFrame] = {}

    # 1) K 线缺失
    miss = check_kline_missing(cache, codes, cal)
    if not miss.empty:
        rate = float(miss["missing_rate"].mean())
        worst = miss.iloc[0]
        lvl, trig = flag(rate, WARN_MISSING, ERROR_MISSING)
        if trig:
            problems.append((lvl, "kline_missing",
                             f"平均缺失率 {rate:.1%}，最差 {worst['code']} {worst['missing_rate']:.1%}"))
        log.info("K线缺失率: 平均 %.2f%%（最差 %s %.2f%%）", rate * 100, worst["code"], worst["missing_rate"] * 100)
        details["kline_missing"] = miss.head(20)

    # 2) 财务 NaN
    fin = check_financial_nan(cache, codes)
    if not fin.empty:
        rate = float(fin["max_nan_rate"].mean())
        lvl, trig = flag(rate, WARN_NAN_RATE, None)
        if trig:
            problems.append((lvl, "financial_nan", f"财务字段平均 NaN 率 {rate:.1%}"))
        log.info("财务字段平均 NaN 率: %.2f%%", rate * 100)
        details["financial_nan"] = fin.head(20)

    # 3) 复权因子跳变（已剔除除权除息日 + 限研究区间）
    adj = check_adjust_factor_jumps(cache, codes, cal)
    n_adj = len(adj)
    if n_adj > 0:
        lvl, trig = flag(n_adj / max(len(codes) * len(cache.get_backward_factor(codes).columns), 1),
                          WARN_ADJ_JUMP, None)
        if trig:
            problems.append((lvl, "adj_factor_jump", f"复权因子异常跳变样本 {n_adj} 条（已排除除权日）"))
        log.info("复权因子异常跳变(非除权日,>15%%): %d 条", n_adj)
        details["adj_factor_jump"] = adj.head(20)

    # 4) 价格异常
    px = check_price_anomalies(cache, codes, cal)
    if not px.empty:
        lvl, trig = flag(len(px) / max(len(codes) * len(cal), 1), WARN_PRICE_JUMP, None)
        if trig:
            problems.append((lvl, "price_anomaly", f"价格异常样本 {len(px)} 条"))
        log.info("价格异常(close<=0 或 |收益|>30%%): %d 条", len(px))
        details["price_anomaly"] = px.head(20)

    # 5) 成分覆盖
    cov = check_coverage(cache, codes, cal)
    lvl, trig = flag(1.0 - cov, None, 1.0 - WARN_COVERAGE)
    if trig:
        problems.append((lvl, "coverage", f"成分覆盖度 {cov:.1%}"))
    log.info("成分覆盖度: %.2f%%", cov * 100)

    # 汇总输出
    if problems:
        print("\n===== 数据质量告警 =====")
        for lvl, item, msg in problems:
            print(f"  [{lvl}] {item}: {msg}")
    else:
        print("\n===== 数据质量检查通过（无告警）=====")

    if args.out or details:
        out_path = Path(args.out) if args.out else Path("reports") / f"data_quality_{datetime.now():%Y%m%d_%H%M%S}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8-sig") as f:
            f.write(f"index,{index_code}\n")
            f.write(f"fingerprint,{cache.get_fingerprint()}\n")
            f.write(f"n_codes,{len(codes)}\nn_days,{len(cal)}\n\n")
            for name, df in details.items():
                f.write(f"### {name}\n")
                f.write(df.to_csv(index=False))
        log.info("质量明细已保存: %s", out_path)

    has_error = any(lvl == "ERROR" for lvl, _, _ in problems)
    return 1 if (args.strict and has_error) else 0

if __name__ == "__main__":
    raise SystemExit(main())
