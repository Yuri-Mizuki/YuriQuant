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
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("check_data_quality")

# 阈值（缺省即告警的边界）
WARN_MISSING = 0.05      # K 线缺失率 > 5% WARN
ERROR_MISSING = 0.20     # K 线缺失率 > 20% ERROR
WARN_ADJ_JUMP = 0.001    # 复权因子单日跳变 > 0.1% 的样本占比 WARN
WARN_PRICE_JUMP = 0.0005  # 单日 |收益| > 30% 的样本占比 WARN
WARN_NAN_RATE = 0.20     # 财务关键字段 NaN 率 WARN
WARN_COVERAGE = 0.90     # 成分覆盖度 < 90% WARN


# ---------------------------------------------------------------------------
# 检查项
# ---------------------------------------------------------------------------
def check_kline_missing(cache, codes, cal) -> pd.DataFrame:
    """逐股 K 线缺失率（按交易日历应有行 vs 实际行）。"""
    kline = cache.get_daily_kline(codes, cal[0], cal[-1])
    if kline.empty:
        return pd.DataFrame(columns=["code", "n_expected", "n_actual", "missing_rate"])
    kline = kline.reset_index()
    counts = kline.groupby("code")["date"].nunique()
    n_expected = len(cal)
    rows = []
    for c in codes:
        actual = int(counts.get(c, 0))
        rows.append({"code": c, "n_expected": n_expected, "n_actual": actual,
                     "missing_rate": 1.0 - actual / n_expected if n_expected else 0.0})
    return pd.DataFrame(rows).sort_values("missing_rate", ascending=False)


def check_financial_nan(cache, codes) -> pd.DataFrame:
    """财务三表关键字段的 NaN 率（按 code）。"""
    fields = {"income": "OPERA_REV", "balance": "TOTAL_ASSETS", "cash_flow": "WS_OPERA_ACT"}
    frames = []
    for table, field in fields.items():
        fn = {"income": cache.get_income, "balance": cache.get_balance_sheet,
              "cash_flow": cache.get_cash_flow}[table]
        try:
            df = fn(codes)
        except Exception as e:
            log.warning("财务表 %s 读取失败: %s", table, e)
            continue
        if df is None or df.empty or field not in df.columns:
            continue
        rate = df.groupby("code")[field].apply(lambda s: s.isna().mean())
        frames.append(pd.DataFrame({"code": rate.index, f"nan_rate_{table}": rate.values}))
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="code", how="outer")
    nan_cols = [c for c in out.columns if c.startswith("nan_rate_")]
    out["max_nan_rate"] = out[nan_cols].max(axis=1)
    return out.sort_values("max_nan_rate", ascending=False)


def check_adjust_factor_jumps(cache, codes, cal=None) -> pd.DataFrame:
    """复权因子跳变（排除除权除息日）：|Δ/prev| > 15% 且当日无除权/除息标记。

    复权因子在**除权除息日**跳变是正常现象（10 送 X 会跳 30-50%），必须用
    ``history_stock_status`` 的 is_ex_dividend/is_ex_rights 标记剔除，否则
    大量误报（真实 HS300 一年 1275 条"跳变"几乎全是除权）。此外早期历史
    （90 年代高送转）无除权标记，因此只检查 ``cal`` 指定的研究区间
    （数据错误检查只对近期数据有意义）。
    未标记且跳变 > 15% 的样本才记入（疑似数据错误）。
    """
    backward = cache.get_backward_factor(codes)
    if backward is None or backward.empty:
        return pd.DataFrame(columns=["date", "code", "prev", "curr", "jump"])
    if cal:
        lo, hi = pd.Timestamp(str(cal[0])), pd.Timestamp(str(cal[-1]))
        backward = backward.loc[lo:hi]
    if backward.empty:
        return pd.DataFrame(columns=["date", "code", "prev", "curr", "jump"])
    # 除权除息日标记（date, code → bool），只拉研究区间
    ex_days: set[tuple] = set()
    try:
        status = cache.get_history_stock_status(codes, int(cal[0]) if cal else int(backward.index.min().strftime("%Y%m%d")),
                                                int(cal[-1]) if cal else int(backward.index.max().strftime("%Y%m%d")))
        if status is not None and not status.empty:
            for col in ("is_ex_dividend", "is_ex_rights"):
                if col in status.columns:
                    sub = status[status[col].fillna(False)]
                    ex_days |= set(zip(pd.to_datetime(sub["date"]), sub["code"]))
    except Exception:
        pass
    prev = backward.shift(1)
    jump = (backward / prev - 1.0).abs()
    bad = jump.where(jump > 0.15)
    rows = []
    for d in bad.index:
        s = bad.loc[d].dropna()
        for c in s.index:
            if (d, c) in ex_days:
                continue        # 除权除息日 → 正常跳变
            rows.append({"date": d, "code": c,
                         "prev": float(backward.loc[d, c] / (1 + s[c])),
                         "curr": float(backward.loc[d, c]), "jump": float(s[c])})
    return pd.DataFrame(rows)


def check_price_anomalies(cache, codes, cal) -> pd.DataFrame:
    """价格异常：close<=0 与单日 |收益|>30%（涨跌停外）的明细。"""
    kline = cache.get_daily_kline(codes, cal[0], cal[-1])
    if kline.empty:
        return pd.DataFrame(columns=["date", "code", "close", "ret"])
    close = kline["close"].unstack("code")
    ret = close.pct_change()
    bad_price = (close <= 0).stack()
    bad_ret = (ret.abs() > 0.30).stack()
    rows = []
    for idx in bad_price[bad_price].index:
        rows.append({"date": idx[0], "code": idx[1], "close": float(close.loc[idx[0], idx[1]]), "ret": np.nan})
    for idx in bad_ret[bad_ret].index:
        rows.append({"date": idx[0], "code": idx[1],
                     "close": float(close.loc[idx[0], idx[1]]),
                     "ret": float(ret.loc[idx[0], idx[1]])})
    return pd.DataFrame(rows)


def check_coverage(cache, codes, cal) -> float:
    """指数成分覆盖度：成分股在 K 线中至少出现一天的比例。"""
    kline = cache.get_daily_kline(codes, cal[0], cal[-1])
    if kline.empty:
        return 0.0
    present = set(kline.index.get_level_values("code"))
    return len(present) / len(codes) if codes else 1.0


# ---------------------------------------------------------------------------
# 汇总 + 告警
# ---------------------------------------------------------------------------
def _flag(v, *warn_bounds) -> tuple[str, bool]:
    """(级别, 是否触发)。warn_bounds = (warn_at, error_at)；None 表示该级别不设。"""
    if warn_bounds[1] is not None and v > warn_bounds[1]:
        return "ERROR", True
    if warn_bounds[0] is not None and v > warn_bounds[0]:
        return "WARN", True
    return "OK", False


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
    codes = uni.get_constituent(index_code, target)
    log.info("检查 %s：%d 只成分股，%d 个交易日，数据指纹 %s",
             index_code, len(codes), len(cal), cache.get_fingerprint())

    problems: list[tuple[str, str, str]] = []   # (级别, 检查项, 说明)
    details: dict[str, pd.DataFrame] = {}

    # 1) K 线缺失
    miss = check_kline_missing(cache, codes, cal)
    if not miss.empty:
        rate = float(miss["missing_rate"].mean())
        worst = miss.iloc[0]
        lvl, trig = _flag(rate, WARN_MISSING, ERROR_MISSING)
        if trig:
            problems.append((lvl, "kline_missing",
                             f"平均缺失率 {rate:.1%}，最差 {worst['code']} {worst['missing_rate']:.1%}"))
        log.info("K线缺失率: 平均 %.2f%%（最差 %s %.2f%%）", rate * 100, worst["code"], worst["missing_rate"] * 100)
        details["kline_missing"] = miss.head(20)

    # 2) 财务 NaN
    fin = check_financial_nan(cache, codes)
    if not fin.empty:
        rate = float(fin["max_nan_rate"].mean())
        lvl, trig = _flag(rate, WARN_NAN_RATE, None)
        if trig:
            problems.append((lvl, "financial_nan", f"财务字段平均 NaN 率 {rate:.1%}"))
        log.info("财务字段平均 NaN 率: %.2f%%", rate * 100)
        details["financial_nan"] = fin.head(20)

    # 3) 复权因子跳变（已剔除除权除息日 + 限研究区间）
    adj = check_adjust_factor_jumps(cache, codes, cal)
    n_adj = len(adj)
    if n_adj > 0:
        lvl, trig = _flag(n_adj / max(len(codes) * len(cache.get_backward_factor(codes).columns), 1),
                          WARN_ADJ_JUMP, None)
        if trig:
            problems.append((lvl, "adj_factor_jump", f"复权因子异常跳变样本 {n_adj} 条（已排除除权日）"))
        log.info("复权因子异常跳变(非除权日,>15%%): %d 条", n_adj)
        details["adj_factor_jump"] = adj.head(20)

    # 4) 价格异常
    px = check_price_anomalies(cache, codes, cal)
    if not px.empty:
        lvl, trig = _flag(len(px) / max(len(codes) * len(cal), 1), WARN_PRICE_JUMP, None)
        if trig:
            problems.append((lvl, "price_anomaly", f"价格异常样本 {len(px)} 条"))
        log.info("价格异常(close<=0 或 |收益|>30%%): %d 条", len(px))
        details["price_anomaly"] = px.head(20)

    # 5) 成分覆盖
    cov = check_coverage(cache, codes, cal)
    lvl, trig = _flag(1.0 - cov, None, 1.0 - WARN_COVERAGE)
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
