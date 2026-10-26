"""
ETF 资金流 / 净值数据更新脚本
==============================

用法:
    python -m scripts.ingest.update_etf_flow                    # 全池回补（份额/净值/IOPV）
    python -m scripts.ingest.update_etf_flow --tables share,nav
    python -m scripts.ingest.update_etf_flow --codes 510300.SH,510500.SH
    python -m scripts.ingest.update_etf_flow --force             # 忽略覆盖判断，强制回源

说明:
    - 数据源：AmazingData ``InfoData.get_fund_share / get_fund_nav / get_fund_iopv``。
    - 落盘（``Config.paths()["cache_root"]``）:
        fund_share.parquet  份额变动，(date=CHANGE_DATE, code) 索引
        fund_nav.parquet    基金净值，(date=PRICE_DATE, code) 索引
        fund_iopv.parquet   ETF 参考净值，(date=PRICE_DATE, code) 索引
    - **逐只代码调用**：SDK 多代码一次拉取会因下载量大而超时（B2 审计实测）。
    - 增量：默认按「本地末日期 >= 最近交易日 - 3 天」跳过已覆盖标的；
      ``--force`` 强制重拉。份额 PIT 用 ``ann_date``（实测 ANN_DATE>=CHANGE_DATE=1.000）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

import pandas as pd  # noqa: E402

from config import Config  # noqa: E402
from data.cache import DataCache  # noqa: E402
from data.datasource import create_datasource  # noqa: E402
from data.etf_universe import ETF_CANDIDATES  # noqa: E402
from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("update_etf_flow")

# 表名 -> (数据源方法名, 缓存文件名, 日期列名)
TABLES: dict[str, tuple[str, str, str]] = {
    "share": ("get_fund_share", "fund_share.parquet", "CHANGE_DATE"),
    "nav": ("get_fund_nav", "fund_nav.parquet", "PRICE_DATE"),
    "iopv": ("get_fund_iopv", "fund_iopv.parquet", "PRICE_DATE"),
}


def _merge_save(root: Path, filename: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """本地全量 + 新数据合并去重落盘，返回合并后的全量。"""
    p = root / filename
    local = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    if new_df is None or new_df.empty:
        return local
    combined = pd.concat([local, new_df]) if not local.empty else new_df
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(p, compression="snappy")
    return combined


def _last_dates(df: pd.DataFrame) -> dict[str, pd.Timestamp]:
    if df is None or df.empty:
        return {}
    ts = df.index.get_level_values("date")
    codes = df.index.get_level_values("code")
    return pd.DataFrame({"date": ts, "code": codes}).groupby("code")["date"].max().to_dict()


def main() -> None:
    ap = argparse.ArgumentParser(description="ETF 份额 / 净值 / IOPV 更新")
    ap.add_argument("--tables", default="share,nav,iopv",
                    help="逗号分隔：share,nav,iopv（默认全量）")
    ap.add_argument("--codes", default=None, help="逗号分隔代码，默认取 ETF_CANDIDATES")
    ap.add_argument("--force", action="store_true", help="忽略覆盖判断，强制回源")
    ap.add_argument("--min-lag-days", type=int, default=3,
                    help="本地末日期距最近交易日不超过该天数则跳过")
    args = ap.parse_args()

    root = Path(Config.paths()["cache_root"])
    codes = args.codes.split(",") if args.codes else list(ETF_CANDIDATES)
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    bad = [t for t in tables if t not in TABLES]
    if bad:
        raise SystemExit(f"未知表: {bad}，可选 {list(TABLES)}")

    ds = create_datasource()
    cache = DataCache(ds)
    cal = cache.get_calendar(20100101, None)
    last_trade = pd.Timestamp(str(cal[-1])) if cal else pd.Timestamp.now().normalize()

    log.info("池 %d 只；表 %s；最近交易日 %s；cache_root=%s",
             len(codes), tables, last_trade.date(), root)

    for table in tables:
        meth, filename, _date_col = TABLES[table]
        fn = getattr(ds, meth, None)
        if fn is None:
            log.warning("%s 数据源不支持，跳过", table)
            continue
        local = pd.read_parquet(root / filename) if (root / filename).exists() else None
        covered = _last_dates(local)
        ok, fail, skip = 0, [], 0
        for code in codes:
            last = covered.get(code)
            if (not args.force and last is not None
                    and (last_trade - last).days <= args.min_lag_days):
                skip += 1
                continue
            try:
                df = fn([code])
            except Exception as e:  # noqa: BLE001 - 单只失败不阻断其余
                log.warning("  %s %s 失败: %s: %s", table, code, type(e).__name__, e)
                fail.append(code)
                continue
            if df is None or df.empty:
                log.warning("  %s %s 空表（代码可能不支持该表）", table, code)
                fail.append(code)
                continue
            _merge_save(root, filename, df)
            ok += 1
            log.info("  %s %s 新增 %d 行 [%s -> %s]", table, code, len(df),
                     df.index.get_level_values("date").min().date(),
                     df.index.get_level_values("date").max().date())
        total = _merge_save(root, filename, pd.DataFrame())
        log.info("%s 完成：成功 %d / 跳过 %d / 失败 %d；累计 %d 行 / %d 只",
                 table, ok, skip, len(fail),
                 len(total), total.index.get_level_values("code").nunique() if len(total) else 0)
        if fail:
            log.warning("%s 失败代码: %s", table, fail)

        if table == "share" and not total.empty and "ann_date" in total.columns:
            flat = total.reset_index()
            m = flat["ann_date"].notna()
            ratio = float((flat.loc[m, "ann_date"] >= flat.loc[m, "date"]).mean())
            log.info("份额 PIT 校验：ANN_DATE>=CHANGE_DATE 占比 %.3f（应=1.000）", ratio)

    log.info("ETF 资金流数据更新完成。")


if __name__ == "__main__":
    main()
