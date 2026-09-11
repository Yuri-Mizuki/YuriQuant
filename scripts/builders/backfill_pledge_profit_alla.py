"""回补全A（含退市）股权质押 / 业绩预告 / 业绩快报三张新表到本地 parquet。

供 B+ 基本面族因子构建（build_alla_pledge_factors）使用：
- equity_pledge_freeze: 股权质押/冻结
- profit_notice        : 业绩预告（净利变动区间，前瞻信息代理）
- profit_express       : 业绩快报（净利/营收同比）

按 master_codes（含退市，来自 _status_refetch/master_codes.parquet）逐批拉取，
断点续跑（_fetch_progress_{table}.json 记录已完成的批号）。列名统一归一化：
MARKET_CODE→code, ANN_DATE→ann_date, REPORTING_PERIOD→report_period 等。

用法（系统 Python 3.12，真数据源）:
    python -m scripts.builders.backfill_pledge_profit_alla --batch 80
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402

TABLES = {
    "equity_pledge_freeze": {
        "renames": {"MARKET_CODE": "code", "ANN_DATE": "ann_date", "HOLDER_NAME": "holder_name",
                    "TOTAL_PLEDGE_SHR": "total_pledge_shr", "FRO_SHARES": "fro_shares",
                    "FRO_SHR_TO_TOTAL_RATIO": "fro_shr_to_total_ratio",
                    "TOTAL_HOLDING_SHR_RATIO": "total_holding_shr_ratio",
                    "IS_EQUITY_PLEDGE_REPO": "is_equity_pledge_repo"},
        "date_cols": ["ann_date", "begin_date", "end_date", "disfrozen_time"],
    },
    "profit_notice": {
        "renames": {"MARKET_CODE": "code", "ANN_DATE": "ann_date",
                    "REPORTING_PERIOD": "report_period",
                    "P_CHANGE_MAX": "p_change_max", "P_CHANGE_MIN": "p_change_min",
                    "NET_PROFIT_MAX": "net_profit_max", "NET_PROFIT_MIN": "net_profit_min",
                    "P_NUMBER": "p_number", "FIRST_ANN_DATE": "first_ann_date"},
        "date_cols": ["ann_date", "report_period", "first_ann_date"],
    },
    "profit_express": {
        "renames": {"MARKET_CODE": "code", "ANN_DATE": "ann_date",
                    "REPORTING_PERIOD": "report_period",
                    "YOY_GR_NET_PROFIT_PARENT": "yoy_gr_net_profit_parent",
                    "YOY_GR_GROSS_REV": "yoy_gr_gross_rev", "YOY_GR_REV": "yoy_gr_rev",
                    "ROE_WEIGHTED": "roe_weighted"},
        "date_cols": ["ann_date", "report_period"],
    },
    "equity_restricted": {
        "renames": {"MARKET_CODE": "code", "LIST_DATE": "list_date",
                    "SHARE_RATIO": "share_ratio", "SHARE_LST": "share_lst",
                    "SHARE_LST_TYPE_NAME": "share_lst_type_name",
                    "SHARE_LST_IS_ANN": "share_lst_is_ann",
                    "CLOSE_PRICE": "close_price",
                    "SHARE_LST_MARKET_VALUE": "share_lst_market_value"},
        "date_cols": ["list_date"],
    },
    "margin_detail": {
        "renames": {"MARKET_CODE": "code", "TRADE_DATE": "trade_date",
                    "BORROW_MONEY_BALANCE": "borrow_money_balance",
                    "PURCH_WITH_BORROW_MONEY": "purch_with_borrow_money",
                    "REPAYMENT_OF_BORROW_MONEY": "repayment_of_borrow_money",
                    "SEC_LENDING_BALANCE": "sec_lending_balance",
                    "SALES_OF_BORROWED_SEC": "sales_of_borrowed_sec",
                    "MARGIN_TRADE_BALANCE": "margin_trade_balance"},
        "date_cols": ["trade_date"],
    },
    "long_hu_bang": {
        "renames": {"MARKET_CODE": "code", "TRADE_DATE": "trade_date",
                    "BUY_AMOUNT": "buy_amount", "SELL_AMOUNT": "sell_amount",
                    "FLOW_MARK": "flow_mark", "TOTAL_AMOUNT": "total_amount",
                    "TOTAL_VOLUME": "total_volume"},
        "date_cols": ["trade_date"],
    },
    "block_trading": {
        "renames": {"MARKET_CODE": "code", "TRADE_DATE": "trade_date",
                    "B_SHARE_PRICE": "b_share_price", "B_SHARE_VOLUME": "b_share_volume",
                    "B_SHARE_AMOUNT": "b_share_amount", "B_FREQUENCY": "b_frequency",
                    "BLOCK_AVG_VOLUME": "block_avg_volume"},
        "date_cols": ["trade_date"],
    },
}

NUMERIC_SUFFIX = (
    "_shr", "_ratio", "_max", "_min", "_weighted", "_rev", "_yoy", "_np",
    "_time", "_date",
)


def _normalize(raw, spec: dict) -> pd.DataFrame:
    if isinstance(raw, dict):
        frames = [df.assign(_code=code) for code, df in raw.items()
                  if df is not None and not (hasattr(df, "empty") and df.empty)]
        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, ignore_index=True)
    elif isinstance(raw, pd.DataFrame) and not raw.empty:
        frame = raw.copy()
    else:
        return pd.DataFrame()

    renames = {k: v for k, v in spec["renames"].items() if k in frame.columns}
    out = frame.rename(columns=renames)
    if "code" not in out.columns:
        out["code"] = out["_code"]
    for c in spec["date_cols"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    for c in out.columns:
        if c.endswith(NUMERIC_SUFFIX):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=80)
    ap.add_argument("--tables", nargs="+", default=list(TABLES))
    args = ap.parse_args()

    root = Path(str(Config.cache()["root"]))
    master = sorted(pd.read_parquet(root / "_status_refetch" / "master_codes.parquet")
                    ["code"].astype(str))
    master = [c for c in master if c.endswith((".SH", ".SZ", ".BJ"))]
    print(f"主清单(含退市) {len(master)} 只", flush=True)

    from data.datasource import create_datasource
    ds = create_datasource()
    lpath = ds._local_path

    for t in args.tables:
        spec = TABLES[t]
        fn = getattr(ds._info, f"get_{t}")
        p = root / f"{t}.parquet"
        if p.exists():
            have = set(pd.read_parquet(p, columns=["code"])["code"].astype(str))
        else:
            have = set()
        missing = sorted(set(master) - have)
        print(f"[{t}] 已有 {len(have)} 只 | 缺失 {len(missing)} 只", flush=True)
        if not missing:
            print(f"[{t}] 无缺失，跳过", flush=True)
            continue

        prog_path = root / f"_fetch_progress_{t}.json"
        prog = set(json.loads(prog_path.read_text(encoding="utf-8"))
                   ) if prog_path.exists() else set()
        batches = [missing[i:i + args.batch] for i in range(0, len(missing), args.batch)]
        t0 = time.time()
        for bi, batch in enumerate(batches):
            if bi in prog:
                continue
            for attempt in (1, 2, 3):
                try:
                    out = fn(batch, local_path=lpath, is_local=False)
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"  批{bi} 第{attempt}次失败: {str(exc)[:110]}", flush=True)
                    time.sleep(4 * attempt)
            df = _normalize(out, spec) if out is not None else pd.DataFrame()
            if not df.empty:
                # 逐批增量落盘：中断也不丢已拉批次（避免进程被杀时内存累积丢失）
                new = pd.concat([pd.read_parquet(p), df], ignore_index=True) \
                    if p.exists() else df
                new.to_parquet(p)
            prog.add(bi)
            prog_path.write_text(json.dumps(sorted(prog)), encoding="utf-8")
            print(f"[{t}] 批{bi}/{len(batches)} 完成 "
                  f"{len(df) if not df.empty else 0} 行 "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()