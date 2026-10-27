"""B2 一次性审计：SDK 未用表（ETF 份额 / 指数日权重 / 可转债 / 期权 / 国债）。

对应 TODO_ARCHIVE.md 册Ⅰ §六 B2 与 `reports/frontier_audit_2026-10-19/sdk_field_audit.md`。
本脚本只做**字段级实测**（字段清单 / 覆盖与历史 / PIT / 频率），不做策略探针。

口径：
- 每表取**代表性样本**（不全量拉取），全量吞吐另估；
- PIT 只做**结构判定**（有无公告时间戳、公告日 vs 数据日先后），不下结论式断言；
- 任一表失败只记录原因，不阻断其余表（审计的价值一半在「不可得/格式未明」上）。

产出：`reports/frontier_audit_2026-10-19/sdk_field_audit/`
  - `{table}.csv`   字段清单（列名 + dtype + 非空率 + 示例值）
  - `summary.csv`   每表一行：样本 / 行数 / 覆盖标的 / 起止 / 频率 / PIT 结构
"""

from __future__ import annotations

import argparse
import queue
import threading
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "frontier_audit_2026-10-19" / "sdk_field_audit"

# 样本选择：尽量覆盖沪深两市 / 不同年份挂牌的标的
SAMPLES = {
    "index_weight": ["000300.SH", "000905.SH", "000852.SH"],
    "fund_share": ["510300.SH", "510500.SH", "159915.SZ"],
    "kzz_conv": ["113050.SH", "128136.SZ"],
    "option_basic_info": None,   # 运行时从 get_option_code_list 取前 3 个
    "treasury_yield": None,      # 代码格式未明，运行时探测
}

# 表名 → SDK 方法名（表名沿用审计清单编号，方法名带 get_ 前缀）
METHOD = {
    "index_weight": "get_index_weight",
    "fund_share": "get_fund_share",
    "kzz_conv": "get_kzz_conv",
    "option_basic_info": "get_option_basic_info",
    "treasury_yield": "get_treasury_yield",
}


def _call(fn, seconds=180.0):
    """SDK 可能挂死（无超时），用 daemon 线程 + queue 硬超时。"""
    q: queue.Queue = queue.Queue()

    def _run():
        try:
            q.put(("ok", fn()))
        except BaseException as exc:  # noqa: BLE001 - 审计要看到任何异常
            q.put(("err", f"{type(exc).__name__}: {exc}"))

    threading.Thread(target=_run, daemon=True).start()
    try:
        return q.get(timeout=seconds)
    except queue.Empty:
        return ("timeout", f">{seconds:.0f}s 无返回")


def _field_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        nonnull = float(s.notna().mean())
        sample = ""
        if len(s.dropna()):
            sample = str(s.dropna().iloc[0])[:40]
        rows.append({"column": col, "dtype": str(s.dtype),
                     "nonnull": round(nonnull, 4), "sample": sample,
                     "n_unique": int(s.nunique(dropna=True))})
    return pd.DataFrame(rows)


def _date_col(df: pd.DataFrame) -> str | None:
    """优先取「数据日」列（TRADE_DATE / CHANGE_DATE），其次 ANN_DATE。"""
    for c in ("TRADE_DATE", "CHANGE_DATE", "END_DATE", "ANN_DATE", "DATE"):
        if c in df.columns:
            return c
    return None


def _summarize(table: str, df: pd.DataFrame, note: str) -> dict:
    dcol = _date_col(df)
    dmin = dmax = ""
    if dcol is not None:
        d = pd.to_numeric(df[dcol], errors="coerce").dropna()
        if len(d):
            dmin, dmax = str(int(d.min())), str(int(d.max()))
    id_col = next((c for c in ("CON_CODE", "MARKET_CODE", "FUND_CODE", "CODE",
                               "SECURITY_CODE", "INDEX_CODE")
                   if c in df.columns and df[c].nunique() < len(df)), None)
    n_codes = int(df[id_col].nunique()) if id_col else None
    # PIT 结构：公告日是否 >= 数据日（份额表典型：ANN_DATE >= CHANGE_DATE）
    pit = "无公告时间戳"
    if {"ANN_DATE", "CHANGE_DATE"} <= set(df.columns):
        ann = pd.to_numeric(df["ANN_DATE"], errors="coerce")
        chg = pd.to_numeric(df["CHANGE_DATE"], errors="coerce")
        ok = (ann >= chg).mean()
        pit = f"ANN_DATE>=CHANGE_DATE 占比 {ok:.3f}（公告后才可知）"
    elif dcol is not None:
        pit = f"仅数据日 {dcol}，无公告时间戳 ⇒ 需按 T+1 使用"
    return {"table": table, "n_rows": len(df), "n_codes": n_codes,
            "date_col": dcol or "", "date_min": dmin, "date_max": dmax,
            "pit_structure": pit, "note": note}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="B2 SDK 未用表字段审计")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from config import Config
    from data.datasource import create_datasource

    local = Config.paths()["sdk_local_path"]
    ds = create_datasource()
    info = ds._info

    def fetch(meth: str, codes):
        fn = getattr(info, meth)
        return fn(codes, local_path=local, is_local=False)

    if SAMPLES["option_basic_info"] is None:
        st, codes = _call(lambda: ds._base.get_option_code_list(), seconds=60)
        SAMPLES["option_basic_info"] = (codes[:3] if st == "ok" and codes else [])
        print(f"option codes: {st}, n={len(codes) if st == 'ok' else '?'}")

    if SAMPLES["treasury_yield"] is None:
        # 代码格式未知：先试几种常见写法，能取到就记下（审计的价值在此）
        for guess in ("CN", "CN10Y", "1001", "TB.CN", "000001.SH"):
            st, val = _call(lambda g=guess: fetch("get_treasury_yield", [g]),
                            seconds=60)
            if st == "ok" and isinstance(val, dict) and val:
                det = next(iter(val.values()))
                if getattr(det, "empty", True) is False:
                    SAMPLES["treasury_yield"] = [guess]
                    print(f"treasury code 命中: {guess}")
                    break
            print(f"  treasury 试探 {guess}: {st}")

    summary = []
    for table, codes in SAMPLES.items():
        if not codes:
            summary.append({"table": table, "n_rows": 0, "n_codes": 0,
                            "date_col": "", "date_min": "", "date_max": "",
                            "pit_structure": "", "note": "样本代码未取到"})
            print(f"=== {table}: 跳过（无样本代码）")
            continue
        meth = METHOD[table]
        # **按代码逐个取**：SDK 多代码一次拉取会因下载量大而超时（实测 3 只
        # 指数 >180s），逐只取既能硬超时也能看清是哪个标的问题。
        profiled, frames, notes = False, [], []
        for code in codes:
            st, val = _call(lambda c=code: fetch(meth, [c]), seconds=600)
            print(f"  {table} {code} -> {st}")
            if st != "ok":
                notes.append(f"{code}: {st}")
                continue
            if isinstance(val, dict) and not val:
                notes.append(f"{code}: 空 dict（代码格式可能不对）")
                continue
            df = val if isinstance(val, pd.DataFrame) else next(iter(val.values()))
            if getattr(df, "empty", True):
                notes.append(f"{code}: 空表")
                continue
            if not profiled:
                _field_profile(df).to_csv(out / f"{table}.csv", index=False,
                                          encoding="utf-8-sig")
                profiled = True
            row = _summarize(table, df, note=code)
            frames.append(row)
            print(f"    rows={row['n_rows']} dates={row['date_min']}->"
                  f"{row['date_max']}")
        if frames:
            agg = dict(frames[0])
            agg.update({"table": table, "n_codes": len(frames),
                        "note": f"样本 {len(frames)}/{len(codes)} 只可取; "
                                + ("; ".join(notes) if notes else "全部可取")})
            summary.append(agg)
            print(f"   === {table}: {agg['n_rows']} 行/只, "
                  f"{agg['date_min']}->{agg['date_max']}, pit={agg['pit_structure']}")
        else:
            summary.append({"table": table, "n_rows": 0, "n_codes": 0,
                            "date_col": "", "date_min": "", "date_max": "",
                            "pit_structure": "", "note": "; ".join(notes)})

    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False,
                                 encoding="utf-8-sig")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
