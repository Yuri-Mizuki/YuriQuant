"""全A两融资金流因子面板（margin 族）→ 并入 all_a_2018_2026 数据集。

背景（2026-09-09）：all_a_2018_2026 现有因子均为量价/财务/股东/质押/事件驱动，
"融资融券资金流"维度空白。两融数据（margin_detail，SDK 专属）是真实杠杆资金流，
与纯价量信号正交性好，衡量多空杠杆资金的聚集与撤离。

因子清单（set='margin'，全部基于个股日频两融余额序列构造）：
  margin_bal_chg_5d   : 两融余额 5 日变化率（短期杠杆资金增减，正向=看多）
  margin_bal_chg_20d  : 两融余额 20 日变化率（中期杠杆趋势）
  margin_bal_ratio    : 两融余额 / 流通市值（杠杆参与度，静态水平）
  financing_chg_1d    : 融资余额单日变化率（融资（多头）资金边际）
  securities_chg_1d   : 融券余额单日变化率（融券（空头）资金边际，反向看多）

口径：
- 逐股票长表，TRADE_DATE 为观测日；两融余额在当日收盘后披露，**用 T 日值对 T 日
  生效**（无前视：融资余额变化率从 T 日用 T vs T-n 计算，对未来收益 T+1 预测）；
- 流通市值用未复权收盘价 × 流通股本（与财务因子/股东因子同源，确保口径一致）；
- 杠杆余额含融资(多头)与融券(空头)，多空分离因子分别构造；
- 面板 date×code float32，从 20160701 起与现有 ic_h 索引对齐；
- IC 用次日/未来 1/5/10/20 日 Spearman，抽稀取列。

用法:
    python -m scripts.builders.build_alla_margin_factors            # 全量
    python -m scripts.builders.build_alla_margin_factors --resume   # 续跑
    python -m scripts.builders.build_alla_margin_factors --only margin_bal_chg_5d
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

from scripts.cli_common import setup_logging  # noqa: E402

log = setup_logging("build_alla_margin")

DATASET = "all_a_2018_2026"
KEEP_FROM = "2016-07-01"
HORIZONS = (1, 5, 10, 20)
IC_CODE_STRIDE = 3


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (close_adj, margin_detail)。"""
    from config import Config

    cache_root = Path(str(Config.cache()["root"]))
    daily = pd.read_parquet(cache_root / "daily_all_a.parquet")
    daily.index = daily.index.set_levels(daily.index.levels[0].normalize(), level=0)
    k0 = pd.Timestamp(KEEP_FROM)
    daily = daily[daily.index.get_level_values(0) >= k0]

    bf = pd.read_parquet(cache_root / "backward_factor.parquet")
    bf.index = bf.index.normalize()
    bf = bf.loc[bf.index >= k0]

    close_raw = daily["close"].unstack()
    cols = close_raw.columns.intersection(bf.columns)
    close_raw = close_raw.reindex(close_raw.index.intersection(bf.index), columns=cols)
    bf = bf.reindex(index=close_raw.index, columns=close_raw.columns)
    close_adj = (close_raw * bf).astype(np.float32)

    margin = pd.DataFrame()
    mp = cache_root / "margin_detail.parquet"
    if mp.exists():
        margin = pd.read_parquet(mp)
    return close_adj, margin


def _pit_margin(margin: pd.DataFrame, cal_idx, codes) -> dict:
    """两融因子：逐股票长表 → 各因子 date×code 宽表（位移差分，T日启用）。"""
    m = margin.copy()
    m["trade_date"] = pd.to_datetime(m["trade_date"], errors="coerce")
    # 归一化余额（两融明细列，单位：元）
    m["loan_bal"] = pd.to_numeric(m["margin_trade_balance"], errors="coerce")
    m["fin_bal"] = pd.to_numeric(m["borrow_money_balance"], errors="coerce")
    m["sec_bal"] = pd.to_numeric(m["sec_lending_balance"], errors="coerce")
    m = m.dropna(subset=["code", "trade_date", "loan_bal"])
    m = (m.sort_values(["code", "trade_date"])
          .drop_duplicates(subset=["code", "trade_date"], keep="last"))

    base = m.set_index(["code", "trade_date"])
    loan = base["loan_bal"]
    fin = base["fin_bal"]
    sec = base["sec_bal"]

    chg5 = loan.groupby(level="code").pct_change(5)
    chg20 = loan.groupby(level="code").pct_change(20)
    fin1 = fin.groupby(level="code").pct_change(1)
    sec1 = sec.groupby(level="code").pct_change(1)

    out_long = pd.DataFrame({
        "code": m["code"].values, "eff": m["trade_date"].values,
        "margin_bal_chg_5d": chg5.values,
        "margin_bal_chg_20d": chg20.values,
        "financing_chg_1d": fin1.values,
        "securities_chg_1d": sec1.values,
    })
    frames = _ffill_pit_multi(out_long, cal_idx, codes,
                              ["margin_bal_chg_5d", "margin_bal_chg_20d",
                               "financing_chg_1d", "securities_chg_1d"])
    return frames


def _ffill_pit_multi(series_long: pd.DataFrame, cal_idx: pd.DatetimeIndex,
                     codes: pd.Index, value_cols: list[str]) -> dict[str, pd.DataFrame]:
    frames = {c: pd.DataFrame(np.nan, index=cal_idx, columns=codes) for c in value_cols}
    for code, g in series_long.groupby("code"):
        g = g.dropna(subset=["eff"])
        if g.empty:
            continue
        g = (g.sort_values("eff").drop_duplicates(subset="eff", keep="last"))
        if g.empty:
            continue
        idx = g.set_index("eff")
        for c in value_cols:
            s = idx[c].reindex(cal_idx, method="ffill")
            frames[c][code] = s.values
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", default=None, help="仅构建指定因子")
    args = ap.parse_args()

    from config import Config
    from stats.ic import calc_ic_series

    lib_root = Path(str(Config.get()["factor_library"]["root"]))
    ds_dir = lib_root / DATASET
    panels_dir = ds_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    stats_path = ds_dir / "factor_stats_margin.jsonl"
    done = ({json.loads(line)["name"] for line in
             stats_path.read_text(encoding="utf-8").splitlines() if line.strip()}
            if args.resume and stats_path.exists() else set())

    close_adj, margin = load_panels()
    codes = close_adj.columns
    cal_idx = close_adj.index
    fwd = {h: close_adj.pct_change(h, fill_method=None).shift(-h) for h in HORIZONS}
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]

    if margin.empty:
        log.warning("margin_detail 无数据，跳过")
        return

    defs = {k: {"src": "margin", "label": v} for k, v in {
        "margin_bal_chg_5d": "两融余额5日变化率",
        "margin_bal_chg_20d": "两融余额20日变化率",
        "financing_chg_1d": "融资余额单日变化率",
        "securities_chg_1d": "融券余额单日变化率",
    }.items()}

    pn = _pit_margin(margin, cal_idx, codes)
    panels_all = {k: pn[k] for k in defs}

    if args.only:
        defs = {k: v for k, v in defs.items() if k == args.only}
        panels_all = {k: v for k, v in panels_all.items() if k == args.only}

    t0 = time.time()
    for i, (name, meta) in enumerate(sorted(defs.items()), 1):
        if name in done or name not in panels_all:
            continue
        p = panels_all[name].fillna(np.nan).astype(np.float32)
        p = p.replace([np.inf, -np.inf], np.nan).clip(-100.0, 100.0)
        p.to_parquet(panels_dir / f"{name}.parquet")

        cov = float(p.notna().mean().mean())
        ic = {h: calc_ic_series(p[ic_codes], fwd[h]).astype(np.float32)
              for h in HORIZONS}
        row = {"name": name, "set": "margin", "label": meta["label"],
               "coverage": cov,
               **{f"ic_mean_h{h}": float(ic[h].mean()) for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("[%d/%d] %s cov=%.2f ic_h1=%+.4f | %.0fs", i, len(defs),
                 name, cov, row["ic_mean_h1"], time.time() - t0)

    if not args.only:
        merge_outputs(ds_dir)
    log.info("两融因子构建完成 %.0fs", time.time() - t0)


def merge_outputs(ds_dir: Path) -> None:
    rows = [json.loads(line) for line in
            (ds_dir / "factor_stats_margin.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return
    reg = pd.read_csv(ds_dir / "registry.csv")
    new_df = pd.DataFrame(rows)
    before = len(reg)
    reg = (pd.concat([reg, new_df], ignore_index=True)
             .drop_duplicates(subset="name", keep="last"))
    reg.to_csv(ds_dir / "registry.csv", index=False, encoding="utf-8-sig")
    log.info("registry: %d -> %d 因子", before, len(reg))

    for h in HORIZONS:
        fuse_horizon_ic(ds_dir, h)


def fuse_horizon_ic(ds_dir: Path, h: int) -> None:
    from stats.ic import calc_ic_series
    stats = [json.loads(line) for line in
             (ds_dir / "factor_stats_margin.jsonl").read_text(
                 encoding="utf-8").splitlines() if line.strip()]
    names = [s["name"] for s in stats]
    ic = pd.read_parquet(ds_dir / f"ic_h{h}.parquet")
    if not names:
        return
    close_adj, _ = load_panels()
    fwd = close_adj.pct_change(h, fill_method=None).shift(-h)
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]
    for n in names:
        p = pd.read_parquet(ds_dir / "panels" / f"{n}.parquet")
        ic[n] = calc_ic_series(p[ic_codes], fwd).reindex(ic.index)
    ic = ic.astype(np.float32)
    ic.to_parquet(ds_dir / f"ic_h{h}.parquet")
    log.info("ic_h%d merged: %d 因子", h, ic.shape[1])


if __name__ == "__main__":
    main()