"""大宗折价率 + 股东明细动态因子 → 并入 all_a_2018_2026 数据集。

背景（2026-09-11）：moneyflow 族已有 b_share_amount/volume 的"大宗额/量"因子
（block_amt_20d/block_vol_20d），但缺**折价率**——大宗相对收盘价的折溢价方向，
是机构大额接盘的定价信号，与纯量价正交。holder 族已有截面存量（top10 集中度、
机构占比、户数变化），但缺**跨报告期动态**（前十大股东比例环比、更替率）。

  moneyflow 折价族（set='moneyflow'）：
    block_disc_1d   : 当日大宗加权折价率 = Σ[量权×(收盘-成交)/收盘]，>0 折价(机构低价接盘)
    block_disc_20d  : 过去 20 交易日折价率累积（折价强度趋势）
    block_prem_20d  : 过去 20 交易日溢价率累积（成交>收盘，机构溢价抢筹）

  holder 动态族（set='holder'）：
    top10_hold_chg  : 前十大股东持股比例(前次报告期→本报告期) 变化（吸筹/派发）
    holder_stability: 前十大股东跨报告期更替率（本期在榜∩上期在榜比例，治理稳定度）

口径：
- 折价：block 按 trade_date 与当日收盘价对齐（未复权收盘，同为当下市价）；每股
  当日多条按量权加权；事件窗口聚合取滚动累积。
- 股东动态：按 (code, holder_end_date) 聚合 top10 pct，跨报告期 diff / 更替率；
  PIT 用 ann_date 前向填充，无前视。
- 面板 date×code float32，从 2016-07-01 起与 ic_h 索引对齐；
- IC 用次日/未来 1/5/10/20 日 Spearman，抽稀取列。

用法:
    python -m scripts.builders.build_alla_disc_holder_dyn            # 全量
    python -m scripts.builders.build_alla_disc_holder_dyn --resume   # 续跑
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

log = setup_logging("build_alla_disc_holder_dyn")
from scripts.builders import common  # noqa: E402
from scripts.builders.common import KEEP_FROM, HORIZONS, IC_CODE_STRIDE  # noqa: E402

DATASET = "all_a_2018_2026"
WINDOW = 20


# ---------------------------------------------------------------------------
# PIT / 事件窗口 helper
# ---------------------------------------------------------------------------
def _pit_panel(series_long: pd.DataFrame, cal_idx: pd.DatetimeIndex,
               codes: pd.Index, value_col: str) -> pd.DataFrame:
    """按 eff 前向填充 → date×code 宽表。"""
    frame = pd.DataFrame(np.nan, index=cal_idx, columns=codes)
    for code, g in series_long.groupby("code"):
        g = g.dropna(subset=["eff", value_col])
        if g.empty:
            continue
        g = (g.sort_values("eff").drop_duplicates(subset="eff", keep="last"))
        if g.empty:
            continue
        s = g.set_index("eff")[value_col].reindex(cal_idx, method="ffill")
        frame[code] = s.values
    return frame


def _event_rolling(events: pd.DataFrame, cal_idx, codes,
                   value_col: str, agg: str = "sum") -> dict[str, pd.DataFrame]:
    """事件长表 → 各 code 在交易日历上的滚动窗口聚合（rolling sum）。"""
    e = events.dropna(subset=["code", "eff", value_col]).copy()
    e["eff"] = pd.to_datetime(e["eff"])
    e = e[e["eff"].isin(cal_idx)]
    if e.empty:
        return {value_col: pd.DataFrame(np.nan, index=cal_idx, columns=codes)}
    daily = (e.groupby(["code", "eff"], sort=False)[value_col].sum().reset_index())
    piv = daily.pivot(index="eff", columns="code", values=value_col)
    piv = piv.reindex(index=cal_idx, columns=codes).fillna(0.0)
    roll = piv.rolling(WINDOW, min_periods=1).sum()
    roll = roll.replace(0.0, np.nan)
    return {value_col: roll.astype(np.float32)}


def _to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# ① 大宗折价率
# ---------------------------------------------------------------------------
def _build_block_disc(block: pd.DataFrame, daily_close: pd.DataFrame,
                      cal_idx, codes) -> dict[str, pd.DataFrame]:
    """大宗折价率：事件(成交日)折价率 → 1d PIT / 20d 滚动累积。"""
    if block is None or block.empty:
        return {}
    b = block.copy()
    b["trade_date"] = pd.to_datetime(b["trade_date"], errors="coerce")
    b = _to_num(b, ["b_share_price", "b_share_amount", "b_share_volume"])
    b = b.dropna(subset=["code", "trade_date", "b_share_price"])
    b["eff"] = b["trade_date"]
    b = b[b["eff"].isin(cal_idx)]

    # 当日收盘价（未复权，date×code 宽表 → 摊平长表查表）
    close_stack = daily_close.stack().rename("close").reset_index()
    close_stack.columns = ["date", "code", "close"]
    m = b.merge(close_stack, left_on=["eff", "code"],
                right_on=["date", "code"], how="inner")
    m = m.dropna(subset=["close"])
    m = m[m["close"] > 0]
    m["disc"] = (m["close"] - m["b_share_price"]) / m["close"]  # >0 折价

    # 每股当日多条 → 量权(amount)加权折价
    m["wamt"] = m["b_share_amount"].replace(0.0, np.nan)
    m["_discw"] = m["disc"] * m["wamt"]
    daily = (m.groupby(["code", "eff"], sort=False)
              .agg(sum_w=("wamt", "sum"), sum_dw=("_discw", "sum"))
              .reset_index())
    daily["disc_1d"] = (daily["sum_dw"] / daily["sum_w"].replace(0.0, np.nan))

    out_long = pd.DataFrame({"code": daily["code"], "eff": daily["eff"],
                             "block_disc_1d": daily["disc_1d"]})
    out = {"block_disc_1d": _pit_panel(out_long, cal_idx, codes,
                                       "block_disc_1d").clip(-0.5, 0.5)}

    # 20日折价累积（拉开同方向强度）与 20日溢价累积（成交高于收盘的抢筹）
    disc20 = daily.rename(columns={"disc_1d": "block_disc_20d"})
    out.update(_event_rolling(disc20, cal_idx, codes, "block_disc_20d"))
    prem = daily.copy()
    prem["block_prem_20d"] = (-daily["disc_1d"]).clip(lower=0.0)
    out.update(_event_rolling(prem, cal_idx, codes, "block_prem_20d"))
    return out


# ---------------------------------------------------------------------------
# ② 股东明细动态
# ---------------------------------------------------------------------------
def _build_holder_dyn(share_holder: pd.DataFrame, cal_idx,
                      codes) -> dict[str, pd.DataFrame]:
    """股东明细动态：前十大持股环变化 + 更替率（PIT 向量化）。"""
    if share_holder is None or share_holder.empty:
        return {}
    sh = share_holder.copy()
    for c in ("ann_date", "holder_end_date"):
        sh[c] = pd.to_datetime(sh[c], errors="coerce")
    sh = _to_num(sh, ["holder_pct"])
    sh = sh.dropna(subset=["code", "holder_end_date", "holder_pct"])
    sh["pct"] = sh["holder_pct"]
    sh = sh.sort_values(["code", "holder_end_date", "pct"],
                        ascending=[True, True, False])

    # 每股每报告期：组内 rank 取前10，聚合 top10 比例/名称/ann_date
    sh["_rk"] = sh.groupby(["code", "holder_end_date"])["pct"].rank(
        method="first", ascending=False)
    top10 = sh[sh["_rk"] <= 10]
    top10_sum = top10.groupby(["code", "holder_end_date"], sort=False)["pct"].sum()
    names_map = top10.groupby(["code", "holder_end_date"], sort=False)[
        "holder_name"].agg(lambda s: set(s))
    ann = sh.groupby(["code", "holder_end_date"], sort=False)["ann_date"].max()

    agg = pd.DataFrame({
        "ann": ann.reindex(top10_sum.index),
        "top10": top10_sum,
        "names": names_map.reindex(top10_sum.index),
    }).reset_index()
    agg = agg.dropna(subset=["ann", "top10"])
    agg = agg.sort_values(["code", "holder_end_date"])

    # 环比变化 & 更替率（按每股报告期序列）
    agg["top10_chg"] = agg.groupby("code")["top10"].diff()
    agg["names_prev"] = agg.groupby("code", sort=False)["names"].shift(1)

    def _stability(cur, prev):
        if not isinstance(cur, set) or not isinstance(prev, set) \
                or not cur or cur is None or prev is None:
            return np.nan
        inter = len(cur & prev)
        return inter / max(len(cur), 1)

    agg["holder_stability"] = [
        _stability(c, p) for c, p in zip(agg["names"], agg["names_prev"])]

    out_long = pd.DataFrame({
        "code": agg["code"], "eff": pd.to_datetime(agg["ann"]),
        "top10_hold_chg": agg["top10_chg"],
        "holder_stability": agg["holder_stability"],
    })
    out = _pit_long(out_long, cal_idx, codes,
                    ["top10_hold_chg", "holder_stability"])
    for k, v in out.items():
        out[k] = v.clip(-100.0, 100.0)
    return out


def _pit_long(series_long, cal_idx, codes, value_cols) -> dict[str, pd.DataFrame]:
    frames = {c: pd.DataFrame(np.nan, index=cal_idx, columns=codes)
              for c in value_cols}
    for code, g in series_long.groupby("code"):
        g = g.dropna(subset=["eff"])
        if g.empty:
            continue
        g = (g.sort_values("eff").drop_duplicates(subset="eff", keep="last"))
        if g.empty:
            continue
        idx = g.set_index("eff")
        for c in value_cols:
            frames[c][code] = idx[c].reindex(cal_idx, method="ffill").values
    return frames


# ---------------------------------------------------------------------------
# 数据 & 主流程
# ---------------------------------------------------------------------------
def load_panels():
    from config import Config
    cache_root = Path(str(Config.cache()["root"]))
    daily = pd.read_parquet(cache_root / "daily_all_a.parquet")
    daily.index = daily.index.set_levels(daily.index.levels[0].normalize(),
                                         level=0)
    k0 = pd.Timestamp(KEEP_FROM)
    daily = daily[daily.index.get_level_values(0) >= k0]
    bf = pd.read_parquet(cache_root / "backward_factor.parquet")
    bf.index = bf.index.normalize()
    bf = bf.loc[bf.index >= k0]
    close_raw = daily["close"].unstack()
    cols = close_raw.columns.intersection(bf.columns)
    close_raw = close_raw.reindex(close_raw.index.intersection(bf.index),
                                  columns=cols)
    bf = bf.reindex(index=close_raw.index, columns=close_raw.columns)
    close_adj = (close_raw * bf).astype(np.float32)

    tabs = {
        "block": (pd.read_parquet(cache_root / "block_trading.parquet")
                  if (cache_root / "block_trading.parquet").exists() else None),
        "holder": (pd.read_parquet(cache_root / "share_holder.parquet")
                   if (cache_root / "share_holder.parquet").exists() else None),
    }
    return close_adj, close_raw, tabs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from config import Config
    from stats.ic import calc_ic_series

    lib_root = Path(str(Config.get()["factor_library"]["root"]))
    ds_dir = lib_root / DATASET
    panels_dir = ds_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    stats_path = ds_dir / "factor_stats_disc_holder_dyn.jsonl"
    done = {json.loads(l)["name"] for l in
            stats_path.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if args.resume and stats_path.exists() else set()

    close_adj, close_raw, tabs = load_panels()
    codes = close_adj.columns
    cal_idx = close_adj.index
    fwd = {h: close_adj.pct_change(h, fill_method=None).shift(-h)
           for h in HORIZONS}
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]

    panels: dict[str, pd.DataFrame] = {}
    panels.update(_build_block_disc(tabs.get("block"), close_raw, cal_idx, codes))
    panels.update(_build_holder_dyn(tabs.get("holder"), cal_idx, codes))

    labels = {
        "block_disc_1d": "大宗当日加权折价率",
        "block_disc_20d": "大宗20日折价率累积",
        "block_prem_20d": "大宗20日溢价率累积",
        "top10_hold_chg": "前十大股东持股比环比",
        "holder_stability": "前十大股东更替率",
    }
    sets = {("block_", "moneyflow"), ("top10_", "holder"), ("holder_", "holder")}

    if not panels:
        log.warning("无可构建因子，跳过")
        return

    t0 = time.time()
    for i, name in enumerate(sorted(panels), 1):
        if name in done:
            continue
        p = panels[name].fillna(np.nan).astype(np.float32)
        p = p.replace([np.inf, -np.inf], np.nan)
        p.to_parquet(panels_dir / f"{name}.parquet")
        cov = float(p.notna().mean().mean())
        ic = {h: calc_ic_series(p[ic_codes], fwd[h]).astype(np.float32)
              for h in HORIZONS}
        family = next((s for pre, s in sets if name.startswith(pre)), "fundamental")
        row = {"name": name, "set": family, "label": labels.get(name, name),
               "coverage": cov,
               **{f"ic_mean_h{h}": float(ic[h].mean()) for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("[%d/%d] %s cov=%.2f ic_h1=%+.4f | %.0fs", i, len(panels),
                 name, cov, row["ic_mean_h1"], time.time() - t0)

    common.merge_outputs(ds_dir, 'disc_holder_dyn')
    log.info("折价+股东动态因子构建完成 %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()