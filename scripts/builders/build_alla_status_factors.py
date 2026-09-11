"""停牌 / ST状态因子 → 并入 all_a_2018_2026 数据集。

背景（2026-09-11）：history_stock_status 表（date×code 多级索引）长期落盘但
从未构建成因子。其 is_st / is_suspended / 涨跌停界是干净的风险状态标签——
业界共识是这些属"流动性/治理风险"维度，与纯量价正交。本脚本构造停牌与
ST状态因子族，并 append 进 registry.csv 与 ic_h{1,5,10,20}.parquet。

因子清单（set='status'）：
  停牌类：
    suspend_streak_60d : 近 60 交易日内最长连续停牌天数（深度停牌风险）
    suspend_count_60d  : 近 60 交易日停牌次数（异常交易频度）
    suspend_ratio_60d  : 近 60 交易日停牌占比（流动性缺失程度）
  状态类：
    is_st              : 是否 ST/*ST 风险警示（1=风险，0=正常）
    st_days            : 当前已连续处于 ST 状态的天数（治理恶化持续时间）
    limit_pos          : 当日收盘相对涨跌停的价格位置 (close-pre_close)/(high_limited-pre_close)，
                         逼近 1=封涨停热度，逼近 0=跌停承压（量价热度，信息独立于纯涨跌）

口径：
- history_stock_status 是 transaction-date 维度的**当日状态**，天然 PIT（当日可见），
  无前视、无需 ffill；
- 停牌/ST 连续天数与近 60 日统计在**交易日序**上滚动（用 unstack 宽表，跨 code 沿
  date 序累计）；
- 面板 date×code float32，从 2016-07-01 起与现有 ic_h 索引严格对齐；
- IC 用次日/未来 1/5/10/20 日 Spearman（与量价 calc_ic_series 同口径），抽稀取列。

用法:
    python -m scripts.builders.build_alla_status_factors            # 单机
    python -m scripts.builders.build_alla_status_factors --resume   # 断点续跑
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

log = setup_logging("build_alla_status")

DATASET = "all_a_2018_2026"
KEEP_FROM = "2016-07-01"
HORIZONS = (1, 5, 10, 20)
IC_CODE_STRIDE = 3
W60 = 60


def _load_status() -> pd.DataFrame:
    """date×code 多级索引 → 各状态列宽表（date, code 列 + 值列）。"""
    df = pd.read_parquet(Path("e:/data/parquet/history_stock_status.parquet"))
    df = df.reset_index()
    for c in ["date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _pivot(df: pd.DataFrame, col: str, cal_idx, codes) -> pd.DataFrame:
    """从 date/code 长表 pivot 成 date×code 宽表，对齐日历与股票池。"""
    piv = (df.dropna(subset=["date", "code", col])
             .pivot(index="date", columns="code", values=col))
    piv = piv.reindex(index=cal_idx, columns=codes)
    return piv


def _st_days(is_st: pd.DataFrame) -> pd.DataFrame:
    """当前连续 ST 天数（交易日序累计，命中=+1，未命中=归零）。"""
    vals = is_st.values.astype(np.float32)
    out = np.zeros_like(vals)
    for i in range(vals.shape[0]):
        out[i] = (out[i - 1] + 1) * vals[i] if i > 0 else vals[i]
    return pd.DataFrame(out, index=is_st.index, columns=is_st.columns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from config import Config
    from stats.ic import calc_ic_series

    # 交易日历与股票池：取 daily_all_a 的 (date, code) 宽 close_adj
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
    codes = close_adj.columns
    cal_idx = close_adj.index

    lib_root = Path(str(Config.get()["factor_library"]["root"]))
    ds_dir = lib_root / DATASET
    panels_dir = ds_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    stats_path = ds_dir / "factor_stats_status.jsonl"
    done = {json.loads(l)["name"] for l in
            stats_path.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if args.resume and stats_path.exists() else set()

    st = _load_status()

    # ---- 构建各因子宽表 ----
    is_sus = _pivot(st, "is_suspended", cal_idx, codes).fillna(False)
    is_sus_b = is_sus.astype(bool)
    is_st_p = _pivot(st, "is_st", cal_idx, codes)

    # 停牌因子
    suspend_ratio = is_sus_b.rolling(W60, min_periods=5).mean().astype(np.float32)
    suspend_count = is_sus_b.rolling(W60, min_periods=5).sum().astype(np.float32)
    # ST 状态（PIT：当日已知）
    st_flag = is_st_p.fillna(False).astype(np.float32)
    st_enter = _st_days(st_flag)

    # 涨跌停位置：limit_pos(封板热度)。daily 的 close 与 status 的 pre_close/high_limited 合并
    dclose = daily.reset_index().rename(columns={"close": "close"})
    dclose = dclose[["date", "code", "close"]] if "close" in dclose \
        else dclose.rename(columns={"levels_0": "date", "level_1": "code"})
    lim = (st.dropna(subset=["pre_close", "high_limited"])
             .assign(den=lambda d: d["high_limited"] - d["pre_close"]))
    lim = lim[lim["den"] > 0]
    lp = lim.merge(dclose, on=["date", "code"], how="inner")
    lp["limit_pos"] = ((lp["close"] - lp["pre_close"]) / lp["den"]).clip(0, 1)
    limit_pos = _pivot(lp, "limit_pos", cal_idx, codes)

    panels = {
        "suspend_ratio_60d": suspend_ratio.replace(0.0, np.nan),
        "suspend_count_60d": suspend_count.replace(0.0, np.nan),
        "is_st": st_flag.replace(0.0, np.nan),
        "st_days": st_enter.replace(0.0, np.nan),
        "limit_pos": limit_pos,
    }

    labels = {
        "suspend_ratio_60d": "近60日停牌占比",
        "suspend_count_60d": "近60日停牌次数",
        "is_st": "ST风险警示",
        "st_days": "连续ST天数",
        "limit_pos": "封板位置(涨停热度)",
    }

    if not panels:
        log.warning("无可构建因子")
        return

    fwd = {h: close_adj.pct_change(h, fill_method=None).shift(-h)
           for h in HORIZONS}
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]
    order = ["suspend_ratio_60d", "suspend_count_60d", "is_st", "st_days",
             "limit_pos"]

    t0 = time.time()
    for i, name in enumerate(order, 1):
        if name not in panels or name in done:
            continue
        p = panels[name].fillna(np.nan).astype(np.float32)
        p = p.replace([np.inf, -np.inf], np.nan)
        p.to_parquet(panels_dir / f"{name}.parquet")
        cov = float(p.notna().mean().mean())
        ic = {h: calc_ic_series(p[ic_codes], fwd[h]).astype(np.float32)
              for h in HORIZONS}
        row = {"name": name, "set": "status", "label": labels.get(name, name),
               "coverage": cov,
               **{f"ic_mean_h{h}": float(ic[h].mean()) for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("[%d/%d] %s cov=%.2f ic_h1=%+.4f | %.0fs", i, len(order),
                 name, cov, row["ic_mean_h1"], time.time() - t0)

    merge_outputs(ds_dir)
    log.info("停牌/ST因子构建完成 %.0fs", time.time() - t0)


def merge_outputs(ds_dir: Path) -> None:
    sp = ds_dir / "factor_stats_status.jsonl"
    rows = [json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines()
            if l.strip()]
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
    from scripts.builders.build_alla_fundamental_factors import load_panels as _fp
    sp = ds_dir / "factor_stats_status.jsonl"
    stats = [json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    names = [s["name"] for s in stats]
    if not names:
        return
    icp = ds_dir / f"ic_h{h}.parquet"
    ic = pd.read_parquet(icp)
    # 本脚本独立构造 close_adj（与 main 相同），从 ic 现有索引取股票池
    cache_root = Path("e:/data/parquet")
    daily = pd.read_parquet(cache_root / "daily_all_a.parquet")
    daily.index = daily.index.set_levels(daily.index.levels[0].normalize(),
                                         level=0)
    daily = daily[daily.index.get_level_values(0) >= pd.Timestamp(KEEP_FROM)]
    bf = pd.read_parquet(cache_root / "backward_factor.parquet")
    bf.index = bf.index.normalize()
    bf = bf.loc[bf.index >= pd.Timestamp(KEEP_FROM)]
    close_raw = daily["close"].unstack()
    cols = close_raw.columns.intersection(bf.columns)
    close_raw = close_raw.reindex(close_raw.index.intersection(bf.index),
                                  columns=cols)
    bf = bf.reindex(index=close_raw.index, columns=close_raw.columns)
    close_adj = (close_raw * bf).astype(np.float32)
    fwd = close_adj.pct_change(h, fill_method=None).shift(-h)
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]
    for n in names:
        p = pd.read_parquet(ds_dir / "panels" / f"{n}.parquet")
        ic[n] = calc_ic_series(p[ic_codes], fwd).reindex(ic.index)
    ic = ic.astype(np.float32)
    ic.to_parquet(icp)
    log.info("ic_h%d merged: %d 因子", h, ic.shape[1])


if __name__ == "__main__":
    main()