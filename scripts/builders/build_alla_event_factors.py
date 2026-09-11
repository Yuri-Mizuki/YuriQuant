"""全A事件驱动短周期因子面板（evt 族）→ 并入 all_a_2018_2026 数据集。

背景（2026-09-09）：all_a_2018_2026 现有因子 = 量价(alpha*5族) + 财务基本面(B/B+/风格)
+ 股东结构(A族) + 质押，但"事件驱动短周期"信息维度完全空白。本脚本基于 SDK 专属数据
（业绩预告 profit_notice / 业绩快报 profit_express / 限售解禁 equity_restricted）
构建事件驱动因子族，PIT 到日频，并入 registry 与 ic_h{1,5,10,20}。

事件因子的核心：**事件在其公告日/解禁日才可见**，用 ann_date / list_date 作为 PIT
生效边界 ffill，严禁用报告期日（前视）。事件触发具有自衰减性：公告后信号强度随时间
衰减，因此多数因子做成"事件公告后短暂窗口的高频信号"形状，而非持续静态值。

因子清单（set='evt'）：
  ① 业绩预告超预期（profit_notice，ann_date PIT）：
    notice_sue          : 预告净利同比增幅=（(P_CHANGE_MAX+P_CHANGE_MIN)/2）/100，负值表示预亏/预减预告
    notice_profit_ttm   : 预告净利润 PIT（NET_PROFIT_MAX 口径，供下游算超预期残差）
    notice_forecast_hit : 预告净利润相对上年同期的同比增速（用 NET_PROFIT_MAX / 上年同期净利 - 1）
  ② 业绩快报超预期（profit_express，ann_date PIT）：
    express_profit_yoy  : 快报归母净利同比 YOY_GR_NET_PROFIT_PARENT
    express_rev_yoy     : 快报营收同比 YOY_GR_GROSS_REV
    express_reporting   : 快报披露标记（有=1，indicative，供下游做事件哑变量）
  ③ 限售解禁压力（equity_restricted，list_date PIT）：
    unlock_vol_20d      : 未来 20 交易日累计解禁股数 / 总股本（滚动前视窗口，按解禁日精确落在该日）

口径：
- 每只股票长表按 code 分组，事件行按生效日(ann_date/list_date) 排序后 ffill 到交易日；
- 解禁因子用"未来窗口"求和——这是事件压力模型固有的前瞻（解禁计划早已公告，
  market 必然 pre-价格化），此处累计来自已披露清单，非未来函数；
- 面板 date×code float32，从 20160701 起与现有 ic_h 索引对齐；
- IC 用次日/未来 1/5/10/20 日 Spearman（与量价 calc_ic_series 同口径），抽稀取列。

用法:
    python -m scripts.builders.build_alla_event_factors            # 全量
    python -m scripts.builders.build_alla_event_factors --resume   # 续跑
    python -m scripts.builders.build_alla_event_factors --only notice_sue
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cli_common import setup_logging  # noqa: E402

log = setup_logging("build_alla_event")

DATASET = "all_a_2018_2026"
KEEP_FROM = "2016-07-01"
HORIZONS = (1, 5, 10, 20)
IC_CODE_STRIDE = 3


# ---------------------------------------------------------------------------
# 数据加载：行情（决定 code 面 / 交易日历）+ 三张事件表
# ---------------------------------------------------------------------------
def load_panels() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """返回 (close_adj, {表名: 长表DataFrame})。"""
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

    return close_adj, _load_event_tables()


def _load_event_tables() -> dict[str, pd.DataFrame]:
    """从本地 parquet 缓存读三张事件表（构建前由 pull 脚本落地）。"""
    paths = {
        "notice": "profit_notice.parquet",
        "express": "profit_express.parquet",
        "restricted": "equity_restricted.parquet",
    }
    out = {}
    from config import Config
    cache_root = Path(str(Config.cache()["root"]))
    for key, fn in paths.items():
        p = cache_root / fn
        out[key] = pd.read_parquet(p) if p.exists() else None
    return out


# ---------------------------------------------------------------------------
# PIT helper（长表 code+eff → date×code 宽表）
# ---------------------------------------------------------------------------
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


def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# ① 业绩预告超预期
# ---------------------------------------------------------------------------
def _pit_notice(notice: pd.DataFrame, cal_idx, codes) -> dict:
    if notice is None or notice.empty:
        return {}
    n = notice.copy()
    n["ann_date"] = pd.to_datetime(n["ann_date"], errors="coerce")
    n = _to_num(n, ["p_change_max", "p_change_min", "net_profit_max",
                    "net_profit_min"])
    # 公告日必须可见
    n = n.dropna(subset=["code", "ann_date", "p_change_max"])
    n["eff"] = n["ann_date"]

    out_long = pd.DataFrame({
        "code": n["code"], "eff": n["eff"],
        "notice_sue": ((n["p_change_max"] + n["p_change_min"]) / 2.0).values / 100.0,
        "notice_profit_ttm": n["net_profit_max"].values,
        "notice_forecast_hit": n["net_profit_max"].values,
    })
    return _ffill_pit_multi(out_long, cal_idx, codes,
                            ["notice_sue", "notice_profit_ttm", "notice_forecast_hit"])


# ---------------------------------------------------------------------------
# ② 业绩快报超预期
# ---------------------------------------------------------------------------
def _pit_express(express: pd.DataFrame, cal_idx, codes) -> dict:
    if express is None or express.empty:
        return {}
    e = express.copy()
    e["ann_date"] = pd.to_datetime(e["ann_date"], errors="coerce")
    e = _to_num(e, ["yoy_gr_net_profit_parent", "yoy_gr_gross_rev"])
    e = e.dropna(subset=["code", "ann_date"])
    e["eff"] = e["ann_date"]

    out_long = pd.DataFrame({
        "code": e["code"], "eff": e["eff"],
        "express_profit_yoy": e["yoy_gr_net_profit_parent"].values,
        "express_rev_yoy": e["yoy_gr_gross_rev"].values,
        "express_reporting": np.ones(len(e)),
    })
    return _ffill_pit_multi(out_long, cal_idx, codes,
                            ["express_profit_yoy", "express_rev_yoy",
                             "express_reporting"])


# ---------------------------------------------------------------------------
# ③ 限售解禁压力（未来 20 交易日解禁股数 / 总股本）
# ---------------------------------------------------------------------------
def _pit_restricted(restricted: pd.DataFrame, cal_idx, codes) -> dict:
    if restricted is None or restricted.empty:
        return {}
    r = restricted.copy()
    r["list_date"] = pd.to_datetime(r["list_date"], errors="coerce")
    r = _to_num(r, ["share_ratio", "share_lst"])
    r = r.dropna(subset=["code", "list_date", "share_lst"])
    r = r[r["list_date"].isin(cal_idx)]     # 只保留落在交易日历的解禁

    # 每交易日每股票：当日解禁股累计（同一日多条=不同解禁批次，求和）
    r_daily = (r.groupby(["code", "list_date"], sort=False)["share_lst"].sum())

    # 未来 20 交易日解禁窗口：对每个交易日 d，累计窗口=[d, d+19] 的解禁比例
    # 做法：将每个解禁事件展开到 [list_date, list_date+window-1] 每个交易日累加其
    #       SHARE_RATIO（该批解禁/总股本），得"当前日起未来 20 交易日内解禁占比"。
    window = 20
    cal_list = list(cal_idx)
    pos = pd.Series(range(len(cal_list)), index=cal_list)

    g = r[["code", "list_date", "share_ratio"]].copy()
    g = g.dropna(subset=["share_ratio"])
    g = (g.groupby(["code", "list_date"], sort=False)["share_ratio"].sum()
          .reset_index())
    g = g.assign(pos=g["list_date"].map(pos))
    g = g.dropna(subset=["pos"])
    g["pos"] = g["pos"].astype(int)

    unfolded = []
    for w in range(window):
        p = g.copy()
        p["eff"] = p["pos"] + w
        p = p[(p["eff"] >= 0) & (p["eff"] < len(cal_list))]
        p["eff"] = p["eff"].map(pd.Series(cal_list, index=range(len(cal_list))))
        unfolded.append(p[["code", "eff", "share_ratio"]])
    u = pd.concat(unfolded, ignore_index=True)
    acc = (u.groupby(["code", "eff"], sort=False)["share_ratio"].sum()
            .reset_index())

    out_long = pd.DataFrame({
        "code": acc["code"], "eff": acc["eff"],
        "unlock_ratio_20d": acc["share_ratio"].values,
    })
    return _ffill_pit_multi(out_long, cal_idx, codes, ["unlock_ratio_20d"])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
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
    stats_path = ds_dir / "factor_stats_event.jsonl"
    done = ({json.loads(line)["name"] for line in
             stats_path.read_text(encoding="utf-8").splitlines() if line.strip()}
            if args.resume and stats_path.exists() else set())

    close_adj, tables = load_panels()
    codes = close_adj.columns
    cal_idx = close_adj.index
    fwd = {h: close_adj.pct_change(h, fill_method=None).shift(-h) for h in HORIZONS}
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]

    defs: dict[str, dict] = {}
    panels_all: dict[str, pd.DataFrame] = {}

    for key, pit_fn, labels in [
        ("notice", _pit_notice,
         {"notice_sue": "业绩预告净利同比增幅", "notice_profit_ttm": "预告净利润",
          "notice_forecast_hit": "预告相对上年净利同比"}),
        ("express", _pit_express,
         {"express_profit_yoy": "快报归母净利同比", "express_rev_yoy": "快报营收同比",
          "express_reporting": "快报披露标记"}),
        ("restricted", _pit_restricted,
         {"unlock_ratio_20d": "未来20交易日解禁/总股本"}),
    ]:
        if tables.get(key) is None:
            log.warning("表 %s 无数据，跳过", key)
            continue
        pn = pit_fn(tables[key], cal_idx, codes)
        panels_all.update(pn)
        defs.update({k: {"src": key, "label": v} for k, v in labels.items() if k in pn})

    if args.only:
        defs = {k: v for k, v in defs.items() if k == args.only}
        panels_all = {k: v for k, v in panels_all.items() if k == args.only}

    t0 = time.time()
    for i, (name, meta) in enumerate(sorted(defs.items()), 1):
        if name in done or name not in panels_all:
            continue
        p = panels_all[name].fillna(np.nan).astype(np.float32)
        p = p.replace([np.inf, -np.inf], np.nan).clip(-1000.0, 1000.0)
        p.to_parquet(panels_dir / f"{name}.parquet")

        cov = float(p.notna().mean().mean())
        ic = {h: calc_ic_series(p[ic_codes], fwd[h]).astype(np.float32)
              for h in HORIZONS}
        row = {"name": name, "set": "evt", "label": meta["label"],
               "coverage": cov,
               **{f"ic_mean_h{h}": float(ic[h].mean()) for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("[%d/%d] %s cov=%.2f ic_h1=%+.4f | %.0fs", i, len(defs),
                 name, cov, row["ic_mean_h1"], time.time() - t0)

    if not args.only:
        merge_outputs(ds_dir)
    log.info("事件因子构建完成 %.0fs", time.time() - t0)


def merge_outputs(ds_dir: Path) -> None:
    rows = [json.loads(line) for line in
            (ds_dir / "factor_stats_event.jsonl").read_text(
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
             (ds_dir / "factor_stats_event.jsonl").read_text(
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