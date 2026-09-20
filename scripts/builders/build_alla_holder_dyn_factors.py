"""全A 股东/内部人**动态持股**因子面板（holder_dyn 族）→ 并入 all_a_2018_2026。
================================================================================

背景（2026-09-20，另类数据 P0 首轮落地）
----------------------------------------
all_a_2018_2026 已有的事件族（``evt``）只覆盖业绩预告 / 业绩快报 / 限售解禁；
``holder`` 族是**静态结构**（股东户数 / 十大股东，``build_alla_holder_factors``）。
**「谁在买、谁在卖」这一支完全空白** —— 本脚本补的正是它。

数据源（全部免自建，见 ``data/altdata/``）
------------------------------------------
======================  =============================  ==============  ===========
表                       内容                            粒度            实测历史
======================  =============================  ==============  ===========
``mgmt_hold``            高管/董监高增减持（巨潮）        事件（人-日）   滚动 1 年
``inner_trade``          内部人交易（雪球）              事件（人-日）   20.5 月
``holder_control``       实际控制人持股变动（巨潮）      事件（控制人）  2010-12 起
======================  =============================  ==============  ===========

🚨 PIT 红线（三张表质量**不同**，逐表施加滞后，见 ``source_akshare.PIT_QUALITY``）
----------------------------------------------------------------------------------
- ``mgmt_hold``：有 ``公告日期`` ⇒ **直接用公告日**，滞后 0；
- ``inner_trade``：只有 ``变动日期`` ⇒ 法定披露期限 T+2 ⇒ **滞后 2 交易日**；
- ``holder_control``：只有 ``变动日期``（定期报告/权益变动书口径）⇒ 披露可晚
  数月 ⇒ **滞后 60 交易日**（宁可错杀不可前视；见 ``SUGGESTED_LAG_TRADING_DAYS``）。

**没有公告日的表绝不按变动日直接对齐** —— 那是前视，会让回测虚高。

口径
----
- 事件→日频：每个事件贡献到 ``[eff, eff+window-1]`` 的每个交易日，等价于
  「在日 d 回看 window 天内是否覆盖该事件」；``eff = 可得日 + 滞后``；
- **一律用股数口径**（``chg_qty`` 100% 非空），除以**流通股本**归一化 ——
  实测巨潮 ``期末市值`` 整列全空、``成交均价`` 仅 52% 非空，**金额口径会丢一半样本**；
- 未上市（流通股本 NaN）期间一律 NaN，不填 0；
- 面板 date×code float32，从 20160701 起与现有 ic_h 索引对齐。

用法:
    python -m scripts.builders.build_alla_holder_dyn_factors            # 全量
    python -m scripts.builders.build_alla_holder_dyn_factors --only mgmt_netbuy_60d
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

log = setup_logging("build_alla_holder_dyn")
from scripts.builders import common  # noqa: E402
from scripts.builders.common import HORIZONS, IC_CODE_STRIDE  # noqa: E402

DATASET = "all_a_2018_2026"
FAMILY = "holder_dyn"


# ---------------------------------------------------------------------------
# 输入：行情（日历/股票池）+ 流通股本 + 三张另类数据表
# ---------------------------------------------------------------------------
def load_close_adj():
    close_adj = common.load_close_adj()
    return close_adj


def load_float_shares(cal_idx, codes) -> pd.DataFrame:
    """日频流通股本面板（单位：股，上市前 NaN）。"""
    from config import Config
    from data.market_cap import build_shares_panel

    root = Path(str(Config.cache()["root"]))
    p = root / "equity_structure.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"缺少 {p} —— 流通股本面板是归一化分母，不能缺。"
            "先跑 scripts/builders 的股本回补。")
    es = pd.read_parquet(p)
    return build_shares_panel(es, cal_idx, codes, share_field="float_share")


def load_alt_tables() -> dict[str, pd.DataFrame]:
    """读 ``data/altdata`` 的缓存表（缺失返回空表，由调用方记 warning）。

    🚨 **``mgmt`` 键优先取自建的巨潮全历史表**（``alt_cninfo_holder.parquet``，
    2010 起），仅在它缺失时退回 akshare 的 1 年滚动表（``alt_mgmt_hold.parquet``）。

    为什么必须切源（2026-09-20）：OOS 窗口是 **2018–2026**，而 akshare 封装
    只能给**滚动 1 年** ⇒ 用它建的 ``mgmt_*`` 因子在 7/8 的年份是 NaN，
    基于它做的 IC / 消融实际只覆盖最近一年，**Δ 会被误读成「因子无效」**。
    切到全历史后覆盖率从 ≈12% 提到 ≈100%，这一步不是优化而是**前提修正**。
    """
    from config import Config

    root = Path(str(Config.cache()["root"]))
    out: dict[str, pd.DataFrame] = {}
    for key, fn in (("mgmt", "alt_cninfo_holder.parquet"),
                    ("mgmt_fallback", "alt_mgmt_hold.parquet"),
                    ("inner_trade", "alt_inner_trade.parquet"),
                    ("control", "alt_holder_control.parquet")):
        p = root / fn
        out[key] = pd.read_parquet(p) if p.exists() else pd.DataFrame()

    if not out["mgmt"].empty:
        log.info("  mgmt 源 = 巨潮全历史（alt_cninfo_holder）%d 行", len(out["mgmt"]))
    elif not out["mgmt_fallback"].empty:
        log.warning("  mgmt 源 = akshare 1 年滚动（alt_mgmt_hold）%d 行 —— "
                    "全历史表缺失，因子覆盖率会严重受限", len(out["mgmt_fallback"]))
        out["mgmt"] = out["mgmt_fallback"]
    for k in ("mgmt", "inner_trade", "control"):
        if out[k].empty:
            log.warning("缺少/%为空: %s（先跑 scripts/pipelines/fetch_altdata_daily.py）", k)
    return out


# ---------------------------------------------------------------------------
# PIT helper：事件长表 → 滚动窗口面板
# ---------------------------------------------------------------------------
def ann_to_datetime(s: pd.Series) -> pd.Series:
    """``ann_date`` → ``datetime64``，**对三种 dtype 都成立**。

    🚨 为什么必须有这个函数（2026-09-20 实测事故）：
    各数据源的 ``ann_date`` **dtype 不统一** —— akshare 系给 ``datetime64``，
    而本项目自建的 ``cninfo_holder`` 按口径要求给 **int YYYYMMDD**。
    直接 ``pd.DatetimeIndex(ev["ann_date"])`` 遇到 int 会把它当 **纳秒时间戳**
    （项目既有 PIT 铁律），于是 26 万条事件全部落到 1970 → ``searchsorted``
    一律返回 0 → **所有事件被塞进日历第一天**，产出一个人造尖峰面板。
    **全程不抛异常**，只在覆盖率指标上留下一个不起眼的数字。

    ⇒ 任何把 ``ann_date`` 变成时间轴的地方**都必须过这个函数**。
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s.astype(str).str.replace("-", "", regex=False).str[:8],
                          format="%Y%m%d", errors="coerce")


def _apply_lag(events: pd.DataFrame, lag_days: int, cal_idx) -> pd.DataFrame:
    """把可得日 ``ann_date`` 推迟 ``lag_days`` 个**交易日**作为生效日 ``eff``。

    lag_days=0 时 eff = ann_date 当日起的第一个交易日（``side="left"``）。

    ⚠️ **日历起点之前的事件必须丢弃，不能钳到第一天**：
    ``searchsorted`` 对「早于首元素」与「等于首元素」都返回 0，二者无法区分。
    若只判 ``pos < len(cal_idx)``（老实现），所有样本外事件会被**堆到第一天**，
    制造一个跨截面的假信号（实测 5 万条事件堆在 2016-07-01，max 面板值 47.8）。
    """
    if events is None or events.empty:
        return pd.DataFrame(columns=["code", "eff"])
    ev = events.dropna(subset=["code", "ann_date"]).copy()
    if ev.empty:
        return pd.DataFrame(columns=["code", "eff"])
    d = ann_to_datetime(ev["ann_date"])
    pos = cal_idx.searchsorted(pd.DatetimeIndex(d), side="left") + int(lag_days)
    ok = (d >= cal_idx[0]).to_numpy() & (pos < len(cal_idx))
    n_drop = int((~ok).sum())
    if n_drop:
        log.info("  _apply_lag: 丢弃 %d 条（早于日历起点 %s 或滞后越界）",
                 n_drop, cal_idx[0].date())
    ev = ev[ok].copy()
    if ev.empty:
        return pd.DataFrame(columns=["code", "eff"])
    ev["eff"] = cal_idx[np.asarray(pos[ok], dtype=int)]
    return ev


def _rolling_sum_panel(events: pd.DataFrame, value_col: str, window: int,
                       cal_idx, codes, mask: pd.DataFrame) -> pd.DataFrame:
    """事件长表 → 日频滚动窗口累计面板（date×code）。

    每个事件贡献到 ``[eff, eff+window-1]`` 的每个交易日 —— 对日 d 而言，
    这等价于「d 回看 window 个交易日内发生的全部事件」。

    ``mask``：仅在 mask 非 NaN 处给值（未上市期间不填 0）。
    """
    out = pd.DataFrame(np.nan, index=cal_idx, columns=codes, dtype=np.float32)
    if events is None or events.empty or value_col not in events.columns:
        return out
    ev = events.dropna(subset=["code", "eff", value_col])
    if ev.empty:
        return out
    ev = ev[ev["code"].isin(codes)]
    if ev.empty:
        return out
    g = ev.groupby(["eff", "code"], sort=False)[value_col].sum().unstack()
    g = g.reindex(index=cal_idx)          # 事件日→交易日历（非交易日已由 _apply_lag 归位）
    g = g.reindex(columns=codes)
    g = g.fillna(0.0).astype(np.float32)
    r = g.rolling(window=window, min_periods=1).sum()
    r = r.where(mask.notna())             # 未上市/退市后 → NaN
    return r.astype(np.float32)


def _level_panel(events: pd.DataFrame, value_col: str, cal_idx, codes,
                 mask: pd.DataFrame) -> pd.DataFrame:
    """事件长表 → 日频**水平量**面板（按 eff 前向填充）。

    用于实控人「控股比例」这类只在新事件时刷新的状态量。
    """
    out = pd.DataFrame(np.nan, index=cal_idx, columns=codes, dtype=np.float32)
    if events is None or events.empty or value_col not in events.columns:
        return out
    ev = events.dropna(subset=["code", "eff", value_col])
    ev = ev[ev["code"].isin(codes)]
    if ev.empty:
        return out
    g = (ev.sort_values("eff").drop_duplicates(subset=["code", "eff"], keep="last")
           .pivot(index="eff", columns="code", values=value_col))
    g = g.reindex(index=cal_idx.union(g.index)).sort_index().ffill()
    g = g.reindex(index=cal_idx, columns=codes)
    return g.where(mask.notna()).astype(np.float32)


# ---------------------------------------------------------------------------
# 各表 → 因子
# ---------------------------------------------------------------------------
#: 因子集（set=FAMILY）路由：表名 → 构建函数
def build_panels(cal_idx, codes, shares: pd.DataFrame,
                 tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """在给定日历/股票池上构建 holder_dyn 族面板。"""
    from data.altdata.source_akshare import SUGGESTED_LAG_TRADING_DAYS as LAG

    out: dict[str, pd.DataFrame] = {}
    mask = shares

    # --- ① 高管/董监高增减持（有公告日 ⇒ lag 0）---
    mh = tables.get("mgmt")
    if mh is not None and not mh.empty:
        ev = _apply_lag(mh, LAG["mgmt_hold"], cal_idx)
        # 带符号股数（增持正/减持负）与事件方向
        ev = ev.assign(
            signed_qty=pd.to_numeric(mh.loc[ev.index, "signed_qty"], errors="coerce").values,
            chg_qty=pd.to_numeric(mh.loc[ev.index, "chg_qty"], errors="coerce").values,
            direction=mh.loc[ev.index, "direction"].values)
        ev["evt_sign"] = np.where(ev["direction"].astype(str) == "增持", 1.0, -1.0)
        for w in (20, 60):
            net = _rolling_sum_panel(ev, "signed_qty", w, cal_idx, codes, mask)
            out[f"mgmt_netbuy_{w}d"] = (net / shares).astype(np.float32)
        out["mgmt_netbuy_cnt_20d"] = _rolling_sum_panel(
            ev, "evt_sign", 20, cal_idx, codes, mask)
        # 减持强度单列（增持与减持的信息含义不对称，不合并成净额）
        sell = ev[ev["direction"].astype(str) == "减持"].copy()
        sell_qty = _rolling_sum_panel(sell, "chg_qty", 60, cal_idx, codes, mask)
        out["mgmt_sell_ratio_60d"] = (sell_qty / shares).astype(np.float32)
        span = (f"{ann_to_datetime(ev['ann_date']).min().date()}"
                f"~{ann_to_datetime(ev['ann_date']).max().date()}")
        log.info("  mgmt: %d 事件（公告日 %s）→ 5 因子", len(ev), span)
    else:
        log.warning("  mgmt 无数据，跳过其 5 个因子")

    # --- ② 内部人交易（无公告日 ⇒ lag 2 交易日）---
    it = tables.get("inner_trade")
    if it is not None and not it.empty:
        ev = _apply_lag(it, LAG["inner_trade"], cal_idx)
        ev = ev.assign(chg_qty=pd.to_numeric(it.loc[ev.index, "chg_qty"], errors="coerce").values)
        ev["evt_sign"] = np.sign(ev["chg_qty"].fillna(0.0))
        for w in (20, 60):
            net = _rolling_sum_panel(ev, "chg_qty", w, cal_idx, codes, mask)
            out[f"inner_netbuy_{w}d"] = (net / shares).astype(np.float32)
        out["inner_netbuy_cnt_60d"] = _rolling_sum_panel(
            ev, "evt_sign", 60, cal_idx, codes, mask)
        log.info("  inner_trade: %d 事件 → 3 因子", len(ev))
    else:
        log.warning("  inner_trade 无数据，跳过其 3 个因子")

    # --- ③ 实际控制人变动（无公告日 ⇒ lag 60 交易日）---
    hc = tables.get("control")
    if hc is not None and not hc.empty:
        ev = _apply_lag(hc, LAG["holder_control"], cal_idx)
        ev = ev.assign(
            hold_ratio=pd.to_numeric(hc.loc[ev.index, "hold_ratio"], errors="coerce").values,
            one=1.0)
        out["ctrl_hold_ratio"] = _level_panel(ev, "hold_ratio", cal_idx, codes, mask)
        out["ctrl_change_cnt_60d"] = _rolling_sum_panel(
            ev, "one", 60, cal_idx, codes, mask)
        log.info("  holder_control: %d 事件 → 2 因子", len(ev))
    else:
        log.warning("  control 无数据，跳过其 2 个因子")

    return out


#: 因子名 → 中文标签（写入 registry）
LABELS: dict[str, str] = {
    "mgmt_netbuy_20d": "高管净增持/流通股本·20日",
    "mgmt_netbuy_60d": "高管净增持/流通股本·60日",
    "mgmt_netbuy_cnt_20d": "高管净增持事件数·20日",
    "mgmt_sell_ratio_60d": "高管减持/流通股本·60日",
    "inner_netbuy_20d": "内部人净买入/流通股本·20日",
    "inner_netbuy_60d": "内部人净买入/流通股本·60日",
    "inner_netbuy_cnt_60d": "内部人净买入事件数·60日",
    "ctrl_hold_ratio": "实际控制人控股比例",
    "ctrl_change_cnt_60d": "实际控制人变动次数·60日",
}


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
    stats_path = ds_dir / f"factor_stats_{FAMILY}.jsonl"
    done = ({json.loads(line)["name"] for line in
             stats_path.read_text(encoding="utf-8").splitlines() if line.strip()}
            if args.resume and stats_path.exists() else set())

    close_adj = load_close_adj()
    codes = close_adj.columns
    cal_idx = close_adj.index
    shares = load_float_shares(cal_idx, codes)
    tables = load_alt_tables()
    fwd = {h: close_adj.pct_change(h, fill_method=None).shift(-h) for h in HORIZONS}
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]

    log.info("日历 %d 日 / %s ~ %s；股票池 %d 只",
             len(cal_idx), cal_idx[0].date(), cal_idx[-1].date(), len(codes))
    log.info("流通股本非空率 %.1f%%", shares.notna().mean().mean() * 100)

    t0 = time.time()
    panels_all = build_panels(cal_idx, codes, shares, tables)
    defs = {k: v for k, v in LABELS.items() if k in panels_all}
    if args.only:
        defs = {k: v for k, v in defs.items() if k == args.only}
        panels_all = {k: v for k, v in panels_all.items() if k == args.only}
    log.info("构建 %d 个因子（%.0fs）", len(defs), time.time() - t0)

    for i, (name, label) in enumerate(sorted(defs.items()), 1):
        if name in done or name not in panels_all:
            continue
        p = panels_all[name].replace([np.inf, -np.inf], np.nan).clip(-1e4, 1e4).astype(np.float32)
        if p.notna().sum().sum() == 0:
            log.warning("[%d/%d] %s 全 NaN，跳过落盘", i, len(defs), name)
            continue
        p.to_parquet(panels_dir / f"{name}.parquet")

        cov = float(p.notna().mean().mean())
        # 🚨 ``coverage``（非空率）**测不出「稠密但退化」**：本族面板是 0 填充的，
        # 一个因子可以在 90% 的交易日上「非空但截面全 0」而 coverage 仍有 0.76。
        # 实测（2026-09-20）：mgmt_* 因子在 2018–2025 段全 0，coverage=0.76，
        # 但**只有 9.7% 的交易日有非零截面** —— 基于它的 IC / 消融等于只测了一年。
        # 故必须同时记录「有效交易日占比」：截面标准差 > 0 的日期才有信息。
        row_std = p.std(axis=1, skipna=True)
        active = row_std > 0
        active_frac = float(active.mean()) if len(active) else 0.0
        ic = {h: calc_ic_series(p[ic_codes], fwd[h]).astype(np.float32) for h in HORIZONS}
        row = {"name": name, "set": FAMILY, "label": label, "coverage": cov,
               "active_day_frac": active_frac,
               "active_first": (str(p.index[active].min().date()) if active.any() else None),
               "active_last": (str(p.index[active].max().date()) if active.any() else None),
               **{f"ic_mean_h{h}": float(ic[h].mean()) for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if active_frac < 0.5:
            log.warning("[%d/%d] %s ⚠️ 有效交易日仅 %.1f%%（%s ~ %s）—— "
                        "IC/消融只覆盖这段，别按全样本解读", i, len(defs), name,
                        active_frac * 100, row["active_first"], row["active_last"])
        log.info("[%d/%d] %s cov=%.3f eff=%.3f ic_h1=%+.4f ic_h5=%+.4f "
                 "ic_h10=%+.4f ic_h20=%+.4f | %.0fs",
                 i, len(defs), name, cov, active_frac, row["ic_mean_h1"], row["ic_mean_h5"],
                 row["ic_mean_h10"], row["ic_mean_h20"], time.time() - t0)

    if not args.only:
        common.merge_outputs(ds_dir, FAMILY)
    log.info("holder_dyn 构建完成 %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
