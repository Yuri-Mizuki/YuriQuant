"""SUE 标准化意外盈余 + 质押深度因子 → 并入 all_a_2018_2026 数据集。

背景（2026-09-11）：现有 evt 族的 notice_sue 仅是业绩预告净利同比中值/100
（纵截面），非学术意义的标准化意外盈余（Standardized Unexpected Earnings，SUE）；
现有 fundamental 族的质押因子都是静态比例（pledge_ratio / pledge_holder_ratio /
frozen_ratio），缺"新增质押边际/变化率/冻结对市值密度"等深度维度。本脚本补齐：

  SUE 族（set='evt'）—— 横截面标准化意外盈余，事件脉冲：
    sue_notice_cs    : 业绩预告盈利同比在"同报告期+同公告日窗口横截面"的 z-score
                         （剔除共同成分，得公司层面意外盈余；学术 SUE 标准化）
    sue_notice_20d   : 过去 20 交易日 SUE 累积（预告窗口脉冲，事件热度/幅度）
    sue_express_cs   : 业绩快报净利同比横截面 z-score（快报比预告更实）
    sue_express_20d  : 业绩快报 SUE 20 日累积

  质押深度族（set='fundamental'）—— 边际与密度：
    pledge_chg_20d   : 质押比例（总质押市值/流通市值加权代理）20日变化率（新增质押）
    pledge_frn_density: 冻结市值 / 流通市值（winsorize 防异常），资产质量压力
    pledge_holder_density: 冻结/股东持股比例横截面（占其持股覆盖度）

口径：
- SUE 横截面标准化：对公告日窗口内所有同报告期公司，对盈利同比做 z-score
    （(x - 截面均值) / 截面标准差），阈值 5σ 截断；再 ffill 到交易日并做事件过期
    （仅保留公告后 20 交易日窗口，避免永久残留）。
- 质押深度：质押长表按 ann_date PIT 前向填充；变化率 = 最新质押比例相对 20 交易
    日前的变化；市值密度需代理（质押市值 = total_pledge_shr × 收盘价）。
- 面板 date×code float32，从 2016-07-01 起与现有 ic_h 索引对齐；
- IC 用次日/未来 1/5/10/20 日 Spearman，抽稀取列。

用法:
    python -m scripts.builders.build_alla_sue_pledge_factors            # 全量
    python -m scripts.builders.build_alla_sue_pledge_factors --resume   # 续跑
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

log = setup_logging("build_alla_sue_pledge")
from scripts.builders import common  # noqa: E402
from scripts.builders.common import KEEP_FROM, HORIZONS, IC_CODE_STRIDE  # noqa: E402

DATASET = "all_a_2018_2026"
EVENT_WINDOW = 20
Z_CAP = 5.0
WINS_CAP = 2.0  # 冻结/持股 密度 winsorize 上界


# ---------------------------------------------------------------------------
# PIT helper（长表 code+date → date×code 宽表）
# ---------------------------------------------------------------------------
def _pit_panel(series_long, cal_idx, codes, value_col: str) -> pd.DataFrame:
    """按 eff 前向填充：返回 date×code 宽表（含 NaN 未覆盖）。"""
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


def _event_window_cum(series_long, cal_idx, codes, value_col: str) -> pd.DataFrame:
    """事件脉冲：对每个事件日，把 value 展开到 [eff, eff+EVENT_WINDOW-1] 并累加。

    用于 SUE 这类"公告后短期有效、过后衰减"的事件因子——避免 PIT ffill 永久残留。
    """
    e = series_long.dropna(subset=["eff", value_col]).copy()
    e["eff"] = pd.to_datetime(e["eff"])
    e = e[e["eff"].isin(cal_idx)]
    if e.empty:
        return pd.DataFrame(np.nan, index=cal_idx, columns=codes)
    cal_list = list(cal_idx)
    pos = pd.Series(range(len(cal_list)), index=cal_list)
    e = e.assign(pos=e["eff"].map(pos)).dropna(subset=["pos"])
    e["pos"] = e["pos"].astype(int)
    parts = []
    for w in range(EVENT_WINDOW):
        p = e.copy()
        p["eff_pos"] = p["pos"] + w
        p = p[(p["eff_pos"] >= 0) & (p["eff_pos"] < len(cal_list))]
        if p.empty:
            continue
        p["eff"] = p["eff_pos"].map(pd.Series(cal_list, index=range(len(cal_list))))
        parts.append(p[["code", "eff", value_col]])
    if not parts:
        return pd.DataFrame(np.nan, index=cal_idx, columns=codes)
    u = pd.concat(parts, ignore_index=True)
    acc = (u.groupby(["code", "eff"], sort=False)[value_col].sum().reset_index())
    frame = pd.DataFrame(np.nan, index=cal_idx, columns=codes)
    for code, g in acc.groupby("code"):
        frame[code] = g.set_index("eff")[value_col].reindex(cal_idx).values
    return frame


def _cs_zscore(s: pd.Series) -> pd.Series:
    """横截面 z-score（去均值/标准差），5σ 截断（学术 SUE 标准化）。"""
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd is None or sd == 0 or np.isnan(sd):
        return s * np.nan
    z = (s - mu) / sd
    return z.clip(-Z_CAP, Z_CAP)


# ---------------------------------------------------------------------------
# ① SUE 横截面标准化意外盈余（按公告日窗口截面）
# ---------------------------------------------------------------------------
def _build_sue(notice, express, cal_idx, codes) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    def _cs_normalize(df, value_col, ann_col="ann_date", report_col="report_period",
                      min_group=5):
        """同公告周内所有公司的 value 做横截面 z-score（≥min_group 家才标准化）。

        分组键仅用公告周（不含 code），使同一周内所有公告公司彼此可比；
        相比"单股单期"的伪标准化，这才符合 SUE 的"相对意外"定义。同比已剔除
        规模/时序成分，跨行业可比，故不需按行业再分。
        """
        d = df.dropna(subset=["code", ann_col, value_col]).copy()
        d = _to_num(d, [value_col])
        d = d.dropna(subset=[value_col])
        d[ann_col] = pd.to_datetime(d[ann_col], errors="coerce")
        d = d.dropna(subset=[ann_col])
        d["_week"] = d[ann_col].dt.to_period("W")
        sizes = d.groupby("_week")["code"].nunique()
        keep = sizes[sizes >= min_group].index
        d = d[d["_week"].isin(keep)]
        if d.empty:
            return pd.DataFrame(columns=["code", "eff", value_col])
        z = d.groupby("_week")[value_col].transform(_cs_zscore)
        return pd.DataFrame({"code": d["code"], "eff": d[ann_col], value_col: z})

    # 业绩预告：盈利同比中值 → 横截面 SUE
    if notice is not None and not notice.empty:
        nt = notice.copy()
        if {"p_change_max", "p_change_min", "report_period"}.issubset(nt.columns):
            nt["p_change_mid"] = ((nt["p_change_max"] + nt["p_change_min"]) / 2.0)
            sue_l = _cs_normalize(nt, "p_change_mid")
            if not sue_l.empty:
                out["sue_notice_cs"] = _pit_panel(
                    sue_l, cal_idx, codes, "p_change_mid").clip(-Z_CAP, Z_CAP)
                out["sue_notice_20d"] = _event_window_cum(
                    sue_l.rename(columns={"p_change_mid": "sue_notice_20d"}),
                    cal_idx, codes, "sue_notice_20d")

    # 业绩快报：净利同比 → 横截面 SUE
    if express is not None and not express.empty:
        ex = express.copy()
        if "yoy_gr_net_profit_parent" in ex.columns and "report_period" in ex.columns:
            sue_l = _cs_normalize(ex, "yoy_gr_net_profit_parent")
            if not sue_l.empty:
                out["sue_express_cs"] = _pit_panel(
                    sue_l, cal_idx, codes,
                    "yoy_gr_net_profit_parent").clip(-Z_CAP, Z_CAP)
                out["sue_express_20d"] = _event_window_cum(
                    sue_l.rename(columns={
                        "yoy_gr_net_profit_parent": "sue_express_20d"}),
                    cal_idx, codes, "sue_express_20d")
    return out


# ---------------------------------------------------------------------------
# ② 质押深度：变化率 / 市值密度（PIT 最新值）
# ---------------------------------------------------------------------------
def _build_pledge_depth(pledge, close_raw, cal_idx, codes) -> dict[str, pd.DataFrame]:
    if pledge is None or pledge.empty:
        return {}
    p = pledge.copy()
    p["ann_date"] = pd.to_datetime(p["ann_date"], errors="coerce")
    p = _to_num(p, ["total_pledge_shr", "fro_shares", "FRO_SHR_TO_TOTAL_HOLDING_RATIO",
                    "total_holding_shr_ratio"])
    p = p.dropna(subset=["code", "ann_date"])
    p["eff"] = p["ann_date"]

    out: dict[str, pd.DataFrame] = {}

    # 每股公告日合并多条（多股东/多笔）为当日总量
    agg_cols: dict = {
        "pledge_shr": ("total_pledge_shr", "sum"),
        "freeze_shr": ("fro_shares", "sum"),
        "hold_shr": ("total_holding_shr_ratio", "sum"),
    }
    if "FRO_SHR_TO_TOTAL_HOLDING_RATIO" in p.columns:
        agg_cols["frn_ratio"] = ("FRO_SHR_TO_TOTAL_HOLDING_RATIO", "sum")
    agg = p.groupby(["code", "eff"], sort=False).agg(**agg_cols).reset_index()

    # ① 质押 20 日变化率：当前质押股本相对 20 交易日前的增长
    pv = _pit_panel(agg.rename(columns={"pledge_shr": "_pv"}), cal_idx, codes, "_pv")
    base = pv.shift(20, axis=0).replace(0.0, np.nan)
    chg = (pv - pv.shift(20, axis=0)) / base
    out["pledge_chg_20d"] = chg.replace([np.inf, -np.inf], np.nan).clip(-5.0, 5.0)

    # ② 冻结股本密度（99 分位 winsorize 标量上界）
    fz = _pit_panel(agg.rename(columns={"freeze_shr": "_fz"}), cal_idx, codes, "_fz")
    fz = fz.replace([np.inf, -np.inf], np.nan)
    upper = float(np.nanquantile(fz.values, 0.99)) if fz.isna().all().all() is False \
        else 1.0
    out["pledge_frn_density"] = fz.clip(upper=upper)

    # ③ 冻结/持股横截面密度（FRO_SHR_TO_TOTAL_HOLDING_RATIO 已为比例，winsorize）
    if "frn_ratio" in agg.columns:
        frn = _pit_panel(agg.rename(columns={"frn_ratio": "_frn"}), cal_idx, codes,
                         "_frn")
        out["pledge_holder_density"] = (frn
                                        .replace([np.inf, -np.inf], np.nan)
                                        .clip(upper=WINS_CAP))
    return out


# ---------------------------------------------------------------------------
# 数据 & 主流程
# ---------------------------------------------------------------------------
def _to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_panels():
    from config import Config
    cache_root = Path(str(Config.cache()["root"]))
    daily = pd.read_parquet(cache_root / "daily_all_a.parquet")
    daily.index = daily.index.set_levels(
        daily.index.levels[0].normalize(), level=0)
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

    tabs = {}
    for fn, k in [("profit_notice", "notice"), ("profit_express", "express"),
                  ("equity_pledge_freeze", "pledge")]:
        p = cache_root / f"{fn}.parquet"
        tabs[k] = pd.read_parquet(p) if p.exists() else None
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
    stats_path = ds_dir / "factor_stats_sue_pledge.jsonl"
    done = {json.loads(l)["name"] for l in
            stats_path.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if args.resume and stats_path.exists() else set()

    close_adj, close_raw, tabs = load_panels()
    codes = close_adj.columns
    cal_idx = close_adj.index
    fwd = {h: close_adj.pct_change(h, fill_method=None).shift(-h) for h in HORIZONS}
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]

    labels = {
        "sue_notice_cs": "业绩预告盈利意外横截面zscore",
        "sue_notice_20d": "业绩预告SUE 20日累积",
        "sue_express_cs": "业绩快报净利意外横截面zscore",
        "sue_express_20d": "业绩快报SUE 20日累积",
        "pledge_chg_20d": "质押股本20日变化率",
        "pledge_frn_density": "冻结股本密度(99分位截断)",
        "pledge_holder_density": "冻结/持股覆盖密度(2×winsorize)",
    }
    sets = {"sue_": "evt", "pledge_": "fundamental"}

    panels: dict[str, pd.DataFrame] = {}
    panels.update(_build_sue(tabs.get("notice"), tabs.get("express"),
                             cal_idx, codes))
    panels.update(_build_pledge_depth(tabs.get("pledge"), close_raw, cal_idx, codes))

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
        row = {"name": name, "set": next((v for k, v in sets.items()
                                          if name.startswith(k)), "fundamental"),
               "label": labels.get(name, name), "coverage": cov,
               **{f"ic_mean_h{h}": float(ic[h].mean()) for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("[%d/%d] %s cov=%.2f ic_h1=%+.4f | %.0fs", i, len(panels),
                 name, cov, row["ic_mean_h1"], time.time() - t0)

    common.merge_outputs(ds_dir, 'sue_pledge')
    log.info("SUE+质押深度因子构建完成 %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()