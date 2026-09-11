"""全A机构资金流因子面板（moneyflow 族）→ 并入 all_a_2018_2026 数据集。

背景（2026-09-09）：all_a_2018_2026 已有量价/财务/股东/质押/事件/两融因子，
但"龙虎榜上榜资金"与"大宗交易"两类机构席位资金流维度空白。二者均由 SDK 专属
数据（long_hu_bang / block_trading）支撑，代表机构/游资的显性大额交易行为，
与散户主导的纯量价信号正交性好。

因子清单（set='moneyflow'）：
  ① 龙虎榜（long_hu_bang，按 trade_date 事件日）：
    lhb_net_buy       : 龙虎榜净买入额（BUY_AMOUNT - SELL_AMOUNT），近端日事件
    lhb_count_20d     : 过去 20 交易日上榜次数（事件热度）
    lhb_net_20d       : 过去 20 交易日累计净买入额（资金净流入强度）
  ② 大宗交易（block_trading，按 trade_date 事件日）：
    block_amt_20d     : 过去 20 交易日大宗成交额（机构大额吞吐）
    block_freq_20d    : 过去 20 交易日大宗笔数（B_FREQUENCY 之和）

口径：
- 龙虎榜/大宗事件具有短期自衰减性，因子做成"事件窗口聚合"（20 日滚动）形态，
  非持续静态值；净额同时保留日度信号（lhb_net_buy 当日）。
- 事件按 trade_date 生效（当日收盘后披露，T 日对 T+1 预测，无前视）；
- 面板 date×code float32，从 20160701 起与现有 ic_h 索引对齐；
- IC 用次日/未来 1/5/10/20 日 Spearman，抽稀取列。

用法:
    python -m scripts.builders.build_alla_moneyflow_factors            # 全量
    python -m scripts.builders.build_alla_moneyflow_factors --resume   # 续跑
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

log = setup_logging("build_alla_moneyflow")
from scripts.builders import common  # noqa: E402
from scripts.builders.common import KEEP_FROM, HORIZONS, IC_CODE_STRIDE  # noqa: E402

DATASET = "all_a_2018_2026"
WINDOW = 20


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (close_adj, {long_hu_bang, block_trading})。"""
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

    tables = {}
    for key in ("long_hu_bang", "block_trading"):
        p = cache_root / f"{key}.parquet"
        tables[key] = pd.read_parquet(p) if p.exists() else None
    return close_adj, tables


def _event_rolling(events: pd.DataFrame, cal_idx, codes,
                   value_col: str, agg: str = "sum", count: bool = False) -> dict:
    """事件长表 → 各 code 在交易日历上的滚动窗口聚合因子。

    events: (code, eff, value)。对每个交易日 d，聚合 eff 落在 [d-WINDOW, d] 的
    事件（rolling sum / rolling count），生成 date×code 宽表。需逐交易日展开，
    事件稀疏且日度，用 cal_idx 展开成日频再滚动。
    """
    e = events.copy()
    e = e.dropna(subset=["code", "eff", value_col] if not count else ["code", "eff"])
    e["eff"] = e["eff"].astype("datetime64[ns]")
    # 只保留落在日历内的事件
    e = e[e["eff"].isin(cal_idx)]
    if e.empty:
        return {c: pd.DataFrame(np.nan, index=cal_idx, columns=codes)
                for c in (["count"] if count else [value_col])}

    if count:
        daily = (e.groupby(["code", "eff"], sort=False).size().rename("count"))
    else:
        daily = (e.groupby(["code", "eff"], sort=False)[value_col].sum())
    daily = daily.reset_index()

    # 透视成稀疏 (date×code) → 滚动聚合（ffill 后 rolling 等价于窗口求和）
    piv = daily.pivot(index="eff", columns="code", values=daily.columns[-1])
    piv = piv.reindex(index=cal_idx, columns=codes).fillna(0.0)
    roll = piv.rolling(WINDOW, min_periods=1).sum()
    roll = roll.replace(0.0, np.nan)   # 无事件=NaN（无信号，而非 0）
    key = list(set([("count" if count else value_col)]))
    return {value_col: roll.astype(np.float32)}


def _pit_lhb(lhb: pd.DataFrame, cal_idx, codes) -> dict:
    if lhb is None or lhb.empty:
        return {}
    l = lhb.copy()
    l["trade_date"] = pd.to_datetime(l["trade_date"], errors="coerce")
    l["buy"] = pd.to_numeric(l.get("buy_amount"), errors="coerce")
    l["sell"] = pd.to_numeric(l.get("sell_amount"), errors="coerce")
    l = l.dropna(subset=["code", "trade_date", "buy"])
    l["eff"] = l["trade_date"]
    l["net"] = (l["buy"] - l["sell"].fillna(0.0))

    # 当日净买额（取净额>0 on 上榜日；多席位取合计）
    daily_net = (l.groupby(["code", "eff"], sort=False)["net"].sum()).reset_index()
    net_panels = common.ffill_pit_multi(daily_net.rename(columns={"net": "lhb_net_buy"}),
                                  cal_idx, codes, ["lhb_net_buy"])
    # 20 日上榜次数 + 累计净额
    cnt = _event_rolling(l, cal_idx, codes, "net", agg="sum", count=True)
    net20 = _event_rolling(l, cal_idx, codes, "net", agg="sum")
    return {**net_panels, "lhb_count_20d": cnt["net"], "lhb_net_20d": net20["net"]}


def _pit_block(block: pd.DataFrame, cal_idx, codes) -> dict:
    if block is None or block.empty:
        return {}
    b = block.copy()
    b["trade_date"] = pd.to_datetime(b["trade_date"], errors="coerce")
    b["amt"] = pd.to_numeric(b.get("b_share_amount"), errors="coerce")
    b["vol"] = pd.to_numeric(b.get("b_share_volume"), errors="coerce")
    b = b.dropna(subset=["code", "trade_date", "amt"])
    b["eff"] = b["trade_date"]

    amt20 = _event_rolling(b[["code", "eff", "amt"]], cal_idx, codes, "amt", "sum")
    vol20 = _event_rolling(b[["code", "eff", "vol"]].dropna(subset=["vol"]),
                           cal_idx, codes, "vol", "sum")
    return {"block_amt_20d": amt20["amt"], "block_vol_20d": vol20["vol"]}


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
    stats_path = ds_dir / "factor_stats_moneyflow.jsonl"
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
    if tables.get("long_hu_bang") is not None:
        pn = _pit_lhb(tables["long_hu_bang"], cal_idx, codes)
        panels_all.update(pn)
        defs.update({k: {"src": "long_hu_bang", "label": v} for k, v in {
            "lhb_net_buy": "龙虎榜当日净买入额",
            "lhb_count_20d": "龙虎榜20日上榜次数",
            "lhb_net_20d": "龙虎榜20日累计净买入额",
        }.items() if k in pn})
    if tables.get("block_trading") is not None:
        pn = _pit_block(tables["block_trading"], cal_idx, codes)
        panels_all.update(pn)
        defs.update({k: {"src": "block_trading", "label": v} for k, v in {
            "block_amt_20d": "大宗20日成交额",
            "block_vol_20d": "大宗20日成交量",
        }.items() if k in pn})

    if not defs:
        log.warning("无资金流数据，跳过")
        return

    t0 = time.time()
    for i, (name, meta) in enumerate(sorted(defs.items()), 1):
        if name in done or name not in panels_all:
            continue
        p = panels_all[name].fillna(np.nan).astype(np.float32)
        p = p.replace([np.inf, -np.inf], np.nan).clip(-1e11, 1e11)
        p.to_parquet(panels_dir / f"{name}.parquet")

        cov = float(p.notna().mean().mean())
        ic = {h: calc_ic_series(p[ic_codes], fwd[h]).astype(np.float32)
              for h in HORIZONS}
        row = {"name": name, "set": "moneyflow", "label": meta["label"],
               "coverage": cov,
               **{f"ic_mean_h{h}": float(ic[h].mean()) for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("[%d/%d] %s cov=%.2f ic_h1=%+.4f | %.0fs", i, len(defs),
                 name, cov, row["ic_mean_h1"], time.time() - t0)

    common.merge_outputs(ds_dir, 'moneyflow')
    log.info("资金流因子构建完成 %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()