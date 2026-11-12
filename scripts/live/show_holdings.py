"""yqcore-solo 持仓查询（只读，随时可跑）

背景
----
`yqcore-solo` 是**月频**策略：信号日 = 当月首个交易日，成交日 = 次一交易日开盘，
持仓整月不变，下一期才换。所以「看持仓」不是每天生成新快照，而是
**查当前生效的那一期**。

`solo_live.py` 负责**生成**一期持仓（有副作用，落 holdings/orders/meta 三个文件）；
本脚本只负责**查询**：定位当期、按最新收盘重估、打印。它**不写任何文件** ——
随时跑都不会污染 `live/` 目录，也不会让「当前持仓是哪一份」变得含糊。

用法
----
    python -m scripts.live.show_holdings            # 查当期（最新一期）
    python -m scripts.live.show_holdings --all      # 列出所有期次一览
    python -m scripts.live.show_holdings --date 20261102   # 查指定期次

与 `solo_live.py` 的关系
------------------------
- 持仓**构成**（哪几只、多少股）以落盘的 `holdings_<ds>.csv` 为准，本脚本不改；
- **估值**（现价、盈亏）每次都按日线最新收盘重算，所以同一期在不同日子查
  会得到不同市值 —— 这是「随时可查」的意义，不是数据不一致。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import Config

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "reports" / "yq_core" / "live"


# ---------------------------------------------------------------------------
# 定位期次
# ---------------------------------------------------------------------------
def list_periods(live_dir: Path | None = None) -> list[str]:
    """所有已生成的持仓期次（``signal_date``，升序）。"""
    d = live_dir or LIVE_DIR
    out = []
    for p in d.glob("holdings_*.csv"):
        stem = p.stem.rsplit("_", 1)[-1]
        if len(stem) == 8 and stem.isdigit():
            out.append(stem)
    return sorted(out)


def latest_period(live_dir: Path | None = None) -> str | None:
    ps = list_periods(live_dir)
    return ps[-1] if ps else None


def load_holdings(ds: str, live_dir: Path | None = None) -> pd.DataFrame:
    d = live_dir or LIVE_DIR
    p = d / f"holdings_{ds}.csv"
    if not p.exists():
        raise FileNotFoundError(f"没有 {ds} 这一期的持仓：{p}")
    return pd.read_csv(p, dtype={"code": str}, index_col="code")


def load_meta(ds: str, live_dir: Path | None = None) -> dict:
    d = live_dir or LIVE_DIR
    p = d / f"meta_{ds}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 估值
# ---------------------------------------------------------------------------
def latest_close(codes: list[str], cache_root: str | None = None) -> tuple[pd.Series, pd.Timestamp]:
    """取每只股票**最新可得**收盘价，返（价格 Series, 该日期）。

    用未复权收盘（与 ``solo_live.load_prices`` 同源），因为持仓成本价是按
    实际成交价记的，复权价对不上。
    """
    root = Path(cache_root or str(Config.cache()["root"]))
    px = pd.read_parquet(root / "daily_all_a.parquet", columns=["close"])
    dates = pd.to_datetime(px.index.get_level_values(0))
    px.index = pd.MultiIndex.from_arrays(
        [dates, px.index.get_level_values(1).astype(str)])
    wide = px["close"].unstack(level=1)
    wide = wide.reindex(columns=[c for c in codes if c in wide.columns])
    # 最后一行的有效值：部分股票当天停牌，则该列取它自己最后一个非空
    last = wide.ffill().iloc[-1]
    return last, wide.index[-1]


def value_position(h: pd.DataFrame, last: pd.Series,
                   asof: pd.Timestamp) -> pd.DataFrame:
    """按最新价重估，返回带 latest/ret/pnl 的表。"""
    df = h.copy()
    df["latest"] = df.index.map(lambda c: last.get(c, float("nan")))
    df["ret_total"] = df["latest"] / df["open_exec"] - 1.0
    df["mv_now"] = df["shares"] * df["latest"]
    df["pnl_now"] = df["cost"] * df["ret_total"]
    df = df.rename(columns={"open_exec": "cost_px"})
    keep = ["rank", "industry", "shares", "cost_px", "latest",
            "ret_total", "mv_now", "pnl_now"]
    return df[[c for c in keep if c in df.columns]]


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def show_period(ds: str, live_dir: Path | None = None,
                cache_root: str | None = None) -> None:
    h = load_holdings(ds, live_dir)
    meta = load_meta(ds, live_dir)
    last, asof = latest_close(list(h.index), cache_root)
    v = value_position(h, last, asof)

    cap = meta.get("capital")
    sig = meta.get("signal_date", ds)
    exe = meta.get("exec_date")
    print("=" * 78)
    print(f"yqcore-solo 当期持仓  信号日 {sig} → 成交 {exe or '?'}（T+1 开盘）")
    print(f"目标 {meta.get('k', '?')} 只 | 实持 {len(v)} 只"
          f" | 每申万一级 ≤{meta.get('max_per_sector', '?')} 只"
          f" | 整手 {meta.get('lot', '?')} 股")
    if cap:
        print(f"资金 {cap:,.0f} 元 | 投入 {v['mv_now'].sum() * 0 + h['cost'].sum():,.0f}"
              f"（{h['cost'].sum() / cap * 100:.2f}%）")
    print(f"估值日 {asof.date()}（按最新可得收盘，非成交日）")
    print("-" * 78)

    show = v.copy()
    show["shares"] = show["shares"].astype(int)
    for c in ("cost_px", "latest"):
        show[c] = show[c].map(lambda x: f"{x:.2f}")
    show["ret_total"] = show["ret_total"].map(lambda x: f"{x * 100:+.2f}%")
    show["mv_now"] = show["mv_now"].map(lambda x: f"{x:,.0f}")
    show["pnl_now"] = show["pnl_now"].map(lambda x: f"{x:+,.0f}")
    show = show.rename(columns={
        "rank": "榜内排名", "industry": "行业", "shares": "股数",
        "cost_px": "成本价", "latest": "现价", "ret_total": "期间收益",
        "mv_now": "市值", "pnl_now": "盈亏"})
    print(show.to_string())

    mv = float(v["mv_now"].sum())
    pnl = float(v["pnl_now"].sum())
    cost = float(h["cost"].sum())
    print("-" * 78)
    line = f"持仓市值 {mv:,.0f} | 成本 {cost:,.0f} | 浮盈亏 {pnl:+,.0f}（{pnl / cost * 100:+.2f}%）"
    if cap:
        line += f" | 占总资金 {mv / cap * 100:.2f}% | 现金 {cap - mv:,.0f}"
    print(line)
    if meta.get("tradable", {}).get("n_flipped"):
        t = meta["tradable"]
        print(f"⚠️ 该期曾按信号日实时状态纠正 {t['n_flipped']} 只"
              f"（榜内 tradable 列已过期），名单以此份为准")
    print("=" * 78)


def show_all(live_dir: Path | None = None) -> None:
    periods = list_periods(live_dir)
    if not periods:
        print("live/ 下还没有任何持仓期次")
        return
    rows = []
    for ds in periods:
        h = load_holdings(ds, live_dir)
        m = load_meta(ds, live_dir)
        rows.append({
            "信号日": ds,
            "成交日": m.get("exec_date", "?"),
            "只数": len(h),
            "成本": round(float(h["cost"].sum())),
            "组合收益": (f"{m['portfolio_ret'] * 100:+.2f}%"
                    if m.get("portfolio_ret") is not None else "?"),
        })
    print("=" * 62)
    print("yqcore-solo 历史期次（持仓一个月一期）")
    print("-" * 62)
    print(pd.DataFrame(rows).to_string(index=False))
    print("=" * 62)
    print(f"共 {len(periods)} 期；最新一期 {periods[-1]}"
          f"（查详情：--date {periods[-1]}）")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="yqcore-solo 持仓查询（只读，随时可跑，不写文件）")
    ap.add_argument("--date", default=None, metavar="YYYYMMDD",
                    help="查指定期次（信号日）；缺省 = 最新一期")
    ap.add_argument("--all", action="store_true", help="列出全部期次一览")
    ap.add_argument("--live-dir", default=None, help="持仓目录（缺省 reports/yq_core/live）")
    args = ap.parse_args()

    live_dir = Path(args.live_dir) if args.live_dir else None

    if args.all:
        show_all(live_dir)
        return

    ds = args.date or latest_period(live_dir)
    if ds is None:
        raise SystemExit("live/ 下还没有任何持仓期次，先跑 solo_live.py 生成第一期")
    show_period(ds, live_dir)


if __name__ == "__main__":
    main()