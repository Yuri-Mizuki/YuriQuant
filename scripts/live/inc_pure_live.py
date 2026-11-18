"""`inc_pure` 实盘出单器：董监高主动增持 → T+1 开盘集合竞价买入，持 20 个交易日

它与回测（`scripts/evaluation/event_driven_probe.py`）的关系
-----------------------------------------------------------
**同一个口径，两种用途**：

- 回测回答「这个策略赚不赚钱」，用全历史一次性算完，允许无限资金、无并发上限；
- 本脚本回答「今天该挂哪些单」，逐日滚动，**必须处理回测刻意回避的约束**：
  现金不够、并发太多、整手取整、ST/停牌/涨停买不到、单笔超过成交额 1%。

所以本脚本**不重算收益**，只复刻事件定义与买卖时点：

| 环节 | 口径 | 来源 |
|---|---|---|
| 事件 | `direction=增持` + `reason∈{竞价交易,二级市场买卖,大宗交易}` + `relation=本人` + 合计 ≥0.1% | `fam_inc_pure` |
| 聚合 | 同 (code, 公告日) 多条记录合并成一条，比例求和 | `_inc_refined` |
| 可见日 | 公告日当天或其后第一个交易日（周末公告顺延） | `_avail` |
| 建仓 | 可见日的**次一交易日开盘**（T+1 集合竞价） | `build_positions` |
| 卖出 | 建仓日 + 20 个交易日的**开盘** | 同上 |
| 重叠 | 同股在持有窗内再出事件 ⇒ **延长持有**，不加仓 | 同上 |

**实盘新增、回测没有的四条闸门**（都写进 `--help`，可关）：

1. `is_st` / `is_suspended`（`history_stock_status`）：ST 涨跌幅 5%，回测未剔；
   T 日停牌的票第二天开盘可能仍无流动性。
2. **上市不足 60 个交易日**：回测踩过「买一只还不存在的股票」的坑
   （`_base/tradable_mask` 不含上市日），实盘必须自己拦。
3. **单笔 ≤ T 日成交额 1%**：`exec_fill_probe` 的容量假设，实盘要真按它截断。
4. **并发上限**：回测峰值 122 只、平均 19.5 只，400 万本金下 122 只每笔只剩 3.3 万，
   整手都买不了几只。故设 `--max-open`（默认 30），超出按事件强度截断并记录。

用法
----
    # 只出单，不落盘（默认 dry-run）
    python -m scripts.live.inc_pure_live signal --asof 20260923 --capital 4000000

    # 确认后落盘：orders/buy_<成交日>.csv、orders/sell_<成交日>.csv、state.json
    python -m scripts.live.inc_pure_live signal --asof 20260923 --capital 4000000 --commit

    # 补跑：从上次 asof 的次日起，一次补齐多天事件
    python -m scripts.live.inc_pure_live signal --asof 20260925

    # 看当前持仓与到期日
    python -m scripts.live.inc_pure_live check

⚠️ 数据前置：`alt_cninfo_holder.parquet` 的 `ann_date` 必须已推到 **T 日当晚**，
`daily_all_a` / `calendar` 至少到 T 日。两者任一滞后，本脚本会直接报错退出，
不会用旧数据静默出单（旧数据出单 = 用过期事件下单，比不出单更危险）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PQ = Path("D:/data/parquet")
DEFAULT_OUT = ROOT / "reports" / "inc_pure_live"

HOLD = 20                      # 持有交易日数（与回测 H=20 一致）
ACTIVE = ("竞价交易", "二级市场买卖", "大宗交易")
MIN_RATIO = 0.1                # 合计增持占总股本 ≥0.1%
MIN_LISTED_DAYS = 60           # 上市不足 60 个交易日不建仓
LOT = 100
COST_ONE_WAY = 0.00135


# ============================================================ 数据
def _die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_cal() -> pd.DatetimeIndex:
    p = PQ / "calendar.parquet"
    if not p.exists():
        _die(f"缺交易日历 {p}")
    s = pd.read_parquet(p)["date"]
    # 该表存的是 int YYYYMMDD（20260923），不是 Timestamp
    if pd.api.types.is_integer_dtype(s):
        s = pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")
    return pd.DatetimeIndex(s).normalize().sort_values()


def load_market(since: pd.Timestamp) -> pd.DataFrame:
    """长表 (date, code) → open/close/amount，**只读 since 之后**。

    直接 ``read_parquet`` 全表要 12 秒（1160 万行），实盘链路每天跑一次不值得；
    用 pyarrow 的 row-group 过滤后是 0.1 秒。``since`` 取 asof − 250 自然日，
    足够 `listed_days` 判断「上市不足 60 个交易日」。
    """
    p = PQ / "daily_all_a.parquet"
    if not p.exists():
        _die(f"缺日线 {p}")
    t = pq.read_table(
        p, columns=["date", "code", "open", "close", "amount"],
        filters=[("date", ">=", since.to_pydatetime())],
    )
    d = t.to_pandas().reset_index()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    return d


def load_status(asof: pd.Timestamp) -> pd.DataFrame:
    """T 日的状态快照：ST / 停牌 / 涨停价。"""
    p = PQ / "history_stock_status.parquet"
    if not p.exists():
        print("[warn] 缺 history_stock_status，ST/停牌闸门降级为不启用")
        return pd.DataFrame(columns=["code", "is_st", "is_suspended", "high_limited"])
    t = pq.read_table(p, filters=[("date", "==", asof.to_pydatetime())])
    s = t.to_pandas().reset_index()
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()
    return s.copy()


def load_events(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """公告日落在 (lo, hi] 的主动增持事件，已按 (code, 公告日) 聚合。"""
    p = PQ / "alt_cninfo_holder.parquet"
    if not p.exists():
        _die(f"缺增持明细 {p}")
    df = pd.read_parquet(
        p, columns=["code", "sec_name", "ann_date", "direction", "chg_ratio",
                    "reason", "relation"]
    )
    df = df[df["direction"].astype(str).str.contains("增持")]
    df = df[df["reason"].astype(str).isin(ACTIVE)]
    df = df[df["relation"].astype(str) == "本人"]
    d = pd.to_datetime(df["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.assign(ann=d)
    df = df[(df["ann"] > lo) & (df["ann"] <= hi)]
    df["r"] = pd.to_numeric(df["chg_ratio"], errors="coerce").abs()
    df = df.dropna(subset=["r", "ann"])
    df["code"] = _sym(df["code"])
    if df.empty:
        return pd.DataFrame(columns=["code", "name", "ann", "strength", "n"])
    g = (df.groupby(["code", "ann"])
           .agg(name=("sec_name", "first"), strength=("r", "sum"), n=("r", "size"))
           .reset_index())
    return g[g["strength"] >= MIN_RATIO].reset_index(drop=True)


def _sym(code: pd.Series) -> pd.Series:
    c = code.astype(str).str.strip()
    has = c.str.contains(r"\.", regex=True)
    six = c.str.zfill(6)
    p = six.str[0]
    suf = np.select([p.eq("6"), p.isin(["0", "3"]), p.isin(["4", "8", "9"])],
                    [".SH", ".SZ", ".BJ"], default="")
    return pd.Series(np.where(has, c, six + suf), index=c.index)


# ============================================================ 状态
def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"asof": None, "capital": 0.0, "cash": 0.0, "positions": []}


def save_state(path: Path, st: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================ 定价
def latest_px(mk: pd.DataFrame, asof: pd.Timestamp, codes: list[str]) -> pd.Series:
    sub = mk[(mk["date"] == asof) & (mk["code"].isin(codes))]
    return sub.set_index("code")["close"]


def listed_days(mk: pd.DataFrame, code: str, asof: pd.Timestamp) -> int:
    s = mk[(mk["code"] == code) & (mk["date"] <= asof) & mk["close"].notna()]
    return int(len(s))


# ============================================================ 主流程
def cmd_signal(args: argparse.Namespace) -> int:
    out = Path(args.out)
    orders, state_p = out / "orders", out / "state.json"
    cal = load_cal()
    asof = pd.Timestamp(str(args.asof))

    # ---- 数据新鲜度硬闸门：宁可不跑，也不要用过期事件下单
    ann_latest = pd.to_datetime(
        pd.read_parquet(PQ / "alt_cninfo_holder.parquet", columns=["ann_date"])["ann_date"]
        .astype(str), format="%Y%m%d", errors="coerce").max()
    if ann_latest < asof:
        _die(f"增持表只到 {ann_latest.date()}，asof={asof.date()} "
             f"⇒ 盘后公告链路未更新。拒绝出单。")
    if cal.max() < asof:
        _die(f"交易日历只到 {cal.max().date()}，asof={asof.date()}")
    if asof not in cal:
        _die(f"{asof.date()} 不是交易日（日历里没有）")

    i_asof = cal.get_indexer([asof])[0]
    t1 = cal[i_asof + 1] if i_asof + 1 < len(cal) else None
    if t1 is None:
        _die("日历末尾没有下一交易日，无法给出成交日")
    i_t1 = i_asof + 1

    st = load_state(state_p)
    # 本金只在**从未初始化**时写入；已运行过的账户一律以 state 里的 cash 为准，
    # 否则一次误传 --capital 就会把现金余额悄悄改掉。
    if args.capital and args.capital > 0 and st.get("asof") is None:
        st["capital"] = float(args.capital)
        st["cash"] = float(args.capital)
    capital = st["capital"] or float(args.capital or 0)
    if capital <= 0:
        _die("未设本金：首次运行请带 --capital")

    #: 事件区间 (lo, asof]。首次运行（无状态）取 asof 当天，即 lo = asof − 1 天。
    lo = pd.Timestamp(st["asof"]) if st.get("asof") else asof - pd.Timedelta(days=1)
    gap = int(np.searchsorted(cal.values, np.datetime64(asof))
              - np.searchsorted(cal.values, np.datetime64(lo)))
    if gap > 3:
        print(f"[warn] 距上次运行（{lo.date()}）已过 {gap} 个交易日：本次会一次性补处理"
              f"这 {gap} 天的全部事件。补跑可以，但**到期的持仓仍按今天统一处理**，"
              f"与逐日运行不等价。")
    if lo > asof:
        _die(f"状态 asof={lo.date()} 晚于本次 asof={asof.date()}，拒绝倒序运行")

    mk = load_market(asof - pd.Timedelta(days=250))
    stat = load_status(asof)
    ev = load_events(lo, asof)
    px = latest_px(mk, asof, ev["code"].tolist())

    # ---- 1. 卖出：今天到期（sell_date <= t1），T+1 开盘挂卖
    sells, keep, frozen = [], [], []
    for p in st["positions"]:
        if p.get("sell_date") and pd.Timestamp(p["sell_date"]) <= t1:
            q = float(latest_px(mk, asof, [p["code"]]).get(p["code"], np.nan))
            if not np.isfinite(q):
                # 到期日无收盘价 ⇒ 停牌/退市，卖不掉，只能继续持有并报出来
                frozen.append(p["code"])
                keep.append(p)
                continue
            sells.append(dict(p, px_ref=q))
        else:
            keep.append(p)

    # ---- 2. 现金（卖出所得按 T 日收盘价估算，实际以 T+1 开盘成交为准）
    cash = float(st["cash"]) + sum(
        s["shares"] * s["px_ref"] * (1 - COST_ONE_WAY) for s in sells)

    # ---- 3. 买入候选：逐条过闸门
    open_codes = {p["code"] for p in keep}
    #: 本批新事件对应的卖出日（建仓日 t1 + HOLD 个交易日）；日历不够长则不出单
    if i_t1 + HOLD < len(cal):
        new_sell = cal[i_t1 + HOLD]
    else:
        print(f"[warn] 日历只剩 {len(cal) - i_t1} 个交易日 < HOLD={HOLD}，"
              f"无法确定卖出日 ⇒ 本次不建新仓（先补日历）")
        new_sell = None

    rows, skipped, extend = [], [], []
    for e in ev.itertuples(index=False):
        code, name = e.code, e.name
        if code in open_codes:                      # 重叠 ⇒ 延长持有，不新开仓
            for p in keep:
                if p["code"] == code and new_sell is not None:
                    old = str(p["sell_date"])
                    p["sell_date"] = max(old, str(new_sell.date()))
                    if p["sell_date"] != old:
                        extend.append({"code": code, "from": old, "to": p["sell_date"]})
            skipped.append((code, name, "已持仓 ⇒ 延长持有，不加仓"))
            continue
        if args.gate_status:
            s = stat[stat["code"] == code]
            if not s.empty:
                if bool(s["is_st"].iloc[0]):
                    skipped.append((code, name, "ST"))
                    continue
                if bool(s["is_suspended"].iloc[0]):
                    skipped.append((code, name, "T日停牌"))
                    continue
        if code not in px.index or not np.isfinite(px[code]):
            skipped.append((code, name, "T日无收盘价"))
            continue
        if listed_days(mk, code, asof) < MIN_LISTED_DAYS:
            skipped.append((code, name, f"上市不足{MIN_LISTED_DAYS}日"))
            continue
        rows.append({"code": code, "name": name, "px": float(px[code]),
                     "strength": float(e.strength), "n_person": int(e.n)})
    cand = pd.DataFrame(rows)

    # ---- 4. 并发上限 + 资金分配
    room = max(0, int(args.max_open) - len(keep))
    cand = cand.sort_values("strength", ascending=False).reset_index(drop=True) if not cand.empty else cand
    if not cand.empty and len(cand) > room:
        for _, r in cand.iloc[room:].iterrows():
            skipped.append((r["code"], r["name"], f"并发上限 {args.max_open}"))
        cand = cand.iloc[:room].reset_index(drop=True)

    buys = []
    if not cand.empty:
        per = min(capital * args.max_single_pct, cash / len(cand))
        adv = mk[(mk["date"] == asof) & (mk["code"].isin(cand["code"]))].set_index("code")["amount"]
        for r in cand.itertuples(index=False):
            budget = per
            a = float(adv.get(r.code, np.nan))
            if np.isfinite(a) and a > 0:                     # 单笔 ≤ 成交额 1%
                budget = min(budget, a * args.max_adv_pct)
            sh = int(budget / r.px // LOT) * LOT
            if sh < LOT:
                skipped.append((r.code, r.name, f"整手不足（预算 {budget:,.0f} / 价 {r.px:.2f}）"))
                continue
            amt = sh * r.px * (1 + COST_ONE_WAY)
            if amt > cash:
                skipped.append((r.code, r.name, "现金不足"))
                continue
            cash -= amt
            buys.append({"code": r.code, "name": r.name, "shares": sh, "px": r.px,
                         "amount": amt, "strength": r.strength, "n_person": r.n_person,
                         "buy_date": t1, "sell_date": new_sell})

    # ---- 5. 落盘
    new_pos = keep + [
        {"code": b["code"], "name": b["name"], "buy_date": str(b["buy_date"].date()),
         "sell_date": str(b["sell_date"].date()), "shares": b["shares"],
         "price": round(b["px"], 4), "strength": round(b["strength"], 4)}
        for b in buys
    ]
    new_state = {"asof": str(asof.date()), "capital": capital,
                 "cash": round(cash, 2), "positions": new_pos}

    SELL_COLS = ["code", "name", "side", "shares", "px_ref", "buy_date", "note"]
    BUY_COLS = ["code", "name", "side", "shares", "px_ref", "amount", "strength",
                "n_person", "sell_date", "note"]
    sell_df = pd.DataFrame([{
        "code": s["code"], "name": s["name"], "side": "SELL_OPEN_AUCTION",
        "shares": s["shares"], "px_ref": round(float(s.get("px_ref", np.nan)), 4),
        "buy_date": s["buy_date"], "note": "开盘集合竞价市价卖出（卖出端对价格不敏感）",
    } for s in sells]).reindex(columns=SELL_COLS)
    buy_df = pd.DataFrame([{
        "code": b["code"], "name": b["name"], "side": "BUY_OPEN_AUCTION",
        "shares": b["shares"], "px_ref": round(b["px"], 4),
        "amount": round(b["amount"], 2), "strength": round(b["strength"], 4),
        "n_person": b["n_person"], "sell_date": str(b["sell_date"].date()),
        "note": "9:15–9:25 挂开盘集合竞价限价单，价格挂涨停价保成交",
    } for b in buys]).reindex(columns=BUY_COLS)

    print(f"\n=== inc_pure 出单  asof={asof.date()}  成交日={t1.date()} ===")
    print(f"本金 {capital:,.0f}  可用现金 {cash:,.0f}  当前持仓 {len(new_pos)} 只"
          f"（上限 {args.max_open}）")
    if frozen:
        print(f"[warn] 已到期但无收盘价（停牌/退市），继续持有：{', '.join(frozen)}")
    print(f"\n[卖出 {len(sell_df)} 笔]")
    if len(sell_df):
        print(sell_df.to_string(index=False))
    else:
        print("  （今日无到期）")
    print(f"\n[买入 {len(buy_df)} 笔]")
    if len(buy_df):
        print(buy_df.to_string(index=False))
    else:
        print("  （今日无新建仓）")
    if skipped:
        print(f"\n[跳过 {len(skipped)} 条]")
        for c, n, why in skipped[:40]:
            print(f"  {c} {n:<6} {why}")
        if len(skipped) > 40:
            print(f"  … 另 {len(skipped) - 40} 条")

    if not args.commit:
        print("\n[dry-run] 未写任何文件。确认无误后加 --commit。")
        return 0

    orders.mkdir(parents=True, exist_ok=True)
    ds = t1.strftime("%Y%m%d")
    buy_df.to_csv(orders / f"buy_{ds}.csv", index=False, encoding="utf-8-sig")
    sell_df.to_csv(orders / f"sell_{ds}.csv", index=False, encoding="utf-8-sig")
    save_state(state_p, new_state)
    log = {"asof": str(asof.date()), "trade_date": ds,
           "run_at": datetime.now().isoformat(timespec="seconds"),
           "n_buy": len(buy_df), "n_sell": len(sell_df), "n_skip": len(skipped),
           "skipped": [{"code": c, "name": n, "why": w} for c, n, w in skipped],
           "extend": extend}
    (orders / f"run_{asof.strftime('%Y%m%d')}.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {orders/('buy_'+ds+'.csv')}\n[out] {orders/('sell_'+ds+'.csv')}"
          f"\n[out] {state_p}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    st = load_state(Path(args.out) / "state.json")
    if not st["positions"]:
        print("无持仓（state.json 为空或不存在）")
        return 0
    cal = load_cal()
    asof = cal.max()
    mk = load_market(asof - pd.Timedelta(days=250))
    df = pd.DataFrame(st["positions"])
    px = latest_px(mk, asof, df["code"].tolist())
    df["last"] = df["code"].map(px)
    df["mv"] = df["shares"] * df["last"]
    df["ret"] = df["last"] / df["price"] - 1
    i_asof = int(np.searchsorted(cal.values, np.datetime64(asof)))
    df["days_left"] = df["sell_date"].map(
        lambda s: int(np.searchsorted(cal.values, np.datetime64(s))) - i_asof)
    mv = float(df["mv"].sum())
    print(f"asof={asof.date()}  本金 {st['capital']:,.0f}  现金 {st['cash']:,.0f}  "
          f"持仓市值 {mv:,.0f}  合计 {mv + st['cash']:,.0f}")
    print(f"持仓 {len(df)} 只  平均单笔收益 {df['ret'].mean():+.2%}")
    print(df.sort_values("sell_date")[
        ["code", "name", "shares", "price", "last", "ret", "buy_date", "sell_date", "days_left"]
    ].to_string(index=False))
    over = df[df["days_left"] < 0]
    if len(over):
        print(f"\n[warn] {len(over)} 笔已过卖出日却没卖掉（days_left<0）——"
              f"说明中间有交易日没跑 signal，或当天卖出单没成交。"
              f"补跑一天 signal 即可把它们排进卖出清单。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="inc_pure 实盘出单器")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("signal", help="生成 T+1 开盘买卖清单")
    s.add_argument("--asof", required=True, help="信号日 YYYYMMDD（= 公告日，盘后跑）")
    s.add_argument("--capital", type=float, default=0.0, help="初始本金（仅首次）")
    s.add_argument("--max-open", type=int, default=30, help="并发持仓上限（默认 30）")
    s.add_argument("--max-single-pct", type=float, default=0.05,
                   help="单笔占本金上限（默认 5%% = 1/20 等权）")
    s.add_argument("--max-adv-pct", type=float, default=0.01,
                   help="单笔 ≤ T 日成交额的百分比（默认 1%%）")
    s.add_argument("--gate-status", action="store_true", default=True,
                   help="启用 ST/停牌闸门（默认开）")
    s.add_argument("--no-gate-status", dest="gate_status", action="store_false")
    s.add_argument("--commit", action="store_true", help="落盘（默认 dry-run）")
    s.add_argument("--out", default=str(DEFAULT_OUT), help="产物目录（默认 reports/inc_pure_live）")
    s.set_defaults(func=cmd_signal)

    c = sub.add_parser("check", help="查看当前持仓与到期")
    c.add_argument("--out", default=str(DEFAULT_OUT), help="产物目录")
    c.set_defaults(func=cmd_check)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
