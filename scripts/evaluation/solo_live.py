"""yqcore-solo 生产持仓生成器（2026-11-10 建立）。

**输入** = 出榜链的 ``ranking_<ds>.csv``（solo 信号口径：882 ortho ·
``gbdt_mid13`` = h1:h5:h10:h20 = 2:1:1:2 秩平均），**输出** = 当期实际可下单
持仓（TopK + 每申万一级 ≤1 只 + 整手 100 股 + 指定资金）。

为什么需要它
------------
``scripts/pipelines/alla_daily_rank.py`` 的执行层只有两种产物：①
``picks``（top-frac + 分位缓冲带）②``--deliverable-k``（**无行业约束**的
可交易 TopN）。而 ``yqcore-solo`` 的定义是
``SectorCappedBufferedTopKLongOnly(k=8, exit_mult=1.0, max_per_sector=1)``
**+ 整手 100 股** —— 两者都不在出榜链里（2026-11-10 复核：见
``reports/yq_core/README.md`` §未做项）。

本脚本把**回测用的同一个策略类**接到出榜链的排名上：不另写一套选股逻辑，
所以「回测口径」与「今日持仓」走同一条代码路径。

口径（与 ``scripts/evaluation/caliber.run_once`` 逐条对齐）
----------------------------------------------------------
- 信号日 T = 每月第一个交易日；**成交 T+1 开盘**（production profile）。
- 整手股数 ``trunc(w·capital/close_T/lot)·lot`` —— 与引擎
  ``backtest.engine._apply_lot_rounding`` **同式**：用 **T 收盘价**定量、
  T+1 开盘价成交（``run_once`` 的 ``price_panel=close_raw.ffill().shift(1)``
  正是这个语义，``shift(1)`` 是无前视的关键）。
- 可交易性取 ``ranking.tradable``（信号日停牌 / ST / 收盘封板 = 买不进），
  与回测的 ``sig.shift(1).where(mask)`` 同为「进候选前先掩码」。
- 缓冲带状态：``exit_mult=1.0`` ⇒ 保留门槛 ``k·1.0`` = 建仓门槛 ``k``，
  **退化为无缓冲**；但上一期持仓仍要喂进去 —— 同行业撞车时「谁先占名额」
  依赖它（Rank 6 的旧仓会挤掉 Rank 3 的同行业新丁）。故用 ``seed_holdings``
  按 ``holdings_*.csv`` 时序链式传递，冷启动 = 空集（等价回测首期）。

用法::

    python -m scripts.evaluation.solo_live --date 20261102 \
        --ranking reports/alla_daily_solo/ranking_20261102.csv \
        --capital 100000

产物落 ``reports/yq_core/live/``：``holdings_<ds>.csv``（含股数/金额）、
``orders_<ds>.csv``（仅「买什么、几股、限价参考」）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from strategy.examples import SectorCappedBufferedTopKLongOnly  # noqa: E402

LIVE_DIR = ROOT / "reports" / "yq_core" / "live"


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------
def load_ranking(path: str | Path) -> pd.DataFrame:
    """读出榜链的 ``ranking_<ds>.csv``（index = 6 位代码）。"""
    df = pd.read_csv(Path(path), dtype={"code": str}, index_col="code")
    df.index = df.index.astype(str)
    for col in ("score", "pct_rank"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "tradable" in df.columns:
        df["tradable"] = df["tradable"].astype(str).str.lower().isin(
            ("true", "1", "1.0"))
    return df


def load_prices(cols: tuple[str, ...] = ("close",)) -> tuple[dict[str, pd.DataFrame],
                                                             pd.DatetimeIndex]:
    """从日线缓存取**未复权**价格宽表（date × code）+ 交易日历。"""
    fp = Path(str(Config.cache()["root"])) / "daily_all_a.parquet"
    px = pd.read_parquet(fp, columns=list(cols))
    dates = pd.to_datetime(px.index.get_level_values(0))
    px.index = pd.MultiIndex.from_arrays(
        [dates, px.index.get_level_values(1).astype(str)])
    out = {c: px[c].unstack(level=1) for c in cols}
    return out, pd.DatetimeIndex(dates.unique()).sort_values()


# ---------------------------------------------------------------------------
# 选股（复用回测策略类）
# ---------------------------------------------------------------------------
def refresh_tradable(ranking: pd.DataFrame,
                     date: pd.Timestamp) -> pd.Series:
    """用**信号日实时状态**重算可交易性，覆盖 ``ranking.csv`` 里可能过期的列。

    为什么需要（2026-11-10 事故）
    ---------------------------
    ``ranking.tradable`` 是出榜链在**出榜当时**算的。一旦状态表断更，那一列
    就永久停在旧口径上：本次状态表止于 10-22，11-02 的排名里 4 只 ST 全是
    ``tradable=True``，直接进持仓（Top10 里 5 只 ST）。持仓生成器如果只信
    这份 CSV，修好出榜链也救不回已经落盘的榜。

    这里直接调用出榜链同一个 ``signal_day_tradable``（同函数、同语义 ——
    状态表缺失日会降级为「日线涨跌停 + 最近可得 ST」），不另写判断。
    """
    from scripts.pipelines.alla_daily_rank import signal_day_tradable

    px, _ = load_prices(("close",))
    close_t = px["close"].reindex(columns=ranking.index)
    if date not in close_t.index:
        raise KeyError(f"日线缓存无信号日 {date.date()}，无法重算可交易性")
    one = pd.DataFrame([close_t.loc[date]], index=pd.DatetimeIndex([date]))
    return (signal_day_tradable(one, date)
            .reindex(ranking.index).fillna(False).astype(bool))


def solo_weights(ranking: pd.DataFrame, date: pd.Timestamp, k: int = 8,
                 max_per_sector: int = 1, prev: set | None = None,
                 gross_exposure: float = 1.0) -> pd.Series:
    """solo 目标权重（等权，行业去重，缓冲带由 ``prev`` 驱动）。"""
    scores = ranking["score"].dropna()
    if "tradable" in ranking.columns:
        scores = scores.where(ranking.loc[scores.index, "tradable"].fillna(False))
    sector = (ranking["industry_l1"].reindex(scores.index).fillna("未知")
              .astype(str) if "industry_l1" in ranking.columns
              else pd.Series("未知", index=scores.index))
    sectors = pd.DataFrame([sector], index=pd.DatetimeIndex([date]))
    strat = SectorCappedBufferedTopKLongOnly(
        sectors, k=k, exit_mult=1.0, max_per_sector=max_per_sector,
        gross_exposure=gross_exposure)
    strat.seed_holdings(prev or set())
    return strat.get_weights_at(date, scores)


def lot_sizing(weights: pd.Series, prices: pd.Series,
               capital: float, lot: int = 100) -> pd.DataFrame:
    """整手取整（``trunc`` 向零，与 ``_apply_lot_rounding`` 同式）。"""
    px = prices.reindex(weights.index).astype(float)
    out = pd.DataFrame({"weight_target": weights})
    out["price_ref"] = px
    ok = np.isfinite(px.to_numpy()) & (px.to_numpy() > 0)
    shares = np.zeros(len(out))
    w = weights.to_numpy()
    shares[ok] = np.trunc(w[ok] * capital / px.to_numpy()[ok] / lot) * lot
    out["shares"] = shares.astype(int)
    out["amount"] = (out["shares"] * px).to_numpy()
    gross = float(w.sum())
    out["weight_actual"] = (out["amount"] / capital).to_numpy()
    out.attrs["invest"] = float(out["amount"].sum() / capital)
    out.attrs["gross_target"] = gross
    return out


# ---------------------------------------------------------------------------
# 时序状态
# ---------------------------------------------------------------------------
def load_prev_holdings(live_dir: Path, before: pd.Timestamp) -> tuple[set, str]:
    """读**严格早于** ``before`` 的最近一期持仓（无则空集 = 冷启动）。"""
    ds = before.strftime("%Y%m%d")
    cands = sorted((p for p in live_dir.glob("holdings_*.csv")
                    if p.stem.rsplit("_", 1)[-1].isdigit()
                    and p.stem.rsplit("_", 1)[-1] < ds),
                   key=lambda p: p.stem.rsplit("_", 1)[-1])
    if not cands:
        return set(), ""
    df = pd.read_csv(cands[-1], dtype={"code": str}, index_col="code")
    return set(df.index.astype(str)), cands[-1].name


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="yqcore-solo 生产持仓生成器")
    ap.add_argument("--ranking", required=True, help="出的榜 ranking_<ds>.csv")
    ap.add_argument("--date", required=True, help="信号日 YYYYMMDD")
    ap.add_argument("--capital", type=float, default=100000.0, help="资金（元）")
    ap.add_argument("--k", type=int, default=8, help="目标只数（solo 定版 8）")
    ap.add_argument("--max-per-sector", type=int, default=1,
                    help="每申万一级最多几只（solo 定版 1）")
    ap.add_argument("--lot", type=int, default=100, help="最小交易单位")
    ap.add_argument("--prev", default=None,
                    help="显式指定上期持仓 CSV（缺省 = 自动取最近一期）")
    ap.add_argument("--exec-date", default=None,
                    help="成交日（缺省 = 信号日的下一个交易日，T+1 开盘）")
    ap.add_argument("--mark-date", default=None,
                    help="估值日（缺省 = 缓存最新交易日）：填了才算浮盈")
    ap.add_argument("--out-dir", default=None, help="产物目录（缺省 reports/yq_core/live）")
    ap.add_argument("--no-refresh-tradable", action="store_true",
                    help="沿用 ranking.csv 的 tradable 列（默认改为信号日实时重算）")
    args = ap.parse_args()

    date = pd.Timestamp(args.date)
    out_dir = Path(args.out_dir) if args.out_dir else LIVE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking = load_ranking(args.ranking)
    traded = {"refreshed": False}
    if not args.no_refresh_tradable:
        new_ok = refresh_tradable(ranking, date)
        old_ok = (ranking["tradable"].astype(bool) if "tradable" in ranking.columns
                  else pd.Series(True, index=ranking.index))
        flipped = old_ok & ~new_ok
        ranking["tradable"] = new_ok
        traded = {"refreshed": True,
                  "n_now_blocked": int((~new_ok).sum()),
                  "n_flipped": int(flipped.sum()),
                  "flipped_codes": ranking.index[flipped].tolist()}
        if int(flipped.sum()):
            print(f"可交易性重算：{int(flipped.sum())} 只由「可交易」翻为「不可交易」")
    if args.prev:
        prev = set(pd.read_csv(args.prev, dtype={"code": str},
                               index_col="code").index.astype(str))
        prev_src = Path(args.prev).name
    else:
        prev, prev_src = load_prev_holdings(out_dir, date)

    weights = solo_weights(ranking, date, k=args.k,
                           max_per_sector=args.max_per_sector, prev=prev)
    px, tdays = load_prices(("open", "close"))
    close_t = px["close"].loc[date]
    sized = lot_sizing(weights, close_t, args.capital, lot=args.lot)

    # 成交日 = 信号日的下一个交易日；估值日 = 缓存最新交易日（可覆盖）
    later = tdays[tdays > date]
    exec_d = pd.Timestamp(args.exec_date) if args.exec_date else (
        later[0] if len(later) else pd.NaT)
    mark_d = pd.Timestamp(args.mark_date) if args.mark_date else tdays[-1]
    if pd.notna(exec_d) and exec_d in px["open"].index:
        fill = px["open"].loc[exec_d].reindex(sized.index)
    else:
        fill = pd.Series(np.nan, index=sized.index)
    last = px["close"].loc[mark_d].reindex(sized.index) if mark_d in px["close"].index \
        else pd.Series(np.nan, index=sized.index)
    sized["open_exec"] = fill
    sized["cost"] = sized["shares"] * fill
    sized["last_close"] = last
    sized["mv"] = sized["shares"] * last
    sized["ret_since_exec"] = last / fill - 1.0
    sized["pnl"] = sized["mv"] - sized["cost"]

    hold = ranking.reindex(sized.index)
    sized["rank"] = hold["rank"] if "rank" in hold.columns else np.nan
    sized["name"] = hold["name"] if "name" in hold.columns else ""
    sized["industry"] = hold["industry_l1"] if "industry_l1" in hold.columns else ""
    sized = sized.sort_values("weight_target", ascending=False)

    ds = date.strftime("%Y%m%d")
    hpath = out_dir / f"holdings_{ds}.csv"
    sized.index.name = "code"
    cols = ["rank", "name", "industry", "weight_target", "price_ref",
            "shares", "amount", "weight_actual", "open_exec", "cost",
            "last_close", "mv", "ret_since_exec", "pnl"]
    sized[cols].to_csv(hpath, encoding="utf-8-sig")
    sized[cols].to_csv(out_dir / f"orders_{ds}.csv", encoding="utf-8-sig")

    spend = float(sized["cost"].sum())
    mv = float(sized["mv"].sum())
    meta = {
        "signal_date": ds,
        "exec_date": None if pd.isna(exec_d) else str(exec_d.date()),
        "mark_date": str(mark_d.date()),
        "capital": args.capital,
        "k": args.k,
        "max_per_sector": args.max_per_sector,
        "lot": args.lot,
        "n_holdings": int(len(sized)),
        "cash_ratio": round(1.0 - spend / args.capital, 6),
        "portfolio_ret": round(mv / spend - 1.0, 6) if spend > 0 else None,
        "prev_source": prev_src,
        "prev_n": len(prev),
        "ranking": str(args.ranking),
        "tradable": traded,
    }
    (out_dir / f"meta_{ds}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"信号日 {ds} → 成交 {meta['exec_date']}（T+1 开盘）| 估值 {meta['mark_date']}")
    print(f"资金 {args.capital:,.0f} | 目标 {args.k} 只 | 实持 {len(sized)} 只 "
          f"| 投入 {spend:,.0f}（{spend / args.capital * 100:.2f}%）"
          f"| 持仓市值 {mv:,.0f}（组合 {meta['portfolio_ret'] * 100:+.2f}%）")
    print(f"上期持仓 ← {prev_src or '无（冷启动）'}（{len(prev)} 只）")
    show = sized[cols].copy()
    for c in ("weight_target", "weight_actual", "ret_since_exec"):
        show[c] = (show[c] * 100).round(2)
    for c in ("amount", "cost", "mv", "pnl"):
        show[c] = show[c].round(0)
    print(show.to_string())
    print(f"\n产物: {hpath}")


if __name__ == "__main__":
    main()
