"""
ETF 盘中决策时点探针（14:50 / 14:55 / 收盘）
===============================================

用途：给「盘中出信号」改造（模式 B，见 `reports/etf_contest5/README.md` §9.2）定
**决策时点**。14:55 决策只剩 120 秒到 14:57 收盘集合竞价；14:50 有 420 秒。提前
5 分钟换来 5 倍预算，代价是价格离收盘更远 —— 本探针量化这个代价。

三件事：

1. **耗时**：`MarketData.query_snapshot(code_list, d, d)` 拉全池单日快照的真实耗时。
   ⚠️ 接口签名只有日期（int YYYYMMDD）、**没有时间窗参数** ⇒ 盘中拉当天同样是
   拉「开盘到调用时刻」的全部条数，耗时与决策时点无关，只能整日拉完再截断。
2. **价格偏离**：各时点 `last` 相对当日 `close` 的偏离分布。
3. **信号翻转**：把 `close` 换成各时点的 `last` 重算 `mom12_1` / `pre10` 筛子 /
   Top3 选标，看最终持仓是否改变。

用法::

    python -m scripts.portfolio.etf_intraday_clock_probe --date 20261029
    python -m scripts.portfolio.etf_intraday_clock_probe --dates 20261022:20261029 \\
        --json reports/etf_contest5/clock_probe.json

⚠️ 每次运行都会 login 一次 AmazingData；单账号互斥，不要与出榜/更新任务并发。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

#: 待测决策时点（时分秒）。14:57 是收盘集合竞价开始，故 14:55 是原方案的时点。
#: 可用 ``--clocks`` 覆盖（如 ``14:45,14:50,14:52,14:53,14:55,14:57``）做逐分钟权衡扫描。
CLOCKS = {"c1450": "14:50:00", "c1455": "14:55:00", "c1457": "14:57:00"}


def _pool() -> list[str]:
    """比赛可投池（本地池 − EXCLUDED_5）。"""
    from data.etf_universe import ETF_CANDIDATES
    from scripts.portfolio.etf_contest_spec import EXCLUDED_5
    return [c for c in ETF_CANDIDATES if c not in set(EXCLUDED_5)]


def _snap(ad, codes: list[str], day: int,
          cache_dir: Path | None = None) -> tuple[dict[str, pd.DataFrame], float]:
    """拉单日快照，返回 ({code: DataFrame(trade_time 索引, 含 last)}, 拉取秒数)。

    `cache_dir` 非空时按 ``snapshot_{day}.parquet`` 缓存整日原始快照：
    单日 45 只约 55~57 秒，多日扫描时重复拉取纯属浪费。
    """
    import pyarrow.parquet as pq

    if cache_dir is not None:
        cp = cache_dir / f"snapshot_{day}.parquet"
        if cp.exists():
            t0 = time.perf_counter()
            raw = pq.read_table(cp).to_pandas()
            dt = time.perf_counter() - t0
            out = {}
            for code, df in raw.groupby("code", sort=False):
                out[code] = df.set_index("ts").sort_index()
            return out, -dt          # 负值 = 命中缓存（秒数为读盘时间）

    t0 = time.perf_counter()
    rsp = ad.query_snapshot(codes, day, day)
    dt = time.perf_counter() - t0
    data = rsp[0] if isinstance(rsp, (list, tuple)) else rsp
    inner = data.get(day, data) if isinstance(data, dict) else {}
    out, frames = {}, []
    for code, df in inner.items():
        if df is None or not len(df):
            continue
        if "trade_time" in df.columns:
            idx = pd.to_datetime(df["trade_time"], errors="coerce")
            df = df.set_index(idx)
            df.index.name = "ts"
        df = df.sort_index()
        out[code] = df
        if cache_dir is not None:
            keep = df[["code", "last", "close", "volume", "amount",
                       "trading_phase_code"]].copy()
            frames.append(keep)
    if cache_dir is not None and frames:
        import pyarrow as pa
        cache_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(pd.concat(frames).reset_index()),
                       cache_dir / f"snapshot_{day}.parquet", compression="snappy")
    return out, dt


def _close_at(df: pd.DataFrame, hhmmss: str) -> float | None:
    """取 <= hhmmss 的最后一条 `last`。"""
    if df is None or not len(df):
        return None
    t = pd.Timestamp(hhmmss).time()
    sub = df[df.index.time <= t]
    if not len(sub):
        return None
    v = float(sub["last"].iloc[-1]) if "last" in sub else np.nan
    return v if np.isfinite(v) and v > 0 else None


def _ref_close(snaps: dict[str, pd.DataFrame], codes: list[str],
               daily: pd.DataFrame | None, ts: pd.Timestamp) -> pd.Series:
    """当日参照收盘价。

    优先用快照自身 15:00 之后的最后一条 `last`（同一数据源，避免与日线口径差异）；
    退回到本地日线（当日未落盘时用前一日，仅为兜底，会在输出里标注）。
    """
    ref = {}
    for c in codes:
        ref[c] = _close_at(snaps.get(c), "15:00:00")
    s = pd.Series(ref, dtype=float)
    if daily is not None and ts in daily.index:
        s = s.where(s.notna(), daily.loc[ts].reindex(s.index))
    return s



def _daily_close(codes: list[str]) -> pd.DataFrame:
    """本地日线原始收盘（直接读 parquet，不登录数据源）。"""
    import pyarrow.parquet as pq

    from config import Config
    p = Path(Config.paths()["cache_root"]) / "daily_etf.parquet"
    if p.exists():
        d = pq.read_table(p).to_pandas()
    else:                                   # 兜底：走 DataCache（需 login）
        from data.cache import DataCache
        from data.datasource import create_datasource
        d = DataCache(create_datasource()).read_daily("etf")
    if "date" not in d.columns:
        d = d.reset_index()
    wide = d.pivot_table(index="date", columns="code", values="close", aggfunc="last")
    # parquet 里 date 已是 datetime；旧缓存可能是 int YYYYMMDD，两种都要认
    if pd.api.types.is_datetime64_any_dtype(wide.index):
        wide.index = pd.to_datetime(wide.index)
    else:
        wide.index = pd.to_datetime(wide.index.astype(str), format="%Y%m%d",
                                    errors="coerce")
    wide = wide[wide.index.notna()].sort_index()
    return wide.reindex(columns=[c for c in codes if c in wide.columns])


def probe(dates: list[int], verbose: bool = True,
          cache_dir: Path | None = None,
          clocks: dict[str, str] | None = None) -> dict:
    from data.datasource import create_datasource

    clocks = clocks or CLOCKS
    codes = _pool()
    ds = create_datasource()
    ad = ds._market
    daily = _daily_close(codes)

    rows, tcost, ref_src, failed = [], [], {}, []
    base = daily.copy()
    for day in dates:
        try:
            snaps, dt = _snap(ad, codes, day, cache_dir)
        except Exception as exc:                      # 单日失败不中断整批
            failed.append({"date": int(day), "err": repr(exc)[:200]})
            print(f"  {day}: ⚠️ 拉取失败 {repr(exc)[:120]}", flush=True)
            continue
        tcost.append(dt)
        if verbose:
            flag = "缓存" if dt < 0 else "拉取"
            print(f"  {day}: {len(snaps)}/{len(codes)} 只, {flag} {abs(dt):.1f}s",
                  flush=True)
        ts = pd.Timestamp(str(day))
        ref = _ref_close(snaps, codes, daily, ts)
        ref_src[str(day)] = "snapshot" if ref.notna().any() else "daily"
        # 当日不在本地日线时（当天尚未跑批），用快照收盘价补一行，供 mom/pre10 计算
        if ts not in base.index:
            base.loc[ts] = ref
            base = base.sort_index()
        else:
            base.loc[ts] = base.loc[ts].where(base.loc[ts].notna(), ref)

        for tag, hhmmss in clocks.items():
            for code in codes:
                px = _close_at(snaps.get(code), hhmmss)
                r = float(ref.get(code, np.nan))
                if px is None or not np.isfinite(r) or r <= 0:
                    continue
                rows.append({"date": day, "clock": tag, "code": code,
                             "px": px, "close": r, "dev": px / r - 1.0})

    df = pd.DataFrame(rows)
    out: dict = {"n_codes": len(codes), "ref_source": ref_src,
                 "failed": failed, "fetch_sec": tcost,
                 "fetch_sec_mean": float(np.mean([abs(t) for t in tcost if t > 0]))
                 if any(t > 0 for t in tcost) else None,
                 "n_samples": len(df)}
    if not len(df):
        return out

    # ---- 2. 价格偏离 ----
    dev = {}
    for tag, g in df.groupby("clock"):
        a = g["dev"].abs()
        dev[tag] = {"mean": float(a.mean()), "median": float(a.median()),
                    "p95": float(a.quantile(0.95)), "p99": float(a.quantile(0.99)),
                    "max": float(a.max()),
                    "unchanged_rate": float((a < 1e-9).mean())}
    out["deviation"] = dev

    # ---- 3. 信号翻转：用各时点价替换当日收盘重算 ----
    # mom12_1 = close.shift(21)/close.shift(252) - 1；pre10 = 剔除 close/shift(20)-1 > 10%
    # （回测用后复权，这里只有原始价 ⇒ 只用「当日一行的替换」做横向比较，
    #   三时点用同一套原始价基准，故偏离可比；绝对水平不作结论。）
    flip = {}
    for tag, hhmmss in clocks.items():
        g = df[df["clock"] == tag]
        # 逐日构造「当日价 = 时点价」的面板
        for day, gg in g.groupby("date"):
            ts = pd.Timestamp(str(day))
            if ts not in base.index:
                continue
            repl = gg.set_index("code")["px"]
            days = base.index[base.index <= ts]
            if len(days) < 253:
                continue
            i_now, i20, i252 = days[-1], days[-21], days[-253]
            c20 = base.loc[i20]
            c252 = base.loc[i252]
            c_now = base.loc[i_now].copy()
            c_true = base.loc[i_now]
            c_now.loc[repl.index.intersection(c_now.index)] = repl
            mom = c_now / c252 - 1.0
            mom_true = c_true / c252 - 1.0
            pre = c_now / c20 - 1.0
            pre_true = c_true / c20 - 1.0
            valid = pre.notna() & pre_true.notna()
            flipped = ((pre[valid].abs() > 0.10) != (pre_true[valid].abs() > 0.10))
            # 🚨 生产口径是「先过 pre10 筛，再按动量排序取 Top3」：
            # 直接拿未过滤的 Top3 对比会把「筛子本身的作用」误记成「时点造成的改变」
            # （2026-10-29 踩过：三个时点都显示 7/15 日改变）。
            fin_now = pre[pre.abs() <= 0.10].dropna().index
            fin_true = pre_true[pre_true.abs() <= 0.10].dropna().index
            top = mom[mom.index.isin(fin_now)].dropna().nlargest(3).index.tolist()
            top_true = (mom_true[mom_true.index.isin(fin_true)]
                        .dropna().nlargest(3).index.tolist())
            r = flip.setdefault(tag, {"pre10_flip": 0, "pre10_n": 0,
                                      "top3_changed": 0, "days": 0,
                                      "cand_changed": 0,
                                      "mom_dev": [], "rank_corr": [],
                                      "top3_lists": []})
            r["pre10_flip"] += int(flipped.sum())
            r["pre10_n"] += int(valid.sum())
            r["top3_changed"] += int(top != top_true)
            r["cand_changed"] += int(set(fin_now) != set(fin_true))
            r["days"] += 1
            both = pd.concat([mom, mom_true], axis=1, keys=["a", "b"]).dropna()
            if len(both) > 5:
                r["mom_dev"].append(float((both["a"] - both["b"]).abs().mean()))
                r["rank_corr"].append(float(both["a"].corr(both["b"], method="spearman")))
            r["top3_lists"].append({"date": int(day), "intraday": top,
                                    "close": top_true})

    for tag, r in flip.items():
        r["mom_dev_mean"] = float(np.mean(r.pop("mom_dev"))) if r["mom_dev"] else None
        r["rank_corr_min"] = float(np.min(r["rank_corr"])) if r["rank_corr"] else None
        r.pop("rank_corr")
    out["flip"] = flip
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ETF 盘中决策时点探针（14:50/14:55/收盘）")
    ap.add_argument("--date", type=int, default=None, help="单个交易日 YYYYMMDD")
    ap.add_argument("--dates", default=None, help="区间 begin:end（YYYYMMDD）")
    ap.add_argument("--json", default=None, help="结果落盘路径")
    ap.add_argument("--cache-dir", default=None,
                    help="整日原始快照缓存目录（多日扫描时避免重复 55s/日）")
    ap.add_argument("--rebalance-only", action="store_true",
                    help="只测周频调仓日（每周首个交易日）")
    ap.add_argument("--clocks", default=None,
                    help="逗号分隔的决策时点 HH:MM（默认 14:50,14:55,14:57）；"
                         "给 14:45,14:50,14:52,14:53,14:55,14:57 可得逐分钟权衡曲线")
    args = ap.parse_args(argv)

    clocks = CLOCKS
    if args.clocks:
        clocks = {}
        for hhmm in args.clocks.split(","):
            hhmm = hhmm.strip()
            tag = "c" + hhmm.replace(":", "")
            clocks[tag] = hhmm if len(hhmm) > 5 else f"{hhmm}:00"

    if args.dates:
        b, e = args.dates.split(":")
        rng = pd.bdate_range(pd.Timestamp(b), pd.Timestamp(e))
        dates = [int(d.strftime("%Y%m%d")) for d in rng]
    elif args.date:
        dates = [args.date]
    else:
        dates = [int(pd.Timestamp.today().strftime("%Y%m%d"))]

    if args.rebalance_only:
        # 周频 = 每周首个交易日（与 etf_champ_signals._next_rebalance 同义）
        seq = pd.to_datetime([str(d) for d in dates], format="%Y%m%d")
        first = pd.Series(seq).groupby(seq.isocalendar().week.to_numpy()).min()
        dates = [int(t.strftime("%Y%m%d")) for t in first]

    print(f"决策时点探针：{len(dates)} 日 × {len(_pool())} 只", flush=True)
    out = probe(dates, cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                clocks=clocks)

    if out.get("fetch_sec_mean"):
        fresh = [t for t in out["fetch_sec"] if t > 0]
        print(f"\n快照拉取耗时：均值 {out['fetch_sec_mean']:.1f}s "
              f"(新拉 {len(fresh)} 日, "
              f"范围 {min(fresh):.1f}~{max(fresh):.1f}s)", flush=True)
    print("\n=== 价格偏离（|时点价/收盘价−1|）===", flush=True)
    for tag, d in out.get("deviation", {}).items():
        print(f"  {tag}: 均值 {d['mean']:.3%} 中位 {d['median']:.3%} "
              f"p95 {d['p95']:.3%} p99 {d['p99']:.3%} 最大 {d['max']:.3%} "
              f"不变 {d['unchanged_rate']:.1%}", flush=True)
    print("\n=== 信号翻转（vs 用收盘价）===", flush=True)
    for tag, r in out.get("flip", {}).items():
        md = r["mom_dev_mean"]
        rc = r["rank_corr_min"]
        print(f"  {tag}: pre10 翻转 {r['pre10_flip']}/{r['pre10_n']} "
              f"({r['pre10_flip']/max(r['pre10_n'],1):.2%}) · "
              f"Top3 改变 {r['top3_changed']}/{r['days']} 日 · "
              f"mom 偏离 {md:.3%} · 秩相关最低 {rc:.5f}"
              if md is not None and rc is not None else
              f"  {tag}: 样本不足", flush=True)

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        import json
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
                     encoding="utf-8")
        print(f"\n  → {p}", flush=True)


if __name__ == "__main__":
    main()
