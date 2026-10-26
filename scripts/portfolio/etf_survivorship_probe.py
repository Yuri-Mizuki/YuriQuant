"""
ETF 轮动幸存者偏差一次性探针
============================

量化「现 41 只 ETF 池只含仍在交易的标的」带来的幸存者偏差：

  - 全 ETF 清单 = SDK 当前 ``get_code_list("EXTRA_ETF")`` ∪ 历史
    ``get_hist_code_list("EXTRA_ETF")``（后者含已退市/清盘）；
  - 退市判定 = 末交易日 < 2026-06-30；
  - PIT 动态池 = 上市满 3 年（首个交易日 + 3×365.25 天）且滚动 60 日均成交额
    ≥ 3000 万元，逐日施加（不预先用今天的流动性挑池）；
  - A（旧口径）= 现 41 只池 + 仅流动性的 PIT 掩码（与
    ``scripts/portfolio/etf_rotation_probe.py`` 的 ``dual_ivvt15_m5`` 一致）；
  - B（PIT 全池）= 上述「上市满 3 年 + 流动性」动态池；
  - **选标前置掩码**：不可入池标的的因子置 NaN，策略只在可投集合内排序，
    不再依赖引擎事后清零 + 重新归一化（那会让持仓只数被动缩水）；
  - 两者同引擎同成本：12-1 动量 + 绝对动量 + inverse-vol + 15% 波动目标，
    月频、TopK=5、初始 100 万，``EtfCosts(commission_rate=5e-4, min=0.1, slip=5bp)``。

判据（先验固定）：
  - 复现校验：A 的 Sharpe 与现基线 0.62 差 ≤ 0.05、年化与 7.14% 差 ≤ 0.5pp、
    平均持仓 ≈ 5；
  - 主判据：B − A 的 Sharpe 下降 ≤ 0.10 → 幸存者偏差可忽略；
    > 0.20 → 结论需大幅打折；其间 → 需标注；
  - 退市标的中实际进过 PIT 池的 < 5 只 ⇒ 样本不足以量化（另报最高 60 日均额分位）。

用法::

    python -m scripts.portfolio.etf_survivorship_probe

产物：``reports/etf_survivorship_probe/{summary.csv, panels.csv, run_config.json}``
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import EtfCosts
from backtest.engine import VectorBacktest
from config import Config
from data.datasource import create_datasource
from data.etf_universe import ETF_CANDIDATES
from strategy.etf_rotation import EtfRotationAdvanced

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "etf_survivorship_probe"

MIN_AMOUNT = 3e7              # 滚动 60 日均成交额下限（元）
MIN_YEARS = 3.0               # 上市年限
DELIST_CUTOFF = "2026-06-30"  # 末交易日早于此 ⇒ 退市
SEGMENTS = [(2020, 2021), (2022, 2023), (2024, 2026)]

STRATEGY = dict(top_k=5, absolute_filter=True, weighting="inverse_vol",
                target_vol=0.15, cash_code="CASH")
# 与正式探针 etf_rotation_probe 一致：佣金 5bp（最低 0.1 元）+ 滑点 5bp
COSTS = dict(commission_rate=0.0005, commission_min=0.1, slippage_bp=5.0)
BASELINE_SHARPE = 0.62        # reports/etf_rotation_probe 的 dual_ivvt15_m5
BASELINE_ANNUAL = 0.0714


# ---------------------------------------------------------------------------
# SDK 调用（单点登录互斥：本进程内顺序调用，超时用 daemon 线程兜底）
# ---------------------------------------------------------------------------
def _call(fn, seconds: float = 420.0):
    """SDK 可能挂死（无超时），用 daemon 线程 + queue 硬超时。"""
    q: queue.Queue = queue.Queue()

    def _run():
        try:
            q.put(("ok", fn()))
        except BaseException as exc:  # noqa: BLE001 - 拉数要看到任何异常
            q.put(("err", f"{type(exc).__name__}: {exc}"))

    threading.Thread(target=_run, daemon=True).start()
    try:
        return q.get(timeout=seconds)
    except queue.Empty:
        return ("timeout", f">{seconds:.0f}s 无返回")


def _cache_path() -> Path:
    return Path(str(Config.cache()["root"])) / "etf_all_kline.parquet"


def _write_cache(path: Path, frames: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(frames)
    df = (df[~df.index.duplicated(keep="last")]
          .sort_index())
    df.to_parquet(path, compression="snappy")
    return df


def fetch_klines(ds, codes: list[str], begin: int, end: int, path: Path,
                 batch: int = 80, timeout: float = 420.0) -> pd.DataFrame:
    """分批拉全 ETF 清单日 K 线（close/amount），断点续拉并落盘。

    已落盘过的代码跳过；单批失败时**换一种批量策略**（拆成 20 只一组）重试一次，
    仍失败则记录跳过。返回合并后的长表 ``(date, code)``。
    """
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    have = set(existing.index.get_level_values("code")) if len(existing) else set()
    todo = [c for c in codes if c not in have]
    frames: list[pd.DataFrame] = [existing] if len(existing) else []
    failed: list[str] = []
    n_batch = max((len(todo) + batch - 1) // batch, 1)
    t0 = time.time()
    print(f"[kline] 清单 {len(codes)} 只 | 缓存已有 {len(have)} 只 | 待拉 {len(todo)} 只 "
          f"| {n_batch} 批", flush=True)

    def _add(part: pd.DataFrame) -> None:
        if part is None or not len(part):
            return
        cols = [c for c in ("close", "amount") if c in part.columns]
        frames.append(part[cols].astype(float))

    for bi in range(n_batch):
        sub = todo[bi * batch:(bi + 1) * batch]
        st, val = _call(lambda: ds.get_daily_kline(sub, begin, end), timeout)
        if st == "ok":
            if not len(val):
                print(f"  批 {bi + 1}/{n_batch}: 空（{len(sub)} 只）", flush=True)
            else:
                _add(val)
                print(f"  批 {bi + 1}/{n_batch}: {len(sub)} 只 -> +{len(val)} 行 "
                      f"[{time.time() - t0:.0f}s]", flush=True)
            if (bi + 1) % 5 == 0 or bi + 1 == n_batch:
                _write_cache(path, frames)
            continue
        # 换一种批量策略重试一次：拆成 20 只一组
        print(f"  批 {bi + 1}/{n_batch}: {st}（{val}），改 20 只/组重试", flush=True)
        for j in range(0, len(sub), 20):
            small = sub[j:j + 20]
            st2, val2 = _call(lambda: ds.get_daily_kline(small, begin, end), timeout)
            if st2 != "ok":
                failed.extend(small)
                print(f"    子批 {small[0]}..{small[-1]}: {st2}（{val2}）", flush=True)
                continue
            _add(val2)
        _write_cache(path, frames)

    df = _write_cache(path, frames)
    if failed:
        print(f"[kline] 未取到 {len(failed)} 只: {failed[:10]}...", flush=True)
    print(f"[kline] 完成 {len(df):,} 行 | {df.index.get_level_values('code').nunique()} 只 "
          f"| 写入 {path}", flush=True)
    return df


def attach_adj_factor(ds, raw: pd.DataFrame, codes: list[str], path: Path,
                      timeout: float = 900.0) -> tuple[pd.DataFrame, str]:
    """给长表补 ``adj_factor``（后复权累积因子，1.0=不复权）。

    优先 SDK ``get_backward_factor``（一次全量，超时 900s）；失败时退化到本地
    ``backward_factor.parquet`` 已有的列（覆盖现 41 只池），再失败则留空（用原始价）。
    """
    st, bf = _call(lambda: ds.get_backward_factor(codes), timeout)
    src = "sdk_backward_factor"
    if st != "ok" or bf is None or getattr(bf, "empty", True):
        print(f"[adj] SDK get_backward_factor {st}（{bf if st != 'ok' else '空表'}），"
              "退化到本地 backward_factor.parquet", flush=True)
        src = "local_backward_factor_cache"
        p = Path(str(Config.cache()["root"])) / "backward_factor.parquet"
        bf = pd.DataFrame()
        if p.exists():
            cached = pd.read_parquet(p)
            bf = cached[[c for c in codes if c in cached.columns]]
        if bf.empty:
            raw["adj_factor"] = np.nan
            _write_cache(path, [raw])
            return raw, "none"

    close = raw["close"].unstack("code")
    cols = list(close.columns)
    adj = (bf.reindex(index=close.index, columns=cols)
             .ffill().fillna(1.0))
    adj_long = adj.stack()
    adj_long.index = adj_long.index.set_names(["date", "code"])
    raw["adj_factor"] = adj_long.reindex(raw.index).astype(float)
    _write_cache(path, [raw])
    print(f"[adj] 复权因子来源 {src}，非 1.0 行占比 "
          f"{(raw['adj_factor'] > 1.0).mean():.3f}", flush=True)
    return raw, src


def recheck_coverage(ds, missing: list[str], begin: int, end: int, path: Path,
                     batch: int = 80, timeout: float = 420.0) -> dict:
    """对缓存里没有数据的代码再拉一次，记录有多少只其实有 K 线。

    用于区分「SDK 本地缓存不全」与「数据源本身没有退市标的行情」。
    命中数据会并入 ``etf_all_kline.parquet``。
    """
    hit: set[str] = set()
    frames: list[pd.DataFrame] = []
    if path.exists():
        frames.append(pd.read_parquet(path))
    for bi in range(0, len(missing), batch):
        sub = missing[bi:bi + batch]
        st, val = _call(lambda: ds.get_daily_kline(sub, begin, end), timeout)
        got: set[str] = set()
        if st == "ok" and val is not None and len(val):
            got = set(val.index.get_level_values("code"))
            frames.append(val[["close", "amount"]].astype(float))
            hit |= got
        print(f"  [coverage] 批 {bi // batch + 1}: 查询 {len(sub)} 只 -> 命中 {len(got)} 只 "
              f"({st})", flush=True)
    if len(frames) > 1:
        _write_cache(path, frames)
    return {"n_queried": len(missing), "n_success": len(hit),
            "codes_sample": sorted(hit)[:50]}


# ---------------------------------------------------------------------------
# 面板构建与回测
# ---------------------------------------------------------------------------
def build_panels(raw: pd.DataFrame) -> dict:
    close = raw["close"].unstack("code").sort_index()
    close = close.loc[:, close.notna().any()]
    amount = raw["amount"].unstack("code").reindex(index=close.index, columns=close.columns)
    adj_factor = None
    if "adj_factor" in raw.columns and raw["adj_factor"].notna().any():
        adj_factor = (raw["adj_factor"].unstack("code")
                      .reindex(index=close.index, columns=close.columns)
                      .ffill().fillna(1.0))
        adj_close = close * adj_factor
    else:
        adj_close = close.copy()

    ret = adj_close.pct_change(fill_method=None)
    first = close.apply(lambda s: s.first_valid_index())
    last = close.apply(lambda s: s.last_valid_index())
    ready = first + pd.Timedelta(days=365.25 * MIN_YEARS)
    age_ok = pd.DataFrame(
        np.greater_equal.outer(close.index.values, ready.values),
        index=close.index, columns=close.columns,
    )
    liq = amount.rolling(60, min_periods=40).mean() >= MIN_AMOUNT
    elig = (age_ok & liq & close.notna()).astype(bool)
    delisted = last < pd.Timestamp(DELIST_CUTOFF)

    return {
        "close_raw": close,
        "close": adj_close,
        "adj_factor": adj_factor,
        "amount": amount,
        "mom": adj_close.shift(21) / adj_close.shift(252) - 1.0,
        "vol": ret.rolling(20).std() * np.sqrt(252.0),
        "liq": liq.astype(bool),                     # 正式探针口径：仅流动性
        "elig": elig,                                # 上市满 3 年 + 流动性
        "first": first,
        "last": last,
        "delisted": delisted,
    }


def run_window(panels: dict, codes: list[str], begin: int, end: int,
               mask_override: pd.DataFrame | None = None) -> dict:
    """在 [begin, end] 上跑单一组合，返回指标。

    因子面板先按可投掩码置 NaN（选标前置掩码），``executable_mask`` 仍照传。
    """
    lo = pd.Timestamp(str(begin))
    hi = pd.Timestamp(str(end))
    idx = panels["close"].index
    m = (idx >= lo) & (idx <= hi)

    close = panels["close"].loc[m, codes]
    returns = close.assign(CASH=1.0).pct_change(fill_method=None)
    if mask_override is not None:
        mask = mask_override.loc[m, codes].copy()
    else:
        mask = panels["elig"].loc[m, codes].copy()
    mask = mask.fillna(False).astype(bool)
    # 选标前置掩码：不可入池标的的因子置 NaN，策略只在可投集合内排序。
    # 若让引擎事后清零再归一化，持仓只数会被动缩水（Top5 → 2 只）。
    factor = panels["mom"].loc[m, codes].where(mask).copy()
    factor["CASH"] = np.nan
    vol = panels["vol"].loc[m, codes].copy()
    vol["CASH"] = np.nan
    mask["CASH"] = True

    strategy = EtfRotationAdvanced(vol_panel=vol, **STRATEGY)
    bt = VectorBacktest(strategy=strategy, rebalance_freq="M",
                        initial_capital=1_000_000.0, costs=EtfCosts(**COSTS))
    res = bt.run(factor_panel=factor, returns_panel=returns,
                 executable_mask=mask, horizon=1)
    met = res.metrics()
    w = res.weights_history
    held = [c for c in w.columns if c != "CASH"]
    live = w[w.abs().sum(axis=1) > 1e-9]
    avg_holdings = (float((live[held].abs() > 1e-9).sum(axis=1).mean())
                    if len(live) else 0.0)
    return {
        "annual_return": met["annual_return"],
        "sharpe": met["sharpe"],
        "max_drawdown": met["max_drawdown"],
        "avg_holdings": avg_holdings,
    }


def run_variant(panels: dict, codes: list[str], begin: int, end: int,
                mask_override: pd.DataFrame | None = None) -> dict:
    """全窗口指标 + 三段年化（每段独立重置资金，与现有探针口径一致）。"""
    full = run_window(panels, codes, begin, end, mask_override)
    segs = []
    for y0, y1 in SEGMENTS:
        segs.append(run_window(panels, codes, max(begin, y0 * 10000 + 101),
                               y1 * 10000 + 1231, mask_override)["annual_return"])
    return {**full, **{f"seg_{i + 1}": s for i, s in enumerate(segs)}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ETF 轮动幸存者偏差一次性探针")
    ap.add_argument("--begin", type=int, default=20200101)
    ap.add_argument("--end", type=int, default=20261026)
    ap.add_argument("--kline-begin", type=int, default=20050101,
                    help="K 线起点；取 2005 以覆盖全部 ETF 的真实首个交易日")
    ap.add_argument("--batch", type=int, default=80)
    ap.add_argument("--skip-fetch", action="store_true",
                    help="直接用已有 etf_all_kline.parquet")
    ap.add_argument("--recheck-coverage", action="store_true",
                    help="对缓存里无数据的代码再拉一次，确认是缓存缺失还是数据源缺口")
    args = ap.parse_args(argv)

    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path()
    end_kline = int(pd.Timestamp(str(args.end)).strftime("%Y%m%d"))

    ds = create_datasource()
    cur = sorted(set(ds.get_code_list("EXTRA_ETF")))
    # 当前代码表只有仍在交易者；退市/清盘 ETF 须从历史代码表补齐
    hist_codes: list[str] = []
    uni_src = "get_code_list('EXTRA_ETF')"
    base = getattr(ds, "_base", None)
    if base is not None:
        st, hist = _call(lambda: base.get_hist_code_list(
            security_type="EXTRA_ETF", start_date=20150101, end_date=end_kline,
            local_path=Config.paths()["sdk_local_path"]), 900)
        if st == "ok" and hist:
            hist_codes = sorted({c for c in hist if isinstance(c, str)})
            uni_src = ("get_code_list('EXTRA_ETF') ∪ "
                       "get_hist_code_list('EXTRA_ETF')")
            print(f"[universe] 历史代码表 {len(hist_codes)} 只 "
                  f"(当前表之外 {len(set(hist_codes) - set(cur))} 只)", flush=True)
        else:
            print(f"[universe] 历史代码表 {st}（{hist}），退化为当前表", flush=True)
    codes = sorted(set(cur) | set(hist_codes))
    print(f"[universe] 合并清单 {len(codes)} 只（当前 {len(cur)} 只）", flush=True)

    if args.skip_fetch and cache_path.exists():
        raw = pd.read_parquet(cache_path)
        adj_src = "cache"
        if args.recheck_coverage:
            # 缓存覆盖诊断：其余代码再拉一次，看数据源是否真的没有历史行情
            have = set(raw.index.get_level_values("code"))
            missing = [c for c in codes if c not in have]
            print(f"[coverage] 缓存 {len(have)} 只 / 清单 {len(codes)} 只，"
                  f"缺 {len(missing)} 只，重拉一次", flush=True)
            cov = recheck_coverage(ds, missing, args.kline_begin, end_kline, cache_path)
            print(f"[coverage] 重查 {cov['n_queried']} 只 -> 命中 {cov['n_success']} 只，"
                  f"其余为数据源无历史行情", flush=True)
            if cov["n_success"]:
                raw = pd.read_parquet(cache_path)
            coverage = cov
        else:
            coverage = {}
    else:
        raw = fetch_klines(ds, codes, args.kline_begin, end_kline, cache_path,
                           batch=args.batch)
        raw, adj_src = attach_adj_factor(ds, raw, codes, cache_path)
        coverage = {}

    panels = build_panels(raw)
    delisted = panels["delisted"]
    n_delisted = int(delisted.sum())

    # A 臂是**历史口径**：固定为 2026-10-26 首次扩池时的 41 只，不随
    # ETF_CANDIDATES 增长（二次扩池 41→50 的 9 只只在 P2 池扩展 A/B 中评估）。
    pool_41 = [c for c in ETF_CANDIDATES if c not in {
        "159516.SZ", "159992.SZ", "159755.SZ", "588200.SH", "562500.SH",
        "159985.SZ", "159980.SZ", "159981.SZ", "513520.SH"}]
    codes_a = [c for c in pool_41 if c in panels["close"].columns]
    codes_b = [c for c in panels["close"].columns if bool(panels["elig"][c].any())]
    delisted_in_b = [c for c in codes_b if bool(delisted.get(c, False))]

    print(f"[pool] 覆盖 {len(panels['close'].columns)} 只 | 退市 {n_delisted} 只 | "
          f"A 池 {len(codes_a)} 只 | B 池 {len(codes_b)} 只 "
          f"(其中退市 {len(delisted_in_b)} 只)", flush=True)

    rows = []
    for name, sub, mo in (("A_old_41", codes_a, panels["liq"]),
                          ("B_pit_all", codes_b, None)):
        r = run_variant(panels, sub, args.begin, args.end, mask_override=mo)
        rows.append({"variant": name, "n_assets": len(sub), **r})
        print(f"{name:<12} n={len(sub):<4} ann={r['annual_return']:+.2%} "
              f"sh={r['sharpe']:.2f} mdd={r['max_drawdown']:.2%} "
              f"seg={r['seg_1']:+.1%}/{r['seg_2']:+.1%}/{r['seg_3']:+.1%} "
              f"hold={r['avg_holdings']:.1f}", flush=True)

    summary = pd.DataFrame(rows)[["variant", "n_assets", "annual_return", "sharpe",
                                  "max_drawdown", "seg_1", "seg_2", "seg_3",
                                  "avg_holdings"]]
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")

    # ---- 退市标的的 PIT 入池能力（结构性原因）----
    amt60 = panels["amount"].rolling(60, min_periods=40).mean()
    del_codes = list(delisted[delisted].index)
    max_amt: dict[str, float] = {}
    for c in del_codes:
        ready = panels["first"][c] + pd.Timedelta(days=365.25 * MIN_YEARS)
        s = amt60[c].loc[amt60.index >= ready].dropna()
        max_amt[c] = float(s.max()) if len(s) else float("nan")
    ma = pd.Series(max_amt, dtype=float).dropna()
    pool_max_amt = amt60[codes_b].max().dropna() if codes_b else pd.Series(dtype=float)
    del_stat = {
        "n_delisted_with_data": len(del_codes),
        "n_delisted_with_3y_window": int(len(ma)),
        "n_delisted_ever_eligible": len(delisted_in_b),
        "n_delisted_max_amt60_ge_min": int((ma >= MIN_AMOUNT).sum()),
        "min_amount_threshold": MIN_AMOUNT,
        "max_amt60_after_3y_quantiles": {
            f"p{p}": float(ma.quantile(p / 100.0))
            for p in (10, 25, 50, 75, 90, 100)},
        "max_amt60_after_3y_max": float(ma.max()) if len(ma) else float("nan"),
        "max_amt60_after_3y_over_threshold": (
            float(ma.max() / MIN_AMOUNT) if len(ma) else float("nan")),
        "b_pool_max_amt60_median": (
            float(pool_max_amt.median()) if len(pool_max_amt) else float("nan")),
        "delisted_top_codes": ma.nlargest(5).round(0).to_dict(),
    }
    print(f"[退市] {len(del_codes)} 只有数据 | 有满 3 年窗口 {len(ma)} 只 | "
          f"曾入池 {len(delisted_in_b)} 只 | 最高 60 日均额 "
          f"p50={ma.quantile(0.5) / 1e4:.0f} 万 / p90={ma.quantile(0.9) / 1e4:.0f} 万 "
          f"/ max={ma.max() / 1e4:.0f} 万（门槛 {MIN_AMOUNT / 1e4:.0f} 万）", flush=True)

    # ---- panels.csv：逐年平均入池数量 + 退市标的占比 ----
    elig = panels["elig"]
    prow = []
    for year in range(2018, 2027):
        m = (elig.index >= pd.Timestamp(f"{year}-01-01")) & \
            (elig.index <= pd.Timestamp(f"{year}-12-31"))
        if not m.any():
            continue
        in_year = elig.loc[m]
        ever = in_year.columns[in_year.any()]
        prow.append({
            "year": year,
            "avg_n_eligible": float(in_year.sum(axis=1).mean()),
            "n_ever_eligible": int(len(ever)),
            "n_delisted_ever_eligible": int(sum(bool(delisted.get(c, False)) for c in ever)),
            "delisted_share": (float(sum(bool(delisted.get(c, False)) for c in ever))
                               / max(len(ever), 1)),
        })
    pd.DataFrame(prow).to_csv(out / "panels.csv", index=False, encoding="utf-8-sig")

    # ---- 判据 ----
    a = summary.loc[summary["variant"] == "A_old_41"].iloc[0]
    b = summary.loc[summary["variant"] == "B_pit_all"].iloc[0]
    d_sharpe = float(a["sharpe"] - b["sharpe"])
    reproduce = abs(float(a["sharpe"]) - BASELINE_SHARPE) <= 0.05
    if len(delisted_in_b) < 5:
        verdict = "样本不足（退市标的进池 < 5 只）"
    elif d_sharpe <= 0.10:
        verdict = "幸存者偏差可忽略"
    elif d_sharpe > 0.20:
        verdict = "结论需大幅打折"
    else:
        verdict = "需标注"

    (out / "run_config.json").write_text(json.dumps({
        "begin": args.begin, "end": args.end, "kline_begin": args.kline_begin,
        "universe": uni_src,
        "n_codes_current": len(cur),
        "n_codes_hist": len(hist_codes),
        "n_codes_hist_not_current": len(set(hist_codes) - set(cur)),
        "n_codes_listed": len(codes),
        "n_codes_with_data": int(len(panels["close"].columns)),
        "n_codes_hist_not_current_with_data": int(
            len(set(panels["close"].columns) - set(cur))),
        "coverage_recheck": coverage,
        "n_delisted": n_delisted,
        "delist_cutoff": DELIST_CUTOFF,
        "min_amount": MIN_AMOUNT, "min_years": MIN_YEARS,
        "segments": SEGMENTS,
        "strategy": STRATEGY, "costs": COSTS,
        "selection_mask": "因子面板按可投掩码置 NaN（选标前置）",
        "adj_factor_source": adj_src,
        "n_assets_A": len(codes_a), "n_assets_B": len(codes_b),
        "mask_A": "仅流动性（60 日均额 ≥ 3000 万，与正式探针一致）",
        "mask_B": "上市满 3 年 + 60 日均额 ≥ 3000 万",
        "delisted_pool_stats": del_stat,
        "cache": str(cache_path),
        "baseline_sharpe": BASELINE_SHARPE, "baseline_annual": BASELINE_ANNUAL,
        "reproduce_ok": bool(reproduce),
        "reproduce_detail": {
            "sharpe_A": float(a["sharpe"]), "sharpe_diff": float(a["sharpe"]) - BASELINE_SHARPE,
            "annual_A": float(a["annual_return"]),
            "annual_diff": float(a["annual_return"]) - BASELINE_ANNUAL,
            "avg_holdings_A": float(a["avg_holdings"]),
        },
        "sharpe_drop_B_vs_A": d_sharpe,
        "judge": "复现 |ΔSharpe|<=0.05 & |Δ年化|<=0.5pp & 持仓≈5；"
                 "主判据 B-A 下降 <=0.10 可忽略 / >0.20 需大幅打折",
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n判据：复现={'OK' if reproduce else 'FAIL'} | B-A Sharpe 下降 {d_sharpe:+.2f} "
          f"| {verdict}\n写入 {out}", flush=True)


if __name__ == "__main__":
    main()
