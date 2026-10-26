"""
ETF 层 α 可行性探针（S2 持仓映射 α 的前置验证）
================================================

把个股 α 按 ETF 跟踪指数的**在册成分权重**聚合到 ETF 层，检验 ETF 层 α 对
下一月 ETF 收益是否有截面预测力。**只做 IC 检验，不做组合回测。**

口径（先验固定，跑完不改）:
  - 权重：调仓日 t 的在册成分（INDATE<=t 且 OUTDATE 空或 >t），
    权重 ∝ 未复权 close × FLOAT_SHARE（PIT 前向填充）；缺自由流通股本时用
    总市值（`market_cap.parquet`）兜底；每个 ETF 内归一化。
  - α_i：α 面板先按日期 rolling(20).mean()，再取 t 行。
  - ETF 层 α = Σ(w_i × α_i) / Σw_i；α 权重覆盖率 < 60% 的 ETF-月跳过。
  - 检验：月末 t 的 ETF 层 α vs 下一月收益（复权 close 的 t_next/t-1）的
    截面 Spearman rank IC（至少 6 只 ETF 有值才算一期）。同时算 α 与
    12-1 动量的截面 rank 相关，以及动量前 50% 子集内的 α IC。

判据:
  - 主判据：mean IC ≥ 0.05 且 t_stat ≥ 2。
  - 增量：mean_ic_mom_top50 显著为正（t ≥ 1.5）。

数据来源:
  - 个股 α:    reports/alla_rolling_ortho/pred/ens_h1h5__h1.parquet
  - 未复权收盘: reports/alla_rolling/_base/close_raw.parquet
  - 总市值:    reports/alla_rolling/_base/market_cap.parquet
  - 自由流通股本: E:/data/parquet/equity_structure.parquet（变更流水表，PIT 展开）
  - ETF 行情:  DataCache + EtfUniverse（后复权 close）
  - 指数成分:  SDK get_index_constituent，缓存到
              E:/data/sdk_cache/infodata/etf_alpha/constituent_<index>.parquet

用法:
    python -m scripts.portfolio.etf_alpha_probe
    python -m scripts.portfolio.etf_alpha_probe --begin 20200101 --end 20261026
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "etf_alpha_probe"
ALPHA_PATH = ROOT / "reports" / "alla_rolling_ortho" / "pred" / "ens_h1h5__h1.parquet"
CLOSE_RAW_PATH = ROOT / "reports" / "alla_rolling" / "_base" / "close_raw.parquet"
MCAP_PATH = ROOT / "reports" / "alla_rolling" / "_base" / "market_cap.parquet"
EQUITY_STRUCTURE_PATH = Path("E:/data/parquet/equity_structure.parquet")
SDK_CACHE = Path("E:/data/sdk_cache/infodata/etf_alpha")

# ETF → 跟踪指数（只列可映射 A 股成分的 30 只；跨境 6 / 债券 4 / 黄金 1 不映射）
ETF_INDEX: dict[str, str] = {
    # 宽基
    "510300.SH": "000300.SH",   # 沪深300
    "510050.SH": "000016.SH",   # 上证50
    "510500.SH": "000905.SH",   # 中证500
    "512100.SH": "000852.SH",   # 中证1000
    "159915.SZ": "399006.SZ",   # 创业板指
    "588000.SH": "000688.SH",   # 科创50
    "159901.SZ": "399330.SZ",   # 深证100
    "510880.SH": "000015.SH",   # 上证红利
    # 行业
    "512010.SH": "000933.SH",   # 中证医药
    "512170.SH": "399989.SZ",   # 中证医疗
    "512690.SH": "399997.SZ",   # 中证酒
    "512880.SH": "399975.SZ",   # 证券公司
    "515030.SH": "399976.SZ",   # 新能源汽车
    "512400.SH": "000819.SH",   # 有色金属
    "512660.SH": "399967.SZ",   # 中证军工
    "512800.SH": "399986.SZ",   # 中证银行
    "512980.SH": "399971.SZ",   # 中证传媒
    "515220.SH": "399998.SZ",   # 中证煤炭
    "159928.SZ": "000932.SH",   # 中证主要消费
    "159949.SZ": "399673.SZ",   # 创业板50
    # 主题
    "515790.SH": "931151.CSI",  # 中证光伏产业
    "159869.SZ": "930901.CSI",  # 中证动漫游戏
    "512290.SH": "930726.CSI",  # 中证生物医药
    "512070.SH": "399809.SZ",   # 中证800证券保险
    "512200.SH": "931775.CSI",  # 中证全指房地产
    "512480.SH": "931865.CSI",  # 中证全指半导体
    # 516160 新能源ETF 实为跟踪中证新能源（399808.SZ）；任务草稿给的 931160.CSI
    # 与 515880 通信ETF 重复，故以 399808.SZ 为准。
    "516160.SH": "399808.SZ",
    "512760.SH": "990001.CSI",  # 中华半导体芯片
    "159825.SZ": "000949.SH",   # 中证农业
    "515880.SH": "931160.CSI",  # 中证全指通信设备
}

BEGIN = 20200101
END = 20261026
MIN_COVERAGE = 0.60     # α 可得的成分权重覆盖率下限
MIN_ETF_PER_PERIOD = 6  # 一期截面至少需要的 ETF 数
MOM_SKIP, MOM_WINDOW = 21, 252
ALPHA_WINDOW = 20
TOP_FRAC = 0.5
MIN_TOP_N = 4


# ---------------------------------------------------------------------------
# SDK 串行调用 + 硬超时
# ---------------------------------------------------------------------------
def _call(fn, seconds: float = 600.0):
    """SDK 可能挂死（无超时），daemon 线程 + queue 硬超时。"""
    q: queue.Queue = queue.Queue()

    def _run():
        try:
            q.put(("ok", fn()))
        except BaseException as exc:  # noqa: BLE001 - 探针要看到任何异常
            q.put(("err", f"{type(exc).__name__}: {exc}"))

    threading.Thread(target=_run, daemon=True).start()
    try:
        return q.get(timeout=seconds)
    except queue.Empty:
        return ("timeout", f">{seconds:.0f}s 无返回")


def _aliases(code: str) -> list[str]:
    """代码别名：只给 1 个常见备选（.CSI ↔ .SH），不做无限试探。"""
    if code.endswith(".CSI"):
        return [code, f"{code[:-4]}.SH"]
    if code.endswith(".SH"):
        return [code, f"{code[:-3]}.CSI"]
    if code.endswith(".SZ"):
        return [code, f"{code[:-3]}.CSI"]
    return [code]


def _normalize_constituent(df: pd.DataFrame) -> pd.DataFrame:
    """SDK 原始表 → (con_code, in_date, out_date)，日期统一为 Timestamp。"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["con_code", "in_date", "out_date"])
    out = df.rename(columns={"CON_CODE": "con_code", "INDATE": "in_date",
                             "OUTDATE": "out_date"}).copy()
    for col in ("con_code", "in_date", "out_date"):
        if col not in out.columns:
            out[col] = np.nan
    out = out[["con_code", "in_date", "out_date"]]
    out["in_date"] = pd.to_datetime(out["in_date"], errors="coerce")
    out["out_date"] = pd.to_datetime(out["out_date"], errors="coerce")
    return out.dropna(subset=["con_code", "in_date"]).reset_index(drop=True)


def load_constituents(index_codes: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """逐指数加载成分（本地 parquet 缓存优先；无缓存才登录 SDK，全部串行）。

    Returns:
        (index → 成分事件表, index → 备注)
    """
    SDK_CACHE.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    notes: dict[str, str] = {}

    def _read_cached(code: str) -> pd.DataFrame | None:
        p = SDK_CACHE / f"constituent_{code}.parquet"
        if not p.exists():
            return None
        return _normalize_constituent(pd.read_parquet(p))

    missing = [c for c in index_codes if _read_cached(c) is None]
    ds = local = None
    if missing:
        from config import Config
        from data.datasource import create_datasource

        local = Config.paths()["sdk_local_path"]
        ds = create_datasource()

    for code in index_codes:
        cached = _read_cached(code)
        if cached is not None and not cached.empty:
            frames[code] = cached
            notes[code] = f"cache ({len(cached)} 行)"
            continue

        got = None
        tried = []
        for cand in _aliases(code)[:2]:   # 主代码 + 至多 1 个别名
            tried.append(cand)
            st, val = _call(
                lambda c=cand: ds._info.get_index_constituent(
                    [c], local_path=local, is_local=False),
                seconds=600.0,
            )
            if st != "ok":
                notes[code] = f"{cand}: {st}"
                continue
            raw = val.get(cand) if isinstance(val, dict) else val
            norm = _normalize_constituent(raw)
            if norm.empty:
                notes[code] = f"{cand}: 空表"
                continue
            got = (cand, norm)
            break

        if got is None:
            frames[code] = pd.DataFrame(columns=["con_code", "in_date", "out_date"])
            notes[code] = "unmapped: " + "; ".join(tried)
            continue
        cand, norm = got
        norm.to_parquet(SDK_CACHE / f"constituent_{code}.parquet", compression="snappy")
        frames[code] = norm
        notes[code] = f"ok via {cand} ({len(norm)} 行)"
    return frames, notes


# ---------------------------------------------------------------------------
# 权重面板（自由流通市值代理，PIT）
# ---------------------------------------------------------------------------
def build_weight_panels(dates: pd.DatetimeIndex, columns: pd.Index,
                        close_raw: pd.DataFrame, mcap: pd.DataFrame):
    """返回 (float_mv, mcap_mv) 两张只在调仓日取值的面板（单位：元）。"""
    from data.market_cap import build_shares_panel

    eq = (pd.read_parquet(EQUITY_STRUCTURE_PATH)
          if EQUITY_STRUCTURE_PATH.exists() else pd.DataFrame())
    shares = build_shares_panel(eq, index=dates, columns=columns,
                                share_field="float_share")
    float_mv = shares * close_raw.reindex(index=dates, columns=columns)
    float_mv = float_mv.where(float_mv > 0)          # 缺股本 / 为 0 → NaN，走兜底
    mcap_mv = mcap.reindex(index=dates, columns=columns).where(lambda x: x > 0)
    return float_mv, mcap_mv


def members_at(cons: pd.DataFrame, t: pd.Timestamp) -> list[str]:
    m = (cons["in_date"] <= t) & (cons["out_date"].isna() | (cons["out_date"] > t))
    return cons.loc[m, "con_code"].dropna().unique().tolist()


def month_end_dates(idx: pd.DatetimeIndex, begin: int, end: int) -> pd.DatetimeIndex:
    sel = idx[(idx >= pd.Timestamp(str(begin))) & (idx <= pd.Timestamp(str(end)))]
    if len(sel) == 0:
        return pd.DatetimeIndex([])
    s = pd.Series(sel, index=sel)
    return pd.DatetimeIndex(sorted(s.groupby([s.index.year, s.index.month]).max().values))


def _spearman(a: pd.Series, b: pd.Series, min_n: int = 3) -> float:
    ok = a.notna() & b.notna()
    if int(ok.sum()) < min_n:
        return np.nan
    x, y = a[ok], b[ok]
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(x.corr(y, method="spearman"))


def _stats(s: pd.Series) -> dict:
    v = s.dropna()
    n = len(v)
    if n < 2:
        return {"mean": (float(v.mean()) if n else None), "icir": None,
                "t_stat": None, "n": n}
    mean, std = float(v.mean()), float(v.std(ddof=1))
    return {"mean": mean,
            "icir": (mean / std) if std > 0 else None,
            "t_stat": (mean / std * np.sqrt(n)) if std > 0 else None,
            "n": n}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ETF 层 α 可行性探针（S2 前置验证）")
    ap.add_argument("--begin", type=int, default=BEGIN)
    ap.add_argument("--end", type=int, default=END)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from data.cache import DataCache
    from data.datasource import create_datasource
    from data.etf_universe import EtfUniverse

    # ---- 1. ETF 行情（后复权）----
    cache = DataCache(create_datasource())
    uni = EtfUniverse(cache)
    close_all = uni.load_close(adjust=True)
    close_all = close_all[[c for c in ETF_INDEX if c in close_all.columns]]
    etf_close = close_all.loc[(close_all.index >= pd.Timestamp(str(args.begin)))
                              & (close_all.index <= pd.Timestamp(str(args.end)))]
    print(f"ETF 池: {len(etf_close.columns)} 只, {etf_close.index.min().date()} -> "
          f"{etf_close.index.max().date()}", flush=True)

    # ---- 2. 个股 α / 未复权收盘 / 总市值 ----
    alpha_raw = pd.read_parquet(ALPHA_PATH)
    alpha_roll = alpha_raw.rolling(ALPHA_WINDOW, min_periods=ALPHA_WINDOW).mean()
    close_raw = pd.read_parquet(CLOSE_RAW_PATH)
    mcap = pd.read_parquet(MCAP_PATH)

    # ---- 3. 指数成分（串行 + 缓存）----
    index_codes = list(dict.fromkeys(ETF_INDEX.values()))
    cons_map, cons_notes = load_constituents(index_codes)
    mapped = {i: c for i, c in cons_map.items() if not c.empty}
    print(f"成分命中: {len(mapped)}/{len(index_codes)} 个指数", flush=True)

    # ---- 4. 调仓日与权重面板 ----
    me = month_end_dates(etf_close.index, args.begin, args.end)
    print(f"调仓日: {len(me)} 个, {me.min().date()} -> {me.max().date()}", flush=True)

    codes_union = sorted({c for df in mapped.values()
                          for c in df["con_code"].dropna().unique()})
    codes_union = [c for c in codes_union if c in close_raw.columns]
    print(f"成分并集（有未复权收盘）: {len(codes_union)}", flush=True)
    float_mv, mcap_mv = build_weight_panels(me, pd.Index(codes_union), close_raw, mcap)

    # ---- 5. 逐 ETF-月聚合 α ----
    alphas = pd.DataFrame(index=me, columns=list(ETF_INDEX), dtype=float)
    cov_log: dict[str, list[dict]] = {etf: [] for etf in ETF_INDEX}

    for etf, idx_code in ETF_INDEX.items():
        cons = mapped.get(idx_code)
        if cons is None:
            continue
        for t in me:
            if t not in alpha_roll.index:
                continue                              # α 面板无此交易日
            if pd.isna(etf_close.at[t, etf]):
                continue                              # 未上市 / 当日无价
            members = [x for x in members_at(cons, t) if x in float_mv.columns]
            if not members:
                continue
            w = float_mv.loc[t, members].astype(float)
            w = w.where(w > 0, mcap_mv.loc[t, members])   # 总市值兜底
            w = w.where(w > 0).dropna()
            if w.empty:
                continue
            a = alpha_roll.loc[t].reindex(w.index).astype(float)
            total = float(w.sum())
            cov = float(w[a.notna()].sum()) / total
            used = cov >= MIN_COVERAGE
            cov_log[etf].append({"date": t, "n_cons": len(w),
                                 "coverage": cov, "used": used})
            if not used or float(w[a.notna()].sum()) <= 0:
                continue
            alphas.at[t, etf] = float((w * a).sum()) / float(w[a.notna()].sum())

    cov_rows = []
    for etf, rows in cov_log.items():
        if not rows:
            continue
        cov_rows.append({
            "etf": etf,
            "index_code": ETF_INDEX[etf],
            "n_constituents_avg": round(float(np.mean([r["n_cons"] for r in rows])), 2),
            "alpha_coverage_avg": round(float(np.mean([r["coverage"] for r in rows])), 4),
            "months": int(sum(r["used"] for r in rows)),
        })

    # ---- 6. 截面检验（月末 → 下一月末）----
    close_me = etf_close.reindex(me)
    ret_fwd = close_me.shift(-1) / close_me - 1.0
    mom = (close_all.shift(MOM_SKIP) / close_all.shift(MOM_WINDOW) - 1.0).reindex(me)

    ic_rows = []
    for t in me:
        a = alphas.loc[t].dropna()
        r = ret_fwd.loc[t]
        ok = a.index.intersection(r.dropna().index)
        if len(ok) < MIN_ETF_PER_PERIOD:
            continue
        a, r, m = a[ok], r[ok], mom.loc[t, ok]
        row = {"date": t.strftime("%Y-%m-%d"), "n_etf": int(len(ok)),
               "ic": _spearman(a, r, MIN_ETF_PER_PERIOD),
               "ic_mom_top50": np.nan, "corr_alpha_mom": np.nan}
        if m.notna().sum() >= 2:
            row["corr_alpha_mom"] = _spearman(a, m, 3)
            mm = m.dropna()
            n_top = max(1, int(np.ceil(len(mm) * TOP_FRAC)))
            top = mm.sort_values(ascending=False).index[:n_top]
            if len(top) >= MIN_TOP_N:
                row["ic_mom_top50"] = _spearman(a[top], r[top], MIN_TOP_N)
        ic_rows.append(row)

    ic_df = pd.DataFrame(ic_rows)
    if ic_df.empty:
        raise SystemExit("无任何有效截面期（检查成分映射 / α 覆盖率）")

    # ---- 7. 落盘 ----
    ic_df[["date", "n_etf", "ic", "ic_mom_top50"]].to_csv(
        out / "ic.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cov_rows).to_csv(out / "coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"etf": etf, "index_code": idx_code,
         "status": "ok" if idx_code in mapped else "unmapped",
         "note": cons_notes.get(idx_code, "")}
        for etf, idx_code in ETF_INDEX.items()
    ]).to_csv(out / "mapping.csv", index=False, encoding="utf-8-sig")

    st_main = _stats(ic_df["ic"])
    st_top = _stats(ic_df["ic_mom_top50"])
    corr_am = ic_df["corr_alpha_mom"].dropna()
    years = pd.to_datetime(ic_df["date"]).dt.year
    ok_rows = ic_df["ic"].notna()
    by_year = {str(int(y)): float(v) for y, v in
               ic_df.loc[ok_rows, "ic"].groupby(years[ok_rows]).mean().items()}

    evaluated = sum(len(v) for v in cov_log.values())
    used = sum(r["used"] for v in cov_log.values() for r in v)
    all_cov = [r["coverage"] for v in cov_log.values() for r in v]
    used_cov = [r["coverage"] for v in cov_log.values() for r in v if r["used"]]

    summary = {
        "period": {"first": ic_df["date"].iloc[0], "last": ic_df["date"].iloc[-1],
                   "months_selected": int(len(me))},
        "alpha_window": ALPHA_WINDOW,
        "min_coverage": MIN_COVERAGE,
        "mean_ic": st_main["mean"],
        "icir": st_main["icir"],
        "t_stat": st_main["t_stat"],
        "n_periods": st_main["n"],
        "by_year": by_year,
        "mean_corr_alpha_mom": float(corr_am.mean()) if len(corr_am) else None,
        "mean_ic_mom_top50": st_top["mean"],
        "icir_mom_top50": st_top["icir"],
        "t_stat_mom_top50": st_top["t_stat"],
        "n_periods_mom_top50": st_top["n"],
        "coverage": {
            "n_etf_mapped": len(ETF_INDEX),
            "n_index_attempted": len(index_codes),
            "n_index_ok": len(mapped),
            "n_etf_with_alpha": int(alphas.notna().any().sum()),
            "etf_months_evaluated": int(evaluated),
            "etf_months_used": int(used),
            "used_rate": (used / evaluated) if evaluated else None,
            "mean_coverage_all": float(np.mean(all_cov)) if all_cov else None,
            "mean_coverage_used": float(np.mean(used_cov)) if used_cov else None,
            "avg_etf_per_period": float(ic_df["n_etf"].mean()),
            "unmapped_note": {k: v for k, v in cons_notes.items() if k not in mapped},
        },
        "judge": {
            "main_rule": "mean_ic>=0.05 and t_stat>=2",
            "pass_main": bool(st_main["mean"] is not None and st_main["mean"] >= 0.05
                              and st_main["t_stat"] is not None and st_main["t_stat"] >= 2.0),
            "increment_rule": "mean_ic_mom_top50>0 and t_stat_mom_top50>=1.5",
            "pass_increment": bool(st_top["mean"] is not None and st_top["mean"] > 0
                                   and st_top["t_stat"] is not None
                                   and st_top["t_stat"] >= 1.5),
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nmean IC={st_main['mean']:.4f} ICIR={st_main['icir']:.2f} "
          f"t={st_main['t_stat']:.2f} n={st_main['n']}")
    print(f"mom_top50 IC={st_top['mean']} t={st_top['t_stat']} n={st_top['n']}")
    print(f"corr(α,mom)={summary['mean_corr_alpha_mom']}  "
          f"ETF-月 used/eval={used}/{evaluated}")
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
