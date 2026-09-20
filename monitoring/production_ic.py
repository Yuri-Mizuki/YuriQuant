"""生产口径每日 IC / 风格暴露监控（全A · ens_h1h5 · ortho 预处理）。

问题
----
``monitoring/`` 现有链路监控的是因子库数据集（``hs300_2022_2025``），其行情源
``daily_hs300.parquet`` 自 2026-08-21 起停更；而生产在跑的是**全A**口径
（``scripts/pipelines/alla_daily_rank.py``）。两条链路互不相干 —— 生产模型
实际上**没有任何按日推进的 IC 记录**，无法区分「单日噪音」与「持续退化」。

本模块把生产口径接到一条按日推进的时间序列上，且历史段与每日段**共用同一段
指标代码**，可直接拼成一条序列：

- 历史种子：``reports/alla_rolling_ortho/pred/ens_h1h5__h1.parquet``（回测 OOS
  预测面板）与 ``_base/{close_adj,cov_*}.parquet``（**与历史消融网格完全同一套
  风格协变量**，保证口径可比）。
- 每日增量：``reports/alla_daily/ranking_<ds>.csv``（当日出榜打分）+ 本地行情缓存。

产物
----
``reports/monitoring/production_ic_daily.csv``（按 ``predict_date`` 幂等覆盖）。

口径
----
- 收益：后复权 close 的 T→T+1 收益（信号 T 日收盘产生、T+1 收盘计收益）。
- ``ic_raw`` = 每日截面 Spearman(score, 次日收益)。
- ``ic_neutral`` = 先用 :func:`factor.preprocessing.neutralize` 对当日 score 做
  五风格中性化（size/industry/mom/vol/turn）取残差，再算 Spearman。
- ``style_exposure_ratio`` = ``1 − ic_neutral / ic_raw``（raw IC 被风格解释的比例）。
- ``z_*`` = 组合持仓在对应风格上的横截面 z 均值（事前可算，不需未来收益）。

注意
----
历史段是回测 OOS 面板、当前段是每日链实时打分，**模型配置相同但特征来源路径
不同**（历史段走回测面板，当前段走 ``load_tail``），两者的 IC 水平可能有系统差；
序列首尾衔接处应看趋势而非绝对水平。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("monitoring.production_ic")

#: 生产配置：集成 h1+h5 秩平均、horizon=1 的 OOS 预测面板
PRED_STEM = "ens_h1h5__h1"
#: 组合名义分位（与出榜链 frac 一致：Top 10%）
TOP_FRAC = 0.10
#: 风格协变量键（与 ``_base/cov_*.parquet`` 对齐）
STYLE_KEYS = ("size", "mom", "vol", "turn")
INDUSTRY_KEY = "industry"
#: 参与 IC 计算的最少股票数（低于此值当天记 NaN，避免小样本伪相关）
MIN_SAMPLES = 30


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def daily_rank_ic(score: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    """逐日截面 Spearman 相关（秩变换后 Pearson，向量化）。

    仅在两边都有值的 (date, code) 上计算；当天有效样本 < ``MIN_SAMPLES`` 记 NaN。
    """
    valid = score.notna() & ret.notna()
    if not valid.any().any():
        return pd.Series(dtype=float)
    s = score.where(valid).rank(axis=1)
    r = ret.where(valid).rank(axis=1)
    n = valid.sum(axis=1)
    sc = s.sub(s.mean(axis=1), axis=0)
    rc = r.sub(r.mean(axis=1), axis=0)
    num = (sc * rc).sum(axis=1)
    den = np.sqrt((sc ** 2).sum(axis=1) * (rc ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    return ic.where(n >= MIN_SAMPLES)


def _zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    """横截面 rank 后标准化（对极值稳健；与 09-18 复盘脚本同口径）。"""
    r = df.rank(axis=1, pct=True)
    mu = r.mean(axis=1)
    sd = r.std(axis=1, ddof=0).replace(0, np.nan)
    return r.sub(mu, axis=0).div(sd, axis=0)


def _monotonicity(score: pd.DataFrame, ret: pd.DataFrame,
                  n_quantiles: int = 10) -> pd.Series:
    """分位单调性：各分位次日均收益 与 分位序号 的横截面逐日 Spearman。"""
    out: dict[pd.Timestamp, float] = {}
    for d in score.index:
        s = score.loc[d]
        r = ret.loc[d] if d in ret.index else None
        if r is None:
            continue
        sub = pd.concat([s.rename("s"), r.rename("r")], axis=1).dropna()
        if len(sub) < MIN_SAMPLES * 2:
            continue
        try:
            q = pd.qcut(sub["s"].rank(method="first"), n_quantiles,
                        labels=False, duplicates="drop")
        except ValueError:
            continue
        means = sub.groupby(q)["r"].mean()
        if len(means) < 3:
            continue
        out[d] = float(pd.Series(means.index).corr(
            pd.Series(means.values), method="spearman"))
    return pd.Series(out, dtype=float).sort_index()


def _neutralize_style(score: pd.DataFrame, cov: dict[str, pd.DataFrame],
                      keys: tuple[str, ...]) -> pd.DataFrame:
    """按给定风格键对 score 做逐日截面中性化（取残差）。

    复用 :func:`factor.preprocessing.neutralize`（与历史网格同一实现），
    ``size`` 走 ``market_cap_panel``（内部取 log），``industry`` 走哑变量，
    其余连续协变量走 ``extra_covariates``。
    """
    from factor.preprocessing import neutralize

    if not keys:
        return score
    extra = {k: cov[k] for k in keys
             if k in cov and k not in ("size", INDUSTRY_KEY)}
    return neutralize(
        score,
        market_cap_panel=cov.get("size") if "size" in keys else None,
        industry_panel=cov.get(INDUSTRY_KEY) if INDUSTRY_KEY in keys else None,
        extra_covariates=extra or None,
    )


def _combine(*panels: pd.DataFrame | None) -> pd.DataFrame:
    """纵向拼接若干面板，后出现者覆盖先出现者的同名日期。"""
    keep = [p for p in panels if p is not None and not p.empty]
    if not keep:
        return pd.DataFrame()
    keys = sorted({k for p in keep for k in p.columns})
    out = pd.concat([p.reindex(columns=keys) for p in keep], axis=0)
    return out[~out.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------------------
# 数据装配
# ---------------------------------------------------------------------------
def load_live_base(cache_root: str | Path, begin: str = "20260601") -> dict:
    """从本地缓存重建 ``begin`` 起的复权价与风格协变量面板（对齐 stage_prep）。

    与 ``scripts/pipelines/rolling_grid_alla.stage_prep`` 同公式、同数据源，
    因此与 ``_base/cov_*.parquet`` 在重叠区间应当逐值一致（调用方做校验）。
    """
    from data.cache import DataCache
    from data.industry import IndustryClassification
    from data.offline import OfflineQuietDataSource
    from factor.preprocessing import build_style_covariates

    root = Path(cache_root)
    daily = pd.read_parquet(root / "daily_all_a.parquet")
    daily.index = daily.index.set_levels(daily.index.levels[0].normalize(), level=0)
    daily = daily[daily.index.get_level_values(0) >= pd.Timestamp(begin)]
    d = daily.reset_index()
    d["date"] = d["date"].dt.normalize()

    def _panel(col: str) -> pd.DataFrame:
        return d.pivot(index="date", columns="code", values=col).sort_index()

    o, hi, lo, c = _panel("open"), _panel("high"), _panel("low"), _panel("close")
    v, amt = _panel("volume"), _panel("amount")
    raw_close = c.copy()

    bf = pd.read_parquet(root / "backward_factor.parquet")
    bf = bf[[x for x in c.columns if x in bf.columns]]
    f = bf.reindex(index=c.index, columns=c.columns).ffill()
    for pnl in (o, hi, lo, c):
        pnl[:] = pnl.values * f.values
    vwap = ((amt / v.replace(0, np.nan)) * f).astype(np.float32)
    vwap = vwap.replace([np.inf, -np.inf], np.nan)

    eq = pd.read_parquet(root / "equity_structure.parquet")
    shares = _shares_panel(eq, c.index, c.columns)
    mktcap = (shares * raw_close).astype(np.float32)

    cache = DataCache(OfflineQuietDataSource())
    industry = IndustryClassification(cache, level=1).get_industry_panel(
        list(c.columns), c.index)

    cov = build_style_covariates(
        {"close": c, "volume": v, "tot_share": shares},
        market_cap_panel=mktcap, industry_panel=industry)
    px = {"open": o, "high": hi, "low": lo, "close": c, "volume": v,
          "amount": amt, "vwap": vwap, "close_raw": raw_close}
    return {"close_adj": c, "cov": cov, "px": px, "bwd": f}


def _shares_panel(eq: pd.DataFrame, index: pd.Index,
                  columns: pd.Index) -> pd.DataFrame:
    """日频总股本面板（万股 → 股），转调 ``data.market_cap.build_shares_panel``。

    与 ``scripts/pipelines/rolling_grid_alla._shares_panel`` 同公式——回测基线
    ``_base/cov_size`` 即由后者生成，重叠区间校验 max|Δ|=0 已证等价。
    """
    from data.market_cap import build_shares_panel

    return build_shares_panel(eq, index, columns, share_field="tot_share")


def load_hist_base(hist_dir: str | Path) -> dict:
    """读回测 OOS 预测面板与 ``_base`` 复权价 / 风格协变量。"""
    d = Path(hist_dir)
    pred_p = d / "pred" / f"{PRED_STEM}.parquet"
    close_p = d / "_base" / "close_adj.parquet"
    if not pred_p.exists() or not close_p.exists():
        raise FileNotFoundError(
            f"回测基线缺失：{pred_p} 或 {close_p}（先跑 rolling_grid_alla --stage prep）")
    pred = pd.read_parquet(pred_p)
    close = pd.read_parquet(close_p)
    cov: dict[str, pd.DataFrame] = {}
    for k in (*STYLE_KEYS, INDUSTRY_KEY):
        p = d / "_base" / f"cov_{k}.parquet"
        if p.exists():
            cov[k] = pd.read_parquet(p)
    return {"pred": pred, "close_adj": close, "cov": cov}


def collect_daily_scores(rank_dir: str | Path,
                         exclude_suffixes: tuple[str, ...] = ("_prestatusfix",),
                         ) -> dict[str, pd.DataFrame]:
    """收集每日出榜产物里的打分截面：``{YYYYMMDD: score Series}``。"""
    d = Path(rank_dir)
    out: dict[str, pd.DataFrame] = {}
    for p in sorted(d.glob("ranking_*.csv")):
        ds = p.stem.replace("ranking_", "")
        if not ds.isdigit():
            continue
        if any(ds.endswith(sfx) for sfx in exclude_suffixes):
            continue
        try:
            df = pd.read_csv(p, index_col=0)
        except Exception as e:  # noqa: BLE001
            log.warning("榜单读取失败，跳过 %s: %s", p.name, str(e)[:80])
            continue
        if "score" not in df.columns:
            log.warning("榜单无 score 列，跳过 %s", p.name)
            continue
        ts = pd.to_datetime(ds, format="%Y%m%d")
        out[ds] = df["score"].rename(ts).to_frame().T
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_production_ic(
    rank_dir: str | Path = "reports/alla_daily",
    hist_dir: str | Path = "reports/alla_rolling_ortho",
    cache_root: str | Path | None = None,
    out_path: str | Path | None = None,
    live_begin: str = "20260601",
    check_overlap: bool = True,
) -> pd.DataFrame:
    """跑一轮生产口径 IC/暴露监控，返回并落盘逐日指标表。

    Args:
        rank_dir: 每日出榜产物目录（``ranking_<ds>.csv``）。
        hist_dir: 回测基线目录（``pred/`` + ``_base/``）。
        cache_root: 行情缓存根（默认 ``Config.cache()["root"]``）。
        out_path: 输出 CSV（默认 ``reports/monitoring/production_ic_daily.csv``）。
        live_begin: 实时区间的面板起点（需留出风格协变量滚动窗口的预热期）。
        check_overlap: 重叠区间校验实时重建的协变量与 ``_base`` 是否一致。
    """
    from config import Config

    cache_root = Path(cache_root or str(Config.cache()["root"]))
    hist = load_hist_base(hist_dir)
    live = load_live_base(cache_root, begin=live_begin)

    # 行情面板：历史段 + 实时段（实时覆盖重叠）
    close = _combine(hist["close_adj"], live["close_adj"])
    cov: dict[str, pd.DataFrame] = {}
    for k in set(hist["cov"]) | set(live["cov"]):
        cov[k] = _combine(hist["cov"].get(k), live["cov"].get(k))

    if check_overlap:
        report_overlap(hist["cov"], live["cov"], close.index)

    # 打分面板：历史 OOS + 每日实时
    score = hist["pred"].copy()
    daily = collect_daily_scores(rank_dir)
    if daily:
        score = _combine(score, *daily.values())
    score = score.reindex(columns=close.columns)
    log.info("打分面板 %d 日 × %d 股（%s ~ %s）", len(score), score.shape[1],
             score.index[0].date(), score.index[-1].date())

    return compute_metrics(score, close, cov, out_path=out_path)


def report_overlap(hist_cov: dict[str, pd.DataFrame],
                   live_cov: dict[str, pd.DataFrame],
                   close_index: pd.Index) -> None:
    """校验实时重建的协变量在重叠区间是否复现 ``_base``（口径一致性证据）。"""
    for k, h in hist_cov.items():
        lv = live_cov.get(k)
        if lv is None or k == INDUSTRY_KEY:
            continue
        common = h.index.intersection(lv.index)
        if not len(common):
            continue
        cols = h.columns.intersection(lv.columns)
        d = (h.loc[common, cols] - lv.loc[common, cols]).abs()
        denom = h.loc[common, cols].abs().median().median()
        rel = float(d.max().max()) / float(denom) if denom == denom and denom else np.nan
        log.info("协变量重叠校验 cov_%s: n=%d 日  max|Δ|=%.3e 相对中位=%.4f",
                 k, len(common), float(d.max().max()), rel)


def compute_metrics(score: pd.DataFrame, close: pd.DataFrame,
                    cov: dict[str, pd.DataFrame],
                    out_path: str | Path | None = None) -> pd.DataFrame:
    """由打分面板 / 复权价 / 风格协变量算逐日 IC 与暴露，落盘并返回。"""
    # 停牌/退市边缘可能让 pct_change 产出 ±inf；不清洗会污染组合均值
    # （实测 2026-09-16 截面 top_excess 曾出现 -inf）。
    fwd = close.pct_change(fill_method=None).shift(-1)
    fwd = fwd.replace([np.inf, -np.inf], np.nan)
    ic_raw = daily_rank_ic(score, fwd)

    keys_all = tuple(k for k in (*STYLE_KEYS, INDUSTRY_KEY) if k in cov)
    resid = _neutralize_style(score, cov, keys_all)
    ic_neu = daily_rank_ic(resid, fwd)
    ic_ind = daily_rank_ic(_neutralize_style(score, cov, (INDUSTRY_KEY,)), fwd)
    mono = _monotonicity(score, fwd)

    # 组合口径：Top 10% 等权（与出榜链 frac 一致），事后 + 事前指标
    zs = {k: _zscore_cs(cov[k]) for k in STYLE_KEYS if k in cov}
    rows = []
    for d in score.index:
        s = score.loc[d].dropna()
        if len(s) < MIN_SAMPLES:
            continue
        k = max(int(len(s) * TOP_FRAC), 1)
        top = s.nlargest(k).index
        r = fwd.loc[d].dropna() if d in fwd.index else pd.Series(dtype=float)
        _r = r.reindex(s.index).dropna()
        row: dict[str, Any] = {
            "predict_date": d,
            "next_date": next_trading_day(d, close.index),
            "n_scored": int(len(s)),
            "n_eval": int(len(_r)),
            "n_hold": int(len(top)),
            "ic_raw": ic_raw.get(d, np.nan),
            "ic_neutral": ic_neu.get(d, np.nan),
            "ic_ind_neutral": ic_ind.get(d, np.nan),
            "monotonicity": mono.get(d, np.nan),
            "top_ret": float(_r.reindex(top).mean()) if len(_r) else np.nan,
            "univ_ret": float(_r.mean()) if len(_r) else np.nan,
        }
        row["top_excess"] = (
            row["top_ret"] - row["univ_ret"] if row["top_ret"] == row["top_ret"] else np.nan)
        if row["ic_raw"] == row["ic_raw"] and abs(row["ic_raw"]) > 1e-9 \
                and row["ic_neutral"] == row["ic_neutral"]:
            row["style_exposure_ratio"] = 1.0 - row["ic_neutral"] / row["ic_raw"]
        else:
            row["style_exposure_ratio"] = np.nan
        for kk, zd in zs.items():
            row[f"z_{kk}"] = float(zd.loc[d].reindex(top).mean()) if d in zd.index else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        log.warning("无可用截面，未产出")
        return out
    out = out.sort_values("predict_date").reset_index(drop=True)
    out["source"] = np.where(out["predict_date"] <= pd.Timestamp("2026-09-12"),
                             "backtest", "live")
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False, encoding="utf-8-sig")
        log.info("已落盘 %s（%d 日）", p, len(out))
    return out


def next_trading_day(d: pd.Timestamp, calendar: pd.Index) -> pd.Timestamp | None:
    """面板日历里 ``d`` 的下一交易日（无则 None）。"""
    later = calendar[calendar > d]
    return later[0] if len(later) else None


def summarize(df: pd.DataFrame, window: int = 60) -> dict:
    """把逐日 IC 折成「近期 vs 全期」摘要（复用 stats.monitor 口径）。"""
    from stats.monitor import monitor_ic_series

    if df.empty:
        return {}
    out: dict[str, Any] = {"n_days": int(df["ic_raw"].notna().sum())}
    for col in ("ic_raw", "ic_neutral"):
        s = df.set_index("predict_date")[col].dropna()
        base = monitor_ic_series(s, window=window)
        out[f"{col}_full"] = base["ic_mean_full"]
        out[f"{col}_recent"] = base["ic_mean_recent"]
        out[f"{col}_t_nw"] = base["ic_t_nw_recent"]
    ex = df["top_excess"].dropna()
    if len(ex):
        out["top_excess_recent"] = float(ex.tail(window).mean())
        out["top_excess_full"] = float(ex.mean())
    return out
