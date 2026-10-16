"""生产口径每日 IC / 风格暴露监控（全A · 历史段 ens_h1h5/ortho · 实时段 defv/dingban）。

问题
----
``monitoring/`` 现有链路监控的是因子库数据集（``hs300_2022_2025``），其行情源
``daily_hs300.parquet`` 自 2026-08-21 起停更；而生产在跑的是**全A**口径
（``scripts/pipelines/alla_daily_rank.py``）。两条链路互不相干 —— 生产模型
实际上**没有任何按日推进的 IC 记录**，无法区分「单日噪音」与「持续退化」。

本模块把生产口径接到一条按日推进的时间序列上，且历史段与每日段**共用同一段
指标代码**，可直接拼成一条序列：

- 历史种子：``reports/alla_rolling_ortho/pred/ens_h1h5__h1.parquet``（回测 OOS
  预测面板，**882 ortho 口径**，止于 2026-09-07）与
  ``_base/{close_adj,cov_*}.parquet``（**与历史消融网格完全同一套风格协变量**，
  保证口径可比）。本机无 fundind 回测 pred（真源 ``alla_rolling_ortho920tl``
  不在本机、regen 需数小时），故历史段**保留 ortho 口径**——见下方「注意」。
- 每日增量：``reports/alla_daily_defv/ranking_<ds>.csv``（2026-10-10 起定版
  dingban 口径出榜）+ 本地行情缓存。

产物
----
``reports/monitoring/production_ic_daily.csv``（按 ``predict_date`` 幂等覆盖）。
``source`` 列动态划分：``predict_date <= hist pred 面板末日`` → ``backtest``，
否则 → ``live``（不再硬编码日期，口径切换日变动会自动跟随）。

口径
----
- 收益：后复权 close 的 T→T+1 收益（信号 T 日收盘产生、T+1 收盘计收益）。
- ``ic_raw`` = 每日截面 Spearman(score, 次日收益)。
- ``ic_neutral`` = 先用 :func:`factor.preprocessing.neutralize` 对当日 score 做
  五风格中性化（size/industry/mom/vol/turn）取残差，再算 Spearman。
- ``style_exposure_ratio`` = ``1 − ic_neutral / ic_raw``（raw IC 被风格解释的比例）。
- ``signal_mktcap_spearman`` = 逐日截面 Spearman(score, size)。信号-市值漂移
  监控（国泰海通多粒度 DL 复盘 → 方案 D①）：多粒度因子 2026 失效的病根即此
  漂移（研报口径 corr 与市值 −0.3 且逐年偏移）。本项目因子层已全正交，但 gbdt
  非线性组合可能重新长出市值暴露，故须逐年监控。注意：``cov["size"]`` 是 raw
  市值面板（log 只在 neutralize 内部做），与 882 口径一次性诊断
  （``scripts/oneoff/_signal_mktcap_drift.py``，中性化面板）存在系统差——
  raw 口径语义更直接，红牌判据看**逐年恶化趋势**而非绝对值对齐。
- ``z_*`` = 组合持仓在对应风格上的横截面 z 均值（事前可算，不需未来收益）。

注意
----
🚨 **口径断点（2026-10-16 披露，联动件④）**：历史段 = 882 ortho 口径回测 OOS
（``ens_h1h5``），实时段 = 定版 dingban（defv 榜，``h1-h20 秩平均 ⊕ s0.5``）。
两段**信号口径不同**，切换日（≈2026-09-07 后首个 defv 榜单日）两侧 IC 不可直接
比绝对水平，只比段内趋势。原因：本机无 fundind 回测 pred（``config/baselines.yaml``
``v1-dingban`` 条目 ``reproducible_local: false``），regen 需数小时，等面板重建
链路空出内存后再补——届时只需重跑本模块（历史种子换成 fundind pred 即可）。
此外历史段是回测面板、实时段是每日链实时打分，**模型配置相同但特征来源路径
不同**（历史段走回测面板，实时段走 ``load_tail``），本身也有系统差。
``source`` 列已把两段分开，跨段汇总（如 ``summarize`` 全期均值）解读时须注明。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("monitoring.production_ic")

#: 历史种子文件名：882 ortho 口径回测 OOS 预测面板（h1+h5 秩平均、horizon=1）。
#: 注意：实时段（defv dingban）信号是 h1+h5+h10+h20 秩平均 ⊕ s0.5 慢信号，
#: 与历史种子**不同口径**——详见模块 docstring「注意」。
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
    rank_dir: str | Path = "reports/alla_daily_defv",
    hist_dir: str | Path = "reports/alla_rolling_ortho",
    cache_root: str | Path | None = None,
    out_path: str | Path | None = None,
    live_begin: str = "20260601",
    check_overlap: bool = True,
) -> pd.DataFrame:
    """跑一轮生产口径 IC/暴露监控，返回并落盘逐日指标表。

    Args:
        rank_dir: 每日出榜产物目录（``ranking_<ds>.csv``）。**2026-10-10 起默认
            ``reports/alla_daily_defv``**（定版 dingban 口径）；旧 legacy 口径产物
            在 ``reports/alla_daily``（停于 2026-09-22）。
        hist_dir: 回测基线目录（``pred/`` + ``_base/``）。历史段仍是 882 ortho
            口径（本机无 fundind 回测 pred）⇒ 与实时段（dingban）之间存在口径
            断点，见模块 docstring「注意」；``source`` 列分界 = hist pred 末日。
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

    # source 分界 = 历史种子（回测 OOS pred）面板末日：其后出现的任何打分截面
    # （无论来自 legacy 榜还是 defv 榜）都算 live。口径断点见模块 docstring。
    hist_end = hist["pred"].index.max()
    return compute_metrics(score, close, cov, out_path=out_path, hist_end=hist_end)


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
                    out_path: str | Path | None = None,
                    hist_end: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """由打分面板 / 复权价 / 风格协变量算逐日 IC 与暴露，落盘并返回。

    Args:
        hist_end: 历史段（backtest）与实时段（live）的分界日——
            ``predict_date <= hist_end`` 标 ``backtest``，否则 ``live``。
            调用方（:func:`run_production_ic`）传 hist pred 面板末日；
            传 ``None`` 时全部标 ``live``（无历史段概念的场景，如测试）。
    """
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

    # 信号-市值漂移（方案 D①）：逐日 Spearman(score, raw 市值)。
    # daily_rank_ic 对两个已 rank 面板即逐日截面 Spearman（秩变换后 Pearson）。
    if "size" in cov:
        drift = daily_rank_ic(score, cov["size"])
    else:
        drift = pd.Series(dtype=float)
        log.warning("缺 cov['size']，signal_mktcap_spearman 列将全 NaN")

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
            "signal_mktcap_spearman": drift.get(d, np.nan),
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
    # source 动态划分（联动件④）：分界 = hist pred 面板末日，不再硬编码日期。
    # 硬编码 "2026-09-12" 是 legacy 榜切换残留，与 defv 首榜（2026-10-08）不符。
    if hist_end is not None:
        he = pd.Timestamp(hist_end)
        out["source"] = np.where(out["predict_date"] <= he, "backtest", "live")
    else:
        out["source"] = "live"
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
    # 信号-市值漂移摘要：全期 vs 近窗均值 + 逐年序列（红牌判据 = 逐年恶化趋势）
    if "signal_mktcap_spearman" in df.columns:
        dr = df.set_index("predict_date")["signal_mktcap_spearman"].dropna()
        if len(dr):
            out["mktcap_drift_full"] = float(dr.mean())
            out["mktcap_drift_recent"] = float(dr.tail(window).mean())
            out["mktcap_drift_yearly"] = {
                int(y): round(float(v), 4)
                for y, v in dr.groupby(dr.index.year).mean().items()}
    return out
