"""申万一级行业上下文：PIT 行业归属 + 自由流通市值权重 + 调仓日
================================================================

**重建说明（2026-11-01）**
--------------------------
本模块替代已丢失的 `scripts/oneoff/etf_industry_factor_p0.py` 中**被下游复用**的
三个函数（`build_context` / `agg_industry`（原名 `_agg`）/ `float_share_panel`）。

丢失原因不是意外，是路径策略：`scripts/oneoff/` 在 `.gitignore:45` 里被整目录忽略，
该文件从未进过 git，也不在 stash、不在这台机器的任何副本里（全盘 find 无结果）。
因此**只能按调用点反推重建**，原件的研究主体（IC 落盘、FDR 聚类、分位表格）没有
下游依赖，其产物 `reports/etf_contest5/industry_factor_p0.*` 同样不在盘上，故未重建。

放这里而不是原地重建，是为了让下次不会再丢：`scripts/builders/` 是受版本管理的目录。

## 调用点（重建时逐条对齐，共 9 处）

| 调用方 | 用到 | 行 |
|---|---|---|
| `scripts/portfolio/report_replica.py` | `build_context`(400) / `_agg`(205, 409) / `float_share_panel`(290) | 4 |
| `scripts/model_train/eval_dl_industry.py` | `_agg`(50) | 1 |
| `scripts/model_train/replica_deep.py` | `float_share_panel`(91) | 1 |
| `scripts/model_train/decomp_dl_industry.py` | `_agg`(30) | 1 |
| `scripts/portfolio/industry_rotation.py` | `build_context`(88) | 1 |

## 契约（调用点能唯一确定的部分）

- `build_context(...) -> dict`，键只有五个：`dates` / `universe` / `lbls` / `wts` / `inds`
- `dates`：调仓日 `list[pd.Timestamp]`
- `universe`：跨期固定的股票池 `pd.Index`；`lbls[i]`、`wts[i]` 与之**等长同序**
- `lbls[i]`：**整数行业序号**（对应 `inds` 的位置），`-1` = 当日无归属
  （`industry_rotation.py:105` 用 `ctx["lbls"][i] >= 0` 过滤 ⇒ 必须是整数而非代码串）
- `wts[i]`：正权重，`0` = 当日该股不可用；`_agg` 内部做 `Σ(w·x)/Σ(w)`，故**不必归一**
  （`industry_rotation.py:109-111` 的等价实现即此结构）
- `inds`：31 个申万一级行业代码（升序），同时是 `_agg` 输出的列序
  （`agg_matrix` 之后 `reindex(columns=px.columns)`，`px` 来自 `sw_industry_daily.parquet`）

## 无法从调用点确定、按研报口径取值的假设（⚠️ 与原件不保证逐位一致）

- **A. 权重口径三选一**（`build_context(weight=...)`，2026-11-01 起）：
  - `float_mcap`（历史基线）= **流通股本 × 当日收盘价**。库内只有 `float_share`
    （**流通**股本，`data/datasource.py:203`）。对 A+H 公司这个字段直接等于总股本
    （601398 流通 = 总股本 3564 亿股）⇒ 工行流通市值算成 2.89 万亿、比 A 股真实流通
    市值高约 50%，方向与「信号在小票、权重在大票」同向。
  - `free_mcap`（本次新增）= **自由流通股本近似 × 收盘价**，其中
    自由流通 ≈ 流通股本 − 持股 ≥5% 的股东所持流通股（`free_float_share_panel`，
    PIT 按**公告日**可用，不按报告期末，避免偷看）。这是近似不是中证口径。
  - `equal`（本次新增）= 行业内**等权**。行业内部怎么加权不影响「买行业 ETF」的
    执行，所以这是**信号质量**臂而非落地臂，恰好对应「等权 IC 0.0423 vs 市值加权 0.0025」
    那个差别到底来自权重还是来自信号本身。
- **B. `agg_industry` 内含 1/99 缩尾，且缩尾在行业内做**。依据是两个调用点的 docstring
  （`eval_dl_industry.py:49`、`report_replica.py:405`）都写「行业内自由流通市值加权 + 1/99 winsorize」；
  原件把缩尾写在 `_agg` 内还是调用前无法区分（`industry_rotation.agg_sue` 在前面自己又做了一次
  全截面 clip，属重复实现，不构成证据）。成员 < 20 只的行业跳过缩尾，避免小样本分位数失真。
- **C. 行业内有效成员 < 5 只 ⇒ 该行业当日记 NaN**（`industry_rotation.py:111` 的等价实现用 `cnt >= 5`）。
- **D. 调仓日 = 每周/每月首个交易日**（`report_replica.rebal_dates` 的原样复刻：先按完整日历
  分组取首日，再按 `start` 过滤，两处顺序不能换，否则 start 落在周中会多吃一天）。
- **E. universe = 行业分类表出现过、且行情表里有收盘价的全部代码（升序）**，跨期固定；
  当日无归属的股票靠 `lbls == -1` 剔除，而不是从池里删掉（删掉会让各期池不同，`lbls[i]`
  与 `wts[i]` 的长度契约就破了）。
- **F. 日历下界 = 2015-01-05**（`calendar.parquet` 起点，也是 `daily_all_a` 起点）。
  `industry_rotation.agg_sue` 会拿 2000 年起的行业指数日期来要上下文，早于 2015 的期
  自然拿不到（`i < 0` 跳过 ⇒ 该期 NaN），不会崩，但也**不要**把它读成"因子缺失"。

## 用法

    python -m scripts.builders.industry_context --check          # 体检：覆盖率/权重/成员数
    python -m scripts.builders.industry_context --check --freq M --start 2021-01-01
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import Config  # noqa: E402
from data.market_cap import build_shares_panel  # noqa: E402
from data.point_in_time import lookup_intervals  # noqa: E402

PQ = Path(Config.paths()["cache_root"])

N_IND = 31            # 申万一级行业数（2021 版）
MIN_MEMBERS = 5       # 行业内最少有效成员数，低于此记 NaN（假设 C）
WINSOR = (1.0, 99.0)  # 行业内缩尾分位（假设 B）
WINSOR_MIN_N = 20     # 成员少于此数不缩尾（分位数不可靠）
SHARE_FIELD = "float_share"
WEIGHT_MODES = ("float_mcap", "free_mcap", "equal")


# --------------------------------------------------------------------- 原始表
@functools.lru_cache(maxsize=1)
def _classification() -> pd.DataFrame:
    """申万一级 PIT 行业归属事件表（长表：code / industry_code / in_date / out_date）。"""
    return pd.read_parquet(PQ / "industry_classification_level1.parquet")


@functools.lru_cache(maxsize=1)
def _equity_structure() -> pd.DataFrame:
    """股本变动事件表（长表：code / change_date / tot_share / float_share，单位万股）。"""
    return pd.read_parquet(PQ / "equity_structure.parquet")


@functools.lru_cache(maxsize=1)
def _share_holder() -> pd.DataFrame:
    """十大股东持股（长表：code / ann_date / holder_end_date / holder_pct /
    holder_quantity / float_quantity）。`float_quantity` 是该股东持有的**流通股数**。"""
    d = pd.read_parquet(PQ / "share_holder.parquet")
    d["code"] = d["code"].astype(str)
    return d


@functools.lru_cache(maxsize=1)
def _calendar() -> pd.DatetimeIndex:
    """A 股交易日历。优先 `calendar.parquet`（轻量），缺失时退回行情索引。"""
    p = PQ / "calendar.parquet"
    if p.exists():
        s = pd.read_parquet(p).iloc[:, 0]
        if not pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s.astype("int64").astype(str), format="%Y%m%d",
                               errors="coerce")
        return pd.DatetimeIndex(s).dropna().sort_values()
    d = pd.read_parquet(PQ / "daily_all_a.parquet", columns=["close"])
    return pd.DatetimeIndex(d.index.get_level_values(0).unique()).sort_values()


@functools.lru_cache(maxsize=4)
def _close_panel(dates: tuple[pd.Timestamp, ...]) -> pd.DataFrame:
    """指定日期 × 全部代码的**原始**收盘价（沪深不复权，市值口径必须用它）。"""
    d = pd.read_parquet(PQ / "daily_all_a.parquet", columns=["close"])
    wide = d["close"].unstack("code")
    return wide.reindex(index=pd.DatetimeIndex(list(dates))).astype(np.float32)


# ------------------------------------------------------------------ 公开函数（一）
def float_share_panel(dates, codes, share_field: str = SHARE_FIELD) -> pd.DataFrame:
    """股本面板（单位：**股**），PIT 前向填充到给定日期。

    Args:
        dates: 目标日期（调仓日或逐日皆可）。
        codes: 股票代码。
        share_field: `"float_share"`（流通股本，默认）或 `"tot_share"`。

    Returns:
        DataFrame(index=dates, columns=codes)，单位股；无数据为 NaN。

    Note:
        直接复用 `data.market_cap.build_shares_panel`（该实现与
        `scripts/pipelines/rolling_grid_alla._shares_panel` 等价、已有测试覆盖），
        避免再写一遍股本事件表的前向填充。
    """
    dates = pd.DatetimeIndex(pd.DatetimeIndex(dates).unique()).sort_values()
    codes = pd.Index(list(codes))
    out = build_shares_panel(_equity_structure(), dates, codes, share_field=share_field)
    out.index.name = "date"
    out.columns.name = "code"
    return out


def rebal_dates(idx: pd.DatetimeIndex, freq: str = "W",
                start: pd.Timestamp | None = None) -> list[pd.Timestamp]:
    """周度：每周首个交易日；月频：每月首个交易日；`"D"`：每个交易日。

    与 `scripts/portfolio/report_replica.rebal_dates` 逐字同义（先分组取首日、
    再按 start 过滤）。这里复刻而**不** import，是为了避免
    `report_replica` ↔ 本模块的循环依赖（前者在函数内 import 本模块）。

    `"D"` 是给**日频调仓诊断**（`scripts/model_train/diag_daily_rebal.py`）用的：
    它能直接检验「alpha 只活一天」这条判断——只有逐日调仓才吃得到日频 alpha。
    """
    idx = pd.DatetimeIndex(idx).sort_values()
    s = pd.Series(idx, index=idx)
    if freq == "D":
        g = s  # 每个交易日都是一个调仓日
    elif freq == "M":
        g = s.groupby([idx.year, idx.month]).first()
    else:
        iso = idx.isocalendar()
        g = s.groupby([iso["year"].to_numpy(), iso["week"].to_numpy()]).first()
    out = [pd.Timestamp(x) for x in g.tolist()]
    if start is not None:
        out = [d for d in out if d >= pd.Timestamp(start)]
    return out


def free_float_share_panel(dates, codes, holder_pct_min: float = 5.0,
                           floor_ratio: float = 0.05,
                           fallback: pd.DataFrame | None = None) -> pd.DataFrame:
    """自由流通股本**近似**（单位：股）：流通股本 − 持股 ≥`holder_pct_min`% 的股东所持流通股。

    研报是「行业内自由流通市值加权」，但库内只有 `float_share`（**流通**股本）——
    对 601398 这类 A+H 公司，`float_share` 直接等于 `tot_share`（含 H 股），
    流通市值会被高估约 50%。本函数用十大股东披露数据扣掉大股东持股做近似。

    Args:
        dates: 目标日期。
        codes: 股票代码。
        holder_pct_min: 扣减门槛（占总股本比例，%）。5% 是《证券法》「举牌线」，
            也是中证自由流通口径里「非自由流通」的常见分界；只扣 top-10 里的
            小比例股东会严重低估自由流通（基金/北向是自由流通盘）。
        floor_ratio: 下限比例，`free ≥ floor_ratio × float_share`，防止扣成 0 或负数。
        fallback: 无股东记录时的回退股本面板（默认用 `float_share`）。

    Returns:
        DataFrame(index=dates, columns=codes)，单位股。

    Note:
        **PIT 口径 = 公告日 `ann_date`**（不是报告期末 `holder_end_date`）：财报要等
        公告才可用，用期末日会偷看。无任何股东记录的股票回退到流通股本（等价于
        「不扣减」），覆盖率见 `describe_free_float`。
    """
    dates = pd.DatetimeIndex(pd.DatetimeIndex(dates).unique()).sort_values()
    codes = pd.Index(sorted(set(map(str, codes))))
    if fallback is None:
        fallback = float_share_panel(dates, codes)

    sh = _share_holder()
    sub = sh[sh["code"].isin(codes) & (sh["holder_pct"] >= holder_pct_min)]
    sub = sub.dropna(subset=["float_quantity"])
    if sub.empty:
        return fallback
    # 🚨 同一份报表里同一股东会出现多次（「十大股东」与「十大流通股东」两张表并列），
    # 不去重会把大股东重复扣 2~4 倍 —— 实测 601398 扣减额 11124 亿股 > 流通股本
    # 3564 亿股，所有股票都被压到 floor。先按 (code, 期末, 股东名) 取 max 再求和。
    g = (sub.groupby(["code", "holder_end_date", "holder_name"], observed=True)
         ["float_quantity"].max().reset_index())
    held = (g.groupby(["code", "holder_end_date"], observed=True)["float_quantity"]
            .sum().rename("held").reset_index())
    # 可用日 = 该期报表的**最早**公告日（拿最晚的会把信息推后一个季度）
    ann = (sub.groupby(["code", "holder_end_date"], observed=True)["ann_date"]
           .min().rename("avail_date").reset_index())
    rep = held.merge(ann, on=["code", "holder_end_date"], how="left")

    # 报表期末的流通股本（用它做被减数，比用公告日的股本更贴报表口径）
    fs_end = float_share_panel(pd.DatetimeIndex(rep["holder_end_date"].unique()), codes)
    fs_long = fs_end.stack().rename("float_sh").reset_index()
    fs_long.columns = ["holder_end_date", "code", "float_sh"]
    rep = rep.merge(fs_long, on=["holder_end_date", "code"], how="left")

    with np.errstate(invalid="ignore"):
        free = rep["float_sh"] - rep["held"].fillna(0.0)
    lo = rep["float_sh"] * floor_ratio
    free = np.where(rep["float_sh"].isna(), np.nan, np.maximum(free, lo))
    free = np.minimum(free, rep["float_sh"])
    rep = rep.assign(free_sh=free)

    wide = rep.pivot_table(index="avail_date", columns="code", values="free_sh",
                           aggfunc="last").sort_index()
    wide = wide.reindex(sorted(set(wide.index) | set(dates))).ffill().reindex(index=dates)
    wide = wide.reindex(columns=codes)
    return wide.fillna(fallback)


def describe_free_float(dates, codes=None) -> pd.DataFrame:
    """自由流通口径体检：扣减前后市值之比、逐行业中位数、覆盖率。"""
    codes = pd.Index(sorted(set(map(str, codes)))) if codes is not None else \
        pd.Index(sorted(_classification()["code"].astype(str).unique()))
    dates = pd.DatetimeIndex(dates)
    fs = float_share_panel(dates, codes)
    ff = free_float_share_panel(dates, codes, fallback=fs)
    sh = _share_holder()
    has = pd.Index(sh.loc[sh["holder_pct"] >= 5.0, "code"].unique())
    cov = len(set(codes) & set(has)) / max(len(codes), 1)
    ratio = (ff / fs.where(fs > 0)).stack().rename("free_over_float")
    out = pd.DataFrame({"free_over_float": ratio})
    print(f"自由流通/流通 比值：中位 {out['free_over_float'].median():.3f}，"
          f"10 分位 {out['free_over_float'].quantile(0.1):.3f}；"
          f"至少一个 ≥5% 股东的股票 {cov:.1%}")
    return out


def build_context(start="2018-01-01", freq: str = "W", end=None,
                  universe=None, weight: str = "float_mcap") -> dict:
    """构建行业上下文（PIT 分类 + 权重 + 调仓日）。

    Args:
        start: 首个调仓日下界（含）。
        freq: `"D"`=日频（每个交易日）、`"W"`=周度（每周首个交易日）、
            `"M"`=月度（每月首个交易日）。
        end: 末个调仓日上界（含），None = 到数据末端。
        universe: 显式指定股票池（必须覆盖所有会用到的代码），None = 自动。
        weight: 权重口径 —— `float_mcap`（流通市值，历史基线）/
            `free_mcap`（自由流通市值近似，见 `free_float_share_panel`）/ `equal`（等权）。
            行业内部怎么加权**不影响**「买行业 ETF」这一步的执行，
            所以 equal 与 mcap 的差异是信号质量问题，不是落地问题。

    Returns:
        dict(dates, universe, lbls, wts, inds, weight)，语义见模块 docstring「契约」。
    """
    if weight not in WEIGHT_MODES:
        raise ValueError(f"未知 weight={weight!r}，只支持 {WEIGHT_MODES}")
    start = pd.Timestamp(start)

    dates = rebal_dates(_calendar(), freq, start)
    if end is not None:
        dates = [d for d in dates if d <= pd.Timestamp(end)]
    if not dates:
        raise ValueError(f"调仓日为空：start={start.date()} freq={freq}")

    rec = _classification()
    close = _close_panel(tuple(dates))
    inds = pd.Index(sorted(rec["industry_code"].dropna().astype(str).unique()))

    if universe is None:
        # 假设 E：分类表里出现过 + 行情表里有价，跨期固定、升序
        uni = pd.Index(sorted(set(rec["code"].astype(str).unique())
                              & set(close.columns.astype(str))))
    else:
        uni = pd.Index(sorted(set(map(str, universe))))

    # 行业归属：PIT 区间展开 → 行业代码 → 整数序号（-1 = 当日无归属）
    panel = lookup_intervals(rec, pd.DatetimeIndex(dates), uni,
                             code_col="code", value_col="industry_code")
    vals = panel.to_numpy(dtype=object)
    ids = np.full(vals.shape, -1, dtype=np.int16)
    for k, name in enumerate(inds):
        ids[vals == name] = k

    # 权重（三种口径，见 `weight` 参数）。非正/缺失记 0 = 当日不可用
    px = close.reindex(columns=uni).to_numpy(dtype=np.float64)
    fs = float_share_panel(pd.DatetimeIndex(dates), uni).to_numpy(dtype=np.float64)
    if weight == "equal":
        w = np.where(np.isfinite(px) & (px > 0), 1.0, 0.0)
    else:
        sh = fs if weight == "float_mcap" else free_float_share_panel(
            pd.DatetimeIndex(dates), uni, fallback=float_share_panel(
                pd.DatetimeIndex(dates), uni)).to_numpy(dtype=np.float64)
        w = px * sh
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

    return {
        "dates": dates,
        "universe": uni,
        "lbls": [ids[i] for i in range(len(dates))],
        "wts": [w[i] for i in range(len(dates))],
        "inds": inds,
        "weight": weight,
    }


# ------------------------------------------------------------------ 公开函数（二）
def agg_industry(v, lbl, w, min_members: int = MIN_MEMBERS,
                 winsor: tuple[float, float] | None = WINSOR,
                 n_ind: int = N_IND) -> np.ndarray:
    """个股截面 → 行业截面：行业内自由流通市值加权均值（含 1/99 缩尾）。

    Args:
        v: 个股值（长度 = 股票数），NaN 忽略。
        lbl: 个股行业序号（`-1` 或超界 = 剔除）。
        w: 个股权重（自由流通市值），`<=0` 或 NaN = 剔除；无需归一。
        min_members: 行业内有效成员下限，不足则该行业 NaN。
        winsor: 行业内缩尾分位，None = 不缩尾。
        n_ind: 行业数（输出长度）。

    Returns:
        np.ndarray(n_ind)，无有效成员的行业为 NaN。

    Note:
        与 `industry_rotation.agg_sue:107-113` 的等价实现同结构
        （`Σ(w·v)/Σ(w)`、`cnt >= 5`），差别只在多一层缩尾（假设 B）。
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    lbl = np.asarray(lbl).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    out = np.full(n_ind, np.nan)
    if v.size == 0 or not (v.size == lbl.size == w.size):
        return out

    ok = np.isfinite(v) & np.isfinite(w) & (w > 0) & (lbl >= 0) & (lbl < n_ind)
    if not ok.any():
        return out
    x, L, W = v[ok], lbl[ok].astype(np.int64), w[ok]

    if winsor is not None:
        g = pd.DataFrame({"x": x, "g": L}).groupby("g")["x"]
        q = g.quantile([winsor[0] / 100.0, winsor[1] / 100.0]).unstack()
        cnt = g.size()
        lo = q.iloc[:, 0].reindex(q.index).to_numpy()
        hi = q.iloc[:, 1].reindex(q.index).to_numpy()
        big = cnt.reindex(q.index).to_numpy() >= WINSOR_MIN_N
        lo = np.where(big, lo, -np.inf)
        hi = np.where(big, hi, np.inf)
        lo_map = pd.Series(lo, index=q.index).reindex(L).to_numpy()
        hi_map = pd.Series(hi, index=q.index).reindex(L).to_numpy()
        x = np.clip(x, lo_map, hi_map)

    num = np.bincount(L, weights=x * W, minlength=n_ind)
    den = np.bincount(L, weights=W, minlength=n_ind)
    cnt_all = np.bincount(L, minlength=n_ind)
    good = (cnt_all >= min_members) & (den > 0)
    out[good] = num[good] / den[good]
    return out


# 兼容旧名（原 `_agg`）；新代码请用 `agg_industry`
_agg = agg_industry


# --------------------------------------------------------------------- 体检
def describe(ctx: dict, n_head: int = 5) -> pd.DataFrame:
    """逐调仓日体检：覆盖股票数、行业数、权重合计、单行业成员数分位。"""
    rows = []
    for i, dt in enumerate(ctx["dates"]):
        lbl, w = ctx["lbls"][i], ctx["wts"][i]
        use = (lbl >= 0) & (w > 0)
        if not use.any():
            rows.append({"date": dt, "n_stock": 0, "n_ind": 0, "w_sum": 0.0,
                         "mem_min": 0, "mem_med": 0, "mem_max": 0})
            continue
        mem = np.bincount(lbl[use], minlength=len(ctx["inds"]))
        mem = mem[mem > 0]
        rows.append({"date": dt, "n_stock": int(use.sum()), "n_ind": int(mem.size),
                     "w_sum": float(w[use].sum()), "mem_min": int(mem.min()),
                     "mem_med": int(np.median(mem)), "mem_max": int(mem.max())})
    df = pd.DataFrame(rows).set_index("date")
    if len(df):
        print(f"调仓日 {df.index[0].date()} ~ {df.index[-1].date()}，共 {len(df)} 期")
        print(f"  行业数 {int(df['n_ind'].min())}~{int(df['n_ind'].max())}"
              f"（期望 ≤ {len(ctx['inds'])}，2021-12 前少 3 个）")
        print(f"  有效股票数 {int(df['n_stock'].min())}~{int(df['n_stock'].max())}"
              f"，中位 {int(df['n_stock'].median())}")
        print(f"  单行业成员数 中位 {int(df['mem_med'].median())}，"
              f"最小 {int(df['mem_min'].min())}（< {MIN_MEMBERS} 会被记 NaN）")
        print(f"  权重合计 中位 {df['w_sum'].median() / 1e8:.1f} 亿元")
        print(df.head(n_head).to_string())
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="行业上下文体检")
    ap.add_argument("--check", action="store_true", help="只体检不落盘（本脚本恒不落盘）")
    ap.add_argument("--weight", default="float_mcap", choices=list(WEIGHT_MODES),
                    help="权重口径：float_mcap / free_mcap / equal")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--freq", default="W", choices=["W", "M"])
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    ctx = build_context(args.start, freq=args.freq, end=args.end, weight=args.weight)
    print(f"universe {len(ctx['universe'])} 只，行业 {len(ctx['inds'])} 个，"
          f"权重={ctx['weight']}：{', '.join(ctx['inds'][:4])} … {ctx['inds'][-1]}")
    describe(ctx)


if __name__ == "__main__":
    main()
