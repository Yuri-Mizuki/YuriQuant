"""聚宽因子看板候选探针（2026-10-26）。

## 为什么有这个脚本

`https://www.joinquant.com/view/factorlib/list` 的因子看板（285 个）是在
**HS300 + 近 10 年 + 多空组合Ⅰ + 过滤涨跌停停牌** 单一口径下算的 IC/收益。
本项目此前只按这个口径挑过 4 个候选，2026-10-26 用 **5 个股票池
（hs300/zz500/zz800/zz1000/zzall）× 4 个区间（10y/3y/1y/3m）= 20 个口径**
复核后，结论变了：

- 285 个里 **176 个**在 20 口径下方向全一致（最多 1 个口径反向）；
- 但绝大多数被 `alpha101/alpha158/alpha191/alpha360` 覆盖，真正**未被 950
  特征库覆盖**的是三族：**换手/流动性**、**成交金额**、**残差波动**；
- `momentum`（504d 滞后 21d、半衰 126）**方向不一致**（hs300/zz500 正、
  zz1000/zzall 负）→ 原候选 `jq_mom_504` 降级为「不通过」；
- 数据源其实**不缺**：`daily_all_a`（OHLCV + amount）+ `_base`（复权价、
  流通市值、换手）能算出上述全部量价因子；真缺数据源的只有**分析师一致预期**
  类（earnings_yield / earnyild / predicted_EP / predicted_growth）。

本脚本把「方向一致 + 数据可得」的候选搬到本地因子库口径上复算，回答三件事：

1. **强不强**：全 A、T+1 可交易掩码下的 Rank IC / NW t / 逐年方向；
2. **新不新**：与 `panels_neu` 950 张面板的截面秩相关（重复度）；
3. **能不能用**：对「申万一级行业 + lncap」取残差后的增量 IC，以及月度十分位
   多空的净收益（双边千三成本、5 窗检验）。

   残差基线**刻意不含** `turn20 / vol20 / mom`：换手正是本轮要测的缺口，
   把它们放进基线等于「自己控自己」，会把 VOL20 的增量 IC 压成 0；风格
   暴露是否已由库覆盖，改由第 2 项的 `dup_corr` 回答。

## 候选与口径来源

定义逐字跟着聚宽 `algorithmIntro` 走（见 `CANDIDATES` 里的注释），只有三处
显式近似：

- `jq_dastd_ew`：聚宽用「252 日窗 + 半衰 42 日」的加权标准差；这里用
  `ewm(halflife=42)`（252 日外权重已衰减到 1.6%，可忽略）。
- `jq_resvol` 里的 CMRA：沿用 `build_alla_barra_styles._cmra` 的月度
  max−min 口径（与 `panels_barra` 同源，便于横向对照），非聚宽原式。
- `jq_mom_504`：聚宽是 504 日窗 + 半衰 126 日加权和；这里用
  `log1p(ret).shift(21).ewm(halflife=126)`。

`jq_std20` / `jq_std60` 是**近似对照**（不是严格正对照）：聚宽 Variance20/60
是**收益**标准差，Qlib `STD20/60` 是**价格**标准差/收盘价，两者口径不同但
高度相关。本地实测截面 Spearman：原始 `panels` 0.62~0.85、中性化
`panels_neu` 0.42~0.78。因此预期 `dup_corr1_exact` 落在 0.4~0.9，
**不能**指望接近 1；若 top1 不是 `alpha158_STD20/60`，才说明去重链路有问题。

## 用法

    # 冒烟（3 个候选 + 跳过全库去重）
    python -m scripts.evaluation.jq_factor_probe --only jq_vol5 jq_price_no_fq --no-dup

    # 全量（写 reports/jq_factor_probe/）
    python -m scripts.evaluation.jq_factor_probe

    # 换股票池 / 换 horizon
    python -m scripts.evaluation.jq_factor_probe --universe hs300 --horizons 1,20

    # 只跑某一族
    python -m scripts.evaluation.jq_factor_probe --families turnover liquidity

输出：`jq_probe_h{h}.csv`（逐候选全指标）、`top_jq_probe_h{h}.csv`、
`meta_h{h}.json`。**只读本地 parquet，不依赖 AmazingData SDK。**
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

log = setup_logging("jq_factor_probe")

#: 双边成本（与 `scripts/evaluation/gtja_repro_eval.FEE_RT` 同口径）。
FEE_RT = 0.003

#: 候选注册表：name -> (jq 原名, 族, 说明)。族用于 `--families` 过滤。
CANDIDATES: dict[str, tuple[str, str, str]] = {
    # ---- 换手 / 流动性（950 特征库未覆盖）----
    "jq_vol5": ("VOL5", "turnover", "5 日均换手率"),
    "jq_vol20": ("VOL20", "turnover", "20 日均换手率"),
    "jq_vol60": ("VOL60", "turnover", "60 日均换手率"),
    "jq_vol120": ("VOL120", "turnover", "120 日均换手率"),
    "jq_turn_std20": ("turnover_volatility", "turnover", "20 日换手率标准差"),
    "jq_stom": ("share_turnover_monthly", "turnover", "ln(过去 21 日换手率和)"),
    "jq_stoq": ("avg_share_turnover_quarterly", "turnover",
                "ln(过去 63 日换手率和/3)"),
    "jq_stoa": ("average_share_turnover_annual", "turnover",
                "ln(过去 252 日换手率和/12)"),
    "jq_liq": ("liquidity", "liquidity",
               "0.35·STOM + 0.35·STOQ + 0.30·STOA（聚宽再对市值取残差，"
               "残差化由本脚本的 resid_ic 统一做）"),
    "jq_davol5": ("DAVOL5", "turnover", "5 日 / 120 日均换手"),
    "jq_davol10": ("DAVOL10", "turnover", "10 日 / 120 日均换手"),
    # ---- 波动（950 特征库只有 std/beta，没有残差波动）----
    "jq_dastd_ew": ("daily_standard_deviation", "volatility",
                    "252 日半衰 42 日加权收益标准差"),
    "jq_hsigma": ("historical_sigma", "volatility",
                  "对市场收益回归残差的 252 日标准差"),
    "jq_resvol": ("resvol", "volatility",
                  "0.50·DASTD + 0.42·HSIGMA + 0.08·CMRA"),
    "jq_boll_down": ("boll_down", "volatility",
                     "(MA20 − 2·STD20)/close"),
    "jq_std20": ("Variance20", "control", "正对照：≈ alpha158_STD20"),
    "jq_std60": ("Variance60", "control", "正对照：≈ alpha158_STD60"),
    # ---- 成交金额（alpha158 的 VMA/VSTD 是成交量口径，不是金额）----
    "jq_tvstd6": ("TVSTD6", "amount", "6 日成交金额标准差"),
    "jq_tvstd20": ("TVSTD20", "amount", "20 日成交金额标准差"),
    "jq_tvma6": ("TVMA6", "amount", "6 日成交金额均值"),
    "jq_tvma20": ("TVMA20", "amount", "20 日成交金额均值"),
    # ---- 价格 / 动量 ----
    "jq_price_no_fq": ("price_no_fq", "price", "不复权收盘价（低价效应）"),
    "jq_mom_504": ("momentum", "momentum",
                   "504 日窗滞后 21 日、半衰 126 日加权对数收益"),
    # ---- 需要 OHLC（950 特征库有 alpha158 的 KLEN/KMID，但无这些形态）----
    "jq_mfi14": ("MFI14", "ohlc", "14 日资金流量指标"),
    "jq_cci20": ("CCI20", "ohlc", "20 日顺势指标"),
    "jq_atr14": ("ATR14", "ohlc", "14 日均幅"),
    "jq_vosc": ("VOSC", "ohlc", "VEMA12 与 VEMA26 的差 / VEMA12"),
}

#: 需要 OHLC/volume/amount 的族。
_OHLC_FAMILIES = {"ohlc", "amount"}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _ret(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None)


def _log1p(x: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(invalid="ignore"):
        return np.log1p(x.where(x > -1))


def _ic_stats(ic: pd.Series) -> dict:
    """IC 均值 / IR / 胜率 / Newey-West t / 有效天数（与 residual_ic_probe 同口径）。"""
    from stats.robust_stats import nw_tstat

    s = ic.dropna()
    if len(s) < 2:
        return dict(ic=float("nan"), ir=float("nan"), win=float("nan"),
                    nw_t=float("nan"), n=0)
    sd = float(s.std())
    t_nw, _se, _lag = nw_tstat(s.to_numpy(dtype=float))
    return dict(ic=float(s.mean()),
                ir=float(s.mean() / sd) if sd else float("nan"),
                win=float((s > 0).mean()), nw_t=float(t_nw), n=int(len(s)))


def _yearly(ic: pd.Series) -> dict:
    """逐年 IC 均值 + 同号年数。"""
    out: dict = {}
    if ic.empty:
        return out
    for y, sub in ic.groupby(ic.index.year):
        out[f"ic_{y}"] = float(sub.mean())
    vals = [v for k, v in out.items() if k.startswith("ic_")]
    out["years_pos"] = int(sum(v > 0 for v in vals))
    out["years_n"] = int(len(vals))
    out["years_sign_consistent"] = int(max(out["years_pos"],
                                           out["years_n"] - out["years_pos"]))
    return out


def _rank_panel(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True).astype(np.float32)


def _cs_ic(x_rank: pd.DataFrame, fwd_rank: pd.DataFrame,
           valid: pd.DataFrame) -> pd.Series:
    """逐日截面 Pearson（两边都已 rank → Spearman）；valid 之外的置 NaN。"""
    a = x_rank.where(valid)
    b = fwd_rank.where(valid)
    return a.corrwith(b, axis=1).astype(float)


# ---------------------------------------------------------------------------
# 候选面板构建
# ---------------------------------------------------------------------------
def build_panel(name: str, ctx: dict) -> pd.DataFrame:
    """按候选名构建因子面板（float32）。"""
    turn = ctx["turn"]
    ret = ctx["ret"]
    close = ctx["close"]

    if name.startswith("jq_vol") and name[6:].isdigit():
        n = int(name[6:])
        return turn.rolling(n, min_periods=max(2, n // 2)).mean()
    if name == "jq_turn_std20":
        return turn.rolling(20, min_periods=10).std()
    if name == "jq_stom":
        return _log1p(turn.rolling(21, min_periods=10).sum())
    if name == "jq_stoq":
        return _log1p(turn.rolling(63, min_periods=30).sum() / 3.0)
    if name == "jq_stoa":
        return _log1p(turn.rolling(252, min_periods=120).sum() / 12.0)
    if name == "jq_liq":
        stom = _log1p(turn.rolling(21, min_periods=10).sum())
        stoq = _log1p(turn.rolling(63, min_periods=30).sum() / 3.0)
        stoa = _log1p(turn.rolling(252, min_periods=120).sum() / 12.0)
        return 0.35 * stom + 0.35 * stoq + 0.30 * stoa
    if name == "jq_davol5":
        return (turn.rolling(5, min_periods=3).mean()
                / turn.rolling(120, min_periods=60).mean())
    if name == "jq_davol10":
        return (turn.rolling(10, min_periods=5).mean()
                / turn.rolling(120, min_periods=60).mean())
    if name == "jq_dastd_ew":
        return ret.ewm(halflife=42, min_periods=126).std()
    if name == "jq_hsigma":
        return _hsigma(ret)
    if name == "jq_resvol":
        return (0.50 * ret.ewm(halflife=42, min_periods=126).std()
                + 0.42 * _hsigma(ret)
                + 0.08 * _cmra(ret, close.index))
    if name == "jq_boll_down":
        ma = close.rolling(20, min_periods=10).mean()
        sd = close.rolling(20, min_periods=10).std()
        return (ma - 2.0 * sd) / close
    if name == "jq_std20":
        return ret.rolling(20, min_periods=10).std()
    if name == "jq_std60":
        return ret.rolling(60, min_periods=30).std()
    if name == "jq_tvstd6":
        return ctx["amount"].rolling(6, min_periods=3).std()
    if name == "jq_tvstd20":
        return ctx["amount"].rolling(20, min_periods=10).std()
    if name == "jq_tvma6":
        return ctx["amount"].rolling(6, min_periods=3).mean()
    if name == "jq_tvma20":
        return ctx["amount"].rolling(20, min_periods=10).mean()
    if name == "jq_price_no_fq":
        return ctx["close_raw"]
    if name == "jq_mom_504":
        return _log1p(ret).shift(21).ewm(halflife=126, min_periods=126).mean()
    if name == "jq_mfi14":
        return _mfi(ctx)
    if name == "jq_cci20":
        return _cci(ctx)
    if name == "jq_atr14":
        return _atr(ctx)
    if name == "jq_vosc":
        v = ctx["volume"]
        e12 = v.ewm(span=12, min_periods=6).mean()
        e26 = v.ewm(span=26, min_periods=13).mean()
        return (e12 - e26) / e12.replace(0.0, np.nan) * 100.0
    raise KeyError(f"未注册的候选：{name}")


def _hsigma(ret: pd.DataFrame) -> pd.DataFrame:
    """个股收益对市场等权收益的 252 日回归残差标准差。"""
    mkt = ret.mean(axis=1)
    var_m = mkt.rolling(252, min_periods=126).var()
    beta = ret.rolling(252, min_periods=126).cov(mkt).div(var_m, axis=0)
    resid = ret.sub(beta.mul(mkt, axis=0))
    return resid.rolling(252, min_periods=126).std()


def _cmra(ret: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """CMRA 近似：月度超额对数收益的 12 个月滚动 max−min（与 panels_barra 同源）。"""
    exc = ret.sub(ret.mean(axis=1), axis=0)
    monthly = (1.0 + exc).resample("ME").prod(min_count=15) - 1.0
    log1 = np.log1p(monthly)
    rng = (log1.rolling(12, min_periods=9).max()
           - log1.rolling(12, min_periods=9).min())
    return rng.reindex(idx, method="ffill").astype(float)


def _ohlc(ctx: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """复权后的 (high, low, open)：原始价 × (close_adj / close_raw)。"""
    adj = ctx["close"] / ctx["close_raw"].replace(0.0, np.nan)
    return (ctx["high"] * adj, ctx["low"] * adj, ctx["open"] * adj)


def _mfi(ctx: dict) -> pd.DataFrame:
    """14 日资金流量指标。

    注意：非上涨日要按 **0 流量**参与滚动求和，不能留空——否则 ``min_periods``
    要求 14 日窗口内「上涨日 ≥7 且下跌日 ≥7」同时成立，只剩恰好 7:7 的样本
    （实测覆盖率 0.12）。聚宽口径是窗口内所有交易日的资金流之和。
    """
    h, low, _ = _ohlc(ctx)
    c = ctx["close"]
    tp = (h + low + c) / 3.0
    mf = tp * ctx["volume"]
    ok = mf.notna()
    up = mf.where(tp.diff() > 0, 0.0).where(ok)
    dn = mf.where(tp.diff() < 0, 0.0).where(ok)
    pos = up.rolling(14, min_periods=7).sum()
    neg = dn.rolling(14, min_periods=7).sum()
    ratio = pos / neg.replace(0.0, np.nan)
    out = (100.0 - 100.0 / (1.0 + ratio)).where(neg > 0, 100.0)
    return out.where(neg.notna())


def _cci(ctx: dict) -> pd.DataFrame:
    """20 日顺势指标。"""
    h, low, _ = _ohlc(ctx)
    tp = (h + low + ctx["close"]) / 3.0
    ma = tp.rolling(20, min_periods=10).mean()
    md = (tp - ma).abs().rolling(20, min_periods=10).mean()
    return (tp - ma) / (0.015 * md.replace(0.0, np.nan))


def _atr(ctx: dict) -> pd.DataFrame:
    """14 日均幅：TR = max(H−L, |H−C₋₁|, |L−C₋₁|)。"""
    h, low, _ = _ohlc(ctx)
    c = ctx["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - low).stack(), (h - pc).abs().stack(),
                    (low - pc).abs().stack()], axis=1).max(axis=1).unstack()
    return tr.rolling(14, min_periods=7).mean()


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------
def _load_raw_field(path: Path, field: str) -> pd.DataFrame:
    """从 daily_all_a 取单字段宽表（float32）。"""
    d = pd.read_parquet(path, columns=[field])
    wide = d[field].unstack("code")
    wide.index = pd.DatetimeIndex(wide.index).normalize()
    return wide.astype(np.float32)


def _universe_mask(name: str, idx: pd.DatetimeIndex,
                   cols: pd.Index) -> pd.DataFrame | None:
    """指数成分时序掩码（hs300 / zz1000）；其它返回 None。"""
    files = {"hs300": "index_constituent_000300_SH.parquet",
             "zz1000": "index_constituent_000852_SH.parquet"}
    f = files.get(name)
    if f is None:
        return None
    from config import Config

    p = Path(str(Config.cache()["root"])) / f
    if not p.exists():
        log.warning("成分文件缺失，退回全 A：%s", p)
        return None
    d = pd.read_parquet(p)
    d["in_date"] = pd.to_datetime(d["in_date"].astype(str), format="%Y%m%d",
                                  errors="coerce")
    d["out_date"] = pd.to_datetime(d["out_date"].astype(str), format="%Y%m%d",
                                   errors="coerce")
    out = pd.DataFrame(False, index=idx, columns=cols)
    for _, r in d.iterrows():
        code = str(r["con_code"])
        if code not in cols:
            continue
        lo = r["in_date"] if pd.notna(r["in_date"]) else idx[0]
        hi = r["out_date"] if pd.notna(r["out_date"]) else idx[-1]
        seg = out.loc[(out.index >= lo) & (out.index <= hi), code]
        out.loc[seg.index, code] = True
    log.info("universe=%s：成分覆盖 %.3f", name,
             float(out.to_numpy(dtype=float).mean()))
    return out


# ---------------------------------------------------------------------------
# 增量 IC：因子对 lncap + 申万一级取残差（+ 可选库内最相似因子）
# ---------------------------------------------------------------------------
def _resid_ic_np(xr: np.ndarray, yr: np.ndarray, ok: np.ndarray,
                 ind: np.ndarray, regs: list[np.ndarray],
                 n_ind: int) -> pd.Series:
    """逐日「因子对 size+行业 取残差」后的 IC（残差化**只做在因子侧**）。

    口径与 `research.factor_library._residual_ic` 一致：残差化因子，再与
    **原始**未来收益秩求相关。先对行业哑变量残差化（FWL：等价于行业内
    去均值），再对连续协变量（size 秩等）做 4 元以内 OLS，避免逐日
    5800×35 的 lstsq。
    """
    n_days = xr.shape[0]
    ic = np.full(n_days, np.nan)
    for t in range(n_days):
        m = ok[t] & np.isfinite(xr[t]) & np.isfinite(yr[t])
        for r in regs:
            m &= np.isfinite(r[t])
        codes = ind[t]
        m &= codes >= 0
        n = int(m.sum())
        if n < 50:
            continue
        c = codes[m].astype(np.int64)
        y = xr[t][m].astype(np.float64)
        xv = [r[t][m].astype(np.float64) for r in regs]
        # 行业内去均值（含截距等价于行业哑变量）
        cnt = np.bincount(c, minlength=n_ind)[c].astype(np.float64)
        g = np.bincount(c, weights=y, minlength=n_ind)[c] / cnt
        y = y - g
        for j, v in enumerate(xv):
            gm = np.bincount(c, weights=v, minlength=n_ind)[c] / cnt
            xv[j] = v - gm
        X = np.column_stack(xv)
        try:
            coef = np.linalg.solve(X.T @ X + 1e-12 * np.eye(X.shape[1]),
                                   X.T @ y)
        except np.linalg.LinAlgError:
            continue
        res = y - X @ coef
        if np.std(res) < 1e-12:
            ic[t] = 0.0
            continue
        rt = yr[t][m].astype(np.float64)
        ic[t] = float(np.corrcoef(np.argsort(np.argsort(res)),
                                  np.argsort(np.argsort(rt)))[0, 1])
    return pd.Series(ic, index=pd.RangeIndex(n_days))


# ---------------------------------------------------------------------------
# 月度十分位多空
# ---------------------------------------------------------------------------
def _rebalance_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """月频调仓日 = 每月第一个交易日（与 backtest 引擎同口径）。"""
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).first().values)


def monthly_ls(panel: pd.DataFrame, close: pd.DataFrame, mask: pd.DataFrame,
               sign: int, n_grp: int = 10) -> dict:
    """月度十分位多空（等权、双边千三、5 窗检验）。"""
    reb = _rebalance_dates(close.index)
    reb = reb[reb.isin(panel.index)]
    if len(reb) < 13:
        return {}
    pos = {d: i for i, d in enumerate(close.index)}
    d_next = []
    for d in reb:
        i = pos[d]
        d_next.append(close.index[min(i + 21, len(close.index) - 1)])
    d_next = pd.DatetimeIndex(d_next)

    gross, turns, longs, dates = [], [], [], []
    prev_long: set = set()
    prev_short: set = set()
    for i in range(len(reb) - 1):
        sig = panel.loc[reb[i]] * sign
        nxt = d_next[i]
        pr = close.loc[nxt] / close.loc[reb[i]] - 1.0
        ok = (sig.notna() & pr.notna() & mask.loc[reb[i]]
              & close.loc[reb[i]].notna() & close.loc[nxt].notna())
        if int(ok.sum()) < 100:
            continue
        s = sig[ok]
        r = pr[ok]
        q = pd.qcut(s.rank(method="first"), n_grp, labels=False)
        top = s.index[q == n_grp - 1]
        bot = s.index[q == 0]
        if len(top) < 10 or len(bot) < 10:
            continue
        g = float(r[top].mean() - r[bot].mean())
        cur_l, cur_s = set(top), set(bot)
        to = (len(cur_l - prev_long) / max(1, len(cur_l))
              + len(cur_s - prev_short) / max(1, len(cur_s))) / 2.0
        prev_long, prev_short = cur_l, cur_s
        gross.append(g)
        turns.append(to)
        longs.append(float(r[top].mean()))
        dates.append(reb[i])

    if len(gross) < 12:
        return {}
    gross = pd.Series(gross, index=pd.DatetimeIndex(dates))
    turns = pd.Series(turns, index=gross.index)
    net = gross - FEE_RT * turns
    n_win = 5
    chunks = np.array_split(np.arange(len(net)), n_win)
    w_means = [float(net.iloc[c].mean()) for c in chunks if len(c) >= 6]
    return {
        "ls_ann": float((1 + net).prod() ** (12.0 / len(net)) - 1.0),
        "ls_sharpe": float(net.mean() / net.std() * np.sqrt(12.0))
        if net.std() > 0 else float("nan"),
        "ls_mean_m": float(net.mean()),
        "ls_win_m": float((net > 0).mean()),
        "ls_turnover_m": float(turns.mean()),
        "ls_ann_long": float((1 + pd.Series(longs,
                                           index=gross.index)).prod()
                             ** (12.0 / len(gross)) - 1.0),
        "ls_windows_pass": int(sum(w >= 0.01 for w in w_means)),
        "ls_windows_n": int(len(w_means)),
    }


# ---------------------------------------------------------------------------
# 全库去重（近似 + top1 精确复核）
# ---------------------------------------------------------------------------
def _dup_batch(cand_ranks: dict[str, pd.DataFrame], dup_dir: Path,
               stride: int, top_k: int = 3) -> dict[str, list[tuple[float, str]]]:
    """候选面板 vs 库内全部面板的截面秩相关（近似 Spearman，逐日交集 Pearson）。

    秩在各面板自身有效域内计算，交集上取 Pearson。与库内 `_run_dup_check`
    的「交集上重排名」在覆盖率接近时差异 <0.02；top1 会再精确复核一次
    （见 :func:`_dup_exact`）。
    """
    files = sorted(dup_dir.glob("*.parquet"))
    if not files:
        return {}
    dates = next(iter(cand_ranks.values())).index[::stride]
    codes = next(iter(cand_ranks.values())).columns
    names = list(cand_ranks)
    X = np.stack([cand_ranks[n].reindex(index=dates, columns=codes)
                  .to_numpy(dtype=np.float32) for n in names])  # C×D×N
    Xm = ~np.isfinite(X)
    Xz = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    best: dict[str, list[tuple[float, str]]] = {n: [] for n in names}
    t0 = time.time()
    for i, fp in enumerate(files, 1):
        try:
            d = pd.read_parquet(fp)
        except Exception:  # noqa: BLE001
            continue
        d = d.reindex(index=dates, columns=codes)
        if d.shape[1] == 0 or d.notna().to_numpy(dtype=float).mean() < 0.05:
            continue
        y = d.rank(axis=1).to_numpy(dtype=np.float32)
        Ym = ~np.isfinite(y)
        Yz = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        V = ~(Xm | Ym[None, :, :])
        n = V.sum(axis=2).astype(np.float64)          # C×D
        sx = np.einsum("cdn,cdn->cd", Xz, V)
        sy = np.einsum("cdn,dn->cd", V, Yz)
        sxx = np.einsum("cdn,cdn->cd", Xz * Xz, V)
        syy = (Yz * Yz * V).sum(axis=2)
        sxy = np.einsum("cdn,cdn->cd", Xz * Yz[None, :, :], V)
        with np.errstate(invalid="ignore", divide="ignore"):
            num = sxy - sx * sy / n
            den = np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))
            corr = num / den
        corr[~np.isfinite(corr) | (n < 100)] = np.nan
        mean_corr = np.nanmean(corr, axis=1)
        for j, nm in enumerate(names):
            c = mean_corr[j]
            if np.isfinite(c):
                best[nm].append((float(c), fp.stem))
        if i % 200 == 0:
            log.info("  去重比对 %d/%d（%.0fs）", i, len(files), time.time() - t0)
    return {k: sorted(v, reverse=True)[:top_k] for k, v in best.items()}


def _dup_exact(panel: pd.DataFrame, dup_dir: Path, name: str) -> float:
    """top1 精确复核：交集上重排名后的逐日 Spearman 均值。"""
    p = dup_dir / f"{name}.parquet"
    if not p.exists():
        return float("nan")
    old = pd.read_parquet(p)
    dates = panel.index.intersection(old.index)
    codes = panel.columns.intersection(old.columns)
    if len(dates) < 30 or len(codes) < 10:
        return float("nan")
    a = panel.loc[dates, codes]
    b = old.loc[dates, codes]
    v = a.notna() & b.notna()
    ra = a.where(v).rank(axis=1)
    rb = b.where(v).rank(axis=1)
    return float(ra.corrwith(rb, axis=1).mean())


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--universe", default="all",
                    choices=["all", "hs300", "zz1000"],
                    help="股票池（成分时序由 index_constituent_* 提供）")
    ap.add_argument("--horizons", default="1,20",
                    help="逗号分隔的评估 horizon（默认 1,20）")
    ap.add_argument("--families", default=None,
                    help="逗号分隔族过滤（turnover,liquidity,volatility,"
                         "amount,price,momentum,ohlc,control）")
    ap.add_argument("--only", nargs="+", default=None, help="只跑指定候选名")
    ap.add_argument("--dup-dir", default=None,
                    help="去重对照面板目录（默认 factor_library/.../panels_neu）")
    ap.add_argument("--dup-stride", type=int, default=20,
                    help="去重抽样步长（默认每 20 个交易日取 1 天）")
    ap.add_argument("--no-dup", action="store_true", help="跳过全库去重")
    ap.add_argument("--out-dir", default="reports/jq_factor_probe")
    ap.add_argument("--min-coverage", type=float, default=0.05)
    args = ap.parse_args(argv)

    from scripts.common.alla_common import ds_root, load_base

    base = load_base()
    close = base["close"]
    close_raw = pd.read_parquet(
        ROOT / "reports" / "alla_rolling" / "_base" / "close_raw.parquet")
    mask_all = base["mask"].astype(bool)
    idx, cols = close.index, close.columns

    uni = None
    if args.universe != "all":
        uni = _universe_mask(args.universe, idx, cols)

    names = list(CANDIDATES)
    if args.only:
        names = [n for n in names if n in set(args.only)]
        missing = set(args.only) - set(names)
        if missing:
            log.warning("未注册候选被忽略：%s", sorted(missing))
    if args.families:
        fam = {s.strip() for s in args.families.split(",") if s.strip()}
        names = [n for n in names if CANDIDATES[n][1] in fam]
    if not names:
        raise SystemExit("没有可跑的候选")
    log.info("候选 %d 个：%s", len(names), ", ".join(names))

    need_ohlc = any(CANDIDATES[n][1] in _OHLC_FAMILIES for n in names)
    from config import Config

    raw_p = Path(str(Config.cache()["root"])) / "daily_all_a.parquet"
    ctx = {"close": close, "close_raw": close_raw,
           "turn": base["cov"]["turn"], "ret": _ret(close)}
    if need_ohlc:
        if not raw_p.exists():
            raise SystemExit(f"缺少 {raw_p}，无法构建金额/OHLC 候选")
        log.info("装载 daily_all_a 的 high/low/open/volume/amount ...")
        for f in ("high", "low", "open", "volume", "amount"):
            ctx[f] = _load_raw_field(raw_p, f).reindex(index=idx, columns=cols)

    # 残差化协变量（仅风格控制）：lncap 的**截面秩**（行业以去均值方式等价进入）
    # 统一在秩空间回归，与候选因子的 rank 同尺度。
    size_ln = np.log(base["market_cap"].where(base["market_cap"] > 0))
    ind = base["cov"]["industry"]
    # 申万一级以字符串代码（如 801780.SI）存储 → 全表 factorize 成整数（NaN→-1）
    ind_codes, _ = pd.factorize(ind.to_numpy().ravel())
    ind_np = ind_codes.reshape(ind.shape)
    n_ind = int(ind_np.max()) + 1
    regs_np = [_rank_panel(size_ln).to_numpy(np.float32)]

    dup_dir = (Path(args.dup_dir) if args.dup_dir
               else ds_root() / "panels_neu")
    if not args.no_dup and not dup_dir.exists():
        log.warning("去重目录不存在，自动跳过：%s", dup_dir)
        args.no_dup = True

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 预计算各 horizon 的 fwd rank / valid
    fwd_cache = {}
    for h in horizons:
        fwd = (close.pct_change(h, fill_method=None).shift(-h)
               .replace([np.inf, -np.inf], np.nan).astype(np.float32))
        valid = mask_all & fwd.notna()
        if uni is not None:
            valid = valid & uni.reindex(index=idx, columns=cols)
        fwd_cache[h] = (fwd, _rank_panel(fwd.where(valid)), valid)
    ok_np = fwd_cache[horizons[0]][2].to_numpy()

    rows: list[dict] = []
    dup_ranks: dict[str, pd.DataFrame] = {}
    t0 = time.time()
    for i, name in enumerate(names, 1):
        jq, family, desc = CANDIDATES[name]
        ts = time.time()
        panel = build_panel(name, ctx)
        panel = panel.replace([np.inf, -np.inf], np.nan).astype(np.float32)
        if uni is not None:
            panel = panel.where(uni)
        cov = float(panel.notna().to_numpy(dtype=float).mean())
        if cov < args.min_coverage:
            log.info("[%2d/%d] %-16s 跳过（cov=%.3f）", i, len(names), name, cov)
            continue
        row: dict = {"name": name, "jq_name": jq, "family": family,
                     "desc": desc, "coverage": cov}
        for h in horizons:
            fwd, fwd_r, valid = fwd_cache[h]
            xr = _rank_panel(panel)
            ic = _cs_ic(xr, fwd_r, valid)
            st = _ic_stats(ic)
            row.update({f"ic_h{h}": st["ic"], f"ir_h{h}": st["ir"],
                        f"t_h{h}": st["nw_t"], f"win_h{h}": st["win"],
                        f"n_h{h}": st["n"]})
            if h == horizons[0]:
                row.update(_yearly(ic))
        # 增量 IC（首 horizon）：① 只控 size+行业（回答"是不是风格代理"）
        # ② 再控库内最相似因子（回答"相对现有特征还有没有增量"，去重阶段补算）
        h0 = horizons[0]
        _, fwd_r0, valid0 = fwd_cache[h0]
        sig_ic = row[f"ic_h{h0}"]
        sign0 = int(np.sign(sig_ic)) if np.isfinite(sig_ic) else 1
        resid_ic = _resid_ic_np(
            xr.to_numpy(np.float32), fwd_r0.to_numpy(np.float32),
            ok_np, ind_np, regs_np, n_ind)
        rst = _ic_stats(resid_ic)
        row.update({"resid_si_ic": rst["ic"], "resid_si_t": rst["nw_t"],
                    "resid_si_n": rst["n"]})
        # 月度十分位多空
        row.update(monthly_ls(panel, close, valid0, sign0))
        rows.append(row)
        log.info("[%2d/%d] %-16s cov=%.2f | IC h%s=%+.4f t=%+.2f | "
                 "resid(si)=%+.4f t=%+.2f | LSann=%+.2f%% mean_m=%+.2f%% 5窗%d/5 "
                 "(%.0fs)",
                 i, len(names), name, cov, h0, row[f"ic_h{h0}"],
                 row[f"t_h{h0}"], row["resid_si_ic"], row["resid_si_t"],
                 100 * row.get("ls_ann", float("nan")),
                 100 * row.get("ls_mean_m", float("nan")),
                 row.get("ls_windows_pass", -1), time.time() - ts)
        if not args.no_dup:
            dup_ranks[name] = _rank_panel(panel)
        del panel
        if i % 4 == 0:
            _flush(rows, out_dir, args, horizons)

    _flush(rows, out_dir, args, horizons)
    df = pd.read_csv(out_dir / f"jq_probe_h{horizons[0]}.csv") \
        if (out_dir / f"jq_probe_h{horizons[0]}.csv").exists() else pd.DataFrame()
    if df.empty:
        log.warning("没有产出（全部跳过）")
        return

    if not args.no_dup and dup_ranks:
        log.info("全库去重：%d 候选 × %s", len(dup_ranks), dup_dir)
        best = _dup_batch(dup_ranks, dup_dir, args.dup_stride)
        fwd_r0_np = fwd_cache[horizons[0]][1].to_numpy(np.float32)
        extra = {}
        for name, cands in best.items():
            top1 = cands[0] if cands else (float("nan"), "")
            exact = (_dup_exact(dup_ranks[name], dup_dir, top1[1])
                     if top1[1] else float("nan"))
            rec = {
                "dup_top1": top1[1], "dup_corr1": top1[0],
                "dup_corr1_exact": exact,
                "dup_top2": cands[1][1] if len(cands) > 1 else "",
                "dup_corr2": cands[1][0] if len(cands) > 1 else np.nan,
                "dup_top3": cands[2][1] if len(cands) > 2 else "",
                "dup_corr3": cands[2][0] if len(cands) > 2 else np.nan,
            }
            # 相对库内最相似因子的增量（控 size + 行业 + dup_top1）
            if top1[1]:
                try:
                    old = pd.read_parquet(dup_dir / f"{top1[1]}.parquet")
                    old = old.reindex(index=idx, columns=cols)
                    old_r = _rank_panel(old)
                    ric = _resid_ic_np(
                        dup_ranks[name].to_numpy(np.float32), fwd_r0_np,
                        ok_np, ind_np, regs_np + [old_r.to_numpy(np.float32)],
                        n_ind)
                    st = _ic_stats(ric)
                    rec.update({"resid_dup_ic": st["ic"], "resid_dup_t": st["nw_t"]})
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s 增量 IC 计算失败：%s", name, exc)
            extra[name] = rec
        add = pd.DataFrame([{"name": k, **v} for k, v in extra.items()])
        df = df.merge(add, on="name", how="left")

    df = df.sort_values(f"t_h{horizons[0]}", key=lambda s: s.abs(),
                        ascending=False)
    out_csv = out_dir / f"jq_probe_h{horizons[0]}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df.head(30).to_csv(out_dir / f"top_jq_probe_h{horizons[0]}.csv",
                       index=False, encoding="utf-8-sig")
    meta = {
        "universe": args.universe,
        "horizons": horizons,
        "n_eval_days": int(len(idx)),
        "eval_start": str(idx[0].date()),
        "eval_end": str(idx[-1].date()),
        "dup_dir": None if args.no_dup else str(dup_dir),
        "dup_stride": args.dup_stride,
        "fee_rt": FEE_RT,
        "n_candidates": len(names),
        "n_evaluated": int(len(df)),
        "elapsed_s": round(time.time() - t0, 1),
        "resid_baseline": "申万一级行业内去均值 + lncap 截面秩（刻意不含换手/波动/动量）",
        "ls_rule": "月度首交易日调仓、十分位等权、双边千三、5 窗各 2 年",
        "caveat": ("IC 口径 = 全 A（或指定池）T+1 可交易掩码、截面 Spearman；"
                   "聚宽看板数字来自其自有数据源与单口径，量级可比、逐位不可比。"
                   "dup_corr1 为抽样近似，dup_corr1_exact 为 top1 交集重排名精确值。"),
    }
    (out_dir / f"meta_h{horizons[0]}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("完成 %d 个（%.0fs）-> %s", len(df), time.time() - t0, out_csv)
    show = [c for c in ("name", f"ic_h{horizons[0]}", f"t_h{horizons[0]}",
                        "resid_si_ic", "resid_si_t", "dup_corr1_exact",
                        "resid_dup_ic", "resid_dup_t", "ls_ann",
                        "ls_windows_pass") if c in df.columns]
    log.info("按 |t| 排序前 10：\n%s", df.head(10)[show].to_string(index=False))


def _flush(rows: list[dict], out_dir: Path, args, horizons: list[int]) -> None:
    """增量落盘（保留旧行，按 name 去重）。"""
    if not rows:
        return
    out_csv = out_dir / f"jq_probe_h{horizons[0]}.csv"
    new = pd.DataFrame(rows)
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        # 去重阶段的列由本轮重算，避免重跑时 merge 出 _x/_y 双份
        stale = [c for c in old.columns
                 if c.startswith(("dup_", "resid_dup_"))]
        old = old.drop(columns=stale, errors="ignore")
        new = pd.concat([old[~old["name"].isin(new["name"])], new],
                        ignore_index=True)
    new.to_csv(out_csv, index=False, encoding="utf-8-sig")
    rows.clear()


if __name__ == "__main__":
    main()
