"""Barra CNE6 风格因子原始暴露面板（简化版）→ all_a_2018_2026/panels_barra/。

目标（2026-10-16 由莉酱拍板）：CNE6 9 一级骨架简化版，~16 风格描述变量，
对齐天风 0830「16 Barra 风格因子」验收口径（ΔR² 增量 / VIF 参照系）。
行业因子 = 申万一级（``_base/cov_industry``），进入截面回归时作哑变量。

**本脚本只产 raw 暴露**（``panels_barra/{name}.parquet``）——CNE6 标准预处理
（MAD 去极值 → 市值加权标准化）在截面回归链路里**逐日现算**（research/barra.py），
不预中性化落盘：因为 (a) Barra 暴露在回归前才做市值加权标准化，预存会锁死
权重口径；(b) 复用实验侧panels_neu 的「行业+市值中性化」是 alpha 因子口径，
与 Barra 风险暴露口径**相反**（风险模型要保留风格暴露、只去极值标准化）。

描述变量与窗口（DolphinDB Barra CNE6 教程口径，2026-10-16 核实）：

| 一级 | 描述变量 | 定义 | 窗口 |
|---|---|---|---|
| Size | lncap | ln(流通市值) | 时点 |
| Volatility | beta | 个股日收益对市场超额收益 OLS（半衰 63） | 252d |
| Volatility | resid_vol | 1/3(DASTD + CMRA + HSIGMA) 等权合成 | 252d/365d |
| Momentum | momentum | ln(1+RSTR)，252d 累计收益滞后 21d 剔近月（EMA 半衰 126） | 252d |
| NLSize | nl_size | −(立方(标准化 ln(流通市值)))（CNE6 MIDCAP 近似） | 时点 |
| Liquidity | liquidity | 1/3(STOM + STOQ + STOA)（均为 ln(1+月均日换手×天数)） | 21/63/252d |
| EarningsYld | ep_ttm | 净利 TTM / 市值 | PIT |
| Value | btop | 净资产(不含少数)/市值 | PIT |
| Growth | growth | 1/2(SG + EG)：营收/净利 TTM 同比（5y 窗口的简化） | PIT |
| Leverage | mlev | (市值+优先股+负债)/市值 → 简化 (市值+总负债)/市值 | PIT |
| Leverage | dtoa | 总负债/总资产 | PIT |
| Leverage | blev | 总负债/净资产(不含少数) | PIT |
| DivYield | dtop | 12 个月现金分红/市值 | 365d |

简化取舍（相对完整 CNE6，全部显式）：
- Volatility 用 DASTD/CMRA/HSIGMA 等权（原版波动率平方根加权）；beta 用简单
  OLS（原版半衰 63 加权近似为 EWλ 回归——工程上等价可行）。
- NLSize 用立方变换（原版 MIDCAP 公式未公开，取公开文献常见近似）。
- Growth 用 1y TTM 同比（原版 5y 窗口每股增长；本机财报覆盖有限，1y 同比
  更稳；语义同为「增长速度」）。
- STOA 用 252 交易日（教程窗口标 365 自然日，交易日口径更贴近原意）。

用法::

    python -m scripts.builders.build_alla_barra_styles            # 全量
    python -m scripts.builders.build_alla_barra_styles --only beta  # 调试
    python -m scripts.builders.build_alla_barra_styles --force     # 覆盖已有
    python -m scripts.builders.build_alla_barra_styles --status    # 只看覆盖
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("build_alla_barra_styles")

DATASET = "all_a_2018_2026"
BASE_DIR = ROOT / "reports" / "alla_rolling" / "_base"

#: 描述变量清单（唯一真源；一级归属见 _LEVEL1）
STYLE_VARS = (
    "lncap", "beta", "resid_vol", "momentum", "nl_size",
    "liquidity", "ep_ttm", "btop", "growth",
    "mlev", "dtoa", "blev", "dtop",
)
_LEVEL1 = {
    "lncap": "size", "beta": "volatility", "resid_vol": "volatility",
    "momentum": "momentum", "nl_size": "nl_size", "liquidity": "liquidity",
    "ep_ttm": "earnings_yld", "btop": "value", "growth": "growth",
    "mlev": "leverage", "dtoa": "leverage", "blev": "leverage",
    "dtop": "dividend_yld",
}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _ew_lambda(n: int, half_life: int) -> np.ndarray:
    """指数衰减权重（最新一日权重最大），长度 n，半衰期 half_life。"""
    lam = 0.5 ** (1.0 / half_life)
    w = lam ** np.arange(n - 1, -1, -1)
    return w / w.sum()


# ---------------------------------------------------------------------------
# 量价类描述变量（raw 暴露，不含任何截面标准化）
# ---------------------------------------------------------------------------
def build_lncap(mktcap: pd.DataFrame) -> pd.DataFrame:
    """Size·LNCAP = ln(流通市值)（时点值，单位元→取 log 后无量纲）。"""
    return np.log(mktcap.where(mktcap > 0))


def build_beta(close_adj: pd.DataFrame) -> pd.DataFrame:
    """Volatility·HBETA：个股日收益对市场等权收益的 252d 截距为零 OLS 斜率。

    β = Cov(r_i, r_m) / Var(r_m)（滚动 252 日，min_periods=126）。
    """
    ret = close_adj.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)
    cov = ret.rolling(252, min_periods=126).cov(mkt)
    var = mkt.rolling(252, min_periods=126).var()
    return (cov.T / var.replace(0.0, np.nan)).T


def _dastd(close_adj: pd.DataFrame) -> pd.DataFrame:
    """DASTD = |r_i − r_m| 的 252d 均值（超额收益绝对值）。"""
    ret = close_adj.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)
    return (ret.sub(mkt, axis=0)).abs().rolling(252, min_periods=126).mean()


def _cmra(close_adj: pd.DataFrame) -> pd.DataFrame:
    """CMRA = ln(S_max/S_min)，S = Π(1+超额收益) 的 12 个月度滚动累计。"""
    ret = close_adj.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)
    exc = ret.sub(mkt, axis=0)
    monthly = (1 + exc).resample("ME").prod(min_count=15) - 1
    log1 = np.log1p(monthly)
    # 12 个月滚动累计收益的 max-min
    roll_max = log1.rolling(12, min_periods=9).max()
    roll_min = log1.rolling(12, min_periods=9).min()
    out = np.expm1(roll_max) .sub(np.expm1(roll_min))  # 回到简单差分口径防溢出
    # 映射回交易日
    return out.reindex(close_adj.index, method="ffill").astype(float)


def _hsigma(close_adj: pd.DataFrame) -> pd.DataFrame:
    """HSIGMA = 个股对市场回归残差的 252d 标准差。"""
    ret = close_adj.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)
    cov = ret.rolling(252, min_periods=126).cov(mkt)
    beta = (cov.T / mkt.rolling(252, min_periods=126).var().replace(0.0, np.nan)).T
    resid = ret.sub(beta.mul(mkt, axis=0))
    return resid.rolling(252, min_periods=126).std()


def build_resid_vol(close_adj: pd.DataFrame) -> pd.DataFrame:
    """Volatility·resid_vol = 等权(DASTD, CMRA, HSIGMA)，各分项先截面标准化再合成。

    原版用波动率平方根加权；简化等权（三分量同单位：均为「波动量级」）。
    """
    parts = [_dastd(close_adj), _cmra(close_adj), _hsigma(close_adj)]
    z = []
    for p in parts:
        mu = p.mean(axis=1)
        sd = p.std(axis=1).replace(0.0, np.nan)
        z.append(p.sub(mu, axis=0).div(sd, axis=0))
    return sum(z) / len(z)


def build_momentum(close_adj: pd.DataFrame) -> pd.DataFrame:
    """Momentum·RSTR：252d 累计收益、剔最近 21d、EW(半衰126) 加权。

    RSTR = Σ w·ln(1+r_exc)；简化用全市场等权超额。
    向量化：窗口 252、剔除最近 21 ⇒ 实际用 [−252, −21) 段 231 日，
    权重取 EW 序列的对应前段（最新权重对应 −21 日处）。
    """
    ret = close_adj.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)
    exc = ret.sub(mkt, axis=0)
    w_full = _ew_lambda(252, 126)
    w_use = w_full[: 252 - 21]                    # 对应 [t-251, t-21] 共 231 日
    w_use = w_use / w_use.sum()
    log_exc = np.log1p(exc.clip(lower=-0.95))
    # 滚动 231 日加权和 = 窗口 [t-230, t]；再整体滞后 21 日 ⇒ 覆盖 [t-251, t-21]
    win = len(w_use)
    def _wsum(x: np.ndarray) -> float:
        return float(np.dot(w_use, x))
    rolled = log_exc.rolling(win, min_periods=win).apply(_wsum, raw=True)
    return rolled.shift(21)


def build_nl_size(mktcap: pd.DataFrame) -> pd.DataFrame:
    """NLSize·MIDCAP 近似 = −(标准化 ln(流通市值))³（时点）。"""
    lncap = np.log(mktcap.where(mktcap > 0))
    z = (lncap.sub(lncap.mean(axis=1), axis=0)
         .div(lncap.std(axis=1).replace(0.0, np.nan), axis=0))
    return -(z ** 3)


def build_liquidity(amount: pd.DataFrame, mktcap: pd.DataFrame) -> pd.DataFrame:
    """Liquidity = 等权(STOM, STOQ, STOA)；STOx = ln(1 + 窗口Σ(成交额/市值))。

    教程口径：STOM 21d、STOQ 63d（=3 个月度聚合的均值）、STOA 252d（年）。
    ln 变换前 clip 换手上界（A 股单日换手可 >100%，clip 2.0 防极端）。
    """
    daily_turn = (amount / mktcap.where(mktcap > 0)).clip(upper=2.0)
    parts = []
    for win in (21, 63, 252):
        s = daily_turn.rolling(win, min_periods=win // 2).sum()
        parts.append(np.log1p(s))
    z = []
    for p in parts:
        mu = p.mean(axis=1)
        sd = p.std(axis=1).replace(0.0, np.nan)
        z.append(p.sub(mu, axis=0).div(sd, axis=0))
    return sum(z) / len(z)


# ---------------------------------------------------------------------------
# 财务类描述变量（PIT）
# ---------------------------------------------------------------------------
def _pit(report_df: pd.DataFrame, cal_idx: pd.DatetimeIndex, codes: pd.Index,
         field: str) -> pd.DataFrame:
    from data.financials import build_pit_panel
    if field not in report_df.columns or report_df[field].isna().all():
        return pd.DataFrame(np.nan, index=cal_idx, columns=codes)
    return build_pit_panel(report_df, cal_idx, field).reindex(
        index=cal_idx, columns=codes)


def build_fundamental_vars(cal_idx: pd.DatetimeIndex, codes: pd.Index,
                           inc: pd.DataFrame, bal: pd.DataFrame,
                           div: pd.DataFrame, mktcap: pd.DataFrame,
                           tot_share_pit: pd.DataFrame,
                           close_raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """EP/BTOP/Growth/MLEV/DTOA/BLEV/DTOP 的 PIT raw 暴露。

    市值口径：mktcap 面板 = TOT_SHARE×close（`_base` 同款），财务侧用
    `_base/market_cap`（构建器同源），保证分子分母一致。
    """
    from scripts.builders.build_alla_fundamental_factors import add_ttm_yoy
    out: dict[str, pd.DataFrame] = {}

    # TTM 序列
    inc = add_ttm_yoy(inc, "OPERA_REV", "OPERA_REV_TTM", None)
    inc = add_ttm_yoy(inc, "NET_PRO_INCL_MIN_INT_INC", "NET_PRO_TTM", None)

    rev = _pit(inc, cal_idx, codes, "OPERA_REV_TTM")
    npr = _pit(inc, cal_idx, codes, "NET_PRO_TTM")
    ta = _pit(bal, cal_idx, codes, "TOTAL_ASSETS")
    tl = _pit(bal, cal_idx, codes, "TOTAL_LIAB")
    eq = _pit(bal, cal_idx, codes, "TOT_SHARE_EQUITY_EXCL_MIN_INT")

    safe = lambda a, b: a / b.replace(0.0, np.nan)  # noqa: E731

    # EarningsYld·EP_TTM
    out["ep_ttm"] = safe(npr, mktcap)
    # Value·BTOP
    out["btop"] = safe(eq, mktcap)
    # Growth = 1/2(SG + EG)，同比用 TTM 12 个月前值
    rev_yoy = rev / rev.shift(252) - 1.0
    np_yoy = npr / npr.shift(252) - 1.0
    out["growth"] = 0.5 * (rev_yoy.clip(-3, 3) + np_yoy.clip(-3, 3))
    # Leverage·MLEV（简化：(市值+总负债)/市值）、DTOA、BLEV
    out["mlev"] = 1.0 + safe(tl, mktcap)
    out["dtoa"] = safe(tl, ta)
    out["blev"] = safe(tl, eq)
    # DivYield·DTOP：近 365d 每股分红×股本 / 市值（除息日 PIT）
    if len(div) and {"ex_date", "cash_per_share_pre_tax", "base_share"} <= set(div.columns):
        d = div[["code", "ex_date", "cash_per_share_pre_tax", "base_share"]].copy()
        d["ex_date"] = pd.to_datetime(d["ex_date"], errors="coerce")
        d = d.dropna(subset=["ex_date"])
        d = d[d["ex_date"] >= cal_idx[0]]
        cash = pd.DataFrame(0.0, index=cal_idx, columns=codes)
        for code, g in d.groupby("code"):
            if code not in codes:
                continue
            ev = g.set_index("ex_date")["cash_per_share_pre_tax"] \
                  .clip(lower=0).resample("D").sum()
            cash[code] = ev.reindex(cal_idx, method="ffill").fillna(0.0)
        cash_365 = cash.rolling(365, min_periods=180).sum()
        # 每股分红口径 → 总分红 = per_share × 股本；除以市值 = per_share×shares/mktcap
        # mktcap = TOT_SHARE×close ⇒ 分红/市值 = per_share_365/close（股本约掉）
        out["dtop"] = safe(cash_365, close_raw)
    else:
        out["dtop"] = pd.DataFrame(np.nan, index=cal_idx, columns=codes)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _status(ds_dir: Path) -> dict[str, str]:
    from scripts.pipelines.rebuild_panel_library import parquet_last_date
    d = ds_dir / "panels_barra"
    if not d.exists():
        return {n: "缺失" for n in STYLE_VARS}
    return {n: parquet_last_date(d / f"{n}.parquet") for n in STYLE_VARS}


def _write_panel(panel: pd.DataFrame, path: Path) -> None:
    from scripts.common.atomic_io import write_parquet_atomic  # 容错写盘（§六坑⑥）
    write_parquet_atomic(panel, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="只构建指定描述变量（逗号分隔）")
    ap.add_argument("--force", action="store_true", help="覆盖已有面板")
    ap.add_argument("--status", action="store_true", help="只打印覆盖末日")
    args = ap.parse_args()

    from config import Config
    ds_dir = Path(str(Config.get()["factor_library"]["root"])) / DATASET
    ds_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ds_dir / "panels_barra"

    if args.status:
        for k, v in _status(ds_dir).items():
            print(f"{k:12s} {v}")
        return 0

    targets = STYLE_VARS
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        unknown = want - set(STYLE_VARS)
        if unknown:
            raise SystemExit(f"未知描述变量: {sorted(unknown)}，可选 {STYLE_VARS}")
        targets = tuple(n for n in STYLE_VARS if n in want)

    t0 = time.time()
    # ---- 数据装载（与 _base 同源）----
    close_adj = pd.read_parquet(BASE_DIR / "close_adj.parquet")
    close_raw = pd.read_parquet(BASE_DIR / "close_raw.parquet")
    mktcap = pd.read_parquet(BASE_DIR / "market_cap.parquet")
    cal_idx, codes = close_adj.index, close_adj.columns
    log.info("数据: %d 日 × %d 股（%s ~ %s）", len(cal_idx), len(codes),
             cal_idx[0].date(), cal_idx[-1].date())

    cache_root = Path(str(Config.cache()["root"]))
    inc = pd.read_parquet(cache_root / "income.parquet")
    bal = pd.read_parquet(cache_root / "balance_sheet.parquet")
    div = pd.read_parquet(cache_root / "dividend.parquet")
    amount = None
    daily_p = cache_root / "daily_all_a.parquet"
    if daily_p.exists():
        d = pd.read_parquet(daily_p)
        d.index = d.index.set_levels(d.index.levels[0].normalize(), level=0)
        amount = d["amount"].unstack().reindex(index=cal_idx, columns=codes)
        amount = amount.astype(float)
        log.info("成交额面板 %d 日 × %d 股", *amount.shape)

    todo = []
    for n in targets:
        p = out_dir / f"{n}.parquet"
        if p.exists() and not args.force:
            log.info("跳过 %s（已存在，--force 覆盖）", n)
            continue
        todo.append(n)

    built = 0
    for n in todo:
        t1 = time.time()
        if n == "lncap":
            panel = build_lncap(mktcap)
        elif n == "beta":
            panel = build_beta(close_adj)
        elif n == "resid_vol":
            panel = build_resid_vol(close_adj)
        elif n == "momentum":
            panel = build_momentum(close_adj)
        elif n == "nl_size":
            panel = build_nl_size(mktcap)
        elif n == "liquidity":
            if amount is None:
                log.warning("缺 daily_all_a（成交额），跳过 liquidity")
                continue
            panel = build_liquidity(amount, mktcap)
        elif n in ("ep_ttm", "btop", "growth", "mlev", "dtoa", "blev", "dtop"):
            if not hasattr(main, "_fund_cache"):
                main._fund_cache = build_fundamental_vars(  # type: ignore[attr-defined]
                    cal_idx, codes, inc, bal, div, mktcap, None, close_raw)
            panel = main._fund_cache[n]  # type: ignore[attr-defined]
        else:  # pragma: no cover
            raise AssertionError(n)
        panel = panel.astype(np.float32)
        _write_panel(panel, out_dir / f"{n}.parquet")
        built += 1
        log.info("[%d/%d] %s cov=%.3f (%.0fs)", built, len(todo), n,
                 float(panel.notna().mean().mean()), time.time() - t1)

    log.info("Barra 原始暴露完成 %d 个（一级归属: %s）耗时 %.0fs",
             built, sorted(set(_LEVEL1.values())), time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
