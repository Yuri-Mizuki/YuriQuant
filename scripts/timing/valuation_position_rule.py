"""估值定仓位规则（个人投资用）：沪深300 估值/风险状态 → 目标仓位。

规则（参数一次定死，不在样本内调参）::

    W_val  = clip(0.60 + 0.90 * (0.50 - P_pe), 0.10, 1.00)   # P_pe = PE(TTM) 10年分位（EMA 半衰期 1 个月）
    tilt   = 0.45 * clip((5日均价/200日线 - 1) / 0.15, -1, 1)# 趋势加成：距年线 ±15% 饱和 → ±45pp，平滑
    scale  = clip(0.20 / vol60_ann, 0.50, 1.00)              # 只减不加（风险预算）
    W_raw  = clip((W_val + tilt) * scale, 0.10, 1.00)         # 最终仓位区间 10%~100%
    W_t    = 惰性执行：|W_raw - W_{t-1}| < 10pp 不动；单次变动 ≤ 10pp（月频）

仓位语义 = 股票（沪深300，含股息代理）占比，其余配国债指数（000012.SH）。
三个状态量：贵不贵（PE 分位）、风险高不高（60 日波动）、趋势在不在（200 日线）。
**方向预测类指标（择时线已证伪）一律不进信号**，无信息量的指标不上面板。

回测口径（判据预注册，**纯研究口径**）：
- 收益率口径：沪深300 价格指数 + 2.2%/年 股息（全收益代理），各腿同口径。
- 月频：PE 为月末观测值（乐咕乐股），决策用 t 月末收盘数据，**t+1 交易日收盘执行**。
- 成本：研究口径不扣交易成本（实盘 12bp 的敏感性见 README §三）。
- 对照：① 固定 60/40（60% 国证A指 + 40% 国债指数，月频再平衡）；
  ② 满仓国证A指（全市场基准）；③ 满仓沪深300（规则标的对照）。
- 通过判据（三条全过才算过）：Calmar 高于 60/40；Sharpe 不低于 60/40 − 0.05；
  最大回撤小于 60/40 与满仓两者。

用法::

    # ① 刷新估值/利率缓存（需装有 akshare 的解释器，本机 D:\\Python\\Python312）
    D:\\Python\\Python312\\python.exe -m scripts.timing.valuation_position_rule ^
        --refresh-data --root E:/data/parquet
    # ② 回测 + 生成面板（项目 .venv）
    .venv\\Scripts\\python.exe -m scripts.timing.valuation_position_rule
    .venv\\Scripts\\python.exe -m scripts.timing.valuation_position_rule --self-check
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "valuation_position_rule"
INDEX = "000300.SH"
BOND = "000012.SH"          # 上证国债指数（含票息，作为现金端的可投资替代）
BENCH = "399317.SZ"         # 国证A指（全市场），纯研究基准：比沪深300 更代表「大盘」
BEGIN = "2015-05-01"        # PE 10年滚动分位在 2015-04 起有满 120 个月观测
COST_BP = 0.0               # 研究口径：不扣交易成本（实盘 12bp 的敏感性见 README）
DEADBAND = 0.10             # 惰性区间（10pp 内不动）
MAX_STEP = 0.10             # 单次最大变动（10pp）
VAL_SLOPE = 0.90            # W = 0.60 + 0.90*(0.5 - P)
VAL_BASE = 0.60
W_LO, W_HI = 0.10, 1.00     # 最终仓位区间（含 scale/trend 之后）
VOL_TARGET = 0.20           # 目标年化波动（scale 分子）
SCALE_LO, SCALE_HI = 0.50, 1.00
DIV_YIELD = 0.022           # 沪深300 股息率假设：价格指数 → 全收益代理（三条腿同口径）
TREND_MA = 200              # 趋势均线（日）
TREND_PX_MA = 5             # 趋势分子用 5 日均价（去掉单日长阴线的冲击，仍无未来信息）
TREND_SPAN = 0.15           # 距年线 ±15% 映射到趋势分数 ±1（饱和）
TREND_TILT = 0.45           # 趋势加成幅度：±45pp（幅度要求取网格上端；k=0.35~0.45 全过判据）
VAL_SMOOTH = 21             # 估值分位 EMA 半衰期（交易日，≈1 个月）：摊平月末阶梯，代价是滞后半个月


# ------------------------------------------------------------------ 路径 / 数据
def _cache_root(root: str | None = None) -> Path:
    """parquet 缓存根：显式参数 > $YQ_DATA_ROOT > config > 本机缺省。"""
    if root:
        return Path(root)
    env = os.environ.get("YQ_DATA_ROOT")
    if env:
        return Path(env) / "parquet"
    try:
        from config import Config
        return Path(str(Config.cache()["root"]))
    except Exception:  # noqa: BLE001 - 无项目依赖时的兜底（akshare 解释器常见）
        return Path("D:/data/parquet")


def refresh_data(root: Path) -> None:
    """拉取并缓存 PE(TTM) 月度序列与 10 年国债收益率（需要 akshare）。"""
    import akshare as ak

    root.mkdir(parents=True, exist_ok=True)

    # --- 沪深300 PE(TTM)：乐咕乐股，月末观测，2005-04 起 -------------------
    df = ak.stock_index_pe_lg(symbol="沪深300")
    pe = pd.DataFrame({
        "pe_ttm": pd.to_numeric(df["滚动市盈率"], errors="coerce").values,
        "pe_static": pd.to_numeric(df["静态市盈率"], errors="coerce").values,
    }, index=pd.to_datetime(df["日期"]))
    pe.index.name = "date"
    pe = pe.dropna().sort_index()
    pe.to_parquet(root / "index_valuation_000300.parquet")
    print(f"[refresh] PE(TTM) {len(pe)} 行  {pe.index.min():%Y-%m-%d} → {pe.index.max():%Y-%m-%d}")

    # --- 中债国债收益率曲线：按年分段拉取（长区间查询会返回空）-------------
    frames = []
    end_year = pd.Timestamp.today().year
    for y in range(2015, end_year + 1):
        try:
            frames.append(ak.bond_china_yield(start_date=f"{y}0101", end_date=f"{y}1231"))
        except Exception as e:  # noqa: BLE001 - 单年失败不拖垮整段
            print(f"[refresh] bond {y} 失败: {e}")
    raw = pd.concat(frames, ignore_index=True)
    g = raw[raw["曲线名称"] == "中债国债收益率曲线"].copy()
    bond = pd.DataFrame({
        "y1y": pd.to_numeric(g["1年"], errors="coerce").values,
        "y10y": pd.to_numeric(g["10年"], errors="coerce").values,
    }, index=pd.to_datetime(g["日期"]))
    bond.index.name = "date"
    bond = bond.dropna().sort_index()
    bond = bond[~bond.index.duplicated(keep="last")]
    bond.to_parquet(root / "bond_yield.parquet")
    print(f"[refresh] 国债收益率 {len(bond)} 行  {bond.index.min():%Y-%m-%d} → {bond.index.max():%Y-%m-%d}")


def load_pe(root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / "index_valuation_000300.parquet").sort_index()


def load_bond(root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / "bond_yield.parquet").sort_index()


def _index_frame(code: str, root: Path) -> pd.DataFrame:
    """指数日线（date 索引）：open/high/low/close/volume/amount。"""
    safe = code.replace(".", "_")
    df = pd.read_parquet(root / f"index_daily_{safe}.parquet")
    if "code" in getattr(df.index, "names", ()):
        df = df.xs(df.index.get_level_values("code").unique()[0], level="code")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = pd.DatetimeIndex(df.index)
    return df


# ------------------------------------------------------------------ 指标面板
def _rolling_pct_rank(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """滚动分位：当前值在过去 window 个观测中的位置（0~1）。"""
    return s.rolling(window, min_periods=min_periods).apply(
        lambda w: float((w[:-1] < w[-1]).mean()), raw=True)


def build_panel(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (日频指标面板, 月频决策面板)。

    全部指标只用 t 日（或 t 月末）收盘及更早的数据，无未来信息。
    """
    px = _index_frame(INDEX, root)
    bench_px = _index_frame(BENCH, root)
    bond = _index_frame(BOND, root)
    pe_m = load_pe(root)
    y = load_bond(root)

    dates = px.index[px.index >= pd.Timestamp(BEGIN)]
    ret_s = px["close"].pct_change(fill_method=None)
    ret_b = bond["close"].pct_change(fill_method=None).reindex(dates).ffill()

    # --- 估值：PE(TTM) 月末 + 10 年滚动分位（120 个月） ---------------------
    pe_m = pe_m[pe_m.index >= pd.Timestamp("2004-01-01")]
    p_pe_m = _rolling_pct_rank(pe_m["pe_ttm"], 120, 60)
    y10_m = y["y10y"].reindex(pe_m.index, method="ffill")
    erp_m = 100.0 / pe_m["pe_ttm"] - y10_m
    p_erp_m = _rolling_pct_rank(erp_m, 120, 60)

    # --- 风险/情绪：日频 ---------------------------------------------------
    vol60 = (ret_s.rolling(60, min_periods=40).std() * np.sqrt(252)).reindex(dates)
    ma200 = px["close"].rolling(TREND_MA, min_periods=int(TREND_MA * 0.6)).mean()
    px_ma200 = (px["close"] / ma200 - 1.0).reindex(dates)
    # 趋势项用 5 日均价而非单日收盘：避免一根长阴线把仓位一次性打下去（仍是因果、无未来信息）
    px_ma200_s = (px["close"].rolling(TREND_PX_MA, min_periods=1).mean() / ma200 - 1.0).reindex(dates)
    ret_tr = (ret_s + DIV_YIELD / 252).reindex(dates)   # 全收益代理（股息按日摊）
    ret_bench = (bench_px["close"].pct_change(fill_method=None) + DIV_YIELD / 252).reindex(dates)

    # 两融余额 20 日动量（全市场融资余额合计）
    md = pd.read_parquet(root / "margin_detail.parquet",
                         columns=["code", "trade_date", "borrow_money_balance"])
    md_dt = pd.to_datetime(md["trade_date"], unit="ns")
    margin_total = md.groupby(md_dt)["borrow_money_balance"].sum().sort_index()
    margin_mom = margin_total.pct_change(20, fill_method=None)
    margin_mom.index = pd.DatetimeIndex(margin_mom.index)
    margin_last = margin_mom.index.max()
    margin_mom = margin_mom.reindex(dates, method="ffill")
    del md, md_dt

    panel = pd.DataFrame({
        "close": px["close"].reindex(dates),
        "ret": ret_tr,
        "ret_px": ret_s.reindex(dates),
        "ret_bench": ret_bench,
        "bench_close": bench_px["close"].reindex(dates),
        "ret_bond": ret_b,
        "pe_ttm": pe_m["pe_ttm"].reindex(dates, method="ffill"),
        "p_pe": p_pe_m.reindex(dates, method="ffill"),
        "y10": y10_m.reindex(dates, method="ffill"),
        "erp": erp_m.reindex(dates, method="ffill"),
        "p_erp": p_erp_m.reindex(dates, method="ffill"),
        "vol60": vol60,
        "margin_mom20": margin_mom,
        "px_ma200": px_ma200,
        "px_ma200_s": px_ma200_s,
    })
    # 估值分位平滑：月度观测在日频图上会形成 15~20pp 的月末阶梯（如 2026-11-20 分位 47%→65%），
    # 用半衰期 1 个月（21 个交易日）的因果 EMA 把阶梯摊平；代价是信号滞后约半个月
    panel["p_pe_s"] = panel["p_pe"].ewm(halflife=VAL_SMOOTH, min_periods=1).mean()

    # --- 月频决策面板：PE 月末观测 ∩ 交易日 ---------------------------------
    m_dates = pe_m.index[pe_m.index >= pd.Timestamp(BEGIN)]
    m_dates = m_dates[m_dates <= dates.max()]
    monthly = panel.reindex(m_dates, method="ffill")[
        ["pe_ttm", "p_pe", "p_pe_s", "erp", "p_erp", "vol60", "margin_mom20",
         "px_ma200", "px_ma200_s"]].copy()
    monthly["decision_date"] = monthly.index
    monthly.attrs["margin_last"] = margin_last
    return panel, monthly


# ------------------------------------------------------------------ 规则
def trend_tilt(ma_pct: float) -> float:
    """平滑趋势加成：距年线 ±15% 饱和，线性映射到 ±TREND_TILT，无跳变。"""
    if not np.isfinite(ma_pct):
        return 0.0
    return float(TREND_TILT * np.clip(ma_pct / TREND_SPAN, -1.0, 1.0))


def target_weight(p_pe: float, vol60: float, ma_pct: float = 0.0,
                  trend_gate: bool = True) -> tuple[float, float, float]:
    """原始目标仓位 → (W_val, scale, W_raw)。

    p_pe 用 EMA 平滑后的 PE 分位（`p_pe_s`）；ma_pct = 5 日均价/200日线 − 1。
    """
    if not np.isfinite(p_pe):
        return np.nan, np.nan, np.nan
    w_val = float(np.clip(VAL_BASE + VAL_SLOPE * (0.5 - p_pe), W_LO, W_HI))
    if not np.isfinite(vol60) or vol60 <= 0:
        scale = 1.0
    else:
        scale = float(np.clip(VOL_TARGET / vol60, SCALE_LO, SCALE_HI))
    w = w_val + (trend_tilt(ma_pct) if trend_gate else 0.0)
    return w_val, scale, float(np.clip(w * scale, W_LO, W_HI))


def apply_lazy(raw: pd.Series, init: float = 0.60) -> pd.Series:
    """惰性执行：deadband 内不动 + 单次变动上限 MAX_STEP。"""
    out, prev = {}, init
    for dt, w in raw.items():
        if not np.isfinite(w):
            out[dt] = prev
            continue
        w = prev if abs(w - prev) < DEADBAND else prev + float(np.clip(w - prev, -MAX_STEP, MAX_STEP))
        out[dt] = w
        prev = w
    return pd.Series(out, name="target")


def daily_raw_weights(panel: pd.DataFrame) -> pd.Series:
    """日频「实时参考仓位」：把当日平滑估值分位 / 60日波动 / 5日均价趋势代入公式，不过惰性带。"""
    p_pe = panel["p_pe_s"].values.astype(float)
    vol = panel["vol60"].values.astype(float)
    ma = panel["px_ma200_s"].values.astype(float)
    w_val = np.clip(VAL_BASE + VAL_SLOPE * (0.5 - p_pe), W_LO, W_HI)
    scale = np.ones(len(panel))
    ok = np.isfinite(vol) & (vol > 0)
    scale[ok] = np.clip(VOL_TARGET / vol[ok], SCALE_LO, SCALE_HI)
    tilt = TREND_TILT * np.clip(ma / TREND_SPAN, -1.0, 1.0)
    tilt[~np.isfinite(ma)] = 0.0
    w = np.clip((w_val + tilt) * scale, W_LO, W_HI)
    w[~np.isfinite(p_pe)] = np.nan
    return pd.Series(w, index=panel.index, name="w_raw_daily")


# ------------------------------------------------------------------ 回测
def simulate(panel: pd.DataFrame, targets: pd.Series,
             init_w: float, cost_bp: float = COST_BP,
             ret_col: str = "ret") -> pd.DataFrame:
    """按目标仓位模拟：t 月末决策、t+1 交易日收盘执行，权重日间随涨跌漂移。"""
    dates = panel.index
    ret_s = panel[ret_col].fillna(0.0).values
    ret_b = panel["ret_bond"].fillna(0.0).values
    # 决策日 → 下一个交易日（执行日）
    exec_map = {}
    pos = {d: i for i, d in enumerate(dates)}
    for dt, w in targets.items():
        if dt not in pos:
            continue
        j = pos[dt] + 1
        if j < len(dates):
            exec_map[j] = float(w)

    w = float(init_w)
    rows = []
    for i, dt in enumerate(dates):
        r = w * ret_s[i] + (1 - w) * ret_b[i]
        ws = w * (1 + ret_s[i])
        wb = (1 - w) * (1 + ret_b[i])
        tot = ws + wb
        w = ws / tot if tot > 0 else w
        cost = 0.0
        if i in exec_map:
            tgt = exec_map[i]
            cost = abs(tgt - w) * cost_bp * 1e-4
            r -= cost
            w = tgt
        rows.append((dt, r, w, cost))
    out = pd.DataFrame(rows, columns=["date", "ret", "weight", "cost"]).set_index("date")
    out["nav"] = (1 + out["ret"]).cumprod()
    return out


def bench_fixed(panel: pd.DataFrame, w_fixed: float, rebal_dates: pd.Index,
                cost_bp: float = COST_BP, ret_col: str = "ret") -> pd.DataFrame:
    """固定权重基准（月频再平衡，同成本）；w_fixed=1 即满仓无成本。"""
    targets = pd.Series(w_fixed, index=rebal_dates)
    return simulate(panel, targets, init_w=w_fixed,
                    cost_bp=cost_bp if w_fixed < 1 else 0.0, ret_col=ret_col)


def perf(series: pd.Series) -> dict:
    from backtest.metrics import (annual_return, annual_volatility,
                                  calmar_ratio, max_drawdown, sharpe_ratio)
    r = series.dropna()
    return {
        "annual_return": annual_return(r),
        "annual_vol": annual_volatility(r),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar_ratio(r),
        "total_return": float((1 + r).prod() - 1),
    }


# ------------------------------------------------------------------ 面板
def _level_pe(p: float) -> str:
    """PE 分位 → 便宜/昂贵。"""
    if not np.isfinite(p):
        return "-"
    for cut, name in ((0.20, "便宜"), (0.40, "偏便宜"), (0.60, "中性"),
                      (0.80, "偏贵"), (1.01, "昂贵")):
        if p < cut:
            return name
    return "昂贵"


def _level_erp(p: float) -> str:
    """ERP 分位 → 股债性价比（越高越偏股）。"""
    if not np.isfinite(p):
        return "-"
    if p >= 0.80:
        return "股性价比高"
    if p <= 0.20:
        return "股性价比低"
    return "中性"


def _pct(v: float, digits: int = 1) -> str:
    return "-" if not np.isfinite(v) else f"{v * 100:.{digits}f}%"


def _render_panel(monthly: pd.DataFrame, panel: pd.DataFrame, res: dict,
                  margin_last) -> Path:
    from research.html_report import page

    last = monthly.dropna(subset=["p_pe"]).iloc[-1]
    tgt = res["strategy"]["targets"]
    today = panel.index.max()

    # 参考仓位 = 公式当日原始输出（不受惰性带约束）；月频执行值另行标注
    p_pe_live = float(last["p_pe_s"])               # 平滑后的分位（进公式）
    vol_live = float(panel["vol60"].iloc[-1])
    ma_live = float(panel["px_ma200_s"].iloc[-1])   # 5 日均价 vs 200 日线（进公式）
    trend_live = trend_tilt(ma_live)             # 趋势加成（pp），可正可负、平滑
    w_val, scale, w_raw = target_weight(p_pe_live, vol_live, ma_live)
    last_target = float(tgt.iloc[-1])           # 最近月频确认的执行值（含惰性）
    live = float(w_raw) if np.isfinite(w_raw) else last_target
    gap = live - last_target
    gap_txt = ("与公式一致" if abs(gap) < 1e-9
               else f"差 {gap * 100:+.0f}pp 惰性带内" if abs(gap) < DEADBAND
               else f"差 {gap * 100:+.0f}pp 已越带")
    try:
        margin_stale = pd.Timestamp(margin_last) < today - pd.Timedelta(days=30)
    except (ValueError, TypeError):
        margin_stale = True

    def _rank(col: str) -> float:
        """指标在近 10 年日频窗口中的分位（以最新交易日为准）。"""
        s = panel[col].dropna()
        if len(s) < 252:
            return float("nan")
        return float((s.iloc[-7560:] < s.iloc[-1]).mean())

    def _gauge(pct: float, good_high: bool = False) -> str:
        """彩色数据条：统一「左红右绿」，标记位 = 对仓位有利程度（越靠右越好）。"""
        if not np.isfinite(pct):
            return "<div class='gauge empty'></div>"
        pos = float(np.clip(pct if good_high else 1.0 - pct, 0.0, 1.0)) * 100
        return f"<div class='gauge'><span style='left:{pos:.1f}%'></span></div>"

    def _ind(name: str, tag: str, value: str, pct: float, status: str,
             reverse: bool = False, better: str = "") -> str:
        """reverse=True 表示越高越好；两条路径都换算成「越靠右越有利」。"""
        pct_txt = "" if not np.isfinite(pct) else f" · 十年分位 {_pct(pct, 0)}"
        dir_txt = f"<span class='dir'>{better}</span>" if better else ""
        return (f"<div class='ind'><div class='ind-head'>"
                f"<span class='ind-name'>{name}<span class='tag'>{tag}</span>{dir_txt}</span>"
                f"<span class='ind-val'><b>{value}</b>{pct_txt} <em>{status}</em></span></div>"
                f"{_gauge(pct, reverse)}</div>")

    tilt_val = trend_live
    rows = [
        _ind("沪深300 PE TTM", "① 估值", f"{last['pe_ttm']:.2f}", last["p_pe"],
             _level_pe(last["p_pe"]), better="↓ 越低越好"),
        _ind("ERP", "参考", f"{last['erp']:.2f}%", last["p_erp"],
             _level_erp(last["p_erp"]), reverse=True, better="↑ 越高越好"),
        _ind("60日波动", "风险", _pct(vol_live), _rank("vol60"),
             "高" if vol_live > 0.25 else ("低" if vol_live < 0.15 else "中"),
             better="↓ 越低越好"),
        _ind("5日均价 vs 200日线", "② 趋势", _pct(ma_live), _rank("px_ma200"),
             "趋势上" if ma_live >= 0 else "趋势下", reverse=True, better="↑ 越高越好"),
        _ind("两融余额 20日动量", "参考",
             _pct(last["margin_mom20"]) if np.isfinite(last["margin_mom20"]) else "-",
             _rank("margin_mom20"),
             "加杠杆" if last["margin_mom20"] > 0.02 else
             ("去杠杆" if last["margin_mom20"] < -0.02 else "平稳"),
             better="↓ 越低越好"),
    ]

    chart_data = {
        "strategy": {"dates": [str(d.date()) for d in res["strategy"]["sim"].index],
                     "nav": [round(float(v), 4) for v in res["strategy"]["sim"]["nav"]]},
        "full": {"dates": [str(d.date()) for d in res["full"]["sim"].index],
                 "nav": [round(float(v), 4) for v in res["full"]["sim"]["nav"]]},
    }

    # 历史仓位曲线：公式日频输出（实时参考仓位） vs 月频执行值
    pos_dates = panel.index[panel["p_pe"].notna() & panel["vol60"].notna()]
    raw_d = res["w_raw_daily"].reindex(pos_dates)
    exec_d = tgt.reindex(pos_dates, method="ffill")

    def _pct_series(s: pd.Series) -> list:
        return [None if not np.isfinite(v) else round(float(v) * 100, 1) for v in s]

    chart_data["pos"] = {"dates": [str(d.date()) for d in pos_dates],
                         "raw": _pct_series(raw_d), "exec": _pct_series(exec_d),
                         "bench": [None if not np.isfinite(v) else round(float(v), 1)
                                   for v in panel.loc[pos_dates, "bench_close"]]}
    data_json = json.dumps(chart_data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    body = f"""
<div class="hero">
  <div class="hero-main">
    <div class="hero-label">今日参考仓位 · {today:%Y-%m-%d}</div>
    <div class="hero-num">{live * 100:.0f}<small>%</small></div>
    <div class="hero-sub">沪深300 ETF 510300 {live * 100:.0f}% · 国债 ETF {100 - live * 100:.0f}%</div>
    <div class="hero-note">月频执行值 {last_target * 100:.0f}% · {gap_txt}</div>
  </div>
  <div class="hero-decomp">
    <div><span>① 估值维度</span><b>{w_val * 100:.0f}%</b></div>
    <div><span>② 趋势加成</span><b>{tilt_val * 100:+.0f}pp</b></div>
    <div><span>③ 波动缩放</span><b>×{scale:.2f}</b></div>
    <div><span>④ 月频执行</span><b>{last_target * 100:.0f}%</b></div>
  </div>
</div>

<div class="card">
  <h2>历史参考仓位</h2>
  <div class="chartbox pos"><canvas id="posChart"></canvas></div>
  <div class="posctl">
    <span class="posctl-end">最初</span>
    <input id="posRange" class="posrng" type="range" min="0" max="0" value="0" step="1">
    <span class="posctl-end">最新</span>
  </div>
  <div class="foot" id="posLabel"></div>
</div>

<div class="card">
  <h2>指标当前位置</h2>
  {''.join(rows)}
</div>

<div class="card">
  <h2>回测对比 · 未扣成本</h2>
  <div class="chartbox"><canvas id="navChart"></canvas></div>
</div>
"""
    extra_css = """
.hero{display:flex;gap:24px;background:#16304f;color:#fff;border-radius:14px;padding:22px 26px;margin-bottom:16px;flex-wrap:wrap}
.hero-main{min-width:220px}
.hero-label{font-size:13px;opacity:.75}
.hero-num{font-size:64px;font-weight:700;line-height:1.05;margin:4px 0}
.hero-num small{font-size:26px;font-weight:500;margin-left:2px}
.hero-sub{font-size:13px;opacity:.8}
.hero-note{font-size:12px;opacity:.6;margin-top:6px}
.hero-decomp{display:grid;grid-template-columns:repeat(2,minmax(150px,1fr));gap:8px 26px;margin-left:auto}
.hero-decomp span{display:block;font-size:12px;opacity:.7}
.hero-decomp b{font-size:18px;font-weight:600}
.ind{padding:12px 0;border-bottom:1px solid #eef1f5}
.ind:last-child{border-bottom:none}
.ind-head{display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap}
.ind-name{font-size:14px;font-weight:600;color:#1f2937}
.dir{font-size:11px;color:#9aa4b2;margin-left:6px;font-weight:500}
.tag{display:inline-block;margin-left:8px;padding:1px 7px;border-radius:9px;background:#eef2f7;color:#5b6b7f;font-size:11px;font-weight:500;vertical-align:1px}
.ind-val{font-size:13px;color:#6b7280}
.ind-val b{font-size:15px;color:#111827;margin-right:4px}
.ind-val em{font-style:normal;font-weight:600;color:#1f4e79;margin-left:6px}
.gauge{position:relative;height:10px;border-radius:5px;margin:8px 0 2px;
  background:linear-gradient(90deg,#c62828,#ef9a9a,#ffe082,#a5d6a7,#2e7d32)}
.gauge span{position:absolute;top:-4px;width:4px;height:18px;background:#111827;border-radius:2px;
  transform:translateX(-2px);box-shadow:0 0 0 2px #fff}
.gauge.empty{background:#eef1f5}
.chartbox{position:relative;height:340px}
.chartbox.small{height:240px}
.chartbox.pos{height:360px}
.posctl{display:flex;align-items:center;gap:10px;margin-top:12px}
.posctl-end{font-size:11px;color:#9aa4b2;flex:0 0 auto}
.posrng{-webkit-appearance:none;appearance:none;flex:1;height:6px;border-radius:3px;
  background:#dde3ea;outline:none;cursor:pointer}
.posrng::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:18px;height:18px;
  border-radius:50%;background:#c62828;border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,.35);cursor:pointer}
.posrng::-moz-range-thumb{width:18px;height:18px;border-radius:50%;background:#c62828;
  border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,.35);cursor:pointer}
.posrng::-moz-range-track{height:6px;border-radius:3px;background:#dde3ea}
.foot{font-size:12px;color:#9aa4b2;margin-top:8px}
@media(max-width:900px){.hero-decomp{margin-left:0}}
"""
    scripts = f"""
<script>const D = {data_json};</script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
window.addEventListener('DOMContentLoaded', function() {{
  var pos = document.getElementById('posChart');
  var rng = document.getElementById('posRange');
  var lab = document.getElementById('posLabel');
  if (pos && rng && window.Chart) {{
    var N = D.pos.dates.length, WIN = 500;   // 一屏两年（约 500 个交易日）
    var maxStart = Math.max(0, N - WIN);
    rng.min = 0; rng.max = maxStart; rng.step = 1; rng.value = maxStart;
    var posChart = null;
    function windowRange(start) {{
      var a = Math.max(0, Math.min(+start, maxStart)), b = Math.min(N, a + WIN);
      // 左轴按当前窗口自适应：留 8pp 余量、至少 20pp 跨度，避免仓位摆动被压成一条平线
      var vals = D.pos.raw.slice(a, b).concat(D.pos.exec.slice(a, b))
        .filter(function(v) {{ return v !== null && isFinite(v); }});
      var loY = vals.length ? Math.min.apply(null, vals) : 0;
      var hiY = vals.length ? Math.max.apply(null, vals) : 100;
      var pad = Math.max(8, (hiY - loY) * 0.25);
      var yMin = Math.max(0, Math.floor(loY - pad)), yMax = Math.min(100, Math.ceil(hiY + pad));
      if (yMax - yMin < 20) {{ var c = (yMax + yMin) / 2; yMin = Math.max(0, c - 10); yMax = Math.min(100, c + 10); }}
      // 右轴（国证A指）按窗口自适应
      var bs = D.pos.bench.slice(a, b).filter(function(v) {{ return v !== null && isFinite(v); }});
      var bLo = bs.length ? Math.min.apply(null, bs) : 0;
      var bHi = bs.length ? Math.max.apply(null, bs) : 1;
      var bPad = Math.max((bHi - bLo) * 0.08, bHi * 0.01);
      return {{a: a, b: b, yMin: yMin, yMax: yMax,
               y1Min: Math.floor(bLo - bPad), y1Max: Math.ceil(bHi + bPad)}};
    }}
    function draw(start) {{
      var r = windowRange(start), dates = D.pos.dates;
      if (!posChart) {{
        posChart = new Chart(pos, {{type:'line', data:{{labels:dates, datasets:[
          {{label:'实时参考仓位', data:D.pos.raw, borderColor:'#c62828', borderWidth:1.8, pointRadius:0, tension:.15, yAxisID:'y'}},
          {{label:'月频执行值', data:D.pos.exec, borderColor:'#1f4e79', borderWidth:1.5, pointRadius:0, stepped:true, borderDash:[6,3], yAxisID:'y'}},
          {{label:'国证A指', data:D.pos.bench, borderColor:'#7b8a9e', borderWidth:1.2, pointRadius:0, tension:.15, yAxisID:'y1'}}
        ]}}, options:{{responsive:true, maintainAspectRatio:false, animation:false,
          interaction:{{mode:'index', intersect:false}},
          scales:{{x:{{min:dates[r.a], max:dates[r.b - 1], ticks:{{maxTicksLimit:8}}}},
            y:{{min:r.yMin, max:r.yMax, ticks:{{callback:function(v){{return v + '%';}}}}}},
            y1:{{position:'right', grid:{{drawOnChartArea:false}},
              ticks:{{maxTicksLimit:5, callback:function(v){{return v.toFixed(0);}}}}}}}}}}}});
      }} else {{
        // 连续滚动：只平移 x 轴窗口、按窗口重设三轴范围，不重建 datasets
        posChart.options.scales.x.min = dates[r.a];
        posChart.options.scales.x.max = dates[r.b - 1];
        posChart.options.scales.y.min = r.yMin;
        posChart.options.scales.y.max = r.yMax;
        posChart.options.scales.y1.min = r.y1Min;
        posChart.options.scales.y1.max = r.y1Max;
        posChart.update('none');
      }}
      lab.textContent = '显示 ' + dates[r.a] + ' ~ ' + dates[r.b - 1]
        + ' · 灰线 = 国证A指 右轴';
    }}
    rng.addEventListener('input', function() {{ draw(rng.value); }});
    draw(maxStart);
  }}
  var nav = document.getElementById('navChart');
  if (nav && window.Chart) {{
    new Chart(nav, {{type:'line', data:{{labels:D.strategy.dates, datasets:[
      {{label:'估值定仓位', data:D.strategy.nav, borderColor:'#c62828', borderWidth:1.8, pointRadius:0, tension:.1}},
      {{label:'满仓国证A指', data:D.full.nav, borderColor:'#7b8a9e', borderWidth:1.2, pointRadius:0, tension:.1}}
    ]}}, options:{{responsive:true, maintainAspectRatio:false, interaction:{{mode:'index', intersect:false}},
      scales:{{x:{{ticks:{{maxTicksLimit:8}}}}, y:{{ticks:{{callback:function(v){{return v.toFixed(1);}}}}}}}}}}}});
  }}
}});
</script>
"""
    html = page(
        "估值定仓位规则 · 面板",
        meta=f"更新 {today:%Y-%m-%d} · 生成 {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        body=body, extra_css=extra_css, scripts=scripts,
    )
    out = OUT_DIR / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# ------------------------------------------------------------------ 主流程
def run(root: Path) -> dict:
    panel, monthly = build_panel(root)
    valid = monthly.dropna(subset=["p_pe_s", "vol60"])
    raw = pd.Series({
        dt: target_weight(r.p_pe_s, r.vol60, r.px_ma200_s)[2]
        for dt, r in valid.iterrows()}, name="w_raw")
    targets = apply_lazy(raw)

    strategy = simulate(panel, targets, init_w=float(targets.iloc[0]))
    # 基准：国证A指（全市场）60/40 与满仓；另留满仓沪深300 作对照
    fixed6040 = bench_fixed(panel, 0.60, targets.index, ret_col="ret_bench")
    full = bench_fixed(panel, 1.00, targets.index, ret_col="ret_bench")
    full300 = bench_fixed(panel, 1.00, targets.index, ret_col="ret")

    res = {
        "strategy": {"sim": strategy, "perf": perf(strategy["ret"]), "targets": targets, "raw": raw},
        "fixed6040": {"sim": fixed6040, "perf": perf(fixed6040["ret"])},
        "full": {"sim": full, "perf": perf(full["ret"])},
        "full300": {"sim": full300, "perf": perf(full300["ret"])},
        "period": f"{strategy.index.min():%Y-%m-%d} ~ {strategy.index.max():%Y-%m-%d}",
    }

    # 预注册判据（只打印，不上面板）
    sp, bp, fp = res["strategy"]["perf"], res["fixed6040"]["perf"], res["full"]["perf"]
    verdict = {
        "calmar": sp["calmar"] > bp["calmar"],
        "sharpe": sp["sharpe"] >= bp["sharpe"] - 0.05,
        "mdd": sp["max_drawdown"] < min(bp["max_drawdown"], fp["max_drawdown"]),
    }
    verdict["pass_all"] = all(verdict.values())
    res["verdict"] = verdict
    res["panel"] = panel
    res["monthly"] = monthly
    res["w_raw_daily"] = daily_raw_weights(panel)
    res["margin_last"] = str(monthly.attrs.get("margin_last", "-"))[:10]
    return res


def run_variants(root: Path) -> None:
    """敏感性：候选状态量/替代锚/二元仓位（只在控制台报数，不影响主规则）。"""
    panel, monthly = build_panel(root)
    valid = monthly.dropna(subset=["p_pe_s", "vol60"])
    px = _index_frame(INDEX, root)
    amt = px["amount"].reindex(panel.index)
    amt_z = (amt - amt.rolling(20).mean()) / amt.rolling(20).std()
    all_a = pd.read_parquet(root / "daily_all_a.parquet", columns=["close"])
    r_a = all_a["close"].groupby(level="code").pct_change(fill_method=None)
    breadth = (r_a > 0).groupby(level="date").mean()
    breadth.index = pd.DatetimeIndex(breadth.index)
    breadth20 = breadth.rolling(20, min_periods=10).mean().reindex(panel.index)
    del all_a, r_a

    trend = {d: r.px_ma200_s for d, r in valid.iterrows()}
    base_raw = {d: target_weight(r.p_pe_s, r.vol60, trend[d])[2] for d, r in valid.iterrows()}

    def _trend_tilt(k: float) -> pd.Series:
        """趋势加成幅度对照：k=0.45 为现行；k=0 即无趋势项。"""
        out = {}
        for d, r in valid.iterrows():
            w_val = float(np.clip(VAL_BASE + VAL_SLOPE * (0.5 - r.p_pe_s), W_LO, W_HI))
            scale = 1.0 if not np.isfinite(r.vol60) or r.vol60 <= 0 else float(
                np.clip(VOL_TARGET / r.vol60, SCALE_LO, SCALE_HI))
            tilt = 0.0 if not np.isfinite(r.px_ma200_s) else k * float(
                np.clip(r.px_ma200_s / TREND_SPAN, -1.0, 1.0))
            out[d] = float(np.clip((w_val + tilt) * scale, W_LO, W_HI))
        return apply_lazy(pd.Series(out))

    def _tilt(series: pd.Series, k: float) -> pd.Series:
        out = dict(base_raw)
        for d in out:
            z = series.get(d, np.nan)
            if np.isfinite(z):
                out[d] = float(np.clip(out[d] + k * z, W_LO, W_HI))
        return apply_lazy(pd.Series(out))

    variants = {
        "现行 PE+vol+趋势加成": apply_lazy(pd.Series(base_raw)),
        "趋势加成 k=0.35": _trend_tilt(0.35),
        "趋势加成 k=0.55": _trend_tilt(0.55),
        "无趋势项": apply_lazy(pd.Series(
            {d: target_weight(r.p_pe_s, r.vol60, trend_gate=False)[2] for d, r in valid.iterrows()})),
        "ERP 替代 PE": apply_lazy(pd.Series(
            {d: target_weight(r.p_erp, r.vol60, trend[d])[2] for d, r in valid.iterrows()})),
        "+涨跌广度 tilt": _tilt(breadth20, 0.30),
        "+成交额 z tilt": _tilt(amt_z, 0.03),
        "二元(PE<50%满仓)": pd.Series(
            {d: 1.0 if r.p_pe_s < 0.5 else 0.0 for d, r in valid.iterrows()}),
    }
    print(f"{'变体':<22}{'年化':>8}{'Sharpe':>8}{'MDD':>8}{'Calmar':>8}{'均仓':>7}")
    for name, t in variants.items():
        s = simulate(panel, t, init_w=float(t.iloc[0]))
        m = perf(s["ret"])
        print(f"{name:<22}{m['annual_return']*100:>7.1f}%{m['sharpe']:>8.2f}"
              f"{m['max_drawdown']*100:>7.1f}%{m['calmar']:>8.2f}{s['weight'].mean()*100:>6.0f}%")

    print("\n现金腿替代（规则不变）：")
    for label, rate in (("现金 0%", 0.0), ("现金 1.5%", 0.015), ("现金 2.0%", 0.020)):
        s = simulate(panel.assign(ret_bond=rate / 252), variants["现行 PE+vol+趋势加成"],
                     init_w=float(variants["现行 PE+vol+趋势加成"].iloc[0]))
        m = perf(s["ret"])
        print(f"  {label:<10} 年化 {m['annual_return']*100:>4.1f}%  Sharpe {m['sharpe']:.2f}  "
              f"MDD {m['max_drawdown']*100:.1f}%")
    print(f"  债指数(基准)  年化 {perf(panel['ret_bond'])['annual_return']*100:>4.1f}%")


def self_check() -> None:
    """规则纯函数自检（不依赖数据）。"""
    # 估值腿：中性分位 60%，极低分位封顶 100%，最高分位 15%
    assert abs(target_weight(0.5, 0.20, 0.0)[2] - 0.60) < 1e-9
    assert abs(target_weight(0.0, 0.20, 0.0)[2] - 1.00) < 1e-9
    assert abs(target_weight(1.0, 0.20, 0.0)[2] - 0.15) < 1e-9
    _, s_hi, _ = target_weight(0.5, 0.60)
    assert abs(s_hi - 0.50) < 1e-9                          # 高波动只减到 0.5
    _, s_lo, _ = target_weight(0.5, 0.10)
    assert abs(s_lo - 1.0) < 1e-9                           # 低波动不加仓
    # 趋势加成：距年线 ±15% 饱和 → ±TREND_TILT，贴线为 0，线性平滑
    assert abs(trend_tilt(0.0)) < 1e-9
    assert abs(trend_tilt(0.15) - TREND_TILT) < 1e-9
    assert abs(trend_tilt(-0.15) + TREND_TILT) < 1e-9
    assert abs(trend_tilt(0.30) - TREND_TILT) < 1e-9        # 饱和不加倍
    assert abs(trend_tilt(-0.075) + TREND_TILT / 2) < 1e-9
    _, _, w_mid = target_weight(0.5, 0.20, 0.075)
    assert abs(w_mid - (0.60 + TREND_TILT / 2)) < 1e-9      # 60% + 半个加成
    _, _, w_off = target_weight(0.5, 0.20, 0.15, trend_gate=False)
    assert abs(w_off - 0.60) < 1e-9
    # 最终仓位区间
    assert abs(target_weight(1.0, 0.60, -0.30)[2] - W_LO) < 1e-9  # (15%-35%)*0.5 → 封底 10%
    lazy = apply_lazy(pd.Series([0.65, 0.60, 0.40, 0.40],
                                index=pd.date_range("2020-01-31", periods=4, freq="ME")),
                      init=0.60)
    assert abs(lazy.iloc[0] - 0.60) < 1e-9                  # +5pp 在惰性区间内
    assert abs(lazy.iloc[1] - 0.60) < 1e-9
    assert abs(lazy.iloc[2] - 0.50) < 1e-9                  # 一次最多 -10pp
    print("[self-check] OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="估值定仓位规则：回测 + 面板")
    ap.add_argument("--refresh-data", action="store_true", help="用 akshare 刷新 PE / 国债收益率缓存")
    ap.add_argument("--root", default=None, help="parquet 缓存根（默认按 YQ_DATA_ROOT/config 解析）")
    ap.add_argument("--self-check", action="store_true", help="只跑规则纯函数自检")
    ap.add_argument("--variants", action="store_true", help="敏感性：候选状态量/现金腿对照")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return
    root = _cache_root(args.root)
    if args.refresh_data:
        refresh_data(root)
        return
    if args.variants:
        run_variants(root)
        return

    res = run(root)
    out = _render_panel(res["monthly"], res["panel"], res, res["margin_last"])
    m = res["strategy"]["perf"]
    print(f"[rule] {res['period']}  年化 {m['annual_return']*100:.1f}%  "
          f"Sharpe {m['sharpe']:.2f}  MDD {m['max_drawdown']*100:.1f}%  "
          f"Calmar {m['calmar']:.2f}")
    print(f"[rule] 判定：{'通过' if res['verdict']['pass_all'] else '未通过'} {res['verdict']}")
    print(f"[rule] 面板 → {out}")


if __name__ == "__main__":
    main()
