"""大盘方向准入检验：**新信息集**能不能预测下月沪深300？

为什么单独开这个探针
--------------------
`timing_probe` 已把**同源**信息测穿：价格趋势 7 规则 0/7、月度价格因子 0/7、
资金面 0/4、日历/月内位置全负；`regime_probe` 两道 kill 点全触发；
`model_state_probe` 5 个模型自诊断状态 0/5。用户追问「还有没有别的办法」后，
本轮只测**没被证伪过、且本地能取到**的少数信息集：

| 信息集 | 与已测的区别 | 数据 |
|---|---|---|
| 期权隐含波动 QVIX（50ETF 期权） | 前瞻性（forward-looking），非价格历史 | optbbs，2015-02+ |
| 隐含减已实现（方差风险溢价 VRP） | 期权定价 vs 实际波动之差，文献方向为**负** | 同上 |
| 股债跷跷板（300 动量 − 国债ETF 动量） | 跨资产相对强弱，与单一股指价格不同源 | sina，2013-04+ |

宏观（PMI/社融/M1-M2/10Y）本轮**取不到**：akshare 的宏观接口走 eastmoney，
在本机整段超时（`datacenter-web.eastmoney.com` ReadTimeout），已实测记录在案。

口径（🚨 与项目纪律一致）
------------------------
- **目标**：沪深300 **T+1 开盘起算**的月度收益（`first close of m+1 → last close of m+1`），
  不用月末收盘当成交价，避免 1 天的乐观偏差。
- **状态**：月末收盘可知；滚动统计一律 PIT（扩张窗口，不看未来）。
- **判据（先验固定，不按结果挑）**：本帧共 **7 个候选状态**，Bonferroni 门槛
  |t| ≥ 2.5（≈ 0.05/7 双尾的保守取整）+ 扩张窗口样本外符号一致 + 五分位单调。
- **样本量现实**：2013-04 起 162 个月，QVIX 2015-02 起仅 ~140 个月。105~162 个
  观测下要让 t 到 2.5，rank IC 需 ≈ 0.20~0.25 —— 这是**很高的效应量**，也是本
  路径先验低的原因。

用法
----
    python -m scripts.evaluation.market_direction_probe            # 离线优先（有缓存不联网）
    python -m scripts.evaluation.market_direction_probe --refresh  # 重新拉数据

产物
----
    reports/market_direction_probe/{data/*.csv, states_monthly.csv,
    predictability.csv, oos.csv, README.md 由人工撰写}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "market_direction_probe"
DATA = OUT / "data"

BEGIN = "2013-04-01"

#: 候选状态（**先验登记**，不按结果增删）：名称 → (含义, 预期方向)
CANDIDATES = {
    "qvix_lvl": "50ETF 期权隐含波动率（月末水平）",
    "qvix_z": "隐含波动率扩张 z-score（PIT）",
    "qvix_chg": "隐含波动率月度变化（上行=恐慌加剧）",
    "vrp": "隐含 − 已实现波动（方差风险溢价，文献预期为负）",
    "realvol60": "沪深300 60 日已实现年化波动",
    "stock_bond": "300 20 日动量 − 国债ETF 20 日动量（股债跷跷板）",
    "mom20": "沪深300 20 日动量（对照组，已知死）",
}


# ---------------------------------------------------------------------------
# 数据（sina 源可用；eastmoney 源在本机超时，故不取宏观）
# ---------------------------------------------------------------------------
def _fetch_with_timeout(fetch, seconds: float = 20.0):
    """akshare 的 sina/optbbs 接口在本机**偶发无超时挂死**（实测两次各挂 >3 分钟）。

    用线程 + 硬超时兜底：超时返回 ``None``，让探针降级继续，而不是整条线卡死。
    """
    import queue
    import threading

    q: queue.Queue = queue.Queue()

    def _run():
        try:
            q.put((True, fetch()))
        except Exception as exc:  # 网络异常
            q.put((False, exc))

    # 🚨 必须是 daemon 线程：ThreadPoolExecutor 的工作线程在解释器退出时会被 join，
    #    挂死的 requests 会让整个进程退不出去（本脚本首版实测卡死两次）。
    threading.Thread(target=_run, daemon=True).start()
    try:
        ok, val = q.get(timeout=seconds)
    except queue.Empty:
        return None
    return val if ok else None


def _cached(name: str, fetch, refresh: bool = False,
            optional: bool = False) -> pd.DataFrame:
    """带本地缓存的抓取；``optional=True`` 时失败返回空表（不抛异常）。"""
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / f"{name}.csv"
    if p.exists() and not refresh:
        df = pd.read_csv(p, parse_dates=["date"])
        return df.set_index("date").sort_index()
    raw = _fetch_with_timeout(fetch)
    if raw is None or len(raw) == 0:
        if optional:
            print(f"  [不可得] {name}：抓取超时/失败，相关状态本轮跳过")
            return pd.DataFrame()
        raise RuntimeError(f"{name} 抓取失败且无本地缓存")
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.to_csv(p, encoding="utf-8-sig")
    return df


def load_inputs(refresh: bool = False) -> pd.DataFrame:
    """沪深300 / 国债ETF / QVIX 的日频收盘（并集索引，ffill 仅用于对齐）。"""
    import akshare as ak

    hs300 = _cached("hs300", lambda: ak.stock_zh_index_daily(symbol="sh000300"), refresh)
    bond = _cached("bond_etf_511010",
                   lambda: ak.fund_etf_hist_sina(symbol="sh511010"), refresh)
    qvix = _cached("qvix_50etf", lambda: ak.index_option_50etf_qvix(), refresh,
                   optional=True)
    df = pd.DataFrame({
        "hs300": hs300["close"],
        "bond": bond["close"],
        "qvix": qvix["close"] if len(qvix) else np.nan,
    })
    df = df[df.index >= BEGIN]
    return df


# ---------------------------------------------------------------------------
# 状态与目标
# ---------------------------------------------------------------------------
def build_monthly_states(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """月末状态（PIT）与**次月**收益（T+1 起算）。"""
    d = df.ffill()
    r = d["hs300"].pct_change(fill_method=None)

    daily = pd.DataFrame({
        "qvix_lvl": d["qvix"],
        "realvol60": r.rolling(60, min_periods=40).std() * np.sqrt(252) * 100,
        "mom20": d["hs300"] / d["hs300"].shift(20) - 1.0,
        "bond_mom20": d["bond"] / d["bond"].shift(20) - 1.0,
    })
    daily["stock_bond"] = daily["mom20"] - daily["bond_mom20"]
    daily["vrp"] = daily["qvix_lvl"] - daily["realvol60"]
    # 扩张 z 只用当时及以前的数据（PIT），前 24 个月缺失
    m = daily["qvix_lvl"].expanding(min_periods=24)
    daily["qvix_z"] = (daily["qvix_lvl"] - m.mean()) / m.std()

    me = daily.groupby(daily.index.to_period("M")).last()
    st = pd.DataFrame({
        "qvix_lvl": me["qvix_lvl"],
        "qvix_z": me["qvix_z"],
        "vrp": me["vrp"],
        "realvol60": me["realvol60"],
        "stock_bond": me["stock_bond"],
        "mom20": me["mom20"],
        "qvix_chg": me["qvix_lvl"].diff(),
    })
    st.index = pd.DatetimeIndex(st.index.to_timestamp(how="end").normalize())

    # 目标：下月**第一个交易日收盘 → 月末收盘**（不在月末收盘成交）
    close = d["hs300"]
    fwd = []
    idx = []
    periods = close.index.to_period("M")
    for p in pd.unique(periods):
        days = close.index[periods == p]
        if len(days) < 2:
            continue
        idx.append(days[-1])
        fwd.append(close.loc[days[-1]] / close.loc[days[0]] - 1.0)
    target = pd.Series(fwd, index=pd.DatetimeIndex(idx)).sort_index()
    return st, target


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def _stats(state: pd.Series, target: pd.Series) -> dict:
    j = pd.concat([state.rename("s"), target.shift(-1).rename("r")], axis=1).dropna()
    if len(j) < 30:
        return {"n": len(j), "rank_ic": np.nan, "t_stat": np.nan, "q1": np.nan,
                "q5": np.nan, "monotone": False, "hit": np.nan}
    ic = float(j["s"].corr(j["r"], method="spearman"))
    q = pd.qcut(j["s"], 5, labels=False, duplicates="drop")
    gm = j.groupby(q)["r"].mean()
    # 方向命中率：状态取当期扩张中位以上/以下，与次月收益符号是否一致
    med = state.expanding(min_periods=24).median().reindex(j.index)
    hi = j["s"] > med
    hit = float((np.sign(j["r"]) == np.where(hi, 1, -1)).mean()) if hi.notna().all() else np.nan
    return {"n": len(j), "rank_ic": ic, "t_stat": ic * np.sqrt(len(j) - 1),
            "q1": float(gm.iloc[0]), "q5": float(gm.iloc[-1]),
            "monotone": bool((gm.diff().dropna() >= 0).all()), "hit": hit}


def _oos_sign(state: pd.Series, target: pd.Series, min_train: int = 60) -> dict:
    """扩张窗口 OLS：用 t 之前的样本拟合，猜 t+1 的符号。**唯一的样本外证据**。"""
    j = pd.concat([state.rename("s"), target.shift(-1).rename("r")], axis=1).dropna()
    if len(j) < min_train + 24:
        return {"oos_n": 0, "oos_hit": np.nan, "oos_t": np.nan}
    preds, actual = [], []
    x, y = j["s"].to_numpy(float), j["r"].to_numpy(float)
    for i in range(min_train, len(j)):
        xs, ys = x[:i], y[:i]
        if np.std(xs) == 0:
            continue
        b = np.polyfit(xs, ys, 1)
        preds.append(b[0] * x[i] + b[1])
        actual.append(y[i])
    preds, actual = np.array(preds), np.array(actual)
    hit = float((np.sign(preds) == np.sign(actual)).mean())
    # 方向能力的 t：把"预测符号×实际收益"当成一条策略收益序列检验均值>0
    edge = preds * actual
    t = float(edge.mean() / (edge.std(ddof=1) / np.sqrt(len(edge)))) if len(edge) > 2 else np.nan
    return {"oos_n": int(len(edge)), "oos_hit": hit, "oos_t": t}


def run(refresh: bool = False) -> pd.DataFrame:
    df = load_inputs(refresh)
    st, target = build_monthly_states(df)
    print(f"数据：{df.index.min().date()} ~ {df.index.max().date()}  "
          f"月频样本 {len(target)}（QVIX 自 2015-02 起）")

    rows, skipped = [], []
    for name, desc in CANDIDATES.items():
        s = st[name]
        if s.notna().sum() < 30:
            skipped.append(name)
            continue
        rows.append({"state": name, "desc": desc, **_stats(s, target),
                     **_oos_sign(s, target)})
    tbl = pd.DataFrame(rows).set_index("state")
    if skipped:
        print(f"  跳过（本机取不到数据）：{skipped}")

    OUT.mkdir(parents=True, exist_ok=True)
    st.assign(target_next=target.shift(-1)).to_csv(
        OUT / "states_monthly.csv", encoding="utf-8-sig")
    tbl.to_csv(OUT / "predictability.csv", encoding="utf-8-sig")

    print("\n=== 大盘方向准入检验（目标 = 下月沪深300，T+1 起算）===")
    cols = ["n", "rank_ic", "t_stat", "q1", "q5", "monotone", "hit",
            "oos_n", "oos_hit", "oos_t"]
    print(tbl[cols].to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n判据（先验固定）：|t| ≥ 2.5（7 候选 Bonferroni）且 oos_t > 0 且五分位单调。")
    print(f"产物: {OUT}\\predictability.csv")
    return tbl


# ---------------------------------------------------------------------------
# 个人投资口径：长周期估值，而不是月频方向
# ---------------------------------------------------------------------------
def load_valuation(refresh: bool = False) -> pd.DataFrame:
    """沪深300 / 全A 的**月度 PE-TTM**（乐咕乐股，sina/legulegu 源，本机可用）。"""
    import akshare as ak

    def hs300_pe():
        d = ak.stock_index_pe_lg(symbol="沪深300")
        d = d.rename(columns={"日期": "date", "滚动市盈率": "pe"})
        return d[["date", "pe"]]

    def all_a_pe():
        d = ak.stock_a_ttm_lyr()
        d = d.rename(columns={"date": "date"})
        return d[["date", "middlePETTM"]].rename(columns={"middlePETTM": "pe"})

    hs = _cached("pe_hs300_ttm", hs300_pe, refresh).rename(columns={"pe": "pe_hs300"})
    allA = _cached("pe_all_a", all_a_pe, refresh).rename(columns={"pe": "pe_all"})
    return pd.concat([hs, allA], axis=1).sort_index()


def _roll_pct(pe: pd.Series, window: int = 120, min_periods: int = 60) -> pd.Series:
    """**滚动 10 年**分位（PIT）：当前 PE 在过去 window 个月中的位置。

    不用扩张窗口：本序列长期下移（2008 的低 PE 会永久压低后面所有分位），
    实测扩张分位最高只到 0.75，**永远触发不了"贵"**。
    """
    return pe.rolling(window, min_periods=min_periods).apply(
        lambda w: float((w.iloc[:-1] <= w.iloc[-1]).mean()), raw=False)


def _fwd(close: pd.Series, h: int) -> pd.Series:
    return close.shift(-h) / close - 1.0


def _cond_stats(sig: pd.Series, ret: pd.Series, lo: float = 0.20,
                hi: float = 0.80, step: int = 12) -> dict:
    """极值条件化：**便宜**（分位 < lo）与**贵**（> hi）之后的条件收益差。

    目标不是"预测每个月"，而是"极端时点有没有信息"——这是个人投资真正需要的
    统计口径（两样本），不是月频 rank IC。重叠窗口会抬高 t，故同时报
    **不重叠子样本**（每 h 个取一个）的 t。
    """
    j = pd.concat([sig.rename("p"), ret.rename("r")], axis=1).dropna()
    cheap, rich = j[j["p"] <= lo]["r"], j[j["p"] >= hi]["r"]
    if len(j) < 24 or len(cheap) < 5 or len(rich) < 5:
        return {"n": len(j), "cheap_n": len(cheap), "rich_n": len(rich)}
    naive_t = ((cheap.mean() - rich.mean())
               / np.sqrt(cheap.var(ddof=1) / len(cheap) + rich.var(ddof=1) / len(rich)))
    # 去重叠：每 step 个月取一个（h=12 → 12，h=36 → 36），保证窗口不重叠
    j_ns = j.iloc[::step]
    c2, r2 = j_ns[j_ns["p"] <= lo]["r"], j_ns[j_ns["p"] >= hi]["r"]
    t_ns = np.nan
    if len(c2) >= 3 and len(r2) >= 3:
        t_ns = ((c2.mean() - r2.mean())
                / np.sqrt(c2.var(ddof=1) / len(c2) + r2.var(ddof=1) / len(r2)))
    return {"n": len(j), "cheap_n": len(cheap), "rich_n": len(rich),
            "cheap_n_ds": len(c2), "rich_n_ds": len(r2),
            "cheap_mean": cheap.mean(), "rich_mean": rich.mean(),
            "spread": cheap.mean() - rich.mean(),
            "t_naive": float(naive_t), "t_deoverlap": float(t_ns),
            "hit_cheap": float((cheap > 0).mean()), "hit_rich": float((rich > 0).mean())}


def _dca(close: pd.Series, sig: pd.Series, rule: dict) -> dict:
    """个人定投对照：每月固定投入 1 份，按估值分位调整倍数。

    返回**资金加权**（money-weighted）结果：终值 / 累计投入，以及等效年化。
    """
    j = pd.concat([close.rename("px"), sig.rename("p")], axis=1).dropna()
    mults = np.ones(len(j))
    if rule["cheap_mult"] != 1.0 or rule["rich_mult"] != 1.0:
        p = j["p"].to_numpy()
        mults = np.where(p <= rule["cheap"], rule["cheap_mult"],
                         np.where(p >= rule["rich"], rule["rich_mult"], 1.0))
    if rule.get("normalize", True):
        # 🚨 必须归一化到「同等总投入」：否则"便宜月加倍"天然多投 20% 的钱，
        #    终值高只是本金多，不是择时更好。
        mults = mults * (len(mults) / mults.sum())
    units = float((mults / j["px"].to_numpy()).sum())
    invested = float(mults.sum())
    final = units * j["px"].iloc[-1]
    n_years = len(j) / 12.0
    return {"months": len(j), "invested": invested, "final": final,
            "multiple": final / invested,
            "cagr": (final / invested) ** (1 / n_years) - 1 if n_years > 0 else np.nan}


def run_valuation(refresh: bool = False) -> pd.DataFrame:
    """个人投资口径：估值分位 → 未来 1/3 年收益；以及定投倍数规则对照。"""
    val = load_valuation(refresh)
    # 估值检验要用**全历史**（sina 指数 2002+），不能被 run() 的 BEGIN 截断
    px_full = pd.read_csv(DATA / "hs300.csv", parse_dates=["date"]).set_index("date")["close"]
    close_me = px_full.groupby(px_full.index.to_period("M")).last()

    rows = []
    for col, label in [("pe_hs300", "沪深300 PE-TTM"), ("pe_all", "全A PE-TTM")]:
        pe = val[col].dropna()
        pe.index = pd.PeriodIndex(pd.to_datetime(pe.index), freq="M")
        pct = _roll_pct(pe)
        # 统一到「月」索引再对齐，避免月末交易日与月末日历日错位
        j = pd.concat([pct.rename("pct"), close_me.rename("px")], axis=1, join="inner")
        for h, hlabel in [(12, "未来1年"), (36, "未来3年")]:
            r = _fwd(j["px"], h)
            rows.append({"signal": f"{label} 分位", "horizon": hlabel,
                         **_cond_stats(j["pct"], r, step=h)})

    tbl = pd.DataFrame(rows).set_index(["signal", "horizon"])
    OUT.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(OUT / "valuation_longhorizon.csv", encoding="utf-8-sig")

    print("\n=== 个人投资口径：估值分位 → 未来收益（便宜20% vs 贵20%）===")
    print(tbl.to_string(float_format=lambda x: f"{x:.3f}"))

    # 定投规则对照（用沪深300 全样本）
    pe = val["pe_hs300"].dropna()
    pe.index = pd.PeriodIndex(pd.to_datetime(pe.index), freq="M")
    pct = _roll_pct(pe)
    j = pd.concat([close_me.rename("px"), pct.rename("p")], axis=1,
                  join="inner").dropna()
    base = _dca(j["px"], j["p"], {"cheap": 0.0, "rich": 1.1, "cheap_mult": 1.0,
                                 "rich_mult": 1.0})
    rule = _dca(j["px"], j["p"], {"cheap": 0.2, "rich": 0.8, "cheap_mult": 2.0,
                                 "rich_mult": 0.5})
    print("\n=== 定投规则对照（每月 1 份；便宜加倍 / 贵减半）===")
    for name, d in [("恒定定投", base), ("估值加权定投", rule)]:
        print(f"  {name}: 投入 {d['invested']:.0f} 份 → 终值 {d['final']:.1f} 份  "
              f"倍数 {d['multiple']:.2f}x  年化 {d['cagr']:.2%}  ({d['months']} 月)")
    print(f"  分位中枢: 便宜(<20%)触发 {int((j['p'] <= 0.2).sum())} 月，"
          f"贵(>80%)触发 {int((j['p'] >= 0.8).sum())} 月")
    return tbl


def main() -> None:
    ap = argparse.ArgumentParser(description="大盘方向准入检验")
    ap.add_argument("--refresh", action="store_true", help="重新拉取数据")
    ap.add_argument("--valuation", action="store_true",
                    help="个人投资口径：长周期估值分位 → 未来 1/3 年收益")
    args = ap.parse_args()
    if args.valuation:
        run_valuation(refresh=args.refresh)
        return
    run(refresh=args.refresh)


if __name__ == "__main__":
    main()
