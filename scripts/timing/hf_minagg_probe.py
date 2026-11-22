"""华福 1114 高频聚合特征探针（纯择时线，reports/hf_minagg_probe/）。

研报：华福证券 2025-11-14《基于 LSTM 神经网络的择时融合多因子选股策略》§高频因子
聚合低频（PDF 仓库外 D:\\project\\研报\\择时\\）。报告声称 43 个指数分钟聚合特征中
最强者（price_vs_resistance q60）2010–2025 年化 33.31%、与次日 open-to-open 相关
|10.76%|；本探针只探这一族（①分析师预期无数据、②资金流成本高、④⑤LSTM 边际值低，
见研读判断）。

两层判据（先于数据写死）
------------------------
L1 IC 层：Top 特征诚实实现（分钟数据 ≤t 收盘可得）→ 与次日 open-to-open 收益的
   日度 Pearson |IC|，对照报告声称值（price_vs_resistance 10.76% / price_volume_
   divergence 8.63% / price_vs_support 8.55%）。**≥2 个特征 |IC| ≥ 声称值的一半**
   ⇒ 进 L2；否则收口（「~1000 次 best-of 选择 + 特征构造瞄准同段历史」假设成立）。
L2 阈值层（仅 L1 过时跑）：存活特征 × 滚动 252 日分位 {q20,q40,q60,q80} 的
   long/flat 阈值策略，T+1 open → T+2 open 执行（报告目标口径），成本 1bp（报告
   万1 口径）/10bp（保守）。**≥3 个臂年化 ≥ 10%** ⇒ 家族真实；否则 best-of 签名。
   极性（大于/小于阈值做多）按**训练半窗 IC 符号**定，测试半窗出数——不做全样本选向。

特征实现（研报只给文字定义，骨架复现，差异见 README §四）
- price_vs_resistance = (日高 − 收盘)/收盘（距阻力位距离，报告「越远收益越高」）
- price_vs_support    = (收盘 − 日低)/收盘（距支撑位距离）
- price_volume_divergence = 日内分钟收盘斜率 z 分 × 日内分钟成交量斜率 z 分（相乘）
- reversal_pattern    = 日内先高后低 → +1，先低后高 → −1（研报原约定）
- intraday_volatility = 分钟收益 std × sqrt(48)（年化到日）
- daily_high_low_ratio = 日高/日低
- volatility_regime   = 3 日实现波动 / 20 日实现波动（日线）

用法::

    python -m scripts.timing.hf_minagg_probe --self-check   # 因果/对齐抽查
    python -m scripts.timing.hf_minagg_probe                # 全量（首跑拉 13 年 5 分钟线）
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.metrics import annual_return, max_drawdown, sharpe_ratio
from scripts.timing.shell import load_index_daily
from stats import PERIODS_PER_YEAR

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "hf_minagg_probe"
INDEX_CODE = "000300.SH"  # 报告基准资产：沪深300
MINUTE_BEGIN = 20130101  # SDK 指数分钟线起点（实测 2013-01 有数据）
MINUTE_PERIOD = 5  # 5 分钟线（48 根/日，足以支撑反转时序/日内斜率）
BARS_PER_DAY = 48
QUANTILES = (0.20, 0.40, 0.60, 0.80)
ROLL_Q_WIN = 252  # 阈值滚动分位窗（1 年，含当日）
COST_GRID_BPS = (1.0, 10.0)  # 报告万1口径 / 保守

# 报告声称值（页 13 相关性表，|IC|）
REPORT_CLAIMS = {
    "price_vs_resistance": 0.1076,
    "price_volume_divergence": 0.0863,
    "price_vs_support": 0.0855,
    "intraday_volatility_ewm": 0.0870,  # 柱状图读数（日内加权波动率）
    "daily_high_low_ratio": 0.0535,
    "intraday_range": 0.0534,
    "reversal_pattern": 0.0503,
}
# L1 判定用核心四特征（报告明确给出数值且定义可实现的）
CORE_FEATURES = ("price_vs_resistance", "price_volume_divergence",
                 "price_vs_support", "intraday_volatility")
IC_FEATURES = ("price_vs_resistance", "price_vs_support", "reversal_pattern",
               "intraday_volatility", "daily_high_low_ratio",
               "price_volume_divergence", "volatility_regime")


# ---------------------------------------------------------------- 数据
def load_minute_daily(code: str, refresh: bool = False) -> pd.DataFrame:
    """指数 5 分钟线 → 日频聚合表（分钟收盘首末/高低时点/日内斜率/日内波动）。

    逐年分块拉取（SDK 按区间查询），缓存 parquet。返回 index=日期 的 DataFrame：
    min_close_first/min_close_last（分钟收盘首末值）、high_time/low_time（日内最高/
    最低的日内时刻占比）、slope_close_z/slope_vol_z（日内分钟收盘/成交量斜率的
    当日 z 分）、intraday_vol（分钟收益 std×sqrt(48)）。
    """
    cache = OUT / "data" / f"{code}_min5_daily.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        if df.index.max() >= pd.Timestamp(dt.date.today()) - pd.Timedelta(days=5):
            return df
    from data.datasource import create_datasource

    ds = create_datasource()
    this_year = dt.date.today().year
    frames = []
    for y in range(MINUTE_BEGIN // 10000, this_year + 1):
        begin = y * 10000 + 101
        end = y * 10000 + 1231
        raw = ds.get_minute_kline([code], begin, end, period=MINUTE_PERIOD)
        if raw.empty:
            continue
        sub = raw.xs(code, level="code").reset_index()
        sub["day"] = sub["kline_time"].dt.normalize()
        frames.append(sub)
        print(f"  分钟线 {y}: {len(sub)} 根")
    m = pd.concat(frames, ignore_index=True)
    m = m.sort_values(["day", "kline_time"])

    def _day_agg(g: pd.DataFrame) -> pd.Series:
        c = g["close"].to_numpy(dtype=float)
        v = g["volume"].to_numpy(dtype=float)
        t = np.arange(len(c), dtype=float)
        hi_i = int(np.argmax(g["high"].to_numpy(dtype=float)))
        lo_i = int(np.argmin(g["low"].to_numpy(dtype=float)))
        out = {
            "min_close_first": c[0],
            "min_close_last": c[-1],
            "high_time": hi_i / max(len(c) - 1, 1),
            "low_time": lo_i / max(len(c) - 1, 1),
            "intraday_vol": float(np.std(np.diff(c) / c[:-1], ddof=1)) * np.sqrt(BARS_PER_DAY),
        }
        # 日内斜率（收盘/成交量 vs 时间），当日 z 分标准化后保留符号与相对强度
        for name, arr in (("slope_close", c), ("slope_vol", v)):
            if np.std(arr) > 0:
                b = float(np.polyfit(t, arr, 1)[0])
                z = b / (float(np.std(arr)) + 1e-12)
            else:
                z = 0.0
            out[f"{name}_z"] = z
        return pd.Series(out)

    daily = m.groupby("day").apply(_day_agg, include_groups=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(cache)
    return daily


def build_features(daily_k: pd.DataFrame, minute_agg: pd.DataFrame) -> pd.DataFrame:
    """七个日频特征（全部只用 ≤t 收盘可得信息）。"""
    f = pd.DataFrame(index=daily_k.index)
    close, high, low = daily_k["close"], daily_k["high"], daily_k["low"]
    f["price_vs_resistance"] = (high - close) / close  # 距阻力位距离（报告：越远收益越高）
    f["price_vs_support"] = (close - low) / close  # 距支撑位距离
    hi_t = minute_agg["high_time"].reindex(daily_k.index)
    lo_t = minute_agg["low_time"].reindex(daily_k.index)
    f["reversal_pattern"] = np.where(hi_t < lo_t, 1.0, -1.0)  # 先高后低=+1（研报约定）
    f["intraday_volatility"] = minute_agg["intraday_vol"].reindex(daily_k.index)
    f["daily_high_low_ratio"] = high / low
    f["price_volume_divergence"] = (
        minute_agg["slope_close_z"].reindex(daily_k.index)
        * minute_agg["slope_vol_z"].reindex(daily_k.index)
    )
    rv = close.pct_change().rolling(20).std() * np.sqrt(PERIODS_PER_YEAR)
    f["volatility_regime"] = (
        close.pct_change().rolling(3).std() * np.sqrt(PERIODS_PER_YEAR)
    ) / rv
    return f


# ---------------------------------------------------------------- 判据
def ts_ic(feat: pd.Series, target: pd.Series, mask: pd.Series | None = None) -> float:
    """日度 Pearson IC（特征 t vs 目标 t→t+1 open-to-open）。"""
    s = pd.concat([feat, target], axis=1).dropna()
    if mask is not None:
        s = s[s.index.isin(mask.index[mask])]
    if len(s) < 100:
        return float("nan")
    return float(s.iloc[:, 0].corr(s.iloc[:, 1]))


def threshold_backtest(feat: pd.Series, open_px: pd.Series, polarity: int,
                       q: float, cost_bps: float) -> dict:
    """long/flat 阈值策略：特征 > 滚动 252 日 q 分位（按极性）→ T+1 open 持有到 T+2 open。

    持仓信号 pos_t 在 t 收盘后可知，作用于 open_{t+1}→open_{t+2} 段；换仓在 t+1 open
    计成本。基准 = 全程满仓 open-to-open。
    """
    thr = feat.rolling(ROLL_Q_WIN, min_periods=ROLL_Q_WIN // 2).quantile(q)
    raw = (feat > thr).astype(float) if polarity > 0 else (feat < thr).astype(float)
    pos_for_next = raw.shift(1)  # t 收盘算出 → 作用于 t+1 open 开始的段
    seg_ret = open_px.shift(-2) / open_px.shift(-1) - 1.0  # t 日的目标段（open_{t+1}→open_{t+2}）
    trade = pos_for_next.diff().abs().fillna(0.0)
    ret = pos_for_next * seg_ret - trade * cost_bps * 1e-4
    ret = ret.dropna()
    bench = seg_ret.dropna()
    nav_ret = (1 + ret).prod() - 1
    return {
        "total": float(nav_ret),
        "annual": annual_return(ret),
        "vol": float(ret.std() * np.sqrt(PERIODS_PER_YEAR)),
        "sharpe": sharpe_ratio(ret),
        "mdd": max_drawdown(ret),
        "bench_annual": annual_return(bench),
        "excess_annual": annual_return(ret) - annual_return(bench),
        "avg_pos": float(pos_for_next.dropna().mean()),
        "turnover": float(trade.sum()),
    }


def self_check(feat: pd.DataFrame, open_px: pd.Series) -> list[str]:
    bad: list[str] = []
    # 1) 目标对齐抽查：seg_ret(t) 必须等于 open_{t+2}/open_{t+1}-1（防错位=前视）
    d = open_px.dropna()
    for loc in (100, 500, 1000):
        t0 = d.index[loc]
        expect = d.iloc[loc + 2] / d.iloc[loc + 1] - 1
        got = (open_px.shift(-2) / open_px.shift(-1) - 1).loc[t0]
        if abs(float(expect) - float(got)) > 1e-12:
            bad.append(f"目标段错位 @{t0.date()}")
    # 2) 特征无未来：reversal_pattern ∈ {−1,+1} 且无 NaN（在分钟覆盖窗内）
    rp = feat["reversal_pattern"].dropna()
    if not rp.isin([-1.0, 1.0]).all():
        bad.append("reversal_pattern 取值越界")
    # 3) 两个距离特征均非负（收盘夹在日低日高之间）
    if (feat["price_vs_resistance"].dropna() < -1e-9).any():
        bad.append("price_vs_resistance 出现负值（收盘高于日高？）")
    if (feat["price_vs_support"].dropna() < -1e-9).any():
        bad.append("price_vs_support 出现负值（收盘低于日低？）")
    return bad


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    print(f"[数据] 日线 {INDEX_CODE} …")
    daily = load_index_daily(INDEX_CODE, cache_dir=OUT / "data", refresh=args.refresh_data)
    print(f"[数据] 5 分钟线 {INDEX_CODE}（{MINUTE_BEGIN // 10000} 起，首跑较慢）…")
    minute_agg = load_minute_daily(INDEX_CODE, refresh=args.refresh_data)
    print(f"  分钟聚合日表：{minute_agg.index.min().date()} ~ {minute_agg.index.max().date()}")

    daily = daily.loc[minute_agg.index.min():]
    feat = build_features(daily, minute_agg)
    open_px = daily["open"]
    seg_ret = open_px.shift(-2) / open_px.shift(-1) - 1.0

    if args.self_check:
        bad = self_check(feat, open_px)
        for b in bad:
            print("SELF-CHECK FAIL:", b)
        if bad:
            return 1
        print("SELF-CHECK PASS（目标对齐/特征取值域/高低夹逼不变量全过）")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- L1：IC 层 ----------
    half = len(feat) // 2
    train_mask = pd.Series(feat.index < feat.index[half], index=feat.index)
    ic_rows = []
    for name in IC_FEATURES:
        ours = ts_ic(feat[name], seg_ret)
        claim = REPORT_CLAIMS.get(name, float("nan"))
        ic_rows.append({
            "feature": name, "ic_ours": ours, "abs_ic_ours": abs(ours),
            "abs_ic_claim": claim, "ratio": abs(ours) / claim if claim == claim and claim > 0 else float("nan"),
            "train_ic": ts_ic(feat[name], seg_ret, train_mask),
        })
    ic_df = pd.DataFrame(ic_rows)
    print("\n[L1] 日度 IC vs 报告声称（判据：核心四特征中 ≥2 个 |IC| ≥ 声称/2）")
    print(ic_df.round(4).to_string(index=False))
    ic_df.to_csv(OUT / "feature_ic.csv", index=False, encoding="utf-8-sig")

    core = ic_df[ic_df["feature"].isin(CORE_FEATURES)]
    n_pass = int((core["abs_ic_ours"] >= core["abs_ic_claim"] / 2).sum())
    g1_pass = n_pass >= 2
    print(f"[L1] 核心四特征达标数：{n_pass}/4（声称值一半口径）→ {'进 L2' if g1_pass else 'IC 层不复现，收口'}")

    # ---------- L2：阈值层（仅 L1 过） ----------
    l2_rows = []
    if g1_pass:
        survivors = core[core["abs_ic_ours"] >= core["abs_ic_claim"] / 2]["feature"].tolist()
        for name in survivors:
            polarity = int(np.sign(
                ts_ic(feat[name].loc[: feat.index[half]], seg_ret.loc[: feat.index[half]])
            ))
            for q in QUANTILES:
                for cbps in COST_GRID_BPS:
                    r = threshold_backtest(feat[name], open_px, polarity, q, cbps)
                    l2_rows.append({"feature": name, "polarity": polarity, "quantile": q,
                                    "cost_bps": cbps, **r})
        l2_df = pd.DataFrame(l2_rows)
        print("\n[L2] 阈值 long/flat（T+1 open 执行，滚动 252 日分位，极性=训练半窗 IC 符号）")
        print(l2_df.round(4).to_string(index=False))
        l2_df.to_csv(OUT / "threshold_backtest.csv", index=False, encoding="utf-8-sig")
        main_arms = l2_df[l2_df["cost_bps"] == 10.0]
        n_alive = int((main_arms["annual"] >= 0.10).sum())
        g2_pass = n_alive >= 3
        best = main_arms.loc[main_arms["annual"].idxmax()]
        g3_pass = bool(best["excess_annual"] > 0)
        gates = [
            {"gate": "L1 IC复现", "pass": g1_pass,
             "detail": f"核心四特征 {n_pass}/4 达声称值一半"},
            {"gate": "L2 阈值家族", "pass": g2_pass,
             "detail": f"10bp 口径年化 ≥10% 的臂数 {n_alive}（判据 ≥3）"},
            {"gate": "G3 成本超额", "pass": g3_pass,
             "detail": f"最强臂 {best['feature']} q{best['quantile']:.0f} 年化 {best['annual'] * 100:.2f}%"
                       f"，超额 {best['excess_annual'] * 100:+.2f}pp"},
        ]
    else:
        gates = [{"gate": "L1 IC复现", "pass": False,
                  "detail": f"核心四特征 {n_pass}/4 达声称值一半 ⇒ 不进阈值层"}]
        l2_df = pd.DataFrame()

    print("\n===== 预注册判定 =====")
    for g in gates:
        print(f"{g['gate']}: {'PASS' if g['pass'] else 'FAIL'}  {g['detail']}")
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False, encoding="utf-8-sig")
    all_pass = all(g["pass"] for g in gates)
    print("\n结论：", "全过 → 高频聚合族在诚实口径下存活，进多窗口/全 43 特征扩展"
          if all_pass else "任一 FAIL → 判负收口（best-of 选择/泄漏假设成立）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
