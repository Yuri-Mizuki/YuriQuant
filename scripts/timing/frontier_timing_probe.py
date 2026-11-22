"""择时前沿三源探针（纯择时线，reports/frontier_timing_probe/）。

背景：multisrc_timing_probe（2026-11-22 判负）证明非价格源在**单资产合成**里无增量，
但 L1 层两融/预告 IC 为真；重启条件①的残余前沿 = 期权隐含 / 跨市场 / 情绪 / 利率。
本探针测其中数据已就位的三张牌（期权源撤下、情绪源待接入，见 §四 边界）：

  S1 跨市场领先滞后：
     us_ret1  = 标普500「上一交易日」收盘收益（北京时间当日凌晨 04:00 收盘，
                早于 A 股 t 日开盘，更早于 t 收盘 ⇒ 因果）
     hk_ret1  = 恒生指数「同日」收盘收益（16:00 收盘，晚于 A 股 15:00 ⇒ t 收盘后
                可知，作用于 t+1 段 ⇒ 因果）
  S2 利率（akshare 中债国债收益率曲线，替代坏掉的 SDK treasury_yield）：
     term_spread = 10Y−1Y 期限利差；term_chg5 = 利差 5 日变化；y10_chg5 = 10Y 5 日变化
  S3 高频结构（hf_minagg 缓存对照族）：hf_pvr / hf_pvd / hf_revpat（已知日 IC 4~7%）

预注册 L1 判据（与 multisrc 一致）：每特征在**日**（open_{t+1}→open_{t+2}）或**周**
（open_{t+1}→open_{t+6}）目标上 |IC| ≥ 2%/3% ⇒ 过门。门的意义 = 「前沿源有没有超过
已封底价格类的信息」，不构成合成承诺（multisrc 已证单资产合成无增量，本探针**不做
L2 合成**——若 L1 出强源，另行立项）。

用法::

    python -m scripts.timing.frontier_timing_probe --self-check
    python -m scripts.timing.frontier_timing_probe
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.timing.hf_minagg_probe import build_features, load_minute_daily
from scripts.timing.shell import load_index_daily
from scripts.timing.shell import ROOT

OUT = ROOT / "reports" / "frontier_timing_probe"
INDEX_CODE = "000300.SH"
BEGIN = "2013-01-04"  # 分钟数据起点（hf 对照族需要）
IC_DAILY = 0.02
IC_WEEKLY = 0.03

FEATURES = ["us_ret1", "hk_ret1", "term_spread", "term_chg5", "y10_chg5",
            "hf_pvr", "hf_pvd", "hf_revpat"]
NONPRICE = {"us_ret1", "hk_ret1", "term_spread", "term_chg5", "y10_chg5"}


# ---------------------------------------------------------------- 数据源
def load_us_daily(refresh: bool = False) -> pd.Series:
    """标普500 日收盘（akshare sina）。"""
    cache = OUT / "data" / "spx.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)["close"]
    import akshare as ak

    df = ak.index_us_daily_sina(symbol=".INX")
    s = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float).sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame("close").to_parquet(cache)
    return s


def load_hk_daily(refresh: bool = False) -> pd.Series:
    """恒生指数 日收盘（akshare sina）。"""
    cache = OUT / "data" / "hsi.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)["close"]
    import akshare as ak

    df = ak.index_hk_daily_sina(symbol="HSI")
    s = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float).sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame("close").to_parquet(cache)
    return s


def load_bond_yield(refresh: bool = False) -> pd.DataFrame:
    """中债国债收益率（1年/10年，%），akshare bond_china_yield。"""
    cache = OUT / "data" / "bond_yield.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    import akshare as ak

    # akshare 东财接口：2015 年起才有数据，且**长区间查询返回空**（单查 ≤1 年可行，
    # 实测 2026-11-22）⇒ 按年分段拉取再拼接
    frames = []
    for y in range(2015, 2027):
        frames.append(ak.bond_china_yield(start_date=f"{y}0101", end_date=f"{y}1231"))
    df = pd.concat(frames, ignore_index=True)
    g = df[df["曲线名称"] == "中债国债收益率曲线"].copy()
    g.index = pd.to_datetime(g["日期"].astype(str))
    out = g[["1年", "10年"]].astype(float).sort_index()
    out.columns = ["y1y", "y10y"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def build_frontier_features(daily_k: pd.DataFrame, minute_agg: pd.DataFrame) -> pd.DataFrame:
    """八个特征（全部 ≤t 收盘可知）。"""
    f = pd.DataFrame(index=daily_k.index)
    # S1 跨市场
    us = load_us_daily()
    us_ret = us.pct_change()
    # 标普 t-1（最后一个严格早于 A 股 t 的美股交易日）→ A 股 t 日特征
    f["us_ret1"] = pd.Series(us_ret.reindex(us_ret.index.union(daily_k.index)).ffill()
                             .reindex(daily_k.index - pd.Timedelta(days=1))
                             .values, index=daily_k.index)
    hk = load_hk_daily()
    hk_ret = hk.pct_change().reindex(daily_k.index)  # 恒指同日（16:00 收盘 > A 股 15:00）
    f["hk_ret1"] = hk_ret
    # S2 利率
    y = load_bond_yield()
    spread = (y["y10y"] - y["y1y"]).reindex(daily_k.index).ffill()
    y10 = y["y10y"].reindex(daily_k.index).ffill()
    f["term_spread"] = spread
    f["term_chg5"] = spread - spread.shift(5)
    f["y10_chg5"] = y10 - y10.shift(5)
    # S3 高频结构（对照族）
    hf = build_features(daily_k, minute_agg)
    f["hf_pvr"] = hf["price_vs_resistance"]
    f["hf_pvd"] = hf["price_volume_divergence"]
    f["hf_revpat"] = hf["reversal_pattern"]
    return f


def ic_of(x: pd.Series, y: pd.Series) -> float:
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) < 200:
        return float("nan")
    return float(s.iloc[:, 0].corr(s.iloc[:, 1]))


def self_check(feat: pd.DataFrame, daily_k: pd.DataFrame) -> list[str]:
    bad = []
    us = feat["us_ret1"].dropna()
    if us.abs().max() > 0.15:
        bad.append("us_ret1 出现 >±15% 的日收益（映射错位？）")
    hk = feat["hk_ret1"].dropna()
    if hk.abs().max() > 0.15:  # 2025-04-07 关税崩盘日恒指单日 −13.2% 为真实行情
        bad.append("hk_ret1 出现 >±15% 的日收益（映射错位？）")
    sp = feat["term_spread"].dropna()
    if len(sp) < 1000 or not ((sp > -2).all() and (sp < 5).all()):
        bad.append("期限利差量级异常")
    close = daily_k["close"].reindex(feat.index)
    if ((close.shift(1) - close.shift(2)) / close.shift(2)).abs().max() > 0.15:
        bad.append("指数日收益量级异常")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    daily = load_index_daily(INDEX_CODE, cache_dir=OUT / "data",
                             refresh=args.refresh_data).loc[BEGIN:]
    minute_agg = load_minute_daily(INDEX_CODE)
    feat = build_frontier_features(daily, minute_agg)

    if args.self_check:
        bad = self_check(feat, daily)
        for b in bad:
            print("SELF-CHECK FAIL:", b)
        if bad:
            return 1
        print("SELF-CHECK PASS（跨市场映射/利差量级/日收益不变量全过）")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    open_px = daily["open"]
    tgt_d = open_px.shift(-2) / open_px.shift(-1) - 1.0
    tgt_w = open_px.shift(-6) / open_px.shift(-1) - 1.0

    rows = []
    for c in FEATURES:
        ic_d = ic_of(feat[c], tgt_d)
        ic_w = ic_of(feat[c], tgt_w)
        rows.append({"feature": c, "source": "非价格" if c in NONPRICE else "高频对照",
                     "ic_daily": ic_d, "ic_weekly": ic_w,
                     "abs_ic_daily": abs(ic_d),
                     "pass": abs(ic_d) >= IC_DAILY or abs(ic_w) >= IC_WEEKLY,
                     "n_days": int(pd.concat([feat[c], tgt_d], axis=1).dropna().shape[0])})
    ic_df = pd.DataFrame(rows)
    print("\n[L1] 前沿源 IC 门（日 ≥2% / 周 ≥3%；目标 = 次日/次周 open-to-open）")
    print(ic_df.round(4).to_string(index=False))
    ic_df.to_csv(OUT / "sources_ic.csv", index=False, encoding="utf-8-sig")

    np_pass = ic_df[(ic_df["source"] == "非价格") & ic_df["pass"]]["feature"].tolist()
    gates = [{"gate": "L1 前沿非价格源", "pass": bool(len(np_pass) >= 1),
              "detail": f"过门非价格特征：{np_pass or '无'}"
                        f"（对照：高频族 {int(ic_df[ic_df['source'] == '高频对照']['pass'].sum())}/3 过门）"}]
    print("\n===== 判定 =====")
    for g in gates:
        print(f"{g['gate']}: {'PASS' if g['pass'] else 'FAIL'}  {g['detail']}")
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False, encoding="utf-8-sig")
    print("\n结论：", "前沿源有超门信息 → 对过门源另行立项（不并入已判负的合成）"
          if gates[0]["pass"] else "前沿源无超门信息 → 该方向收口，价格类 IC 保持唯一已知来源")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
