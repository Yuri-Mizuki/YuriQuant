"""
ETF 轮动·两个月比赛方案探针（2026-10-27）
=========================================

场景：100 万初始资金、两个月按收益率排名。与长线定版 `champ` 的目标不同——
比赛只在意**两个月窗口的收益分布**，不要求低回撤与容量，因此放开：

  - 集中度：TopK = 1 / 2 / 3（长线是 5）；
  - 信号：20 日 / 60 日 / 12-1 动量（短窗口用快信号）；
  - 频率：周频 / 双周 / 月频；
  - 去掉 15% 波动目标与债金防御腿（它们会留 40%+ 现金、压低上行）；
  - 成本按实际费率（佣金 1bp + 滑点 5bp）。

评价口径：对每个变体跑一次全样本回测，按**自然月首日**切出所有
「t → t+2 个月」窗口，统计收益的均值/中位/25 分位/最差、窗口内最大回撤、
以及相对沪深300 的胜率。排名是看单窗口收益，所以**分布下沿比均值更重要**。

用法:
    python -m scripts.portfolio.etf_contest_probe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from data.cache import DataCache
from data.datasource import create_datasource
from scripts.portfolio.etf_rotation_p2 import (
    build_crowding,
    build_defense,
    level_from_returns,
    month_first_dates,
)
from scripts.portfolio.etf_rotation_probe import (
    BENCH,
    MIN_AMOUNT,
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "etf_contest"
CACHE_ROOT = Path(str(Config.cache()["root"]))
ALL_ETF_FILE = "etf_all_kline.parquet"
WL_CSV = ROOT / "reports" / "etf_contest5_wl" / "whitelist_ocr.csv"
# 只数不足时的占位腿：国债 / 十年国债 / 黄金（均为白名单内、流动性充足）
FILL_CODES = ["511010.SH", "511260.SH", "518880.SH"]

BASE = {"freq": "W", "top_k": 3, "absolute": True, "weighting": "equal",
        "commission_bp": 1.0, "slippage_bp": 5.0}

SPECS: list[dict] = [
    {**BASE, "name": "m20_k1_w", "signal": "mom20", "top_k": 1},
    {**BASE, "name": "m20_k2_w", "signal": "mom20", "top_k": 2},
    {**BASE, "name": "m20_k3_w", "signal": "mom20", "top_k": 3},
    {**BASE, "name": "m20_k2_2w", "signal": "mom20", "top_k": 2, "every_n": 10},
    {**BASE, "name": "m20_k2_m", "signal": "mom20", "top_k": 2, "freq": "M"},
    {**BASE, "name": "m60_k2_w", "signal": "mom60", "top_k": 2},
    {**BASE, "name": "m60_k3_w", "signal": "mom60", "top_k": 3},
    {**BASE, "name": "m12_1_k2_w", "signal": "mom12_1", "top_k": 2},
    {**BASE, "name": "m12_1_k3_w", "signal": "mom12_1", "top_k": 3},
    {**BASE, "name": "m12_1_k2_2w", "signal": "mom12_1", "top_k": 2, "every_n": 10},
    {**BASE, "name": "champ_sig_k2_w", "signal": "mom_crowd_f20", "top_k": 2},
    {**BASE, "name": "champ_sig_k3_w", "signal": "mom_crowd_f20", "top_k": 3},
    {**BASE, "name": "champ_sig_k2_2w", "signal": "mom_crowd_f20", "top_k": 2, "every_n": 10},
    {**BASE, "name": "champ_sig_k3_2w", "signal": "mom_crowd_f20", "top_k": 3, "every_n": 10},
    {**BASE, "name": "champ_sig_k3_m", "signal": "mom_crowd_f20", "top_k": 3, "freq": "M"},
    {**BASE, "name": "dual_k3_w", "signal": "mom_dual", "top_k": 3},
    {**BASE, "name": "m20_k2_w_nofilter", "signal": "mom20", "top_k": 2, "absolute": False},
    {**BASE, "name": "m20_crowd_k2_w", "signal": "mom20_crowd_f20", "top_k": 2},
]

# 正式赛规则要求「配置不少于 5 只产品」，故 TopK=1/2/3 全部不合规。以下为
# 合规组（top_k ≥ 5），新增去重开关，检验分散到 5 只后收益下沿是否守住。
K5_SPECS: list[dict] = [
    {**BASE, "name": "k5_m20_w", "signal": "mom20", "top_k": 5},
    {**BASE, "name": "k5_m20_2w", "signal": "mom20", "top_k": 5, "every_n": 10},
    {**BASE, "name": "k5_m12_1_w", "signal": "mom12_1", "top_k": 5},
    {**BASE, "name": "k5_champ_w", "signal": "mom_crowd_f20", "top_k": 5},
    {**BASE, "name": "k5_champ_2w", "signal": "mom_crowd_f20", "top_k": 5, "every_n": 10},
    {**BASE, "name": "k5_m20_w_dd90", "signal": "mom20", "top_k": 5, "dedup_corr": 0.90},
    {**BASE, "name": "k5_m20_2w_dd90", "signal": "mom20", "top_k": 5, "every_n": 10,
     "dedup_corr": 0.90},
    {**BASE, "name": "k5_m12_1_w_dd90", "signal": "mom12_1", "top_k": 5,
     "dedup_corr": 0.90},
    {**BASE, "name": "k5_champ_w_dd90", "signal": "mom_crowd_f20", "top_k": 5,
     "dedup_corr": 0.90},
    {**BASE, "name": "k5_champ_2w_dd90", "signal": "mom_crowd_f20", "top_k": 5,
     "every_n": 10, "dedup_corr": 0.90},
    # 去重后若同簇不足 5 只，允许放宽到 0.95 以观察敏感度
    {**BASE, "name": "k5_champ_w_dd95", "signal": "mom_crowd_f20", "top_k": 5,
     "dedup_corr": 0.95},
    # 硬性 ≥5 只：去重后不足时用原排名补回（否则部分调仓日只持 4 只，不合规）
    {**BASE, "name": "k5_champ_w_dd90_fill", "signal": "mom_crowd_f20", "top_k": 5,
     "dedup_corr": 0.90, "dedup_fill": True},
    {**BASE, "name": "k5_champ_2w_dd90_fill", "signal": "mom_crowd_f20", "top_k": 5,
     "every_n": 10, "dedup_corr": 0.90, "dedup_fill": True},
    {**BASE, "name": "k5_m12_1_w_dd90_fill", "signal": "mom12_1", "top_k": 5,
     "dedup_corr": 0.90, "dedup_fill": True},
    # 核心-卫星：仍持 5 只，但权重按排名线性衰减（5:4:3:2:1），保留前几名集中度
    {**BASE, "name": "k5_champ_2w_dd90_rd", "signal": "mom_crowd_f20", "top_k": 5,
     "every_n": 10, "dedup_corr": 0.90, "weighting": "rank_decay"},
    {**BASE, "name": "k5_champ_w_dd90_rd", "signal": "mom_crowd_f20", "top_k": 5,
     "dedup_corr": 0.90, "weighting": "rank_decay"},
    {**BASE, "name": "k5_m12_1_w_dd90_rd", "signal": "mom12_1", "top_k": 5,
     "dedup_corr": 0.90, "weighting": "rank_decay"},
    # 「凑满 5 只但集中在一个方向」：不做去重，允许同主题双胞胎占多个名额，
    # 再用排名衰减把权重压到前 2–3 只。用 5 个产品名额表达 1–2 个观点。
    {**BASE, "name": "k5_m12_1_w_nodd_rd", "signal": "mom12_1", "top_k": 5,
     "weighting": "rank_decay"},
    {**BASE, "name": "k5_m12_1_2w_nodd_rd", "signal": "mom12_1", "top_k": 5,
     "every_n": 10, "weighting": "rank_decay"},
    {**BASE, "name": "k5_m12_1_w_dd95_rd", "signal": "mom12_1", "top_k": 5,
     "dedup_corr": 0.95, "weighting": "rank_decay"},
    # 权重方案对比：同一选标（Top5 + 去重 0.90 + 周频），只换组合构造
    {**BASE, "name": "k5_m12_1_w_dd90_iv", "signal": "mom12_1", "top_k": 5,
     "dedup_corr": 0.90, "weighting": "inverse_vol"},
    {**BASE, "name": "k5_m12_1_w_dd90_erc", "signal": "mom12_1", "top_k": 5,
     "dedup_corr": 0.90, "weighting": "erc"},
    {**BASE, "name": "k5_m12_1_w_dd90_hrp", "signal": "mom12_1", "top_k": 5,
     "dedup_corr": 0.90, "weighting": "hrp"},
    {**BASE, "name": "k5_champ_2w_dd90_iv", "signal": "mom_crowd_f20", "top_k": 5,
     "every_n": 10, "dedup_corr": 0.90, "weighting": "inverse_vol"},
    # HRP 的稳健性：协方差窗口 40/120 + 双周频，检验是不是"60 日窗口的尖峰"
    {**BASE, "name": "k5_m12_1_w_dd90_hrp40", "signal": "mom12_1", "top_k": 5,
     "dedup_corr": 0.90, "weighting": "hrp", "cov_window": 40},
    {**BASE, "name": "k5_m12_1_w_dd90_hrp120", "signal": "mom12_1", "top_k": 5,
     "dedup_corr": 0.90, "weighting": "hrp", "cov_window": 120},
    {**BASE, "name": "k5_m12_1_2w_dd90_hrp", "signal": "mom12_1", "top_k": 5,
     "every_n": 10, "dedup_corr": 0.90, "weighting": "hrp"},
    # HRP 单只权重会飙到 49%（低波品种），加 30% 上限看是否保留优势
    {**BASE, "name": "k5_m12_1_w_dd90_hrp_mw30", "signal": "mom12_1", "top_k": 5,
     "dedup_corr": 0.90, "weighting": "hrp", "max_weight": 0.30},
]

# 集中版（2026-10-27 晚）：用户确认「配置不少于 5 只产品」按**赛期内累计
# 买入过 5 只不同产品**理解，不再要求每时点同时持有 ≥5 只。故放开 top_k=1/2/3，
# 用 5 个产品名额只作「先买 5 只」的一次性合规动作（回测只多几笔成本，不入收益）。
CONC_SPECS: list[dict] = [
    {**BASE, "name": "c1_m12_1_w_dd90", "signal": "mom12_1", "top_k": 1,
     "dedup_corr": 0.90, "dedup_fill": True},
    {**BASE, "name": "c2_m12_1_w_dd90", "signal": "mom12_1", "top_k": 2,
     "dedup_corr": 0.90, "dedup_fill": True},
    {**BASE, "name": "c3_m12_1_w_dd90", "signal": "mom12_1", "top_k": 3,
     "dedup_corr": 0.90, "dedup_fill": True},
    # Top5 作参照（旧合规版）
    {**BASE, "name": "c5_m12_1_w_dd90", "signal": "mom12_1", "top_k": 5,
     "dedup_corr": 0.90, "dedup_fill": True},
    {**BASE, "name": "c5_m12_1_w_dd90_hrp", "signal": "mom12_1", "top_k": 5,
     "dedup_corr": 0.90, "dedup_fill": True, "weighting": "hrp"},
    # 集中版 + HRP：k=2/3 时 HRP 仍有分散意义
    {**BASE, "name": "c2_m12_1_w_dd90_hrp", "signal": "mom12_1", "top_k": 2,
     "dedup_corr": 0.90, "dedup_fill": True, "weighting": "hrp"},
    {**BASE, "name": "c3_m12_1_w_dd90_hrp", "signal": "mom12_1", "top_k": 3,
     "dedup_corr": 0.90, "dedup_fill": True, "weighting": "hrp"},
    # 优化点2：拥挤度软惩罚（超出 0.80 的部分线性扣 10pp 动量）
    {**BASE, "name": "c3_m12_1_cs10_w_dd90", "signal": "mom12_1_crowdsoft",
     "top_k": 3, "dedup_corr": 0.90, "dedup_fill": True},
    {**BASE, "name": "c3_m12_1_cs10_w_dd90_hrp", "signal": "mom12_1_crowdsoft",
     "top_k": 3, "dedup_corr": 0.90, "dedup_fill": True, "weighting": "hrp"},
    # 软惩罚在更集中档位上的补充（k=1/2 只需在池内跑，扩池缺份额数据）
    {**BASE, "name": "c1_m12_1_cs10_w_dd90", "signal": "mom12_1_crowdsoft",
     "top_k": 1, "dedup_corr": 0.90, "dedup_fill": True},
    {**BASE, "name": "c2_m12_1_cs10_w_dd90_hrp", "signal": "mom12_1_crowdsoft",
     "top_k": 2, "dedup_corr": 0.90, "dedup_fill": True, "weighting": "hrp"},
    # 优化点3：6-1 动量（跳最近一月、看半年）
    {**BASE, "name": "c3_m6_1_w_dd90", "signal": "mom6_1", "top_k": 3,
     "dedup_corr": 0.90, "dedup_fill": True},
    {**BASE, "name": "c3_m6_1_w_dd90_hrp", "signal": "mom6_1", "top_k": 3,
     "dedup_corr": 0.90, "dedup_fill": True, "weighting": "hrp"},
    # 优化点5：周五调仓（引擎 freq="W" 默认取每周首个交易日 = 周一）
    {**BASE, "name": "c3_m12_1_w_dd90_fri", "signal": "mom12_1", "top_k": 3,
     "dedup_corr": 0.90, "dedup_fill": True, "rebal_weekday": "last"},
    {**BASE, "name": "c5_m12_1_w_dd90_hrp_fri", "signal": "mom12_1", "top_k": 5,
     "dedup_corr": 0.90, "dedup_fill": True, "weighting": "hrp",
     "rebal_weekday": "last"},
]

# 长线定版 champ 放进同一 2 月窗口口径，用于「比赛方案 vs 长线方案」对照。
# champ = 月频、Top5、逆波动、15% 年化波动目标、债金防御腿（DEF 合成列）、去重 0.90。
LT_SPECS: list[dict] = [
    {**BASE, "name": "lt_champ", "signal": "mom_crowd_f20", "top_k": 5, "freq": "M",
     "weighting": "inverse_vol", "target_vol": 0.15, "defense": "mix",
     "dedup_corr": 0.90},
    {**BASE, "name": "lt_champ_nodd", "signal": "mom_crowd_f20", "top_k": 5, "freq": "M",
     "weighting": "inverse_vol", "target_vol": 0.15, "defense": "mix"},
]


def build_contest_panels(close: pd.DataFrame, amount: pd.DataFrame,
                         with_crowd: bool = True) -> dict:
    """在 P1 面板上加 20/60/6-1 日动量与拥挤度（过滤版 + 软惩罚版）。"""
    p = build_panels(close, amount)
    p["mom20"] = close / close.shift(20) - 1.0
    p["mom60"] = close / close.shift(60) - 1.0
    # 优化点3：6-1 动量 = 跳过最近一个月、看之前半年。
    p["mom6_1"] = close.shift(21) / close.shift(126) - 1.0
    if with_crowd:
        crowd = build_crowding(close)
        p["crowd"] = crowd
        p["mom20_crowd_f20"] = p["mom20"].where(crowd <= 0.80)
        p["mom_crowd_f20"] = p["mom12_1"].where(crowd <= 0.80)
        # 优化点2：软惩罚——份额分位超出 0.80 的部分线性扣最多 10pp 动量，
        # 不硬剔除，避免把可选标的砍到不足只数。缺份额数据的标的按 0 惩罚。
        over = ((crowd - 0.80) / 0.20).clip(lower=0.0, upper=1.0).fillna(0.0)
        p["mom12_1_crowdsoft"] = p["mom12_1"] - 0.10 * over
        # 双周期确认：12-1 中期动量，且 20 日短期动量也为正。
        p["mom_dual"] = p["mom12_1"].where(p["mom20"] > 0)
    return p


def load_wl_bundle(cache, begin: int = WARMUP_BEGIN, end: int | None = None,
                   min_years: float = 3.0, min_amount: float = MIN_AMOUNT):
    """大赛白名单 ∩ 上市满 3 年 ∩ 流动性达标者，从 etf_all_kline 构建面板束。

    返回与 ``load_bundle`` 同构的 ``(uni, close, amount, bench)``（uni 置 None，
    下游只用于标签展示）。价格为 ``close × adj_factor``（后复权）。
    """
    w = pd.read_csv(WL_CSV, dtype={"code": str}).drop_duplicates("code")
    codes = [s + (".SZ" if s.startswith("1") else ".SH") for s in w["code"]]
    raw = pd.read_parquet(CACHE_ROOT / ALL_ETF_FILE)
    raw = raw[raw.index.get_level_values("code").isin(codes)]
    close_raw = raw["close"].unstack("code").sort_index()
    adj = (raw["adj_factor"].unstack("code")
           .reindex(index=close_raw.index, columns=close_raw.columns)
           .ffill().fillna(1.0))
    close = close_raw * adj
    amount = raw["amount"].unstack("code").reindex(
        index=close.index, columns=close.columns)

    first = close_raw.apply(lambda s: s.first_valid_index())
    last = close_raw.apply(lambda s: s.last_valid_index())
    ready = first + pd.Timedelta(days=365.25 * min_years)
    age_ok = pd.DataFrame(
        np.greater_equal.outer(close.index.values, ready.values),
        index=close.index, columns=close.columns)
    liq = amount.rolling(60, min_periods=40).mean() >= min_amount
    alive = last >= close.index.max() - pd.Timedelta(days=15)
    keep = [c for c in close.columns
            if bool(alive.get(c, False)) and bool((age_ok[c] & liq[c]).any())]
    close, amount = close[keep], amount[keep]

    hi = pd.Timestamp(str(end)) if end else close.index.max()
    lo = pd.Timestamp(str(begin))
    m = (close.index >= lo) & (close.index <= hi)
    close, amount = close.loc[m], amount.loc[m]

    idx = cache.get_index_daily(BENCH, begin, int(hi.strftime("%Y%m%d")))
    bench_close = idx["close"].unstack("code")
    bcol = [c for c in bench_close.columns if c == BENCH]
    bench = (bench_close[bcol[0]].reindex(close.index).ffill()
             .pct_change(fill_method=None) if bcol else pd.Series(dtype=float))
    print(f"[wl] 白名单 {len(codes)} 只（源内 {raw.index.get_level_values('code').nunique()}）"
          f" -> 满 {min_years:.0f} 年+流动+仍在交易 {len(keep)} 只", flush=True)
    return None, close, amount, bench


def two_month_table(eq: pd.Series, bench: pd.Series, mfs: pd.DatetimeIndex) -> pd.DataFrame:
    """按自然月首日切 t → t+2 个月窗口，返回逐窗口收益与窗口内回撤。"""
    rows = []
    for i in range(len(mfs) - 2):
        t0, t2 = mfs[i], mfs[i + 2]
        if t0 not in eq.index or t2 not in eq.index:
            continue
        seg = eq.loc[t0:t2]
        if len(seg) < 20:
            continue
        ret = float(eq.loc[t2] / eq.loc[t0] - 1.0)
        dd = float((seg / seg.cummax() - 1.0).min())
        b = (float(bench.loc[t2] / bench.loc[t0] - 1.0)
             if t0 in bench.index and t2 in bench.index else np.nan)
        rows.append({"start": t0, "end": t2, "ret": ret, "dd": dd,
                     "bench": b, "excess": ret - b if np.isfinite(b) else np.nan})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ETF 两个月比赛方案探针")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--specs", default="base",
                    choices=["base", "k5", "lt", "conc"],
                    help="base=TopK1/2/3 旧组；k5=合规组（≥5 只）；"
                         "lt=长线 champ 对照；conc=累计口径下的集中版（Top1/2/3）")
    ap.add_argument("--universe", choices=["pool", "wl"], default="pool",
                    help="pool=现有 51 只池；wl=大赛白名单扩池（满 3 年+流动性）")
    ap.add_argument("--exclude", default="",
                    help="不可交易的代码（逗号分隔），如主办方指定池不含的品种；"
                         "把 mask 置 False 后全部变体在同一可投集合内重跑")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = {"base": SPECS, "k5": K5_SPECS, "lt": LT_SPECS,
             "conc": CONC_SPECS}[args.specs]
    if args.universe == "wl":
        # 份额数据只覆盖现有 50 只池（白名单内 29 只），扩池上跑拥挤度会
        # 退化成「绝大多数标的 0 惩罚」，与基准无差别，故不跑含 crowd 的变体。
        specs = [s for s in specs if "crowd" not in s["signal"]]

    cache = DataCache(create_datasource())
    bundle = (load_wl_bundle(cache) if args.universe == "wl"
              else load_bundle(cache, WARMUP_BEGIN, None))
    close = bundle[1]
    panels = build_contest_panels(close, bundle[2],
                                  with_crowd=(args.universe == "pool"))
    excluded = [c.strip() for c in args.exclude.split(",") if c.strip()]
    if excluded:
        miss = [c for c in excluded if c not in panels["mask"].columns]
        if miss:
            raise SystemExit(f"--exclude 含池外代码：{miss}")
        # 只改 mask：所有变体在同一可投集合内选标与建仓，因子历史不变（无前视）
        for c in excluded:
            panels["mask"].loc[:, c] = False
        print(f"排除不可交易 {len(excluded)} 只：{excluded}", flush=True)
    bench = (1.0 + bundle[3].fillna(0.0)).cumprod()
    mfs = month_first_dates(close.index, 20200101,
                            int(close.index.max().strftime("%Y%m%d")))

    rows, per_window = [], []
    for spec in specs:
        rb = None
        if spec.get("every_n"):
            idx = close.index[close.index >= pd.Timestamp("20200101")]
            rb = set(idx[:: int(spec["every_n"])])
        elif spec.get("rebal_weekday") == "last":
            # 引擎 freq="W" 取每周首个交易日（周一）；这里改取每周最后一个交易日。
            idx = close.index[close.index >= pd.Timestamp("20200101")]
            s = pd.Series(idx, index=idx)
            rb = set(s.groupby(s.index.to_period("W")).last())
        extra = None
        if spec.get("defense"):
            # 防御腿是合成资产（债+金 50/50），作为净值列注入回测
            extra = pd.DataFrame({"DEF": level_from_returns(build_defense(close)["mix"])})
        r = run_variant(bundle, panels, spec, 20200101, None,
                        extra_close=extra, rebalance_days=rb)
        tab = two_month_table(r["equity"], bench, mfs)
        tab["name"] = spec["name"]
        per_window.append(tab)
        # 正式赛窗口是 10 月中 → 12 月中，季节位置固定：单独统计 10 月起跑的窗口
        oct_tab = tab[tab["start"].dt.month == 10]
        m = r
        hc = r["hold_counts"]
        min_hold = float(hc.min()) if len(hc) else np.nan
        lt5 = float((hc < 5).mean()) if len(hc) else np.nan
        rows.append({
            "name": spec["name"], "signal": spec["signal"], "top_k": spec["top_k"],
            "universe": args.universe,
            "freq": "2W" if spec.get("every_n") else spec["freq"],
            "dedup": spec.get("dedup_corr", np.nan),
            "dedup_window": spec.get("dedup_window", np.nan),
            "absolute": spec["absolute"],
            "n_windows": len(tab),
            "mean": tab["ret"].mean(), "median": tab["ret"].median(),
            "p25": tab["ret"].quantile(0.25), "p10": tab["ret"].quantile(0.10),
            "worst": tab["ret"].min(), "best": tab["ret"].max(),
            "win_rate": float((tab["ret"] > 0).mean()),
            "p_gt10": float((tab["ret"] > 0.10).mean()),
            "p_gt20": float((tab["ret"] > 0.20).mean()),
            "p_lt_m10": float((tab["ret"] < -0.10).mean()),
            "beat_bench": float((tab["excess"] > 0).mean()),
            "median_dd": tab["dd"].median(), "worst_dd": tab["dd"].min(),
            "oct_n": len(oct_tab), "oct_mean": oct_tab["ret"].mean(),
            "oct_min": oct_tab["ret"].min() if len(oct_tab) else np.nan,
            "ann": m["annual_return"], "sharpe": m["sharpe"],
            "full_mdd": m["max_drawdown"], "avg_turnover": m["avg_turnover"],
            "avg_holdings": m["avg_holdings"],
            "min_holdings": min_hold, "pct_days_lt5": lt5,
            "cost_drag": m["cost_drag"],
        })
        print(f"{spec['name']:<20} 2M均值={tab['ret'].mean():+.2%} "
              f"中位={tab['ret'].median():+.2%} p10={tab['ret'].quantile(0.10):+.2%} "
              f"最差={tab['ret'].min():+.2%} 胜率={rows[-1]['win_rate']:.0%} "
              f"10月起={rows[-1]['oct_mean']:+.2%} "
              f"持仓={rows[-1]['avg_holdings']:.1f}/最少{min_hold:.0f}", flush=True)

    summary = pd.DataFrame(rows).sort_values("median", ascending=False)
    tag = {"base": "", "k5": "_k5", "lt": "_lt", "conc": "_conc"}[args.specs]
    if args.universe == "wl":
        tag += "_wl"
    summary.to_csv(out / f"contest{tag}_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(per_window).to_csv(out / f"contest{tag}_windows.csv", index=False,
                                 encoding="utf-8-sig")
    (out / f"run_config{tag}.json").write_text(json.dumps({
        "scenario": "2 个月收益排名赛，初始 100 万，正式赛 2026-10-12 → 12-18",
        "constraint": "配置不少于 5 只产品 + 至少 5 笔成交 + 普通交易（无两融/套利）",
        "begin": 20200101, "bench": BENCH, "min_amount": MIN_AMOUNT,
        "cost": "commission 1bp + slippage 5bp",
        "window": "自然月首日 t → t+2 个月",
        "specs": specs,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["name", "top_k", "freq", "dedup", "n_windows", "mean", "median", "p10",
            "worst", "win_rate", "p_gt10", "p_gt20", "p_lt_m10", "worst_dd",
            "oct_mean", "oct_min", "min_holdings", "pct_days_lt5"]
    print("\n[按中位数排序]\n" + summary[cols].to_string(
        index=False, float_format=lambda x: f"{x:+.2%}"))
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
