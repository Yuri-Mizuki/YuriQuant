"""
日内收益分解 + 时段效应分析
============================

5 分钟频率研究的第一步（解释性分析）：先用 5 分钟 + 日线数据回答
"alpha 在收益的哪一半、在一天的哪个时段"，而不是直接挖因子。

输出两类结果：
1. 收益分解（日线口径）：隔夜收益 vs 日内收益的均值/方差贡献/相关。
2. 时段效应（5 分钟口径）：成交量 U 型曲线、时段波动率结构、
   开盘/尾盘行为、首根 bar 对全天的预测方向。

口径说明
--------
- 收益全部用**未复权价**计算，除权/除息日样本剔除（分钟线无复权参数，
  除权日 bar 有跳变；剔除比折算更干净）。
- 组合层面时间序列：每日对成分股取横截面均值（等权），得到逐日序列后
  用 Newey-West t 做显著性检验（面板 pooled 的普通 t 会高估）。
- 时段收益用 bar 内收益（close/open-1）累加近似，避免跨日拼接。

用法
----
    python -m scripts.intraday_analysis --mock          # mock 验证管线（2023-2024）
    python -m scripts.intraday_analysis                  # 真实 HS300 2025（默认）
    python -m scripts.intraday_analysis --begin 20250101 --end 20251231
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cli_common import add_real_mock_args, setup_logging  # noqa: E402


matplotlib.use("Agg")

from data.cache import DataCache  # noqa: E402
from data.cache_helpers import load_daily  # noqa: E402
from data.datasource import create_datasource  # noqa: E402
from data.offline import OfflineDataSource  # noqa: E402
from data.universe import Universe  # noqa: E402
from research.robust_stats import nw_tstat  # noqa: E402

log = setup_logging("intraday_analysis")

# 中文字体（Windows SimHei；macOS/linux 可改）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_PERIOD = 5

# ---- 时段切分：5 分钟档 48 根 ----

def _seg_of(time_label: str) -> str:
    """把 5 分钟 bar 标签（HH:MM）映射到四段。"""
    h, m = int(time_label[:2]), int(time_label[3:])
    t = h * 60 + m
    if t < 600:      # 9:30-9:59 开盘 30 分钟
        return "open30"
    if t < 690:      # 10:00-11:25
        return "mid_morning"
    if t < 870:      # 13:00-14:25
        return "mid_afternoon"
    return "close30"  # 14:30-14:55 尾盘 30 分钟

def _filter_ex_dividend(df, status) -> pd.DataFrame:
    """剔除除权/除息日的样本（收益分解与时段效应共用）。

    df: 长表，含 date 列（分钟数据先 normalize 出 date）。
    status: history_stock_status 长表（date, code, is_ex_dividend, is_ex_rights），
        date/code 可能在 MultiIndex 上（缓存层写盘形式），也可能在列上。
    返回过滤后的 df。
    """
    if status is None or status.empty:
        return df
    st = status.copy()
    if isinstance(st.index, pd.MultiIndex):
        st = st.reset_index()
    flags = st[st["is_ex_dividend"] | st["is_ex_rights"]]
    if flags.empty:
        return df
    keys = set(zip(pd.to_datetime(flags["date"]).dt.normalize(), flags["code"]))
    mask = pd.Series(
        [ (d, c) not in keys for d, c in zip(df["date"].dt.normalize(), df["code"]) ],
        index=df.index,
    )
    return df[mask]

def load_data(cache, uni, index_code, begin, end, period):
    """拉取股票池、日线、5 分钟 K 线、状态表。返回各 DataFrame。"""
    codes, cal, daily = load_daily(cache, uni, index_code, begin, end)
    if not codes:
        raise RuntimeError(f"{index_code} 在 {end or cal[-1]} 无成分股")
    target = end or cal[-1]
    log.info("日线: %d 行", len(daily))
    mk = cache.get_minute_kline(codes, begin, target, period=period)
    log.info("%d 分钟K线: %d 行", period, len(mk))
    status = cache.get_history_stock_status(codes, begin, target)
    log.info("历史状态: %d 行", len(status))
    return codes, daily, mk, status

# ---- 1. 收益分解 ----

def decompose_returns(daily, status, year: int) -> dict:
    """隔夜 vs 日内收益分解。

    daily: (date, code) 多索引，open/close。
    status: 除权除息日剔除用。
    返回汇总 dict + 逐日时间序列 DataFrame。
    """
    d = daily.reset_index()
    d = _filter_ex_dividend(d, status) if status is not None and not status.empty else d
    d = d[["date", "code", "open", "close"]].dropna()
    d["date"] = d["date"].dt.normalize()

    open_w = d.pivot(index="date", columns="code", values="open")
    close_w = d.pivot(index="date", columns="code", values="close")
    # 对齐后逐日
    open_w, close_w = open_w.align(close_w, join="inner")
    prev_close = close_w.shift(1)

    r_o = open_w / prev_close - 1.0   # 隔夜收益
    r_i = close_w / open_w - 1.0      # 日内收益
    r_d = close_w / prev_close - 1.0  # 全日收益
    # 恒等式校验：r_d ≈ (1+r_o)(1+r_i)-1
    ident = ((1 + r_o) * (1 + r_i) - 1 - r_d).abs().max().max()

    # 组合层面时间序列（每日横截面均值）
    ts = pd.DataFrame({
        "r_overnight": r_o.mean(axis=1, skipna=True),
        "r_intraday": r_i.mean(axis=1, skipna=True),
        "r_daily": r_d.mean(axis=1, skipna=True),
    }).dropna()

    def _stat(s: pd.Series) -> dict:
        t, se, lag = nw_tstat(s.values)
        return {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "t_nw": t,
            "se_nw": se,
            "lag": lag,
            "annualized": float(s.mean()) * len(ts),
        }

    stats = {k: _stat(ts[k]) for k in ts.columns}

    # 方差分解（组合层面）：Var(r_d) ≈ Var(r_o) + Var(r_i) + 2Cov
    var_o, var_i, var_d = ts["r_overnight"].var(), ts["r_intraday"].var(), ts["r_daily"].var()
    cov_oi = ts["r_overnight"].cov(ts["r_intraday"])
    var_recon = var_o + var_i + 2 * cov_oi
    corr_oi = ts["r_overnight"].corr(ts["r_intraday"])

    # 横截面（pooled）：隔夜-日内相关、个股层面均值分布
    pooled = pd.DataFrame({"r_o": r_o.stack(), "r_i": r_i.stack()}).dropna()
    corr_pooled = pooled["r_o"].corr(pooled["r_i"])
    by_code = pooled.groupby(level="code").agg(
        mean_o=("r_o", "mean"), mean_i=("r_i", "mean"),
        corr=("r_o", lambda x: x.corr(pooled["r_i"].loc[x.index])),
    )
    # 简化个股相关：逐 code 算
    by_code = pooled.groupby(level="code").apply(
        lambda g: pd.Series({
            "mean_o": g["r_o"].mean(), "mean_i": g["r_i"].mean(),
            "corr": g["r_o"].corr(g["r_i"]),
        }), include_groups=False,
    ).dropna()

    summary = {
        "year": year,
        "n_days": len(ts),
        "identity_max_err": float(ident),
        "corr_overnight_intraday_ts": float(corr_oi),
        "corr_overnight_intraday_pooled": float(corr_pooled),
        "var_decomp": {
            "var_daily": float(var_d),
            "var_overnight": float(var_o),
            "var_intraday": float(var_i),
            "cov_oi": float(cov_oi),
            "var_reconstructed": float(var_recon),
            "overnight_share": float(var_o / var_d) if var_d else 0.0,
            "intraday_share": float(var_i / var_d) if var_d else 0.0,
            "cov_share": float(2 * cov_oi / var_d) if var_d else 0.0,
        },
        "stats": stats,
        "by_code": by_code,
    }
    return summary, ts

# ---- 2. 时段效应 ----

def time_of_day_effects(mk, status, year: int) -> dict:
    """5 分钟时段效应：成交量 U 型、时段波动率、开盘/尾盘、首根预测。

    mk: (kline_time, code) 多索引 5 分钟 K 线。
    """
    df = mk.reset_index()
    df["date"] = df["kline_time"].dt.normalize()
    df = _filter_ex_dividend(df, status) if status is not None and not status.empty else df
    df = df.dropna(subset=["open", "close", "volume"])
    df["bar_ret"] = df["close"] / df["open"] - 1.0
    df["time_label"] = df["kline_time"].dt.strftime("%H:%M")
    df["seg"] = df["time_label"].map(_seg_of)

    # 成交量分布（U 型）：各时段成交量占全天比例
    vol_by_time = df.groupby("time_label")["volume"].sum()
    vol_share = vol_by_time / vol_by_time.sum()

    # 时段波动率与收益（bar 内收益）
    bar_stats = df.groupby("time_label")["bar_ret"].agg(["mean", "std"])
    bar_stats["abs_mean"] = df.groupby("time_label")["bar_ret"].apply(lambda s: s.abs().mean())

    # 四段累计收益：每 (date, code) 段内 bar_ret 求和 → 组合时间序列 → NW t
    seg_sum = df.groupby(["date", "code", "seg"])["bar_ret"].sum().unstack("seg")
    seg_stats = {}
    for seg in ["open30", "mid_morning", "mid_afternoon", "close30"]:
        if seg not in seg_sum.columns:
            continue
        # seg_sum[seg] 是 Series（index=(date, code)），组合层面 = 每日横截面均值
        ts = seg_sum[seg].groupby(level="date").mean().dropna()
        t, se, lag = nw_tstat(ts.values)
        seg_stats[seg] = {"mean": float(ts.mean()), "t_nw": t, "n_days": len(ts)}

    # 首根 bar 方向 vs 全天日内收益（开盘延续 or 反转）
    first = df[df["time_label"] == "09:30"].set_index(["date", "code"])["bar_ret"]
    intraday = df.groupby(["date", "code"])["bar_ret"].sum()
    pair = pd.DataFrame({"first": first, "intraday": intraday}).dropna()
    up = pair[pair["first"] > 0]["intraday"]
    dn = pair[pair["first"] < 0]["intraday"]
    # 组合层面时间序列
    up_ts = up.groupby(level="date").mean()
    dn_ts = dn.groupby(level="date").mean()
    first_bar = {
        "n_up_days": int(up_ts.count()),
        "n_dn_days": int(dn_ts.count()),
        "intraday_after_up_open": float(up_ts.mean()),
        "intraday_after_dn_open": float(dn_ts.mean()),
        "t_nw_up": nw_tstat(up_ts.values)[0],
        "t_nw_dn": nw_tstat(dn_ts.values)[0],
        # 首根收益与全天日内收益的相关（pooled 横截面）
        "corr_first_intraday": float(pair["first"].corr(pair["intraday"])),
    }

    return {
        "year": year,
        "vol_share_by_time": vol_share,
        "bar_stats_by_time": bar_stats,
        "seg_stats": seg_stats,
        "first_bar": first_bar,
    }

# ---- 输出 ----

def make_plots(summary: dict, tod: dict, out_path: Path, period: int) -> None:
    """2x2 面板：收益分解、成交量 U 型、时段波动率、首根 bar 预测。"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. 收益分解
    ax = axes[0, 0]
    stats = summary["stats"]
    means = [stats["r_overnight"]["mean"] * 100, stats["r_intraday"]["mean"] * 100,
             stats["r_daily"]["mean"] * 100]
    labels = ["隔夜", "日内", "全日"]
    bars = ax.bar(labels, means, color=["#185FA5", "#0F6E56", "#888780"])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
                f"{m:.3f}%", ha="center", fontsize=10)
    vd = summary["var_decomp"]
    ax.set_title(f"{summary['year']} 收益分解（日均 %）\n"
                 f"方差占比 隔夜 {vd['overnight_share']*100:.0f}% · "
                 f"日内 {vd['intraday_share']*100:.0f}% · 相关 {summary['corr_overnight_intraday_ts']:.2f}")
    ax.axhline(0, color="#B4B2A9", linewidth=0.8)

    # 2. 成交量 U 型
    ax = axes[0, 1]
    vs = tod["vol_share_by_time"]
    ax.plot(range(len(vs)), vs.values * 100, marker="o", markersize=2.5,
            color="#185FA5", linewidth=1.2)
    ax.set_xticks(range(0, len(vs), 6))
    ax.set_xticklabels(vs.index[::6], rotation=45, fontsize=8)
    ax.set_ylabel("成交量占比 %")
    ax.set_title(f"{tod['year']} 成交量日内分布（{period} 分钟，U 型）")

    # 3. 时段波动率结构
    ax = axes[1, 0]
    bs = tod["bar_stats_by_time"]
    ax.plot(range(len(bs)), bs["abs_mean"].values * 100, marker="s", markersize=2.5,
            color="#0F6E56", linewidth=1.2, label="|bar 收益| 均值")
    ax.set_xticks(range(0, len(bs), 6))
    ax.set_xticklabels(bs.index[::6], rotation=45, fontsize=8)
    ax.set_ylabel("%")
    ax.set_title("时段波动率结构（bar 内收益绝对均值）")
    ax.legend(fontsize=9)

    # 4. 首根 bar 预测
    ax = axes[1, 1]
    fb = tod["first_bar"]
    vals = [fb["intraday_after_up_open"] * 100, fb["intraday_after_dn_open"] * 100]
    bars = ax.bar(["首根上涨后\n全天日内", "首根下跌后\n全天日内"], vals,
                  color=["#A32D2D", "#0F6E56"])
    for b, m in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
                f"{m:.3f}%", ha="center", fontsize=10)
    ax.axhline(0, color="#B4B2A9", linewidth=0.8)
    ax.set_title(f"开盘首根方向 vs 全天日内收益（相关 {fb['corr_first_intraday']:.2f}）")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("图表已保存: %s", out_path)

def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"

def main():
    parser = argparse.ArgumentParser(description="日内收益分解 + 时段效应分析")
    add_real_mock_args(parser, offline=True, mock_help="用 mock 数据验证管线（2023-2024）")
    parser.add_argument("--index", default="000300.SH", help="指数代码")
    parser.add_argument("--begin", type=int, default=None, help="起始日 YYYYMMDD")
    parser.add_argument("--end", type=int, default=None, help="结束日 YYYYMMDD")
    parser.add_argument("--period", type=int, default=DEFAULT_PERIOD, help="分钟档位，默认 5")
    parser.add_argument("--out-dir", default="reports", help="输出目录")
    args = parser.parse_args()

    if args.mock:
        import tempfile
        from tests.conftest import MockDataSource
        ds = MockDataSource()
        begin = args.begin or 20230103
        end = args.end or 20241231
        # mock 用独立临时缓存，避免污染真实缓存目录（e:/data/parquet 是 2025 数据）
        cache_root = tempfile.mkdtemp(prefix="mock_cache_")
    elif args.offline:
        ds = OfflineDataSource()
        begin = args.begin or 20250101
        end = args.end or 20251231
        cache_root = None
    else:
        ds = create_datasource()
        begin = args.begin or 20250101
        end = args.end or 20251231
        cache_root = None
    cache = DataCache(ds, cache_root=cache_root)
    uni = Universe(cache)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    year = int(str(end)[:4])

    codes, daily, mk, status = load_data(cache, uni, args.index, begin, end, args.period)
    if daily.empty or mk.empty:
        log.error("数据为空，无法分析（--mock 需要先确认 mock 日历，真实模式需先 update_data）")
        sys.exit(1)

    # 1. 收益分解
    log.info("计算收益分解 ...")
    summary, ts = decompose_returns(daily, status, year)

    # 2. 时段效应
    log.info("计算时段效应 ...")
    tod = time_of_day_effects(mk, status, year)

    # 3. 图表
    png = out_dir / f"intraday_analysis_{year}.png"
    make_plots(summary, tod, png, args.period)

    # 4. 落盘
    ts_path = out_dir / f"intraday_ts_{year}.csv"
    ts.to_csv(ts_path)
    rows = []
    stats = summary["stats"]
    for k, v in stats.items():
        rows.append({"metric": f"mean_{k}", "value": _fmt(v["mean"], 6)})
        rows.append({"metric": f"t_nw_{k}", "value": _fmt(v["t_nw"])})
        rows.append({"metric": f"annualized_{k}", "value": _fmt(v["annualized"], 6)})
    rows.append({"metric": "corr_overnight_intraday_ts", "value": _fmt(summary["corr_overnight_intraday_ts"])})
    rows.append({"metric": "corr_overnight_intraday_pooled", "value": _fmt(summary["corr_overnight_intraday_pooled"])})
    vd = summary["var_decomp"]
    rows.append({"metric": "var_share_overnight", "value": _fmt(vd["overnight_share"])})
    rows.append({"metric": "var_share_intraday", "value": _fmt(vd["intraday_share"])})
    for seg, s in tod["seg_stats"].items():
        rows.append({"metric": f"seg_mean_{seg}", "value": _fmt(s["mean"], 6)})
        rows.append({"metric": f"seg_t_{seg}", "value": _fmt(s["t_nw"])})
    fb = tod["first_bar"]
    rows.append({"metric": "firstbar_intraday_after_up", "value": _fmt(fb["intraday_after_up_open"], 6)})
    rows.append({"metric": "firstbar_intraday_after_dn", "value": _fmt(fb["intraday_after_dn_open"], 6)})
    rows.append({"metric": "firstbar_corr_first_intraday", "value": _fmt(fb["corr_first_intraday"])})
    summary_df = pd.DataFrame(rows)
    sum_path = out_dir / f"intraday_summary_{year}.csv"
    summary_df.to_csv(sum_path, index=False)

    # 5. 控制台摘要
    log.info("===== 收益分解（组合层面，%d 日）=====", summary["n_days"])
    for k in ("r_overnight", "r_intraday", "r_daily"):
        s = stats[k]
        log.info("%-12s mean=%+.5f%%  annual=%+.2f%%  t_nw=%+.2f  std=%.4f%%",
                 k, s["mean"] * 100, s["annualized"] * 100, s["t_nw"], s["std"] * 100)
    log.info("隔夜×日内相关: 时间序列 %.3f / pooled %.3f", summary["corr_overnight_intraday_ts"],
             summary["corr_overnight_intraday_pooled"])
    log.info("方差分解: 隔夜 %.1f%% + 日内 %.1f%% + 协方差 %.1f%% = %.4f (实际 %.4f)",
             vd["overnight_share"] * 100, vd["intraday_share"] * 100, vd["cov_share"] * 100,
             vd["var_reconstructed"], vd["var_daily"])

    log.info("===== 时段效应（%d 分钟）=====", args.period)
    for seg, s in tod["seg_stats"].items():
        log.info("%-12s 均值 %+.5f%%  t_nw=%+.2f", seg, s["mean"] * 100, s["t_nw"])
    log.info("首根 bar 上涨后全天日内 %+.5f%% / 下跌后 %+.5f%% / 相关 %.3f",
             fb["intraday_after_up_open"] * 100, fb["intraday_after_dn_open"] * 100,
             fb["corr_first_intraday"])

    # 6. 实验记录
    try:
        from research.experiments import record_experiment
        record_experiment(
            kind="intraday_analysis",
            command=" ".join(sys.argv),
            params={"index": args.index, "begin": begin, "end": end,
                    "period": args.period, "mock": args.mock, "n_codes": len(codes)},
            data_fingerprint=cache.get_fingerprint(),
            result_path=str(out_dir),
            metrics={
                "mean_r_overnight": stats["r_overnight"]["mean"],
                "mean_r_intraday": stats["r_intraday"]["mean"],
                "corr_oi": summary["corr_overnight_intraday_ts"],
                "var_share_intraday": vd["intraday_share"],
                "seg_open30": tod["seg_stats"]["open30"]["mean"],
                "seg_close30": tod["seg_stats"]["close30"]["mean"],
            },
            note="日内收益分解 + 时段效应",
        )
    except Exception as e:
        log.warning("实验记录写入失败: %s", e)

    log.info("完成。输出: %s, %s, %s", ts_path, sum_path, png)

if __name__ == "__main__":
    main()