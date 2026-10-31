"""ETF 轮动 · 档位评测（80 个滚动 2 月窗口，口径统一 T+1 open）

给定一个 profile（或 `FINAL` 上的变体），输出对外引用所需的全套指标：

    年化 / Sharpe / MDD / 2 月窗口 均值·中位·p10·最差·胜率 / 换手 / 最大单只

以及与**定版基线**的同窗口配对检验（Δ均值、t、bootstrap CI、胜率）。

口径与 ``reports/etf_contest5/README.md`` §零 严格一致：

  - 单次连续回测 ``20200101 → 末日``，**事后**按月首切 2 月窗口（80 个，相邻重叠 1 月）
    —— 不要逐窗独立回测，否则 dedup/HRP 会冷启动失效（见 ``run_variant`` docstring）；
  - T+1 open 的 ``execution_split`` 由**该变体自身**的 ``rebalance_dates``（信号日 T）构造，
    与 §零 的 30.63% 同源。若改用「全索引周分组的第一个交易日」，contest3 曾得 26.17%
    （差 0.06pp，来自首/末各一个调仓日），两者不可混用。

用法::

    python -m scripts.portfolio.etf_contest5_eval                 # 默认四档
    python -m scripts.portfolio.etf_contest5_eval --only contest3
    python -m scripts.portfolio.etf_contest5_eval --csv out.csv   # 落盘

结论指向 ``reports/etf_contest5/README.md``；本脚本只产数，不做判断。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import build_execution_split
from data.cache import DataCache
from data.datasource import create_datasource
from scripts.portfolio.etf_champ_signals import PROFILES
from scripts.portfolio.etf_contest_spec import EXCLUDED_5
from scripts.portfolio.etf_rotation_panels import (
    build_defense,
    build_extra_panels,
    level_from_returns,
)
from scripts.portfolio.etf_rotation_probe import (
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)

BEGIN = 20200101
ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(20261029)

_BASE = dict(PROFILES["contest3"])
#: 待评档位。基线 ``contest3`` = 定版「三锋」（``etf_contest_spec.FINAL``）。
#: 🚨 buffer 的 enter/exit **必须写全**：45 只池里 15/40、10/30、5/20 的窄档位
#: 会退化成「还在清单就不卖」，三者结果逐位相同（README §七 #12 注）。
VARIANTS: dict[str, dict] = {
    "contest3": _BASE,
    "contest5h": dict(PROFILES["contest5h"]),
    "c3_buffer10": {**_BASE, "enter_rank": 5, "exit_rank": 10},
    "c3_buffer15_40": {**_BASE, "enter_rank": 15, "exit_rank": 40},
    "c3_buffer5_20": {**_BASE, "enter_rank": 5, "exit_rank": 20},
    "c3_buffer30": {**_BASE, "enter_rank": 5, "exit_rank": 30},
    "champ": dict(PROFILES["champ"]),
}
#: 带合成腿的档位：`champ`（磐石）有 DEF 防御腿。🚨 它的 open 口径必须把 DEF 补进
#: **执行价面板**，否则 `build_execution_split` 拿到的 fill 面板缺 DEF 列 ⇒ 该腿拿不到
#: 执行价、权重被吞，年化会从 13.2% 掉到 9.5%（2026-10-29 查明的复现失败根因）。
DEF_VARIANTS = {"champ"}


def two_month_windows(eq: pd.Series, mfs: pd.DatetimeIndex) -> pd.Series:
    """按月首切 t → t+2 月的窗口收益，索引 = 窗口起始日。"""
    out: dict[pd.Timestamp, float] = {}
    for i in range(len(mfs) - 2):
        t0, t2 = mfs[i], mfs[i + 2]
        if t0 not in eq.index or t2 not in eq.index:
            continue
        if len(eq.loc[t0:t2]) < 20:
            continue
        out[t0] = float(eq.loc[t2] / eq.loc[t0] - 1.0)
    return pd.Series(out)


def yearly(eq: pd.Series) -> pd.Series:
    r = eq.pct_change().fillna(0.0)
    return r.groupby(r.index.year).apply(lambda x: float((1.0 + x).prod() - 1.0))


def stats(ret: pd.Series) -> dict:
    return {"n": len(ret), "mean": float(ret.mean()), "median": float(ret.median()),
            "p10": float(ret.quantile(0.10)), "worst": float(ret.min()),
            "win_rate": float((ret > 0).mean())}


def paired(a: pd.Series, b: pd.Series) -> dict:
    """同窗口配对差 a − b（重叠窗口 ⇒ naive t 偏乐观，只作量级参考）。"""
    idx = a.index.intersection(b.index)
    d = (a[idx] - b[idx]).to_numpy()
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    t = mean / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    boot = np.array([RNG.choice(d, n, replace=True).mean() for _ in range(10000)])
    # 非重叠敏感性：每 2 窗取 1（相邻窗共享 1 个月）
    d2 = d[::2]
    t2 = float(d2.mean() / (d2.std(ddof=1) / np.sqrt(len(d2)))) if len(d2) > 2 else float("nan")
    return {"n": n, "mean": mean, "t": t, "t_nonoverlap": t2,
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
            "win_rate": float((d > 0).mean())}


def _vwap_panel(raw: pd.DataFrame, close: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """由 ``amount / volume`` 造 VWAP 执行价面板，与 ``close`` 同口径（后复权）。

    ETF 的 ``volume`` 单位是**份**、``amount`` 是**元** ⇒ 比值为当日成交均价（不复权），
    再乘后复权因子 ``close / close_raw``。``volume <= 0`` 的停牌/异常日置 NaN。

    ⚠️ 语义：这是"**次日全天按成交均价分批买入**"的算法单近似，隐含「能把全天成交量
    均匀吃到」⇒ 对 45 只日成交额均在亿级的池子是可行的，但仍是**上限估计**（真实
    VWAP 单有滑点、且我们无法预知当日 VWAP 形状）。
    """
    idx, cols = close.index, close.columns
    amt = raw["amount"].unstack("code").reindex(index=idx, columns=cols)
    vol = raw["volume"].unstack("code").reindex(index=idx, columns=cols)
    rc = raw["close"].unstack("code").reindex(index=idx, columns=cols)
    lo = raw["low"].unstack("code").reindex(index=idx, columns=cols)
    hi = raw["high"].unstack("code").reindex(index=idx, columns=cols)
    ratio = close / rc.where(rc > 0)
    vw = (amt / vol.where(vol > 0)) * ratio
    vw = vw.where(vw > 0)
    # 自检：VWAP 必须落在当日 [low, high] 内（价格 3 位小数，留 5bp 容差）
    ok = vw.notna() & lo.notna() & hi.notna()
    inband = (vw >= lo * ratio * 0.9995) & (vw <= hi * ratio * 1.0005)
    n_ok = int(ok.to_numpy().sum())
    # ⚠️ 布尔 DataFrame 索引 DataFrame 是**掩码选择**（仍是矩阵），median 会按列返回
    # Series ⇒ 必须先降成 numpy 一维再取分位。
    diff = (vw / close - 1.0).to_numpy()
    pick = ok.to_numpy() & np.isfinite(diff)
    d = diff[pick]
    diag = {
        "coverage": n_ok / max(int(close.notna().to_numpy().sum()), 1),
        "inband": int((inband & ok).to_numpy().sum()) / max(n_ok, 1),
        "rel_med": float(np.median(d)) if d.size else float("nan"),
        "rel_p10": float(np.percentile(d, 10)) if d.size else float("nan"),
        "rel_p90": float(np.percentile(d, 90)) if d.size else float("nan"),
    }
    return vw.astype(np.float32), diag


def exec_ladder(bundle, panels, spec: dict, close: pd.DataFrame,
                open_adj: pd.DataFrame, mfs: pd.DatetimeIndex,
                raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """成交价位的实测对照（README §9.1）。fill=None ⇒ T 日收盘成交。

    ① `T 日收盘` 是**不 shift** 的口径（收盘出信号、收盘成交），语义上等价于
    「信号与成交同一时点」，只作理论参照；②③④ 才是「T 收盘出信号 + 次日成交」。
    ``raw`` 非空时追加 ④ `T+1 VWAP`（同日同批信号，仅换执行价）。
    """
    r0 = run_variant(bundle, panels, spec, BEGIN, None)
    rb = set(pd.DatetimeIndex(r0["rebalance_dates"]))
    rows = []
    steps: list[tuple[str, pd.DataFrame | None]] = [
        ("① T 日收盘", None), ("② T+1 开盘", open_adj), ("③ T+1 收盘", close)]
    vwap_adj, diag = None, None
    if raw is not None:
        vwap_adj, diag = _vwap_panel(raw, close)
        # 执行日缺 VWAP 时回退开盘价（只影响停牌/极低量日，比例见下方自检）
        steps.append(("④ T+1 VWAP", vwap_adj.fillna(open_adj)))
    vw_res = None
    for tag, fill in steps:
        if fill is None:
            r = r0
        else:
            r = run_variant(bundle, panels, spec, BEGIN, None,
                            execution_split=build_execution_split(fill, close, rb),
                            execution_shift=True)
            if tag.startswith("④"):
                vw_res = r
        w = two_month_windows(r["equity"], mfs)
        rows.append({"caliber": tag, "ann": r["annual_return"], "sharpe": r["sharpe"],
                     "mean": float(w.mean()), "p10": float(w.quantile(0.10)),
                     "worst": float(w.min()), "mdd": r["max_drawdown"]})
    if vwap_adj is not None and vw_res is not None and diag is not None:
        w = vw_res["weights"]
        cols = [c for c in w.columns if c not in ("CASH", "DEF") and c in vwap_adj.columns]
        held = (w[cols].abs() > 1e-12).to_numpy()
        nan_ = vwap_adj[cols].isna().reindex(index=w.index, fill_value=True).to_numpy()
        n_held, n_bad = int(held.sum()), int((held & nan_).sum())
        print("\n== ④ VWAP 口径自检（不改信号，仅换执行价）==")
        print(f"覆盖率：close 有效样本上 {diag['coverage']:.2%} 有 VWAP；"
              f"落在当日 [low, high] 内 {diag['inband']:.2%}")
        print(f"VWAP / close − 1：中位 {diag['rel_med']:+.2%}  p10 {diag['rel_p10']:+.2%}"
              f"  p90 {diag['rel_p90']:+.2%}（<0 表示成交均价低于收盘）")
        print(f"持仓位（执行日 × 实际权重>0）缺 VWAP：{n_bad}/{n_held} = "
              f"{n_bad / max(n_held, 1):.2%}，已回退 T+1 开盘价")
    return pd.DataFrame(rows).set_index("caliber")


def _read_pool_file(path: str) -> set[str]:
    """读自定义池清单：每行一个代码，或首列为代码的 CSV（忽略表头/空行/注释）。"""
    keep: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        s = line.split(",")[0].strip()
        if not s or s.startswith("#") or s.lower() in ("code", "pool_code", "代码"):
            continue
        keep.add(s)
    return keep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ETF 轮动档位评测（80 个 2 月窗口）")
    ap.add_argument("--only", default=None, help="只跑指定档位（逗号分隔）")
    ap.add_argument("--csv", default=None, help="窗口收益落盘路径")
    ap.add_argument("--exec-ladder", action="store_true",
                    help="只跑 §9.1 成交价位（① T收盘 / ② T+1开盘 / ③ T+1收盘 / ④ T+1 VWAP）")
    ap.add_argument("--legacy-no-pit", action="store_true",
                    help="关掉权重层 PIT 修正，复现 2026-10-29 之前的历史数字（不外引用）")
    ap.add_argument("--keep-excluded", action="store_true",
                    help="不屏蔽 EXCLUDED_5（= 50 只「自有策略」口径）。"
                         "🚨 比赛对外口径必须关闭本开关；两口径不可混用")
    ap.add_argument("--pool", default=None,
                    help="自定义可投池清单（每行一个代码，或首列为代码的 CSV）；"
                         "池外代码全部 mask=False。🚨 仅用于池结构实验，非对外口径")
    args = ap.parse_args(argv)
    if args.legacy_no_pit:
        import scripts.portfolio.etf_rotation_probe as _RP
        _RP._pit_shift = lambda df: df
        print("⚠️ --legacy-no-pit：权重层 PIT 修正已关闭，结果仅为历史对照", flush=True)
    names = list(VARIANTS) if not args.only else [s.strip() for s in args.only.split(",")]

    cache = DataCache(create_datasource())
    bundle = load_bundle(cache, WARMUP_BEGIN, None)
    uni, close, amount, bench = bundle
    raw = cache.read_daily("etf")
    rc = raw["close"].unstack("code").reindex(index=close.index, columns=close.columns)
    ro = raw["open"].unstack("code").reindex(index=close.index, columns=close.columns)
    open_adj = (ro * (close / rc)).astype(np.float32)

    panels = build_extra_panels(close, build_panels(close, amount), rc)
    if not args.keep_excluded:
        for x in EXCLUDED_5:
            panels["mask"].loc[:, x] = False
    else:
        print(f"⚠️ --keep-excluded：{len(EXCLUDED_5)} 只非指定池标的已放回"
              f"（自有策略口径，**不是**比赛对外口径）", flush=True)
    if args.pool:
        keep = _read_pool_file(args.pool)
        unknown = sorted(keep - set(close.columns))
        drop = [c for c in close.columns if c not in keep]
        for x in drop:
            panels["mask"].loc[:, x] = False
        print(f"⚠️ --pool：自定义池 {len(keep)} 只，已 mask 掉池外 {len(drop)} 只"
              f"（**池结构实验口径，非对外数字**）", flush=True)
        if unknown:
            print(f"   清单里 {len(unknown)} 只不在面板中，已忽略：{unknown[:5]}", flush=True)
    mfs = pd.DatetimeIndex(
        pd.Series(close.index, index=close.index)
        .groupby(close.index.to_period("M")).first().to_numpy())
    # 合成防御腿（仅 champ 用）。DEF 是净值水平线，其「执行价」就是当日净值。
    extra_def = (pd.DataFrame({"DEF": level_from_returns(build_defense(close)["mix"])})
                 if any(n in DEF_VARIANTS for n in names) else None)
    close_def = close.join(extra_def) if extra_def is not None else close
    open_def = open_adj.join(extra_def) if extra_def is not None else open_adj

    if args.exec_ladder:
        d = exec_ladder(bundle, panels, VARIANTS[names[0]], close, open_adj, mfs, raw=raw)
        print("\n== 成交价位实测（§9.1，同一批信号）==")
        print(f"{'口径':<14}{'2月均值':>9}{'p10':>9}{'最差':>9}{'Sharpe':>9}{'年化':>9}{'MDD':>9}")
        for k, r in d.iterrows():
            print(f"{k:<14}{r['mean']:>+9.2%}{r['p10']:>+9.2%}{r['worst']:>+9.2%}"
                  f"{r['sharpe']:>9.2f}{r['ann']:>9.2%}{r['mdd']:>9.2%}")
        if args.csv:
            out = Path(args.csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            d.to_csv(out, encoding="utf-8-sig")
            print(f"\n落盘 → {out}")
        return 0

    rows, wins, anos, eqs = [], {}, {}, {}
    for name in names:
        spec = VARIANTS[name]
        ex = extra_def if name in DEF_VARIANTS else None
        cl = close_def if ex is not None else close
        od = open_def if ex is not None else open_adj
        r0 = run_variant(bundle, panels, spec, BEGIN, None, extra_close=ex)
        spl = build_execution_split(od, cl,
                                    set(pd.DatetimeIndex(r0["rebalance_dates"])))
        r1 = run_variant(bundle, panels, spec, BEGIN, None, extra_close=ex,
                         execution_split=spl, execution_shift=True)
        w = r1["weights"]
        live = w[w.abs().sum(axis=1) > 1e-9]
        held = [c for c in w.columns if c not in ("CASH", "DEF")]
        maxw = float(live[held].max(axis=1).median()) if len(live) else float("nan")
        wt = two_month_windows(r1["equity"], mfs)
        wins[name] = wt
        anos[name] = yearly(r1["equity"])
        eqs[name] = r1["equity"]
        rows.append({"variant": name,
                     "ann_close": r0["annual_return"], "ann_open": r1["annual_return"],
                     "sharpe_close": r0["sharpe"],
                     "sharpe": r1["sharpe"], "mdd": r1["max_drawdown"],
                     "turnover": r1["avg_turnover"], "maxw": maxw,
                     # ⚠️ 引擎的 cost_drag = 成本金额 / **初始**资金 / 年数。因为资本在
                     # 复利增长、成本按当期 capital 计提，这个比率会被放大（contest3
                     # 实测 7.2%，而按 `换手 × 双边 7bp` 只有 3.2%）⇒ **不能当"扣掉的
                     # 年化收益"读**，只在同口径横向比较时用。
                     "cost_init": r1["cost_drag"], "n_rb": r1["n_rebalance"],
                     **{f"w2m_{k}": v for k, v in stats(wt).items()}})
        print(f"[done] {name}", flush=True)

    d = pd.DataFrame(rows).set_index("variant")
    print("\n== 档位指标（T+1 open，80 个 2 月窗口）==")
    hdr = (f"{'档位':<16}{'年化':>8}{'Sharpe':>8}{'MDD':>8}{'均值':>8}{'中位':>8}"
           f"{'p10':>8}{'最差':>8}{'胜率':>7}{'换手':>7}{'最大单只':>9}{'成本/年*':>9}")
    print(hdr)
    print(f"{'':<16}{'':>8}{'':>8}{'':>8}{'':>8}{'':>8}{'':>8}{'':>8}{'':>7}{'':>7}"
          f"{'':>9}  * 占初始资金，非收益拖累")
    for k, r in d.iterrows():
        print(f"{k:<16}{r.ann_open:>8.2%}{r.sharpe:>8.2f}{r.mdd:>8.2%}"
              f"{r.w2m_mean:>+8.2%}{r.w2m_median:>+8.2%}{r.w2m_p10:>+8.2%}"
              f"{r.w2m_worst:>+8.2%}{r.w2m_win_rate:>7.1%}{r.turnover:>7.1%}"
              f"{r.maxw:>9.1%}{r.cost_init:>9.2%}")

    print("\n== close → open 执行税（同一变体两口径，符号可跨档翻转）==")
    print(f"{'档位':<16}{'close年化':>10}{'open年化':>10}{'差值':>9}{'closeSR':>9}{'openSR':>9}")
    for k, r in d.iterrows():
        print(f"{k:<16}{r.ann_close:>10.2%}{r.ann_open:>10.2%}"
              f"{r.ann_open - r.ann_close:>+9.2%}"
              f"{r.sharpe_close:>9.2f}{r.sharpe:>9.2f}")

    base = "contest3"
    if base in wins:
        print(f"\n== 对齐「{base}」的同窗口配对差（open 口径）==")
        print(f"{'档位':<16}{'Δ均值':>9}{'t':>7}{'非重叠t':>9}{'95%CI':>22}{'胜率':>8}")
        for k in wins:
            if k == base:
                continue
            s = paired(wins[k], wins[base])
            ci = f"[{s['ci_lo']:+.2%},{s['ci_hi']:+.2%}]"
            print(f"{k:<16}{s['mean']:>+9.2%}{s['t']:>7.2f}{s['t_nonoverlap']:>9.2f}"
                  f"{ci:>22}{s['win_rate']:>8.1%}")

    # 🚨 非重叠 2 月期 = 赛制真实计分单位。上面的 80 窗互相重叠，**最差窗口/波动/Sharpe
    # 都对切窗方式敏感**：曾据重叠窗的「单点最差窗口」判否 buffer，换非重叠后方向反转。
    # 判断档位优劣一律看本表；重叠窗表只用于配对检验（同窗对齐）与平滑分布。
    print("\n== 非重叠 2 月期（赛制真实粒度，判断主口径）==")
    per = pd.DataFrame({k: v.resample("2ME").last().pct_change() for k, v in eqs.items()}).dropna()
    print(f"{'档位':<16}{'均值':>9}{'中位':>9}{'波动':>8}{'期Sharpe':>9}"
          f"{'p10':>9}{'最差':>9}{'胜率':>8}")
    for k in per.columns:
        s = per[k]
        print(f"{k:<16}{s.mean():>+9.2%}{s.median():>+9.2%}{s.std():>8.2%}"
              f"{s.mean() / s.std():>9.2f}{s.quantile(0.10):>+9.2%}"
              f"{s.min():>+9.2%}{(s > 0).mean():>8.1%}")
    print("相关矩阵（非重叠）：")
    print(per.corr().round(2).to_string())

    print("\n== 分年度收益（open）==")
    print((pd.DataFrame(anos) * 100).round(1).to_string())

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(wins).to_csv(out, encoding="utf-8-sig")
        per_out = out.with_name(out.stem + "_periods" + out.suffix)
        per.to_csv(per_out, encoding="utf-8-sig")
        print(f"\n窗口收益已落盘：{out}")
        print(f"非重叠期收益已落盘：{per_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
