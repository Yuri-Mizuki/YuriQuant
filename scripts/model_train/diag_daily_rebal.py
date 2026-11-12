"""日频调仓诊断：既然「alpha 只活一天」，那把调仓频率提到每天能不能收到钱？

## 为什么要做这个

`diag_exec_timing.py` 给出一条判断：信号日 d 的因子，超额集中在「d 收盘 → d+1 收盘」
这 1 天（期均 +0.1436%、t 3.62），持有整周（S2 口径）只剩 4.66%/年。于是自然要问：
**如果 alpha 只在 1 天内有效，为什么不每天调一次、每天只吃这 1 天？**

这个问题必须实测，不能靠推理，原因是成本随频率线性放大：

| 持有期 h | 毛超额（若每日 alpha 不变） | 年化成本 @10bp | 净值 |
|---|---|---|---|
| h=1（日频） | 50 倍于单日期均 | 2×10bp×252 = **50%/yr** | 取决于两者之差 |
| h=5（周频） | 1/5 | 10%/yr | — |
| h=10 | 1/10 | 5%/yr | — |

单日期均只要超过 2×bp（10bp ⇒ 0.20%/日），日频调仓就是划算的；低于它就被成本反噬。
**这是一个纯经验问题**，答案只能来自逐日预测面板（`pred_*.parquet` 覆盖 2851 天，
日频完全够用），而 `eval_dl_industry.py` 只做周频（`freq="W"`）。

## 与 `diag_exec_timing.py` 的口径关系

- 价格面板同源（个股 `open/close` 按 PIT 行业归属 + 流通市值权重合成，申万官方指数
  没有开盘价），**只能在面板内部横向比较**，不能与官方指数口径的数字混用。
- 新增两个东西：① **持有期权** h ∈ {1,2,3,5,10}（重叠建仓 = 稳态组合持有 h 个 vintage，
  每个 1/h 权重）；② **按权重变化的单边换手**（`0.5·Σ|w_t − w_{t−1}|`，与
  `report_replica.port_backtest` 逐字同义），不再用集合对称差，避免成本口径二次放大。

## 成本模型

每个 vintage 一次完整往返 = 卖旧 + 买新 = `2 × bp`；持有 h 天摊到每天 `2×bp/h`，
年化 `2×bp×252/h`。另外单独报「按实际换手算」的成本（日频信号有日内漂移，
换手率不一定等于 1.0）。

用法：
    python -m scripts.model_train.diag_daily_rebal --pred TCN_4_day60_10d__s1full
    python -m scripts.model_train.diag_daily_rebal --pred TCN_4_day60_10d__s1full --top 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from stats.significance import mean_t

PQ = Path(Config.paths()["cache_root"])
DL_DIR = Path("reports/report_replica/dl")
OUT_DIR = Path("reports/report_replica")
TD = 252            # 年化交易日
MIN_N = 20          # 一期至少多少个行业才计入


# --------------------------------------------------------------- 价格面板
def synth_panel(ctx: dict) -> dict[str, np.ndarray]:
    """按每期 PIT 权重合成行业价格：`pc`(d 收盘) / `po1`(d+1 开) / `pc1`(d+1 收)。

    与 `diag_exec_timing._synth_panel` 同一算法，这里复刻是为了让本脚本可以
    独立指定 `freq="D"`（上游那个默认周频、且不带日频选项）。
    """
    dates, uni, inds = pd.DatetimeIndex(ctx["dates"]), ctx["universe"], ctx["inds"]
    n_ind = len(inds)

    d = pd.read_parquet(PQ / "daily_all_a.parquet", columns=["open", "close"])
    d = d[~d.index.duplicated(keep="last")]
    op, cl = d["open"].unstack("code"), d["close"].unstack("code")
    row = {t: i for i, t in enumerate(cl.index)}

    def agg(col: pd.DataFrame, j: int, wo: np.ndarray, lb: np.ndarray) -> np.ndarray:
        px = col.iloc[j].reindex(uni).to_numpy(float)
        ok = np.isfinite(px) & (wo > 0)
        if not ok.any():
            return np.full(n_ind, np.nan)
        num = np.bincount(lb[ok], weights=px[ok] * wo[ok], minlength=n_ind)
        den = np.bincount(lb[ok], weights=wo[ok], minlength=n_ind)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    out: dict[str, list[np.ndarray]] = {"pc": [], "po1": [], "pc1": []}
    last_row = len(cl) - 1
    for i, dd in enumerate(dates):
        j = row[dd]
        j1 = min(j + 1, last_row)
        w = ctx["wts"][i]
        lb = ctx["lbls"][i]
        w = np.where((lb >= 0) & (w > 0), w, 0.0)
        wo = w / w.sum()
        out["pc"].append(agg(cl, j, wo, lb))
        out["po1"].append(agg(op, j1, wo, lb))
        out["pc1"].append(agg(cl, j1, wo, lb))
    return {k: np.vstack(v) for k, v in out.items()}


# --------------------------------------------------------------- 超额序列
def tranches(px: dict, fac: np.ndarray, dates: pd.DatetimeIndex, inds,
             kind: str, hold: int, top: int | None,
             step: int = 1) -> tuple[pd.Series, pd.Series]:
    """逐个建仓 vintage 的「头部组 − 全体」超额，返回 (超额序列, 单边换手序列)。

    Args:
        kind: `"oo"`=开盘买→开盘卖（可执行）、`"cc"`=收盘买→收盘卖（研报口径）、
            `"id"`=开盘买→当日收盘卖（纯日内，只能 h=1）。
        hold: 持有交易日数。
        step: 每隔 `step` 个交易日建一次仓（`step=1` 日频、`step=5` 约周频）。
    """
    ent, ext = {"oo": ("po1", "po1"), "cc": ("pc", "pc"), "id": ("po1", "pc1")}[kind]
    ex, turn, prev = {}, {}, None
    for k in range(0, len(dates) - hold - 1, step):
        # po1[i] = dates[i] 后一交易日的开盘价、pc[i] = dates[i] 收盘价 ⇒ 入场都用 i=k
        p0, p1 = px[ent][k], px[ext][k + hold]
        r = p1 / p0 - 1.0
        f = fac[k]
        ok = np.isfinite(r) & np.isfinite(f)
        if ok.sum() < MIN_N:
            continue
        fv, rv = f[ok], r[ok]
        names = np.asarray(inds)[ok]
        if top is None:
            pct = pd.Series(fv).rank(method="average", pct=True).to_numpy()
            sel = pct > 0.8
        else:
            sel = np.zeros(len(fv), bool)
            sel[np.argsort(fv)[-top:]] = True
        if sel.sum() < 3:
            continue
        sub = pd.Series(0.0, index=inds)
        sub.loc[names[sel]] = 1.0 / sel.sum()
        t = np.nan if prev is None else float((sub - prev).abs().sum() / 2)
        prev = sub
        ex[dates[k + hold]] = float(rv[sel].mean() - rv.mean())
        turn[dates[k + hold]] = t
    return pd.Series(ex).sort_index(), pd.Series(turn).sort_index()


def nw_t(s: pd.Series, lag: int) -> float:
    """Newey-West t：重叠持有 h 天会让朴素 t 高估约 √h 倍。"""
    x = s.to_numpy(float)
    n = len(x)
    if n < 10:
        return np.nan
    mu = x.mean()
    d = x - mu
    g0 = float(d @ d) / n
    var = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(d[k:] @ d[:-k]) / n
        var += 2 * (1 - k / (lag + 1)) * gk
    if var <= 0:
        return np.nan
    return mu / np.sqrt(var / n)


def _ann(mean_per: float, hold: int) -> float:
    """单期均值（持有 `hold` 个交易日、一次完整往返）→ 组合年化。

    `mean_per` 是**已扣成本**的期均（算术），日化后按 252 复利。
    """
    if not np.isfinite(mean_per) or (1 + mean_per) <= 0:
        return np.nan
    return (1 + mean_per) ** (TD / hold) - 1


def weekday_breakdown(s: pd.Series, dates: pd.DatetimeIndex,
                      hold: int) -> pd.DataFrame:
    """按【建仓信号日】星期几拆期均超额 —— 检验收益是否只挂在某一天。

    卖出日 = 建仓日 + hold 个交易日，所以建仓日要用位置回推，不能直接减日历日。
    """
    pos = {d: i for i, d in enumerate(dates)}
    # `sig` 与 `s.index` 是一条记录的两种日期（信号日 vs 卖出日），必须**按位置配对**；
    # 早先写成 `s.reindex(sig)` 会按标签重新取值，把每条超额错配到前一天的信号日上。
    sig = pd.DatetimeIndex([dates[pos[t] - hold] for t in s.index])
    g = pd.DataFrame({"ex": s.to_numpy(), "wd": sig.dayofweek})
    r = g.groupby("wd")["ex"].agg(["mean", "count", "std"])
    r["mean"] *= 1e4
    r["t"] = r["mean"] / 1e4 / (r["std"] / np.sqrt(r["count"]))
    return r[["mean", "count", "t"]]


def decompose_week(px: dict, fac: np.ndarray, dates: pd.DatetimeIndex,
                   top: int | None) -> pd.DataFrame:
    """周频 S2（T+1 开 → 下次 T+1 开）拆成「首日日内 + 后 4 天」。

    上一版把「1 天（T 收→T+1 收）的 7.75%」直接减「S2 的 5.51%」当作「后 4 天」，
    两个窗口**不嵌套**（前者含隔夜、后者不含），所以那个 −1.2% 是错的。
    """
    rows = []
    for k in range(len(dates) - 1):
        tot = px["po1"][k + 1] / px["po1"][k] - 1
        d1 = px["pc1"][k] / px["po1"][k] - 1
        m = np.isfinite(tot) & np.isfinite(fac[k])
        if m.sum() < MIN_N:
            continue
        fv = fac[k][m]
        pick = np.argsort(fv)[-top:]                 # 头部组的**位置下标**
        ex_t = np.nanmean(tot[m][pick]) - np.nanmean(tot[m])
        ex_1 = np.nanmean(d1[m][pick]) - np.nanmean(d1[m])
        rows.append((dates[k], ex_t * 1e4, ex_1 * 1e4))
    d = pd.DataFrame(rows, columns=["sig", "week", "day1"]).set_index("sig")
    d["rest"] = ((1 + d["week"] / 1e4) / (1 + d["day1"] / 1e4) - 1) * 1e4
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description="日频调仓诊断")
    ap.add_argument("--pred", default="TCN_4_day60_10d__s1full")
    ap.add_argument("--weight", default="float_mcap")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--top", type=int, default=5, help="固定 TopN 行业；默认 5")
    ap.add_argument("--cost-bp", type=float, nargs="+", default=[10.0, 5.0, 3.0])
    args = ap.parse_args()

    from scripts.model_train.eval_dl_industry import agg_one
    from scripts.portfolio.report_replica import build_ctx

    pred = pd.read_parquet(DL_DIR / f"pred_{args.pred}.parquet")

    ctx = build_ctx("D", args.start, weight=args.weight)     # ← 日频上下文
    dates = pd.DatetimeIndex(ctx["dates"])
    inds = ctx["inds"]
    fac = agg_one(pred, ctx).reindex(index=dates, columns=inds).to_numpy()
    valid = np.isfinite(fac).sum(1) >= MIN_N
    print(f"预测 {args.pred} · 加权 {args.weight} · 日频 {len(dates)} 天 "
          f"({dates[0].date()}~{dates[-1].date()}) · 有效信号 {valid.sum()} 天 · "
          f"Top{args.top}")

    px = synth_panel(ctx)
    rows = []
    arms = [
        ("日频 h=1 开→次日开（全可执行）", "oo", 1, 1),
        ("日频 h=1 开→当日收（纯日内）", "id", 1, 1),
        ("日频 h=1 收→次日收（研报口径）", "cc", 1, 1),
        ("日频 h=2 开→开", "oo", 2, 1),
        ("日频 h=3 开→开", "oo", 3, 1),
        ("日频 h=5 开→开", "oo", 5, 1),
        ("日频 h=10 开→开", "oo", 10, 1),
    ]
    print("\n=== 各持有期（期均 = 一个 vintage 的「TopN − 全体」超额）===")
    for label, kind, hold, step in arms:
        s, tu = tranches(px, fac, dates, inds, kind, hold, args.top, step=step)
        if len(s) < 30:
            print(f"  {label} 样本不足 n={len(s)}")
            continue
        tv = tu.dropna()
        line = (f"  {label:26s} n={len(s):4d} 期均 {s.mean()*1e4:+6.2f}bp "
                f"t_NW {nw_t(s, max(1, hold - 1)):+5.2f} "
                f"毛年化 {_ann(s.mean(), hold)*100:+7.2f}% 单边换手 {tv.mean():.2f}")
        print(line)
        for bp in args.cost_bp:
            net = s.mean() - 2 * bp / 1e4            # 一次完整往返 = 买+卖
            rows.append({
                "arm": label, "kind": kind, "hold": hold,
                "n": len(s), "period_mean_bp": s.mean() * 1e4,
                "t_nw": nw_t(s, max(1, hold - 1)),
                "gross_ann": _ann(s.mean(), hold),
                "cost_bp": bp, "net_ann": _ann(net, hold),
                "turn_single": tv.mean() if len(tv) else np.nan,
            })

    df = pd.DataFrame(rows)
    print("\n=== 扣成本后年化（%，一次往返 = 2×bp，不看换手）===")
    piv = df.pivot_table(index="arm", columns="cost_bp", values="net_ann") * 100
    print(piv.round(2).to_string())

    print("\n=== 按建仓信号日星期几（0=周一）===")
    for hold in (1, 5):
        s, _ = tranches(px, fac, dates, inds, "oo", hold, args.top, step=1)
        r = weekday_breakdown(s, dates, hold)
        print(f"  h={hold} 开→开：")
        print("   " + r.round(2).to_string().replace("\n", "\n   "))

    print("\n=== 周频 S2 拆解：首日日内 vs 后 4 天 ===")
    ctxw = build_ctx("W", args.start, weight=args.weight)
    datesw = pd.DatetimeIndex(ctxw["dates"])
    facw = agg_one(pred, ctxw).reindex(index=datesw, columns=ctxw["inds"]).to_numpy()
    d = decompose_week(synth_panel(ctxw), facw, datesw, args.top)
    for c, name in (("week", "整周 S2"), ("day1", "首日 T+1开→T+1收"),
                    ("rest", "后 4 天")):
        x = d[c] / 1e4
        t = mean_t(x)
        print(f"  {name:16s} 期均 {x.mean()*1e4:+6.2f}bp t {t:+5.2f} "
              f"折年 {((1+x.mean())**52-1)*100:+6.2f}%  n={len(x)}")

    # 周频 S2 的净值：只交易「换掉的那部分」，成本 = 2×bp×单边换手
    s5, tu5 = tranches(synth_panel(ctxw), facw, datesw, ctxw["inds"],
                       "oo", 1, args.top, step=1)
    tr = tu5.dropna().mean()
    print(f"\n=== 周频 S2（T+1 开→下次 T+1 开）净值 ===")
    print(f"  期均 {s5.mean()*1e4:+.2f}bp · 单边换手 {tr:.2f} · n={len(s5)}")
    for bp in args.cost_bp:
        net = s5.mean() - 2 * bp / 1e4 * tr
        print(f"  单边 {bp:4.1f}bp ⇒ 每期成本 {2*bp*tr:5.2f}bp ⇒ 净 {net*1e4:+6.2f}bp/周 "
              f"⇒ 年化 {(1+net)**52-1:+.2%}")

    tag = f"_{args.weight}" if args.weight != "float_mcap" else ""
    out = OUT_DIR / f"diag_daily_rebal{tag}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
