"""成交时点诊断：把行业因子的超额拆到「隔夜 / 日内 / 整周」三段上。

## 为什么需要这个脚本

研报 §2 的分组测试口径是「**行业指数当天收盘价成交、不考虑交易费用**」，隐含两个
不成立的前提：信号用 T 日 OHLCV 构造（T 日收盘后才算得出）⇒「T 日收盘价成交」
严格不可执行；零成本 ⇒ 周频调仓的实际摩擦被抹掉。

而 `report_replica.quintile_excess(lag=1)` 只测了「T+1 **收盘**成交」一种可执行写法，
它恰好是最差的写法（见 `--help` 下的口径表）。本脚本把成交时点拆开，给出完整相位曲线。

## 为什么用合成指数而不是申万官方指数

`sw_industry_daily.parquet` 只有收盘价、没有开盘价，**无法表达「开盘成交」**。
所以这里用个股 `open/close` 按 `build_context` 的同一套 PIT 行业归属 + 流通市值权重
合成行业价格。合成口径的水平与官方差 1~2 个点（t4 全样本 6.30% vs 7.63%），
**只能在脚本内部横向比较，不能与报告里的官方口径数字混用**。

## 口径定义（信号日 d，下一调仓日 d'）

| 代号 | 买入 | 卖出 | 可执行性 |
|---|---|---|---|
| `S0` | d 收盘 | d' 收盘 | 需 T 日尾盘下单（研报口径） |
| `S1` | d+1 收盘 | d'+1 收盘 | ✅ 收盘集合竞价 |
| `S2` | d+1 开盘 | d'+1 开盘 | ✅ 开盘集合竞价 |
| `ON` | d 收盘 | d+1 开盘 | 单夜（需尾盘买入） |
| `ID` | d+1 开盘 | d+1 收盘 | ✅ 单日日内 |

用法：
    python -m scripts.model_train.diag_exec_timing --pred TCN_4_day60_10d__s1full
    python -m scripts.model_train.diag_exec_timing --pred GRU_2_day60_10d__s1full --top 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PQ = Path("D:/data/parquet")
DL_DIR = Path("reports/report_replica/dl")
TOP_Q = 0.8          # Top20%
MIN_N = 20           # 一期至少多少个行业才计入


def _synth_panel(ctx: dict) -> dict[str, np.ndarray]:
    """按每期 PIT 权重合成行业价格：d 收盘 / d+1 开盘 / d+1 收盘。

    Returns:
        dict(pc, po1, pc1)，各 `(n_dates, n_inds)`。
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
    for i, dd in enumerate(dates):
        j, j1 = row[dd], min(row[dd] + 1, len(cl) - 1)
        w = ctx["wts"][i]
        lb = ctx["lbls"][i]
        w = np.where((lb >= 0) & (w > 0), w, 0.0)
        wo = w / w.sum()
        out["pc"].append(agg(cl, j, wo, lb))
        out["po1"].append(agg(op, j1, wo, lb))
        out["pc1"].append(agg(cl, j1, wo, lb))
    return {k: np.vstack(v) for k, v in out.items()}


def _excess(num: np.ndarray, den: np.ndarray, fac: np.ndarray,
            dates: pd.DatetimeIndex, inds: np.ndarray,
            *, shift: bool, top_n: int | None) -> tuple[pd.Series, pd.Series]:
    """逐期「头部组 − 全体平均」超额，返回 (超额序列, 每期成分)。

    `shift=True` 时收益取 `num[k+1]/den[k]-1`（跨期持有）；
    `shift=False` 取 `num[k]/den[k]-1`（同期单段）。
    """
    ex, picks = {}, {}
    last = len(dates) - 1
    for k in range(last):
        r = (num[k + 1] / den[k] - 1.0) if shift else (num[k] / den[k] - 1.0)
        f = fac[k]
        ok = np.isfinite(r) & np.isfinite(f)
        if ok.sum() < MIN_N:
            continue
        fv, rv = f[ok], r[ok]
        names = np.asarray(inds)[ok]
        if top_n is None:
            rk = pd.Series(fv).rank(method="average", pct=True).to_numpy()
            sel = rk > TOP_Q
        else:
            sel = np.zeros(len(fv), bool)
            sel[np.argsort(fv)[-top_n:]] = True
        if sel.sum() < 3:
            continue
        key = dates[k + 1] if shift else dates[k]
        ex[key] = float(rv[sel].mean() - rv.mean())
        picks[key] = set(names[sel])
    return pd.Series(ex).sort_index(), pd.Series(picks)


def _turnover(picks: pd.Series) -> float:
    """周均**单边**换手 = `0.5·(卖出比例 + 买入比例)`，与 `port_backtest` 的
    `0.5·Σ|Δw|` 同量纲。

    ⚠️ 早先用的是「集合对称差 / 并集」（`|A△B| / |A∪B|`），那是另一个口径：
    Top5 全换时它给 2.0、而单边换手是 1.0，直接乘 `2×bp` 会把成本高估约 25%。
    """
    ks = list(picks.index)
    t = []
    for k0, k1 in zip(ks, ks[1:]):
        a, b = picks[k0], picks[k1]        # 每期选中的行业名集合
        if not a or not b:
            continue
        sell = len(a - b) / len(a)
        buy = len(b - a) / len(b)
        t.append(0.5 * (sell + buy))
    return float(np.mean(t)) if t else np.nan


def _stat(s: pd.Series) -> tuple[float, float]:
    n = len(s)
    geo = (1 + s).prod() ** (52 / n) - 1
    return geo, s.mean() / (s.std(ddof=1) / np.sqrt(n))


def main() -> None:
    ap = argparse.ArgumentParser(description="成交时点诊断")
    ap.add_argument("--pred", default="TCN_4_day60_10d__s1full")
    ap.add_argument("--weight", default="float_mcap")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--top", type=int, default=None, help="固定 TopN 行业；默认 Top20%%")
    ap.add_argument("--cost-bp", type=float, nargs="+", default=[10.0, 5.0, 3.0],
                    help="单边成本 bp，可给多个")
    args = ap.parse_args()

    from scripts.model_train.eval_dl_industry import agg_one
    from scripts.portfolio.report_replica import build_ctx

    ctx = build_ctx("W", args.start, weight=args.weight)
    dates = pd.DatetimeIndex(ctx["dates"])
    inds = ctx["inds"]
    pred = pd.read_parquet(DL_DIR / f"pred_{args.pred}.parquet")
    fac = agg_one(pred, ctx).reindex(index=dates, columns=inds).to_numpy()
    grp = f"Top{args.top}" if args.top else "Top20%"

    print(f"预测 {args.pred} · 加权 {args.weight} · {len(dates)} 期 "
          f"({dates[0].date()}~{dates[-1].date()}) · 分组 {grp}")
    px = _synth_panel(ctx)

    print("\n=== 整周持有：三种成交时点 ===")
    arms = {
        "S0 d收盘→d'收盘（研报口径）": ("pc", "pc"),
        "S1 d+1收盘→d'+1收盘": ("pc1", "pc1"),
        "S2 d+1开盘→d'+1开盘（可执行）": ("po1", "po1"),
    }
    keep = {}
    for name, (nk, dk) in arms.items():
        s, picks = _excess(px[nk], px[dk], fac, dates, inds, shift=True, top_n=args.top)
        if len(s) < 10:
            continue
        geo, t = _stat(s)
        tr = _turnover(picks)
        print(f"  {grp} {name:30s} 年化 {geo*100:+7.2f}%  t {t:+6.2f}  n={len(s)}"
              f"  周单边换手 {tr:.2f}")
        # 净值必须**逐期扣成本再复利**：`geo − 成本年化` 会把波动拖累算两遍
        for bp in args.cost_bp:
            c = tr * 2 * bp / 1e4              # 每期成本（单边换手 × 买+卖）
            net = (1 + s - c).prod() ** (52 / len(s)) - 1
            print(f"  {'':<10}{'':30s} └ {bp:4.1f}bp/单边 · 每期成本 {c*1e4:5.2f}bp"
                  f"（{c*52*100:5.2f}%/yr）· 净 {net*100:+6.2f}%/yr")
        keep[name] = s

    print("\n=== 单期拆解（信号日 d 的因子，只持 1 天；期均为单次超额）===")
    for nk, dk, label in (("pc1", "pc", "d收→d+1收（1天，需尾盘）"),
                          ("po1", "pc", "隔夜 d收→d+1开"),
                          ("pc1", "po1", "日内 d+1开→d+1收（可执行）")):
        s, _ = _excess(px[nk], px[dk], fac, dates, inds, shift=False, top_n=args.top)
        if len(s) < 10:
            continue
        n = len(s)
        print(f"  {grp} {label:34s} 期均 {s.mean()*100:+.4f}%  t "
              f"{s.mean()/(s.std(ddof=1)/np.sqrt(n)):+6.2f}  n={n}  "
              f"折年（每周一次）{((1+s.mean())**52-1)*100:+6.2f}%")

    if keep:
        print("\n=== 分年（%）===")
        tab = pd.DataFrame({
            k: v.groupby(v.index.year).apply(lambda x: (1 + x).prod() ** (52 / len(x)) - 1) * 100
            for k, v in keep.items()})
        print(tab.round(1).to_string())


if __name__ == "__main__":
    main()
