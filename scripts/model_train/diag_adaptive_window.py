"""「不固定窗口」到底能不能成立：三个前置前提的实测。

背景
----
RAFT 式检索的启发：不预设窗口长度，而是让数据决定「看多远」。落法有三条：

  A. 检索驱动 —— 相似片段的长度决定窗口（用户提的这条）
  B. 变点驱动 —— regime 起点决定窗口
  C. 多尺度并行 —— 不选，把所有尺度一起给模型学权重

三条都建立在一个**未被检验的前提**上：固定窗口（我们一直用 60 日）确实是瓶颈。
本脚本不重训 ANN，只做三个便宜的判别实验，因为只要第一个不成立，后面全无意义：

  实验 1  窗口敏感性曲线：固定窗口 L 在 5~250 之间，IC 是一条陡峭曲线还是平台？
          —— 平台 ⇒ 换窗口没有收益，「选窗口」这件事本身不值钱。
  实验 2  最优窗口是否可预判（这是「动态窗口」的生死线）：
          a) argmax_L 的稳定性（转移矩阵 / 与上一期的相关）
          b) 严格 PIT 的可实现策略：用过去 26 期 IC 选下一个 L，能否跑赢
             固定 L=60？（样本内 argmax 只作上界参考，不可实现）
          c) 按市场状态（波动分位）分组，看 L 偏好是否有稳定规律
          —— 若 b 跑输固定，说明 argmax 是噪声，动态选窗口是负和游戏。
  实验 3  多尺度并行 vs 最优单尺度：这是「不固定窗口」最便宜、最常见的实现，
          若它都赢不了单尺度，变长输入（贵得多）更没戏。

口径
----
行业指数（`sw_industry_daily`）直接算，不依赖个股面板——顺带绕开「行情只到 2015」
这个短板。周度调仓日、与主口径同；IC 用 `report_replica.ic_series`（当期因子 vs
下一调仓日收益，lag=0）。两族因子分别测，避免「哪个族更强」污染结论：
  · `ma_L`   = px / MA_L - 1        （水平偏离）
  · `ret_L`  = px / px.shift(L) - 1 （区间动量）

用法:
    python -m scripts.model_train.diag_adaptive_window
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("reports/report_replica")
LS = (5, 10, 20, 30, 40, 60, 80, 120, 160, 250)


def _nw_t(x: np.ndarray, lag: int = 4) -> float:
    """Newey-West t（周频重叠样本必须校正自相关）。"""
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    mu = x.mean()
    e = x - mu
    n = len(x)
    v = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(mu / np.sqrt(v / n))


def build_family(px: pd.DataFrame, kind: str, Ls=LS) -> dict[int, pd.DataFrame]:
    """按窗口长度 L 构造一族因子面板，键为 L。"""
    out: dict[int, pd.DataFrame] = {}
    for L in Ls:
        if kind == "ma":
            out[L] = px / px.rolling(L).mean() - 1
        elif kind == "ret":
            out[L] = px / px.shift(L) - 1
        else:
            raise ValueError(kind)
    return out


def ic_matrix(fam: dict[int, pd.DataFrame], dates, px: pd.DataFrame) -> pd.DataFrame:
    """(期 × L) 的 IC 矩阵：每期每个窗口长度各自的截面 IC。"""
    from scripts.portfolio.report_replica import ic_series

    tdays = px.index
    cols = {}
    for L, sig in fam.items():
        s = sig.reindex(index=tdays, columns=px.columns)
        cols[L] = ic_series(s, dates, px, tdays, lag=0)
    return pd.DataFrame(cols, index=pd.DatetimeIndex(dates[: len(next(iter(cols.values())))]))


def _zrow(df: pd.DataFrame, i: int) -> pd.Series:
    """取第 i 期的横截面并做 z-score（用于多尺度等权合成）。"""
    x = df.iloc[i].astype(float)
    x = x.clip(*np.percentile(x.dropna(), [1, 99])) if x.notna().sum() > 5 else x
    sd = x.std()
    return (x - x.mean()) / sd if sd and sd > 1e-12 else x * np.nan


def multiscale_signal(fam: dict[int, pd.DataFrame], Ls_pick) -> pd.DataFrame:
    """多尺度等权：各 L 的因子逐期 z-score 后平均，得到单一信号面板。"""
    idx = next(iter(fam.values())).index
    cols = next(iter(fam.values())).columns
    acc = np.zeros((len(idx), len(cols)))
    cnt = np.zeros((len(idx), 1))
    for L in Ls_pick:
        m = fam[L].to_numpy(float)
        mu = np.nanmean(m, axis=1, keepdims=True)
        sd = np.nanstd(m, axis=1, keepdims=True)
        z = (m - mu) / np.where(sd > 1e-12, sd, np.nan)
        ok = np.isfinite(z)
        acc += np.where(ok, z, 0.0)
        cnt += ok.sum(1, keepdims=True)
    with np.errstate(invalid="ignore"):
        out = np.where(cnt > 0, acc / cnt, np.nan)
    return pd.DataFrame(np.clip(out, -5, 5), index=idx, columns=cols)


def ic_of_panel(sig: pd.DataFrame, dates, px: pd.DataFrame) -> np.ndarray:
    from scripts.portfolio.report_replica import ic_series

    return ic_series(sig.reindex(index=px.index, columns=px.columns), dates, px, px.index, lag=0)


def report_sensitivity(icm: pd.DataFrame, tag: str) -> None:
    print(f"\n=== 实验 1 · {tag}：窗口敏感性（固定 L，2018 起周度）===")
    rows = []
    for L in icm.columns:
        x = icm[L].to_numpy()
        rows.append({"L": L, "IC": np.nanmean(x), "t_nw": _nw_t(x), "n": int(np.isfinite(x).sum())})
    d = pd.DataFrame(rows).set_index("L")
    best = d["IC"].idxmax()
    print(d.round(4).to_string())
    print(f"  → 最优 L={best}（IC {d.loc[best,'IC']:+.4f}）；"
          f"最优值 / 中位数 = {d.loc[best,'IC'] / d['IC'].median():.2f}x；"
          f"IC 极差 {d['IC'].max() - d['IC'].min():+.4f}")
    # 平台宽度：IC ≥ 最优值 90% 的连续区间
    ok = d["IC"] >= 0.9 * d.loc[best, "IC"]
    print(f"  → IC ≥ 最优 90% 的窗口长度：{list(d.index[ok])}")
    print(f"  → 固定 60 日 vs 最优：{d.loc[60,'IC']:+.4f} vs {d.loc[best,'IC']:+.4f}"
          f"（差 {(d.loc[best,'IC']-d.loc[60,'IC'])*100:+.2f}pt IC）")


def report_predictability(icm: pd.DataFrame, px: pd.DataFrame, dates, tag: str,
                          win: int = 26) -> pd.DataFrame:
    """实验 2：最优窗口能不能被预先选出来。"""
    print(f"\n=== 实验 2 · {tag}：最优窗口是否可预判 ===")
    X = icm.to_numpy()
    cols = list(icm.columns)
    best_all = int(np.nanargmax(np.nanmean(X, axis=0)))
    print(f"  样本内最优 L={cols[best_all]}（这是不可实现的上界，只作参考）")

    # a) 稳定性：每期 argmax 的游走 + 与上一期的相关
    am = pd.Series([cols[i] for i in np.nanargmax(X, axis=1)], index=icm.index)
    # 相邻两期 argmax 相同的比例，与「随机猜中」的基准 1/len(cols) 比
    same = (am.to_numpy()[1:] == am.to_numpy()[:-1]).mean()
    # 各 L 的 IC 序列一阶自相关（L 的「优劣」是否有惯性）
    ac1 = {L: icm[L].autocorr(1) for L in cols}
    print(f"  相邻期 argmax 相同比例 {same:.1%}（随机基准 {1/len(cols):.1%}）")
    print(f"  各 L 的 IC 一阶自相关：中位数 {np.nanmedian(list(ac1.values())):+.3f}"
          f"（{min(ac1.values()):+.3f} ~ {max(ac1.values()):+.3f}）")
    print(f"  argmax 分布：{am.value_counts().to_dict()}")

    # b) PIT 可实现策略：用过去 win 期均值 IC 选 L
    dyn = np.full(len(X), np.nan)
    pick = []
    for k in range(win, len(X)):
        past = np.nanmean(X[k - win:k], axis=0)     # 严格只用 k 之前
        j = int(np.nanargmax(past))
        pick.append(cols[j])
        dyn[k] = X[k, j]
    pk = pd.Series(pick)
    print(f"\n  PIT 动态选窗（回看 {win} 期）：IC {np.nanmean(dyn):+.4f} t_nw {_nw_t(dyn):+.2f}"
          f" n={int(np.isfinite(dyn).sum())}")
    print(f"    被选中的 L 分布：{pk.value_counts().to_dict()}")

    base = {
        f"固定 L={cols[best_all]}（样本内上界）": icm[cols[best_all]].to_numpy(),
        "固定 L=60（我们一直用的）": icm[60].to_numpy(),
        "固定 L=20（短窗对照）": icm[20].to_numpy(),
        "固定 L=120（长窗对照）": icm[120].to_numpy(),
        "全部 L 等权平均（多尺度基线）": np.nanmean(X, axis=1),
    }
    rows = []
    for name, x in base.items():
        rows.append({"scheme": name, "IC": np.nanmean(x), "t_nw": _nw_t(x)})
    rows.append({"scheme": f"PIT 动态选窗（回看 {win} 期）", "IC": np.nanmean(dyn), "t_nw": _nw_t(dyn)})
    d = pd.DataFrame(rows).set_index("scheme")
    print("\n  " + d.round(4).to_string().replace("\n", "\n  "))
    gap = d.loc[f"PIT 动态选窗（回看 {win} 期）", "IC"] - d.loc["固定 L=60（我们一直用的）", "IC"]
    print(f"  → 动态选窗 − 固定 60 日 = {gap*100:+.2f}pt IC")
    return d


def report_state_conditioned(icm: pd.DataFrame, px: pd.DataFrame, dates, tag: str) -> None:
    """实验 2c：按当日波动状态分组，看 L 偏好有没有稳定规律。"""
    print(f"\n=== 实验 2c · {tag}：按波动状态分组的 L 偏好 ===")
    lr = np.log(px).diff()
    vol = lr.rolling(20).std().mean(axis=1)          # 全行业平均波动
    pos = {d: i for i, d in enumerate(px.index)}
    v = np.array([vol.iloc[pos[d]] if d in pos else np.nan for d in icm.index])
    q = pd.qcut(pd.Series(v), [0, 1 / 3, 2 / 3, 1.0], labels=["低波", "中波", "高波"])
    out = icm.groupby(q.to_numpy()).mean()
    out.loc["全样本"] = icm.mean()
    print(out.round(4).to_string())
    best = {g: out.loc[g].idxmax() for g in out.index}
    print(f"  → 各组最优 L：{best}")
    # 高波组是否系统性偏好短窗
    short = [L for L in icm.columns if L <= 40]
    print(f"  → 短窗(L≤40)在高波组的 IC 均值 {out.loc['高波', short].mean():+.4f}"
          f" vs 低波组 {out.loc['低波', short].mean():+.4f}")


def vol_state(px: pd.DataFrame, dates, lookback: int = 250) -> pd.Series:
    """每期 t 的波动状态（严格 PIT）：当日全行业 20 日波动在**过去 lookback 日**的分位。

    注意不能对全样本做 qcut —— 那会用上未来的波动分布（评测期整体偏高的分位阈值）。
    """
    vol = np.log(px).diff().rolling(20).std().mean(axis=1)
    pos = {d: i for i, d in enumerate(px.index)}
    out = {}
    v = vol.to_numpy()
    for d in dates:
        i = pos.get(d)
        if i is None or i < lookback:
            out[d] = np.nan
            continue
        hist = v[i - lookback:i + 1]
        hist = hist[np.isfinite(hist)]
        out[d] = float((hist < v[i]).mean()) if len(hist) > 50 else np.nan
    return pd.Series(out)


def report_state_pit(icm: pd.DataFrame, px: pd.DataFrame, dates, tag: str,
                     lookback: int = 250) -> None:
    """实验 4：按**当期可观测**的波动状态选窗口——真正可实现的「动态窗口」。"""
    print(f"\n=== 实验 4 · {tag}：PIT 波动状态驱动的动态窗口（可实现）===")
    st = vol_state(px, list(icm.index), lookback).reindex(icm.index)
    cols = list(icm.columns)
    Ls = np.array(cols)
    short = [L for L in cols if L <= 40]
    long_ = [L for L in cols if L >= 120]

    # 状态规则来自实验 2c（低波偏好短窗、中波偏好长窗），用**当期**分位判定
    def pick(r):
        if not np.isfinite(r):
            return np.nan
        if r < 1 / 3:
            return cols.index(max(short, key=lambda L: icm[L].mean()))
        if r < 2 / 3:
            return cols.index(max(long_, key=lambda L: icm[L].mean()))
        return cols.index(max(short, key=lambda L: icm[L].mean()))

    j = np.array([pick(r) for r in st.to_numpy()], dtype=float)
    X = icm.to_numpy()
    dyn = np.array([X[k, int(j[k])] if np.isfinite(j[k]) else np.nan for k in range(len(X))])
    print(f"  分位缺失/不可判 {int((~np.isfinite(j)).sum())} 期；"
          f"状态分布 低波 {int((st<1/3).sum())} / 中波 {int(((st>=1/3)&(st<2/3)).sum())}"
          f" / 高波 {int((st>=2/3).sum())}")
    rows = [{"scheme": "PIT 状态选窗", "IC": np.nanmean(dyn), "t_nw": _nw_t(dyn)}]
    for L in (20, 60, 250):
        x = icm[L].to_numpy()
        rows.append({"scheme": f"固定 L={L}", "IC": np.nanmean(x), "t_nw": _nw_t(x)})
    # 高波期空仓（信号×0）：检验「状态择时」是否比「状态选窗」更值钱
    w = np.where(st.to_numpy() >= 2 / 3, 0.0, 1.0)
    g = np.where(np.isfinite(X[:, cols.index(60)]), X[:, cols.index(60)], np.nan) * w
    rows.append({"scheme": "固定 L=60 + 高波期空仓", "IC": np.nanmean(g), "t_nw": _nw_t(g)})
    d = pd.DataFrame(rows).set_index("scheme")
    print("  " + d.round(4).to_string().replace("\n", "\n  "))
    cmp60 = d.loc["PIT 状态选窗", "IC"] - d.loc["固定 L=60", "IC"]
    print(f"  → 状态选窗 − 固定 60 日 = {cmp60*100:+.2f}pt IC")


def main() -> None:
    ap = argparse.ArgumentParser(description="不固定窗口的三个前置检验")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--win", type=int, default=26, help="PIT 选窗的回看期数")
    args = ap.parse_args()

    from scripts.portfolio.report_replica import load_px, rebal_dates

    px = load_px()
    px = px[px.index >= "2013-01-01"]
    dates = rebal_dates(px.index, pd.Timestamp(args.start), "W")
    print(f"行业 {px.shape[1]} 个 · 面板 {px.index[0].date()}~{px.index[-1].date()}"
          f" · 调仓日 {len(dates)} 期（{args.start} 起周度）")

    for kind in ("ma", "ret"):
        fam = build_family(px, kind)
        icm = ic_matrix(fam, dates, px)
        tag = "MA 偏离" if kind == "ma" else "区间动量"
        report_sensitivity(icm, tag)
        report_predictability(icm, px, dates, tag, win=args.win)
        report_state_conditioned(icm, px, dates, tag)
        report_state_pit(icm, px, dates, tag)

        # 实验 3：多尺度并行 vs 最优单尺度
        print(f"\n=== 实验 3 · {tag}：多尺度并行 vs 最优单尺度 ===")
        for name, pick in (
            ("全尺度 10 腿", list(icm.columns)),
            ("三尺度 20/60/120", [20, 60, 120]),
            ("四尺度 20/60/120/250", [20, 60, 120, 250]),
        ):
            sig = multiscale_signal(fam, pick)
            x = ic_of_panel(sig, dates, px)
            print(f"  {name:24s} IC {np.nanmean(x):+.4f} t_nw {_nw_t(x):+.2f}"
                  f"   n={int(np.isfinite(x).sum())}")
        print(f"  {'固定最优单尺度':24s} IC {icm.mean().max():+.4f}"
              f" t_nw {_nw_t(icm[icm.mean().idxmax()].to_numpy()):+.2f}")

    # 落盘
    out = OUT_DIR / "diag_adaptive_window.csv"
    rows = []
    for kind in ("ma", "ret"):
        fam = build_family(px, kind)
        icm = ic_matrix(fam, dates, px)
        for L in icm.columns:
            x = icm[L].to_numpy()
            rows.append({"family": kind, "L": L, "IC": np.nanmean(x), "t_nw": _nw_t(x)})
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
