"""长窗腿：把「固定窗口选错了长度」这件事量化，并分辨 IC 增益与组合层拟合。

由来
----
用户问「RAFT 那种检索相似片段的思路，能不能不固定窗口、让数据决定看多远」。
`diag_adaptive_window.py` 已把这个问题拆成三个前提并全部证否：

  1. 窗口敏感性 —— 最优是 **250 日**，我们一直用的 60 日在曲线谷底（IC 0.0010）。
  2. 最优窗口可否预判 —— **不可以**。相邻期 argmax 相同率 10.9%（随机基准 10.0%），
     各窗口 IC 的一阶自相关中位数 −0.06，PIT 动态选窗 IC −0.0102（比固定 60 日还差）。
  3. 多尺度并行 —— 输给最优单尺度（0.0095 vs 0.0175）。

⇒ 「让数据决定窗口」这条路走不通，但第 1 条指向一个便宜得多的替代：
   **不动模型，只把长窗作为独立一条腿加进去。**

本脚本验证这条腿，并且**先把「拟合」排除掉**再报数：
  · IC 的权重曲线（应平滑单峰；锯齿 = 拟合）
  · 长窗长度 L 的曲线（同理）
  · 分段稳定性（2018–21 / 2022–26）
  · 组合层 TopN 敏感性（Top5 的放大倍数是否只在 Top5 成立）
  · 连续 rank 分权（去掉 Top5 的离散选择）

口径：行业指数 → 与主口径同的周度调仓日；收益用合成面板 `synth_panel`（T+1 开盘，
S2 = 本期末 → 下期末）；IC 用 `ic_series(lag=0)`。

用法:
    python -m scripts.model_train.diag_long_window --pred TCN_4_day60_10d__s1full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("reports/report_replica")


def _ic(sig: pd.DataFrame, dates, px: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    from scripts.portfolio.report_replica import ic_series, stats

    s = ic_series(sig.reindex(index=px.index, columns=px.columns), list(dates), px, px.index, lag=0)
    st = stats(s)
    return st["ic"], st["t_nw"], s


def _linear_long(sig: pd.DataFrame, P: dict, dates) -> tuple[float, float, int]:
    """连续 rank 分权多头：权重 ∝ clip(rank_pct−0.5, 0)，无 TopN 的离散跳跃。"""
    rows = []
    arr = sig.to_numpy()
    for k in range(len(dates) - 1):
        r = P["po1"][k + 1] / P["po1"][k] - 1
        f = arr[k]
        ok = np.isfinite(r) & np.isfinite(f)
        if ok.sum() < 20:
            continue
        rk = pd.Series(f[ok]).rank(pct=True).to_numpy() - 0.5
        w = np.clip(rk, 0, None)
        w = w / w.sum()
        rows.append(float(w @ r[ok] - r[ok].mean()))
    s = pd.Series(rows)
    return float((1 + s).prod() ** (52 / len(s)) - 1), float(
        s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))), len(s)


def main() -> None:
    ap = argparse.ArgumentParser(description="长窗腿验证")
    ap.add_argument("--pred", default="TCN_4_day60_10d__s1full")
    ap.add_argument("--weight", default="float_mcap")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--L", type=int, default=250, help="长窗动量回望天数")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    from scripts.model_train.eval_dl_industry import agg_one, DL_DIR
    from scripts.model_train.diag_exec_timing import synth_panel, head_excess, weekly_turnover, geo_tstat
    from scripts.portfolio.report_replica import build_ctx, load_px, zrank

    px = load_px()
    ctx = build_ctx("W", args.start, weight=args.weight)
    dates = pd.DatetimeIndex(ctx["dates"])
    inds = list(ctx["inds"])
    dl = agg_one(pd.read_parquet(DL_DIR / f"pred_{args.pred}.parquet"), ctx)
    dl = dl.reindex(index=dates, columns=inds)
    mom = (px / px.shift(args.L) - 1).reindex(index=dates, columns=inds)
    P = synth_panel(ctx)

    print(f"预测 {args.pred} · 长窗 L={args.L} · {len(dates)} 期 · Top{args.top}")

    # 1) IC 的权重曲线 —— 判真信号 vs 拟合
    print("\n=== 1. IC 的权重曲线（应平滑单峰）===")
    best_w, best_ic = 1.0, -9.9
    for w in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        s = dl if w == 1.0 else w * zrank(dl) + (1 - w) * zrank(mom)
        i, t, _ = _ic(s, dates, px)
        star = ""
        if i > best_ic:
            best_ic, best_w = i, w
            star = " ←"
        print(f"  DL {w:.1f} + 长窗 {1-w:.1f}: IC {i:+.4f}  t_nw {t:+.2f}{star}")
    print(f"  → 峰值在 DL 权重 {best_w}（IC {best_ic:+.4f}）；单 DL 为 "
          f"{_ic(dl, dates, px)[0]:+.4f}（+{(best_ic-_ic(dl, dates, px)[0])*100:.2f}pt）")

    # 2) 长窗长度曲线
    print("\n=== 2. 长窗长度 L 的曲线（应呈宽峰，不要单点尖峰）===")
    for L in (120, 150, 200, 250, 300, 400):
        m = (px / px.shift(L) - 1).reindex(index=dates, columns=inds)
        i, t, _ = _ic(0.6 * zrank(dl) + 0.4 * zrank(m), dates, px)
        print(f"  L={L:<4d} IC {i:+.4f}  t_nw {t:+.2f}")

    # 3) 分段
    print("\n=== 3. 分段稳定性（合成 IC 与单腿 IC）===")
    comb = best_w * zrank(dl) + (1 - best_w) * zrank(mom)
    for tag, sig in (("DL 单独", dl), ("合成", comb), ("长窗单独", mom)):
        _, _, s = _ic(sig, dates, px)
        idx = dates[: len(s)]
        parts = []
        for lo, hi in ((2018, 2021), (2022, 2026)):
            x = s[(idx.year >= lo) & (idx.year <= hi)]
            x = x[np.isfinite(x)]
            parts.append(f"{lo}-{hi} IC {x.mean():+.4f} (t {x.mean()/(x.std(ddof=1)/np.sqrt(len(x))):+.2f})")
        print(f"  {tag:8s} " + "  ".join(parts))

    # 4) 组合层：TopN 敏感性 + 连续分权
    print(f"\n=== 4. 组合层（Top{args.top} 超额，毛年化；连续分权为对照）===")
    print(f"  {'方案':22s} {'Top5':>9} {'Top8':>9} {'Top10':>9} {'Top15':>9} {'换手':>6}")
    for label, sig in (("DL 单独", dl), (f"合成({best_w:.1f}/{1-best_w:.1f})", comb)):
        cells = []
        tr = np.nan
        for top in (5, 8, 10, 15):
            ex, picks = head_excess(P["po1"], P["po1"], sig.to_numpy(), dates, inds,
                                shift=True, top_n=top)
            g, _ = geo_tstat(ex)
            cells.append(f"{g*100:+8.2f}%")
            if top == 5:
                tr = weekly_turnover(picks)
        print(f"  {label:22s} " + " ".join(cells) + f" {tr:>6.3f}")

    print("\n  连续 rank 分权（无 TopN 跳跃）：")
    for label, sig in (("DL 单独", dl), (f"合成({best_w:.1f}/{1-best_w:.1f})", comb)):
        g, t, n = _linear_long(sig, P, dates)
        print(f"    {label:22s} 毛年化 {g*100:+6.2f}%  t {t:+.2f}  n={n}")

    # 5) 净收益（按峰值权重，逐期扣成本复利）
    print(f"\n=== 5. 净收益（合成 {best_w:.1f}/{1-best_w:.1f}，Top{args.top}，逐期扣成本）===")
    ex, picks = head_excess(P["po1"], P["po1"], comb.to_numpy(), dates, inds,
                        shift=True, top_n=args.top)
    tr = weekly_turnover(picks)
    g, t = geo_tstat(ex)
    print(f"  毛 {g*100:+.2f}%  t {t:+.2f}  单边换手 {tr:.3f}")
    for bp in (10, 5, 3):
        net = (1 + ex - tr * 2 * bp / 1e4).prod() ** (52 / len(ex)) - 1
        print(f"    @{bp}bp 净 {net*100:+.2f}%")
    print("\n  ⚠️ 判读纪律：若权重/L 曲线出现锯齿（非单峰）、或增益只在 Top5 成立而 Top8+ 消失，"
          "\n     则视为拟合，只可引用 IC 层的结论，不可引用组合层的年化。")

    rows = []
    for w in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        s = dl if w == 1.0 else w * zrank(dl) + (1 - w) * zrank(mom)
        i, t, _ = _ic(s, dates, px)
        rows.append({"dl_weight": w, "ic": i, "t_nw": t})
    out = OUT_DIR / "diag_long_window.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
