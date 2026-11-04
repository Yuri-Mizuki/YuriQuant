"""RAFT 思路的轻量原型：行业层 k-NN 检索预测（不做 MLP，只看「检索本身有没有信息」）。

研报 §2.1.2 的 RAFT = Retrieval-Augmented Forecasting of Time-series：预测前先从历史库
检索与当前状态最相似的若干片段，把**这些片段之后实际发生的结果**聚合为上下文，与当前
输入一起送入预测器——即「参照历史上相似情形后来怎样了」，而不是全靠参数记忆。

研报的实现：行业某日的 94 维技术因子 → 取 70 维算 Pearson 相似度 → Top-M 邻居 →
softmax 加权其未来 N 日收益 → 汇总 10 个统计量 → 与原因子拼成 104 维 → 两层 MLP。

本脚本只做**前半段**（检索 + softmax 加权），故意不接 MLP。理由：
  1. 若「检索」这一步本身就没有截面信息，接 MLP 也救不回来，先证伪最省事；
  2. MLP 融合需要调参，会把「检索有没有用」和「网络调得好不好」两件事混在一起。

与研报的已知差距（必须随结论一起引用）：
  · 因子维度：研报 94 维（含 LEVEL2），我们只有 close + amount ⇒ 约 28 维；
  · 相似度：研报 Pearson，我们用余弦（对标准化后等价）；
  · 无 MLP 融合层；
  · 检索池：研报「30 行业共享」，我们主口径跨行业共享、附同行业对照。

严格 PIT：t 日检索只用 `t' + h <= t` 的邻居（其未来收益已实现），且要求 `|t - t'| >= h`
排除时间重叠泄漏。

用法:
    python -m scripts.model_train.diag_raft_knn --h 5 --m 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SW_FILE = Path("D:/data/parquet/sw_industry_daily.parquet")
OUT_DIR = Path("reports/report_replica")
START = "2013-01-01"


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """行业收盘价与成交额面板（行=日期，列=行业代码）。"""
    s = pd.read_parquet(SW_FILE)
    s["date"] = pd.to_datetime(s["date"])
    s = s[s["date"] >= START]
    px = s.pivot_table(index="date", columns="code", values="close", aggfunc="last").sort_index()
    amt = s.pivot_table(index="date", columns="code", values="amount", aggfunc="last").sort_index()
    px, amt = px.align(amt, join="inner")
    return px.ffill(), amt.fillna(0.0)


def build_factors(px: pd.DataFrame, amt: pd.DataFrame) -> np.ndarray:
    """构造约 28 维技术因子 `(T, N, D)`，逐列做 250 日滚动 z-score（PIT）。"""
    lr = np.log(px).diff()
    feats: dict[str, pd.DataFrame] = {}

    # 动量：多尺度
    for k in (1, 5, 10, 20, 60, 120):
        feats[f"ret{k}"] = px.pct_change(k)
    # 波动：多尺度 + 波动比
    v20 = lr.rolling(20).std()
    v60 = lr.rolling(60).std()
    feats["vol20"] = v20
    feats["vol60"] = v60
    feats["volratio"] = v20 / v60.replace(0, np.nan)
    # 均线偏离
    for k in (5, 10, 20, 60, 120):
        feats[f"ma{k}"] = px / px.rolling(k).mean() - 1
    # 位置：距 N 日高低点
    for k in (20, 60):
        feats[f"hi{k}"] = px / px.rolling(k).max() - 1
        feats[f"lo{k}"] = px / px.rolling(k).min() - 1
        feats[f"pos{k}"] = ((px - px.rolling(k).min())
                            / (px.rolling(k).max() - px.rolling(k).min()).replace(0, np.nan))
    # 成交额：量比与量能变化
    a20 = amt.rolling(20).mean()
    a60 = amt.rolling(60).mean()
    feats["amt20"] = np.log(amt.replace(0, np.nan) / a20.replace(0, np.nan))
    feats["amt60"] = np.log(amt.replace(0, np.nan) / a60.replace(0, np.nan))
    feats["amt5_20"] = a20 / a60.replace(0, np.nan) - 1
    # 量价相关 / 偏度 / RSI
    feats["corr20"] = lr.rolling(20).corr(np.log(amt.replace(0, np.nan)).diff())
    feats["skew20"] = lr.rolling(20).skew()
    up = lr.clip(lower=0).rolling(14).mean()
    dn = (-lr.clip(upper=0)).rolling(14).mean()
    feats["rsi14"] = up / (up + dn).replace(0, np.nan)
    # 动量差（加速度）
    feats["accel"] = px.pct_change(20) - px.pct_change(60)

    cols = list(feats)
    stack = np.stack([feats[c].to_numpy(dtype=np.float64) for c in cols], axis=-1)  # (T,N,D)

    # 滚动 z-score：**必须按行业分别做**。若把 (T,N,D) 拉平成 (T*N,D) 再 rolling，
    # 窗口会跨行业边界（每 N 行换一个行业，250 行窗口横跨约 8 个行业），
    # 等于把「行业间水平差」混进「时序标准化」，相似度就退化成「找同类行业」。
    z = np.full_like(stack, np.nan)
    for j in range(stack.shape[1]):
        sub = pd.DataFrame(stack[:, j, :])
        mu = sub.rolling(250, min_periods=120).mean().to_numpy()
        sd = sub.rolling(250, min_periods=120).std().to_numpy()
        z[:, j, :] = (stack[:, j, :] - mu) / sd
    return np.clip(z, -5, 5)


def knn_predict(F: np.ndarray, fwd: np.ndarray, qidx: np.ndarray,
                h: int, M: int, tau: float, same_ind: bool) -> pd.DataFrame:
    """对每个查询日 t（qidx）做检索预测。

    Args:
        F: `(T, N, D)` 标准化因子，NaN 表示不可用。
        fwd: `(T, N)` 未来 h 日收益（`px[t+h]/px[t]-1`）。
        qidx: 查询日位置数组。
        M: 邻居数。
        tau: softmax 温度（越小越尖）。
        same_ind: True=只在同行业历史里检索；False=跨行业共享池。

    Returns:
        长表 `(date, ind, pred, hit_rate, spread)`。
    """
    T, N, D = F.shape
    rows = []
    # 单位化，便于余弦
    norm = np.linalg.norm(F, axis=2, keepdims=True)
    U = np.divide(F, norm, out=np.zeros_like(F), where=norm > 0)

    for t in qidx:
        # PIT：邻居的「未来 h 日收益」必须在 t 之前实现完 ⇒ t' <= t - h
        tmax = t - h
        if tmax < 250:
            continue

        # 先把 [250, tmax] 的整段池子摊平，之后每个查询行业都用**同一个池**
        # ⚠️ 不能按行业分别算相似度再拼接——那样每段用的查询向量不同（U[t,j]），
        # 拼接后不可比，跨行业口径会退化成「所有行业拿到同一个答案」。
        pool = U[250:tmax + 1].reshape(-1, D)              # (P*N, D)
        pfwd = fwd[250:tmax + 1].reshape(-1)               # (P*N,)
        pind = np.repeat(np.arange(N), tmax + 1 - 250)     # (P*N,)
        ok = np.isfinite(pool).all(1) & np.isfinite(pfwd)
        pool, pfwd, pind = pool[ok], pfwd[ok], pind[ok]
        if len(pool) < M:
            continue

        for j in range(N):
            if not np.isfinite(U[t, j]).all():
                continue
            sim = pool @ U[t, j]
            if same_ind:
                m = pind == j
                if m.sum() < M:
                    continue
                sj, rj = sim[m], pfwd[m]
            else:
                sj, rj = sim, pfwd
            top = np.argpartition(-sj, M)[:M]
            st, rt = sj[top], rj[top]
            w = np.exp((st - st.max()) / tau)
            w /= w.sum()
            rows.append((t, j, float(w @ rt), float((rt > 0).mean()),
                         float(np.quantile(rt, 0.8) - np.quantile(rt, 0.2))))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["t", "j", "pred", "hit_rate", "spread"])


def main() -> None:
    ap = argparse.ArgumentParser(description="RAFT 思路轻量原型")
    ap.add_argument("--h", type=int, default=5, help="预测标签天数")
    ap.add_argument("--m", type=int, default=50, help="邻居数 M")
    ap.add_argument("--tau", type=float, default=0.05, help="softmax 温度")
    ap.add_argument("--freq", default="W")
    args = ap.parse_args()

    from scripts.portfolio.report_replica import build_ctx, load_px, ic_series, stats
    from scripts.model_train.eval_dl_industry import agg_one, DL_DIR

    px, amt = load_panels()
    F = build_factors(px, amt)
    T, N, D = F.shape
    print(f"行业面板 {T} 日 × {N} 行业 × {D} 维因子（{px.index[0].date()}~{px.index[-1].date()}）")

    # 标签：未来 h 日收益（分母用 t 日收盘，与主口径一致）；尾部 h 行天然为 NaN
    fwd = px.shift(-args.h).to_numpy() / px.to_numpy() - 1

    # 查询日：用主口径的周度调仓日，保证可比
    ctx = build_ctx(args.freq, "2018-01-01", weight="float_mcap")
    dates = pd.DatetimeIndex(ctx["dates"])
    pos = {d: i for i, d in enumerate(px.index)}
    qidx = np.array([pos[d] for d in dates if d in pos], dtype=int)
    print(f"查询日 {len(qidx)} 个（周频，2018-01-01 起，与主口径同）")

    pxall = load_px()          # 只加载一次：这个文件 11M 行，循环里重复读会白等 10 秒
    tdays = pxall.index
    out = {}
    for same in (False, True):
        res = knn_predict(F, fwd, qidx, args.h, args.m, args.tau, same_ind=same)
        if res.empty:
            print("检索无结果")
            continue
        # 直接用 (t, j) 映射回日期与行业代码，**不再假设每个查询日都恰好输出 N 行**
        # （只要有行业因邻居不足被跳过，按行数整除重建就会整体错位，IC 会变成 NaN）
        fac = pd.DataFrame({
            "date": px.index[res["t"].to_numpy()],
            "ind": px.columns.to_numpy()[res["j"].to_numpy()],
            "pred": res["pred"].to_numpy(),
            "hit": res["hit_rate"].to_numpy(),
            "spread": res["spread"].to_numpy(),
        })
        name = "同行业检索" if same else "跨行业共享池"
        out[name] = fac
        n_date = fac["date"].nunique()
        print(f"\n=== {name}（h={args.h}, M={args.m}, τ={args.tau}）"
              f"→ {n_date} 期 × 平均 {len(fac)/max(n_date,1):.1f} 行业 ===")

        for col, label in (("pred", "检索预测值"), ("hit", "正收益占比"), ("spread", "分位差(8-2)")):
            sig = fac.pivot_table(index="date", columns="ind", values=col)
            sig = sig.reindex(index=pxall.index, columns=pxall.columns)
            ic = ic_series(sig, list(dates), pxall, tdays, lag=0)
            st = stats(ic)
            print(f"  {label:14s} 行业层 IC {st['ic']:+.4f}  t_nw {st['t_nw']:+.2f}  n={st['n']}")

    # 与现有 TCN 因子的 IC 相关（判断是否是「第三条腿」）
    if out:
        pred = pd.read_parquet(DL_DIR / "pred_TCN_4_day60_10d__s1full.parquet")
        tcn = agg_one(pred, ctx).reindex(index=pxall.index, columns=pxall.columns)
        ic_tcn = ic_series(tcn, list(dates), pxall, tdays, lag=0)
        print("\n=== 与 TCN(close 口径) 的 IC 序列相关（判是否正交）===")
        for name, fac in out.items():
            sig = fac.pivot_table(index="date", columns="ind", values="pred")
            sig = sig.reindex(index=pxall.index, columns=pxall.columns)
            ic_r = ic_series(sig, list(dates), pxall, tdays, lag=0)
            m = np.isfinite(ic_r) & np.isfinite(ic_tcn)
            c = np.corrcoef(ic_r[m], ic_tcn[m])[0, 1] if m.sum() > 20 else np.nan
            # 两腿等权合成的 IC
            z1 = sig.rank(axis=1, pct=True)
            z2 = tcn.rank(axis=1, pct=True)
            comb = ic_series((z1 + z2) / 2, list(dates), pxall, tdays, lag=0)
            st = stats(comb)
            print(f"  {name:10s} IC 相关 {c:+.3f}  | 与 TCN 合成 IC {st['ic']:+.4f} "
                  f"t_nw {st['t_nw']:+.2f}（TCN 单独 {stats(ic_tcn)['ic']:+.4f}）")

        all_fac = pd.concat(out.values())
        all_fac.to_csv(OUT_DIR / f"diag_raft_knn_h{args.h}.csv", index=False, encoding="utf-8-sig")
        print(f"\n→ {OUT_DIR / f'diag_raft_knn_h{args.h}.csv'}")


if __name__ == "__main__":
    main()
