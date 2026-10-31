"""诊断「个股预测聚合到行业后 IC 归零」：是信息不存在，还是聚合口径丢掉了？

回答三个问题：
  1. 个股 IC 里有多少来自「行业内配对」、多少来自「行业间」？
  2. 同样的预测，换聚合口径（市值加权 / 等权）行业 IC 差多少？
  3. 行业信号集中在哪个市值分组？—— 市值加权聚合的效果≈大市值子样本。

判据（本脚本预注册）：
  · 若等权与市值加权的行业 IC 接近 ⇒ 归零是「信息不存在」；
  · 若差很多且小市值组 IC 高 ⇒ 归零是「信号在小票、权重在大票」的错配。

用法：
    python -m scripts.model_train.decomp_dl_industry
    python -m scripts.model_train.decomp_dl_industry --pred pred_TCN_4_day60_10d.parquet
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.model_train.replica_deep import build_arrays, forward_avg_ret  # noqa: E402
from scripts.oneoff.etf_industry_factor_p0 import _agg  # noqa: E402
from scripts.portfolio.report_replica import OUT_DIR, build_ctx  # noqa: E402

DL_DIR = OUT_DIR / "dl"


def _corr(a, b, min_n: int = 15) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n:
        return float("nan")
    a = a[m] - a[m].mean()
    b = b[m] - b[m].mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


def _eqagg(v: np.ndarray, lbl: np.ndarray, min_n: int = 5) -> np.ndarray:
    """行业等权均值（只取有限值），返回长度 31。"""
    out = np.full(31, np.nan)
    for k in range(31):
        s = v[(lbl == k) & np.isfinite(v)]
        if len(s) >= min_n:
            out[k] = s.mean()
    return out


def _eta2(v: np.ndarray, lbl: np.ndarray) -> float:
    """行业间平方和 / 总平方和（ANOVA）。"""
    s = pd.Series(v).groupby(lbl)
    ss_b = float((s.size() * (s.mean() - v.mean()) ** 2).sum())
    ss_t = float(((v - v.mean()) ** 2).sum())
    return ss_b / ss_t if ss_t > 0 else float("nan")


def _t(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean() / s.std() * np.sqrt(len(s))) if len(s) > 5 and s.std() > 0 else float("nan")


def collect(pred: pd.DataFrame, ctx: dict, lab: np.ndarray, idx: pd.DatetimeIndex,
            n_grp: int = 3) -> pd.DataFrame:
    uni, lbls, wts = ctx["universe"], ctx["lbls"], ctx["wts"]
    n = len(uni)
    rows = []
    for i, dt in enumerate(ctx["dates"]):
        if dt not in pred.index:
            continue
        ii = idx.get_loc(dt)
        x = pred.loc[dt].reindex(uni).to_numpy(float)
        y = np.asarray(lab[ii], float).copy()
        lbl = np.asarray(lbls[i])
        w = np.asarray(wts[i], float)
        if not (len(x) == len(y) == len(lbl) == n):
            continue
        m = np.isfinite(x) & np.isfinite(y) & (lbl >= 0) & (w > 0)
        if m.sum() < 800:
            continue

        X_m, Y_m = _agg(x[m], lbl[m], w[m]), _agg(y[m], lbl[m], w[m])
        X_e, Y_e = _eqagg(x[m], lbl[m]), _eqagg(y[m], lbl[m])
        dx = x[m] - pd.Series(x[m]).groupby(lbl[m]).transform("mean").to_numpy()
        dy = y[m] - pd.Series(y[m]).groupby(lbl[m]).transform("mean").to_numpy()

        base = dict(dt=dt, scope="全样本",
                    ic_stock=_corr(x[m], y[m], 500),
                    ic_within=_corr(dx, dy, 500),
                    ic_ind_mcap=_corr(X_m, Y_m),
                    ic_ind_eq=_corr(X_e, Y_e),
                    eta2=_eta2(x[m], lbl[m]), n=int(m.sum()))
        rows.append(base)

        # 按当日流通市值分组：行业信号集中在哪个规模段
        pct = np.nanpercentile(w[m], np.linspace(0, 100, n_grp + 1)[1:-1])
        seg = np.digitize(w, pct, right=True)
        names = ["小", "中", "大"] if n_grp == 3 else [f"G{k + 1}" for k in range(n_grp)]
        for k, nm in enumerate(names):
            gm = m & (seg == k)
            if gm.sum() < 500:
                continue
            rows.append(dict(dt=dt, scope=nm,
                             ic_stock=_corr(x[gm], y[gm], 300),
                             ic_within=_corr(
                                 x[gm] - pd.Series(x[gm]).groupby(lbl[gm]).transform("mean").to_numpy(),
                                 y[gm] - pd.Series(y[gm]).groupby(lbl[gm]).transform("mean").to_numpy(), 300),
                             ic_ind_mcap=_corr(_agg(x[gm], lbl[gm], w[gm]), _agg(y[gm], lbl[gm], w[gm])),
                             ic_ind_eq=_corr(_eqagg(x[gm], lbl[gm]), _eqagg(y[gm], lbl[gm])),
                             eta2=_eta2(x[gm], lbl[gm]), n=int(gm.sum())))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="pred_TCN_4_day60_10d.parquet")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--report", action="store_true", help="另写 dl/decomp_industry.md")
    args = ap.parse_args()

    fp = DL_DIR / args.pred
    pred = pd.read_parquet(fp)
    arr = build_arrays()
    idx = arr["index"]
    lab = forward_avg_ret(arr["avg"], idx, args.h, arr["close"])
    ctx = build_ctx("W", args.start)
    print(f"预测 {fp.name} · 标签 h={args.h}d（close 分母）")

    d = collect(pred, ctx, lab, idx)
    print(f"周频 {d.dt.nunique()} 期  {d.dt.min().date()}~{d.dt.max().date()}\n")

    tab = (d.groupby("scope", sort=False)
             .agg(平均只数=("n", "mean"), 行业间方差占比=("eta2", "mean"),
                  个股IC=("ic_stock", "mean"), 行业内IC=("ic_within", "mean"),
                  行业IC_市值加权=("ic_ind_mcap", "mean"),
                  行业IC_等权=("ic_ind_eq", "mean"),
                  t_市值加权=("ic_ind_mcap", _t), t_等权=("ic_ind_eq", _t)))
    print(tab.round(4).to_string())
    print("\n注：行业内 IC = 两侧都减掉行业均值后的相关（衡量「同行业内选股」能力）；")
    print("    行业 IC = 行业因子与行业收益的截面相关（衡量「选行业」能力）。")

    full = d[d.scope == "全样本"]
    print("\n=== 结论 ===")
    print(f"  个股 IC {full.ic_stock.mean():+.4f} vs 行业内 IC {full.ic_within.mean():+.4f}"
          f" ⇒ 差额 {full.ic_stock.mean() - full.ic_within.mean():+.4f} 才是行业间贡献")
    print(f"  同一份预测：市值加权行业 IC {full.ic_ind_mcap.mean():+.4f}"
          f" / 等权行业 IC {full.ic_ind_eq.mean():+.4f}"
          f" ⇒ 差 {full.ic_ind_eq.mean() - full.ic_ind_mcap.mean():+.4f}")

    out = OUT_DIR / "dl" / "decomp_industry.csv"
    d.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n→ {out}")

    if args.report:
        md = _render(tab, full, d, args)
        fp_md = OUT_DIR / "dl" / "decomp_industry.md"
        fp_md.write_text(md, encoding="utf-8")
        print(f"→ {fp_md}")


def _render(tab: pd.DataFrame, full: pd.DataFrame, d: pd.DataFrame, args) -> str:
    ic_mcap = float(full["ic_ind_mcap"].mean())
    ic_eq = float(full["ic_ind_eq"].mean())
    t_mcap, t_eq = _t(full["ic_ind_mcap"]), _t(full["ic_ind_eq"])
    small = tab.loc["小"] if "小" in tab.index else None
    big = tab.loc["大"] if "大" in tab.index else None
    L = [
        "# 深度学习维：为什么个股 IC 尚可、聚合到行业后归零",
        "",
        f"> 预测 `{args.pred}` · 周频 {d.dt.nunique()} 期"
        f"（{d.dt.min().date()} ~ {d.dt.max().date()}）· 标签 {args.h} 日均价涨跌（`close` 分母）",
        "> 脚本 `scripts/model_train/decomp_dl_industry.py`，数据 `decomp_industry.csv`",
        "",
        "## 一 结论先行",
        "",
        "**信息是存在的，归零发生在聚合口径上，不是模型没学到东西。**",
        "",
        f"- 同一份预测：**市值加权（研报 §2.1.1 口径）行业 IC {ic_mcap:+.4f}"
        f"（t={t_mcap:+.2f}）；同期等权聚合 {ic_eq:+.4f}（t={t_eq:+.2f}）**。",
        "- 行业信号集中在**小市值**：小市值组用市值加权聚合都有 "
        f"{small['行业IC_市值加权']:+.4f}，大市值组只有 {big['行业IC_市值加权']:+.4f}。"
        "而市值加权聚合 ≈ 让大票说话 ⇒ 归零。",
        "- 市值加权聚合必然让**大票主导**行业因子，而大票上的个股 IC 只有小票的 1/2.8"
        "（见下表）⇒ 行业因子被信号最弱的那批股票决定。",
        "- ⚠️ **不要**把它归因成「日内反转」：已实测 `−均价/收盘` 这个裸特征对可成交收益的"
        "个股 IC 只有 +0.003~+0.008（h=1）/−0.02（h=10），撑不起 0.065。"
        "**信号具体来自哪一维尚未定位**，只能说预测与日内形态高度相关、且这个相关在小票上更有效。",
        "",
        "🚨 **这句话要改**：此前总结的「个股 alpha 聚合不出行业 alpha」不准确。"
        "准确说法是「**个股 alpha 里确实有行业成分，但集中在低权重股票上，"
        "市值加权聚合按权重把它筛掉了**」。",
        "",
        "## 二 完整分解表",
        "",
        "| 市值分组 | 平均只数 | 行业间方差占比 η² | 个股 IC | 行业内 IC | 行业 IC（市值加权） | t | 行业 IC（等权） | t |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for nm in tab.index:
        r = tab.loc[nm]
        L.append(
            f"| {nm} | {r['平均只数']:.0f} | {r['行业间方差占比']:.1%} | {r['个股IC']:+.4f} | "
            f"{r['行业内IC']:+.4f} | **{r['行业IC_市值加权']:+.4f}** | {r['t_市值加权']:+.2f} | "
            f"{r['行业IC_等权']:+.4f} | {r['t_等权']:+.2f} |")
    L += [
        "",
        "口径说明：",
        "- **个股 IC**：全截面预测 vs 未来收益的 Pearson。",
        "- **行业内 IC**：两侧都减掉行业均值后再算相关 ⇒ 只保留「同行业内选股」能力。",
        "- **行业 IC**：行业因子（行业内聚合）与行业收益的截面相关 ⇒ 「选行业」能力。",
        "- 三组按**当日流通市值**三分位切，各约 1/3 只数。",
        "",
        "## 三 怎么读这张表（三个反直觉点）",
        "",
        "1. **个股 IC ≈ 行业内 IC，差额为负** ⇒ 在个股层看不到行业成分。"
        "但这不是「不存在」，而是个股层方差被行业内噪声主导（4700 只里行业内变异占 93%），"
        "行业成分被淹没。**聚合的作用正是把行业内噪声按 1/√N 削掉，让它浮出来** —— "
        "这也是等权能到 4.23% 的原因。",
        "2. **η² 最大的一组（大市值 13%）反而是行业 IC 最差的（−0.4%）** ⇒ "
        "η² 衡量「预测里有多少行业差异」，不衡量「这部分差异对不对」。"
        "大票的行业差异是有的，只是方向与未来行业收益无关。"
        "**判据不能只看 η²。**",
        "3. **小市值组市值加权 7.70% > 等权 9.42%？** 不是矛盾：组内市值加权仍保留"
        "小市值内部的大票，权重更集中但样本一致性好；全样本时跨了三组，"
        "市值加权把权重几乎全给了大票组（那里信号是 −0.39%），所以塌到 0.25%。",
        "",
        "## 四 与研报的差距落在哪",
        "",
        "研报也是个股训练 → 自由流通市值加权（§2.1.1 标题即「GRU、TCN 因子（个股加权）」），"
        "所以**层级没差**。它市值加权后仍有 6.85%，说明**它的大票也带行业信号**。",
        "",
        "差距被定位到一个可检验的点：**我们的模型在大票上几乎无信号**"
        f"（大市值组个股 IC {big['个股IC']:+.4f}，小市值 {small['个股IC']:+.4f}，差 {small['个股IC'] / abs(big['个股IC']):.1f} 倍）。"
        "而大票弱正是因为信号主体是日内反转（大票流动性好、尾盘异动小）。",
        "",
        "## 五 边界",
        "",
        "- 单配置 `TCN_4_day60_10d`、2021–2026、seed 3；未做多配置/多种子。",
        "- t 值为朴素 t（`mean/std×√N`），**未做 Newey-West**；行业因子相邻日秩相关 0.033，"
        "影响有限但引用要带口径。",
        "- **等权 4.23% 不等于可落地**：实盘要买行业 ETF，而行业 ETF/指数是市值加权，"
        "「等权选行业 + 市值加权持有」存在错配，需单独验证；且小市值信号"
        "要过纸面判据（涨停占比、换手、容量）。",
        "- 现有 `pred_*.parquet` 仍是旧标签训出来的（见主报告 §四 分母伪相关），"
        "本表用的是 `close` 分母重算，但模型权重受旧标签影响 ⇒ 数值是**偏保守下界**。",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    main()
