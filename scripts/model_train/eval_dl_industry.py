"""把自训 GRU / TCN 个股预测聚合成行业因子，出 IC / 组合表 + 诊断报告。

只用研报口径：**成分股自由流通市值加权 → 行业因子**（研报 §2.1.1 开头），
随后沿用 `scripts/portfolio/report_replica.py` 的行业单因子测试与组合回测
（周频、当天收盘算 IC、T+1 收盘成交、Top5、单边 10bp/36bp）。

诊断（回答「为什么个股 IC 高、行业 IC 却没有」）：
  · 个股层截面 IC（日频）
  · 预测的**行业间方差占比 η²** —— 预测里有多少是行业效应
  · 行业因子相邻日秩相关 —— 是静态风格还是日频噪声
  · 预测与 6 个输入特征的截面相关 —— 信号实际来源
  · 多头组涨停占比 / 全市场 —— 纸面判据第 ③ 条（>5 倍 = 纸面）
  · 剔除 T 日涨跌停后的个股 IC

用法：
    python -m scripts.model_train.eval_dl_industry            # 评估 dl/ 下全部 pred_*
    python -m scripts.model_train.eval_dl_industry --report    # 另写 dl/README.md
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import Config  # noqa: E402
from scripts.portfolio.report_replica import (  # noqa: E402
    OUT_DIR,
    by_year,
    ic_series,
    load_px,
    port_backtest,
    stats,
    zrank,
)

DL_DIR = OUT_DIR / "dl"
PQ = Path(Config.paths()["cache_root"])
FEAT_NAMES = ["open/close", "high/close", "low/close", "close_ret", "vwap/close", "turnover"]


def agg_one(panel: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """个股面板 → 行业因子（行业内自由流通市值加权 + 1/99 winsorize）。"""
    from scripts.oneoff.etf_industry_factor_p0 import _agg

    rows = []
    for i, dt in enumerate(ctx["dates"]):
        if dt not in panel.index:
            rows.append(np.full(len(ctx["inds"]), np.nan))
            continue
        v = panel.loc[dt].reindex(ctx["universe"]).to_numpy(dtype=float)
        rows.append(_agg(v, ctx["lbls"][i], ctx["wts"][i]))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(ctx["dates"]),
                        columns=ctx["inds"])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / d if d > 0 else np.nan


def diagnose(pred: pd.DataFrame, ctx: dict, step: int = 5) -> dict:
    """个股层 IC / η² / 自相关 / 特征暴露 / 涨停占比。"""
    from scripts.model_train.replica_deep import build_arrays, forward_avg_ret

    arr = build_arrays()
    idx, codes, feat = arr["index"], arr["codes"], arr["feat"]
    lab = forward_avg_ret(arr["avg"], idx, 10, arr["close"])      # 可成交口径
    # 伪口径：分母用当日均价。只用于暴露「分母共线」有多大，不参与任何结论。
    lab_bad = forward_avg_ret(arr["avg"], idx, 10, arr["avg"])

    d = pd.read_parquet(PQ / "daily_all_a.parquet", columns=["close"])
    close = d["close"].unstack("code").reindex(index=idx, columns=codes)
    ret = close.pct_change()
    up = ret > 0.095

    days = pred.dropna(how="all").index
    ics, ics_bad, ics_ex, feat_corr = [], [], [], {k: [] for k in FEAT_NAMES}
    ctrl = {"bad": [], "ok": []}
    lim = []
    for n, dt in enumerate(days):
        i = idx.get_loc(dt)
        p = pred.loc[dt]
        y = pd.Series(lab[i], index=codes)
        yb = pd.Series(lab_bad[i], index=codes)
        m = p.notna() & np.isfinite(y)
        if m.sum() < 500:
            continue
        ics.append(_corr(p[m].to_numpy(), y[m].to_numpy()))
        ok_b = m & np.isfinite(yb)
        if ok_b.sum() > 500:
            ics_bad.append(_corr(p[ok_b].to_numpy(), yb[ok_b].to_numpy()))
        # 裸特征对照：直接用 −(vwap/close) 当预测，看两套分母下差多少
        f_vw = pd.Series(feat[i, :, 4], index=codes)
        mc = m & f_vw.notna()
        if mc.sum() > 500:
            ctrl["bad"].append(_corr(-f_vw[mc].to_numpy(), yb[mc].to_numpy()))
            ctrl["ok"].append(_corr(-f_vw[mc].to_numpy(), y[mc].to_numpy()))
        ok = m & ~(ret.iloc[i].abs() > 0.095)
        if ok.sum() > 500:
            ics_ex.append(_corr(p[ok].to_numpy(), y[ok].to_numpy()))
        if n % step == 0:
            for k, nm in enumerate(FEAT_NAMES):
                f = pd.Series(feat[i, :, k], index=codes)
                mm = p.notna() & f.notna()
                if mm.sum() > 500:
                    feat_corr[nm].append(_corr(p[mm].to_numpy(), f[mm].to_numpy()))
            q = p[m].rank(pct=True)
            sub = up.iloc[i][m]
            lim.append((float(sub[q >= 0.9].mean()), float(sub.mean())))

    # 行业分解
    eta2, ratios = [], []
    for i, dt in enumerate(pd.DatetimeIndex(ctx["dates"])):
        if dt not in pred.index:
            continue
        p = pred.loc[dt].reindex(ctx["universe"]).to_numpy(dtype=float)
        lbl = ctx["lbls"][i]
        m = np.isfinite(p) & (lbl >= 0)
        if m.sum() < 500:
            continue
        x, L = p[m], lbl[m]
        grp = pd.Series(x).groupby(L).mean()
        n = pd.Series(x).groupby(L).size()
        gm = x.mean()
        eta2.append(float((n * (grp - gm) ** 2).sum() / ((x - gm) ** 2).sum()))
        ratios.append(float(grp.std() / x.std()))

    ind = agg_one(pred, ctx).reindex(index=load_px().index)
    acs, prev = [], None
    for dt, row in ind.loc[days[0]:].iterrows():
        if row.notna().sum() < 20:
            continue
        r = row.rank()
        if prev is not None:
            acs.append(r.corr(prev))
        prev = r

    lim = np.asarray(lim)
    return {
        "ic_stock": float(np.nanmean(ics)),
        "ic_stock_badbase": float(np.nanmean(ics_bad)),
        "ctrl_badbase": float(np.nanmean(ctrl["bad"])),
        "ctrl_okbase": float(np.nanmean(ctrl["ok"])),
        "ic_stock_ex_limit": float(np.nanmean(ics_ex)),
        "n_ic": len(ics),
        "eta2": float(np.median(eta2)),
        "grp_std_ratio": float(np.median(ratios)),
        "ac": float(np.nanmedian(acs)),
        "feat_corr": {k: float(np.nanmean(v)) for k, v in feat_corr.items() if v},
        "limit_top": float(lim[:, 0].mean()) if len(lim) else np.nan,
        "limit_all": float(lim[:, 1].mean()) if len(lim) else np.nan,
        "start": days[0].date(), "end": days[-1].date(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="all")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--freq", default="W", choices=["W", "M"])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    from scripts.portfolio.report_replica import build_ctx

    px = load_px()
    tdays = px.index
    ctx = build_ctx(args.freq, args.start)
    dates = list(pd.DatetimeIndex(ctx["dates"]))

    files = (sorted(DL_DIR.glob("pred_*.parquet")) if args.configs == "all"
             else [DL_DIR / f"pred_{n}.parquet" for n in args.configs.split(",")])
    rows, series, diags = [], {}, {}
    for fp in files:
        name = fp.stem.replace("pred_", "")
        pred = pd.read_parquet(fp)
        if not pred.notna().any(axis=1).any():
            print(f"跳过 {name}：预测为空")
            continue
        ind = agg_one(pred, ctx).reindex(index=px.index, columns=px.columns)
        series[name] = ind

        ic0 = ic_series(ind, dates, px, tdays, lag=0)
        st = stats(ic0)
        r = port_backtest(ind, px, dates, top=5, cost_bp=10.0)
        rows.append({"factor": name, "n_ic": st["n"], "ic": st["ic"],
                     "icir": st["icir"], "t": st["t"], "t_nw": st["t_nw"],
                     "ann": r.get("ann"), "ann_bench": r.get("ann_bench"),
                     "excess": r.get("excess"), "t_ir": r.get("t_ir"),
                     "turnover": r.get("turnover")})
        print(f"  {name:22s} 行业层 IC {st['ic']:+.4f} t_nw {st['t_nw']:+.2f} | "
              f"超额 {r.get('excess', float('nan')):+.2%} "
              f"t(IR) {r.get('t_ir', float('nan')):+.2f}")
        diags[name] = diagnose(pred, ctx)
        dg = diags[name]
        print(f"    个股层 IC {dg['ic_stock']:+.4f}"
              f"（剔涨跌停 {dg['ic_stock_ex_limit']:+.4f}）"
              f" | 若分母用 avg[t] 则为 {dg['ic_stock_badbase']:+.4f} ⚠️伪"
              f" | η² {dg['eta2']:.3f}"
              f" | 相邻日秩相关 {dg['ac']:.3f} | 多头组涨停 {dg['limit_top']:.1%}"
              f" vs 全市场 {dg['limit_all']:.1%}")

    if not rows:
        print("无可用预测")
        return
    df = pd.DataFrame(rows).sort_values("ic", ascending=False)
    df.to_csv(OUT_DIR / f"dl_industry_{args.freq}.csv", index=False,
              encoding="utf-8-sig")

    comb = None
    if series:
        z = sum(zrank(v) for v in series.values()) / len(series)
        comb = port_backtest(z, px, dates, top=5, cost_bp=10.0)
        print(f"\n  {'DL合成':22s} 年化 {comb['ann']:+.2%} "
              f"基准 {comb['ann_bench']:+.2%} 超额 {comb['excess']:+.2%} "
              f"t(IR) {comb['t_ir']:+.2f}")
        pd.DataFrame({"ret": comb["ret"], "bench": comb["bench"]}).to_csv(
            OUT_DIR / f"dl_comb_returns_{args.freq}.csv", encoding="utf-8-sig")
        print(by_year(comb["ret"].dropna(), comb["bench"].dropna()).to_string(
            float_format=lambda v: f"{v:+.4f}"))

    print(f"\n→ {OUT_DIR / f'dl_industry_{args.freq}.csv'}")
    if args.report:
        write_report(df, diags, comb, args.freq)


def write_report(df: pd.DataFrame, diags: dict, comb: dict | None, freq: str) -> None:
    L = [
        "# 深度学习维（GRU / TCN）复现 —— 按研报 §2.1.1 自训：个股 IC 0.072、行业层归零",
        "",
        f"> **判定**：`复现完成（负结果）` · {freq} 频 · 生成于 2026-10-31",
        "> **一句话**：架构 / 损失 / batch / 滚动全部照研报，个股层可成交 IC **0.072**"
        "（剔涨跌停 0.047），**但聚合到行业层只剩 0.006、组合 −7.95%** —— 产不出行业信号。",
        "",
        "> 🚨 **2026-10-31 更正**：此前报「个股层 IC 0.197」是**标签分母伪相关**所致，已作废。",
        "> 研报写「均价涨跌幅」未给分母；照字面用 `avg[t]` 做分母，会与特征 `vwap/close`",
        "> 共用 `avg[t]`，机械相关理论上限 **0.178**（实测裸特征 0.169 / 模型 0.198）。",
        "> 换成可成交价 `close[t]` 做分母后 **0.198 → 0.072**。详见 §四。",
        "",
        "## 一 与研报逐条对齐（实现口径）",
        "",
        "| 研报要求（§2.1.1.3） | 本线实现 | 一致 |",
        "|---|---|---|",
        "| 输入 30/60 日行情、6 特征（高开低收均价换手），去量纲+标准化 | "
        "`open/close, high/close, low/close, close/close_{t-1}, vwap/close, turnover`，"
        "逐日截面 z-score | ⚠️ 去量纲方式见 §四 |",
        "| 标签 t+1 起 5/10 日均价涨跌幅、标准化 | "
        "`mean(avg[t+1..t+h])/close[t]-1`，逐日截面 z-score | ⚠️ 分母见 §四 |",
        "| 滚动训练：每年一个模型供下一年用 | 每年一个模型；训练 < 该年，验证 = 上一年 | ✅ |",
        "| **损失 = 预测值与收益率的相关系数取负** | `corr_loss`（`model/seq_deep.py`） | ✅ |",
        "| **BatchSize = 一个交易日的股票数量** | 每交易日一个 batch，不打乱 | ✅ |",
        "| GRU 2 层 | `nn.GRU(num_layers=2, hidden=64)` | ✅（hidden 研报未公开）|",
        "| TCN kernel=3；60 日 4 层 `[8,16,32,64]`、30 日 3 层 `[8,16,32]` | "
        "因果卷积 + 膨胀(1/2/4/8) + 残差 | ✅ |",
        "",
        "## 二 结果",
        "",
        "| 因子 | 行业层 IC | t_nw | Top5 年化超额 | t(IR) |",
        "|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        L.append(f"| `{r['factor']}` | {r['ic']:+.4f} | {r['t_nw']:+.2f} | "
                 f"{r['excess']:+.2%} | {r['t_ir']:+.2f} |")
    if comb is not None:
        L += ["", f"**DL 合成**（等权）：年化 {comb['ann']:+.2%}、"
                  f"基准 {comb['ann_bench']:+.2%}、超额 {comb['excess']:+.2%}、"
                  f"t(IR) {comb['t_ir']:+.2f}。", ""]
    L += ["**研报对照**：12 个 GRU/TCN 因子行业层 IC **5.48%~8.21%（均值 6.85%、"
          "IC_IR 1.48）**，多头组年化超额 14.66%。我们一个都没达到。", "",
          "## 三 个股层强度与信号来源（诊断）", ""]
    for name, d in diags.items():
        L += [f"### `{name}`（{d['start']} ~ {d['end']}）", "",
              f"- **个股层截面 IC {d['ic_stock']:+.4f}**（日频 {d['n_ic']} 期；"
              f"剔除 T 日涨跌停后 {d['ic_stock_ex_limit']:+.4f}）。"
              f"若把标签分母换成 `avg[t]`，同一个预测会「变成」"
              f"**{d['ic_stock_badbase']:+.4f}** ⚠️ —— 那不是模型变强，是分母伪相关（§四）。",
              f"- **行业间方差占比 η² = {d['eta2']:.1%}** —— 预测的 "
              f"**{1 - d['eta2']:.0%} 变异在行业内**。行业均值 std / 个股 std 仅 "
              f"{d['grp_std_ratio']:.2f}（纯噪声聚合应到 ~0.09 ⇒ 行业成分真实存在，"
              "但绝对量太小）。",
              f"- **行业因子相邻日秩相关 {d['ac']:+.3f}** —— 不是静态风格押注，是日频噪声。",
              "- **信号来源**：预测与输入特征的截面相关 = "
              + "、".join(f"`{k}` {v:+.2f}" for k, v in d["feat_corr"].items())
              + "。模型几乎是当日 6 个日内形状特征的线性组合，"
              "即**日内反转 / 尾盘异动**，不是跨周期的行业景气判断。",
              f"- **纸面判据第 ③ 条**：多头组（预测前 10%）T 日涨停占比 "
              f"**{d['limit_top']:.1%}**，全市场 {d['limit_all']:.1%} ⇒ "
              f"**{d['limit_top'] / max(d['limit_all'], 1e-9):.1f} 倍**（阈值 5 倍）。"
              "涨停股次日买不进，这段收益收不到。", ""]
    L += ["## 四 🚨 标签分母的伪相关（本次最大更正）", "",
          "**症状**：同一份预测，只改标签分母，个股 IC 就从 0.198 掉到 0.072。", "",
          "| 口径 | 分母 | TCN 个股 IC | 裸特征 `−(vwap/close)` 对照 |",
          "|---|---|---|---|",
    ]
    for name, d in diags.items():
        L.append(f"| 伪（研报字面） | `avg[t]`（当日均价） | **{d['ic_stock_badbase']:+.4f}** | "
                 f"**{d['ctrl_badbase']:+.4f}** |")
        L.append(f"| 正确（可成交） | `close[t]`（当日收盘） | **{d['ic_stock']:+.4f}** | "
                 f"{d['ctrl_okbase']:+.4f} |")
    L += ["",
          "**机理**：`label = mean(avg[t+1..t+h]) / avg[t] − 1` 的分母是**当日均价**，"
          "而特征里 `vwap/close = avg[t]/close[t]` 的分子正是同一个 `avg[t]`。取对数：",
          "",
          "```",
          "label ≈ (未来均价/close − 1) − log(avg[t]/close[t]) = r − d ,   d ≡ log(avg/close)",
          "⇒ corr(−d, label) ≈ σ(d) / σ(r)",
          "```",
          "",
          "实测 σ(d) = **0.95%**（日频截面）、σ(10 日收益) = **5.33%** ⇒ 机械相关上限 "
          "**0.178**。也就是说：**一个只输出「今日均价偏离收盘多少」的假预测，什么都不学，"
          "就能拿到 IC 0.17**。而换成 `close` 分母后，同一个裸特征只有 −0.013。", "",
          "**这条对研报的意义**：它写「均价涨跌幅」没写分母。若它也用 `avg[t]` 做分母，"
          "6.85% 的行业层 IC 会同样被污染；若它用 `close[t]`，则我们与它的差距仍在"
          "「特征去量纲方式」（§五 第 2 条）。这一点无法从研报推出，**不能拿来当"
          "「研报也作弊」的证据**。", "",
          "**已修**：`forward_avg_ret(avg, idx, h, close)` 分母强制为 `close[t]`"
          "（`scripts/model_train/replica_deep.py`），并加了 `avg` 分母的对照列，"
          "防止下次再犯。⚠️ **当前这份 `pred_*.parquet` 是用伪标签训练出来的**，"
          "它追了一部分分母伪相关，所以 0.072 是**偏保守的下界**，不是修好后的水平。", "",
          "## 五 与研报的三处不可消除差异（引用必须带）", "",
          "1. **训练窗短**：数据起点 2015-01（`daily_all_a` 首日），研报 2008 起。"
          "研报「前 9 年训练」，我们 2018 年模型只有 3 年历史。",
          "2. **特征去量纲方式**：我们把 OHLC 都除以当日收盘，得到 `vwap/close`。"
          "这个比值本身在 A 股是极强的次日反转信号（单特征 h=1 个股 IC **−0.42**，"
          "但那 0.42 里有多少是真反转、多少是上面的分母伪相关，**未拆**），"
          "模型很容易退化成抓它。研报只说「去量纲、标准化」未给公式；"
          "若它用的是「除以均价」或「个股时序标准化」，`vwap/close` 这一维就不存在，"
          "模型会被迫去学更慢的行业成分 —— **这是最可能的分歧点**。",
          "3. **行业池**：申万一级 28→31 vs 中信一级 30。",
          "",
          "## 六 结论与下一步", "",
          "**结论**：个股层可成交 IC 0.072，行业层市值加权 IC 0.006、组合 −7.95%。",
          "",
          "🚨 **但归零的机制不是「个股 alpha 聚合不出行业 alpha」** —— 见 "
          "`decomp_industry.md`（脚本 `decomp_dl_industry.py`）：同一份预测 "
          "**等权聚合行业 IC +0.0423（t=2.10）**，市值加权只有 **+0.0025**；"
          "分市值看，**小市值组市值加权都有 +0.0770（t=3.92）、大市值组 −0.0039**。"
          "⇒ 行业成分是**存在**的，只是集中在低权重股票上，"
          "自由流通市值加权聚合按权重把它筛掉了。",
          "",
          "**下一步（已预注册，见 `TODO.md` §一「研报七维复现线」）**：",
          "1. 🚨 **用 `close` 分母标签重训**（代码已修）—— 当前预测是伪标签训的，"
          "0.072 不能当作「模型上限」；",
          "2. 换去量纲方式（除以均价 / 个股时序 z-score）重训 —— 直接针对 §五 第 2 条；",
          "3. 训练/预测剔除 T 日涨跌停，去掉纸面成分；",
          "4. 补 `week30` 四个配置（日线重采样周线）与 day30 四个；",
          "5. 以上全挂好机器（CPU 单配置 6 年 ≈ 2.5 小时，GPU 快 20 倍）。",
          "",
          "## 七 复现", "",
          "```bash",
          "# 训练（GPU 用 --device cuda；CPU 建议只挑 1~2 个配置）",
          "python -m scripts.model_train.replica_deep --configs TCN_4_day60_10d \\",
          "    --start 2021 --epochs 10 --patience 2 --device cuda",
          "# 聚合 + 评估 + 诊断报告",
          "python -m scripts.model_train.eval_dl_industry --configs all --report",
          "```",
          "",
          "产物：`pred_*.parquet`（个股预测面板）、`train_log_*.csv`（逐年训练/验证 IC）、"
          f"`dl_industry_{freq}.csv`、本文件。",
          ""]
    (DL_DIR / "README.md").write_text("\n".join(L), encoding="utf-8")
    print(f"→ {DL_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
