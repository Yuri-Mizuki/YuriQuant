"""容量与真实成本诊断：把「10bp」这个假设换成按 AUM 估出来的成本。

结论一直挂在一个从未实测的数上——单边 10bp。本脚本用平方根冲击模型，结合
申万一级行业的**真实日成交额**与**真实日波动率**，反解：

    单边成本(AUM) = 半价差 + k · σ_d · sqrt(Q_ind / ADV_ind) + 佣金

其中 Q_ind = AUM × 单边换手 / TopN 是单个行业每次调仓要动的钱，ADV 取调仓日前
20 日均额（PIT，不看未来）。k 是平方根系数，A 股经验区间 0.5~1.5，脚本给三档。

两条交易路径的差别要说清：
  · 路 A「一篮子成分股复制」——ADV 用行业成交额，成本低，但要买几十只股票，
    跟踪误差小、操作重；
  · 路 B「买行业 ETF」——ADV 换成 ETF 自身成交额（比行业成分股小 1~2 个数量级），
    操作轻，但溢价/折价 + 冲击会把成本推高数倍。本脚本只能算路 A（无 ETF 数据），
    路 B 用 --etf-adv-ratio 做敏感性。

用法:
    python -m scripts.model_train.diag_capacity --pred TCN_4_day60_10d__s1full --top 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

DL_DIR = Path("reports/report_replica/dl")
OUT_DIR = Path("reports/report_replica")
SW_FILE = Path(Config.paths()["cache_root"]) / "sw_industry_daily.parquet"

TRADING_DAYS = 252
MIN_N = 20


def load_industry_micro(ind_codes: list[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """行业日成交额（亿元）与日收益率，仅保留需要的行业与区间。"""
    s = pd.read_parquet(SW_FILE)
    s["date"] = pd.to_datetime(s["date"])
    s = s[s["date"] >= start].copy()

    # 申万代码 801010.SI ↔ ctx 里的行业键（形如 801010 或 801010.SI），做双向映射
    key = s["code"].astype(str).str.split(".").str[0]
    s["key"] = key
    codes = pd.Index(ind_codes).astype(str)
    codes0 = codes.str.split(".").str[0]
    keep = np.isin(s["key"].to_numpy(), codes0.to_numpy())
    s = s[keep].copy()

    amt = s.pivot_table(index="date", columns="key", values="amount", aggfunc="last").sort_index()
    px = s.pivot_table(index="date", columns="key", values="close", aggfunc="last").sort_index()
    ret = px.pct_change()
    return amt, ret


def main() -> None:
    ap = argparse.ArgumentParser(description="容量与真实成本诊断")
    ap.add_argument("--pred", default="TCN_4_day60_10d__s1full")
    ap.add_argument("--weight", default="float_mcap")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--blend-L", type=int, default=0,
                    help="与长窗动量合成的回望天数；0=不合成")
    ap.add_argument("--blend-w", type=float, default=0.4,
                    help="长窗腿权重（合成时 DL 权重 = 1-w）")
    ap.add_argument("--aum", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
                    help="组合规模（亿元）")
    ap.add_argument("--k", type=float, nargs="+", default=[0.5, 1.0, 1.5],
                    help="平方根冲击系数")
    ap.add_argument("--spread-bp", type=float, default=3.0, help="单边半价差 bp")
    ap.add_argument("--fee-bp", type=float, default=0.5, help="佣金+过户费 bp")
    ap.add_argument("--etf-adv-ratio", type=float, default=1.0,
                    help="ETF 路径敏感性：行业 ETF 成交额 / 行业成分股成交额")
    args = ap.parse_args()

    from scripts.model_train.eval_dl_industry import agg_one
    from scripts.portfolio.report_replica import build_ctx
    # 周频必须用 `diag_exec_timing` 的 `head_excess`（`shift=True` 跨期持有）。
    # ⚠️ 不能用 `diag_daily_rebal.tranches`：那是**日频**实现，给它周度 dates 配
    # `hold=5` 会把「5 个交易日」读成「5 个调仓期 = 5 周」，毛年化会从 +5.5% 掉到 −17.7%。
    from scripts.model_train.diag_exec_timing import synth_panel, head_excess, weekly_turnover

    pred = pd.read_parquet(DL_DIR / f"pred_{args.pred}.parquet")
    ctx = build_ctx("W", args.start, weight=args.weight)
    dates = pd.DatetimeIndex(ctx["dates"])
    inds = list(ctx["inds"])
    fac = agg_one(pred, ctx).reindex(index=dates, columns=inds)
    if args.blend_L:
        from scripts.portfolio.report_replica import load_px, zrank
        mom = (load_px() / load_px().shift(args.blend_L) - 1)
        mom = mom.reindex(index=dates, columns=inds)
        fac = (1 - args.blend_w) * zrank(fac) + args.blend_w * zrank(mom)
    fac = fac.to_numpy()

    amt, ret = load_industry_micro(inds, args.start)

    # 行业键对齐：ctx 的 inds → sw 的 6 位 code
    sw_key = {c: str(c).split(".")[0] for c in inds}
    amt.columns = [str(c) for c in amt.columns]
    ret.columns = [str(c) for c in ret.columns]

    # 1) 逐期取 TopN，记录该期所选行业的 ADV20 与 σ20（都用调仓日**之前**的数据）
    rows = []
    for k, d in enumerate(dates):
        if d not in amt.index:
            continue
        f = fac[k]
        ok = np.isfinite(f)
        if ok.sum() < MIN_N:
            continue
        idx = np.where(ok)[0]
        order = idx[np.argsort(f[idx])]
        picks = [inds[i] for i in order[-args.top:]]

        pos = amt.index.get_loc(d)
        if pos < 20:
            continue
        adv = amt.iloc[pos - 20:pos].mean()      # 亿元/日
        sig = ret.iloc[pos - 20:pos].std() * 100  # %/日

        advs, sigs = [], []
        for c in picks:
            j = sw_key[c]
            if j in adv.index and np.isfinite(adv.get(j, np.nan)):
                advs.append(float(adv[j]))
                sigs.append(float(sig[j]) if np.isfinite(sig.get(j, np.nan)) else np.nan)
        if len(advs) < args.top:
            continue
        rows.append({"sig": d, "adv_min": min(advs), "adv_mean": float(np.mean(advs)),
                     "sig_mean": float(np.nanmean(sigs))})

    m = pd.DataFrame(rows).set_index("sig")
    print(f"预测 {args.pred} · {args.weight} · 周频 {len(dates)} 期 · 有效 {len(m)} 期 · Top{args.top}")
    print(f"所选行业 ADV20（亿元/日）中位数 {m['adv_mean'].median():.0f}"
          f" / 最差行业 {m['adv_min'].median():.0f} · 日波动率中位数 {m['sig_mean'].median():.2f}%")

    # 2) 基准：毛收益与换手（周频 S2 = T+1 开盘 → 下次 T+1 开盘）
    px = synth_panel(ctx)
    s, picks = head_excess(px["po1"], px["po1"], fac, dates, inds, shift=True, top_n=args.top)
    tr = float(weekly_turnover(picks))
    gross = float((1 + s).prod() ** (52 / len(s)) - 1)
    print(f"\n毛年化 {gross*100:+.2f}% · 单边换手 {tr:.3f} · 期数 {len(s)}")

    # 3) AUM × k → 单边成本 → 净年化
    print(f"\n=== 单边成本（bp）· 半价差 {args.spread_bp:.1f}bp + 冲击 + 佣金 {args.fee_bp:.1f}bp ===")
    print(f"{'AUM(亿)':>8}" + "".join(f"{'k='+str(k):>10}" for k in args.k))
    out = []
    for aum in args.aum:
        q_ind = aum * tr / args.top          # 亿元，单个行业每次要动的钱
        line = f"{aum:>8.1f}"
        for k in args.k:
            # 冲击bp = k × σ(%) × 100 × sqrt(Q/ADV)，σ 用中位数（保守取最差行业会更好，见下）
            q_adv = q_ind / (m["adv_mean"].median() * args.etf_adv_ratio)
            imp = k * m["sig_mean"].median() * 100 * np.sqrt(q_adv)
            one_way = args.spread_bp + args.fee_bp + imp
            line += f"{one_way:>10.1f}"
            out.append({"aum": aum, "k": k, "q_ratio": q_adv, "impact_bp": imp,
                        "one_way_bp": one_way})
        print(line)

    df = pd.DataFrame(out)
    df["per_period_bp"] = df["one_way_bp"] * tr * 2
    df["cost_ann"] = df["per_period_bp"] * 52 / 1e4
    # 逐期扣成本再复利（与 diag_exec_timing 一致）
    net = []
    for _, r in df.iterrows():
        c = r["per_period_bp"] / 1e4
        net.append(float((1 + s - c).prod() ** (52 / len(s)) - 1))
    df["net_ann"] = net

    print(f"\n=== 净年化（%，毛 {gross*100:.2f}%）===")
    piv = df.pivot_table(index="aum", columns="k", values="net_ann") * 100
    print(piv.round(2).to_string())

    print(f"\n=== 成本年化（%）===")
    piv2 = df.pivot_table(index="aum", columns="k", values="cost_ann") * 100
    print(piv2.round(2).to_string())

    # 4) 平衡点：净 = +3%（M2）与净 = 0
    print("\n=== 平衡点（k=1.0）===")
    sub = df[df["k"] == 1.0].sort_values("aum")
    for target, name in ((0.03, "M2 +3%"), (0.0, "打平")):
        hit = sub[sub["net_ann"] >= target]
        if len(hit):
            print(f"  过 {name}：AUM ≤ {hit['aum'].max():.1f} 亿"
                  f"（该点单边 {hit.sort_values('aum').iloc[-1]['one_way_bp']:.1f}bp）")
        else:
            print(f"  过 {name}：**任何规模都过不了**（最小 AUM 也只有 "
                  f"{sub['net_ann'].max()*100:+.2f}%）")

    # 5) 要多少 IC 才能过 M2（成本固定为「1 亿规模」下的真实值）
    base = sub[sub["aum"] == 1.0]
    if len(base):
        c1 = float(base["per_period_bp"].iloc[0]) / 1e4
        need_net = 0.03
        # 近似：净年化 ≈ 毛年化 − 成本年化，反解所需毛
        need_gross = need_net + c1 * 52
        ratio = need_gross / gross if gross > 0 else np.nan
        ic_now = 0.0395
        if gross >= need_gross:
            print(f"\n  1 亿规模下成本 {c1*52*100:.2f}%/年；要过 M2 需毛 {need_gross*100:.2f}%，"
                  f"实际毛 {gross*100:.2f}% ⇒ **已过线**（余量 {(gross-need_gross)*100:+.2f}pt）")
        else:
            print(f"\n  1 亿规模下成本 {c1*52*100:.2f}%/年；要过 M2 需毛 {need_gross*100:.2f}%"
                  f" ⇒ 毛收益需 ×{ratio:.2f} ⇒ 若 IC 线性映射，IC 需 {ic_now*ratio:.4f}"
                  f"（研报披露 0.0548，我们 {ic_now}）")

    tag = f"_{args.weight}" if args.weight != "float_mcap" else ""
    out_file = OUT_DIR / f"diag_capacity{tag}.csv"
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    m.to_csv(OUT_DIR / f"diag_capacity_picks{tag}.csv", encoding="utf-8-sig")
    print(f"\n→ {out_file}")


if __name__ == "__main__":
    main()
