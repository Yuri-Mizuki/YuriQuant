"""`rd_to_cap`（研发强度）的生产口径叠加验证 —— P0。

为什么还要再做一遍
------------------
`reports/factor_ev_rd/` 已经量出 `rd_to_cap` 是全部候选里最强的一个
（单因子 Top10% 等权超额 **+9.16pp**、neu IC +0.0407 t=8.78、三段 t 3.99/1.96/3.67），
但那份结论有三个口径缺口，**不能直接当"能提升定版"用**：

1. **没有行业中性化**（原表用原始面板 + 收盘价撮合）⇒ 超额可能来自行业/市值暴露；
2. **没有 T+1 open 撮合**（月频 close→close，与生产的 T+1 open 差一层）；
3. **交集池剔除 28% 无研发公司** ⇒ "更强"里混进了"池子不同"。

本脚本补这三项，回答的是**唯一有意义的问题**：
把它叠加到定版信号上，走 production 口径（27bp 时变 / T+1 open / 整手），
组合收益有没有增量 —— 且这个增量必须**跑赢安慰剂臂**。

判据（预注册，2026-11-13）
--------------------------
- **Δann ≥ +1.00pp**（对同档 base 臂），且
- **Δann > 安慰剂臂 Δann + 1pp**，安慰剂两形态：
  `rand`（同 NaN 掩码的随机值）与 `shuf`（逐日打乱 rd 截面次序，保留边缘分布）。
  两者都过 ⇒ 说明增量来自 rd 的**截面排序信息**，不是"多叠一路信号把组合摊匀"。

🚨 必须在**21 个相位上配对**后再判
-------------------------------
solo 自己的相位极差是 **17.98pp**（`reports/rtl_phase_tiers/`），单相位下任何
±4pp 的臂间差都落在噪声里（2026-11-13 实测：同一臂在 phase 0 差 +3.88pp、
在 phase 10 差 −3.15pp，量级全由相位决定）。所以本脚本**逐相位跑、按相位配对**
取 Δ 的均值与 t（`Δ_p = ann_arm,p − ann_base,p`，t = mean/(std/√21)）——
相位是共同冲击，配对后抵消。**只报单相位的结论一律不可信。**

叠加方式与 fill 的选择
----------------------
`rd_to_cap` 覆盖只有 ~70%（研发披露缺失），而 `weighted_rank_average` 取**交集**
⇒ 直接复合会把股票池砍掉 30%，那会与"因子有没有用"混在一起。所以这里自己做复合：

    signal = (rank_pct(base) + w · rank_pct(rd)) / (1 + w)

`rank_pct(rd)` 的缺失格**填 0.5（中性）**，池子保持与 base 逐日完全相同
（`n_hold` 必须与 base 逐位相等，这是自检）。另给一条 `fill=0.0`（未知研发视作最差）
作为敏感性检验 —— 若只有 fill0 有效，说明是"惩罚无研发公司"而不是研发强度本身。

用法::

    python -m scripts.evaluation.rd_overlay_probe                 # 三档 × 7 臂（约 5 分钟）
    python -m scripts.evaluation.rd_overlay_probe --tiers yqcore-solo
    python -m scripts.evaluation.rd_overlay_probe --diag-only     # 只跑诊断（30 秒）
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.evaluation.caliber as cal
from scripts.evaluation import yq_core_report as ycr
from strategy.registry import get_tier

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "rd_overlay"
RD_RAW = ROOT / "reports" / "factor_ev_rd" / "panels" / "rd_to_cap.parquet"
NEU_BASE = ROOT / "reports" / "alla_rolling" / "_base"
CACHE = OUT / "rd_to_cap_neu.parquet"
TIERS = ("yqcore-solo", "yqcore-focus", "yqcore-breadth")

#: 产物文件名后缀（`--suffix`），分批跑时避免互相覆盖
_SUFFIX = ""

#: (臂名, 权重 w, 缺失格的 rank 填充, 值来源)
ARMS = (
    ("base",       0.00, None, "rd"),
    ("rd_w05",     0.05, 0.5,  "rd"),
    ("rd_w10",     0.10, 0.5,  "rd"),
    ("rd_w20",     0.20, 0.5,  "rd"),
    ("rd_w20_f0",  0.20, 0.0,  "rd"),      # 未知研发视作最差
    ("rd_neg_w05", 0.05, 0.5,  "neg"),     # 剂量-反应：反号应反向亏
    ("rd_ma60_w05", 0.05, 0.5, "ma60"),    # 分母去日频波动：滚动 60 日均值
    ("pl_rand",    0.20, 0.5,  "rand"),    # 安慰剂：同掩码随机值
    ("pl_shuf",    0.20, 0.5,  "shuf"),    # 安慰剂：逐日打乱次序
)


# ---------------------------------------------------------------- 面板

def rd_neutralized() -> pd.DataFrame:
    """`rd_to_cap` → 生产同口径中性化面板（行业 + log 市值，MAD 去极值 + zscore）。

    与 `scripts/builders/common.py::neutralize_one` 逐字同参，故与 `panels_neu` 可比。
    结果缓存到 `reports/rd_overlay/`（~1 分钟，算一次就够）。
    """
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    from factor.preprocessing import preprocess_factor

    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(RD_RAW)
    if not isinstance(raw.index, pd.DatetimeIndex):
        raw.index = pd.to_datetime(raw.index.astype(str), format="%Y%m%d")
    mc = pd.read_parquet(NEU_BASE / "market_cap.parquet")
    ind = pd.read_parquet(NEU_BASE / "cov_industry.parquet")
    common = mc.index.intersection(raw.index)
    raw = raw.reindex(index=common)
    x = preprocess_factor(raw, market_cap_panel=mc.reindex(index=common),
                          industry_panel=ind.reindex(index=common))
    x = x.replace([np.inf, -np.inf], np.nan).astype(np.float32).clip(-10.0, 10.0)
    x.to_parquet(CACHE)
    print(f"  rd_to_cap 中性化完成 → {CACHE.name} 覆盖 "
          f"{float(x.notna().to_numpy(dtype=float).mean()):.3f}", flush=True)
    return x


def _rank_pct(p: pd.DataFrame) -> pd.DataFrame:
    return p.rank(axis=1, pct=True)


def composite(base: pd.DataFrame, rd: pd.DataFrame, w: float,
              fill: float | None) -> pd.DataFrame:
    """`(rank(base) + w·rank(rd)) / (1+w)`；`fill` 只作用于 rd 的缺失格。

    池子约束与 base 逐日相同（`where(base.notna())`），**不取交集** —— 见模块头。
    """
    rb = _rank_pct(base)
    if w == 0:
        return rb.where(base.notna()).astype("float32")
    rr = _rank_pct(rd.reindex(index=base.index, columns=base.columns))
    if fill is not None:
        rr = rr.fillna(fill)
    return ((rb + w * rr) / (1.0 + w)).where(base.notna()).astype("float32")


def placebo_panels(rd: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """两形态安慰剂 + 一条「分母去噪」对照。均保留 NaN 位置。

    `ma60` 不是安慰剂，是**替代解释的排除**：`rd_to_cap = RD_ANN / 市值` 的分母是
    日频市值，逐日随涨跌变化 ⇒ 该因子天然带一层**短期反转**。滚动 60 日均值把日频
    市值churn 抹掉、只留「研发强度」的慢排序：若 `rd_ma60` 的增量消失，说明
    `rd_to_cap` 的增益来自反转而非研发强度。
    """
    rng = np.random.default_rng(20261113)
    v = rd.to_numpy(dtype=float, copy=True)
    nan = ~np.isfinite(v)
    rand = v.copy()
    rand[~nan] = rng.standard_normal(int((~nan).sum()))
    rand[nan] = np.nan
    shuf = v.copy()
    for i in range(v.shape[0]):
        row = v[i]
        ok = np.isfinite(row)
        if ok.sum() > 1:
            shuf[i, ok] = rng.permutation(row[ok])
    slow = rd.rolling(60, min_periods=20).mean()
    # 滚动后截面尺度会漂，按日重做 zscore（保持与其它臂同量纲）
    slow = slow.sub(slow.mean(axis=1), axis=0).div(slow.std(axis=1), axis=0)
    return {"rand": pd.DataFrame(rand, index=rd.index, columns=rd.columns),
            "shuf": pd.DataFrame(shuf, index=rd.index, columns=rd.columns),
            "ma60": slow.astype("float32")}


# ---------------------------------------------------------------- 诊断

def diagnose(base: pd.DataFrame, rd: pd.DataFrame) -> dict:
    """模型是否**已经**吃掉了研发强度：截面秩相关 + Top 名单重合度。"""
    b = base.reindex(index=rd.index, columns=rd.columns)
    r = rd.reindex(index=rd.index, columns=rd.columns)
    rho, cover, top_overlap = [], [], []
    for d in base.index.intersection(rd.index):
        s, t = b.loc[d], r.loc[d]
        ok = s.notna() & t.notna()
        if ok.sum() < 100:
            continue
        rho.append(float(s[ok].corr(t[ok], method="spearman")))
        cover.append(int(ok.sum()) / int(s.notna().sum()))
        # 模型头部（按档位最宽的 K=15）落在 rd 前 10% 的比例
        top = set(s[ok].nlargest(15).index)
        q90 = t[ok].quantile(0.90)
        top_overlap.append(float(np.mean([c in set(t[ok][t[ok] >= q90].index)
                                          for c in top])))
    return {"rho_pred_rd": float(np.mean(rho)),
            "cover_in_model_universe": float(np.mean(cover)),
            "top15_in_rd_top10pct": float(np.mean(top_overlap)),
            "n_days": len(rho)}


# ---------------------------------------------------------------- 显著性

def block_bootstrap(d: pd.Series, block: int = 20, n_boot: int = 4000,
                    seed: int = 20261113) -> dict:
    """日收益差序列的**循环分块** bootstrap（time 轴，不用相位轴）。

    🚨 为什么不能用相位轴 t：21 个相位的日收益相关约 0.82 ⇒ 有效样本量仅 ~1.2，
    相位轴上的标准误被系统性低估（`rtl_rebalance_timing` 已量过）。差异序列本身
    有自相关，故用 20 日分块重采样，按算术年化（×252）给点估计与 95% 区间。
    """
    x = d.dropna().to_numpy()
    k = len(x) // block
    if k < 5:
        return {"ann": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
    blocks = x[: k * block].reshape(k, block)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    means = blocks[idx].mean(axis=(1, 2)) * 252
    point = float(x.mean() * 252)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"ann": point, "lo": float(lo), "hi": float(hi),
            "p": float(min(p, 1.0))}


def analyze(tiers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读 `daily_*.parquet`（逐相位日收益），在 time 轴上判显著性。"""
    rows, yrows = [], []
    for t in tiers:
        f = OUT / f"daily_{t}{_SUFFIX}.parquet"
        if not f.exists():
            continue
        D = pd.read_parquet(f)
        arms = sorted({c.rsplit("__", 1)[0] for c in D.columns})
        avg = {a: D[[c for c in D.columns if c.rsplit("__", 1)[0] == a]].mean(axis=1)
               for a in arms}
        base = avg["base"]
        for a in arms:
            if a == "base":
                continue
            r = block_bootstrap(avg[a] - base)
            # 与安慰剂的直接配对（`pl_rand` 是最保守的那条：同掩码随机值）
            rp = block_bootstrap(avg[a] - avg["pl_rand"]) if "pl_rand" in avg else {}
            rows.append({"tier": t, "arm": a, "delta_ann": r["ann"],
                         "lo": r["lo"], "hi": r["hi"], "p": r["p"],
                         "vs_pl_rand": rp.get("ann", np.nan),
                         "vs_pl_p": rp.get("p", np.nan)})
            # 年度轴：8 个配对观测，与分块 bootstrap 互为对照（两者不一致要写出来）
            y = {}
            for yr in sorted(set(base.index.year)):
                m = base.index.year == yr
                y[yr] = float((1 + avg[a][m]).prod() - (1 + base[m]).prod())
            ys = pd.Series(y)
            n = len(ys)
            tt = (ys.mean() / (ys.std(ddof=1) / np.sqrt(n))) if n > 1 else np.nan
            yrows.append({"tier": t, "arm": a, **{f"y{k}": v for k, v in y.items()},
                          "mean": float(ys.mean()), "t_year": float(tt),
                          "n_pos": int((ys > 0).sum())})
    return pd.DataFrame(rows), pd.DataFrame(yrows)


# ---------------------------------------------------------------- 回测

def run_tier(name: str, ctxs: dict, rd: pd.DataFrame, rd_pl: dict,
             phases: list[int], arms=ARMS) -> pd.DataFrame:
    spec = get_tier(name)
    ctx = ctxs[spec.runnable_signal]
    overrides = {}
    if spec.panel is not None:
        overrides[spec.panel] = ycr.panel_for(spec.panel, ctx)
    capital = ycr.NATIVE_CAPITAL[name]
    px = ycr.price_panel(ctx)
    cost = cal.cost_profile(ycr.PROFILE)
    base = ctx.pred

    rows = []
    dret: dict[str, pd.Series] = {}
    for arm, w, fill, src in arms:
        if src == "rd":
            panel = composite(base, rd, w, fill)
        elif src == "neg":
            panel = composite(base, -rd, w, fill)
        else:
            panel = composite(base, rd_pl[src], w, fill)
        t0 = time.time()
        for p in phases:
            strat = spec.build(**overrides)
            bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                    initial_capital=float(capital), costs=cost)
            res = cal.run_once(bt, ctx, "open", signal=panel, freq="M", phase=p,
                               price_panel=px, lot_size=ycr.LOT)
            m = cal.metrics_row(res, capital)
            rows.append({"tier": name, "arm": arm, "w": w, "fill": fill,
                         "src": src, "phase": p, **m})
            dret[f"{arm}__{p}"] = res.daily_returns.rename(f"{arm}__{p}")
        sub = pd.DataFrame(rows)
        sub = sub[(sub.tier == name) & (sub.arm == arm)]
        print(f"  [{name:<14}] {arm:<10} {len(phases)} 相位均值 "
              f"ann={sub.ann.mean() * 100:7.2f}% "
              f"SR={sub.sharpe_arith.mean():6.3f} 换手={sub.turnover.mean():5.3f} "
              f"持仓={sub.n_active.mean():6.1f} ({time.time() - t0:.0f}s)", flush=True)
    # 日收益落盘：相位轴 t 不可读（相位间日收益相关 0.82，有效样本量仅 1.2），
    # 显著性必须走 time 轴（分块 bootstrap / 逐年配对），故原始序列要留下。
    pd.concat(dret.values(), axis=1).to_parquet(
        OUT / f"daily_{name}{_SUFFIX}.parquet")
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame) -> pd.DataFrame:
    """逐相位配对：Δ_p = arm_p − base_p，给出均值 / std / t / 胜率。"""
    out = []
    for (tier, arm), g in df.groupby(["tier", "arm"]):
        b = df[(df.tier == tier) & (df.arm == "base")].set_index("phase")
        d = (g.set_index("phase")["ann"] - b["ann"]).dropna()
        n = len(d)
        sd = float(d.std(ddof=1))
        out.append({"tier": tier, "arm": arm, "n_phase": n,
                    "ann_base": float(b["ann"].mean()),
                    "ann_arm": float(g["ann"].mean()),
                    "delta": float(d.mean()),
                    "delta_std": sd,
                    "t": float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan,
                    "win": float((d > 0).mean()),
                    "turnover": float(g["turnover"].mean()),
                    "cost_bp": float(g["cost_bp"].mean())})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--phases", default="0-20")
    ap.add_argument("--arms", default=None,
                    help="逗号分隔的臂名（base 会被自动带上）；默认全跑")
    ap.add_argument("--diag-only", action="store_true")
    ap.add_argument("--suffix", default="", help="产物文件名后缀（分批跑时避免覆盖）")
    ap.add_argument("--analyze", action="store_true",
                    help="不跑回测，只读 daily_*.parquet 做 time 轴（分块 bootstrap）检验")
    a = ap.parse_args(argv)
    tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]
    if a.arms:
        want = {x.strip() for x in a.arms.split(",")}
        want.add("base")
        arms = tuple(x for x in ARMS if x[0] in want)
    else:
        arms = ARMS
    if "-" in a.phases:
        lo, hi = a.phases.split("-")
        phases = list(range(int(lo), int(hi) + 1))
    else:
        phases = [int(x) for x in a.phases.split(",")]
    global _SUFFIX
    _SUFFIX = a.suffix
    OUT.mkdir(parents=True, exist_ok=True)

    if a.analyze:
        sig, yr = analyze(tiers)
        sig.to_csv(OUT / f"significance{_SUFFIX}.csv", index=False,
                   encoding="utf-8-sig")
        yr.to_csv(OUT / f"yearly{_SUFFIX}.csv", index=False, encoding="utf-8-sig")
        print("\n=== time 轴显著性（日收益差 20 日分块 bootstrap，×252 算术年化）===")
        print(sig.round(3).to_string())
        print("\n=== 年度轴（Δ 复利，8 年配对）===")
        print(yr.round(4).to_string())
        return 0

    rd_raw = rd_neutralized()
    keys = sorted({get_tier(t).runnable_signal for t in tiers})
    ctxs = {k: cal.load_context(signal=k, window="prod", end=ycr.END_ALIGN,
                                execution="open") for k in keys}

    diag = []
    for k, ctx in ctxs.items():
        d = diagnose(ctx.pred, rd_raw)
        d["signal"] = k
        diag.append(d)
        print(f"\n=== 诊断 {k} ===")
        print(f"  与 rd_to_cap(neu) 截面秩相关 = {d['rho_pred_rd']:+.4f}")
        print(f"  模型池中 rd 有值的比例       = {d['cover_in_model_universe']:.1%}")
        print(f"  模型 Top15 落在 rd 前 10% 的比例 = "
              f"{d['top15_in_rd_top10pct']:.1%}（随机基准 10%）")
    pd.DataFrame(diag).to_csv(OUT / "diagnostics.csv", index=False,
                              encoding="utf-8-sig")
    if a.diag_only:
        return 0

    rd_pl = placebo_panels(rd_raw)
    out = []
    for name in tiers:
        print(f"\n=== {name} ===", flush=True)
        out.append(run_tier(name, ctxs, rd_raw, rd_pl, phases, arms))
    df = pd.concat(out, ignore_index=True)
    df.to_csv(OUT / f"arms_phase{_SUFFIX}.csv", index=False, encoding="utf-8-sig")

    sm = paired(df)
    sm.to_csv(OUT / f"paired{_SUFFIX}.csv", index=False, encoding="utf-8-sig")

    print("\n=== 逐相位配对（Δann vs base，pp；21 相位）===")
    for tier, g in sm.groupby("tier"):
        print(f"\n[{tier}]  base ann = {g.iloc[0]['ann_base'] * 100:.2f}%")
        v = g.set_index("arm")[["delta", "delta_std", "t", "win", "turnover",
                                "cost_bp"]].copy()
        v["delta"] = (v["delta"] * 100).round(2)
        v["delta_std"] = (v["delta_std"] * 100).round(2)
        v["win"] = (v["win"] * 100).round(0)
        print(v.round(3).to_string())

    print("\n判据：Δ ≥ +1.00pp 且 t ≥ 2 且 > 安慰剂 +1pp")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
