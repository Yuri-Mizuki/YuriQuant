"""focus + 行业去重（每申万一级 ≤N 只）的生产口径完整评估。

背景（2026-11-13 追问）
----------------------
`sector_constraint_tier` 的初步结果：focus 去掉行业约束后最大行业权重 **25.2%**（p90 40%、
max 60%），比 solo 定版的 17.4% 更集中；而 `focus + 每行业≤1` 的 21 相位均值
**25.68% / SR 0.9709 / 持仓 9.16** 在四项上都优于 solo 定版（25.07 / 0.9296 / 5.83）。
但那只做了「21 相位配对均值 + 配对 t=1.46」，**缺两道项目硬要求**：
持仓数匹配对照、安慰剂。本脚本补齐，参数网格一并跑。

预注册判据（写死，先于结果）
----------------------------
准入需**同时**满足：
1. ``sec1 − base`` 的 21 相位配对均值 **Δann ≥ +1.00pp**，且 **≥ 14/21 相位为正**
2. **持仓数匹配对照**：``sec1`` 必须优于「无约束但持仓数相当」的臂（`k9_match`/`k10_match`）
   —— 否则增量只是「少持几只 ⇒ 杠杆放大」，与行业无关
3. **随机行业安慰剂**：``sec1`` 必须优于「同样的约束强度 + 打乱行业标签」的臂（`pl_rand`）
   —— 否则增量来自「多一层筛选」而非行业信息
4. **time 轴显著**：日收益差 20 日分块 bootstrap 的 95% 区间不含 0（相位轴有效样本量仅
   ~1.2，不可用作显著性 ⇒ 见 `reports/rtl_rebalance_timing/README.md`）

四条全过才判「可进影子/定版」；过 1/2/3 但不过 4 ⇒ 判「一致但未显著」。

口径
----
production 27bp 时变 · T+1 open · 整手 100 · focus 资金 5e5 · 窗口 prod · 末日 2026-09-07。
`base` / `solo_base` 必须逐位复现 `yq_core` 的 29.40% / 36.58%（phase 0）。

用法::

    python -m scripts.evaluation.focus_sector_probe --phases 0        # 冒烟（约 20s）
    python -m scripts.evaluation.focus_sector_probe --phases 0-20     # 全量（约 6min）
    python -m scripts.evaluation.focus_sector_probe --analyze         # 只做显著性
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.evaluation.caliber as cal
from scripts.evaluation import yq_core_report as ycr

OUT = Path("reports/focus_sector")
SEED = 20261113

#: (臂名, 资金档, 构造) 。``None``=无约束 BufferedTopKLongOnly(k, exit_mult)。
ARMS: tuple[tuple[str, str, dict | None], ...] = (
    ("base",       "yqcore-focus", None),
    ("sec1",       "yqcore-focus", dict(k=15, exit_mult=2.0, max_per_sector=1)),
    ("sec2",       "yqcore-focus", dict(k=15, exit_mult=2.0, max_per_sector=2)),
    ("sec3",       "yqcore-focus", dict(k=15, exit_mult=2.0, max_per_sector=3)),
    ("k20_sec1",   "yqcore-focus", dict(k=20, exit_mult=2.0, max_per_sector=1)),
    ("k12_sec1",   "yqcore-focus", dict(k=12, exit_mult=2.0, max_per_sector=1)),
    ("k8_plain",   "yqcore-focus", None),   # 持仓数匹配对照：无约束 K8（配 k12_sec1）
    ("k9_match",   "yqcore-focus", None),   # 持仓数匹配对照（无约束）
    ("k10_match",  "yqcore-focus", None),   # 同上
    ("pl_rand",    "yqcore-focus", dict(k=15, exit_mult=2.0, max_per_sector=1)),
    ("pl_rand_k12", "yqcore-focus", dict(k=12, exit_mult=2.0, max_per_sector=1)),
    ("solo_base",  "yqcore-solo",  dict(k=8, exit_mult=1.0, max_per_sector=1)),
)
#: 每个臂的 k / exit_mult（``None`` 的 kwargs 也要知道用几）。
K_EXIT = {"base": (15, 2.0), "k8_plain": (8, 2.0), "k9_match": (9, 2.0),
          "k10_match": (10, 2.0)}


def industry_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    ind = pd.read_parquet(cal.BASE_DIR / "industry.parquet")
    return ind.reindex(index=ctx.pred.index, columns=ctx.pred.columns)


def shuffle_industry(ind: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """逐日打乱行业标签：保留每日行业边际分布，破坏「股票↔行业」对应。

    这是行业约束的**机制安慰剂**：若真实行业约束只是「多一层筛选」，它应与
    随机分组一样好；若行业信息真有用，真实臂应显著更优。
    """
    rng = np.random.default_rng(seed)
    out = ind.copy()
    vals = out.to_numpy(dtype=object, copy=True)
    for i in range(vals.shape[0]):
        row = vals[i]
        ok = pd.notna(row)
        n = int(ok.sum())
        if n > 1:
            vals[i, ok] = rng.permutation(row[ok])
    return pd.DataFrame(vals, index=ind.index, columns=ind.columns)


def block_bootstrap(diff: pd.Series, block: int = 20, n_boot: int = 4000,
                    seed: int = SEED) -> dict:
    """日收益差的**分块** bootstrap（保留自相关）。返回年化 Δ 与 95% 区间。"""
    x = diff.dropna().to_numpy()
    n = len(x)
    if n < block * 3:
        return {"ann": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
    nb = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)
    samp = x[idx[:, :n]]
    est = samp.mean(axis=1) * 252
    lo, hi = np.percentile(est, [2.5, 97.5])
    p = float(2 * min((est <= 0).mean(), (est >= 0).mean()))
    return {"ann": float(x.mean() * 252), "lo": float(lo), "hi": float(hi),
            "p": p}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="0")
    ap.add_argument("--arms", default=",".join(a[0] for a in ARMS))
    ap.add_argument("--capital", type=float, default=None,
                    help="覆盖档位 native 资金（判跨档可比性用；默认按 tier 取）")
    ap.add_argument("--suffix", default="",
                    help="产物后缀，避免覆盖同臂不同资金的 daily/phase 文件")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args(argv)

    ctx = cal.load_context(signal="gbdt_mid13", window="prod", end=20260907,
                           execution="open")
    px = ycr.price_panel(ctx)
    ind = industry_panel(ctx)
    cost = cal.cost_profile(ycr.PROFILE)

    if a.analyze:
        return analyze()

    from strategy.examples import BufferedTopKLongOnly, SectorCappedBufferedTopKLongOnly

    phases = (list(range(int(a.phases.split("-")[0]), int(a.phases.split("-")[1]) + 1))
              if "-" in a.phases
              else [int(x) for x in a.phases.split(",")])
    want = {x.strip() for x in a.arms.split(",") if x.strip()}

    ind_pl = None
    for arm, tier, kw in ARMS:
        if arm not in want:
            continue
        capital = (a.capital if a.capital is not None
                   else float(ycr.NATIVE_CAPITAL[tier]))
        sectors = ind
        if arm.startswith("pl_rand"):
            if ind_pl is None:
                print("  构造随机行业 placebo 面板 ...", flush=True)
                ind_pl = shuffle_industry(ind)
            sectors = ind_pl
        k, em = K_EXIT.get(arm, ((kw or {}).get("k"), (kw or {}).get("exit_mult")))
        rets, daily = {}, {}
        for p in phases:
            t0 = time.time()
            strat = (BufferedTopKLongOnly(k=k, exit_mult=em) if kw is None
                     else SectorCappedBufferedTopKLongOnly(sectors=sectors, **kw))
            bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                    initial_capital=capital, costs=cost)
            res = cal.run_once(bt, ctx, "open", freq="M", phase=p,
                               price_panel=px, lot_size=ycr.LOT)
            m = cal.metrics_row(res, capital)
            rets[p] = {"ann": m["ann"], "sr": m["sharpe_arith"],
                       "max_dd": m["max_dd"], "turnover": m["turnover"],
                       "cost_bp": m["cost_bp"], "n_hold": m["n_active"],
                       "phase": p, "arm": arm, "tier": tier}
            daily[p] = res.daily_returns
            print(f"  [{arm:<10}] p={p:<3} ann={m['ann'] * 100:7.2f}% "
                  f"SR={m['sharpe_arith']:6.3f} 持仓={m['n_active']:6.1f} "
                  f"换手={m['turnover']:5.3f} ({time.time() - t0:.0f}s)", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        sfx = a.suffix
        pd.DataFrame(daily).to_parquet(OUT / f"daily_{arm}{sfx}.parquet")
        pd.DataFrame(list(rets.values())).to_csv(
            OUT / f"phase_{arm}{sfx}.csv", index=False, encoding="utf-8-sig")
    return 0


def analyze() -> int:
    """相位配对 + time 轴分块 bootstrap。相位轴有效样本量 ~1.2，不能当显著性。"""
    files = sorted(OUT.glob("phase_*.csv"))
    if not files:
        print("没有 phase_*.csv，先跑回测")
        return 1
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    piv = df.pivot_table(index="phase", columns="arm", values="ann")

    print("\n=== 21 相位均值（production 口径）===")
    sm = df.groupby("arm")[["ann", "sr", "max_dd", "turnover", "cost_bp", "n_hold"]].mean()
    sm["ann_std"] = df.groupby("arm")["ann"].std(ddof=1)
    sm = sm.sort_values("ann", ascending=False)
    print(sm.round(4).to_string())

    base = "base"
    rows = []
    for arm in piv.columns:
        if arm == base:
            continue
        x = (piv[arm] - piv[base]).dropna()
        rows.append({"arm": arm, "delta_ann": x.mean() * 100,
                     "sd": x.std(ddof=1) * 100,
                     "t": float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))),
                     "pos": int((x > 0).sum()), "n": len(x)})
    out = pd.DataFrame(rows).sort_values("delta_ann", ascending=False)
    print("\n=== 相位配对 Δann vs base（pp）===")
    print(out.round(3).to_string(index=False))

    # time 轴：各臂先对 21 相位取日收益均值（= 等权叠加），再与 base 配对
    dr = {}
    for f in sorted(OUT.glob("daily_*.parquet")):
        arm = f.stem.replace("daily_", "")
        D = pd.read_parquet(f)
        dr[arm] = D.mean(axis=1)
    if base in dr:
        sig = []
        for arm, s in dr.items():
            if arm == base:
                continue
            r = block_bootstrap(s - dr[base])
            sig.append({"arm": arm, "delta_ann": r["ann"], "lo": r["lo"],
                        "hi": r["hi"], "p": r["p"],
                        "含0": bool(r["lo"] <= 0 <= r["hi"])})
        sg = pd.DataFrame(sig).sort_values("delta_ann", ascending=False)
        print("\n=== time 轴：21 相位等权叠加的日收益差，20 日分块 bootstrap ===")
        print(sg.round(3).to_string(index=False))
        sg.to_csv(OUT / "significance.csv", index=False, encoding="utf-8-sig")
        out = out.merge(sg, on="arm", how="left")

    out.to_csv(OUT / "paired.csv", index=False, encoding="utf-8-sig")
    sm.to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
