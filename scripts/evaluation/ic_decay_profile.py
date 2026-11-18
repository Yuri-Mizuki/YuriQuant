"""因子 IC 衰减剖面：有效期到底有多长（纯信号口径，**不含成本、不含组合**）
================================================================================

为什么必须与 §9.3 分开测
------------------------
§9.3 那张「净收益随 H 变化」的表是**复合量**：净 = 毛 − 成本，
而毛既含「信号衰减」又含「组合构造（topK 等权）的非线性」。
所以从那张表读不出「因子有效期」，只能读出「这条策略在哪个持有期能赚钱」。

本脚本只问一件事：**IC(h) 相对 IC(1) 衰减多快。**
不含成本、不含 topK、不含 buffer —— 这是因子的**信息半衰期**。

⚠️ 但需要两个附加量才能判读，否则会被 horizon 口径骗：

1. **重叠窗口会机械放大 |t|**：h=20 的 IC 序列用 20 日重叠收益算，
   有效独立样本只有 n/20 ⇒ 必须报 ``t/√h``（本项目 §九 已踩过这个坑）。
2. **IC 大小的横向不可比**：不同因子的 IC 量级差几个数量级，
   衰减倍率 ``IC(h)/IC(1)`` 才可比。但也因此**分母很小时倍率会爆**，
   所以只保留 ``|IC(1)|`` 排名靠前且稳定的因子。

判读口径（本脚本先定死，避免事后挑）
------------------------------------
- **能量半衰期** ``h_half``：使 ``|IC(h)|/|IC(1)| = 0.5`` 的 h（线性插值，超范围则记 ``>20``）。
- **衰减形态**：
  - ``fast``：h=5 已跌到 <30%（≈1 天生命）
  - ``slow``：h=20 仍 >50%（长记忆）
  - ``mid``：其余
- **有效方向数** ``n_eff``（participation ratio ``(Σλ)²/Σλ²``）：
  在每个 h 上各算一次。**若衰减快，n_eff 会随 h 上升**——因为快因子在长 H 上
  退化成噪声，噪声彼此不相关 ⇒ 谱变平 ⇒ n_eff 虚高。
  🚨 这是 n_eff 的一个**重要陷阱**，必须同时报 h=1 与 h=20 两个值。

用法::

    .venv/Scripts/python.exe -m scripts.evaluation.ic_decay_profile
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("ic_decay_profile")

LIB = Path(str(Config.get()["factor_library"]["root"])) / "all_a_2018_2026"
OUT = ROOT / "reports" / "intraday_daily"

HS: tuple[int, ...] = (1, 5, 10, 20)
MIN_OBS = 250       # IC 序列最少非空天数
MIN_ABS_IC1 = 0.01  # |IC(1)| 门槛：低于此倍率无意义（分母太小）


def _nw_t(x: pd.Series, lags: int) -> tuple[float, float]:
    """均值 + Newey-West t（重叠窗口必用）。"""
    x = x.dropna()
    n = len(x)
    if n < 30:
        return float("nan"), float("nan")
    mu = float(x.mean())
    e = (x - mu).to_numpy()
    var = float((e @ e) / n)
    for k in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * float((e[k:] @ e[:-k]) / n)
    if var <= 0:
        return mu, float("nan")
    return mu, mu / np.sqrt(var / n)


def _pr(corr: np.ndarray) -> float:
    """participation ratio = (Σλ)²/Σλ²。"""
    lam = np.clip(np.linalg.eigvalsh(corr), 1e-12, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def n_eff_at(ic: pd.DataFrame, cols: list[str]) -> tuple[float, float]:
    """某 h 上的 n_eff 与 PC1 解释方差比。"""
    sub = ic[cols]
    c = np.nan_to_num(sub.corr(min_periods=MIN_OBS).to_numpy(), nan=0.0)
    np.fill_diagonal(c, 1.0)
    lam, vec = np.linalg.eigh(c)
    lam = np.clip(lam, 1e-12, None)
    cp = (vec * lam) @ vec.T
    d = np.sqrt(np.diag(cp))
    cp = cp / np.outer(d, d)
    lam2 = np.clip(np.linalg.eigvalsh(cp), 1e-12, None)
    return _pr(cp), float(lam2.max() / lam2.sum())


def half_life(ratio: dict[int, float]) -> float:
    """使 |IC(h)|/|IC(1)| 降到 0.5 的 h（线性插值；单调前提下有效）。"""
    if ratio.get(1, 0) <= 0:
        return float("nan")
    hs = sorted(ratio)
    for a, b in zip(hs, hs[1:]):
        ra, rb = ratio[a], ratio[b]
        if ra >= 0.5 >= rb:
            if ra == rb:
                return float(a)
            return a + (ra - 0.5) / (ra - rb) * (b - a)
    return float(max(hs)) if ratio[max(hs)] >= 0.5 else float(min(hs))


def main() -> None:
    ap = argparse.ArgumentParser(description="因子 IC 衰减剖面")
    ap.add_argument("--top", type=int, default=25, help="按 |IC(1)| 取前 N 个展示")
    args = ap.parse_args()

    ics = {h: pd.read_parquet(LIB / f"ic_h{h}.parquet") for h in HS}
    reg = pd.read_csv(LIB / "registry.csv")
    fam = dict(zip(reg["name"], reg.get("family", pd.Series(dtype=object))))

    # 样本内 IC 均值与 NW t（h 档各自校正）
    stat: dict[int, pd.DataFrame] = {}
    for h in HS:
        d = ics[h]
        rows = {}
        for c in d.columns:
            s = d[c]
            if s.notna().sum() < MIN_OBS:
                continue
            mu, t = _nw_t(s, lags=max(h * 2, 5))
            rows[c] = {"ic": mu, "t": t, "t_adj": t / np.sqrt(h), "n": int(s.notna().sum())}
        stat[h] = pd.DataFrame(rows).T
        log.info("h=%2d 有效因子 %d", h, len(stat[h]))

    # 只保留四档都有的因子
    common = set(stat[1].index)
    for h in HS[1:]:
        common &= set(stat[h].index)
    common = sorted(common)
    log.info("四档齐全因子 %d 个", len(common))

    df = pd.DataFrame({f"ic{h}": stat[h].loc[common, "ic"] for h in HS})
    df["t1"] = stat[1].loc[common, "t"]
    df["t1_adj"] = stat[1].loc[common, "t_adj"]
    df["family"] = [fam.get(c, "") for c in df.index]

    # 方向统一：以 h=1 的符号为准，翻正后算衰减（否则正负抵消）
    sign = np.sign(df["ic1"]).replace(0, 1)
    for h in HS:
        df[f"r{h}"] = df[f"ic{h}"] * sign / df["ic1"].abs()

    cand = df[(df["ic1"].abs() >= MIN_ABS_IC1) & (df["t1"].abs() >= 2.0)].copy()
    log.info("候选（|IC1|≥%.3f 且 |t1|≥2）%d 个", MIN_ABS_IC1, len(cand))

    cand["h_half"] = cand.apply(
        lambda r: half_life({h: float(r[f"r{h}"]) for h in HS}), axis=1)

    def shape(r: pd.Series) -> str:
        if r["r5"] < 0.30:
            return "fast"
        if r["r20"] > 0.50:
            return "slow"
        return "mid"

    cand["shape"] = cand.apply(shape, axis=1)

    # ---------- 输出 1：分族汇总 ----------
    print("=" * 78)
    print("IC 衰减剖面（纯信号口径：无成本、无 topK、无 buffer）")
    print(f"  因子库 {len(common)} 个四档齐全，候选 {len(cand)} 个（|IC1|≥{MIN_ABS_IC1}, |t1|≥2）")
    print("=" * 78)
    print("\n【一】分族：IC 衰减形态")
    print("  %-14s %4s %8s %8s %8s %8s  %s" % (
        "族", "n", "IC1", "r5", "r10", "r20", "半衰期中位"))
    if len(cand):
        g = cand.groupby("family").agg(
            n=("ic1", "size"), ic1=("ic1", lambda s: float(s.abs().median())),
            r5=("r5", "median"), r10=("r10", "median"), r20=("r20", "median"),
            hh=("h_half", "median")).sort_values("n", ascending=False)
        for k, r in g.iterrows():
            print("  %-14s %4d %+8.4f %8.2f %8.2f %8.2f  %.1f" % (
                str(k)[:14], int(r["n"]), r["ic1"], r["r5"], r["r10"], r["r20"], r["hh"]))

    # ---------- 输出 2：逐因子 ----------
    print("\n【二】按 |IC(1)| 前 %d 个逐因子（r_h = 同向 IC(h)/|IC(1)|）" % args.top)
    print("  %-24s %-10s %8s %7s %6s %6s %6s %7s %s" % (
        "因子", "族", "IC1", "t(NW)", "r5", "r10", "r20", "h_half", "形态"))
    top = cand.reindex(cand["ic1"].abs().sort_values(ascending=False).index)[:args.top]
    for n, r in top.iterrows():
        print("  %-24s %-10s %+8.4f %+7.2f %6.2f %6.2f %6.2f %7.1f %s" % (
            str(n)[:24], str(r["family"])[:10], r["ic1"], r["t1"],
            r["r5"], r["r10"], r["r20"], r["h_half"], r["shape"]))

    # ---------- 输出 3：衰减形态分布 ----------
    print("\n【三】衰减形态分布（fast=r5<0.30；slow=r20>0.50；其余 mid）")
    if len(cand):
        vc = cand["shape"].value_counts()
        for k in ("fast", "mid", "slow"):
            n = int(vc.get(k, 0))
            print("  %-6s %4d 个 (%.1f%%)" % (k, n, 100 * n / len(cand)))

    # ---------- 输出 4：n_eff 随 h 的变化（陷阱） ----------
    print("\n【四】n_eff 随 h 的变化（participation ratio）")
    print("  🚨 快因子在长 H 退化为噪声 ⇒ 噪声互不相关 ⇒ n_eff 虚高，不能读成「更分散」")
    print("  %-8s %8s %8s %10s %8s" % ("h", "n_eff", "ratio", "PC1占比", "因子数"))
    neff_rows = []
    for h in HS:
        cols = [c for c in common if stat[h].loc[c, "n"] >= MIN_OBS]
        ne, pc1 = n_eff_at(ics[h], cols)
        print("  %-8d %8.2f %8.2f %10.2f %8d" % (
            h, ne, ne / len(cols), pc1, len(cols)))
        neff_rows.append({"h": h, "n_eff": ne, "ratio": ne / len(cols),
                          "pc1": pc1, "n": len(cols)})

    # ---------- 输出 5：同族内衰减一致性 ----------
    print("\n【五】同族内衰减是否一致（族内 r5 的离散度）")
    print("  判读：族内 std 大 ⇒ 「族的有效期」这个说法不成立，因子各衰各的")
    if len(cand):
        g2 = cand.groupby("family").agg(n=("r5", "size"), r5_std=("r5", "std")).dropna()
        g2 = g2[g2["n"] >= 3].sort_values("r5_std", ascending=False)
        for k, r in g2.iterrows():
            print("  %-14s n=%-4d r5 std=%.3f" % (str(k)[:14], int(r["n"]), r["r5_std"]))

    # ---------- 输出 6：🚨 重叠窗口的机械效应（决定性） ----------
    print("\n【六】🚨 重叠窗口校正：r_h > 1 是假象，不能读成「信号变强」")
    print("  IC(h) 用 h 日**重叠**收益算 ⇒ IC 序列高度自相关 ⇒ 有效独立天数只有 n/h")
    print("  %-6s %7s %8s %10s %9s %10s %9s" % (
        "h", "n_days", "IC自相关", "有效天数", "中位|IC|", "IC/√有效", "相对h=1"))
    corr_rows = []
    base = None
    for h in HS:
        d = ics[h]
        acs, mus = [], []
        for c in d.columns:
            s = d[c].dropna()
            if len(s) < MIN_OBS:
                continue
            a = s.autocorr(1)
            if not np.isfinite(a):
                continue
            acs.append(float(a))
            mus.append(float(s.mean()))
        ac = float(np.nanmedian(acs))
        neff_days = len(d) * (1 - ac) / (1 + ac)
        mic = float(np.nanmedian(np.abs(mus)))
        q = mic / np.sqrt(neff_days)
        if base is None:
            base = q
        print("  %-6d %7d %8.3f %10.1f %9.4f %10.6f %9.3f" % (
            h, len(d), ac, neff_days, mic, q, q / base))
        corr_rows.append({"h": h, "n_days": len(d), "ac1": ac,
                          "n_eff_days": neff_days, "med_abs_ic": mic,
                          "ic_per_sqrt_neff": q, "rel_to_h1": q / base})
    print("\n  🔑 判读：若「IC/√有效」随 h 单调上升，说明 |IC| 的增长**追不上**")
    print("     有效样本的收缩 ⇒ 长 H 的 IC 是「少数几次窗口内的巧合」，不是更稳的信号。")

    # 落盘
    cand.to_csv(OUT / "ic_decay_profile.csv", encoding="utf-8-sig")
    pd.DataFrame(neff_rows).to_csv(OUT / "ic_decay_neff.csv", index=False,
                                   encoding="utf-8-sig")
    pd.DataFrame(corr_rows).to_csv(OUT / "ic_decay_overlap.csv", index=False,
                                   encoding="utf-8-sig")
    print("\n已写 ic_decay_profile.csv / ic_decay_neff.csv / ic_decay_overlap.csv")


if __name__ == "__main__":
    main()