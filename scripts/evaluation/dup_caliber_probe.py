"""判重口径探针：同一批因子在三种相关口径下到底有多像。

背景（2026-11-12）
------------------
`scripts/factors/dup_audit.py` 只走 `FactorLibrary._resolve_panel_path`，而该方法
**只解析 `panels/`（原始面板）**。定版策略消费的却是 `all_a_2018_2026/panels_neu_fundind`
（行业+风格中性化后的面板，见 `config/baselines.yaml` 的 `panel_source`）。

于是存在一个没有验证过的隐含假设：**原始面板上的"重复"在中性化面板上仍然重复**。
中性化会剥掉行业与风格暴露，两个原始因子可能因为共同行业成分而 ρ=0.95，
剥掉后只剩 α 部分、ρ 掉到 0.3；反过来也可能升高。若两条口径的排序不一致，
按 `dup_audit` 的结果删因子就是**在错误的层上做决策**。

三个口径
--------
1. ``raw``  —— `panels/`，逐日截面 Spearman 相关（= dup_audit 口径）
2. ``neu``  —— `panels_neu_fundind/`，生产链实际消费的面板
3. ``ic``   —— `ic_h20.parquet` 的 IC 时间序列相关（= factor_library_diversity 口径）

口径 3 便宜且直接衡量"预测力同质性"，但对慢变量不敏感（h20 采样重叠 ⇒ 自相关高）；
口径 1/2 衡量"因子值是否同一个东西"，但对噪声敏感。三者同时看才分得清
"字面重复"（三口径全 1）与"仅共同暴露"（raw 高、neu 低）。

判据（本探针回答的问题）
------------------------
- A：raw 与 neu 的 ρ 排序一致性（Spearman 秩相关），以及同一因子对的 |Δρ| 分布
- B：按 raw 的 |ρ|≥0.95 判为重复的因子对，有多少在 neu 上仍 ≥0.95（移植率）
- C：定版 selection 历史并集里的因子，落在 neu 重复簇内的比例

用法::

    python -m scripts.evaluation.dup_caliber_probe
    python -m scripts.evaluation.dup_caliber_probe --sample-dates 100 --max-codes 1200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("dup_caliber_probe")

LIB_ROOT = Path("e:/data/factor_library/all_a_2018_2026")
SELECTION_DIR = ROOT / "reports" / "alla_rolling_ortho920fundind" / "selection"
OUT_DIR = ROOT / "reports" / "library_governance"

HS = ("h1", "h5", "h10", "h20")


# ---------------------------------------------------------------- 定版因子集
def load_selection() -> tuple[set[str], set[str], dict]:
    """定版滚动 selection：历史并集 + 当期（最后一期）在用集合。"""
    sets: dict[tuple[str, str], list[str]] = {}
    for f in sorted(SELECTION_DIR.glob("y20*__h*.json")):
        if f.name.endswith(".fp.json"):
            continue
        y, h = f.stem.split("__")
        sets[(y, h)] = json.loads(f.read_text(encoding="utf-8"))
    years = sorted({y for y, _ in sets})
    union = set().union(*sets.values())
    last = years[-1]
    cur = set().union(*[set(sets[(last, h)]) for h in HS if (last, h) in sets])
    # 每因子被选中的期数（年 × horizon 共 36 期）
    cnt: dict[str, int] = {}
    for lst in sets.values():
        for n in set(lst):
            cnt[n] = cnt.get(n, 0) + 1
    return union, cur, {"years": years, "n_periods": len(sets), "count": cnt}


# ---------------------------------------------------------------- 样本与签名
def _dates_from(dirpath: Path, names: list[str], k: int) -> pd.DatetimeIndex:
    """用若干代表面板的 index 交集作为真实公共日历，等距抽 k 个。"""
    idx: pd.DatetimeIndex | None = None
    for nm in names[:3] + names[-3:]:
        p = dirpath / f"{nm}.parquet"
        if not p.exists():
            continue
        cur = pd.DatetimeIndex(pd.read_parquet(p, columns=[]).index)
        idx = cur if idx is None else idx.intersection(cur)
    if idx is None:
        raise RuntimeError(f"读不到任何面板日期索引: {dirpath}")
    idx = pd.DatetimeIndex(pd.to_datetime(idx)).normalize().sort_values()
    if len(idx) <= k:
        return idx
    pos = np.linspace(0, len(idx) - 1, k).round().astype(int)
    return idx[np.unique(pos)]


def _codes(max_codes: int, dates: pd.DatetimeIndex) -> list[str]:
    from config import Config

    root = Path(str(Config.cache()["root"]))
    d = pd.read_parquet(root / "daily_all_a.parquet", columns=["amount"]).reset_index()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d[d["date"].isin(set(dates))]
    g = d.groupby("code")
    pool = pd.DataFrame({"cover": g["date"].nunique() / max(1, len(dates)),
                         "amount": g["amount"].median()})
    pool = pool.sort_values(["cover", "amount"], ascending=False)
    if len(pool) > max_codes:
        log.info("股票池截断 %d → %d", len(pool), max_codes)
        pool = pool.head(max_codes)
    return pool.index.tolist()


def _signatures(dirpath: Path, names: list[str], dates: pd.DatetimeIndex,
                codes: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(K,F,C) 签名 + 实际读到的名 + 每个面板的非空率。"""
    from scripts.factors.dup_audit import _row_rank_zscore

    Z = np.zeros((len(dates), len(names), len(codes)), dtype=np.float32)
    cover = np.zeros(len(names), dtype=np.float32)
    got: list[str] = []
    t0 = time.time()
    for fi, nm in enumerate(names):
        p = dirpath / f"{nm}.parquet"
        if not p.exists():
            got.append(nm)
            continue
        try:
            df = pd.read_parquet(p).reindex(index=dates, columns=codes)
        except Exception as e:  # noqa: BLE001
            log.warning("读取失败 %s: %s", nm, e)
            got.append(nm)
            continue
        a = df.to_numpy(dtype=np.float64)
        cover[fi] = float(np.isfinite(a).sum()) / max(1, a.size)
        Z[:, fi, :] = _row_rank_zscore(a)
        got.append(nm)
        if (fi + 1) % 50 == 0:
            log.info("签名 %d/%d（%.0fs）", fi + 1, len(names), time.time() - t0)
    return Z, got, cover


def _corr_from_sig(Z: np.ndarray) -> np.ndarray:
    from scripts.factors.dup_audit import correlation_matrix
    return correlation_matrix(Z)


def _ic_corr(names: list[str], h: int = 20) -> tuple[np.ndarray, list[str]]:
    p = LIB_ROOT / f"ic_h{h}.parquet"
    ic = pd.read_parquet(p)
    have = [n for n in names if n in ic.columns]
    sub = ic[have].dropna(axis=0, how="all")
    # 秩相关对非线性单调关系更稳，且与值口径可比
    c = sub.rank().corr().to_numpy(dtype=np.float64)
    return c, have


# ---------------------------------------------------------------- 分析
def pair_stats(c1: np.ndarray, c2: np.ndarray, names: list[str],
               keep: list[str], thresh: float) -> dict:
    """只在 keep 名单上比较两个矩阵（索引对齐后取上三角）。"""
    i1 = {n: i for i, n in enumerate(names)}
    idx = [i1[n] for n in keep]
    a = c1[np.ix_(idx, idx)].astype(np.float64)
    b = c2[np.ix_(idx, idx)].astype(np.float64)
    iu = np.triu_indices(len(idx), k=1)
    av, bv = a[iu], b[iu]
    ok = np.isfinite(av) & np.isfinite(bv)

    def _rank(x):
        return pd.Series(x).rank().to_numpy()

    rank_r = float(np.corrcoef(_rank(av[ok]), _rank(bv[ok]))[0, 1])
    hi = ok & (np.abs(av) >= thresh)
    transplanted = int((hi & (np.abs(bv) >= thresh)).sum())
    return {
        "n_pairs": int(ok.sum()),
        "rank_corr": rank_r,
        "mean_abs_drho": float(np.mean(np.abs(bv[ok] - av[ok]))),
        "p95_abs_drho": float(np.percentile(np.abs(bv[ok] - av[ok]), 95)),
        "n_hi_raw": int(hi.sum()),
        "n_hi_neu": int((ok & (np.abs(bv) >= thresh)).sum()),
        "transplant": transplanted,
        "transplant_rate": transplanted / max(1, int(hi.sum())),
    }


def clusters_named(corr: np.ndarray, names: list[str], thresh: float) -> dict:
    from scripts.factors.dup_audit import clusters
    cl = clusters(corr, names, thresh)
    out: dict[str, list[str]] = {}
    for cid, g in cl.groupby("cluster"):
        m = g["name"].tolist()
        if len(m) >= 2:
            out[f"c{int(cid)}"] = m
    return out


def write_report(res: dict, meta: dict, out: Path) -> None:
    ps, ps_ic = res["pair_raw_neu"], res["pair_raw_ic"]
    lines = [
        "# 去重判据口径探针（判重层 vs 生产层）",
        "",
        f"> 生成于 {pd.Timestamp.now():%Y-%m-%d %H:%M} · "
        f"样本 {meta['n_dates']} 日 × {meta['n_codes']} 只 · "
        f"因子 {meta['n_factors']} 个（定版 selection 历史并集）",
        "",
        "## 〇 结论先行",
        "",
    ]
    rate = ps["transplant_rate"]
    verdict = ("**判重层需改**" if rate < 0.7 else "**判重层基本可移植**")
    lines += [
        f"1. 原始面板判出的重复，在**生产消费的中性化面板**上只有 "
        f"**{rate:.1%}** 仍 ≥ {meta['thresh']:.2f} ⇒ {verdict}。",
        f"2. 两口径的相关矩阵秩一致度 **{ps['rank_corr']:.3f}**"
        f"（1.0 = 排序完全一致），同一因子对平均 |Δρ| = **{ps['mean_abs_drho']:.3f}**，"
        f"95 分位 **{ps['p95_abs_drho']:.3f}**。",
        f"3. 与 IC 序列口径的一致度 **{ps_ic['rank_corr']:.3f}** —— "
        "值相关与预测力同质是两件事，互相不能替代。",
        "",
        "## 一、三口径的重复对数",
        "",
        "| 口径 | |ρ| ≥ 阈值 的对数 | 说明 |",
        "|---|---:|---|",
        f"| raw（`panels/`，dup_audit 现口径） | {ps['n_hi_raw']} | 现行去重依据 |",
        f"| neu（`panels_neu_fundind/`，生产层） | {ps['n_hi_neu']} | **应在此层判重** |",
        f"| ic（`ic_h20` 序列） | {ps_ic['n_hi_raw']} | 预测力同质 |",
        "",
        f"移植率 = raw 判重复的对里，neu 仍判重复的比例 = **{rate:.1%}**。"
        "低于 70% 说明中性化显著改变了相关结构，按 raw 删因子会误杀。",
        "",
        "## 二、raw → neu 的变化最大的因子对（top 20）",
        "",
        "| 因子 A | 因子 B | ρ_raw | ρ_neu | Δ |",
        "|---|---|---:|---:|---:|",
    ]
    for a, b, r, n in res["top_delta"]:
        lines.append(f"| `{a}` | `{b}` | {r:+.3f} | {n:+.3f} | {n-r:+.3f} |")
    lines += [
        "",
        "## 三、neu 层重复簇与定版 selection 的关系",
        "",
        f"neu 层 |ρ| ≥ {meta['thresh']:.2f} 的簇：**{len(res['neu_clusters'])}** 个，"
        f"覆盖 **{sum(len(v) for v in res['neu_clusters'].values())}** 个因子；"
        f"其中含定版**当期在用**成员的簇 **{res['clusters_with_current']}** 个。",
        "",
        "| 簇 | 成员数 | 含当期在用 | 成员 |",
        "|---|---:|---|---|",
    ]
    for cid, members in sorted(res["neu_clusters"].items(),
                               key=lambda kv: -len(kv[1]))[:15]:
        cur = [m for m in members if m in res["current"]]
        lines.append(f"| {cid} | {len(members)} | {len(cur)} | "
                     f"{'|'.join(members[:6])}{'…' if len(members) > 6 else ''} |")
    lines += [
        "",
        "## 四、口径与局限",
        "",
        f"1. 日期等距抽 {meta['n_dates']} 个，股票池按覆盖度+成交额取 {meta['n_codes']} 只；"
        "NaN 记 0 ⇒ 相关偏低估（漏报不误报）。",
        "2. `neu` 面板是 `panels_neu_fundind`（中性化 + 基本面保留席位），"
        "即 `config/baselines.yaml` 中 `panel_source` 指向的生产输入。",
        "3. `ic` 口径用 h20 且日频重叠 ⇒ 自相关抬高所有 ρ，只作旁证不比绝对值。",
        "4. 本表只回答「在哪一层像」，不回答「删了会不会亏」——那要组合层 A/B。",
        "",
    ]
    out.mkdir(parents=True, exist_ok=True)
    (out / "dup_caliber_probe.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("报告：%s", out / "dup_caliber_probe.md")


def main() -> None:
    ap = argparse.ArgumentParser(description="判重口径探针")
    ap.add_argument("--sample-dates", type=int, default=120)
    ap.add_argument("--max-codes", type=int, default=1500)
    ap.add_argument("--thresh", type=float, default=0.95)
    args = ap.parse_args()

    union, current, sel_meta = load_selection()
    names = sorted(union)
    log.info("定版 selection：历史并集 %d，当期在用 %d，期数 %d",
             len(union), len(current), sel_meta["n_periods"])

    d_raw = LIB_ROOT / "panels"
    d_neu = LIB_ROOT / "panels_neu_fundind"
    missing_raw = [n for n in names if not (d_raw / f"{n}.parquet").exists()]
    missing_neu = [n for n in names if not (d_neu / f"{n}.parquet").exists()]
    log.info("缺面板：raw %d 个 / neu %d 个", len(missing_raw), len(missing_neu))

    dates = _dates_from(d_neu, [n for n in names if n not in missing_neu], args.sample_dates)
    codes = _codes(args.max_codes, dates)
    log.info("样本 %d 日 × %d 只", len(dates), len(codes))

    Zraw, _, cov_raw = _signatures(d_raw, names, dates, codes)
    Craw = _corr_from_sig(Zraw)
    log.info("raw 相关矩阵完成")
    Zneu, _, cov_neu = _signatures(d_neu, names, dates, codes)
    Cneu = _corr_from_sig(Zneu)
    log.info("neu 相关矩阵完成")

    # IC 口径只覆盖有的列
    Cic, ic_names = _ic_corr(names, h=20)
    log.info("IC 口径覆盖 %d/%d 个因子", len(ic_names), len(names))

    # 两两对比：raw↔neu 用全名单，raw↔ic 用 ic 覆盖的名单
    ps_rn = pair_stats(Craw, Cneu, names, names, args.thresh)
    common_ic = [n for n in names if n in set(ic_names)]
    i1 = {n: i for i, n in enumerate(ic_names)}
    # 把 Craw 重排到 ic 顺序
    ia = {n: i for i, n in enumerate(names)}
    Craw_ic = Craw[np.ix_([ia[n] for n in ic_names], [ia[n] for n in ic_names])]
    ps_ri = pair_stats(Craw_ic, Cic, ic_names, ic_names, args.thresh)
    log.info("raw↔neu 移植率 %.1f%%；raw↔ic 秩一致 %.3f",
             100 * ps_rn["transplant_rate"], ps_ri["rank_corr"])

    # 变化最大的对
    iu = np.triu_indices(len(names), k=1)
    d = Cneu[iu].astype(np.float64) - Craw[iu].astype(np.float64)
    order = np.argsort(-np.abs(d))[:20]
    top = [(names[iu[0][o]], names[iu[1][o]], float(Craw[iu][o]), float(Cneu[iu][o]))
           for o in order]

    neu_cl = clusters_named(Cneu, names, args.thresh)
    n_with_cur = sum(1 for m in neu_cl.values() if any(x in current for x in m))

    res = {"pair_raw_neu": ps_rn, "pair_raw_ic": ps_ri, "top_delta": top,
           "neu_clusters": neu_cl, "clusters_with_current": n_with_cur,
           "current": current}
    meta = {"n_dates": len(dates), "n_codes": len(codes), "n_factors": len(names),
            "thresh": args.thresh}
    write_report(res, meta, OUT_DIR)

    print("\n=== 判重口径探针 ===")
    print(f"raw 判重复对 {ps_rn['n_hi_raw']} → neu 仍重复 {ps_rn['n_hi_neu']}"
          f"（移植率 {ps_rn['transplant_rate']:.1%}）")
    print(f"raw↔neu 秩一致 {ps_rn['rank_corr']:.3f}，平均 |Δρ| {ps_rn['mean_abs_drho']:.3f}")
    print(f"raw↔ic  秩一致 {ps_ri['rank_corr']:.3f}")
    print(f"neu 重复簇 {len(neu_cl)} 个，其中 {n_with_cur} 个含定版当期在用因子")
    print(f"报告 → {OUT_DIR/'dup_caliber_probe.md'}")


if __name__ == "__main__":
    main()
