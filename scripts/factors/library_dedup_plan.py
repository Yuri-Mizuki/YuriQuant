"""因子库去重决策表：把「相关 → 质量 → 生产依赖」三条判据并成一张可拍板的清单。

为什么不直接用 `dup_audit` 的结论
----------------------------------
`dup_audit` 回答的是「两个因子有多像」，它的输出是**几何结构**（簇）。但
「删不删」还取决于两件它不管的事：

1. **质量**：簇内留谁？`dup_audit` 原先按 `sharpe_ls_M` 取最大，而全 A 库该列
   `0/950` 有值（`evals/` 只覆盖 26 个因子）⇒ 代表退化为字母序第一个（2026-11-12 修）。
   本脚本改用 `ic_mean_h20` + **IC 三段稳定性**双判据。
2. **生产依赖**：定版 `yqcore` 族的信号来自 `panels_neu_fundind`，每期经 DPP 从
   候选池选 50 个因子（`reports/alla_rolling_ortho920fundind/selection/y*__h*.json`）。
   一个因子若在 36 期里被选中过，删它就会改历史回测；若从未被选中，删它是
   纯库存清理。这一层 `dup_audit` 完全看不到。

分层判据（tier）
----------------
- ``T1 硬重复``  ρ ≥ 0.999 —— 截面秩相关接近 1，等价于同一因子的线性变换（含反向）。
  此类删非代表**不损失信息**，但若代表/成员在定版在用，仍需按生产依赖处理。
- ``T2 近重复``  0.95 ≤ ρ < 0.999 —— 同族近邻（如 `alpha360_LOW20` 与 `LOW21`，
  都是 `low.shift(i)/close` 的滞后快照）。**不能只按阈值删**：残差 IC 与 IC 稳定性
  才是取舍依据。
- ``T3 同族``    0.85 ≤ ρ < 0.95 —— 只标注，不建议删。

动作（action）
--------------
- ``keep_prod``    定版当期在用 → 不动，要动必须走组合层 A/B
- ``keep_rep``     簇代表 → 保留
- ``keep_quality`` 非代表但质量显著更好（IC 与稳定性双优）→ 保留，代表该换人
- ``safe_drop``    从未进定版 + 非代表 + 质量不占优 → 可删（仍需人工过一遍）
- ``review``       其余（曾进定版、或质量接近）→ 逐个人工判断

用法::

    python -m scripts.factors.library_dedup_plan --dataset all_a_2018_2026
    python -m scripts.factors.library_dedup_plan --thresh 0.95 --hard 0.999
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("dedup_plan")

SELECTION_DIR = ROOT / "reports" / "alla_rolling_ortho920fundind" / "selection"
OUT_DIR = ROOT / "reports" / "library_governance"
HS = ("h1", "h5", "h10", "h20")


# ---------------------------------------------------------------- 生产依赖
def production_dependence() -> tuple[dict[str, int], set[str], dict]:
    """定版滚动 selection：每个因子被选中的期数 + 最后一期在用集合。

    36 期 = 9 年 × 4 horizon。选中期数直接衡量「这个因子在定版里出现过几次」，
    比任何代理指标都硬——它是 `panels_neu_fundind → DPP → pred` 链条的实际输出。
    """
    sets: dict[tuple[str, str], list[str]] = {}
    for f in sorted(SELECTION_DIR.glob("y20*__h*.json")):
        if f.name.endswith(".fp.json"):
            continue
        y, h = f.stem.split("__")
        sets[(y, h)] = json.loads(f.read_text(encoding="utf-8"))
    if not sets:
        log.warning("读不到 selection 目录：%s，生产依赖列全记 0", SELECTION_DIR)
        return {}, set(), {"n_periods": 0, "years": []}
    years = sorted({y for y, _ in sets})
    cnt: dict[str, int] = {}
    for lst in sets.values():
        for n in set(lst):
            cnt[n] = cnt.get(n, 0) + 1
    cur = set().union(*[set(sets[(years[-1], h)]) for h in HS if (years[-1], h) in sets])
    return cnt, cur, {"n_periods": len(sets), "years": years}


# ---------------------------------------------------------------- IC 稳定性
def ic_stability(names: list[str], h: int = 20, seg: int = 3) -> pd.DataFrame:
    """IC 序列的三段均值 + 全期 NW t。

    为什么不用单期均值：`ic_mean_h20` 是一个数，它可能是「三段都 +0.03」也可能是
    「+0.09 / 0 / −0.03」—— 后者在样本外很容易翻面。去重时留谁，稳定性比均值重要。

    分段按**交易日**等分（不按自然时间），因为不同因子的 IC 有效区间一致（同一份
    `ic_h{h}.parquet`）。
    """
    from stats.robust_stats import nw_tstat

    p = LIB.ic_path(h)
    if not p.exists():
        log.warning("缺 IC 序列 %s，稳定性列记 NaN", p)
        return pd.DataFrame(index=names, columns=["ic_all", "t_nw", "seg_neg", "seg_min"])
    ic = pd.read_parquet(p)
    have = [n for n in names if n in ic.columns]
    sub = ic[have]
    n = len(sub)
    bounds = [int(round(n * i / seg)) for i in range(seg + 1)]
    rows = {}
    for nm in have:
        s = sub[nm].dropna()
        if len(s) < 60:
            continue
        mu = float(s.mean())
        t = float(nw_tstat(s.to_numpy())[0])
        segs = [float(sub[nm].iloc[bounds[i]:bounds[i + 1]].mean()) for i in range(seg)]
        rows[nm] = {
            "ic_all": mu, "t_nw": t,
            # 三段里与全期反号的段数：0 最稳，2 以上基本不可信
            "seg_neg": int(sum(1 for x in segs if np.isfinite(x) and mu * x < 0)),
            "seg_min": float(np.nanmin(segs)) if segs else np.nan,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out = out.reindex(names)
    return out


# ---------------------------------------------------------------- 主流程
class LIB:  # 轻量命名空间，避免把 FactorLibrary 的重依赖拉进来
    root = Path("e:/data/factor_library")

    @classmethod
    def base(cls, ds: str) -> Path:
        return cls.root / ds

    @classmethod
    def ic_path(cls, h: int) -> Path:
        return cls.root / CFG["dataset"] / f"ic_h{h}.parquet"


CFG = {"dataset": "all_a_2018_2026"}


def load_corr(dataset: str) -> tuple[np.ndarray, list[str]]:
    f = LIB.base(dataset) / "dup_matrix.npz"
    if not f.exists():
        raise SystemExit(f"没有相关矩阵：{f}\n先跑 `python -m scripts.factors.dup_audit "
                         f"--dataset {dataset}`")
    z = np.load(f)
    return z["corr"], [str(n) for n in z["names"]]


def build_decision(dataset: str, thresh: float, hard: float) -> pd.DataFrame:
    from scripts.factors.dup_audit import clusters, neighbors

    corr, names = load_corr(dataset)
    reg = pd.read_csv(LIB.base(dataset) / "registry.csv").set_index("name")
    reg = reg.reindex(names)

    cnt, cur, prod_meta = production_dependence()
    log.info("定版：%d 期，历史并集 %d，当期在用 %d",
             prod_meta["n_periods"], len(cnt), len(cur))

    nb = neighbors(corr, names, top_k=1).set_index("name")
    cl = clusters(corr, names, thresh).set_index("name")

    stab = ic_stability(names, h=20)
    icm = pd.to_numeric(reg.get("ic_mean_h20"), errors="coerce")
    resid_ic = pd.to_numeric(reg.get("resid_ic"), errors="coerce")
    resid_t = pd.to_numeric(reg.get("resid_t_nw"), errors="coerce")

    df = pd.DataFrame({
        "name": names,
        "cluster": cl["cluster"].reindex(names).to_numpy(),
        "max_corr": nb["max_corr"].reindex(names).to_numpy(),
        "nearest": nb["top"].reindex(names).to_numpy(),
        "ic_h20": icm.to_numpy(),
        "ic_seg_neg": stab["seg_neg"].reindex(names).to_numpy(),
        "ic_t_nw": stab["t_nw"].reindex(names).to_numpy(),
        "resid_ic": resid_ic.to_numpy(),
        "resid_t": resid_t.to_numpy(),
        "cover": pd.to_numeric(reg.get("coverage"), errors="coerce").to_numpy(),
        "family": reg.get("family").astype(str).to_numpy(),
        "n_sel": [int(cnt.get(n, 0)) for n in names],
        "in_prod_now": [n in cur for n in names],
    })
    df["n_periods"] = prod_meta["n_periods"]
    df["tier"] = np.where(df["max_corr"] >= hard, "T1硬重复",
                          np.where(df["max_corr"] >= thresh, "T2近重复",
                                   np.where(df["max_corr"] >= 0.85, "T3同族", "独立")))

    # ---- 簇代表：质量优先（IC 均值 + 稳定性），不看 sharpe（该列空）
    size = df.groupby("cluster")["name"].transform("size")
    df["cluster_size"] = size
    # 质量分：|IC| 打折 —— 三段反号 2 次以上视为不可信，直接砍半
    q = df["ic_h20"].abs().fillna(0.0)
    q = q * np.where(df["ic_seg_neg"].fillna(3) >= 2, 0.5, 1.0)
    q = q * np.where(df["ic_seg_neg"].fillna(3) >= 1, 0.8, 1.0)
    df["quality"] = q
    # 簇内最高质量者 = 代表（只在 size≥2 的簇里有意义）
    rep_idx = df.groupby("cluster")["quality"].idxmax()
    is_rep = pd.Series(False, index=df.index)
    is_rep.loc[rep_idx.to_numpy()] = True
    df["is_rep"] = is_rep.to_numpy() & (df["cluster_size"] >= 2)

    # ---- 动作分层：贪心最大独立集
    # ⚠️ 不能用「与最近邻的 ρ ≥ 阈值」当删除判据——最近邻自己也可能被删。
    # 实测反例：`alpha158_ROC30` 与 `alpha360_CLOSE30` 互为最近邻（ρ=1.000），
    # 两个都被判成 safe_drop ⇒ 该簇这一支被掏空，留下的代表与它们并不等价。
    # 正确判据是**覆盖**：删 x 的前提是 ∃ 一个**确定保留**的 y 使 ρ(x,y) ≥ 阈值。
    pos = {n: i for i, n in enumerate(names)}
    n_row = len(df)
    action = np.full(n_row, "keep", dtype=object)
    cover_by = np.full(n_row, "", dtype=object)
    cover_rho = np.full(n_row, np.nan, dtype=float)
    for i, nm in enumerate(names):
        if df.at[i, "in_prod_now"]:
            action[i] = "keep_prod"
        elif df.at[i, "is_rep"]:
            action[i] = "keep_rep"
    # 保留集：簇代表 + 定版用过的（含 review）——它们不会被删，可作覆盖源
    keep = set(df.loc[df["is_rep"], "name"]) | set(df.loc[df["n_sel"] > 0, "name"])
    for _cid, g in df.groupby("cluster"):
        if len(g) < 2:
            continue
        for i in g.sort_values("quality", ascending=False).index:
            if action[i] in ("keep_prod", "keep_rep"):
                continue
            if df.at[i, "tier"] not in ("T1硬重复", "T2近重复"):
                action[i] = "keep"
                keep.add(names[i])
                continue
            best_rho, best_nm = -1.0, ""
            for k in keep:
                if k == names[i]:
                    continue
                r = abs(float(corr[pos[names[i]], pos[k]]))
                if r > best_rho:
                    best_rho, best_nm = r, k
            # 残差增量门槛（2026-11-12 加）：ρ 高不等于可删。
            # 反例：`alpha158_VSUMN30` vs `VSUMD30` 覆盖 ρ=1.000，但残差 IC=+0.0381、
            # NW t=25.4 —— 量比信号（涨量/跌量差）就在残差里。只看 ρ 会把它删掉。
            # 判据沿用 dup_audit 报告第四节的分档：|IC| ≥ 0.02 或 |t| ≥ 2 ⇒ 有可用增量。
            # ⚠️ 口径差异：registry 的 resid_* 是对 `dup_top`（最近邻）算的，
            # 不是对这里选出的 `cover_by`。作为「是否有增量」的筛子足够，若要精确
            # 到覆盖者需重算逐日回归（约 0.5 s/个），本表不做。
            _ic = df.at[i, "resid_ic"]
            _t = df.at[i, "resid_t"]
            inc_ic = abs(float(_ic)) if pd.notna(_ic) else 0.0
            inc_t = abs(float(_t)) if pd.notna(_t) else 0.0
            has_inc = inc_ic >= 0.02 or inc_t >= 2.0
            if best_rho >= thresh and not has_inc:
                action[i] = "safe_drop" if df.at[i, "n_sel"] == 0 else "review"
                cover_by[i], cover_rho[i] = best_nm, best_rho
            else:
                # 无覆盖者、或残差有独立增量 ⇒ 必须留
                action[i] = "keep"
                keep.add(names[i])
    df["action"] = action
    df["cover_by"] = cover_by
    df["cover_rho"] = cover_rho
    # 自检：safe_drop 必须都带覆盖者（否则就是「无人覆盖却判可删」的漏网）
    bad = df[(df["action"] == "safe_drop") & (df["cover_by"] == "")]
    if len(bad):
        log.error("自检失败：%d 个 safe_drop 没有覆盖者，改为 keep：%s",
                  len(bad), bad["name"].head(5).tolist())
        df.loc[bad.index, "action"] = "keep"
    return df.sort_values(["tier", "cluster", "quality"], ascending=[True, True, False])


def write_report(df: pd.DataFrame, dataset: str, thresh: float, hard: float) -> None:
    n = len(df)
    vc = df["action"].value_counts()
    tiers = df["tier"].value_counts()
    sel = df[df["n_sel"] > 0]
    lines = [
        "# 因子库去重决策表",
        "",
        f"> `{dataset}` · 因子 {n} 个 · 相关阈值 {hard:.3f}（硬）/ {thresh:.2f}（近）"
        f" · 定版期数 {int(df['n_periods'].iloc[0]) if n else 0}",
        "",
        "## 〇 结论先行",
        "",
        f"1. 定版历史用过 **{len(sel)}** 个（当期在用 **{int(df['in_prod_now'].sum())}** 个），"
        f"从未进过 selection 的 **{n - len(sel)}** 个。",
        f"2. 按相关性分层：T1 硬重复 {int(tiers.get('T1硬重复', 0))} · "
        f"T2 近重复 {int(tiers.get('T2近重复', 0))} · T3 同族 {int(tiers.get('T3同族', 0))} · "
        f"独立 {int(tiers.get('独立', 0))}。",
        f"3. 动作建议：可删（safe_drop）**{int(vc.get('safe_drop', 0))}** 个 · "
        f"待人工（review）**{int(vc.get('review', 0))}** 个 · "
        f"保留 {int(vc.get('keep_prod', 0)) + int(vc.get('keep_rep', 0)) + int(vc.get('keep', 0))} 个。",
        "",
        "> **本表不删任何东西。** 删除清单需人工过一遍后，走快照 + 回收站流程。",
        "",
        "## 一、动作 × 是否进过定版",
        "",
        "| 动作 | 从未进 selection | 曾进 selection | 合计 |",
        "|---|---:|---:|---:|",
    ]
    for a in ["safe_drop", "review", "keep_prod", "keep_rep", "keep"]:
        if a not in vc:
            continue
        sub = df[df["action"] == a]
        lines.append(f"| `{a}` | {int((sub['n_sel'] == 0).sum())} | "
                     f"{int((sub['n_sel'] > 0).sum())} | {len(sub)} |")
    lines += [
        "",
        "## 二、可删清单（safe_drop，按覆盖相关度降序，前 40）",
        "",
        "判据（**贪心最大独立集**）：删 x 的前提是存在一个**确定保留**的因子 y，使 "
        f"|ρ(x,y)| ≥ {thresh:.2f}，**且 x 的残差没有独立增量**（|resid_ic| < 0.02 且 "
        "|resid_t| < 2）；并且 x 从未进定版 selection、不是簇代表。"
        "「确定保留」= 簇代表 ∪ 定版用过的 ∪ 先被判定要留的（质量更高的先留）。"
        "这样既不会出现**互为最近邻的两个因子同时被删**，也不会把**高相关但残差有增量**"
        "的因子（如量比类 `VSUMN` vs `VSUMD`，ρ=1.000 但残差 t=25）误删。"
        "",
        "> ⚠️ 表中 ρ 为**绝对值**，含反向重复。例如 `alpha158_SUMD10` 与 `SUMP10` 的"
        "真实 ρ = **−1.000**（涨量 vs 跌量，镜像），表中显示 1.000；这类只留一个并"
        "保留其 IC 符号即可，删掉另一个不损失信息。",
        "",
        "| 因子 | 覆盖者 | ρ | 簇大小 | 残差 IC | 残差 t | IC_h20 | 家族 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    sd = df[df["action"] == "safe_drop"].sort_values("cover_rho", ascending=False)
    for _, r in sd.head(40).iterrows():
        lines.append(f"| `{r['name']}` | `{r['cover_by']}` | {r['cover_rho']:.3f} | "
                     f"{int(r['cluster_size'])} | {r['resid_ic']:+.4f} | "
                     f"{r['resid_t']:+.2f} | {r['ic_h20']:+.4f} | {r['family']} |")
    if len(sd) > 40:
        lines.append(f"| … | 其余 {len(sd) - 40} 个见 `dedup_decision.csv` | | | | | |")

    lines += [
        "",
        "## 三、需人工判断（review：高冗余，但定版用过）",
        "",
        "这些因子与某个保留因子 ρ ≥ 阈值，但**在 36 期 selection 里被选中过**（当期未必用）。"
        "删它们会让**对应年份的历史回测不可复现**——不是收益高低的问题，是复现性问题。"
        "要动必须重跑那几年的 `--stage select` 并记录差异。",
        "",
        "| 因子 | 覆盖者 | ρ | 定版选中期数 | IC_h20 |",
        "|---|---|---:|---:|---:|",
    ]
    rv = df[df["action"] == "review"].sort_values("n_sel", ascending=False)
    for _, r in rv.head(30).iterrows():
        lines.append(f"| `{r['name']}` | `{r['cover_by']}` | {r['cover_rho']:.3f} | "
                     f"{int(r['n_sel'])} | {r['ic_h20']:+.4f} |")
    if len(rv) > 30:
        lines.append(f"| … | 其余 {len(rv) - 30} 个见 CSV | | | |")

    lines += [
        "",
        "## 四、口径与边界",
        "",
        f"1. 相关来自 `dup_matrix.npz`：{200} 日 × {3500} 只抽样、逐日截面 Spearman、"
        "NaN 记 0 ⇒ **偏低估**（漏报不误报）。",
        "2. 质量分 = `|ic_mean_h20|` 打折（三段反号 1 次 ×0.8、≥2 次 ×0.5）。"
        "**不用 `sharpe_ls_M`**：该列全库 0/950 有值，用它选代表会退化成字母序。",
        "3. 「定版在用」= `reports/alla_rolling_ortho920fundind/selection/y*__h*.json`，"
        "即 `panel_source: panels_neu_fundind` 那条线（`yqcore` 族）。"
        "**其他档位（solo/focus）同源同信号，一并覆盖**。",
        "4. 相关是**截面**口径；两个因子截面不相关但多空收益时序同步的风险同质，"
        "本表测不到（`evals/` 仅 26 个因子有序列）。",
        "5. `safe_drop` 不等价于「删了没影响」：从未进 selection 只说明**这套 DPP 没选它**，"
        "不排除换配置后被选中。清理前仍要留清单归档。",
        "",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dedup_plan.md").write_text("\n".join(lines), encoding="utf-8")
    df.to_csv(OUT_DIR / "dedup_decision.csv", index=False, encoding="utf-8-sig")
    log.info("报告 → %s（CSV %d 行）", OUT_DIR / "dedup_plan.md", len(df))


def main() -> None:
    ap = argparse.ArgumentParser(description="因子库去重决策表")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--thresh", type=float, default=0.95, help="近重复阈值")
    ap.add_argument("--hard", type=float, default=0.999, help="硬重复（换皮）阈值")
    args = ap.parse_args()
    CFG["dataset"] = args.dataset

    df = build_decision(args.dataset, args.thresh, args.hard)
    write_report(df, args.dataset, args.thresh, args.hard)

    vc = df["action"].value_counts()
    print("\n=== 去重决策 ===")
    print(f"因子 {len(df)} 个；定版用过 {(df['n_sel'] > 0).sum()}，"
          f"当期在用 {int(df['in_prod_now'].sum())}")
    for a in ["safe_drop", "review", "keep_prod", "keep_rep", "keep"]:
        if a in vc:
            print(f"  {a:14s} {vc[a]:4d}")
    print(f"\n报告 → {OUT_DIR / 'dedup_plan.md'}")


if __name__ == "__main__":
    main()
