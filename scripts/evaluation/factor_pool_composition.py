"""因子池构成体检：量价占比到底多少、口径在哪、哪些族是真被用上了。

背景（2026-10-10）：此前口头结论「因子池 83.6% 是量价」未落文件，且分母口径
不明。本脚本把四个候选池（registry / panels / ic 缓存 / panels_neu）的族构成
一次性算清，并给出**三层占比**——池 → 入选 → 归因——避免再拿单一数字说事。

产出（stdout，可重定向到 reports/factor_pool_composition/composition.txt）：
  1. 四池的族构成对比（哪个分母给出 83.6%）
  2. 模型池按 set 拆解（四个 WorldQuant Alpha 集 = 量价全部）
  3. 定版 920 口径 y2026 入选 50 特征的族构成
  4. 每日特征重要性 gain 份额（按族）
  5. 数据面一致性：registry / panels / panels_neu 三方差集

用法：
    python -m scripts.evaluation.factor_pool_composition
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import Config
from research.factor_library import SET_TO_FAMILY

DS = "all_a_2018_2026"
_SEL_DIR = Path("reports/alla_rolling_ortho920fundind/selection")
_FI = Path("reports/alla_daily_defv/feature_importance_20261008.csv")
_HORIZONS = (1, 5, 10, 20)


def _root() -> Path:
    return Path(str(Config.get()["factor_library"]["root"])) / DS


def main() -> None:
    root = _root()
    reg = pd.read_csv(root / "registry.csv").set_index("name")
    ic = pd.read_parquet(root / "ic_h1.parquet")

    def fam(n: str) -> str:
        return str(reg["family"].get(n, "")) if n in reg.index else "(未登记)"

    def setn(n: str) -> str:
        return str(reg["set"].get(n, "")) if n in reg.index else "(未登记)"

    def comp(names, label: str) -> None:
        s = pd.Series([fam(n) for n in names])
        vc = s.value_counts()
        tot = len(names)
        q = int(vc.get("量价", 0))
        print(f"=== {label}  共 {tot}  量价 {q} = {100 * q / tot:.1f}% ===")
        for k, v in vc.items():
            print(f"  {str(k):8s} {v:5d}  {100 * v / tot:5.1f}%")
        print()

    panels = [p.stem for p in (root / "panels").glob("*.parquet")]
    neu = [p.stem for p in (root / "panels_neu").glob("*.parquet")]

    print("#### 1. 四池族构成 ####\n")
    comp(list(reg.index), "registry（全库登记）")
    comp(panels, "panels/（原始面板）")
    comp(list(ic.columns), "ic_h1 列（模型入模候选）")
    comp(neu, "panels_neu（生产预处理库）")
    # alt 入库前的 955 口径
    pre_alt = [n for n in reg.index if setn(n) != "altf"]
    comp(pre_alt, "registry 去掉 20 个 altf（= 旧 955 口径）")

    print("#### 2. 模型池按 set 拆解 ####\n")
    sets = pd.Series([setn(n) for n in ic.columns])
    vc = sets.value_counts()
    for k, v in vc.items():
        print(f"  {str(k):14s} {v:4d}  {100 * v / len(ic.columns):5.1f}%")
    print()

    print("#### 3. 定版 920 入选特征族构成（y2026）####\n")
    union: list[str] = []
    for h in _HORIZONS:
        p = _SEL_DIR / f"y2026__h{h}.json"
        if not p.exists():
            print(f"  h{h}: 缺 {p}")
            continue
        feats = json.loads(p.read_text(encoding="utf-8"))
        union += feats
        vc = pd.Series([fam(n) for n in feats]).value_counts()
        q = int(vc.get("量价", 0))
        print(f"  h{h:<2d} n={len(feats)}  量价 {q} ({100 * q / len(feats):.0f}%)  "
              + " | ".join(f"{k}:{v}" for k, v in vc.items()))
    uvc = pd.Series([fam(n) for n in sorted(set(union))]).value_counts()
    print(f"  四周期并集 {len(set(union))} 个  "
          + " | ".join(f"{k}:{v}" for k, v in uvc.items()))
    print()

    print("#### 4. 特征重要性 gain 份额（按族）####\n")
    if _FI.exists():
        fi = pd.read_csv(_FI)
        fi["fam"] = fi["feature"].map(fam)
        g = fi.groupby("fam")["importance_gain"].sum().sort_values(ascending=False)
        for k, v in g.items():
            print(f"  {k or '(未登记)':8s} {v:12.1f}  {100 * v / g.sum():5.1f}%")
    else:
        print(f"  {_FI} 不存在，跳过")
    print()

    print("#### 5. 数据面一致性 ####\n")
    ns, ps, us = set(reg.index), set(panels), set(neu)
    print(f"  registry {len(ns)} / panels {len(ps)} / panels_neu {len(us)}")
    print(f"  registry 有但 panels 缺 : {len(ns - ps)} "
          f"（族 {pd.Series([fam(n) for n in ns - ps]).value_counts().to_dict()}）")
    print(f"  panels 有但 registry 缺 : {len(ps - ns)}（哈希命名，GP 挖掘产物）")
    print(f"  panels 有但 panels_neu 缺: {len(ps - us)} "
          f"（族 {pd.Series([fam(n) for n in ps - us]).value_counts().to_dict()}）")
    print()

    print("#### 6. 各族 IC 质量（ic_h1 口径）####\n")
    r2 = reg.reindex(ic.columns)
    rows = []
    for f, idx in r2.groupby("family").groups.items():
        sub = r2.loc[idx]
        icm = sub["ic_mean_h1"].astype(float)
        rows.append((f, len(sub), icm.abs().mean(), icm.abs().median()))
    for f, n, m, md in sorted(rows, key=lambda x: -x[2]):
        print(f"  {f:8s} n={n:4d}  mean|IC|={m:.4f}  median|IC|={md:.4f}")
    print()
    # SET_TO_FAMILY 一致性（防分族漂移）
    bad = [s for s in pd.Series([setn(n) for n in ic.columns]).unique()
           if s not in SET_TO_FAMILY and s != "(未登记)"]
    if bad:
        print(f"  [warn] set 不在 SET_TO_FAMILY: {bad}")


if __name__ == "__main__":
    main()
