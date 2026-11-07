"""registry 元数据回填：``formula`` 与 ``family``（只补空缺，不动任何统计列）。

为什么补这两列
--------------
``formula``
    唯一代码消费方是 ``research/factor_extension.py:extend_gp_factors``：

        fp = formula_builder(str(row.get("formula") or name), features=feats)(panel)

    空 formula 会退化成因子名。名称**本身是表达式**的（GP 挖出的
    ``div(sub(low,...),...)``）能侥幸跑通，而 ``alpha101_001`` / ``alpha158_BETA30``
    这类名字必然解析失败 ⇒ 这批面板**不可重算、不可迁移**。更根本的是：拿到一个
    registry 行，应当能回答"这个面板是什么"，现在回答不了。

``family``
    ``FAMILY_WHITELIST`` 是受控词表，``list_all(family=)`` / 族统计 / 组合报告都按它
    切片。all_a 965 行里 827 行空（85.7%），等于这批因子在任何按族的视图里都不存在。

⚠️ 两者的补法不同：``formula`` 在这里填的是**定义引用**（论文/因子集出处 + 编号），
不是可执行表达式——可执行的唯一真源是 builder 代码。这与 hs300 子库既有 505 行的
写法一致（``WorldQuant Alpha101 (Kakushadze 2016) #001``），所以回填后两个库终于一致。

规则（每条都能追到仓库里的一个文件，不发明内容）
------------------------------------------------
===== ============================= ==========================================
集    formula                       依据
===== ============================= ==========================================
alpha101  ``<SET_LABELS> #001``      ``factor/alpha_base.py::SET_LABELS``（论文）
alpha191  同上                     同上（GTJA 191）
alpha158  ``<SET_LABELS> · BETA30`` 同上；尾串本就是"算子+窗口"的定义
alpha360  ``<SET_LABELS> · CLOSE59`` 同上；尾串是"字段+滞后"
e4x       ``group_zscore(px | tercile(buk))``
                                    ``scripts/archive/build_alla_e4_interaction_factors.py``
                                    的文件头设计说明 + ``_group_zscore`` 实现
llmmcts   ``label``                 该批的 label **就是** ``formula_project``
                                    （``build_alla_llmmcts_factors.py:51``）
带定义式  ``label``                  一批财务比率的 label 本身就是定义式
                                    （``研发强度=研发TTM/营收TTM``）；判据是等号左边
                                    不含括号/"："（否则是正文里的"="，如
                                    ``披露时滞相对偏移(晚披露=坏消息)``）
其余      ——                        定义散在 builder 的 Python 代码里，抄一份 label
                                    只是制造"看起来有 formula"的假象 ⇒ 留空并计数
===== ============================= ==========================================

``family`` 一律走 ``SET_TO_FAMILY``（库内唯一真源）；集名缺失的旧子库（hs300 /
zz1000 的 ``set`` 列全空）用名字前缀兜底，仍无法判定的留空——**不猜**。

用法：
    python -m scripts.factors.backfill_metadata                 # dry-run，打印计划
    python -m scripts.factors.backfill_metadata --apply
    python -m scripts.factors.backfill_metadata --dataset zz1000_2022_2025 --apply
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("backfill_metadata")

#: 名字前缀 → 集名（只用于 ``set`` 列缺失的旧子库；映射本身很短，不需要"猜"）
PREFIX_TO_SET = {"alpha101": "alpha101", "alpha158": "alpha158",
                 "alpha191": "alpha191", "alpha360": "alpha360",
                 "e4": "e4x", "llm": "llmmcts"}

#: 编号型因子集（formula 写成 ``... #001``）与特征型（``... · BETA30``）
NUMBERED_SETS = ("alpha101", "alpha191")
FEATURE_SETS = ("alpha158", "alpha360")


def _alpha_formula(set_name: str, name: str) -> str:
    """alpha 四族的定义引用（与 hs300 既有 505 行的写法对齐）。"""
    from factor.alpha_base import SET_LABELS

    label = SET_LABELS[set_name]
    tail = name[len(set_name) + 1:] if name.startswith(set_name + "_") else name
    return f"{label} #{tail}" if set_name in NUMBERED_SETS else f"{label} · {tail}"


#: label 里"等号左边是干净的因子名"才算定义式；这条正则同时挡掉
#: ``披露时滞相对偏移(晚披露=坏消息)`` 这种正文里的等号。
_LABEL_DEF = re.compile(r"^[^()（）:：=]{1,20}=")

#: 白名单外的历史缩写（``FAMILY_WHITELIST`` 之外的值）→ 受控族。
#: 原缩写不丢，落到 ``subfamily``（该列是自由文本，正是为细分准备的）。
FAMILY_ALIASES = {"rnd": "基本面", "lev": "基本面", "liq": "基本面",
                  "pro": "基本面", "qual": "基本面", "cfq": "基本面"}


def _definition_from_label(label: str) -> str:
    """label 本身就是定义式时返回原文，否则返回空。"""
    return label if _LABEL_DEF.match(label) else ""


def _e4_formula(name: str) -> str:
    """``e4_{bucket}_x_{px}`` → 组内 zscore 的读法。

    bucket 里自己带下划线（``np_growth_yoy``），只能按 ``_x_`` 切。
    """
    body = name[3:] if name.startswith("e4_") else name
    if "_x_" not in body:
        return ""
    buk, px = body.split("_x_", 1)
    return f"group_zscore({px} | tercile({buk}))"


def _txt(v) -> str:
    """把 registry 里的标量读成干净字符串。

    坑：``str(np.nan or "")`` 是 ``"nan"`` 而不是 ``""``——nan 是**真值**。
    旧写法 ``str(row.get("set") or "")`` 因此让「空 set」永远走不到前缀兜底，
    ``family_for_set("nan")`` 直接返回空（等于子库整列白填）。
    """
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    out = str(v).strip()
    return "" if out.lower() in ("nan", "none") else out


def propose(row: pd.Series) -> dict:
    """给一行 registry 算 (set, family, formula)；算不出的留空。"""
    from research.factor_library import family_for_set

    name = _txt(row.get("name"))
    set_name = _txt(row.get("set"))
    if not set_name:
        head = name.split("_")[0]
        set_name = PREFIX_TO_SET.get(head, "")

    fam = family_for_set(set_name)
    formula = ""
    if set_name in NUMBERED_SETS or set_name in FEATURE_SETS:
        formula = _alpha_formula(set_name, name)
    elif set_name == "e4x":
        formula = _e4_formula(name)
    elif set_name == "llmmcts":
        # 这批的 label 就是 formula_project 原文（见模块 docstring 的规则表）
        lab = _txt(row.get("label"))
        formula = lab if "(" in lab else ""
    else:
        formula = _definition_from_label(_txt(row.get("label")))
    return {"name": name, "set": set_name, "family": fam, "formula": formula}


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip().isin(["", "nan", "None"]))


def borrow_definitions(dataset: str, *, base_root: Path | None = None) -> dict[str, str]:
    """从**兄弟数据集**借同名因子的定义串（hs300 / zz1000 里有一批手写定义）。

    只在两个前提成立时才借：① 同名因子在本库为空；② 兄弟库之间对同一名字**不矛盾**
    （2026-11-07 实测 hs300 与 zz1000 的 346 个同名定义逐字一致 ⇒ 可以互借）。
    矛盾的名字直接剔除，不选边。
    """
    from research.factor_library import FactorLibrary

    root = Path(base_root) if base_root else None
    seen: dict[str, set[str]] = {}
    for ds in FactorLibrary.list_datasets(root=root):
        if ds == dataset:
            continue
        try:
            sib = FactorLibrary(root=root, dataset=ds).list_all()
        except Exception:  # noqa: BLE001 —— 兄弟库读不动不该阻断回填
            continue
        if "formula" not in sib.columns:
            continue
        for n, f in zip(sib["name"].astype(str), sib["formula"]):
            f = "" if pd.isna(f) else str(f).strip()
            if f:
                seen.setdefault(n, set()).add(f)
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def build_plan(reg: pd.DataFrame, borrow: dict[str, str] | None = None) -> pd.DataFrame:
    """按行算回填计划；只为**当前为空**的字段填值。

    集级护栏：若某个集里**已有人工族标签**且与该集名映射不一致（或集内不止一种），
    说明人已经把这一族的粒度定得比"集名"更细（如 fundamental 里商誉/质押属「股东」、
    moneyflow 里大宗折价属「股东」）。这种集**不提案**——按集名硬填会把粗标签
    盖到细分族旁边，制造同一集内两套粒度。
    """
    from research.factor_library import family_for_set

    prop = pd.DataFrame([propose(r) for _, r in reg.iterrows()])
    fam_old = reg.get("family", pd.Series("", index=reg.index)).reset_index(drop=True)
    fml_old = reg.get("formula", pd.Series("", index=reg.index)).reset_index(drop=True)

    fam_blank = _blank(fam_old)
    known = prop.loc[~fam_blank].assign(fam=fam_old[~fam_blank].astype(str).str.strip())
    per_set = known.groupby("set")["fam"].agg(lambda s: sorted(set(s)))
    refined = {s for s, vals in per_set.items()
               if s and (len(vals) > 1 or vals[0] != family_for_set(s))}
    if refined:
        log.info("族标签已人工细化的集（不提案）：%s", ", ".join(sorted(refined)))
    prop.loc[prop["set"].isin(refined), "family"] = ""

    # 本地规则给不出定义时，才去兄弟库借（借的是别人手写过的原文，不改写）
    borrowed = pd.Series(False, index=prop.index)
    if borrow:
        miss = prop["formula"] == ""
        got = prop.loc[miss, "name"].map(borrow)
        prop.loc[miss, "formula"] = got.fillna("")
        borrowed = miss & (prop["formula"] != "")
        log.info("从兄弟库借到 %d 条定义", int(borrowed.sum()))

    plan = pd.DataFrame({
        "name": prop["name"], "set": prop["set"],
        "family_old": fam_old, "family_new": prop["family"],
        "formula_old": fml_old, "formula_new": prop["formula"],
    })
    plan["set_refined"] = plan["set"].isin(refined)
    plan["borrowed"] = borrowed.to_numpy()
    plan["fill_family"] = fam_blank & (prop["family"] != "")
    plan["fill_formula"] = _blank(fml_old) & (prop["formula"] != "")
    # 冲突：已有非空值与提案不一致 —— 不回写，只报（静默覆盖会掩盖口径漂移）
    plan["conflict_family"] = (~fam_blank) & (prop["family"] != "") & \
        (fam_old.astype(str).str.strip() != prop["family"])
    plan["conflict_formula"] = (~_blank(fml_old)) & (prop["formula"] != "") & \
        (fml_old.astype(str).str.strip() != prop["formula"])
    return plan


def summarize(plan: pd.DataFrame) -> pd.DataFrame:
    g = plan.groupby("set", dropna=False)
    out = pd.DataFrame({
        "n": g.size(),
        "family_fill": g["fill_family"].sum(),
        "formula_fill": g["fill_formula"].sum(),
        "borrowed": g["borrowed"].sum(),
        "refined": g["set_refined"].first(),
        "family_conflict": g["conflict_family"].sum(),
        "formula_conflict": g["conflict_formula"].sum(),
    }).reset_index()
    out["set"] = out["set"].fillna("（未判定）").replace("", "（未判定）")
    return out.sort_values("n", ascending=False)


def apply_plan(lib, plan: pd.DataFrame, *, dataset: str) -> dict:
    """按集分组写回（``fill_missing_only=True`` ⇒ 非空值永远不会被清空）。"""
    from research.factor_library import source_for

    todo = plan[plan["fill_family"] | plan["fill_formula"]].copy()
    stats = {"before": 0, "after": 0, "added": 0, "updated": 0}
    for set_name, sub in todo.groupby("set", sort=True):
        if not set_name:
            log.warning("%d 行无法判定集名，跳过", len(sub))
            continue
        recs = []
        for _, r in sub.iterrows():
            rec = {"name": r["name"]}
            if r["fill_family"]:
                rec["family"] = r["family_new"]
            if r["fill_formula"]:
                rec["formula"] = r["formula_new"]
            recs.append(rec)
        st = lib.upsert_rows(
            recs, source=source_for(str(set_name), producer="backfill_metadata",
                                    scope=dataset),
            kind="", maturity="", frequency="", set_name=str(set_name),
            fill_missing_only=True)
        for k in stats:
            stats[k] += st[k]
        log.info("[%s] 回填 %d 行 → registry %d 行", set_name, len(recs), st["after"])
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="registry formula/family 回填（默认 dry-run）")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--root", default=None,
                    help="库根（缺省走 config；测试用临时目录）")
    ap.add_argument("--apply", action="store_true", help="真正写盘（默认只打印计划）")
    ap.add_argument("--no-borrow", action="store_true",
                    help="不从兄弟数据集借同名因子的定义串")
    args = ap.parse_args(argv)

    from research.factor_library import STAT_COLUMNS, FactorLibrary

    lib = FactorLibrary(root=args.root, dataset=args.dataset)
    reg = lib.list_all()
    if reg.empty:
        raise SystemExit(f"{lib.root}/registry.csv 为空或不存在")

    borrow = {} if args.no_borrow else borrow_definitions(
        args.dataset, base_root=Path(lib.base_root))
    plan = build_plan(reg, borrow)
    tab = summarize(plan)
    pd.set_option("display.width", 160)
    print(f"\n=== {args.dataset}：{len(reg)} 行 ===")
    print(tab.to_string())

    n_fam = int(plan["fill_family"].sum())
    n_fml = int(plan["fill_formula"].sum())
    n_fam_c = int(plan["conflict_family"].sum())
    n_fml_c = int(plan["conflict_formula"].sum())
    print(f"\n待补 family {n_fam} 行（剩 {int(_blank(reg['family']).sum()) - n_fam} 行无依据）")
    print(f"待补 formula {n_fml} 行（剩 {int(_blank(reg['formula']).sum()) - n_fml} 行无依据）")
    if n_fam_c or n_fml_c:
        print(f"⚠️ 与现有值冲突 {n_fam_c} + {n_fml_c} 行（不回写，仅报告）")

    if not args.apply:
        print("\n[dry-run] 加 --apply 写盘")
        return

    stat_cols = [c for c in sorted(STAT_COLUMNS) if c in reg.columns]
    # 按 name 对齐而不是按行号：upsert_rows 会重排行，行号对齐会造出满屏假漂移
    before = reg.set_index("name")[stat_cols]
    st = apply_plan(lib, plan, dataset=args.dataset)
    after = lib.list_all()

    # 自检 1：统计列一个数都不能动
    b, a = before, after.set_index("name")[stat_cols].reindex(before.index)
    drift = {}
    for c in stat_cols:
        # bool 列（significant）不能直接相减，先统一成 float
        x = pd.to_numeric(b[c], errors="coerce").astype(float)
        y = pd.to_numeric(a[c], errors="coerce").astype(float)
        d = float((x - y).abs().max()) if x.notna().any() else 0.0
        # 1e-12 是 CSV 往返的浮点量化噪声（实测 p_value_nw 会差 6e-24），不是漂移
        if not pd.isna(d) and d > 1e-12:
            drift[c] = d
    # 自检 2：没有新增行
    if st["added"]:
        raise SystemExit(f"❌ 意外新增 {st['added']} 行——回填不应造行")

    post = lib.list_all()
    print(f"\n写盘完成：family {int(_blank(post['family']).sum())} 空 / "
          f"formula {int(_blank(post['formula']).sum())} 空"
          f"（{len(post)} 行）")
    print("统计列漂移：" + ("无 ✅" if not drift else f"❌ {drift}"))
    if drift:
        raise SystemExit("统计列被改动，回填越界")


if __name__ == "__main__":
    main()
