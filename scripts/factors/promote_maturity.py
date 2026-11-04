"""成熟度晋升：把 :data:`research.factor_library.MATURITY_GATES` 落到全库。

跑一次回答两个问题：
1. **965 个因子里，有几个真的够格进白名单？**（绝对门槛，不是"排名前 N"）
2. **进了白名单的那几个，彼此是不是还在互相重复？**（队列去重）

用法::

    python -m scripts.factors.promote_maturity --dataset all_a_2018_2026
    python -m scripts.factors.promote_maturity --dataset all_a_2018_2026 --apply
    python -m scripts.factors.promote_maturity --gate min_sharpe=0.7   # 敏感性

默认 **dry-run**（只出报告与明细，不改 registry）；``--apply`` 才写盘。

为什么门槛写在库里而不是这个脚本里：晋升标准是"什么算可用"的定义，属于因子库
的语义（``MATURITY_VALUES`` 的邻居），脚本只是 IO。改阈值只有一个地方可改。
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from research.factor_library import (MATURITY_GATES, MATURITY_REASONS,
                                     FactorLibrary)

log = logging.getLogger("promote_maturity")

#: 门槛施加顺序（报告里的漏斗按此顺序累加，与 :meth:`assess_maturity` 一致）。
GATE_ORDER: list[str] = ["no_eval", "stale_caliber", "dup_unknown", "low_sharpe",
                         "redundant", "unstable", "fading",
                         "crowded", "family_cap", "uncovered"]

LEVEL_RANK = {"invalid": 0, "retired": 1, "experimental": 1,
              "oos_verified": 2, "active": 3}


def load_corr(lib: FactorLibrary, path: str | None = None
              ) -> tuple[np.ndarray | None, list[str] | None, str]:
    """读全库截面相关矩阵；缺本地矩阵时回退到 all_a 的快照。

    子库（hs300 / zz1000 / top10）的因子名与 all_a 同名同义，截面签名也几乎一致，
    所以跨库复用矩阵比"每个子库重算 7 分钟"划算。回退时**必须打印来源**，
    否则报告里的"独立"结论会被误当成该子库自己的证据。
    """
    cands: list[Path] = []
    if path:
        cands.append(Path(path))
    else:
        cands.append(Path(lib.root) / "dup_matrix.npz")
        if lib.dataset and lib.dataset != "all_a_2018_2026":
            cands.append(Path(lib.base_root) / "all_a_2018_2026" / "dup_matrix.npz")
    for p in cands:
        if p.exists():
            z = np.load(p, allow_pickle=True)
            return z["corr"], [str(x) for x in z["names"]], str(p)
    log.warning("找不到相关矩阵（试过 %s）：本次不给 active 档",
                ", ".join(str(c) for c in cands))
    return None, None, ""


def funnel(df: pd.DataFrame) -> list[tuple[str, int, int]]:
    """门槛漏斗：``[(门槛, 被它单独挡掉的数, 累积通过数)]``。

    "被挡掉"只统计**此前的门槛都没挡**的行——否则一个烂因子会在每一关都被数一次，
    读起来像 8 个问题，实际只是 1 个。
    """
    out: list[tuple[str, int, int]] = []
    alive = np.ones(len(df), dtype=bool)
    for k in GATE_ORDER:
        hit = np.array([k in f for f in df["fails"]])
        n = int((hit & alive).sum())
        alive &= ~hit
        out.append((k, n, int(alive.sum())))
    return out


def gate_hits(df: pd.DataFrame) -> dict[str, int]:
    """每道门槛**单独**命中的次数（不累积）。

    与 :func:`funnel` 互补：漏斗回答"按顺序筛到最后剩几个"，本表回答"如果只看这一条
    会挡多少"。子库那种"前面一道就全挡光"的情况，只有本表能看出后面几道其实也挡了
    很多（例：hs300 的 `dup_unknown` 挡掉全部 505，但 `low_sharpe` 单独也挡 504
    ——真实结论是"业绩本来就不够"，不是"缺冗余审计"）。
    """
    return {k: int(sum(k in f for f in df["fails"])) for k in GATE_ORDER}


def _standin_check(df: pd.DataFrame, act: pd.DataFrame,
                   corr: np.ndarray | None, corr_names) -> pd.DataFrame | None:
    """找出每个 active 的"嫌疑替身"：夏普更高、相关 ≥0.9、却被队列门槛挡掉的因子。

    贪心的固有副作用：被 ``family_cap`` 挡掉的强因子退出候选池后，与它近乎同义的
    "异名近亲"（如 LOW37 vs OPEN36，同窗口不同价位字段，相关 0.98）仍能通过相关
    检查——白名单两两都 <0.85，但选中的不是该族最强的那个。这不是 bug，是"先到先得"
    的代价；摊开写出来比假装没有好。
    """
    if corr is None or corr_names is None or act.empty:
        return None
    idx = {str(n): i for i, n in enumerate(list(corr_names))}
    blocked = df[df["verdict"] == "oos_verified"]
    rows = []
    for nm, r in act.iterrows():
        if nm not in idx:
            continue
        best = None
        for other, o in blocked.iterrows():
            if other not in idx or o["sharpe"] <= r["sharpe"]:
                continue
            c = abs(float(corr[idx[str(nm)], idx[str(other)]]))
            if c >= 0.9 and (best is None or c > best[1]):
                best = (str(other), c, float(o["sharpe"]), o["fails"])
        if best:
            rows.append({"active": nm, "standin": best[0], "corr": best[1],
                         "standin_sharpe": best[2],
                         "why": "；".join(MATURITY_REASONS.get(k, k) for k in best[3])})
    return pd.DataFrame(rows)


def write_report(lib: FactorLibrary, df: pd.DataFrame, pf: list[tuple[str, int, int]],
                 corr_src: str, gates: dict, out_path: Path, applied: int,
                 corr: np.ndarray | None = None, corr_names=None) -> None:
    vc = df["verdict"].value_counts()
    n = len(df)
    cur = df["current"].fillna("").astype(str).str.strip().replace("", "（空）")
    moves = pd.DataFrame({"from": cur, "to": df["verdict"]})
    moves = moves[moves["from"] != moves["to"]]

    L: list[str] = []
    L.append(f"# 因子成熟度评定：{lib.dataset}")
    L.append("")
    state = f"本次写盘 {applied} 行" if applied else "registry 无需改动（dry-run / 已是最新）"
    L.append(f"生成于 {pd.Timestamp.now():%Y-%m-%d %H:%M}；库内 {n} 个因子；{state}。")
    L.append("")
    L.append("## 一、结论")
    L.append("")
    L.append("| 档位 | 个数 | 占比 |")
    L.append("|---|---:|---:|")
    for k in ("active", "oos_verified", "experimental", "retired", "invalid"):
        c = int(vc.get(k, 0))
        L.append(f"| `{k}` | {c} | {c / n:.1%} |")
    L.append("")
    n_act = int(vc.get("active", 0))
    n_ok = n_act + int(vc.get("oos_verified", 0))
    if n_act == 0:
        n_low = gate_hits(df).get("low_sharpe", 0)
        L.append(f"**没有因子进 `active` 白名单**。个体证据齐全的 {n_ok} 个也没有；"
                 f"{n_low}/{n} 个在净多空夏普这一条上就不过线。"
                 "这个库当前不该被用来选因子——不是缺审计（`dup_unknown` 会把没查过"
                 "冗余的都挡掉），而是**扣成本后的业绩本身不够**。")
    else:
        L.append(f"**只有 {n_act} 个进 `active` 白名单**"
                 f"（{n_act / n:.1%}）。这个数字小不是标准苛刻："
                 f"个体证据齐全（七条硬门槛全过）的有 {n_ok} 个，"
                 f"但其中 {sum(h for k, h, _ in pf if k in ('crowded', 'family_cap'))} 个"
                 "是同一批公式的窗口网格或近亲——`alpha360` 六个字段 × 60 个窗口就占了"
                 "半个库，它们彼此相关 0.4~0.9，统计上「各自独立」，构造上是同一个想法。")
    L.append("")
    L.append("## 二、门槛漏斗（按施加顺序，每关只数此前没被挡的）")
    L.append("")
    L.append("| # | 门槛 | 本条挡掉 | 累计剩余 | 含义 |")
    L.append("|---:|---|---:|---:|---|")
    tot = n
    for i, (k, hit, alive) in enumerate(pf, 1):
        L.append(f"| {i} | `{k}` | {hit} | {alive} | {MATURITY_REASONS.get(k, k)} |")
    L.append("")
    L.append(f"起点 {tot} 个；最后剩下的 {pf[-1][2]} 个即 `active`，"
             f"另有 {int(vc.get('oos_verified', 0))} 个个体证据齐全但因队列类门槛"
             "（`crowded`/`family_cap`/`uncovered`）进不了白名单。")
    L.append("")
    L.append("### 各门槛单独命中数（不累积，看\"如果只看这一条\"）")
    L.append("")
    L.append("| 门槛 | 单独命中 | | 门槛 | 单独命中 |")
    L.append("|---|---:|---|---|---:|")
    gh = gate_hits(df)
    items = list(gh.items())
    half = (len(items) + 1) // 2
    for i in range(half):
        k1, v1 = items[i]
        if i + half < len(items):
            k2, v2 = items[i + half]
            L.append(f"| `{k1}` | {v1} | | `{k2}` | {v2} |")
        else:
            L.append(f"| `{k1}` | {v1} | | | |")
    L.append("")
    L.append("`unstable`/`fading` 只在前面全过之后才判，单独命中数天然偏小；"
             "其余各条相互独立。")
    L.append("")
    L.append("## 三、晋升前后的档位迁移")
    L.append("")
    if moves.empty:
        L.append("（无变化）")
    else:
        t = moves.groupby(["from", "to"]).size().reset_index(name="n")
        L.append("| 原档 | 新档 | 个数 |")
        L.append("|---|---|---:|")
        for _, r in t.sort_values("n", ascending=False).iterrows():
            L.append(f"| `{r['from']}` | `{r['to']}` | {r['n']} |")
    L.append("")
    L.append("## 四、active 白名单")
    L.append("")
    act = df[df["verdict"] == "active"].sort_values("sharpe", ascending=False)
    if act.empty:
        L.append("（空——没有因子同时过七条门槛并证明与已选 active 独立）")
        L.append("")
    else:
        L.append("| # | 因子 | 净多空夏普 | 后 30% 夏普 | 近 250 日 | IC 后段 | 最近邻相关 | 残差 IC |")
        L.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for i, (nm, r) in enumerate(act.iterrows(), 1):
            rid = "-" if not np.isfinite(r["resid_ic"]) else f"{r['resid_ic']:.3f}"
            L.append(f"| {i} | `{nm}` | {r['sharpe']:.2f} | {r['sharpe_fwd']:.2f} | "
                     f"{r['sharpe_recent']:.2f} | {r['ic_fwd']:.3f} | "
                     f"{r['dup_corr_max']:.3f} | {rid} |")
        L.append("")
        L.append("""
「后 30% 夏普」是时间分段的样本外近似，不是严格 OOS：因子是在全样本上被挑出来的，
后段数字自带选择偏差。它能筛掉"靠单段行情撑起来"的因子，但不能当作未来收益的证据。
""")
        # 替身检查：贪心按夏普降序跑，被族配额挡掉的强因子不再参与相关检查，
        # 它的"异名近亲"因此可能补位。这里把每一位的嫌疑替身摊开。
        sub = _standin_check(df, act, corr, corr_names)
        if sub is not None:
            L.append("### 白名单的替身检查")
            L.append("")
            if sub.empty:
                L.append("每个 active 都没有「夏普更高且相关 ≥0.9」的被挡因子——没有替身。")
            else:
                L.append("| active | 嫌疑替身 | 两者相关 | 替身夏普 | 替身被挡原因 |")
                L.append("|---|---|---:|---:|---|")
                for _, r in sub.iterrows():
                    L.append(f"| `{r['active']}` | `{r['standin']}` | {r['corr']:.3f} | "
                             f"{r['standin_sharpe']:.2f} | {r['why']} |")
                L.append("")
                L.append("这些位子严格说是替身：白名单内部仍满足「两两 |corr| < 0.85」，"
                         "但选中的人不是该族最强的那一个。")
            L.append("")
    L.append("## 五、差一步的因子（个体达标、被队列门槛挡掉）")
    L.append("")
    near = df[(df["verdict"] == "oos_verified")
              & df["fails"].map(lambda f: bool(f))]
    if near.empty:
        L.append("（无）")
    else:
        # 按族聚合：同族 40 个 VOLUME 逐个列出来没有信息量，读者要看的是
        # "哪个族被挡住了、它最好的是谁、被挡的是相关还是配额"。
        L.append("| 族代表 | 夏普 | 本族达标数 | 被挡原因 |")
        L.append("|---|---:|---:|---|")
        groups = []
        for fk, grp in near.groupby("family"):
            top = grp.sort_values("sharpe", ascending=False).iloc[0]
            groups.append((str(top.name), float(top["sharpe"]), len(grp), top["fails"]))
        for nm, sh, cnt, fails in sorted(groups, key=lambda r: -r[1]):
            why = "；".join(MATURITY_REASONS.get(k, k) for k in fails)
            L.append(f"| `{nm}` | {sh:.2f} | {cnt} | {why} |")
        L.append("")
        L.append(f"共 {len(near)} 个达标因子散在 {len(groups)} 个族里；"
                 "族数才是这批因子的真实信息量。")
    L.append("")
    L.append("## 六、降级与口径")
    L.append("")
    L.append(f"* 相关矩阵来源：`{corr_src or '（缺失，本次未产生 active）'}`。"
             "子库复用 all_a 的截面签名——因子名同名同义，但严格说这是**借用**证据。")
    L.append("* 门槛现值："
             + "、".join(f"`{k}={v:g}`" for k, v in gates.items()) + "。")
    L.append("* `retired` 只对「曾达标、现在 `unstable`/`fading`」生效；证据缺失"
             "（没 eval / 旧口径 / 没查冗余）只退回 `experimental`——那不是因子失效，"
             "是账没记清。")
    L.append("* `invalid` 由 `dup_audit` 判定，本脚本不改判（面板修好后需人工复核）。")
    L.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="因子成熟度评定与晋升")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--root", default=None, help="库根（默认走 config）")
    ap.add_argument("--apply", action="store_true", help="写回 registry（默认只报告）")
    ap.add_argument("--demote", action="store_true",
                    help="允许把已达标的因子降级（unstable/fading → retired）")
    ap.add_argument("--matrix", default=None, help="相关矩阵 npz 路径")
    ap.add_argument("--no-note", action="store_true", help="不把失败原因写进 note")
    ap.add_argument("--gate", action="append", default=[],
                    help="覆盖门槛：--gate min_sharpe=0.7（可重复）")
    ap.add_argument("--report", default="reports/library_governance/maturity.md",
                    help="报告落盘路径（空串 = 不写）")
    ap.add_argument("--csv", default="reports/library_governance/maturity_all.csv",
                    help="明细落盘路径（空串 = 不写）")
    ap.add_argument("--top", type=int, default=0, help="额外打印前 N 名（按夏普）")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    gates = dict(MATURITY_GATES)
    for kv in args.gate:
        k, _, v = kv.partition("=")
        if k not in gates:
            raise SystemExit(f"未知门槛 {k}；可选：{', '.join(gates)}")
        gates[k] = float(v)

    lib = FactorLibrary(root=args.root, dataset=args.dataset)
    corr, names, src = load_corr(lib, args.matrix)
    print(f"库 {args.dataset}：{lib.root}")
    print(f"相关矩阵：{src or '（缺失）'}")

    df = lib.assess_maturity(corr=corr, corr_names=names, demote=args.demote,
                             gates=gates)
    if df.empty:
        print("空库，无事可做。")
        return

    pf = funnel(df)
    vc = df["verdict"].value_counts()
    print("\n=== 门槛漏斗 ===")
    for i, (k, hit, alive) in enumerate(pf, 1):
        print(f"{i}. {k:14s} 挡掉 {hit:4d}  剩余 {alive:4d}  {MATURITY_REASONS.get(k,'')}")
    print("\n=== 各门槛单独命中（不累积）===")
    for k, v in gate_hits(df).items():
        print(f"  {k:14s} {v:4d}")
    print("\n=== 档位 ===")
    for k in ("active", "oos_verified", "experimental", "retired", "invalid"):
        print(f"  {k:14s} {int(vc.get(k, 0)):4d}")

    act = df[df["verdict"] == "active"].sort_values("sharpe", ascending=False)
    if not act.empty:
        print("\n=== active 白名单 ===")
        for i, (nm, r) in enumerate(act.iterrows(), 1):
            print(f"{i:2d}. {nm:26s} sharpe {r['sharpe']:5.2f}  "
                  f"后段 {r['sharpe_fwd']:5.2f}  近250 {r['sharpe_recent']:5.2f}  "
                  f"ic_fwd {r['ic_fwd']:+.3f}  dup {r['dup_corr_max']:.2f}")
    if args.top:
        print(f"\n=== 夏普前 {args.top}（不论档位）===")
        for nm, r in df.sort_values("sharpe", ascending=False).head(args.top).iterrows():
            why = "；".join(MATURITY_REASONS.get(k, k) for k in r["fails"]) or "-"
            print(f"  {nm:26s} {r['sharpe']:5.2f}  {r['verdict']:12s} {why}")

    applied = 0
    if args.apply:
        applied = lib.apply_maturity(df, note=not args.no_note)
        print(f"\n已写回 registry：{applied} 行变化。")
        bad = lib.contract_violations()
        print(f"契约检查：{len(bad)} 处违规")
        for b in bad[:10]:
            print(f"  {b['name']}: {b['rule']} ({b['detail']})")
    else:
        print("\n（dry-run：未写盘；加 --apply 生效）")

    if args.csv:
        p = Path(args.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        out["fails"] = out["fails"].map(lambda f: ";".join(f))
        out.reset_index().to_csv(p, index=False, encoding="utf-8-sig")
        print(f"明细 → {p}")
    if args.report:
        p = Path(args.report)
        write_report(lib, df, pf, src, gates, p, applied,
                     corr=corr, corr_names=names)
        print(f"报告 → {p}")


if __name__ == "__main__":
    main()
