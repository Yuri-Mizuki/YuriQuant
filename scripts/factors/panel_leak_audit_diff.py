"""把两次体检的 CSV 放到**同一套判据**下重判，输出可比的前后对照。

为什么需要它
------------
`panel_leak_audit` 的判据集合会随认知增补（2026-11-07 就补过 `cov_last < 0.05`）。
一旦在两次扫描之间改了规则，"冻结 33 → 11"这种直接相减是**假的**：被新规则抓到的
面板会被误记成"这次才坏"，反过来旧规则抓到而新规则不管的会被误记成"修好了"。

真实发生过：`is_st` / `st_days` / `suspend_*` / `alpha101_097` / `alpha191_138`
六块面板的原始指标（rows / cov_last / chg_frac_250 / stale_now / jump）两次**逐位相同**，
却因为 before 那份 CSV 是补规则之前生成的（连 `const_all` 列都没有），
`frozen` 列一边 NaN 一边 True。若不重判，会得出"重建引入了 6 块新冻结"的错误结论。

做法
----
判据常量**从** :mod:`scripts.factors.panel_leak_audit` **导入**，不在这里复制一份 ——
复制就会再次漂移。只依赖 CSV 里已经落盘的列，不重读 parquet，所以秒级完成。

用法::

    python -m scripts.factors.panel_leak_audit_diff \\
        reports/panel_leak_audit/before/audit_raw.csv \\
        reports/panel_leak_audit/after/audit_raw.csv

判据缺列时（老版本 CSV 没有 `const_all`）按 False 处理，并在输出里标注
``const_all 不可用``，避免把"没测"当成"没违反"却不吭声。
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

from scripts.factors.panel_leak_audit import (  # noqa: E402
    COV_MIN,
    JUMP_MIN,
    STALE_MIN,
    STATIC_MIN,
)

REQUIRED = ("name", "jump", "stale_now", "chg_frac_250", "cov_last")


def rescore(path: Path) -> tuple[pd.DataFrame, bool]:
    """按当前判据重判一份体检 CSV，返回 (带 frozen/reason 的表, const_all 是否可用)。"""
    t = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in t.columns]
    if missing:
        raise SystemExit(f"{path} 缺列 {missing}，不是 panel_leak_audit 的产物？")

    has_const = "const_all" in t.columns
    # 老 CSV 没有 const_all 列：按 False 处理（真常量面板必然也满足后面的静态/低覆盖规则，
    # 所以不会漏判，只是 reason 会落到"整窗零变化"而不是"全历史常量"）。
    c = t["const_all"].fillna(False).astype(bool) if has_const \
        else pd.Series(False, index=t.index)

    t["frozen"] = (
        c
        | ((t["jump"] >= JUMP_MIN) & (t["stale_now"] >= STALE_MIN))
        | (t["chg_frac_250"] <= STATIC_MIN)
        | (t["cov_last"] < COV_MIN)
    )
    # reason 的优先级与 audit_panel() 保持一致
    t["reason"] = np.where(c, "全历史常量",
                  np.where(t["cov_last"] < COV_MIN, "尾部无覆盖",
                  np.where(t["jump"] >= JUMP_MIN, "尾部新冻结",
                  np.where(t["chg_frac_250"] <= STATIC_MIN, "整窗零变化", ""))))
    return t, has_const


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="两次体检同判据对照")
    ap.add_argument("before", help="重建前 audit CSV")
    ap.add_argument("after", help="重建后 audit CSV")
    ap.add_argument("--out", default=None, help="对照表落盘路径（可选）")
    args = ap.parse_args(argv)

    b, b_const = rescore(Path(args.before))
    a, a_const = rescore(Path(args.after))
    bf, af = set(b.loc[b.frozen, "name"]), set(a.loc[a.frozen, "name"])

    print(f"同判据（jump>={JUMP_MIN:.0f} & stale>={STALE_MIN:.0f} / "
          f"chg<={STATIC_MIN} / cov<{COV_MIN}）")
    if not b_const:
        print("  ⚠️ before 无 const_all 列（旧版 CSV）⇒ 真常量面板归入'整窗零变化'")
    if not a_const:
        print("  ⚠️ after 无 const_all 列（旧版 CSV）")
    print(f"\n冻结：before {len(bf)}/{len(b)}  →  after {len(af)}/{len(a)}")
    print("\n-- 分因（before）--")
    print(b.loc[b.frozen, "reason"].value_counts().to_string() or "  （无）")
    print("\n-- 分因（after）--")
    print(a.loc[a.frozen, "reason"].value_counts().to_string() or "  （无）")
    print(f"\n-- 本次修好（前冻后不冻）{len(bf - af)} 块 --")
    print("  " + (", ".join(sorted(bf - af)) or "（无）"))
    print(f"\n-- 仍冻结 {len(af)} 块 --")
    print("  " + (", ".join(sorted(af)) or "（无）"))
    print(f"\n-- ⚠️ 新增冻结（回归）{len(af - bf)} 块 --")
    print("  " + (", ".join(sorted(af - bf)) or "（无）"))

    if args.out:
        rows = [{"name": n,
                 "state": "修复" if n in bf - af else ("回归" if n in af - bf else "仍冻结"),
                 "before_reason": b.set_index("name").loc[n, "reason"] if n in bf else "",
                 "after_reason": a.set_index("name").loc[n, "reason"] if n in af else ""}
                for n in sorted(bf | af)]
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(dest, index=False, encoding="utf-8-sig")
        print(f"\n明细 → {dest}")


if __name__ == "__main__":
    main()
