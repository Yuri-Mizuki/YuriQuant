"""全库面板前视体检（只读）：把「哪几块面板还在用未来数据」从猜测变成清单。

为什么不是查覆盖率
------------------
`year_offset` 的前视 bug（2026-11-04 发现）写出来的列是 ``_off(p) = v(p) − v(p+1y)``。
最近一年的报告期拿不到 ``p+1y`` ⇒ 那些报告期的 ``_off`` 是 NaN ⇒ PIT 展开时
**从一年前最后一个可算的报告期一直前填**。于是面板的**覆盖率不掉**（前填让分母里
一直有值），掉的是**信息**：值不动了。所以「按覆盖率体检」看不见这个 bug，必须看
「值最后一次变化离今天多远」。

方法
----
对每只股票算「最后一次变化的行号」，得到停滞天数 ``stale``；取**中位数**。
然后**在同一个面板上把 as-of 往回退 250 行再算一次**：

* 干净面板：两个 as-of 的中位停滞都≈它自己的更新节奏（季频约 20 日），``jump ≈ 0``；
* 前视面板：as-of 退回一年时正常（约 11 日），as-of 贴到末日时爆到 130~250 日，
  ``jump`` 上百。

只看绝对值不行：年报类因子本来就一年动一次（中位停滞可以到 120 日），绝对阈值会把
它们全打成嫌疑。**``jump`` 是相对它自己的历史节奏**，所以对更新频率免疫。

标定（2026-11-07 实测，250 日窗口）
----------------------------------
===================  ===========  ==========
面板                  中位停滞(末日) jump
===================  ===========  ==========
asset_growth_yoy        172         161
np_accel_sq             166         155
margin_delta_ttm        131         119
sue_q                   ≥250        ≥239
-------------------  -----------  ----------
roe_ttm / inv_rev_gap    20           9
gross_margin             21           9
bp                       0           0
===================  ===========  ==========

``jump ≥ 60 且 stale_now ≥ 100`` 判「尾部新冻结」。已知污染四个全部命中，已知干净三个全部不误伤。
另有 ``sue_q`` / ``margin_delta_ttm`` 这类**早就停更**的：两个 as-of 都饱和在 250 日、
``jump`` 反而是 0，靠第二条规则（近一年窗口内动过的股票占比 ≤1%）抓。

⚠️ 这个体检**测不出部分污染**：`piotroski_f` 九项里六项走 ``_off``，但另外三项（ROA>0、
CFO>0、CFO>NP 的水平项）每天在动，面板整体不冻结（jump 只有 14）。所以静态的
:data:`TAINTED_BY_YEAR_OFFSET` 才是这条 bug 的权威清单，经验扫描只负责抓**没预料到的**
新泄漏。两者是交集关系，不是二选一。

用法::

    python -m scripts.factors.panel_leak_audit                    # 扫 all_a 原始面板
    python -m scripts.factors.panel_leak_audit --layer neu        # 扫中性化面板
    python -m scripts.factors.panel_leak_audit --only sue_q bp    # 抽查
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("panel_leak_audit")

#: 经验扫描的参数（改这些要重跑标定，见模块 docstring 的表）。
WINDOW = 250          # as-of 回看窗口（约一年交易日）
LOOKBACK = 250        # 第一个 as-of 往回退多少行
JUMP_MIN = 60.0       # 判定冻结：最近比一年前多停滞这么多天
STALE_MIN = 100.0     # 且最近中位停滞本身已经这么长
STATIC_MIN = 0.01     # 整窗变化率低于此：一年内几乎没有一只股票动过
COV_MIN = 0.05        # 近一年覆盖率低于此：面板根本没跟上，不是"值没变"

#: `year_offset()` 前视 bug 的**权威**污染清单。
#:
#: 判定依据是代码路径而不是面板症状：这些因子直接读 ``_off_*`` 列，而该列在修复前
#: 是 ``v(p) − v(p+1y)``。逐条锚点见括号（文件:行，行号会漂，锚的是因子名与该文件
#: 里有 ``year_offset(`` 这一事实——`tests/test_panel_leak_audit.py` 会守住）。
TAINTED_BY_YEAR_OFFSET: dict[str, str] = {
    # build_alla_constructed_factors.py:119,122,175,176
    "np_accel_sq": "build_alla_constructed_factors",
    "margin_delta_ttm": "build_alla_constructed_factors",
    # build_alla_p5_factors.py:76-86,170-178
    "sue_q": "build_alla_p5_factors",
    "asset_growth_yoy": "build_alla_p5_factors",
    "piotroski_f": "build_alla_p5_factors",
    "inv_rev_gap": "build_alla_p5_factors",
}


def _staleness(v: np.ndarray, end_row: int, window: int = WINDOW) -> tuple[float, float]:
    """以 ``end_row`` 为 as-of，返回 (各股停滞天数中位数, 窗口内动过的股票占比)。

    「变化」= 相邻两行都非空且值不同，或从 NaN 变非空。窗口内**一次都没动**的股票
    按窗口长度计（最坏情况），不计入分母——否则一只常年静止的股票会把中位数拉没。
    """
    lo = max(0, end_row - window + 1)
    sub = v[lo:end_row + 1]
    diff = np.zeros(sub.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        diff[1:] = (~np.isnan(sub[1:])) & (~np.isnan(sub[:-1])) & \
            (np.abs(sub[1:] - sub[:-1]) > 1e-9)
    appear = np.zeros(sub.shape, dtype=bool)
    appear[1:] = np.isnan(sub[:-1]) & (~np.isnan(sub[1:]))
    chg = diff | appear
    # 反向 argmax 找每列最后一个 True
    rev = chg[::-1]
    any_chg = rev.any(axis=0)
    last_from_end = np.argmax(rev, axis=0).astype(float)   # 无变化时为 0（argmax 全 False）
    stale = np.where(any_chg, last_from_end, float(len(sub) - 1))
    return float(np.median(stale)), float(any_chg.mean())


def audit_panel(path: Path) -> dict:
    """单个面板的体检行。"""
    d = pd.read_parquet(path)
    v = d.to_numpy(dtype=np.float64, copy=False)
    n = len(v)
    cov = np.isfinite(v).mean(axis=1)

    end = n - 1
    stale_now, chg_now = _staleness(v, end)
    stale_prev, _ = _staleness(v, max(WINDOW, end - LOOKBACK))
    jump = stale_now - stale_prev

    # 全历史常量：整块面板就是把首行那一份截面复制了 N 遍（广播 bug 的签名）
    base0 = np.nan_to_num(v[0], nan=0.0)
    const_all = bool(np.all(np.abs(np.nan_to_num(v, nan=0.0) - base0) <= 1e-12))
    cov_last = float(cov[-WINDOW:].mean())

    frozen = bool(const_all) or (jump >= JUMP_MIN and stale_now >= STALE_MIN) or \
        (chg_now <= STATIC_MIN) or (cov_last < COV_MIN)
    if const_all:
        reason = "全历史常量"
    elif cov_last < COV_MIN:
        reason = "尾部无覆盖"
    elif jump >= JUMP_MIN:
        reason = "尾部新冻结"
    elif chg_now <= STATIC_MIN:
        reason = "整窗零变化"
    elif stale_now >= STALE_MIN:
        reason = "长期低刷新（更新慢的因子正常）"
    else:
        reason = ""
    return {
        "name": path.stem,
        "rows": n,
        "last_date": d.index[-1].strftime("%Y-%m-%d"),
        "cov_first": round(float(cov[:WINDOW].mean()), 4),
        "cov_last": round(float(cov[-WINDOW:].mean()), 4),
        "cov_shape": round(float(cov[-WINDOW:].mean() - cov[:WINDOW].mean()), 4),
        "chg_frac_250": round(chg_now, 4),
        "const_all": const_all,
        "stale_now": round(stale_now, 1),
        "stale_prev": round(stale_prev, 1),
        "jump": round(jump, 1),
        "frozen": bool(frozen),
        "reason": reason,
    }


def _resolve_dir(root: Path, dataset: str, layer: str) -> Path:
    sub = {"raw": "panels", "neu": "panels_neu"}[layer]
    return root / dataset / sub


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="全库面板前视体检（只读）")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--root", default=None, help="因子库根（缺省读 config）")
    ap.add_argument("--layer", choices=["raw", "neu"], default="raw")
    ap.add_argument("--only", nargs="+", default=None, help="只扫这些面板名")
    ap.add_argument("--out", default="reports/panel_leak_audit")
    ap.add_argument("--top", type=int, default=25, help="控制台列出前多少个嫌疑")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    if args.root:
        root = Path(args.root)
    else:
        from config import Config
        root = Path(str(Config.get()["factor_library"]["root"]))
    pdir = _resolve_dir(root, args.dataset, args.layer)
    if not pdir.is_dir():
        raise SystemExit(f"目录不存在：{pdir}")

    files = sorted(pdir.glob("*.parquet"))
    if args.only:
        keep = set(args.only)
        files = [f for f in files if f.stem in keep]
        missing = keep - {f.stem for f in files}
        if missing:
            log.warning("这些名字不在目录里：%s", ", ".join(sorted(missing)))
    log.info("体检 %s：%d 个面板", pdir, len(files))

    rows = []
    for i, f in enumerate(files, 1):
        try:
            rows.append(audit_panel(f))
        except Exception as e:                      # 单块坏文件不该毁掉整轮
            log.warning("%s 读取失败：%s", f.stem, e)
            rows.append({"name": f.stem, "rows": 0, "last_date": "ERR", "frozen": False,
                         "err": str(e)[:80]})
        if i % 100 == 0:
            log.info("  已扫 %d/%d", i, len(files))

    table = pd.DataFrame(rows)
    table["tainted"] = table["name"].map(
        lambda n: TAINTED_BY_YEAR_OFFSET.get(n, "") != "")
    table["verdict"] = np.where(
        table["tainted"] & table["frozen"].fillna(False), "tainted+frozen",
        np.where(table["tainted"], "tainted",
                 np.where(table["frozen"].fillna(False), "frozen", "")))

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    order = ["verdict", "frozen", "jump", "stale_now", "stale_prev", "tainted"]
    table = table.sort_values(["frozen", "jump"], ascending=[False, False])
    table.to_csv(dest / f"audit_{args.layer}.csv", index=False, encoding="utf-8-sig")

    frozen = table[table["frozen"].fillna(False)]
    tainted = table[table["tainted"]]
    only_tainted = tainted[~tainted["frozen"].fillna(False)]

    print()
    print(f"== 体检结果（{args.layer}，阈值 jump>={JUMP_MIN:.0f} 且 stale>={STALE_MIN:.0f}）")
    print(f"   面板 {len(table)} 个 · 经验判冻结 {len(frozen)} 个 · 代码静态污染 {len(tainted)} 个")
    if len(frozen):
        cols = ["name", "reason", "jump", "stale_now", "chg_frac_250", "cov_last", "verdict"]
        print("\n-- 经验判冻结（前 %d）--" % min(args.top, len(frozen)))
        print(frozen[cols].head(args.top).to_string(index=False))
    if len(only_tainted):
        print("\n-- 代码污染但面板未冻结（部分污染，只靠经验扫描会漏）--")
        print(only_tainted[["name", "jump", "stale_now", "chg_frac_250"]].to_string(index=False))
    missed = sorted(set(TAINTED_BY_YEAR_OFFSET) - set(table["name"]))
    if missed:
        print("\n-- ⚠️ 静态清单里有、但这次没扫到（换层/换数据集）--")
        print("   " + ", ".join(missed))

    (dest / f"audit_{args.layer}.json").write_text(json.dumps(
        {"window": WINDOW, "lookback": LOOKBACK, "jump_min": JUMP_MIN,
         "stale_min": STALE_MIN, "n_panels": len(table),
         "n_frozen": int(len(frozen)), "n_tainted": int(len(tainted)),
         "frozen": frozen["name"].tolist(),
         "tainted": sorted(TAINTED_BY_YEAR_OFFSET)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("明细 → %s", dest / f"audit_{args.layer}.csv")


if __name__ == "__main__":
    main()
