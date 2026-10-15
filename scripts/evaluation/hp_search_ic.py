"""超参 × 预测 IC 小规模对照（**滚动口径**，只比预测 IC，不跑组合回测）。

与 ``reports/hp_sensitivity/``（2026-10-10, P2）的关系
------------------------------------------------------
那份探针已经用 13 配置 one-at-a-time 扫描回答过「当前固化点周围有没有低垂果实」，
结论是**没有**（小幅调整全在 seed 噪声地板 7.6e-4 内；「更大/更快」明确劣化）。
但它有两条边界，本脚本专门补这两条：

  1. **单切分**（1 折拟合/配置，2022–2026）⇒ 其 README 自标「只作敏感度读、
     不作选型依据」。本脚本改成 **滚动重训**（``--refit``，默认 60≈季度，与定版
     回测 ``rolling_grid_alla::N_FOLDS=4`` 同源），给出多折 OOS IC 的稳定性。
  2. **one-at-a-time 单维**，未探参数**交互**，且**容量轴只往上/邻近测**
     （leaves 15→7→31，未测更小的 3/5/10；mcs 只测 100）。本脚本沿
     「**减容量**」方向细扫（hp 已证方向 = 过拟合主导 ⇒ 减容量才是对的方向），
     并加少量的 leaves×mcs 交互。

口径（与 ``retrain_freq_ic.py`` / ``--preset dingban`` 快信号段逐项一致）
------------------------------------------------------------------------
  特征源 = ``panels_neu_fundind``；清单 = ``y{year}__h{h}.json``（按年冻结）
  窗 = 500 个有效标签日滚动；embargo = horizon；模型 = gbdt（定版超参为基准）
  标签 = h 日前瞻 rank + 可交易掩码；评价 = 可交易口径逐日 Spearman IC
  含 ``existence_mask`` 幽灵股守卫。**复用 retrain_freq_ic 的滚动预测实现**。

判据
----
  Δ(配置, default) 只有在**幅值 > seed 噪声地板**时才构成证据。本地板由
  ``seed ∈ {1,7,42}`` 三档 + default 的 IC 散布估计（σ / 极差）。

用法
----
    # 冒烟（OOS 末尾 15 日，只跑 3 个配置，约 2 分钟）
    python -m scripts.evaluation.hp_search_ic --quick

    # 正式：OOS 2025-01-01~2026-09-01，季度重训，14 配置
    python -m scripts.evaluation.hp_search_ic --refit 60 \
        --oos-begin 20250101 --oos-end 20260901
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402
from scripts.evaluation.retrain_freq_ic import (  # noqa: E402
    SEL_DIR,
    TRADABLE_LABELS,
    _rolling_pred,
)

log = setup_logging("hp_search_ic")

# 基准 = 定版固化超参（真源 model.params.DEFAULT_MODEL_PARAMS["gbdt"]）。
# 此处显式复制，避免 import 顺序耦合；若 model/params.py 变动需同步。
DEFAULT_HP = dict(n_estimators=150, learning_rate=0.03, num_leaves=15,
                  min_child_samples=50, seed=0)

# 网格设计（override 叠加在 DEFAULT_HP 上）
#   A. 容量轴细扫（往**更小**推——hp 已证"过拟合主导"，减容量是正确方向，
#      但旧探针只测到 leaves=7）
#   B. 少量交互（one-at-a-time 的盲区）
#   C. seed 地板（估计 Δ 的证据阈值）
GRID: list[tuple[str, dict]] = [
    ("default",   {}),
    # --- A 容量轴：叶子数往小 ---
    ("leaves=10", dict(num_leaves=10)),
    ("leaves=7",  dict(num_leaves=7)),
    ("leaves=5",  dict(num_leaves=5)),
    ("leaves=3",  dict(num_leaves=3)),
    # --- A 容量轴：叶内最小样本往大 ---
    ("mcs=100",   dict(min_child_samples=100)),
    ("mcs=200",   dict(min_child_samples=200)),
    # --- A 容量轴：树深往浅 ---
    ("depth=4",   dict(max_depth=4)),
    ("depth=3",   dict(max_depth=3)),
    # --- B 交互（单维扫描的盲区）---
    ("L5+mcs100", dict(num_leaves=5, min_child_samples=100)),
    ("L7+mcs200", dict(num_leaves=7, min_child_samples=200)),
    # --- C seed 噪声地板 ---
    ("seed=1",    dict(seed=1)),
    ("seed=7",    dict(seed=7)),
    ("seed=42",   dict(seed=42)),
]

FLOOR_KEYS = ("default", "seed=1", "seed=7", "seed=42")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refit", type=int, default=60,
                    help="重训间隔（交易日）；60≈季度（与定版回测 N_FOLDS=4 同源）")
    ap.add_argument("--oos-begin", default=None, help="OOS 起点 YYYYMMDD")
    ap.add_argument("--oos-end", default=None, help="OOS 终点 YYYYMMDD")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--sel-year", type=int, default=None)
    ap.add_argument("--out", default="reports/hp_search_ic")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--quick", action="store_true",
                    help="冒烟：OOS 取末尾 15 日、只跑 default/leaves=7/seed=1")
    args = ap.parse_args(argv)

    from model.labels import build_labels
    from scripts.evaluation.news_alpha_probe import ic_stats, masked_ic
    from scripts.pipelines.rolling_grid_alla import (
        FeatureStore,
        ds_root,
        existence_mask,
        load_base,
    )

    t0 = time.time()
    h = args.horizon
    refit = args.refit
    dest = Path(args.out) / args.out_tag if args.out_tag else Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    grid = GRID[:3] if args.quick else GRID
    if args.quick and refit > 5:
        refit = 5  # 冒烟 OOS 仅 15 日，季度 refit 会触发 "OOS < refit" 守卫

    base = load_base()
    close = base["close"]
    mask = base["mask"].astype(bool)
    all_days = close.index

    # 特征面板覆盖末日：close 可到未来（如 2026-10），但 panels_neu_fundind 未必——
    # OOS 一旦越界，existence_mask 全 False ⇒ IC 静默变 NaN（冒烟实测踩过）。
    _pf = sorted((ds_root() / "panels_neu_fundind").glob("*.parquet"))
    panel_last = (pd.read_parquet(_pf[0], columns=[]).index.max()
                  if _pf else all_days[-1])
    log.info("特征面板覆盖末日：%s（close 末日 %s）",
             pd.Timestamp(panel_last).date(), all_days[-1].date())
    coverage_end = min(pd.Timestamp(all_days[-1]), pd.Timestamp(panel_last))

    # ---- 特征清单（定版 E5 fundind，按年冻结）----
    if args.sel_year is None:
        cands = sorted(SEL_DIR.glob(f"y*__h{h}.json"))
        if not cands:
            raise SystemExit(f"{SEL_DIR} 下无 y*__h{h}.json")
        sel_year = int(cands[-1].name[1:5])
    else:
        sel_year = args.sel_year
    sel_path = SEL_DIR / f"y{sel_year}__h{h}.json"
    if not sel_path.exists():
        raise SystemExit(f"特征清单缺失：{sel_path}")
    names = json.loads(sel_path.read_text(encoding="utf-8"))
    log.info("特征清单 %s：%d 个 | 网格 %d 配置 | refit=%d",
             sel_path.name, len(names), len(grid), refit)

    # ---- OOS 区间（终点受特征面板覆盖约束）----
    if args.quick:
        oos_days = all_days[all_days <= coverage_end][-15:]
    else:
        b = pd.Timestamp(args.oos_begin) if args.oos_begin else all_days[0]
        e = min(pd.Timestamp(args.oos_end), coverage_end) \
            if args.oos_end else coverage_end
        oos_days = all_days[(all_days >= b) & (all_days <= e)]
    if len(oos_days) < refit:
        raise SystemExit(f"OOS {len(oos_days)} 日 < refit {refit}")
    n_folds = len(range(0, len(oos_days), refit))
    log.info("OOS: %s ~ %s（%d 日 / %d 折）",
             oos_days[0].date(), oos_days[-1].date(), len(oos_days), n_folds)

    # ---- 裁剪特征（50 面板 × 全史 ≈ 2.4GB，裁到所需最小范围）----
    i0 = int(all_days.get_indexer([oos_days[0]])[0])
    keep = all_days[max(0, i0 - args.window - h - 20):]
    store = FeatureStore(ds_root() / "panels_neu_fundind")
    feats = store.get_many(names)
    feats = {k: v.reindex(index=keep, columns=close.columns) for k, v in feats.items()}
    log.info("特征加载完成：%d 个 × %d 日（%s ~ %s）",
             len(feats), len(keep), keep[0].date(), keep[-1].date())

    labels, embargo_days = build_labels(
        close, horizon=h, mode="rank",
        tradable_mask=mask if TRADABLE_LABELS else None)
    log.info("标签：rank / embargo=%d / 训练标签掩码=%s", embargo_days, TRADABLE_LABELS)
    fwd = close.pct_change(h, fill_method=None).shift(-h)

    valid = existence_mask(feats, close, oos_days)
    n_ok = int(valid.sum(axis=1).median())
    log.info("existence_mask: 可用股票/日中位 %d", n_ok)
    if n_ok == 0:
        raise SystemExit(
            "existence_mask 全 0 ⇒ OOS 落在特征面板覆盖之外，IC 会全 NaN；"
            "请把 --oos-end 收到面板覆盖日之内")

    rows = []
    for name, override in grid:
        t1 = time.time()
        params = {**DEFAULT_HP, **override}
        raw, n_fit = _rolling_pred(feats, labels, oos_days, all_days,
                                   args.window, embargo_days, 250,
                                   refit, params, {})
        pred = raw.where(valid.reindex(index=oos_days))
        ic = masked_ic(pred, fwd, mask)
        st = ic_stats(ic)
        ic_t = (st["ic"] / (ic.dropna().std() / np.sqrt(st["n"]))
                if st["n"] > 1 and ic.dropna().std() else float("nan"))
        rows.append({
            "config": name,
            "n_estimators": params["n_estimators"],
            "learning_rate": params["learning_rate"],
            "num_leaves": params["num_leaves"],
            "max_depth": params.get("max_depth"),
            "min_child_samples": params["min_child_samples"],
            "seed": params["seed"],
            "n_fits": n_fit,
            "n_ic_days": st["n"],
            "trad_ic": st["ic"],
            "trad_ir": st["ir"],
            "trad_win": st["win"],
            "ic_t": ic_t,
            "sec": round(time.time() - t1, 1),
        })
        log.info("%-11s | 训 %2d 次 | IC=%.5f IR=%.2f win=%.1f%% t=%.2f | %.0fs",
                 name, n_fit, st["ic"], st["ir"], 100 * st["win"], ic_t,
                 time.time() - t1)

    df = pd.DataFrame(rows)
    base_ic = float(df.loc[df["config"] == "default", "trad_ic"].iloc[0])
    df["delta_e4"] = (df["trad_ic"] - base_ic) * 1e4

    # seed 噪声地板：default + 三档 seed 的 IC 散布
    floor_vals = df[df["config"].isin(FLOOR_KEYS)]["trad_ic"].to_numpy()
    floor_sigma = float(np.std(floor_vals, ddof=1)) if len(floor_vals) > 1 else float("nan")
    floor_range = float(np.ptp(floor_vals)) if len(floor_vals) > 1 else float("nan")
    df["within_floor"] = df["delta_e4"].abs() <= abs(floor_sigma) * 1e4

    df = df.sort_values("trad_ic", ascending=False)
    df.to_csv(dest / "summary.csv", index=False, encoding="utf-8-sig")
    meta = {
        "horizon": h, "window": args.window, "refit": refit,
        "sel_year": sel_year, "n_features": len(names),
        "oos_begin": str(oos_days[0].date()), "oos_end": str(oos_days[-1].date()),
        "n_oos_days": len(oos_days), "n_folds": n_folds,
        "tradable_labels": TRADABLE_LABELS, "default_hp": DEFAULT_HP,
        "seed_floor_sigma": floor_sigma, "seed_floor_range": floor_range,
        "seed_floor_keys": list(FLOOR_KEYS), "n_configs": len(grid),
        "boundary_note": "滚动口径补 hp_sensitivity 的单切分边界；"
                         "容量轴往更小推 + 少量交互，为旧探针盲区。",
    }
    (dest / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("完成 %.0fs | seed 地板 σ=%.6f（=%.2fe-4）| 汇总 → %s",
             time.time() - t0, floor_sigma, floor_sigma * 1e4,
             dest / "summary.csv")
    print(df[["config", "trad_ic", "delta_e4", "within_floor",
              "trad_ir", "ic_t", "n_fits"]].to_string(index=False))


if __name__ == "__main__":
    main()
