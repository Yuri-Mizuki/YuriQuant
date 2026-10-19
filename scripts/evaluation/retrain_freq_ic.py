"""重训频率 × 预测 IC 小规模对照（只比预测 IC，**不跑组合回测**）。

问题
----
定版口径的**回测**是"年内 4 折（季度）"重训（`rolling_grid_alla::N_FOLDS=4`，
h>=10 用 2 折），而**生产出榜链**是"每个交易日"重训
（`alla_daily_rank::train_and_predict`）——两者不同源，且全项目**从未做过**
重训频率对照（`reports/experiments.csv` 里出现过 n_folds 变动的记录全是 mock
小样本或别的算法口径）。本脚本只回答一个问题：

    **在定版快信号口径下，重训频率（季度/月度/周度/逐日）对 OOS 预测 IC 有何影响？**

口径（= `--preset dingban` 的**快信号**部分，逐项对齐）
--------------------------------------------------------
  特征源 = `panels_neu_fundind`（E5 fundind 正交化）
  清单   = `reports/alla_rolling_ortho920fundind/selection/y{year}__h{h}.json`
  窗     = 500 个**有效标签日**滚动；embargo = horizon
  模型   = gbdt（`model.params.DEFAULT_MODEL_PARAMS["gbdt"]`，seed=0）
  标签   = h 日前瞻收益截面 rank + 可交易掩码（定版 `tradable_labels=True`）
  评价   = 可交易口径逐日 Spearman IC（复用 `news_alpha_probe::masked_ic`）
          + `existence_mask` 幽灵股守卫（同 `stage_predict`）

为什么只比 IC
-------------
重训频率只影响"模型新鲜度"这一件事；组合层（buffer 门槛 / 成本 / 换手）会把
它稀释甚至掩盖。IC 是纯信号口径、无成本假设，且比完整回测快一个数量级。

效率
----
各档折起点都是 `oos_days` 的等距采样且同起点 ⇒ **refit=1 的折起点集合
是其他所有档的超集**，训练结果按折起点缓存复用 ⇒ 总训练次数 = |oos_days|
（即逐日档的折数），其余档位零额外成本。

用法
----
    # 冒烟（约 2 分钟）：OOS 取末尾 15 日 × {5,1}
    python -m scripts.evaluation.retrain_freq_ic --quick

    # 正式对照：OOS=2026-01-01~2026-09-01，档位 {60,20,5,1}
    python -m scripts.evaluation.retrain_freq_ic --refits 60,20,5,1 \
        --oos-begin 20260101 --oos-end 20260901
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
from stats.significance import mean_inference  # noqa: E402

log = setup_logging("retrain_freq_ic")

SEL_DIR = ROOT / "reports" / "alla_rolling_ortho920fundind" / "selection"
# 定版口径的训练标签是否用可交易掩码（真源 = alla_daily_rank.PRESETS["dingban"]
# ["tradable_labels"]，2026-10-15 由莉酱拍板 True）。此处显式复制而非 import
# `alla_daily_rank`——避免为取一个常量扛入该脚本的整条依赖链。
TRADABLE_LABELS = True


def rolling_pred(feats: dict[str, pd.DataFrame], labels: pd.DataFrame,
                 oos_days: pd.DatetimeIndex, all_days: pd.DatetimeIndex,
                 window: int, embargo: int, min_train: int,
                 refit: int, params: dict,
                 cache: dict) -> tuple[pd.DataFrame, int]:
    """每 ``refit`` 个交易日重训一次；折起点相同的模型跨档位复用（见模块 docstring）。

    返回 (OOS 预测面板, 本次新增训练次数)。

    公开名（2026-10-16）：``hp_search_ic`` 跨模块消费同一实现，原 ``_rolling_pred``
    触发 ``test_layering`` 私有名守卫；重训频率/超参双探针共用此滚动预测核心。
    """
    from model.predictor import LGBMPredictor

    out = pd.DataFrame(np.nan, index=oos_days, columns=labels.columns)
    n_fit = 0
    for i in range(0, len(oos_days), refit):
        chunk = oos_days[i:i + refit]
        key = chunk[0]                      # 训练段只由折起点决定 ⇒ 可作为缓存键
        model = cache.get(key)
        if model is None:
            cut = all_days[all_days < chunk[0]]
            if embargo and len(cut) > embargo:
                cut = cut[:-embargo]
            tr = cut[-window:]
            if len(tr) < min_train:
                log.warning("  折 %s 训练段不足（%d < %d），跳过",
                            chunk[0].date(), len(tr), min_train)
                continue
            model = LGBMPredictor(**params)
            model.fit({k: v.loc[tr] for k, v in feats.items()}, labels.loc[tr])
            cache[key] = model
            n_fit += 1
        out.loc[chunk] = model.predict(
            {k: v.loc[chunk] for k, v in feats.items()})
    return out, n_fit


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refits", default="60,20,5,1",
                    help="重训间隔（交易日）列表，逗号分隔；60≈季度(4折)/20≈月度/"
                         "5≈周度/1=逐日")
    ap.add_argument("--oos-begin", default=None, help="OOS 起点 YYYYMMDD（默认全期）")
    ap.add_argument("--oos-end", default=None, help="OOS 终点 YYYYMMDD（默认全期）")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--window", type=int, default=500, help="滚动训练窗（交易日）")
    ap.add_argument("--sel-year", type=int, default=None,
                    help="用哪一年的特征清单（默认 = 清单目录里最新年份）")
    ap.add_argument("--out", default="reports/retrain_freq_ic")
    ap.add_argument("--out-tag", default="", help="产出子目录后缀")
    ap.add_argument("--quick", action="store_true",
                    help="冒烟：OOS 取末尾 15 日、档位 {5,1}（覆盖 --oos-*/--refits）")
    args = ap.parse_args(argv)

    if args.quick:
        args.refits = "5,1"

    from model.labels import build_labels
    from model.params import DEFAULT_MODEL_PARAMS
    from scripts.evaluation.news_alpha_probe import ic_stats, masked_ic
    from scripts.common.alla_common import (
        FeatureStore,
        ds_root,
        existence_mask,
        load_base,
    )

    t0 = time.time()
    refits = [int(x) for x in str(args.refits).split(",") if x.strip()]
    params = dict(DEFAULT_MODEL_PARAMS["gbdt"])
    h = args.horizon
    dest = Path(args.out) / args.out_tag if args.out_tag else Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    base = load_base()
    close = base["close"]
    mask = base["mask"].astype(bool)
    all_days = close.index

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
    log.info("特征清单 %s：%d 个", sel_path.name, len(names))

    # ---- OOS 区间 ----
    if args.quick:
        oos_days = all_days[-15:]
    else:
        b = pd.Timestamp(args.oos_begin) if args.oos_begin else all_days[0]
        e = pd.Timestamp(args.oos_end) if args.oos_end else all_days[-1]
        oos_days = all_days[(all_days >= b) & (all_days <= e)]
    if len(oos_days) < max(refits):
        raise SystemExit(f"OOS {len(oos_days)} 日 < 最大档位 {max(refits)}")
    log.info("OOS: %s ~ %s（%d 日）| 档位 %s",
             oos_days[0].date(), oos_days[-1].date(), len(oos_days),
             {r: ("季度" if r >= 60 else "月度" if r >= 20
                  else "周度" if r > 1 else "逐日") for r in refits})

    # ---- 裁剪特征到所需最小日期范围（省内存：50 面板 × 全史 ≈ 2.4GB）----
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

    # 幽灵股守卫（同 stage_predict：未上市/无行情/特征不可用）
    valid = existence_mask(feats, close, oos_days)
    log.info("existence_mask: 可用股票/日中位 %d", int(valid.sum(axis=1).median()))

    cache: dict = {}
    rows = []
    for refit in sorted(refits, reverse=True):
        t1 = time.time()
        raw, n_fit = rolling_pred(feats, labels, oos_days, all_days,
                                  args.window, embargo_days, 250,
                                  refit, params, cache)
        pred = raw.where(valid.reindex(index=oos_days))
        ic = masked_ic(pred, fwd, mask)
        st = ic_stats(ic)
        n_chunks = len(range(0, len(oos_days), refit))
        rows.append({
            "refit_every": refit,
            "n_folds": n_chunks,
            "n_fits": n_fit,
            "n_fits_cached": n_chunks - n_fit,
            "n_ic_days": st["n"],
            "ic_mean": st["ic"],
            "ic_ir": st["ir"],
            "ic_win": st["win"],
            "ic_t": mean_inference(ic, robust=False)["t_stat"],
            "sec": round(time.time() - t1, 1),
        })
        ic.to_frame("ic").to_csv(dest / f"ic_refit{refit:03d}.csv",
                                 encoding="utf-8-sig")
        log.info("refit=%3d 日 | 新训 %2d 次 | IC=%.4f IR=%.2f win=%.1f%% "
                 "t=%.2f | %.0fs", refit, n_fit, st["ic"], st["ir"],
                 100 * st["win"], rows[-1]["ic_t"], time.time() - t1)

    df = pd.DataFrame(rows).sort_values("refit_every")
    df.to_csv(dest / "summary.csv", index=False, encoding="utf-8-sig")
    meta = {"horizon": h, "window": args.window, "sel_year": sel_year,
            "n_features": len(names), "oos_begin": str(oos_days[0].date()),
            "oos_end": str(oos_days[-1].date()), "n_oos_days": len(oos_days),
            "tradable_labels": TRADABLE_LABELS,
            "gbdt_params": params, "refits": refits,
            "total_fits": len(cache)}
    (dest / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("完成 %.0fs | 总训练 %d 次 | 汇总 → %s",
             time.time() - t0, len(cache), dest / "summary.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
