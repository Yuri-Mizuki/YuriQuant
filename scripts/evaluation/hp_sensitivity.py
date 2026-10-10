"""模型层超参敏感度探针（P2）。

背景
----
`model/params.py` 的 gbdt 超参（n_est=150 / lr=0.03 / leaves=15 / mcs=50 / seed=0）
**从未做过敏感度检查**——项目自评「模型层成熟度 ≈30%」，但历次结论都是「换模型族
无增量」，从未验证**当前点是否落在一个平台**上。

本探针不搜最优，只回答一个问题：**当前固化点周围有没有被漏掉的低垂果实？**
（单维 one-at-a-time 扫描 + **seed 噪声地板**——后者是关键：若扫描幅度小于 seed 散布，
则整个「超参」维度就是噪声，不必再投预算。）

口径
----
- 特征：`reports/alla_rolling_ortho/selection/y2022__h1.json` 的 50 个因子
  （用 ≤2021 数据选出 ⇒ 对 2022+ 测试段**无前视**），读 `panels_neu` + zscore/±10σ。
- 训练窗：测试段之前 500 个交易日；测试段：`--test-begin`（默认 2022-01-01）之后。
- **单切分**（1 次拟合/配置）⇒ 结果只作**敏感度**读，不作选型依据。
- 标签 = IC 口径 `close_adj.pct_change(1).shift(-1)`；指标 = **可交易 IC**
  （`news_alpha_probe.masked_ic`，T+1 掩码）。

用法
----
    python -m scripts.evaluation.hp_sensitivity
    python -m scripts.evaluation.hp_sensitivity --test-begin 20230101
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

log = setup_logging("hp_sensitivity")

PANEL_DIR = Path(r"E:/data/factor_library/all_a_2018_2026/panels_neu")
SEL = Path(r"reports/alla_rolling_ortho/selection/y2022__h1.json")
TRAIN_WINDOW = 500

# one-at-a-time：以 DEFAULT_MODEL_PARAMS["gbdt"] 为基准
DEFAULT = dict(n_estimators=150, learning_rate=0.03, num_leaves=15,
               min_child_samples=50, seed=0)
GRID: list[tuple[str, dict]] = [
    ("default", {}),
    ("n_est=75", dict(n_estimators=75)),
    ("n_est=300", dict(n_estimators=300)),
    ("n_est=600", dict(n_estimators=600)),
    ("lr=0.015", dict(learning_rate=0.015)),
    ("lr=0.06", dict(learning_rate=0.06)),
    ("leaves=7", dict(num_leaves=7)),
    ("leaves=31", dict(num_leaves=31)),
    ("mcs=100", dict(min_child_samples=100)),
    ("max_depth=4", dict(max_depth=4)),      # 项目从未设过 max_depth（类默认 6）
    ("seed=1", dict(seed=1)),                # ↓ 三档 seed = 噪声地板
    ("seed=7", dict(seed=7)),
    ("seed=42", dict(seed=42)),
]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test-begin", default="20220101")
    ap.add_argument("--out", default="reports/hp_sensitivity")
    args = ap.parse_args(argv)

    from factor.preprocessing import standardize_zscore
    from model.predictor import LGBMPredictor
    from scripts.evaluation.news_alpha_probe import ic_stats, masked_ic
    from scripts.pipelines.rolling_grid_alla import load_base
    from stats.ic import calc_ic_series

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    base = load_base()
    close, mask = base["close"], base["mask"].astype(bool)
    days, cols = close.index, close.columns
    names = json.loads(SEL.read_text())
    log.info("特征 %d 个（%s）· 面板 %s", len(names), SEL.name, PANEL_DIR.name)

    feats = {}
    for n in names:
        p = pd.read_parquet(PANEL_DIR / f"{n}.parquet").reindex(index=days,
                                                               columns=cols)
        feats[n] = (standardize_zscore(p).clip(-10.0, 10.0).astype(np.float32)
                    .replace([np.inf, -np.inf], np.nan))
    log.info("特征载入完成 %.0fs", time.time() - t0)

    labels = close.pct_change(1, fill_method=None).shift(-1)
    test_begin = pd.Timestamp(args.test_begin)
    test_days = days[days >= test_begin]
    cut = days[days < test_begin][-TRAIN_WINDOW:]
    log.info("训练 %s ~ %s（%d 日）| 测试 %s ~ %s（%d 日）",
             cut[0].date(), cut[-1].date(), len(cut),
             test_days[0].date(), test_days[-1].date(), len(test_days))

    f_train = {k: v.loc[cut] for k, v in feats.items()}
    f_test = {k: v.loc[test_days] for k, v in feats.items()}
    y_train = labels.loc[cut]
    fwd_ic = labels.reindex(index=test_days, columns=cols)
    mask_test = mask.reindex(index=test_days, columns=cols).fillna(True)

    rows = []
    for tag, over in GRID:
        params = {**DEFAULT, **over}
        t1 = time.time()
        m = LGBMPredictor(**params)
        m.fit(f_train, y_train)
        pred = m.predict(f_test).reindex(index=test_days, columns=cols)
        full = ic_stats(calc_ic_series(pred, fwd_ic))
        trad = ic_stats(masked_ic(pred, fwd_ic, mask_test))
        rows.append(dict(config=tag, **{k: v for k, v in params.items()},
                         full_ic=full["ic"], trad_ic=trad["ic"],
                         trad_ir=trad["ir"], trad_win=trad["win"],
                         secs=round(time.time() - t1, 1)))
        log.info("%-14s 全样本IC=%+.4f | 可交易IC=%+.4f IR=%.3f 胜率=%.0f%% (%.0fs)",
                 tag, full["ic"], trad["ic"], trad["ir"], trad["win"] * 100,
                 time.time() - t1)

    df = pd.DataFrame(rows)
    df.to_csv(dest / "results.csv", index=False, encoding="utf-8-sig")

    base_ic = float(df.loc[df.config == "default", "trad_ic"].iloc[0])
    seeds = df[df.config.str.startswith("seed=")]["trad_ic"]
    log.info("\n===== 敏感度（相对 default 的可交易 IC Δ，单位 1e-4）=====")
    for _, r in df.iterrows():
        log.info("%-14s Δ=%+7.1f  可交易IC=%.4f", r["config"],
                 (r["trad_ic"] - base_ic) * 1e4, r["trad_ic"])
    if len(seeds) > 1:
        log.info("seed 噪声地板：跨 %d 个 seed 的可交易 IC 散布 = %.1f (1e-4) "
                 "（σ=%.1f）— 任何小于该幅度的 Δ 都不构成证据",
                 len(seeds), (seeds.max() - seeds.min()) * 1e4, seeds.std() * 1e4)
    log.info("完成 %.0fs -> %s", time.time() - t0, dest / "results.csv")


if __name__ == "__main__":
    main()
