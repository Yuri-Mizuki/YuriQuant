"""
国君复现·三段纪律评估：train 报告 → valid 筛选 → test 一次性
==============================================================

对应 config.discipline：train 2022-2023（挖掘）/ valid 2024（筛选定权重）/
test 2025 起（冻结，只碰一次）。本脚本在挖掘完成后运行：

1. 全区间 (20220101–20251231) 构建一次环境（面板 + VWAP 执行链 + 可交易掩码），
   按日期切片评估三段——深窗口因子在 valid/test 有天然预热，无泄漏
   （切片只影响统计窗口，因子值仅用当日及之前数据）。
2. **train 段**：报告全部池内因子的费后指标（方向按 train 符号归一）。
3. **valid 段**：筛选——费后夏普 > 0 且按 valid 夏普降序取前 10（研报实践口径）。
4. **test 段**：入选因子 + 等权复合的一次性评估（唯一可信的"未来"数字）。

用法:
    python scripts/gtja_discipline_eval.py --pool reports/gtja_repro/pool.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cli_common import setup_logging  # noqa: E402


import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


log = setup_logging("gtja_discipline_eval")

SIX = ["open", "high", "low", "close", "volume", "vwap"]
TRAIN = (20220101, 20231231)
VALID = (20240101, 20241231)
TEST = (20250101, 20251231)
# 研报第八节滚动实践（2022 以来）：多空费后年化 28.37% / 夏普 1.52 / 回撤 -11.70%
REPORT_PRACTICE = {"ls_ann": 0.2837, "ls_sharpe": 1.52, "ls_dd": -0.1170}


def _slice(df: pd.DataFrame, rng: tuple[int, int]) -> pd.DataFrame:
    return df.loc[[d for d in df.index
                   if rng[0] <= int(d.strftime("%Y%m%d")) <= rng[1]]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="reports/gtja_repro/baseline_pool.csv")
    ap.add_argument("--top", type=int, default=10, help="valid 筛选保留的因子数")
    ap.add_argument("--out", default="reports/gtja_repro/discipline_eval.csv")
    args = ap.parse_args()

    from config import Config
    from data.cache_helpers import build_panel, build_tradable_mask
    from scripts.gtja_repro_eval import TOP_FRAC, FEE_RT, SIX, ls_metrics, load_backward_once
    from factor.gtja import build_vwap_exec_returns
    from factor.formula import formula_builder
    from factor.genetic_mining import _ls_net_stats

    pool = pd.read_csv(args.pool)
    log.info("池内因子 %d 个", len(pool))

    cfg = Config.get()
    cfg["universe"]["default"] = "all_a"
    bwd = load_backward_once()
    panel, _ = build_panel(cfg, 20220101, 20251231, offline=True)
    rets = build_vwap_exec_returns(panel, bwd=bwd)
    if "vwap" not in panel and "amount" in panel:
        panel["vwap"] = panel["amount"] / panel["volume"]
    bwd_al = (bwd.reindex(index=panel["close"].index, columns=panel["close"].columns)
              .ffill() if bwd is not None and len(bwd) else None)
    mask = build_tradable_mask(panel["close"], bwd=bwd_al)
    panel = {k: panel[k] for k in SIX if k in panel}
    feats = list(panel.keys())

    rets_tr, rets_va, rets_te = (_slice(rets, r) for r in (TRAIN, VALID, TEST))
    mask_tr, mask_va, mask_te = (_slice(mask, r) for r in (TRAIN, VALID, TEST))

    rows = []
    for f in pool["formula"]:
        try:
            build = formula_builder(f, features=feats)
            fp = build(panel)
            st_tr = _ls_net_stats(_slice(fp, TRAIN), rets_tr,
                                  top_frac=TOP_FRAC, fee_rt=FEE_RT, tradable=mask_tr)
            if st_tr["n"] < 60 or not np.isfinite(st_tr["ann_ret"]):
                log.warning("train 段无效，跳过: %s", f)
                continue
            sign = 1.0 if st_tr["ann_ret"] >= 0 else -1.0
            fp = fp * sign
            m = {"formula": f, "sign": int(sign)}
            for tag, rng, r_, mk in (("train", TRAIN, rets_tr, mask_tr),
                                     ("valid", VALID, rets_va, mask_va),
                                     ("test", TEST, rets_te, mask_te)):
                mm = ls_metrics(_slice(fp, rng), r_, tradable=mk)
                m.update({f"{tag}_{k}": v for k, v in mm.items()})
            rows.append(m)
        except Exception as exc:
            log.warning("评估失败 %s: %s", f, exc)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("无有效因子")
    # train 段夏普降序
    df = df.sort_values("train_ls_sharpe", ascending=False).reset_index(drop=True)
    df.to_csv(args.out, index=False)
    log.info("已保存 %s", args.out)

    # ---- valid 筛选（研报实践：费后夏普为正且最高的 10 个）----
    sel = df[df["valid_ls_sharpe"] > 0].sort_values("valid_ls_sharpe",
                                                    ascending=False).head(args.top)
    log.info("valid 筛选：%d/%d 因子费后夏普>0，入选 %d 个",
             int((df["valid_ls_sharpe"] > 0).sum()), len(df), len(sel))

    show = ["formula", "sign", "train_ls_ann", "train_ls_sharpe", "train_rankic",
            "valid_ls_sharpe", "valid_rankic",
            "test_ls_ann", "test_ls_dd", "test_ls_sharpe", "test_rankic", "test_ic_win"]
    with pd.option_context("display.width", 250, "display.float_format",
                           lambda v: f"{v:.4f}"):
        print("\n===== 全部因子（train 夏普降序）=====")
        print(df[show].to_string(index=False))
        print("\n===== valid 入选（前 %d）=====" % len(sel))
        print(sel[show].to_string(index=False))
        print("\n===== test 段一次性评估（唯一可信的未来数字）=====")
        print("入选因子 test 指标：")
        print(sel[[c for c in show if c.startswith("test") or c == "formula"]].to_string(index=False))

        comp = pd.DataFrame({
            "研报滚动实践(2022以来)": [REPORT_PRACTICE["ls_ann"], REPORT_PRACTICE["ls_dd"],
                              REPORT_PRACTICE["ls_sharpe"]],
        }, index=["多空年化收益", "最大回撤", "费后夏普"])
        if len(sel):
            comp["本次test入选均值"] = [sel["test_ls_ann"].mean(), sel["test_ls_dd"].mean(),
                                 sel["test_ls_sharpe"].mean()]
            comp["本次test最优"] = [sel["test_ls_ann"].max(), sel["test_ls_dd"].max(),
                             sel["test_ls_sharpe"].max()]
        print("\n===== 与研报滚动实践对比（test 段）=====")
        print(comp.to_string())


if __name__ == "__main__":
    main()
