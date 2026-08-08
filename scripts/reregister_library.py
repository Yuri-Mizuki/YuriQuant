"""
用新回测口径重注册因子库全部因子（空头腿成本：借券费/保证金占用）
================================================================

引擎接入 ShortCostModel（借券费按日计提 + 保证金占用报告，2026-08-05）后，
库内 canonical 回测（ls_M / lo_M / ls_W）的指标仍是旧口径（无借券费）——
多空策略空头腿收益系统性虚高。本脚本按当前引擎默认（或指定参数）重算
每个因子的 IC + 3 套 canonical 回测，使库内 sharpe/年化/回撤反映真实
空头持有成本。

- 真实数据集（hs300_2025 等）：从本地缓存重建收益面板（离线，无需 SDK 在线）。
- mock 数据集：用 AR(1) 信号面板重建（seed=0，确定性）。
- 注册前自动备份 registry.csv（registry.backup_<ts>.csv），可回滚。
- --no-short-cost 可回退旧口径（借券费=0），用于前后对比验证。

用法
----
    python -m scripts.reregister_library --dataset hs300_2025
    python -m scripts.reregister_library --dataset mock
    python -m scripts.reregister_library --dataset hs300_2025 --no-short-cost
    python -m scripts.reregister_library --dataset hs300_2025 --borrow-rate 0.10
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import ShortCostModel
from config import Config
from research.factor_library import FactorLibrary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("reregister_library")


def build_returns(dataset: str, begin: int, end: int) -> pd.DataFrame:
    """按数据集重建次日收益面板（与原始注册同源，保证 IC 不变）。"""
    if dataset == "mock":
        from scripts.mine_factors import gen_mock_panel_with_signal
        panel = gen_mock_panel_with_signal()
        return panel["close"].pct_change().shift(-1)
    from scripts.mine_factors import build_real_panel
    cfg = Config.get()
    _panel, returns = build_real_panel(cfg, begin, end)
    return returns


def main():
    parser = argparse.ArgumentParser(description="用新回测口径重注册因子库（空头腿成本）")
    parser.add_argument("--dataset", default="hs300_2025", help="数据集名（mock 或真实数据集）")
    parser.add_argument("--root", default=None, help="因子库根目录（默认读 settings）")
    parser.add_argument("--begin", type=int, default=20250101, help="收益面板起始（真实数据）")
    parser.add_argument("--end", type=int, default=20251231, help="收益面板结束（真实数据）")
    parser.add_argument("--no-short-cost", action="store_true",
                        help="关闭空头腿成本（借券费=0，旧口径；用于对比验证）")
    parser.add_argument("--borrow-rate", type=float, default=None, help="年化借券费率（默认读配置 0.08）")
    parser.add_argument("--margin-ratio", type=float, default=None, help="融券保证金比例（默认读配置 1.0）")
    parser.add_argument("--deleverage", action="store_true",
                        help="1 倍资金约束：总保证金需求 > 1 时按比例降杠杆")
    args = parser.parse_args()

    cfg = Config.get()
    cfg_bt = dict(cfg.get("backtest", {}))
    short_costs = ShortCostModel(
        borrow_rate=0.0 if args.no_short_cost else (args.borrow_rate if args.borrow_rate is not None
                                                    else cfg_bt.get("short_borrow_rate", 0.08)),
        margin_ratio=args.margin_ratio if args.margin_ratio is not None
                      else cfg_bt.get("short_margin_ratio", 1.0),
    )
    if short_costs.borrow_rate > 0:
        log.info("新口径: 借券费年化 %.2f%% 按日计提, 保证金比例 %.1f%% (--no-short-cost 可回退旧口径)",
                 short_costs.borrow_rate * 100, short_costs.margin_ratio * 100)
    else:
        log.info("旧口径: 借券费=0（--no-short-cost）")

    lib = FactorLibrary(root=args.root, dataset=args.dataset)
    reg = lib.list_all()
    if reg.empty:
        log.error("数据集 %s 为空，无可重注册因子", args.dataset)
        sys.exit(1)

    # 备份 registry（可回滚）
    backup = lib._registry_path.with_name(
        f"registry.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    shutil.copy2(lib._registry_path, backup)
    log.info("registry 已备份: %s", backup)

    returns = build_returns(args.dataset, args.begin, args.end)
    log.info("收益面板: %d 日 × %d 只, 待重注册 %d 个因子",
             returns.shape[0], returns.shape[1], len(reg))

    # 先快照元数据（register 覆盖同名行，避免迭代中 registry 变化）
    meta = []
    for _, r in reg.iterrows():
        parents = [p for p in str(r.get("parents", "")).split("|") if p]
        meta.append({
            "name": r["name"],
            "kind": r["kind"],
            "formula": r.get("formula", r["name"]),
            "parents": parents,
            "source": r.get("source", ""),
        })

    before = {r["name"]: {"ls_M_sharpe": r.get("sharpe_ls_M"), "ls_W_sharpe": r.get("sharpe_ls_W")}
              for _, r in reg.iterrows()}
    after = {}
    for m in meta:
        panel = lib.get_panel(m["name"])
        if panel is None or panel.empty:
            log.warning("  SKIP（无面板）: %s", m["name"])
            continue
        rp = returns.reindex(index=panel.index, columns=panel.columns)
        row = lib.register(
            m["name"], panel, rp,
            kind=m["kind"], formula=m["formula"],
            parents=m["parents"] or None, source=m["source"],
            short_costs=short_costs, deleverage=args.deleverage,
        )
        after[m["name"]] = {"ls_M_sharpe": row.get("sharpe_ls_M"), "ls_W_sharpe": row.get("sharpe_ls_W")}
        b = before.get(m["name"], {})
        log.info("  %-30s kind=%-9s IC=%+.4f  sharpe_ls_M: %+.3f→%+.3f (Δ%+.3f)  best=%+.3f@%s",
                 m["name"], m["kind"], row["ic_mean"],
                 _f(b.get("ls_M_sharpe")), _f(row.get("sharpe_ls_M")),
                 _f(row.get("sharpe_ls_M")) - _f(b.get("ls_M_sharpe")),
                 row["best_sharpe"], row["best_config"])

    # 汇总对比
    print("\n===== 重注册汇总（%s）=====" % args.dataset)
    print(f"{'因子':<32s} {'ls_M 旧→新':>22s} {'ls_W 旧→新':>22s}")
    for name in before:
        b_m, a_m = before[name].get("ls_M_sharpe"), after.get(name, {}).get("ls_M_sharpe")
        b_w, a_w = before[name].get("ls_W_sharpe"), after.get(name, {}).get("ls_W_sharpe")
        print(f"{name:<32s} {_f(b_m):>8.3f}→{_f(a_m):<8.3f} {_f(b_w):>8.3f}→{_f(a_w):<8.3f}")
    print("\n完成。如需回滚: 用 registry.backup_*.csv 覆盖 registry.csv 后重跑。")


def _f(v) -> float:
    try:
        return float(v) if v is not None and not pd.isna(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    main()
