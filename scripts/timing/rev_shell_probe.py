"""周频均值回归壳稳健性探针（纯择时线缝合清单③，reports/rev_shell_probe/）。

背景：sws_timing_probe（2026-11-21 判负）里唯一正超额形态是**纯反转退化腿**——
每周三按回看期收益反向调仓（每满 1% 动 0.1 仓，钳 [0,1]，首日 0.5），2020-2025 上证
年化 6.88% / Sharpe 0.81，vs 半仓持有 +4.4pp/年。单资产单路径，不能直接缝合；
本探针按预注册六门验证其稳健性。

预注册六门（跑之前写死；被门臂 = **报告参数臂** step=1%/0.1 仓/周三，成本 10bp/边；
参数网格只做敏感性报告、**不用于选峰**）：

| 门 | 条件 |
|---|---|
| G1 边际 | 全样本（2016-01-01 起，上证）年化 − 半仓持有 ≥ +2.0pp |
| G2 逐年 | 正超额年份 ≥ 6/11（2016–2026） |
| G3 成本 | 30bp/边 口径年化超额仍 > 0 |
| G4 回撤 | 全样本 MDD ≤ 半仓持有 MDD + 3pp |
| G5 急涨踏空 | 指数单月涨幅 Top5 月份的策略月超额均值 ≥ −3pp |
| G6 跨资产 | 沪深300 / 中证500 同参数 10bp 全样本年化超额，≥ 2/3 资产为正 |

全过 ⇒ 记「采纳候选」（可进组合层缝合评估）；任一 FAIL ⇒ 判负收口。
统计附件：全样本日差（壳 − 半仓）配对 t（Newey-West lag5）。

用法::

    python -m scripts.timing.rev_shell_probe               # 全量跑
    python -m scripts.timing.rev_shell_probe --self-check  # 不变量抽查
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.metrics import annual_return, max_drawdown, sharpe_ratio
from scripts.timing.shell import load_index_daily, run_rebalance_shell
from stats import PERIODS_PER_YEAR

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "rev_shell_probe"

FULL_BEGIN = "2016-01-01"  # 全样本窗（数据从 2014 拉，留缓冲）
COST_MAIN_BPS = 10.0
COST_GRID = (0.0, 10.0, 30.0)
STEP_PCT_GRID = (0.5, 1.0, 2.0)  # 每满 X% 动 0.1 仓（敏感性，不选峰）
PRIMARY = "000001.SH"  # 上证指数（被门资产）
CROSS_ASSETS = ("000300.SH", "000905.SH")  # 沪深300 / 中证500（G6）


# ---------------------------------------------------------------- 指标
def metrics_row(ret: pd.Series) -> dict:
    return {
        "total": float((1 + ret).prod() - 1),
        "annual": annual_return(ret),
        "vol": float(ret.std() * np.sqrt(PERIODS_PER_YEAR)),
        "sharpe": sharpe_ratio(ret),
        "mdd": max_drawdown(ret),
    }


# NW t 唯一真源在 stats.robust_stats（2026-11-22 收口，删本地第二实现；
# 旧 /n 分母口径与真源 /(n-j) 有 ~1-2% 差异，本探针未出过结论、无历史包袱）。
from stats.robust_stats import nw_t  # noqa: E402


def run_arm(close: pd.Series, cost_bps: float, step_pct: float) -> dict:
    """纯反转臂：sig 恒 0（永远执行震荡/均值回归逻辑）。"""
    sig = pd.Series(0.0, index=close.index)
    return run_rebalance_shell(close, sig, cost_bps=cost_bps, step_pct=step_pct)


# ---------------------------------------------------------------- self-check
def self_check(close: pd.Series) -> list[str]:
    bad: list[str] = []
    win = close.loc[FULL_BEGIN:]
    r = run_arm(win, COST_MAIN_BPS, 1.0)
    p = r["pos"]
    if not ((p >= 0).all() and (p <= 1).all()) or not np.isfinite(p).all():
        bad.append("仓位越界 [0,1] 或含非有限值")
    if not ((0.0 <= r["avg_pos"] <= 1.0)):
        bad.append(f"avg_pos 越界: {r['avg_pos']}")
    # 对称性：纯动量臂（sig 恒 1）与纯反转臂的换手应几乎相同（同一条回看收益序列）
    r_mom = run_rebalance_shell(win, pd.Series(1.0, index=win.index), cost_bps=COST_MAIN_BPS)
    if abs(r["turnover"] - r_mom["turnover"]) > 1e-6 * max(1.0, r["turnover"]):
        bad.append(f"反转/动量换手不对称: {r['turnover']:.4f} vs {r_mom['turnover']:.4f}")
    return bad


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    codes = (PRIMARY,) + CROSS_ASSETS
    data = {}
    for code in codes:
        df = load_index_daily(code, cache_dir=OUT / "data", refresh=args.refresh_data)
        data[code] = df["close"]
        print(f"{code}: {df.index.min().date()} ~ {df.index.max().date()}（{len(df)} 行）")

    if args.self_check:
        bad = self_check(data[PRIMARY])
        for b in bad:
            print("SELF-CHECK FAIL:", b)
        if bad:
            return 1
        print("SELF-CHECK PASS（仓位/换手对称性不变量全过）")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- 主资产：全样本 × 成本 × 步长网格（网格只报告，门在 step=1.0 判） ----------
    close = data[PRIMARY].loc[FULL_BEGIN:]
    bench_half = close.pct_change().fillna(0.0) * 0.5
    bench_full = close.pct_change().fillna(0.0)
    rows = []
    for sp in STEP_PCT_GRID:
        for cbps in COST_GRID:
            r = run_arm(close, cbps, sp)
            m = metrics_row(r["ret"])
            rows.append({"asset": PRIMARY, "step_pct": sp, "cost_bps": cbps, **m,
                         "avg_pos": r["avg_pos"], "turnover": r["turnover"]})
    for name, ret, w in (("bench_half_bh", bench_half, 0.5), ("bench_full_bh", bench_full, 1.0)):
        rows.append({"asset": PRIMARY, "step_pct": float("nan"), "cost_bps": 0.0,
                     **metrics_row(ret), "avg_pos": w, "turnover": 0.0})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "grid_primary.csv", index=False, encoding="utf-8-sig")
    print("\n[网格] 上证全样本（门在 step=1.0 / 10bp 判；网格不选峰）")
    print(grid.round(4).to_string(index=False))

    arm = grid[(grid["step_pct"] == 1.0) & (grid["cost_bps"] == COST_MAIN_BPS)].iloc[0]
    arm30 = grid[(grid["step_pct"] == 1.0) & (grid["cost_bps"] == 30.0)].iloc[0]
    bh = grid[grid["step_pct"].isna() & (grid["avg_pos"] == 0.5)].iloc[0]

    # ---------- G1/G2/G4/G5：报告参数臂 vs 半仓持有 ----------
    ret_arm = run_arm(close, COST_MAIN_BPS, 1.0)["ret"]
    diff = ret_arm - bench_half
    g1_excess = float(arm["annual"] - bh["annual"])
    t_nw = nw_t(diff)

    yearly = []
    for y in sorted(set(diff.index.year)):
        d = diff.loc[diff.index.year == y]
        if len(d) < 60:
            continue
        yearly.append({"year": int(y), "excess_annual": annual_return(d),
                       "excess_total": float((1 + d).prod() - 1)})
    yearly_df = pd.DataFrame(yearly)
    n_pos = int((yearly_df["excess_total"] > 0).sum())
    yearly_df.to_csv(OUT / "yearly_excess.csv", index=False, encoding="utf-8-sig")

    m_ret = run_arm(close, COST_MAIN_BPS, 1.0)["ret"]
    m_month = (1 + m_ret).resample("ME").prod() - 1
    b_month = (1 + bench_half).resample("ME").prod() - 1
    i_month = (1 + close.pct_change().fillna(0.0)).resample("ME").prod() - 1
    top5 = i_month.nlargest(5)
    g5_mean = float((m_month.loc[top5.index] - b_month.loc[top5.index]).mean())

    g3_excess = float(arm30["annual"] - bh["annual"])
    g4_ok = float(arm["mdd"]) <= float(bh["mdd"]) + 0.03

    gates = {
        "G1 边际": (g1_excess >= 0.02, f"全样本超额 {g1_excess * 100:+.2f}pp（判据 ≥ +2.0pp）；NW-t {t_nw:.2f}"),
        "G2 逐年": (n_pos >= 6, f"正超额年份 {n_pos}/{len(yearly_df)}（判据 ≥ 6）"),
        "G3 成本": (g3_excess > 0, f"30bp 口径超额 {g3_excess * 100:+.2f}pp（判据 > 0）"),
        "G4 回撤": (g4_ok, f"壳 MDD {arm['mdd'] * 100:.2f}% vs 半仓 {bh['mdd'] * 100:.2f}% + 3pp"),
        "G5 急涨": (g5_mean >= -0.03, f"指数最强 5 个月均超额 {g5_mean * 100:+.2f}%（判据 ≥ −3pp）"),
    }

    # ---------- G6：跨资产 ----------
    cross_rows = []
    n_pos_asset = 1  # 上证本身按 G1 的方向计（G1 过门才计入；此处先按超额符号）
    n_pos_asset = int(g1_excess > 0)
    for code in CROSS_ASSETS:
        c = data[code].loc[FULL_BEGIN:]
        b_half = c.pct_change().fillna(0.0) * 0.5
        ra = run_arm(c, COST_MAIN_BPS, 1.0)["ret"]
        m = metrics_row(ra)
        ex = m["annual"] - annual_return(b_half)
        n_pos_asset += int(ex > 0)
        cross_rows.append({"asset": code, **m, "annual_excess": ex,
                           "avg_pos": run_arm(c, COST_MAIN_BPS, 1.0)["avg_pos"]})
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(OUT / "cross_assets.csv", index=False, encoding="utf-8-sig")
    gates["G6 跨资产"] = (n_pos_asset >= 2, f"年化超额为正资产 {n_pos_asset}/3："
                          + "；".join(f"{r.asset} {r.annual_excess * 100:+.2f}pp"
                                      for r in cross_df.itertuples()))

    print("\n[跨资产] 同参数 10bp 全样本")
    print(cross_df.round(4).to_string(index=False))

    # ---------- 判定 ----------
    print("\n===== 预注册六门（被门臂 = step1%/0.1仓/周三/10bp，上证 2016→now） =====")
    for g, (ok, msg) in gates.items():
        print(f"{g}: {'PASS' if ok else 'FAIL'}  {msg}")
    verdict = pd.DataFrame(
        [{"gate": g, "pass": ok, "detail": msg} for g, (ok, msg) in gates.items()]
    )
    verdict.to_csv(OUT / "gates.csv", index=False, encoding="utf-8-sig")
    all_pass = all(ok for ok, _ in gates.values())
    print("\n结论：", "六门全过 → 记「采纳候选」，可进组合层缝合评估" if all_pass
          else "任一 FAIL → 判负收口（壳留作备忘，不缝合）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
