"""buffer 档位选择验证（网格 + 样本外 + 权重稳健性）

为什么单独一个脚本
==================

`etf_contest5_eval` 只评**固定档位**；本脚本回答另一个问题：**某个档位值不值得选**。
2026-10-29 在 `exit_rank=20` 上踩过三次坑，教训逐条固化成默认输出：

  1. **`enter_rank` 是死参数**：45 只池 + Top3 持仓下，任何 ``enter_rank >= 3`` 净值逐位相同
     （``strategy/etf_rotation.py::_select`` 有「不够 top_k 就用剩余排名补齐」兜底）。
     ``--full-grid`` 会打印退化分组，直接看到这件事。
  2. **单点最差窗口不能判档**：它随切窗方式翻转。默认给**非重叠 2 月期**（赛制真实计分单位），
     并给 p10（分位数尾部）而不是只给 min。
  3. **全样本优势可能是时段特定**：默认做前后半分割 + 逐年滚动选档，两者都用「t 之前的信息」
     选档，再看实现收益 —— 这是唯一能识别「事后挑档」的检验。
  4. **动态权重多半没用**：默认对比滚动定权 vs 固定权重，用实现收益说话。

用法::

    python -m scripts.portfolio.etf_contest5_buffer_oos              # 默认：4 档 + OOS + 权重
    python -m scripts.portfolio.etf_contest5_buffer_oos --full-grid  # 另跑 5x5 网格查退化

所有口径均为 T+1 open（与生产同源）。证据落 ``reports/etf_contest5/README.md`` §4.1–4.2。
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest.engine import build_execution_split
from data.cache import DataCache
from data.datasource import create_datasource
from scripts.portfolio.etf_champ_signals import PROFILES
from scripts.portfolio.etf_contest_spec import EXCLUDED_5
from scripts.portfolio.etf_rotation_panels import build_extra_panels
from scripts.portfolio.etf_rotation_probe import (
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)

BEGIN = 20200101
BASE = dict(PROFILES["contest3"])

#: 有效档位（``enter_rank`` 已证明无效，只列 ``exit_rank``）。None = 三锋（无缓冲）。
EXITS: dict[str, int | None] = {"c3": None, "x10": 10, "x15": 15, "x20": 20, "x30": 30}
GRID_E = (3, 5, 8, 10, 15)
GRID_X = (10, 15, 20, 30, 40)


def _load():
    cache = DataCache(create_datasource())
    bundle = load_bundle(cache, WARMUP_BEGIN, None)
    _uni, close, amount, _bench = bundle
    raw = cache.read_daily("etf")
    rc = raw["close"].unstack("code").reindex(index=close.index, columns=close.columns)
    ro = raw["open"].unstack("code").reindex(index=close.index, columns=close.columns)
    open_adj = (ro * (close / rc)).astype(np.float32)
    panels = build_extra_panels(close, build_panels(close, amount), rc)
    for x in EXCLUDED_5:
        panels["mask"].loc[:, x] = False
    return bundle, panels, open_adj, close


def _run_one(bundle, panels, open_adj, close, exit_rank: int | None) -> dict:
    """跑单档，返回 equity 与 open 口径指标。

    🚨 ``enter_rank`` 必须**同时**设置：`strategy/etf_rotation.py::_select` 里
    ``if not self.enter_rank: return ranked[:top_k]`` —— 它是 buffer 逻辑的**开关**，
    只设 ``exit_rank`` 会**静默退化**成无缓冲（2026-10-29 实测踩过：5 档结果逐位相同）。
    开关一旦打开，值本身在 ``>= top_k`` 时无影响（3/5/8/10/15 净值逐位相同）。
    """
    spec = dict(BASE)
    if exit_rank is not None:
        spec.update({"enter_rank": 3, "exit_rank": exit_rank})
    r0 = run_variant(bundle, panels, dict(spec), BEGIN, None)
    spl = build_execution_split(open_adj, close,
                                set(pd.DatetimeIndex(r0["rebalance_dates"])))
    r1 = run_variant(bundle, panels, dict(spec), BEGIN, None,
                     execution_split=spl, execution_shift=True)
    return {"equity": r1["equity"], "ann": r1["annual_return"], "sharpe": r1["sharpe"],
            "mdd": r1["max_drawdown"], "turnover": r1["avg_turnover"]}


def _periods(equity: pd.Series) -> pd.Series:
    """非重叠 2 月期收益（赛制真实计分单位）。"""
    return equity.resample("2ME").last().pct_change()


def _fmt_period_row(name: str, s: pd.Series) -> str:
    return (f"{name:<6}{s.mean():>+8.2%}{s.std():>8.2%}{s.mean() / s.std():>8.2f}"
            f"{s.quantile(0.10):>+9.2%}{s.min():>+9.2%}{(s > 0).mean():>8.1%}")


PER_HDR = f"{'档位':<6}{'均值':>8}{'波动':>8}{'期SR':>8}{'p10':>9}{'最差':>9}{'胜率':>8}"


def report_grid(bundle, panels, open_adj, close) -> None:
    print("\n== 5x5 网格（enter_rank x exit_rank）：查退化 ==")
    eqs, rows = {}, []
    for e in GRID_E:
        for x in GRID_X:
            spec = {**BASE, "enter_rank": e, "exit_rank": x}
            r0 = run_variant(bundle, panels, dict(spec), BEGIN, None)
            spl = build_execution_split(open_adj, close,
                                        set(pd.DatetimeIndex(r0["rebalance_dates"])))
            r1 = run_variant(bundle, panels, dict(spec), BEGIN, None,
                             execution_split=spl, execution_shift=True)
            tag = f"e{e}_x{x}"
            eqs[tag] = r1["equity"]
            per = _periods(r1["equity"]).dropna()
            rows.append({"enter": e, "exit": x, "ann": r1["annual_return"],
                         "psr": per.mean() / per.std()})
            print(f"[done] {tag} ann={r1['annual_return']:.4f}", flush=True)

    d = pd.DataFrame(rows)
    tags, seen, groups = list(eqs), set(), []
    for a in tags:
        if a in seen:
            continue
        same = [b for b in tags if eqs[a].equals(eqs[b])]
        seen.update(same)
        groups.append(same)
    print("\n相同净值的档位组（退化）：")
    for g in groups:
        if len(g) > 1:
            print("  ", g)
    print("\n期 Sharpe（行=enter，列=exit，x100）：")
    print((d.pivot(index="enter", columns="exit", values="psr") * 100).round(1).to_string())
    print("\n年化（%，行=enter，列=exit）：")
    print((d.pivot(index="enter", columns="exit", values="ann") * 100).round(1).to_string())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="buffer 档位选择验证（网格 / 样本外 / 权重）")
    ap.add_argument("--full-grid", action="store_true", help="另跑 5x5 网格查退化（约 2 分钟）")
    ap.add_argument("--csv", default=None, help="非重叠期收益落盘路径")
    args = ap.parse_args(argv)

    bundle, panels, open_adj, close = _load()
    if args.full_grid:
        report_grid(bundle, panels, open_adj, close)

    print("\n== 单档指标（非重叠 2 月期，T+1 open）==")
    print(PER_HDR)
    res, per = {}, {}
    for tag, xr in EXITS.items():
        r = _run_one(bundle, panels, open_adj, close, xr)
        p = _periods(r["equity"]).dropna()
        res[tag], per[tag] = r, p
        print(_fmt_period_row(tag, p), flush=True)

    P = pd.DataFrame(per).dropna()
    # 防呆：缓冲档若与 c3 逐位相同，说明 enter_rank 开关没打开（静默退化），结果无意义
    dup = [c for c in P.columns if c != "c3" and P[c].equals(P["c3"])]
    if dup:
        print(f"\n🚨 缓冲未生效：{dup} 与 c3 逐位相同，检查 enter_rank 是否设置")
    print("\n相关矩阵（非重叠；重叠窗会系统性低估相关）：")
    print(P.corr().round(2).to_string())

    print("\n== A. 前后半分割（IS 按累计收益选档 → OOS 检验）==")
    half = len(P) // 2
    for nm, is_sl, oos_sl in (("前段选 → 后段验", slice(0, half), slice(half, None)),
                              ("后段选 → 前段验", slice(half, None), slice(0, half))):
        IS, OOS = P.iloc[is_sl], P.iloc[oos_sl]
        cum = (1 + IS).prod() - 1
        pick = cum.idxmax()
        got = (1 + OOS[pick]).prod() - 1
        ref = (1 + OOS["c3"]).prod() - 1
        print(f"\n{nm}")
        print("  IS 各档累计: " + "  ".join(f"{c}:{100 * v:+.1f}%" for c, v in cum.items()))
        print(f"  IS 选中 {pick} ⇒ OOS {100 * got:+.1f}% vs 三锋 {100 * ref:+.1f}%"
              f" ⇒ Δ {100 * (got - ref):+.1f}pp")

    print("\n== B. 逐年滚动选档（用此前所有期选档，检验当年）==")
    Py = P.copy()
    Py["year"] = Py.index.year
    for y in sorted(Py["year"].unique())[1:]:
        hist = Py[Py["year"] < y].drop(columns="year")
        cur = Py[Py["year"] == y].drop(columns="year")
        if len(cur) < 3:
            continue
        pick = ((1 + hist).prod() - 1).idxmax()
        got = (1 + cur[pick]).prod() - 1
        ref = (1 + cur["c3"]).prod() - 1
        print(f"  {y}: 历史选中 {pick:<4} 当年 {100 * got:>+7.1f}% | 三锋 {100 * ref:>+7.1f}%"
              f" | Δ {100 * (got - ref):>+6.1f}pp")

    print("\n== C. 两腿权重（w·x20 + (1-w)·三锋）==")
    print(f"{'w':>6}{'均值':>9}{'波动':>8}{'期SR':>7}{'累计':>9}")
    for w in (0, .2, .3, .4, .5, .6, .7, .8, 1.0):
        r = w * P["x20"] + (1 - w) * P["c3"]
        print(f"{w:>6.0%}{r.mean():>+9.2%}{r.std():>8.2%}{r.mean() / r.std():>7.2f}"
              f"{(1 + r).prod() - 1:>+9.1%}")

    print("\n== D. 滚动定权（每期用此前 12 期按 Sharpe 定 w）==")
    K, rec = 12, []
    for i in range(K, len(P)):
        hist = P.iloc[i - K:i]
        best_w, best_sr = 0.0, -9e9
        for w in np.arange(0, 1.01, 0.1):
            r = w * hist["x20"] + (1 - w) * hist["c3"]
            sr = r.mean() / r.std() if r.std() > 0 else -9e9
            if sr > best_sr:
                best_w, best_sr = float(w), sr
        row = P.iloc[i]
        rec.append({"date": P.index[i], "w": best_w,
                    "roll": best_w * row["x20"] + (1 - best_w) * row["c3"],
                    "fix50": 0.5 * row["x20"] + 0.5 * row["c3"],
                    "c3": row["c3"], "x20": row["x20"]})
    R = pd.DataFrame(rec).set_index("date")
    print("权重序列: " + " ".join(f"{w:.0f}" for w in R["w"]))
    print("权重 均值 %.2f 中位 %.2f 最小 %.1f 最大 %.1f"
          % (R["w"].mean(), R["w"].median(), R["w"].min(), R["w"].max()))
    for c, nm in (("roll", "滚动定权"), ("fix50", "固定50/50"),
                  ("c3", "纯三锋"), ("x20", "纯exit20")):
        s = R[c]
        print(f"{nm:<9}{len(s):>4}期 均值 {s.mean():>+7.2%} 波动 {s.std():>7.2%}"
              f" 期SR {s.mean() / s.std():>5.2f} 累计 {(1 + s).prod() - 1:>+8.1%}")

    if args.csv:
        pd.DataFrame(per).to_csv(args.csv, encoding="utf-8-sig")
        print(f"\n非重叠期收益已落盘：{args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
