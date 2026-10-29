"""ETF 轮动 · 权重层 PIT（信息渗漏）探针。

## 待验证的指控

``execution_shift`` 模式下引擎在执行日 E 调 ``strategy.get_weights_at(E, …)``
（``backtest/engine.py``），而 ``run_variant`` 把 **未 shift** 的
``close.pct_change()`` 当作 ``ret_panel`` 交给策略 ⇒ ``_risk_parity`` 的
``ret_panel.loc[:E].tail(60)`` 与 ``_dedup`` 的 ``loc[:E].tail(120)`` 都**包含 E 日
当天的收益**。真实流程在 E 开盘下单时只有 ≤T（=E−1 收盘）的信息。

同一族还有两条同类路径：
  - ``_weights`` 的 ``self.vol_panel.loc[date]``（``inverse_vol`` / ``target_vol`` 用；
    磐石 ``champ`` 走这条）；
  - ``_dedup`` 的 120 日相关矩阵（所有 ``dedup_corr`` 档位）。

## 本脚本做三件事

1. **直接取证**：包住 ``_risk_parity``，打印每次调用传入的 ``date`` 与
   实际用到的最后一根收益日期。二者相等 ⇒ 渗漏成立（不需要推断）。
2. **无渗漏对照**：只把*策略内部*面板 ``shift(1)``（引擎结算panel 不动），
   即「E 日只看到 T 及之前」。等价于 ``loc[< date]``。
3. 量化差异：年化/Sharpe/80 个 2 月窗口 + 非重叠 2 月期 + 选标改变率 +
   权重 L1 差 + 日收益相关 —— 判断「渗漏是虚高还是噪声」。

用法::

    python -m scripts.portfolio.etf_leak_probe                    # contest3（定版）
    python -m scripts.portfolio.etf_leak_probe --profile contest5h,champ
    python -m scripts.portfolio.etf_leak_probe --csv out.csv

⚠️ 本脚本**不改生产代码**，只用 ``__init__`` 包装做局部替换。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import build_execution_split
from data.cache import DataCache
from data.datasource import create_datasource
from scripts.portfolio.etf_champ_signals import PROFILES
from scripts.portfolio.etf_contest_spec import EXCLUDED_5
from scripts.portfolio.etf_rotation_panels import (
    build_defense,
    build_extra_panels,
    level_from_returns,
)
from scripts.portfolio.etf_rotation_probe import (
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)
from strategy.etf_rotation import EtfRotationAdvanced

BEGIN = 20200101
ROOT = Path(__file__).resolve().parents[2]

#: 渡让开关：True = 策略内部面板额外 shift(1)（模拟真实流程）
_NOLEAK = {"ret": False, "vol": False}
#: 取证记录：(调用日 date, 实际用到的最后一根收益行)
_TRACE: list[tuple[str, str]] = []
_TRACE_MAX = 6

_ORIG_INIT = EtfRotationAdvanced.__init__
_ORIG_RP = EtfRotationAdvanced._risk_parity


def _rp_trace(self, sel, date, fallback):
    """原样调用，附带记录「传入的 date」vs「实际最后一根收益」。"""
    if len(_TRACE) < _TRACE_MAX and self.ret_panel is not None:
        sub = self.ret_panel.loc[:date].tail(self.cov_window)
        if len(sub):
            _TRACE.append((pd.Timestamp(date).strftime("%Y-%m-%d"),
                           pd.Timestamp(sub.index[-1]).strftime("%Y-%m-%d")))
    return _ORIG_RP(self, sel, date, fallback)


def _init(inst, *a, **k):
    _ORIG_INIT(inst, *a, **k)
    if _NOLEAK["ret"] and inst.ret_panel is not None:
        inst.ret_panel = inst.ret_panel.shift(1)
    if _NOLEAK["vol"] and getattr(inst, "vol_panel", None) is not None:
        inst.vol_panel = inst.vol_panel.shift(1)


EtfRotationAdvanced.__init__ = _init
EtfRotationAdvanced._risk_parity = _rp_trace


def two_month_windows(eq: pd.Series, mfs: pd.DatetimeIndex) -> pd.Series:
    out: dict[pd.Timestamp, float] = {}
    for i in range(len(mfs) - 2):
        t0, t2 = mfs[i], mfs[i + 2]
        if t0 not in eq.index or t2 not in eq.index:
            continue
        if len(eq.loc[t0:t2]) < 20:
            continue
        out[t0] = float(eq.loc[t2] / eq.loc[t0] - 1.0)
    return pd.Series(out)


def _sel_sets(w: pd.DataFrame, days) -> dict[pd.Timestamp, frozenset]:
    """调仓日的持仓集合（权重 > 1e-9 的标的，排除现金/合成腿）。"""
    out = {}
    for d in days:
        if d not in w.index:
            continue
        row = w.loc[d]
        out[d] = frozenset(c for c in row.index
                           if abs(float(row[c])) > 1e-9 and c not in ("CASH", "DEF"))
    return out


def run_one(bundle, panels, spec: dict, extra_close, noleak: bool) -> dict:
    """跑一次 T+1 open 口径（与 etf_contest5_eval 同源）。"""
    _NOLEAK["ret"] = noleak
    _NOLEAK["vol"] = noleak
    r0 = run_variant(bundle, panels, spec, BEGIN, None, extra_close=extra_close)
    open_adj = panels["_open_adj"]
    spl = build_execution_split(open_adj, bundle[1],
                                set(pd.DatetimeIndex(r0["rebalance_dates"])))
    r1 = run_variant(bundle, panels, spec, BEGIN, None, extra_close=extra_close,
                     execution_split=spl, execution_shift=True)
    _NOLEAK["ret"] = False
    _NOLEAK["vol"] = False
    return r1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ETF 权重层 PIT 渗漏探针")
    ap.add_argument("--profile", default="contest3", help="逗号分隔，见 PROFILES")
    ap.add_argument("--csv", default=None, help="窗口对照落盘")
    args = ap.parse_args(argv)
    names = [s.strip() for s in args.profile.split(",")]

    cache = DataCache(create_datasource())
    bundle = load_bundle(cache, WARMUP_BEGIN, None)
    uni, close, amount, bench = bundle
    raw = cache.read_daily("etf")
    rc = raw["close"].unstack("code").reindex(index=close.index, columns=close.columns)
    ro = raw["open"].unstack("code").reindex(index=close.index, columns=close.columns)
    open_adj = (ro * (close / rc)).astype(np.float32)

    panels = build_extra_panels(close, build_panels(close, amount), rc)
    for x in EXCLUDED_5:
        panels["mask"].loc[:, x] = False
    panels["_open_adj"] = open_adj          # 供 run_one 复用，避免重复构造
    extra = pd.DataFrame({"DEF": level_from_returns(build_defense(close)["mix"])})

    mfs = pd.DatetimeIndex(
        pd.Series(close.index, index=close.index)
        .groupby(close.index.to_period("M")).first().to_numpy())

    print("== ① 直接取证：_risk_parity 用到的最后一根收益 vs 调用日 ==", flush=True)
    run_one(bundle, panels, PROFILES[names[0]], extra, noleak=False)
    for d_call, d_last in _TRACE:
        flag = "🚨 含当日" if d_call == d_last else "ok"
        print(f"  调用日 {d_call} → 最后一根收益 {d_last}   {flag}", flush=True)

    rows = []
    for name in names:
        spec = PROFILES[name]
        b = run_one(bundle, panels, spec, extra, noleak=False)
        n = run_one(bundle, panels, spec, extra, noleak=True)
        wl = two_month_windows(b["equity"], mfs)
        ln = two_month_windows(n["equity"], mfs)
        idx = wl.index.intersection(ln.index)

        # 选标 / 权重差异（只在调仓日比较）
        rb = sorted(set(pd.DatetimeIndex(b["rebalance_dates"]))
                    & set(pd.DatetimeIndex(n["rebalance_dates"])))
        sb, sn = _sel_sets(b["weights"], rb), _sel_sets(n["weights"], rb)
        sel_diff = sum(1 for d in rb if d in sb and d in sn and sb[d] != sn[d])
        # 逐日权重 L1 差（去掉 252 日暖机）
        wb, wn = b["weights"], n["weights"]
        common = wb.index.intersection(wn.index)
        common = common[common >= close.index[min(252, len(close.index) - 1)]]
        l1 = (wb.loc[common] - wn.loc[common]).abs().sum(axis=1)
        dr_b = b["equity"].pct_change().reindex(common)
        dr_n = n["equity"].pct_change().reindex(common)

        rows.append({
            "profile": name,
            "ann_leak": b["annual_return"], "ann_noleak": n["annual_return"],
            "sh_leak": b["sharpe"], "sh_noleak": n["sharpe"],
            "mdd_leak": b["max_drawdown"], "mdd_noleak": n["max_drawdown"],
            "w2m_leak": float(wl[idx].mean()), "w2m_noleak": float(ln[idx].mean()),
            "w2m_med_leak": float(wl[idx].median()), "w2m_med_noleak": float(ln[idx].median()),
            "worst_leak": float(wl[idx].min()), "worst_noleak": float(ln[idx].min()),
            "n_rb": len(rb), "sel_diff": sel_diff,
            "l1_med": float(l1.median()), "l1_p90": float(l1.quantile(0.90)),
            "r_corr": float(dr_b.corr(dr_n)),
        })
        print(f"[done] {name}", flush=True)

    d = pd.DataFrame(rows).set_index("profile")
    print("\n== ② 对照（T+1 open，80 个重叠 2 月窗 / 非重叠期另算）==")
    print(f"{'档位':<12}{'':>10}{'年化':>9}{'Sharpe':>9}{'2月均值':>9}{'2月中位':>9}{'最差':>9}")
    for k, r in d.iterrows():
        print(f"{k:<12}{'现行(含E日)':>10}{r.ann_leak:>9.2%}{r.sh_leak:>9.2f}"
              f"{r.w2m_leak:>+9.2%}{r.w2m_med_leak:>+9.2%}{r.worst_leak:>+9.2%}")
        print(f"{'':<12}{'无渗漏(真实)':>10}{r.ann_noleak:>9.2%}{r.sh_noleak:>9.2f}"
              f"{r.w2m_noleak:>+9.2%}{r.w2m_med_noleak:>+9.2%}{r.worst_noleak:>+9.2%}")
        print(f"{'':<12}{'Δ':>10}{r.ann_noleak - r.ann_leak:>+9.2%}"
              f"{r.sh_noleak - r.sh_leak:>+9.2f}"
              f"{r.w2m_noleak - r.w2m_leak:>+9.2%}"
              f"{r.w2m_med_noleak - r.w2m_med_leak:>+9.2%}"
              f"{r.worst_noleak - r.worst_leak:>+9.2%}")

    print("\n== ③ 机制诊断 ==")
    for k, r in d.iterrows():
        print(f"  {k}: 调仓日 {int(r.n_rb)} 个，选标改变 {int(r.sel_diff)} 个；"
              f"逐日权重 L1 中位 {r.l1_med:.2%} / p90 {r.l1_p90:.2%}；"
              f"日收益相关 {r.r_corr:.4f}")

    if args.csv:
        p = Path(args.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(p, encoding="utf-8-sig")
        print(f"\n落盘 → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
