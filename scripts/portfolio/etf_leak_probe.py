"""ETF 轮动 · 权重层 PIT（信息渗漏）验证与回归探针。

## 背景（已修复，本脚本作为回归守卫）

``execution_shift`` 模式下引擎在**执行日 E** 调 ``strategy.get_weights_at(E, …)``
（``backtest/engine.py``），而策略内部一律按 ``ret_panel.loc[:date]`` /
``vol_panel.loc[date]`` 取窗口 ⇒ 若不修正就会读到 **E 日当天**的收益与波动，
而 E 日开盘下单时它们尚未发生。三条实际路径：

  - ``_risk_parity``：60 日协方差（三锋 / 五衡）；
  - ``_dedup``：120 日相关（所有 ``dedup_corr`` 档位）；
  - ``_weights``：``vol_panel.loc[date]``（``inverse_vol`` / ``target_vol``，磐石走这条）。

2026-10-29 由莉酱发现，当日修复：``run_variant`` 在 ``execution_shift=True`` 时
把**策略侧**面板整体 ``shift(1)``（``etf_rotation_probe._pit_shift``），
引擎结算用的 ``returns`` 不动。

## 本脚本做三件事

1. **取证**：包住 ``_risk_parity``，打印每次调用传入的 ``date`` 与**实际用到的
   最后一根收益日期**。修复后二者应相差 1 个交易日；相等 ⇒ 修复失效。
2. **对照**：``--compare`` 时关掉修复（把 ``_pit_shift`` 还原成 identity）跑一遍，
   与修复后逐项对比 —— 这正是历史数字（26.11% / 24.61%）的复现通道。
3. **量化**：年化 / Sharpe / 80 个重叠 2 月窗 / 选标改变数 / 逐日权重 L1 / 日收益相关。

用法::

    python -m scripts.portfolio.etf_leak_probe                    # contest3，只验修复生效
    python -m scripts.portfolio.etf_leak_probe --compare --csv out.csv
    python -m scripts.portfolio.etf_leak_probe --profile contest3,contest5h --compare

⚠️ ``--compare`` 会 monkeypatch 回修复前的行为，**仅供复现历史数字**，不要用于出成绩。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.portfolio.etf_rotation_probe as RP
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

_ORIG_PIT_SHIFT = RP._pit_shift
_IDENTITY = lambda df: df  # noqa: E731  —— 关掉修复 = 还原 2026-10-29 前的行为
#: 取证记录：(调用日 date, 实际用到的最后一根收益行)
_TRACE: list[tuple[str, str]] = []
_TRACE_MAX = 6

_ORIG_RP = EtfRotationAdvanced._risk_parity
#: 未 shift 的原始收益面板，取证时作参照（main 里注入）
_RAW_RET: dict[str, pd.DataFrame] = {}


def _rp_trace(self, sel, date, fallback):
    """原样调用，附带判定「策略实际用到的最后一行收益是哪一天的」。

    🚨 判据必须看**值**、不能看索引：``_pit_shift`` 只移动值不改索引，修复后
    ``loc[:E]`` 最后一行的 index 仍然是 E，但装的是 E−1 的收益。故与原始面板的
    E 行、E−1 行分别求 L1 距离，取更近的那个。
    """
    if len(_TRACE) < _TRACE_MAX and self.ret_panel is not None:
        sub = self.ret_panel.loc[:date].tail(self.cov_window)
        raw = _RAW_RET.get("df")
        if len(sub) and raw is not None and date in raw.index:
            used = sub.iloc[-1].reindex(raw.columns)
            pos = raw.index.get_loc(date)
            d_e = float((used - raw.loc[date].reindex(raw.columns)).abs().sum())
            d_prev = (float((used - raw.loc[raw.index[pos - 1]].reindex(raw.columns)).abs().sum())
                      if pos >= 1 else float("nan"))
            _TRACE.append((pd.Timestamp(date).strftime("%Y-%m-%d"), d_e, d_prev))
    return _ORIG_RP(self, sel, date, fallback)


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


def run_one(bundle, panels, spec: dict, extra_close, pit: bool = True,
            trace: bool = False) -> dict:
    """跑一次 T+1 open 口径（与 etf_contest5_eval 同源）。

    ``pit=False`` 会把 ``_pit_shift`` 还原成 identity，等价于修复前的历史行为。
    ``trace=True`` 只对**第二次**（``execution_shift=True``）回测做取证 ——
    第一次是「T 收盘成交」的辅助跑，不 shift 是**正确**的，不能当证据。
    """
    RP._pit_shift = _ORIG_PIT_SHIFT if pit else _IDENTITY
    try:
        r0 = run_variant(bundle, panels, spec, BEGIN, None, extra_close=extra_close)
        spl = build_execution_split(panels["_open_adj"], bundle[1],
                                    set(pd.DatetimeIndex(r0["rebalance_dates"])))
        if trace:
            _TRACE.clear()
        return run_variant(bundle, panels, spec, BEGIN, None, extra_close=extra_close,
                           execution_split=spl, execution_shift=True)
    finally:
        RP._pit_shift = _ORIG_PIT_SHIFT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ETF 权重层 PIT 验证/回归探针")
    ap.add_argument("--profile", default="contest3", help="逗号分隔，见 PROFILES")
    ap.add_argument("--compare", action="store_true",
                    help="额外跑一遍关闭修复的版本（复现历史数字）")
    ap.add_argument("--csv", default=None, help="对照结果落盘")
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

    print("== ① 取证：策略实际用到的最后一行收益是哪一天 ==", flush=True)
    _RAW_RET["df"] = close.assign(CASH=1.0).pct_change(fill_method=None)
    run_one(bundle, panels, PROFILES[names[0]], extra, pit=True, trace=True)
    for d_call, d_e, d_prev in _TRACE:
        if d_e == 0.0 and d_prev == 0.0:
            # shift 后首行必为 NaN ⇒ 距离算成 0 的假象，不是证据
            flag = "⚠️ 该行整行为 NaN（窗口未满），不可判定"
        elif d_prev <= d_e:
            flag = "✅ 已 PIT（只到信号日）"
        else:
            flag = "🚨 仍含当日，修复失效"
        print(f"  调用日 {d_call} → 距 E {d_e:.3e} / 距 E−1 {d_prev:.3e}   {flag}",
              flush=True)

    rows = []
    for name in names:
        spec = PROFILES[name]
        fixed = run_one(bundle, panels, spec, extra, pit=True)
        if not args.compare:
            continue
        legacy = run_one(bundle, panels, spec, extra, pit=False)
        wf = two_month_windows(fixed["equity"], mfs)
        wl = two_month_windows(legacy["equity"], mfs)
        idx = wf.index.intersection(wl.index)

        # 选标 / 权重差异（只在调仓日比较）
        rb = sorted(set(pd.DatetimeIndex(fixed["rebalance_dates"]))
                    & set(pd.DatetimeIndex(legacy["rebalance_dates"])))
        sf, sl = _sel_sets(fixed["weights"], rb), _sel_sets(legacy["weights"], rb)
        sel_diff = sum(1 for d in rb if d in sf and d in sl and sf[d] != sl[d])
        # 逐日权重 L1 差（去掉 252 日暖机）
        wfx, wlg = fixed["weights"], legacy["weights"]
        common = wfx.index.intersection(wlg.index)
        common = common[common >= close.index[min(252, len(close.index) - 1)]]
        l1 = (wfx.loc[common] - wlg.loc[common]).abs().sum(axis=1)
        dr_f = fixed["equity"].pct_change().reindex(common)
        dr_l = legacy["equity"].pct_change().reindex(common)

        rows.append({
            "profile": name,
            "ann_fixed": fixed["annual_return"], "ann_legacy": legacy["annual_return"],
            "sh_fixed": fixed["sharpe"], "sh_legacy": legacy["sharpe"],
            "mdd_fixed": fixed["max_drawdown"], "mdd_legacy": legacy["max_drawdown"],
            "w2m_fixed": float(wf[idx].mean()), "w2m_legacy": float(wl[idx].mean()),
            "w2m_med_fixed": float(wf[idx].median()), "w2m_med_legacy": float(wl[idx].median()),
            "worst_fixed": float(wf[idx].min()), "worst_legacy": float(wl[idx].min()),
            "n_rb": len(rb), "sel_diff": sel_diff,
            "l1_med": float(l1.median()), "l1_p90": float(l1.quantile(0.90)),
            "r_corr": float(dr_f.corr(dr_l)),
        })
        print(f"[done] {name}", flush=True)

    if not rows:
        return 0

    d = pd.DataFrame(rows).set_index("profile")
    print("\n== ② 对照（T+1 open，80 个重叠 2 月窗）==")
    print(f"{'档位':<12}{'':>12}{'年化':>9}{'Sharpe':>9}{'2月均值':>9}{'2月中位':>9}{'最差':>9}")
    for k, r in d.iterrows():
        print(f"{k:<12}{'旧(含E日)':>12}{r.ann_legacy:>9.2%}{r.sh_legacy:>9.2f}"
              f"{r.w2m_legacy:>+9.2%}{r.w2m_med_legacy:>+9.2%}{r.worst_legacy:>+9.2%}")
        print(f"{'':<12}{'修复后(PIT)':>12}{r.ann_fixed:>9.2%}{r.sh_fixed:>9.2f}"
              f"{r.w2m_fixed:>+9.2%}{r.w2m_med_fixed:>+9.2%}{r.worst_fixed:>+9.2%}")
        print(f"{'':<12}{'Δ':>12}{r.ann_fixed - r.ann_legacy:>+9.2%}"
              f"{r.sh_fixed - r.sh_legacy:>+9.2f}"
              f"{r.w2m_fixed - r.w2m_legacy:>+9.2%}"
              f"{r.w2m_med_fixed - r.w2m_med_legacy:>+9.2%}"
              f"{r.worst_fixed - r.worst_legacy:>+9.2%}")

    print("\n== ③ 机制诊断 ==")
    for k, r in d.iterrows():
        print(f"  {k}: 调仓日 {int(r.n_rb)} 个，持仓集合差异 {int(r.sel_diff)} 个；"
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
