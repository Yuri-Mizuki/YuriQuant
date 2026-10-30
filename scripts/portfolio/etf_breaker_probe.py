"""ETF 轮动 · 熔断降仓实验：「回撤触 −15% 就减仓」到底划不划算
================================================================

背景：赛内纪律原本是「组合回撤触 −15% ⇒ 临时降档到五衡（Top5/HRP）」。2026-10-29
权重层 PIT 修复后五衡 20.65% vs 三锋 30.63%（**落后 9.98pp/年**），且五衡的 MDD
（33.27%）比三锋（24.97%）还深 ⇒ 降档的钱白花了（README §八 P0）。

本脚本测另一条路：**不减选标、只减敞口**——回撤触 −15% 后把组合整体降到 k（余量
落 CASH），回撤修复后回到满仓。要回答三个问题：

  1. 减仓能削掉多少尾部（p10 / 最差窗口 / MDD），代价是多少年化？
  2. 与「降档五衡」（−9.98pp/年、尾部反而更差）相比谁更划算？
  3. 这点削尾是**真择时**还是「少拿仓位少波动」？⇒ 随机时点安慰剂 + 平均敞口匹配。

方法要点（与 README §零 口径严格一致）：

  - 单次连续回测 20200101 → 末日，**事后**按月首切 2 月窗口；非重叠 2 月期为主判据；
  - T+1 open 三件套（策略面板 shift(1) + 调仓日 E=T+1 + split 按信号日构造）；
  - 回撤用**账户净值**（含 CASH）的 `eq / cummax(eq) − 1`，在信号日 T 判定、
    E=T+1 开盘执行 ⇒ 无前视；
  - 减仓不换选标（`BreakerOverlay`：沿用持仓按收盘漂移 × k），恢复满仓时按当日
    信号重选标（对应"退出熔断回三锋"）；
  - 🚨 **自洽迭代**：减仓后回撤路径变了 ⇒ 触发日不能直接用基线路径定。用不动点迭代
    （跑 → 按自身净值重算状态表 → 再跑），收敛判据 = 状态表逐日相同。

已知近似（引用时须带）：

  - CASH 按 **0 收益**处理（无货币基金利息）⇒ 对减仓略微保守；
  - 漂移用收盘价近似，换手成本略被高估；
  - 触发次数是个位数 ⇒ 统计功效极低，结论只能是「机制上削尾多少 + 付多少」，不能
    断言"熔断有效/无效"。

用法::

    python -m scripts.portfolio.etf_breaker_probe                    # 主表：k×恢复规则
    python -m scripts.portfolio.etf_breaker_probe --placebo 60       # 加随机安慰剂
    python -m scripts.portfolio.etf_breaker_probe --k 0.5 --rules band5

产物：控制台表格（可直接重定向成 `reports/etf_contest5/breaker_*.txt`）+ `--csv` 明细。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import build_execution_split
from data.cache import DataCache
from data.datasource import create_datasource
from scripts.portfolio.etf_champ_signals import PROFILES
from scripts.portfolio.etf_contest5_eval import two_month_windows
from scripts.portfolio.etf_contest_spec import EXCLUDED_5
from scripts.portfolio.etf_rotation_panels import build_extra_panels
from scripts.portfolio.etf_rotation_probe import (
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)

BEGIN = 20200101
ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(20261029)
BASE = dict(PROFILES["contest3"])          # 定版三锋
THR = 0.15                                  # 赛制熔断线 −15%
#: 恢复规则：``band:x`` = 账户净值回撤收窄到 −x 即回满仓；``newhigh`` = 创新高才回；
#: ``time:n`` = 降仓后满 n 个交易日无条件回。⚠️ 注意「减仓后回撤自动减半」的陷阱：
#: 触发时 −15% 会立即变成 ≈−7.5%，所以 `band:0.075` 会在次日就解除（抖动），
#: 必须用更紧的 band 才是有意义的滞留区间。
RULES = {
    "band5": ("band", 0.05),
    "band3": ("band", 0.03),
    "newhigh": ("newhigh", 0.0),
    "time21": ("time", 21),
    "time63": ("time", 63),
}


@dataclass
class Env:
    bundle: object
    panels: dict
    close: pd.DataFrame
    open_adj: pd.DataFrame
    mfs: pd.DatetimeIndex
    spl: tuple = None
    base_eq: pd.Series = field(default=None)


def build_env() -> Env:
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
    mfs = pd.DatetimeIndex(
        pd.Series(close.index, index=close.index)
        .groupby(close.index.to_period("M")).first().to_numpy())
    env = Env(bundle=bundle, panels=panels, close=close, open_adj=open_adj, mfs=mfs)
    # 基线（无熔断）跑一次：拿调仓日构造 split，也拿它的净值做事件对照
    r0 = run_variant(bundle, panels, BASE, BEGIN, None)
    rb = set(pd.DatetimeIndex(r0["rebalance_dates"]))
    env.spl = build_execution_split(open_adj, close, rb)
    r1 = run_variant(bundle, panels, BASE, BEGIN, None,
                     execution_split=env.spl, execution_shift=True)
    env.base_eq = r1["equity"]
    return env


def run_open(env: Env, states: dict | None) -> dict:
    return run_variant(env.bundle, env.panels, BASE, BEGIN, None,
                       execution_split=env.spl, execution_shift=True,
                       breaker=({"states": states} if states else None))


def derive_states(eq: pd.Series, thr: float, k: float, rule: tuple,
                  min_hold: int = 0) -> dict:
    """按账户净值回撤生成状态表（key = 信号日 T，值 = 敞口）。

    ``min_hold`` = 降仓最短持续交易日。🚨 必须 >0，否则「减仓后回撤自动减半」
    会让 band 类规则在**次日**就解除（−15% ⇒ −7.5%），产生抖动并使不动点不收敛。
    """
    dd = (eq / eq.cummax() - 1.0).to_numpy()
    days = list(eq.index)
    out: dict[pd.Timestamp, float] = {}
    state, since = 1.0, 0
    for i, d in enumerate(days):
        if state >= 1.0 - 1e-12:
            if dd[i] <= -thr:
                state, since = k, i
        elif (i - since) < min_hold:
            pass
        else:
            kind, arg = rule
            hit = {
                "band": lambda: dd[i] >= -float(arg),
                "newhigh": lambda: dd[i] >= 0.0,
                "time": lambda: (i - since) >= int(arg),
            }[kind]()
            if hit:
                state = 1.0
        out[d] = state
    return out


HOLD = {"band": 10, "newhigh": 0, "time": 0}


def solve(env: Env, thr: float, k: float, rule: tuple, max_iter: int = 10) -> dict:
    """不动点迭代：状态表 ↔ 自身净值路径自洽。"""
    states: dict = {}
    hold = HOLD.get(rule[0], 0)
    for it in range(max_iter):
        r = run_open(env, states or None)
        new = derive_states(r["equity"], thr, k, rule, min_hold=hold)
        same = bool(states) and all(
            abs(new[d] - states.get(d, 1.0)) < 1e-12 for d in new)
        states = new
        if same:
            r["iters"], r["converged"] = it, True
            r["states"] = states
            return r
    r["iters"], r["converged"] = max_iter, False
    # 未收敛（振荡）：用最后一轮的状态表重跑，保证「净值 ↔ 状态表」至少对应一致，
    # 便于读逐次干预账；但这条路径本身不自洽，数字只能当区间端点看。
    r = run_open(env, states)
    r["iters"], r["converged"] = max_iter, False
    r["states"] = states
    return r


def runs_of(states: dict, k: float) -> list[tuple[pd.Timestamp, pd.Timestamp | None]]:
    """降仓区段 [(起, 止)]（止 = 恢复日，None = 到期末）。"""
    s = pd.Series(states).sort_index()
    off = s < 1.0 - 1e-12
    out, start = [], None
    for d, v in off.items():
        if v and start is None:
            start = d
        elif (not v) and start is not None:
            out.append((start, d))
            start = None
    if start is not None:
        out.append((start, None))
    return out


def event_table(env: Env, r: dict, k: float, thr: float) -> pd.DataFrame:
    """逐次干预的因果账：同一区间内「熔断」vs「基线」的累计收益差。"""
    rows = []
    eq_b, eq_x = env.base_eq, r["equity"]
    for t0, t1 in runs_of(r["states"], k):
        end = t1 if t1 is not None else eq_x.index[-1]
        if t0 not in eq_x.index or end not in eq_x.index:
            continue
        rb_ = float(eq_b.loc[end] / eq_b.loc[t0] - 1.0)
        rx_ = float(eq_x.loc[end] / eq_x.loc[t0] - 1.0)
        dd_b = float((eq_b.loc[t0:end] / eq_b.loc[t0:end].cummax() - 1.0).min())
        rows.append({"trigger": t0, "recover": t1,
                     "days": int((eq_x.loc[t0:end].index.size) - 1),
                     "base_ret": rb_, "breaker_ret": rx_,
                     "delta": rx_ - rb_, "base_dd_in": dd_b})
    return pd.DataFrame(rows)


def placebo(env: Env, real: list[tuple[pd.Timestamp, pd.Timestamp | None]],
            k: float, n: int) -> pd.Series:
    """块状随机安慰剂：保持「降仓次数 + 每次持续天数」，只把时点打乱。"""
    idx = env.base_eq.index
    pos = {d: i for i, d in enumerate(idx)}
    blocks = []
    for t0, t1 in real:
        i0 = pos[t0]
        i1 = pos[t1] if t1 is not None else len(idx) - 1
        blocks.append(max(i1 - i0, 1))
    out = []
    n_days = len(idx)
    for _ in range(n):
        states = {d: 1.0 for d in idx}
        taken = np.zeros(n_days, dtype=bool)
        ok = True
        for L in blocks:
            for _try in range(200):
                s = int(RNG.integers(0, n_days - L - 1))
                if not taken[s: s + L].any():
                    break
            else:
                ok = False
                break
            taken[s: s + L] = True
            for j in range(s, s + L):
                states[idx[j]] = k
        if not ok:
            continue
        r = run_open(env, states)
        out.append(r["annual_return"])
    return pd.Series(out)


def run_static(env: Env, k: float) -> dict:
    """恒定敞口对照：每期按 k 缩放、余量落 CASH（无任何择时）。

    🚨 这是熔断的真正对手盘：若"平均敞口 k̄ 的恒定仓位"能达到同样的 MDD 改善，
    熔断的择时价值就是 0，只是把「少拿仓位」包装成纪律。
    """
    fac = pd.Series(float(k), index=env.close.index)
    return run_variant(env.bundle, env.panels, BASE, BEGIN, None,
                       execution_split=env.spl, execution_shift=True, exposure=fac)


def _selfcheck(r: dict, k: float) -> tuple[float, float, int]:
    """静默失效自检：降仓日的多头腿合计应 ≈ k（不是 k 就说明覆盖层没生效）。

    状态表的键是**信号日 T**，权重在执行日 E=T+1 生效 ⇒ 检查时后移一天。
    """
    w = r["weights"]
    idx = list(w.index)
    pos = {d: i for i, d in enumerate(idx)}
    st = pd.Series(r.get("states", {}))
    off = [idx[pos[d] + 1] for d, v in st.items()
           if v < 1.0 - 1e-12 and d in pos and pos[d] + 1 < len(idx)]
    if not off:
        return float("nan"), float("nan"), 0
    held = w.drop(columns=[c for c in ("CASH", "DEF") if c in w.columns])
    expo = held.loc[off].abs().sum(axis=1)
    return float(expo.median()), float((expo - k).abs().max()), len(off)


def dd_conditional(eq: pd.Series, thr: float,
                   fwds=(21, 63)) -> pd.DataFrame:
    """机制诊断：处于不同回撤区间时，后续 21/63 日的收益分布。

    若「回撤 ≤ −thr」组的前瞻收益**不显著差于全样本**，说明"跌够了"没有可
    利用的时序信息 ⇒ 熔断只能靠"少拿仓位"降低波动，不可能改善风险调整后收益。
    """
    dd = eq / eq.cummax() - 1.0
    rows = []
    for tag, m in (("全样本", pd.Series(True, index=eq.index)),
                   ("回撤 > −8%", dd > -0.08),
                   ("−15%~−8%", (dd <= -0.08) & (dd > -thr)),
                   (f"回撤 ≤ −{thr:.0%}", dd <= -thr)):
        row = {"state": tag, "days": int(m.sum())}
        for h in fwds:
            f = (eq.shift(-h) / eq - 1.0)[m].dropna()
            row[f"n{h}_med"] = float(f.median()) if len(f) else float("nan")
            row[f"n{h}_mean"] = float(f.mean()) if len(f) else float("nan")
            row[f"n{h}_win"] = float((f > 0).mean()) if len(f) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("state")


def _row(env: Env, r: dict, tag: str, k: float, rule: str) -> dict:
    w = two_month_windows(r["equity"], env.mfs)
    per = r["equity"].resample("2ME").last().pct_change().dropna()
    st = pd.Series(r.get("states", {})).sort_index()
    return {"tag": tag, "k": k, "rule": rule,
            "ann": r["annual_return"], "sharpe": r["sharpe"],
            "mdd": r["max_drawdown"], "turnover": r["avg_turnover"],
            "w_mean": float(w.mean()), "w_p10": float(w.quantile(0.10)),
            "w_worst": float(w.min()), "win": float((w > 0).mean()),
            "p_mean": float(per.mean()), "p_std": float(per.std()),
            "p_sr": float(per.mean() / per.std()), "p_p10": float(per.quantile(0.10)),
            "p_worst": float(per.min()), "p_win": float((per > 0).mean()),
            "n_off": len(runs_of(r.get("states", {}), k)) if len(st) else 0,
            "off_share": float((st < 1.0 - 1e-12).mean()) if len(st) else 0.0,
            "iters": r.get("iters", 0), "conv": r.get("converged", True)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ETF 熔断降仓实验（T+1 open，80 窗/41 期）")
    ap.add_argument("--k", default="0.5", help="减仓后的敞口（逗号分隔，如 0.3,0.5,0.7）")
    ap.add_argument("--thr", type=float, default=THR, help="触发线（回撤，正数，默认 0.15）")
    ap.add_argument("--rules", default="band5,band3,newhigh,time21",
                    help=f"恢复规则（逗号分隔）：{','.join(RULES)}")
    ap.add_argument("--placebo", type=int, default=0, help="随机安慰剂次数（0=不跑）")
    ap.add_argument("--placebo-tags", default="",
                    help="只对这些档位跑安慰剂（逗号分隔；空=全部）")
    ap.add_argument("--static", action="store_true",
                    help="加跑「恒定敞口」对照（每期按 k 缩放、无择时）")
    ap.add_argument("--static-k", default="",
                    help="恒定敞口对照的敞口列表（逗号分隔；空=沿用 --k）")
    ap.add_argument("--csv", default=None, help="明细落盘路径")
    args = ap.parse_args(argv)

    ks = [float(x) for x in args.k.split(",")]
    rules = [x.strip() for x in args.rules.split(",")]
    for x in rules:
        if x not in RULES:
            raise SystemExit(f"未知规则 {x}，可选：{','.join(RULES)}")

    print(f"载入数据（BEGIN={BEGIN}）…", flush=True)
    env = build_env()
    print(f"基线 三锋 open：年化 {env.base_eq.pct_change().agg('mean') * 252:.2%}"
          f"（末日 {env.close.index[-1].date()}）", flush=True)

    rows, evs, store = [], {}, {}
    rb = run_open(env, None)
    b = _row(env, rb, "baseline", 1.0, "-")
    b["conv"] = True
    rows.append(b)
    print(f"[base] 年化 {b['ann']:.2%} SR {b['sharpe']:.2f} MDD {b['mdd']:.2%} "
          f"2月均值 {b['w_mean']:+.2%} p10 {b['w_p10']:+.2%} 最差 {b['w_worst']:+.2%}",
          flush=True)

    print(f"\n== 机制诊断：回撤状态下基线的前瞻收益（thr={args.thr:.0%}）==")
    print(dd_conditional(env.base_eq, args.thr).round(4).to_string())

    for k in ks:
        for rule in rules:
            r = solve(env, args.thr, k, RULES[rule])
            tag = f"k{k:g}_{rule}"
            row = _row(env, r, tag, k, rule)
            rows.append(row)
            store[tag] = r
            evs[tag] = event_table(env, r, k, args.thr)
            med, mx, n_off = _selfcheck(r, k)
            row["expo_med"], row["expo_dev"] = med, mx
            print(f"[{tag}] 年化 {row['ann']:.2%} SR {row['sharpe']:.2f} "
                  f"MDD {row['mdd']:.2%} 2月p10 {row['w_p10']:+.2%} "
                  f"最差 {row['w_worst']:+.2%} 触发 {row['n_off']} 次 "
                  f"降仓占比 {row['off_share']:.1%} 迭代 {row['iters']} "
                  f"{'收敛' if row['conv'] else '🚨未收敛'}"
                  f" | 自检 敞口中位 {med:.2f} 偏差 {mx:.3f} (n={n_off})", flush=True)

    d = pd.DataFrame(rows).set_index("tag")
    print("\n== 主表（T+1 open；80 重叠窗）==")
    print(f"{'档位':<18}{'年化':>8}{'SR':>6}{'MDD':>8}{'2月均值':>9}{'p10':>8}"
          f"{'最差':>8}{'胜率':>7}{'换手':>7}")
    for t, r in d.iterrows():
        print(f"{t:<18}{r.ann:>8.2%}{r.sharpe:>6.2f}{r.mdd:>8.2%}"
              f"{r.w_mean:>+9.2%}{r.w_p10:>+8.2%}{r.w_worst:>+8.2%}"
              f"{r.win:>7.1%}{r.turnover:>7.1%}")

    print("\n== 非重叠 2 月期（赛制真实计分单位）==")
    print(f"{'档位':<18}{'均值':>9}{'波动':>8}{'期SR':>7}{'p10':>9}{'最差':>9}{'胜率':>7}")
    for t, r in d.iterrows():
        print(f"{t:<18}{r.p_mean:>+9.2%}{r.p_std:>8.2%}{r.p_sr:>7.2f}"
              f"{r.p_p10:>+9.2%}{r.p_worst:>+9.2%}{r.p_win:>7.1%}")

    print("\n== 相对基线的增量（Δ 年化 / ΔMDD / Δp10_非重叠）==")
    b0 = d.loc["baseline"]
    print(f"{'档位':<18}{'Δ年化':>9}{'ΔMDD':>9}{'Δp10':>9}{'Δ最差':>9}{'降仓占比':>9}")
    for t, r in d.iterrows():
        if t == "baseline":
            continue
        print(f"{t:<18}{r.ann - b0.ann:>+9.2%}{r.mdd - b0.mdd:>+9.2%}"
              f"{r.p_p10 - b0.p_p10:>+9.2%}{r.p_worst - b0.p_worst:>+9.2%}"
              f"{r.off_share:>9.1%}")

    if args.static:
        print("\n== 恒定敞口对照（无择时，每期按 k 缩放；判「熔断有没有择时价值」）==")
        print(f"{'敞口':<8}{'年化':>9}{'SR':>7}{'MDD':>9}{'期均值':>9}{'期SR':>7}"
              f"{'期p10':>9}{'期最差':>9}")
        sks = ([float(x) for x in args.static_k.split(",")] if args.static_k
               else sorted(set(ks)))
        for k in sks:
            rs = run_static(env, k)
            x = _row(env, rs, f"static{k:g}", k, "-")
            print(f"{k:<8.2f}{x['ann']:>9.2%}{x['sharpe']:>7.2f}{x['mdd']:>9.2%}"
                  f"{x['p_mean']:>+9.2%}{x['p_sr']:>7.2f}{x['p_p10']:>+9.2%}"
                  f"{x['p_worst']:>+9.2%}", flush=True)

    if evs:
        print("\n== 逐次干预账（同区间：熔断 − 基线）==")
        for t, e in evs.items():
            if e.empty:
                print(f"{t}: 未触发")
                continue
            print(f"\n[{t}] 触发 {len(e)} 次，合计差额 {e['delta'].sum():+.2%}")
            print(f"  {'触发日':<12}{'恢复日':<12}{'天数':>5}{'基线':>9}{'熔断':>9}"
                  f"{'差额':>9}{'基线内回撤':>11}")
            for _, x in e.iterrows():
                rec = x["recover"].date() if x["recover"] is not None else "期末"
                print(f"  {str(x['trigger'].date()):<12}{str(rec):<12}{x['days']:>5}"
                      f"{x['base_ret']:>+9.2%}{x['breaker_ret']:>+9.2%}"
                      f"{x['delta']:>+9.2%}{x['base_dd_in']:>11.2%}")

    if args.placebo and evs:
        print(f"\n== 随机安慰剂（块状，保持次数与持续天数，n={args.placebo}）==")
        want = {x.strip() for x in args.placebo_tags.split(",") if x.strip()}
        for t, r in store.items():
            if want and t not in want:
                continue
            real = runs_of(r["states"], float(d.loc[t, "k"]))
            if not real:
                continue
            sim = placebo(env, real, float(d.loc[t, "k"]), args.placebo)
            real_ann = float(d.loc[t, "ann"])
            pct = float((sim < real_ann).mean())
            print(f"{t:<18}真实 Δ年化 {real_ann - b0.ann:+.2%}  "
                  f"随机分位 {(sim < real_ann).mean():.0%}  "
                  f"随机 5/50/95% = {sim.quantile(0.05) - b0.ann:+.2%} / "
                  f"{sim.median() - b0.ann:+.2%} / {sim.quantile(0.95) - b0.ann:+.2%}"
                  f"  (n={len(sim)})"
                  + ("  ⇒ 真实方案在随机分布内" if 0.05 < pct < 0.95 else "  ⇒ 超出随机分布"))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(out, encoding="utf-8-sig")
        parts = [e.assign(tag=t) for t, e in evs.items() if not e.empty]
        all_ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not all_ev.empty:
            all_ev.to_csv(out.with_name(out.stem + "_events" + out.suffix),
                          index=False, encoding="utf-8-sig")
        print(f"\n落盘 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
