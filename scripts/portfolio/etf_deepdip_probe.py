"""ETF 轮动 · 「深跌建仓」收益腿的崩盘区制存活检验（2026-10-30）
=================================================================

背景（`TODO.md` §一 ② 的 2026-10-29 追问升级 + 10-30 预注册判据）
    10-27 探针组曾得出「深跌建仓不是飞刀，是收益腿」：

      · `mom20 ∈ [−20%, −10%)` 的 15 笔建仓，fwd42 均值 **+5.36%**，其余 +0.30%；
      · 「`mom12_1` 前 1/3 × 近 60 日回撤 ≤ −20%」格 fwd42 中位 **+8.69%**
        （对照「无回撤」格 −4.33%）。

    🚨 但这两条的全部样本都在 2020 年之后——从没见过 2015 式主题崩盘、2018 熊市。
    于是 10-27 的处置（「当前 2 只该形态持仓维持不动」）建立在 n≈15 的正期望上，
    而 n≈15 且全是单一行情，正是幸存者偏差的典型形态。

判据（v2，2026-10-30 同日修订 —— **修订理由公开**）
    🚨 v1 的判据写的是「该格 fwd42 **均值 < 0**」。跑完发现它**设计不当**：
    24 只池在 2014–2019 的 fwd42 几乎每格都是正的（+0.8%~+1.8%，池子自身漂移），
    绝对正负号测的是 beta，没有区分力；而 10-27 的原始主张本身就是**相对**的
    （+8.69% vs 无回撤格 −4.33%）。故判据改为**差分**：

      主判据 D1：长样本「前期强 × dd ≤ −20%」的 fwd42 均值**减去**同期同组
                「前期强 × dd > −10%」的均值。
                 · 差 ≥ 0 且 95% CI 下界 > 0 → 优势**强存活**
                 · 差 ≥ 0 但 CI 跨 0        → **弱存活**（不能宣称更好，也不说更差）
                 · 差 < 0                   → **推翻超额表述**
      副判据 D2：2015（主题崩盘）/ 2018（熊市）单年，差分是否转负。

    🚨 改判据是**事后**行为，故 v1（绝对号）结果也一并留在报告里，
    让读者自己看两种读法的差别 —— 不许只报对自己有利的那个。

口径（与生产逐项同源）
    · 价格：ETF 自身后复权 close/open（`etf_long_sample.load_env`，已与生产
      `load_close` 逐位核对）；成交：T+1 open（`FINAL` 口径）；
    · 事件 = **建仓**（某只从上一调仓日不在仓 → 本调仓日在仓）；
      🚨 调仓日一律取自引擎的 `rebalance_dates`，**不能用 `weights` 的非零行**
      （那是逐日漂移快照，会把 fwd42 变成相邻日重叠 41/42 的重复观测）；
    · `fwd42` 主口径 = `open[T+1] → close[T+42]`；
      附 `fwd42_close` = `close[T] → close[T+42]`（与 10-27 旧探针口径可比，仅对照）；
    · 分组：`mom12_1` 在**当日可交易池内**的截面分位（前 1/3 = 前期强），
      回撤 = `close / close.rolling(60).max() − 1`。

两个样本
    · ``entry``：真实建仓事件（与 10-27 结论同构，但样本受池限制）；
    · ``pool``：每个调仓日池内**所有**可交易标的（不筛排名）⇒ 把 n 拉起来、
      与选标逻辑无关，用来判「该格在截面上是否有超额」。**主判据读 pool。**
    🚨 24 只池在 2020+ 的 entry 深跌格只有个位数（原结论是 15 笔）⇒ 原样本是
    45 只生产池的产物，**两个样本不可直接比**。故另给 ``--window near45``
    做样本量对齐性核对，只核对 n 与方向，不作判据。

用法::

    python -m scripts.portfolio.etf_deepdip_probe                  # long + near
    python -m scripts.portfolio.etf_deepdip_probe --window near45  # 45 只池对齐核对
    python -m scripts.portfolio.etf_deepdip_probe --min-amount 0   # 不加流动性门槛

产物（落 `reports/etf_contest5/`）：
    deepdip_cells_2026-10-30.txt / deepdip_cells.csv / deepdip_events.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.portfolio.etf_contest_spec import FINAL  # noqa: E402
from scripts.portfolio.etf_long_sample import (  # noqa: E402
    LONG_CODES,
    LONG_END,
    LONG_START,
    NEAR_END,
    NEAR_START,
    WARMUP,
    _run,
    load_env,
)

OUT = Path("reports/etf_contest5")
TXT = OUT / "deepdip_cells_2026-10-30.txt"
FWD = 42                 # 前瞻窗口（交易日）
DD_WIN = 60              # 回撤回看的窗口
STRONG_Q = 2.0 / 3.0     # 「前期强」= mom12_1 截面分位 ≥ 2/3
DD_BUCKETS = [(-9.99, -0.20, "dd ≤ −20%"), (-0.20, -0.10, "−20% < dd ≤ −10%"),
              (-0.10, 9.99, "dd > −10%")]
MOM20_BUCKETS = [(-9.99, -0.20, "mom20 ≤ −20%"), (-0.20, -0.10, "mom20 ∈ [−20,−10)%"),
                 (-0.10, 0.0, "mom20 ∈ [−10,0)"), (0.0, 9.99, "mom20 ≥ 0")]
#: 主判据关键格与对照格：(列, 下界, 上界)
KEY_CELL = ("dd60", -9.99, -0.20)
CTRL_CELL = ("dd60", -0.10, 9.99)


# ------------------------------------------------------------------ 事件构造
def _holdings(r1: dict) -> pd.DataFrame:
    """只留多头腿的权重快照（与 `etf_weight_probe.hold_matrix` 同实现）。"""
    w = r1["weights"]
    held = [c for c in w.columns if c not in ("CASH", "DEF")]
    live = w[w.abs().sum(axis=1) > 1e-9]
    return live[held]


def _entry_events(hold: pd.DataFrame, rb_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """建仓事件 = 本调仓日在仓、上一调仓日不在仓。

    🚨 ``rb_dates`` 必须是引擎调仓日；用 ``hold.index``（逐日漂移）会把「上一
    调仓日」退化成「上一交易日」，pool 样本随之虚增约 5 倍且相邻重叠 41/42
    （2026-10-30 实测：13.7k → 2.5k）。
    """
    rows, prev = [], None
    for d in rb_dates:
        if d not in hold.index:
            continue
        cur = set(hold.loc[d][hold.loc[d] > 1e-9].index)
        before = (set(hold.loc[prev][hold.loc[prev] > 1e-9].index)
                  if prev is not None and prev in hold.index else set())
        for c in sorted(cur - before):
            rows.append({"date": d, "code": c, "kind": "entry"})
        prev = d
    return pd.DataFrame(rows)


def _pool_events(mask: pd.DataFrame, rb_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """全可交易池版「建仓格」= 每个调仓日池内所有可交易标的（不筛排名）。"""
    rows = []
    for d in rb_dates:
        if d not in mask.index:
            continue
        ok = mask.loc[d]
        for c in ok.index[ok.fillna(False).astype(bool)]:
            rows.append({"date": d, "code": c, "kind": "pool"})
    return pd.DataFrame(rows)


def _fwd(close: pd.DataFrame, open_: pd.DataFrame, idx: dict,
         d: pd.Timestamp, c: str) -> tuple[float, float]:
    """(T+1 open → T+FWD close, T close → T+FWD close)。"""
    i = idx[d]
    if i + FWD >= len(close) or c not in close.columns:
        return np.nan, np.nan
    d1, d2 = close.index[i + 1], close.index[i + FWD]
    o, c0, c2 = float(open_.at[d1, c]), float(close.at[d, c]), float(close.at[d2, c])
    if not np.isfinite(o) or o <= 0 or not np.isfinite(c2):
        return np.nan, np.nan
    return (c2 / o - 1.0,
            (c2 / c0 - 1.0) if np.isfinite(c0) and c0 > 0 else np.nan)


def build_events(env: dict, r1: dict, start: int) -> tuple[pd.DataFrame, int]:
    """用引擎调仓日生成 entry / pool 两套事件，并附特征 + fwd42。"""
    close, open_ = env["close"], env["open_adj"]
    p = env["panels"]
    mom, mask = p["mom12_1"], p["mask"]
    dd = close / close.rolling(DD_WIN, min_periods=40).max() - 1.0
    mom20 = close / close.shift(20) - 1.0

    rb = pd.DatetimeIndex(r1["rebalance_dates"])
    rb = rb[rb >= pd.Timestamp(str(start))]
    hold = _holdings(r1)
    ev = pd.concat([_entry_events(hold, rb), _pool_events(mask, rb)],
                   ignore_index=True)
    idx = {d: i for i, d in enumerate(close.index)}
    rank = mom.where(mask).rank(axis=1, pct=True)   # 截面分位（可交易池内）

    recs = []
    for d, c, kind in ev.itertuples(index=False):
        if c not in close.columns:
            continue
        fw_o, fw_c = _fwd(close, open_, idx, d, c)
        recs.append({
            "date": d, "code": c, "kind": kind,
            "mom12_1": float(mom.at[d, c]) if c in mom.columns else np.nan,
            "mom_pct": float(rank.at[d, c]) if c in rank.columns else np.nan,
            "mom20": float(mom20.at[d, c]),
            "dd60": float(dd.at[d, c]),
            "fwd42_open": fw_o, "fwd42_close": fw_c,
        })
    ev = pd.DataFrame(recs)
    ev["year"] = ev["date"].dt.year
    ev["strong"] = ev["mom_pct"] >= STRONG_Q
    return ev, len(rb)


# ------------------------------------------------------------------ 统计工具
def _stat(s: pd.Series) -> str:
    s = s.dropna()
    if not len(s):
        return f"{'—':>9}{'—':>9}{'—':>8}{0:>8d}"
    return (f"{s.mean():>+8.2%}{s.median():>+8.2%}"
            f"{(s > 0).mean():>8.1%}{len(s):>8d}")


def _boot_diff(cell: pd.DataFrame, ctrl: pd.DataFrame, n_boot: int = 2000,
               seed: int = 7) -> dict:
    """块状 bootstrap（**季度**为重采样单位）求「格 − 对照」均值差与 95% CI。

    🚨 为什么按季度而非按事件：fwd42 在相邻调仓周之间重叠 41/42，事件不独立；
    按季度整块重采样保留块内相关结构，是这里最小可信的做法（按事件重采样会把
    CI 压窄数倍）。可用季度 < 8 时不给 CI，只报点估计。
    """
    a, b = cell["fwd42_open"].dropna(), ctrl["fwd42_open"].dropna()
    if len(a) < 5 or len(b) < 5:
        return {"diff": (float(a.mean() - b.mean()) if len(a) and len(b) else np.nan),
                "lo": np.nan, "hi": np.nan, "p": np.nan, "n": len(a)}
    point = float(a.mean() - b.mean())
    qa = pd.Series(a.to_numpy(),
                   index=pd.PeriodIndex(cell.loc[a.index, "date"], freq="Q"))
    qb = pd.Series(b.to_numpy(),
                   index=pd.PeriodIndex(ctrl.loc[b.index, "date"], freq="Q"))
    qs = sorted(set(qa.index) | set(qb.index))
    if len(qs) < 8:
        return {"diff": point, "lo": np.nan, "hi": np.nan, "p": np.nan, "n": len(a)}
    ga = {q: v.to_numpy() for q, v in qa.groupby(level=0)}
    gb = {q: v.to_numpy() for q, v in qb.groupby(level=0)}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = [qs[i] for i in rng.choice(len(qs), size=len(qs), replace=True)]
        va = [ga[q] for q in pick if q in ga]
        vb = [gb[q] for q in pick if q in gb]
        if va and vb:
            diffs.append(np.concatenate(va).mean() - np.concatenate(vb).mean())
    d = np.asarray(diffs)
    return {"diff": point, "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "p": float(np.mean(d <= 0)),
            "n": len(a)}


def cell_table(ev: pd.DataFrame, col: str, buckets: list, kind: str) -> list[str]:
    out = [f"  {'分组':<8}{'档位':<20}{'均值':>9}{'中位':>9}{'胜率':>8}{'n':>8}"]
    sub = ev[ev["kind"] == kind]
    for strong, g in (("前期强", sub[sub["strong"]]), ("不强", sub[~sub["strong"]])):
        for lo, hi, lab in buckets:
            m = (g[col] >= lo) & (g[col] < hi)
            out.append(f"  {strong:<8}{lab:<20}{_stat(g.loc[m, 'fwd42_open'])}")
    out.append("  ↳ 口径对照（T 收盘成交，仅与 10-27 旧探针可比，不作判据）：")
    for lo, hi, lab in buckets:
        m = (sub[col] >= lo) & (sub[col] < hi) & sub["strong"]
        out.append(f"  {'前期强':<8}{lab:<20}{_stat(sub.loc[m, 'fwd42_close'])}")
    return out


def diff_line(ev: pd.DataFrame, kind: str, tag: str, n_boot: int) -> str:
    """主判据一行：关键格 − 对照格 的差分与 bootstrap CI。"""
    sub = ev[(ev["kind"] == kind) & ev["strong"]]
    col, lo, hi = KEY_CELL
    ccol, clo, chi = CTRL_CELL
    cell = sub[(sub[col] >= lo) & (sub[col] < hi)]
    ctrl = sub[(sub[ccol] >= clo) & (sub[ccol] < chi)]
    r = _boot_diff(cell, ctrl, n_boot=n_boot)
    if not np.isfinite(r["diff"]):
        return f"  {tag:<8}{kind:<6}样本不足"
    ci = ("CI 不可算（季度 <8）" if not np.isfinite(r["lo"])
          else f"95% CI [{r['lo']:+.2%}, {r['hi']:+.2%}] p={r['p']:.3f}")
    return (f"  {tag:<8}{kind:<6}n={r['n']:>5}  深跌格 {cell['fwd42_open'].mean():>+7.2%}"
            f"  −  无回撤格 {ctrl['fwd42_open'].mean():>+7.2%}"
            f"  =  **{r['diff']:>+7.2%}**  {ci}")


def yearly_diff(ev: pd.DataFrame, kind: str, n_boot: int) -> list[str]:
    out = []
    sub = ev[ev["kind"] == kind]
    hit = False
    for y in (2015, 2018):
        g = sub[sub["year"] == y]
        if not len(g):
            continue
        hit = True
        out.append(diff_line(g.assign(kind=kind), kind, str(y), n_boot))
    if not hit:
        out.append("  （该窗口不含 2015 / 2018）")
    return out


# ------------------------------------------------------------------ 主流程
def main(argv=None) -> int:
    from data.cache import DataCache
    from data.datasource import create_datasource

    ap = argparse.ArgumentParser(description="深跌建仓 fwd42 崩盘区制存活检验")
    ap.add_argument("--window", default="long,near",
                    help="long / near / near45（45 只生产池，仅样本量对齐核对）")
    ap.add_argument("--min-amount", type=float, default=3e7,
                    help="流动性门槛（与 etf_long_sample 同默认）；0 = 不加约束")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap 次数")
    args = ap.parse_args(argv)
    wins = [w.strip() for w in args.window.split(",") if w.strip()]

    cache = DataCache(create_datasource())
    lines: list[str] = []
    rows: list[dict] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("=" * 100)
    say("ETF 轮动 · 「深跌建仓」收益腿的崩盘区制存活检验  2026-10-30")
    say("=" * 100)
    say(f"口径 = FINAL（Top3/HRP/周频/dedup 0.9）+ T+1 open；"
        f"fwd42 = open[T+1] → close[T+{FWD}]")
    say(f"「前期强」= mom12_1 在当日可交易池内的截面分位 ≥ 2/3；"
        f"流动性门槛 {args.min_amount / 1e4:.0f} 万（near45 走生产池默认）")
    say("🚨 判据为**差分**（关键格 − 同期同组无回撤格），不是绝对正负号 —— 理由见 docstring")

    for tag in wins:
        if tag == "long":
            env = load_env(cache, LONG_CODES, WARMUP, LONG_END, args.min_amount)
            start = LONG_START
            name = f"2014–2019（{len(LONG_CODES)} 只，含 2015 崩盘 / 2018 熊市）"
            r1 = _run(env, dict(FINAL))
        elif tag == "near":
            env = load_env(cache, LONG_CODES, 20180101, NEAR_END, args.min_amount)
            start = NEAR_START
            name = f"2020–2026（同 {len(LONG_CODES)} 只，旧探针样本区的同池版）"
            r1 = _run(env, dict(FINAL))
        elif tag == "near45":
            from scripts.portfolio.etf_contest5_eval import BEGIN
            from scripts.portfolio.etf_weight_probe import load_env as wp_load_env
            from scripts.portfolio.etf_weight_probe import run as wp_run
            env = wp_load_env()
            start = BEGIN
            name = "2020–2026（45 只生产池，仅供样本量对齐核对）"
            _, r1 = wp_run(env, dict(FINAL))
        else:
            raise SystemExit(f"未知窗口 {tag}")

        ev, n_rb = build_events(env, r1, start)
        ev["window"] = tag
        rows.append(ev)
        say()
        say(f"== {tag} · {name} ==")
        say(f"  面板 {env['close'].shape} {env['close'].index.min().date()}"
            f"~{env['close'].index.max().date()}；调仓 **{n_rb}** 次；"
            f"事件 {len(ev)} 条（entry {(ev['kind'] == 'entry').sum()} / "
            f"pool {(ev['kind'] == 'pool').sum()}）")
        say()
        say(f"  ── A. 前期强 × 近 {DD_WIN} 日回撤（pool 样本，主判据用）──")
        lines.extend(cell_table(ev, "dd60", DD_BUCKETS, "pool"))
        say()
        say("  ── B. 仅实际建仓事件（与 10-27 旧结论同构）──")
        lines.extend(cell_table(ev, "dd60", DD_BUCKETS, "entry"))
        say()
        say("  ── C. mom20 分档（复现 §七 #19 的 15 笔口径）──")
        lines.extend(cell_table(ev, "mom20", MOM20_BUCKETS, "entry"))
        say()
        say("  ── D. 主判据差分（bootstrap 以季度为块）──")
        say(diff_line(ev, "pool", "全窗口", args.boot))
        say(diff_line(ev, "entry", "全窗口", args.boot))
        say()
        say("  ── E. 崩盘 / 熊市单年 ──")
        for s in yearly_diff(ev, "pool", args.boot):
            say(s)

    all_ev = pd.concat(rows, ignore_index=True)

    # ---------------------------------------------------------- 判读
    say()
    say("=" * 100)
    say("== 判读（主判据读 long / pool）==")
    lg = all_ev[(all_ev["window"] == "long") & (all_ev["kind"] == "pool")]
    r_main = _boot_diff(lg[(lg["dd60"] < -0.20) & lg["strong"]],
                        lg[(lg["dd60"] > -0.10) & lg["strong"]], n_boot=args.boot)
    if not np.isfinite(r_main["diff"]):
        say("  🚨 长样本关键格样本不足 ⇒ **检验无结论**（不是通过）")
    else:
        ci = ("CI 不可算" if not np.isfinite(r_main["lo"])
              else f"95% CI [{r_main['lo']:+.2%}, {r_main['hi']:+.2%}]")
        say(f"  长样本 前期强×dd≤−20% 相对无回撤格的超额 = {r_main['diff']:+.2%}"
            f"（{ci}，n={r_main['n']}）")
        if r_main["diff"] >= 0 and np.isfinite(r_main["lo"]) and r_main["lo"] > 0:
            say("  ⇒ D1 **强存活**：优势在 2014–2019 显著为正；仍不改 FINAL。")
        elif r_main["diff"] >= 0:
            say("  ⇒ D1 **弱存活**：点估计 ≥ 0 但 CI 跨 0 ⇒ 不能宣称「深跌格更好」，")
            say("     只能说「没有证据说它更差」；§七 #19–22 维持封闭。")
        else:
            say("  ⇒ D1 **推翻超额表述**：「深跌建仓 = 收益腿」在 2014–2019 不成立，")
            say("     10-27 的 +5.36%/+8.69% 应读作 2020+ 单一区制的产物。")
            say("     🚨 但这**不改变** 10-27 的处置：结论仍是「不改 FINAL」——")
            say("     「深跌格不更好」只意味着别去加规则，不是要减仓或反向。")
    say()
    say("== v1 判据（已弃用，留档对照）==")
    key_abs = lg[(lg["dd60"] < -0.20) & lg["strong"]]["fwd42_open"].dropna()
    if len(key_abs):
        say(f"  绝对均值 {key_abs.mean():+.2%}、中位 {key_abs.median():+.2%}、"
            f"胜率 {(key_abs > 0).mean():.1%}、n={len(key_abs)}")
        say("  🚨 24 只池 2014–2019 几乎每格绝对收益都是正的（池子自身漂移）⇒")
        say("     v1 的「均值 < 0 即判负」测的是 beta、无区分力，已改用差分判据。")
    say()
    say("== 必读边界 ==")
    say("  ① 24 只本身是「今天还在的 ETF」⇒ 产品层幸存者偏差不可消除；")
    say("  ② 2014–2017 可交易日均 ≤10 只，截面宽度窄 ⇒ 「前 1/3」区分度弱；")
    say("  ③ `entry` 样本受池限制：24 只池 2020+ 的深跌格仅个位数，而 10-27 原结论是")
    say("     15 笔（45 只生产池）⇒ **两份样本不可直接比**，`near45` 只用于核对这一点；")
    say("  ④ bootstrap 以季度为块，只吸收块内相关；fwd42 重叠仍会让 CI 偏窄，只作量级参考；")
    say("  ⑤ 本检验只回答「要不要重开讨论」，**不改 FINAL、不构成采纳证据**。")

    OUT.mkdir(parents=True, exist_ok=True)
    TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    all_ev.to_csv(OUT / "deepdip_events.csv", index=False, encoding="utf-8-sig")
    pool = all_ev[all_ev["kind"] == "pool"].copy()
    pool["dd_bucket"] = pd.cut(pool["dd60"], [-9.99, -0.2, -0.1, 9.99],
                               labels=["≤−20%", "−20~−10%", ">−10%"])
    cells = (pool.groupby(["window", "strong", "dd_bucket"], observed=True)["fwd42_open"]
             .agg(n="count", mean="mean", median="median",
                  win_rate=lambda s: float((s > 0).mean())))
    cells.to_csv(OUT / "deepdip_cells.csv", encoding="utf-8-sig")
    print(f"\n落盘 → {TXT}\n落盘 → {OUT / 'deepdip_events.csv'}"
          f"\n落盘 → {OUT / 'deepdip_cells.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
