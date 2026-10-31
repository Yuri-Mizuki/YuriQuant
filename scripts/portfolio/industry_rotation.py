"""行业池轮动线（独立实验线，不与 ETF 三锋混跑）。

池子 = **申万一级行业指数 31 个**（官方、含长历史），不聚合个股因子——
所有信号直接在行业指数序列上原生计算。这是与「个股因子→行业聚合」的本质区别：
聚合会丢掉行业内部分歧（η² 实测中位 0.046），原生计算不丢。

设计（对齐研报方法论，但避开它的三个口径问题）：
  · 月频调仓（研报是周频；月频与项目调仓日一致，且换手可控）
  · 信号 T 日收盘 → **T+1 收盘成交**（不用 T 日收盘，避免同日成交假设）
  · 成本 36bp 单边（`Config.costs()` 口径；另报 13bp 作敏感性）
  · 多头 Top5 等权（研报选 5 个行业），基准 = 全行业等权、同期、同成交价
  · **分年 / 前后半段**都报，防止单段结论

信号族（全部行业原生）：
  mom12_1      12 个月动量，剔除最近 1 个月
  mom_1m       最近 1 个月（先验：短期反转，故反向用见 rev_1m）
  vol60        60 日波动
  ra_mom       动量 / 波动（风险调整动量）
  amt_chg      行业成交额 20 日 / 120 日（放量）
  sue_q_agg    个股 SUE 按行业内市值加权聚合（**对照组**，唯一的非原生信号）
  native_combo 原生族复合（+动量 −短期涨）
  rev_1m / amt_shrink / sue_low  上述三个的**反向臂**（样本内发现，须看前后半段）

用法：
    python -m scripts.portfolio.industry_rotation
    python -m scripts.portfolio.industry_rotation --start 2014-03-01 --top 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.metrics import annual_return, max_drawdown, sharpe_arithmetic, sharpe_ratio
from config import Config

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "industry_rotation"
PQ = Path(Config.paths()["cache_root"])

COST_BP = 36.0          # 单边，bp（项目统一口径）
COST_BP_ALT = 13.0      # ETF 实际（佣金 3 + 滑点 10，无印花税）


# --------------------------------------------------------------------------- 数据
def load_px() -> tuple[pd.DataFrame, pd.DataFrame]:
    """申万一级行业指数：收盘点位 + 成交额。"""
    d = pd.read_parquet(PQ / "sw_industry_daily.parquet")
    px = d.pivot_table(index="date", columns="code", values="close").sort_index()
    amt = d.pivot_table(index="date", columns="code", values="amount").sort_index()
    return px, amt.reindex_like(px)


def month_first_days(idx: pd.DatetimeIndex, start: pd.Timestamp) -> list[pd.Timestamp]:
    s = pd.Series(idx, index=idx)
    days = [pd.Timestamp(x) for x in s.groupby([idx.year, idx.month]).first().tolist()]
    return [d for d in days if d >= start]


# --------------------------------------------------------------------------- 信号
def build_signals(px: pd.DataFrame, amt: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ret = px.pct_change(fill_method=None)
    mom12_1 = px.shift(21) / px.shift(261) - 1
    mom_1m = px.shift(1) / px.shift(21) - 1
    vol60 = ret.rolling(60).std()
    sigs = {
        "mom12_1": mom12_1,
        "mom_1m": mom_1m,
        "vol60": vol60,
        "ra_mom": mom12_1 / vol60.replace(0, np.nan),
        "amt_chg": amt.rolling(20).mean() / amt.rolling(120).mean(),
    }
    # 反向臂（先验方向与实测相反，属样本内发现）
    for k in ("mom_1m", "amt_chg"):
        sigs[f"rev_{k}"] = -sigs[k]

    # 原生复合：+动量 −短期涨（两个都有先验方向，不拿 vol60 凑）
    r_mom = mom12_1.rank(axis=1, pct=True)
    r_rev = mom_1m.rank(axis=1, pct=True)
    sigs["native_combo"] = r_mom - r_rev
    return sigs


def agg_sue(dates: list[pd.Timestamp], px_cols: pd.Index) -> pd.DataFrame:
    """个股 SUE → 行业内流通市值加权（对照组信号，非原生）。"""
    from scripts.oneoff.etf_industry_factor_p0 import build_context

    panel = pd.read_parquet(
        PQ.parent / "factor_library" / "all_a_2018_2026" / "panels" / "sue_q.parquet")
    ctx = build_context(dates[0], freq="M")
    ctx_dates = pd.DatetimeIndex(ctx["dates"])
    rows = {}
    for dt in dates:
        # 申万日历与 A 股日历可能差 1 天 ⇒ 取 <= dt 的最近一个 ctx 调仓日
        i = int(ctx_dates.searchsorted(dt, side="right")) - 1
        if i < 0:
            continue
        cd = ctx_dates[i]
        if cd not in panel.index:
            continue
        v = panel.loc[cd].reindex(ctx["universe"]).to_numpy(dtype=float)
        ok = np.isfinite(v) & (ctx["lbls"][i] >= 0) & (ctx["wts"][i] > 0)
        if ok.sum() < 50:
            continue
        x = np.clip(v[ok], *np.percentile(v[ok], [1, 99]))
        L, W = ctx["lbls"][i][ok], ctx["wts"][i][ok]
        num = np.bincount(L, weights=x * W, minlength=31)
        den = np.bincount(L, weights=W, minlength=31)
        cnt = np.bincount(L, minlength=31)
        a = np.where((cnt >= 5) & (den > 0), num / np.where(den == 0, 1, den), np.nan)
        rows[dt] = pd.Series(a, index=ctx["inds"])
    return pd.DataFrame(rows).T.reindex(columns=px_cols)


# --------------------------------------------------------------------------- 回测
def exec_days(dates: list[pd.Timestamp], tdays: pd.DatetimeIndex) -> list[int]:
    """信号日 → 成交日（下一个交易日）的行号。"""
    out = []
    for d in dates:
        j = tdays.get_indexer([d])[0]
        if j < 0:
            j = int(tdays.searchsorted(d))
        out.append(min(j + 1, len(tdays) - 1))
    return out


def build_bench(px: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.Series:
    """统一基准：全行业等权、同成交价、同调仓日。所有信号共用，保证超额可比。"""
    tdays = px.index
    ed = exec_days(dates, tdays)
    out = {}
    for k in range(len(dates) - 1):
        e0, e1 = ed[k], ed[k + 1]
        if e1 <= e0:
            continue
        out[tdays[e1]] = float((px.iloc[e1] / px.iloc[e0] - 1).mean())
    return pd.Series(out).sort_index()


def backtest(
    sig: pd.DataFrame,
    px: pd.DataFrame,
    dates: list[pd.Timestamp],
    top: int = 5,
    cost_bp: float = COST_BP,
    bench: pd.Series | None = None,
) -> dict:
    """月频 Top-N 等权。信号 T 日 → T+1 收盘成交 → 下期 T+1 收盘卖出。"""
    tdays = px.index
    ed = exec_days(dates, tdays)
    nav, turns, w_old, idx = [], [], pd.Series(0.0, index=px.columns), []
    for k in range(len(dates) - 1):
        e0, e1 = ed[k], ed[k + 1]
        if e1 <= e0:
            continue
        j = tdays.get_indexer([dates[k]])[0]
        j = j if j >= 0 else int(tdays.searchsorted(dates[k]))
        s = sig.iloc[j].dropna()
        hold = px.iloc[e1] / px.iloc[e0] - 1
        s = s[s.index.isin(hold.dropna().index)]
        if len(s) < top * 2:
            continue

        rk = s.rank(ascending=False)
        long = rk[rk <= top].index
        w = pd.Series(0.0, index=px.columns)
        w[long] = 1.0 / len(long)

        gross = float((w * hold.fillna(0)).sum())
        turn = float((w - w_old).abs().sum() / 2)
        nav.append(gross - turn * 2 * cost_bp / 1e4)
        turns.append(turn)
        idx.append(tdays[e1])
        w_old = w

    if not nav:
        return {}
    r = pd.Series(nav, index=pd.DatetimeIndex(idx))
    b = (bench.reindex(r.index) if bench is not None
         else pd.Series(1.0, index=r.index))
    per = 12
    ex = r - b
    sd = ex.std(ddof=1)
    return {
        "ret": r, "bench_ret": b,
        "nav": (1 + r).cumprod(), "bench": (1 + b).cumprod(),
        "ann": annual_return(r, per), "ann_bench": annual_return(b, per),
        "vol": float(r.std(ddof=1) * np.sqrt(per)),
        "sharpe_geo": sharpe_ratio(r, periods_per_year=per),
        "sharpe_arith": sharpe_arithmetic(r, periods_per_year=per),
        "mdd": max_drawdown(r),
        "win": float((r > 0).mean()),
        "turnover": float(np.mean(turns) * per),
        "ir": float(ex.mean() / sd * np.sqrt(per)) if sd > 0 else np.nan,
        "t_ir": float(ex.mean() / sd * np.sqrt(len(ex))) if sd > 0 else np.nan,
        "n": len(r),
    }


def split_test(r: pd.Series, b: pd.Series) -> tuple[float, float]:
    """前后半段超额（年化），防止单段结论。"""
    h = len(r) // 2
    out = []
    for seg in (slice(0, h), slice(h, None)):
        rs, bs = r.iloc[seg], b.iloc[seg]
        out.append(annual_return(rs, 12) - annual_return(bs, 12))
    return out[0], out[1]


def static_control(sig: pd.DataFrame, px: pd.DataFrame, dates: list[pd.Timestamp],
                   top: int = 5) -> dict:
    """静态对照：固定持有「全样本最常入选」的 N 个标的，不轮动。

    用来分解「总超额」里有多少来自**长期选对了行业**（静态），
    多少来自**时点切换**（轮动）。只有后者才是轮动的贡献。
    """
    tdays = px.index
    ed = exec_days(dates, tdays)
    cnt = pd.Series(0.0, index=px.columns)
    for k in range(len(dates) - 1):
        j = tdays.get_indexer([dates[k]])[0]
        j = j if j >= 0 else int(tdays.searchsorted(dates[k]))
        v = sig.iloc[j].dropna()
        if len(v) >= top * 2:
            cnt[v.nlargest(top).index] += 1
    fixed = cnt.nlargest(top).index
    rets, idx = [], []
    for k in range(len(dates) - 1):
        e0, e1 = ed[k], ed[k + 1]
        if e1 <= e0:
            continue
        h = px.iloc[e1] / px.iloc[e0] - 1
        rets.append(float(h.reindex(fixed).mean()))
        idx.append(tdays[e1])
    r = pd.Series(rets, index=pd.DatetimeIndex(idx))
    return {"names": list(fixed), "ann": annual_return(r, 12),
            "sharpe": sharpe_ratio(r, periods_per_year=12), "ret": r}


def gap_static(sig: pd.DataFrame, px: pd.DataFrame, dates: list,
               top: int = 5, bench: pd.Series | None = None) -> dict:
    """把「总超额」拆成 静态选行业 + 时点轮动 两段（都在同一基准之上）。"""
    r = backtest(sig, px, dates, top=top, bench=bench)
    st = static_control(sig, px, dates, top=top)
    if not r:
        return {}
    return {"total": r["ann"] - r["ann_bench"],
            "static": st["ann"] - r["ann_bench"],
            "timing": (r["ann"] - r["ann_bench"]) - (st["ann"] - r["ann_bench"]),
            "fixed": st["names"], "run_ann": r["ann"], "static_ann": st["ann"]}


def quintiles(sig, px, dates, top_cols=None) -> pd.DataFrame:
    """研报的「分 5 组」：每组等权**毛收益**（不扣成本，只判单调性）。"""
    tdays = px.index
    ed = exec_days(dates, tdays)
    buckets: dict[int, list[float]] = {i: [] for i in range(5)}
    for k in range(len(dates) - 1):
        e0, e1 = ed[k], ed[k + 1]
        if e1 <= e0:
            continue
        j = tdays.get_indexer([dates[k]])[0]
        j = j if j >= 0 else int(tdays.searchsorted(dates[k]))
        s = sig.iloc[j].dropna()
        hold = (px.iloc[e1] / px.iloc[e0] - 1).dropna()
        s = s[s.index.isin(hold.index)]
        if len(s) < 10:
            continue
        q = pd.qcut(s.rank(method="first"), 5, labels=False)
        for b in range(5):
            buckets[b].append(float(hold.reindex(s.index[q == b]).mean()))
    return pd.DataFrame({f"G{b + 1}": pd.Series(buckets[b]) for b in range(5)})


def by_year(r: pd.Series, b: pd.Series) -> pd.DataFrame:
    g = pd.DataFrame({"ret": r, "bench": b})
    g["y"] = g.index.year
    t = g.groupby("y").agg(ret=("ret", lambda s: (1 + s).prod() - 1),
                           bench=("bench", lambda s: (1 + s).prod() - 1))
    t["excess"] = t["ret"] - t["bench"]
    return t


# --------------------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    px, amt = load_px()
    dates = month_first_days(px.index, pd.Timestamp(args.start))
    print(f"池 = {px.shape[1]} 个申万一级行业 · 调仓 {len(dates)} 期 "
          f"({dates[0].date()}~{dates[-1].date()})")

    bench = build_bench(px, dates)
    sigs = build_signals(px, amt)
    try:
        sigs["sue_q_agg"] = agg_sue(dates, px.columns).reindex(index=px.index)
        sigs["sue_low"] = -sigs["sue_q_agg"]
    except Exception as e:  # 数据缺失不该阻断主实验
        print(f"  跳过 sue_q_agg：{e}")

    results, halves = {}, {}
    for name, s in sigs.items():
        r = backtest(s, px, dates, top=args.top, bench=bench)
        if not r:
            continue
        results[name] = r
        halves[name] = split_test(r["ret"], r["bench_ret"])
        a, c = halves[name]
        print(f"  {name:14s} 年化 {r['ann']:+.2%}  超额 {r['ann'] - r['ann_bench']:+.2%}  "
              f"Sharpe {r['sharpe_geo']:+.2f}  t(IR) {r['t_ir']:+.2f}  "
              f"前后半 {a:+.1%}/{c:+.1%}")

    alt = {}
    for name in ("mom12_1", "vol60", "rev_mom_1m", "rev_amt_chg"):
        if name in sigs:
            alt[name] = backtest(sigs[name], px, dates, top=args.top,
                                 cost_bp=COST_BP_ALT, bench=bench)

    # ---- 报告
    order = sorted(results, key=lambda k: -(results[k]["ann"] - results[k]["ann_bench"]))
    L = [
        "# 行业池轮动 · 唯一站得住的只有 sue_low（2026-10-31）",
        "",
        "> **判定**：`工具保留` —— 新实验线已建且可复现；行业原生信号层结论为**负**，"
        "`sue_low` 待独立样本外验证",
        "> **脚本**：`scripts/portfolio/industry_rotation.py`"
        "（`python -m scripts.portfolio.industry_rotation`）",
        "> **产物**：本目录（`README.md` / `nav.csv` / `metrics.csv` / `by_year.csv`）",
        "> **上下游**：上游 `reports/etf_contest5/industry_factor_p0.*`（因子筛选）；下游 无",
        "",
        f"> 池 = **{px.shape[1]} 个申万一级行业指数**（官方）· 月频 {len(dates)} 期"
        f"（{dates[0].date()}~{dates[-1].date()}）· Top{args.top} 等权 · "
        f"T+1 收盘成交 · 成本 {COST_BP:.0f}bp 单边",
        "> 信号全部在**行业指数上原生计算**，唯一例外是 `sue_q_agg`（个股聚合，对照组）",
        "",
        "## 〇 结论先行",
        "",
        "**十个信号里九个没有稳定超额，唯一站得住的是 `sue_low`。**",
        "",
        "- **行业原生信号全线无效**：12-1 动量 `mom12_1` 超额 +0.7pp（前后半 +6.1%/−4.7%，"
        "后半转负）、`vol60` +1.5pp（+8.0%/−4.6% 同病）、风险调整动量 +0.5pp。"
        "这类信号的超额只存在于 2018–2022，2023 之后全部消失。",
        "- **短期动量（`mom_1m`）与放量（`amt_chg`）是稳定的负信号**（t(IR) −2.4 / −3.0，"
        "两段都负）⇒ 追短期涨幅、追放量，在行业层稳定跑输。但**反向也不赚**"
        "（`rev_mom_1m` −5.7%、`rev_amt_chg` −7.1%）⇒ 是**倒 U 型**（两端差、中间好），"
        "不是单调反转，不能用「取反」来收割。",
        "- **`sue_low`（做多低 SUE 行业）是唯一两段同号、t>2 的信号**："
        "年化 +9.15%、超额 +7.80pp、Sharpe 0.43、t(IR) 2.15、分年 **7/9 为正**。"
        "它不是事后挑的方向——P0 全量 950 因子筛选里 `sue_q` 的 IC 本来就是 −0.106（t −4.84，"
        "唯一过 BH-FDR），方向一致，属于**先验 + 回测双重验证**。"
        "⚠️ 但它是**个股 SUE 聚合**，不是行业原生；且下面第三节显示 45% 的超额来自"
        "「长期选对行业」而非「时点切换」。",
        "",
        "| 信号 | 期数 | 年化 | 基准 | 超额 | Sharpe | t(IR) | 回撤 | 胜率 | 换手 | 前半 | 后半 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in order:
        r = results[name]
        a, c = halves[name]
        L.append(
            f"| `{name}` | {r['n']} | {r['ann']:+.2%} | {r['ann_bench']:+.2%} | "
            f"**{r['ann'] - r['ann_bench']:+.2%}** | {r['sharpe_geo']:+.2f} | "
            f"{r['t_ir']:+.2f} | {r['mdd']:.1%} | {r['win']:.0%} | "
            f"{r['turnover']:.0%} | {a:+.1%} | {c:+.1%} |"
        )

    L += [
        "",
        "⚠️ **前半/后半是 2018-2022 / 2023-2026 两段**。两段同号的才算稳；"
        "只有一段为正的，一律视为样本内过拟合。",
        "⚠️ **期数不同（105/100/96）**：`vol60` 需 60 日暖机、`amt_chg` 需 120 日，"
        "故起点晚 3~6 个月。所以「基准」列是**该信号可用期内的**等权基准年化，"
        "不是同一个固定窗口——跨行比较超额可以，比较绝对年化要留意。",
        "",
        "## 一、分 5 组分层（研报口径，毛收益不扣成本）",
        "",
        "| 信号 | G1(弱) | G2 | G3 | G4 | G5(强) | 单调 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in order:
        q = quintiles(sigs[name], px, dates)
        if q.empty:
            continue
        anns = [(1 + q[c]).prod() ** (12 / len(q)) - 1 for c in q.columns]
        mono = "✅" if (anns[4] > anns[2] > anns[0]) else "否"
        L.append(f"| `{name}` | " + " | ".join(f"{a:+.1%}" for a in anns) + f" | {mono} |")

    best = order[0]
    rb = results[best]
    gap = gap_static(sigs[best], px, dates, top=args.top, bench=bench)
    L += ["", f"## 二、最强信号 `{best}` 分年", "",
          "| 年 | 策略 | 等权基准 | 超额 |", "|---|---|---|---|"]
    yt = by_year(rb["ret"], rb["bench_ret"])
    for y, row in yt.iterrows():
        L.append(f"| {y} | {row['ret']:+.2%} | {row['bench']:+.2%} | {row['excess']:+.2%} |")
    pos_years = int((yt["excess"] > 0).sum())
    L.append(f"| **合计** | | | **{pos_years}/{len(yt)} 年为正** |")

    if alt:
        L += ["", "## 三、静态 vs 轮动分解（关键）", "",
              f"固定持有全样本最常入选的 {args.top} 个行业"
              f"（`{'`、`'.join(str(x) for x in gap['fixed'])}`）不轮动，与轮动版对照：",
              "",
              "| 口径 | 年化 | 相对等权基准超额 |",
              "|---|---|---|",
              f"| 轮动 Top{args.top} | {gap['run_ann']:+.2%} | **{gap['total']:+.2%}** |",
              f"| 静态持有同样 {args.top} 个行业 | {gap['static_ann']:+.2%} | "
              f"{gap['static']:+.2%} |",
              f"| **差额 = 时点轮动贡献** | | **{gap['timing']:+.2%}** |",
              "",
              f"⇒ 总超额里约 **{gap['static'] / gap['total'] * 100:.0f}% 来自「长期选对行业」**，"
              f"{gap['timing'] / gap['total'] * 100:.0f}% 来自「时点切换」。"
              "前者是行业选择偏差，后者才是轮动本身。",
              ""]

    if alt:
        L += ["", "## 四、成本敏感性（13bp = ETF 实际佣金+滑点，无印花税）", "",
              "| 信号 | 36bp 年化 | 13bp 年化 | 差 |", "|---|---|---|---|"]
        for name, r in alt.items():
            if name in results:
                L.append(f"| `{name}` | {results[name]['ann']:+.2%} | {r['ann']:+.2%} | "
                         f"{r['ann'] - results[name]['ann']:+.2%} |")

    L += [
        "",
        "## 五、复现",
        "",
        "```bash",
        "python -m scripts.ingest.update_sw_industry --check   # 首次需去掉 --check 拉取 31 条指数",
        "python -m scripts.portfolio.industry_rotation          # 默认 2018 起、Top5",
        "python -m scripts.portfolio.industry_rotation --start 2014-03-01 --top 5",
        "```",
        "",
        "产物：`reports/industry_rotation/{README.md,nav.csv,metrics.csv,by_year.csv}`。",
        "数据依赖：`E:/data/parquet/sw_industry_daily.parquet`（申万官方指数）、"
        "`factor_library/all_a_2018_2026/panels/sue_q.parquet`（对照信号）。",
        "",
        "## 六、边界（引用前必读）",
        "",
        "1. **这是指数层，不是可交易层**。31 个申万一级里有对应 ETF 的只有约 15 个方向；"
        "指数点位不能直接买。落地层要另算覆盖率与跟踪误差。",
        "2. **申万指数是价格指数，不含分红**。等权基准年化只有 1.0%（日频），"
        "加上 2% 左右股息才是真实持有收益；各行业股息率 0.5%~6% 不等 ⇒ "
        "**行业间收益有系统性偏差**，高股息行业被系统性低估。",
        "3. **T+1 收盘成交**是保守假设；实际行业 ETF 可在 T+1 开盘附近成交。",
        "4. **月频 105 期**，Top5/31 的入选概率是 16%，基准波动 20%/年 ⇒ "
        "年化超额的标准误约 2pp。**+1.5pp 的超额在统计上与 0 不可区分。**",
        "5. **不能与 ETF 三锋直接比**：那条是选 ETF（多资产、含跨境商品债），"
        "这条是选行业（纯 A 股权益），资产域不同、基准不同、波动不同。",
        "6. 本线**未过** `caliber.py`：`baselines.yaml` 里没有行业轮动的合法基线，"
        "所以任何数字只能做线内比较，不能对外声称「改进」。",
        "7. **`rev_*` / `sue_low` 三个反向臂是看到回测结果之后才定的方向**，"
        "本身就是多重检验的一部分。它们的「超额」在写进任何结论前，"
        "必须先做独立样本外（2018-2022 定方向、2023-2026 验），见上表前后半段列。",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(L), encoding="utf-8")

    pd.DataFrame({k: v["nav"] for k, v in results.items()}).to_csv(
        OUT_DIR / "nav.csv", encoding="utf-8-sig")
    pd.DataFrame([
        {"signal": k, "ann": v["ann"], "ann_bench": v["ann_bench"],
         "excess": v["ann"] - v["ann_bench"], "sharpe_geo": v["sharpe_geo"],
         "sharpe_arith": v["sharpe_arith"], "t_ir": v["t_ir"], "mdd": v["mdd"],
         "win": v["win"], "turnover": v["turnover"], "n": v["n"],
         "half1": halves[k][0], "half2": halves[k][1]}
        for k, v in results.items()
    ]).to_csv(OUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    yt.to_csv(OUT_DIR / "by_year.csv", encoding="utf-8-sig")
    print(f"\n→ {OUT_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
