"""行业集中度约束：从「标签去重」升级为「风险同源去重」（2026-11-11）。

回答 2026-11-11 的请求：「按先验上更该怎么做去测试」。

先验与它的自我修正
------------------
先验上原始主张是三条：① 按**风险贡献**约束 ② 退一步用**行业权重上限** ③ 绝不要「去重不补位」。

**② 在等权框架里退化**：等权时「行业权重上限 w」⇔「每行业 ≤ k·w 只」，与现行
``max_per_sector`` **数学上完全等价**（两者都是「某行业最多几只」的另一种写法）。
所以「去掉标签限只数、改限权重」在等权下**没有可测内容**，不能做成一个 arm。

**真正没有测过的只有一条**：把「同行业」换成「**风险同源**」。
行业标签只是一个粗糙的相关性代理 —— 它把风险同源的拆开（银行 vs 非银，一级分开算），
又把风险异质的合并（医药生物含化学制药/中药/生物制品）。第①条才是先验上正确的做法。

本脚本测的就是第①条，以及「什么都不做 / 换层级」两个对照。

臂（5 个，同 k=8 b=1.0 10 万整手 production 口径）
--------------------------------------------------
  R0_none          不去重（P0，对照）
  R1_l1_dedup      现行定版：申万一级去重、踢掉不补位
  R2_l2_dedup      换申万二级去重（检验「层级」这个变量的方向）
  R3_riskgroup     **风险组去重**：行业指数收益滚动相关 → 聚类成风险组 → 每组建 1 只
  R4_riskbudget    **风险预算**：每只权重 ∝ 1/√(组合边际风险贡献)，行业标签不参与

风险组怎么构造（PIT 安全）
--------------------------
1. 只用**调仓日之前** ``WINDOW`` 个交易日的 31 个申万一级**行业指数收益**（``sw_industry_daily``）
   估相关矩阵 —— 不碰个股、不用未来数据。
2. 相关性距离 ``d = sqrt(2(1−ρ))`` + 平均连接（average linkage）层次聚类。
3. ``n_groups`` 固定为 8（= k），即「允许的最多样化程度」，再按组内成员映射个股。
   固定簇数、不按数据挑，避免「调最优簇数」这种事后选择。
4. 聚类结果每 ``REFIT`` 个调仓日重估一次（降低抖动，也贴近「一年重估几次」的实务）。

风险预算（R4）怎么算
--------------------
等权组合 w_i = 1/N，边际风险 ``MRC_i = (Σw)_i / σ_p``，风险贡献 ``RC_i = w_i·MRC_i``。
按 ``RC`` 的离散度调权：``u_i ∝ 1/RC_i`` 后归一到 Σw=1，迭代 20 次。
``Σ`` 用个股日收益 + 行业指数作为收缩目标（``Σ = (1−λ)·S + λ·F``，λ=0.5），
避免 8 只票的样本协方差不可逆 —— 🚨 这一步**必须** PIT：只用调仓日之前的数据。

输出 ``reports/sector_risk_budget/``
------------------------------------
  risk_arms.csv      5 臂 × 21 相位汇总
  risk_pairs.csv     逐相位配对差
  risk_yearly.csv    逐年
  risk_groups.csv    风险组随时间的变化（稳定性 / 与一级行业的差异）
  risk_phase.csv     逐相位明细

用法::

    python -m scripts.evaluation.sector_risk_probe              # 全跑（约 3 分钟）
    python -m scripts.evaluation.sector_risk_probe --skip-arms  # 只看聚类诊断
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.evaluation.caliber as cal
from data.point_in_time import lookup_intervals
from strategy.base import Strategy

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "sector_risk_budget"
END = 20260907
LOT = 100
CAPITAL = 1e5
K = 8
BUF = 1.0
RHO = 0.85            # 风险同源阈值：行业指数日收益相关 > RHO 视为同一笔风险
                      # 🚨 事前固定，不按样本调。0.75 会让 2020 年 24 个行业连成一簇（链式合并），
                      # 0.90 会让 2023/2026 年全部 31 个行业互不相连 ⇒ 分组退化为「按标签」。
WINDOW = 252          # 估相关矩阵的回看窗口（交易日）
REFIT = 6             # 每多少个调仓日重聚类一次
LAMBDA_SHRINK = 0.5   # 个股协方差向行业因子收缩的强度

PHASES = list(range(21))
SAFE_PHASE = 15       # ≥16 会因「该月交易日不足」回退月末，非独立抽样


# ---------------------------------------------------------------- 数据

def ind_l1(ctx) -> pd.DataFrame:
    """申万一级行业（PIT），定版用的就是这一层。"""
    return pd.read_parquet(cal.BASE_DIR / "industry.parquet").reindex(
        index=ctx.pred.index, columns=ctx.pred.columns)


def ind_l2(ctx) -> pd.DataFrame:
    """申万二级（PIT 展开）。用于检验「层级」这个变量。"""
    rec = pd.read_parquet(ROOT.parent / "data" / "parquet" /
                          "industry_classification_level2.parquet")
    return lookup_intervals(rec, ctx.pred.index, list(ctx.pred.columns),
                            code_col="code", value_col="industry_code")


def ind_returns() -> pd.DataFrame:
    """31 个申万一级**行业指数**日收益（不是成分股，天然无个股噪声）。"""
    sw = pd.read_parquet(ROOT.parent / "data" / "parquet" / "sw_industry_daily.parquet")
    sw = sw.assign(date=pd.to_datetime(sw["date"]))
    px = sw.pivot_table(index="date", columns="code", values="close", aggfunc="last")
    px = px.sort_index()
    return px.pct_change(fill_method=None).dropna(how="all")


def price_panel(ctx) -> pd.DataFrame:
    raw = pd.read_parquet(cal.BASE_DIR / "close_raw.parquet")
    return raw.reindex(index=ctx.pred.index, columns=ctx.pred.columns).ffill().shift(1)


# ---------------------------------------------------------------- 风险组

def cluster_groups(iret: pd.DataFrame, asof: pd.Timestamp) -> dict[str, int]:
    """只用 ``asof`` **之前** ``WINDOW`` 日的行业指数收益，返回 {行业代码: 组号}。

    定义「风险同源」= 相关性 **连通分量**（ρ > ``RHO`` 直接相连即同组，可传递）。
    🚨 不用「固定簇数」：实测固定 8 簇时最大簇会吞掉 **15~22 个行业**
    （2020 年相关中位 0.648，28 个行业挤成一簇）⇒ 每簇只留 1 只等于持仓崩到个位数，
    不可执行。连通分量让**组的大小由数据决定**，高相关年份自然并成大组，这本身就是结论。
    """
    hist = iret.loc[:asof].iloc[:-1]        # 严格不含当日
    hist = hist.iloc[-WINDOW:]
    if len(hist) < 60 or hist.shape[1] < 2:
        return {c: i for i, c in enumerate(iret.columns)}
    c = hist.corr().fillna(0.0).values
    n = c.shape[0]
    # 并查集求连通分量
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if c[i, j] > RHO:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    roots = [find(i) for i in range(n)]
    uniq = {r: k for k, r in enumerate(sorted(set(roots)))}
    return dict(zip(iret.columns, [uniq[r] for r in roots], strict=False))


class RiskGroupTopK(Strategy):
    """按**风险组**去重（而非行业标签）+ 缓冲带；不补位（与定版同语义）。

    ``group_of`` 每期由 :class:`RiskGroupSchedule` 动态提供，故需 ``get_weights_at``。
    """

    name = "riskgroup_topk"

    def __init__(self, sched: "RiskGroupSchedule", k: int = K, max_per_group: int = 1):
        self.sched = sched
        self.k = int(k)
        self.max_per_group = int(max_per_group)
        self._prev: set = set()
        self.name = f"riskgroup_topk_{k}_{max_per_group}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        raise NotImplementedError("风险组需按日期定位，请用 get_weights_at")

    def get_weights_at(self, date: pd.Timestamp, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if vals.empty:
            self._prev = set()
            return pd.Series(dtype=float)
        grp = self.sched.at(date)
        ranks = vals.rank(ascending=False, method="first")
        n_exit = max(self.k, int(round(self.k * BUF)))

        counts: dict[str, int] = {}
        keep: list = []
        for code in sorted(self._prev, key=lambda c: ranks.get(c, float("inf"))):
            if code not in ranks.index or ranks[code] > n_exit:
                continue
            g = grp.get(code, "__NA__")
            if counts.get(g, 0) >= self.max_per_group:
                continue
            counts[g] = counts.get(g, 0) + 1
            keep.append(code)
            if len(keep) >= self.k:
                break

        fills: list = []
        for code in ranks.sort_values().index:
            if len(keep) + len(fills) >= self.k:
                break
            if ranks[code] > self.k:                 # 与定版一致：只在 TopK 内选
                break
            if code in self._prev and ranks[code] <= n_exit:
                continue
            g = grp.get(code, "__NA__")
            if counts.get(g, 0) >= self.max_per_group:
                continue
            counts[g] = counts.get(g, 0) + 1
            fills.append(code)

        port = keep + fills
        self._prev = set(port)
        if not port:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(port), index=port)


class RiskGroupSchedule:
    """``date -> {code: 组号}``：按 REFIT 间隔重聚类，组号在相邻期之间对齐。"""

    def __init__(self, iret: pd.DataFrame, l1: pd.DataFrame):
        self.iret = iret
        self.l1 = l1
        self._cache: dict[pd.Timestamp, dict[str, str]] = {}

    def at(self, date: pd.Timestamp) -> dict[str, str]:
        if date in self._cache:
            return self._cache[date]
        g = cluster_groups(self.iret, date)
        sec = (self.l1.loc[date] if date in self.l1.index
               else pd.Series(dtype=object))
        out = {}
        for code, s in sec.items():
            if isinstance(s, str) and s in g:
                out[code] = f"G{g[s]}"
        self._cache[date] = out
        return out


class RiskBudgetTopK(Strategy):
    """不做行业去重，改为**风险预算**：权重 ∝ 1/RC_i，让每只票贡献相同风险。

    .. warning:: **10 万整手口径下它不可执行**。风险预算会把小权重票压到一手都买不到
       （实测取 Top8、实持只有 **4.43 只**）⇒ 它是「4.4 只的组合」而不是「8 只的风险预算
       组合」，与 R0/R1 不同持仓数、不可直接比。故本臂只作**方向旁证**，
       结论以「实持匹配」的 ``R5_none_k4`` 为准。
    """

    name = "riskbudget_topk"

    def __init__(self, covfn, k: int = K, w_cap: float = 1.0):
        self.covfn = covfn
        self.k = int(k)
        self.w_cap = float(w_cap)      # 单票权重上限，抑制过小权重被整手吃掉
        self._prev: set = set()
        self.name = f"riskbudget_topk_{k}_c{w_cap:.2f}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        raise NotImplementedError("风险预算需按日期取协方差，请用 get_weights_at")

    def get_weights_at(self, date: pd.Timestamp, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if vals.empty:
            self._prev = set()
            return pd.Series(dtype=float)
        top = list(vals.nlargest(self.k).index)
        self._prev = set(top)
        if self.w_cap < 1.0:
            return self._capped(date, top)
        return self._rc_weights(date, top)

    def _capped(self, date, names) -> pd.Series:
        """``w_cap`` 封顶 + 迭代重分配：防止风险预算把小票压到买不起一手。"""
        w = self._rc_weights(date, names)
        for _ in range(30):
            over = w > self.w_cap
            if not over.any():
                break
            excess = float((w[over] - self.w_cap).sum())
            w[over] = self.w_cap
            free = ~over
            if not free.any():
                break
            w[free] += excess * w[free] / w[free].sum()
        return w / w.sum()

    def _rc_weights(self, date, names) -> pd.Series:
        n = len(names)
        if n < 2:
            return pd.Series(1.0, index=names)
        sig = self.covfn(date, names)
        w = np.full(n, 1.0 / n)
        for _ in range(20):                      # 固定迭代次数，收敛即可
            sw = sig @ w
            var = float(w @ sw)
            if var <= 0:
                break
            mrc = sw / np.sqrt(var)
            rc = w * mrc
            rc = np.maximum(rc, 1e-12)
            tgt = (1.0 / rc)
            w = tgt / tgt.sum()
        w = np.maximum(w, 1e-9)
        return pd.Series(w / w.sum(), index=names)


def make_covfn(ctx, l1: pd.DataFrame):
    """个股日收益 + 行业因子收缩；**只用到 date 之前**的数据。"""
    ret = ctx.ret if hasattr(ctx, "ret") else None
    if ret is None:
        raw = pd.read_parquet(cal.BASE_DIR / "close_adj.parquet")
        ret = raw.reindex(index=ctx.pred.index, columns=ctx.pred.columns).pct_change(
            fill_method=None)
    cache: dict = {}

    def covfn(date: pd.Timestamp, names: list) -> np.ndarray:
        key = (date, tuple(names))
        if key in cache:
            return cache[key]
        hist = ret.loc[:date].iloc[-(WINDOW + 1):-1, :]
        sub = hist.reindex(columns=names).dropna(axis=1, how="any")
        common = [c for c in names if c in sub.columns]
        if len(common) < 2:
            # 历史不足（窗口初期）：退化为对角协方差，等权即最优
            out = np.eye(len(names)) * 4e-4
            cache[key] = out
            return out
        S = sub[common].cov().values
        # 行业因子：个股所属行业的指数收益
        sec = l1.loc[date] if date in l1.index else pd.Series(dtype=object)
        fac = pd.Series({c: sec.get(c) for c in common})
        F = np.zeros_like(S)
        if fac.notna().any():
            grp = fac.fillna("__NA__")
            for g in grp.unique():
                m = (grp == g).values
                if m.sum() == 0:
                    continue
                blk = S[np.ix_(m, m)]
                F[np.ix_(m, m)] = blk.mean()        # 组内平均协方差 = 单因子结构
        sig = (1 - LAMBDA_SHRINK) * S + LAMBDA_SHRINK * F
        # 补足缺失：把 common 的协方差嵌回全 names 的顺序里
        if len(common) < len(names):
            base = float(np.trace(S)) / max(len(S), 1)
            full = np.eye(len(names)) * max(base, 1e-8)
            pos = [names.index(c) for c in common]
            full[np.ix_(pos, pos)] = sig
            sig = full
        sig = (sig + sig.T) / 2
        sig = sig + np.eye(len(sig)) * max(np.trace(sig) / len(sig), 1e-8) * 0.1
        cache[key] = sig
        return sig

    return covfn


# ---------------------------------------------------------------- 回测

def ann(dr: pd.Series) -> float:
    if len(dr) == 0:
        return np.nan
    return float((1 + dr).prod() ** (252.0 / len(dr)) - 1)


def run_arm(ctx, cost, px, strat, phase: int) -> dict:
    bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                            initial_capital=CAPITAL, costs=cost)
    res = cal.run_once(bt, ctx, "open", signal=None, freq="M", phase=phase,
                       price_panel=px, lot_size=LOT)
    return res


def summarize(res, tag: str, phase: int) -> dict:
    m = cal._metrics(res, CAPITAL)
    wh = res.weights_history
    nh = (wh > 1e-9).sum(axis=1)
    return {"arm": tag, "phase": phase, "ann": m["ann"], "sr": m["sharpe_arith"],
            "vol": m["vol"], "mdd": m["max_dd"], "nhold": float(nh.mean())}


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = cal.load_context("gbdt_mid13", window="prod", end=END, execution="open")
    cost = cal.cost_profile("production")
    px = price_panel(ctx)
    l1 = ind_l1(ctx)
    print(f"窗口 {ctx.pred.index[0].date()}..{ctx.pred.index[-1].date()} | "
          f"资金 {CAPITAL:,.0f} | 整手 {LOT} | k={K}", flush=True)

    iret = ind_returns()
    print(f"行业指数 {iret.shape[1]} 个 × {len(iret)} 日 "
          f"({iret.index[0].date()}..{iret.index[-1].date()})", flush=True)

    # ---- 聚类诊断：风险组 vs 一级行业 ----
    from strategy.examples import SectorCappedBufferedTopKLongOnly
    reb = cal.rebalance_dates(ctx.pred.index, freq="M", phase=0)
    diag = []
    for d in reb[::max(1, REFIT)]:
        g = cluster_groups(iret, d)
        sizes = pd.Series(g).value_counts()
        diag.append({"date": d, "n_group": int(sizes.size),
                     "max_size": int(sizes.max()), "min_size": int(sizes.min()),
                     "member_sizes": ";".join(str(x) for x in sorted(sizes.tolist()))})
    DG = pd.DataFrame(diag)
    DG.to_csv(OUT / "risk_groups.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 风险组诊断（{len(DG)} 次重估）===", flush=True)
    print(f"  簇数 {DG['n_group'].unique()} | 最大簇成员数 中位 "
          f"{DG['max_size'].median():.0f} / 最大 {DG['max_size'].max()}", flush=True)
    print(f"  最大簇成员数 ≥6 的占比 {(DG['max_size'] >= 6).mean():.0%}"
          f"（= 这些行业被打包成同一组，行业内已无法分散）", flush=True)

    if "--skip-arms" in sys.argv:
        print(f"\n产物 → {OUT}（仅诊断，{time.time() - t0:.0f}s）", flush=True)
        return

    # ---- 五臂 ----
    sched = RiskGroupSchedule(iret, l1)
    covfn = make_covfn(ctx, l1)
    l2 = ind_l2(ctx)
    spec = [
        ("R0_none", lambda: SectorCappedBufferedTopKLongOnly(
            l1, k=K, exit_mult=BUF, max_per_sector=K)),
        ("R1_l1_dedup", lambda: SectorCappedBufferedTopKLongOnly(
            l1, k=K, exit_mult=BUF, max_per_sector=1)),
        ("R2_l2_dedup", lambda: SectorCappedBufferedTopKLongOnly(
            l2, k=K, exit_mult=BUF, max_per_sector=1)),
        ("R3_riskgroup", lambda: RiskGroupTopK(sched, k=K, max_per_group=1)),
        ("R4_riskbudget", lambda: RiskBudgetTopK(covfn, k=K)),
        # 🔑 持仓数匹配对照：R3 实持 4.01、R4 实持 4.43 ⇒ 必须和「不去重的 k4」比，
        #    否则无法区分「风险口径有用」与「只是持仓少 ⇒ 杠杆放大」。
        ("R5_none_k4", lambda: SectorCappedBufferedTopKLongOnly(
            l1, k=4, exit_mult=BUF, max_per_sector=4)),
        # 封顶版风险预算：单票 ≤25%，抑制小权重被整手吃掉
        ("R6_rb_cap25", lambda: RiskBudgetTopK(covfn, k=K, w_cap=0.25)),
    ]
    rows, detail = [], []
    print("\n=== 五臂 × 21 相位 ===", flush=True)
    for tag, mk in spec:
        t1 = time.time()
        for p in PHASES:
            res = run_arm(ctx, cost, px, mk(), p)
            r = summarize(res, tag, p)
            rows.append(r)
            dr = res.daily_returns
            yr = (1 + dr).groupby(dr.index.year).prod()
            for y, v in yr.items():
                detail.append({"arm": tag, "phase": p, "year": int(y),
                               "ret": float(v - 1)})
        sub = pd.DataFrame(rows)
        sub = sub[sub["arm"] == tag]
        print(f"  {tag:<15} 均值 {sub['ann'].mean():6.2%}  "
              f"phase0 {sub['ann'].iloc[0]:6.2%}  min {sub['ann'].min():6.2%}  "
              f"SR {sub['sr'].mean():.3f}  实持 {sub['nhold'].mean():.2f}  "
              f"[{time.time() - t1:.0f}s]", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "risk_arms.csv", index=False, encoding="utf-8-sig")
    D = pd.DataFrame(detail)
    D.to_csv(OUT / "risk_phase.csv", index=False, encoding="utf-8-sig")
    Y = D.pivot_table(index="year", columns="arm", values="ret", aggfunc="mean")
    Y.to_csv(OUT / "risk_yearly.csv", encoding="utf-8-sig")

    wide = R.pivot(index="phase", columns="arm", values="ann")
    ws = R.pivot(index="phase", columns="arm", values="sr")
    prow = []
    print("\n=== 逐相位配对差 ===", flush=True)
    for a, b, note in (("R1_l1_dedup", "R0_none", "一级去重 vs 不去重"),
                       ("R2_l2_dedup", "R1_l1_dedup", "二级 − 一级"),
                       ("R3_riskgroup", "R1_l1_dedup", "风险组 − 一级"),
                       ("R3_riskgroup", "R5_none_k4", "风险组 − 不去重k4（持仓匹配）"),
                       ("R4_riskbudget", "R5_none_k4", "风险预算 − 不去重k4（持仓匹配）"),
                       ("R6_rb_cap25", "R0_none", "封顶风险预算 vs 不去重"),
                       ("R6_rb_cap25", "R1_l1_dedup", "封顶风险预算 − 一级")):
        d = (wide[a] - wide[b]).dropna()
        ds = (ws[a] - ws[b]).dropna()
        se = d.std(ddof=1) / np.sqrt(len(d))
        prow.append({"a": a, "b": b, "note": note, "n": len(d),
                     "diff_mean": d.mean(), "diff_se": se,
                     "t_nominal": d.mean() / se, "n_pos": int((d > 0).sum()),
                     "sr_diff": ds.mean()})
        print(f"  {note:<20} {a} − {b}: {d.mean() * 100:+6.2f}pp ±{se * 100:4.2f} "
              f"(名义 t={d.mean() / se:4.1f})  正 {int((d > 0).sum())}/{len(d)}  "
              f"SR差 {ds.mean():+.4f}", flush=True)
    pd.DataFrame(prow).to_csv(OUT / "risk_pairs.csv", index=False,
                              encoding="utf-8-sig")

    print("\n=== 逐年（21 相位均值）===", flush=True)
    print((Y * 100).round(2).to_string(), flush=True)
    print(f"\n产物 → {OUT}（{time.time() - t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()