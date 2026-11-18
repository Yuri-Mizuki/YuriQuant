"""挖两个缺维度：横截面相对量（xs）+ 时序非线性结构（nl）
================================================================

回答「横截面相对量和时序非线性结构的因子可以继续挖吗」——不靠猜，四步实证：

**S1 对照臂（先证明方法论对）**：spearman IC 对单调变换不敏感，
所以 ``mkt_z``（全截面 z-score）的 IC **必须**与原始因子逐位相等。
若不等 ⇒ 后面的数全不可信（列对齐/掩码有问题）。

**S2 横截面相对量**：``ind_diff`` / ``ind_rank`` / ``ind_z``（行业内去均值/
排名/z）。三条路径检验：
  a) 对**已有 95 个因子**做行业相对化，IC 是否提升？
  b) 提升是不是只来自「剔掉行业共同波动」（=降低分母 σ 而非提高分子）？
  c) 新增的 ``nl_*`` 做行业相对化后是否比绝对版更强？

**S3 时序非线性结构**：9 个新因子，逐一看 IC/t，并与全部 95 个旧因子求
**横截面秩相关**，判「是不是旧因子的换皮」。

**S4 增量价值（唯一算数的检验）**：拿新因子做**两级聚合**（锚因子切 Q1 →
Q1 内二级排序，即 §六 建议 1 的正确形态），看能否把单因子 47.21% 的毛收益
推高。**IC 相关低 ≠ 有增量**，必须落到组合层。

口径与 `pit_intraday_probe` 完全一致（PIT 421 池 / 300 在册 / T+1 open 买
T+2 open 卖 / 单边 13.5bp），**不可与定版主链并列**。

用法::

    python -m scripts.evaluation.pit_xs_nl_probe --stage all
    python -m scripts.evaluation.pit_xs_nl_probe --stage ctrl   # 只做对照臂
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.cache import DataCache  # noqa: E402
from data.industry import IndustryClassification  # noqa: E402
from data.offline import OfflineDataSource  # noqa: E402
from factor.intraday_xs_nl import (  # noqa: E402
    NL_DOCS,
    cross_variant,
    extract_nl_all,
)
from scripts.common.cli_common import setup_logging  # noqa: E402
from scripts.evaluation.pit_intraday_probe import (  # noqa: E402
    ONE_WAY_BP,
    OUT,
    build_features,
    ic_stats,
    load_context,
    orient,
    run,
    stats,
)

log = setup_logging("pit_xs_nl_probe")

#: 对照臂容差（spearman IC 理论上逐位相等，实测只允许浮点误差）
CTRL_TOL = 1e-9


def ic_of(p: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """逐日截面 spearman IC（向量化）。

    数学上与 ``for dt: p[dt].corr(fwd[dt], method="spearman")`` 逐位相等，
    但把 1083 次 Python 级调用压成一次 ``rank + corrwith``（实测 0.07s vs 8s）。
    🚨 必须先在**成对非空**掩码下 rank，否则两边各自 rank 会错。
    """
    idx = p.index.intersection(fwd.index)
    cols = p.columns.intersection(fwd.columns)
    a, b = p.loc[idx, cols], fwd.loc[idx, cols]
    m = a.notna() & b.notna()
    ok = m.sum(axis=1) >= 30
    c = a.where(m).rank(axis=1).corrwith(b.where(m).rank(axis=1), axis=1)
    return c.where(ok).dropna()


FEAT_CACHE = OUT / "xs_nl_feats.pkl"


def load_feats(store, days, pool_cols) -> dict[str, pd.DataFrame]:
    """加载（或构建并缓存）95 个旧特征 + 9 个新非线性特征。

    ⚠️ 提取全量特征约 17 分钟，S1~S4 各阶段共用同一份，必须缓存。
    缓存键带上池列数与面板天数，防止换了池/窗口后误用旧缓存。
    """
    key = f"{len(pool_cols)}_{len(days)}_{days[0]}_{days[-1]}"
    meta = FEAT_CACHE.with_suffix(".key")
    if FEAT_CACHE.exists() and meta.exists() and meta.read_text() == key:
        feats = pd.read_pickle(FEAT_CACHE)
        log.info("复用特征缓存：%d 个（%s）", len(feats), key)
        return feats

    log.info("加载旧特征（95 个）...")
    feats = build_features(store, days[0], days[-1], with_lag=False)
    log.info("提取新非线性特征（9 个）...")
    nl = extract_nl_all(store, days[0], days[-1])
    feats.update({k: v.reindex(columns=pool_cols) for k, v in nl.items()})
    feats = {k: v.reindex(columns=pool_cols) for k, v in feats.items()}
    pd.to_pickle(feats, FEAT_CACHE)
    meta.write_text(key)
    log.info("特征已缓存：%d 个 → %s", len(feats), FEAT_CACHE)
    return feats


def panel_nan_ratio(p: pd.DataFrame) -> float:
    return float(p.isna().to_numpy().mean())


def xs_rank_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """两因子的**截面秩相关**均值（逐日 spearman 后再平均）。

    🚨 不能用整体 flatten 算——那会把时序水平差混进截面。
    🚨 必须先取**成对非空**再 rank：直接对各自 rank 再 corrwith 会因两边
    NaN 不同而各自 rank 全体（实测偏差 1.7e-4），必须 ``a.where(m).rank()``。
    """
    idx = a.index.intersection(b.index)
    cols = a.columns.intersection(b.columns)
    a, b = a.loc[idx, cols], b.loc[idx, cols]
    m = a.notna() & b.notna()
    ra = a.where(m).rank(axis=1)
    rb = b.where(m).rank(axis=1)
    c = ra.corrwith(rb, axis=1)
    return float(c.mean()) if len(c) else np.nan


# ----------------------------------------------------------------------
def stage_ctrl(feats: dict[str, pd.DataFrame], fwd: pd.DataFrame,
               ic_tab: pd.DataFrame) -> pd.DataFrame:
    """S1：全截面 z-score 对照臂 —— IC 必须与原始因子逐位相等。"""
    log.info("=" * 70)
    log.info("S1 对照臂：mkt_z（全截面 z-score）IC 必须 == 原始 IC")
    names = list(ic_tab.nlargest(8, "t", keep="all")["name"])
    names += list(ic_tab.nsmallest(4, "t", keep="all")["name"])
    rows = []
    for n in names:
        base = ic_of(feats[n], fwd)
        z = ic_of(cross_variant(feats[n], None, "mkt_z"), fwd)
        d = (base - z).abs()
        rows.append({"name": n, "ic_base": base.mean(), "ic_z": z.mean(),
                     "max_abs_diff": float(d.max()), "n_days": len(base),
                     "pass": bool(d.max() < CTRL_TOL)})
        log.info("  %-28s ic %+.4f vs %+.4f  maxdiff %.2e  %s",
                 n, base.mean(), z.mean(), d.max(),
                 "OK" if d.max() < CTRL_TOL else "🚨 FAIL")
    return pd.DataFrame(rows)


def stage_xs(feats: dict[str, pd.DataFrame], ind: pd.DataFrame,
             fwd: pd.DataFrame, ic_tab: pd.DataFrame) -> pd.DataFrame:
    """S2：已有因子 + 新非线性因子的行业相对化前后 IC 对比。

    🚨 **必须同时看 IC（分子）和 IC_std（分母）**：``t = IC/IC_std·√n``。
    行业去均值会削掉「行业共同波动」这一截，IC 与 IC_std **同时下降**；
    若 IC_std 降得更多，|t| 就会上升，但**组合层的毛收益是按 IC 赚的**，
    IC 变小必然赚得少。只报 |t| 上升 = 把「降噪」误读成「变强」。
    """
    log.info("=" * 70)
    log.info("S2 横截面相对量：行业内去均值 / 排名 / z")
    # 取旧因子里 |t| 最强的 12 个作为代表（全量 95 个跑 4 变体会很慢）
    names = list(ic_tab.reindex(ic_tab["t"].abs().sort_values(
        ascending=False).index)["name"].head(12))
    rows = []
    for n in names:
        p = feats[n]
        base = ic_of(p, fwd)
        sb = ic_stats(base)
        r = {"name": n, "ic": sb.get("ic", np.nan), "ic_std": sb.get("ic_std", np.nan),
             "t": sb.get("t", np.nan), "nan_ratio": panel_nan_ratio(p)}
        for how in ("ind_diff", "ind_rank", "ind_z"):
            s = ic_stats(ic_of(cross_variant(p, ind, how), fwd))
            r[f"ic_{how}"] = s.get("ic", np.nan)
            r[f"std_{how}"] = s.get("ic_std", np.nan)
            r[f"t_{how}"] = s.get("t", np.nan)
            r[f"nan_{how}"] = panel_nan_ratio(cross_variant(p, ind, how))
        # 分解：|t| 的变化来自分子还是分母
        r["d_ic_pct"] = (abs(r["ic_ind_diff"]) / abs(r["ic"]) - 1) * 100
        r["d_std_pct"] = (r["std_ind_diff"] / r["ic_std"] - 1) * 100
        r["d_t_pct"] = (abs(r["t_ind_diff"]) / abs(r["t"]) - 1) * 100
        rows.append(r)
        log.info("  %-24s ic %+.4f→%+.4f (%+5.1f%%)  std %.4f→%.4f (%+5.1f%%)  "
                 "|t| %+.2f→%+.2f (%+5.1f%%)",
                 n, r["ic"], r["ic_ind_diff"], r["d_ic_pct"],
                 r["ic_std"], r["std_ind_diff"], r["d_std_pct"],
                 r["t"], r["t_ind_diff"], r["d_t_pct"])
    return pd.DataFrame(rows)


def stage_nl(feats: dict[str, pd.DataFrame], ind: pd.DataFrame,
             fwd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S3：9 个非线性因子的 IC + 与全部旧因子的最大截面秩相关。"""
    log.info("=" * 70)
    log.info("S3 时序非线性结构：%d 个新因子", len(NL_DOCS))
    old = {k: v for k, v in feats.items() if not k.startswith("nl_")}
    rows, corr_rows = [], []
    for n in NL_DOCS:
        if n not in feats:
            continue
        p = feats[n]
        base = ic_of(p, fwd)
        s = ic_stats(base)
        v = cross_variant(p, ind, "ind_diff")
        s_ind = ic_stats(ic_of(v, fwd))
        # 与旧因子的截面秩相关（取旧因子里 |t| 最强 40 个，省时且最有威胁）
        cors = {m: xs_rank_corr(p, q) for m, q in old.items()}
        cs = pd.Series(cors).dropna()
        top = cs.abs().nlargest(3)
        rows.append({
            "name": n, "doc": NL_DOCS[n], "ic": s.get("ic", np.nan),
            "t": s.get("t", np.nan), "win": s.get("win", np.nan),
            "n": s.get("n", 0), "nan_ratio": panel_nan_ratio(p),
            "ic_ind": s_ind.get("ic", np.nan), "t_ind": s_ind.get("t", np.nan),
            "max_abs_corr": float(cs.abs().max()) if len(cs) else np.nan,
            "argmax_corr": top.index[0] if len(top) else "",
        })
        corr_rows.append(pd.Series(cs, name=n))
        log.info("  %-24s ic %+.4f t %+.2f (ind t %+.2f) | 最高相关 %.3f (%s)",
                 n, s.get("ic", np.nan), s.get("t", np.nan),
                 s_ind.get("t", np.nan),
                 rows[-1]["max_abs_corr"], rows[-1]["argmax_corr"])
    return pd.DataFrame(rows), pd.DataFrame(corr_rows)


def stage_portfolio_xs(feats: dict[str, pd.DataFrame], ind: pd.DataFrame,
                       fwd: pd.DataFrame, names: list[str],
                       top: int = 10) -> pd.DataFrame:
    """S2b：行业相对化在**组合层**是赚还是亏（|t| 升而 |IC| 降时的唯一判据）。

    IC 只说明「排序对不对」，组合层才说明「赚不赚」。行业去均值削掉行业共同
    波动后 IC 变小，如果毛收益也变小，那 |t| 上升只是降噪的副产品、**没有经济价值**。
    """
    log.info("=" * 70)
    log.info("S2b 组合层：行业相对化 vs 原始（top%d）", top)
    rows = []
    for n in names:
        s = ic_stats(ic_of(feats[n], fwd))
        for tag, p in (("raw", feats[n]),
                       ("ind_diff", cross_variant(feats[n], ind, "ind_diff"))):
            sig = orient(p, s["ic"])
            r = run(sig, fwd, top=top, cost_bp=ONE_WAY_BP)
            gs, ns = stats(r["gross"]), stats(r["net"])
            rows.append({"name": n, "variant": tag,
                         "gross": gs.get("ann", np.nan),
                         "net": ns.get("ann", np.nan),
                         "sr": ns.get("sr", np.nan),
                         "mdd": ns.get("mdd", np.nan),
                         "turn": float(r["turn"].mean() * 252)})
        a, b = rows[-2], rows[-1]
        log.info("  %-24s 毛 %+6.2f%% → %+6.2f%% (%+.2fpp)  净 %+6.2f%% → %+6.2f%%",
                 n, a["gross"] * 100, b["gross"] * 100,
                 (b["gross"] - a["gross"]) * 100, a["net"] * 100, b["net"] * 100)
    return pd.DataFrame(rows)


def stage_incremental(feats: dict[str, pd.DataFrame], fwd: pd.DataFrame,
                      ic_tab: pd.DataFrame, nl_tab: pd.DataFrame,
                      anchor: str = "mom_last_bar", top: int = 10,
                      q: float = 0.1) -> pd.DataFrame:
    """S4：增量价值 —— 新因子能否把单因子 47% 的毛收益推高（两级聚合）。"""
    log.info("=" * 70)
    log.info("S4 增量：锚因子 %s 切 Q1 → Q1 内用新因子二级排序", anchor)
    # 🚨 ics 必须**从 feats 现算**，不能用旧 IC 表：缓存表来自
    # `pit_intraday_probe`，不含本轮新增的 nl_*，会 KeyError（踩过）。
    # 且定向符号必须与本次数据一致——用别的窗口的 IC 符号会整体反向。
    ics: dict[str, float] = {}
    for n in feats:
        s = ic_stats(ic_of(feats[n], fwd))
        if s:
            ics[n] = float(s["ic"])
    log.info("  现算 IC %d 个（含 nl_*），锚因子 %s ic %+.4f",
             len(ics), anchor, ics.get(anchor, np.nan))
    rows = []

    def add(tag: str, sig: pd.DataFrame) -> None:
        r = run(sig, fwd, top=top, cost_bp=ONE_WAY_BP)
        g, net, turn = r["gross"], r["net"], r["turn"]
        gs, ns = stats(g), stats(net)
        rows.append({"arm": tag, "gross": gs.get("ann", np.nan),
                     "net": ns.get("ann", np.nan), "sr": ns.get("sr", np.nan),
                     "mdd": ns.get("mdd", np.nan),
                     "turn": float(turn.mean() * 252) if len(turn) else np.nan})
        log.info("  %-22s 毛 %+7.2f%%  净 %+7.2f%%  SR %+.3f  换手 %.1f",
                 tag, rows[-1]["gross"] * 100, rows[-1]["net"] * 100,
                 rows[-1]["sr"] or 0.0, rows[-1]["turn"] or 0.0)

    # 基线：单因子全截面
    add(f"single_{anchor}", orient(feats[anchor], ics[anchor]))

    # 锚因子 gate（无二级排序）—— 与 single 应等价，确认 gate 没走偏
    from scripts.evaluation.pit_intraday_probe import composite_2stage
    add(f"gate_only_q{q:g}", composite_2stage(anchor, [], feats, ics, q=q))

    # 各新因子单独进二级排序
    nl_names = [n for n in nl_tab["name"] if n in feats]
    for n in nl_names:
        add(f"gate+{n.replace('nl_', '')}", composite_2stage(anchor, [n], feats, ics, q=q))

    # 全部新因子一起
    if len(nl_names) > 1:
        add("gate+all_nl", composite_2stage(anchor, nl_names, feats, ics, q=q))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="横截面相对量 + 时序非线性结构探针")
    ap.add_argument("--stage", default="all",
                    choices=["all", "ctrl", "xs", "nl", "incr"])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--begin", type=int, default=20220101)
    ap.add_argument("--end", type=int, default=20260630)
    ap.add_argument("--anchor", default="mom_last_bar")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    store, codes, _daily, fwd = load_context(args.begin, args.end)
    days = store.all_days()
    pool_cols = [c for c in store.codes if c in fwd.columns]
    feats = load_feats(store, days, pool_cols)

    ic_cache = OUT / "pit_ic_table.parquet"
    ic_tab = pd.read_parquet(ic_cache)
    # 🚨 缓存表含 190 个 lag 因子（with_lag=True 时生成），本轮 feats 只有当日版
    # （lag 已证伪，§6.4）。必须按 feats 的键过滤，否则 KeyError 或更糟：
    # 混进 lag 因子让「最强因子」排行榜被伪维度污染。
    ic_tab = ic_tab[ic_tab["name"].isin(feats)].reset_index(drop=True)
    log.info("旧 IC 表 %d 个因子（已对齐当日版，剔除 lag）", len(ic_tab))

    # ---- S1 对照臂 ----
    ctrl = stage_ctrl(feats, fwd, ic_tab)
    ctrl.to_csv(OUT / "xs_nl_ctrl.csv", index=False, encoding="utf-8-sig")
    if not bool(ctrl["pass"].all()):
        log.error("🚨 对照臂失败：mkt_z 的 IC 与原始不等 ⇒ 后续结论不可信，中止")
        return

    # ---- S2 横截面相对量 ----
    cache = DataCache(OfflineDataSource())
    ind = IndustryClassification(cache, level=1).get_industry_panel(pool_cols,
                                                                   fwd.index)
    log.info("行业面板 %s，覆盖率 %.1f%%，行业数 %d", ind.shape,
             float(ind.notna().to_numpy().mean()) * 100, ind.nunique().max())
    xs = stage_xs(feats, ind, fwd, ic_tab)
    xs.to_csv(OUT / "xs_nl_xs.csv", index=False, encoding="utf-8-sig")

    # ---- S3 非线性 ----
    nl_tab, nl_corr = stage_nl(feats, ind, fwd)
    nl_tab.to_csv(OUT / "xs_nl_nl.csv", index=False, encoding="utf-8-sig")
    nl_corr.to_csv(OUT / "xs_nl_nlcorr.csv", encoding="utf-8-sig")

    # ---- S2b 行业相对化的组合层代价 ----
    port_names = ["mom_last_bar", "mom_seg12_ret", "pv_zero_ret_share",
                  "evt_down_tail_share"]
    port = stage_portfolio_xs(feats, ind, fwd, port_names, top=args.top)
    port.to_csv(OUT / "xs_nl_port.csv", index=False, encoding="utf-8-sig")

    # ---- S4 增量 ----
    incr = stage_incremental(feats, fwd, ic_tab, nl_tab,
                             anchor=args.anchor, top=args.top, q=args.q)
    incr.to_csv(OUT / "xs_nl_incr.csv", index=False, encoding="utf-8-sig")

    log.info("=" * 70)
    log.info("非线性显著（|t|>1.96）%d / %d",
             int((nl_tab["t"].abs() > 1.96).sum()), len(nl_tab))
    log.info("行业相对化后显著 %d", int((nl_tab["t_ind"].abs() > 1.96).sum()))
    log.info("产物 %s", OUT)


if __name__ == "__main__":
    main()