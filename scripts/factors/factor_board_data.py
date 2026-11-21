"""因子看板数据层：为因子库里每个因子算「分位组合净值 + IC 序列 + 行业 IC」。

口径（与生产一致，不另造一套）
------------------------------
- 组合：每月首个交易日收盘出信号 → **T+1 开盘**成交（``build_execution_split``），
  组内等权；分位组 = 截面秩的 20% 切分（``Q1`` 最小分位 / ``Q5`` 最大分位）。
- 成本：``backtest.costs.default_costs()``（生产口径，含 **2023-08-28 时变印花税**）；
  另跑一遍 ``factor_cost=False`` 的无成本路径，供页面「手续费」开关对照。
- 业绩：直接走 :class:`backtest.engine.VectorBacktest`，**不自己写回测循环**——
  与 ``calibration`` 主链同源，避免「看板数字和定版数字对不上」。
- IC：截面 spearman(因子, 次一期收益)，收益与库评估同一张面板
  （``FactorLibrary.default_returns_panel()``，复权收盘 1 日远期）。
- 价格 / 掩码 / 基准：收盘、开盘、可交易掩码、全A等权基准全部从
  ``FactorLibrary`` 取（2026-11-19 起不再读 ``reports/alla_rolling/_base``
  快照）；``_base`` 只剩行业分类一个用途。

产物（默认 ``reports/factor_board/cache/``）
--------------------------------------------
    <name>.parquet   日频 nav_q1/q5（净/毛）+ ic 序列
    <name>.json      换手序列、行业 IC、各窗口指标（页面直接读）
    _summary.csv     全因子横向指标（列表页用）

用法::

    python -m scripts.factors.factor_board_data --workers 6
    python -m scripts.factors.factor_board_data --names mom20,rev1 --limit 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# 多进程 + OpenBLAS 默认线程数会在每个子进程里各自开一堆缓冲区，
# 8 进程时实测直接 "Memory allocation still failed" 把整池打断（2026-11-17）。
# 本项目全程不依赖 BLAS 多线程（pandas 逐行运算），锁 1 线程既省内存也避免抢核。
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest.costs import default_costs  # noqa: E402
from backtest.engine import VectorBacktest, build_execution_split  # noqa: E402
from backtest.metrics import drawdown_series, sharpe_arithmetic  # noqa: E402
from config import Config  # noqa: E402
from scripts.evaluation.caliber import next_day_rebalance, rebalance_dates  # noqa: E402
from stats import PERIODS_PER_YEAR  # noqa: E402
from stats.ic import calc_ic_series, ic_stats  # noqa: E402
from strategy.base import Strategy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "alla_rolling" / "_base"
OUT = ROOT / "reports" / "factor_board" / "cache"
START = "2019-01-01"
WINDOWS = {"1y": 1, "3y": 3, "5y": 5, "all": None}
PPY = PERIODS_PER_YEAR  # 年化口径唯一真源（禁止再写 244/252 字面量）
MIN_NAMES = 100      # 当日有效样本少于此值不建组合
MIN_UNIQUE = 10      # 截面有效取值少于 10 种 ⇒ 分位切不开（见下）


class _Group(Strategy):
    """分位组合：``side='top'`` 取最高 20%，``'bottom'`` 取最低 20%。"""

    def __init__(self, side: str, q: float = 0.2):
        self.side = side
        self.q = q
        self.name = f"quantile_{side}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        v = factor_values.dropna()
        if len(v) < MIN_NAMES:
            return pd.Series(dtype=float)
        r = v.rank(pct=True, method="average")
        keep = r[r > 1.0 - self.q].index if self.side == "top" else r[r <= self.q].index
        if len(keep) == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(keep), index=keep)


# ---------------------------------------------------------------------------
# 共享上下文（每进程只加载一次）
# ---------------------------------------------------------------------------
CTX: dict = {}


def _init(lib_root: str, dataset: str, base: Path, panel_dir: Path) -> None:
    """加载共享上下文：价格/收益/基准全部走库（`FactorLibrary`），不再读 _base 快照。

    2026-11-19 统一：``_base/close_adj.parquet`` 是 ``rolling_grid_alla --stage prep``
    的**快照**，与库 ``default_returns_panel()`` 同源但时点可能落后，导致看板
    净值与 registry 指标漂移。现在收盘/开盘/掩码/基准都从 ``FactorLibrary`` 取，
    只有行业分类仍读 ``_base/industry.parquet``（库不提供行业面板）。
    """
    from research.factor_library import FactorLibrary

    lib = FactorLibrary(root=Path(lib_root), dataset=dataset)
    reg_all = lib.list_all()
    panel_paths: dict[str, str] = {}
    if "panel_path" in reg_all.columns:
        for nm, pp in zip(reg_all["name"], reg_all["panel_path"]):
            if isinstance(pp, str) and pp.strip():
                panel_paths[str(nm)] = pp
    close = lib.adjusted_close_panel()
    if close is None:
        raise RuntimeError("复权收盘不可用（缺 backward_factor / daily_all_a）")
    close = close.loc[close.index >= pd.Timestamp(START)]
    fill = lib.adjusted_open_panel()
    if fill is None:
        raise RuntimeError("复权开盘不可用（T+1 开盘执行口径需要 open×后复权）")
    fill = fill.reindex(index=close.index, columns=close.columns)
    # 信号日 T 收盘出信号、T+1 开盘成交 ⇒ 掩码取执行日状态（execution_lag=1）
    mask = lib.tradable_mask(1)
    if mask is None:
        mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    else:
        mask = mask.reindex(index=close.index, columns=close.columns).fillna(True)
    ind = pd.read_parquet(base / "industry.parquet").reindex(
        index=close.index, columns=close.columns)
    # 价格指数口径（T 行 = T-1→T），与画基准净值同向
    bench = lib.benchmark_eqw_returns(forward=False)
    bench = (bench.replace([np.inf, -np.inf], np.nan).dropna()
             .loc[lambda s: s.index >= pd.Timestamp(START)])
    bench_nav = (1 + bench).cumprod()

    rb_sig = set(rebalance_dates(close.index))
    rb_exec = next_day_rebalance(close.index)
    split = build_execution_split(fill, close, rb_sig)

    # 价格面板里的 0 / 停牌残值会让 pct_change 产出 ±inf，引擎的 nansum 会把 inf
    # 直接乘进净值（实测 alpha158_KLEN 的 q5 净值某日跳到 inf）。这里把非有限收益
    # 归零、单日 |r|>300% 视为脏值归零——只影响极少数异常点，不改变排序结论。
    def _raw(df: pd.DataFrame) -> pd.DataFrame:
        v = df.replace([np.inf, -np.inf], np.nan)
        return v.where(v.abs() <= 3.0)

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        return _raw(df).fillna(0.0)

    CTX.update(close=close, fill=fill, mask=mask, ind=ind, panel_dir=str(panel_dir),
               panel_paths=panel_paths,
               bench_ret=bench, bench_nav=bench_nav,
               rcc=_clean(close.pct_change(fill_method=None)),
               rcc_ic=_raw(close.pct_change(fill_method=None)),
               split=(_clean(split[0]), _clean(split[1])), rb_exec=rb_exec,
               costs=default_costs(), costs_off=default_costs(factor_cost=False))


# ---------------------------------------------------------------------------
# 单因子计算
# ---------------------------------------------------------------------------
def _industry_ic(panel: pd.DataFrame, fwd: pd.DataFrame, ind: pd.DataFrame,
                 dates: pd.DatetimeIndex, n_sample: int = 36) -> dict:
    """行业 IC：按月抽样，行业内 spearman IC 的时间均值。"""
    pick = pd.DatetimeIndex(dates).to_series().groupby(
        pd.DatetimeIndex(dates).to_period("M")).last().values
    pick = pick[-n_sample:]
    rows: dict[str, list[float]] = {}
    for d in pick:
        if d not in panel.index:
            continue
        f = panel.loc[d]
        r = fwd.loc[d]
        g = ind.loc[d]
        ok = f.notna() & r.notna() & g.notna()
        if ok.sum() < MIN_NAMES:
            continue
        fr = f[ok].rank()
        rr = r[ok].rank()
        gg = g[ok]
        for name, idx in gg.groupby(gg).groups.items():
            if len(idx) < 8:
                continue
            aa, bb = fr.loc[idx], rr.loc[idx]
            if aa.std() == 0 or bb.std() == 0:
                continue
            rows.setdefault(str(name), []).append(float(aa.corr(bb)))
    return {k: float(np.mean(v)) for k, v in rows.items() if v}


def _window_metrics(nav: pd.Series, bench_nav: pd.Series,
                    turnover: pd.Series) -> dict:
    out = {}
    for wname, yrs in WINDOWS.items():
        if yrs is None:
            seg = nav
            bseg = bench_nav.reindex(nav.index).ffill()
        else:
            cut = nav.index[-1] - pd.DateOffset(years=yrs)
            seg = nav.loc[nav.index >= cut]
            bseg = bench_nav.reindex(seg.index).ffill()
            bseg = bseg / bseg.iloc[0] * seg.iloc[0]
        if len(seg) < 20:
            continue
        r = seg.pct_change().dropna()
        n = len(r)
        cum = float(seg.iloc[-1] / seg.iloc[0] - 1)
        ann = float((seg.iloc[-1] / seg.iloc[0]) ** (PPY / n) - 1)
        bann = float((bseg.iloc[-1] / bseg.iloc[0]) ** (PPY / n) - 1)
        dd = drawdown_series(seg)
        to = turnover.loc[turnover.index >= seg.index[0]]
        out[wname] = {
            "cum": cum, "ann": ann, "bench_ann": bann,
            "excess_ann": float((1 + ann) / (1 + bann) - 1),
            "mdd": float(dd.min()),
            "sharpe": float(sharpe_arithmetic(r, PPY)) if r.std() else None,
            "turnover": float(to.mean() * 12) if len(to) else None,
            "n_days": n,
        }
    return out


def _ic_metrics(ic: pd.Series) -> dict:
    out = {}
    for wname, yrs in WINDOWS.items():
        if yrs is None:
            seg = ic
        else:
            cut = ic.index[-1] - pd.DateOffset(years=yrs)
            seg = ic.loc[ic.index >= cut]
        seg = seg.dropna()
        if len(seg) < 20:
            continue
        s = ic_stats(seg)
        out[wname] = {
            "ic_mean": s["ic"],
            "ic_ir": s["ir"],
            "ic_gt": float((seg.abs() > 0.02).mean()),
            "ic_win": s["win"],
            "ic_t": s["nw_t"],
        }
    return out


def _unique_levels(panel: pd.DataFrame, n_sample: int = 40) -> float:
    """抽样估计「每日截面有效取值数」的中位数。

    🚨 2026-11-17 实测：库里有面板**截面只有 5 个取值**（如 `alpha101_001`，
    4424 只里 2234 只并列同一个值）——秩分位切 20% 时两端根本切不出来，
    引擎里 `rank(pct)>0.8` 会是空集 ⇒ 组合变空仓、换手在高频 0 与 0.5 之间跳。
    这类因子必须**显式标「分位不可分」**，不能给一个看似正常的业绩数字。
    """
    idx = panel.index[panel.notna().any(axis=1)]
    if len(idx) == 0:
        return 0.0
    pick = idx[:: max(1, len(idx) // n_sample)][:n_sample]
    vals = [int(panel.loc[d].nunique()) for d in pick]
    return float(np.median(vals)) if vals else 0.0


def compute_one(name: str) -> dict | None:
    # 面板路径以 registry 的 panel_path 为准（register 通道落的是 slug 文件名，
    # 不是 <name>.parquet）；旧 builder 面板没有 panel_path 时回退到 <name>.parquet。
    p = CTX["panel_paths"].get(name)
    panel_path = Path(p) if p else Path(CTX["panel_dir"]) / f"{name}.parquet"
    if not panel_path.exists():
        return None
    close, mask = CTX["close"], CTX["mask"]
    panel = pd.read_parquet(panel_path)
    codes = panel.columns.intersection(close.columns)
    if len(codes) < MIN_NAMES:
        return None
    panel = panel.reindex(index=close.index, columns=codes)
    if panel.notna().sum(axis=1).max() < MIN_NAMES:
        return None

    m = mask.reindex(columns=codes)
    rcc_ic = CTX["rcc_ic"].reindex(columns=codes)
    fwd = rcc_ic.shift(-1)
    ic = calc_ic_series(panel, fwd, tradable=m, min_samples=100)
    n_uniq = _unique_levels(panel)
    ind_ic = _industry_ic(panel, fwd, CTX["ind"].reindex(columns=codes), panel.index)

    base = {"name": name, "start": str(panel.index[0].date()),
            "end": str(panel.index[-1].date()), "n_days": int(len(panel)),
            "unique_levels": n_uniq, "degenerate": bool(n_uniq < MIN_UNIQUE),
            "ic": _ic_metrics(ic), "industry_ic": ind_ic}

    if base["degenerate"]:
        # 分位切不开 ⇒ 只出 IC，绝不给「看起来正常」的分位业绩
        base.update({"metrics": {}, "turnover": {}})
        return {"name": name, "nav": pd.DataFrame({"ic": ic.reindex(panel.index)}),
                "meta": base}

    sig = panel.shift(1).where(m)
    rcc = CTX["rcc"].reindex(columns=codes)
    sp = (CTX["split"][0].reindex(columns=codes),
          CTX["split"][1].reindex(columns=codes))
    res = {}
    for side in ("bottom", "top"):
        for tag, costs in (("net", CTX["costs"]), ("gross", CTX["costs_off"])):
            bt = VectorBacktest(_Group(side), rebalance_freq="M", costs=costs)
            res[(side, tag)] = bt.run(sig, rcc, horizon=1, rebalance_days=CTX["rb_exec"],
                                      execution_split=sp, executable_mask=m)

    nav = pd.DataFrame({
        "nav_q1_net": (1 + res[("bottom", "net")].daily_returns).cumprod(),
        "nav_q5_net": (1 + res[("top", "net")].daily_returns).cumprod(),
        "nav_q1_gross": (1 + res[("bottom", "gross")].daily_returns).cumprod(),
        "nav_q5_gross": (1 + res[("top", "gross")].daily_returns).cumprod(),
    })
    nav["ic"] = ic.reindex(nav.index)

    bench_nav = CTX["bench_nav"].reindex(nav.index).ffill()
    to_q1 = res[("bottom", "net")].turnover_series.dropna()
    to_q5 = res[("top", "net")].turnover_series.dropna()
    base.update({
        "n_days": int(len(nav)),
        "metrics": {
            "q1_net": _window_metrics(nav["nav_q1_net"], bench_nav, to_q1),
            "q5_net": _window_metrics(nav["nav_q5_net"], bench_nav, to_q5),
            "q1_gross": _window_metrics(nav["nav_q1_gross"], bench_nav, to_q1),
            "q5_gross": _window_metrics(nav["nav_q5_gross"], bench_nav, to_q5),
        },
        "turnover": {
            "q1": {"dates": [str(d.date()) for d in to_q1.index],
                   "values": [float(x) for x in to_q1.values]},
            "q5": {"dates": [str(d.date()) for d in to_q5.index],
                   "values": [float(x) for x in to_q5.values]},
        },
    })
    return {"name": name, "nav": nav, "meta": base}


def _summary_row(meta: dict, reg: pd.Series) -> dict:
    row = {"name": meta["name"], "family": reg.get("family"),
           "degenerate": meta.get("degenerate"), "unique_levels": meta.get("unique_levels"),
           "source": str(reg.get("source", "")).split(":")[0],
           "label": reg.get("label"), "formula": reg.get("formula"),
           "coverage": reg.get("coverage"), "n_days": meta["n_days"],
           "start": meta["start"], "end": meta["end"]}
    for key, block in meta["metrics"].items():
        for w, m in block.items():
            for k, v in m.items():
                row[f"{key}_{w}_{k}"] = v
    for w, m in meta["ic"].items():
        for k, v in m.items():
            row[f"ic_{w}_{k}"] = v
    if meta["industry_ic"]:
        vals = list(meta["industry_ic"].values())
        row["ind_ic_max"] = float(np.max(vals))
        row["ind_ic_min"] = float(np.min(vals))
    return row


def _scan_one(job) -> tuple[str, float | None]:
    """扫描单个面板的截面取值数（模块级，供进程池 pickle）。"""
    name, panel_dir = job
    p = Path(panel_dir) / f"{name}.parquet"
    if not p.exists():
        return name, None
    return name, _unique_levels(pd.read_parquet(p))


def _scan_degenerate(names, out, reg, workers, base, panel_dir,
                     lib_root, dataset) -> None:
    """扫描全库面板的截面取值数；对「切不开分位」的因子按新口径重算。"""
    flagged = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        jobs = [(n, str(panel_dir)) for n in names]
        for n, uq in ex.map(_scan_one, jobs, chunksize=4):
            if uq is not None and uq < MIN_UNIQUE:
                flagged.append((n, uq))
    print(f"截面取值 < {MIN_UNIQUE} 的因子：{len(flagged)} 个")
    for n, uq in flagged[:40]:
        print(f"  {n}  unique={uq:.0f}")
    (out / "_degenerate.txt").write_text(
        "\n".join(f"{n}\t{uq:.0f}" for n, uq in flagged), encoding="utf-8")

    if not flagged:
        return
    _init(lib_root, dataset, base, panel_dir)
    for n, _ in flagged:
        r = compute_one(n)
        if r is None:
            continue
        r["nav"].to_parquet(out / f"{n}.parquet")
        (out / f"{n}.json").write_text(json.dumps(r["meta"], ensure_ascii=False),
                                       encoding="utf-8")
    print(f"已按「分位不可分」口径重算 {len(flagged)} 个")


def main() -> None:
    ap = argparse.ArgumentParser(description="因子看板数据层：分位组合净值 + IC + 行业 IC")
    ap.add_argument("--root", default=str(
        Path(Config.paths()["factor_library_root"]) / "all_a_2018_2026"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--base", default=str(BASE),
                    help="行业分类快照目录（只读取 industry.parquet）")
    ap.add_argument("--names", default=None, help="逗号分隔的因子名（缺省 = 库内全部）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--resume", action="store_true", help="跳过已有缓存")
    ap.add_argument("--scan-unique", action="store_true",
                    help="只扫描面板截面取值数，列出「分位不可分」因子后重算它们")
    args = ap.parse_args()

    root = Path(args.root)
    lib_root, dataset = str(root.parent), root.name
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(root / "registry.csv").set_index("name", drop=False)

    names = [n.strip() for n in args.names.split(",")] if args.names else list(reg.index)
    names = [n for n in names if n in reg.index]
    if args.resume:
        names = [n for n in names if not (out / f"{n}.parquet").exists()]
    if args.limit:
        names = names[: args.limit]

    _init(lib_root, dataset, Path(args.base), root / "panels")
    print(f"因子 {len(names)} 个 · 日期 {CTX['close'].index[0].date()} ~ "
          f"{CTX['close'].index[-1].date()} · 进程 {args.workers}")

    if args.scan_unique:
        _scan_degenerate(names, out, reg, args.workers, Path(args.base),
                         root / "panels", lib_root, dataset)
        return

    rows, done, failed = [], 0, []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(lib_root, dataset, Path(args.base),
                                       root / "panels")) as ex:
        futs = {ex.submit(compute_one, n): n for n in names}
        for fut in as_completed(futs):
            n = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:                      # noqa: BLE001
                failed.append((n, repr(exc)[:120]))
                print(f"  ✗ {n}: {exc!r}"[:160])
                continue
            if r is None:
                continue
            r["nav"].to_parquet(out / f"{n}.parquet")
            (out / f"{n}.json").write_text(
                json.dumps(r["meta"], ensure_ascii=False), encoding="utf-8")
            rows.append(_summary_row(r["meta"], reg.loc[n]))
            done += 1
            if done % 25 == 0:
                print(f"  ... {done}/{len(names)}")

    if rows:
        df = pd.DataFrame(rows)
        path = out / "_summary.csv"
        if path.exists() and args.resume:
            old = pd.read_csv(path)
            df = pd.concat([old[~old["name"].isin(df["name"])], df])
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"汇总 {len(df)} 行 → {path}")
    if failed:
        # 2026-11-21：失败必须反映在 rc 上——此前全失败仍 rc=0，下游
        # factor_board 拿空 summary 才以 KeyError 暴雷（当天实战踩了两次）。
        print(f"失败 {len(failed)} 个：{failed[:5]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
