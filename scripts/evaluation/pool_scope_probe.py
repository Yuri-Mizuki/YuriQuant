"""股票池范围探针 —— 同一选股信号在 hs300 / zz500 / zz1000 / zz2000 上的业绩。

问题（2026-11-05 由莉酱）：全A 920 策略放到四个指数池上分别是什么水平？

口径（必须显式，见 `scripts/evaluation/caliber.py`）
--------------------------------------------------
* 信号：``--signal``（默认 ``ens_h1h5`` = 882 ortho，即本机工作基线 v1w-ortho882）。
  🚨 **本机没有 920 fundind 的 pred**（``dingban_fundind`` 的落盘面板缺失），
  本探针跑的是**代理信号**，绝对数字不可与 v1-dingban 并列；池子之间的**相对**
  结论可以迁移，但需注明代理关系。
* 执行层：``BufferedTopFracLongOnly(0.15, 0.40)`` = 生产定版形态。
* 口径组合：``production``（固定 10bp / T+1 open / 1e6）。
* 池子过滤方式：把 pred 面板**池外置 NaN**（``signal.where(mask)``），于是
  ``BufferedTopFracLongOnly`` 的 TopN = ``0.15 × 池内有效截面数``，自动随池子缩放。
  ⚠️ 四臂的**持仓数不同**（hs300≈45 / zz500≈75 / zz1000≈150 / zz2000≈300 /
  全A≈640）——比较的是「池子 + 分散度」的联合效果，不是纯池子效应。
* **同池等权基准**（``bench_*`` 列）：池内等权、与策略同频（月首调仓）持有，
  零成本。策略 **只有跑赢这一列才算该池内有选股 alpha**；绝对收益的池子间差异
  大部分是 β（大盘 vs 小盘），不能直接横向比。

用法
----
    python -m scripts.evaluation.pool_scope_probe --begin 20190101 \
        --pools all_a,hs300,zz500,zz1000 --out reports/pool_scope_probe/by_pool_2019.csv
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

from backtest.engine import VectorBacktest  # noqa: E402
from backtest.metrics import (  # noqa: E402
    annual_return,
    annual_volatility,
    max_drawdown,
    sharpe_ratio,
)
from config import Config  # noqa: E402
from data.universe import Universe  # noqa: E402
from scripts.evaluation.caliber import (  # noqa: E402
    _metrics,          # 汇总指标唯一实现（含 NW t / years_to_prove），避免另写一份
    cost_profile,
    describe,
    get_profile,
    load_context,
    rebalance_dates,
    run_once,
)
from strategy.cross_section import BufferedTopFracLongOnly, _rank_desc  # noqa: E402
from strategy.base import Strategy  # noqa: E402


class BufferedFixedNLongOnly(Strategy):
    """缓冲带 + **固定持仓数**（与 ``BufferedTopFracLongOnly`` 同规则，只把
    ``n_entry = frac·n`` 换成常数）。

    存在理由：池子臂若各按 15% 取，全A 持 ~644 只、hs300 只持 45 只 ⇒
    池子效应与分散度效应混在一起。固定 N 臂把分散度锁死，只留池子差异。
    ``n_exit`` 缺省按生产带宽 ``0.40/0.15 ≈ 2.67×`` 推。
    """

    def __init__(self, n_entry: int, n_exit: int | None = None):
        self.n_entry = max(1, int(n_entry))
        self.n_exit = max(self.n_entry, int(n_exit) if n_exit
                          else int(round(self.n_entry * 0.40 / 0.15)))
        self._prev: set = set()
        self.name = f"buffered_fixedN_{self.n_entry}_{self.n_exit}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if len(vals) == 0:
            self._prev = set()
            return pd.Series(dtype=float)
        ranks = _rank_desc(vals)
        keep = [c for c in self._prev if c in ranks.index and ranks[c] <= self.n_exit]
        candidates = [c for c in ranks.sort_values().index
                      if c not in set(keep) and c not in self._prev
                      and ranks[c] <= self.n_entry]
        portfolio = keep + candidates[:max(0, self.n_entry - len(keep))]
        if len(portfolio) > self.n_entry:
            portfolio = sorted(portfolio, key=lambda c: ranks[c])[:self.n_entry]
        if not portfolio:
            self._prev = set()
            return pd.Series(dtype=float)
        self._prev = set(portfolio)
        return pd.Series(1.0 / len(portfolio), index=portfolio)

#: 指数代码（AmazingData 口径）。``zz2000`` 用 ``932000.CSI``（中证2000 官方代码）。
POOLS: dict[str, str] = {
    "hs300": "000300.SH",
    "zz500": "000905.SH",
    "zz1000": "000852.SH",
    "zz2000": "932000.CSI",
}


class _LocalOnlyCache:
    """只读本地 parquet 的成分表访问器。

    不走 ``DataCache._refresh_pit_event_table``：那张表按 ``_meta.json`` 的
    ``refreshed_on`` 判过期，手动落盘的表不在 meta 里 ⇒ 每次调用都尝试回源
    （离线桩返回空表 + 一条告警）。本探针只需读，直接读文件更快也更安静。
    """

    def __init__(self) -> None:
        root = str(Config.paths()["cache_root"]).replace("//", "/")
        self._root = Path(root)

    def get_index_constituent(self, index_code: str) -> pd.DataFrame:
        p = self._root / f"index_constituent_{index_code.replace('.', '_')}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"成分表缺失：{p}\n先跑一次 "
                f"DataCache(create_datasource()).get_index_constituent({index_code!r}) 落盘。")
        return pd.read_parquet(p)


def _pool_mask(index_code: str, dates: pd.DatetimeIndex,
               columns: pd.Index) -> pd.DataFrame:
    """按日 PIT 归属矩阵（date × code，bool），对齐到信号面板的列。"""
    uni = Universe(_LocalOnlyCache())
    m = uni.get_membership_mask(index_code, dates)
    out = m.reindex(index=dates, columns=columns).astype("boolean").fillna(False)
    return out.astype(bool)


def _bench_returns(fwd: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    """池内等权、月首调仓、零成本的日收益序列（β 基准）。

    段内权重按价格漂移（买入后不动），不做事后再平衡；下一个月首重置为等权。
    与策略臂同频（``caliber.rebalance_dates`` = 每月首个交易日）⇒ 可直接比。

    🚨 2026-11-05 修复：调仓日权重**只给当日有收益数据的股票**。
    此前用 ``mask`` 全域归一，而面板列（5801）远多于当时有价的股票
    （2018 年仅 3134 只）⇒ 42% 的权重落在「收益记 0」的空列上，基准被严重稀释：
    2018 年真实全A 等权 **−30.6%** 被算成 **−18.4%**，导致当年超额从 +4.0%
    误报为 −8.2%。凡 2026-11-05 之前本脚本产出的 ``bench_*`` / ``excess_*``
    数字均作废。
    """
    dates = fwd.index
    flag = np.array([d in rebalance_dates(dates) for d in dates])
    seg = pd.Series(np.cumsum(flag), index=dates)
    out = pd.Series(np.nan, index=dates)
    for _, sub in fwd.groupby(seg):
        first = sub.index[0]
        # 权重只分配给「该段内有收益数据」的池内股票。
        # 用段内 any 而非段首 notna：窗口首日（或调仓日停牌）整列 NaN 时，
        # 用段首会把整段 1 个月 skip 掉（2024-01 微盘暴跌月就这么丢过）。
        w = (mask.loc[first].to_numpy(dtype=float)
             * sub.notna().any(axis=0).to_numpy(dtype=float))
        n = w.sum()
        if n == 0:
            continue
        w = w / n
        cum = (1.0 + sub.fillna(0.0)).cumprod()
        prev = cum.shift(1).fillna(1.0)
        out.loc[sub.index] = (cum / prev).to_numpy() @ w - 1.0
    return out.fillna(0.0)


def _raw_fwd(ctx) -> pd.DataFrame:
    """**未填 0** 的前瞻收益（基准专用）。

    两个坑（2026-11-05 实测）：
    1. ``caliber.load_context`` 返回的 ``ctx.fwd`` 已 ``fillna(0.0)`` ⇒「当年还没
       价格的列」与「真实 0 收益」不可区分，等权基准被稀释（2018 年真实 −30.6%
       被算成 −18.4%）。
    2. ``ctx.close`` 是**窗口裁剪后**的，窗口首日没有前值 ⇒ 首行全 NaN。若窗口首日
       恰是月首（2024-01-02 就是），整段 1 月会被跳过——而 2024 年 1 月是微盘股
       暴跌月 ⇒ 基准从 4.6% 被抬到 28.9%。
    ⇒ 必须从**完整** close 面板算 pct_change，再裁到窗口。
    """
    from scripts.evaluation.caliber import BASE_DIR

    full = pd.read_parquet(BASE_DIR / "close_adj.parquet").reindex(columns=ctx.pred.columns)
    f = full.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return f.reindex(index=ctx.pred.index)


def _bench_metrics(dr: pd.Series) -> dict:
    return {
        "bench_ann": float(annual_return(dr)),
        "bench_sharpe": float(sharpe_ratio(dr)),
        "bench_vol": float(annual_volatility(dr)),
        "bench_max_dd": float(max_drawdown(dr)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="ens_h1h5")
    ap.add_argument("--begin", type=int, default=20190101)
    ap.add_argument("--end", type=int, default=20260907)
    ap.add_argument("--pools", default="all_a," + ",".join(POOLS),
                    help="逗号分隔；all_a = 不过滤（对照）")
    ap.add_argument("--profile", default="production")
    ap.add_argument("--entry", type=float, default=0.15)
    ap.add_argument("--exit", type=float, default=0.40)
    ap.add_argument("--fixed-n", type=int, default=0,
                    help=">0 时改用固定持仓数（不分池缩放），用于隔离分散度效应")
    ap.add_argument("--keep-bj", action="store_true",
                    help="保留北交所成分（默认剔除，与生产「BJ 不入榜」一致）")
    ap.add_argument("--out", default="reports/pool_scope_probe/by_pool.csv")
    args = ap.parse_args()

    prof = get_profile(args.profile)
    ctx = load_context(signal=args.signal, begin=args.begin, end=args.end,
                       execution=prof.execution)
    cost = cost_profile(prof.cost)
    print(describe(ctx, prof.cost, prof.capital, prof.name))
    print(f"面板 {ctx.pred.shape[0]} 日 × {ctx.pred.shape[1]} 股 · "
          f"{ctx.pred.index.min().date()}..{ctx.pred.index.max().date()}")

    rows = []
    fwd_raw = _raw_fwd(ctx)          # 完整面板算一次，循环内复用（读盘 ~1s/次）
    for pool in [p.strip() for p in args.pools.split(",") if p.strip()]:
        if pool == "all_a":
            m = pd.DataFrame(True, index=ctx.pred.index, columns=ctx.pred.columns)
        else:
            if pool not in POOLS:
                raise KeyError(f"未知池子 {pool!r}，可选 all_a / {sorted(POOLS)}")
            m = _pool_mask(POOLS[pool], ctx.pred.index, ctx.pred.columns)

        sig = ctx.pred.where(m)
        if not args.keep_bj:
            # 生产出榜「BJ 不入榜」（alla_daily_rank.py）；中证2000 成分含少量北交所
            # （实测约 0.7%），不剔除会让 zz2000 臂与生产口径不同源。
            bj = [c for c in sig.columns if c.endswith(".BJ")]
            if bj:
                sig = sig.copy()
                sig.loc[:, bj] = np.nan
                m = m.copy()
                m.loc[:, bj] = False
        n_eff = sig.notna().sum(axis=1)
        n_eff_pos = n_eff[n_eff > 0]
        n_top = (pd.Series(float(args.fixed_n), index=n_eff_pos.index)
                 if args.fixed_n > 0 else (args.entry * n_eff_pos).round())

        bt = VectorBacktest(
            strategy=(BufferedFixedNLongOnly(args.fixed_n)
                      if args.fixed_n > 0
                      else BufferedTopFracLongOnly(args.entry, args.exit)),
            rebalance_freq="M", initial_capital=float(prof.capital), costs=cost)
        res = run_once(bt, ctx, prof.execution, signal=sig)
        m_tr = _metrics(res, prof.capital)
        dr = res.daily_returns

        bench = _bench_returns(fwd_raw, m)
        bm = _bench_metrics(bench)
        excess = dr - bench.reindex(dr.index).fillna(0.0)

        row = {
            "pool": pool,
            "index_code": POOLS.get(pool, "全A(不过滤)"),
            "signal": args.signal,
            "profile": prof.name,
            "fixed_n": args.fixed_n,
            "begin": ctx.pred.index.min().strftime("%Y%m%d"),
            "end": ctx.pred.index.max().strftime("%Y%m%d"),
            "n_days": int(len(dr)),
            "years": round(len(dr) / 252.0, 2),
            "n_valid_median": float(n_eff_pos.median()) if len(n_eff_pos) else 0.0,
            "n_valid_min": int(n_eff_pos.min()) if len(n_eff_pos) else 0,
            "n_top_median": float(n_top.median()) if len(n_top) else 0.0,
            "ann": m_tr["ann"],
            "sharpe": m_tr["sharpe"],
            "sharpe_arith": m_tr["sharpe_arith"],
            "sharpe_t_nw": m_tr["sharpe_t_stat"],
            "vol": m_tr["vol"],
            "turnover": m_tr["turnover"],
            "max_dd": m_tr["max_dd"],
            "n_active": m_tr["n_active"],
            # 超额（策略 − 同池等权基准，日频相减后重算年化/Sharpe）
            "excess_ann": float(annual_return(excess)),
            "excess_sharpe": float(sharpe_ratio(excess)),
            **bm,
        }
        for y, sub in dr.groupby(dr.index.year):
            row[f"ret_{y}"] = float((1 + sub).prod() - 1)
            bs = bench.reindex(sub.index).fillna(0.0)
            row[f"exc_{y}"] = float((1 + sub).prod() - (1 + bs).prod())
        rows.append(row)
        print(f"[{pool:8s}] ann={row['ann']*100:6.2f}%  SR={row['sharpe']:.3f}  "
              f"超额={row['excess_ann']*100:+6.2f}%  "
              f"基准={row['bench_ann']*100:6.2f}%/SR{row['bench_sharpe']:.3f}  "
              f"MDD={row['max_dd']*100:5.1f}%  实际持仓={row['n_active']:.0f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\n已写 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
