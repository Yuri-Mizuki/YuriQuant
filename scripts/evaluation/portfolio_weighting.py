"""组合层权重对照（P1）：等权 vs 波动率倒数加权 vs 风险平价。

背景
----
定版组合 = **Top10% 等权**（`reports/alla_daily_defv/picks_*.csv` 权重恒为 1/742）；
deliverable 口径 = **TopK 45 等权**（17.44% / Sharpe 0.687，882 ortho）。
优化层的 QP/HRP/BL **只在 hs300/zz1000 试过**（RL/QP 24 臂无稳健超额），
**全A 口径下"权重分配"这一维从未单独测过**。

本实验：固定信号与选股规则，**只改权重分配**，回答「风险加权能否胜过等权」。

权重模式
--------
- `equal`   : 1/N（现行定版）
- `inv_vol` : ∝ 1/σ（60 日滚动收益波动），归一
- `erc`     : 风险平价近似（等风险贡献，用对角协方差的迭代解）

口径边界
--------
pred = 882 ortho（`reports/alla_rolling_ortho/pred`）。**920 定版 pred 本机无副本**
（`reports/alla_rolling_ortho920fundind/` 只有 selection + _base）⇒ 本实验的
**相对结论可迁移、绝对数字属 882 口径**，920 复验须在好机器跑。

用法
----
    python -m scripts.evaluation.portfolio_weighting
    python -m scripts.evaluation.portfolio_weighting --ks 45,100 --modes equal,inv_vol
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:/YuriQuant")

from strategy.base import Strategy  # noqa: E402

log = logging.getLogger("portfolio_weighting")
ROOT = Path(r"E:/YuriQuant")
VOL_WINDOW = 60
VOL_MIN_PERIODS = 20


def _risk_parity_diag(sigma: np.ndarray, iters: int = 200) -> np.ndarray:
    """对角协方差下的等风险贡献权重（闭式迭代解）。"""
    n = len(sigma)
    if n == 0:
        return sigma
    w = 1.0 / np.maximum(sigma, 1e-12)
    w = w / w.sum()
    for _ in range(iters):
        rc = w * sigma                      # 风险贡献 ∝ w_i σ_i
        target = rc.sum() / n
        w = w * (target / np.maximum(rc, 1e-18)) ** 0.5
        w = np.maximum(w, 1e-12)
        w = w / w.sum()
    return w


class VolWeightedTopKLongOnly(Strategy):
    """Top-K 纯多头 + 风险加权（可选排名缓冲带）。

    ``vol`` 为 date×code 的波动率面板（PIT：调仓日收盘后可算）。
    ``exit_mult`` 非空时启用指数式缓冲带（保留门槛 = 排名 ≤ K×exit_mult），
    此时策略**有状态**，每个回测须用新实例。
    缺波动率的标的回退到截面中位数，避免整臂因单只缺数而崩。
    """

    def __init__(self, vol: pd.DataFrame, k: int = 45, mode: str = "inv_vol",
                 exit_mult: float | None = None):
        self.vol = vol
        self.k = k
        self.mode = mode
        self.exit_mult = exit_mult
        self._prev: set = set()
        self.name = (f"topk_lo_{k}_{mode}"
                     + (f"_buf{exit_mult:g}" if exit_mult else ""))

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        raise NotImplementedError("需按 date 取波动率，请走 get_weights_at")

    def get_weights_at(self, date: pd.Timestamp,
                       factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if len(vals) == 0:
            self._prev = set()
            return pd.Series(dtype=float)
        k = min(self.k, len(vals))
        if k <= 0:
            return pd.Series(dtype=float)

        if self.exit_mult is None:
            codes = list(vals.nlargest(k).index)
        else:
            ranks = vals.rank(ascending=False, method="first")
            n_exit = max(k, int(round(k * self.exit_mult)))
            keep = [c for c in self._prev if c in ranks.index and ranks[c] <= n_exit]
            cands = [c for c in ranks.sort_values().index
                     if c not in self._prev and ranks[c] <= k]
            port = keep + cands[:max(0, k - len(keep))]
            if len(port) > k:
                port = sorted(port, key=lambda c: ranks[c])[:k]
            self._prev = set(port)
            codes = port
        if len(codes) == 0:
            return pd.Series(dtype=float)

        if self.mode == "equal":
            return pd.Series(1.0 / len(codes), index=codes)

        if date in self.vol.index:
            v = self.vol.loc[date].reindex(codes).astype(float)
            fallback = float(np.nanmedian(self.vol.loc[date].values)) or 1.0
            v = v.fillna(fallback).clip(lower=1e-8)
        else:
            v = pd.Series(1.0, index=codes)

        if self.mode == "inv_vol":
            w = 1.0 / v.values
        elif self.mode == "erc":
            w = _risk_parity_diag(v.values)
        else:
            raise ValueError(f"未知 weight mode: {self.mode}")
        w = w / w.sum()
        return pd.Series(w, index=codes)


class BufferedTopKLongOnly(Strategy):
    """固定 K 多头的排名缓冲带（**有状态**，每回测须用新实例）。

    与 `strategy.examples.BufferedTopFracLongOnly` 同规则，但目标持仓数固定为 K
    （集中组合口径 45 只），而非按截面比例。保留门槛 = 排名 ≤ K×exit_mult。
    """

    def __init__(self, k: int = 45, exit_mult: float = 1.5):
        self.k = k
        self.exit_mult = exit_mult
        self._prev: set = set()
        self.name = f"buffered_topk_lo_{k}_{exit_mult}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if len(vals) == 0:
            self._prev = set()
            return pd.Series(dtype=float)
        n_exit = max(self.k, int(round(self.k * self.exit_mult)))
        ranks = vals.rank(ascending=False, method="first")
        keep = [c for c in self._prev if c in ranks.index and ranks[c] <= n_exit]
        cands = [c for c in ranks.sort_values().index
                 if c not in self._prev and ranks[c] <= self.k]
        port = keep + cands[:max(0, self.k - len(keep))]
        if len(port) > self.k:
            port = sorted(port, key=lambda c: ranks[c])[:self.k]
        self._prev = set(port)
        return pd.Series(1.0 / max(len(port), 1), index=port)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho/pred")
    ap.add_argument("--signal", default="ens_h1h5")
    ap.add_argument("--out", default="reports/portfolio_weighting")
    ap.add_argument("--ks", default="45,100")
    ap.add_argument("--modes", default="equal,inv_vol,erc")
    ap.add_argument("--buffer-mults", default="2.0",
                    help="固定 K 缓冲带臂的 exit 倍数（逗号分隔；空串=不加）")
    ap.add_argument("--freq", default="M", choices=["M", "W"])
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    from backtest.costs import default_costs
    from backtest.engine import VectorBacktest
    from scripts.evaluation.deliverable_portfolio import load_signal
    from scripts.pipelines.rolling_grid_alla import _ann, load_base, res_metrics

    dest = ROOT / args.out
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    base = load_base()
    close = base["close"]
    mask = base["mask"].astype(bool)
    bench_idx = base["bench_index"]
    bench_eqw = base["bench_eqw"]
    pred = load_signal(Path(ROOT / args.pred_dir), args.signal, base)

    vol = close.pct_change(fill_method=None).rolling(
        VOL_WINDOW, min_periods=VOL_MIN_PERIODS).std()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bms = [float(x) for x in args.buffer_mults.split(",") if x.strip()]
    arms: list[tuple[str, object]] = []
    for k in ks:
        for mode in modes:
            arms.append((f"TopK{k}_{mode}",
                         VolWeightedTopKLongOnly(vol, k=k, mode=mode)))
        for bm in bms:
            arms.append((f"TopK{k}_buf{bm:g}",
                         BufferedTopKLongOnly(k=k, exit_mult=bm)))
        if bms:
            bm = bms[0]
            arms.append((f"TopK{k}_inv_vol_buf{bm:g}",
                         VolWeightedTopKLongOnly(vol, k=k, mode="inv_vol",
                                                 exit_mult=bm)))

    oos = pred.index
    fwd = close.pct_change(fill_method=None).loc[oos]
    mask_oos = mask.reindex(index=oos, columns=close.columns).fillna(True)
    b_eqw = bench_eqw.reindex(oos).fillna(0.0)
    costs = default_costs()
    log.info("信号 %s %s | 波动率窗 %d 日 | 对照组 %d 臂",
             args.signal, pred.shape, VOL_WINDOW, len(ks) * len(modes))

    rows = []
    for tag, strat in arms:
        bt = VectorBacktest(strategy=strat, rebalance_freq=args.freq,
                            initial_capital=1_000_000.0, costs=costs)
        res = bt.run(pred, fwd, executable_mask=mask_oos, horizon=1,
                     rebalance_days=None)
        dr = res.daily_returns
        m = res_metrics(dr, bench_idx.reindex(dr.index).fillna(0.0),
                        res.turnover_series)
        wh = res.weights_history
        nh = (wh > 0).sum(axis=1)
        nh = nh[nh > 0]
        hhi = float(((wh ** 2).sum(axis=1).replace(0, np.nan)).mean())
        rows.append(dict(
            arm=tag, k=getattr(strat, "k", None),
            mode=tag.split("_", 1)[-1],
            annual=m["annual"], excess_eqw=m["annual"] - _ann(b_eqw),
            sharpe=m["sharpe"], max_dd=m["max_dd"], turnover=m["turnover"],
            n_hold_mean=float(nh.mean()) if len(nh) else 0.0,
            hhi=hhi))
        log.info("%-16s 年化=%+6.2f%% 超额=%+6.2f%% Sharpe=%.3f 回撤=%5.1f%% "
                 "换手=%5.1f%% 持仓=%4.0f HHI=%.5f",
                 rows[-1]["arm"], m["annual"] * 100, rows[-1]["excess_eqw"] * 100,
                 m["sharpe"], m["max_dd"] * 100, m["turnover"] * 100,
                 rows[-1]["n_hold_mean"], hhi)

    df = pd.DataFrame(rows)
    csv = dest / "results.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    log.info("完成 %.0fs -> %s", time.time() - t0, csv)

    if not df.empty:
        show = df.set_index("arm")[["annual", "sharpe", "max_dd", "turnover"]]
        show["annual"] = (show["annual"] * 100).round(2)
        show["max_dd"] = (show["max_dd"] * 100).round(1)
        show["turnover"] = (show["turnover"] * 100).round(1)
        show["sharpe"] = show["sharpe"].round(3)
        log.info("逐臂对照（年化%%/Sharpe/回撤%%/换手%%）：\n%s", show.to_string())


if __name__ == "__main__":
    main()
