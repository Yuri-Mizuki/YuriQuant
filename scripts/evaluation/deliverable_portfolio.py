"""可交付组合：集中度 × 行业约束（P0 落地，2026-10-10 固化）。

问题：定版榜单的 ``picks`` 是 band 上限（全A Top10% ≈ 550 只），**不是能下单的
组合**。本脚本回答"从 ~550 只集中到 45~50 只、再加行业约束"绩效怎么变，并把
最优可交付形态钉到网格上。

2026-10-09 首测（882 ortho 口径）三条结论：
  1. **集中化是赚的**：~550 只 → 45 只，年化 15.52% → 17.44%（+1.9pp）、
     Sharpe 0.631 → 0.687；代价换手 72.9% → 91.0%。
  2. **行业硬只数上限是净代价**：「每申万一级 ≤2 只」−2.64pp（17.44% → 14.80%），
     四个 (k,m) 组合一致 −2.4 ~ −3.1pp ⇒ 不要用硬只数上限。
  3. **回撤几乎不动**（38.4 ~ 40.4%）。

口径（与 ``reports/deliverable_portfolio/README.md`` 逐行对齐）：
  h=1 日频收益 · M 月频调仓 · equal 等权 · ``default_costs`` ·
  base = ``reports/alla_rolling/_base``（行业 = 申万一级，31 个）。

⚠️ 口径边界：默认 pred 是 882 ortho 主实验（``ens_h1h5__h1``）。920 定版口径
（``--preset dingban`` = fundind + h1020⊕s0.5）的 rolling pred **本机无副本**，
复验请在好机器上用 ``--pred-dir reports/alla_rolling_<tag>/pred --signal defv``
重跑（本脚本已支持该信号形态，无需改代码）。

用法:
    python -m scripts.evaluation.deliverable_portfolio
    python -m scripts.evaluation.deliverable_portfolio --pred-dir reports/alla_rolling_ortho/pred
    python -m scripts.evaluation.deliverable_portfolio --signal defv \\
        --pred-dir reports/alla_rolling_ortho920tl/pred   # 定版口径复验
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("deliverable_portfolio")

# 行业分类默认位置（882/920 的 _base 里 industry.parquet 只在 882 目录下）
DEFAULT_INDUSTRY = "reports/alla_rolling/_base/industry.parquet"


def load_signal(pred_dir: Path, kind: str, base: dict) -> pd.DataFrame:
    """载入/构造预测信号面板。

    kind:
      - ``ens_h1h5``：直接读 ``ens_h1h5__h1.parquet``（882 主实验口径）；
      - ``h1020``：``gbdt__h1/5/10/20`` 四 horizon 逐日秩平均（920 快信号）；
      - ``defv``：h1020 ⊕ 慢信号 s0.5（定版 ``--preset dingban`` 口径）。
    """
    if kind == "ens_h1h5":
        fp = pred_dir / "ens_h1h5__h1.parquet"
        if not fp.exists():
            raise SystemExit(f"{fp} 缺失（--signal ens_h1h5 需要该文件）")
        return pd.read_parquet(fp)

    from scripts.pipelines import rolling_grid_alla as RG

    hs = []
    for h in (1, 5, 10, 20):
        fp = pred_dir / f"gbdt__h{h}.parquet"
        if not fp.exists():
            raise SystemExit(f"{fp} 缺失（--signal {kind} 需要四 horizon gbdt pred）")
        hs.append(pd.read_parquet(fp))
    sig = RG._rank_average(hs)

    if kind == "defv":
        from scripts.evaluation.fundamental_blend import build_slow_panel

        slow, _slow_max, _w = build_slow_panel(
            base, sig.index, 24, 2, 0.5, 12, log=print)
        # 0.5·rank(h1020) + 0.5·rank(slow)，与 buffer_arm_920 --variant defv 同口径
        sig = (0.5 * sig.rank(axis=1, pct=True)
               + 0.5 * slow.rank(axis=1, pct=True))
        log.info("定版变体：h1020⊕s0.5 构造完成（slow 面板 %d 日）", slow.shape[0])
    return sig


def build_arms(args, sectors: pd.DataFrame):
    """构造待评估臂列表 [(tag, strategy_factory), ...]。"""
    from strategy.examples import SectorCappedTopKLongOnly, TopFracLongOnly, TopKLongOnly

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    ms = [int(x) for x in args.sector_ms.split(",") if x.strip()]
    arms = [(f"基线 TopFrac{args.frac:.2f}(~550只)",
             lambda: TopFracLongOnly(frac=args.frac, weight_mode="equal"))]
    arms += [(f"TopK {k}（无约束）", lambda k=k: TopKLongOnly(k=k)) for k in ks]
    arms += [(f"SectorCap k={args.sector_k} m={m}",
              lambda m=m: SectorCappedTopKLongOnly(
                  sectors, k=args.sector_k, max_per_sector=m)) for m in ms]
    return arms


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho/pred",
                    help="pred 目录（默认 882 ortho 主实验）")
    ap.add_argument("--signal", default="ens_h1h5",
                    choices=["ens_h1h5", "h1020", "defv"],
                    help="信号形态；defv = 定版 h1020⊕s0.5")
    ap.add_argument("--out", default="reports/deliverable_portfolio")
    ap.add_argument("--industry", default=DEFAULT_INDUSTRY,
                    help="申万一级行业面板（date×code）")
    ap.add_argument("--frac", type=float, default=0.10, help="基线 TopFrac 比例")
    ap.add_argument("--ks", default="30,35,40,45,50,55,60,80,100,200",
                    help="无约束 TopK 网格（逗号分隔）")
    ap.add_argument("--sector-k", type=int, default=45,
                    help="行业约束臂的 k（固定 k，扫 m）")
    ap.add_argument("--sector-ms", default="2,3,4,5,6,8",
                    help="行业约束臂的 max_per_sector 网格（逗号分隔）")
    ap.add_argument("--freq", default="M", choices=["M", "W"], help="调仓频率")
    ap.add_argument("--slippage-bp", type=float, default=None,
                    help="覆盖滑点（bp）；不给则用 config 真源。成本敏感性扫描用"
                         "（集中化把换手推到 90%%+，小盘滑点更大）。"
                         "单边成本 = 2·佣金 + 印花税 + 2·滑点")
    ap.add_argument("--exclude-baseline", action="store_true",
                    help="跳过基线 TopFrac 臂（省时间）")
    args = ap.parse_args(argv)

    from backtest.costs import default_costs
    from backtest.engine import VectorBacktest
    from scripts.pipelines.rolling_grid_alla import _ann, load_base, res_metrics

    pred_dir = Path(args.pred_dir)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    base = load_base()
    pred = load_signal(pred_dir, args.signal, base)
    close = base["close"]
    mask = base["mask"].astype(bool)
    bench_idx = base["bench_index"]
    bench_eqw = base["bench_eqw"]
    sectors = pd.read_parquet(ROOT / args.industry)
    log.info("signal=%s pred=%s 行业数=%d",
             args.signal, pred.shape, sectors.stack().nunique())

    oos = pred.index
    fwd = close.pct_change(fill_method=None).loc[oos]
    mask_oos = mask.reindex(index=oos, columns=close.columns).fillna(True)
    b_eqw = bench_eqw.reindex(oos).fillna(0.0)
    costs = default_costs()
    if args.slippage_bp is not None:
        # 成本敏感性扫描：单边成本率 = 2·佣金 + 印花税 + 2·滑点
        # （slippage_bp 是**基点数值**，须 /1e4 换算，与 costs.calc 同口径）
        def _onesided_bp(c) -> float:
            return (2 * c.commission_rate + c.stamp_duty
                    + 2 * c.slippage_bp / 1e4) * 1e4

        prev_bp = _onesided_bp(costs)
        costs.slippage_bp = float(args.slippage_bp)
        log.info("滑点覆盖 → %.0fbp：单边成本 %.1f → %.1f bp",
                 args.slippage_bp, prev_bp, _onesided_bp(costs))

    arms = build_arms(args, sectors)
    if args.exclude_baseline:
        arms = [a for a in arms if not a[0].startswith("基线")]

    rows = []
    for tag, strat in arms:
        bt = VectorBacktest(strategy=strat(), rebalance_freq=args.freq,
                            initial_capital=1_000_000.0, costs=costs)
        res = bt.run(pred, fwd, executable_mask=mask_oos, horizon=1,
                     rebalance_days=None)
        dr = res.daily_returns
        m = res_metrics(dr, bench_idx.reindex(dr.index).fillna(0.0),
                        res.turnover_series)
        wh = res.weights_history
        nhold = (wh > 0).sum(axis=1)
        nhold = nhold[nhold > 0]
        row = dict(arm=tag, annual=m["annual"],
                   excess_eqw=m["annual"] - _ann(b_eqw),
                   sharpe=m["sharpe"], max_dd=m["max_dd"],
                   turnover=m["turnover"],
                   n_hold_mean=float(nhold.mean()) if len(nhold) else 0.0,
                   n_hold_min=int(nhold.min()) if len(nhold) else 0)
        rows.append(row)
        log.info("%-26s 年化=%+6.2f%% 超额=%+6.2f%% Sharpe=%.3f 回撤=%5.1f%% "
                 "换手=%5.1f%% 持仓均值=%5.0f",
                 tag, row["annual"] * 100, row["excess_eqw"] * 100,
                 row["sharpe"], row["max_dd"] * 100, row["turnover"] * 100,
                 row["n_hold_mean"])

    df = pd.DataFrame(rows)
    csv = dest / "results.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    log.info("完成 %.0fs -> %s", time.time() - t0, csv)


if __name__ == "__main__":
    main()
