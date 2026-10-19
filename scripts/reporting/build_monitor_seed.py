"""构造 monitor_production_ic 的生产口径历史种子面板（一次性，2026-10-17）。

联动件④（TODO §一）：IC 监控的历史段种子须与生产口径同构——但该信号**不是
任何 rolling_grid 落盘文件**（caliber 教训：``ens_h1020__h1.parquet`` 永远不会
生成），须现场构造：

    seed = 0.5·rank(fast(w)) + 0.5·rank(slow)
    fast(w) = w·rank(h1) + (1−w)·mean(rank(h5), rank(h10), rank(h20))
    slow   = build_slow_panel(trailing 24 月 ICIR, embargo 2 月末, min_cov 0.5)

与 ``scripts/evaluation/cheap_lever_revalidate_920.py`` **同口径**（直接复用其
``wrank_avg``）。快信号基座 = ``reports/alla_rolling_ortho920fundind/pred``
（v2 数据态冻结面板）。默认 w=0.25 = **现行生产 dingban** 形态；v1w2-cheap
（切换 2026-10-17 搁置）通过后以 ``--h1-weight 0.125
--out reports/monitoring/seed_v1w2_h1020slow.parquet`` 重建。

⚠️ 环境注记：慢信号因子数随本机 ``panels_neu`` 实体而变（本机 2026-10-17
补齐 P5 后 72/72 就绪，div_yield_ttm 因 dividend 表仅 520 只为空面板被自然
排除；好机器记录为 81 因子族）——换机或面板变更后**须重跑本脚本**再跑监控。

用法:
    python scripts/reporting/build_monitor_seed.py                 # dingban（h1w=0.25）
    python scripts/reporting/build_monitor_seed.py --h1-weight 0.125  # v1w2 形态
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("seed_prod")

PRED_DIR = ROOT / "reports" / "alla_rolling_ortho920fundind"
OUT = ROOT / "reports" / "monitoring" / "seed_dingban_h1020slow.parquet"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", default=str(PRED_DIR),
                    help="920 fundind 定版 pred 目录（v2 数据态冻结面板）")
    ap.add_argument("--h1-weight", type=float, default=0.25,
                    help="快信号 h1 份额（现行生产 dingban 等权 = 0.25；"
                         "v1w2-cheap = 0.125）")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    from scripts.evaluation.cheap_lever_revalidate_920 import wrank_avg
    from scripts.evaluation.fundamental_blend import build_slow_panel
    from scripts.evaluation.horizon_mix import load
    from scripts.common import alla_common as AC

    t0 = time.time()
    pred_dir = Path(args.pred_dir)
    w = float(args.h1_weight)

    h = {k: load(pred_dir, f"gbdt__h{k}") for k in (1, 5, 10, 20)}
    ranked = {k: p.rank(axis=1, pct=True) for k, p in h.items()}
    rest = (1.0 - w) / 3.0
    fast = wrank_avg(ranked, {1: w, 5: rest, 10: rest, 20: rest})

    base = AC.load_base()
    slow, _mx, _wt = build_slow_panel(base, fast.index, 24, 2, 0.5, 12, log=print)
    slow_rank = slow.rank(axis=1, pct=True)
    seed = 0.5 * fast.rank(axis=1, pct=True) + 0.5 * slow_rank

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seed.to_parquet(out)
    log.info("种子落盘 %s：%d 日 × %d 股（%s ~ %s），h1w=%.3f，耗时 %.0fs",
             out, seed.shape[0], seed.shape[1], seed.index[0].date(),
             seed.index[-1].date(), w, time.time() - t0)


if __name__ == "__main__":
    main()
