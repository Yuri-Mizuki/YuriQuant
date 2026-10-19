"""A1 同目标交叉评估 + 组合层对照（2026-10-19）。

## 定位

`TODO.md` §六 A1 下一步。首个 h60 模型（h20 清单）IC 0.141 vs h20 基线
0.1237，但**两模型评的是各自目标**（h60 模型评 h60 标签、h20 模型评 h20
标签）——且 h60 目标本身更可预测（主信号 h60 IC 0.148 vs h20 0.132），
IC 差不能归因于「h60 训练更好」。本脚本做两件事：

1. **同目标交叉**：两模型 pred 面板 × {h20, h60} 标签的 2×2 IC 矩阵 +
   配对日 IC 差（NW t）——同目标下才能比模型；
2. **组合层**：`--portfolio` 时把各 pred 当信号，与 882 工作基线同窗、
   同口径（production profile + `BufferedTopFracLongOnly(0.15,0.40)` 月频）
   回测净 Sharpe / Calmar / 换手——A1 的立项判据（长 horizon 的卖点是
   低换手高容量，不以年化为唯一判据）。

## 口径

- IC：可交易掩码 + 截面 Spearman（`stats.ic.calc_ic_series`），与
  `horizon_probe` / `h60_probe` 完全同源；
- 配对差：两模型**同日** IC 相减后的 NW t（消除目标日重叠导致的伪显著）；
- 组合层：pred 窗口短于全史（walk-forward OOS）⇒ `--begin` 截窗口，
  baseline 臂（882 `ens_h1h5`）同窗截齐，绝对数字不与登记表全窗口径并列。

## 用法

    python -m scripts.evaluation.horizon_cross_eval \
        --preds m_h20=path/to/pred_refit060.parquet,m_h60=path/to/pred2.parquet \
        --horizons 20,60
    # 组合层（同窗对照，窗口取两 pred 交集）
    python -m scripts.evaluation.horizon_cross_eval ... --portfolio
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("horizon_cross_eval")


def ic_stats(ic: pd.Series) -> dict:
    from stats.robust_stats import nw_tstat

    s = ic.dropna()
    if len(s) < 2:
        return dict(ic_mean=float("nan"), ic_ir=float("nan"),
                    ic_win=float("nan"), ic_nw_t=float("nan"), n=0)
    sd = float(s.std())
    t_nw, _se, _lag = nw_tstat(s.to_numpy(dtype=float))
    return dict(ic_mean=float(s.mean()),
                ic_ir=float(s.mean() / sd) if sd else float("nan"),
                ic_win=float((s > 0).mean()), ic_nw_t=float(t_nw), n=int(len(s)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds", required=True,
                    help="逗号分隔 name=path 列表（第一个为基准臂）")
    ap.add_argument("--horizons", default="20,60")
    ap.add_argument("--baseline", default="ens_h1h5",
                    help="组合层的基线信号（caliber SIGNALS 键）")
    ap.add_argument("--portfolio", action="store_true", help="跑组合层对照")
    ap.add_argument("--out", default="reports/frontier_audit_2026-10-19/cross_eval")
    args = ap.parse_args(argv)

    from scripts.common.alla_common import load_base
    from stats.ic import calc_ic_series

    preds: dict[str, pd.DataFrame] = {}
    for tok in args.preds.split(","):
        name, _, path = tok.partition("=")
        p = pd.read_parquet(path.strip())
        p = p.replace([np.inf, -np.inf], np.nan).astype(np.float32)
        preds[name.strip()] = p
        log.info("pred[%s] = %s（%s ~ %s，%d 日）", name, path,
                 p.index[0].date(), p.index[-1].date(), len(p))
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    base = load_base()
    close, mask_all = base["close"], base["mask"].astype(bool)
    idx = preds[next(iter(preds))].index
    for p in preds.values():
        idx = idx.intersection(p.index)
    cols = close.columns
    mask = mask_all.reindex(index=idx).fillna(False)

    rows: list[dict] = []
    for name, p in preds.items():
        pr = p.reindex(index=idx)
        for h in horizons:
            fwd = (close.pct_change(h, fill_method=None).shift(-h)
                   .reindex(index=idx).replace([np.inf, -np.inf], np.nan)
                   .astype(np.float32))
            st = ic_stats(calc_ic_series(pr, fwd, tradable=mask))
            rows.append({"model": name, "target": h, **st})
            log.info("IC | %-8s → h%-3d = %+.4f (NW t=%.2f, win=%.1f%%, n=%d)",
                     name, h, st["ic_mean"], st["ic_nw_t"], 100 * st["ic_win"],
                     st["n"])
            del fwd

    # 配对差：vs 基准臂（第一个 pred），同目标
    ref_name = next(iter(preds))
    for name in list(preds)[1:]:
        for h in horizons:
            fwd = (close.pct_change(h, fill_method=None).shift(-h)
                   .reindex(index=idx).replace([np.inf, -np.inf], np.nan)
                   .astype(np.float32))
            ic_a = calc_ic_series(preds[ref_name].reindex(index=idx), fwd,
                                  tradable=mask)
            ic_b = calc_ic_series(preds[name].reindex(index=idx), fwd,
                                  tradable=mask)
            both = pd.concat([ic_a, ic_b], axis=1).dropna()
            st = ic_stats(both.iloc[:, 1] - both.iloc[:, 0])
            rows.append({"model": f"{name} − {ref_name}", "target": h,
                         **st, "paired": True})
            log.info("配对差 | %s − %s → h%d = %+.4f (NW t=%.2f, n=%d)",
                     name, ref_name, h, st["ic_mean"], st["ic_nw_t"], st["n"])
            del fwd

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    ic_df = pd.DataFrame(rows)
    ic_df.to_csv(dest / "ic_cross.csv", index=False, encoding="utf-8-sig")

    pf_rows = []
    if args.portfolio:
        from scripts.evaluation.caliber import (
            cost_profile,
            evaluate_strategy,
            get_profile,
            load_context,
        )
        from strategy.examples import BufferedTopFracLongOnly

        prof = get_profile("production")
        b0 = max(p.index[0] for p in preds.values())
        b0i = int(str(b0.date()).replace("-", ""))
        ctx = load_context(signal=args.baseline, window="prod", begin=b0i,
                           execution=prof.execution)
        log.info("组合层窗口 %s ~ %s（%d 日，production 口径）",
                 ctx.begin, ctx.pred.index[-1].date(), ctx.n_days)
        cost = cost_profile(prof.cost)
        arms = {args.baseline: ctx.pred}
        for name, p in preds.items():
            arms[name] = p.reindex(index=ctx.pred.index,
                                   columns=ctx.pred.columns)
        for name, sig in arms.items():
            m = evaluate_strategy(BufferedTopFracLongOnly(0.15, 0.40), ctx,
                                  cost, prof.capital, signal=sig, yearly=True)
            calmar = (m["ann"] / m["max_dd"] if m["max_dd"] > 0 else np.nan)
            pf_rows.append({"arm": name, "calmar": calmar, **m})
            log.info("组合 | %-10s ann=%.2f%% sharpe=%.3f calmar=%.2f "
                     "mdd=%.3f turn=%.1f%%", name, 100 * m["ann"], m["sharpe"],
                     calmar, m["max_dd"], 100 * m["turnover"])
        pf_df = pd.DataFrame(pf_rows)
        pf_df.to_csv(dest / "portfolio.csv", index=False, encoding="utf-8-sig")

    meta = {"preds": {t.partition("=")[0].strip(): t.partition("=")[2].strip()
                      for t in args.preds.split(",")},
            "horizons": horizons, "baseline": args.baseline,
            "n_days": int(len(idx)), "oos": f"{idx[0].date()} ~ {idx[-1].date()}",
            "portfolio": bool(args.portfolio)}
    (dest / "meta.json").write_text(json.dumps(meta, ensure_ascii=False,
                                               indent=2), encoding="utf-8")
    log.info("完成 -> %s", dest)


if __name__ == "__main__":
    main()
