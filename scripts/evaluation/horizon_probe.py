"""Horizon 探针：主信号在 h1–h120 上的 IC 衰减与独立样本数（2026-10-19）。

## 定位

TODO_ARCHIVE.md 册Ⅰ §六 A1 的**最便宜前置**：在决定是否训练 h60/h120 模型之前，先回答
「主信号的预测力能活到多长」。如果 h60 的 IC 已经衰减到 0，训练 h60 模型大概率
只是拟合噪声；如果仍有 0.02–0.03 的可交易 IC，才值得做标签/训练/回测全链。

## 口径

- 主信号默认 `reports/alla_rolling_ortho/pred/ens_h1h5__h1.parquet`（882 ortho）。
- 收益 = `close_adj.pct_change(h).shift(-h)`；IC = 截面 Spearman；T+1 可交易掩码。
- 每个 h 报三组：
  1. **daily**：全部重叠日 IC 的均值 / NW t（重叠会高估显著性）；
  2. **non-overlap**：每 h 个交易日取 1 天（近似独立期），均值 / NW t / n；
  3. **effective n**：按 IC 一阶自相关折算 `n_eff = n(1-ρ)/(1+ρ)`（粗估）。
- 同时报 `fwd` 覆盖率与 IC 胜率。

## 用法

    python -m scripts.evaluation.horizon_probe
    python -m scripts.evaluation.horizon_probe --horizons 1,5,10,20,60,120
    python -m scripts.evaluation.horizon_probe --signals ens_h1h5__h1,gbdt__h20
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

log = setup_logging("horizon_probe")


def _stats(ic: pd.Series) -> dict:
    from stats.robust_stats import nw_tstat

    s = ic.dropna()
    if len(s) < 2:
        return dict(ic=float("nan"), ir=float("nan"), win=float("nan"),
                    nw_t=float("nan"), n=0, ac1=float("nan"),
                    n_eff=float("nan"))
    sd = float(s.std())
    t_nw, _se, _lag = nw_tstat(s.to_numpy(dtype=float))
    ac1 = float(s.autocorr(1)) if len(s) > 2 else float("nan")
    n_eff = (len(s) * (1 - ac1) / (1 + ac1)
             if ac1 == ac1 and -1 < ac1 < 1 else float("nan"))
    return dict(ic=float(s.mean()), ir=float(s.mean() / sd) if sd else float("nan"),
                win=float((s > 0).mean()), nw_t=float(t_nw), n=int(len(s)),
                ac1=ac1, n_eff=float(n_eff))


def _nonoverlap(ic: pd.Series, h: int) -> pd.Series:
    """每 h 个交易日取 1 天（从第一个有效日开始），近似非重叠标签。"""
    s = ic.dropna()
    if s.empty:
        return s
    return s.iloc[:: max(1, h)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho/pred")
    ap.add_argument("--signals", default="ens_h1h5__h1",
                    help="逗号分隔的信号文件名（不含 .parquet）")
    ap.add_argument("--horizons", default="1,5,10,20,60,120")
    ap.add_argument("--out", default="reports/frontier_audit_2026-10-19")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args(argv)

    from scripts.common.alla_common import load_base
    from stats.ic import calc_ic_series

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.out_tag}" if args.out_tag else ""
    out_csv = dest / f"horizon_ic{suffix}.csv"
    meta_json = dest / f"meta_horizon{suffix}.json"

    base = load_base()
    close = base["close"]
    mask = base["mask"].astype(bool)
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    t0 = time.time()
    rows: list[dict] = []

    for sig_name in signals:
        fp = Path(args.pred_dir) / f"{sig_name}.parquet"
        if not fp.exists():
            log.warning("缺信号，跳过：%s", fp)
            continue
        main = pd.read_parquet(fp).replace([np.inf, -np.inf], np.nan)
        main = main.astype(np.float32)
        cols = main.columns.intersection(close.columns)
        idx = main.index
        main = main.reindex(index=idx, columns=cols)
        mask_s = mask.reindex(index=idx, columns=cols).fillna(False).astype(bool)
        log.info("信号 %s: %d 日 × %d 列 | %s ~ %s", sig_name, len(idx), len(cols),
                 idx[0].date(), idx[-1].date())
        for h in horizons:
            fwd = (close.pct_change(h, fill_method=None)
                   .shift(-h)
                   .reindex(index=idx, columns=cols)
                   .replace([np.inf, -np.inf], np.nan)
                   .astype(np.float32))
            ic = calc_ic_series(main, fwd, tradable=mask_s)
            daily = _stats(ic)
            non = _stats(_nonoverlap(ic, h))
            row = {"signal": sig_name, "horizon": h,
                   "fwd_cov": float(fwd.notna().to_numpy(dtype=float).mean()),
                   "daily_ic": daily["ic"], "daily_ir": daily["ir"],
                   "daily_nw_t": daily["nw_t"], "daily_win": daily["win"],
                   "daily_n": daily["n"], "daily_ac1": daily["ac1"],
                   "daily_n_eff": daily["n_eff"],
                   "nonoverlap_ic": non["ic"], "nonoverlap_nw_t": non["nw_t"],
                   "nonoverlap_n": non["n"], "nonoverlap_win": non["win"]}
            rows.append(row)
            log.info("h=%-3d | daily IC=%+.4f (NW t=%.2f, n=%d) | "
                     "non-overlap IC=%+.4f (NW t=%.2f, n=%d) | fwd cov=%.2f",
                     h, daily["ic"], daily["nw_t"], daily["n"],
                     non["ic"], non["nw_t"], non["n"], row["fwd_cov"])
            del fwd, ic
        del main, mask_s

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    meta = {"signals": signals, "horizons": horizons,
            "n_rows": int(len(df)), "elapsed_s": round(time.time() - t0, 1),
            "caveat": ("主信号为 882 ortho，且训练标签是 h1/h5；h60/h120 的 IC 是"
                       "信号衰减诊断，不等于「用 h60 标签重训」的模型表现。"
                       "重叠标签的 daily NW t 会高估显著性，判读以 non-overlap 列为准。")}
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    log.info("完成 %.0fs -> %s", time.time() - t0, out_csv)
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
