#!/usr/bin/env python
"""P1b —— 生产基线同口径复验：定版因子 gbdt vs alpha360 gbdt / GRU。

承接 P1（`reports/p1_sequence_predictor/`）：P1 在**完全相同的 alpha360 输入**上
发现 GRU 比 gbdt 强 +57%，但那个 gbdt 用的输入是**原始 360 维 alpha360**，
不是生产在用的定版因子 —— 也就是说 **P1 的对照基线可能是弱基线**。

本脚本把**生产基线**（`y2026__h5` DPP 选中的 50 因子 gbdt）放进**同一口径**
（同池 / 同 horizon / 同折 / 同标签 / 同评估），回答一个问题：

    **GRU 相对 gbdt 的 +57%，换成生产输入后还在吗？**

口径对齐（逐项复用 `seq_predictor_compare`，不做任何重实现）：
- 池与价格：`build_px_panels`（与 P1 完全同一函数）
- 标签：`build_labels(h, mode="rank")` / `forward_returns`
- 折：`forward_roll_folds`
- 评估：`evaluate_model`（rank IC）
- 模型：`GbdtArm` / `SeqArm`（P1 同款）

因子来源：`E:/data/factor_library/all_a_2018_2026/panels_neu/`（定版正交化面板）。
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from scripts.evaluation.seq_predictor_compare import (
    FIELDS,
    GbdtArm,
    SeqArm,
    build_px_panels,
    cs_z,
    panel_from_vec,
    row_idx,
)

ROOT = Path(__file__).resolve().parents[2]
# 多机适配：因子库根走 config 单一真源（本机在 E:，另一台可能在 D:）
PANEL_ROOT = (Path(Config.paths()["factor_library_root"])
              / "all_a_2018_2026" / "panels_neu")
SELECTION = ROOT / "reports" / "alla_rolling_ortho920fundind" / "selection"
OUT_DIR = ROOT / "reports" / "p1b_prod_baseline"

log = logging.getLogger("p1b")


# ---------------------------------------------------------------------------
# 1. 定版因子面板加载
# ---------------------------------------------------------------------------
def load_prod_factors(dates: pd.DatetimeIndex, codes: pd.Index,
                      names: list[str]) -> np.ndarray:
    """读定版因子面板 → (n_d, n_c, n_f) float32，reindex 到目标 (dates, codes)。"""
    out = np.full((len(dates), len(codes), len(names)), np.nan, dtype=np.float32)
    missing = []
    for j, name in enumerate(names):
        fp = PANEL_ROOT / f"{name}.parquet"
        if not fp.exists():
            missing.append(name)
            continue
        p = pd.read_parquet(fp)
        v = p.reindex(index=dates, columns=codes).values
        out[:, :, j] = v.astype(np.float32)
    if missing:
        raise FileNotFoundError(f"缺少因子面板: {missing}")
    log.info("[prod] %d 因子面板 → %s (%.2f GB)",
             len(names), out.shape, out.nbytes / 1e9)
    return out


def load_alpha360(panels: dict, dates: pd.DatetimeIndex, codes: pd.Index, n_lags: int):
    """构造 alpha360 张量（与 P1 逐位同口径）：(n_d, n_c, T, 6)。

    复用 `run()` 已经建好的后复权面板，不重复读盘。
    """
    from factor.alpha158 import compute_alpha360
    from factor.alpha_base import AlphaData

    d = AlphaData(panels)
    p360 = compute_alpha360(d)
    arr = np.full((len(dates), len(codes), n_lags, len(FIELDS)), np.nan, dtype=np.float32)
    for fi, f in enumerate(FIELDS):
        for t in range(n_lags):
            v = cs_z(p360.pop(f"alpha360_{f}{t}")).reindex(
                index=dates, columns=codes).values
            arr[:, :, n_lags - 1 - t, fi] = v.astype(np.float32)
    del p360, d
    log.info("[a360] arr=%s (%.2f GB)", arr.shape, arr.nbytes / 1e9)
    return arr


def build_prod_seq(flat_prod: np.ndarray, T: int) -> np.ndarray:
    """(n_d, n_c, F) → (n_d, n_c, T, F)：第 t 步 = 往前 (T-1-t) 日的因子值。

    与 `alpha360` 的窗口方向一致（0 = 最早，T-1 = 最新），保证与 P1 的
    序列模型输入同构 —— 差别只在「特征维度」：alpha360 是 6 字段，这里是 50 因子。
    """
    n_d, n_c, F = flat_prod.shape
    out = np.full((n_d, n_c, T, F), np.nan, dtype=np.float32)
    for t in range(T):
        shift = T - 1 - t
        if shift == 0:
            out[:, :, t, :] = flat_prod
        else:
            out[shift:, :, t, :] = flat_prod[:-shift]
    log.info("[seq] prod 因子序列 T=%d → %s (%.2f GB)", T, out.shape, out.nbytes / 1e9)
    return out


def prod_factor_names(args) -> list[str]:
    if args.factor_set == "prod50":
        fp = SELECTION / f"y{args.select_year}__h{args.horizon}.json"
        names = json.loads(fp.read_text(encoding="utf-8"))
        log.info("[sel] %s → %d 因子", fp.name, len(names))
        return names
    if args.factor_set == "all":
        names = sorted(p.stem for p in PANEL_ROOT.glob("*.parquet"))
        log.info("[sel] panels_neu 全量 → %d 因子", len(names))
        return names
    raise ValueError(args.factor_set)


# ---------------------------------------------------------------------------
# 2. 滚动评估（口径与 P1 逐项一致）
# ---------------------------------------------------------------------------
def run(args):
    from factor.cv import forward_roll_folds
    from model.evaluation import evaluate_model
    from model.labels import build_labels, forward_returns

    panels, close_adj = build_px_panels(args.index, args.warmup, args.end)
    dates, codes = close_adj.index, close_adj.columns

    b0, e0 = pd.Timestamp(str(args.begin)), pd.Timestamp(str(args.end))
    keep = (dates >= b0) & (dates <= e0)
    close_adj = close_adj.loc[dates[keep]]
    dates = dates[keep]
    log.info("[slice] %s~%s %d 日 × %d 股", dates[0].date(), dates[-1].date(),
             len(dates), len(codes))

    flat_parts: dict[str, np.ndarray] = {}
    nf = prod_factor_names(args)
    flat_parts["prod"] = load_prod_factors(dates, codes, nf)
    if args.seq_lags > 0:
        flat_parts["prod_seq"] = build_prod_seq(flat_parts["prod"], args.seq_lags)
    if args.with_a360:
        flat_parts["a360"] = load_alpha360(panels, dates, codes, args.n_lags)
    del panels

    labels, embargo = build_labels(close_adj, horizon=args.horizon, mode="rank")
    fwd = forward_returns(close_adj, horizon=args.horizon)
    log.info("[label] horizon=%dd embargo=%dd", args.horizon, embargo)

    n = len(dates)
    n_test = max(120, int(n * args.test_frac))
    test_days = dates[-n_test:]
    folds = forward_roll_folds(dates, test_days, args.n_folds, embargo)
    log.info("[cv] test %s~%s，%d 折", test_days[0].date(), test_days[-1].date(), len(folds))

    pos = {d: i for i, d in enumerate(dates)}
    valid_2d = close_adj.notna().values
    n_codes = len(codes)
    arms = _build_arms(args)
    preds = {k: pd.DataFrame(np.nan, index=test_days, columns=codes) for k in arms}

    for fi, fold in enumerate(folds, 1):
        tr_days, te_days = fold.train_days, fold.test_days
        if len(tr_days) < args.min_train_days or len(te_days) == 0:
            continue
        tri = np.array([pos[d] for d in tr_days])
        tei = np.array([pos[d] for d in te_days])
        y_tr = labels.loc[tr_days].values.ravel().astype(np.float32)
        ok = valid_2d[tri].ravel() & ~np.isnan(y_tr)
        rows_tr = row_idx(tri, n_codes)[ok]
        rows_te = row_idx(tei, n_codes)
        log.info("[fold %d/%d] train %s~%s (%d 日, 有效 %d/%d) -> test %s~%s",
                 fi, len(folds), tr_days[0].date(), tr_days[-1].date(), len(tr_days),
                 len(rows_tr), len(valid_2d[tri].ravel()),
                 te_days[0].date(), te_days[-1].date())

        yv = y_tr[ok]
        for name, (kind, kw) in arms.items():
            src = flat_parts[kw["src"]]
            flat = src.reshape(-1, *src.shape[2:])
            Xtr, Xte = flat[rows_tr], flat[rows_te]
            if kind == "gbdt":
                m = GbdtArm(**{k: v for k, v in kw.items() if k != "src"})
                m.fit(Xtr.reshape(len(rows_tr), -1), yv)
                p = m.predict(Xte.reshape(len(rows_te), -1))
            elif kind == "seq":
                m = SeqArm(n_lags=src.shape[2],
                           **{k: v for k, v in kw.items() if k != "src"}).fit(Xtr, yv)
                p = m.predict(Xte)
            else:
                raise ValueError(kind)
            preds[name].loc[te_days] = panel_from_vec(p, te_days, codes)
            del Xtr, Xte

    rows = []
    for name, pred in preds.items():
        pz = cs_z(pred)
        ev = evaluate_model(pz, fwd.loc[pred.index])
        rows.append({"arm": name, "ic_mean": ev["ic_mean"], "ic_ir": ev["ic_ir"],
                     "ic_t_nw": ev["ic_t_nw"],
                     "n_days": int(ev["ic_series"].notna().sum())})
    table = pd.DataFrame(rows).sort_values("ic_mean", ascending=False)
    return table, preds, fwd, test_days


def _build_arms(args) -> dict:
    """返回 {name: (kind, kwargs)}；kwargs 里 `src` 指向 flat_parts 的键。"""
    arms: dict = {}
    if args.seq_lags > 0:
        # Q2：同输入（prod 因子 × T 日窗口）下 gbdt vs 序列模型
        arms["gbdt_prod_seq"] = ("gbdt", dict(src="prod_seq", seed=0,
                                              n_estimators=args.n_estimators,
                                              threads=args.threads))
        for arch in [a.strip() for a in args.archs.split(",") if a.strip()]:
            arms[f"{arch}_prod_seq"] = ("seq", dict(src="prod_seq", arch=arch, seed=0,
                                                    threads=args.threads,
                                                    hidden=args.hidden,
                                                    epochs=args.epochs))
    else:
        arms[f"gbdt_prod{args.factor_set}"] = ("gbdt", dict(
            src="prod", seed=0, n_estimators=args.n_estimators, threads=args.threads))
    if args.with_a360:
        arms["gbdt_a360"] = ("gbdt", dict(src="a360", seed=0,
                                          n_estimators=args.n_estimators,
                                          threads=args.threads))
    return arms


def main():
    ap = argparse.ArgumentParser(description="P1b 生产基线同口径复验")
    ap.add_argument("--index", default="000300.SH", help="池（PIT 成分）")
    ap.add_argument("--begin", type=int, default=20180101)
    ap.add_argument("--end", type=int, default=20260831)
    ap.add_argument("--warmup", type=int, default=20160601)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--n-lags", type=int, default=60, help="alpha360 序列长度")
    ap.add_argument("--n-folds", type=int, default=2)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--min-train-days", type=int, default=240)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--factor-set", default="prod50", choices=["prod50", "all"])
    ap.add_argument("--select-year", type=int, default=2026,
                    help="读哪一年的 DPP 选择结果（默认 2026）")
    ap.add_argument("--with-a360", action="store_true", default=True,
                    help="同时跑 alpha360 gbdt（P1 基线复现校验）")
    ap.add_argument("--no-a360", dest="with_a360", action="store_false")
    ap.add_argument("--seq-lags", type=int, default=0,
                    help=">0 时把 prod 因子做成 T 日窗口序列，跑 Q2（同输入 gbdt vs 序列）")
    ap.add_argument("--archs", default="gru", help="--seq-lags>0 时的序列臂")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-tag", default="hs300_h5")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    t0 = time.time()
    table, preds, fwd, test_days = run(args)
    print(f"\n===== P1b 生产基线对照（test {test_days[0].date()}~{test_days[-1].date()}，"
          f"h={args.horizon}，池={args.index}）=====")
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.4f}"):
        print(table.to_string(index=False))
    print(f"耗时 {time.time() - t0:.0f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / f"arms_{args.out_tag}.csv", index=False)
    for name, pred in preds.items():
        cs_z(pred).to_parquet(OUT_DIR / f"pred_{name}_{args.out_tag}.parquet")
    (OUT_DIR / f"meta_{args.out_tag}.json").write_text(json.dumps({
        "index": args.index, "horizon": args.horizon, "factor_set": args.factor_set,
        "n_factors": len(prod_factor_names(args)), "test_days": len(test_days),
        "test_begin": str(test_days[0].date()), "test_end": str(test_days[-1].date()),
        "elapsed_s": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
