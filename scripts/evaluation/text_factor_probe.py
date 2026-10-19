"""论文族 `pap_*` / 情绪族 `txt_*` 的生产口径收尾评估（P0）。

## 定位

TODO「建议推进顺序」第 4 项：论文族 21 + 情绪族 5 的**低成本收尾评估**，
明确限定为「预期上界 +1~3% IR，不作新增长极」。

## ⚠️ 口径修正（本脚本的第一个产出）

此前 TODO / `reports/consensus_availability` 记为「registry 有 / **面板无**
⇒ 从未评估」。实测**不成立**：这 26 个因子的面板以**哈希名**落在
``<ds>/panels/<hash>.parquet``（``registry.panel_path`` 指向它，实体存在），
``<ds>/evals/<hash>.parquet`` 亦有逐日 IC + 净值，registry 的 IC/IR/t 列**已填**。

真正的缺口不是「没评估」，而是**没进生产池**：``panels_neu/``（949 =
975 − 论文 21 − 情绪 5）不含它们 ⇒ 从未中性化、从未与量价池做过 DPP 竞争。
故本脚本的定位 = **把它们放进生产口径判一次**。

## 判据（`reports/four_paths/README.md` §三）

> 新因子族准入门槛 = **自身可交易 IR ≥ 0.3**（组合层 Sharpe 尺度，与主策略
> S₁ = 0.714 同尺度），**不是**「与主模型低相关」。

故本脚本报两个尺度、不混用：
- **IC 尺度**：逐日 IC 均值 + iid t 值（``ic_summary.csv``）；
- **组合尺度**：`VectorBacktest` + 月频 TopFrac0.10 / TopK45 的年化与 Sharpe
  （``portfolios.csv``）—— 判据只看这一列。

## 与既有探针的关系

判据件（``masked_ic`` / ``fuse_rank`` / ``ic_stats``）与回测口径**逐项复用**
``scripts/evaluation/news_alpha_probe.py``、``pead_alpha_probe.py``，
主信号同为 ``reports/alla_rolling_ortho/pred/ens_h1h5__h1.parquet``
⇒ 与 PEAD / news 两次探针**同尺度可比**。

## 路径（2026-10-13 起）

因子库根走 ``Config.paths()["factor_library_root"]``（多机适配的单一真源，
本机经 `config/settings_local.yaml` 解析到 E:）。**不再探测盘符候选。**

## ⚠️ 内存约束

前台 shell 的进程内存被限在 ~1.3 GB（后台约 4 GB，本机物理 31 GB
但空闲另论）⇒ 全部面板不得同时驻留，本脚本**逐因子读 → 算 → 释放**。

用法::

    python -m scripts.evaluation.text_factor_probe                     # txt+pap 全量
    python -m scripts.evaluation.text_factor_probe --sets txt          # 只情绪族
    python -m scripts.evaluation.text_factor_probe --only txt_senti_res
    python -m scripts.evaluation.text_factor_probe --skip-portfolio    # 只跑 IC 三层
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("text_factor_probe")

DATASET = "all_a_2018_2026"


def resolve_ds_root(explicit: str | None) -> Path:
    """定位因子库数据集根：显式路径 > `Config.paths()` 单一真源。"""
    from config import Config

    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    root = Config.paths().get("factor_library_root")
    if root:
        cands.append(Path(root) / DATASET)
    for c in cands:
        if (c / "registry.csv").exists():
            log.info("因子库根 = %s", c)
            return c
    raise FileNotFoundError(
        f"因子库 root 不可用：{[str(c) for c in cands]}；"
        "检查 config/settings_local.yaml 的 factor_library.root")


# ---------------------------------------------------------------------------
# 中性化
# ---------------------------------------------------------------------------
def neutralize_panel(panel: pd.DataFrame, mc: pd.DataFrame, ind: pd.DataFrame,
                     name: str) -> tuple[pd.DataFrame, dict]:
    """按面板形态分派中性化（复用 altf 回填脚本判定，防 MAD 静默死面板）。

    - 零膨胀（行内 0 占比中位 > 0.5）⇒ 跳过 MAD，其余照常；
    - 市场级广播（行 nunique ≤ 1）⇒ 截面中性化无定义，**原样返回**只告警；
    - 其余 ⇒ MAD → 行业+市值中性化 → zscore。
    """
    from factor.preprocessing import preprocess_factor
    from scripts.builders.build_alla_altfactor_neu import classify

    mode, diag = classify(panel)
    if mode == "broadcast":
        log.warning("%s 判为市场级广播（行 nunique≤1），跳过中性化", name)
        return panel.astype(np.float32), {"mode": mode, **diag}

    common = mc.index.intersection(panel.index)
    p = panel.reindex(index=common)
    m = mc.reindex(index=common, columns=p.columns)
    g = ind.reindex(index=common, columns=p.columns)
    winsorize = None if mode == "zero_inflated" else "mad"
    x = preprocess_factor(p, market_cap_panel=m, industry_panel=g,
                          winsorize=winsorize)
    x = x.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return x.clip(-10.0, 10.0), {"mode": mode, **diag}


# ---------------------------------------------------------------------------
# IC 判据
# ---------------------------------------------------------------------------
def residual_panel(sig_rank: pd.DataFrame,
                   main_rank: pd.DataFrame) -> pd.DataFrame:
    """逐日把 `sig_rank` 对 `main_rank` 做一元 OLS，返回残差面板（全向量化）。

    残差**不重排秩**——非单调变换会改掉 spearman 口径，故残差 IC 用同一套
    `masked_ic` 计算，与原始 IC 严格可比。
    """
    mm = main_rank.reindex(index=sig_rank.index, columns=sig_rank.columns)
    v = sig_rank.notna() & mm.notna()
    a = sig_rank.where(v)
    b = mm.where(v)
    ya = a.sub(a.mean(axis=1), axis=0)
    yb = b.sub(b.mean(axis=1), axis=0)
    var = (yb ** 2).mean(axis=1)
    cov = (ya * yb).mean(axis=1)
    beta = cov / var.replace(0.0, np.nan)
    return ya.sub(yb.mul(beta, axis=0)).where(v)


def _ic_row(sig: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame,
            prefix: str) -> dict:
    """全样本 IC / 可交易 IC，均含 iid t 值（t = 日 IC 均值/标准差 × √n）。

    ⚠️ `ic_stats` 返回的 `ir` **是日频比值 mean/std，不是年化 IR**，
    故 t 值 = `ir × √n`；误乘 √252 会把 t 缩小 16 倍（本脚本首轮踩过）。
    """
    from scripts.evaluation.news_alpha_probe import ic_stats, masked_ic
    from stats.ic import calc_ic_series

    def _t(st: dict) -> float:
        n = st.get("n", 0)
        if not n or st.get("ir") != st.get("ir"):
            return float("nan")
        return float(st["ir"]) * np.sqrt(n)

    full = ic_stats(calc_ic_series(sig, fwd))
    trad = ic_stats(masked_ic(sig, fwd, mask))
    out = {
        f"{prefix}_full_ic": full["ic"], f"{prefix}_full_t": _t(full),
        f"{prefix}_trad_ic": trad["ic"], f"{prefix}_trad_t": _t(trad),
        f"{prefix}_trad_win": trad["win"], f"{prefix}_trad_n": trad["n"],
    }
    del full, trad
    return out


# ---------------------------------------------------------------------------
# 组合层
# ---------------------------------------------------------------------------
def run_arms(arms: list[tuple[str, pd.DataFrame]], fwd_engine: pd.DataFrame,
              mask: pd.DataFrame, bench_eqw: pd.Series, bench_index: pd.Series,
              oos_days: pd.DatetimeIndex, holdings: list[str]) -> list[dict]:
    """逐臂回测；信号**按自身列宽**对齐（不回填到全 A 宽度，省内存）。"""
    from backtest.costs import default_costs
    from backtest.engine import VectorBacktest
    from scripts.common.alla_common import ann, res_metrics
    from strategy.examples import TopFracLongOnly, TopKLongOnly

    factory = {
        "TopFrac0.10": lambda: TopFracLongOnly(frac=0.10, weight_mode="equal"),
        "TopK45": lambda: TopKLongOnly(k=45),
    }
    costs = default_costs()
    b_eqw = bench_eqw.reindex(oos_days).fillna(0.0)
    b_idx = bench_index.reindex(oos_days).fillna(0.0)
    rows: list[dict] = []
    for hold_tag in holdings:
        for tag, sig in arms:
            sig = sig.reindex(index=oos_days)
            mk = mask.reindex(index=oos_days, columns=sig.columns).fillna(False)
            bt = VectorBacktest(strategy=factory[hold_tag](), rebalance_freq="M",
                                initial_capital=1_000_000.0, costs=costs)
            res = bt.run(sig, fwd_engine, executable_mask=mk.astype(bool),
                         horizon=1, rebalance_days=None)
            m = res_metrics(res.daily_returns,
                            b_idx.reindex(res.daily_returns.index).fillna(0.0),
                            res.turnover_series)
            rows.append(dict(hold=hold_tag, arm=tag, annual=m["annual"],
                             excess_eqw=m["annual"] - ann(b_eqw),
                             sharpe=m["sharpe"], ir=m["ir"],
                             excess_t=m["excess_t_stat"], max_dd=m["max_dd"],
                             turnover=m["turnover"]))
            del sig, mk, res
    del b_eqw, b_idx
    return rows


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", default="txt,pap", help="逗号分隔：txt / pap")
    ap.add_argument("--only", nargs="+", default=None, help="只评指定因子名")
    ap.add_argument("--ds-root", default=None, help="因子库数据集根（默认探测）")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho/pred")
    ap.add_argument("--main-signal", default="ens_h1h5__h1.parquet")
    ap.add_argument("--out", default="reports/text_factor_probe")
    ap.add_argument("--skip-portfolio", action="store_true")
    ap.add_argument("--fuse-weights", default="0.5,0.7,0.8,0.9")
    args = ap.parse_args(argv)

    from scripts.common.alla_common import ROLL_DIR as ROLL_OUT

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ds = resolve_ds_root(args.ds_root)
    reg = pd.read_csv(ds / "registry.csv").set_index("name")
    if args.only:
        names = list(args.only)
    else:
        want = [s.strip() for s in args.sets.split(",") if s.strip()]
        prefixes = tuple(p if p.endswith("_") else p + "_" for p in want)
        names = sorted(n for n in reg.index
                       if n.startswith(prefixes)
                       and isinstance(reg.loc[n, "panel_path"], str))
    log.info("待评估 %d 个：%s", len(names), names)

    bdir = ROLL_OUT / "_base"
    close = pd.read_parquet(bdir / "close_adj.parquet")
    cols = close.columns
    mask = pd.read_parquet(bdir / "tradable_mask.parquet").astype(bool)
    mc = pd.read_parquet(bdir / "market_cap.parquet")
    ind = pd.read_parquet(bdir / "cov_industry.parquet")
    bench_eqw = pd.read_parquet(bdir / "bench_eqw.parquet")["ret"]
    bench_index = pd.read_parquet(bdir / "bench_index.parquet")["ret"]

    main = pd.read_parquet(Path(args.pred_dir) / args.main_signal)
    oos_days = main.index
    log.info("主信号 %s: %d 日 | OOS %d 日（%s ~ %s）", args.main_signal,
             main.shape[0], len(oos_days), oos_days[0].date(), oos_days[-1].date())

    labels = close.pct_change(args.horizon, fill_method=None).shift(-args.horizon)
    fwd_all = labels.reindex(index=oos_days).astype(np.float32)
    mask_all = mask.reindex(index=oos_days)
    main_all = main.reindex(index=oos_days, columns=cols).astype(np.float32)
    del labels
    gc.collect()

    fwd_engine = close.pct_change(fill_method=None).loc[oos_days].astype(np.float32)
    del close
    gc.collect()

    holdings = ["TopFrac0.10", "TopK45"]
    rows: list[dict] = []
    rows2: list[dict] = []
    rows3: list[dict] = []
    if not args.skip_portfolio:
        rows2 = run_arms([("主模型 only", main_all)], fwd_engine, mask, bench_eqw,
                          bench_index, oos_days, holdings)
        if any(n.startswith("txt_") for n in names):
            rows3 = run_arms([("主模型 only", main_all)], fwd_engine, mask,
                              bench_eqw, bench_index, oos_days, ["TopFrac0.10"])

    for i, n in enumerate(names, 1):
        raw = pd.read_parquet(Path(str(reg.loc[n, "panel_path"])))
        cn = raw.columns.intersection(cols)
        sig = raw.reindex(index=oos_days, columns=cn).replace(
            [np.inf, -np.inf], np.nan).astype(np.float32)
        fwd_n = fwd_all[cn]
        mask_n = mask_all[cn].fillna(True)
        main_n = main_all[cn]
        main_rank_n = main_n.rank(axis=1, pct=True)

        row: dict = {"name": n, "set": reg.loc[n, "set"],
                     "cov": float(sig.notna().to_numpy(dtype=float).mean())}
        row["rank_corr_main"] = float(
            sig.rank(axis=1, pct=True).corrwith(main_rank_n, axis=1).mean())
        row.update(_ic_row(sig, fwd_n, mask_n, "raw"))
        del main_n

        neu, diag = neutralize_panel(raw, mc, ind, n)
        neu = neu.reindex(index=oos_days, columns=cn).astype(np.float32)
        row.update(_ic_row(neu, fwd_n, mask_n, "neu"))
        resid = residual_panel(sig.rank(axis=1, pct=True),
                               main_rank_n).astype(np.float32)
        row.update(_ic_row(resid, fwd_n, mask_n, "res"))
        row.update({"mode": diag["mode"], "zero_frac": diag["zero_frac_med"],
                    "neu_cov": float(neu.notna().to_numpy(dtype=float).mean())})
        rows.append(row)
        log.info("[%2d/%d] %-18s cov=%.2f | 全样本IC=%+.4f t=%.2f | 可交易IC=%+.4f"
                 " t=%.2f | 中性化后 IC=%+.4f t=%.2f | 残差IC=%+.4f t=%.2f | "
                 "corr(main)=%+.3f | mode=%s",
                 i, len(names), n, row["cov"], row["raw_full_ic"], row["raw_full_t"],
                 row["raw_trad_ic"], row["raw_trad_t"], row["neu_trad_ic"],
                 row["neu_trad_t"], row["res_trad_ic"], row["res_trad_t"],
                 row["rank_corr_main"], row["mode"])
        del resid, main_rank_n, fwd_n, mask_n
        gc.collect()

        if not args.skip_portfolio:
            rows2.extend(run_arms([(f"{n}·raw", sig), (f"{n}·neu", neu)],
                                   fwd_engine, mask, bench_eqw, bench_index,
                                   oos_days, holdings))
            if n.startswith("txt_"):
                from scripts.evaluation.news_alpha_probe import fuse_rank
                main_sub = main_all[cn]
                for w in [float(x) for x in args.fuse_weights.split(",") if x.strip()]:
                    fused = fuse_rank(main_sub, neu, w, neu.notna())
                    rows3.extend(run_arms([(f"主⊕{n} w={w:.1f}", fused)],
                                           fwd_engine, mask, bench_eqw,
                                           bench_index, oos_days, ["TopFrac0.10"]))
                    del fused
                    gc.collect()
                del main_sub
            pd.DataFrame(rows2).to_csv(dest / "portfolios.csv", index=False,
                                       encoding="utf-8-sig")

        del raw, sig, neu
        gc.collect()

    pd.DataFrame(rows).to_csv(dest / "ic_summary.csv", index=False,
                              encoding="utf-8-sig")
    if args.skip_portfolio:
        log.info("完成（--skip-portfolio） %.0fs -> %s", time.time() - t0, dest)
        return
    pd.DataFrame(rows2).to_csv(dest / "portfolios.csv", index=False,
                               encoding="utf-8-sig")
    if rows3:
        pd.DataFrame(rows3).to_csv(dest / "fusion.csv", index=False,
                                   encoding="utf-8-sig")
    log.info("完成 %.0fs -> %s", time.time() - t0, dest)


if __name__ == "__main__":
    main()
