"""A2 二阶 GBDT：残差收益目标 walk-forward + 融合组合层（2026-10-19）。

## 定位

`TODO.md` §六 A2 下一步的第一臂。残差 IC 首轮结论：量价残差 = 已有量价的
非线性成分（top200 里 55/60 过 t≥3），非量价残差集中在负向。本脚本检验
「这些残差方向能不能被一个二阶 GBDT 组合出可交易的增量」：

1. **特征池（无选择前视）**：从首轮 CSV 的**早年列**（ic_2018~ic_2023）重排
   |t| 榜（首轮 `both_nw_t` 是全期 in-sample，直接用它挑特征会把选择前视
   带进 OOS）；量价 top35 + 非量价 |early_t|≥1.5 全部；
2. **DPP 去冗余**到 40 个（相关矩阵用 2023 年前采样窗，`research.dpp_selection`）；
3. **标签 = 残差收益**：h1 前瞻收益 rank 对主信号 rank 逐日一元 OLS 残差化
   （与 `residual_ic_probe` 的 `both` 口径同实现，测试已锁逐位一致）；
4. **walk-forward**：500 日窗 / embargo 1 / 季度重训（复用
   `retrain_freq_ic.rolling_pred`，OOS 2024-01-02~2026-09-01 与 A1 探针对齐）；
5. **组合层裁决**：主信号 ⊕ α·rank(stage2)，α∈{0.25, 0.50}，production 口径
   `BufferedTopFracLongOnly(0.15,0.40)`；**安慰剂** = stage2 pred 时间循环平移
   250 日后同 α 融合（ΔSharpe 应≈0）。

## 已知边界

- 特征池/去冗余用 2023 年前数据（无选择前视），但超参（α 档位、窗口、refit）
  是项目既有口径，未为本实验重调；
- 主信号 882 ortho（本机唯一 pred）；920 定版复验挂好机器；
- 标签残差化把主信号线性成分拿掉 ⇒ 二阶模型对**原始收益**的 IC 应显著低于
  主信号——它只在融合时提供正交增量，单独不构成策略。

## 用法

    python -m scripts.evaluation.residual_stack_probe
    python -m scripts.evaluation.residual_stack_probe --alphas 0.25,0.50
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("residual_stack_probe")

PV_TOP = 35          # 量价早年榜取前 N
NONPV_MIN_T = 1.5    # 非量价早年 |t| 门槛
K_FINAL = 40         # DPP 后特征数
YEAR_END = "2023-12-31"
OOS = ("20240102", "20260901")


def resid_rank(sig_rank: pd.DataFrame, main_rank: pd.DataFrame) -> pd.DataFrame:
    """逐日一元 OLS 残差；与 residual_ic_probe.resid_rank 同实现（测试锁一致）。"""
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


def early_table(csvs: list[Path]) -> pd.DataFrame:
    """合并首轮 CSV，按早年列（ic_2018~ic_2023）重算 |t| 榜。"""
    frames = []
    for c in csvs:
        if c.exists():
            frames.append(pd.read_csv(c))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["name"])
    early = [c for c in df.columns
             if re.fullmatch(r"ic_20(18|19|2[0-3])", c)]
    df["early_ic"] = df[early].mean(axis=1)
    df["early_t"] = df["early_ic"] / (df[early].std(axis=1)
                                      / np.sqrt(len(early)))
    df["ysc"] = df[["years_pos", "years_n"]].apply(
        lambda r: max(int(r.years_pos), int(r.years_n - r.years_pos)), axis=1)
    return df


def dedup_family(names: pd.Series) -> list[str]:
    """粗去族：把 `PREFIX<数字>` 系列归组只留首个（如 alpha360_VOLUME41..59）。"""
    seen: dict[str, str] = {}
    out: list[str] = []
    for n in names:
        key = re.sub(r"(VOLUME|VSUM[DPN]?|MA\d+|MIN\d+|KLOW|KLEN)\d+$",
                     r"\1", n)
        if key in seen:
            continue
        seen[key] = n
        out.append(n)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho/pred")
    ap.add_argument("--main-signal", default="ens_h1h5__h1.parquet")
    ap.add_argument("--panels-dir", default="panels_neu")
    ap.add_argument("--alphas", default="0.25,0.50")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--refit", type=int, default=60)
    ap.add_argument("--placebo-shift", type=int, default=250)
    ap.add_argument("--out",
                    default="reports/frontier_audit_2026-10-19/residual_stack")
    args = ap.parse_args(argv)

    from model.params import DEFAULT_MODEL_PARAMS
    from research.dpp_selection import corr_matrix, dpp_select
    from scripts.common.alla_common import ds_root, load_base
    from scripts.evaluation.caliber import (
        cost_profile,
        evaluate_strategy,
        get_profile,
        load_context,
    )
    from scripts.evaluation.retrain_freq_ic import rolling_pred
    from stats.ic import calc_ic_series

    t0 = time.time()
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    audit = Path("reports/frontier_audit_2026-10-19")
    pool_tab = early_table([audit / "residual_ic_h1_top200.csv",
                            audit / "residual_ic_h1_nonpv_s3.csv"])
    pv = pool_tab[pool_tab["name"].str.startswith(("alpha", "gtja"))]
    nonpv = pool_tab[~pool_tab["name"].str.startswith(("alpha", "gtja"))]
    pv_pool = dedup_family(pv.sort_values("early_t", key=abs,
                                          ascending=False)["name"])
    pv_pool = pv_pool[:PV_TOP]
    nonpv_pool = list(nonpv[nonpv["early_t"].abs() >= NONPV_MIN_T]["name"])
    pool = pv_pool + nonpv_pool
    log.info("特征池 %d = 量价 %d（去族后 top%d）+ 非量价 %d（|t|≥%.1f）",
             len(pool), len(pv_pool), PV_TOP, len(nonpv_pool), NONPV_MIN_T)

    base = load_base()
    close, mask_all = base["close"], base["mask"].astype(bool)
    all_days = close.index
    main = pd.read_parquet(Path(args.pred_dir) / args.main_signal)
    main = main.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    cols = main.columns.intersection(close.columns)

    cut = pd.Timestamp(YEAR_END)
    sel_days = all_days[all_days <= cut][-500:][::4]
    sample_codes = cols[::2]
    panels: dict[str, pd.DataFrame] = {}
    pdir = ds_root() / args.panels_dir
    for n in pool:
        x = pd.read_parquet(pdir / f"{n}.parquet")
        if not isinstance(x.index, pd.DatetimeIndex):
            x.index = pd.to_datetime(x.index)
        panels[n] = (x.reindex(index=sel_days, columns=sample_codes)
                     .astype(np.float32))
        del x
    corr = corr_matrix(panels, method="cross")
    qual = pool_tab.set_index("name")["early_t"].abs().reindex(corr.index)
    sel = dpp_select(corr, k=min(K_FINAL, len(corr.index)),
                     quality=qual)["selected"]
    log.info("DPP 选中 %d 个（%.0fs）", len(sel), time.time() - t0)
    (dest / "features.json").write_text(
        json.dumps({"pool": pool, "selected": sel}, ensure_ascii=False,
                   indent=2), encoding="utf-8")
    del panels

    # ---- 标签：残差收益（h1，both 口径）----
    fwd = (close.pct_change(1, fill_method=None).shift(-1)
           .reindex(columns=cols)
           .replace([np.inf, -np.inf], np.nan).astype(np.float32))
    mask = mask_all.reindex(columns=cols).fillna(False)
    main_r = main.reindex(columns=cols)
    v = fwd.notna() & main_r.notna()
    labels = resid_rank(fwd.where(v).rank(axis=1),
                        main_r.where(v).rank(axis=1)).astype(np.float32)
    log.info("残差标签构建完成（覆盖 %.3f）",
             float(labels.notna().to_numpy(dtype=float).mean()))
    del fwd

    # ---- 特征加载（裁剪到训练+OOS 最小范围）----
    b, e = pd.Timestamp(OOS[0]), pd.Timestamp(OOS[1])
    oos_days = all_days[(all_days >= b) & (all_days <= e)]
    i0 = int(all_days.get_indexer([oos_days[0]])[0])
    keep = all_days[max(0, i0 - args.window - 21):]
    feats: dict[str, pd.DataFrame] = {}
    for n in sel:
        x = pd.read_parquet(pdir / f"{n}.parquet")
        feats[n] = (x.reindex(index=keep, columns=cols)
                    .replace([np.inf, -np.inf], np.nan).astype(np.float32))
        del x
    log.info("特征 %d 张 × %d 日（%s ~ %s）", len(feats), len(keep),
             keep[0].date(), keep[-1].date())

    params = dict(DEFAULT_MODEL_PARAMS["gbdt"])
    raw, n_fit = rolling_pred(feats, labels, oos_days, all_days,
                              args.window, 1, 250, args.refit, params, {})
    stage2 = raw.where(mask.reindex(index=oos_days))
    labels_oos = labels.reindex(index=oos_days)   # IC 评估用，del 前抽出
    del feats, labels, raw          # 组合层前释放（峰值内存控制）
    log.info("二阶模型训练完成（%d 折，%.0fs）", n_fit, time.time() - t0)
    stage2.astype(np.float32).to_parquet(dest / "stage2_pred.parquet")

    # ---- 二阶模型对残差收益 / 原始收益的 OOS IC ----
    fwd_oos = (close.pct_change(1, fill_method=None).shift(-1)
               .reindex(index=oos_days, columns=cols)
               .replace([np.inf, -np.inf], np.nan).astype(np.float32))
    mask_oos = mask.reindex(index=oos_days)
    ic_res = calc_ic_series(stage2, labels_oos, tradable=mask_oos)
    ic_raw = calc_ic_series(stage2, fwd_oos, tradable=mask_oos)
    rows = [{"item": "stage2_ic_on_residual", "ic": float(ic_res.mean()),
             "ir": float(ic_res.mean() / ic_res.std()),
             "win": float((ic_res > 0).mean()), "n": int(ic_res.notna().sum())},
            {"item": "stage2_ic_on_raw", "ic": float(ic_raw.mean()),
             "ir": float(ic_raw.mean() / ic_raw.std()),
             "win": float((ic_raw > 0).mean()), "n": int(ic_raw.notna().sum())}]
    log.info("stage2 IC(残差收益)=%.4f IR=%.2f | IC(原始收益)=%.4f",
             rows[0]["ic"], rows[0]["ir"], rows[1]["ic"])

    # ---- 融合组合层（production 口径）----
    prof = get_profile("production")
    ctx = load_context(pred_path=Path(args.pred_dir) / args.main_signal,
                       window="prod", begin=int(OOS[0]),
                       execution=prof.execution)
    cost = cost_profile(prof.cost)
    main_rank = ctx.pred.rank(axis=1, pct=True).astype(np.float32)
    s2 = stage2.reindex(index=ctx.pred.index, columns=ctx.pred.columns)
    s2_rank = s2.rank(axis=1, pct=True).astype(np.float32)
    s2_placebo = pd.DataFrame(
        np.roll(s2_rank.to_numpy(), args.placebo_shift, axis=0),
        index=s2_rank.index, columns=s2_rank.columns)

    from strategy.examples import BufferedTopFracLongOnly

    arms: list[tuple[str, pd.DataFrame | None]] = [("main", None)]
    for a in [float(x) for x in args.alphas.split(",")]:
        arms.append((f"fuse_a{int(round(a*100)):03d}",
                     ((1 - a) * main_rank + a * s2_rank).astype(np.float32)))
        arms.append((f"plac_a{int(round(a*100)):03d}",
                     ((1 - a) * main_rank + a * s2_placebo)
                     .astype(np.float32)))
    pf = []
    for name, sig in arms:
        m = evaluate_strategy(BufferedTopFracLongOnly(0.15, 0.40), ctx, cost,
                              prof.capital, signal=sig, yearly=True)
        pf.append({"arm": name, **m, "sec": 0})
        log.info("组合 | %-12s ann=%.2f%% sharpe=%.3f turn=%.1f%%",
                 name, 100 * m["ann"], m["sharpe"], 100 * m["turnover"])

    pd.DataFrame(rows).to_csv(dest / "stage2_ic.csv", index=False,
                              encoding="utf-8-sig")
    pf_df = pd.DataFrame(pf)
    ref = pf_df[pf_df.arm == "main"].iloc[0]
    for c in ("ann", "sharpe", *[c for c in pf_df.columns
                                 if c.startswith("yearly_ret_")]):
        pf_df[f"d_{c.replace('yearly_ret_', 'yr')}"] = pf_df[c] - ref[c]
    pf_df.to_csv(dest / "portfolio.csv", index=False, encoding="utf-8-sig")
    meta = {"oos": list(OOS), "window": args.window, "refit": args.refit,
            "alphas": args.alphas, "n_features": len(sel),
            "n_pool": len(pool), "n_fits": n_fit,
            "main_signal": args.main_signal, "panels_dir": args.panels_dir,
            "selection": ("特征池按早年列 ic_2018~ic_2023 的 |t| 重排"
                          "（无选择前视）；DPP 相关矩阵亦只用 2023 前窗"),
            "elapsed_s": round(time.time() - t0, 1)}
    (dest / "meta.json").write_text(json.dumps(meta, ensure_ascii=False,
                                               indent=2), encoding="utf-8")
    log.info("完成 %.0fs -> %s", time.time() - t0, dest)


if __name__ == "__main__":
    main()
