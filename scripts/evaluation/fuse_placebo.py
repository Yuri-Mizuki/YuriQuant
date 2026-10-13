"""融合增益的**安慰剂检验**：区分「真信号」与「覆盖率掩码 / 暴露」假象。

## 为什么需要它

`fuse_rank(main, aux, w, cov)` 只在 ``aux`` 非 NaN 的股票上调整排名，其余保持
纯 ``rank(main)``。当 ``aux`` 是**覆盖不均匀**的族（研报类覆盖 = 有分析师覆盖的
股票 ≈ 大市值/热门股；alt/news 族同理）时，**任何**带此掩码的融合都会把权重
从小盘移向大盘 —— 从而降低波动与回撤、抬高 Sharpe，**与 aux 的取值内容无关**。

实测到过这个假象：`text_factor_probe` 里一个自身可交易 IR 仅 **0.008** 的因子
（`txt_report_score`，IC t=0.33）融合后 Sharpe +0.116。按组合数学
（ρ≈0 时 ΔS ≈ S₂²/(2S₁)）应只有 +0.00005 ⇒ 必然不是信号。

## 做法

安慰剂 = 把 ``aux`` 的**逐行截面值在「被覆盖的列」内随机置换**：
保留 ① 覆盖了哪些股票（掩码不变）② 值的截面分布，只**破坏值与未来收益的关联**。
用同一融合 + 同一回测重复 ``--reps`` 次，与真实融合对照。

- 真实 ΔSharpe 落在安慰剂分布内（≲2σ）⇒ **改善来自掩码/暴露，不是信号**；
- 真实显著超出安慰剂分布 ⇒ 才有真信号成分（仍需过 IR ≥ 0.3 准入门槛）。

用法::

    python -m scripts.evaluation.fuse_placebo                       # 默认 txt_* 全族
    python -m scripts.evaluation.fuse_placebo --only txt_report_score --reps 20
    python -m scripts.evaluation.fuse_placebo --auxs news_alpha     # 换族（需自行补载入）

产物：``placebo.csv``（逐次）+ ``placebo_summary.csv``（真实 vs 安慰剂均值/σ）。
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

from scripts.common.cli_common import setup_logging  # noqa: E402
from scripts.evaluation.news_alpha_probe import fuse_rank  # noqa: E402
from scripts.evaluation.text_factor_probe import (  # noqa: E402
    _run_arms,
    neutralize_panel,
    resolve_ds_root,
)

log = setup_logging("fuse_placebo")


def row_shuffle_among_covered(neu: pd.DataFrame, cov: pd.DataFrame,
                              rng: np.random.Generator) -> pd.DataFrame:
    """逐行在 ``cov==True`` 的列内随机置换 ``neu`` 的值（其余保持 NaN）。

    保留覆盖掩码与截面分布，破坏与未来收益的关联。
    """
    vals = neu.to_numpy(dtype=np.float32, copy=True)
    c = cov.to_numpy(dtype=bool)
    for i in range(vals.shape[0]):
        idx = np.flatnonzero(c[i] & ~np.isnan(vals[i]))
        if idx.size > 1:
            vals[i, idx] = vals[i, idx][rng.permutation(idx.size)]
    return pd.DataFrame(vals, index=neu.index, columns=neu.columns)


def _default_txt_names(ds_root: Path) -> list[str]:
    reg = pd.read_csv(ds_root / "registry.csv")
    return sorted(n for n in reg["name"] if str(n).startswith("txt_"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", default=None,
                    help="辅助因子名（默认全 txt_*）")
    ap.add_argument("--ds-root", default=None, help="因子库数据集根（默认单一真源）")
    ap.add_argument("--weights", default="0.5,0.7,0.8,0.9")
    ap.add_argument("--reps", type=int, default=15, help="安慰剂重复次数")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--hold", default="TopFrac0.10")
    ap.add_argument("--out", default="reports/text_factor_probe/placebo")
    args = ap.parse_args(argv)

    from scripts.pipelines.rolling_grid_alla import OUT as ROLL_OUT

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    ds = resolve_ds_root(args.ds_root)
    names = list(args.only) if args.only else _default_txt_names(ds)
    reg = pd.read_csv(ds / "registry.csv").set_index("name")
    log.info("安慰剂检验 %d 个辅助因子：%s", len(names), names)

    bdir = ROLL_OUT / "_base"
    close = pd.read_parquet(bdir / "close_adj.parquet")
    cols = close.columns
    mask = pd.read_parquet(bdir / "tradable_mask.parquet").astype(bool)
    mc = pd.read_parquet(bdir / "market_cap.parquet")
    ind = pd.read_parquet(bdir / "cov_industry.parquet")
    bench_eqw = pd.read_parquet(bdir / "bench_eqw.parquet")["ret"]
    bench_index = pd.read_parquet(bdir / "bench_index.parquet")["ret"]

    main_path = Path("reports/alla_rolling_ortho/pred") / "ens_h1h5__h1.parquet"
    main = pd.read_parquet(main_path)
    oos_days = main.index
    fwd_engine = close.pct_change(fill_method=None).loc[oos_days].astype(np.float32)
    del close
    main_all = main.reindex(index=oos_days, columns=cols).astype(np.float32)

    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    base = _run_arms([("主模型 only", main_all)], fwd_engine, mask, bench_eqw,
                     bench_index, oos_days, [args.hold])
    base_sh = float(base[0]["sharpe"])
    for r in base:
        rows.append({"tag": "主模型 only", "rep": -1, "kind": "base", **r})
    log.info("主模型 only: 年化=%+.2f%% Sharpe=%.4f 回撤=%.1f%%",
             base[0]["annual"] * 100, base_sh, base[0]["max_dd"] * 100)

    for n in names:
        raw = pd.read_parquet(Path(str(reg.loc[n, "panel_path"])))
        cn = raw.columns.intersection(cols)
        neu, _ = neutralize_panel(raw, mc, ind, n)
        neu = neu.reindex(index=oos_days, columns=cn).astype(np.float32)
        cov = neu.notna()
        main_sub = main_all[cn]

        for w in weights:  # 真实融合
            fused = fuse_rank(main_sub, neu, w, cov)
            for r in _run_arms([("x", fused)], fwd_engine, mask, bench_eqw,
                               bench_index, oos_days, [args.hold]):
                rows.append({"tag": f"{n}|真实|w={w:.1f}", "rep": -1,
                             "kind": "real", "w": w, **r})
            del fused

        for rep in range(args.reps):  # 安慰剂
            placebo = row_shuffle_among_covered(neu, cov, rng)
            for w in weights:
                fused = fuse_rank(main_sub, placebo, w, cov)
                for r in _run_arms([("x", fused)], fwd_engine, mask, bench_eqw,
                                   bench_index, oos_days, [args.hold]):
                    rows.append({"tag": f"{n}|安慰剂|w={w:.1f}", "rep": rep,
                                 "kind": "placebo", "w": w, **r})
                del fused
            del placebo
            log.info("%-18s 安慰剂 rep %2d/%d", n, rep + 1, args.reps)
        del raw, neu, cov, main_sub

    df = pd.DataFrame(rows)
    df.to_csv(dest / "placebo.csv", index=False, encoding="utf-8-sig")

    body = df[df["kind"] != "base"].copy()
    body["arm"] = body["tag"].str.split("|").str[0]
    agg = (body.groupby(["arm", "kind", "w"], sort=False)
           .agg(sharpe=("sharpe", "mean"), sharpe_sd=("sharpe", "std"),
                annual=("annual", "mean"), max_dd=("max_dd", "mean"),
                n_rep=("sharpe", "size"))
           .reset_index())
    agg["delta"] = agg["sharpe"] - base_sh
    wide = agg.pivot_table(index=["arm", "w"], columns="kind",
                           values=["delta", "sharpe_sd"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index().rename(columns={"delta_real": "delta_real"})
    wide["delta_excess_over_placebo"] = (wide["delta_real"]
                                         - wide["delta_placebo"])
    wide["sigma_excess"] = (wide["delta_excess_over_placebo"]
                            / wide["sharpe_sd_placebo"])
    wide = wide[["arm", "w", "delta_real", "delta_placebo", "sharpe_sd_placebo",
                 "delta_excess_over_placebo", "sigma_excess"]]
    summ = pd.concat([pd.DataFrame([dict(arm="主模型 only", w=np.nan,
                                         delta_real=0.0)]), wide],
                     ignore_index=True)
    summ.to_csv(dest / "placebo_summary.csv", index=False, encoding="utf-8-sig")
    log.info("完成 -> %s", dest)


if __name__ == "__main__":
    main()
